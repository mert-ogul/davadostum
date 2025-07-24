from __future__ import annotations

"""mcp_scraper.py – Fetch Yargıtay decisions through the Yargı MCP Bedesten API.

Run:
    # 1️⃣ Önce ayrı bir terminalde MCP sunucusunu başlatın
    #    uvx yargi-mcp &
    # 2️⃣ Sonra bu modülü çalıştırın
    #    python -m legalrag.mcp_scraper

Bu versiyon Selenium kullanmaz; Bedesten API JSON çıktısı üzerinden veri çeker ve
reCAPTCHA sorununu tamamen ortadan kaldırır.
"""

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List

import anyio
from fastmcp import Client

from .settings import load_config
from .utils import ensure_dirs, get_connection

logging.basicConfig(level=logging.INFO)

# MCP sunucusunun varsayılan adresi. Değiştirmek isterseniz config.toml altına
# [mcp] section ekleyip host & port belirtin.
DEFAULT_MCP_URL = "http://127.0.0.1:3333"
SEARCH_TOOL = "search_bedesten_unified"
GET_TOOL = "get_bedesten_document_markdown"
CHECKPOINT_FILE = "data/scraper_checkpoint.json"


def _load_checkpoint() -> Dict[str, Any]:
    """Checkpoint dosyasından ilerlemeyi yükle."""
    try:
        import json
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_page": 0, "total_processed": 0}


def _save_checkpoint(page: int, total_processed: int):
    """Checkpoint dosyasına ilerlemeyi kaydet."""
    import json
    ensure_dirs()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_page": page, "total_processed": total_processed}, f)


def _get_existing_document_ids(cur) -> set:
    """Veritabanında mevcut olan document ID'leri al."""
    cur.execute("SELECT url FROM decisions WHERE url != ''")
    urls = {row[0] for row in cur.fetchall()}
    return urls


def _call_mcp(client: Client, tool_name: str, **params):
    """fastmcp.Client aracı senkron olarak çağır (anyio.run wrap)."""

    async def _runner():
        res = await client.call_tool(tool_name, params)
        # Bedesten aracı structured_content döndürür
        if res.structured_content is not None:
            return res.structured_content
        # Downgrade path: if data set
        if res.data is not None:
            return res.data
        # Eğer sadece ContentBlock ise boş sözlük döndür
        return {}

    return anyio.run(_runner)


def _save_decision(cur, meta: Dict[str, Any], raw_md: str):
    """Kararı decisions tablosuna ekle. Meta alanlarını olabildiğince eşleştirir."""
    # Bedesten'den gelen meta örnek alanlar: decisionDate, decisionNumber, courtChamber…
    url = meta.get("source_url") or meta.get("url") or meta.get("detailUrl") or ""
    daire = meta.get("birimAdi") or meta.get("courtChamber") or meta.get("daire") or ""
    esas = meta.get("esasNo") or meta.get("mainId") or meta.get("esas") or ""
    karar = meta.get("kararNo") or meta.get("decisionNumber") or meta.get("karar") or ""
    tarih = meta.get("kararTarihiStr") or meta.get("decisionDate") or meta.get("tarih") or ""

    cur.execute(
        """
        INSERT INTO decisions
        (url, daire, esas, karar, tarih, raw_text)
        VALUES (?,?,?,?,?,?)
        """,
        (url, daire, esas, karar, tarih, raw_md),
    )


async def _async_main():
    ensure_dirs()
    cfg = load_config()
    db_path: str = cfg["paths"]["db"]

    mcp_url = cfg.get("mcp", {}).get("url", DEFAULT_MCP_URL)
    client = Client(mcp_url + "/mcp/")

    scrape_cfg = cfg["scraping"]
    start_year = int(scrape_cfg.get("start_year", 2015))
    end_year = int(scrape_cfg.get("end_year", _dt.datetime.now().year))
    phrase = scrape_cfg.get("phrase", "içtihat")

    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                url TEXT,
                daire TEXT,
                esas TEXT,
                karar TEXT,
                tarih TEXT,
                raw_text TEXT
            )
            """
        )

        # Checkpoint'ten ilerlemeyi yükle
        checkpoint = _load_checkpoint()
        start_page = checkpoint["last_page"] + 1
        total_saved = checkpoint["total_processed"]
        
        # Mevcut document ID'leri al
        existing_urls = _get_existing_document_ids(cur)
        
        logging.info("Checkpoint'ten devam ediliyor: Sayfa %s, Toplam işlenen: %s", start_page, total_saved)
        logging.info("Mevcut karar sayısı: %s", len(existing_urls))

        async with client:
            logging.info("Tüm Yargıtay kararları aranıyor…")
            search_params = dict(
                phrase=phrase,
                court_types=["YARGITAYKARARI"],
                pageNumber=start_page,
            )

            while True:
                results = await client.call_tool(SEARCH_TOOL, search_params)
                data = results.structured_content or results.data or {}
                docs: List[Dict[str, Any]] = data.get("decisions") or []
                logging.info("Year all page %s: %s decisions, total_records: %s", 
                           "all", search_params["pageNumber"], len(docs), data.get("total_records", 0))
                if not docs:
                    break

                logging.info("Sayfa %s → %s karar", search_params["pageNumber"], len(docs))

                page_saved = 0
                for doc in docs:
                    doc_id = doc.get("documentId") or doc.get("id")
                    if not doc_id:
                        continue
                    
                    # URL oluştur
                    doc_url = f"https://bedesten.adalet.gov.tr/document/{doc_id}"
                    
                    # Eğer bu karar zaten varsa atla
                    if doc_url in existing_urls:
                        continue
                    
                    try:
                        md_res = await client.call_tool(GET_TOOL, {"documentId": doc_id})
                        md_data = md_res.structured_content or md_res.data
                        if hasattr(md_data, 'markdown_content'):
                            raw_md = md_data.markdown_content
                        elif isinstance(md_data, dict):
                            raw_md = md_data.get("markdown_content") or md_data.get("markdown") or md_data.get("text") or ""
                        else:
                            raw_md = ""
                        
                        # URL'yi meta'ya ekle
                        doc["source_url"] = doc_url
                        _save_decision(cur, doc, raw_md)
                        total_saved += 1
                        page_saved += 1
                        existing_urls.add(doc_url)
                    except Exception as e:
                        logging.warning("Document %s retrieval failed: %s", doc_id, str(e))
                        # Save decision without content
                        doc["source_url"] = doc_url
                        _save_decision(cur, doc, "")
                        total_saved += 1
                        page_saved += 1
                        existing_urls.add(doc_url)

                conn.commit()
                
                # Checkpoint'i kaydet
                _save_checkpoint(search_params["pageNumber"], total_saved)
                
                logging.info("Sayfa %s tamamlandı: %s yeni karar eklendi", search_params["pageNumber"], page_saved)

                # Bedesten returns page_size & total_records but not hasNextPage
                if len(docs) == data.get("page_size", 10):
                    search_params["pageNumber"] += 1
                else:
                    break

    logging.info("Toplam %s karar veritabanına eklendi.", total_saved)


def main():
    anyio.run(_async_main)


if __name__ == "__main__":
    main() 