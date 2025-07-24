"""
Türk Hukuk Emsal Karar Sistemi - Optimized for Turkish Lawyers
Retrieves beneficial precedent decisions for legal petitions.
"""

import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import json
from typing import List, Dict, Any, Tuple
import subprocess
import tempfile
import os
import re

class TurkishLegalRetriever:
    def __init__(self):
        """Initialize the Turkish Legal RAG system optimized for CPU usage."""
        # Load SBERT model for semantic search
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        # Load FAISS index
        self.index = faiss.read_index("data/faiss.index")
        
        # Load metadata
        with open("data/meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        # Initialize BM25 for keyword matching
        corpus = []
        for item in self.meta:
            text = item["snippet"][:500] if item["snippet"] else ""
            corpus.append(text.split())
        self.bm25 = BM25Okapi(corpus)
        
        # Mistral model path for minimal LLM usage
        self.mistral_path = "models/mistral-7b-instruct-v0.2.Q4_0.gguf"
        self.top_k = 20
        
        # Performance optimizations
        self._llm = None  # Lazy load Mistral
        self._cached_analyses = {}  # Cache analysis results
        self._benefit_keywords = self._load_benefit_keywords()
    
    def _load_benefit_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that indicate beneficial precedents for petitions."""
        return {
            "positive_outcomes": [
                "kabul", "uygun", "doğru", "geçerli", "haklı", "yerinde", "onaylandı", 
                "desteklendi", "kabul edildi", "uygun bulundu", "doğru bulundu",
                "hukuka uygun", "adil", "hakkaniyetli", "memnuniyet verici"
            ],
            "legal_principles": [
                "prensip", "kural", "esas", "hukuki", "yasal", "mevzuat", "kanun",
                "yönetmelik", "tüzük", "anayasa", "temel hak", "insan hakları"
            ],
            "court_hierarchy": [
                "yargıtay", "danıştay", "anayasa mahkemesi", "genel kurul", 
                "büyük genel kurul", "hukuk genel kurul", "ceza genel kurul"
            ]
        }
    
    def _call_mistral_minimal(self, prompt: str, max_tokens: int = 512) -> str:
        """Call Mistral LLM with minimal token usage for CPU optimization."""
        try:
            from llama_cpp import Llama
            
            # Lazy load Mistral model
            if self._llm is None:
                self._llm = Llama(
                    model_path=self.mistral_path,
                    n_ctx=1024,  # Reduced context for speed
                    n_threads=4,
                    temperature=0.1,
                    repeat_penalty=1.1
                )
            
            # Call Mistral with minimal tokens
            response = self._llm(
                f"[INST] {prompt} [/INST]",
                max_tokens=max_tokens,
                stop=["</s>", "[INST]"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
                
        except Exception as e:
            print(f"Mistral çağrısı başarısız: {e}")
            return ""
    
    def _extract_search_keywords(self, query: str) -> List[str]:
        """Extract important legal keywords from user input using LLM."""
        # Check cache first
        if query in self._cached_analyses:
            return self._cached_analyses[query].get("keywords", [])
        
        prompt = f"""
Sen Türk hukuk uzmanısın. Bu dava açıklamasından arama için önemli keywordleri çıkar:

"{query}"

Sadece şu formatta yanıtla:
KEYWORDS: [virgülle ayrılmış 5-8 önemli hukuki terim]

Örnek:
KEYWORDS: nafaka, boşanma, velayet, tazminat, aile hukuku
KEYWORDS: kasten öldürme, meşru müdafaa, ceza hukuku, savunma
KEYWORDS: iş kazası, tazminat, işveren sorumluluğu, güvenlik önlemi
"""
        
        response = self._call_mistral_minimal(prompt, max_tokens=200)
        
        # Parse keywords
        keywords = []
        try:
            if "KEYWORDS:" in response:
                keywords_text = response.split("KEYWORDS:")[1].strip()
                keywords = [k.strip() for k in keywords_text.split(',')]
        except:
            pass
        
        # Fallback: extract basic keywords from query
        if not keywords:
            # Basic Turkish legal keyword extraction
            legal_terms = [
                "nafaka", "boşanma", "velayet", "tazminat", "aile", "evlilik", "eş",
                "kasten", "öldürme", "yaralama", "meşru müdafaa", "ceza", "hapis",
                "iş kazası", "işveren", "güvenlik", "sorumluluk", "borç", "alacak",
                "sözleşme", "miras", "taşınmaz", "mülkiyet", "hırsızlık", "dolandırıcılık"
            ]
            query_lower = query.lower()
            keywords = [term for term in legal_terms if term in query_lower]
            
            # If still no keywords, extract basic words
            if not keywords:
                # Extract important words from query
                important_words = ["nafaka", "boşanma", "dava", "karar", "emsal", "miktar", "konu"]
                keywords = [word for word in important_words if word in query_lower]
                
                # If still no keywords, use the most important words from query
                if not keywords:
                    words = query.lower().split()
                    # Filter out common words and keep legal-sounding ones
                    common_words = ["ve", "ile", "için", "konusunda", "arasında", "hakkında", "davası", "karar", "emsal", "arıyorum", "bul", "istiyorum"]
                    keywords = [word for word in words if word not in common_words and len(word) > 3][:5]
        
        # Cache the result
        if query not in self._cached_analyses:
            self._cached_analyses[query] = {}
        self._cached_analyses[query]["keywords"] = keywords
        
        return keywords
    
    def _keyword_match_score(self, decision_text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score between decision and keywords."""
        if not keywords:
            return 0.0
        
        text_lower = decision_text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
        
        return matches / len(keywords) if keywords else 0.0
    
    def _extract_case_intent(self, query: str) -> Dict[str, Any]:
        """Extract case intent using minimal LLM call."""
        # Check cache first
        if query in self._cached_analyses and "case_type" in self._cached_analyses[query]:
            return self._cached_analyses[query]
        
        prompt = f"""
Sen Türk hukuk uzmanısın. Bu davayı analiz et:

"{query}"

Sadece şu bilgileri ver:
1. Dava türü: [Ceza/Aile/Medeni/İş/İdari]
2. Ana konu: [Tek cümle]
3. Arama terimleri: [3-5 kelime, virgülle ayrılmış]
"""
        
        response = self._call_mistral_minimal(prompt, max_tokens=200)
        
        # Parse response
        intent = {
            "case_type": "Genel",
            "main_topic": "",
            "search_terms": [],
            "is_criminal": False
        }
        
        try:
            lines = response.split('\n')
            for line in lines:
                if "Dava türü:" in line:
                    case_type = line.split(":")[1].strip()
                    intent["case_type"] = case_type
                    intent["is_criminal"] = "Ceza" in case_type
                elif "Ana konu:" in line:
                    intent["main_topic"] = line.split(":")[1].strip()
                elif "Arama terimleri:" in line:
                    terms = line.split(":")[1].strip()
                    intent["search_terms"] = [t.strip() for t in terms.split(',')]
        except:
            # Fallback: use query as search terms
            intent["search_terms"] = query.split()[:5]
        
        # Cache the result
        if query not in self._cached_analyses:
            self._cached_analyses[query] = {}
        self._cached_analyses[query].update(intent)
        
        return intent
    
    def _calculate_benefit_score(self, decision_text: str, court_info: str) -> float:
        """Calculate how beneficial a decision is for petitions."""
        text_lower = decision_text.lower()
        court_lower = court_info.lower() if court_info else ""
        
        benefit_score = 0.0
        
        # Check for positive outcomes
        positive_count = sum(1 for keyword in self._benefit_keywords["positive_outcomes"] 
                           if keyword in text_lower)
        benefit_score += positive_count * 0.1
        
        # Check for legal principles
        principle_count = sum(1 for keyword in self._benefit_keywords["legal_principles"] 
                            if keyword in text_lower)
        benefit_score += principle_count * 0.05
        
        # Court hierarchy bonus
        for keyword in self._benefit_keywords["court_hierarchy"]:
            if keyword in court_lower:
                if "genel kurul" in court_lower or "anayasa" in court_lower:
                    benefit_score += 0.3
                elif "yargıtay" in court_lower or "danıştay" in court_lower:
                    benefit_score += 0.2
                break
        
        # Recency bonus (minimal)
        if "2023" in court_info or "2024" in court_info:
            benefit_score += 0.1
        
        return min(benefit_score, 1.0)
    
    def _search_beneficial_decisions(self, query: str, intent: Dict[str, Any]) -> List[Dict]:
        """Search for decisions that are both similar and beneficial for petitions."""
        # Extract keywords from user input
        keywords = self._extract_search_keywords(query)
        print(f"🔍 Çıkarılan anahtar kelimeler: {', '.join(keywords)}")
        
        # Create enhanced query with search terms
        search_terms = intent["search_terms"]
        enhanced_query = f"{query} {' '.join(search_terms)}"
        
        # Semantic search
        query_embedding = self.model.encode([enhanced_query])
        scores, indices = self.index.search(query_embedding, self.top_k * 3)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.meta):
                continue
            
            # Get full decision data
            try:
                conn = sqlite3.connect("data/decisions.sqlite")
                cur = conn.cursor()
                cur.execute("SELECT daire, esas, karar, tarih, raw_text FROM decisions WHERE id = ?", (self.meta[idx]["id"],))
                result = cur.fetchone()
                if not result:
                    conn.close()
                    continue
                    
                court_info, esas, karar, tarih, raw_text = result
                conn.close()
                
                # Calculate semantic similarity score
                semantic_score = float(scores[0][i])
                semantic_score = 1.0 / (1.0 + np.exp(-(semantic_score - 2.0) * 0.4))
                semantic_score = 0.6 + (semantic_score * 0.3)
                
                # Calculate keyword matching score
                decision_text = raw_text if raw_text else self.meta[idx]["snippet"]
                keyword_score = self._keyword_match_score(decision_text, keywords)
                
                # Calculate benefit score
                benefit_score = self._calculate_benefit_score(decision_text, court_info)
                
                # Combined score: 35% similarity + 35% keyword + 20% benefit
                final_score = (semantic_score * 0.35) + (keyword_score * 0.35) + (benefit_score * 0.30)
                final_score = max(0.60, min(final_score, 0.95))
                
                results.append({
                    "id": self.meta[idx]["id"],
                    "score": final_score,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "benefit_score": benefit_score,
                    "snippet": self.meta[idx]["snippet"],
                    "daire": court_info or "Bilinmiyor",
                    "esas": esas or "Bilinmiyor",
                    "karar": karar or "Bilinmiyor",
                    "tarih": tarih or "Bilinmiyor",
                    "raw_text": raw_text or ""
                })
                
            except Exception as e:
                continue
        
        return results
    
    def _explain_top_3_benefits(self, query: str, top_3_results: List[Dict]) -> str:
        """Explain how to use top 3 decisions for petition benefits."""
        if len(top_3_results) < 3:
            return "Yeterli emsal karar bulunamadı."
        
        # Prepare decision summaries
        decisions_text = ""
        for i, result in enumerate(top_3_results, 1):
            decisions_text += f"""
{i}. {result['daire']} - {result['esas']}/{result['karar']} ({result['tarih']})
   Benzerlik: {result['semantic_score']:.1%} | Keyword: {result['keyword_score']:.1%} | Fayda: {result['benefit_score']:.1%}
   Özet: {result['snippet'][:150]}...
"""
        
        prompt = f"""
Sen deneyimli bir Türk avukatısın. Bu 3 emsal kararı analiz et:

DAVA: "{query}"

EMSAL KARARLAR:
{decisions_text}

Her karar için şunları açıkla:
1. Bu karar nasıl emsal olarak kullanılabilir?
2. Dilekçede hangi argümanları destekler?
3. Dikkat edilmesi gereken noktalar nelerdir?

Sonra genel değerlendirme:
- Bu kararların gücü nedir?
- Hangi noktalara odaklanmalı?
- Tahmini başarı şansı nedir?

SADECE TÜRKÇE YANITLA. Kısa ve öz ol.
"""
        
        explanation = self._call_mistral_minimal(prompt, max_tokens=800)
        
        if not explanation:
            return self._fallback_explanation(top_3_results)
        
        return explanation
    
    def _fallback_explanation(self, top_3_results: List[Dict]) -> str:
        """Fallback explanation when LLM fails."""
        explanation = "**En İyi 3 Emsal Kararın Değerlendirmesi:**\n\n"
        
        for i, result in enumerate(top_3_results, 1):
            explanation += f"{i}. **{result['daire']}** ({result['esas']}/{result['karar']}):\n"
            explanation += f"   - Benzerlik: {result['semantic_score']:.1%} | Keyword: {result['keyword_score']:.1%} | Fayda: {result['benefit_score']:.1%}\n"
            explanation += f"   - Bu karar, sorunuzla ilgili hukuki prensipleri içermektedir.\n"
            explanation += f"   - Dilekçenizde emsal olarak kullanılabilir.\n\n"
        
        explanation += "**Genel Değerlendirme:** Bu kararlar, sorunuzla ilgili en uygun ve faydalı hukuki emsalleri içermektedir."
        return explanation
    
    def search_beneficial_precedents(self, case_description: str) -> Dict[str, Any]:
        """Main function to search for beneficial precedent decisions."""
        print("🔍 Dava analizi yapılıyor...")
        
        # Clear entire cache for fresh analysis
        self._cached_analyses = {}
        
        # Extract keywords directly
        keywords = self._extract_search_keywords(case_description)
        print(f"🔍 Çıkarılan anahtar kelimeler: {', '.join(keywords)}")
        
        # Step 1: Extract case intent (minimal LLM call)
        intent = self._extract_case_intent(case_description)
        
        print(f"📋 Dava Türü: {intent['case_type']}")
        print(f"🎯 Ana Konu: {intent['main_topic']}")
        
        # Step 2: Search for beneficial decisions
        print("🔎 Faydalı emsal kararlar aranıyor...")
        all_results = self._search_beneficial_decisions(case_description, intent)
        
        # Step 3: Sort by combined score (similarity + keyword + benefit)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_20_results = all_results[:20]
        top_3_results = all_results[:3]
        
        # Step 4: Generate explanation for top 3 (minimal LLM call)
        print("🧠 En iyi 3 karar analiz ediliyor...")
        explanation = self._explain_top_3_benefits(case_description, top_3_results)
        
        return {
            "top_20_results": top_20_results,
            "top_3_results": top_3_results,
            "explanation": explanation,
            "intent": intent,
            "total_found": len(all_results)
        }

# Backward compatibility
Retriever = TurkishLegalRetriever
