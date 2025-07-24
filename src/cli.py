"""
Türk Hukuk Emsal Karar Sistemi - Command Line Interface
Terminal-based interface for Turkish lawyers to find beneficial precedent decisions.
"""

import argparse
import sys
from .retriever import TurkishLegalRetriever

def main():
    parser = argparse.ArgumentParser(
        description="Türk Hukuk Emsal Karar Sistemi - Faydalı Emsal Kararlar Bulma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanım:
  python -m legalrag.cli "Müvekkilim kasten öldürme suçundan yargılanıyor, meşru müdafaa savunması yapmak istiyoruz"
  python -m legalrag.cli "Boşanma davasında nafaka miktarı konusunda emsal karar arıyorum"
  python -m legalrag.cli "İş kazası sonrası tazminat davası için faydalı kararlar bul"
        """
    )
    parser.add_argument("case_description", help="Dava açıklaması (doğal dil)")
    parser.add_argument("--show-all", action="store_true", help="Tüm 20 kararı göster")
    parser.add_argument("--show-snippets", action="store_true", help="Karar özetlerini göster")
    parser.add_argument("--show-scores", action="store_true", help="Benzerlik, keyword ve fayda skorlarını göster")
    parser.add_argument("--show-content", action="store_true", help="Kararların tam içeriğini göster")
    parser.add_argument("--content-limit", type=int, default=1000, help="İçerik karakter limiti (varsayılan: 1000)")
    
    args = parser.parse_args()
    
    try:
        print("⚖️  Türk Hukuk Emsal Karar Sistemi")
        print("=" * 60)
        print(f"📝 Dava: {args.case_description}")
        print("=" * 60)
        
        # Initialize retriever
        retriever = TurkishLegalRetriever()
        
        # Search for beneficial precedents
        result = retriever.search_beneficial_precedents(args.case_description)
        
        # Display results
        print(f"\n✅ {len(result['top_20_results'])} faydalı emsal karar bulundu")
        print("=" * 60)
        
        # Show top 20 results if requested
        if args.show_all:
            print(f"\n📋 İlk 20 Emsal Karar:")
            print("-" * 60)
            
            for i, decision in enumerate(result['top_20_results'], 1):
                print(f"{i:2d}. {decision['daire']} - {decision['esas']}/{decision['karar']} ({decision['tarih']})")
                
                if args.show_scores:
                    print(f"     Benzerlik: {decision['semantic_score']:.1%} | Keyword: {decision['keyword_score']:.1%} | Fayda: {decision['benefit_score']:.1%} | Toplam: {decision['score']:.1%}")
                
                if args.show_snippets:
                    snippet = decision['snippet'][:200] + "..." if len(decision['snippet']) > 200 else decision['snippet']
                    print(f"     Özet: {snippet}")
                
                if args.show_content and decision.get('raw_text'):
                    content = decision['raw_text'][:args.content_limit]
                    if len(decision['raw_text']) > args.content_limit:
                        content += "..."
                    print(f"     İçerik: {content}")
                    print()
                else:
                    print()
        
        # Always show top 3 with detailed analysis
        print(f"\n🏆 En İyi 3 Emsal Karar (Dilekçe İçin En Faydalı):")
        print("=" * 60)
        
        for i, decision in enumerate(result['top_3_results'], 1):
            print(f"{i}. {decision['daire']} - {decision['esas']}/{decision['karar']} ({decision['tarih']})")
            print(f"   Benzerlik: {decision['semantic_score']:.1%} | Keyword: {decision['keyword_score']:.1%} | Fayda: {decision['benefit_score']:.1%} | Toplam: {decision['score']:.1%}")
            
            if args.show_snippets:
                snippet = decision['snippet'][:300] + "..." if len(decision['snippet']) > 300 else decision['snippet']
                print(f"   Özet: {snippet}")
            
            if args.show_content and decision.get('raw_text'):
                content = decision['raw_text'][:args.content_limit]
                if len(decision['raw_text']) > args.content_limit:
                    content += "..."
                print(f"   İçerik: {content}")
            
            print("-" * 40)
        
        # Display LLM explanation
        print(f"\n🧠 Uzman Analizi (En İyi 3 Kararın Kullanımı):")
        print("=" * 60)
        print(result['explanation'])
        
        # Summary
        print(f"\n📊 Özet:")
        print(f"   • Toplam bulunan karar: {result['total_found']}")
        print(f"   • Dava türü: {result['intent']['case_type']}")
        print(f"   • Ana konu: {result['intent']['main_topic']}")
        print(f"   • Arama terimleri: {', '.join(result['intent']['search_terms'])}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
