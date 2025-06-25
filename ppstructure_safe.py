#!/usr/bin/env python3
"""
Script sécurisé pour PPStructureV3 
Avec gestion des erreurs et alternatives en cas de manque de mémoire
"""

import os
import gc
import psutil
from pathlib import Path

def get_memory_info():
    """Obtenir les informations mémoire"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent': memory.percent
    }

def run_ppstructure_safe():
    """Exécuter PPStructureV3 de manière sécurisée"""
    print("🚀 Démarrage de PPStructureV3 sécurisé")
    
    # Vérifier la mémoire disponible
    memory_info = get_memory_info()
    print(f"💾 Mémoire disponible: {memory_info['available_gb']:.2f}GB ({memory_info['percent']:.1f}% utilisée)")
    
    # Image path
    image_path = "/app/data/tableau_compte_resultat_ocr-1.png"
    
    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        print("💡 Placez votre image dans le dossier ./data/ sur l'hôte")
        return
    
    # Créer le dossier de sortie
    output_dir = "/app/output_parts"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from paddleocr import PPStructureV3
        
        # Choisir la configuration selon la mémoire disponible
        if memory_info['available_gb'] >= 3.0:
            print("💪 Assez de mémoire pour PP-DocLayout-L")
            pipeline = PPStructureV3(
                layout_detection_model_name="PP-DocLayout-L",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            print("✅ Pipeline PP-DocLayout-L initialisé")
        else:
            print("⚠️  Mémoire limitée - utilisation du modèle par défaut (plus léger)")
            pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            print("✅ Pipeline léger initialisé")
        
        # Prédiction
        print(f"🔍 Analyse de {image_path}...")
        output = pipeline.predict(image_path)
        
        # Traitement des résultats
        print("📊 Résultats:")
        for i, res in enumerate(output):
            print(f"\n--- Résultat {i+1} ---")
            res.print()
            
            # Sauvegarder les résultats
            json_path = f"{output_dir}/result_{i+1}.json"
            md_path = f"{output_dir}/result_{i+1}.md"
            
            res.save_to_json(save_path=json_path)
            res.save_to_markdown(save_path=md_path)
            
            print(f"💾 JSON sauvegardé: {json_path}")
            print(f"💾 Markdown sauvegardé: {md_path}")
        
        print(f"\n🎉 Analyse terminée! Résultats dans {output_dir}")
        
        # Nettoyer la mémoire
        del pipeline
        gc.collect()
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Assurez-vous que PaddleOCR 3.0.2 est installé")
        return False
        
    except RuntimeError as e:
        if "memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"❌ Erreur de mémoire: {e}")
            print("💡 Solutions:")
            print("  1. Redémarrez le kernel Jupyter")
            print("  2. Fermez d'autres notebooks")
            print("  3. Augmentez la RAM du conteneur Docker")
            print("  4. Utilisez un modèle plus léger")
        else:
            print(f"❌ Erreur RuntimeError: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal"""
    success = run_ppstructure_safe()
    
    if success:
        print("\n✅ Script terminé avec succès!")
    else:
        print("\n❌ Script terminé avec des erreurs")
        print("\n🔧 Débogage:")
        print("  - Exécutez 'python test_ppstructure_debug.py' pour plus de détails")
        print("  - Vérifiez les logs avec 'docker compose logs jupyter'")

if __name__ == "__main__":
    main() 