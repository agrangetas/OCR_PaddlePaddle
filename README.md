# 📊 OCR Table Analysis Tool

Un module Python avancé pour l'extraction, l'analyse et l'export de tableaux à partir de documents numérisés avec PaddleOCR.

## 🚀 Fonctionnalités

- **🔍 Extraction automatique** de la structure des tableaux
- **📝 Assignment intelligent** des textes OCR aux cellules
- **🧹 Nettoyage automatique** et correction des chevauchements
- **📏 Espacement intelligent** basé sur les distances réelles
- **📤 Export HTML/Markdown** avec gestion des fusions de cellules
- **🎯 Visualisation interactive** de la structure détectée

## 📋 Prérequis

```bash
pip install numpy matplotlib pillow opencv-python
```

## ⚡ Utilisation rapide

```python
from src.utils import *

# 1. Charger les données PaddleOCR
layout_boxes, rec_boxes, rec_texts = load_paddleocr_data("donnees.json")

# 2. Extraire la structure du tableau
table_structure = extract_table_structure(layout_boxes, fill_empty_cells=True)

# 3. Assigner les textes OCR
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    force_assignment=True,
    auto_correct_overlaps=True,
    smart_spacing=True
)

# 4. Exporter en HTML
html_output = export_to_html(filled_structure)
save_html_to_file(html_output, "tableau.html")
```

## 🔧 Workflow détaillé

### Étape 1 : Extraction de la structure

```python
table_structure = extract_table_structure(
    layout_boxes,
    tolerance=10,              # Tolérance d'alignement (pixels)
    fill_empty_cells=True,     # Compléter les cellules manquantes
    extend_cells=False         # Étendre les cellules pour combler les espaces
)

# Visualiser la structure détectée
plot_table_structure(table_structure)
```

**Paramètres :**
- `tolerance` : Distance maximale pour considérer des lignes comme alignées
- `fill_empty_cells` : Active la génération automatique de cellules vides
- `extend_cells` : Active l'extension des cellules pour combler les espaces

### Étape 2 : Assignment des textes OCR

```python
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    overlap_threshold=0.5,     # Seuil de recouvrement minimum (0-1)
    force_assignment=True,     # Forcer l'assignment même sans recouvrement
    clean_structure=True,      # Nettoyer après l'assignment
    auto_correct_overlaps=True,# Corriger automatiquement les chevauchements
    smart_spacing=True,        # Espacement intelligent adaptatif
    verbose_overlaps=False     # Afficher les détails de détection
)

# Visualiser le résultat final
plot_final_result(filled_structure)
```

**Paramètres :**
- `overlap_threshold` : Pourcentage minimum de recouvrement requis (0.0 à 1.0)
- `force_assignment` : Force l'assignment du texte à la cellule la plus proche
- `clean_structure` : Active le nettoyage post-assignment
- `auto_correct_overlaps` : Correction automatique des chevauchements (récursive)
- `smart_spacing` : Espacement basé sur les distances réelles entre textes
- `verbose_overlaps` : Active l'affichage détaillé des diagnostics

### Étape 3 : Export et sauvegarde

```python
# Export HTML avec fusions de cellules
html_output = export_to_html(
    filled_structure,
    table_title="Mon Tableau",
    highlight_merged=True      # Colorer les cellules fusionnées
)
save_html_to_file(html_output, "tableau.html")

# Export Markdown
markdown_output = export_to_markdown(filled_structure, "Mon Tableau")
save_markdown_to_file(markdown_output, "tableau.md")
```

## 🎛️ Paramètres avancés

### Correction automatique des chevauchements

Le système applique plusieurs stratégies dans l'ordre de priorité :

1. **Réduction des cellules vides** : Réduit les cellules vides qui chevauchent des cellules pleines
2. **Fusion des duplicatas** : Fusionne les cellules quasi-identiques
3. **Absorption d'inclusions** : Absorbe les cellules complètement incluses
4. **Résolution de conflits** : Résout les conflits de position de grille
5. **Redimensionnement** : Ajuste les cellules avec chevauchement partiel

### Espacement intelligent

L'espacement adaptatif calcule automatiquement :

- **Espaces horizontaux** : Basés sur la largeur moyenne des caractères
- **Lignes verticales** : Basées sur la hauteur moyenne du texte
- **Adaptation contextuelle** : Ajustement selon la taille et densité de la cellule

```python
# Exemple avec paramètres personnalisés
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    overlap_threshold=0.3,     # Plus permissif
    force_assignment=True,     
    clean_structure=True,      
    auto_correct_overlaps=True,
    smart_spacing=True,        # Espacement intelligent
    verbose_overlaps=True      # Mode debug
)
```

## 📊 Exemples d'utilisation

### Exemple 1 : Traitement simple

```python
from src.utils import *

# Charger les données
layout_boxes, rec_boxes, rec_texts = load_paddleocr_data("data.json")

# Pipeline simple
table_structure = extract_table_structure(layout_boxes)
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    force_assignment=True
)

# Export
html_output = export_to_html(filled_structure)
save_html_to_file(html_output, "simple_table.html")
```

### Exemple 2 : Traitement avancé avec corrections

```python
from src.utils import *

# Charger les données
layout_boxes, rec_boxes, rec_texts = load_paddleocr_data("complex_data.json")

# Pipeline avancé
table_structure = extract_table_structure(
    layout_boxes,
    tolerance=15,              # Plus tolérant pour documents complexes
    fill_empty_cells=True,     # Compléter la grille
    extend_cells=True          # Étendre les cellules
)

# Visualiser la structure détectée
plot_table_structure(table_structure)

# Assignment avec toutes les corrections
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    overlap_threshold=0.4,     # Seuil adapté
    force_assignment=True,     
    clean_structure=True,      # Nettoyage post-assignment
    auto_correct_overlaps=True,# Correction automatique
    smart_spacing=True,        # Espacement intelligent
    verbose_overlaps=False     # Mode silencieux
)

# Visualiser le résultat
plot_final_result(filled_structure)

# Export avec mise en forme
html_output = export_to_html(
    filled_structure,
    table_title="Tableau Complexe",
    highlight_merged=True
)
save_html_to_file(html_output, "complex_table.html")
```

### Exemple 3 : Mode debug

```python
from src.utils import *

# Mode debug complet
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    overlap_threshold=0.5,
    force_assignment=True,
    clean_structure=True,
    auto_correct_overlaps=True,
    smart_spacing=True,
    verbose_overlaps=True      # Affichage détaillé
)

# Sortie debug :
# 🔍 Détection des chevauchements entre cellules:
#   ⚠️  Cellule #12 et #34 se chevauchent (DUPLICATE):
#       Cellule #12: (100.0,50.0→200.0,100.0) grille(1,2) span(1×1)
#       ...
# 🔄 Correction automatique des chevauchements (récursive)...
#   🔄 Itération 1/5
#   📊 3 chevauchements détectés à corriger
#   ...
```

## 🔧 Fonctions utilitaires

### Chargement des données

```python
# Charger depuis un fichier JSON PaddleOCR
layout_boxes, rec_boxes, rec_texts = load_paddleocr_data("donnees.json")

# Charger une image
image = load_image("document.jpg")
```

### Visualisation

```python
# Visualiser la structure détectée
plot_table_structure(table_structure, figsize=(12, 8))

# Visualiser le résultat final avec textes
plot_final_result(filled_structure, figsize=(15, 10))
```

### Nettoyage manuel

```python
# Nettoyer la structure manuellement
cleaned_structure = clean_table_structure(
    table_structure,
    tolerance=5,               # Tolérance d'alignement
    verbose_overlaps=False     # Mode silencieux
)
```

## 📈 Structure des données

### Format des cellules

Chaque cellule est représentée par un objet `TableCell` :

```python
class TableCell:
    x1, y1, x2, y2     # Coordonnées physiques (pixels)
    row_start, col_start # Position dans la grille
    row_span, col_span   # Spans de fusion
    texts              # Liste des textes OCR assignés
    final_text         # Texte final ordonné
    is_auto_filled     # Marqueur cellule auto-générée
```

### Format d'entrée PaddleOCR

```python
# layout_boxes : Liste de coordonnées [x1, y1, x2, y2]
layout_boxes = [[100, 50, 200, 100], [200, 50, 300, 100], ...]

# rec_boxes : Boxes des textes OCR [[x1, y1, x2, y2], ...]
rec_boxes = [[105, 55, 195, 95], [205, 55, 295, 95], ...]

# rec_texts : Textes correspondants
rec_texts = ["Texte 1", "Texte 2", ...]
```

## ⚠️ Notes importantes

1. **Ordre des opérations** : Toujours assigner les textes AVANT le nettoyage
2. **Chevauchements** : La correction automatique peut prendre plusieurs itérations
3. **Performance** : L'espacement intelligent est plus lent mais plus précis
4. **Encodage** : Tous les exports gèrent correctement l'UTF-8

## 🐛 Résolution de problèmes

### Cellules mal détectées
- Ajuster le paramètre `tolerance` dans `extract_table_structure()`
- Activer `fill_empty_cells=True` pour combler les trous

### Textes mal assignés
- Réduire `overlap_threshold` pour être plus permissif
- Activer `force_assignment=True` pour forcer l'assignment

### Chevauchements persistants
- Activer `verbose_overlaps=True` pour diagnostiquer
- La correction automatique résout la plupart des cas

### Espacement incorrect
- Désactiver `smart_spacing=False` pour revenir au mode basique
- L'espacement intelligent s'adapte automatiquement aux dimensions

## 📚 API Reference

### Fonctions principales

| Fonction | Description | Paramètres clés |
|----------|-------------|-----------------|
| `extract_table_structure()` | Extrait la structure du tableau | `tolerance`, `fill_empty_cells` |
| `assign_ocr_to_structure()` | Assigne les textes aux cellules | `overlap_threshold`, `auto_correct_overlaps` |
| `export_to_html()` | Export HTML avec rowspan/colspan | `highlight_merged` |
| `export_to_markdown()` | Export Markdown simple | `table_title` |

### Fonctions utilitaires

| Fonction | Description |
|----------|-------------|
| `load_paddleocr_data()` | Charge les données depuis JSON |
| `plot_table_structure()` | Visualise la structure détectée |
| `plot_final_result()` | Visualise le résultat final |
| `clean_table_structure()` | Nettoyage manuel de la structure |

---

**Développé avec ❤️ pour une analyse de tableaux OCR robuste et intelligente** 