#!/usr/bin/env python3
"""
🔧 utils_new.py - Module simplifié pour l'analyse de tableaux OCR

Ce module contient uniquement les fonctions essentielles pour :
1. Extraire la structure d'un tableau à partir du layout
2. Visualiser cette structure
3. Placer les textes OCR dans les cellules
4. Visualiser le résultat final  
5. Exporter en HTML avec rowspan/colspan

Utilisation simple depuis un notebook :
```python
from src.utils_new import *

# 1. Extraire la structure du tableau
table_structure = extract_table_structure(layout_boxes)

# 2. Visualiser la structure
plot_table_structure(table_structure)

# 3. Placer les textes OCR
filled_structure = assign_ocr_to_structure(table_structure, rec_boxes, rec_texts)

# 4. Visualiser le résultat
plot_final_result(filled_structure)

# 5. Exporter en HTML
html_output = export_to_html(filled_structure)
```
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import cv2
from PIL import Image, ImageDraw, ImageFont


# === STRUCTURE DE DONNÉES ===

class TableCell:
    """Cellule de tableau avec position et spans"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 row_start: int, col_start: int, row_span: int = 1, col_span: int = 1):
        self.x1 = x1
        self.y1 = y1  
        self.x2 = x2
        self.y2 = y2
        self.row_start = row_start
        self.col_start = col_start
        self.row_span = row_span
        self.col_span = col_span
        self.texts = []  # Liste des textes OCR assignés
        self.final_text = ""  # Texte final ordonné
        
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def contains_point(self, x: float, y: float, tolerance: float = 5) -> bool:
        """Vérifie si un point est dans la cellule (avec tolérance)"""
        return (self.x1 - tolerance <= x <= self.x2 + tolerance and 
                self.y1 - tolerance <= y <= self.y2 + tolerance)
    
    def overlap_with_box(self, box: List[float]) -> float:
        """Calcule le pourcentage de recouvrement avec une box OCR"""
        x1, y1, x2, y2 = box
        
        # Intersection
        inter_x1 = max(self.x1, x1)
        inter_y1 = max(self.y1, y1)
        inter_x2 = min(self.x2, x2)
        inter_y2 = min(self.y2, y2)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box_area = (x2 - x1) * (y2 - y1)
        
        return inter_area / box_area if box_area > 0 else 0.0


# === FONCTION 1 : EXTRAIRE LA STRUCTURE DU TABLEAU ===

def extract_table_structure(layout_boxes: List[Dict], tolerance: float = 10) -> List[TableCell]:
    """
    Extrait la structure du tableau à partir des boxes de layout.
    
    Args:
        layout_boxes: Liste des boxes de layout (format PaddleOCR)
        tolerance: Tolérance pour détecter les alignements
        
    Returns:
        Liste des cellules du tableau avec rowspan/colspan
    """
    if not layout_boxes:
        return []
    
    # Extraire les coordonnées des cellules
    cells_coords = layout_boxes
    
    if not cells_coords:
        return []
    
    # Trier par ymax (ligne du haut) puis xmin (à gauche)
    cells_coords.sort(key=lambda x: (x[1], x[0]))  # (y1, x1)
    
    # Détecter les lignes et colonnes de la grille
    row_lines = _detect_grid_lines(cells_coords, 'horizontal', tolerance)
    col_lines = _detect_grid_lines(cells_coords, 'vertical', tolerance)
    
    # Créer les cellules avec leurs positions de grille
    table_cells = []
    for x1, y1, x2, y2 in cells_coords:
        row_start, row_end = _find_grid_position(y1, y2, row_lines, tolerance)
        col_start, col_end = _find_grid_position(x1, x2, col_lines, tolerance)
        
        row_span = row_end - row_start
        col_span = col_end - col_start
        
        cell = TableCell(x1, y1, x2, y2, row_start, col_start, row_span, col_span)
        table_cells.append(cell)
    
    return table_cells


def _detect_grid_lines(cells_coords: List[List[float]], direction: str, tolerance: float) -> List[float]:
    """Détecte les lignes de grille horizontales ou verticales"""
    if direction == 'horizontal':
        # Utiliser y1 et y2 pour les lignes horizontales
        positions = []
        for x1, y1, x2, y2 in cells_coords:
            positions.extend([y1, y2])
    else:  # vertical
        # Utiliser x1 et x2 pour les lignes verticales
        positions = []
        for x1, y1, x2, y2 in cells_coords:
            positions.extend([x1, x2])
    
    # Grouper les positions similaires
    positions.sort()
    lines = []
    current_group = [positions[0]]
    
    for pos in positions[1:]:
        if pos - current_group[-1] <= tolerance:
            current_group.append(pos)
        else:
            # Nouvelle ligne
            lines.append(sum(current_group) / len(current_group))
            current_group = [pos]
    
    # Ajouter le dernier groupe
    lines.append(sum(current_group) / len(current_group))
    
    return sorted(lines)


def _find_grid_position(start: float, end: float, grid_lines: List[float], tolerance: float) -> Tuple[int, int]:
    """Trouve la position d'une cellule dans la grille"""
    start_idx = 0
    end_idx = len(grid_lines) - 1
    
    # Trouver l'index de début
    for i, line in enumerate(grid_lines):
        if abs(start - line) <= tolerance:
            start_idx = i
            break
    
    # Trouver l'index de fin
    for i, line in enumerate(grid_lines):
        if abs(end - line) <= tolerance:
            end_idx = i
            break
    
    return start_idx, end_idx


# === FONCTION 2 : VISUALISER LA STRUCTURE ===

def plot_table_structure(table_structure: List[TableCell], 
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualise la structure du tableau détectée.
    
    Args:
        table_structure: Liste des cellules du tableau
        figsize: Taille de la figure
    """
    if not table_structure:
        print("Aucune cellule à afficher")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Déterminer les dimensions du tableau
    min_x = min(cell.x1 for cell in table_structure)
    max_x = max(cell.x2 for cell in table_structure)
    min_y = min(cell.y1 for cell in table_structure)
    max_y = max(cell.y2 for cell in table_structure)
    
    # Définir les limites avec un peu de marge
    margin = 20
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(max_y + margin, min_y - margin)  # Inverser Y pour avoir l'origine en haut
    
    # Fond blanc
    ax.set_facecolor('white')
    
    # Dessiner chaque cellule
    for i, cell in enumerate(table_structure):
        # Rectangle de la cellule
        rect = patches.Rectangle(
            (cell.x1, cell.y1), 
            cell.x2 - cell.x1, 
            cell.y2 - cell.y1,
            linewidth=2, 
            edgecolor='blue', 
            facecolor='lightblue',
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # Texte avec les informations de la cellule
        center_x, center_y = cell.center()
        info_text = f"({cell.row_start},{cell.col_start})\n{cell.row_span}×{cell.col_span}"
        ax.text(center_x, center_y, info_text, 
                ha='center', va='center', 
                fontsize=10, color='darkblue', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax.set_title(f'Structure du tableau détectée ({len(table_structure)} cellules)')
    ax.set_xlabel('Position X (pixels)')
    ax.set_ylabel('Position Y (pixels)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# === FONCTION 3 : PLACER LES TEXTES OCR ===

def assign_ocr_to_structure(table_structure: List[TableCell], 
                          rec_boxes: List[List[float]], 
                          rec_texts: List[str],
                          overlap_threshold: float = 0.1,
                          force_assignment: bool = False) -> List[TableCell]:
    """
    Place les textes OCR dans les cellules du tableau.
    
    Args:
        table_structure: Structure du tableau
        rec_boxes: Boxes des textes OCR [x1, y1, x2, y2]
        rec_texts: Textes correspondants
        overlap_threshold: Seuil de recouvrement minimal
        force_assignment: Si True, place TOUS les textes même sans recouvrement (par centre)
        
    Returns:
        Structure du tableau avec textes assignés
    """
    # Copier la structure pour ne pas modifier l'originale
    filled_structure = [TableCell(cell.x1, cell.y1, cell.x2, cell.y2, 
                                 cell.row_start, cell.col_start, 
                                 cell.row_span, cell.col_span) 
                       for cell in table_structure]
    
    # ÉTAPE 1 : Liaison des textes OCR aux cellules
    used_texts = set()
    
    for text_idx, (rec_box, rec_text) in enumerate(zip(rec_boxes, rec_texts)):
        if not rec_text.strip() or text_idx in used_texts:
            continue
            
        best_cell = None
        best_overlap = 0
        all_overlaps = []
        
        # Trouver la cellule avec le meilleur recouvrement
        for cell in filled_structure:
            overlap = cell.overlap_with_box(rec_box)
            all_overlaps.append(overlap)
            if overlap > best_overlap and overlap >= overlap_threshold:
                best_overlap = overlap
                best_cell = cell
        
        # Assigner le texte à la meilleure cellule
        if best_cell is not None:
            best_cell.texts.append({
                'text': rec_text.strip(),
                'box': rec_box,
                'center': ((rec_box[0] + rec_box[2]) / 2, (rec_box[1] + rec_box[3]) / 2)
            })
            used_texts.add(text_idx)
        else:
            # DIAGNOSTIC : Texte non matché
            max_overlap = max(all_overlaps) if all_overlaps else 0
            box_center = ((rec_box[0] + rec_box[2]) / 2, (rec_box[1] + rec_box[3]) / 2)
            box_size = (rec_box[2] - rec_box[0], rec_box[3] - rec_box[1])
            
            print(f"❌ TEXTE NON MATCHÉ: '{rec_text.strip()}'")
            print(f"   📦 Box: {rec_box} | Centre: {box_center} | Taille: {box_size}")
            print(f"   📊 Meilleur recouvrement: {max_overlap:.3f} (seuil: {overlap_threshold})")
            
            if max_overlap > 0:
                print(f"   🔍 Raison: Recouvrement {max_overlap:.3f} < seuil {overlap_threshold}")
            else:
                print(f"   🔍 Raison: Aucun recouvrement avec les cellules du tableau")
            
            # OPTION FORCE : Placer par centre de la box
            if force_assignment:
                # Trouver la cellule qui contient le centre de la box de texte
                target_cell = None
                for cell in filled_structure:
                    if (cell.x1 <= box_center[0] <= cell.x2 and 
                        cell.y1 <= box_center[1] <= cell.y2):
                        target_cell = cell
                        break
                
                if target_cell:
                    target_cell.texts.append({
                        'text': rec_text.strip(),
                        'box': rec_box,
                        'center': box_center
                    })
                    used_texts.add(text_idx)
                    print(f"   🔧 FORCE: Placé dans cellule ({target_cell.row_start},{target_cell.col_start}) par centre")
                else:
                    print(f"   🔧 FORCE: Centre hors de toutes les cellules - texte ignoré")
            else:
                # Vérifier si le texte est proche d'une cellule
                min_distance = float('inf')
                closest_cell = None
                for cell in filled_structure:
                    cell_center = cell.center()
                    distance = ((box_center[0] - cell_center[0])**2 + (box_center[1] - cell_center[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_cell = cell
                
                if closest_cell:
                    print(f"   📍 Cellule la plus proche: centre {closest_cell.center()}, distance: {min_distance:.1f}px")
            print()
    
    # ÉTAPE 2 : Ordonnancement spatial des textes dans chaque cellule
    for cell in filled_structure:
        if cell.texts:
            cell.final_text = _order_texts_spatially(cell.texts)
    
    return filled_structure


def _order_texts_spatially(texts: List[Dict]) -> str:
    """
    Ordonne les textes dans une cellule selon leur position spatiale.
    
    Args:
        texts: Liste des textes avec leurs positions
        
    Returns:
        Texte final ordonné avec espaces et retours à la ligne
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]['text']
    
    # Trier par position Y (haut vers bas) puis X (gauche vers droite)
    texts.sort(key=lambda t: (t['center'][1], t['center'][0]))
    
    # Grouper par lignes approximatives
    lines = []
    current_line = [texts[0]]
    y_tolerance = 10
    
    for text in texts[1:]:
        if abs(text['center'][1] - current_line[0]['center'][1]) <= y_tolerance:
            current_line.append(text)
        else:
            lines.append(current_line)
            current_line = [text]
    lines.append(current_line)
    
    # Construire le texte final
    result_lines = []
    for line in lines:
        # Trier les textes de la ligne par X (gauche vers droite)
        line.sort(key=lambda t: t['center'][0])
        line_text = ' '.join(t['text'] for t in line)
        result_lines.append(line_text)
    
    return '\n'.join(result_lines)


# === FONCTION 4 : VISUALISER LE RÉSULTAT ===

def plot_final_result(filled_structure: List[TableCell], 
                     figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualise le résultat final avec les textes placés.
    
    Args:
        filled_structure: Structure avec textes assignés
        figsize: Taille de la figure
    """
    if not filled_structure:
        print("Aucune cellule à afficher")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Déterminer les dimensions du tableau
    min_x = min(cell.x1 for cell in filled_structure)
    max_x = max(cell.x2 for cell in filled_structure)
    min_y = min(cell.y1 for cell in filled_structure)
    max_y = max(cell.y2 for cell in filled_structure)
    
    # Définir les limites avec un peu de marge
    margin = 20
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(max_y + margin, min_y - margin)  # Inverser Y pour avoir l'origine en haut
    
    # Fond blanc
    ax.set_facecolor('white')
    
    # Dessiner chaque cellule avec son contenu
    for cell in filled_structure:
        # Rectangle de la cellule
        if cell.final_text.strip():
            color = 'green'
            facecolor = 'lightgreen'
        else:
            color = 'gray'
            facecolor = 'lightgray'
        
        rect = patches.Rectangle(
            (cell.x1, cell.y1), 
            cell.x2 - cell.x1, 
            cell.y2 - cell.y1,
            linewidth=2, 
            edgecolor=color, 
            facecolor=facecolor,
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # Texte de la cellule
        if cell.final_text.strip():
            center_x, center_y = cell.center()
            # Limiter la longueur du texte affiché
            display_text = cell.final_text[:50] + "..." if len(cell.final_text) > 50 else cell.final_text
            ax.text(center_x, center_y, display_text, 
                    ha='center', va='center', 
                    fontsize=8, color='darkgreen', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                    wrap=True)
    
    # Statistiques
    filled_cells = len([cell for cell in filled_structure if cell.final_text.strip()])
    total_cells = len(filled_structure)
    
    ax.set_title(f'Résultat final: {filled_cells}/{total_cells} cellules remplies')
    ax.set_xlabel('Position X (pixels)')
    ax.set_ylabel('Position Y (pixels)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# === FONCTION 5 : EXPORT HTML ===

def export_to_html(filled_structure: List[TableCell], 
                  table_title: str = "Tableau OCR", 
                  table_class: str = "ocr-table",
                  highlight_merged: bool = True) -> str:
    """
    Exporte la structure en HTML avec rowspan/colspan.
    
    Args:
        filled_structure: Structure avec textes assignés
        table_title: Titre du tableau
        table_class: Classe CSS
        highlight_merged: Si True, colorie les cellules fusionnées en jaune
        
    Returns:
        Code HTML du tableau
    """
    if not filled_structure:
        return f"<p>Tableau vide</p>"
    
    # Déterminer la taille de la grille
    max_row = max(cell.row_start + cell.row_span for cell in filled_structure)
    max_col = max(cell.col_start + cell.col_span for cell in filled_structure)
    
    # Créer une grille pour suivre les cellules occupées
    grid_occupied = [[False for _ in range(max_col)] for _ in range(max_row)]
    
    # Marquer les cellules occupées
    for cell in filled_structure:
        for r in range(cell.row_start, cell.row_start + cell.row_span):
            for c in range(cell.col_start, cell.col_start + cell.col_span):
                if r < max_row and c < max_col:
                    grid_occupied[r][c] = True
    
    # Générer le CSS avec ou sans couleur pour les cellules fusionnées
    merged_cell_style = """
.{table_class} .merged-cell {{
    background-color: #fff9c4;
}}
""" if highlight_merged else ""
    
    # Générer le HTML
    html = f"""
<style>
.{table_class} {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-family: Arial, sans-serif;
}}

.{table_class} th, .{table_class} td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}}

.{table_class} th {{
    background-color: #f2f2f2;
    font-weight: bold;
}}{merged_cell_style}
</style>

<div>
    <h3>{table_title}</h3>
    <table class="{table_class}">
"""
    
    # Trier les cellules par position de grille pour un rendu correct
    sorted_cells = sorted(filled_structure, key=lambda c: (c.row_start, c.col_start))
    
    # Construire le tableau ligne par ligne
    for row in range(max_row):
        html += "        <tr>\n"
        
        for col in range(max_col):
            # Vérifier si cette position est occupée par une cellule
            cell_at_position = None
            for cell in sorted_cells:
                if (cell.row_start == row and cell.col_start == col):
                    cell_at_position = cell
                    break
            
            if cell_at_position:
                # Générer la cellule avec rowspan/colspan
                cell_class = "merged-cell" if highlight_merged and (cell_at_position.row_span > 1 or cell_at_position.col_span > 1) else ""
                rowspan_attr = f' rowspan="{cell_at_position.row_span}"' if cell_at_position.row_span > 1 else ""
                colspan_attr = f' colspan="{cell_at_position.col_span}"' if cell_at_position.col_span > 1 else ""
                
                # Calculer l'alignement automatiquement
                h_align, v_align = "left", "top"
                if cell_at_position.texts:
                    # Prendre la première box pour déterminer l'alignement
                    first_box = cell_at_position.texts[0]['box']
                    box_center_x = (first_box[0] + first_box[2]) / 2
                    box_center_y = (first_box[1] + first_box[3]) / 2
                    cell_center_x = (cell_at_position.x1 + cell_at_position.x2) / 2
                    cell_center_y = (cell_at_position.y1 + cell_at_position.y2) / 2
                    cell_width = cell_at_position.x2 - cell_at_position.x1
                    cell_height = cell_at_position.y2 - cell_at_position.y1
                    
                    # Alignement horizontal
                    if box_center_x < cell_center_x - cell_width * 0.15:
                        h_align = "left"
                    elif box_center_x > cell_center_x + cell_width * 0.15:
                        h_align = "right"
                    else:
                        h_align = "center"
                    
                    # Alignement vertical
                    if box_center_y < cell_center_y - cell_height * 0.15:
                        v_align = "top"
                    elif box_center_y > cell_center_y + cell_height * 0.15:
                        v_align = "bottom"
                    else:
                        v_align = "middle"
                
                # Style d'alignement
                align_style = f' style="text-align: {h_align}; vertical-align: {v_align};"'
                
                # Convertir les retours à la ligne en <br>
                cell_content = cell_at_position.final_text.replace('\n', '<br>')
                if not cell_content.strip():
                    cell_content = "&nbsp;"
                
                html += f'            <td class="{cell_class}"{rowspan_attr}{colspan_attr}{align_style}>{cell_content}</td>\n'
            elif not grid_occupied[row][col]:
                # Cellule vide non occupée par un span
                html += f'            <td>&nbsp;</td>\n'
        
        html += "        </tr>\n"
    
    html += """    </table>
</div>
"""
    
    return html


# === FONCTIONS UTILITAIRES ===

def load_paddleocr_data(json_file_path: str) -> Tuple[List[Dict], List[List[float]], List[str]]:
    """
    Charge les données PaddleOCR depuis un fichier JSON.
    
    Args:
        json_file_path: Chemin vers le fichier JSON de sortie PaddleOCR
        
    Returns:
        Tuple (layout_boxes, rec_boxes, rec_texts)
    """
    import json
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    layout_boxes = data.get('table_res_list', {})[0].get('cell_box_list', [])
    rec_boxes = data.get('table_res_list', {})[0]['table_ocr_pred'].get('rec_boxes', [])
    rec_texts = data.get('table_res_list', {})[0]['table_ocr_pred'].get('rec_texts', [])
    
    return layout_boxes, rec_boxes, rec_texts


def load_image(image_path: str) -> np.ndarray:
    """
    Charge une image depuis un fichier.
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Image au format numpy array
    """
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        from PIL import Image
        return np.array(Image.open(image_path))
    else:
        return cv2.imread(image_path)


# === EXEMPLE D'UTILISATION ===

"""
# Utilisation depuis un notebook:

# 1. Charger les données
layout_boxes, rec_boxes, rec_texts = load_paddleocr_data('path/to/results.json')

# 2. Traitement complet
table_structure = extract_table_structure(layout_boxes)
plot_table_structure(table_structure)

# 3. Assignation avec options
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts,
    overlap_threshold=0.1,        # Seuil normal
    force_assignment=True         # Force placement de TOUS les textes
)
plot_final_result(filled_structure)

# 4. Export HTML avec options
html_output = export_to_html(
    filled_structure, 
    "Mon Tableau",
    highlight_merged=True         # Colorie les cellules fusionnées
)
print(html_output)

# OU sans couleurs :
html_simple = export_to_html(filled_structure, highlight_merged=False)
""" 