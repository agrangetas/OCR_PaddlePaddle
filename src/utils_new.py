#!/usr/bin/env python3
"""
🔧 utils_new.py - Module simplifié pour l'analyse de tableaux OCR

Ce module contient uniquement les fonctions essentielles pour :
1. Extraire la structure d'un tableau à partir du layout
2. Visualiser cette structure
3. Placer les textes OCR dans les cellules
4. Visualiser le résultat final  
5. Exporter en HTML avec rowspan/colspan
6. Sauvegarder en fichiers HTML/Markdown avec encodage UTF-8

Fonctionnalités avancées :
- Complétion automatique des cellules vides
- Gestion intelligente de l'alignement des textes
- Échappement des caractères spéciaux (UTF-8)
- Options de diagnostic pour les textes non matchés

Utilisation simple depuis un notebook :
```python
from src.utils_new import *

# 1. Extraire la structure du tableau (avec complétion automatique des cellules vides)
table_structure = extract_table_structure(layout_boxes, fill_empty_cells=True)

# 2. Visualiser la structure (les cellules auto-générées apparaissent en gris)
plot_table_structure(table_structure)

# 3. Placer les textes OCR (avec option force pour placer tous les textes)
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts, force_assignment=True
)

# 4. Visualiser le résultat
plot_final_result(filled_structure)

# 5. Exporter en HTML (avec ou sans couleurs pour les cellules fusionnées)
html_output = export_to_html(filled_structure, highlight_merged=True)

# 6. Sauvegarder le HTML dans un fichier
save_html_to_file(html_output, "tableau.html")

# 7. Exporter en Markdown avec alignement et fusions
markdown_output = export_to_markdown(filled_structure, "Mon Tableau")
print(markdown_output)

# 8. Sauvegarder le Markdown dans un fichier
save_markdown_to_file(markdown_output, "tableau.md")

# OU sans couleurs :
html_simple = export_to_html(filled_structure, highlight_merged=False)
```

Note: Toutes les fonctions gèrent correctement l'encodage UTF-8 et échappent les caractères spéciaux.
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
        self.is_auto_filled = False  # Marque si la cellule a été auto-générée
        
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

def extract_table_structure(layout_boxes: List[Dict], tolerance: float = 10, 
                            fill_empty_cells: bool = True) -> List[TableCell]:
    """
    Extrait la structure du tableau à partir des boxes de layout.
    
    Args:
        layout_boxes: Liste des boxes de layout (format PaddleOCR)
        tolerance: Tolérance pour détecter les alignements
        fill_empty_cells: Si True, complète automatiquement les espaces vides
        
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
    
    # Compléter les espaces vides avec des cellules vides (si demandé)
    if fill_empty_cells:
        complete_cells = _fill_empty_cells(table_cells, row_lines, col_lines)
        return complete_cells
    else:
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


def _fill_empty_cells(existing_cells: List[TableCell], row_lines: List[float], col_lines: List[float]) -> List[TableCell]:
    """
    Complète les espaces vides du tableau avec des cellules vides.
    
    Args:
        existing_cells: Cellules déjà détectées
        row_lines: Lignes horizontales de la grille
        col_lines: Lignes verticales de la grille
    
    Returns:
        Liste complète des cellules (existantes + nouvelles vides)
    """
    if not existing_cells or len(row_lines) < 2 or len(col_lines) < 2:
        return existing_cells
    
    # Créer une grille pour marquer les zones occupées
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    occupied_grid = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Marquer les zones occupées par les cellules existantes
    for cell in existing_cells:
        for r in range(cell.row_start, cell.row_start + cell.row_span):
            for c in range(cell.col_start, cell.col_start + cell.col_span):
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    occupied_grid[r][c] = True
    
    # Créer les cellules vides pour combler les espaces
    complete_cells = existing_cells.copy()
    
    for row in range(n_rows):
        for col in range(n_cols):
            if not occupied_grid[row][col]:
                # Créer une cellule vide pour cette position
                x1 = col_lines[col]
                y1 = row_lines[row]
                x2 = col_lines[col + 1]
                y2 = row_lines[row + 1]
                
                empty_cell = TableCell(x1, y1, x2, y2, row, col, 1, 1)
                empty_cell.is_auto_filled = True  # Marquer comme cellule auto-générée
                complete_cells.append(empty_cell)
                
                # Marquer cette position comme occupée
                occupied_grid[row][col] = True
    
    return complete_cells


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
        # Différencier les cellules détectées des cellules auto-générées
        if getattr(cell, 'is_auto_filled', False):
            # Cellule auto-générée (vide)
            edgecolor = 'gray'
            facecolor = 'lightgray'
            textcolor = 'gray'
            alpha = 0.2
        else:
            # Cellule détectée par le layout
            edgecolor = 'blue'
            facecolor = 'lightblue'
            textcolor = 'darkblue'
            alpha = 0.3
        
        # Rectangle de la cellule
        rect = patches.Rectangle(
            (cell.x1, cell.y1), 
            cell.x2 - cell.x1, 
            cell.y2 - cell.y1,
            linewidth=2, 
            edgecolor=edgecolor, 
            facecolor=facecolor,
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Texte avec les informations de la cellule
        center_x, center_y = cell.center()
        info_text = f"({cell.row_start},{cell.col_start})\n{cell.row_span}×{cell.col_span}"
        ax.text(center_x, center_y, info_text, 
                ha='center', va='center', 
                fontsize=10, color=textcolor, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Statistiques des cellules
    detected_cells = len([cell for cell in table_structure if not getattr(cell, 'is_auto_filled', False)])
    auto_filled_cells = len([cell for cell in table_structure if getattr(cell, 'is_auto_filled', False)])
    
    title = f'Structure du tableau: {detected_cells} détectées + {auto_filled_cells} complétées = {len(table_structure)} cellules'
    ax.set_title(title)
    ax.set_xlabel('Position X (pixels)')
    ax.set_ylabel('Position Y (pixels)')
    ax.grid(True, alpha=0.3)
    
    # Légende simple
    if auto_filled_cells > 0:
        legend_text = "🔵 Cellules détectées par layout\n⚫ Cellules auto-générées (vides)"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
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

def _escape_html(text: str) -> str:
    """
    Échappe les caractères HTML spéciaux pour éviter les problèmes d'encodage.
    
    Args:
        text: Texte à échapper
        
    Returns:
        Texte avec caractères HTML échappés
    """
    if not text:
        return ""
    
    # Échapper les caractères HTML de base
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    return text


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
    
    # Générer le HTML avec encodage UTF-8
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(table_title)}</title>
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
</head>
<body>
    <div>
        <h3>{_escape_html(table_title)}</h3>
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
                
                # Échapper et convertir les retours à la ligne en <br>
                cell_content = _escape_html(cell_at_position.final_text).replace('\n', '<br>')
                if not cell_content.strip():
                    cell_content = "&nbsp;"
                
                html += f'            <td class="{cell_class}"{rowspan_attr}{colspan_attr}{align_style}>{cell_content}</td>\n'
            elif not grid_occupied[row][col]:
                # Cellule vide non occupée par un span
                html += f'            <td>&nbsp;</td>\n'
        
        html += "        </tr>\n"
    
    html += """        </table>
    </div>
</body>
</html>
"""
    
    return html


def save_html_to_file(html_content: str, filename: str = "tableau.html") -> None:
    """
    Sauvegarde le contenu HTML dans un fichier avec encodage UTF-8.
    
    Args:
        html_content: Contenu HTML à sauvegarder
        filename: Nom du fichier (avec extension .html)
    """
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(html_content)
        print(f"✅ Tableau HTML sauvegardé dans {filename}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")


def _escape_markdown(text: str) -> str:
    """
    Échappe les caractères Markdown spéciaux.
    
    Args:
        text: Texte à échapper
        
    Returns:
        Texte avec caractères Markdown échappés
    """
    if not text:
        return ""
    
    # Échapper les caractères Markdown de base
    text = text.replace('\\', '\\\\')
    text = text.replace('|', '\\|')
    text = text.replace('*', '\\*')
    text = text.replace('_', '\\_')
    text = text.replace('`', '\\`')
    text = text.replace('#', '\\#')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace('(', '\\(')
    text = text.replace(')', '\\)')
    
    return text


def export_to_markdown(filled_structure: List[TableCell], 
                      table_title: str = "Tableau OCR") -> str:
    """
    Exporte la structure en Markdown avec gestion des fusions.
    
    Args:
        filled_structure: Structure avec textes assignés
        table_title: Titre du tableau
        
    Returns:
        Code Markdown du tableau
    """
    if not filled_structure:
        return f"# {table_title}\n\n*Tableau vide*"
    
    # Déterminer la taille de la grille
    max_row = max(cell.row_start + cell.row_span for cell in filled_structure)
    max_col = max(cell.col_start + cell.col_span for cell in filled_structure)
    
    # Créer une grille de contenu
    grid_content = [["" for _ in range(max_col)] for _ in range(max_row)]
    grid_alignment = [["left" for _ in range(max_col)] for _ in range(max_row)]
    
    # Remplir la grille
    for cell in filled_structure:
        content = _escape_markdown(cell.final_text.strip()) if cell.final_text else ""
        
        # Déterminer l'alignement
        alignment = "left"
        if cell.texts:
            first_box = cell.texts[0]['box']
            box_center_x = (first_box[0] + first_box[2]) / 2
            cell_center_x = (cell.x1 + cell.x2) / 2
            cell_width = cell.x2 - cell.x1
            
            if box_center_x < cell_center_x - cell_width * 0.15:
                alignment = "left"
            elif box_center_x > cell_center_x + cell_width * 0.15:
                alignment = "right"
            else:
                alignment = "center"
        
        # Placer le contenu dans la grille
        for r in range(cell.row_start, cell.row_start + cell.row_span):
            for c in range(cell.col_start, cell.col_start + cell.col_span):
                if r < max_row and c < max_col:
                    if r == cell.row_start and c == cell.col_start:
                        # Cellule principale : contenu complet
                        if cell.row_span > 1 or cell.col_span > 1:
                            grid_content[r][c] = f"{content} `[{cell.row_span}×{cell.col_span}]`"
                        else:
                            grid_content[r][c] = content
                        grid_alignment[r][c] = alignment
                    else:
                        # Cellule fusionnée : marquer comme occupée
                        grid_content[r][c] = "~"  # Marque de fusion
                        grid_alignment[r][c] = alignment
    
    # Générer le Markdown
    markdown = f"# {_escape_markdown(table_title)}\n\n"
    
    # En-tête du tableau
    header_line = "|"
    separator_line = "|"
    for col in range(max_col):
        header_line += f" Col {col+1} |"
        # Alignement dans le séparateur
        if grid_alignment[0][col] == "center":
            separator_line += ":---:|"
        elif grid_alignment[0][col] == "right":
            separator_line += "----:|"
        else:
            separator_line += "-----|"
    
    markdown += header_line + "\n"
    markdown += separator_line + "\n"
    
    # Lignes du tableau
    for row in range(max_row):
        line = "|"
        for col in range(max_col):
            content = grid_content[row][col]
            if content == "~":
                content = ""  # Cellule fusionnée = vide
            elif not content:
                content = " "  # Cellule vide
            line += f" {content} |"
        markdown += line + "\n"
    
    # Note sur les fusions
    has_merged = any(cell.row_span > 1 or cell.col_span > 1 for cell in filled_structure)
    if has_merged:
        markdown += "\n*Note: Les cellules fusionnées sont marquées avec `[lignes×colonnes]`*\n"
    
    return markdown


def save_markdown_to_file(markdown_content: str, filename: str = "tableau.md") -> None:
    """
    Sauvegarde le contenu Markdown dans un fichier avec encodage UTF-8.
    
    Args:
        markdown_content: Contenu Markdown à sauvegarder
        filename: Nom du fichier (avec extension .md)
    """
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(markdown_content)
        print(f"✅ Tableau Markdown sauvegardé dans {filename}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")


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

# 2. Traitement complet avec options
table_structure = extract_table_structure(
    layout_boxes,
    tolerance=10,               # Tolérance pour alignement
    fill_empty_cells=True       # Compléter automatiquement les cellules vides
)
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

# 5. Sauvegarder le HTML dans un fichier
save_html_to_file(html_output, "mon_tableau.html")

# 6. Exporter en Markdown
markdown_output = export_to_markdown(filled_structure, "Mon Tableau")
print(markdown_output)

# 7. Sauvegarder le Markdown dans un fichier
save_markdown_to_file(markdown_output, "mon_tableau.md")

# OU sans couleurs :
html_simple = export_to_html(filled_structure, highlight_merged=False)

# OU sans complétion automatique :
table_structure_basic = extract_table_structure(layout_boxes, fill_empty_cells=False)
""" 