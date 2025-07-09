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
7. Nettoyer et simplifier la structure pour une grille cohérente
8. Corriger automatiquement les chevauchements de cellules

Fonctionnalités avancées :
- Complétion automatique des cellules vides
- Extension des cellules pour combler les espaces
- Nettoyage et simplification de la structure (alignement + fusion)
- Correction automatique des chevauchements et duplicatas
- Espacement intelligent adaptatif basé sur les distances réelles entre textes
- Gestion intelligente de l'alignement des textes
- Échappement des caractères spéciaux (UTF-8)
- Options de diagnostic pour les textes non matchés

Utilisation simple depuis un notebook :
```python
from src.utils_new import *

# 1. Extraire la structure du tableau (avec complétion automatique des cellules vides)
table_structure = extract_table_structure(
    layout_boxes, 
    fill_empty_cells=True,     # Compléter les cellules vides
    extend_cells=True          # Étendre les cellules pour combler les espaces
)

# 2. Visualiser la structure (avec cellules auto-générées en gris)
plot_table_structure(table_structure)

# 3. Placer les textes OCR (avec option force pour placer tous les textes)
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts, 
    overlap_threshold=0.5,
    force_assignment=True,     # Forcer l'assignment même sans recouvrement
    clean_structure=False,     # Désactivé par défaut pour éviter les problèmes
    auto_correct_overlaps=True,# Corriger automatiquement les chevauchements
    smart_spacing=True,        # Espacement intelligent basé sur les distances réelles
    verbose_overlaps=False     # Masquer les détails de détection des chevauchements (par défaut)
)

# 4. Visualiser le résultat final
plot_final_result(filled_structure)

# 5. Exporter en HTML avec couleurs
html_output = export_to_html(filled_structure, highlight_merged=True)
save_html_to_file(html_output, "tableau.html")

# 6. Exporter en Markdown
markdown_output = export_to_markdown(filled_structure, "Mon Tableau")
save_markdown_to_file(markdown_output, "tableau.md")

# OU sans couleurs :
html_simple = export_to_html(filled_structure, highlight_merged=False)
```

Note: Toutes les fonctions gèrent correctement l'encodage UTF-8 et échappent les caractères spéciaux.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import cv2
from PIL import Image, ImageDraw, ImageFont
from enum import Enum


# === STRUCTURE DE DONNÉES ===

class OverlapType(Enum):
    """Types de chevauchements possibles"""
    DUPLICATE = "DUPLICATE"  # Cellules identiques
    INCLUSION = "INCLUSION"  # Une cellule dans une autre
    SAME_GRID = "SAME_GRID"  # Même position de grille
    PARTIAL = "PARTIAL"      # Chevauchement partiel

class Overlap:
    """Représente un chevauchement entre deux cellules"""
    def __init__(self, cell1: 'TableCell', cell2: 'TableCell', 
                 cell1_idx: int, cell2_idx: int,
                 overlap_type: OverlapType, 
                 percentage1: float, percentage2: float,
                 intersection_area: float):
        self.cell1 = cell1
        self.cell2 = cell2
        self.cell1_idx = cell1_idx
        self.cell2_idx = cell2_idx
        self.overlap_type = overlap_type
        self.percentage1 = percentage1  # Pourcentage de cell1 qui chevauche
        self.percentage2 = percentage2  # Pourcentage de cell2 qui chevauche
        self.intersection_area = intersection_area
        
        # Calculer la sévérité (0-100, plus élevé = plus urgent)
        self.severity = self._calculate_severity()
    
    def _calculate_severity(self) -> float:
        """Calcule la sévérité du chevauchement pour prioriser les corrections"""
        if self.overlap_type == OverlapType.DUPLICATE:
            return 100.0  # Priorité maximale
        elif self.overlap_type == OverlapType.INCLUSION:
            return 90.0 + max(self.percentage1, self.percentage2) / 10
        elif self.overlap_type == OverlapType.SAME_GRID:
            return 80.0 + max(self.percentage1, self.percentage2) / 10
        else:  # PARTIAL
            return max(self.percentage1, self.percentage2)

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
        self.smart_spacing = True  # Stocker le paramètre dans la cellule
        
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
    
    def add_text(self, text: str, x1: float, y1: float, x2: float, y2: float) -> None:
        """Ajoute un texte OCR à la cellule"""
        self.texts.append({
            'text': text.strip(),
            'box': [x1, y1, x2, y2],
            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
        })
    
    def finalize_text(self) -> None:
        """Finalise le texte de la cellule en ordonnant spatialement les textes avec espacement intelligent"""
        if self.texts:
            # Vérifier si l'espacement intelligent est activé
            use_smart_spacing = getattr(self, 'smart_spacing', True)
            
            if use_smart_spacing:
                # Calculer les dimensions de la cellule
                cell_width = self.x2 - self.x1
                cell_height = self.y2 - self.y1
                
                # Utiliser l'espacement intelligent basé sur les dimensions de la cellule
                self.final_text = _order_texts_spatially_with_cell_context(self.texts, cell_width, cell_height)
            else:
                # Utiliser l'espacement basique
                self.final_text = _order_texts_spatially(self.texts)
        else:
            self.final_text = ""
    
    def is_empty(self) -> bool:
        """Vérifie si la cellule est vide (pas de texte)"""
        return not self.final_text.strip()
    
    def merge_with(self, other: 'TableCell') -> None:
        """Fusionne cette cellule avec une autre cellule"""
        # Étendre les coordonnées pour englober les deux cellules
        self.x1 = min(self.x1, other.x1)
        self.y1 = min(self.y1, other.y1)
        self.x2 = max(self.x2, other.x2)
        self.y2 = max(self.y2, other.y2)
        
        # Combiner les textes
        self.texts.extend(other.texts)
        
        # Recalculer le texte final
        self.finalize_text()
        
        # Garder le statut auto-filled seulement si les deux cellules le sont
        self.is_auto_filled = self.is_auto_filled and other.is_auto_filled


def _calculate_overlap(x1: float, y1: float, x2: float, y2: float, 
                      cell_x1: float, cell_y1: float, cell_x2: float, cell_y2: float) -> float:
    """Calcule le pourcentage de recouvrement entre une box OCR et une cellule"""
    # Intersection
    inter_x1 = max(x1, cell_x1)
    inter_y1 = max(y1, cell_y1)
    inter_x2 = min(x2, cell_x2)
    inter_y2 = min(y2, cell_y2)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box_area = (x2 - x1) * (y2 - y1)
    
    return inter_area / box_area if box_area > 0 else 0.0


# === FONCTION 1 : EXTRAIRE LA STRUCTURE DU TABLEAU ===

def extract_table_structure(layout_boxes: List[Dict], tolerance: float = 10, 
                            fill_empty_cells: bool = True,
                            extend_cells: bool = False) -> List[TableCell]:
    """
    Extrait la structure du tableau à partir des boxes de layout.
    
    Args:
        layout_boxes: Liste des boxes de layout (format PaddleOCR)
        tolerance: Tolérance pour détecter les alignements
        fill_empty_cells: Si True, complète automatiquement les espaces vides
        extend_cells: Si True, étend les cellules pour combler les espaces
        
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
    else:
        complete_cells = table_cells
    
    # Étendre les cellules pour combler les espaces (si demandé)
    if extend_cells:
        complete_cells = extend_cells_to_fill_gaps(complete_cells, tolerance=2)
    
    return complete_cells


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


def extend_cells_to_fill_gaps(table_cells: List[TableCell], tolerance: float = 2) -> List[TableCell]:
    """
    Étend UNIQUEMENT les cellules vides pour combler les espaces.
    Ne touche pas aux cellules avec du texte pour éviter les décalages.
    
    Args:
        table_cells: Liste des cellules du tableau
        tolerance: Tolérance pour considérer deux bords comme alignés
        
    Returns:
        Liste des cellules avec espaces comblés
    """
    if not table_cells:
        return table_cells
    
    # Copier les cellules pour ne pas modifier les originales
    extended_cells = []
    for cell in table_cells:
        new_cell = TableCell(cell.x1, cell.y1, cell.x2, cell.y2, 
                           cell.row_start, cell.col_start, 
                           cell.row_span, cell.col_span)
        new_cell.texts = cell.texts.copy()
        new_cell.final_text = cell.final_text
        new_cell.is_auto_filled = getattr(cell, 'is_auto_filled', False)
        extended_cells.append(new_cell)
    
    # Déterminer les limites du tableau
    min_x = min(cell.x1 for cell in extended_cells)
    max_x = max(cell.x2 for cell in extended_cells)
    min_y = min(cell.y1 for cell in extended_cells)
    max_y = max(cell.y2 for cell in extended_cells)
    
    # Séparer cellules avec texte et cellules vides
    filled_cells = [cell for cell in extended_cells if cell.final_text.strip()]
    empty_cells = [cell for cell in extended_cells if not cell.final_text.strip()]
    
    print(f"🔧 Extension : {len(filled_cells)} cellules avec texte préservées, {len(empty_cells)} cellules vides étendues")
    
    # ÉTENDRE UNIQUEMENT LES CELLULES VIDES
    for cell in empty_cells:
        
        # 1. ÉTENDRE VERS LE HAUT (si pas de cellule au-dessus)
        cells_above = [c for c in extended_cells if c != cell and 
                      c.x1 < cell.x2 and c.x2 > cell.x1 and  # chevauchement horizontal
                      c.y2 <= cell.y1 + tolerance]  # au-dessus ou juste au niveau
        
        if cells_above:
            # Prendre le bas de la cellule la plus proche au-dessus
            closest_bottom = max(c.y2 for c in cells_above)
            cell.y1 = closest_bottom
        else:
            # Étendre jusqu'au haut du tableau
            cell.y1 = min_y
        
        # 2. ÉTENDRE VERS LA DROITE (si pas de cellule à droite)
        cells_right = [c for c in extended_cells if c != cell and 
                      c.y1 < cell.y2 and c.y2 > cell.y1 and  # chevauchement vertical
                      c.x1 >= cell.x2 - tolerance]  # à droite ou juste au niveau
        
        if cells_right:
            # Prendre le gauche de la cellule la plus proche à droite
            closest_left = min(c.x1 for c in cells_right)
            cell.x2 = closest_left
        else:
            # Étendre jusqu'au bord droit du tableau
            cell.x2 = max_x
        
        # 3. LIMITER VERS LE BAS (nettoyer les chevauchements)
        cells_below = [c for c in extended_cells if c != cell and 
                      c.x1 < cell.x2 and c.x2 > cell.x1 and  # chevauchement horizontal
                      c.y1 >= cell.y2 - tolerance and  # en dessous ou juste au niveau
                      c.y1 < cell.y2 + tolerance]  # pas trop loin
        
        if cells_below:
            # Limiter le bas à la cellule la plus proche en dessous
            closest_top = min(c.y1 for c in cells_below)
            if closest_top < cell.y2:  # Seulement si on doit réduire
                cell.y2 = closest_top
    
    return extended_cells


def clean_table_structure(table_cells: List[TableCell], tolerance: float = 5, verbose_overlaps: bool = False) -> List[TableCell]:
    """
    Nettoie légèrement la structure du tableau pour avoir une grille cohérente.
    CONSERVATEUR : ajuste les bords à la marge, fusionne UNIQUEMENT les cellules vides.
    
    Args:
        table_cells: Liste des cellules du tableau
        tolerance: Tolérance pour considérer des bords comme alignés (petite valeur!)
        verbose_overlaps: Si True, affiche les détails de détection des chevauchements
        
    Returns:
        Liste des cellules légèrement nettoyées
    """
    if not table_cells:
        return table_cells
    
    # Copier les cellules
    clean_cells = []
    for cell in table_cells:
        new_cell = TableCell(cell.x1, cell.y1, cell.x2, cell.y2, 
                           cell.row_start, cell.col_start, 
                           cell.row_span, cell.col_span)
        new_cell.texts = cell.texts.copy()
        new_cell.final_text = cell.final_text
        new_cell.is_auto_filled = getattr(cell, 'is_auto_filled', False)
        clean_cells.append(new_cell)
    
    # 1. ALIGNEMENT LÉGER DES BORDS (seulement quelques pixels)
    clean_cells = _align_cell_borders_conservative(clean_cells, tolerance)
    
    # 2. FUSIONNER UNIQUEMENT LES CELLULES VIDES ADJACENTES
    clean_cells = _merge_only_empty_cells(clean_cells, tolerance)
    
    # 3. RECALCULER LES POSITIONS DE GRILLE (sans modifier les cellules avec texte)
    clean_cells = _recalculate_grid_positions_conservative(clean_cells, tolerance)
    
    # 4. DÉTECTER LES CHEVAUCHEMENTS APRÈS NETTOYAGE
    if verbose_overlaps:
        print("🔍 Détection des chevauchements après nettoyage...")
    overlaps = _detect_overlapping_cells(clean_cells, verbose=verbose_overlaps)
    
    return clean_cells


def _align_cell_borders_conservative(cells: List[TableCell], tolerance: float) -> List[TableCell]:
    """Aligne LÉGÈREMENT les bords des cellules - seulement ajustements mineurs."""
    
    # Séparer cellules avec texte et cellules vides
    filled_cells = [cell for cell in cells if cell.final_text.strip()]
    empty_cells = [cell for cell in cells if not cell.final_text.strip()]
    
    # Pour les cellules avec texte : alignement très conservateur
    if filled_cells:
        # Collecter les positions des cellules AVEC TEXTE uniquement
        x_positions = []
        y_positions = []
        for cell in filled_cells:
            x_positions.extend([cell.x1, cell.x2])
            y_positions.extend([cell.y1, cell.y2])
        
        # Grouper avec une tolérance plus petite pour les cellules pleines
        unique_x = _group_similar_values(x_positions, tolerance)
        unique_y = _group_similar_values(y_positions, tolerance)
        
        # Ajustement LÉGER pour les cellules avec texte
        for cell in filled_cells:
            old_x1, old_x2 = cell.x1, cell.x2
            old_y1, old_y2 = cell.y1, cell.y2
            
            new_x1 = _find_closest_value(cell.x1, unique_x)
            new_x2 = _find_closest_value(cell.x2, unique_x)
            new_y1 = _find_closest_value(cell.y1, unique_y)
            new_y2 = _find_closest_value(cell.y2, unique_y)
            
            # Appliquer SEULEMENT si le changement est vraiment petit
            if abs(new_x1 - old_x1) <= tolerance:
                cell.x1 = new_x1
            if abs(new_x2 - old_x2) <= tolerance:
                cell.x2 = new_x2
            if abs(new_y1 - old_y1) <= tolerance:
                cell.y1 = new_y1
            if abs(new_y2 - old_y2) <= tolerance:
                cell.y2 = new_y2
    
    # Pour les cellules vides : alignement plus libre
    if empty_cells and filled_cells:
        # Aligner les cellules vides sur les cellules pleines
        for empty_cell in empty_cells:
            # Chercher la meilleure position par rapport aux cellules pleines
            for filled_cell in filled_cells:
                # Alignement horizontal si proche
                if abs(empty_cell.x1 - filled_cell.x1) <= tolerance:
                    empty_cell.x1 = filled_cell.x1
                if abs(empty_cell.x2 - filled_cell.x2) <= tolerance:
                    empty_cell.x2 = filled_cell.x2
                    
                # Alignement vertical si proche
                if abs(empty_cell.y1 - filled_cell.y1) <= tolerance:
                    empty_cell.y1 = filled_cell.y1
                if abs(empty_cell.y2 - filled_cell.y2) <= tolerance:
                    empty_cell.y2 = filled_cell.y2
    
    return cells


def _merge_only_empty_cells(cells: List[TableCell], tolerance: float) -> List[TableCell]:
    """Fusionne UNIQUEMENT les cellules vides adjacentes - ne touche pas aux cellules avec texte."""
    
    # Séparer cellules vides et pleines
    empty_cells = [cell for cell in cells if not cell.final_text.strip()]
    filled_cells = [cell for cell in cells if cell.final_text.strip()]
    
    if not empty_cells:
        return cells
    
    print(f"🔄 Fusion de {len(empty_cells)} cellules vides (préservation de {len(filled_cells)} cellules avec texte)")
    
    # Fusionner SEULEMENT les cellules vides
    merged_empty = _merge_horizontally(empty_cells, tolerance)
    merged_empty = _merge_vertically(merged_empty, tolerance)
    
    print(f"✅ Résultat : {len(merged_empty)} cellules vides après fusion")
    
    return filled_cells + merged_empty


def _recalculate_grid_positions_conservative(cells: List[TableCell], tolerance: float) -> List[TableCell]:
    """Recalcule les positions de grille de manière conservative - préserve la structure existante."""
    if not cells:
        return cells
    
    # Utiliser TOUTES les cellules pour recréer la grille
    x_positions = []
    y_positions = []
    for cell in cells:
        x_positions.extend([cell.x1, cell.x2])
        y_positions.extend([cell.y1, cell.y2])
    
    col_lines = sorted(set(_group_similar_values(x_positions, tolerance)))
    row_lines = sorted(set(_group_similar_values(y_positions, tolerance)))
    
    # Recalculer les positions de grille pour TOUTES les cellules
    for cell in cells:
        # Trouver les indices de grille les plus proches
        try:
            row_start = row_lines.index(_find_closest_value(cell.y1, row_lines))
            row_end = row_lines.index(_find_closest_value(cell.y2, row_lines))
            col_start = col_lines.index(_find_closest_value(cell.x1, col_lines))
            col_end = col_lines.index(_find_closest_value(cell.x2, col_lines))
            
            # Mettre à jour les propriétés de grille
            cell.row_start = row_start
            cell.col_start = col_start
            cell.row_span = max(1, row_end - row_start)  # Au moins 1
            cell.col_span = max(1, col_end - col_start)  # Au moins 1
        except (ValueError, IndexError):
            # En cas de problème, garder les valeurs actuelles
            pass
    
    return cells


def _group_similar_values(values: list, tolerance: float) -> list:
    """Groupe les valeurs similaires et retourne les moyennes."""
    if not values:
        return []
    
    sorted_values = sorted(set(values))
    groups = []
    current_group = [sorted_values[0]]
    
    for value in sorted_values[1:]:
        if value - current_group[-1] <= tolerance:
            current_group.append(value)
        else:
            # Finir le groupe actuel et en commencer un nouveau
            groups.append(sum(current_group) / len(current_group))
            current_group = [value]
    
    # Ajouter le dernier groupe
    groups.append(sum(current_group) / len(current_group))
    
    return groups


def _find_closest_value(target: float, values: list) -> float:
    """Trouve la valeur la plus proche dans une liste."""
    return min(values, key=lambda x: abs(x - target))


def _merge_horizontally(cells: List[TableCell], tolerance: float) -> List[TableCell]:
    """Fusionne les cellules vides adjacentes horizontalement."""
    if len(cells) <= 1:
        return cells
    
    # Trier par ligne puis par colonne
    sorted_cells = sorted(cells, key=lambda c: (c.y1, c.x1))
    merged = []
    
    i = 0
    while i < len(sorted_cells):
        current = sorted_cells[i]
        
        # Chercher les cellules à fusionner à droite
        j = i + 1
        while j < len(sorted_cells):
            next_cell = sorted_cells[j]
            
            # Vérifier si on peut fusionner (même ligne, côte à côte)
            if (abs(current.y1 - next_cell.y1) <= tolerance and 
                abs(current.y2 - next_cell.y2) <= tolerance and
                abs(current.x2 - next_cell.x1) <= tolerance):
                
                # Fusionner : étendre current vers la droite
                current.x2 = next_cell.x2
                # Réinitialiser les propriétés de grille - elles seront recalculées
                current.row_span = 1
                current.col_span = 1
                j += 1
            else:
                break
        
        merged.append(current)
        i = j
    
    return merged


def _merge_vertically(cells: List[TableCell], tolerance: float) -> List[TableCell]:
    """Fusionne les cellules vides adjacentes verticalement."""
    if len(cells) <= 1:
        return cells
    
    # Trier par colonne puis par ligne
    sorted_cells = sorted(cells, key=lambda c: (c.x1, c.y1))
    merged = []
    
    i = 0
    while i < len(sorted_cells):
        current = sorted_cells[i]
        
        # Chercher les cellules à fusionner vers le bas
        j = i + 1
        while j < len(sorted_cells):
            next_cell = sorted_cells[j]
            
            # Vérifier si on peut fusionner (même colonne, empilées)
            if (abs(current.x1 - next_cell.x1) <= tolerance and 
                abs(current.x2 - next_cell.x2) <= tolerance and
                abs(current.y2 - next_cell.y1) <= tolerance):
                
                # Fusionner : étendre current vers le bas
                current.y2 = next_cell.y2
                # Réinitialiser les propriétés de grille - elles seront recalculées
                current.row_span = 1
                current.col_span = 1
                j += 1
            else:
                break
        
        merged.append(current)
        i = j
    
    return merged


def _detect_overlapping_cells(table_cells: List[TableCell], verbose: bool = False) -> List[Overlap]:
    """
    Détecte les cellules qui se chevauchent et retourne les informations détaillées.
    
    Args:
        table_cells: Liste des cellules du tableau
        verbose: Si True, affiche les détails de détection des chevauchements
        
    Returns:
        Liste des chevauchements détectés
    """
    if verbose:
        print("🔍 Détection des chevauchements entre cellules:")
    
    overlaps = []
    
    for i in range(len(table_cells)):
        for j in range(i + 1, len(table_cells)):
            cell1 = table_cells[i]
            cell2 = table_cells[j]
            
            # Vérifier si les cellules se chevauchent
            if (cell1.x1 < cell2.x2 and cell1.x2 > cell2.x1 and
                cell1.y1 < cell2.y2 and cell1.y2 > cell2.y1):
                
                # Calculer la zone d'intersection
                inter_x1 = max(cell1.x1, cell2.x1)
                inter_y1 = max(cell1.y1, cell2.y1)
                inter_x2 = min(cell1.x2, cell2.x2)
                inter_y2 = min(cell1.y2, cell2.y2)
                
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                cell1_area = cell1.area()
                cell2_area = cell2.area()
                
                overlap_pct_cell1 = (inter_area / cell1_area) * 100 if cell1_area > 0 else 0
                overlap_pct_cell2 = (inter_area / cell2_area) * 100 if cell2_area > 0 else 0
                
                # Déterminer le type de chevauchement
                overlap_type = _classify_overlap(cell1, cell2, overlap_pct_cell1, overlap_pct_cell2)
                
                # Créer l'objet Overlap
                overlap = Overlap(cell1, cell2, i, j, overlap_type, 
                                overlap_pct_cell1, overlap_pct_cell2, inter_area)
                overlaps.append(overlap)
                
                # Afficher les informations seulement si verbose=True
                if verbose:
                    print(f"  ⚠️  Cellule #{i} et #{j} se chevauchent ({overlap_type.value}):")
                    print(f"      Cellule #{i}: ({cell1.x1:.1f},{cell1.y1:.1f}→{cell1.x2:.1f},{cell1.y2:.1f}) "
                          f"grille({cell1.row_start},{cell1.col_start}) span({cell1.row_span}×{cell1.col_span})")
                    print(f"      Cellule #{j}: ({cell2.x1:.1f},{cell2.y1:.1f}→{cell2.x2:.1f},{cell2.y2:.1f}) "
                          f"grille({cell2.row_start},{cell2.col_start}) span({cell2.row_span}×{cell2.col_span})")
                    print(f"      Zone d'intersection: ({inter_x1:.1f},{inter_y1:.1f}→{inter_x2:.1f},{inter_y2:.1f})")
                    print(f"      Pourcentage de chevauchement: {overlap_pct_cell1:.1f}% pour #{i}, {overlap_pct_cell2:.1f}% pour #{j}")
                    print(f"      Sévérité: {overlap.severity:.1f}")
                    print()
    
    if verbose:
        if not overlaps:
            print("  ✅ Aucun chevauchement détecté")
        else:
            print(f"  📊 {len(overlaps)} chevauchements détectés")
        print()
    
    return overlaps


def _classify_overlap(cell1: TableCell, cell2: TableCell, pct1: float, pct2: float) -> OverlapType:
    """Classifie le type de chevauchement entre deux cellules"""
    # Vérifier si les cellules sont quasi-identiques
    if (abs(cell1.x1 - cell2.x1) < 5 and abs(cell1.y1 - cell2.y1) < 5 and
        abs(cell1.x2 - cell2.x2) < 5 and abs(cell1.y2 - cell2.y2) < 5):
        return OverlapType.DUPLICATE
    
    # Vérifier si même position de grille
    if (cell1.row_start == cell2.row_start and cell1.col_start == cell2.col_start):
        return OverlapType.SAME_GRID
    
    # Vérifier si inclusion (une cellule dans l'autre)
    if pct1 >= 95 or pct2 >= 95:
        return OverlapType.INCLUSION
    
    # Sinon, c'est un chevauchement partiel
    return OverlapType.PARTIAL


def _shrink_empty_cell_to_avoid_overlap(cells: List[TableCell], overlap: Overlap, to_remove: set) -> bool:
    """
    Réduit une cellule vide pour éviter le chevauchement avec une cellule pleine.
    
    Args:
        cells: Liste des cellules
        overlap: Informations sur le chevauchement
        to_remove: Set des cellules à supprimer
        
    Returns:
        True si la correction a été appliquée, False sinon
    """
    cell1, cell2 = overlap.cell1, overlap.cell2
    
    # Déterminer quelle cellule est vide et laquelle est pleine
    empty_cell = None
    full_cell = None
    
    if cell1.is_empty() and not cell2.is_empty():
        empty_cell = cell1
        full_cell = cell2
        empty_idx = overlap.cell1_idx
        full_idx = overlap.cell2_idx
    elif cell2.is_empty() and not cell1.is_empty():
        empty_cell = cell2
        full_cell = cell1
        empty_idx = overlap.cell2_idx
        full_idx = overlap.cell1_idx
    else:
        # Les deux cellules sont pleines ou vides -> pas applicable
        return False
    
    # Calculer la zone d'intersection
    inter_x1 = max(empty_cell.x1, full_cell.x1)
    inter_y1 = max(empty_cell.y1, full_cell.y1)
    inter_x2 = min(empty_cell.x2, full_cell.x2)
    inter_y2 = min(empty_cell.y2, full_cell.y2)
    
    # Déterminer comment réduire la cellule vide
    # Priorité : réduire du côté où le chevauchement est le plus important
    
    # Distances de chevauchement dans chaque direction
    overlap_left = inter_x2 - empty_cell.x1    # Chevauchement vers la gauche
    overlap_right = empty_cell.x2 - inter_x1   # Chevauchement vers la droite
    overlap_top = inter_y2 - empty_cell.y1     # Chevauchement vers le haut
    overlap_bottom = empty_cell.y2 - inter_y1  # Chevauchement vers le bas
    
    # Trouver la direction avec le plus petit ajustement nécessaire
    adjustments = []
    
    # Peut-on réduire de la gauche ?
    if empty_cell.x1 < full_cell.x1:
        new_x1 = full_cell.x1
        if new_x1 < empty_cell.x2:  # Vérifier que la cellule reste valide
            adjustments.append(('left', overlap_left, new_x1, None, None, None))
    
    # Peut-on réduire de la droite ?
    if empty_cell.x2 > full_cell.x2:
        new_x2 = full_cell.x2
        if new_x2 > empty_cell.x1:  # Vérifier que la cellule reste valide
            adjustments.append(('right', overlap_right, None, new_x2, None, None))
    
    # Peut-on réduire du haut ?
    if empty_cell.y1 < full_cell.y1:
        new_y1 = full_cell.y1
        if new_y1 < empty_cell.y2:  # Vérifier que la cellule reste valide
            adjustments.append(('top', overlap_top, None, None, new_y1, None))
    
    # Peut-on réduire du bas ?
    if empty_cell.y2 > full_cell.y2:
        new_y2 = full_cell.y2
        if new_y2 > empty_cell.y1:  # Vérifier que la cellule reste valide
            adjustments.append(('bottom', overlap_bottom, None, None, None, new_y2))
    
    if not adjustments:
        return False
    
    # Choisir l'ajustement qui préserve le mieux la cellule (plus petit changement)
    adjustments.sort(key=lambda x: x[1])  # Trier par taille d'ajustement
    direction, _, new_x1, new_x2, new_y1, new_y2 = adjustments[0]
    
    # Appliquer l'ajustement
    if new_x1 is not None:
        empty_cell.x1 = new_x1
    if new_x2 is not None:
        empty_cell.x2 = new_x2
    if new_y1 is not None:
        empty_cell.y1 = new_y1
    if new_y2 is not None:
        empty_cell.y2 = new_y2
    
    print(f"    ✅ Cellule vide #{empty_idx} réduite du côté {direction} pour éviter le chevauchement avec #{full_idx}")
    return True


def _auto_correct_overlaps(table_cells: List[TableCell], verbose_overlaps: bool = False) -> List[TableCell]:
    """
    Corrige automatiquement les chevauchements détectés de manière récursive.
    
    Args:
        table_cells: Liste des cellules du tableau
        verbose_overlaps: Si True, affiche les détails de détection des chevauchements
        
    Returns:
        Liste des cellules corrigées
    """
    print("🔄 Correction automatique des chevauchements (récursive)...")
    
    corrected_cells = table_cells.copy()
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"  🔄 Itération {iteration + 1}/{max_iterations}")
        
        # Détecter les chevauchements
        overlaps = _detect_overlapping_cells(corrected_cells, verbose=verbose_overlaps)
        
        if not overlaps:
            print(f"  ✅ Aucun chevauchement détecté - correction terminée après {iteration + 1} itération(s)")
            break
        
        print(f"  📊 {len(overlaps)} chevauchements détectés à corriger")
        
        # Trier par sévérité (plus urgent en premier)
        overlaps.sort(key=lambda x: x.severity, reverse=True)
        
        # Appliquer les corrections
        cells_to_remove = set()
        corrections_applied = 0
        
        for overlap in overlaps:
            # Vérifier si les cellules existent encore (pas supprimées par correction précédente)
            if overlap.cell1_idx in cells_to_remove or overlap.cell2_idx in cells_to_remove:
                continue
            
            print(f"    🔧 Correction {overlap.overlap_type.value}: cellules #{overlap.cell1_idx} et #{overlap.cell2_idx}")
            
            # STRATÉGIE 1 : Réduire la cellule vide (si applicable)
            if _shrink_empty_cell_to_avoid_overlap(corrected_cells, overlap, cells_to_remove):
                corrections_applied += 1
                continue
            
            # STRATÉGIE 2 : Corrections classiques
            if overlap.overlap_type == OverlapType.DUPLICATE:
                # Fusionner les cellules identiques
                _merge_duplicate_cells(corrected_cells, overlap, cells_to_remove)
                corrections_applied += 1
            
            elif overlap.overlap_type == OverlapType.INCLUSION:
                # Absorber la cellule incluse
                _absorb_included_cell(corrected_cells, overlap, cells_to_remove)
                corrections_applied += 1
            
            elif overlap.overlap_type == OverlapType.SAME_GRID:
                # Résoudre le conflit de position de grille
                _resolve_grid_conflict(corrected_cells, overlap, cells_to_remove)
                corrections_applied += 1
            
            elif overlap.overlap_type == OverlapType.PARTIAL:
                # Redimensionner les cellules qui se chevauchent
                if overlap.percentage1 >= 50 or overlap.percentage2 >= 50:
                    _resize_overlapping_cells(corrected_cells, overlap, cells_to_remove)
                    corrections_applied += 1
        
        # Supprimer les cellules marquées pour suppression
        if cells_to_remove:
            corrected_cells = [cell for i, cell in enumerate(corrected_cells) if i not in cells_to_remove]
            print(f"    🗑️  {len(cells_to_remove)} cellules supprimées après fusion")
        
        # Recalculer les positions de grille si des cellules ont été supprimées
        if cells_to_remove:
            print("    🔄 Recalcul des positions de grille...")
            corrected_cells = _recalculate_grid_positions_conservative(corrected_cells, tolerance=5)
        
        print(f"  ✅ {corrections_applied} corrections appliquées dans cette itération")
        
        # Si aucune correction appliquée, on peut arrêter
        if corrections_applied == 0:
            print("  ⚠️  Aucune correction appliquée - arrêt anticipé")
            break
    
    else:
        # Si on arrive ici, c'est qu'on a atteint la limite d'itérations
        print(f"  ⚠️  Limite de {max_iterations} itérations atteinte")
    
    # Vérification finale
    print("  🔍 Vérification finale des chevauchements...")
    final_overlaps = _detect_overlapping_cells(corrected_cells, verbose=verbose_overlaps)
    
    if final_overlaps:
        print(f"  ⚠️  {len(final_overlaps)} chevauchements subsistent après correction récursive")
        # Afficher les chevauchements restants pour diagnostic
        for overlap in final_overlaps:
            print(f"    - {overlap.overlap_type.value}: cellules #{overlap.cell1_idx} et #{overlap.cell2_idx} "
                  f"(sévérité: {overlap.severity:.1f})")
    else:
        print("  ✅ Tous les chevauchements ont été corrigés avec succès")
    
    return corrected_cells


def _merge_duplicate_cells(cells: List[TableCell], overlap: Overlap, to_remove: set) -> None:
    """Fusionne deux cellules identiques"""
    cell1, cell2 = overlap.cell1, overlap.cell2
    
    # Fusionner cell2 dans cell1
    cell1.merge_with(cell2)
    
    # Marquer cell2 pour suppression
    to_remove.add(overlap.cell2_idx)
    
    print(f"    ✅ Cellules dupliquées fusionnées")


def _absorb_included_cell(cells: List[TableCell], overlap: Overlap, to_remove: set) -> None:
    """Absorbe une cellule incluse dans une autre"""
    cell1, cell2 = overlap.cell1, overlap.cell2
    
    # Déterminer quelle cellule absorbe l'autre
    if overlap.percentage1 >= overlap.percentage2:
        # cell1 est plus incluse dans cell2 -> cell2 absorbe cell1
        cell2.merge_with(cell1)
        to_remove.add(overlap.cell1_idx)
        print(f"    ✅ Cellule #{overlap.cell1_idx} absorbée par #{overlap.cell2_idx}")
    else:
        # cell2 est plus incluse dans cell1 -> cell1 absorbe cell2
        cell1.merge_with(cell2)
        to_remove.add(overlap.cell2_idx)
        print(f"    ✅ Cellule #{overlap.cell2_idx} absorbée par #{overlap.cell1_idx}")


def _resolve_grid_conflict(cells: List[TableCell], overlap: Overlap, to_remove: set) -> None:
    """Résout un conflit de position de grille"""
    cell1, cell2 = overlap.cell1, overlap.cell2
    
    # Prioriser la cellule avec du texte
    if not cell1.is_empty() and cell2.is_empty():
        # cell1 a du texte, absorber cell2
        cell1.merge_with(cell2)
        to_remove.add(overlap.cell2_idx)
        print(f"    ✅ Cellule vide #{overlap.cell2_idx} fusionnée avec cellule pleine #{overlap.cell1_idx}")
    elif cell1.is_empty() and not cell2.is_empty():
        # cell2 a du texte, absorber cell1
        cell2.merge_with(cell1)
        to_remove.add(overlap.cell1_idx)
        print(f"    ✅ Cellule vide #{overlap.cell1_idx} fusionnée avec cellule pleine #{overlap.cell2_idx}")
    else:
        # Les deux ont du texte ou sont vides -> fusionner
        cell1.merge_with(cell2)
        to_remove.add(overlap.cell2_idx)
        print(f"    ✅ Cellules à même position fusionnées: #{overlap.cell2_idx} → #{overlap.cell1_idx}")


def _resize_overlapping_cells(cells: List[TableCell], overlap: Overlap, to_remove: set) -> None:
    """Redimensionne les cellules qui se chevauchent partiellement"""
    cell1, cell2 = overlap.cell1, overlap.cell2
    
    # Stratégie simple : diviser l'espace au milieu du chevauchement
    inter_x1 = max(cell1.x1, cell2.x1)
    inter_y1 = max(cell1.y1, cell2.y1)
    inter_x2 = min(cell1.x2, cell2.x2)
    inter_y2 = min(cell1.y2, cell2.y2)
    
    # Calculer les milieux
    mid_x = (inter_x1 + inter_x2) / 2
    mid_y = (inter_y1 + inter_y2) / 2
    
    # Déterminer si le chevauchement est plutôt horizontal ou vertical
    overlap_width = inter_x2 - inter_x1
    overlap_height = inter_y2 - inter_y1
    
    if overlap_width > overlap_height:
        # Chevauchement horizontal -> ajuster verticalement
        if cell1.y1 < cell2.y1:
            # cell1 est au-dessus, ajuster vers le bas
            cell1.y2 = min(cell1.y2, mid_y)
            cell2.y1 = max(cell2.y1, mid_y)
        else:
            # cell2 est au-dessus, ajuster vers le bas
            cell2.y2 = min(cell2.y2, mid_y)
            cell1.y1 = max(cell1.y1, mid_y)
    else:
        # Chevauchement vertical -> ajuster horizontalement
        if cell1.x1 < cell2.x1:
            # cell1 est à gauche, ajuster vers la droite
            cell1.x2 = min(cell1.x2, mid_x)
            cell2.x1 = max(cell2.x1, mid_x)
        else:
            # cell2 est à gauche, ajuster vers la droite
            cell2.x2 = min(cell2.x2, mid_x)
            cell1.x1 = max(cell1.x1, mid_x)
    
    print(f"    ✅ Cellules redimensionnées pour éliminer le chevauchement")


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
                            overlap_threshold: float = 0.5,
                            force_assignment: bool = False,
                            clean_structure: bool = False,
                            auto_correct_overlaps: bool = False,
                            smart_spacing: bool = True,
                            verbose_overlaps: bool = False) -> List[TableCell]:
    """
    Assigne les textes OCR aux cellules du tableau.
    
    Args:
        table_structure: Structure du tableau (cellules)
        rec_boxes: Boxes des textes OCR
        rec_texts: Textes OCR correspondants
        overlap_threshold: Seuil de recouvrement minimum
        force_assignment: Si True, force l'assignment même sans recouvrement
        clean_structure: Si True, nettoie la structure APRÈS l'assignment
        auto_correct_overlaps: Si True, corrige automatiquement les chevauchements
        smart_spacing: Si True, utilise l'espacement intelligent basé sur les distances réelles
        verbose_overlaps: Si True, affiche les détails de détection des chevauchements
        
    Returns:
        Structure avec textes assignés
    """
    if not table_structure or not rec_boxes or not rec_texts:
        return table_structure
    
    # Copier la structure pour ne pas modifier l'original
    filled_structure = []
    for cell in table_structure:
        new_cell = TableCell(cell.x1, cell.y1, cell.x2, cell.y2, 
                           cell.row_start, cell.col_start, 
                           cell.row_span, cell.col_span)
        new_cell.texts = cell.texts.copy()
        new_cell.final_text = cell.final_text
        new_cell.is_auto_filled = getattr(cell, 'is_auto_filled', False)
        new_cell.smart_spacing = smart_spacing  # Stocker le paramètre dans la cellule
        filled_structure.append(new_cell)
    
    # Suivre quels textes ont été assignés
    assigned_texts = set()
    
    # Pour chaque text box, trouver la meilleure cellule
    for i, (text_box, text_content) in enumerate(zip(rec_boxes, rec_texts)):
        if i in assigned_texts:
            continue
            
        text_x1, text_y1, text_x2, text_y2 = text_box
        best_cell = None
        best_overlap = 0
        
        # Trouver la cellule avec le meilleur recouvrement
        for cell in filled_structure:
            overlap = _calculate_overlap(text_x1, text_y1, text_x2, text_y2, 
                                       cell.x1, cell.y1, cell.x2, cell.y2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cell = cell
        
        # Assigner le texte si le recouvrement est suffisant
        if best_cell and best_overlap >= overlap_threshold:
            best_cell.add_text(text_content, text_x1, text_y1, text_x2, text_y2)
            assigned_texts.add(i)
        elif force_assignment and best_cell:
            # Mode force : assigner même sans recouvrement suffisant
            best_cell.add_text(text_content, text_x1, text_y1, text_x2, text_y2)
            assigned_texts.add(i)
        else:
            # Texte non assigné - diagnostic
            print(f"⚠️  Texte non assigné: '{text_content}' (recouvrement: {best_overlap:.2f} < {overlap_threshold})")
            if best_cell:
                print(f"    Meilleure cellule: ({best_cell.x1}, {best_cell.y1}) -> ({best_cell.x2}, {best_cell.y2})")
    
    # Finaliser les textes dans chaque cellule
    for cell in filled_structure:
        cell.finalize_text()
    
    # Nettoyer la structure APRÈS l'assignment des textes (si demandé)
    if clean_structure:
        print("🧹 Nettoyage de la structure APRÈS assignment des textes...")
        filled_structure = clean_table_structure(filled_structure, tolerance=5, verbose_overlaps=verbose_overlaps)
    
    # Détecter les chevauchements sur la version finale de la grille
    if verbose_overlaps:
        print("🔍 Détection des chevauchements sur la version finale...")
    overlaps = _detect_overlapping_cells(filled_structure, verbose=verbose_overlaps)

    # Corriger automatiquement les chevauchements (si demandé)
    if auto_correct_overlaps:
        print("🔄 Correction automatique des chevauchements...")
        filled_structure = _auto_correct_overlaps(filled_structure, verbose_overlaps=verbose_overlaps)
    
    return filled_structure


def _order_texts_spatially_with_cell_context(texts: List[Dict], cell_width: float, cell_height: float) -> str:
    """
    Ordonne les textes avec espacement intelligent basé sur les dimensions de la cellule.
    
    Args:
        texts: Liste des textes avec leurs positions
        cell_width: Largeur de la cellule
        cell_height: Hauteur de la cellule
        
    Returns:
        Texte final ordonné avec espacement adapté à la cellule
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]['text']
    
    # Calculer les paramètres d'espacement intelligent
    spacing_params = _calculate_smart_spacing(texts, cell_width, cell_height)
    
    # Trier par position Y (haut vers bas) puis X (gauche vers droite)
    texts.sort(key=lambda t: (t['center'][1], t['center'][0]))
    
    # Grouper par lignes avec la tolérance adaptée
    lines = []
    current_line = [texts[0]]
    
    for text in texts[1:]:
        vertical_distance = abs(text['center'][1] - current_line[0]['center'][1])
        
        if vertical_distance <= spacing_params['y_tolerance']:
            # Même ligne approximative
            current_line.append(text)
        else:
            # Nouvelle ligne
            lines.append(current_line)
            current_line = [text]
    
    lines.append(current_line)
    
    # Construire le texte final avec espacement adaptatif
    result_lines = []
    
    for line_idx, line in enumerate(lines):
        # Trier les textes de la ligne par X (gauche vers droite)
        line.sort(key=lambda t: t['center'][0])
        
        # Construire le texte de la ligne avec espacement horizontal adaptatif
        line_parts = []
        for i, text in enumerate(line):
            line_parts.append(text['text'])
            
            # Calculer l'espacement horizontal avec le texte suivant
            if i < len(line) - 1:
                next_text = line[i + 1]
                
                # Distance horizontale entre les textes
                current_right = text['box'][2]  # x2 du texte actuel
                next_left = next_text['box'][0]  # x1 du texte suivant
                horizontal_gap = next_left - current_right
                
                # Calculer le nombre d'espaces à insérer avec adaptation contextuelle
                if horizontal_gap > 0:
                    # Base : largeur moyenne des caractères
                    base_spaces = max(1, int(horizontal_gap / spacing_params['avg_char_width']))
                    
                    # Ajustement basé sur la largeur de la cellule
                    width_ratio = cell_width / 200  # 200px comme référence
                    adjusted_spaces = int(base_spaces * width_ratio)
                    
                    # Ajustement basé sur la densité de texte
                    if spacing_params['text_density'] > 0.001:  # Cellule dense
                        adjusted_spaces = max(1, adjusted_spaces // 2)  # Réduire l'espacement
                    elif spacing_params['text_density'] < 0.0001:  # Cellule peu dense
                        adjusted_spaces = min(adjusted_spaces * 2, 30)  # Augmenter l'espacement
                    
                    # Limiter le nombre d'espaces
                    final_spaces = max(1, min(adjusted_spaces, 25))
                    
                    line_parts.append(' ' * final_spaces)
                else:
                    # Pas d'écart ou textes qui se chevauchent -> un seul espace
                    line_parts.append(' ')
        
        line_text = ''.join(line_parts)
        result_lines.append(line_text)
        
        # Calculer l'espacement vertical avec la ligne suivante
        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1]
            
            # Calculer la distance verticale entre les lignes
            current_line_bottom = max(t['box'][3] for t in line)  # y2 max de la ligne actuelle
            next_line_top = min(t['box'][1] for t in next_line)   # y1 min de la ligne suivante
            vertical_gap = next_line_top - current_line_bottom
            
            # Calculer le nombre de lignes vides à insérer avec adaptation contextuelle
            if vertical_gap > spacing_params['avg_height'] * 0.5:  # Seuil minimum
                # Base : hauteur moyenne des textes
                base_lines = int(vertical_gap / spacing_params['avg_height'])
                
                # Ajustement basé sur la hauteur de la cellule
                height_ratio = cell_height / 100  # 100px comme référence
                adjusted_lines = int(base_lines * height_ratio)
                
                # Ajustement basé sur la densité de texte
                if spacing_params['text_density'] > 0.001:  # Cellule dense
                    adjusted_lines = max(0, adjusted_lines // 2)  # Réduire l'espacement
                elif spacing_params['text_density'] < 0.0001:  # Cellule peu dense
                    adjusted_lines = min(adjusted_lines * 2, 15)  # Augmenter l'espacement
                
                # Limiter le nombre de lignes vides
                final_lines = max(0, min(adjusted_lines, 10))
                
                # Ajouter les lignes vides
                for _ in range(final_lines):
                    result_lines.append('')
    
    return '\n'.join(result_lines)


def _order_texts_spatially(texts: List[Dict]) -> str:
    """
    Ordonne les textes dans une cellule selon leur position spatiale avec espacement intelligent.
    
    Args:
        texts: Liste des textes avec leurs positions
        
    Returns:
        Texte final ordonné avec espaces et retours à la ligne proportionnels aux distances
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]['text']
    
    # Trier par position Y (haut vers bas) puis X (gauche vers droite)
    texts.sort(key=lambda t: (t['center'][1], t['center'][0]))
    
    # Estimer la taille de police moyenne basée sur la hauteur des textes
    text_heights = [t['box'][3] - t['box'][1] for t in texts]
    avg_text_height = sum(text_heights) / len(text_heights) if text_heights else 12
    
    # Estimer la largeur moyenne des caractères (approximation)
    avg_char_width = avg_text_height * 0.6  # Ratio approximatif hauteur/largeur
    
    # Grouper par lignes avec espacement vertical intelligent
    lines = []
    current_line = [texts[0]]
    y_tolerance = avg_text_height * 0.3  # Tolérance basée sur la taille de police
    
    for text in texts[1:]:
        vertical_distance = abs(text['center'][1] - current_line[0]['center'][1])
        
        if vertical_distance <= y_tolerance:
            # Même ligne approximative
            current_line.append(text)
        else:
            # Nouvelle ligne
            lines.append(current_line)
            current_line = [text]
    
    lines.append(current_line)
    
    # Construire le texte final avec espacement intelligent
    result_lines = []
    
    for line_idx, line in enumerate(lines):
        # Trier les textes de la ligne par X (gauche vers droite)
        line.sort(key=lambda t: t['center'][0])
        
        # Construire le texte de la ligne avec espacement horizontal intelligent
        line_parts = []
        for i, text in enumerate(line):
            line_parts.append(text['text'])
            
            # Calculer l'espacement horizontal avec le texte suivant
            if i < len(line) - 1:
                next_text = line[i + 1]
                
                # Distance horizontale entre les textes
                current_right = text['box'][2]  # x2 du texte actuel
                next_left = next_text['box'][0]  # x1 du texte suivant
                horizontal_gap = next_left - current_right
                
                # Calculer le nombre d'espaces à insérer
                if horizontal_gap > 0:
                    # Estimer le nombre d'espaces basé sur la largeur moyenne des caractères
                    num_spaces = max(1, int(horizontal_gap / avg_char_width))
                    
                    # Limiter le nombre d'espaces pour éviter des écarts trop grands
                    num_spaces = min(num_spaces, 20)  # Max 20 espaces
                    
                    line_parts.append(' ' * num_spaces)
                else:
                    # Pas d'écart ou textes qui se chevauchent -> un seul espace
                    line_parts.append(' ')
        
        line_text = ''.join(line_parts)
        result_lines.append(line_text)
        
        # Calculer l'espacement vertical avec la ligne suivante
        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1]
            
            # Calculer la distance verticale entre les lignes
            current_line_bottom = max(t['box'][3] for t in line)  # y2 max de la ligne actuelle
            next_line_top = min(t['box'][1] for t in next_line)   # y1 min de la ligne suivante
            vertical_gap = next_line_top - current_line_bottom
            
            # Calculer le nombre de lignes vides à insérer
            if vertical_gap > avg_text_height * 0.5:  # Seuil minimum pour insérer des lignes
                # Estimer le nombre de lignes basé sur la hauteur moyenne
                num_blank_lines = int(vertical_gap / avg_text_height)
                
                # Limiter le nombre de lignes vides pour éviter des écarts trop grands
                num_blank_lines = min(num_blank_lines, 10)  # Max 10 lignes vides
                
                # Ajouter les lignes vides
                for _ in range(num_blank_lines):
                    result_lines.append('')
    
    return '\n'.join(result_lines)


def _estimate_font_size_from_texts(texts: List[Dict]) -> Tuple[float, float]:
    """
    Estime la taille de police et la largeur moyenne des caractères à partir des textes.
    
    Args:
        texts: Liste des textes avec leurs positions
        
    Returns:
        Tuple (hauteur_moyenne, largeur_moyenne_caractère)
    """
    if not texts:
        return 12.0, 7.2  # Valeurs par défaut
    
    # Calculer la hauteur moyenne des textes
    text_heights = []
    char_widths = []
    
    for text in texts:
        height = text['box'][3] - text['box'][1]
        width = text['box'][2] - text['box'][0]
        text_length = len(text['text'].strip())
        
        text_heights.append(height)
        
        # Estimer la largeur moyenne par caractère
        if text_length > 0:
            char_width = width / text_length
            char_widths.append(char_width)
    
    avg_height = sum(text_heights) / len(text_heights) if text_heights else 12.0
    avg_char_width = sum(char_widths) / len(char_widths) if char_widths else avg_height * 0.6
    
    return avg_height, avg_char_width


def _calculate_smart_spacing(texts: List[Dict], cell_width: float, cell_height: float) -> Dict:
    """
    Calcule l'espacement intelligent basé sur la taille de la cellule et des textes.
    
    Args:
        texts: Liste des textes avec leurs positions
        cell_width: Largeur de la cellule
        cell_height: Hauteur de la cellule
        
    Returns:
        Dictionnaire avec les paramètres d'espacement
    """
    if not texts:
        return {'avg_height': 12.0, 'avg_char_width': 7.2, 'y_tolerance': 4.0}
    
    avg_height, avg_char_width = _estimate_font_size_from_texts(texts)
    
    # Calculer la tolérance Y basée sur la taille de police et la hauteur de cellule
    y_tolerance = avg_height * 0.3
    
    # Ajuster en fonction de la densité de texte dans la cellule
    text_density = len(texts) / (cell_width * cell_height) if cell_width * cell_height > 0 else 0
    
    # Plus la densité est élevée, plus on est strict sur l'espacement
    if text_density > 0.001:  # Cellule dense
        y_tolerance *= 0.7
    elif text_density < 0.0001:  # Cellule peu dense
        y_tolerance *= 1.3
    
    return {
        'avg_height': avg_height,
        'avg_char_width': avg_char_width,
        'y_tolerance': y_tolerance,
        'text_density': text_density
    }


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
    for i, cell in enumerate(filled_structure):
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
        
        # Afficher l'indice de la cellule dans le coin supérieur gauche
        ax.text(cell.x1 + 5, cell.y1 + 15, f"#{i}", 
                ha='left', va='top', 
                fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Texte de la cellule au centre
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
                # Trouver l'indice de cette cellule dans la liste originale
                cell_index = filled_structure.index(cell_at_position)
                
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
                
                # Ajouter l'identifiant de la cellule en petit dans le coin
                cell_id = f'<span style="font-size: 10px; color: #999; float: right;">#{cell_index}</span>'
                
                # Échapper et convertir les retours à la ligne en <br>
                cell_content = _escape_html(cell_at_position.final_text).replace('\n', '<br>')
                if not cell_content.strip():
                    cell_content = "&nbsp;"
                
                # Combiner l'identifiant avec le contenu
                final_content = f'{cell_id}{cell_content}'
                
                html += f'            <td class="{cell_class}"{rowspan_attr}{colspan_attr}{align_style}>{final_content}</td>\n'
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
    fill_empty_cells=True,      # Compléter automatiquement les cellules vides
    extend_cells=True,          # Étendre les cellules pour combler les espaces
    clean_structure=False       # Désactivé par défaut pour éviter les problèmes
)

# 3. Visualiser la structure (avec cellules auto-générées en gris)
plot_table_structure(table_structure)

# 4. Assigner les textes OCR avec nettoyage
filled_structure = assign_ocr_to_structure(
    table_structure, rec_boxes, rec_texts, 
    overlap_threshold=0.5,
    force_assignment=True,     # Forcer l'assignment même sans recouvrement
    clean_structure=False,     # Désactivé par défaut pour éviter les problèmes
    auto_correct_overlaps=True,# Corriger automatiquement les chevauchements
    smart_spacing=True,        # Espacement intelligent basé sur les distances réelles
    verbose_overlaps=False     # Masquer les détails de détection des chevauchements (par défaut)
)

# 5. Visualiser le résultat final
plot_final_result(filled_structure)

# 6. Exporter en HTML avec couleurs
html_output = export_to_html(
    filled_structure, 
    "Mon Tableau",
    highlight_merged=True         # Colorie les cellules fusionnées
)
print(html_output)

# 7. Sauvegarder le HTML dans un fichier
save_html_to_file(html_output, "mon_tableau.html")

# 8. Exporter en Markdown
markdown_output = export_to_markdown(filled_structure, "Mon Tableau")
print(markdown_output)

# 9. Sauvegarder le Markdown dans un fichier
save_markdown_to_file(markdown_output, "mon_tableau.md")

# OU sans couleurs :
html_simple = export_to_html(filled_structure, highlight_merged=False)

# OU sans complétion automatique :
table_structure_basic = extract_table_structure(layout_boxes, fill_empty_cells=False)

# OU pour étendre les cellules sans les compléter automatiquement :
table_structure_extended = extract_table_structure(layout_boxes, extend_cells=True, fill_empty_cells=False)

# OU pour nettoyer la structure séparément :
table_structure = extract_table_structure(layout_boxes, fill_empty_cells=True)
clean_structure = clean_table_structure(table_structure)
""" 