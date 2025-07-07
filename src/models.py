#!/usr/bin/env python3
"""
📋 Modèles Pydantic pour le module OCR Tableaux

Ce fichier définit les modèles de données utilisés pour valider et documenter
les structures de données dans le module OCR Tableaux.

Auteur: Assistant IA
Version: 2.0
Date: 2024
"""

from typing import List, Tuple, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np

# === MODÈLES DE BASE ===

class Box(BaseModel):
    """
    Modèle pour représenter une boîte englobante (rectangle).
    
    Attributes:
        x_min: Coordonnée X minimale
        y_min: Coordonnée Y minimale  
        x_max: Coordonnée X maximale
        y_max: Coordonnée Y maximale
    """
    x_min: float = Field(..., ge=0, description="Coordonnée X minimale")
    y_min: float = Field(..., ge=0, description="Coordonnée Y minimale")
    x_max: float = Field(..., gt=0, description="Coordonnée X maximale")
    y_max: float = Field(..., gt=0, description="Coordonnée Y maximale")
    
    @validator('x_max')
    def x_max_must_be_greater_than_x_min(cls, v, values):
        if 'x_min' in values and v <= values['x_min']:
            raise ValueError('x_max doit être supérieur à x_min')
        return v
    
    @validator('y_max')
    def y_max_must_be_greater_than_y_min(cls, v, values):
        if 'y_min' in values and v <= values['y_min']:
            raise ValueError('y_max doit être supérieur à y_min')
        return v
    
    def to_list(self) -> List[float]:
        """Convertit la box en liste [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]
    
    def center(self) -> Tuple[float, float]:
        """Calcule le centre de la box."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    def area(self) -> float:
        """Calcule l'aire de la box."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def width(self) -> float:
        """Calcule la largeur de la box."""
        return self.x_max - self.x_min
    
    def height(self) -> float:
        """Calcule la hauteur de la box."""
        return self.y_max - self.y_min

class Point(BaseModel):
    """
    Modèle pour représenter un point 2D.
    
    Attributes:
        x: Coordonnée X
        y: Coordonnée Y
    """
    x: float = Field(..., description="Coordonnée X")
    y: float = Field(..., description="Coordonnée Y")
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convertit le point en tuple (x, y)."""
        return (self.x, self.y)

class CompositeCell(BaseModel):
    """
    Modèle pour représenter une cellule composite avec sa position dans la grille.
    
    Attributes:
        row_start: Ligne de début (incluse)
        row_end: Ligne de fin (exclue)
        col_start: Colonne de début (incluse)
        col_end: Colonne de fin (exclue)
        text: Texte contenu dans la cellule
    """
    row_start: int = Field(..., ge=0, description="Ligne de début (incluse)")
    row_end: int = Field(..., gt=0, description="Ligne de fin (exclue)")
    col_start: int = Field(..., ge=0, description="Colonne de début (incluse)")
    col_end: int = Field(..., gt=0, description="Colonne de fin (exclue)")
    text: str = Field(..., min_length=0, description="Texte contenu dans la cellule")
    
    @validator('row_end')
    def row_end_must_be_greater_than_row_start(cls, v, values):
        if 'row_start' in values and v <= values['row_start']:
            raise ValueError('row_end doit être supérieur à row_start')
        return v
    
    @validator('col_end')
    def col_end_must_be_greater_than_col_start(cls, v, values):
        if 'col_start' in values and v <= values['col_start']:
            raise ValueError('col_end doit être supérieur à col_start')
        return v
    
    def to_tuple(self) -> Tuple[int, int, int, int, str]:
        """Convertit en tuple (row_start, row_end, col_start, col_end, text)."""
        return (self.row_start, self.row_end, self.col_start, self.col_end, self.text)
    
    def is_merged(self) -> bool:
        """Vérifie si la cellule est fusionnée (span > 1)."""
        return (self.row_end - self.row_start > 1) or (self.col_end - self.col_start > 1)
    
    def row_span(self) -> int:
        """Calcule le span vertical."""
        return self.row_end - self.row_start
    
    def col_span(self) -> int:
        """Calcule le span horizontal."""
        return self.col_end - self.col_start

# === MODÈLES COMPOSÉS ===

class OCRText(BaseModel):
    """
    Modèle pour représenter un texte OCR avec sa position.
    
    Attributes:
        text: Texte reconnu
        box: Boîte englobante du texte
        confidence: Confiance de la reconnaissance (optionnel)
    """
    text: str = Field(..., description="Texte reconnu par OCR")
    box: Box = Field(..., description="Boîte englobante du texte")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confiance de la reconnaissance")
    
    def is_empty(self) -> bool:
        """Vérifie si le texte est vide."""
        return len(self.text.strip()) == 0

class GridStructure(BaseModel):
    """
    Modèle pour représenter la structure d'une grille.
    
    Attributes:
        row_lines: Positions des lignes horizontales
        col_lines: Positions des lignes verticales
        n_rows: Nombre de lignes
        n_cols: Nombre de colonnes
    """
    row_lines: List[float] = Field(..., description="Positions des lignes horizontales")
    col_lines: List[float] = Field(..., description="Positions des lignes verticales")
    
    @validator('row_lines')
    def row_lines_must_be_sorted(cls, v):
        if v != sorted(v):
            raise ValueError('row_lines doit être trié')
        return v
    
    @validator('col_lines')
    def col_lines_must_be_sorted(cls, v):
        if v != sorted(v):
            raise ValueError('col_lines doit être trié')
        return v
    
    @property
    def n_rows(self) -> int:
        """Nombre de lignes dans la grille."""
        return len(self.row_lines)
    
    @property
    def n_cols(self) -> int:
        """Nombre de colonnes dans la grille."""
        return len(self.col_lines)
    
    def to_tuple(self) -> Tuple[List[float], List[float]]:
        """Convertit en tuple (row_lines, col_lines)."""
        return (self.row_lines, self.col_lines)

class TableData(BaseModel):
    """
    Modèle pour représenter les données complètes d'un tableau.
    
    Attributes:
        cell_boxes: Boxes des cellules détectées
        ocr_texts: Textes OCR avec leurs positions
        grid_structure: Structure de la grille
        composite_cells: Cellules composites finales
    """
    cell_boxes: List[Box] = Field(..., description="Boxes des cellules détectées")
    ocr_texts: List[OCRText] = Field(..., description="Textes OCR avec positions")
    grid_structure: GridStructure = Field(..., description="Structure de la grille")
    composite_cells: List[CompositeCell] = Field(..., description="Cellules composites finales")
    
    def to_grid_2d(self) -> List[List[str]]:
        """Convertit en grille 2D simple."""
        table = [["" for _ in range(self.grid_structure.n_cols)] 
                for _ in range(self.grid_structure.n_rows)]
        
        for cell in self.composite_cells:
            if cell.text.strip():
                target_r = max(0, min(cell.row_start, self.grid_structure.n_rows - 1))
                target_c = max(0, min(cell.col_start, self.grid_structure.n_cols - 1))
                table[target_r][target_c] = cell.text
        
        return table
    
    def get_stats(self) -> dict:
        """Calcule les statistiques du tableau."""
        total_cells = self.grid_structure.n_rows * self.grid_structure.n_cols
        filled_cells = len([cell for cell in self.composite_cells if cell.text.strip()])
        merged_cells = len([cell for cell in self.composite_cells if cell.is_merged()])
        total_chars = sum(len(cell.text) for cell in self.composite_cells)
        
        return {
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "merged_cells": merged_cells,
            "total_chars": total_chars,
            "fill_rate": filled_cells / total_cells if total_cells > 0 else 0.0,
            "merge_rate": merged_cells / filled_cells if filled_cells > 0 else 0.0
        }

# === MODÈLES DE CONFIGURATION ===

class GridConfig(BaseModel):
    """
    Configuration pour la construction de grille.
    
    Attributes:
        y_thresh: Seuil pour regrouper les lignes horizontales
        x_thresh: Seuil pour regrouper les lignes verticales
        tolerance: Tolérance pour l'épaisseur des traits
    """
    y_thresh: float = Field(10.0, gt=0, description="Seuil pour les lignes horizontales")
    x_thresh: float = Field(10.0, gt=0, description="Seuil pour les lignes verticales")
    tolerance: float = Field(5.0, gt=0, description="Tolérance pour l'épaisseur des traits")

class PlacementConfig(BaseModel):
    """
    Configuration pour le placement des textes.
    
    Attributes:
        tolerance: Tolérance pour le placement des textes
        spatial_tolerance: Tolérance pour l'ordonnancement spatial
        merge_threshold: Seuil pour la fusion de textes
    """
    tolerance: float = Field(5.0, gt=0, description="Tolérance pour le placement")
    spatial_tolerance: float = Field(10.0, gt=0, description="Tolérance spatiale")
    merge_threshold: int = Field(3, gt=0, description="Seuil pour la fusion de textes")

class ExportConfig(BaseModel):
    """
    Configuration pour l'export des tableaux.
    
    Attributes:
        include_headers: Inclure les en-têtes automatiques
        cell_alignment: Alignement des cellules
        table_class: Classe CSS pour HTML
        highlight_merged: Surligner les cellules fusionnées
    """
    include_headers: bool = Field(True, description="Inclure les en-têtes automatiques")
    cell_alignment: str = Field("left", regex="^(left|center|right)$", description="Alignement des cellules")
    table_class: str = Field("ocr-table", description="Classe CSS pour HTML")
    highlight_merged: bool = Field(True, description="Surligner les cellules fusionnées")

# === MODÈLES DE RÉSULTATS ===

class ProcessingResult(BaseModel):
    """
    Résultat du traitement d'un tableau.
    
    Attributes:
        table_data: Données du tableau
        markdown_output: Export Markdown
        html_output: Export HTML
        processing_time: Temps de traitement
        errors: Liste des erreurs rencontrées
    """
    table_data: TableData = Field(..., description="Données du tableau")
    markdown_output: Optional[str] = Field(None, description="Export Markdown")
    html_output: Optional[str] = Field(None, description="Export HTML")
    processing_time: Optional[float] = Field(None, ge=0, description="Temps de traitement en secondes")
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs")
    
    def is_successful(self) -> bool:
        """Vérifie si le traitement a réussi."""
        return len(self.errors) == 0
    
    def has_output(self) -> bool:
        """Vérifie si des outputs ont été générés."""
        return self.markdown_output is not None or self.html_output is not None

# === FONCTIONS UTILITAIRES ===

def validate_box_list(boxes: List[List[float]]) -> List[Box]:
    """
    Valide et convertit une liste de boxes en modèles Pydantic.
    
    Args:
        boxes: Liste de boxes au format [x_min, y_min, x_max, y_max]
        
    Returns:
        Liste de modèles Box validés
        
    Raises:
        ValueError: Si une box est invalide
    """
    validated_boxes = []
    for i, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f"Box {i} doit avoir 4 valeurs, trouvé {len(box)}")
        
        try:
            validated_box = Box(
                x_min=box[0],
                y_min=box[1],
                x_max=box[2],
                y_max=box[3]
            )
            validated_boxes.append(validated_box)
        except Exception as e:
            raise ValueError(f"Box {i} invalide: {e}")
    
    return validated_boxes

def validate_composite_cells(cells: List[Tuple[int, int, int, int, str]]) -> List[CompositeCell]:
    """
    Valide et convertit une liste de cellules composites.
    
    Args:
        cells: Liste de tuples (r0, r1, c0, c1, text)
        
    Returns:
        Liste de modèles CompositeCell validés
        
    Raises:
        ValueError: Si une cellule est invalide
    """
    validated_cells = []
    for i, cell in enumerate(cells):
        if len(cell) != 5:
            raise ValueError(f"Cellule {i} doit avoir 5 valeurs, trouvé {len(cell)}")
        
        try:
            validated_cell = CompositeCell(
                row_start=cell[0],
                row_end=cell[1],
                col_start=cell[2],
                col_end=cell[3],
                text=cell[4]
            )
            validated_cells.append(validated_cell)
        except Exception as e:
            raise ValueError(f"Cellule {i} invalide: {e}")
    
    return validated_cells

# === EXEMPLE D'UTILISATION ===

if __name__ == "__main__":
    # Exemple d'utilisation des modèles
    
    # Création d'une box
    box = Box(x_min=10.0, y_min=20.0, x_max=50.0, y_max=60.0)
    print(f"Box: {box}")
    print(f"Centre: {box.center()}")
    print(f"Aire: {box.area()}")
    
    # Création d'une cellule composite
    cell = CompositeCell(
        row_start=0,
        row_end=2,
        col_start=0,
        col_end=1,
        text="Cellule fusionnée"
    )
    print(f"Cellule: {cell}")
    print(f"Fusionnée: {cell.is_merged()}")
    print(f"Span: {cell.row_span()}x{cell.col_span()}")
    
    # Configuration
    config = GridConfig(y_thresh=15.0, x_thresh=12.0)
    print(f"Configuration: {config}") 