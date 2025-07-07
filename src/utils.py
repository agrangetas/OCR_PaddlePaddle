import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
try:
    from IPython.display import display, Markdown
except ImportError:
    # Fallback si IPython n'est pas disponible
    def display(content):
        print(content)
    def Markdown(content):
        return content

# === AFFICHAGE MARKDOWN ===
def show_markdown(md_path: str):
    """Affiche un markdown Jupyter-friendly."""
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        display(Markdown(content))
    else:
        print("❌ Fichier Markdown non trouvé.")


def box_center(box):
    x_min, y_min, x_max, y_max = box
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def is_point_in_box(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def build_grid_structure(cell_box_list, y_thresh=10, x_thresh=10):
    """
    Détermine la structure des lignes et colonnes à partir de cell_box_list.
    Version améliorée avec ajustement dynamique de la grille.
    """
    return build_adaptive_grid_structure(cell_box_list, y_thresh, x_thresh)

def build_adaptive_grid_structure(cell_box_list, y_thresh=10, x_thresh=10, tolerance=5):
    """
    Construit une grille adaptative qui peut ajouter des lignes/colonnes dynamiquement.
    
    Args:
        cell_box_list: Liste des boxes de cellules [x_min, y_min, x_max, y_max]
        y_thresh: Seuil pour regrouper les lignes
        x_thresh: Seuil pour regrouper les colonnes  
        tolerance: Tolérance pour l'épaisseur des traits
    
    Returns:
        tuple: (row_lines, col_lines) listes des positions des lignes et colonnes
    """
    if not cell_box_list:
        return [], []
    
    # Étape 1: Construire une grille initiale avec l'approche classique
    def center_y(box): return (box[1] + box[3]) / 2
    def center_x(box): return (box[0] + box[2]) / 2
    
    # Collecter tous les bords des boxes
    y_positions = []
    x_positions = []
    
    for box in cell_box_list:
        x_min, y_min, x_max, y_max = box
        y_positions.extend([y_min, y_max])
        x_positions.extend([x_min, x_max])
    
    # Trier et dédupliquer avec tolérance
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))
    
    # Fonction pour regrouper les positions similaires
    def cluster_positions(positions, thresh):
        if not positions:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if abs(pos - current_cluster[-1]) <= thresh:
                current_cluster.append(pos)
            else:
                # Prendre la médiane du cluster pour plus de stabilité
                clusters.append(np.median(current_cluster))
                current_cluster = [pos]
        
        clusters.append(np.median(current_cluster))
        return clusters
    
    # Grille initiale
    row_lines = cluster_positions(y_positions, y_thresh)
    col_lines = cluster_positions(x_positions, x_thresh)
    
    # Étape 2: Vérifier chaque box et ajuster la grille si nécessaire
    for box_idx, box in enumerate(cell_box_list):
        x_min, y_min, x_max, y_max = box
        
        # Vérifier si cette box nécessite des lignes supplémentaires
        row_lines, col_lines = _check_and_add_lines(
            box, row_lines, col_lines, y_thresh, x_thresh, tolerance
        )
    
    # Étape 3: Optimisation finale - supprimer les lignes redondantes
    row_lines = _optimize_lines(row_lines, cell_box_list, y_thresh, is_horizontal=True)
    col_lines = _optimize_lines(col_lines, cell_box_list, x_thresh, is_horizontal=False)
    
    return sorted(row_lines), sorted(col_lines)

def _check_and_add_lines(box, row_lines, col_lines, y_thresh, x_thresh, tolerance):
    """
    Vérifie si une box nécessite l'ajout de nouvelles lignes/colonnes.
    """
    x_min, y_min, x_max, y_max = box
    
    # Vérifier les lignes horizontales (rows)
    new_row_lines = list(row_lines)
    
    # Vérifier si y_min et y_max ont des lignes correspondantes
    y_min_match = _find_closest_line(y_min, row_lines, y_thresh)
    y_max_match = _find_closest_line(y_max, row_lines, y_thresh)
    
    if y_min_match is None:
        new_row_lines.append(y_min)
    if y_max_match is None:
        new_row_lines.append(y_max)
    
    # Vérifier les lignes verticales (cols)
    new_col_lines = list(col_lines)
    
    x_min_match = _find_closest_line(x_min, col_lines, x_thresh)
    x_max_match = _find_closest_line(x_max, col_lines, x_thresh)
    
    if x_min_match is None:
        new_col_lines.append(x_min)
    if x_max_match is None:
        new_col_lines.append(x_max)
    
    # Vérifier si la box semble être subdivisée (détection de cellules plus petites)
    if len(new_row_lines) > len(row_lines) or len(new_col_lines) > len(col_lines):
        new_row_lines, new_col_lines = _handle_subdivision(
            box, new_row_lines, new_col_lines, y_thresh, x_thresh, tolerance
        )
    
    return sorted(set(new_row_lines)), sorted(set(new_col_lines))

def _find_closest_line(position, lines, threshold):
    """
    Trouve la ligne la plus proche d'une position donnée.
    Retourne None si aucune ligne n'est dans le seuil.
    """
    if not lines:
        return None
    
    distances = [abs(position - line) for line in lines]
    min_distance = min(distances)
    
    if min_distance <= threshold:
        return lines[distances.index(min_distance)]
    
    return None

def _handle_subdivision(box, row_lines, col_lines, y_thresh, x_thresh, tolerance):
    """
    Gère la subdivision détectée d'une cellule.
    """
    x_min, y_min, x_max, y_max = box
    
    # Analyser si la box semble être une subdivision
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Vérifier les lignes qui pourraient subdiviser cette box
    internal_row_lines = [line for line in row_lines if y_min < line < y_max]
    internal_col_lines = [line for line in col_lines if x_min < line < x_max]
    
    # Si on détecte des subdivisions internes, on peut ajuster la grille
    if internal_row_lines or internal_col_lines:
        # Ici on pourrait implémenter une logique plus sophistiquée
        # pour détecter des patterns de subdivision
        pass
    
    return row_lines, col_lines

def _optimize_lines(lines, cell_box_list, threshold, is_horizontal=True):
    """
    Optimise les lignes en supprimant celles qui ne sont pas nécessaires.
    """
    if len(lines) <= 1:
        return lines
    
    necessary_lines = []
    
    for line in lines:
        # Vérifier si cette ligne est nécessaire pour délimiter des cellules
        if _is_line_necessary(line, cell_box_list, threshold, is_horizontal):
            necessary_lines.append(line)
    
    return necessary_lines

def _is_line_necessary(line, cell_box_list, threshold, is_horizontal):
    """
    Détermine si une ligne est nécessaire pour délimiter des cellules.
    """
    separations = 0
    
    for box in cell_box_list:
        x_min, y_min, x_max, y_max = box
        
        if is_horizontal:
            # Pour les lignes horizontales, vérifier si elles séparent des cellules
            if abs(y_min - line) <= threshold or abs(y_max - line) <= threshold:
                separations += 1
        else:
            # Pour les lignes verticales
            if abs(x_min - line) <= threshold or abs(x_max - line) <= threshold:
                separations += 1
    
    # Une ligne est nécessaire si elle sépare au moins 2 cellules
    return separations >= 2

def assign_cells_to_grid(cell_box_list, row_lines, col_lines):
    """
    Associe chaque box à une zone [row_start:row_end, col_start:col_end] selon sa position.
    """
    grid_map = {}  # key = (row, col), value = index of cell_box
    cell_spans = {}

    for idx, box in enumerate(cell_box_list):
        x_min, y_min, x_max, y_max = box

        row_start = np.argmin([abs(y_min - y) for y in row_lines])
        row_end   = np.argmin([abs(y_max - y) for y in row_lines]) + 1
        col_start = np.argmin([abs(x_min - x) for x in col_lines])
        col_end   = np.argmin([abs(x_max - x) for x in col_lines]) + 1

        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                grid_map[(r, c)] = idx
        cell_spans[idx] = (row_start, row_end, col_start, col_end)

    return grid_map, cell_spans

def fill_grid_with_text(cell_box_list, rec_boxes, rec_text, row_lines, col_lines):
    """
    Construit une grille de texte, même en cas de fusion de cellules.
    """
    n_rows = len(row_lines)
    n_cols = len(col_lines)
    table = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    # Map cell_box to grid
    grid_map, cell_spans = assign_cells_to_grid(cell_box_list, row_lines, col_lines)

    # Attribuer le texte
    cell_contents = {i: [] for i in range(len(cell_box_list))}

    for rbox, text in zip(rec_boxes, rec_text):
        center = box_center(rbox)
        for i, cbox in enumerate(cell_box_list):
            if is_point_in_box(center, cbox):
                cell_contents[i].append(text)
                break

    # Remplir le tableau avec le texte fusionné
    filled = set()
    for i, (r0, r1, c0, c1) in cell_spans.items():
        # Joindre les textes avec des retours à la ligne si multiples
        if len(cell_contents[i]) == 1:
            content = cell_contents[i][0]
        else:
            content = "\n".join(cell_contents[i])
        for r in range(r0, r1):
            for c in range(c0, c1):
                if (r, c) not in filled:
                    table[r][c] = content
                    filled.add((r, c))

    return table
from collections import defaultdict

def get_cell_index_ranges(box, row_lines, col_lines):
    x_min, y_min, x_max, y_max = box

    row_start = np.argmin([abs(y_min - y) for y in row_lines])
    row_end   = np.argmin([abs(y_max - y) for y in row_lines]) + 1
    col_start = np.argmin([abs(x_min - x) for x in col_lines])
    col_end   = np.argmin([abs(x_max - x) for x in col_lines]) + 1

    return row_start, row_end, col_start, col_end

def build_composite_cells(cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance=5):
    """
    Associe chaque rec_box à une cellule du tableau, en tenant compte des tolérances
    et fusionne les rec_texts pour chaque cellule.
    """
    from collections import defaultdict
    import numpy as np

    cell_map = defaultdict(list)

    for rec_box, text in zip(rec_boxes, rec_texts):
        x_min, y_min, x_max, y_max = rec_box
        matched = False

        for cell_box in cell_boxes:
            cx_min, cy_min,cx_max, cy_max = cell_box

            # On applique une tolérance
            if (x_min >= cx_min - tolerance and x_max <= cx_max + tolerance and
                y_min >= cy_min - tolerance and y_max <= cy_max + tolerance):
                r0, r1, c0, c1 = get_cell_index_ranges(cell_box, row_lines, col_lines)
                key = (r0, r1, c0, c1)
                cell_map[key].append(text.strip())
                matched = True
                break  # Stop au premier match (évite les doublons)

        if not matched:
            print(f"[!] Texte non apparié : '{text}' (box = {rec_box})")

    composite_cells = []
    for (r0, r1, c0, c1), texts in cell_map.items():
        # Joindre les textes avec des retours à la ligne si multiples
        if len(texts) == 1:
            merged_text = texts[0]
        else:
            merged_text = "\n".join(texts)
        composite_cells.append((r0, r1, c0, c1, merged_text))

    return composite_cells
    
def fill_grid_from_composites(composite_cells, n_rows, n_cols):
    table = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    used_texts = set()

    for r0, r1, c0, c1, text in composite_cells:
        if text in used_texts:
            continue

        # Chercher la première cellule vide dans la première ligne du bloc
        placed = False
        for c in range(c0, min(c1, n_cols)):
            if 0 <= r0 < n_rows and 0 <= c < n_cols:
                if table[r0][c] == "":
                    table[r0][c] += (" " + text) if table[r0][c] else text                    
                    used_texts.add(text)
                    placed = True
                    break
        if not placed:
            continue  # Aucune cellule dispo pour ce texte, on saute

        # On vide les autres cellules du bloc
        for r in range(r0, min(r1, n_rows)):
            for c in range(c0, min(c1, n_cols)):
                if table[r][c] != text:
                    table[r][c] = ""

    return table


def export_to_markdown(table=None, composite_cells=None, n_rows=None, n_cols=None, 
                      include_headers=True, cell_alignment="left", table_title=None):
    """
    Convertit une grille 2D ou des composite_cells en tableau Markdown.
    
    Args:
        table: Grille 2D optionnelle (format traditionnel)
        composite_cells: Liste de cellules composites (r0, r1, c0, c1, text)
        n_rows, n_cols: Dimensions si utilisation des composite_cells
        include_headers: Si True, inclut une ligne d'en-tête avec Col 1, Col 2, etc.
        cell_alignment: "left", "center", "right"
        table_title: Titre optionnel pour le tableau
    
    Returns:
        str: Tableau Markdown formaté
    """
    
    # Déterminer la source de données
    if table is not None:
        # Utiliser la grille 2D fournie
        working_table = table
        n_rows = len(table)
        n_cols = max(len(row) for row in table) if table else 0
    elif composite_cells is not None and n_rows is not None and n_cols is not None:
        # Construire une grille à partir des composite_cells
        working_table = fill_grid_from_composites_simple(composite_cells, n_rows, n_cols)
    else:
        raise ValueError("Vous devez fournir soit 'table' soit 'composite_cells' avec n_rows/n_cols")
    
    if n_cols == 0:
        return "| (tableau vide) |\n|---|\n"
    
    # Normaliser toutes les lignes à la même longueur
    normalized_table = []
    for row in working_table:
        padded_row = row + [""] * (n_cols - len(row))
        normalized_table.append(padded_row)
    
    # Construire le tableau Markdown
    markdown_lines = []
    
    # Titre optionnel
    if table_title:
        markdown_lines.append(f"### {table_title}")
        markdown_lines.append("")
    
    # En-têtes
    if include_headers:
        header_row = []
        for i in range(n_cols):
            header_row.append(f"Col {i+1}")
        
        markdown_lines.append("| " + " | ".join(header_row) + " |")
        
        # Ligne de séparation avec alignement
        separator_parts = []
        for _ in range(n_cols):
            if cell_alignment == "center":
                separator_parts.append(":---:")
            elif cell_alignment == "right":
                separator_parts.append("---:")
            else:  # left
                separator_parts.append("---")
        
        markdown_lines.append("| " + " | ".join(separator_parts) + " |")
    
    # Lignes de données
    for row in normalized_table:
        # Nettoyer et échapper les caractères spéciaux Markdown
        cleaned_row = []
        for cell in row:
            cell_text = str(cell).strip()
            # Échapper les pipes et autres caractères spéciaux
            cell_text = cell_text.replace("|", "\\|").replace("\n", "<br>")
            cleaned_row.append(cell_text)
        
        markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
    
    return "\n".join(markdown_lines)

def export_to_markdown_advanced(composite_cells, n_rows, n_cols, 
                               show_merged_info=True, compact_empty=True,
                               table_title=None):
    """
    Version avancée d'export Markdown qui préserve les informations de fusion.
    
    Args:
        composite_cells: Liste de cellules composites
        n_rows, n_cols: Dimensions de la grille
        show_merged_info: Si True, indique les fusions avec [FUSIONNÉ]
        compact_empty: Si True, supprime les lignes/colonnes complètement vides
        table_title: Titre optionnel
    
    Returns:
        str: Tableau Markdown avec informations de fusion
    """
    
    # Créer une grille avec informations de fusion
    table = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    merge_info = {}  # (r, c) -> informations de fusion
    
    # Traitement des cellules composites
    for r0, r1, c0, c1, text in composite_cells:
        if not text.strip():
            continue
            
        # Déterminer si c'est une cellule fusionnée
        is_merged = (r1 - r0 > 1) or (c1 - c0 > 1)
        
        # Placer le texte dans la cellule de départ
        if 0 <= r0 < n_rows and 0 <= c0 < n_cols:
            display_text = text.strip()
            if show_merged_info and is_merged:
                span_info = f"[{r1-r0}x{c1-c0}]"
                display_text = f"{display_text} {span_info}"
            
            table[r0][c0] = display_text
            merge_info[(r0, c0)] = {
                "rowspan": r1 - r0,
                "colspan": c1 - c0,
                "is_merged": is_merged
            }
            
            # Marquer les autres cellules de la fusion
            for r in range(r0, min(r1, n_rows)):
                for c in range(c0, min(c1, n_cols)):
                    if r != r0 or c != c0:
                        table[r][c] = "~" if show_merged_info else ""
    
    # Suppression des lignes/colonnes vides si demandé
    if compact_empty:
        table = _compact_empty_rows_cols(table)
    
    # Utiliser la fonction Markdown standard
    return export_to_markdown(
        table=table, 
        include_headers=True, 
        table_title=table_title
    )

def _compact_empty_rows_cols(table):
    """
    Supprime les lignes et colonnes complètement vides.
    """
    if not table:
        return table
    
    n_rows = len(table)
    n_cols = max(len(row) for row in table)
    
    # Normaliser d'abord
    normalized_table = []
    for row in table:
        normalized_table.append(row + [""] * (n_cols - len(row)))
    
    # Identifier les lignes non vides
    non_empty_rows = []
    for r in range(n_rows):
        if any(cell.strip() and cell.strip() != "~" for cell in normalized_table[r]):
            non_empty_rows.append(r)
    
    # Identifier les colonnes non vides
    non_empty_cols = []
    for c in range(n_cols):
        if any(normalized_table[r][c].strip() and normalized_table[r][c].strip() != "~" 
               for r in range(n_rows)):
            non_empty_cols.append(c)
    
    # Construire la table compacte
    compact_table = []
    for r in non_empty_rows:
        compact_row = []
        for c in non_empty_cols:
            compact_row.append(normalized_table[r][c])
        compact_table.append(compact_row)
    
    return compact_table

def export_to_html(composite_cells=None, n_rows=None, n_cols=None, table=None,
                   table_title=None, table_class="ocr-table", cell_padding=4,
                   highlight_merged=True, include_stats=True, use_merges=True):
    """
    Export HTML amélioré avec gestion réelle de colspan et rowspan.
    
    Args:
        composite_cells: Liste de cellules composites
        n_rows, n_cols: Dimensions si utilisation des composite_cells
        table: Grille 2D alternative
        table_title: Titre du tableau
        table_class: Classe CSS pour le tableau
        cell_padding: Espacement des cellules
        highlight_merged: Surligner les cellules fusionnées
        include_stats: Inclure les statistiques
        use_merges: Si True, utilise colspan/rowspan pour les vraies fusions
    
    Returns:
        str: HTML formaté
    """
    
    # Si on a des composite_cells et use_merges=True, utiliser la nouvelle fonction
    if (composite_cells is not None and n_rows is not None and n_cols is not None 
        and use_merges and table is None):
        
        # Vérifier s'il y a des cellules fusionnées
        has_merges = any((r1 - r0 > 1) or (c1 - c0 > 1) 
                        for r0, r1, c0, c1, text in composite_cells if text.strip())
        
        if has_merges:
            return export_to_html_with_merges(
                composite_cells, n_rows, n_cols, table_title, table_class, 
                cell_padding, highlight_merged, include_stats
            )
    
    # Sinon, utiliser l'ancienne méthode (grille 2D)
    # Déterminer la source de données et créer une grille 2D
    if table is not None:
        # Utiliser la grille 2D fournie
        working_table = table
        n_rows = len(table)
        n_cols = max(len(row) for row in table) if table else 0
    elif composite_cells is not None and n_rows is not None and n_cols is not None:
        # Créer une grille 2D à partir des composite_cells (comme export_to_markdown)
        working_table = fill_grid_from_composites_simple(composite_cells, n_rows, n_cols)
    else:
        raise ValueError("Vous devez fournir soit 'composite_cells' soit 'table'")
    
    # Construire le HTML
    html_lines = []
    
    # Titre et statistiques
    if table_title or include_stats:
        html_lines.append('<div class="table-header">')
        if table_title:
            html_lines.append(f'<h3>{table_title}</h3>')
        if include_stats:
            stats = _calculate_table_stats_from_table(working_table, n_rows, n_cols)
            html_lines.append(f'<div class="table-stats">{stats}</div>')
        html_lines.append('</div>')
    
    # CSS intégré avec amélioration du formatage
    html_lines.append('<style>')
    html_lines.append(f'.{table_class} {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }}')
    html_lines.append(f'.{table_class} td {{ ')
    html_lines.append(f'  border: 1px solid #ddd; ')
    html_lines.append(f'  padding: {cell_padding}px; ')
    html_lines.append(f'  vertical-align: top; ')
    html_lines.append(f'  white-space: pre-wrap; /* Préserve les espaces et retours à la ligne */ ')
    html_lines.append(f'  word-wrap: break-word; /* Coupe les mots longs */ ')
    html_lines.append(f'  line-height: 1.4; /* Améliore la lisibilité */ ')
    html_lines.append(f'}}')
    
    # Style pour les cellules remplies
    html_lines.append(f'.{table_class} .filled-cell {{ background-color: #f9f9f9; }}')
    
    # Style pour les cellules fusionnées
    if highlight_merged:
        html_lines.append(f'.{table_class} .merged-cell {{ background-color: #f0f8ff; font-weight: bold; }}')
    
    # Styles pour les paragraphes dans les cellules
    html_lines.append(f'.{table_class} td p {{ margin: 0; margin-bottom: 0.5em; }}')
    html_lines.append(f'.{table_class} td p:last-child {{ margin-bottom: 0; }}')
    
    # Style pour les retours à la ligne
    html_lines.append(f'.{table_class} td br {{ line-height: 1.6; }}')
    
    # Styles pour les statistiques
    html_lines.append('.table-stats { font-size: 0.9em; color: #666; margin-bottom: 10px; }')
    html_lines.append('.table-header { margin-bottom: 15px; }')
    html_lines.append('.table-header h3 { margin: 0; color: #333; }')
    
    html_lines.append('</style>')
    
    # Tableau principal - utilise la grille 2D simple
    html_lines.append(f'<table class="{table_class}">')
    
    for r in range(n_rows):
        html_lines.append("  <tr>")
        for c in range(n_cols):
            # Obtenir le contenu de la cellule
            cell_content = ""
            if r < len(working_table) and c < len(working_table[r]):
                cell_content = working_table[r][c]
            
            # Échapper le HTML dans le texte
            escaped_text = _escape_html(cell_content)
            
            # Classe CSS pour les cellules non vides
            css_classes = []
            if cell_content.strip():
                css_classes.append("filled-cell")
            
            # Construire la cellule
            attrs_str = ""
            if css_classes:
                attrs_str = f' class="{" ".join(css_classes)}"'
            
            html_lines.append(f'    <td{attrs_str}>{escaped_text}</td>')
        
        html_lines.append("  </tr>")
    
    html_lines.append("</table>")
    
    return "\n".join(html_lines)

def _table_to_composite_cells(table):
    """
    Convertit une grille 2D en liste de composite_cells.
    """
    composite_cells = []
    n_rows = len(table)
    n_cols = max(len(row) for row in table) if table else 0
    
    for r in range(n_rows):
        for c in range(min(len(table[r]), n_cols)):
            text = table[r][c]
            if text.strip():
                composite_cells.append((r, r+1, c, c+1, text.strip()))
    
    return composite_cells

def _calculate_table_stats(composite_cells, n_rows, n_cols):
    """
    Calcule des statistiques sur le tableau.
    """
    if not composite_cells:
        return f"Tableau vide ({n_rows}×{n_cols})"
    
    total_cells = n_rows * n_cols
    filled_cells = len([cell for cell in composite_cells if cell[4].strip()])
    merged_cells = len([cell for cell in composite_cells 
                       if (cell[1] - cell[0] > 1) or (cell[3] - cell[2] > 1)])
    
    total_chars = sum(len(cell[4]) for cell in composite_cells)
    
    return (f"Grille {n_rows}×{n_cols} | "
            f"Cellules remplies: {filled_cells}/{total_cells} | "
            f"Fusions: {merged_cells} | "
            f"Caractères: {total_chars}")

def _escape_html(text):
    """
    Échappe les caractères spéciaux HTML et améliore l'affichage des retours à la ligne.
    
    Args:
        text: Texte à échapper
    
    Returns:
        str: Texte HTML sécurisé avec formatage amélioré
    """
    if not text:
        return ""
    
    # Étape 1: Échapper les caractères spéciaux HTML
    escaped = (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    # Étape 2: Améliorer l'affichage des retours à la ligne
    # Remplacer les doubles retours à la ligne par des paragraphes
    escaped = escaped.replace("\n\n", "</p><p>")
    
    # Remplacer les retours à la ligne simples par des <br> avec espacement
    escaped = escaped.replace("\n", "<br>\n")
    
    # Préserver les espaces multiples (utile pour l'alignement de données)
    escaped = escaped.replace("  ", "&nbsp;&nbsp;")
    
    # Si on a des paragraphes, les encapsuler correctement
    if "</p><p>" in escaped:
        escaped = f"<p>{escaped}</p>"
    
    # Nettoyer les paragraphes vides
    escaped = escaped.replace("<p></p>", "")
    escaped = escaped.replace("<p><br>\n</p>", "")
    
    return escaped

def sharpen_laplacian(image):
    """Augmente la netteté en ajoutant les bords laplaciens."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(gray - 0.3 * laplacian)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
def sharpen_kernel(image):
    """Sharpen avec un noyau 3x3 classique."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return cv2.filter2D(image, -1, kernel)


def sharpen_unsharp_mask(image, alpha=1.5, beta=-0.5, sigma=1.0):
    """Applique un filtre Unsharp Mask pour renforcer la netteté.
    
    Params:
    - image: image numpy uint8 (BGR ou RGB)
    - alpha: poids de l'image originale (typiquement 1.0-2.0)
    - beta: poids négatif de l'image floutée (typiquement -0.5 à -1.0)
    - sigma: écart-type du flou gaussien (typiquement 1.0-2.0)
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def pil_to_cv2(pil_img):
    """Convertit une image PIL en image OpenCV (numpy array BGR)."""
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")  # Supprime l'alpha si présent
    cv_img = np.array(pil_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def cv2_to_pil(cv_img):
    """Convertit une image OpenCV (numpy array BGR) en image PIL RGB."""
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img_rgb)

def show_images_side_by_side(img1, img2, title1="Original", title2="Sharpened"):
    """Affiche 2 images PIL côte à côte pour comparaison."""
    plt.figure(figsize=(24,12))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis('off')
    plt.show()

# --- UTILISATION ---
def utils_show_side_by_side(image_name, folder_path):
    #image_path = "ton_image.png"  # <-- remplace par ton chemin d'image
    pil_img = Image.open("input/"+image_name).convert("RGB")

    cv_img = pil_to_cv2(pil_img)
    alpha= 1.5
    sharpened_cv_img = sharpen_unsharp_mask(cv_img, alpha=alpha, beta=1-alpha, sigma=2)
    alpha= 1.6
    sharpened_cv_img_2 = sharpen_unsharp_mask(cv_img, alpha=alpha, beta=1-alpha, sigma=2)

    #sharpened_cv_img = sharpen_laplacian(sharpened_cv_img)
    #sharpened_cv_img_2 = sharpen_kernel(sharpened_cv_img)

    sharpened_pil = cv2_to_pil(sharpened_cv_img)
    sharpened_pil_2 = cv2_to_pil(sharpened_cv_img_2)

    original = Image.open(os.path.join(folder_path,image_name))
    processed = auto_crop_deskew_enhance(original)
    processed = apply_clahe(processed)
    processed_shaprened = sharpen_adaptive(processed, radius=1.0, amount=1.2)
    processed_shaprened.save(f"Preprocessed/{image_name.replace('.png', '').replace('.jpg', '')}_preprocessed_manual.png", format='PNG', compress_level=0)

    original = Image.open(os.path.join(folder_path,image_name))
    processed_shaprened_load = Image.open(f"Preprocessed/{image_name.replace('.png', '').replace('.jpg', '')}_preprocessed_manual.png")

    show_images_side_by_side(processed_shaprened_load, original)

# === NOUVELLES FONCTIONS AMÉLIORÉES AVEC STRATÉGIE EN DEUX TEMPS ===

def build_composite_cells_advanced(cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance=5):
    """
    Version améliorée avec stratégie en deux temps :
    1. Placement théorique initial de chaque texte
    2. Détection des fusions nécessaires et ordonnancement intelligent
    
    Args:
        cell_boxes: Boxes des cellules de layout 
        rec_boxes: Boxes des textes OCR
        rec_texts: Textes reconnus
        row_lines, col_lines: Lignes de grille
        tolerance: Tolérance pour le placement
    
    Returns:
        Liste des cellules composites avec textes ordonnés
    """
    
    # ÉTAPE 1: Placement théorique initial
    initial_placement, unmatched_texts = _create_initial_text_placement(
        cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance
    )
    
    # ÉTAPE 2: Détection des fusions nécessaires  
    fusion_groups = _detect_fusion_requirements(
        initial_placement, cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance
    )
    
    # ÉTAPE 3: Création des cellules composites avec ordonnancement intelligent
    composite_cells = _create_ordered_composite_cells(
        fusion_groups, rec_boxes, rec_texts, row_lines, col_lines
    )
    
    return composite_cells

def _create_initial_text_placement(cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance):
    """
    ÉTAPE 1: Placement théorique initial - associe chaque texte à sa cellule la plus proche.
    """
    from collections import defaultdict
    
    # Structure: cell_grid_id -> {texts: [], boxes: [], original_cell_box: box}
    placement_map = defaultdict(lambda: {"texts": [], "boxes": [], "original_cell_box": None})
    unmatched_texts = []
    
    # Pour chaque cellule de layout, déterminer ses coordonnées de grille
    cell_grid_mapping = {}  # cell_box -> (r0, r1, c0, c1)
    for i, cell_box in enumerate(cell_boxes):
        r0, r1, c0, c1 = get_cell_index_ranges(cell_box, row_lines, col_lines)
        cell_grid_id = (r0, r1, c0, c1)
        cell_grid_mapping[i] = cell_grid_id
        placement_map[cell_grid_id]["original_cell_box"] = cell_box
    
    # Pour chaque texte OCR, trouver la cellule la plus proche
    for rec_box, text in zip(rec_boxes, rec_texts):
        best_cell_id = _find_best_matching_cell(
            rec_box, cell_boxes, cell_grid_mapping, tolerance
        )
        
        if best_cell_id is not None:
            placement_map[best_cell_id]["texts"].append(text.strip())
            placement_map[best_cell_id]["boxes"].append(rec_box)
        else:
            unmatched_texts.append((rec_box, text))
            print(f"[!] Texte non apparié : '{text}' (box = {rec_box})")
    
    return dict(placement_map), unmatched_texts

def _find_best_matching_cell(rec_box, cell_boxes, cell_grid_mapping, tolerance):
    """
    Trouve la cellule la plus proche d'un texte OCR selon plusieurs critères.
    """
    x_min, y_min, x_max, y_max = rec_box
    rec_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    best_cell_id = None
    best_score = float('inf')
    
    for i, cell_box in enumerate(cell_boxes):
        cx_min, cy_min, cx_max, cy_max = cell_box
        cell_center = ((cx_min + cx_max) / 2, (cy_min + cy_max) / 2)
        
        # Critère 1: Le texte est-il contenu dans la cellule (avec tolérance) ?
        is_contained = (x_min >= cx_min - tolerance and x_max <= cx_max + tolerance and
                       y_min >= cy_min - tolerance and y_max <= cy_max + tolerance)
        
        # Critère 2: Distance entre centres
        center_distance = ((rec_center[0] - cell_center[0])**2 + 
                          (rec_center[1] - cell_center[1])**2)**0.5
        
        # Critère 3: Recouvrement des zones
        overlap_area = _calculate_overlap_area(rec_box, cell_box)
        
        # Score composite (plus faible = meilleur)
        if is_contained:
            score = center_distance * 0.1  # Bonus pour containment
        else:
            score = center_distance + (1.0 / (overlap_area + 1e-6))
        
        if score < best_score:
            best_score = score
            best_cell_id = cell_grid_mapping[i]
    
    return best_cell_id

def _calculate_overlap_area(box1, box2):
    """Calcule l'aire de recouvrement entre deux boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    return x_overlap * y_overlap

def _detect_fusion_requirements(initial_placement, cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance):
    """
    ÉTAPE 2: Détecte quelles cellules doivent être fusionnées selon les textes qui débordent.
    """
    fusion_groups = []
    processed_cells = set()
    
    for cell_id, placement_data in initial_placement.items():
        if cell_id in processed_cells:
            continue
            
        # Analyser si cette cellule nécessite une fusion
        fusion_group = _analyze_cell_fusion_needs(
            cell_id, placement_data, initial_placement, row_lines, col_lines, tolerance
        )
        
        if fusion_group:
            fusion_groups.append(fusion_group)
            processed_cells.update(fusion_group["cells"])
        else:
            # Cellule simple sans fusion
            fusion_groups.append({
                "cells": [cell_id],
                "merged_span": cell_id,
                "texts": placement_data["texts"],
                "boxes": placement_data["boxes"]
            })
            processed_cells.add(cell_id)
    
    return fusion_groups

def _analyze_cell_fusion_needs(cell_id, placement_data, all_placements, row_lines, col_lines, tolerance):
    """
    Analyse si une cellule nécessite une fusion avec ses voisines.
    Version améliorée qui détecte aussi les opportunités de consolidation.
    """
    if not placement_data["texts"]:
        return None
    
    r0, r1, c0, c1 = cell_id
    cells_to_merge = [cell_id]
    all_texts = placement_data["texts"][:]
    all_boxes = placement_data["boxes"][:]
    
    # NOUVEAU: Vérifier les fusions basées sur les cellules adjacentes vides
    adjacent_empty_cells = _find_adjacent_empty_cells(cell_id, all_placements, row_lines, col_lines)
    
    # Si on a des cellules adjacentes vides, on peut envisager une fusion
    if adjacent_empty_cells:
        for adj_cell in adjacent_empty_cells:
            if adj_cell not in cells_to_merge:
                cells_to_merge.append(adj_cell)
    
    # Vérifier les textes qui pourraient déborder (logique originale)
    for text_box in placement_data["boxes"]:
        expansion_needed = _check_text_expansion_needs(
            text_box, cell_id, row_lines, col_lines, tolerance
        )
        
        if expansion_needed:
            additional_cells = _find_cells_for_expansion(
                expansion_needed, all_placements, cell_id
            )
            
            for add_cell in additional_cells:
                if add_cell not in cells_to_merge:
                    cells_to_merge.append(add_cell)
                    if add_cell in all_placements:
                        all_texts.extend(all_placements[add_cell]["texts"])
                        all_boxes.extend(all_placements[add_cell]["boxes"])
    
    # NOUVEAU: Vérifier si on peut fusionner avec des cellules voisines ayant du contenu similaire
    similar_adjacent_cells = _find_similar_adjacent_cells(
        cell_id, placement_data, all_placements, row_lines, col_lines
    )
    
    for sim_cell in similar_adjacent_cells:
        if sim_cell not in cells_to_merge:
            cells_to_merge.append(sim_cell)
            if sim_cell in all_placements:
                all_texts.extend(all_placements[sim_cell]["texts"])
                all_boxes.extend(all_placements[sim_cell]["boxes"])
    
    if len(cells_to_merge) > 1:
        # Calculer la span fusionnée
        all_r0 = min(cell[0] for cell in cells_to_merge)
        all_r1 = max(cell[1] for cell in cells_to_merge) 
        all_c0 = min(cell[2] for cell in cells_to_merge)
        all_c1 = max(cell[3] for cell in cells_to_merge)
        
        return {
            "cells": cells_to_merge,
            "merged_span": (all_r0, all_r1, all_c0, all_c1),
            "texts": all_texts,
            "boxes": all_boxes
        }
    
    return None

def _find_adjacent_empty_cells(cell_id, all_placements, row_lines, col_lines):
    """
    Trouve les cellules adjacentes vides qui pourraient être fusionnées.
    """
    r0, r1, c0, c1 = cell_id
    adjacent_cells = []
    
    # Cellules adjacentes possibles
    candidates = [
        (r0, r1, c0-1, c0),      # Gauche
        (r0, r1, c1, c1+1),      # Droite  
        (r0-1, r0, c0, c1),      # Haut
        (r1, r1+1, c0, c1),      # Bas
    ]
    
    for candidate in candidates:
        cr0, cr1, cc0, cc1 = candidate
        # Vérifier que la cellule candidate est dans les limites
        if (cr0 >= 0 and cr1 <= len(row_lines) and 
            cc0 >= 0 and cc1 <= len(col_lines)):
            # Vérifier si cette cellule est vide (pas de texte)
            if candidate in all_placements:
                if not all_placements[candidate]["texts"]:
                    adjacent_cells.append(candidate)
    
    return adjacent_cells

def _find_similar_adjacent_cells(cell_id, placement_data, all_placements, row_lines, col_lines):
    """
    Trouve les cellules adjacentes avec du contenu similaire (ex: même catégorie).
    """
    r0, r1, c0, c1 = cell_id
    similar_cells = []
    
    # Cellules adjacentes possibles
    candidates = [
        (r0, r1, c0-1, c0),      # Gauche
        (r0, r1, c1, c1+1),      # Droite  
        (r0-1, r0, c0, c1),      # Haut
        (r1, r1+1, c0, c1),      # Bas
    ]
    
    current_texts = placement_data["texts"]
    
    for candidate in candidates:
        cr0, cr1, cc0, cc1 = candidate
        # Vérifier que la cellule candidate est dans les limites
        if (cr0 >= 0 and cr1 <= len(row_lines) and 
            cc0 >= 0 and cc1 <= len(col_lines)):
            
            if candidate in all_placements:
                candidate_texts = all_placements[candidate]["texts"]
                
                # Vérifier si les textes sont "similaires" (heuristique simple)
                if candidate_texts and _are_texts_similar(current_texts, candidate_texts):
                    similar_cells.append(candidate)
    
    return similar_cells

def _are_texts_similar(texts1, texts2):
    """
    Heuristique simple pour détecter si deux ensembles de textes sont similaires.
    """
    if not texts1 or not texts2:
        return False
    
    # Heuristiques possibles :
    # 1. Textes très courts (probablement des labels)
    avg_len1 = sum(len(t) for t in texts1) / len(texts1)
    avg_len2 = sum(len(t) for t in texts2) / len(texts2)
    
    if avg_len1 < 5 and avg_len2 < 5:
        return True
    
    # 2. Textes numériques (probablement des valeurs)
    def is_mostly_numeric(text):
        return sum(c.isdigit() or c in '.,% -' for c in text) / len(text) > 0.7
    
    text1_combined = " ".join(texts1)
    text2_combined = " ".join(texts2)
    
    if is_mostly_numeric(text1_combined) and is_mostly_numeric(text2_combined):
        return True
    
    # 3. Textes avec des mots clés similaires
    keywords1 = set(w.lower() for w in text1_combined.split() if len(w) > 2)
    keywords2 = set(w.lower() for w in text2_combined.split() if len(w) > 2)
    
    if keywords1 and keywords2:
        similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        return similarity > 0.3
    
    return False

def _check_text_expansion_needs(text_box, cell_id, row_lines, col_lines, tolerance):
    """
    Vérifie si un texte nécessite une expansion de sa cellule.
    """
    tx_min, ty_min, tx_max, ty_max = text_box
    r0, r1, c0, c1 = cell_id
    
    # Calculer les bordures de la cellule selon les lignes de grille
    cell_y_min = row_lines[r0] if r0 < len(row_lines) else row_lines[-1]
    cell_y_max = row_lines[r1] if r1 < len(row_lines) else row_lines[-1]  
    cell_x_min = col_lines[c0] if c0 < len(col_lines) else col_lines[-1]
    cell_x_max = col_lines[c1] if c1 < len(col_lines) else col_lines[-1]
    
    expansion = {}
    
    # Vérifier débordement horizontal
    if tx_min < cell_x_min - tolerance:
        expansion["left"] = True
    if tx_max > cell_x_max + tolerance:
        expansion["right"] = True
        
    # Vérifier débordement vertical
    if ty_min < cell_y_min - tolerance:
        expansion["up"] = True
    if ty_max > cell_y_max + tolerance:
        expansion["down"] = True
    
    return expansion if expansion else None

def _find_cells_for_expansion(expansion_needed, all_placements, base_cell_id):
    """
    Trouve les cellules adjacentes nécessaires pour l'expansion.
    """
    r0, r1, c0, c1 = base_cell_id
    additional_cells = []
    
    for direction in expansion_needed.keys():
        if direction == "left" and c0 > 0:
            additional_cells.append((r0, r1, c0-1, c0))
        elif direction == "right":
            additional_cells.append((r0, r1, c1, c1+1))
        elif direction == "up" and r0 > 0:
            additional_cells.append((r0-1, r0, c0, c1))
        elif direction == "down":
            additional_cells.append((r1, r1+1, c0, c1))
    
    # Filtrer pour ne garder que les cellules qui existent
    valid_cells = []
    for cell in additional_cells:
        if cell in all_placements:
            valid_cells.append(cell)
    
    return valid_cells

def _create_ordered_composite_cells(fusion_groups, rec_boxes, rec_texts, row_lines, col_lines):
    """
    ÉTAPE 3: Crée les cellules composites finales avec ordonnancement intelligent des textes.
    """
    composite_cells = []
    
    for group in fusion_groups:
        r0, r1, c0, c1 = group["merged_span"]
        
        # Ordonner les textes selon leur position spatiale
        merged_text = _order_texts_spatially(group["texts"], group["boxes"])
        
        composite_cells.append((r0, r1, c0, c1, merged_text))
    
    return composite_cells

def _order_texts_spatially(texts, boxes):
    """
    Ordonne les textes selon leur position : gauche→droite, puis haut→bas.
    Retourne les textes avec séparateurs appropriés (espaces ou retours à la ligne).
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    # Créer des paires (texte, box) avec coordonnées
    text_positions = []
    for text, box in zip(texts, boxes):
        x_min, y_min, x_max, y_max = box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        text_positions.append((text, box, center_x, center_y))
    
    # Si pas de positions, retourner chaîne vide
    if not text_positions:
        return ""
    
    # Trier par ligne (y), puis par colonne (x)
    # Utiliser une tolérance pour les "lignes" (textes à peu près à la même hauteur)
    y_tolerance = 10
    
    # Grouper par lignes approximatives
    lines = []
    text_positions.sort(key=lambda x: x[3])  # Trier par y d'abord
    
    current_line = [text_positions[0]]
    current_y = text_positions[0][3]
    
    for item in text_positions[1:]:
        if abs(item[3] - current_y) <= y_tolerance:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item[3]
    
    if current_line:
        lines.append(current_line)
    
    # Dans chaque ligne, trier par x (gauche → droite), puis combiner les lignes
    ordered_text_parts = []
    for line in lines:
        line.sort(key=lambda x: x[2])  # Trier par x
        # Joindre les textes de la même ligne avec des espaces
        line_texts = [item[0] for item in line]
        ordered_text_parts.append(" ".join(line_texts))
    
    # Joindre les différentes lignes avec des retours à la ligne
    return "\n".join(ordered_text_parts)

def fill_grid_from_composites_simple(composite_cells, n_rows, n_cols):
    """
    Version simplifiée sans debug de fill_grid_from_composites_advanced.
    Conserve TOUS les textes et gère intelligemment les conflits.
    """
    table = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Structure pour collecter tous les textes par position de grille finale
    grid_texts = {}  # (r, c) -> liste de (texte, priorité, positions_originales)
    
    # ÉTAPE 1: Collecter tous les textes avec leur position de grille
    for i, (r0, r1, c0, c1, text) in enumerate(composite_cells):
        if not text.strip():
            continue
            
        # Calculer la priorité basée sur la position (haut-gauche = priorité haute)
        priority = r0 * 1000 + c0  # Plus petit = plus prioritaire
        
        # Déterminer la position de placement dans la grille finale
        target_r = max(0, min(r0, n_rows - 1))
        target_c = max(0, min(c0, n_cols - 1))
        
        # Ajouter le texte à la collection
        key = (target_r, target_c)
        if key not in grid_texts:
            grid_texts[key] = []
        
        grid_texts[key].append((text.strip(), priority, r0, r1, c0, c1))
    
    # ÉTAPE 2: Pour chaque position de grille, combiner les textes intelligemment
    for (target_r, target_c), texts_info in grid_texts.items():
        if not texts_info:
            continue
            
        # Trier par priorité (position haut-gauche d'abord)
        texts_info.sort(key=lambda x: x[1])
        
        # Combiner les textes selon leur nature et position
        combined_text = _combine_texts_simple(texts_info)
        
        # Placer le texte combiné dans la grille
        if 0 <= target_r < n_rows and 0 <= target_c < n_cols:
            table[target_r][target_c] = combined_text
    
    return table

def _combine_texts_simple(texts_info):
    """
    Version simplifiée de combinaison de textes avec gestion des retours à la ligne.
    """
    if not texts_info:
        return ""
    
    if len(texts_info) == 1:
        return texts_info[0][0]  # Un seul texte
    
    # Pour plusieurs textes, analyser leur position spatiale
    texts_with_pos = []
    for text, priority, r0, r1, c0, c1 in texts_info:
        center_r = (r0 + r1) / 2
        center_c = (c0 + c1) / 2
        texts_with_pos.append((text, center_r, center_c))
    
    # Trier par position : d'abord par Y (haut vers bas), puis par X (gauche vers droite)
    texts_with_pos.sort(key=lambda x: (x[1], x[2]))
    
    # Si pas de positions, retourner chaîne vide
    if not texts_with_pos:
        return ""
    
    # Grouper par lignes approximatives (même Y)
    y_tolerance = 0.5  # Tolérance pour considérer que deux textes sont sur la même ligne
    lines = []
    
    current_line = [texts_with_pos[0]]
    current_y = texts_with_pos[0][1]
    
    for item in texts_with_pos[1:]:
        if abs(item[1] - current_y) <= y_tolerance:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item[1]
    
    if current_line:
        lines.append(current_line)
    
    # Combiner les lignes avec des retours à la ligne
    line_texts = []
    for line in lines:
        # Trier par X (gauche vers droite) dans chaque ligne
        line.sort(key=lambda x: x[2])
        # Joindre les textes de la même ligne avec des espaces
        line_content = " ".join(item[0] for item in line)
        line_texts.append(line_content)
    
    # Joindre les différentes lignes avec des retours à la ligne
    return "\n".join(line_texts)

def export_to_html_with_merges(composite_cells, n_rows, n_cols, 
                              table_title=None, table_class="ocr-table", cell_padding=4,
                              highlight_merged=True, include_stats=True):
    """
    Export HTML avec gestion correcte des cellules fusionnées (colspan et rowspan).
    
    Args:
        composite_cells: Liste de cellules composites (r0, r1, c0, c1, text)
        n_rows, n_cols: Dimensions de la grille
        table_title: Titre du tableau
        table_class: Classe CSS pour le tableau
        cell_padding: Espacement des cellules
        highlight_merged: Surligner les cellules fusionnées
        include_stats: Inclure les statistiques
    
    Returns:
        str: HTML formaté avec vraies fusions de cellules
    """
    
    # Construire le HTML
    html_lines = []
    
    # Titre et statistiques
    if table_title or include_stats:
        html_lines.append('<div class="table-header">')
        if table_title:
            html_lines.append(f'<h3>{table_title}</h3>')
        if include_stats:
            stats = _calculate_table_stats(composite_cells, n_rows, n_cols)
            html_lines.append(f'<div class="table-stats">{stats}</div>')
        html_lines.append('</div>')
    
    # CSS intégré avec amélioration du formatage
    html_lines.append('<style>')
    html_lines.append(f'.{table_class} {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }}')
    html_lines.append(f'.{table_class} td {{ ')
    html_lines.append(f'  border: 1px solid #ddd; ')
    html_lines.append(f'  padding: {cell_padding}px; ')
    html_lines.append(f'  vertical-align: top; ')
    html_lines.append(f'  white-space: pre-wrap; ')
    html_lines.append(f'  word-wrap: break-word; ')
    html_lines.append(f'  line-height: 1.4; ')
    html_lines.append(f'}}')
    
    # Style pour les cellules remplies
    html_lines.append(f'.{table_class} .filled-cell {{ background-color: #f9f9f9; }}')
    
    # Style pour les cellules fusionnées
    if highlight_merged:
        html_lines.append(f'.{table_class} .merged-cell {{ background-color: #f0f8ff; font-weight: bold; border: 2px solid #007acc; }}')
    
    # Styles pour les paragraphes dans les cellules
    html_lines.append(f'.{table_class} td p {{ margin: 0; margin-bottom: 0.5em; }}')
    html_lines.append(f'.{table_class} td p:last-child {{ margin-bottom: 0; }}')
    
    # Style pour les retours à la ligne
    html_lines.append(f'.{table_class} td br {{ line-height: 1.6; }}')
    
    # Styles pour les statistiques
    html_lines.append('.table-stats { font-size: 0.9em; color: #666; margin-bottom: 10px; }')
    html_lines.append('.table-header { margin-bottom: 15px; }')
    html_lines.append('.table-header h3 { margin: 0; color: #333; }')
    
    html_lines.append('</style>')
    
    # Créer une grille pour suivre les cellules occupées
    occupied = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Créer une map de cellules par position de départ
    cell_map = {}  # (r, c) -> (r0, r1, c0, c1, text)
    
    for r0, r1, c0, c1, text in composite_cells:
        if text.strip():  # Ignorer les cellules vides
            cell_map[(r0, c0)] = (r0, r1, c0, c1, text)
            
            # Marquer toutes les positions occupées par cette cellule
            for r in range(max(0, r0), min(r1, n_rows)):
                for c in range(max(0, c0), min(c1, n_cols)):
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        occupied[r][c] = True
    
    # Générer le tableau HTML
    html_lines.append(f'<table class="{table_class}">')
    
    for r in range(n_rows):
        html_lines.append("  <tr>")
        
        for c in range(n_cols):
            # Si cette cellule est déjà occupée par une fusion précédente, l'ignorer
            if occupied[r][c] and (r, c) not in cell_map:
                continue
            
            # Vérifier s'il y a une cellule composite qui commence ici
            if (r, c) in cell_map:
                r0, r1, c0, c1, text = cell_map[(r, c)]
                
                # Calculer rowspan et colspan
                rowspan = min(r1 - r0, n_rows - r0)
                colspan = min(c1 - c0, n_cols - c0)
                
                # Échapper le HTML dans le texte
                escaped_text = _escape_html(text)
                
                # Construire les attributs de la cellule
                attrs = []
                if rowspan > 1:
                    attrs.append(f'rowspan="{rowspan}"')
                if colspan > 1:
                    attrs.append(f'colspan="{colspan}"')
                
                # Classes CSS
                css_classes = ["filled-cell"]
                if highlight_merged and (rowspan > 1 or colspan > 1):
                    css_classes.append("merged-cell")
                
                attrs.append(f'class="{" ".join(css_classes)}"')
                attrs_str = " " + " ".join(attrs) if attrs else ""
                
                html_lines.append(f'    <td{attrs_str}>{escaped_text}</td>')
                
            else:
                # Cellule vide normale
                html_lines.append(f'    <td class="empty-cell"></td>')
        
        html_lines.append("  </tr>")
    
    html_lines.append("</table>")
    
    return "\n".join(html_lines)

def _calculate_table_stats_from_table(table, n_rows, n_cols):
    """
    Calcule des statistiques sur un tableau à partir d'une grille 2D.
    
    Args:
        table: Grille 2D du tableau
        n_rows, n_cols: Dimensions du tableau
    
    Returns:
        str: Statistiques formatées
    """
    if not table:
        return f"Tableau vide ({n_rows}×{n_cols})"
    
    total_cells = n_rows * n_cols
    filled_cells = 0
    total_chars = 0
    
    # Compter les cellules remplies et les caractères
    for r in range(n_rows):
        for c in range(n_cols):
            if r < len(table) and c < len(table[r]):
                cell_content = table[r][c]
                if cell_content.strip():
                    filled_cells += 1
                    total_chars += len(cell_content)
    
    empty_cells = total_cells - filled_cells
    
    return (f"Grille {n_rows}×{n_cols} | "
            f"Cellules remplies: {filled_cells}/{total_cells} | "
            f"Cellules vides: {empty_cells} | "
            f"Caractères: {total_chars}")

def build_composite_cells_advanced_v2(cell_boxes, rec_boxes, rec_texts, row_lines, col_lines, tolerance=5):
    """
    Version finale optimisée avec placement intelligent des textes OCR.
    
    Cette version combine les meilleures techniques :
    - Placement théorique initial avec scoring multicritère
    - Filtrage des cellules vides
    - Ordonnancement spatial des textes
    - Conservation totale des textes
    
    Args:
        cell_boxes: Boxes des cellules de layout détectées
        rec_boxes: Boxes des textes OCR
        rec_texts: Textes reconnus par OCR
        row_lines: Lignes horizontales de la grille
        col_lines: Lignes verticales de la grille
        tolerance: Tolérance pour le placement des textes
    
    Returns:
        List[Tuple]: Liste des cellules composites (r0, r1, c0, c1, text)
    
    Examples:
        >>> row_lines, col_lines = build_adaptive_grid_structure(cell_boxes)
        >>> composite_cells = build_composite_cells_advanced_v2(
        ...     cell_boxes, rec_boxes, rec_texts, row_lines, col_lines
        ... )
        >>> print(f"Cellules créées: {len(composite_cells)}")
    """
    from collections import defaultdict
    
    # ÉTAPE 1: Placement théorique initial avec scoring intelligent
    cell_grid_mapping = {}  # index -> (r0, r1, c0, c1)
    placement_scores = {}  # (r0, r1, c0, c1) -> {texts: [], boxes: [], scores: []}
    
    # Mapper chaque cellule de layout à sa position de grille
    for i, cell_box in enumerate(cell_boxes):
        r0, r1, c0, c1 = get_cell_index_ranges(cell_box, row_lines, col_lines)
        cell_grid_mapping[i] = (r0, r1, c0, c1)
    
    # Pour chaque texte OCR, trouver la meilleure cellule avec scoring
    for rec_box, text in zip(rec_boxes, rec_texts):
        if not text.strip():
            continue
        
        best_cell_id = None
        best_score = float('inf')
        
        # Évaluer chaque cellule de layout
        for i, cell_box in enumerate(cell_boxes):
            score = _calculate_placement_score(rec_box, cell_box, tolerance)
            
            if score < best_score:
                best_score = score
                best_cell_id = cell_grid_mapping[i]
        
        # Placer le texte dans la meilleure cellule
        if best_cell_id is not None:
            if best_cell_id not in placement_scores:
                placement_scores[best_cell_id] = {"texts": [], "boxes": [], "scores": []}
            
            placement_scores[best_cell_id]["texts"].append(text.strip())
            placement_scores[best_cell_id]["boxes"].append(rec_box)
            placement_scores[best_cell_id]["scores"].append(best_score)
    
    # ÉTAPE 2: Filtrage et ordonnancement spatial
    composite_cells = []
    
    for (r0, r1, c0, c1), data in placement_scores.items():
        texts = data["texts"]
        boxes = data["boxes"]
        
        if not texts:  # Ignorer les cellules vides
            continue
        
        # Ordonner les textes selon leur position spatiale
        merged_text = _order_texts_spatially(texts, boxes)
        
        # Créer la cellule composite
        composite_cells.append((r0, r1, c0, c1, merged_text))
    
    return composite_cells

def _calculate_placement_score(rec_box, cell_box, tolerance):
    """
    Calcule un score de placement pour un texte OCR dans une cellule.
    
    Args:
        rec_box: Box du texte OCR [x_min, y_min, x_max, y_max]
        cell_box: Box de la cellule [x_min, y_min, x_max, y_max]
        tolerance: Tolérance pour le placement
    
    Returns:
        float: Score de placement (plus faible = meilleur)
    """
    x_min, y_min, x_max, y_max = rec_box
    cx_min, cy_min, cx_max, cy_max = cell_box
    
    # Centres des boxes
    rec_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    cell_center = ((cx_min + cx_max) / 2, (cy_min + cy_max) / 2)
    
    # Critère 1: Le texte est-il contenu dans la cellule (avec tolérance) ?
    is_contained = (x_min >= cx_min - tolerance and x_max <= cx_max + tolerance and
                   y_min >= cy_min - tolerance and y_max <= cy_max + tolerance)
    
    # Critère 2: Distance euclidienne entre centres
    center_distance = ((rec_center[0] - cell_center[0])**2 + 
                      (rec_center[1] - cell_center[1])**2)**0.5
    
    # Critère 3: Aire de recouvrement
    overlap_area = _calculate_overlap_area(rec_box, cell_box)
    
    # Critère 4: Taille relative (éviter les cellules trop petites ou trop grandes)
    rec_area = (x_max - x_min) * (y_max - y_min)
    cell_area = (cx_max - cx_min) * (cy_max - cy_min)
    size_ratio = min(rec_area, cell_area) / max(rec_area, cell_area) if max(rec_area, cell_area) > 0 else 0
    
    # Score composite (plus faible = meilleur)
    if is_contained:
        # Bonus important pour containment
        score = center_distance * 0.1 + (1.0 - size_ratio) * 0.5
    else:
        # Pénalité pour non-containment
        score = center_distance + (1.0 / (overlap_area + 1e-6)) + (1.0 - size_ratio) * 2.0
    
    return score
