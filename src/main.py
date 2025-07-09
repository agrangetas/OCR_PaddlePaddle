## Init:
print("📦 Import PaddleOCR...")
from paddleocr import PPStructureV3
print("✅ Import réussi")

print("🔄 Initialisation du pipeline...")
pipeline = PPStructureV3(layout_detection_model_name="PP-DocLayout-L")#PP-StructureLayout")#"PP-DocLayout-L")
print("✅ Pipeline initialisé avec succès")

import gc
import os
import cv2
import json
import csv
from pathlib import Path
from paddleocr import PPStructureV3
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from IPython.display import Markdown, display
import numpy as np
import time


def pil_to_cv2(pil_img):
    """Convertit une image PIL en image OpenCV (numpy array BGR)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    """Convertit une image OpenCV (numpy array BGR) en image PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def apply_clahe(pil_img):
    """Améliore localement le contraste (CLAHE)."""
    gray = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced).convert("RGB")


def sharpen_adaptive(pil_img, radius=1, amount=0.6):
    """Améliore la netteté avec un filtre Unsharp mask doux."""
    img = pil_to_cv2(pil_img)
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return cv2_to_pil(sharpened)

    
def auto_crop_deskew_enhance(pil_img):
    """Pipeline OCR : correction d’angle limitée à ±20°, crop, contraste, netteté."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Ajustement de l'angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Limiter l'angle à ±20°
    if abs(angle) > 20:
        angle = 0
    

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Post deskew → binarize + crop
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    _, binary_rotated = cv2.threshold(gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary_rotated)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = rotated[y:y+h, x:x+w]

    return cv2_to_pil(cropped)


def predict_safe(pipeline,  folder_path, image_name):
    """Prédiction sécurisée avec prétraitement OCR."""
    try:
        full_path = os.path.join(folder_path,image_name)
        print(f"🔄 Prédiction sécurisée: {full_path}")
        if not os.path.exists(full_path):
            print(f"❌ Image non trouvée: {full_path}")
            return None

        original = Image.open(full_path)
        processed = auto_crop_deskew_enhance(original)
        processed = apply_clahe(processed)
        processed_shaprened = sharpen_adaptive(processed, radius=1.0, amount=1.2)

        processed_shaprened.save(f"Preprocessed/{image_name.replace('.png', '').replace('.jpg', '')}_preprocessed.png", format='PNG', compress_level=0)

        # Redimensionnement
        if max(processed.size) > 1024:
            ratio = 1024 / max(processed.size)
            new_size = (int(processed.size[0] * ratio), int(processed.size[1] * ratio))
            processed = processed.resize(new_size, Image.Resampling.LANCZOS)
            print(f"📏 Image redimensionnée: {new_size}")

        temp_path = "/tmp/safe_input.png"
        processed.save(temp_path, "PNG")

        gc.collect()
        print("🧠 Exécution du modèle...")
        result = pipeline.predict(temp_path)
        print("✅ Prédiction réussie !")

        return result, processed

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        gc.collect()

def create_dirs_if_not_exist(dirs):
    """
    Crée les dossiers listés dans `dirs` s'ils n'existent pas.

    Args:
        dirs (list of str): Liste de chemins de dossiers à créer.
    """
    for folder in dirs:
        os.makedirs(folder, exist_ok=True)


def main_apply_pipeline(image_name, folder_path = "input/" ):
    # === PIPELINE PaddleOCR ===
    start_time = time.time()
    results, img = predict_safe(pipeline, folder_path, image_name)
    print('calculé en ', time.time() - start_time, 'secondes')


    img_name = image_name.split('/')[-1].replace('.','_')

    # === PARAMÈTRES ===
    output_dir = Path("output/"+img_name)
    crop_dir = output_dir / "crops"
    layout_vis_path = output_dir / "layout_vis.jpg"
    markdown_path = output_dir / "safe_input.md"
    json_path = output_dir / "result.json"
    csv_path = output_dir / "zones_summary.csv"


    folders = ["output/"+img_name, "output/"+img_name+"/crops", "output/"+img_name+"/imgs","output/"+img_name+"/json"]
    create_dirs_if_not_exist(folders)

    for res in results:
        res.save_to_img("output/"+img_name+"/imgs/")
        res.save_to_json("output/"+img_name+"/json/")
        
    # Créer les dossiers nécessaires
    for folder in ["table", "figure", "text", "others","header","table_title"]:
        os.makedirs(crop_dir / folder, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    # Charger l'image d'origine
    #image = cv2.imread(image_path)
    img_np = np.array(img) 
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Si tu veux ensuite repasser en RGB pour PIL, ce n’est pas utile ici
    # Mais dans ton code, tu voulais convertir BGR -> RGB, donc :
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ou simplement utiliser img_np directement si tu veux afficher l'image PIL avec Image.fromarray:
    vis_image = Image.fromarray(img_np)

    draw = ImageDraw.Draw(vis_image)
    zone_data = []
    all_texts = []

    # === PARCOURIR LES ZONES ===
    for page_idx, result in enumerate(results):
        for i, block in enumerate(result["layout_det_res"]["boxes"]):
            label = block["label"]
            bbox = list(map(int, block["coordinate"]))  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # Découper l'image
            crop = img_np[y1:y2, x1:x2]
            folder = label if label in ["text", "table", "figure","header","table_title"] else "others"
            filename = f"{label}_{page_idx}_{i}.jpg"
            cv2.imwrite(str(crop_dir / folder / filename), crop)

            # Sauvegarde JSON des coordonnées
            metadata = {
                "label": label,
                "bbox": bbox,
                "image_crop": str(crop_dir / folder / filename),
                "rec_texts": block.get("res", "")
            }
            with open(crop_dir / folder / f"{filename}.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Données CSV
            zone_data.append([
                page_idx,
                label,
                bbox[0], bbox[1], bbox[2], bbox[3],
                block.get("res", "")
            ])

            # Dessiner sur image
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((x1 + 3, y1 + 3), label, fill="red")

    # === ENREGISTRER VISUALISATION AVEC LAYOUT ===
    vis_image.save(layout_vis_path)

    # === SAUVEGARDE JSON + MARKDOWN ===
    for page in results:
        # page.save_to_json(save_path=output_dir)
        # page.save_to_markdown(save_path=output_dir)
        page.save_all(save_path=output_dir)

    # === SAUVEGARDE CSV DES BLOCS ===
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "label", "x1", "y1", "x2", "y2", "rec_texts"])
        writer.writerows(zone_data)

   
    print("\n✅ Traitement terminé. Résultats enregistrés dans le dossier 'output/'")
    print("📷 Aperçu de la mise en page détectée : output/layout_vis.jpg")
    print("📋 Markdown généré : output/result.md\n")
    
    return results, img