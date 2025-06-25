# Configuration PaddleOCR - Versions Validées

## ✅ Versions qui fonctionnent (testées et validées)

### PaddleOCR Stack
- **PaddlePaddle**: `3.0.0`
- **PaddleOCR**: `3.0.0` 
- **PaddleX**: `3.0.0`
- **tokenizers**: `0.19.1` (installé automatiquement par PaddleX)

### Dépendances principales
- **Python**: `3.9`
- **numpy**: `1.24.4`
- **pandas**: `1.5.3`
- **opencv-contrib-python**: `4.10.0.84`
- **scikit-learn**: `1.3.2`

## ⚠️ Versions à éviter

### Problématiques identifiées
- **PaddleOCR 3.0.2** + **tokenizers 0.21.2** → Crash du kernel
- **PaddleOCR 3.0.2** + **PaddleX 3.0.2** → Incompatibilité avec tokenizers
- **Multiples versions d'OpenCV** → Conflits potentiels

## 🔧 Configuration finale

### Dockerfile.jupyter
```dockerfile
# PaddlePaddle 3.0.0 depuis le dépôt officiel chinois
RUN pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# PaddleOCR 3.0.0 avec dépendances manuelles
RUN pip install --no-cache-dir chardet colorlog prettytable py-cpuinfo \
    && pip install --no-cache-dir ruamel.yaml ujson einops ftfy GPUtil \
    && pip install --no-cache-dir regex tiktoken \
    && pip install --no-cache-dir imagesize premailer shapely \
    && pip install --no-cache-dir paddleocr==3.0.0 --no-deps \
    && pip install --no-cache-dir "paddlex[ocr]==3.0.0"
```

### requirements.txt
```txt
# Versions compatibles avec PaddleX 3.0.0
pandas==1.5.3
numpy==1.24.4
opencv-contrib-python==4.10.0.84
scikit-learn==1.3.2
```

## ✅ Code utilisateur validé

```python
from paddleocr import PaddleOCR

# Configuration qui fonctionne
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Import des modules avancés
from paddleocr import LayoutDetection, PPStructureV3

# LayoutDetection
model = LayoutDetection(model_name="PP-DocLayout-L")
output = model.predict("image.png", batch_size=1)

# PPStructureV3
pipeline = PPStructureV3(layout_detection_model_name="PP-DocLayout-L")
output = pipeline.predict("image.png")
```

## 📝 Notes importantes

1. **Cache des modèles**: Les modèles sont téléchargés dans `/app/.paddlex/official_models` au premier usage
2. **Modèles utilisés**: `PP-OCRv5_mobile_det` et `PP-OCRv5_mobile_rec` (versions légères)
3. **Mémoire**: Configuration optimisée pour éviter les crashes
4. **Compatibilité**: Toutes les versions ont été testées ensemble

## 🚀 Statut
- ✅ Installation validée
- ✅ Import PaddleOCR OK
- ✅ Initialisation PaddleOCR OK  
- ✅ LayoutDetection OK
- ✅ PPStructureV3 OK
- ✅ Pas de crash du kernel

**Dernière validation**: $(date)
**Environnement**: Docker Python 3.9 