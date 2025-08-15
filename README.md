# retinal-disease-classification
Retinal Disease Classification with Fundus Images

## 1. Overview
Classify retinal diseases (e.g., DR, AMD, Glaucoma, Normal) from color fundus images using deep learning (transfer learning + augmentations).  
Focus: clean pipeline, reproducibility, extendability.

## 2. Features
- Config-driven training (hyperparams, augmentations, optimizer)
- Transfer learning (e.g., EfficientNet / ResNet)
- Stratified splits + reproducible seeds
- Metrics: accuracy, per-class F1, confusion matrix
- Optional mixed precision + early stopping
- Export to ONNX / TorchScript (inference ready)

## 3. Dataset
Expected structure (example):
```
data/
  raw/
    train/
      DR/
      AMD/
      Glaucoma/
      Normal/
    val/
    test/
```
Public sources you may adapt: APTOS 2019, DDR, Messidor (respect licenses).

## 4. Quick Start
```
git clone <repo-url>
cd retinal-disease-classification
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 5. Configuration (example)
```
configs/
  default.yaml
    model: efficientnet_b3
    image_size: 300
    batch_size: 16
    epochs: 30
    optimizer:
      name: adamw
      lr: 3e-4
    augment:
      horizontal_flip: true
      random_crop: true
```

## 6. Training
```
python train.py \
  --data-root ./data/raw/train \
  --val-root ./data/raw/val \
  --config configs/default.yaml \
  --out-dir runs/exp1
```
Resume:
```
python train.py --resume runs/exp1/checkpoints/last.ckpt
```

## 7. Evaluation
```
python eval.py --checkpoint runs/exp1/checkpoints/best.ckpt --test-root ./data/raw/test
```
Outputs: metrics.json, confusion_matrix.png.

## 8. Inference
```
python predict.py --checkpoint runs/exp1/checkpoints/best.ckpt --image path/to/fundus.jpg
```
Batch:
```
python predict.py --checkpoint ... --input-dir samples/ --output preds.csv
```

## 9. Export
```
python export.py --checkpoint runs/exp1/checkpoints/best.ckpt --format onnx --out model.onnx
```

## 10. Logging
Recommended: integrate TensorBoard or Weights & Biases:
```
--loggers tensorboard
```

## 11. Project Structure (proposed)
```
retinal_disease_classification/
  data_loader.py
  datasets.py
  transforms.py
  models/
    __init__.py
    efficientnet.py
  train.py
  eval.py
  predict.py
  export.py
  utils/
    metrics.py
    seed.py
configs/
runs/
```

## 12. Reproducibility
Set seeds (numpy, torch, random) and enable deterministic flags where possible:
```
python train.py --seed 42 --deterministic
```

## 13. Benchmarks (placeholder)
| Model | Img Size | Acc | Macro F1 | Params |
|-------|----------|-----|----------|--------|
| EfficientNet-B3 | 300 | TBD | TBD | 12M |

(Add after first successful run.)

## 14. Future Work
- Multi-label pathology scoring
- Lesion localization (attention / CAMs)
- Ensemble blending
- Domain adaptation

## 15. Contributing
1. Fork + branch (feat/short-name)
2. Add/modify tests if logic changes
3. Run lint + format
4. PR with concise description

## 16. License
Add your chosen license (e.g., MIT) in LICENSE file.

## 17. Citation
If you publish with this codebase:
```
@misc{retinal-disease-classification,
  title  = {Retinal Disease Classification},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/your/repo}
}
```

## 18. Disclaimer
Not for clinical use. Educational / research only.
