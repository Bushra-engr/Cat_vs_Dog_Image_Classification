# 🐾 Cat vs Dog Image Classification

Binary image classification using **MobileNetV2** transfer learning.

---

## Model
- **Base:** MobileNetV2 (frozen, ImageNet weights)
- **Head:** GlobalAvgPool → Dense(128, relu) → BatchNorm → Dropout(0.5) → Dense(2, softmax)
- **Input:** 128 × 128 × 3
- **Loss:** sparse_categorical_crossentropy

---

## Dataset
- **Classes:** Cat, Dog
- **Split:** 80% train / 20% validation
- **Batch size:** 32

---

## Results
| Metric | Value |
|---|---|
| Train Accuracy | ~92%+ |
| Val Accuracy | ~90%+ |
| Epochs | 15 |

---

## Tech Stack
```
TensorFlow 2.21.0 | Streamlit >=1.35.0 | NumPy | Pillow
```

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
