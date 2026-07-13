# Weather Image Classifier

A two-stage image classifier that answers two questions about a photo:

1. **Is this a weather photo at all?** — an input *gate* rejects images that are not weather (a laptop, a portrait, a clear empty landscape).
2. **Which weather is it?** — if the image passes the gate, the model predicts the weather class and shows the top-3 most likely classes with probabilities.

It is a capstone project built end-to-end: EDA → transfer learning → out-of-distribution (OOD) gate → evaluation → a Gradio demo you can run locally on CPU.

---

## Repo layout

```
weather-recognizer/
├── README.md
├── app.py
├── custom.css
├── notebooks/
│   └── weather_classifier.ipynb     # full pipeline: EDA → train → eval → gate
└── models/
    ├── weather_classifier.keras     # trained model
    └── gate_config.json             # method, threshold, class names, OOD metrics
```

---

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Upload a photo -> the gate decides *weather / not weather* -> if weather, you get the top-3 classes with probabilities. Runs on CPU. The trained weights (`models/weather_classifier.keras`) and `models/gate_config.json` are included.

---

## Approach

```
                                          ┌─ gate score < τ ──►  "Not a weather photo"
photo ─► resize 224² ─► EfficientNet-B0 ─► logits ─► MSP gate ─┤
        + preprocess     (fine-tuned)                          └─ gate score ≥ τ ──►  top-3 weather classes + probs
```

- **Backbone:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on the weather dataset (transfer learning, nothing trained from scratch).
- **Gate:** Maximum Softmax Probability (MSP) of the model's own output. If the model is not confident about *any* weather class, the image is probably not weather. A threshold `τ` calibrated to keep 95 % of real weather decides accept/reject.
- **One model, deep pipeline** — deliberately no backbone comparison; the goal is to understand every step.

---

## Dataset

- **Source:** [`jehanbhathena/weather-dataset`](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) (Kaggle), ~6,862 images.
- **11 classes:** `dew`, `fogsmog`, `frost`, `glaze`, `hail`, `lightning`, `rain`, `rainbow`, `rime`, `sandstorm`, `snow`. Class names are read dynamically from the folder structure - never hard-coded.
- **Split:** stratified 70 / 15 / 15 (train / val / test) with `splitfolders`, `seed=42`. Test set = 1,041 images.

### EDA highlights

- The classes are **imbalanced** (e.g. `rainbow` ≈ 36 test images vs `rime` ≈ 174) -> we use **balanced class weights** during training.
- Several classes are **physically similar** and expected to be confusable: `rime` / `glaze` / `frost` / `snow` are all ice/frost on surfaces.
- Images vary in size → all resized to **224×224**, the native input of EfficientNet-B0.

---

## Model & training

**Architecture:** `EfficientNetB0(include_top=False)` → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(11, softmax)`. ~5M parameters, small enough to run inference on CPU.

**Augmentation** (train only): `rotation ±15°`, `width/height shift 10%`, `zoom 15%`, `horizontal flip`.
Deliberately **no vertical flip** (upside-down sky isn't weather) and **no strong brightness/contrast** jitter (haze vs clear sky *is* a contrast signal — augmenting it away would hurt).

**Two-stage fine-tuning:**

| Stage | What trains | Optimizer | Epochs | Val accuracy |
|-------|-------------|-----------|--------|--------------|
| 1 — head | backbone frozen, only the new head | Adam `1e-3` | 10 | ~0.884 |
| 2 — fine-tune | unfreeze `block6`, `block7`, `top_` (BatchNorm kept frozen) | Adam `1e-5` | 15 | ~0.909 |

Both stages use `CategoricalCrossentropy(label_smoothing=0.1)`, balanced `class_weight`, and callbacks: `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint(save_best_only)`.

Why this recipe:
- **Two stages** — let the fresh head stabilize first so its large early gradients don't wreck the pretrained features.
- **BatchNorm frozen** in stage 2 — the running statistics were learned on ImageNet's huge batches; updating them on small weather batches is unstable.
- **Label smoothing 0.1** — reduces overconfidence, which directly helps the gate (softer, more informative probabilities).

---

## Results

- **Test accuracy: 90.0 %** · **Top-3 accuracy: 98.4 %** · macro-F1: 0.913

Per-class F1 (test):

| class | P | R | F1 | | class | P | R | F1 |
|-------|---|---|----|-|-------|---|---|----|
| dew | 0.96 | 0.93 | **0.95** | | rain | 0.99 | 0.96 | **0.98** |
| fogsmog | 0.93 | 0.95 | **0.94** | | rainbow | 1.00 | 1.00 | **1.00** |
| frost | 0.77 | 0.88 | **0.82** | | rime | 0.86 | 0.82 | **0.84** |
| glaze | 0.72 | 0.78 | **0.75** | | sandstorm | 0.96 | 0.90 | **0.93** |
| hail | 0.98 | 0.99 | **0.98** | | snow | 0.87 | 0.86 | **0.87** |
| lightning | 1.00 | 1.00 | **1.00** | | | | | |

**Most confused pairs** (all physically similar): `rime → glaze` (18), `rime → snow` (10), `glaze → rime` (9), `glaze → frost` (8), `sandstorm → fogsmog` (7). `glaze` and `frost` are the hardest classes — exactly the ice-on-surface group EDA flagged.

**Grayscale ablation** (how much the model relies on color): accuracy drops **0.900 → 0.789** (−0.111) when color is removed. Color matters most for `sandstorm` (−0.457), `dew` (−0.226) and `rainbow` (−0.194) — all colour-defined phenomena. `fogsmog` is essentially colourless and is unaffected. This confirms the model uses genuine visual cues, not artifacts.

---

## The gate (OOD detection)

We compared two score functions on the model's logits:

- **MSP** — `max(softmax(logits))`. High = confident = in-distribution.
- **Energy** — `T · logsumexp(logits / T)`, evaluated at several temperatures.

| method | AUROC (easy OOD) | FPR@95TPR |
|--------|------------------|-----------|
| **MSP** | **0.974** | **0.184** |
| Energy (best, T=0.1) | 0.965 | 0.212 |

MSP won on both AUROC and FPR@95TPR, so it is the deployed gate. **Threshold `τ = 0.396`** was set at the 5th percentile of in-distribution scores — i.e. calibrated to keep **95 % of true weather images** (95 % TPR). Config lives in `models/gate_config.json`.

**OOD benchmark — two difficulty levels:**

| OOD set | what it is | AUROC | FPR@95TPR |
|---------|-----------|-------|-----------|
| easy | non-weather Caltech classes (faces, cars, laptops…) | **0.974** | 0.184 |
| hard | *near-weather* scenes: clear beaches, sunny mountains, empty skies, tree branches | **0.918** | 0.455 |

---

## Limitations

- On **hard, near-weather** images the gate is much weaker: ~46 % of them slip through at the 95 %-TPR threshold. An empty blue sky or a clear beach can look enough like a weather scene that the model stays confident. This is the fundamental limit of using the classifier's own confidence as a gate — it was never trained on "boring landscapes."
- `glaze` / `frost` / `rime` are genuinely hard to separate even for a person; most errors are inside this group.
- The gate threshold is a single global knob — it trades false rejects of real weather against false accepts of non-weather. It can be re-calibrated for a stricter or looser demo.



