---
name: image-preprocessing-augmentation
description: Use when preparing an image dataset for CNN training — splitting, resizing, normalizing, building tf.data pipelines, and choosing data augmentation justified by the domain. Enforces no-leakage splits and class-imbalance handling (class weights). Produces reports/preprocessing_decisions.json. Triggers when user mentions "preprocesamiento de imágenes", "data augmentation", "tf.data", "redimensionar", "normalizar imágenes", "split de imágenes".
---

# Image Preprocessing & Augmentation

## Overview

In computer vision there is no manual feature engineering — the CNN learns the features.
The analog of FE is the **input pipeline**: split → resize → normalize → augment. Each
augmentation must be justified by the domain (does this transform produce a plausible image
of the same class?), exactly as tabular FE justifies each feature by an EDA finding. The
output is `reports/preprocessing_decisions.json` + reproducible `tf.data` pipelines.

## When to use

- After [[image-eda]], before [[cnn-architecture-design]]
- User says "preparar las imágenes" / "data augmentation" / "armar los splits"

Do NOT use:
- For tabular FE (use [[feature-engineering-justified]])
- Before EDA (you need the grayscale/balance findings to choose channels and weights)

## Workflow

1. **Reserve the catch-all class.** Build the label index ONLY from defined classes; keep
   "Other" paths aside for the Task-6 inference step. It never enters train/val/test.
2. **Stratified split 70/15/15** (or per consigna) with a fixed seed, stratifying by class so
   small classes appear in every split.
3. **Decide channels** from the EDA grayscale finding: 1 channel if grayscale; 3 if real color
   (or replicate gray→3 only for transfer-learning backbones).
4. **Resize + normalize.** Fixed target (e.g. 128×128 for a from-scratch CNN; 224×224 for
   ImageNet backbones). Normalize to [0,1] (or backbone-specific preprocessing).
5. **Augmentation — justify each transform by the domain.** Only transforms that yield a
   plausible same-class image. For terrain/aerial: rotations (any angle — no canonical
   orientation), H/V flips, mild zoom, brightness/contrast (illumination). Avoid transforms
   that change the label or are physically implausible (e.g. heavy color jitter on grayscale).
6. **Apply augmentation to TRAIN ONLY.** val/test see clean, resized, normalized images.
7. **Class weights** from train counts (`sklearn compute_class_weight`) to counter imbalance —
   the CV analog of the SMOTE/oversampling the course covered.
8. **Visualize** original vs augmented examples (deliverable + sanity check).

## Code template (tf.data + augmentation, leakage-safe)

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

SEED, IMG, BATCH = 42, 128, 32

def make_splits(paths, labels):
    # stratified 70/15/15, fixed seed → reproducible, every class in each split
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30,
                                                stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50,
                                              stratify=y_tmp, random_state=SEED)
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)

def _load(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)   # grayscale dataset
    img = tf.image.resize(img, [IMG, IMG]) / 255.0
    return img, label

aug = tf.keras.Sequential([                       # domain-justified for terrain
    tf.keras.layers.RandomRotation(0.5),          # any orientation is valid
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),        # illumination changes
])

def pipeline(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(_load, tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1000, seed=SEED).map(lambda x, y: (aug(x, training=True), y),
                                             tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def class_weights(y_train, n_classes):
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    return dict(enumerate(w))
```

## Output spec

`reports/preprocessing_decisions.json`:
```json
{
  "split": {"scheme": "stratified 70/15/15", "seed": 42,
            "counts": {"train": 1755, "val": 376, "test": 377}},
  "image_size": 128, "channels": 1, "normalization": "[0,1]",
  "augmentation": [
    {"transform": "RandomRotation(±180°)", "rationale": "terrain has no canonical orientation"},
    {"transform": "RandomFlip H+V", "rationale": "mirror of terrain is still valid terrain"},
    {"transform": "RandomZoom(0.1)", "rationale": "scale invariance of features"},
    {"transform": "RandomBrightness(0.1)", "rationale": "illumination variation"}
  ],
  "excluded_transforms": [
    {"transform": "color jitter", "reason": "dataset is grayscale; would be meaningless"}
  ],
  "class_weights": {"0": 0.33, "6": 7.8},
  "catch_all_reserved": "other (N images, inference only)"
}
```

## <EXTREMELY-IMPORTANT> Rules

1. **No-leakage split.** Stratify by class, fixed seed, and ensure no image appears in two
   splits. Augmentation NEVER touches val/test.
2. **Catch-all reserved before split.** "Other" must be removed before building the index, or
   it leaks into training (consigna requires it held out).
3. **Every augmentation justified by the domain.** A transform that could change the true label
   (or is physically implausible) is forbidden. Document the rationale per transform AND the
   ones you excluded and why.
4. **Channels follow the EDA grayscale finding.** Don't train 3-channel on grayscale data
   unless a backbone requires it (then replicate, don't fabricate color).
5. **Class weights, not silent imbalance.** Compute from TRAIN counts only.
6. **Augmentation = the CV analog of oversampling/SMOTE** the course taught — frame it that way.

## Auto-review before handoff

Before passing to [[cnn-architecture-design]]:
1. Split counts sum to total defined-class images; "Other" excluded
2. No path overlap across train/val/test (assert on full sets)
3. Augmentation applied to train only (verify val/test pipeline has no aug layer)
4. `preprocessing_decisions.json` lists a rationale for every included AND excluded transform
5. Class weights computed from train only

If any check fails, fix before designing the model — a leaky split invalidates all later metrics.

## Red flags

| Thought | Reality |
|---|---|
| "Augment everything, more data is better" | An implausible transform teaches wrong invariances. Justify each. |
| "Random split is fine" | Small classes can vanish from a split. Stratify. |
| "Apply augmentation to all sets for consistency" | Augmenting val/test corrupts evaluation. Train only. |
| "Include Other, the model sees more" | Other is held out by the consigna. Reserve it. |
| "Color jitter helps robustness" | Not on grayscale data. Exclude it and say why. |
