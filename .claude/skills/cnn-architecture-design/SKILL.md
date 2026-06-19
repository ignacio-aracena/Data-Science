---
name: cnn-architecture-design
description: Use when designing a convolutional neural network from scratch or via transfer learning for image classification, justifying every architectural choice. Covers conv/pool blocks, filters, kernel size, BatchNorm, dropout, the output layer, loss/optimizer, and transfer-learning backbones (MobileNetV2/ResNet50). Triggers when user mentions "diseñar CNN", "arquitectura convolucional", "Conv2D", "transfer learning", "MobileNet", "ResNet", "capa de salida".
---

# CNN Architecture Design

## Overview

The consigna grades "diseño y justificación de la arquitectura" — so the deliverable is not
just a model, it is a model whose every choice (number of filters, kernel size, activation,
pooling, BatchNorm, dropout, output layer) is justified. This skill provides a defensible
default architecture and the reasoning to defend each knob, plus a transfer-learning track.

## When to use

- After [[image-preprocessing-augmentation]], to define v1 of the model
- User says "diseñar la CNN" / "arquitectura" / "transfer learning"

Do NOT use:
- For the iteration loop v1→vN (use [[nn-iteration-with-error-analysis]] — this skill defines
  the FIRST architecture; that one decides each NEXT one from error analysis)
- For tabular MLPs

## The defensible default (from-scratch CNN)

A 3-block convolutional tower + GAP head. Justification per choice in the table below.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(img=128, channels=1, n_classes=7):
    m = models.Sequential([
        layers.Input((img, img, channels)),
        # Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        # Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        # Block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return m
```

## Choice → justification (defend these)

| Choice | Default | Justification |
|---|---|---|
| Conv blocks | 3 (32→64→128) | Doubling filters as spatial size halves keeps compute ~balanced; depth captures simple→complex features. |
| Kernel size | 3×3 | Two stacked 3×3 ≈ one 5×5 receptive field with fewer params and more non-linearity. |
| Activation | ReLU | Cheap, non-saturating, standard for conv nets. |
| Pooling | MaxPool 2×2 | Downsamples, keeps strongest activation, adds small translation invariance. |
| BatchNorm | after each conv | Stabilizes and speeds training, mild regularization; helps with small batches on CPU. |
| Head pooling | GlobalAveragePooling | Far fewer params than Flatten+Dense → less overfit on a small dataset. |
| Dropout | 0.5 before output | Regularizes the dense head; justified by small-dataset overfit risk (verify on curves). |
| Output | Dense(n_classes, softmax) | Multiclass single-label → softmax + categorical cross-entropy. |
| Loss | (sparse) categorical CE | Standard multiclass classification loss. |
| Optimizer | Adam | Robust default; from the course syllabus. |

## Transfer-learning track (bonus)

```python
def build_transfer(img=224, n_classes=7, base="mobilenet"):
    inp = tf.keras.Input((img, img, 3))                 # backbones need 3 channels
    x = tf.keras.layers.Concatenate()([inp, inp, inp]) if False else inp
    Base = tf.keras.applications.MobileNetV2 if base == "mobilenet" else tf.keras.applications.ResNet50
    backbone = Base(include_top=False, weights="imagenet", input_tensor=inp)
    backbone.trainable = False                          # phase 1: freeze, train head
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)
# phase 2 (fine-tune): backbone.trainable = True; unfreeze last N layers; recompile lr=1e-5
```

For a grayscale dataset, replicate the single channel to 3 (`tf.image.grayscale_to_rgb`) so
the ImageNet backbone accepts it — and note in the report that no real color information is added.

## <EXTREMELY-IMPORTANT> Rules

1. **Justify every knob.** The grade is on the justification, not the layer list. Use the table.
2. **Output layer matches the problem.** Single-label multiclass → softmax + CE. Don't ship a
   sigmoid/BCE head for a single-label problem.
3. **Dropout/L2 only with a reason.** Add regularization because the small dataset risks overfit
   (or because curves show it), not "because it's standard". Defer aggressive regularization to
   [[nn-iteration-with-error-analysis]] driven by the v1 curves.
4. **GAP over Flatten on small data.** Flatten+big Dense explodes params and overfits.
5. **Transfer learning: freeze then fine-tune.** Don't unfreeze the whole backbone from step 1 on
   a tiny dataset — you'll wreck the pretrained weights.
6. **Grayscale→3ch is replication, not color.** Say so in the report; it changes interpretability.

## Auto-review before training

1. `model.summary()` param count is reasonable for the dataset size (small data → modest model)
2. Output units == number of defined classes; activation == softmax
3. Loss matches label encoding (sparse vs one-hot)
4. Input shape matches the preprocessing pipeline (size + channels)
5. Each architectural choice has a one-line justification recorded (for the notebook/report)

## Red flags

| Thought | Reality |
|---|---|
| "Deeper is always better" | On ~2k images a deep net overfits. Start modest, grow only if error analysis says so. |
| "Flatten then Dense(512)" | Param explosion on small data. Use GlobalAveragePooling. |
| "Add dropout everywhere by default" | Regularize from evidence (overfit curves), not reflex. |
| "Fine-tune the whole ResNet from epoch 0" | Destroys pretrained features on small data. Freeze first. |
| "Sigmoid output, it's classification" | Single-label multiclass = softmax. Sigmoid is for multi-label. |
| "Color images, feed RGB to MobileNet" | If the data is grayscale, you're replicating channels — note it. |
