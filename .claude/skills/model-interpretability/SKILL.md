---
name: model-interpretability
description: Use when explaining what a trained model learned and why it fails — for TABULAR models (SHAP, permutation importance, partial dependence, calibration) and for VISION CNNs (feature maps, Grad-CAM, occlusion sensitivity, invariance to rotation/flip/color). Crosses failure modes with the confusion matrix / residuals. Triggers when user mentions "interpretabilidad", "explicabilidad", "SHAP", "feature importance", "Grad-CAM", "occlusion", "qué mira el modelo", "por qué falla", "saliency", "invarianza".
---

# Model Interpretability

## Overview

A model is not done until you can explain what it relies on and where it breaks. This skill
covers both data types. Every experiment yields an artifact (figure/table) AND a one-paragraph
interpretation, and the key deliverable is the **cross-reference between failure modes and the
confusion matrix (classification) or residuals (regression)** — the most-rewarded analysis.

## When to use

- After a model is trained and evaluated (you have a confusion matrix / residuals + error cases)
- User says "interpretabilidad", "SHAP", "Grad-CAM", "qué mira el modelo", "por qué falla"

Do NOT use:
- Before there is a trained model and an error breakdown to explain

## Branch by data type

### Tabular models
- **Permutation importance** — shuffle each feature, measure metric drop. Model-agnostic global importance.
- **SHAP** — per-prediction attributions; summary plot (global) + force/waterfall (local).
- **Partial dependence / ICE** — marginal effect of a feature on the prediction.
- **Calibration** — reliability curve + Brier/ECE for probabilistic classifiers.
- **Coefficient/tree inspection** — for linear/tree baselines, read the model directly.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=42)
# r.importances_mean ordered desc → bar plot with feature names
```

### Vision CNNs (do at least 3; all for full marks)
1. **Feature maps — early vs deep.** Activations of the first vs last conv layer for one image;
   early = edges/texture, deep = class-specific patterns. Interpret.
2. **Grad-CAM.** Heatmap from gradients of the predicted class w.r.t. the LAST conv layer.
3. **Occlusion sensitivity.** Slide a patch; map the drop in the true-class probability.
   Model-agnostic cross-check for Grad-CAM.
4. **Shape invariance.** Rotations (90/180/arbitrary) + flips on TEST; measure metric change;
   relate to whether rotation/flip was in augmentation.
5. **Color invariance.** Grayscale, channel swap (RGB→BGR), brightness shift; measure impact.
   On grayscale data, channel-swap ≈ zero effect (R==G==B) — a strong honest finding.

```python
import tensorflow as tf
from tensorflow import keras
def grad_cam(model, img, last_conv):
    gm = keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(img[None]); loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)[0]
    cam = tf.reduce_sum(conv_out[0] * tf.reduce_mean(grads, (0, 1)), -1)
    return (tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)).numpy()
```

## Cross-reference with errors (REQUIRED, both data types)

For each top confusion pair (classification) or worst residual bin (regression), pull example
errors and run the relevant tool (SHAP / Grad-CAM / occlusion). Ask: is the model keying on a
confound (background, a leaky feature, global brightness) instead of the real signal? Does the
error concentrate under a transform (rotation, atypical value)? Write the explanation. This
closes the "¿por qué falla donde falla?".

## Output spec

- Figures in `reports/figures/` (permutation/SHAP, or featuremaps/gradcam/occlusion/robustness)
- `reports/interpretability_findings.md` with one interpretation paragraph per experiment +
  the failure-mode cross-reference

## <EXTREMELY-IMPORTANT> Rules

1. **Every experiment = figure/table + interpretation.** A plot without a sentence is half-done.
2. **Cross-reference failures with the confusion matrix / residuals.** Highest-value analysis.
3. **Tie invariance results back to the data and the augmentation.** "No color effect" is only
   meaningful once you state the data is grayscale; "rotation-robust" relative to augmentation.
4. **Grad-CAM uses the LAST conv layer.** A dense layer gives nothing spatial.
5. **SHAP/permutation on the validation/holdout split**, not on training, to reflect generalization.
6. **Honest negative findings are valuable** ("keys on brightness, not structure").

## Auto-review before handoff

1. ≥3 experiments (vision: of the five), each with a saved artifact
2. `interpretability_findings.md` has an interpretation per experiment
3. ≥1 failure mode explained via the tools and tied to the confusion matrix / residuals
4. Invariance/color findings reference the data nature and augmentation choices
5. Artifacts have interpretable titles

## Red flags

| Thought | Reality |
|---|---|
| "SHAP/Grad-CAM looks cool, ship it" | Without interpretation it's decoration. Explain it. |
| "Global importance is enough" | Local explanations on the error cases reveal the failure mode. |
| "Skip occlusion, Grad-CAM suffices" | They can disagree; occlusion is a model-agnostic check. |
| "Color experiment is irrelevant" | On grayscale data the ~zero effect IS the finding. |
| "Failures are just noise" | Cross with the matrix/residuals; patterns emerge. |
