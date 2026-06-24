---
title: >-
  [Paper Note] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation
description: >-
  [CVPR 2026][Interpretability][Accident Anticipation] RiskProp is proposed as a self-supervised risk propagation paradigm anchored by the collision frame. By utilizing future frame regularization and adaptive monotonic constraint losses, the model learns temporally coherent risk evolution curves relying solely on collision frame annotations, achieving SOTA performance on CAP and Nexar datasets.
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Accident Anticipation"
  - "Self-Supervised Risk Propagation"
  - "Temporal Modeling"
  - "Monotonicity Constraint"
  - "Dashcam Videos"
date: 2026-05-08
content_hash: 093b178f9a2c840f
---

# RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation

**Conference**: CVPR 2026  
**arXiv**: [2603.27165](https://arxiv.org/abs/2603.27165)  
**Code**: [https://github.com/xingyueye5/RiskProp/](https://github.com/xingyueye5/RiskProp/)  
**Area**: Interpretability  
**Keywords**: Accident Anticipation, Self-Supervised Risk Propagation, Temporal Modeling, Monotonicity Constraint, Dashcam Videos

## TL;DR

RiskProp is proposed as a self-supervised risk propagation paradigm anchored by the collision frame. By utilizing future frame regularization and adaptive monotonic constraint losses, the model learns temporally coherent risk evolution curves relying solely on collision frame annotations, achieving SOTA performance on CAP and Nexar datasets.

## Background & Motivation

1. **Background**: Accident anticipation aims to estimate risk scores in real-time from dashcam videos, triggering alerts when scores exceed a threshold. Existing methods model this task as binary supervised learning—labeling all frames in non-accident videos as 0, and accident videos from "anomaly onset" to the collision frame as 1.

2. **Limitations of Prior Work**: The binary label paradigm is fundamentally flawed as it forces the model to treat all pre-collision frames as having equal risk, ignoring the progressive nature of risk evolution. Manually labeling "anomaly onset" frames is subjective and inconsistent, leading to noisy supervision signals.

3. **Key Challenge**: In real driving scenarios, risk is a continuous increasing process (e.g., rising slowly when a driver is distracted vs. spiking when a pedestrian suddenly appears), yet binary labels fail to express these intermediate states and scene-dependent evolutions.

4. **Goal**: To learn temporally coherent and physically plausible risk evolution curves using only reliably labeled collision frames, without relying on manual anomaly onset annotations.

5. **Key Insight**: Two key observations are made: (1) Future frames contain more explicit collision evidence, making future predictions more accurate and suitable as pseudo-supervision for current frames; (2) Pre-collision risk generally follows a non-decreasing trend.

6. **Core Idea**: Use the collision frame as the sole anchor and propagate risk signals backward through future frame predictions as soft labels, combined with adaptive monotonic constraints to model risk evolution without manual interval labeling.

## Method

### Overall Architecture

The core problem addressed by RiskProp is learning a risk curve that increases progressively without being flattened by binary labels, given only the collision frame as a reliable label. For each timestamp $t$, the model processes $O$ consecutive frames (where $O=5$) $\mathbf{x}_t = \{x_{t-O+1}, \dots, x_t\}$. A 3D CNN encoder (SlowOnly) extracts features, followed by a sigmoid layer to output the risk score $a_t = \sigma(f_\theta(\mathbf{x}_t))$. The supervision signal is the primary innovation: the collision frame provides the ground truth anchor, while intermediate frames use future frame predictions as soft targets combined with scene-adaptive monotonic constraints. Three losses are employed: BCE anchors the ends (collision and start frames), Future Frame Regularization (FFR) propagates signals backward, and Adaptive Monotonicity Constraint (AMC) ensures the learned curve is generally non-decreasing.

### Key Designs

**1. Future Frame Regularization (FFR): Using future predictions as soft labels to propagate signals backward**

Binary labels fail by treating all pre-collision frames equally. FFR leverages the fact that as a collision approaches, evidence becomes clearer, making future predictions naturally more reliable than current ones. Using stop-gradient, the prediction of the next frame $\text{detach}(z_{t+1})$ is frozen as the target for $z_t$, minimizing $\mathcal{L}_{\text{reg}} = \sum_{t=1}^{T-1} \|\text{detach}(z_{t+1}) - z_t\|^2$. Since the collision frame $z_T$ has a reliable ground truth $y_T=1$, high-risk signals propagate backward through the chain $z_T \to z_{T-1} \to \cdots$ to early frames. The stop-gradient ensures information flows only backward in time, preventing uncertain early predictions from polluting the collision anchor. This mechanism eliminates the need for teacher models or subjective anomaly onset labels.

**2. Adaptive Monotonicity Constraint (AMC): Ensuring non-decreasing risk while tolerating short-term fluctuations**

Risk signals propagated by FFR may still exhibit local jitter. AMC enforces $a_j \geq a_i$ for sampled frame pairs $(i, j)$ where $j > i$ using a hinge-like structure: $\mathcal{L}_{\text{mono}} = \frac{1}{|\mathcal{D}|} \sum_{(i,j)} \max\big(0,\, a_i - a_j + \delta(\Delta t, \bar{c}_{i:j})\big)$. Instead of a hard constraint, it uses an adaptive margin:

$$\delta = \delta_0 \cdot \Delta t \cdot \bar{c}_{i:j},$$

where the margin scales with the temporal interval $\Delta t$ and the average prediction confidence $\bar{c}_{i:j}$. Larger time gaps or higher confidence segments result in stricter monotonicity constraints, while low-confidence or adjacent frames allow for minor fluctuations. This balances continuous risk growth with noise tolerance.

**3. Collision-Only Annotation Strategy: Labeling only objective collision moments**

Existing methods rely on exponential decay weighting or manual onset windows, which are subjective sources of noise. RiskProp simplifies annotation: only the collision frame is designated as positive ($y_T=1$) and the first frame as negative ($y_0=0$) in accident videos, while non-accident videos are all zeros. Intermediate supervision is provided entirely by FFR soft labels. BCE is applied with higher weights on collision frames to mitigate imbalance. This relies on the objectivity of collision timestamps while FFR and AMC handle the evolution modeling.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{bce}} + \lambda_1 \cdot \mathcal{L}_{\text{reg}} + \lambda_2 \cdot \mathcal{L}_{\text{mono}}$, with $\lambda_1=1.5$ and $\lambda_2=1.1$. Training utilizes SlowOnly weights, SGD optimizer, and 8 A800 GPUs for 50 epochs. The batch size is 64; initial learning rate is 0.002 with 10% decay every 20 epochs. Frame sampling intervals are set to $d_{\min}=0.1, d_{\max}=0.9$, with a base margin $\delta_0=0.01$. Videos are resized to 224x224 and resampled to 10 FPS.

## Key Experimental Results

### Main Results

| Dataset | Method | mAUC0.1 | mAUC | mAP | mTTA0.1 (s) |
|--------|------|---------|------|-----|-------------|
| CAP | AdaLEA | 0.379 | 0.807 | 0.857 | 1.115 |
| CAP | CRASH | 0.401 | 0.842 | 0.887 | 1.085 |
| CAP | **RiskProp** | **0.483** | 0.853 | **0.890** | **1.207** |
| Nexar | CRASH | 0.393 | 0.832 | 0.846 | 0.857 |
| Nexar | **RiskProp** | **0.472** | **0.869** | **0.870** | **0.958** |

On Nexar, RiskProp outperforms the runner-up CRASH across all metrics, with mAUC0.1 improving by 0.079, mAUC by 0.037, and mAP by 0.024.

### Ablation Study

| Config | Annotation Strategy | mAUC0.1 (CAP) | mAUC0.1 (Nexar) | Note |
|------|---------|---------------|-----------------|------|
| Baseline (w/o FFR/AMC) | Only Collision | 0.358 | 0.298 | Only collision frame labeled, no self-supervised constraints |
| +FFR | Only Collision | 0.474 | 0.453 | FFR improves CAP by 0.116 |
| +FFR+AMC | Only Collision | 0.483 | 0.472 | Full model (SOTA) |
| +FFR+AMC | Anomaly Onset | 0.484 | 0.479 | Using manual onset labels |

### Key Findings

- **FFR provides the largest gain**: Under the "Only Collision" setting, adding FFR alone increases mAUC0.1 by 0.116 on CAP and 0.155 on Nexar, proving future frame regularization effectively propagates risk signals.
- **Collision-only labels are sufficient**: The full model with "Only Collision" labels achieves 0.483 (CAP) / 0.472 (Nexar), performing nearly identically to the "Anomaly Onset" dense labels (0.484 / 0.479).
- **Smoother risk curves**: Qualitative analysis shows RiskProp maintains low risk estimates during safe periods and rises sharply only when danger emerges, suppressing early false positives common in traditional methods.

## Highlights & Insights

- **Ingenious self-supervised propagation**: Using stop-gradient to let the next frame's prediction serve as the target for the current frame enables risk signal back-propagation without extra teacher models.
- **Minimal labeling matching dense labeling**: Demonstrating that simplified annotations can achieve dense-label performance under proper self-supervised constraints has significant implications for practical deployment.
- **Transferable adaptive monotonicity**: The adaptive margin based on confidence and temporal distance can be generalized to any task requiring temporal monotonicity, such as disease progression or equipment aging.

## Limitations & Future Work

- Collision frame labels are still required; the method is not fully unsupervised.
- FFR and AMC are disabled on non-accident videos, meaning safety modeling relies solely on BCE.
- Only 3D CNN encoders were validated; Transformer or multi-modal encoders remain unexplored.
- Fixed 10 FPS resampling might lose critical information in fast-changing scenes.
- Risk propagation could be extended to be bidirectional (backward from collision and forward from safety constraints).

## Related Work & Insights

- **vs. AdaLEA/CRASH**: Unlike these methods that rely on exponential decay or manual onset windows, RiskProp removes subjective designs and outperforms them using only collision anchors and self-supervision.
- **vs. DSTA**: While DSTA achieves higher mAUC on CAP (0.895), RiskProp leads significantly in early warning metrics (mAUC0.1, mTTA).
- The self-supervised temporal propagation concept could inspire related fields like video anomaly detection and action prediction.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm of collision anchoring and self-supervised propagation is novel, though individual components (stop-gradient pseudo-labels, monotonicity constraints) have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across two datasets, ablation studies, and qualitative risk curve analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous method description, and intuitive visualizations.
- Value: ⭐⭐⭐⭐ Significant reduction in annotation dependency and improved interpretability are major advantages for safety-critical systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](../../ICML2026/interpretability/interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](../../CVPR2025/interpretability/probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)

</div>

<!-- RELATED:END -->
