---
title: >-
  [Paper Note] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation
description: >-
  [CVPR 2026][Accident Anticipation] This paper proposes RiskProp, a collision-anchored self-supervised risk propagation paradigm that learns temporally coherent risk evolution curves using only collision-frame annotations, via a future-frame regularization loss and an adaptive monotonicity constraint loss, achieving state-of-the-art performance on the CAP and Nexar datasets.
tags:
  - CVPR 2026
  - Accident Anticipation
  - Self-Supervised Risk Propagation
  - Temporal Modeling
  - Monotonicity Constraint
  - Dashcam
date: 2026-05-08
content_hash: 2834e4111234c0a4
---

# RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation

**Conference**: CVPR 2026
**arXiv**: [2603.27165](https://arxiv.org/abs/2603.27165)
**Code**: [https://github.com/xingyueye5/RiskProp/](https://github.com/xingyueye5/RiskProp/)
**Area**: Interpretability
**Keywords**: Accident Anticipation, Self-Supervised Risk Propagation, Temporal Modeling, Monotonicity Constraint, Dashcam

## TL;DR

This paper proposes RiskProp, a collision-anchored self-supervised risk propagation paradigm that learns temporally coherent risk evolution curves using only collision-frame annotations, via a future-frame regularization loss and an adaptive monotonicity constraint loss, achieving state-of-the-art performance on the CAP and Nexar datasets.

## Background & Motivation

1. **Background**: Accident Anticipation aims to estimate risk scores in real time from dashcam videos, triggering warnings when scores exceed a threshold. Existing methods formulate this as binary supervised learning—all frames in non-accident videos are labeled 0, while frames from the anomaly onset to the collision frame in accident videos are labeled 1.

2. **Limitations of Prior Work**: The binary label paradigm has a fundamental flaw—it forces the model to treat all pre-collision frames as equally risky, ignoring the progressive nature of risk escalation. Manual annotation of "anomaly onset" frames is subjective and inconsistent across annotators, producing noisy supervision signals.

3. **Key Challenge**: In real driving scenarios, risk increases continuously (e.g., rising slowly when a driver is distracted, or spiking sharply when a pedestrian suddenly appears), yet binary labels cannot capture these intermediate states or scene-dependent risk dynamics.

4. **Goal**: To learn temporally coherent and physically plausible risk evolution curves without relying on manually annotated anomaly onset frames, using only reliably annotated collision frames.

5. **Key Insight**: The authors identify two key observations: (1) future frames contain stronger collision evidence, making model predictions on future frames more accurate and suitable as pseudo-supervision for current frames; (2) pre-collision risk generally follows a non-decreasing trend.

6. **Core Idea**: Using the collision frame as the sole anchor, risk signals are back-propagated via next-frame predictions as soft labels, combined with adaptive monotonicity constraints, enabling risk evolution modeling without manual annotations.

## Method

### Overall Architecture

The input is a dashcam video clip. At time step $t$, the model receives $O$ consecutive frames ($O=5$ in experiments) $\mathbf{x}_t = \{x_{t-O+1}, \dots, x_t\}$, extracts features via a 3D CNN encoder (SlowOnly), and outputs a risk score $a_t = \sigma(f_\theta(\mathbf{x}_t))$ through sigmoid activation. The training objective consists of three losses: BCE loss (applied only at the collision and onset frames), Future Frame Regularization loss (FFR), and Adaptive Monotonicity Constraint loss (AMC).

### Key Designs

1. **Future Frame Regularization Loss (FFR)**:

    - **Function**: Propagates risk signals backward from the collision frame to earlier frames.
    - **Mechanism**: Uses a stop-gradient operation to treat the next-frame prediction $\text{detach}(z_{t+1})$ as the soft target for the current frame $z_t$, with loss $\mathcal{L}_{\text{reg}} = \sum_{t=1}^{T-1} \|\text{detach}(z_{t+1}) - z_t\|^2$. Through chain propagation $z_T \to z_{T-1} \to \cdots$, the high-risk signal from the collision frame is gradually transmitted to earlier frames.
    - **Design Motivation**: The collision frame has the only reliable ground-truth label $y_T=1$, and future frames always carry more collision evidence than current frames, making their predictions reliable pseudo-supervision. This eliminates dependence on subjective anomaly onset annotations.

2. **Adaptive Monotonicity Constraint Loss (AMC)**:

    - **Function**: Encourages risk scores to follow a non-decreasing trend along the temporal axis.
    - **Mechanism**: For randomly sampled frame pairs $(i, j)$ with $j > i$, the constraint $a_j \geq a_i$ is enforced via $\mathcal{L}_{\text{mono}} = \frac{1}{|\mathcal{D}|} \sum_{(i,j)} \max(0, a_i - a_j + \delta(\Delta t, \bar{c}_{i:j}))$. The adaptive tolerance margin $\delta = \delta_0 \cdot \Delta t \cdot \bar{c}_{i:j}$ adjusts based on temporal distance and prediction confidence: the constraint tightens for larger time spans and higher model confidence.
    - **Design Motivation**: Risk generally increases over the course of a real accident, but short-term fluctuations are permissible. Fixed margins or hard constraints over-regularize; the adaptive mechanism balances flexibility and stability.

3. **Collision-Only Annotation Strategy**:

    - **Function**: A minimal annotation scheme that reduces dependence on subjective labeling.
    - **Mechanism**: In accident videos, only the collision frame is labeled positive ($y_T=1$) and the onset frame is labeled negative ($y_0=0$); intermediate frames are entirely assigned soft labels by FFR. All frames in non-accident videos are labeled 0. BCE loss is weighted with higher weight on collision frames to mitigate class imbalance.
    - **Design Motivation**: Collision timestamps are objectively reliable, whereas anomaly onset annotations are subjective and inconsistent. Combined with FFR and AMC, collision-only annotation achieves performance comparable to dense annotation.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{\text{bce}} + \lambda_1 \cdot \mathcal{L}_{\text{reg}} + \lambda_2 \cdot \mathcal{L}_{\text{mono}}$, with $\lambda_1=1.5$ and $\lambda_2=1.1$. SlowOnly pretrained weights are used with SGD optimizer, trained for 50 epochs on 8×A800 GPUs with batch size 64, initial learning rate 0.002 decayed by 90% every 20 epochs. Frame sampling uses $d_{\min}=0.1$, $d_{\max}=0.9$, $\delta_0=0.01$. Frames are resized to 224×224 and resampled to 10 FPS.

## Key Experimental Results

### Main Results

| Dataset | Method | mAUC0.1 | mAUC | mAP | mTTA0.1 (s) |
|--------|------|---------|------|-----|-------------|
| CAP | AdaLEA | 0.379 | 0.807 | 0.857 | 1.115 |
| CAP | CRASH | 0.401 | 0.842 | 0.887 | 1.085 |
| CAP | **RiskProp** | **0.483** | 0.853 | **0.890** | **1.207** |
| Nexar | CRASH | 0.393 | 0.832 | 0.846 | 0.857 |
| Nexar | **RiskProp** | **0.472** | **0.869** | **0.870** | **0.958** |

On Nexar, RiskProp surpasses the second-best method CRASH on all metrics: mAUC0.1 by 0.079, mAUC by 0.037, and mAP by 0.024.

### Ablation Study

| Configuration | Annotation | mAUC0.1 (CAP) | mAUC0.1 (Nexar) | Notes |
|------|---------|---------------|-----------------|------|
| Baseline (no FFR/AMC) | Only Collision | 0.358 | 0.298 | Collision-only annotation, no self-supervised constraints |
| +FFR | Only Collision | 0.474 | 0.453 | CAP gains 0.116 with FFR |
| +FFR+AMC | Only Collision | 0.483 | 0.472 | Full model, SOTA |
| +FFR+AMC | Anomaly Onset | 0.484 | 0.479 | With manually annotated onset frames |

### Key Findings

- **FFR contributes the most**: Under the Only Collision setting, adding FFR alone yields mAUC0.1 gains of 0.116 on CAP and 0.155 on Nexar, demonstrating that future-frame regularization effectively propagates risk signals.
- **Collision-only annotation suffices**: The full model under Only Collision achieves 0.483 (CAP) / 0.472 (Nexar), nearly matching dense annotation (Anomaly Onset) at 0.484 / 0.479, confirming that subjective anomaly onset labels are unnecessary.
- **Smoother risk curves**: Qualitative analysis shows that RiskProp maintains low risk estimates during safe periods and rises sharply only when genuine danger appears, effectively suppressing the early false positives common in conventional methods.

## Highlights & Insights

- **The self-supervised chain propagation mechanism is elegant**: Using stop-gradient to treat next-frame predictions as targets for the current frame is a simple design that achieves risk signal back-propagation from collision frames to early frames, without requiring additional teacher models or complex architectures.
- **"Collision-only" matching "dense annotation" is the central contribution**: It demonstrates that under well-designed self-supervised constraints, minimal annotation can match the performance of dense annotation, which has significant practical implications.
- **The adaptive monotonicity constraint is transferable**: The confidence- and temporal-distance-based adaptive margin mechanism can be generalized to any task requiring temporal monotonicity constraints, such as disease progression prediction or equipment degradation monitoring.

## Limitations & Future Work

- Collision-frame annotation remains necessary; the method cannot be applied in fully unsupervised settings.
- FFR and AMC are disabled for non-accident videos, meaning the model relies solely on BCE loss to model safe scenarios.
- Only the 3D CNN encoder is evaluated; Transformer or multimodal encoders remain unexplored.
- Fixed resampling to 10 FPS may discard critical information in rapidly evolving scenes.
- Future work could consider bidirectional risk propagation (not only back-propagating from collision frames, but also forward-constraining from safe periods).

## Related Work & Insights

- **vs AdaLEA/CRASH**: These methods rely on exponential decay weighting or manually annotated anomaly onset frames to define positive sample windows. RiskProp completely removes these subjective designs and outperforms them using only collision frames and self-supervised constraints.
- **vs DSTA**: DSTA achieves the highest mAUC on CAP (0.895), but RiskProp substantially outperforms it on early warning metrics (mAUC0.1, mTTA), reflecting different trade-offs under different evaluation priorities.
- The self-supervised temporal propagation paradigm can inspire related areas such as video anomaly detection and action anticipation.

## Rating

- Novelty: ⭐⭐⭐⭐ The collision-anchored self-supervised propagation paradigm is novel, though the core techniques (stop-gradient pseudo-labels, monotonicity constraints) have individual precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, comprehensive ablation, three annotation strategy comparisons, and risk curve visualization—very thorough.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, method description is rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Reducing annotation dependence is highly valuable for real-world deployment; the interpretability of risk curves is an asset for safety-critical systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ICLR 2026\] Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](../../ICLR2026/interpretability/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_advanced_jai.md)

<!-- RELATED:END -->
