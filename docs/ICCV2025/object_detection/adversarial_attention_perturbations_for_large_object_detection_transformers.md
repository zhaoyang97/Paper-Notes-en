---
title: >-
  [Paper Note] Adversarial Attention Perturbations for Large Object Detection Transformers
description: >-
  [ICCV 2025][Object Detection][Adversarial Attack] This paper proposes AFOG (Attention-Focused Offensive Gradient), an architecture-agnostic adversarial attack method that leverages a learnable attention mechanism to conc…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Adversarial Attack"
  - "Detection Transformer"
  - "Learnable Attention"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: a4a795c12fa13425
---

# Adversarial Attention Perturbations for Large Object Detection Transformers

**Conference**: ICCV 2025
**arXiv**: [2508.02987](https://arxiv.org/abs/2508.02987)
**Code**: Available (noted in paper as "Code is available at: Link")
**Area**: Object Detection / Adversarial Security
**Keywords**: Adversarial Attack, Detection Transformer, Learnable Attention, Object Detection, Adversarial Robustness

## TL;DR
This paper proposes AFOG (Attention-Focused Offensive Gradient), an architecture-agnostic adversarial attack method that leverages a learnable attention mechanism to concentrate perturbations on vulnerable image regions. With only 10 iterations and visually imperceptible perturbations, AFOG reduces the mAP of 12 detection Transformers by up to 37.8×, while also outperforming existing methods on CNN-based detectors.

## Background & Motivation
Transformer-based object detectors (e.g., DETR, Swin, EVA) leverage attention mechanisms to capture long-range dependencies, substantially outperforming traditional CNN-based detectors (Faster R-CNN, SSD, YOLOv3). As these large detection Transformers are widely deployed, understanding their vulnerability to adversarial perturbations becomes critical.

However, existing adversarial attack methods perform poorly against detection Transformers: (1) surrogate-model attacks (black-box) such as UEA and RAD suffer poor transferability when the surrogate and victim architectures differ; (2) victim-model attacks (white-box) such as EBAD and OATB are tailored to specific architectures, and AttentionFool targets only self-attention and cannot attack CNN detectors. **Key Challenge**: A unified attack framework that effectively targets both Transformers and CNNs is lacking.

**Core Idea of AFOG**: Inspired by Transformer self-attention, AFOG designs a learnable "adversarial attention map" that dynamically identifies the most vulnerable pixel regions in an image. This attention mechanism is decoupled from the victim model's internal architecture, making it applicable to both Transformers and CNNs.

## Method

### Overall Architecture
AFOG adopts an iterative projected gradient descent (PGD) framework. At each iteration: (1) the perturbed image is forward-propagated through the victim model; (2) the attack loss (bounding box loss + classification loss) is computed; (3) both the attention map $A$ and perturbation map $P$ are updated via backpropagation; (4) the adversarial example is generated via the Hadamard product $x_{adv} = \Pi_{x,\epsilon}(x + A \odot P)$, projected onto a hypersphere centered at the original image with radius $\epsilon$.

### Key Designs

1. **Learnable Adversarial Attention Mechanism**:

    - **Function**: Learns a pixel-wise attention map $A$ to spatially amplify or suppress the perturbation.
    - **Mechanism**: The attention map $A$ is initialized to all ones, and the perturbation map $P$ is initialized from a uniform distribution over $[-\epsilon, \epsilon]$. Adversarial examples are generated as $x_{adv_k} = \Pi_{x,\epsilon}(x + A_k \odot P_k)$. $A$ and $P$ are updated respectively via gradients of the attack loss:
        - $A_{k+1} \leftarrow A_k + \alpha_A \cdot \sigma[\frac{\partial \mathcal{L}_{AFOG}}{\partial A_k}]$ ($\sigma$ denotes a normalization function)
        - $P_{k+1} \leftarrow P_k + \alpha_P \cdot \Gamma[\frac{\partial \mathcal{L}_{AFOG}}{\partial P_k}]$ ($\Gamma$ denotes the sign function)
    - **Design Motivation**: Unlike static attention (e.g., based on foreground/background priors), AFOG's attention is dynamically updated throughout the attack iterations, enabling the discovery of counter-intuitive vulnerable regions (e.g., sky above a boat). In early iterations, attention concentrates on primary objects; in later iterations, it expands to surrounding regions.

2. **Dual-Loss Attack Optimization**:

    - **Function**: Maximizes attack effectiveness by simultaneously disrupting bounding box predictions and classification predictions.
    - **Mechanism**: The attack loss consists of two components:
        - Bounding box loss: $\mathcal{L}_{bbox} = \sum_{i=1}^{N_x}[f_\vartheta(x, o_i) - f_\vartheta(x_{adv}, o_{adv_i})]$
        - Classification loss: $\mathcal{L}_{cls} = \sum_{i=1}^{N_x}[f_\vartheta(x, c_i) - f_\vartheta(x_{adv}, c_{adv_i})]$
        - $\mathcal{L}_{AFOG} = \mathcal{L}_{bbox} + \mathcal{L}_{cls}$

      The attack freezes model parameters $\vartheta$ and updates only $A$ and $P$ via gradients.
    - **Design Motivation**: Simultaneously attacking both localization and classification suppresses the confidence of correct predictions while boosting that of incorrect ones, yielding a compounded degradation effect.

3. **Specialized Attack Modes (AFOG-V and AFOG-F)**:

    - **Function**: AFOG-V (vanishing attack) causes all detections to disappear; AFOG-F (fabrication attack) generates a large number of spurious detections.
    - **Mechanism**: AFOG-V replaces benign predictions with an empty set as the "ground truth" and negates the loss: $\mathcal{L}_{AFOG_V} = -\mathcal{L}_{bbox}(x_{adv}, \varnothing) - \mathcal{L}_{cls}(x_{adv}, \varnothing)$. AFOG-F removes the confidence threshold and sets all low-confidence predictions to 1.0 as "ground truth."
    - **Design Motivation**: To investigate the effect of adversarial perturbations on different detection behaviors — the vanishing attack probes the robustness of objectness detection, while the fabrication attack probes the robustness of bounding box prediction.

### Loss & Training
Attack hyperparameters: maximum perturbation budget $\epsilon = 0.031$ (on images normalized to $[0,1]$), number of iterations $T = 10$, attention learning rate $\alpha_A$ and perturbation learning rate $\alpha_P$ set separately. Ten iterations are applied uniformly across all models.

## Key Experimental Results

### Main Results: AFOG Attack Performance on 12 Detection Transformers

| Model | Params (M) | Benign mAP | AFOG | AFOG-V | AFOG-F | Reduction |
|-------|-----------|-----------|------|--------|--------|-----------|
| DETR-R50 | 39.8 | 42.1 | 4.1 | 4.5 | 9.8 | 10.3× |
| DETR-R101 | 76.0 | 43.5 | 5.2 | 5.1 | 11.3 | 8.4× |
| ViTDet | 108.1 | 54.9 | 3.8 | 0.9 | 2.8 | 14.4× |
| Swin-L | 217.2 | 56.8 | 7.3 | 2.4 | 8.6 | 7.8× |
| AlignDETR | 47.6 | 51.4 | 18.1 | 1.6 | **1.4** | **37.8×** |
| EVA | 1037.2 | 62.1 | 12.2 | 4.1 | 8.7 | 5.1× |

### Ablation Study: Contribution of Learnable Attention

| Configuration | DETR-R50 | Swin-L | InternImage | Avg. (12 models) |
|--------------|---------|--------|-------------|-----------------|
| AFOG w/o attention | Higher mAP | Higher mAP | Higher mAP | — |
| AFOG w/ attention | 4.1 | 7.3 | 7.3 | Avg. gain 15.1% |
| Max gain | — | — | **30.6%** (InternImage) | — |

Comparison with existing methods (DETR-R50):

| Attack | Type | Budget | Iterations | DETR-R50 mAP | Swin mAP |
|--------|------|--------|------------|-------------|---------|
| GARSDC | Surrogate | 0.05 | 3000+ | 6.0 | — |
| AttentionFool | Victim | — | 10–150 | 21.0 | — |
| EBAD | Victim | 0.039 | 10 | 34.9 | — |
| DBA | Victim | — | 50 | — | 56.7 |
| **AFOG** | Victim | **0.031** | **10** | **4.1** | **7.3** |

### Key Findings
- AFOG uses the smallest perturbation budget (0.031) and fewest iterations (10), substantially outperforming all existing attacks on both DETR-R50 and Swin-L.
- On Swin-L, AFOG outperforms the second-best attack by over 82.7% (DBA: 56.7 → AFOG: 7.3).
- AFOG-V (vanishing attack) outperforms the general AFOG on 11 out of 12 Transformers.
- The learnable attention mechanism yields an average improvement of 15.1%, with a maximum of 30.6% on InternImage.
- AFOG is equally effective on CNN detectors: mAP on Faster R-CNN drops from 67.37 to 2.38, surpassing all comparison methods.
- Attack stealthiness is excellent: SSIM > 0.83, L2 norm ≈ 0.032, visually imperceptible.

## Highlights & Insights
- **Architecture-agnostic unified attack**: The same method effectively attacks both Transformer and CNN detectors, filling a critical gap in the field.
- **Analysis of adversarial attention vs. model self-attention** is in-depth, demonstrating how the attack progressively disrupts the correlation structure of model self-attention, leading to a form of "catastrophic forgetting."
- **Failure case analysis** is valuable: when attention fails to focus on foreground objects, the attack fails, revealing an inherent limitation of the method.

## Limitations & Future Work
- White-box attack assumption (requires access to model parameters and gradients) limits applicability in real-world deployment scenarios.
- Failure cases suggest that attention initialization may affect attack outcomes; better strategies to guide attention toward target regions remain an open direction.
- The effect of the attack under defense strategies (e.g., adversarial training) has not been explored.
- Ten iterations may be insufficient for some large models (e.g., EVA).

## Related Work & Insights
- **vs. AttentionFool**: AttentionFool targets the dot-product self-attention in DETR specifically, cannot attack CNNs, and performs inconsistently on DETR-R50 (mAP 21.0 vs. AFOG 4.1).
- **vs. TOG**: TOG can directly attack single-stage CNN detectors but its performance on Transformers is unknown; AFOG outperforms TOG on both SSD and Faster R-CNN.
- **vs. DBA**: DBA prioritizes perturbing backgrounds to improve stealthiness, but is nearly ineffective on Swin (56.7 vs. benign 56.8); AFOG dynamically learns perturbation focus without static assumptions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of learnable adversarial attention is novel; the architecture-agnostic design is practically valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 12 Transformers + 3 CNNs, 11 baselines, comprehensive stealthiness analysis, and failure case study.
- **Writing Quality**: ⭐⭐⭐⭐ Derivations are clear; visualizations are rich (attention map evolution, self-attention disruption process).
- **Value**: ⭐⭐⭐⭐ Provides an effective diagnostic tool for robustness research on detection models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](lmm-det_make_large_multimodal_models_excel_in_object_detection.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[NeurIPS 2025\] Generalizable Insights for Graph Transformers in Theory and Practice](../../NeurIPS2025/object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)

</div>

<!-- RELATED:END -->
