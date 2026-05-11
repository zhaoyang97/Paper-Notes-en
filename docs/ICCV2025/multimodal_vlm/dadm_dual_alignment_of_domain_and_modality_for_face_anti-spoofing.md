---
title: >-
  [Paper Note] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing
description: >-
  [ICCV 2025][Multimodal VLM][Face Anti-Spoofing] This paper proposes the DADM framework, which simultaneously addresses intra-domain modality misalignment and inter-domain modality misalignment in multimodal face anti-spo…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Face Anti-Spoofing"
  - "Multimodal Fusion"
  - "Domain Generalization"
  - "Mutual Information"
  - "Invariant Risk Minimization"
date: 2026-05-08
content_hash: 8407e5b42a9c8e90
---

# DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing

**Conference**: ICCV 2025
**arXiv**: [2503.00429](https://arxiv.org/abs/2503.00429)
**Code**: [GitHub](https://github.com/)
**Area**: Multimodal VLM
**Keywords**: Face Anti-Spoofing, Multimodal Fusion, Domain Generalization, Mutual Information, Invariant Risk Minimization

## TL;DR

This paper proposes the DADM framework, which simultaneously addresses intra-domain modality misalignment and inter-domain modality misalignment in multimodal face anti-spoofing via a Mutual Information Mask (MIM) module and a dual domain-modality alignment optimization strategy, achieving state-of-the-art performance across four evaluation protocols.

## Background & Motivation

Multimodal face anti-spoofing (FAS) leverages complementary information from RGB, Depth, and Infrared modalities to detect spoofing attacks. However, existing multimodal FAS methods face two fundamental alignment challenges:

**Intra-domain modality misalignment**: The defensive capability of each modality varies considerably across different attack types. For instance, the Depth modality offers almost no defense against 3D mask attacks, while the RGB modality may be more sensitive to printed paper attacks. Naive fusion strategies cannot adapt to such dynamic variations and may even allow "harmful" modalities to degrade overall detection performance.

**Inter-domain modality misalignment**: Introducing additional modalities can exacerbate domain shift, as each modality may exhibit independent distribution drift across different datasets or capture devices. The conventional ERM paradigm mixes all source domains during training and cannot guarantee an optimal classification hyperplane for each sub-domain, causing models to rely on spurious correlations.

Prior work MMDG employs Monte Carlo dropout-based uncertainty estimation to identify unreliable modalities, but suffers from considerable randomness. This paper proposes a more principled solution grounded in mutual information maximization and invariant risk minimization.

## Method

### Overall Architecture

DADM is built upon a frozen pre-trained ViT-B/16 and extracts features from RGB, Depth, and Infrared modalities independently. Three MIM modules (handling RGB-D, RGB-I, and D-I pairs, respectively) are inserted after the MHSA output of each ViT layer to achieve intra-domain modality alignment via mutual information maximization. A CDC-Adapter is also introduced to capture fine-grained local features, and a dual alignment optimization strategy is applied for inter-domain modality alignment. Only the MIM modules and CDC-Adapter parameters are trainable.

### Key Designs

1. **Mutual Information Mask (MIM) Module**:

   - Function: Dynamically generates attention masks for each modality pair to enhance reliable regions and suppress unreliable ones.
   - Mechanism: The features of two modalities are concatenated and passed through a lightweight interaction convolution and a mask generation network, producing two sigmoid-activated masks $\mathbf{m}_{m1}, \mathbf{m}_{m2}$ that reweight the original features:
     $\mathbf{z}_{\text{aligned}\_m1} = \mathbf{m}_{m1} \mathbf{z}_{m1}$
     The aligned features are then globally average-pooled to yield MI tokens $z_{\text{mi1}}, z_{\text{mi2}}$, and mutual information is maximized via a simplified MINE estimator:
     $\mathcal{L}_{\text{mi}} = -\left[\mathbb{E}_{p(z_{\text{mi1}}, z_{\text{mi2}})}\left[\frac{z_{\text{mi1}} + z_{\text{mi2}}}{2}\right] - \log\left(\mathbb{E}_{p(z_{\text{mi1}})p(z_{\text{mi2}})}\left[e^{\frac{z_{\text{mi1}} + z_{\text{mi2}}}{2}}\right]\right)\right]$
   - Design Motivation: By directly using the mask-reweighted MI tokens as a special case of the score function, the method avoids the additional burden of training a separate neural network estimator as required in standard MINE. High mask values indicate informative and reliable regions, while low values correspond to redundant or detrimental information regions.

2. **MI-Guided Gradient Modulation (ReGrad)**:

   - Function: Adaptively adjusts the gradient direction of each MIM module based on the magnitude of MI tokens.
   - Mechanism: When the gradient directions of two modalities conflict (dot product < 0), the gradient of the modality with lower MI is projected and corrected; when directions are consistent, the modality with higher MI receives greater weight. This is implemented via four conditional branches:
     $\text{ReGrad}(\mathbf{g}_1, \mathbf{g}_2) = \begin{cases} \mathbf{g}_1 + \frac{\mathbf{g}_1 \cdot \mathbf{g}_2}{\|\mathbf{g}_1\|_2^2} \mathbf{g}_1 \cdot \text{mi}_2, & \text{if } \mathbf{g}_1 \cdot \mathbf{g}_2 < 0, \text{mi}_1 < \text{mi}_2 \\ \cdots & \end{cases}$
   - Design Motivation: Unlike the uncertainty-based ReGrad in MMDG, this approach measures modality reliability through mutual information magnitude, yielding a more deterministic and stable modulation.

3. **Dual Domain-Modality Alignment Optimization Strategy**:

   - Function: Simultaneously aligns sub-domain classification hyperplanes and inter-modality angular margins.
   - Mechanism:
     - **Hyperplane Alignment**: Adopts PG-IRM (Projected Gradient IRM) optimization to ensure that the globally optimal hyperplane is also locally optimal for each sub-domain, i.e., $\beta^* \in \arg\min_\beta R^e(\phi, \beta), \forall e \in \mathcal{E}$
     - **Angular Margin Alignment**: Constrains the cosine consistency between modality features of the same class across different domains:
       $$\mathcal{L}_{\text{angle}} = \sum_{e_1 \neq e_2} \sum_{i \neq j} \mathbb{I}(y_i=1) \cdot \left(\frac{\mathbf{z}_i^{e_1} \cdot \mathbf{z}_i^{e_2}}{\|\mathbf{z}_i^{e_1}\| \|\mathbf{z}_i^{e_2}\|} - \tau_l\right)^2 + \cdots$$
       where $\tau_l=1.0$ (strict alignment for live samples) and $\tau_s=0.85$ (relaxed alignment for spoof samples).
   - Design Motivation: Unimodal DG methods align only classification hyperplanes; however, in multimodal settings, significant domain shift (angular deviation) in any single modality can severely degrade overall performance. Explicitly constraining inter-modality angular consistency is therefore essential.

### Loss & Training

The total loss is the sum of three terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{ce} + \lambda_{\text{mi}} \mathcal{L}_{\text{mi}} + \lambda_{\text{angle}} \mathcal{L}_{\text{angle}}$$

where $\lambda_{\text{mi}} = 0.1$ and $\lambda_{\text{angle}} = 0.3$. PG-IRM is used to optimize the total loss. At inference, predictions are made using the mean of all sub-domain hyperplanes.

## Key Experimental Results

### Main Results

Average HTER(%)↓ / AUC(%)↑ across methods under Protocol 1 (fixed modality, LOO cross-domain):

| Method | Type | Avg. HTER(%)↓ | Avg. AUC(%)↑ | vs. MMDG Gain |
|--------|------|--------------|-------------|---------------|
| SSDG | DG | 34.22 | 70.20 | - |
| SSAN | DG | 29.15 | 77.22 | - |
| SA-FAS | DG | 28.77 | 78.18 | - |
| VP-FAS | FM | 25.45 | 81.08 | - |
| MMDG | MM-DG | 19.25 | 87.96 | - |
| **DADM (Ours)** | **MM-DG** | **13.63** | **92.96** | **HTER↓5.62, AUC↑5.00** |

### Ablation Study

Contribution of each component (Protocol 1 average):

| Configuration | HTER(%)↓ | AUC(%)↑ | Notes |
|---------------|---------|---------|-------|
| ViT + CE | 31.14 | 74.81 | Baseline |
| + U-Adapter + SSP (MMDG) | 24.54 | 83.14 | Reproduced Prev. SOTA |
| + MIM + CDC-Adapter | 21.75 | 86.17 | MIM replaces U-Adapter |
| + MI-Guided ReGrad | 17.17 | 88.14 | MI-guided gradient outperforms uncertainty-guided |
| + PG-IRM | 16.54 | 90.27 | Introduces invariant risk minimization |
| + DADM (dual alignment) | 14.31 | 92.05 | Adds angular margin alignment |
| + MI Loss (full DADM) | 13.63 | 92.96 | Adds mutual information maximization loss |

MI loss comparison: The proposed simplified MI estimator (13.63/92.96) outperforms both MINE (14.40/92.13) and InfoNCE (15.80/91.47) without incurring additional network overhead.

### Key Findings

- Under Protocol 2 (missing modality) and Protocol 3 (flexible modality), DADM also significantly outperforms all baselines, demonstrating that the dual alignment strategy enhances robustness to modality absence.
- Under Protocol 4 (limited source domains), DADM leads by a substantial margin, particularly reducing HTER from 36.60% (MMDG) to 20.40% in the PS→CW scenario.
- CDC-Adapter improves HTER by approximately 1% over standard convolutional adapters, validating the effectiveness of central difference convolution for capturing spoofing artifacts.

## Highlights & Insights

1. **Precise Problem Decomposition**: The alignment challenges in multimodal domain-generalized FAS are clearly decomposed into intra-domain modality alignment and inter-domain modality alignment, addressed with information-theoretic and optimization-theoretic tools respectively.
2. **Lightweight MI Estimation**: The method cleverly repurposes existing MI tokens as a special case of the score function, eliminating the need for an additional scoring network.
3. **Novel Angular Margin Alignment**: Unlike conventional methods that align only classification hyperplanes, explicitly constraining inter-modality angular consistency proves to be a key factor in handling multimodal domain shift.

## Limitations & Future Work

- Validation is currently limited to the RGB + Depth + Infrared three-modality setting; generalizability to broader modality combinations (e.g., near-infrared + thermal imaging) remains to be explored.
- The MIM module requires three instances per layer (one per modality pair), causing computational cost to scale quadratically with the number of modalities.
- The angular margin hyperparameters $\tau_l$ and $\tau_s$ are set manually and may require dataset-specific tuning.

## Related Work & Insights

- The PG-IRM optimization framework provides a unified theoretical perspective for multimodal domain generalization and is generalizable to other multimodal tasks.
- The mutual information masking idea of the MIM module is applicable to any scenario requiring dynamic modality fusion, such as multi-sensor fusion in autonomous driving.
- Angular margin alignment can be combined with contrastive learning to further strengthen cross-domain representation learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual alignment framework and simplified MI estimator are innovative, though the overall design is largely a combination of existing modules.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four protocols cover fixed/missing/flexible/limited source domain scenarios with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and mathematical derivations are complete, though the dense notation presents a moderate reading barrier.
- Value: ⭐⭐⭐⭐ Achieves significant advances in multimodal FAS; the dual alignment paradigm offers valuable reference for other multimodal DG tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/multimodal_vlm/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)

</div>

<!-- RELATED:END -->
