---
title: >-
  [Paper Note] Boosting Multimodal Learning via Disentangled Gradient Learning
description: >-
  [ICCV 2025][Signal & Communication][Multimodal Learning] This paper reveals an optimization conflict between modality encoders and fusion modules in multimodal learning — the fusion module suppresses gradients propagated…
tags:
  - "ICCV 2025"
  - "Signal & Communication"
  - "Multimodal Learning"
  - "Gradient Disentanglement"
  - "Modality Under-optimization"
  - "Gradient Modulation"
  - "Fusion Module Optimization"
date: 2026-05-08
content_hash: a9487b9388869373
---

# Boosting Multimodal Learning via Disentangled Gradient Learning

**Conference**: ICCV 2025
**arXiv**: [2507.10213](https://arxiv.org/abs/2507.10213)  
**Code**: [https://github.com/shicaiwei123/ICCV2025-GDL](https://github.com/shicaiwei123/ICCV2025-GDL)  
**Area**: Signal Communication
**Keywords**: Multimodal Learning, Gradient Disentanglement, Modality Under-optimization, Gradient Modulation, Fusion Module Optimization

## TL;DR
This paper reveals an optimization conflict between modality encoders and fusion modules in multimodal learning — the fusion module suppresses gradients propagated back to individual modality encoders, causing even the dominant modality to underperform its unimodal counterpart. The paper proposes the Disentangled Gradient Learning (DGL) framework, which addresses this issue by cutting the gradient path from the fusion module to the encoders and replacing it with independent unimodal losses.

## Background & Motivation

**Background**: Multimodal learning leverages complementary information from multiple sensors (visual+audio, RGB+depth, etc.) to improve task performance. Research has focused on designing fusion techniques (tensor-based, attention-based, etc.). However, naively combining multiple modalities does not always yield the expected performance gains.

**Limitations of Prior Work**:
   - Known issue: Multimodal models can underperform unimodal models (the "under-optimization" problem).
   - Existing explanation: The dominant modality suppresses optimization of weaker modalities ("modality imbalance"), motivating gradient modulation methods such as OGM, PMR, and AGM to amplify gradients for weaker modalities.
   - **Overlooked issue**: These methods focus solely on improving weaker modalities while ignoring the fact that **the dominant modality also underperforms its own unimodal baseline**. As shown in Fig. 1, audio (the dominant modality) performs worse in the multimodal model than in its unimodal counterpart.

**Key Challenge**: Why does the dominant modality also underperform in multimodal models? The existing "modality imbalance" explanation cannot answer this. The paper identifies the true cause: **the fusion module suppresses gradients propagated back to modality encoders**, and this suppression intensifies as training progresses.

**Goal**: To explain and resolve the under-optimization of all modalities (including the dominant modality) in multimodal learning.

**Key Insight**: Through mathematical derivation, it is shown that in multimodal models, gradients propagated from the fusion module back to the encoders via the chain rule are diminished compared to those in unimodal models. Specifically, the fusion operation introduces a scaling factor such that $\|g_{\theta_1}^{Multi}\| < \|g_{\theta_1}^{Uni}\|$.

**Core Idea**: Cut the gradient path from the fusion module to the encoders, provide each encoder with an undisturbed optimization signal via independent unimodal losses, and simultaneously block the gradient from unimodal losses back to the fusion module to prevent reverse interference.

## Method

### Overall Architecture
DGL introduces three gradient operations on top of a standard multimodal model (as illustrated in Fig. 2): (1) stopping the gradient of the multimodal loss $L^{Multi}$ from propagating back to the modality encoders $\phi_1, \phi_2$; (2) introducing an independent unimodal loss $L^{Uni}$ for each encoder to provide direct gradients; and (3) stopping the gradient of the unimodal losses from propagating back to the fusion module $\phi_\tau$. These three operations fully decouple the optimization of the encoders and the fusion module.

### Key Designs

1. **Theoretical Analysis of Gradient Suppression**:

    - Function: Mathematically proves that the fusion module suppresses encoder gradients.
    - Mechanism: Consider two modalities $m_1, m_2$ with encoder outputs $z^{m_1}, z^{m_2}$ fed into a fusion module $\phi_\tau$ to produce a fused representation $z^\tau$. By the chain rule, the gradient of the multimodal loss with respect to encoder 1 is $g_{\theta_1}^{Multi} = \frac{\partial L}{\partial z^\tau} \cdot \frac{\partial z^\tau}{\partial z^{m_1}} \cdot \frac{\partial z^{m_1}}{\partial \theta_1}$. The critical term is the intermediate Jacobian $\frac{\partial z^\tau}{\partial z^{m_1}}$ — since the fusion module mixes information from both modalities, the norm of this Jacobian is less than 1, causing gradient diminishment. Unimodal models lack this intermediate term and thus receive larger gradients.
    - Design Motivation: Provides a theoretical explanation for why even the dominant modality underperforms (prior work relied solely on empirical observations without theoretical grounding).

2. **Stop Gradient: Fusion → Encoders**:

    - Function: Prevents gradients of the multimodal loss from propagating through the fusion module back to the encoders.
    - Mechanism: A `stop_gradient` operation (in the encoder direction) is applied to $z^{m_1}$ and $z^{m_2}$ before they enter the fusion module. The fusion module still receives gradients from $L^{Multi}$ to optimize itself, but the suppressed gradients are not passed back to the encoders.
    - Design Motivation: Eliminates the suppression effect of the fusion module on encoder gradients.

3. **Independent Unimodal Losses (via Modality Dropout)**:

    - Function: Provides each encoder with an independent optimization signal.
    - Mechanism: A parameter-free modality dropout technique is employed — at each forward pass, one modality's input is randomly masked so that the fusion module receives information from only a single modality, enabling computation of $L^{Uni}_{m_1}$ and $L^{Uni}_{m_2}$. These unimodal losses are backpropagated directly to the corresponding encoders.
    - Design Motivation: After cutting the fusion-to-encoder gradient path, the encoders require an alternative gradient source. Modality dropout introduces no additional parameters, making it an elegant and lightweight solution.

4. **Stop Gradient: Unimodal Losses → Fusion Module**:

    - Function: Prevents gradients of unimodal losses from propagating back to the fusion module.
    - Mechanism: When computing unimodal losses, `stop_gradient` is applied to the fusion module parameters.
    - Design Motivation: Unimodal losses are designed to optimize encoders; allowing their gradients to reach the fusion module may conflict with those from $L^{Multi}$, thereby interfering with the fusion module's normal optimization.

### Loss & Training
- Total loss: $L = L^{Multi} + \lambda_1 L^{Uni}_{m_1} + \lambda_2 L^{Uni}_{m_2}$
- $L^{Multi}$ updates only the fusion module and the classifier.
- $L^{Uni}_{m_1}$ and $L^{Uni}_{m_2}$ update only the respective modality encoders.
- No architectural modifications are required; only the gradient flow is manipulated, endowing the method with strong generality.

## Key Experimental Results

### Main Results

| Method | CREMA-D (A-V) | Kinetics (A-V) | NYU-Depth (RGB-D) | Avg. Gain |
|--------|--------------|---------------|-------------------|-----------|
| Vanilla | 63.2 | 67.5 | 48.3 | baseline |
| OGM | 66.1 | 69.3 | 50.1 | +2.5 |
| PMR | 65.8 | 68.9 | 49.8 | +2.2 |
| AGM | 67.3 | 70.1 | 51.2 | +3.3 |
| MLA | 68.0 | 70.5 | 51.5 | +3.7 |
| **DGL** | **70.2** | **72.1** | **53.4** | **+5.6** |

### Ablation Study

| Configuration | CREMA-D | Note |
|---------------|--------|------|
| Full DGL | 70.2 | Complete model |
| w/o stop grad: fusion→encoders | 66.5 | Gradient decoupling is most critical |
| w/o unimodal losses | 67.1 | Encoders require independent gradient sources |
| w/o stop grad: unimodal→fusion | 68.9 | Reverse interference also has an impact |
| Stop grad: fusion→encoders only | 67.8 | All three components are indispensable |

### Key Findings
- DGL is effective across diverse modality combinations (audio-visual, RGB-depth), tasks (classification, detection, segmentation), and fusion methods (concatenation, attention, tensor).
- Stopping the fusion-to-encoder gradient is the most critical component; removing it causes a 3.7% drop.
- DGL improves not only the weaker modality but also the dominant modality significantly — something prior methods fail to achieve.
- The effect is particularly pronounced in fusion frameworks with dense cross-modal interactions.
- DGL is modality-agnostic, fusion-method-agnostic, and model-agnostic, exhibiting very high generality.

## Highlights & Insights
- **Theoretical Contribution**: The paper mathematically proves that the fusion module suppresses encoder gradients, explaining why even the dominant modality underperforms — a long-standing puzzle. This is a deeper analysis than the purely empirical "modality imbalance" explanation offered by prior work.
- **Minimalist Design**: DGL operates solely on the gradient flow (three `stop_gradient` operations) without modifying the network architecture or introducing additional parameters. The implementation is extremely concise (a few lines of code) yet yields substantial gains.
- **High Generality**: DGL is modality-agnostic, fusion-method-agnostic, and model-agnostic, making it applicable as a near plug-and-play addition to virtually any multimodal model.
- **Resolves Dominant Modality Under-optimization**: Prior methods (OGM, PMR, etc.) focus exclusively on weaker modalities and may even harm the dominant modality. DGL improves all modalities simultaneously.

## Limitations & Future Work
- Modality dropout increases the number of forward passes during training (one per modality), raising training cost.
- The theoretical analysis relies on simplified assumptions (linear classifiers, specific fusion forms); its applicability to complex nonlinear fusion warrants further investigation.
- The hyperparameters $\lambda_1, \lambda_2$ may affect performance and require tuning.
- The method has not been validated on large-scale pretrained multimodal models (e.g., CLIP, LLaVA).
- Adaptive control of the degree of decoupling, rather than complete gradient stopping, could be explored.

## Related Work & Insights
- **vs. OGM/AGM (gradient modulation)**: Gradient modulation methods attempt to balance gradient magnitudes across modalities but are still inherently subject to fusion-module gradient suppression. DGL addresses the problem at its root.
- **vs. MLA (alternating optimization)**: MLA replaces joint training with alternating unimodal training, thereby entirely forfeiting the joint training signal. DGL preserves joint training for the fusion module while decoupling only the encoders.
- **vs. MMPareto**: Pareto-based methods seek Pareto-optimal solutions in multi-objective optimization but do not resolve the fundamental issue of gradient suppression.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Deep theoretical insight (gradient suppression proof) with an elegant and concise solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple modalities, tasks, and fusion methods with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; the motivation–analysis–method–validation chain is logically complete.
- Value: ⭐⭐⭐⭐⭐ Makes a foundational contribution to the multimodal learning field with highly generalizable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](../../ICML2026/signal_comm/meta-learning_structure-preserving_dynamics.md)
- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](../../ICLR2026/signal_comm/learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](../../NeurIPS2025/signal_comm/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](../../NeurIPS2025/signal_comm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)

</div>

<!-- RELATED:END -->
