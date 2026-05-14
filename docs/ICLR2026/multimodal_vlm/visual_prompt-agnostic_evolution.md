---
title: >-
  [Paper Note] Visual Prompt-Agnostic Evolution
description: >-
  [ICLR2026][Multimodal VLM][Visual Prompt Tuning] This paper proposes Prompt-Agnostic Evolution (PAE), which accelerates VPT convergence (average 1.41×) and improves accuracy by 1–3% across 25 datasets through frequency-a…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "Visual Prompt Tuning"
  - "Vision Transformer"
  - "Parameter-Efficient Fine-Tuning"
  - "Koopman Operator"
  - "Frequency-Domain Initialization"
  - "Lyapunov Stability"
date: 2026-05-08
content_hash: 42a5ba2ad4b410c1
---

# Visual Prompt-Agnostic Evolution

**Conference**: ICLR2026
**arXiv**: [2601.20232](https://arxiv.org/abs/2601.20232)
**Code**: [reeive/PAE](https://github.com/reeive/PAE)
**Area**: Multimodal VLM
**Keywords**: Visual Prompt Tuning, Vision Transformer, Parameter-Efficient Fine-Tuning, Koopman Operator, Frequency-Domain Initialization, Lyapunov Stability

## TL;DR
This paper proposes Prompt-Agnostic Evolution (PAE), which accelerates VPT convergence (average 1.41×) and improves accuracy by 1–3% across 25 datasets through frequency-aware task initialization (MPA) and a Koopman-Lyapunov dynamical system (KLD) for cross-layer prompt coupling. PAE is plug-and-play for various VPT variants and introduces no inference overhead.

## Background & Motivation
1. **Success and limitations of VPT**: Visual Prompt Tuning (VPT) inserts a small number of learnable prompt tokens into each frozen ViT layer to enable downstream adaptation, achieving parameter efficiency but suffering from slow convergence and suboptimal accuracy in practice.
2. **Gradient oscillation**: Experiments reveal significant gradient oscillations across multiple VPT variants during training, most severe in the early and middle stages.
3. **Cross-layer mismatch**: Layer-wise gradient analysis shows that shallow-layer prompts experience a sharp gradient spike early in training followed by rapid stagnation, while deep-layer prompts exhibit high-variance oscillations, leading to severely uncoordinated inter-layer optimization.
4. **Task-agnostic initialization**: Existing VPT initialization strategies are insensitive to downstream tasks, causing early gradients to be largely wasted on aligning with the pretrained backbone. Higher learning rates required to compensate further exacerbate instability.
5. **Independent layer-wise optimization**: Prompts at each layer are initialized and optimized independently; gradients must backpropagate through multiple frozen layers, causing severe signal attenuation in shallow layers and over-adjustment in deep layers, with no explicit cross-layer coordination.
6. **Proliferation of VPT variants without resolving root issues**: Improvements in four directions—structured prompts, adaptive prompts, projection-based prompts, and perception-driven prompts—have not fundamentally addressed the aforementioned training dynamics problems.

## Method
PAE comprises two complementary modules: MPA (initialization phase) and KLD (training phase).

### Modal Pre-Alignment (MPA) — Frequency-Aware Initialization
- **Frequency shortcut discovery**: A mini-batch from the training set is processed via 2D Fourier transform. A sliding window ($w=16$, $\text{stride}=8$) generates $S$ binary frequency masks $M_s$, each applied to the spectrum before inverse-transforming the image back. The reconstructed images are ranked by task loss to identify the frequency shortcut regions most relied upon by the model.
- **Prompt construction**: The Top-$T$ frequency masks with the lowest loss are selected. Filtered images are passed through the frozen patch embedding to obtain patch tokens, which are then aggregated into $T$ representative vectors via energy-weighted pooling (squared L2 norm as weights), forming the first-layer prompt $P_1^{\text{init}}$.
- **Cross-layer propagation**: $P_1^{\text{init}}$ is fed layer-by-layer through the frozen Transformer encoder blocks; each layer's output serves as the initialization $P_i^{\text{init}}$ for the corresponding prompt, ensuring the initialization trajectory is consistent with the backbone's hierarchical semantics.

### Koopman-Lyapunov Discrete Dynamical System (KLD) — Cross-Layer Coupled Optimization
- **Linear projection**: A globally learnable projection matrix $U \in \mathbb{R}^{d \times K}$ (Kaiming initialization) projects each layer's prompt $P_i$ into a shared latent space: $z_i = P_i \cdot U$.
- **Koopman operator evolution**: A globally shared operator $K \in \mathbb{R}^{K \times K}$ (identity initialization) models cross-layer prompt evolution as $\hat{z}_{i+1} = z_i \cdot K$, replacing independent layer-wise optimization with explicit cross-layer coupling.
- **Koopman consistency loss $\mathcal{L}_{\text{kp}}$**: Minimizes the Frobenius norm between the predicted evolution state $\hat{z}_{i+1}$ and the actual projected state $z_{i+1}$, so that each layer's prompt gradient is simultaneously constrained by the consistency of adjacent layers.
- **Lyapunov stability regularization $\mathcal{L}_{\text{stab}}$**: Defines the Lyapunov function $V(z) = \text{tr}(zQz^\top)$, where $Q$ is a learnable symmetric positive definite matrix. Penalties are applied only when $V$ increases between adjacent layers, adaptively constraining accumulated evolution errors.
- **Total objective**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{kp}} + \beta \cdot \mathcal{L}_{\text{stab}}$ ($\alpha=0.5$, $\beta=0.2$), trained end-to-end.

### Key Design Advantages
- **Prompt-agnostic**: Seamlessly integrable into any VPT variant without modifying the backbone or inference pipeline.
- **Lightweight**: MPA initialization takes only ~74 seconds (~5.3 epochs); KLD introduces minimal additional parameters ($K=256$-dimensional Koopman space).

## Key Experimental Results

### Table 1: Classification Accuracy and Speedup on FGVC + VTAB-1k (ViT-B/16)

| Method + PAE | Speedup | FGVC | VTAB-Natural | VTAB-Specialized | VTAB-Structured | VTAB Mean |
|---|---|---|---|---|---|---|
| Full Fine-tune | - | 88.54 | 75.88 | 83.36 | 47.64 | 68.96 |
| VPT + PAE | 1.78× | 89.11 (+1.91) | 78.48 (+3.25) | 82.43 (+2.09) | 54.98 (+3.30) | 71.96 (+2.88) |
| E2VPT + PAE | 1.65× | 89.22 (+1.74) | 80.01 (+1.38) | 84.43 (+1.33) | 57.39 (+2.34) | 73.94 (+1.68) |
| VFPT + PAE | 1.27× | 89.24 (+2.24) | 81.35 (+0.72) | 84.93 (+1.03) | 60.19 (+0.77) | 75.39 (+0.94) |
| SA2VP + PAE | 1.60× | 90.08 (+1.12) | 80.97 (+1.89) | 85.73 (+0.85) | 60.80 (+2.25) | 75.83 (+1.66) |
| BPT + PAE | 1.37× | 90.86 (+1.35) | 80.24 (+2.22) | 84.45 (+1.88) | 60.39 (+1.66) | 75.02 (+1.92) |

### Table 2: Ablation Study (VPT baseline, ViT-B/16)

| MPA | $\mathcal{L}_{\text{kp}}$ | $\mathcal{L}_{\text{stab}}$ | FGVC | VTAB Mean |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 89.11 | 71.96 |
| ✓ | ✗ | ✗ | 89.63 | 74.02 |
| ✗ | ✓ | ✗ | 90.56 | 73.13 |
| ✗ | ✓ | ✓ | 90.78 | 74.42 |
| ✓ | ✓ | ✓ | 91.02 | 74.84 |

- MPA alone contributes the largest individual gain (VTAB +2.06%); combining both KLD losses yields a further +1.29%.
- ADE20K semantic segmentation (ViT-L): PAE improves mIoU by 2–3% for VPT/E2VPT/VFPT with 1.15–1.29× speedup.
- Cross-architecture generalization: Consistent gains across ViT-B/16, Swin-B, ViT-L/16, and ViT-H/14.
- Prompt CKA visualization: PAE produces a clear diagonal band structure in prompt representations, indicating progressive depth-aware evolution replacing global redundancy.
- High-variance classes benefit most: Categories with greater intra-class variance obtain larger relative accuracy gains from PAE.

## Highlights & Insights
- **First formalization of VPT as a dynamical systems control problem over prompt trajectories**, offering an entirely new perspective.
- **Frequency-domain initialization (MPA) deeply exploits the backbone's frequency bias**, achieving task-aware initialization without additional data or pretraining.
- **Koopman operator cross-layer coupling** elegantly resolves the core bottleneck of VPT's layer-wise independent optimization—shallow-layer stagnation and deep-layer oscillation.
- **Plug-and-play with zero inference overhead**: integrable into 8 distinct VPT variants with no modifications to the backbone.
- **Highly comprehensive experiments**: covering 25 datasets, 4 backbone architectures, classification and segmentation tasks, and multi-dimensional visualization analyses.
- **Loss landscape analysis**: PAE drives optimization toward wider and flatter minima, with significantly reduced maximum Hessian eigenvalues and condition numbers, theoretically explaining improved generalization.
- **Grad-CAM visualization**: VPT+PAE focuses on class-discriminative regions as early as epoch 5, whereas vanilla VPT remains unstable even at epoch 50.
- **Negligible initialization cost**: The entire MPA initialization takes only 74 seconds, equivalent to ~5 training epochs, offering an excellent cost-performance ratio.

## Limitations & Future Work
- The Koopman operator assumes approximately linear inter-layer prompt evolution, an assumption that may not hold in very deep or heterogeneous architectures.
- The frequency window search in MPA, though lightweight, introduces additional preprocessing time (~74s) that may accumulate in large-scale continual learning scenarios.
- Experiments primarily cover image classification and semantic segmentation; generalization to more complex visual tasks such as detection and video understanding remains unverified.
- No adaptive scheme is provided for selecting hyperparameters $\alpha$ and $\beta$, which may require separate tuning for different datasets.
- The choice of Koopman space dimensionality $K=256$ lacks theoretical justification.
- The paper does not explore combining PAE with textual prompt tuning methods (e.g., CoOp/CoCoOp).
- Classification accuracy improvements on self-supervised pretrained (MAE) backbones are not separately reported; only CKA visualizations are shown.

## Related Work & Insights
- **vs. VPT/E2VPT/ProVP and other structured prompts**: PAE does not alter prompt structural design; rather, it enhances initialization and optimization dynamics, making the two approaches orthogonal and complementary.
- **vs. VFPT (frequency-domain prompts)**: VFPT re-weights prompt features in the frequency domain, whereas PAE uses frequency-domain shortcuts to initialize prompts—distinct starting points. Applying PAE on top of VFPT still yields +0.94% gain.
- **vs. GatePT**: GatePT adjusts prompts via gating mechanisms, but CKA analysis reveals its cross-layer prompts remain highly redundant; PAE's Koopman evolution achieves superior progressive depth differentiation.
- **vs. other PEFT methods (LoRA/Adapter)**: PAE focuses on accelerating optimization within the prompt tuning paradigm and represents a different PEFT direction from LoRA, with potential for combination.
- **vs. LPT (adaptive prompts)**: LPT dynamically combines shared and group-specific prompts for long-tailed distributions; PAE can be stacked on top, achieving 1.44× speedup and +1.81% VTAB mean improvement.
- **vs. Full Fine-tuning**: Multiple VPT+PAE combinations significantly outperform full fine-tuning on VTAB-1k (e.g., SA2VP+PAE: 75.83% vs. Full Fine-tune: 68.96%), using fewer than 1% of the parameters.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Recasting VPT optimization as a dynamical systems problem; the Koopman+Lyapunov theoretical framework is highly original.
- **Overall**: A practically valuable prompt tuning enhancement work that balances theory and practice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 25 datasets, 8 VPT variants, 4 architectures, classification + segmentation, comprehensive ablations and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear and theoretical derivations are complete, though the heavy notation makes reading somewhat demanding.
- **Value**: ⭐⭐⭐⭐ — A general-purpose plug-and-play VPT accelerator with direct practical value for the prompt tuning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[NeurIPS 2025\] VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion](../../NeurIPS2025/multimodal_vlm/vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](../../ICCV2025/multimodal_vlm/pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](../../ICCV2025/multimodal_vlm/attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](../../ICCV2025/multimodal_vlm/cvpt_cross_visual_prompt_tuning.md)

</div>

<!-- RELATED:END -->
