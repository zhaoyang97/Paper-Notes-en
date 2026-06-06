---
title: >-
  [Paper Note] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding
description: >-
  [NeurIPS 2025][Medical Imaging][fMRI visual decoding] This paper proposes MoRE-Brain, a neuroscience-inspired fMRI visual decoding framework that employs a hierarchical Mixture-of-Experts (MoE) architecture to simulate t…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "fMRI visual decoding"
  - "mixture of experts"
  - "interpretability"
  - "cross-subject generalization"
  - "diffusion model"
date: 2026-05-08
content_hash: 7f01fbeae4ce0734
---

# MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2505.15946](https://arxiv.org/abs/2505.15946)  
**Code**: [GitHub](https://github.com/yuxiangwei0808/MoRE-Brain)  
**Area**: 3D Vision
**Keywords**: fMRI visual decoding, mixture of experts, interpretability, cross-subject generalization, diffusion model

## TL;DR

This paper proposes MoRE-Brain, a neuroscience-inspired fMRI visual decoding framework that employs a hierarchical Mixture-of-Experts (MoE) architecture to simulate the specialized processing of the brain's visual pathway. Combined with a dynamic temporal-spatial dual-routing mechanism that guides image generation via a diffusion model, MoRE-Brain achieves high-fidelity reconstruction while enabling efficient cross-subject generalization and unprecedented mechanistic interpretability.

## Background & Motivation

Decoding visual experiences from fMRI signals is a critical avenue for understanding human visual perception and developing brain–computer interfaces. Although current state-of-the-art methods (e.g., MindEye2, MindBridge) have achieved notable advances in reconstruction fidelity, three fundamental problems remain:

**1. Lack of interpretability**: Existing methods treat fMRI signals as monolithic inputs, without architectural designs that reflect neural processing principles. Although some works coarsely partition the brain into "higher visual cortex" and "lower visual cortex," this simplification ignores the complexity of multiple hierarchical processing stages and multiple specialized cortical regions within the visual pathway.

**2. Difficulty in cross-subject generalization**: Due to inter-individual brain variability, models typically require subject-specific training. Existing approaches (e.g., shared-space mapping, subject-specific tokens) demand additional training and do not scale well to new subjects.

**3. Over-reliance on generative priors**: Leading methods may rely primarily on the generative priors learned by the diffusion model rather than on fMRI signals themselves, making reconstruction quality insensitive to variations in fMRI information content.

**Core Idea**: Drawing on neuroscientific principles of functional specialization and hierarchical processing, the paper designs a hierarchical MoE architecture in which different experts process fMRI signals from functionally correlated voxel groups. Inter-individual differences arise primarily from differences in the spatial topology of functional networks rather than from differences in fundamental computation; therefore, expert computations can be shared while only subject-specific routers need to be adapted.

## Method

### Overall Architecture

MoRE-Brain adopts a two-stage pipeline:

- **Stage 1**: A hierarchical MoE maps fMRI signals into a frozen CLIP space (learning specialized decoding).
- **Stage 2**: Multi-level expert embeddings are integrated into the denoising process of SDXL via a dynamic temporal-spatial routing mechanism (guiding image generation).

### Key Designs

#### 1. Hierarchical MoE fMRI Encoder

Given input fMRI voxels $\mathcal{F} \in \mathbb{R}^v$, the MoE architecture learns data-driven voxel assignment without predefined brain parcellations.

**Routing computation**: At layer $l$, the router computes voxel-to-expert affinity scores based on input features:

$$A^{(l)} = W_r^{(l)} X^{(l)}$$

A softmax over the expert dimension yields probabilities $P^{(l)}$, followed by Top-K selection for each expert:

$$(S_j^{(l)}, I_j^{(l)}) = \text{TopK}(P_{:,j}^{(l)}, k)$$

where $k = \lfloor v / e_l \times c_f \rfloor$ and capacity factor $c_f = 1$ enforces non-overlapping assignment.

**Hierarchical structure**: Starting from $e_0 = 2$ experts in the first layer, the number doubles at each layer over $L = 4$ layers, yielding $2^4 = 16$ experts at the final layer—a number that approximates the count of functional ROIs identified in visual cortex atlases. Each expert is a simple MLP processing the subset of voxels assigned to it.

**Design motivation**: This structure simulates the coarse-to-fine hierarchical processing of the brain's visual pathway—lower-level experts integrate signals from broad visual regions, while higher-level experts focus on finer-grained specific cortical sub-regions, forming functionally specialized internal representations.

#### 2. Dynamic Temporal-Spatial Routing Mechanism

**Temporal router $\mathcal{R}_T$**: Determines which hierarchical level of expert embeddings to use based on the current diffusion timestep $t$:

$$P_T = \text{softmax}\left(\frac{(W_Q t_c) \cdot (W_K \phi)^T}{\sqrt{d_k}}\right)$$

where $\phi = [\phi_1, ..., \phi_L]$ are learnable level embeddings. To encourage coarse-to-fine processing, KL divergence regularization against a Gaussian guidance distribution is applied:

$$\bar{P}_{T,l} = \frac{\exp(-(l - \mu_t)^2 / 2\sigma^2)}{\sum_{k=1}^L \exp(-(k - \mu_t)^2 / 2\sigma^2)}, \quad \mu_t = L \cdot \frac{t}{T}$$

**Spatial router $\mathcal{R}_S$**: Given the selected level's expert embeddings, performs spatial modulation conditioned on the current noisy latent $z_t$:

$$C = \text{softmax}\left(\frac{\hat{W}_Q z_t \cdot \hat{W}_K E_{sel}}{\sqrt{d_k}}\right) \cdot \hat{W}_V E_{sel}$$

The result $C$ serves as the conditioning input for the cross-attention layers of the SDXL U-Net.

**Design motivation**: Temporal routing reflects the coarse-to-fine dynamics of visual processing—early diffusion steps (high noise) correspond to global scene layout (lower-level experts), while later steps (low noise) correspond to fine details (higher-level experts). Spatial routing simulates the brain's integration of features from different specialized regions (shape, color, motion) into a spatially coherent percept.

#### 3. Cross-Subject Generalization Strategy

Based on neuroscientific findings that inter-individual differences stem primarily from differences in the spatial topology of functional networks rather than from differences in fundamental computation:

- **Shared**: All expert networks (core decoding computation).
- **Subject-specific**: Only the router weights (mapping individual brain topology to experts).

After training the full model on Subject 1, the experts are frozen and only the routers are fine-tuned to generalize to new subjects.

### Loss & Training

- Stage 1: Expert outputs are aligned with frozen CLIP embeddings.
- Stage 2: Standard LDM noise prediction loss + KL divergence regularization for temporal routing. The SDXL U-Net is fine-tuned via LoRA.
- Training data: NSD dataset (fMRI responses of 8 subjects viewing 30,000+ natural images).

## Key Experimental Results

### Main Results (Bottleneck Analysis Reveals True fMRI Utilization)

| Method | Total Params (M) | Fine-tuned Params (%) | SSIM↑ | DreamSim↓ | InceptionV3↑ |
|--------|-----------------|----------------------|-------|-----------|-------------|
| MindEye2 | 729.3 | 100% | ~0.41 | ~0.53 | ~0.94 |
| MindBridge | 552.9 | 98.48% | ~0.42 | ~0.56 | ~0.90 |
| **MoRE-Brain** | **293.4** | **44.84%** | ~0.42 | ~0.51 | ~0.96 |

MoRE-Brain achieves competitive performance with fewer than half the parameters. Bottleneck analysis reveals that MoRE-Brain is the most sensitive method to information constraints (exhibiting the largest performance drop), demonstrating that it genuinely exploits fMRI signals. In contrast, MindEye2 maintains CLIP-Cos as high as ~0.8 even under extreme bottleneck conditions, suggesting an over-reliance on generative priors.

### Ablation Study

| Configuration | SSIM↑ | Alex(2)↑ | InceptionV3↑ | DreamSim↓ |
|--------------|-------|---------|-------------|-----------|
| Text+Image (TS) | **0.415** | **0.792** | **0.962** | **0.507** |
| Text only | 0.410 | 0.748 | 0.926 | 0.560 |
| Text only (TS) | 0.402 | 0.754 | 0.948 | 0.542 |
| Image only | 0.422 | 0.742 | 0.893 | 0.605 |
| Image only (TS) | 0.382 | 0.802 | 0.957 | 0.528 |
| Text+Image (S only) | 0.375 | 0.762 | 0.943 | 0.546 |
| Text+Image (T, fixed) | 0.403 | 0.765 | 0.940 | 0.533 |
| Text+Image (T, hard) | 0.397 | 0.650 | 0.867 | 0.626 |
| Text+Image (T, soft) | 0.402 | 0.764 | 0.941 | 0.543 |

The combination of dual conditioning (text + image) and dual routing (TS) achieves the best overall performance.

### Key Findings

1. **Genuine fMRI utilization**: Bottleneck analysis shows that MoRE-Brain is the only method whose performance degrades significantly under information constraints, demonstrating its reliance on true neural information rather than generative priors alone.
2. **Emergent hierarchical specialization**: Lower-level experts process signals from broad visual regions, while higher-level experts focus on specific cortical sub-regions; different experts within the same level show preferences for distinct voxel groups.
3. **Semantic specialization**: Higher-level experts develop preferences for specific semantic categories (e.g., one expert preferring "outdoor" scenes), whereas lower-level experts exhibit more diffuse responses.
4. **Efficient cross-subject generalization**: Fine-tuning only the routers with as little as 2.5% of data (1 session) yields reasonable performance on new subjects.
5. **ICA validation**: Independent component analysis shows that the features learned by the model align with known visual processing regions and higher-order association areas.

## Highlights & Insights

- **Deep integration of neuroscience and deep learning**: Rather than being superficially "inspired" by neuroscience, the paper systematically embeds principles such as hierarchical processing, functional specialization, and coarse-to-fine dynamics into the architectural design.
- **Interpretability as a design objective**: The routing weights enable precise tracing of how different modeled brain regions shape the semantic and spatial attributes of reconstructed images across temporal and spatial dimensions.
- **Bottleneck analysis methodology**: A systematic approach for distinguishing "genuine neural decoding" from "over-reliance on generative priors" is proposed, offering methodological significance for the broader fMRI decoding community.

## Limitations & Future Work

- Validation is currently limited to the NSD dataset; generalization to other fMRI datasets remains unknown.
- The number of final-level experts (16) is chosen empirically based on visual cortex atlases; a broader search space may yield better configurations.
- Although cross-subject generalization is efficient, a certain amount of new-subject data is still required for router fine-tuning.
- The coarse-to-fine assumption underlying the temporal router may be overly rigid, as some visual information processing does not strictly follow a coarse-to-fine trajectory.

## Related Work & Insights

- MindEye2 achieves cross-subject shared modeling but lacks interpretability; MoRE-Brain addresses both generalization and interpretability simultaneously through MoE.
- The image conditioning injection approach from IP-Adapter is applied to integrate fMRI embeddings into the diffusion model.
- The application of MoE architectures to fMRI decoding is entirely novel and may influence other neuroimaging analysis tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Deep integration of MoE with neuroscientific principles; the temporal-spatial dual-routing mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Bottleneck analysis, ablation, cross-subject evaluation, and interpretability analysis are comprehensive, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, methodology is rigorous, visualizations are rich, and neuroscientific interpretation is substantive.
- Value: ⭐⭐⭐⭐⭐ — Simultaneously advances interpretability and cross-subject generalization in the fMRI decoding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](../../ICLR2026/medical_imaging/towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/medical_imaging/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](../../ICCV2025/medical_imaging/beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)

</div>

<!-- RELATED:END -->
