---
title: >-
  [Paper Note] LayerLock: Non-collapsing Representation Learning with Progressive Freezing
description: >-
  [3D Vision] This paper proposes LayerLock, a self-supervised video representation learning method that progressively freezes network layers while dynamically shifting prediction targets from pixels to increasingly deep intermediate layer features. It combines the training stability of pixel prediction with the semantic efficiency of latent variable prediction, and is applied to video models with up to 4B parameters.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 2ef7d00ad5767541
---

# LayerLock: Non-collapsing Representation Learning with Progressive Freezing

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2509.10156](https://arxiv.org/abs/2509.10156)
- **Authors**: Goker Erdogan, Nikhil Parthasarathy, Catalin Ionescu, Drew A. Hudson, Alexander Lerchner, Andrew Zisserman, Mehdi S. M. Sajjadi, João Carreira
- **Institution**: Google DeepMind, University of Oxford
- **Area**: 3D Vision / Self-Supervised Representation Learning
- **Keywords**: Self-supervised learning, video representation, progressive freezing, latent variable prediction, Masked Auto-Encoding, V-JEPA

## TL;DR
This paper proposes LayerLock, a self-supervised video representation learning method that progressively freezes network layers while dynamically shifting prediction targets from pixels to increasingly deep intermediate layer features. It combines the training stability of pixel prediction with the semantic efficiency of latent variable prediction, and is applied to video models with up to 4B parameters.

## Background & Motivation

### Problem Definition
The two dominant paradigms of self-supervised visual representation learning each have inherent trade-offs:
- **Pixel prediction** (e.g., VideoMAE): provides stable learning signals and avoids representation collapse, but requires large amounts of training data and tends to capture low-level visual information misaligned with downstream tasks.
- **Latent variable prediction** (e.g., V-JEPA): is more data-efficient and captures higher-level semantic features, but requires various training tricks (asymmetric architectures, EMA teacher networks, stop-gradient) to prevent representation collapse.

### Key Observation
During Video MAE training, **ViT layers converge in order of depth**: shallower layers converge first, followed by deeper layers. This phenomenon is validated through freezing experiments — if freezing the first $L$ layers at a given step and continuing training results in a final loss close to the baseline $L_{base}$, those layers are considered to have converged.

### Core Insight
Exploiting this layer-wise convergence pattern, the paper proposes **progressive freezing with dynamic target switching**: training begins with pixel prediction (stable), and as shallow layers converge, they are progressively frozen while the prediction target shifts to the intermediate activations of increasingly deep frozen layers. This retains the stability of pixel prediction while gradually incorporating the semantic learning advantages of latent variable prediction, all without representation collapse.

## Method

### Overall Architecture
LayerLock builds on the standard MAE architecture, with the core modification being the management of prediction targets during training:
1. The initial phase predicts pixels (standard MAE) for $N_{\text{pixel}}$ steps.
2. Subsequently, every $N$ steps, the next $k$ layers are frozen and the prediction target is switched to the output $h_k$ of the newly frozen layer.
3. This process continues progressively until 3/4 of the network layers are frozen.

### Model Architecture

**Encoder**: A standard ViT backbone. Input video $x \in \mathbb{R}^{T \times H \times W \times 3}$ is divided into $2 \times 16 \times 16$ patches, randomly masked at a 95% ratio, and linearly projected to $D$ dimensions.

**Decoder**: The last 4 layers of the backbone are reused as the decoder. Decoding latent tokens $z \in \mathbb{R}^{N \times D}$ are concatenated before a specified backbone layer, processed through the remaining Transformer blocks, and projected to the target space via a per-patch linear layer.

**3D Rotary Position Encoding (3D RoPE)**: Feature dimensions are split into 4 parts; 1D RoPE is applied independently along the temporal, height, and width axes (at ratios of 10%, 25%, and 25%), while the fourth part carries no positional information. A key finding is that applying RoPE after the first normalization layer is more effective than applying it within the attention mechanism.

### Progressive Freezing and Latent Prediction

**Freezing schedule** (for ViT-G):
- First 160K steps: predict pixels (standard MAE loss).
- Every 10K steps: freeze 1 layer and switch the prediction target to the output of that newly frozen layer.
- Continue until 32 layers (2/3 of the 48 total) are frozen.
- L2 loss is used: $\text{loss} = \|h_k - \hat{h}_k\|^2$

**Schedule determination heuristic**: Freezing begins when the parameter norms of the initial layers plateau (or begin to decrease under weight decay).

**Mini-warmup at each target switch**: The learning rate is gradually increased at each target transition to avoid training instability from abrupt target changes (yielding approximately 1% improvement on SSv2).

### Extension to V-JEPA
LayerLock is also applicable to methods that already use latent variable prediction:
- Training initially targets the first-layer activations of an EMA teacher network.
- Layers are progressively frozen and targets shift to deeper activations.
- A ViT-L backbone with a 12-layer Transformer decoder is used.
- The EMA teacher network is retained (while progressive freezing itself prevents collapse, EMA yields better downstream performance).

## Key Experimental Results

### Main Results: LayerLock vs. Baselines

| Model | LayerLock | Params (M) | SSv2↑ | K700↑ | ScanNet↓ |
|------|-----------|----------|-------|-------|----------|
| 4DS-G (MAE) | ✗ | 1,848 | 63.1 | 52.1 | 1.02 |
| 4DS-G (MAE) | ✓ | 1,868 | **66.1** | **56.3** | **1.00** |
| 4DS-e (MAE) | ✗ | 3,811 | 64.6 | 54.4 | 0.94 |
| 4DS-e (MAE) | ✓ | 3,818 | **67.1** | **57.9** | 0.98 |
| V-JEPA-L | ✗ | 235 | 52.1 | 42.5 | 1.57 |
| V-JEPA-L | ✓ | 303 | **57.0** | **43.5** | **1.51** |

- SSv2 action classification improves by approximately 3% (MAE) and 5% (V-JEPA), which is highly significant.
- K700 action classification also shows notable gains.
- Depth estimation (ScanNet) performance is maintained or slightly improved.

### Ablation Study

**Efficiency ablation**: Progressive freezing saves 9% total FLOPs and 16% peak memory with negligible performance loss.

| Configuration | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| Baseline MAE (no freezing) | 56.1 | 0.15 |
| Progressive freezing MAE | 56.0 | 0.16 |

**MAE + latent loss leads to collapse**:

| Model | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| 4DS-H | 50.1 | 0.19 |
| + latent (const weight) | 3.7 | 0.38 |
| + latent (cosine schedule) | 5.6 | 0.37 |

Directly adding a latent loss to MAE without freezing results in severe representation collapse, confirming that progressive freezing is essential for stable latent prediction.

**3D RoPE ablation**:

| Configuration | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| No RoPE | 56.1 | 0.15 |
| + 3D RoPE | 58.9 | 0.13 |
| + 3D RoPE + LayerLock | 60.1 | 0.13 |

RoPE independently contributes approximately 2.8% SSv2 improvement and is complementary to LayerLock.

**Patch subsampling efficiency ablation**:

| Configuration | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| Baseline MAE | 63.1 | 1.02 |
| LayerLock 5% patches | 64.9 | 1.12 |
| LayerLock 100% patches | 66.1 | 1.00 |

Computing the latent loss on only 5% of patches still significantly outperforms the baseline, though with a slight degradation in depth estimation.

### Key Findings
1. **Layer-wise convergence ordering**: Shallow layers converge before deep layers, offering important guidance for training strategy design.
2. **Single target vs. multi-target**: Predicting only the most recently frozen layer's output (single target) performs comparably to predicting all frozen layers simultaneously (multi-target), making the simpler approach sufficient.
3. **Freezing schedule**: Gradual layer-by-layer freezing (interval=2K, jump=1) performs best; freezing too many layers at once significantly degrades performance.
4. **Longer training benefits more**: At 1B training samples, the FLOP efficiency gain can reach 19%.

## Highlights & Insights

1. **Observation-driven design**: The progressive freezing strategy is naturally derived from the empirical observation that layers converge in order of depth.
2. **Elegant collapse prevention**: Stable latent prediction is achieved without complex techniques such as asymmetric architectures or contrastive losses — progressive freezing with dynamic target switching suffices.
3. **Strong generality**: The method is effective across both pixel prediction (4DS MAE) and latent variable prediction (V-JEPA) paradigms and scales to 4B-parameter models.
4. **Computational efficiency**: Progressive freezing not only improves representation quality but also reduces training FLOPs and peak memory consumption.
5. **Analogy to biological visual development**: The pattern of shallow layers converging and stabilizing first echoes the concept of "critical period plasticity" in neuroscience.

## Limitations & Future Work

- The freezing schedule currently relies on manual heuristics (observing parameter norm trends) and lacks an adaptive mechanism.
- Improvements on low-level tasks such as depth estimation are less pronounced than on high-level semantic tasks.
- Validation is limited to video tasks; extension to image-based self-supervised learning remains unexplored.
- The training scale is extremely large (1B video clips, 256 TPU-v6), making reproduction challenging.

## Related Work & Insights

- **vs. V-JEPA**: LayerLock provides stable training targets via progressive freezing, partially substituting the role of the EMA teacher network (though experiments show that retaining EMA yields better performance).
- **vs. FreezeOut**: FreezeOut uses progressive freezing in supervised learning to accelerate training; LayerLock extends this into a target-switching strategy for self-supervised representation learning.
- **vs. 4DS**: Building on the MAE foundation of 4DS, LayerLock achieves richer semantic features through a gradual transition from pixel to latent prediction targets.
- **Inspiration**: The progressive freezing paradigm can be extended to longer videos, higher resolutions, and deeper models, with computational efficiency gains that become more significant at larger training scales.

## Rating ⭐⭐⭐⭐⭐
The paper offers deep empirical insights and a simple yet elegant method with clear theoretical motivation. It achieves consistent improvements across both dominant self-supervised paradigms while maintaining computational efficiency. The ablation studies are exceptionally comprehensive (covering freezing schedules, patch subsampling, collapse verification, RoPE, etc.), establishing a new training paradigm for large-scale video self-supervised learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)

</div>

<!-- RELATED:END -->
