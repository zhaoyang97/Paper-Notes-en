---
title: >-
  [Paper Note] Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization
description: >-
  [ICLR 2026][Image Restoration][LayerNorm] The authors track the training process of Image Restoration (IR) Transformers and discover that standard LayerNorm causes feature magnitudes to diverge to the million-level scale and channel entropy to collapse sharply. The root cause is identified as LN's "per-token normalization" and "input-independent scaling," whic
tags:
  - ICLR 2026
  - Image Restoration
  - LayerNorm
  - Transformer
date: 2026-05-08
content_hash: 94ef565ebc8602df
---
# Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=SbLj5hJXh6](https://openreview.net/forum?id=SbLj5hJXh6)  
**Code**: https://github.com/2minkyulee/i-LN  
**Area**: Image Restoration  
**Keywords**: Image Restoration, LayerNorm, Transformer, Training Dynamics, Feature Divergence

## TL;DR
The authors track the training process of Image Restoration (IR) Transformers and discover that standard LayerNorm causes feature magnitudes to diverge to the million-level scale and channel entropy to collapse sharply. The root cause is identified as LN's "per-token normalization" and "input-independent scaling," which conflict with IR tasks. Consequently, they propose i-LN—a plug-and-play replacement for LN that performs normalization across the entire spatial-channel dimension and adaptively adds the scaling factor back after each Attention/FFN block. This stabilizes training and consistently improves performance in SR, denoising, deraining, and JPEG artifact removal.

## Background & Motivation
**Background**: Since the popularity of Vision Transformers, the mainstream backbones for image restoration (Super-Resolution SR, Denoising DN, Deraining DR, JPEG Compression Artifact Removal CAR) have shifted to combinations of Transformer + LayerNorm (e.g., SwinIR, HAT, DRCT). LN has become a default standard. However, research mostly focuses on "how many points can be gained by switching to a stronger architecture," with little investigation into the **internal training dynamics** of these IR Transformers.

**Limitations of Prior Work**: By plotting the intermediate feature trajectories of each basic block in HAT for ×4 SR, the authors identified two overlooked anomalies: ① Feature mean squares **diverge exponentially during training, reaching million (or even ten-million) magnitudes**; ② Feature entropy in the channel dimension **drops sharply at the very early stages of training**, indicating that a few extreme-value channels dominate the overall statistics. These "latent anomalous features" are **ubiquitous** across deeper/wider networks and various IR tasks/Transformer backbones, with larger networks exhibiting faster and more severe divergence.

**Key Challenge**: The authors hypothesize that feature divergence is not a network failure but rather the network **actively attempting to bypass LN**. LN exhibits two fundamental mismatches with IR tasks:

- **Per-token normalization breaks spatial correlation**: LN calculates the mean and variance for each token (pixel) separately, ignoring the relative relationships between tokens, whereas IR highly depends on the spatial structure between pixels.
- **Input-independent scaling discards input statistics**: LN maps all intermediate features into a unified normalized space, restricting the flexibility of representation ranges. This erases input-dependent low-level statistical information crucial for faithful reconstruction in IR tasks.

An intuitive solution would be to **entirely remove normalization layers** (as done in early SR works like EDSR), but the authors' experiments show that IR Transformers become **extremely unstable and fail to converge** without normalization.

**Goal**: To find a normalization scheme that maintains stable training without conflicting with IR tasks.

**Key Insight**: Re-examine LN from a geometric perspective of "structure preservation." Treating a set of tokens as a point cloud in $\mathbb{R}^C$, an ideal transformation should only apply global scaling and translation (homothety) to the entire cloud. The authors **prove** that per-token LN is generally not even a conformal transformation and inevitably destroys the shape of the point cloud.

**Core Idea**: Replace per-token LN with "cross-spatial-channel global normalization + input-adaptive rescaling" to preserve spatial structure between tokens while explicitly reintroducing the global scale information removed by normalization.

## Method

### Overall Architecture
i-LN (Image Restoration Transformer Tailored Layer Normalization) is a **plug-and-play replacement** for standard LN. it does not modify Attention/FFN or the backbone structure; it only replaces the normalization operator in each Transformer block. It consists of two complementary modifications: first, changing "per-token normalization" to "global normalization over the entire spatial-channel dimensions (LN*)"; second, using "input-adaptive rescaling" to explicitly add back the scale factor at the output of each Attention/FFN. Together, these ensure the network preserves low-level feature statistics across the entire forward path, eliminating the need to "bypass" normalization via million-magnitude features while maintaining training stability.

### Key Designs

**1. Spatial Global Normalization LN\*: Preserving Inter-pixel Structure**

Standard per-token LN calculates statistics for the $\ell$-th token: $\mathrm{LN}(x_\ell)=\gamma\frac{1}{\sqrt{\sigma_\ell^2+\epsilon}}(x_\ell-\mu_\ell)+\beta$, where $\mu_\ell, \sigma_\ell^2$ are expected values only over the channel dimension $c$. The problem is that each token is scaled by a different $(\mu_\ell, \sigma_\ell)$, distorting relative differences $x_\ell-x_k$. The authors formalize "inter-pixel structure" as the set of relative differences $\Delta x=\{x_\ell-x_k\}$ and prove (Proposition 1) that per-token LN is generally not a homothety, thus destroying spatial structure.

The modification for LN\* is simple: change the mean and variance to be **calculated over both spatial dimension $\ell$ and channel dimension $c$**, i.e., $\mu=\mathbb{E}_{\ell,c}[x_{\ell,c}]$ and $\sigma^2=\mathbb{E}_{\ell,c}[(x_{\ell,c}-\mu)^2]$, so all tokens share the same global statistics. Consequently, the difference between any two tokens remains $T_{\mathrm{LN}^*}(x_\ell)-T_{\mathrm{LN}^*}(x_k)=\frac{1}{\sigma}(x_\ell-x_k)$ (Proposition 2), which is a standard homothety. The entire point cloud is uniformly scaled, and the spatial structure is preserved.

**2. Input-Adaptive Rescaling: Restoring the Global Scale**

LN\* preserves structure but at the cost of losing the global scale scalar $\sigma$—which IR tasks need to retain for faithful reconstruction. The solution is to re-scale the output after each Attention or FFN using the **standard deviation already calculated** during the normalization step. A block $B$ is rewritten as:

$$B(x;f,\text{i-LN}) = x + \sqrt{\sigma^2+\epsilon}\cdot f(\mathrm{LN}^*(x)),$$

where $f$ denotes the Attention or FFN operation. This "yellow branch" (Fig. 3 in the paper) multiplies the $\sigma$ removed during normalization back onto the residual branch. This restores flexibility to the range of intermediate features and preserves per-instance statistics since $\sigma$ varies with each input image.

### Loss & Training
The method introduces no new loss functions and uses standard pixel reconstruction losses for IR tasks. For fair comparison, the authors **re-implemented** all baselines and their method under a unified setting: Rain13K for deraining and DF2K (DIV2K+Flickr2K) for other tasks; only basic augmentations (flipping, rotation, cropping) were used, excluding mixup or progressive patching. Backbones used include SwinIR, HAT, and DRCT, with model sizes adjusted according to computational constraints (denoted by subscripts like HAT1, HAT2).

## Key Experimental Results

### Main Results
As a direct replacement for LN, i-LN consistently improves performance across four IR tasks and various backbones. The following table compares various normalization schemes on ×4 SR (HAT1), where i-LN performs best:

| Dataset (x4 SR) | Metric | LayerNorm | InstanceNorm | i-LN (Ours) |
|--------|------|------|----------|------|
| Set14 | PSNR / SSIM | 28.79 / .7876 | 28.98 / .7907 | **29.01 / .7915** |
| BSD100 | PSNR / SSIM | 27.68 / .7411 | 27.80 / .7445 | **27.84 / .7456** |
| Urban100 | PSNR / SSIM | 26.55 / .8015 | 27.02 / .8136 | **27.17 / .8167** |
| Manga109 | PSNR / SSIM | 31.01 / .9150 | 31.46 / .9199 | **31.82 / .9228** |

LN is the **worst** among all schemes. Per-token variants like LayerScale/RMSNorm are slightly better than LN but still lose to spatial-global schemes. Removing normalization (None) or using ReZero results in poor convergence due to instability. Spatial-global schemes like InstanceNorm/BatchNorm outperform per-token ones, but BN drops significantly in eval mode (indicating a need for per-image statistics), and IN/BN lose channel info—making i-LN optimal.

Across tasks, SR and deraining show the most significant gains (e.g., HAT1 deraining on Rain100L improved from 34.35 → **36.20** dB; SwinIR1 on Test100 from 27.45 → **29.87** dB). Gains in denoising and JPEG artifact removal were smaller, as their uniform degradation spreads the benefits of "preserving specific input features."

### Ablation Study
Ablations on HAT2 for ×4 SR shows the effect of removing i-LN components (SH = Spatial Homogeneity, Rs = Rescaling):

| Configuration | Description | Effect |
|------|------|------|
| i-LN (Full) | LN\* + Input-adaptive Rescaling | Best quality and feature stability |
| Remove Rs → LN\* | Spatial-global norm only, no rescaling | Drop in quality, channel entropy collapses |
| Remove SH → LN | Revert to per-token LN | Channel entropy **collapses exponentially** to vanilla LN levels |

### Key Findings
- **Per-token operation is the culprit for divergence**: Fig. 4 shows that all per-token normalization schemes (LN, RMSNorm, LayerScale) diverge, while all spatial-global schemes (i-LN, BN, IN, ReZero) do not.
- **Complementary components**: Channel entropy collapses exponentially as SH and Rs are removed, proving that LN\* preserves structures and rescaling restores global scale; both are necessary for healthy activation distributions.
- **Scale sensitivity**: Features diverge faster and reach higher magnitudes as networks become deeper/wider. This reveals a unique risk in scaling IR models: expansion increases not just capacity but also pathological feature growth.
- **Compensatory mechanism**: Baselines converge despite severe feature imbalance because the affine bias of the last LN layer aligns with input channel magnitudes (Fig. 8)—a forced "detour" rather than a healthy state.

## Highlights & Insights
- **Model of "Analysis-Driven Design"**: Visualizing a long-overlooked anomaly (million-level divergence + entropy collapse) to identify root causes in LN, followed by a simple two-line code fix.
- **Clever Geometric Perspective**: Viewing tokens as point clouds and defining "good normalization" as a homothety allows for **formal proof** that per-token LN destroys structure while LN\* preserves it.
- **Minimalist and Plug-and-Play**: i-LN adds no parameters and requires no architectural changes. Its "near-zero cost" makes it easily adoptable.
- **Transferable Insights**: The realization that "per-token normalization destroys spatial structure" is a warning for any low-level vision task (optical flow, depth estimation) relying on spatial correlations.

## Limitations & Future Work
- **Task-Dependent Gains**: Improvements in denoising and JPEG removal are small (often 0.05~0.1 dB), suggesting i-LN benefits most when "reliable regions" exist in the input.
- **Theory Based on Simplified Parameters**: Structure preservation proofs ignore affine parameters $\gamma, \beta$; the rigorous behavior with affine parameters is not fully explored.
- **Scope Limited to SR-type Backbones**: Experiments focus on SwinIR/HAT/DRCT for classic degradations; generalization to deblurring, low-light enhancement, or all-in-one models requires verification.

## Related Work & Insights
- **vs. Standard LayerNorm**: Standard LN is per-token and input-independent, leading to divergence and entropy collapse; i-LN is global and adaptive, eliminating these issues for general gains.
- **vs. No Normalization (EDSR)**: Early CNN-based SR avoided normalization to keep input statistics. While Transformers need normalization for stability, i-LN combines stability with statistical preservation.
- **vs. ReZero / LayerScale**: These use zero-initialized scaling to stabilize Transformers. However, ReZero (no norm) and LayerScale (per-token) are inferior to i-LN, indicating the key isn't just "residual scaling" but "spatial homogeneity + statistical preservation."
- **vs. InstanceNorm / BatchNorm**: These avoid divergence but lose channel information or suffer from train/eval mismatch; i-LN uses per-image statistics and preserves channel info.

## Rating
- Novelty: ⭐⭐⭐⭐ Transparent diagnosis of a neglected training dynamic issue with simple, theoretically-grounded solutions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across tasks, backbones, and normalization variants + scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from phenomenon to diagnosis to solution; strong visual evidence and rigorous propositions.
- Value: ⭐⭐⭐⭐ Plug-and-play with zero cost, offering immediate practical utility for the IR Transformer community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers](../../CVPR2026/image_restoration/diffdecompose_layer-wise_decomposition_of_alpha-composited_images_via_diffusion_.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](../../CVPR2025/image_restoration/dpir_dual_prompting_restoration_dit.md)
- [\[ICLR 2026\] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)
- [\[CVPR 2026\] FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model](../../CVPR2026/image_restoration/foundir-v2_optimizing_pre-training_data_mixtures_for_image_restoration_foundatio.md)
- [\[CVPR 2026\] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](../../CVPR2026/image_restoration/reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)

</div>

<!-- RELATED:END -->
