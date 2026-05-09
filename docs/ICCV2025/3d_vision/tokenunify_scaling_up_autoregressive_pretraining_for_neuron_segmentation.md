---
title: >-
  [Paper Note] TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation
description: >-
  [ICCV 2025][3D Vision][Autoregressive pretraining] TokenUnify is proposed to unify three complementary learning objectives—random token prediction, next-token prediction, and next-all-token prediction—enabling hierarchical predictive coding on large-scale electron microscopy data. The method reduces autoregressive error accumulation from $O(K)$ to $O(\sqrt{K})$, achieving a 44% improvement on downstream neuron segmentation.
tags:
  - ICCV 2025
  - 3D Vision
  - Autoregressive pretraining
  - neuron segmentation
  - electron microscopy
  - Mamba architecture
  - hierarchical predictive coding
date: 2026-05-08
content_hash: b4d6783f099e28ae
---

# TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation

**Conference**: ICCV 2025
**arXiv**: [2405.16847](https://arxiv.org/abs/2405.16847)
**Code**: [https://github.com/ydchen0806/TokenUnify](https://github.com/ydchen0806/TokenUnify)
**Area**: 3D Vision / Neuron Segmentation
**Keywords**: Autoregressive pretraining, neuron segmentation, electron microscopy, Mamba architecture, hierarchical predictive coding

## TL;DR

TokenUnify is proposed to unify three complementary learning objectives—random token prediction, next-token prediction, and next-all-token prediction—enabling hierarchical predictive coding on large-scale electron microscopy data. The method reduces autoregressive error accumulation from $O(K)$ to $O(\sqrt{K})$, achieving a 44% improvement on downstream neuron segmentation.

## Background & Motivation

**Background**: Neuron segmentation from electron microscopy (EM) volumetric images is a critical step toward understanding neural circuits. EM data exhibits three distinctive characteristics: (1) high noise (from electron beam interactions), (2) anisotropic voxels (coarser z-axis resolution), and (3) extremely long-range spatial dependencies (spanning thousands of patches).

**Limitations of Prior Work**:
- Contrastive learning (DINO v2) and masked reconstruction (MAE) yield strong representations but lack favorable scaling laws. The estimation error of MAE is $O(\sqrt{s \log p / n})$, with diminishing returns as model capacity grows.
- Autoregressive methods (AIM, LVM) attempt to bridge the gap, but standard autoregressive error **accumulates linearly** as $O(K)$, which is highly unfavorable for long sequences (K can reach thousands in EM data).
- Conventional vision models cannot effectively handle the long-range spatial continuity in EM data.

**Key Challenge**: Visual data structure is more complex than text—no single pretraining objective can simultaneously capture local spatial patterns, sequential dependencies, and global structure. The scaling laws that made autoregression successful in language have not transferred to vision.

**Key Insight**: Grounded in an **information-theoretic** perspective, this work proves that three prediction tasks capture complementary aspects of visual data structure. It leverages Mamba's linear-complexity sequence modeling to handle long-sequence EM data and constructs a large-scale EM dataset with 1.2 billion annotated voxels.

## Method

### Overall Architecture

A two-stage pipeline:
1. **Pretraining stage**: Train a general visual representation $f_{\theta_1}(\cdot)$ on 1 TB+ of unannotated EM data using three complementary prediction tasks.
2. **Fine-tuning stage**: Fine-tune the segmentation model $g_{\theta_2}(\cdot)$ on annotated data, initialized from pretrained weights.

Input 3D EM volumes are divided into $D' \times H' \times W'$ patches, tokenized into a sequence of length $K$, and processed efficiently by Mamba.

### Key Designs

1. **Random Token Prediction (micro-level)**: Analogous to MAE, a fraction $\rho$ of tokens is randomly masked, and masked tokens are predicted from unmasked context:
    $$\mathcal{L}_{random} = -\mathbb{E}_{\mathcal{M} \sim \mathcal{D}_\rho} \left[\sum_{i \in \mathcal{M}} \log p_\theta(x_i | x_{\mathcal{M}^c})\right]$$
    Function: Learns position-invariant local feature detectors, robust to noise, capturing repetitive patterns of cell membranes and organelles.

2. **Next-Token Prediction (meso-level)**: Autoregressive modeling along a predefined path $\pi$:
    $$\mathcal{L}_{next} = -\mathbb{E}_\pi \left[\sum_{i=1}^K \log p_\theta(x_{\pi(i)} | x_{\pi(<i)})\right]$$
    Function: Captures transitional patterns in neuron morphology—membrane continuity, dendrite/axon directional consistency, and other meso-scale structures.

3. **Next-All-Token Prediction (macro-level)**: Predicts **all** subsequent tokens given the preceding context:
    $$\mathcal{L}_{next\text{-}all} = -\mathbb{E}_\pi \left[\sum_{i=1}^K \sum_{j=i}^K \log p_\theta(x_{\pi(j)} | x_{\pi(<i)})\right]$$
    Function: Captures long-range correlations such as branching patterns, cell-type-specific morphology, and regional organization. **Key theoretical contribution**—prediction errors are distributed across multiple positions rather than accumulated, analogous to the central limit theorem, reducing error from $O(K)$ to $O(\sqrt{K})$. A Perceiver Resampler aggregates global sequence information via cross-attention to maintain computational efficiency.

4. **Multi-Resolution Optimization Protocol**: Curriculum learning-style weight scheduling—from easy to hard:
    - $t < T_1$ (30%): random prediction dominant (weight 0.73)
    - $T_1 \leq t < T_2$ (70%): next-token prediction dominant
    - $t \geq T_2$: next-all prediction dominant

    Smooth transitions are achieved via softmax temperature decay, while auxiliary task contributions (~0.18 and ~0.09) are maintained throughout to preserve multi-task synergy.

5. **EMmamba Segmentation Network**: An encoder-decoder architecture improved from SegMamba, using anisotropic downsampling layers (no downsampling along the z-axis) to accommodate the anisotropic resolution of EM data.

### Loss & Training

Unified pretraining objective:
$$\mathcal{L}_{TokenUnify} = \alpha(t) \cdot \mathcal{L}_{random} + \beta(t) \cdot \mathcal{L}_{next} + \gamma(t) \cdot \mathcal{L}_{next\text{-}all}$$

Segmentation fine-tuning uses affinity map prediction with MSE loss; post-processing employs seeded watershed followed by region merging.

## Key Experimental Results

### Main Results: MEC Dataset (Waterz Post-processing)

| Pretraining Method | VOI_M↓ | VOI_S↓ | VOI↓ | ARAND↓ |
|---|---|---|---|---|
| Random (no pretraining) | 0.4915 | 1.2924 | 1.7839 | 0.2052 |
| MAE | 0.2325 | 1.0923 | 1.3248 | 0.0978 |
| BYOL | 0.2584 | 0.9453 | 1.2037 | 0.0891 |
| dbMIM | 0.2342 | 0.8796 | 1.1138 | 0.0742 |
| **TokenUnify** | **0.1953** | **0.7998** | **0.9951** | **0.0509** |

TokenUnify achieves a **44%** improvement over random initialization (VOI: 1.78→1.00) and a **25%** improvement over MAE (1.32→1.00).

### Ablation Study

| Pretraining Strategy | VOI↓ | ARAND↓ |
|---|---|---|
| Random (mask prediction only) | 1.2680 | 0.0862 |
| Next (autoregression only) | 4.0418 | 0.4416 |
| Random + Next | 1.1300 | 0.0692 |
| Random + Next-all | 1.1907 | 0.1203 |
| **Random + Next + Next-all** | **0.9951** | **0.0509** |

Key observations:
- Pure autoregression (Next only) performs poorly (VOI=4.04)—standalone autoregression is unsuitable for vision tasks that require global spatial understanding.
- The full combination of all three strategies is optimal, validating the complementarity hypothesis.
- Random prediction provides spatially consistent initialization (1.27) and is the best single strategy.

| Fine-tuned Module | VOI↓ | ARAND↓ |
|---|---|---|
| Mamba blocks only | 1.1362 | 0.0782 |
| Encoder only | 1.5556 | 0.1370 |
| Decoder only | 1.5295 | 0.1212 |
| Mamba + Encoder | 1.1065 | 0.0629 |
| **Full fine-tuning** | **0.9951** | **0.0509** |

Mamba blocks are the most critical component (core of sequence modeling capability); under resource constraints, they should be fine-tuned first.

### Key Findings

- **Scaling Law**: From 100M to 1B parameters, TokenUnify consistently outperforms other methods, exhibiting language model-like scaling behavior. Mamba achieves comparable performance with fewer parameters than Transformers, validating the efficiency of linear-complexity architectures on long-sequence visual data.
- **AC3/4 small dataset** (only 1/10 of MEC annotation volume): TokenUnify + Mamba approaches the performance of supervised SOTA method PEA and improves over MAE by 11%, demonstrating effectiveness in annotation-scarce scenarios.
- **Preliminary cross-domain validation**: Pretraining on Kodak natural images, TokenUnify achieves reconstruction quality 2–4 dB higher in PSNR than pure autoregression, indicating the framework is not restricted to the EM domain.

## Highlights & Insights

- **Information-theoretic unification**: The three prediction tasks capture $I(x_i; x_{\mathcal{M}^c})$, $I(x_i; x_{<i})$, and $I(\{x_i,...,x_K\}; x_{<i})$ respectively, jointly maximizing total information extraction.
- **Error reduction from $O(K)$ to $O(\sqrt{K})$**: Achieved by distributing errors across multiple positions in next-all prediction, analogous to the $\sqrt{n}$ scaling of the central limit theorem. This represents a significant theoretical advancement over pure autoregressive approaches.
- **First billion-parameter Mamba vision network**: Demonstrates the scaling feasibility of Mamba for long-sequence visual modeling.
- **MEC dataset with 1.2 billion annotated voxels**: Covering 6 functional brain regions, annotated by two experts over 6 months, making it one of the largest annotated EM datasets of its kind.

## Limitations & Future Work

- Next-all prediction uses a Perceiver Resampler as approximation; it is worth exploring whether a more direct global prediction mechanism can be designed.
- The curriculum learning stage boundaries (30%/70%) and weight ratios are set empirically; adaptive scheduling could be considered.
- Mamba shows substantial gains for small models (28M parameters), but the original EMmamba without pretraining underperforms traditional CNNs (e.g., Superhuman with 1.5M parameters), suggesting that the Mamba architecture itself still has room to improve annotation efficiency.

## Related Work & Insights

- AIM [El-Nouby 2024] and LVM [Bai 2023] explore autoregressive pretraining on natural images but lack the multi-task complementary design and theoretical analysis of TokenUnify.
- MAGE [Li 2023] combines masked and generative objectives but is limited to 2D; TokenUnify extends this to 3D long sequences.
- Compared to EM-specific methods dbMIM and MS-Con-EM, TokenUnify shows significant advantages on the same Mamba backbone, demonstrating the importance of pretraining objective design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Information-theoretic unification of three prediction tasks + error accumulation analysis
- Technical Depth: ⭐⭐⭐⭐ — Theoretical analysis (partly in appendix) + multi-resolution optimization protocol
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale dataset construction, multi-method comparison, scaling analysis, comprehensive ablation
- Value: ⭐⭐⭐⭐ — Direct applicability to connectomics and biological image analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](egom2p_egocentric_multimodal_multitask_pretraining.md)
- [\[ICCV 2025\] PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [\[ICCV 2025\] Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing](trace3d_consistent_segmentation_lifting_via_gaussian_instance_tracing.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](../../CVPR2026/3d_vision/hyperbolic_multiview_pretraining_for_robotic_manipulation.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)

</div>

<!-- RELATED:END -->
