---
title: >-
  [Paper Note] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation
description: >-
  [AAAI 2026][Model Compression][Remote sensing semantic segmentation] This paper proposes Earth-Adapter, the first parameter-efficient fine-tuning (PEFT) method specifically designed to address **artifact problems** in re…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Remote sensing semantic segmentation"
  - "parameter-efficient fine-tuning"
  - "frequency-domain decomposition"
  - "mixture of adapters"
  - "artifact mitigation"
date: 2026-05-08
content_hash: a3a911f16ce6fbba
---

# Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation

**Conference**: AAAI 2026
**arXiv**: [2504.06220](https://arxiv.org/abs/2504.06220)  
**Code**: [Available](https://github.com/VisionXLab/Earth-Adapter)  
**Area**: Model Compression
**Keywords**: Remote sensing semantic segmentation, parameter-efficient fine-tuning, frequency-domain decomposition, mixture of adapters, artifact mitigation

## TL;DR

This paper proposes Earth-Adapter, the first parameter-efficient fine-tuning (PEFT) method specifically designed to address **artifact problems** in remote sensing imagery. Through a frequency-guided Mixture of Adapters (MoA), features are decomposed into high- and low-frequency subspaces, independently optimized, and then dynamically aggregated. The method outperforms the baseline Rein across three settings: remote sensing semantic segmentation (SS), domain adaptation (DA), and domain generalization (DG).

## Background & Motivation

Vision Foundation Models (VFMs) such as DINOv2 achieve strong performance on natural images, but exhibit significant performance degradation when combined with existing PEFT methods for remote sensing (RS) semantic segmentation. The authors identify the root cause as:

**Pervasive artifacts in VFM features**. PCA visualization of DINOv2-L feature maps reveals clearly redundant artifacts. The critical distinction is:

- In **natural images**, artifacts typically cluster around foreground objects (e.g., people, animals), resulting in relatively limited interference.
- In **remote sensing images**, the top-down perspective eliminates concentrated subjects, and multiple co-existing multi-scale targets (e.g., large-scale farmlands and fragmented road networks) cause **artifacts to appear nearly everywhere**, severely disrupting pixel-level feature extraction.

Existing PEFT methods (LoRA, VPT, AdaptFormer, etc.) do not address this issue. For instance, LoRA achieves only 17.8% mIoU on P2V (DA), even lower than frozen DINOv2-L.

Core strategy: **Divide and Conquer** — isolate artifact-related information via frequency-domain decomposition, then independently optimize different frequency subspaces.

## Method

### Overall Architecture

Earth-Adapter is embedded into a DINOv2-L (frozen) + Mask2Former architecture and consists of two core components:

1. **MoA (Mixture of Adapters)**: comprising a spatial adapter, a low-frequency adapter, and a high-frequency adapter.
2. **Router (Dynamic Router)**: adaptively assigns aggregation weights to the three adapters based on the original features.

The optimization objective is extended from standard fine-tuning $\arg\min_\theta$ to $\arg\min_{\theta,\epsilon,\xi}$, where $\epsilon$ and $\xi$ denote adapter and router parameters respectively, while the VFM backbone is frozen.

### Key Designs

#### Mixture of Adapters (MoA)

Given the feature at the $i$-th backbone layer $\mathbf{F}_i \in \mathbb{R}^{(hw) \times c}$:

**Spatial Adapter**: applies low-rank projection directly on spatial features:
$$\Delta \mathbf{F}_i^{spatial} = \text{Adapter}_1^i(\mathbf{F}_i^T)$$
Adapter structure: $\text{Adapter} = W_{up}(\text{ReLU}(W_{down}(\cdot)))$, standard bottleneck design.

**Frequency Decomposition**: the spatial feature is reshaped to $(C, H, W)$, a 2D DFT is applied, and a fixed cutoff frequency $\rho$ with frequency mask $\mathbf{M}$ separates low and high frequencies:
$$\mathbf{F}_i^{low} = \mathcal{FT}^{-1}(\mathbf{M} \odot \mathcal{FT}(\mathbf{F}_{spatial}))$$
$$\mathbf{F}_i^{high} = \mathcal{FT}^{-1}((1-\mathbf{M}) \odot \mathcal{FT}(\mathbf{F}_{spatial}))$$

Frequency mask definition: centered in the frequency domain, regions within distance $\leq \rho \frac{H}{2}$ from the center are low-frequency; the remainder are high-frequency.

**Low-/High-Frequency Adapters**: independently process the respective frequency features:
$$\Delta \mathbf{F}_i^{low} = \text{Adapter}_2^i(\mathbf{F}_i^{low}), \quad \Delta \mathbf{F}_i^{high} = \text{Adapter}_3^i(\mathbf{F}_i^{high})$$

Core rationale for "divide": high-frequency signals capture local details (where artifacts primarily concentrate), while low-frequency signals encode global structure. Separation enables targeted artifact suppression.

#### Dynamic Router

Channel attention learns optimal feature combination weights:
$$\mathbf{w}_i = \text{Softmax}(R_\xi(\mathbf{F}_i))$$

Final feature adjustment ("conquer"):
$$\Delta \mathbf{F}_i = \alpha_i \sum_{k=1}^{3} \mathbf{w}_i^{(k)} \Delta \mathbf{F}_i^{(k)}$$

where $\alpha_i$ is a learnable scaling parameter (initialized to a small value), and $k \in \{spatial, low, high\}$.

Frozen and refined features are fused via residual connection: $\bar{\mathbf{F}_i} = \mathbf{F}_i + \Delta \mathbf{F}_i$.

### Loss & Training

- **SS and DG**: standard end-to-end supervised training with Mask2Former, CE loss.
- **DA**: DACS self-training framework; an EMA teacher generates pseudo-labels for the target domain, and training mixes source and target domain samples.
- Optimizer: AdamW; backbone lr=1e-5; decoder and PEFT parameters lr=1e-4.
- Only adapter parameters $\epsilon$, router parameters $\xi$, and decoder parameters $\theta$ are trained; the backbone is fully frozen.

## Key Experimental Results

### Main Results

**Domain Adaptation (DA) + Domain Generalization (DG)** (4 RS datasets: Potsdam, Vaihingen, LoveDA, iSAID):

| Method | P2V DA | V2P DA | R2U DA | U2R DA | DA Mean | P2V DG | V2P DG | R2U DG | U2R DG | DG Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Frozen DINOv2-L | 21.0 | 7.2 | 21.5 | 11.8 | 15.4 | 57.9 | 49.4 | 57.1 | 42.7 | 51.8 |
| LoRA | 17.8 | 18.8 | 24.5 | 26.0 | 15.7 | 20.3 | 25.6 | 29.3 | 21.9 | 24.3 |
| VPT | 66.2 | 59.3 | 55.3 | 48.0 | 57.2 | 59.2 | 52.3 | 57.4 | 44.9 | 53.5 |
| Rein (baseline) | 60.2 | 60.9 | 52.8 | 26.0 | 50.0 | 60.8 | 52.5 | 55.8 | 43.4 | 53.1 |
| **Earth-Adapter** | **67.7** | **62.2** | **55.9** | **50.0** | **59.0** | **64.9** | **55.1** | **59.0** | **45.7** | **56.2** |

DA mean surpasses Rein by **+9.0%**; DG mean surpasses Rein by **+3.1%**. The gain on U2R DA is the most remarkable: **+24.0%**.

**Semantic Segmentation (SS)**:

| Method | Potsdam | Vaihingen | LoveDA | iSAID | Mean |
|---|---|---|---|---|---|
| Rein | 76.2 | 70.8 | 54.9 | 68.4 | 67.6 |
| **Earth-Adapter** | **76.7** | **71.7** | **56.9** | **69.8** | **68.8** |

SS mean surpasses Rein by **+1.2%**.

### Ablation Study

Adapter combination ablation (DG setting):

| Spatial | + HF | + LF | P2V DG | V2P DG | Mean |
|---|---|---|---|---|---|
| 1 | - | - | 61.0 | 52.8 | 56.9 |
| 3 | - | - | 64.0 | 52.7 | 58.4 (+1.5) |
| **1** | **1** | **1** | **64.9** | **55.1** | **60.0 (+3.1)** |

Simply increasing the number of spatial adapters (to 3) yields only +1.5%, whereas 1 spatial + 1 high-frequency + 1 low-frequency adapter yields +3.1%, demonstrating that **frequency-domain decomposition is more effective than simply stacking more adapters**.

Efficiency comparison:

| Method | Trainable Params | DA Mean mIoU |
|---|---|---|
| Full Fine-Tune | 304.2M | 15.4 |
| Rein | 3.0M | 50.0 |
| **Earth-Adapter** | **2.6M~9.6M** | **59.0** |

Backbone ablation: consistent improvements are observed across DINOv2-S/B/L; the method also proves effective on RS-pretrained VFMs (MTP, ScaleMAE, DOFA).

### Key Findings

- **LoRA and Full Fine-Tuning catastrophically fail in RS**: LoRA achieves only 15.7% DA mean; Full Fine-Tuning achieves 15.4%, demonstrating that PEFT strategies designed for natural images cannot be directly transferred to RS.
- **Frequency-domain decomposition is the key**: isolating artifacts in high-frequency components and preserving semantic structure in low-frequency components, the divide-and-conquer strategy significantly outperforms stacking homogeneous adapters.
- **DINOv2 outperforms RS-pretrained VFMs**: RS-pretrained models are limited by training data scale; DINOv2 + Earth-Adapter achieves superior results.
- **Frozen DINOv2 exhibits non-trivial DG capability**: under the DG setting, Frozen DINOv2-L already achieves 51.8%, outperforming LoRA/AdaptFormer, indicating that inappropriate fine-tuning can in fact harm generalization.

## Highlights & Insights

1. **First PEFT method targeting RS artifacts**: precisely identifies the cause of VFM performance degradation in remote sensing — artifact interference in high-dimensional features.
2. **Frequency-guided divide-and-conquer strategy**: DFT decomposition + independent adaptation + dynamic routing, elegant and effective.
3. **Extreme parameter efficiency**: only 2.6M–9.6M trainable parameters yield a substantial +9.0% gain in DA.
4. **Unified across three settings**: a single method applies to SS, DA, and DG simultaneously, demonstrating strong generality.

## Limitations & Future Work

- The frequency cutoff parameter $\rho$ is fixed rather than adaptive; the optimal frequency boundary may vary across different scenes.
- Validation is limited to remote sensing imagery; artifact characteristics differ in natural images, and the generalizability of the method remains to be verified.
- Router weight visualization and quantitative analysis are limited in the paper, leaving the interpretability of MoA's decision process insufficient.
- The method relies on DINOv2 as the backbone; other VFMs (e.g., InternViT, SigLIP) have not been explored.

## Related Work & Insights

- **Rein** (Wei et al. 2024): the first work to apply PEFT to DG, and the direct baseline of this paper.
- **DINOv2 + Register Tokens** (Darcet et al. 2024): mitigates artifacts via register tokens, but with limited effect (DG +1.7% vs. Earth-Adapter +3.1%).
- **DACS** (Tranheden et al. 2021): DA training framework; this paper adopts its self-training paradigm.
- **Revival of frequency-domain methods in vision**: DFT decomposition continues to gain traction in style transfer, domain adaptation, and feature denoising.

## Rating

- Novelty: ⭐⭐⭐⭐ — Precise identification of RS artifact problem + frequency-guided MoA design with strong originality.
- Technical Depth: ⭐⭐⭐⭐ — DFT decomposition + multi-adapter routing + cross-domain training framework, well integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 12 benchmarks, 3 settings, multiple backbones, comprehensive PEFT comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Rich visualizations (PCA, prediction comparisons), clear motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[AAAI 2026\] AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](adafuse_accelerating_dynamic_adapter_inference_via_token-lev.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](../../NeurIPS2025/model_compression/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[AAAI 2026\] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval](prototype-based_semantic_consistency_alignment_for_domain_adaptive_retrieval.md)

</div>

<!-- RELATED:END -->
