---
title: >-
  [Paper Note] MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy
description: >-
  [CVPR 2026][Medical Imaging][Multi-Resolution] This paper proposes MuViT, a multi-resolution Vision Transformer that employs world-coordinate RoPE positional encoding to jointly process crops of the same scene at differe…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Multi-Resolution"
  - "Vision Transformer"
  - "RoPE"
  - "Microscopy"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: 1e26a037c85774cb
---

# MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy

**Conference**: CVPR 2026
**arXiv**: [2602.24222](https://arxiv.org/abs/2602.24222)
**Code**: [github.com/weigertlab/muvit](https://github.com/weigertlab/muvit)
**Area**: Medical Imaging
**Keywords**: Multi-Resolution, Vision Transformer, RoPE, Microscopy, Semantic Segmentation

## TL;DR

This paper proposes MuViT, a multi-resolution Vision Transformer that employs world-coordinate RoPE positional encoding to jointly process crops of the same scene at different physical resolutions within a single encoder, achieving substantial improvements over single-resolution baselines on microscopy image segmentation tasks.

## Background & Motivation

Modern microscopy modalities—including light-sheet fluorescence microscopy, electron microscopy, and digital pathology—routinely produce gigapixel images exceeding 50K×50K pixels, containing structures that span multiple spatial scales ranging from cellular morphology to tissue architecture. Many analysis tasks require simultaneous exploitation of multi-scale information: for instance, semantic segmentation of individual cells demands knowledge of the surrounding tissue region (global context) alongside fine-grained local detail.

The core tension in existing approaches is as follows:

- **CNN/ViT tile-based prediction**: GPU memory constraints restrict processing to fixed-size tiles (e.g., 512×512), imposing an inherent trade-off between field of view and resolution.
- **Hierarchical architectures (Swin/PVT/HIPT)**: These construct internal feature pyramids from a single-resolution input but do not exploit truly multi-resolution observations.
- **Multi-path models (CrossViT/MPViT)**: These operate on artificially constructed scale variants and lack geometric consistency across scales.

The central insight of MuViT is that different spatial scales can serve as complementary input "modalities," provided they share a unified geometric reference frame.

## Method

### Overall Architecture

MuViT receives crops of the same image at $L$ different physical resolutions, $\mathbf{X} \in \mathbb{R}^{L \times C \times H \times W}$, together with their spatial bounding boxes $\mathcal{B} \in \mathbb{R}^{L \times 2 \times 2}$. Patches from all resolutions are projected into a unified world coordinate system, and cross-resolution information fusion is achieved within a single encoder via RoPE-based attention. The notation MuViT$_{[l_1, l_2, \ldots]}$ denotes an encoder using resolution levels $l_1, l_2, \ldots$, where $l=1$ corresponds to the highest resolution and $l>1$ indicates $l\times$ downsampling.

### Key Designs

1. **World-Coordinate RoPE Positional Encoding**: The center coordinate of each patch is mapped to the pixel coordinate system of the highest resolution (referred to as world coordinates), and 2D axial RoPE injects these coordinates into attention:

$$\theta_k^{(a)} = \mathbf{p}_{l,i,j}^{(a)} / b^{2k/d_a}, \quad k=0,\ldots,d_a/2-1$$

where $b$ is a learnable parameter initialized to 10000. This guarantees that patches representing the same spatial location receive identical positional encodings regardless of resolution, enabling effective cross-scale information flow. Experiments demonstrate that **accurate world coordinates are indispensable**—replacing them with naively centered coordinates causes severe performance degradation.

2. **Multi-Resolution Input Encoding**: Each resolution level $l$ has a dedicated linear projection layer $\text{PE}_l$ and a learnable level embedding $\mathbf{e}_l$:

$$\mathbf{z}_l = \text{PE}_l(\mathbf{X}_l) + \mathbf{e}_l$$

Tokens from all levels are concatenated and processed jointly by a 12-layer Transformer encoder with approximately 25M parameters.

3. **Multi-Resolution MAE Pre-training (MuViT-MAE)**: A high masking ratio of $\rho=0.75$ is applied, with the proportion of visible tokens per level sampled from a Dirichlet distribution $\text{Dir}(\alpha=0.5)$ to encourage cross-scale learning. Each resolution level is paired with a lightweight decoder (2-layer Transformer) that accesses all visible encoded outputs via cross-attention. The loss is the mean MSE over masked patches across all levels.

### Loss & Training

- **Segmentation loss**: Cross-entropy and Dice losses with equal weights of 1.0, computed only at the highest resolution level:

$$\mathcal{L} = \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}(\tilde{y}, y) + \lambda_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}(\tilde{y}, y)$$

- **Segmentation decoder**: Supports both UNETR-style (skip connections with progressive upsampling) and Mask2Former-style (learnable mask queries with cross-attention) decoders.
- **Training sampling**: Nested crops are sampled via random coordinate sampling, ensuring that coarse-resolution crops contain the fine-resolution crops; data is stored in Zarr pyramid format and loaded on demand.

## Key Experimental Results

### Main Results

| Dataset | Method | Input Size | mDSC/DSC | Prev. SOTA | Gain |
|--------|------|----------|----------|----------|------|
| Synthetic | MuViT[1,4]+UNETR | 2×256² | **0.9538** | DeepLabV3: 0.4895 | +0.464 |
| Mouse (11 brain regions) | MuViT[1,8,32]+Mask2Former | 3×256² | **0.901** | DeepLabV3@1024²: 0.843 | +0.058 |
| KPIS (kidney pathology) | MuViT[1,8]+UNETR | 2×512² | **0.8958** | HoloHisto-4K@3840×2160: 0.8454 | +0.050 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| MuViT[1,8,32]+Mask2Former (naive bbox) | mDSC=0.820 (Mouse) | Incorrect coordinates; 0.081 below the correct 0.901 |
| MuViT[1,4]+UNETR (naive bbox) | mDSC=0.386 (Synthetic) | Coordinate error causes performance collapse |
| MuViT[1] (single resolution) | mDSC=0.391 (Mouse) | Lacks global context |
| Linear probe: [1] → [1,8] → [1,8,32,64] | AUC: 0.958 → 0.963 → 0.988 | Richer representations with more resolution levels |
| MAE pre-training acceleration | Epoch 10 reaches mDSC=0.843 | Outperforms all baselines at Epoch 50 |

### Key Findings

- World-coordinate alignment is a prerequisite for effective cross-scale fusion in MuViT; naive coordinates cause severe performance degradation even when the architecture and inputs remain unchanged.
- Multi-resolution MAE pre-training greatly accelerates convergence: performance surpasses all fully-trained single-resolution baselines within only 10 epochs.
- Adding more resolution levels yields monotonic improvements in learned representations (linear probe AUC increases from 0.958 to 0.988).
- The model exhibits robustness to coordinate noise, with negligible performance degradation under shifts of ≤32 pixels.

## Highlights & Insights

- **Elegant yet powerful design**: Effective cross-resolution fusion is achieved solely through world-coordinate RoPE, without introducing complex hierarchical structures or dedicated cross-scale attention modules.
- **Genuine multi-resolution vs. pseudo-multi-scale**: The paper is the first in microscopy imaging to rigorously distinguish between "multi-scale features" derived from a single input and "multi-resolution inputs" truly sampled at different physical resolutions.
- The Dirichlet sampling strategy randomizes masking ratios across levels, promoting complementary cross-scale learning.
- A lightweight architecture (~25M parameters) achieves comprehensive state-of-the-art performance across three distinct tasks.

## Limitations & Future Work

- The full attention mechanism causes computational and memory costs to grow linearly with the number of resolution levels; sparse or cross-scale attention mechanisms could be explored in future work.
- Evaluation is limited to semantic segmentation; downstream tasks such as instance segmentation and object detection remain unexplored.
- The method assumes nested crops (coarse containing fine) and does not consider non-nested or arbitrarily arranged multi-resolution inputs.
- Generalization to 3D volumetric data and non-microscopy domains (e.g., remote sensing) has yet to be validated.

## Related Work & Insights

- MuViT shares conceptual similarity with MultiMAE in treating different scales as "modalities," but enforces geometric consistency through world-coordinate constraints.
- The paper represents a distinctive application of RoPE from NLP to vision, with the unique property that rotation angles are determined by actual spatial coordinates.
- HIPT also addresses hierarchical microscopy images but does not support joint encoding or cross-resolution attention.
- This work directly inspires multi-scale analysis of whole-slide images in digital pathology, where different magnification levels can naturally be treated as distinct resolution levels.

## Rating

- Novelty: ⭐⭐⭐⭐ The application of world-coordinate RoPE in multi-resolution ViTs is concise and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, comprehensive ablations, linear probing, and convergence analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with precise conceptual distinctions.
- Value: ⭐⭐⭐⭐ Broadly applicable to multi-scale processing in microscopy and pathology image analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](../../NeurIPS2025/medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)

</div>

<!-- RELATED:END -->
