---
title: >-
  [Paper Note] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba
description: >-
  [CVPR 2025][Model Compression][Feature Matching] JamMa proposes an ultra-lightweight semi-dense feature matcher based on Joint Mamba. Through the JEGO scan-and-merge strategy, it achieves cross-view joint scanning, efficient four-way scanning, global receptive fields, and omnidirectional feature representation, achieving a superior performance-efficiency tradeoff compared to Transformer-based matchers with less than 50% of the parameters and FLOPs.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Feature Matching"
  - "Mamba"
  - "State Space Models"
  - "Lightweight"
  - "Semi-dense Matching"
date: 2026-05-08
content_hash: de7d64e785d223ab
---

# JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba

**Conference**: CVPR 2025  
**arXiv**: [2503.03437](https://arxiv.org/abs/2503.03437)  
**Code**: [https://leoluxxx.github.io/JamMa-page/](https://leoluxxx.github.io/JamMa-page/)  
**Area**: Model Compression  
**Keywords**: Feature Matching, Mamba, State Space Models, Lightweight, Semi-dense Matching

## TL;DR
JamMa proposes an ultra-lightweight semi-dense feature matcher based on Joint Mamba. Through the JEGO scan-and-merge strategy, it achieves cross-view joint scanning, efficient four-way scanning, global receptive fields, and omnidirectional feature representation, achieving a superior performance-efficiency tradeoff compared to Transformer-based matchers with less than 50% of the parameters and FLOPs.

## Background & Motivation

1. **Background**: Feature matching is fundamental for tasks such as SfM and SLAM. Current state-of-the-art matchers are categorized into sparse methods (e.g., SuperGlue, LightGlue, which rely on keypoint detectors) and semi-dense/dense methods (e.g., LoFTR, ASpanFormer, which establish correspondences between grid points). The latter achieve robust matching in textureless scenes by modeling long-range dependencies using Transformers.

2. **Limitations of Prior Work**: The $O(N^2)$ complexity of Transformers leads to high training costs and large inference delays when processing high-resolution images. Even with linear attention, the number of parameters and computation remains relatively high.

3. **Key Challenge**: The trade-off between long-range dependency modeling capability (crucial for matching) and computational efficiency.

4. **Goal**: To build an ultra-lightweight semi-dense matcher by replacing Transformers with Mamba, which has linear complexity $O(N)$. However, because Mamba is a single-sequence causal model, using it for dual-image feature matching faces three challenges: (1) lack of mutual interaction; (2) unidirectionality; (3) causal nature leading to unbalanced receptive fields.

5. **Key Insight**: Design a novel scanning strategy specifically targeting the three challenges of Mamba—joint scanning for cross-view interaction, skip scanning + four directions to maintain efficiency and omnidirectionality, and a local aggregator to compensate for unbalanced receptive fields.

6. **Core Idea**: Joint Mamba achieves Transformer-like feature matching capabilities under linear complexity through the JEGO strategy (Joint scanning + Efficient skip scanning + Global receptive field + Omnidirectional representation).

## Method

### Overall Architecture
The input is an image pair $(I_A, I_B)$, from which a CNN encoder (ConvNeXt V2 with 0.65M parameters) extracts coarse features $F^c$ (1/8 resolution) and fine features $F^f$ (1/2 resolution). Coarse features are processed through JEGO Scan $\to$ 4 independent Mamba blocks $\to$ JEGO Merge to obtain enhanced cross-view features. Finally, a coarse-to-fine (C2F) matching module generates the final matching results (coarse matching $\to$ fine matching $\to$ sub-pixel refinement).

### Key Designs

1. **Joint Scan**:

    - **Function**: Achieving high-frequency mutual interaction between features of the two images.
    - **Mechanism**: The coarse features of the two images are horizontally concatenated as $X^h = [F_A^c | F_B^c]$ and vertically concatenated as $X^v = [F_A^c; F_B^c]$, and then row/column scanning is performed on the concatenated feature maps. The key lies in the scanning directions, which allow features from both images to alternate within the sequence ("joint"), rather than scanning one entire image before the other ("sequential"). For instance, in horizontal scanning, each row contains features from both A and B, so characteristics of A and B enter the state space of Mamba in high-frequency alternation during the scan.
    - **Design Motivation**: Experiments show that joint scanning outperforms sequential scanning by approximately 2.5 percentage points in pose estimation AUC. Intuitively, feature matching requires close interaction between two images, and joint scanning allows the hidden states of Mamba to carry information from both images simultaneously.

2. **JEGO Four-way Efficient Scan and Merge**:

    - **Function**: Achieving omnidirectional scanning and global receptive fields with a total sequence length of $N$ (instead of $2N$ or $4N$).
    - **Mechanism**: The skip scanning strategy of EVMamba (stride $p=2$) is employed to reduce the sequence length of each direction to $N/4$. Meanwhile, the starting points $(m,n)$ of the four sequences are arranged at different positions to cover four directions (right, left, up, down). After the four sequences are processed by independent Mamba blocks, they are restored to 2D feature maps through JEGO Merge, separating the features of the two images. A gated convolutional aggregator ($3\times3$ Conv) is then used to fuse information from the four directions. Aggregator formula: $\sigma = \text{GELU}(\text{Conv}_3(\tilde{F}^c))$, $\hat{F}^c = \text{Conv}_3(\sigma \cdot \text{Conv}_3(\tilde{F}^c))$.
    - **Design Motivation**: Although the four-directional scanning of VMamba is comprehensive, its total sequence length is $4N$. EVMamba's skip scanning is highly efficient but only performs forward scanning (limiting the receptive field to the bottom-right corner and being non-omnidirectional). The JEGO strategy carefully schedules the starting and ending points of the four directions to make their receptive fields spatially complementary—features with small receptive fields are always adjacent to those with large receptive fields. Thus, a simple $3\times3$ convolutional aggregation allows every feature to obtain global and omnidirectional information.

3. **Coarse-to-Fine Matching Module (C2F)**:

    - **Function**: Generating final sub-pixel level matches from the enhanced features.
    - **Mechanism**: In the coarse matching stage, bidirectional probability matrices $P_{A\to B}$ and $P_{B\to A}$ are computed, and their union is taken to obtain many-to-one matches (which are more robust than the one-to-one matches of Dual-Softmax). In the fine matching stage, an MLP-Mixer is used for interaction within a $5\times5$ window, and Dual-Softmax is applied to establish one-to-one fine matches. In the sub-pixel refinement stage, regression is used to predict offsets.
    - **Design Motivation**: Adopting the matching framework of XoFTR, the bidirectional many-to-one strategy for coarse matching is more robust than traditional Dual-Softmax in textureless areas.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_f + \mathcal{L}_s$. Coarse matching loss $\mathcal{L}_c$ is a bidirectional focal loss; fine matching loss $\mathcal{L}_f$ is a focal loss; sub-pixel loss $\mathcal{L}_s$ is the symmetric epipolar distance. Training is conducted on MegaDepth for 30 epochs with a batch size of 2, using the AdamW optimizer with an initial learning rate of 0.0002 and cosine decay. Training takes approximately 50 hours on a single 4090 GPU.

## Key Experimental Results

### Main Results: Relative Pose Estimation on MegaDepth

| Category | Method | Params(M) | FLOPs(G) | Time(ms) | AUC@5° | AUC@10° | AUC@20° |
|------|------|-----------|----------|----------|--------|---------|---------|
| Sparse | SP+LG | 13.2 | 459.9 | 84.2 | 58.8 | 73.6 | 84.1 |
| Semi-dense | LoFTR | 11.6 | 815.4 | 117.5 | 62.1 | 75.5 | 84.9 |
| Semi-dense | ASpanFormer | 15.8 | 882.3 | 155.7 | 62.6 | 76.1 | 85.7 |
| Semi-dense | ELoFTR | 16.0 | 968.8 | 69.6 | 63.7 | 77.0 | 86.4 |
| **Semi-dense** | **JamMa** | **5.7** | **202.9** | **59.9** | **64.1** | **77.4** | **86.5** |
| Dense | RoMa | 111.3 | 2014.3 | 824.9 | 68.5 | 80.6 | 88.8 |

### Ablation Study

| Configuration | Time(ms) | AUC@5° | AUC@10° | AUC@20° | Description |
|------|----------|--------|---------|---------|------|
| JamMa | 3.2 | 64.5 | 77.3 | 86.3 | Full model |
| Sequential scan instead of Joint scan | 3.2 | 62.2 | 74.7 | 83.7 | AUC@5° drops by 2.3% |
| Without aggregator | 3.0 | 62.3 | 75.1 | 84.3 | AUC@5° drops by 2.2% |
| Scan with EVMamba | 3.0 | 61.9 | 74.8 | 84.1 | No omnidirectionality + no global context |
| Scan with VMamba | 9.7 | 64.1 | 77.1 | 86.2 | Similar performance but 3× slower |
| With linear attention | 24.3 | 64.2 | 77.0 | 86.1 | Similar performance but 7.6× slower |
| No interaction layer | 0 | 60.1 | 73.0 | 82.6 | Baseline |

### Key Findings
- JamMa ranks first overall among semi-dense methods (mean rank of 3.0 vs expected 7.5), showing a clear performance-efficiency balance advantage.
- Joint scanning vs. sequential scanning: Joint scanning improves AUC@5° by 2.3%, indicating that high-frequency mutual interaction is crucial for feature matching.
- Although the aggregator is just a simple $3\times3$ Conv, removing it drops AUC@5° by 2.2%, demonstrating that local information aggregation is highly effective at compensating for unbalanced receptive fields.
- JamMa’s coarse stage takes only 3.2 ms vs 24.3 ms for linear attention (7.6× speedup), with a total inference time of only 59.9 ms.
- With only 5.7M parameters, it is 36% of ASpanFormer and 49% of LoFTR.

## Highlights & Insights
- **Joint Scanning + High-Frequency Mutual Interaction**: Interleaving the features of two images rather than arranging them sequentially allows the hidden states of Mamba to continuously carry cross-view information. This insight is inspiring for any Mamba application that requires dual-sequence interaction modeling (e.g., document comparison, stereo matching).
- **Balanced Receptive Fields + Simple Aggregator = Global Omnidirectionality**: Carefully arranging the starting and ending points of the four directions makes the receptive fields spatially complementary, allowing every feature to obtain global information with just a $3\times3$ Conv. Compared to VMamba (4× sequence length) and Vim (2×), JEGO keeps the total sequence length at $N$.
- **First Success of Mamba in Visual Matching**: Proves that linear-complexity SSMs can replace quadratic-complexity Transformers for feature matching with even better performance. This provides a lightweight path for other computationally intensive vision tasks.

## Limitations & Future Work
- Trained only on the MegaDepth dataset and not fine-tuned on other tasks, which may limit generalizability.
- Dense matchers (such as DKM, RoMa) still lead in absolute accuracy; JamMa’s advantage lies primarily on the efficiency side.
- Although Mamba's causality is mitigated by four-direction scanning and the aggregator, it remains less flexible than true global attention.
- Directions for improvement: (1) Introduce the more efficient parallel computing of Mamba2; (2) Apply the JEGO strategy to more vision Mamba tasks; (3) Explore dynamic skip strides to adapt to different resolutions.

## Related Work & Insights
- **vs. LoFTR/ASpanFormer**: Compared to Transformer-based semi-dense matchers, JEGO Mamba achieves better performance with approximately 1/3 of the parameters and 1/4 of the FLOPs. Its core advantage stems from the linear complexity of Mamba and the efficient scanning of JEGO.
- **vs. ELoFTR**: ELoFTR uses efficient attention to speed up LoFTR; JamMa further replaces attention with Mamba, using fewer parameters (5.7M vs 16.0M) and running faster (59.9 ms vs 69.6 ms).
- **vs. VMamba/EVMamba**: Scanning strategy design in vision Mamba models. VMamba is comprehensive in four directions but has 4× sequence length, while EVMamba is efficient but sacrifices omnidirectionality and global receptive fields. The JEGO strategy combines the advantages of both.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design of the JEGO strategy is ingenious and systematic (joint + efficient + global + omnidirectional), and the Joint Mamba concept is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on pose estimation, homography estimation, and detailed ablation studies, but lacks evaluation on more downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures are intuitive and clear (especially the receptive field visualization), with a logically structured progression.
- Value: ⭐⭐⭐⭐⭐ Initiates a new direction of applying Mamba to visual matching, presenting a highly practical solution that is both ultra-lightweight and high-performing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[CVPR 2025\] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation](parameter_efficient_mamba_tuning_via_projector-targeted_diagonal-centric_linear_.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)

</div>

<!-- RELATED:END -->
