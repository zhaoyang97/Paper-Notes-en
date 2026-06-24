---
title: >-
  [Paper Note] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization
description: >-
  [ECCV 2024][LLM Evaluation][video colorization] This paper proposes ColorMNet, a memory-based deep spatial-temporal feature propagation network. By incorporating three components—Pre-trained Vision-Guided Feature Extraction (PVGFE), Memory-based Feature Propagation (MFP), and Local Attention (LA)—this method achieves video colorization performance superior to state-of-the-art (SOTA) models while significantly reducing GPU memory consumption to only 1.9 GB.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "video colorization"
  - "memory-based feature propagation"
  - "DINOv2"
  - "temporal consistency"
  - "exemplar-based"
date: 2026-05-08
content_hash: ce61560419004149
---

# ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization

**Conference**: ECCV 2024  
**arXiv**: [2404.06251](https://arxiv.org/abs/2404.06251)  
**Code**: [GitHub](https://github.com/yyang181/colormnet)  
**Area**: LLM Evaluation  
**Keywords**: video colorization, memory-based feature propagation, DINOv2, temporal consistency, exemplar-based

## TL;DR
This paper proposes ColorMNet, a memory-based deep spatial-temporal feature propagation network. By incorporating three components—Pre-trained Vision-Guided Feature Extraction (PVGFE), Memory-based Feature Propagation (MFP), and Local Attention (LA)—this method achieves video colorization performance superior to state-of-the-art (SOTA) models while significantly reducing GPU memory consumption to only 1.9 GB.

## Background & Motivation
**Background**: Video colorization requires simultaneously processing the spatial colorization of individual frames and ensuring temporal consistency across frames. Directly applying image colorization methods to video frames leads to severe color flickering between adjacent frames.

**Limitations of Prior Work**:
   - Frame-stacking methods (such as DeepRemaster) stack multiple frames along the temporal dimension, which fails to effectively utilize spatial-temporal priors and incurs a massive GPU memory overhead (16.6 GB).
   - Recurrent propagation methods (such as DeepExemplar) cannot effectively exploit long-range frame information, and errors accumulate over time.
   - Bidirectional recurrent propagation methods (such as BiSTNet) improve the exploration of long-range information but treat the features of each frame equally; thus, inaccurate estimations still accumulate errors, and the memory footprint remains as high as 34.9 GB.

**Key Challenge**: How to establish reliable correspondences between long-range frames under limited GPU memory while mitigating error accumulation arising from inaccurate feature estimations?

**Goal**: This work proposes a memory mechanism that adaptively filters the most important historical frame features based on their usage frequency, thereby exploiting long-range frame information while drastically reducing GPU memory.

**Key Insight**: Inspiration is drawn from the human brain's efficient management of long-term memory—information that is more frequently used receives higher attention.

**Core Idea**: Utilizing DINOv2 to guide spatial feature extraction, employing frequency-driven memory filtering to propagate temporal features, and utilizing local attention to leverage the similarity between adjacent frames.

## Method

### Overall Architecture
Given an input grayscale frame sequence $\{X_i\}_{i=1}^N$ and an exemplar color image $R$, PVGFE extracts spatial features for each frame $\rightarrow$ MFP propagates long-range temporal features $\rightarrow$ LA aggregates adjacent frame features $\rightarrow$ the decoder outputs the chrominance channels $Y_i$ $\rightarrow$ the result is merged with the luminance channel to obtain the colorized frames.

### Key Designs

1. **PVGFE (Pre-trained Vision-Guided Feature Estimation Module)**:

    - **Function**: Fuses the global semantic features from DINOv2 (ViT-S/14) and the local detail features from ResNet50.
    - **Mechanism**: Employs a cross-attention mechanism to fuse the two types of features. DINOv2 provides the query $Q_i^G$, while ResNet50 provides the key $K_i^L$ and value $V_i^L$:
    $\hat{\mathbf{F}}_i = \text{softmax}\left(\frac{\hat{\mathbf{Q}}_i^G (\hat{\mathbf{K}}_i^L)^\top}{\alpha}\right) \hat{\mathbf{V}}_i^L$
    - **Design Motivation**: Using ResNet50 alone cannot model non-local semantic structures. Conversely, while DINOv2 provides global semantics, it lacks the local details required for colorization (as verified by PCA visualization). Simple concatenation fails to adaptively filter effective features. The cross-attention mechanism allows the global semantics of DINOv2 to "query" the local features of ResNet50, naturally achieving complementary integration.

2. **MFP (Memory-based Feature Propagation Module)**:

    - **Function**: Adaptively selects the most reliable features from historical colorized frames and propagates them to the current frame.
    - **Mechanism**:
        - Sampling is conducted every $\gamma$ frames, discarding redundant frames to reduce memory.
        - When the number of historical frames $z$ reaches a threshold $N_s$, the "accumulated similarity" (i.e., usage frequency) $\mathbf{S}_\gamma$ of each pixel feature is computed.
        - A top-$M$ operation is applied to select the $M$ pixel features with the highest usage frequencies from the oldest $N_e$ frames, thereby compressing the memory.
        - The similarity between the current frame query and the memory key is computed using the $L_2$ distance, which is then used to weight and reconstruct color features:
    $\mathbf{V}_i = \mathbf{A}_2^V (\mathbf{W}_i)^\top, \quad \mathbf{W}_i = \text{softmax}(-\mathbf{D}^i)$
    - **Design Motivation**: Features that are frequently matched by subsequent frames are more likely to be accurately estimated. Selecting them suppresses error accumulation. Configured as $\gamma=5, N_e=5, N_s=10, M=128$.

3. **LA (Local Attention Module)**:

    - **Function**: Aggregates features from a $\lambda \times \lambda$ neighborhood around each pixel position $p$ across the previous $d$ frames.
    - **Mechanism**: For each position $p$ in the current frame, attention is computed within the local neighborhood $\mathcal{N}(p)$ across the previous $d$ frames:
    $\mathbf{L}_i^p = \text{softmax}\left(\frac{\mathbf{Q}_i^p (\mathbf{K}^{\mathcal{N}(p),c})^\top}{\beta}\right) \mathbf{V}^{\mathcal{N}(p),c}$
    - **Design Motivation**: While MFP handles long-range frame correspondences, fine-grained motion compensation information within adjacent frames is equally important. LA supplements short-range spatial-temporal details.

### Loss & Training
- **Loss Function**: $L_1$ loss (absolute error between the predicted chrominance channels and the ground truth).
- **Color Space**: CIE LAB.
- **Optimizer**: Adam, learning rate $2 \times 10^{-5}$, batch size 4.
- Trained for 160,000 iterations on a single NVIDIA RTX A6000 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | ColorMNet | BiSTNet (prev SOTA) | Gain |
|--------|------|-----------|---------------------|------|
| DAVIS | PSNR↑ | **35.77** | 34.02 | +1.75 dB |
| DAVIS | FID↓ | **38.39** | 44.69 | -6.30 |
| DAVIS | SSIM↑ | **0.970** | 0.964 | +0.006 |
| DAVIS | LPIPS↓ | **0.035** | 0.043 | -0.008 |
| Videvo | PSNR↑ | **34.35** | 34.12 | +0.23 dB |
| NVCC2023 | PSNR↑ | **33.26** | 33.18 | +0.08 dB |
| NVCC2023 | FID↓ | **20.16** | 25.55 | -5.39 |

### Efficiency Comparison

| Method | Memory (GB) | Runtime (s/frame) | CDC↓ |
|------|---------|-----------------|------|
| DeepExemplar | 19.0 | 0.80 | 0.003876 |
| DeepRemaster | 16.6 | 0.61 | 0.004285 |
| BiSTNet | 34.9 | 1.62 | 0.003870 |
| **ColorMNet** | **1.9** | **0.07** | **0.003763** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| w/ ResNet50 only | 35.01 | 0.962 | Lacks global semantics, leading to color deviations |
| w/ DINOv2 only | 35.38 | 0.963 | Lacks local details |
| w/ Concatenation | 35.26 | 0.965 | Simple concatenation is inferior to cross-attention |
| w/ Stacking | 33.94 | 0.961 | Frame stacking, memory 24.1 GB |
| w/ Recurrent | 35.26 | 0.966 | Recurrent propagation, leading to error accumulation |
| w/o LA | 35.44 | 0.967 | Without local attention |
| **Full ColorMNet** | **35.77** | **0.970** | Full model |

### Key Findings
- PVGFE contributes the most: compared to using only ResNet50, it yields a PSNR improvement of 0.76 dB.
- MFP vs Stacking: achieves a 1.83 dB PSNR improvement, reduces GPU memory by 92% (24.1G $\rightarrow$ 1.9G), and is 14 times faster.
- $\gamma=5$ is identified as the optimal sampling interval; too large a value discards information, while too small a value introduces redundancy.
- Even when the exemplar image contains highly diverse colors, the model yields consistent results, demonstrating its robustness.

## Highlights & Insights
- **Memory Compression Ideology**: Feature filtering using the top-$M$ based on usage frequency elegantly addresses the memory constraint while suppressing error accumulation, serving as an effective "retention-forgetting" mechanism.
- **Novel Use of DINOv2**: This work is the first to introduce large pre-trained vision models to guide feature extraction in video colorization, achieving effective integration through cross-attention rather than simple concatenation.
- **Extreme Efficiency**: Relies on only 1.9G GPU memory (5.4% of BiSTNet) and delivers an inference speed at least 8 times faster than current SOTA, making it highly practical for real-world deployment.
- **Temporal Consistency**: Achieves the best CDC metric, proving that the synergy among the three modules effectively maintains temporal consistency during colorization.

## Limitations & Future Work
- The model size is relatively large (123.61M parameters), posing deployment challenges on edge devices.
- Relying solely on $L_1$ loss without incorporating perceptual or adversarial losses may limit the richness of the generated colors.
- Exemplar selection depends on manual effort (using the first frame's ground truth or web search), lacking an automatic optimal reference image selection mechanism.
- Hyperparameters for memory filtering ($\gamma, N_e, N_s, M$) require manual tuning, with no adaptive mechanism currently available.

## Related Work & Insights
- **vs BiSTNet**: BiSTNet utilizes bidirectional propagation and optical flow alignment, which demands substantial memory and is slow. ColorMNet replaces it with a memory mechanism, improving efficiency by 18 times.
- **vs DeepRemaster**: Frame-stacking cannot effectively model spatial-temporal priors, whereas ColorMNet's combination of MFP and LA is far more precise.
- **vs MAMBA / MeMOTR**: MAMBA randomly selects memory features, and MeMOTR employs exponentially decaying weights; neither performs as well as the frequency-based filtering strategy proposed here.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying a memory mechanism to video colorization is innovative, though individual modules (cross-attention, memory banks) have precedents in other domains.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes comprehensive evaluations on three datasets, detailed ablation studies, efficiency analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and complete mathematical derivations, though some notations are somewhat cumbersome.
- Value: ⭐⭐⭐⭐ Significant efficiency improvements that bear important practical value for video colorization deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)
- [\[ECCV 2024\] Eliminating Warping Shakes for Unsupervised Online Video Stitching](eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](../../ACL2026/llm_evaluation/evaluating_temporal_consistency_in_multi-turn_language_models.md)

</div>

<!-- RELATED:END -->
