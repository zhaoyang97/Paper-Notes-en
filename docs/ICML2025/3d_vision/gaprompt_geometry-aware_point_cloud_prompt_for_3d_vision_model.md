---
title: >-
  [Paper Note] GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model
description: >-
  [ICML 2025][3D Vision][Parameter-Efficient Fine-Tuning] This paper proposes GAPrompt, a geometry-aware PEFT method for pre-trained 3D vision models. By synergistically leveraging point cloud geometric information through three modules—Point Prompt, Point Shift Prompter, and Prompt Propagation—it matches or even outperforms full fine-tuning while training only 2.19% of the parameters.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "Parameter-Efficient Fine-Tuning"
  - "Point Cloud"
  - "Geometry-Aware"
  - "Prompt Learning"
  - "3D Transformer"
date: 2026-05-08
content_hash: 3eec7275bb6dc2fc
---

# GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model

**Conference**: ICML 2025  
**arXiv**: [2505.04119](https://arxiv.org/abs/2505.04119)  
**Code**: [GitHub](https://github.com/zhoujiahuan1991/ICML2025-GAPrompt)  
**Area**: 3D Vision  
**Keywords**: Parameter-Efficient Fine-Tuning, Point Cloud, Geometry-Aware, Prompt Learning, 3D Transformer

## TL;DR

This paper proposes GAPrompt, a geometry-aware PEFT method for pre-trained 3D vision models. By synergistically leveraging point cloud geometric information through three modules—Point Prompt, Point Shift Prompter, and Prompt Propagation—it matches or even outperforms full fine-tuning while training only 2.19% of the parameters.

## Background & Motivation

**Background**: Pre-trained 3D vision models (such as Point-MAE, Point-BERT, ReCon, Point-FEMAE, etc.) perform strongly on point cloud understanding tasks. However, full fine-tuning is extremely costly and carries the risk of catastrophic forgetting. Parameter-Efficient Fine-Tuning (PEFT) methods have succeeded in NLP and 2D vision (VPT, Adapter Tuning, LoRA, etc.), but their direct transfer to 3D point clouds yields sub-optimal performance.

**Limitations of Prior Work**:
1. 2D PEFT methods rely on randomly initialized token prompts, but the sparsity and irregularity of point cloud data make it difficult for these prompts to align, leading to convergence difficulties.
2. Existing 3D PEFT methods like IDPT (using EdgeConv to dynamically generate prompts) incur high computational overhead. DAPT (dynamic adapter) and Point-PEFT operate primarily at the token feature level, failing to capture the point cloud's inherent geometric information.

**Key Challenge**: The core information of a point cloud lies in its spatial geometric structure (shape, spatial distribution), but existing PEFT methods operate in the encoded token space, losing original geometric signals.

**Goal**: How to explicitly and effectively utilize the geometric information of point clouds in a parameter-efficient fine-tuning framework to bridge the performance gap between PEFT and full fine-tuning.

**Key Insight**: Simultaneously inject geometry-aware signals into the input space (point cloud level) and the feature space (token level).

**Core Idea**: Enhance geometry-awareness throughout the entire pipeline from the input space to the feature space using learnable point clouds and instance-adaptive shape features.

## Method

### Overall Architecture

GAPrompt freezes the pre-trained backbone and introduces three lightweight modules:
1. **Point Prompt**: A learnable point cloud directly concatenated in the input space.
2. **Point Shift Prompter**: Extracts global shape features from the raw point cloud and generates instance-level point shifts.
3. **Prompt Propagation**: Injects shape information into the feature extraction process of the Transformer.

Flow: Raw point cloud $\mathbf{x}$ $\rightarrow$ Point Shift Prompter generates shifted $\tilde{\mathbf{x}}$ and shape features $\mathbf{f}$ $\rightarrow$ $[\tilde{\mathbf{x}}; \mathcal{P}]$ encoded as input tokens $\rightarrow$ shape features enhance prompt tokens $\rightarrow$ Prompt Propagation injection $\rightarrow$ frozen Transformer blocks $\rightarrow$ classification head.

### Key Designs

1. **Point Prompt**:
    - **Function**: Serves as a learnable auxiliary point cloud encoded together with the raw data, guiding the model to focus on fine-grained geometric details.
    - **Mechanism**: Initialize $P$ learnable 3D points $\mathcal{P} \in \mathbb{R}^{P \times 3}$ (uniformly distributed $z \sim U(-r, +r)$), concatenate them with the raw point cloud to form $[\tilde{\mathbf{x}}; \mathcal{P}] \in \mathbb{R}^{(S+P) \times 3}$, and encode them together via token embedding. During training, these points automatically move to regions rich in geometric information.
    - **Design Motivation**: Operating in the point cloud input space naturally preserves 3D structural information compared to token-level prompts.

2. **Point Shift Prompter**:
    - **Function**: Extracts global shape features and generates unique point coordinate shifts for each instance.
    - **Mechanism**:
        - **Hierarchical Downsampling**: Referring to PointNet++, raw point clouds are hierarchically aggregated via multi-resolution FPS + KNN: $\mathbf{x}_{j+1} = \text{FPS}(\mathbf{x}_j)$, $\mathbf{n}_j = \text{KNN}(\mathbf{x}_j, \mathbf{x}_{j+1})$.
        - **Shape Feature Extraction**: A lightweight PointNet encodes features at each level $\tilde{\mathbf{d}}_j = \text{PointNet}(\mathbf{x}_j)$, which are finally reshaped into a global shape vector $\mathbf{f} = \text{Reshape}(\tilde{\mathbf{d}}_k) \in \mathbb{R}^D$.
        - **Point Shift Generation**: Propagates features back to raw points via upsampling, and generates shifts via Shift Head: $\tilde{\mathbf{x}} = \text{Shift-Head}([\tilde{\mathbf{d}}_1^n, \tilde{\mathbf{d}}_1])$.
        - **Feature Enhancement**: $\mathbf{f}$ is used to enhance prompt tokens $\mathbf{p}_i = \mathbf{p}'_i + \mathbf{f} \cdot \beta_p$ and adapters $\mathbf{h}_{i+1} = \hat{\mathbf{h}}_i + \text{Adapter}(\hat{\mathbf{h}}_i + \mathbf{f} \cdot \beta_a)$.
    - **Design Motivation**: Geometric structures vary greatly across different instances. Fixed prompts cannot adapt, requiring instance-adaptive adjustments.

3. **Prompt Propagation**:
    - **Function**: Actively injects information from the enhanced prompt tokens into the intermediate features of the Transformer.
    - **Mechanism**: In each Transformer block, FPS + KNN is applied to the input tokens to find local neighborhoods. Prompt tokens are randomly injected at center/neighbor positions (Prompt Injection), and then diffused to all tokens via PointNet++ style feature propagation: $\tilde{\mathbf{h}}_i = \text{Propagate}(\text{Inject}(\mathbf{h}_i^c, \mathbf{h}_i^n, \mathbf{p}_i))$. The injection uses a Permutation method, introducing dropout-like randomness.
    - **Design Motivation**: Relying solely on attention to passively diffuse prompt information has limited effect. Active propagation ensures geometric information penetrates deeply into each layer's features. This mechanism does not increase parameters.

### Loss & Training

- Standard cross-entropy loss for classification.
- Freeze the backbone and only train the newly added modules + classification head.
- Hyperparameters: $\beta_a = 0.5$, $\beta_p = 0.5$, $P = 20$ (ScanObjectNN), AdamW, lr=5e-4, cosine schedule, 400 epochs, single RTX 4090 GPU.

## Key Experimental Results

### Main Results: ScanObjectNN + ModelNet40 Classification

| Backbone | Method | Params (M) | OBJ_BG | OBJ_ONLY | PB_T50_RS | ModelNet |
|------|------|----------|--------|----------|-----------|---------|
| Point-MAE | Full FT | 22.1 (100%) | 90.02 | 88.29 | 85.18 | 93.2 |
| Point-MAE | +IDPT | 1.7 (7.69%) | 91.22 | 90.02 | 84.94 | 93.3 |
| Point-MAE | +DAPT | 1.1 (4.97%) | 90.88 | 90.19 | 85.08 | 93.5 |
| Point-MAE | +Point-PEFT | 0.7 (3.17%) | 89.33 | 88.98 | 84.42 | 94.2 |
| Point-MAE | **+GAPrompt** | **0.6 (2.71%)** | **91.91** | **90.19** | **85.57** | **94.2** |
| Point-FEMAE | Full FT | 27.4 (100%) | 95.18 | 93.29 | 90.22 | 94.0 |
| Point-FEMAE | +IDPT | 1.7 (6.20%) | 92.94 | 90.88 | 88.38 | 93.4 |
| Point-FEMAE | +DAPT | 1.1 (4.01%) | 93.98 | 92.25 | 88.51 | 93.2 |
| Point-FEMAE | +Point-PEFT | 0.7 (2.55%) | 94.32 | 92.94 | 89.35 | 94.3 |
| Point-FEMAE | **+GAPrompt** | **0.6 (2.19%)** | **95.53** | **93.63** | **90.67** | **94.5** |
| PointGPT-L | Full FT | 360.5 (100%) | 97.20 | 96.60 | 93.40 | 94.1 |
| PointGPT-L | **+GAPrompt** | **2.0 (0.55%)** | **98.97** | **96.73** | **94.31** | **96.2** |

### Comparison with NLP/2D PEFT Methods (PB_T50_RS, Point-MAE)

| Method | Params (M) | Accuracy |
|------|----------|----------|
| Full FT | 22.1 | 85.18 |
| Linear Probing | 0.3 | 75.99 |
| VPT | 0.4 | 81.09 |
| Adapter Tuning | 0.9 | 83.93 |
| LoRA | 0.9 | 81.74 |
| SSF | 0.4 | 82.58 |
| **GAPrompt** | **0.6** | **85.57** |

### Ablation Study (PB_T50_RS, Point-FEMAE)

| Point Prompt | PS-Prompter | Prompt Propagation | Acc. |
|:---:|:---:|:---:|:---:|
| ✓ | - | - | 87.85 |
| ✓ | ✓ | - | 89.34 (+1.49) |
| ✓ | ✓ | ✓ | **90.67** (+1.33) |

| Shift Head | Prompt Enhancement | Adapter Enhancement | Acc. |
|:---:|:---:|:---:|:---:|
| ✓ | - | - | 88.23 |
| ✓ | ✓ | - | 89.71 |
| ✓ | ✓ | ✓ | **90.67** |

### Key Findings

- GAPrompt achieves the best PEFT performance across all four backbones, outperforming full fine-tuning on Point-MAE and PointGPT-L.
- Based on PointGPT-L, it achieves 96.2% on ModelNet40 (with only 0.55% of the parameters), setting a new SOTA.
- Point Shift Prompter contributes the most (+1.49%), serving as the core source of geometric information.
- Visualization: Shifted point cloud boundaries are clearer and more compact; [CLS] attention focuses precisely on critical parts of the object.
- After training, the Point Prompt automatically moves into the inner space of the point cloud.

## Highlights & Insights

1. **Geometry-awareness is the core differentiator for 3D PEFT**: Token-level operations cannot reach the core of point clouds—the spatial geometric structure.
2. **Instance-adaptive design**: Each point cloud has unique shape features and shifts, which is crucial for classification with large shape variations.
3. **Extreme parameter efficiency**: 0.6M parameters (2.19%) outperform 27.4M full fine-tuning, with virtually no increase in FLOPs.
4. **Zero-parameter Prompt Propagation**: Uses spatial distance for feature interpolation, introducing no extra parameters but contributing significantly.

## Limitations & Future Work

- Only evaluated on classification tasks; complex tasks such as 3D detection/segmentation are not covered.
- Point Shift Prompter uses a simple PointNet; stronger encoders (such as KPConv) might further improve performance.
- Hyperparameters ($\beta_a$, $\beta_p$, $P$) need to be tuned separately for different datasets.
- The adaptability to different pre-training strategies has not been analyzed.

## Related Work & Insights

- **VPT (Jia et al., 2022)**: 2D Prompt Tuning. GAPrompt is its 3D geometry-aware counterpart.
- **IDPT (Zha et al., 2023)**: EdgeConv dynamic prompt, which involves high computational overhead (FLOPs 7.2G vs. GAPrompt 5.0G).
- **DAPT (Zhou et al., 2024)**: Dynamic adapter, which still operates in token space.
- Insight: Geometry-aware prompt methods can be extended to 3D detection/segmentation, and the Point Shift concept can also be applied to 3D data augmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing a learnable point cloud in the input space alongside instance-adaptive offsets is a unique 3D PEFT approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbones, comprehensive ablation studies, and comparisons, but lacks detection/segmentation tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and convincing visualizations.
- Value: ⭐⭐⭐⭐ Provides a clear "geometry-aware" paradigm for 3D PEFT, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](../../ICCV2025/3d_vision/upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](../../ICCV2025/3d_vision/constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)

</div>

<!-- RELATED:END -->
