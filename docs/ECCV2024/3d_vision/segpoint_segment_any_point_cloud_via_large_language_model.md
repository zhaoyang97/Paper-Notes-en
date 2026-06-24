---
title: >-
  [Paper Note] SegPoint: Segment Any Point Cloud via Large Language Model
description: >-
  [ECCV 2024][3D Vision][3D point cloud segmentation] SegPoint is proposed, the first model to utilize multimodal LLM reasoning capabilities in a unified framework to complete four tasks: 3D instruction segmentation, referring segmentation, semantic segmentation, and open-vocabulary segmentation. Additionally, the Instruct3D benchmark (2,565 pairs) is constructed, achieving an mIoU of 27.5%.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D point cloud segmentation"
  - "LLM"
  - "unified framework"
  - "instruction segmentation"
  - "geometric feature"
date: 2026-05-08
content_hash: c0178bb26c2f4c8c
---

# SegPoint: Segment Any Point Cloud via Large Language Model

**Conference**: ECCV 2024  
**arXiv**: [2407.13761](https://arxiv.org/abs/2407.13761)  
**Code**: [https://heshuting555.github.io/SegPoint](https://heshuting555.github.io/SegPoint)  
**Area**: 3D Vision / Point Cloud Segmentation  
**Keywords**: 3D point cloud segmentation, LLM, unified framework, instruction segmentation, geometric feature

## TL;DR
SegPoint is proposed, the first model to utilize multimodal LLM reasoning capabilities in a unified framework to complete four tasks: 3D instruction segmentation, referring segmentation, semantic segmentation, and open-vocabulary segmentation. Additionally, the Instruct3D benchmark (2,565 pairs) is constructed, achieving an mIoU of 27.5%.

## Background & Motivation
**Background**: Significant progress has been made in 3D point cloud segmentation (e.g., Mask3D, SPFormer), but each model typically addresses only one specific segmentation task, lacking unification.

**Limitations of Prior Work**: (a) Existing methods rely on predefined categories or explicit textual descriptions, failing to understand implicit human intent (e.g., "Where can I sit?"); (b) different tasks require different models, which is inefficient and impractical.

**Key Challenge**: Real-world scenarios require models to understand implicit instructions and perform reasoning, yet current methods lack reasoning capability; meanwhile, a unified framework is needed to handle multiple segmentation tasks.

**Key Insight**: Leveraging the reasoning and world knowledge of LLMs to understand complex/implicit instructions, combined with a geometric enhancer module to compensate for the limitations of point cloud encoders in dense prediction.

**Core Idea**: Injecting LLM reasoning capabilities into 3D point cloud segmentation, achieving high-quality point-wise segmentation through geometric enhancement and feature propagation.

## Method

### Overall Architecture
SegPoint consists of four core components: (1) a pretrained point cloud encoder $\mathcal{E}$ (Uni3D) to extract point cloud features; (2) a large language model $\mathcal{F}$ (LLaMA2-7B) to provide reasoning capabilities; (3) a Geometric Enhancer Module (GEM) $\mathcal{G}$ to extract local geometric information and inject it into the encoder; and (4) a Geometric-guided Feature Propagation (GFP) block $\mathcal{P}$ to generate high-quality point-wise embeddings for precise segmentation.

The inputs are the point cloud $\vec{i}_{point} \in \mathbb{R}^{N \times (3+F)}$ and the text instruction $\vec{i}_{txt}$. The embedding corresponding to the `<SEG>` token in the LLM output is dot-producted with the point-wise features to generate the segmentation mask.

### Key Designs

#### 1. **Vanilla Baseline and Its Limitations**
    - Function: Directly inputs point cloud encoder features into the LLM, detects the `<SEG>` token to generate the mask embedding $\vec{h}_{seg} = \gamma(\vec{y}_{[seg]})$, and performs a dot product with the upsampled point-wise embeddings to obtain the mask $\vec{m} = \vec{h}_{seg} \otimes \text{UpS.}(\vec{f}_{point})$.
    - Limitations: (a) The point cloud encoder is trained for scene-level classification and is unsuitable for dense prediction; (b) FPS sampling reduces the points from $N$ to $N_1$, causing detail loss; (c) directly upsampling from $N_1$ to $N$ introduces significant noise.
    - Design Motivation: Identifies two core bottlenecks: the lack of local geometric information and poor upsampling quality.

#### 2. **Geometric Enhancer Module (GEM)**
    - Function: Extracts the local geometric context of the entire scene and injects it into the intermediate features of the point cloud encoder via cross-attention.
    - Mechanism:
      - GEM consists of 3 KPConv + BN + ReLU blocks, outputting geometric features $\vec{g}_f \in \mathbb{R}^{N \times D}$, which preserves the information of all $N$ points.
      - The geometric features are injected into each block of the encoder via cross-attention: $\hat{\vec{f}_i} = \vec{f}_i + g_i \cdot \text{softmax}\left(\frac{\vec{f}_i \vec{g}_f^T}{\sqrt{D}}\right) \vec{g}_f$
      - A learnable gating factor $g_i$ is initialized to zero, ensuring that the feature distribution of the pretrained weights is not abruptly altered.
    - Design Motivation: KPConv is naturally suited for extracting local 3D geometric information (vs. ordinary linear layers); the gating factor protects pretrained weights; this is similar to the concept of using ConvStem to enhance ViT's ability to capture local information in 2D.

#### 3. **Geometric-guided Feature Propagation (GFP)**
    - Function: Performs high-quality upsampling from sparse point features to dense point-wise embeddings.
    - Mechanism:
      - High-level features $\vec{f}_3, \vec{f}_4$ are upsampled to $N_3, N_2$ points via PointNet++ propagation.
      - Geometric features $\vec{g}_f$ are downsampled to the same number of points via FPS.
      - The upsampled and downsampled features are concatenated and fused through FC + ReLU layers.
      - The last-layer feature $\vec{f}_5$ is concatenated with the hidden state embedding output by the LLM to perceive multimodal info.
      - **Attentive Propagation**: Cross-attention is employed to exchange information across different point densities: $\hat{\tilde{\vec{f}}}_4 = \tilde{\vec{f}}_4 + \text{softmax}\left(\frac{\tilde{\vec{f}}_4 \vec{f}_{54}^T}{\sqrt{D}}\right)\vec{f}_{54}$
    - Design Motivation: To avoid the information loss caused by direct upsampling; geometric features act as "golden reference" to guide the upsampling process.

#### 4. **Task Unification and the Instruct3D Dataset**
    - Function: Handles four segmentation tasks within a unified model using task-specific prompts.
    - Semantic segmentation template: "Can you segment the {category} in this point cloud?" $\rightarrow$ "{category} \<SEG\>"
    - Referring segmentation template: "Can you segment the object {description}?" $\rightarrow$ "{category} \<SEG\>"
    - Instruct3D contains 2,565 instruction-point cloud pairs from 280 scenes in ScanNet++, supporting multi-target and zero-target scenarios.
    - Design Motivation: Implicit instructions require reasoning capabilities (e.g., "Where can I sit?" $\rightarrow$ segmenting a chair), which is unsupported by existing datasets.

### Loss & Training
Total Loss: $\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice}$

- $\mathcal{L}_{txt}$: Autoregressive cross-entropy loss (text generation)
- $\mathcal{L}_{bce}$: Binary cross-entropy loss (segmentation mask)
- $\mathcal{L}_{dice}$: DICE loss (segmentation mask)
- Weights: $\lambda_{txt}=1.0, \lambda_{bce}=2.0, \lambda_{dice}=2.0$
- All datasets are jointly trained, with fine-tuning on specific datasets during evaluation.

## Key Experimental Results

### Instruction Segmentation (Instruct3D)

| Stage | Method | Acc | mIoU |
|------|------|-----|------|
| Two-stage | ScanRefer | 12.0 | 6.9 |
| Two-stage | M3DRef-CLIP | 18.1 | 12.8 |
| Single | BUTD-DETR* | 16.3 | 10.9 |
| Single | EDA* | 16.6 | 12.1 |
| Single | SegPoint† (vanilla) | 21.8 | 16.1 |
| Single | **SegPoint** | **31.6** | **27.5** |

SegPoint achieves a **+14.7 mIoU** gain compared to the best baseline (27.5 vs 12.8).

### Semantic Segmentation

| Method | ScanNet | ScanNet200 | S3DIS |
|------|---------|-----------|-------|
| PTv2 | 75.4 | 30.2 | 71.6 |
| OctFormer | 75.7 | 32.6 | - |
| Swin3D | 75.5 | - | 72.5 |
| **SegPoint** | 74.1 | **35.3** | 72.4 |

Beats the SOTA by +2.7% mIoU on the category-rich ScanNet200.

### Referring Segmentation

| Method | ScanRefer | Nr3D | Multi3DRefer |
|------|-----------|------|-------------|
| M3DRef-CLIP | 35.7 | 27.0 | 32.6 |
| 3D-STMN | 39.5 | - | - |
| RefMask3D | 44.8 | - | - |
| **SegPoint** | 41.7 | **32.2** | **36.1** |

Outperforms competitors by +3.5 mIoU on Multi3DRefer.

### Ablation Study

| GEM | GFP | Instruct3D mIoU | ScanRefer mIoU |
|-----|-----|-----------------|----------------|
| ✗ | ✗ | 16.1 | 30.3 |
| ✓ | ✗ | 21.4 | 35.8 |
| ✗ | ✓ | 23.2 | 38.1 |
| ✓ | ✓ | **27.5** | **41.7** |

### Key Findings
- GEM and GFP each make significant independent contributions (+5.3/+7.1 mIoU), and their combination provides even stronger complementary effects.
- GEM outperforms full fine-tuning, LoRA, and MLP adapters, showing that the improvement does not merely stem from increased parameter size.
- Even the vanilla baseline (SegPoint†) outperforms all existing methods $\rightarrow$ validating the effectiveness of LLMs as segmentation engines.
- In open-vocabulary segmentation, it outperforms several supervised methods (ScanNet++ 19.3 vs. supervised KPConv 30.0), demonstrating strong generalization ability.

## Highlights & Insights
- **Elegance of a Unified Framework**: For the first time, four 3D segmentation tasks are unified within a single model. Switching between tasks is achieved via task-specific prompts, eliminating the need to design separate models for each task. This approach aligns with the trend of unified segmentation models in the 2D domain (e.g., SAM).
- **Necessity of Geometric Enhancement**: Pretrained point cloud encoders (e.g., Uni3D) are designed for scene-level classification, leading to poor performance when directly applied to dense prediction. GEM, by extracting local geometry via KPConv paired with gated injection, adapts to dense prediction at minimal cost.
- **Instruct3D Pioneering a New Task**: Extending 3D segmentation from explicit descriptions to implicit instruction understanding requires world knowledge and reasoning capabilities, driving the development of more intelligent 3D perception systems.

## Limitations & Future Work
- The current framework only supports text prompts and does not support non-textual prompts like points or boxes (similar to SAM's prompt mode).
- Training requires 4×A100 GPUs for approximately 3 days, indicating a high computational cost.
- The Instruct3D dataset is relatively small (only 2,565 pairs) and needs expansion in the future.
- Open-vocabulary segmentation relies on GPT-4 for category name matching, introducing an extra dependency.

## Related Work & Insights
- **vs. LISA (2D)**: SegPoint borrows the paradigm of injecting a `<SEG>` token into the LLM from LISA, but performs end-to-end mask generation without relying on external pretrained segmentation models like SAM.
- **vs. Mask3D/SPFormer**: While these Transformer-based methods excel at standard segmentation, they cannot handle language interaction and implicit instructions.
- **vs. 3D-LLM/PointLLM**: These methods focus primarily on scene-level understanding, lacking the capability for fine-grained point-wise segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to unify four 3D segmentation tasks + new Instruct3D task
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks across four tasks + detailed ablation
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, well-motivated module designs
- Value: ⭐⭐⭐⭐ Driving the transition of 3D segmentation toward a unified, intelligent framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PointLLM: Empowering Large Language Models to Understand Point Clouds](pointllm_empowering_large_language_models_to_understand_point_clouds.md)
- [\[CVPR 2026\] AnyPcc: Compressing Any Point Cloud with a Single Universal Model](../../CVPR2026/3d_vision/anypcc_compressing_any_point_cloud_with_a_single_universal_model.md)
- [\[ECCV 2024\] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark](signavatars_a_large-scale_3d_sign_language_holistic_motion_dataset_and_benchmark.md)

</div>

<!-- RELATED:END -->
