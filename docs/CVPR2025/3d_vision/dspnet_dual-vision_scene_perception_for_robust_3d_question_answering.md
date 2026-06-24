---
title: >-
  [Paper Note] DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering
description: >-
  [CVPR 2025][3D Vision][3D Question Answering] DSPNet introduces a dual-vision scene perception network. To overcome limitations regarding fine-grained perception and robust reasoning in 3D QA, it comprehensively integrates point clouds and multi-view images through three joint modules: Text-guided Multi-view Fusion (TGMF), Adaptive Dual-vision Perception (ADVP), and Multimodal Context-guided Reasoning (MCGR), achieving SOTA results on the SQA3D and ScanQA datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Question Answering"
  - "Dual-vision Perception"
  - "Multi-view Fusion"
  - "Point Cloud Feature Enhancement"
  - "Cross-modal Reasoning"
date: 2026-05-08
content_hash: 46d911fdb16f7d3e
---

# DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering

**Conference**: CVPR 2025  
**arXiv**: [2503.03190](https://arxiv.org/abs/2503.03190)  
**Code**: [https://github.com/LZ-CH/DSPNet](https://github.com/LZ-CH/DSPNet)  
**Area**: 3D Vision / Multimodal VLM  
**Keywords**: 3D Question Answering, Dual-vision Perception, Multi-view Fusion, Point Cloud Feature Enhancement, Cross-modal Reasoning

## TL;DR
DSPNet introduces a dual-vision scene perception network. To overcome limitations regarding fine-grained perception and robust reasoning in 3D QA, it comprehensively integrates point clouds and multi-view images through three joint modules: Text-guided Multi-view Fusion (TGMF), Adaptive Dual-vision Perception (ADVP), and Multimodal Context-guided Reasoning (MCGR), achieving SOTA results on the SQA3D and ScanQA datasets.

## Background & Motivation

1. **Background**: 3D Question Answering (3D QA) requires models to understand 3D scenes and answer related questions, which is primarily categorized into 3D Visual QA (ScanQA) and 3D Situated QA (SQA3D). Mainstream methods (e.g., ScanQA, SQA3D, 3DGraphQA) mainly rely on point clouds as the primary source of visual information.
2. **Limitations of Prior Work**: (a) Relying solely on point clouds makes it difficult to depict flat or small objects (e.g., TVs, pictures, carpets, phones), whereas multi-view images can compensate with rich local texture details; (b) Feature degradation occurs when projecting multi-view images back into 3D space—camera pose noise, missing viewpoints, and complex occlusions result in unreliable features in border and occluded regions; (c) During back-projection, the weight of each view is fixed to $\frac{1}{n}$, failing to consider that different questions require different viewpoints.
3. **Key Challenge**: While the texture information of multi-view images is crucial for scene understanding, simple back-projection combined with average fusion fails to utilize this information effectively and instead introduces noise and degraded features.
4. **Goal**: To effectively fuse multi-view images and point cloud information while handling feature degradation during back-projection to achieve robust 3D question answering.
5. **Key Insight**: The authors point out that the original ScanQA work already attempted 3DMV-style back-projection fusion but obtained suboptimal results. There are two root causes: (a) the importance of each viewpoint in multi-view fusion should correspond to the content of the question (text-guided); (b) the degraded features after back-projection should be adaptively filtered rather than accepted entirely.
6. **Core Idea**: Text-guided multi-view fusion ensures that the viewpoints most relevant to the question are prioritized, adaptive dual-vision perception filters out degraded features, and context-guided reasoning achieves efficient cross-modal interaction.

## Method

### Overall Architecture
DSPNet takes 3D point clouds, multi-view images, and question texts as inputs to output answers. The pipeline is as follows: (1) Three encoders separately encode the point clouds (PointNet++), multi-view images (Swin Transformer), and text (Sentence-BERT); (2) The TGMF module back-projects multi-view features into 3D space and performs weighted fusion based on text-guidance; (3) The ADVP module adaptively fuses the back-projected image features and point cloud features into a unified visual representation; (4) The MCGR module performs cross-modal reasoning and answer prediction via multi-layer cross-attention and Transformers.

### Key Designs

1. **Text-guided Multi-view Fusion (TGMF)**:

    - Function: Adaptively weights the contribution of multi-view image features based on the question text content, prioritizing the viewpoints most relevant to the question.
    - Mechanism: First, the multi-view features are back-projected to coordinates of the 3D point clouds using camera parameters, obtaining $U_p \in \mathbb{R}^{N_p \times M \times D_i}$. Then, the attention scores between the global image features $G_i$ and global text features $G_t$ are calculated as $h_s = \frac{Q K^T}{\sqrt{d_k}}$ (where $Q = G_i W_q$, $K = G_t W_k$), deriving the importance weight of each viewpoint as $s = \text{SoftMax}(h)$. Finally, the weighted aggregation yields the fused feature $Z_i = s U_p$.
    - Design Motivation: Simple average pooling treats all viewpoints equally, ignoring the fact that "different questions require different viewpoint information." For instance, for the question "which side of the picture is the TV," the frontal view is much more useful than side views. Text-guided attention enables the model to learn to select the most relevant viewpoints for different questions.

2. **Adaptive Dual-vision Perception (ADVP)**:

    - Function: Adaptively fuses back-projected image features and point cloud features, filtering out degraded features while enhancing high-confidence features.
    - Mechanism: Inspired by SENet, back-projected features $Z_i$ and point cloud features $Z_p$ are concatenated and passed through MLP + Sigmoid to learn point-wise and channel-wise importance weights: $Z_h = \sigma(\text{MLP}([Z_i, Z_p])) \odot [Z_i, Z_p]$. Features are then mapped to a unified dimension via an FC layer to obtain the final visual feature $Z_v$.
    - Design Motivation: Within the back-projection process, features near FOV boundaries and occluded regions are unreliable due to pose noise and missing viewpoints, and the point cloud features themselves are already sufficient in certain areas without needing image complements. ADVP adaptively suppresses low-quality features and enhances high-quality features through a learned gating mechanism, preventing degraded features from interfering with subsequent reasoning.

3. **Multimodal Context-guided Reasoning (MCGR)**:

    - Function: Achieves fine-grained visual-linguistic interactive reasoning while maintaining computational efficiency.
    - Mechanism: FPS is used to sample $K$ sparse candidate features $Z_c$ from dense visual features $Z_v$, which are combined with positional encodings to form dense embeddings $E_v$ and sparse embeddings $E_c$. Every layer of the $L$-layer reasoning process contains: (a) a cross-attention sub-layer: extracting keypoint features with $E_c^{i-1}$ as the query and $E_v$ as the context, $h_c^i = \text{CrossAtt}(E_c^{i-1}, E_v)$; (b) a Transformer sub-layer: concatenating $h_c^i$ with text features $E_t^{i-1}$ for self-attention interaction $[E_c^i, E_t^i] = \text{Transformer}([h_c^i, E_t^{i-1}])$.
    - Design Motivation: Directly computing cross-modal attention on dense visual features is computationally expensive and introduces feature redundancy, while direct downsampling risks losing spatial information. MCGR utilizes a two-stage interaction of sparse candidates + dense context to significantly reduce the computational cost while maintaining spatial precision and semantic specialty.

### Loss & Training
- 3D VQA Task: $L_{3DVQA} = L_{ans} + \lambda_1 L_{cls} + \lambda_2 L_{loc}$ (answer classification + object classification + reference object localization)
- 3D SQA Task: $L_{3DSQA} = L_{ans}$ (answer classification only)
- Employs soft-ranked cross-entropy loss to handle noisy labels
- Samples 20 multi-view images (224×224) and 40,000 points
- 3D encoder utilizes PointNet++ pre-trained with VoteNet (without VoteHead)
- Image encoder uses a frozen Swin Transformer
- AdamW optimizer, 12 epochs, 4 GPUs, batch size of 48

## Key Experimental Results

### Main Results

| Dataset | Metric | DSPNet | 3DGraphQA | 3D-VisTA (pretrained) | SQA3D |
|--------|------|--------|-----------|----------------------|-------|
| SQA3D | Avg Acc | **50.4** | 49.2 | 48.5 | 47.2 |
| SQA3D | What | **38.2** | 36.4 | 34.8 | 33.5 |
| SQA3D | How | **51.2** | 46.1 | 45.4 | 42.4 |
| ScanQA | EM@1 (w/ obj) | **26.5** | 25.6 | 27.0* | 23.5 |
| ScanQA | CIDEr (w/o obj) | **69.6** | 62.9 | 62.6* | 60.2 |

*Note: 3D-VisTA uses external 3D-Text datasets for pre-training, whereas DSPNet does not use pre-training.

### Ablation Study

| Configuration | ScanQA EM@1 | SQA3D Acc | Description |
|------|-------------|-----------|------|
| Baseline | 22.35 | 49.33 | Without TGMF/ADVP/MCGR |
| + TGMF | 22.69 | 49.58 | Text-guided fusion +0.34 / +0.25 |
| + TGMF + ADVP | 22.80 | 49.87 | Adaptive dual-vision +0.11 / +0.29 |
| + TGMF + MCGR | 23.23 | 49.77 | Contextual reasoning +0.54 / +0.19 |
| Full Model | **23.47** | **50.36** | Three-module synergy |
| w/o 2D modality | 22.26 | 49.05 | Removed image information, notable performance drop |

### Key Findings
- **MCGR contributes the most**: On ScanQA, MCGR individually brings a +0.54 improvement (compared to +0.34 for TGMF and +0.11 for ADVP), showing that the cross-modal reasoning mechanism is most critical for 3D QA.
- **The three-module synergy exceeds simple addition**: The Full Model (+1.12) gains more than the sum of the individual contributions of the three modules, demonstrating positive synergistic effects—TGMF and ADVP provide better features, which MCGR utilizes more effectively.
- **Obvious advantage on open-ended questions**: DSPNet shows the largest improvements on questions like "What" (+1.8) and "How" (+5.1) that require deep scene understanding, while showing smaller advantages on simpler questions like "Is/Can/Which" (which might be guessed from the question alone).
- **2D visual information is indispensable**: Removing multi-view images leads to a 1.21 drop in ScanQA and a 1.31 drop in SQA3D, confirming the vital role of multi-view images in 3D QA.
- **SOTA-level performance without pre-training**: Without pre-training, DSPNet is close to the large-scale pre-trained 3D-VisTA on ScanQA EM@1 (26.5 vs 27.0) and outperforms all and any methods on SQA3D.

## Highlights & Insights
- **Question-driven multi-view fusion**: Using text/questions to guide the weighted fusion of multi-view features is a natural and effective design—different questions certainly require information from different viewpoints. This query-aware fusion approach can be generalized to any text + multi-view task.
- **Explicit modeling of back-projection degradation**: Prior methods simply concatenated back-projected features and point cloud features, ignoring the degradation problem. ADVP adaptively filters degraded features using a SENet-style gating mechanism—this concept of modeling data quality can be transferred to other multimodal fusion scenarios.
- **Sparse-to-dense two-stage reasoning**: MCGR uses FPS-sampled sparse candidates to extract key information from dense features, bypassing the computational bottleneck of performing full attention on high-dimensional features.

## Limitations & Future Work
- **Fixed 20-view sampling strategy**: Currently, uniformly sampling 20 views may not be optimal. A smarter view-selection strategy might be needed for different scenes.
- **Limitations of PointNet++ encoder**: Using the older PointNet++ as the 3D encoder without exploring more advanced point cloud backbones (like PointTransformer v3).
- **Underutilized Large Language Models**: Currently using MCAN as the answer prediction head, the potential for integration with LLMs remains unexplored, which may limit open-ended QA capabilities.
- **Limited scene scale**: Only validated on ScanNet indoor scenes; generalization to larger-scale or outdoor scenes remains unknown.
- **Sensitivity to multi-view count**: Ablating shows that performance scales up with 10/15/20 views, but the balance between accuracy and computational overhead for a higher number of views has not been thoroughly analyzed.

## Related Work & Insights
- **vs ScanQA**: ScanQA attempted 3DMV-style back-projection fusion, but with limited success. DSPNet resolves the two core limitations of basic fusion using TGMF (text-guided weights) and ADVP (degradation filtering).
- **vs 3D-VisTA**: 3D-VisTA learns generic 3D-language representations via large-scale pre-training. DSPNet achieves comparable performance without pre-training through better feature fusion and reasoning—suggesting that architectural design and pre-training both have distinct merits.
- **vs 3DGraphQA**: 3DGraphQA models relationships between objects using graph Transformers, also incorporating multi-view information. DSPNet utilizes point-level (rather than object-level) visual representations, preserving richer spatial information.
- **Comparison with 3D detection fusion methods**: ADVP is inspired by SENet and 2D-3D fusion in 3D detection, but innovatively incorporates explicit modeling of degraded features.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined scheme of text-guided multi-view fusion and back-projection degradation filtering is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two standard datasets + comprehensive ablations + view count analysis + qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation (Figs 1 and 2 intuitively demonstrate existing limitations) and comprehensive module descriptions.
- Value: ⭐⭐⭐⭐ Provides an effective paradigm for utilizing multi-view information in 3D QA, and the modules are highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)
- [\[ICML 2026\] Zero-Shot 3D Question Answering via Hierarchical View-to-Token Transportation](../../ICML2026/3d_vision/zero-shot_3d_question_answering_via_hierarchical_view-to-token_transportation.md)
- [\[CVPR 2025\] Continuous 3D Perception Model with Persistent State](continuous_3d_perception_model_with_persistent_state.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_extended_dr_3d.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_point_maps_shape_pose.md)

</div>

<!-- RELATED:END -->
