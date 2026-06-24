---
title: >-
  [Paper Note] Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion
description: >-
  [ECCV 2024][3D Vision][Point Cloud Completion] This work proposes the EGIInet framework, which achieves modal alignment using a unified encoder and utilizes an explicitly guided information interaction strategy (FT-Loss) to enable the network to accurately identify key structural information in images. On view-guided point cloud completion tasks, it outperforms XMFnet by 16% CD with fewer parameters.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Cross-modal Fusion"
  - "View Guidance"
  - "Multi-modal Alignment"
  - "Information Interaction"
date: 2026-05-08
content_hash: a7ccb2bf43d1a24a
---

# Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion

**Conference**: ECCV 2024  
**arXiv**: [2407.02887](https://arxiv.org/abs/2407.02887)  
**Code**: [https://github.com/WHU-USI3DV/EGIInet](https://github.com/WHU-USI3DV/EGIInet)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Cross-modal Fusion, View Guidance, Multi-modal Alignment, Information Interaction

## TL;DR

This work proposes the EGIInet framework, which achieves modal alignment using a unified encoder and utilizes an explicitly guided information interaction strategy (FT-Loss) to enable the network to accurately identify key structural information in images. On view-guided point cloud completion tasks, it outperforms XMFnet by 16% CD with fewer parameters.

## Background & Motivation

**Background**: Point cloud completion is a fundamental task in 3D vision. Due to inherent limitations of scanning sensors, raw point clouds are often sparse, noisy, and occluded. View-guided Point Cloud Completion (ViPC), which introduces a single-view image to assist in completion, is a more practical solution.

**Limitations of Prior Work**: Existing multi-modal fusion methods (ViPC, CSDN) rely on decision-level fusion or directly estimate 3D coordinates from images (an ill-posed problem). Although XMFnet utilizes cross-attention for latent space fusion, it ignores the inherent differences between modalities and lacks explicit guidance for the information fusion process.

**Key Challenge**: XMFnet tends to extract abstract global semantic features from images while neglecting the geometric structural information essentially required for the point cloud completion task, leading to suboptimal completion results.

**Goal**: How to identify the structural information in the image that is most critical for point cloud completion and efficiently integrate it into the completion process.

**Key Insight**: The completion process is decoupled into two stages: "modal alignment" and "information fusion," which are handled by a unified encoder and an explicitly guided feature transfer loss, respectively.

**Core Idea**: Indirect modal information interaction is explicitly guided by a feature transfer loss supervised by Gram matrices, allowing the network to automatically identify the structural information in the image that is most valuable for completion.

## Method

### Overall Architecture

The pipeline of EGIInet consists of three stages: (1) **Tokenizer** converts both images and point clouds into unified token sequences; (2) **Shared Feature Extractor (SFE)** utilizes shared ViT blocks for modal alignment encoding; (3) **Shared Feature Transfer Network (SFTnet)** performs information interaction under the explicit guidance of FT-Loss. Finally, feature fusion is conducted via a simple cross-attention layer, and the decoder outputs the complete point cloud.

### Key Designs

1. **Unified Encoder**:

    - **Function**: Maps inputs of different modalities into an adjacent latent space, reducing the modality gap.
    - **Mechanism**: It contains two parts: Tokenizer and SFE. The image is split into tokens using a large-kernel convolution, while the point cloud tokens are generated via multi-step FPS downsampling + Ball-query clustering. The token sequences of both modalities $\boldsymbol{F}_{pc}, \boldsymbol{F}_{img} \in \mathbb{R}^{N' \times C'}$ share the same self-attention-based ViT block for feature extraction: $\boldsymbol{F}_{pc}^{stc} = \text{SFE}(\boldsymbol{F}_{pc})$, $\boldsymbol{F}_{img}^{stc} = \text{SFE}(\boldsymbol{F}_{img})$.
    - **Design Motivation**: Different 2D/3D backbones exhibit discrepancies in their latent space distributions and semantic structures. Utilizing a unified, shared architecture can map features of different modalities into an adjacent latent space, simplifying subsequent information interaction. Positional embeddings are additionally incorporated into the point cloud tokens to compensate for the irregularity of point clouds.

2. **Shared Feature Transfer Network (SFTnet)**:

    - **Function**: Provides an information interaction process independent of the encoding stage, enabling indirect interaction between point cloud features and image features while preserving their respective information organization patterns.
    - **Mechanism**: SFTnet is also built upon a shared ViT block structure and processes features of the two modalities separately: $\boldsymbol{F}_{pc}' = \text{SFTnet}(\boldsymbol{F}_{pc}^{stc})$, $\boldsymbol{F}_{img}' = \text{SFTnet}(\boldsymbol{F}_{img}^{stc})$. Crucially, features from the two modalities do not interact directly; instead, they undergo explicitly guided information interaction at the loss level via FT-Loss.
    - **Design Motivation**: Decoupling information interaction from the encoding process provides specific learning objectives for the network at different stages, reducing the overall optimization difficulty. Direct fusion changes the feature organization pattern and increases the learning burden.

3. **Feature Transfer Loss (FT-Loss)**:

    - **Function**: Explicitly guides the transfer of structural info between image and point cloud features through Gram matrix supervision.
    - **Mechanism**: FT-Loss consists of information loss $\mathcal{L}_{infor}$ and structural loss $\mathcal{L}_{stc}$. The information loss enables cross-modal structural information transfer by aligning Gram matrices:
    $\mathcal{L}_{infor} = \frac{(\boldsymbol{G}(\boldsymbol{F}_{img}^{stc}) - \boldsymbol{G}(\boldsymbol{F}_{pc}'))^2 + (\boldsymbol{G}(\boldsymbol{F}_{pc}^{stc}) - \boldsymbol{G}(\boldsymbol{F}_{img}'))^2}{N \times C}$
      where the Gram matrix $\boldsymbol{G}(\boldsymbol{F}) = \boldsymbol{F}^T \cdot \boldsymbol{F}$. The structural loss prevents the information structure of point cloud features from being disrupted: $\mathcal{L}_{stc} = (\boldsymbol{F}_{pc}^{stc} - \boldsymbol{F}_{pc}')^2$.
    - **Design Motivation**: The Gram matrix can describe the channel-wise global structural criticality of features. Through bidirectional Gram matrix alignment, missing relationships in point cloud features can be transferred to image features, while structural information corresponding to missing parts in the image can be transferred to point cloud features. The structural loss ensures that the information structure of the 3D point cloud features is preserved during the transfer process (since 2D features struggle to directly predict 3D coordinates).

### Loss & Training

Total loss function: $\mathcal{L}_{total} = \alpha \times \mathcal{L}_{transfer} + \mathcal{L}_{l_1\text{-}CD}$, where $\alpha = 0.01$ (as the magnitude of transfer loss is much larger than the CD loss). $\mathcal{L}_{transfer} = \mathcal{L}_{infor} + \mathcal{L}_{stc}$. $\mathcal{L}_{l_1\text{-}CD}$ is the standard L1 Chamfer Distance.

## Key Experimental Results

### Main Results

Comparison with existing methods on the ShapeNet-ViPC dataset:

| Method | Avg CD×10³↓ | Airplane | Lamp | F-Score↑ (Avg) |
|------|------------|----------|------|----------------|
| ViPC | 3.308 | 1.760 | 2.867 | 0.591 |
| CSDN | 2.570 | 1.251 | 2.554 | 0.695 |
| XMFnet | 1.443 | 0.572 | 1.810 | 0.796 |
| **EGIInet (Ours)** | **1.211** | **0.534** | **0.776** | **0.836** |

Compared with XMFnet, the average CD metric drops by 16%, the F-Score improves by 5%, and the parameter count is smaller (9.03M < 9.57M). The improvement on the Lamp category is particularly significant (from 1.810 to 0.776, a 57% reduction).

### Ablation Study

| Configuration | Avg CD×10³↓ | Description |
|------|------------|------|
| Full model | 1.211 | Full model |
| w/o sharing | 1.429 | Remove shared structure, CD increases by 18% |
| w/o FT-Loss | 1.354 | Remove feature transfer loss, CD increases by 12% |
| w/o SFTnet | 1.454 | Remove SFTnet (using only simplified loss), CD increases by 20% |
| w/o image | 1.383 | Remove image input, CD increases by 14% |

### Key Findings

- The independent information interaction process of SFTnet is the module contributing most to performance gains (CD increases by 20% when removed), verifying the necessity of separating information interaction from encoding.
- The shared structure is crucial for modal alignment; removing it significantly degrades performance despite doubling the parameters.
- Visualization of FT-Loss demonstrates that the image features accurately focus on structural regions useful for completion under supervision, while they degrade to capturing abstract global features when unsupervised.
- Even without image input (w/o image), relying solely on point clouds is still competitive compared to XMFnet, indicating that the unified encoder and SFTnet themselves also benefit point cloud feature learning.

## Highlights & Insights

- **Applying the Gram matrix for structural info transfer** is the most ingenious design: it reformulates the question of "which image regions are critical for completion" into a differentiable loss constraint, achieving human-controllable guidance for information interaction.
- **Indirect interaction outperforms direct fusion**: Implementing information interaction via a shared network combined with loss supervision is more effective than direct cross-attention fusion, as it preserves the unique feature organization patterns of each modality.
- Attention weight map visualizations intuitively demonstrate the difference between this method and XMFnet: EGIInet precisely localizes structural edges in the image, whereas XMFnet can only capture blurry global semantics.

## Limitations & Future Work

- Under limited parameter constraints, the shared structure underperforms compared to non-shared models on certain complex categories (e.g., categories with few valid pixels), implying that the shared structure may suffer from optimization conflicts across different categories.
- Evaluated only on ShapeNet-ViPC; the generalization ability on real-world datasets remains to be validated.
- $\alpha$ in FT-Loss is fixed at 0.01, and its sensitivity was not thoroughly discussed.
- Only a single view is used, without exploring the possibility of multi-view fusion.

## Related Work & Insights

- **vs XMFnet**: XMFnet directly stacks cross-attention to achieve feature fusion without explicit guidance, which makes the network tend to learn global semantics rather than local structures. EGIInet explicitly guides this process through FT-Loss, focusing the network on critical geometric structural details.
- **vs CSDN**: CSDN uses IPAdaIN to let image features influence the point cloud deformation process, which is an implicit fusion method. The explicit guidance strategy of EGIInet is more transparent and yields better performance.
- **vs ViPC**: ViPC directly converts images to skeleton point clouds and then concatenates them, which is essentially decision-level fusion rather than feature-level fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using Gram matrices to guide cross-modal information interaction is highly creative, though the overall framework still follows the encoder-decoder paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation studies cover all key components, and the visualizations are intuitive and persuasive, but the evaluation is limited to only one dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed method descriptions, and well-illustrated diagrams alongside textual explanations.
- Value: ⭐⭐⭐⭐ The concept of "explicitly guided information interaction" can be generalized to other cross-modal fusion tasks, and the Gram matrix loss design is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](../../AAAI2026/3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[ECCV 2024\] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)
- [\[ECCV 2024\] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)

</div>

<!-- RELATED:END -->
