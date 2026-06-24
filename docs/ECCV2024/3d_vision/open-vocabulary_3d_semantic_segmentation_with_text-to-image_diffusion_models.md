---
title: >-
  [Paper Note] Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models
description: >-
  [ECCV 2024][3D Vision][Open-vocabulary 3D semantic segmentation] Diff2Scene is proposed, marking the first attempt to leverage a pretrained text-to-image diffusion model (Stable Diffusion) for open-vocabulary 3D semantic segmentation. Through an innovative mask distillation method, semantically rich mask embeddings from the 2D foundation model are transferred to a 3D geometry-aware mask model, outperforming the state-of-the-art by 12% on ScanNet200.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Open-vocabulary 3D semantic segmentation"
  - "Diffusion models"
  - "Mask distillation"
  - "Mask2Former"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 045c6da07dd0293c
---

# Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2407.13642](https://arxiv.org/abs/2407.13642)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Open-vocabulary 3D semantic segmentation, Diffusion models, Mask distillation, Mask2Former, Stable Diffusion  

## TL;DR

Diff2Scene is proposed, marking the first attempt to leverage a pretrained text-to-image diffusion model (Stable Diffusion) for open-vocabulary 3D semantic segmentation. Through an innovative mask distillation method, semantically rich mask embeddings from the 2D foundation model are transferred to a 3D geometry-aware mask model, outperforming the state-of-the-art by 12% on ScanNet200.

## Background & Motivation

- **Background**: Open-vocabulary 3D semantic segmentation aims to assign semantic labels described by arbitrary text to each 3D point. Existing methods are primarily based on point-wise feature distillation from CLIP (e.g., OpenScene).
- **Limitations of Prior Work**: CLIP foundation models perform poorly on fine-grained categories and compositional text queries, and their global representation optimization objective is unsuitable for dense prediction tasks requiring fine-grained local representations.
- **Key Challenge**: The rich local representations and text alignment capabilities exhibited by diffusion models in generative tasks have not yet been utilized for 3D semantic understanding. However, their generative features cannot be directly used for point-wise distillation in perception tasks.
- **Goal**: How to effectively utilize the semantically rich representations of diffusion models for open-vocabulary 3D segmentation, particularly overcoming the issue that diffusion features cannot be directly distilled point-wise.
- **Key Insight**: Adopting a mask-based segmentation paradigm (Mask2Former-style) to decouple semantic and spatial information via mask embeddings, enabling cross-modal mask distillation from 2D to 3D.
- **Core Idea**: Using the semantically rich mask embeddings from the 2D branch as fixed classifiers, allowing the 3D branch to learn and generate geometrically accurate 3D masks, achieving a synergy of "semantics from 2D, geometry from 3D".

## Method

### Overall Architecture

Diff2Scene consists of two branches: a **2D semantic understanding branch** (based on the open-vocabulary 2D segmentation model ODISE driven by a diffusion model) and a **3D geometry-aware mask model** (based on MinkowskiNet). The 2D branch predicts salient masks and their semantic embeddings from RGB images, while the 3D branch takes the point cloud and 2D mask embeddings as inputs to predict geometric masks. During inference, predictions from both masks are integrated to fuse saliency patterns and geometric information.

### Key Designs

**Module 1: 2D Semantic Understanding Model (Diffusion Backbone + Mask2Former)**

ODISE is adopted as the 2D branch, using Stable Diffusion (pre-trained on Laion-5B) as the feature backbone paired with a Mask2Former segmentation head. The model takes a 2D image as input and predicts $N$ 2D probability masks $\{\mathcal{B}_i^{2d}\}_{i=1}^N$ and their corresponding semantic embeddings $\{f_i^{2d}\}_{i=1}^N$. The diffusion feature dimension is 256, the CLIP feature dimension is 768, and the number of mask queries is $N=100$.

**Key Advantages**: The generative pre-training of the diffusion model provides powerful local representation capabilities, which are more suitable for dense prediction than the global contrastive learning of CLIP; the mask-based paradigm naturally decouples semantic and spatial information.

**Module 2: Geometry-Aware 3D Mask Model**

The 3D branch extracts features $\mathbf{F}^{3d} \in \mathbb{R}^{M \times D}$ from point clouds using MinkowskiNet18A. The 2D mask embeddings are used as linear classifiers to compute the logit for each 3D point belonging to the corresponding category:

$$\mathcal{S}_i = \langle \mathbf{F}^{3d}, f_i^{2d} \rangle$$

A 3D probability mask $\mathcal{B}'^{3d}_i$ is obtained via a sigmoid function.

**Module 3: Cross-Modal Mask Distillation**

The core innovation lies in the mask distillation loss. 2D masks are unprojected into the 3D space utilizing pixel-to-point correspondence to obtain $\mathcal{B}_i^{3d}$. The predicted masks from the 3D branch are then constrained to align with them:

$$\mathcal{L} = \sum_{i=1}^{N} 1 - \cos(\mathcal{B}'^{3d}_i, \mathcal{B}_i^{3d})$$

This loss implicitly forces the 3D model to learn high-resolution, semantically rich feature representations without requiring direct point-wise distillation of frozen diffusion features (the latter leads to training non-convergence).

### Loss & Training

- **Distillation Loss**: Mask-level cosine similarity loss (Eq. 2), rather than traditional point-wise feature distance.
- **Inference Ensemble**: Fusing predictions of the salient mask and the geometric mask: $\mathbf{p}^c = \lambda \sum_i p_i^c \cdot \mathcal{B}_i^{3d} + (1-\lambda) \sum_i p_i^c \cdot \mathcal{B}'^{3d}_i$, where $\lambda = 0.5$.
- **Training Configuration**: 200 epochs, batch size 8, Adam optimizer with lr=0.0001, and polynomial decay scheduler with power=0.9.
- **Zero-Annotation Training**: No 3D ground-truth annotations are required; only RGB images and reconstructed point clouds from the training set are used.
- **Text Inference**: Text embeddings are extracted using ViT-L/14 CLIP, simultaneously fusing discriminative (CLIP) and generative (Stable Diffusion) semantic features.

## Key Experimental Results

### Main Results

| Method | ScanNet | Matterport3D | ScanNet200 Head | ScanNet200 Common | ScanNet200 Tail | ScanNet200 All | Replica All |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenScene (2D/3D) | 47.5 | 42.6 | 20.0 | 9.7 | 5.1 | 11.6 | 14.9 |
| OpenMask3D | 34.0 | - | 19.6 | 7.5 | 4.5 | 10.5 | 4.8 |
| ConceptFusion | 33.3 | - | 17.5 | 6.3 | 2.8 | 8.8 | 4.6 |
| **Diff2Scene** | **48.6** | **45.5** | **25.6** | **11.5** | **6.9** | **14.2** | **17.5** |

Overall mIoU of 14.2% on ScanNet200 vs 11.6% for OpenScene (+22% relative gain), and 6.9% vs 5.1% in tail categories (+35% relative gain).

### Ablation Study

| Ablation Item | Replica mIoU |
|--------|:---:|
| Full Model | 17.5 |
| w/o 2D salient mask | 12.8 |
| w/o 3D geometric mask | 16.5 |
| w/o CLIP discriminative features | 15.5 |
| w/o Stable Diffusion generative features | 15.3 |

| Distillation Method | Distillation Type | Replica Head | Replica Tail | Replica All |
|---------|---------|:---:|:---:|:---:|
| Fine-tuned CLIP (OpenScene) | Point-based | 32.6 | 7.7 | 11.1 |
| Frozen diffusion feature | Point-based | Diverges | Diverges | Diverges |
| **Multimodal mask distillation** | **Mask-based** | **43.3** | **8.0** | **12.8** |

### Key Findings

- **Diffusion features cannot be directly distilled point-wise**: Employing frozen Stable Diffusion features for point-wise distillation leads to non-convergence during training, motivating the design of mask distillation.
- **Discriminative and generative features are complementary**: Removing either type of feature leads to an approximate 2% decrease in mIoU, while using both jointly achieves the best performance.
- **Salient mask vs geometric mask**: The 2D salient mask contributes more (a decrease of 4.7% when removed), but the 3D geometric mask is also indispensable (a decrease of 1.0% when removed).
- Significant lead on the unseen Replica dataset (17.5 vs 14.9), demonstrating generalization capability.
- Performance on tail classes is close to fully supervised methods (6.9 vs 7.9 of CSC-Pretrain), showcasing potential for handling long-tail distributions.

## Highlights & Insights

- **First to introduce diffusion models to open-vocabulary 3D segmentation**: Pioneeringly validates the utility of generative pretrained representations in 3D understanding.
- **Ingenious mask distillation paradigm**: By using "semantic embeddings as classifiers + 3D features as classified query objects," it bypasses the challenge where diffusion features cannot be directly distilled.
- **Strong compositional query capabilities**: Capable of handling complex compound text queries involving color, shape, location, and utility (e.g., "find the white sneakers closer to the desk chair").
- **Clear design philosophy**: Division of labor between "semantics from 2D foundation models, and geometric accuracy from 3D models" is highly logical.

## Limitations & Future Work

- Small-scale rare categories (e.g., rail) are still prone to misclassification.
- Fine-grained categories with high semantic similarity are easily confused (e.g., windowsill vs window).
- Inference requires running both the 2D diffusion model and the 3D model, resulting in high computational overhead.
- The ODISE model used in the 2D branch relies heavily on external pre-training, and end-to-end training has not been extensively explored.
- Has not been validated on outdoor datasets (e.g., nuScenes).

## Related Work & Insights

- **OpenScene**: The major baseline, pioneering the annotation-free point-wise distillation paradigm but constrained by the local representation capabilities of CLIP features.
- **ODISE**: Direct source of the 2D branch, which utilizes Stable Diffusion for open-vocabulary 2D segmentation.
- **Mask2Former**: The foundation work for mask-based segmentation paradigms, decoupling mask-prediction from semantic classification.
- **Insights**: Diffusion models acting as a "semantic bridge" for 3D feature extraction can be extended to tasks like 3D object detection and instance segmentation; the mask distillation paradigm can also be generalized to other cross-modal knowledge transfer scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to apply diffusion models to 3D open-vocabulary segmentation; the mask distillation design is novel and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation across four datasets alongside complete ablations, though outdoor scenes and efficiency analyses are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; the comparison figure with baselines (Fig. 2) is highly intuitive.
- **Value**: ⭐⭐⭐⭐ — Excellent performance on ScanNet200 tail categories, offering solid references for practical application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation](open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)

</div>

<!-- RELATED:END -->
