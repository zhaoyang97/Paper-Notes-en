---
title: >-
  [Paper Note] Occlusion-Aware Seamless Segmentation
description: >-
  [ECCV 2024][Segmentation][Panoramic Segmentation] Proposes a new task, Occlusion-Aware Seamless Segmentation (OASS), and the UnmaskFormer framework to simultaneously address three major challenges: unlocking the narrow field-of-view of panoramic images, complete amodal segmentation of occluded objects, and pinhole-to-panoramic cross-domain adaptation, achieving SOTA performance on the self-created BlendPASS dataset.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Panoramic Segmentation"
  - "Occlusion-Aware"
  - "Seamless Segmentation"
  - "Unsupervised Domain Adaptation"
  - "Amodal Segmentation"
date: 2026-05-08
content_hash: f0acba3fe6ac5a2b
---

# Occlusion-Aware Seamless Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.02182](https://arxiv.org/abs/2407.02182)  
**Code**: [https://github.com/yihong-97/OASS](https://github.com/yihong-97/OASS)  
**Area**: Image Segmentation  
**Keywords**: Panoramic Segmentation, Occlusion-Aware, Seamless Segmentation, Unsupervised Domain Adaptation, Amodal Segmentation

## TL;DR

Proposes a new task, Occlusion-Aware Seamless Segmentation (OASS), and the UnmaskFormer framework to simultaneously address three major challenges: unlocking the narrow field-of-view of panoramic images, complete amodal segmentation of occluded objects, and pinhole-to-panoramic cross-domain adaptation, achieving SOTA performance on the self-created BlendPASS dataset.

## Background & Motivation

**Background**: Panoramic scene understanding and occlusion-aware amodal segmentation have both made individual progress, but these two directions have developed independently for a long time. Panoramic segmentation methods (e.g., Trans4PASS, DATR) can handle 360° image distortion but cannot reason about occluded objects, whereas amodal methods (e.g., ORCNN, SLN) can predict complete occluded silhouettes but fail to generalize to panoramic images.

**Limitations of Prior Work**:
   - Panoramic images exhibit severe distortions, leading to a drastic performance drop when directly applying traditional segmentation models;
   - Existing semantic/instance segmentation can only predict visible regions, failing to reason about the complete shape of occluded areas;
   - Panoramic image annotation is extremely expensive (approx. 210 minutes per image), resulting in a scarcity of labels and necessitating unsupervised domain adaptation (UDA) to transfer knowledge from the label-rich pinhole domain.

**Key Challenge**: FoV occlusion, in-field object occlusion, and cross-domain gaps are entangled. Existing methods only address one of these issues individually, failing to achieve a "seamless" and comprehensive understanding.

**Goal**: Unify and solve three "masking" issues: (1) unmasking narrow FoV $\rightarrow$ panoramic 360°; (2) unmasking object occlusion $\rightarrow$ amodal complete segmentation; (3) unmasking domain gap $\rightarrow$ UDA from pinhole to panoramic.

**Key Insight**: Define a brand new task named OASS, construct a dedicated dataset BlendPASS, and design a unified framework UnmaskFormer to solve distortion handling, occlusion reasoning, and domain adaptation in one go.

**Core Idea**: Process distortion and occlusion via Unmasking Attention, and enhance cross-domain adaptation and occlusion reconstruction through Amodal-oriented Mix, seamlessly performing five types of segmentation tasks within a single transformer framework.

## Method

### Overall Architecture

UnmaskFormer consists of three major components:
- **UA-based Backbone**: A four-stage Transformer-based feature extractor containing Deformable Patch Embedding (DPE) and Unmasking Attention (UA) to simultaneously handle panoramic distortions and object occlusions;
- **Three-branch Decoder**: A semantic branch (per-pixel semantic classification), an instance branch (top-down instance segmentation based on Mask R-CNN), and an amodal instance branch (predicting complete occluded regions);
- **OAFusion Module**: Fuses the outputs of the three branches to generate five types of segmentation results simultaneously: semantic, instance, amodal instance, panoptic, and amodal panoptic segmentation.

### Key Designs

1. **Unmasking Attention (UA)**:

    - **Function**: Introduces an enhanced pooling layer after self-attention to generate occlusion-aware features.
    - **Mechanism**: After the feature vector $\boldsymbol{X'}$ passes through self-attention, global average pooling is used to obtain the pooled query $\boldsymbol{q} = GAP(\boldsymbol{X'})$. Then, cross-attention generates $\boldsymbol{q'} \in \mathbb{R}^{1 \times 1 \times C}$. A sigmoid function $\phi(\boldsymbol{q'})$ is applied to produce an occlusion-aware mask, which is element-wise multiplied by the original features to obtain the occlusion-aware features $\boldsymbol{X''} = \phi(\boldsymbol{q'}) \odot \boldsymbol{X'}$.
    - **Design Motivation**: Global pooling captures global image context, while the sigmoid mask selectively enhances or suppresses feature channels, enabling the network to learn to focus on information in occluded areas.

2. **Interleaved DPE**:

    - **Function**: Alternates Deformable Patch Embedding from being used solely in the initial stage to being interleaved in Stage 2 and Stage 4.
    - **Mechanism**: DPE captures local geometric variations through learnable adaptive offsets $\boldsymbol{\Delta}^{DPE}(i,j)$. Using DPE in late stages is more effective than in early ones.
    - **Design Motivation**: Panoramic images exhibit distortions at different feature levels; addressing them only in the initial stage is insufficient. Experiments show that employing DPE in deeper stages (Stage 2, 4) yields better results than in shallower stages (Stage 1, 3).

3. **Amodal-oriented Mix (AoMix)**:

    - **Function**: A cross-domain data augmentation strategy that utilizes amodal annotations to generate occluded training samples and blends source-target domain images.
    - **Mechanism**:
        - Randomly sample amodal instance masks $\{M_r^{(i)}\}_{i=1}^z$, and apply random scaling $RS(\cdot)$ and random padding $RP(\cdot)$ to generate a new mask $M_r = H(\sum_i RP(RS(M_r^{(i)})))$.
        - Occlude the "Thing" regions of the source image using $M_r$: $\hat{x}_s = (1 - M_r \cap M_s) \odot x_s$.
        - Randomly sample half of the semantic classes from the occluded source image $\hat{x}_s$ and paste them onto the target image $x_t$ to generate the mixed image $\hat{x}_m$.
    - **Design Motivation**: Using real object shapes (rather than random patches) to create occlusions mimics real-world scenes more closely, while mixing the source and target domains alleviates the domain gap.

4. **OAFusion (Occlusion-Aware Fusion)**:

    - **Function**: Fuses the outputs of the three branches to generate five types of segmentation results.
    - **Mechanism**: Semantic segmentation is output directly. The categories of instances / amodal instances are determined by a majority vote from the semantic branch. The key improvement is that in amodal scenarios, the voting only considers regions that do not overlap with other objects.
    - **Design Motivation**: Traditional fusion can cause misclassification when heavy occlusion occurs (e.g., a pedestrian heavily occluded by a car is misclassified as a car). OAFusion avoids this issue by ignoring overlapping regions during voting.

### Loss & Training

- **Source domain supervised loss** $\mathcal{L}_S$: Cross-entropy loss for the semantic branch, and standard Mask R-CNN losses (bbox + mask) for the instance and amodal branches.
- **Target domain self-training loss** $\mathcal{L}_T = -\omega \sum_{h,w,c} p_t^{(h,w,c)} \log \hat{y}_t^{(h,w,c)}$, where pseudo-labels are generated by an EMA teacher model of Mean-Teacher, and the weight $\omega$ is estimated based on a confidence threshold $\tau = 0.968$.
- **Total loss**: $\mathcal{L}_{total} = \mathcal{L}_S + \mathcal{L}_T$.
- Training configuration: AdamW optimizer, lr = $6 \times 10^{-5}$, weight decay 0.01, batch size 4, crop size 376×376, trained for 40k iterations.

## Key Experimental Results

### Main Results (KITTI360-APS → BlendPASS)

| Method | mIoU (SS) | mAPQ (APS) | mAP (Instance) | mAAP (Amodal) |
|------|-----------|------------|-----------------|---------------|
| DATR | 34.91% | 20.26% | 8.66% | 8.68% |
| Trans4PASS | 40.66% | 22.94% | 10.01% | 9.85% |
| UniDAPS | 38.46% | — | 3.43% | n.a. |
| EDAPS | 40.17% | 23.14% | 10.28% | 10.68% |
| Source-Only | 38.65% | 22.13% | 10.54% | 10.22% |
| **UnmaskFormer** | **43.66%** | **26.58%** | **11.10%** | **10.50%** |

### Panoramic Semantic Segmentation (Other Datasets)

| Dataset | Metric | UnmaskFormer | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| SynPASS | mIoU | **45.34%** | 44.80% (Trans4PASS) | +0.54% |
| DensePASS | mIoU | **48.08%** | 45.89% (Trans4PASS) | +2.19% |

### Ablation Study

| Component Config | mIoU | mAPQ | Description |
|----------|------|------|------|
| PE (baseline) | 41.07% | 22.00% | Original Patch Embedding |
| DPE (early stage) | 40.70% | 22.90% | DPE only in early stage |
| AvgPool | 42.10% | 23.90% | Standard Average Pooling |
| SimPool | 43.04% | 24.74% | SimPool replacement |
| **UA (ours)** | **43.06%** | **25.04%** | Unmasking Attention |
| UA + AoMix | 42.39% | 25.17% | Adding AoMix |
| UA + AoMix + OAFusion | **43.66%** | **26.58%** | Full model |

### Ablation of AoMix Strategies

| Strategy | mIoU | mAPQ | Description |
|------|------|------|------|
| T for S | 42.53% | 23.98% | Occlusion only on source image |
| T for M | 43.18% | 24.78% | Occlusion only on mixed image |
| P for S&M | 42.52% | 21.87% | Occlusion with random patches |
| W for S&M | 41.64% | 24.12% | Occlude all regions |
| **AoMix (ours)** | **42.39%** | **25.17%** | Occlude only Thing categories |

### Key Findings

- UA achieves a +3.04% improvement in mAPQ compared to the baseline PE, demonstrating the effectiveness of the occlusion-aware pooling attention.
- AoMix using real object shapes for occlusion achieves significantly better results than random patch-based occlusion (mAPQ +3.3%), validating the importance of amodal-oriented augmentation.
- OAFusion successfully resolves the misclassification issue of traditional fusion methods in heavily occluded scenarios.
- DPE performs better in the deep stages (Stages 2, 4) compared to the early stages (Stages 1, 3).
- UnmaskFormer has only 13.96M parameters, which is comparable to Trans4PASS (13.93M) but delivers significantly better performance.

## Highlights & Insights

- **Novelty of Task Definition**: OASS unifies panoramic segmentation, amodal segmentation, and UDA into a single task for the first time, filling a gap in the literature.
- **BlendPASS Dataset**: Contains 2,000 panoramic images for training and 100 finely annotated ones for testing, including 2,960 Thing instances where 43% contain occlusions. The annotation quality is guaranteed by three-person cross-validation.
- **Simple Yet Effective UA Design**: Simply adding a pooling attention layer after self-attention significantly boosts the occlusion-awareness capacity.
- **Practical Value of OAFusion**: Solves the fundamental flaw of semantic voting in amodal scenarios, ensuring heavily occluded objects are no longer misclassified into the category of the occluding object.
- A single model simultaneously produces five types of segmentation outputs (semantic/instance/amodal instance/panoptic/amodal panoptic) under a unified and efficient design.

## Limitations & Future Work

- The test set contains only 100 images, which is relatively small and might lead to performance evaluation fluctuations.
- Panoramic image annotation is extremely costly (210 mins/image/person), making it difficult to scale up the dataset.
- Performance on certain minor categories (e.g., van, traffic-light) remains quite low, showing that cross-domain adaptation struggles with long-tailed classes.
- Only validated on driving scenarios; indoor panoramic scenes are not yet explored.
- AoMix relies on source-domain amodal annotations, which limits the model's generalizability.
- The mAAP of amodal instance segmentations is only 10.50%, leaving a large room for improvement in overall performance.

## Related Work & Insights

- **Trans4PASS** [CVPR 2022]: Proposes DPE to handle panoramic distortions, serving as the baseline for UnmaskFormer's backbone.
- **EDAPS** [CVPR 2023]: SOTA for UDA panoramic segmentation, upon which UnmaskFormer builds to incorporate amodal capabilities.
- **DAFormer/DACS**: Offers the framework for self-training and class-mix UDA strategies.
- **ORCNN**: An amodal segmentation method that infers occluded masks through visible masks.
- **Insight**: Integrating amodal information into data augmentation (AoMix) is a clever design, which is more suited for occlusion scenarios than traditional CutMix/ClassMix.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Defines the OASS task for the first time, unifying three independent challenges through a pioneering task formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on three datasets with detailed ablation and rich visualizations, though the test set scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic and beautiful figures, with the triple "unmasking" concept consistently thread throughout the narrative.
- **Value**: ⭐⭐⭐⭐ The dataset and benchmarks hold long-term value, though absolute performance remains relatively low, leaving a gap for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] EAFormer: Scene Text Segmentation with Edge-Aware Transformers](eaformer_scene_text_segmentation_with_edge-aware_transformers.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation](../../CVPR2026/segmentation/learning_and_aligning_click-aware_shape_prior_for_interactive_amodal_instance_se.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)

</div>

<!-- RELATED:END -->
