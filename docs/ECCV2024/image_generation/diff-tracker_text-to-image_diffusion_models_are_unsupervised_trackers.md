---
title: >-
  [Paper Note] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers
description: >-
  [ECCV 2024][Image Generation][None] This paper proposes Diff-Tracker, which is the first to leverage the rich visual semantic knowledge embedded in pre-trained text-to-image diffusion models (Stable Diffusion) for unsupervised object tracking. It achieves continuous tracking by learning a prompt representing the target and updating it online.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "None"
date: 2026-05-08
content_hash: 50149267ca9d94d1
---

# Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers

**Conference**: ECCV 2024  
**arXiv**: [2407.08394](https://arxiv.org/abs/2407.08394)  
**Code**: None  
**Area**: Visual Tracking / Diffusion Models  
**Keywords**: None  

## TL;DR

This paper proposes Diff-Tracker, which is the first to leverage the rich visual semantic knowledge embedded in pre-trained text-to-image diffusion models (Stable Diffusion) for unsupervised object tracking. It achieves continuous tracking by learning a prompt representing the target and updating it online.

## Background & Motivation

### Background
Visual object tracking is a core task in computer vision. Deep learning-based trackers (such as SiamFC and MixFormer) rely heavily on large amounts of annotated data for supervised training. Due to the high cost of annotation, **unsupervised visual tracking** has gained significant attention in recent years. Existing unsupervised methods (such as UDT, USOT, and ULAST) mostly adopt Siamese network architectures, trained via self-supervised strategies like forward-backward consistency or adversarial occlusion.

### Limitations of Prior Work
Unsupervised tracking faces two major challenges:

**Insufficient utilization of semantic and structural information**: It is difficult to effectively leverage the rich semantic and spatial structural information within video frames.

**Inadequate exploration of contextual relationships**: The temporal contextual relationships in videos are underutilized.

### Key Challenge & Key Insight
Pre-trained text-to-image diffusion models (such as Stable Diffusion) have demonstrated a comprehensive understanding of visual representations—ranging from pixel-level semantic details to spatial layouts (object textures, shapes, and spatial arrangement). Researchers have also verified that diffusion models possess a solid understanding of video contextual relationships, even without being trained on video datasets.

**Key Insight**: The **cross-attention map** in diffusion models associates the semantics of the text prompt with the image content—meaning the prompt can activate semantically-related regions on the attention map. If a prompt representing the tracked target can be learned, the rich knowledge of the diffusion model can be leveraged to activate the target region in any frame, thereby achieving tracking.

### Two Technical Challenges
1. The learned prompt needs to encode relationship information between the target and the background (which is crucial for handling occlusions and deformations).
2. Target motion causes appearance changes; hence, the prompt needs to be dynamically updated online to adapt to these variations.

## Method

### Overall Architecture

Diff-Tracker consists of two stages:
1. **Initial Prompt Learner**: Learns an initial prompt on the first frame that can activate the target region.
2. **Online Prompt Updater**: Dynamically updates the prompt based on target motion information to achieve continuous tracking in subsequent frames.

### Key Designs

#### 1. Attention Harmonization

**Function**: Fuses the cross-attention map and the self-attention map to enable the learned prompt to encode relationship information between the target and the background.

**Mechanism**: The self-attention map $M_s$ of the diffusion model captures semantic correlations among pixels. For each pixel on the cross-attention map $M_c$, it is enhanced using its corresponding self-attention map:

$$M_c'(:,:) = \sum_{i=1}\sum_{j=1} M_c(i,j) \cdot M_s(i,j,:,:)$$

The enhanced cross-attention map $M_c'$ encodes the relationships between pixels. Finally, the output attention map is obtained through weighted fusion:

$$\mathcal{M} = (1-\alpha) \cdot M_c' + \alpha \cdot M_c$$

where $\alpha=0.5$ is a balancing hyperparameter.

**Design Motivation**: Pure cross-attention only reflects the correlation between the prompt and individual pixels, lacking modeling of target-background relationships. Introducing self-attention allows the prompt to simultaneously perceive "what the target is" and "the relationship between the target and its surrounding environment," which is particularly critical in scenarios with occlusions and similar distractors.

#### 2. Initial Prompt Learning

**Function**: Learns a prompt embedding from the first frame that can accurately activate the target bounding box region.

**Mechanism**:
1. Encode the first frame $f_1$ into the latent space and add noise.
2. Feed the noisy latent representation $z$ and the prompt to be learned $p_1$ into the denoising UNet $\epsilon_\theta$.
3. Extract the cross-attention map $M_c$ and obtain $\mathcal{M}$ through attention harmonization.
4. Calculate the MSE loss between $\mathcal{M}$ and the ground-truth (GT) attention map (where only the bounding box region is activated).

Total loss:

$$L = \|\mathcal{M} - \mathcal{F}_1\|_2^2 + L_{DM}$$

where $L_{DM}$ is the diffusion model denoising loss, which constrains the prompt to stay within the text embedding space understandable by the diffusion model. During training, the parameters of the diffusion model are frozen, and only the prompt embedding (dimension 1024) is updated.

**Design Motivation**: Since the target can be arbitrary (and hard to describe with text), a learnable prompt embedding is used instead of textual words. The GT attention map is defined as active only within the bounding box, which is simple yet effective.

#### 3. Online Prompt Updater

**Function**: Dynamically updates the prompt based on the target's motion information in subsequent frames to adapt to appearance changes.

**Mechanism**:

**Motion Information Extractor**: Simultaneously extracts long-term and short-term motion information:
- **Short-term motion** $m_k^S$: The current frame and the previous frame are fed into a motion encoder (ResNet18 + Conv3D).
- **Long-term motion** $m_k^L$: The current frame and the previous $T$ frames are stacked and fed into another motion encoder.

Through cross-attention, the global motion information is associated with the target's appearance features to obtain target-conditioned motion information:

$$l_k^L = \text{Cross-Attn}(Q_k, m_k^L), \quad l_k^S = \text{Cross-Attn}(Q_k, m_k^S)$$

where $Q_k$ query is obtained from the appearance features of the target template in the first frame. A fusion head (MLP) fuses the long-term and short-term motion details into $l_k$.

**Prompt Update**: The prompt is updated via a blending head $\mathcal{H}_b$ and a residual connection:

$$p_k = (1-\beta) \cdot \mathcal{H}_b(p_{k-1} + l_k) + \beta \cdot p_{k-1}$$

$\beta=0.7$ ensures update stability, keeping the previous frame's prompt as the primary component, supplemented by motion information.

**Design Motivation**: Relying solely on short-term motion can be unreliable due to occlusions or illumination changes. Thus, long-term motion is introduced to enhance spatio-temporal continuity. The residual update mechanism prevents drastic prompt changes, avoiding tracking drift.

### Loss & Training

- Initial prompt learning: MSE attention loss + diffusion model denoising loss, Adam optimizer, lr=$5 \times 10^{-3}$, 3 epochs.
- Online updater training: MSE loss (cross-attention map of updated prompt vs. GT), Adam optimizer, lr=$5 \times 10^{-4}$, 35 epochs.
- Pseudo-label source: Generated using an optical flow model on GOT-10k, ImageNet VID, LaSOT, and YouTube-VOS.
- At test time: Learn the initial prompt on the first frame, and update the prompt online every frame starting from the 6th frame.

## Key Experimental Results

### Main Results

**TrackingNet / VOT2016 / VOT2018 Results (Comparison with Unsupervised Methods)**:

| Method | Supervised | TrackingNet Suc↑ | TrackingNet Pre↑ | VOT2016 EAO↑ | VOT2018 EAO↑ |
|------|------|-----------------|-----------------|-------------|-------------|
| KCF | No | 0.447 | 0.419 | 0.192 | 0.135 |
| USOT* | No | 0.616 | 0.566 | 0.402 | 0.344 |
| ULAST*-on | No | 0.654 | 0.592 | 0.417 | 0.355 |
| **Diff-Tracker (Ours)** | **No** | **0.675** | **0.614** | **0.430** | **0.365** |
| SiamFC | Yes | 0.571 | 0.533 | 0.235 | 0.188 |
| DiMP | Yes | 0.740 | 0.687 | - | 0.440 |

**OTB2015 / LaSOT Results**:

| Method | Supervised | OTB2015 Suc↑ | OTB2015 Pre↑ | LaSOT Suc↑ | LaSOT Pre↑ |
|------|------|-------------|-------------|-----------|-----------|
| USOT* | No | 0.574 | 0.775 | 0.358 | 0.340 |
| ULAST*-on | No | 0.648 | 0.879 | 0.471 | 0.451 |
| **Diff-Tracker (Ours)** | **No** | **0.661** | **0.898** | **0.486** | **0.472** |

It sets new state-of-the-art (SOTA) results for unsupervised tracking across all five datasets, and is competitive with supervised methods on some metrics (e.g., outperforming classical supervised methods like SiamFC).

### Ablation Study

**Component Ablation on VOT2018**:

| Configuration | EAO↑ | Acc↑ | Rob↓ | Description |
|------|------|------|------|------|
| w/o attention harmonization | 0.359 | 0.577 | 0.280 | Attention harmonization encodes target-background relationship |
| w/o online prompt updater | 0.349 | 0.571 | 0.292 | Online update adapts to target motion variations |
| w/o long-term motion | 0.355 | 0.572 | 0.286 | Long-term motion enhances spatio-temporal continuity |
| w/o short-term motion | 0.360 | 0.577 | 0.281 | Short-term motion captures immediate changes |
| **Full (Ours)** | **0.365** | **0.580** | **0.273** | Coordination of all components yields optimal results |

### Key Findings

1. The online prompt updater yields the most significant contribution (excluding it drops the EAO from 0.365 to 0.349), confirming the necessity of dynamically adapting to target variations.
2. Attention harmonization effectively encodes the target-background relationship (improving EAO by 0.006).
3. Long-term and short-term motion information are complementary; removing either individually leads to performance degradation.
4. Pre-trained diffusion models can be effectively transferred to tracking tasks without fine-tuning, demonstrating strong generalization capability in visual understanding.

## Highlights & Insights

1. **Novel Perspective Shift**: Reinterprets text-to-image diffusion models as "semantic-to-image-region bridges," achieving tracking through prompt learning.
2. **No Textual Description Required**: Tracked targets may be impossible to describe with text; the use of learnable embeddings elegantly bypasses this limitation.
3. **Clever Harnessing of Self-Attention**: Self-attention in diffusion models naturally encodes pixel-to-pixel relationships. Attention harmonization gracefully introduces target-background relationships.

## Limitations & Future Work

1. The prompt needs to be learned/updated for every frame at test time, which may result in slow inference speed.
2. The GT attention map is defined as a simple bounding-box activation, which does not utilize fine-grained segmentation information.
3. The target is represented by only a single prompt token, which has limited capacity to describe complex targets.
4. It relies on the pre-trained knowledge of Stable Diffusion and may perform poorly on target categories that are rare in the diffusion model's training data.
5. Future work could explore using more efficient diffusion models (e.g., SDXL Turbo) to accelerate inference.

## Related Work & Insights

- **ULAST** (CVPR 2024): The previous SOTA in unsupervised tracking, utilizing Siamese networks and adversarial learning; Ours outperforms it.
- **Prompt-to-Prompt** (ICLR 2023): Reveals the precise control of cross-attention maps in diffusion models over image regions.
- **DiffPose** (CVPR 2023): Applies diffusion models to pose estimation, sharing the same "diffusion model transfer" paradigm as Ours.
- Insight: The representational knowledge embedded in large-scale pre-trained generative models has the potential to be transferred to various discriminative visual tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first to apply text-to-image diffusion models to unsupervised tracking, featuring a novel perspective and an elegant approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive SOTA across five benchmarks with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivational derivation with self-consistent logic from observations to the proposed approach.
- **Value**: ⭐⭐⭐⭐ — Establishes a new paradigm for diffusion models in tracking, offering strong inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)

</div>

<!-- RELATED:END -->
