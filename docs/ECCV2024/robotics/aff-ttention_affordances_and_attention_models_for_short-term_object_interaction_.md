---
title: >-
  [Paper Note] AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation
description: >-
  [ECCV 2024][Robotics][Short-Term Anticipation] Proposes the STAformer architecture and two affordance-based modules (an environment affordance database + interaction hotspots), improving the relative performance of Short-Term object Interaction Anticipation (STA) in egocentric videos by 30-45% on Ego4D and EPIC-Kitchens.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Short-Term Anticipation"
  - "Affordances"
  - "Egocentric Video"
  - "Attention"
  - "Object Interaction"
date: 2026-05-08
content_hash: 75e266c7f0350d01
---

# AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation

**Conference**: ECCV 2024  
**arXiv**: [2406.01194](https://arxiv.org/abs/2406.01194)  
**Code**: [https://github.com/lmur98/AFFttention](https://github.com/lmur98/AFFttention)  
**Area**: Robotics  
**Keywords**: Short-Term Anticipation, Affordances, Egocentric Video, Attention, Object Interaction

## TL;DR
Proposes the STAformer architecture and two affordance-based modules (an environment affordance database + interaction hotspots), improving the relative performance of Short-Term object Interaction Anticipation (STA) in egocentric videos by 30-45% on Ego4D and EPIC-Kitchens.

## Background & Motivation
**Background**: Short-Term Object Interaction Anticipation (STA) requires simultaneously predicting the location (bbox), class (noun), action (verb), and time-to-contact of the next interacting object from egocentric videos. This is a crucial capability for wearable assistants and human-robot interaction.

**Limitations of Prior Work**:
   - Most prior methods (StillFast, TransFusion, GANO) are based on convolutions or simple feature fusion, leading to insufficient modeling of video dynamics.
   - The fusion of image and video branches is coarse (e.g., summation or concatenation), failing to fully exploit complementary information from both.
   - Predictions lack prior constraints concerning human behavior patterns, leading to false positives that are inconsistent with scene functionality.

**Key Challenge**: STA requires understanding both high-resolution spatial details (for object localization) and temporal dynamics (for predicting future actions), but existing methods only fuse these two dimensions shallowly.

**Goal**: (a) How to achieve better fusion of image and video features for STA? (b) How to leverage interaction priors in the environment to constrain predictions?

**Key Insight**: Approaching from two complementary directions: structured attention fusion (STAformer) and affordance priors based on patterns of human behavior (Environment Affordances + Interaction Hotspots).

**Core Idea**: Utilizes frame-guided temporal pooling and dual cross-attention to achieve fine-grained image-video fusion, while employing an affordance database and interaction hotspots to constrain predictions on both semantic and spatial levels.

## Method

### Overall Architecture
The input consists of a high-resolution image $I_T$ (the last frame) and a low-resolution video clip $\mathcal{V}_{T-t:T}$. DINOv2 is used to extract 2D image tokens, and TimeSformer is used to extract 3D video tokens. A three-stage pipeline is applied: (1) Frame-guided Temporal Pooling aligns 3D video features to the 2D space of the last frame; (2) Dual Image-Video Attention fuses image and video features via bidirectional cross-attention; and (3) a feature pyramid and prediction heads output detection results. Finally, two affordance modules refine the predictions post-hoc.

### Key Designs

1. **Frame-guided Temporal Pooling Attention**:

    - **Function**: Compresses 3D spatiotemporal tokens from TimeSformer into a 2D space, aligning them with the last frame.
    - **Mechanism**: Uses video tokens of the last frame as the query, and tokens of all frames as keys/values, achieving adaptive temporal pooling via residual cross-attention: $\Phi_{3D}^{2D} = \Phi_{3D}(\mathcal{V}_T) + A(Q_{last}, K_{all}, V_{all})$
    - **Design Motivation**: More flexible than mean pooling or convolutional pooling, as it adaptively aggregates temporal information based on the spatial positions of the last frame. Ablative experiments show a +0.76 mAP All improvement over mean pooling.

2. **Dual Image-Video Attention**:

    - **Function**: Bidirectional cross-attention that allows image tokens to incorporate video dynamics, while enabling video tokens to capture high-resolution spatial details.
    - **Mechanism**: Two parallel residual cross-attention layers—image-guided (image as Q, video as KV) and video-guided (video as Q, image as KV)—each outputting refined tokens.
    - **Design Motivation**: Unidirectional attention (only image-to-video or video-to-image) is less effective than bidirectional, as each modality carries complementary information. Ablation shows that bidirectional attention improves by ~0.1 mAP All over sum fusion.

3. **Environment Affordance Module**:

    - **Function**: Constructs an "environment-interaction" database from the training set; at inference time, it matches the current scene to functionally similar regions in the database to obtain possible noun/verb distributions.
    - **Mechanism**: (a) Clusters activity-centric zones from training videos using a Siamese network; (b) Records the set of co-occurring nouns/verbs for each zone; (c) During inference, encodes the current video using EgoVLP-v2 and finds the closest 2K zones via KNN (K visual matches + K textual matches); (d) Computes a weighted affordance probability distribution by similarity: $p_{aff}(n|\mathcal{V}') \propto \exp(\sum S_i \cdot \mathbb{1}_{n \in \mathcal{N}^{Z_i}})$; (e) Performs Bayesian fusion between this distribution and the STA model predictions.
    - **Design Motivation**: Human behaviors in similar environments exhibit consistency (e.g., kitchen $\rightarrow$ picking up a knife to cut vegetables). This prior can constrain the predicted noun/verb distribution, reducing illogical predictions.

4. **Interaction Hotspot Module**:

    - **Function**: Predicts the spatial distribution of where future interactions are likely to occur by observing hand and object trajectories.
    - **Mechanism**: Modified based on the hand motion prediction method by Liu et al.—using a hand-object detector finetuned on Ego4D combined with EgoVLP features to output an interaction probability map $p_{ih}(x,y)$ for each pixel, which is then used to re-weight STA detection confidence at the bounding box centers.
    - **Design Motivation**: The affordance database addresses the "what" of physical interactions, whereas the hotspot module addresses the "where", making them complementary.

### Loss & Training
- Adopts the Faster-RCNN prediction heads from StillFast, including bbox regression, classification, and confidence loss.
- The last 3 blocks of DINOv2 and the last 3 blocks of TimeSformer are finetuned.
- The Affordance and Hotspot modules are post-processing steps and not involved in end-to-end training.

## Key Experimental Results

### Main Results

| Dataset | Metrics (mAP All) | STAformer+AFF | Prev. SOTA | Gain |
|--------|---------------|---------------|-----------|---------|
| Ego4D v1 val | All Top-5 mAP | 3.77 | 2.60 (TransFusion) | +45.0% |
| Ego4D v2 val | All Top-5 mAP | 5.67 | 3.99 (GANO v2) | +42.1% |
| Ego4D v2 test | All Top-5 mAP | 6.75 | 5.18 (Language NAO) | +30.3% |
| EPIC-Kitchens val | All Top-5 mAP | 4.69 | 3.28 (StillFast) | +42.9% |

### Ablation Study

| Configuration | N mAP | N+V mAP | All mAP | Description |
|------|-------|---------|---------|------|
| StillFast baseline | 16.21 | 7.47 | 2.48 | CNN baseline |
| DINOv2 image only (A1) | 17.48 | 8.64 | 2.52 | Image-only already outperforms baseline |
| + TimeSformer mean pool (B1) | 16.67 | 8.38 | 2.63 | Mean pooling yields limited benefit |
| + Frame-guided pooling (B3) | 19.78 | 10.04 | 3.39 | Key design, brings massive improvement |
| + Dual attention (C1) | 20.08 | 10.21 | 3.47 | Bidirectional attention brings minor improvement |
| + Finetune DINOv2 (C2=STAformer) | 21.71 | 10.75 | 3.53 | Finetuning the image encoder is important |
| + Multi-head variant (C5) | 23.02 | 11.57 | 3.85 | Multi-head attention yields further improvement |

### Key Findings
- **Frame-guided temporal pooling contributes the most**: Improves All mAP by 0.76 from B1 to B3, making it the most critical component in STAformer.
- **Environment affordance is also effective on StillFast**: Improves from 2.48 to 2.85 (StillFast+AFF), showing that the affordance module is model-agnostic.
- **Bidirectional attention outperforms unidirectional variants**: C2 (bidirectional) outperforms C3 (I $\rightarrow$ V) and C4 (V $\rightarrow$ I), indicating both information flows are valuable.
- **N+V metrics benefit the most**: Affordance mainly improves semantic prediction by constraining noun/verb distributions; the relative gain in N+V reaches up to 58.9%.

## Highlights & Insights
- **Affordance database as external memory**: This is an elegant, non-parametric way to inject domain knowledge—requiring no model retraining, only database matching. This retrieval-augmented paradigm can be transferred to any task requiring scene priors.
- **Model-agnostic nature of post-processing modules**: Both environment affordance and interaction hotspot modules are post-hoc filters that can be plugged into any STA model (validated on both StillFast and STAformer), significantly increasing practical utility.
- **Dual-route retrieval (visual + textual) for affordance**: Uses visual similarity to retrieve scenes with similar appearance, and cross-modal similarity to find scenes with similar functionality but different appearances (e.g., kitchens in different countries). This dual-route strategy is highly intuitive and effective.

## Limitations & Future Work
- **Affordance database requires extensive training data annotations**: Constructing zones involves training Siamese networks and clustering, which might limit generalization to new domains.
- **Non-end-to-end post-processing**: Being applied post-hoc, the affordance and hotspot modules cannot be optimized via backpropagation. Integrating them into the training loop could potentially yield better performance.
- **Time complexity**: KNN retrieval over the affordance database adds extra computation during inference; the paper does not discuss real-time viability.
- **EPIC-Kitchens STA annotations are self-generated by the authors**: The annotation quality and evaluation fairness of this new benchmark require verification by the community.

## Related Work & Insights
- **vs StillFast**: StillFast uses a CNN architecture (R50+X3D) with sum fusion, whereas this work upgrades the backbone to a Transformer architecture (DINOv2+TimeSformer) with dual attention, resulting in a comprehensive upgrade at the architectural level.
- **vs EGO-TOPO**: EGO-TOPO directly predicts affordances using neural networks, whereas this work adopts KNN database retrieval. Experiments show the retrieval paradigm is superior (EGO-TOPO's affordance actually degraded StillFast's performance when integrated).
- **vs TransFusion**: TransFusion introduces a language encoder for multimodal fusion, whereas this work's affordance method achieves superior semantic constraint without requiring auxiliary language inputs during inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The affordance database and STAformer designs are novel, though the individual components are not radically groundbreaking on their own.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, exhaustive ablations, validation of module transferability, and comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations, though the heavy notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ Achieves state-of-the-art results by a large margin on the STA task; the affordance retrieval concept is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](../../CVPR2026/robotics/ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[ICLR 2026\] Nonparametric Teaching of Attention Learners](../../ICLR2026/robotics/nonparametric_teaching_of_attention_learners.md)
- [\[AAAI 2026\] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](../../AAAI2026/robotics/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)
- [\[ECCV 2024\] DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control](disco_embodied_navigation_and_interaction_via_differentiable_scene_semantics_and.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](../../AAAI2026/robotics/theory_of_mind_for_explainable_human-robot_interaction.md)

</div>

<!-- RELATED:END -->
