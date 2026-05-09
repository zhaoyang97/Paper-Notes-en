---
title: >-
  [Paper Note] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation
description: >-
  [CVPR2026][foot contact estimation] This paper proposes FECO, a framework that achieves robust dense foot contact estimation from a single RGB image via shoe style–content randomization (adversarial training) and ground-aware learning (pixel height maps + ground normals), significantly outperforming existing methods on multiple benchmarks.
tags:
  - CVPR2026
  - foot contact estimation
  - shoe style invariance
  - ground-aware learning
  - adversarial training
  - dense contact prediction
date: 2026-05-08
content_hash: 62ecc2e04c3aa70f
---

# Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation

**Conference**: CVPR2026  
**arXiv**: [2511.22184](https://arxiv.org/abs/2511.22184)  
**Code**: [dqj5182/FECO_RELEASE](https://github.com/dqj5182/FECO_RELEASE)  
**Area**: Others  
**Keywords**: foot contact estimation, shoe style invariance, ground-aware learning, adversarial training, dense contact prediction

## TL;DR

This paper proposes FECO, a framework that achieves robust dense foot contact estimation from a single RGB image via shoe style–content randomization (adversarial training) and ground-aware learning (pixel height maps + ground normals), significantly outperforming existing methods on multiple benchmarks.

## Background & Motivation

**Importance of foot contact**: Human locomotion and balance fundamentally depend on foot–environment interaction; accurately capturing foot contact regions is critical for understanding body motion dynamics and modeling physically plausible behavior.

**Limitations of prior work**: Existing methods mostly reduce foot contact to joint-level predictions and rely on geometric heuristics such as zero-velocity constraints, failing to capture dense contact patterns distributed across fine-grained regions of the foot sole.

**Insufficient accuracy of general-purpose models**: Although dense human contact estimation methods exist (POSA, BSTRO, DECO), their prediction accuracy on foot regions remains poor; body-part-specific contact models (e.g., HACO for hands) have been shown to outperform general-purpose ones.

**Challenge of shoe appearance diversity**: In real-world settings, feet are typically covered by shoes whose color, texture, material, and style vary greatly, causing models to overfit to spurious correlations between shoe appearance and contact patterns in the training data (e.g., sneakers → skateboarding motions).

**Ambiguity of ground information**: Ground surfaces (carpet, asphalt, flooring) often have monotonous or repetitive textures, providing sparse visual cues; since contact occurs along directions parallel to the ground, the absence of explicit ground geometry reasoning leads to inaccurate predictions.

**Occlusion and viewpoint variation**: The above challenges are compounded by occlusion, viewpoint, and illumination changes, highlighting the need for representation learning that captures geometric and physical context rather than surface appearance.

## Method

### Overall Architecture

FECO consists of five core modules: (1) low-level style randomization, (2) shoe style–content randomization, (3) ground feature learning, (4) spatial attention fusion, and (5) a foot contact decoder. During training, each sample is processed simultaneously as one clean image and two low-level style-randomized images, with all modules trained end-to-end jointly.

### Key Designs

**Low-level Style Randomization**: Pro-RandConv is adopted to apply random local texture transformations to input images by sampling random convolution weights, deformable convolution offsets, and affine parameters (deformable convolution → Instance Normalization → affine transform → tanh), eliminating the model's reliance on local low-level texture statistics.

**Shoe Style-Content Randomization**: The external shoe image dataset UT Zappos50K (50K images covering shoes/sandals/slippers/boots) is used as an independent style source (rather than sampling within mini-batches). Shoe features extracted by a ViT are fed into two parallel branches:

- **Content randomization branch**: Adversarial adapters $\mathbf{A}_{\text{prev}}$ and $\mathbf{A}_{\text{after}}$ (zero-initialized 3×3 convolutions with learnable scaling factor γ=0.02) apply AdaIN style transfer to shoe features, preserving shoe content while injecting the style statistics of the input image. This branch is used for adversarial training to prevent the predictor from overfitting to style cues in the input.
- **Style randomization branch**: An interpolation weight α is sampled from a uniform distribution and used to interpolate between the channel statistics of input features and shoe features before executing AdaIN, generating shoe-style-invariant representations and exposing the model to diverse visual styles during training.

**Ground-Aware Learning**: A ground feature encoder is introduced, producing multi-scale features used for:

- **Pixel Height Map**: A DPT decoder predicts per-pixel height, scaled to pixel units by the maximum image edge length, providing dense geometric context.
- **Ground Normal**: Foot segmentation masks suppress foot-region features to prevent shortcut learning; global average pooling followed by two fully connected layers, tanh activation, and L2 normalization produces a unit-length ground normal vector.

**Spatial Attention Fusion**: Randomized features and ground features are concatenated along the channel dimension, passed through a 3×3 convolution reducing to 256 channels → ReLU → Dropout(0.2) → 1×1 convolution outputting two-way softmax weights, which adaptively weight-fuse ground features and style-invariant features.

**Foot Contact Decoder**: A Transformer architecture (self-attention + cross-attention) takes contact tokens and image features as input and outputs contact logits for 265 foot mesh vertices; sigmoid activation yields contact probabilities, which are further projected via a regressor to multi-level predictions for 11 joints and 3 keypoints (OpenPose-defined).

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{main}} + \mathcal{L}_{\text{style}} + \mathcal{L}_{\text{style-adv}} + \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{ground}}$$

- $\mathcal{L}_{\text{main}}$: BCE loss for multi-level predictions from the main branch
- $\mathcal{L}_{\text{style}}$: BCE loss for the style branch (gradients backpropagated only to the style branch decoder)
- $\mathcal{L}_{\text{style-adv}}$: BCE between style branch predictions and a uniform distribution (trains only the adversarial adapters)
- $\mathcal{L}_{\text{mask}}$: Average of BCE and Dice losses for foot segmentation
- $\mathcal{L}_{\text{ground}} = \mathcal{L}_{\text{pixel-height}}(\text{MAE}) + \mathcal{L}_{\text{ground-normal}}(\text{cosine similarity})$

All losses are computed separately for the clean image and two ProRandConv-augmented images, then averaged.

## Key Experimental Results

### Datasets & Setup

Training uses 10 datasets (PROX/BEHAVE/InterCap/EgoBody/RICH/MOYO/Hi4D/MMVP/MotionPRO + the authors' own COFE), covering foot–scene/object/ground/body interactions with millions of images. The primary evaluation set is MMVP. ViT-Huge backbone, AdamW (lr=1e-5), batch=4, trained for 10 epochs on a single A6000.

### Main Results

| Method | Precision ↑ | Recall ↑ | F1-Score ↑ |
|--------|------------|----------|------------|
| POSA | 0.276 | 0.308 | 0.255 |
| BSTRO | 0.436 | 0.538 | 0.464 |
| DECO | 0.374 | 0.511 | 0.409 |
| **FECO (Ours)** | **0.563** | **0.613** | **0.577** |

FECO surpasses BSTRO by 11.3% and DECO by 16.8% in F1 on MMVP. On joint-level foot contact estimation (COFE dataset video sequences), FECO—the only method not using temporal information—achieves F1=0.515, substantially outperforming WHAM (0.363) and Footskate Reducer (0.301).

### Ablation Study

| Ablation | F1-Score |
|----------|----------|
| w/o low-level randomization | 0.555 |
| + low-level randomization | 0.577 (+4.0%) |
| w/o style/content randomization | 0.522 |
| + content randomization | 0.531 |
| + style randomization | 0.554 |
| + both combined | 0.577 (+10.5%) |
| w/o ground learning | 0.506 |
| + ground normal | 0.527 |
| + pixel height map | 0.569 |
| + spatial attention | 0.577 (+14.0%) |

### Key Findings

- **Complementarity of style–content randomization**: Content randomization improves recall (coverage) while style randomization improves precision (robustness); combining both achieves optimal F1 balance.
- **Clear incremental gains from ground geometry**: Ground normals provide global orientation → pixel height maps provide dense geometric context → spatial attention adaptively fuses them; stacking all three yields a 14% F1 improvement.
- **Effectiveness of the COFE dataset**: Adding COFE improves F1 from 0.450 to 0.515 (+14.4%), as its diverse in-the-wild appearances and varied foot interactions effectively complement 3D mocap data.
- **Comparison with other style generalization techniques**: Shoe style–content randomization (0.577) substantially outperforms BIN (0.396), MixStyle (0.448), SagNets (0.511), and LatentDR (0.542).

## Highlights & Insights

- First dedicated dense foot contact estimation framework, filling a gap in the field
- The shoe style–content dual-branch randomization design is elegant, leveraging an external shoe dataset for style disentanglement in a manner transferable to other tasks with appearance bias
- Ground-aware learning introduces two complementary geometric signals (pixel height maps and ground normals), and the use of foot masks to suppress shortcut learning is a noteworthy implementation detail
- The COFE dataset (31K+ annotated samples) is constructed and publicly released, providing a standardized in-the-wild foot contact evaluation benchmark for the community
- Single-image inference requires no temporal information, yet surpasses video-based methods on joint-level evaluation

## Limitations & Future Work

- Dense contact estimation relies on the SMPL-X foot mesh topology (265 vertices); generalization to non-humanoid foot shapes or extreme footwear types remains unknown
- The vast majority of training data originates from controlled 3D mocap environments; although COFE adds in-the-wild samples, its scale remains limited (31K)
- Ground-truth generation for pixel height maps and ground normals depends on existing depth/geometry estimation tools, potentially introducing cumulative bias
- Quantitative evaluation is conducted only on MMVP and COFE, lacking validation in more diverse real-world scenarios (outdoor/complex terrain)
- The ViT-Huge backbone entails substantial computational cost, raising questions about feasibility for real-time applications

## Related Work & Insights

- **Joint-level contact**: Footskate Reducer (zero-velocity constraints) → HuMoR/PIP/WHAM (learning joint contact for motion) → Foot Stabilization (SMPL distance thresholds)
- **Dense contact**: POSA (cVAE conditional generation) → BSTRO (Transformer with visual input) → DECO (in-the-wild annotation) → HACO (hand-specific; decoder design adopted in this work)
- **Style generalization**: BIN (BN/IN gating) → SagNets (content/style dual-network adversarial training) → RandConv (random convolutions) → this paper's shoe-specific style–content randomization
- **Ground representation**: Pixel Height (shadow generation) → PixHt-Lab/ORG (3D reconstruction) → this paper extends pixel height maps to contact estimation

## Rating

- Novelty: ⭐⭐⭐⭐ — First dedicated dense foot contact estimation framework; the combination of shoe style–content randomization and ground-aware learning is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Training on 10 datasets, detailed ablations (five groups), multi-method and multi-granularity comparisons; however, test scenarios are somewhat controlled
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, well-coordinated figures and tables
- Value: ⭐⭐⭐⭐ — Opens a new direction in dense foot contact estimation; open-source dataset and code; applicable to motion capture, VR, and robotic gait analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/others/learning_dense_hand_contact_estimation_from_imbalanced_data.md)
- [\[CVPR 2026\] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries](nailia_multimodal_nail_design_retrieval_based_on_dense_intent_descriptions_and_p.md)
- [\[AAAI 2026\] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)
- [\[ICCV 2025\] Magic Insert: Style-Aware Drag-and-Drop](../../ICCV2025/others/magic_insert_style-aware_drag-and-drop.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)

</div>

<!-- RELATED:END -->
