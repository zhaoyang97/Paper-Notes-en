---
title: >-
  [Paper Note] EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance
description: >-
  [CVPR 2025][Medical Imaging][Echocardiography] This paper proposes EchoWorld, a motion-aware world modeling framework for echocardiography probe guidance. It first undergoes pre-training via spatial world modeling (masked reconstruction) and motion world modeling (predicting visual changes based on probe motion) to encode cardiac anatomical knowledge. In the fine-tuning stage, a motion-aware attention mechanism is introduced to fuse historical visual-motion sequences…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Echocardiography"
  - "probe guidance"
  - "world models"
  - "motion-aware attention"
  - "representation learning"
date: 2026-05-08
content_hash: a55335eeddf66e39
---

# EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance

**Conference**: CVPR 2025  
**arXiv**: [2504.13065](https://arxiv.org/abs/2504.13065)  
**Code**: [https://github.com/LeapLabTHU/EchoWorld](https://github.com/LeapLabTHU/EchoWorld)  
**Area**: Medical Image  
**Keywords**: Echocardiography, probe guidance, world models, motion-aware attention, representation learning

## TL;DR

This paper proposes EchoWorld, a motion-aware world modeling framework for echocardiography probe guidance. It first undergoes pre-training via spatial world modeling (masked reconstruction) and motion world modeling (predicting visual changes based on probe motion) to encode cardiac anatomical knowledge. In the fine-tuning stage, a motion-aware attention mechanism is introduced to fuse historical visual-motion sequences, significantly reducing guidance errors across 10 standard views.

## Background & Motivation

**Background**: Echocardiography is a non-invasive, low-cost, and widely available cardiac diagnostic tool. However, its operation is highly dependent on operator experience, requiring precise control of the probe's position, angle, and pressure to acquire standard view images. The global shortage of qualified sonographers limits the accessibility of ultrasound services. Probe guidance systems aim to predict the required probe motion vectors from the current frame to the target standard view, providing support for assisted scanning or fully automated robotic scanning.

**Limitations of Prior Work**: Ultrasound probe guidance is inherently a visual-motion sequence prediction problem. The model must not only understand complex cardiac anatomy (chambers, valves, blood vessels) but also how probe motion causes changes in visual signals. Existing methods (such as imitation-learning-based US-GuideNet) mainly employ general vision backbones (e.g., ResNet, DINOv2), which lack specialized encoding of ultrasound domain knowledge and lack effective mechanisms to integrate motion information with visual features. Standard interleaved visual-action sequence encoding fails to fully exploit the motion relationships among all frames.

**Key Challenge**: Probe guidance requires two simultaneous capabilities: understanding "what is in the image" (anatomical knowledge) and understanding "what will be seen when the probe moves" (motion-visual dynamics). Existing methods lack a framework to systematically learn both types of knowledge. Meanwhile, the standard attention mechanism models inter-frame motion relationships implicitly, failing to explicitly utilize the rich probe pose data.

**Goal**: (1) How to design a pre-training strategy to systematically encode ultrasound cardiac anatomy and motion dynamics? (2) How to effectively integrate historical visual-motion information during guidance inference to improve prediction accuracy?

**Key Insight**: The authors draw an analogy to human sonographers possessing a mental model of an "internal cardiac map"—knowing the location of various cardiac structures and what will be seen after moving the probe. Inspired by the concept of world models, they design two pre-training tasks: spatial world modeling (learning anatomical structure) and motion world modeling (learning the relationship between motion and visual changes). For downstream guidance, they introduce motion-aware attention to encode 3D relative poses between frames into keys/values, replacing standard position encodings.

**Core Idea**: Use dual spatial and motion world model pre-training to encode ultrasound domain knowledge, and use motion-aware attention to explicitly integrate inter-frame pose relationships to enhance probe guidance accuracy.

## Method

### Overall Architecture

EchoWorld adopts a two-stage design. **Pre-training stage**: Based on the JEPA (Joint-Embedding Predictive Architecture) framework, spatial world modeling (MAE-style masked region reconstruction) and motion world modeling (predicting visual changes given probe motion) are performed simultaneously. **Fine-tuning stage**: Freeze/fine-tune the pre-trained visual and motion encoders, and add a motion-aware attention module to process historical visual-motion sequences $\{(I_{t_i}, p_{t_i})\}_{i=1}^N$ to output predicted motion vectors pointing to each target standard view.

### Key Designs

1. **Spatial World Modeling**:

    - **Function**: Encode spatial knowledge of cardiac anatomy, such as chamber positions, valve morphology, and tissue textures.
    - **Mechanism**: A masked reconstruction task based on JEPA. A ViT serves as the context encoder, processing only visible patches, while the target encoder (an EMA version of the context encoder) processes the full image. The predictor predicts the feature representation of the masked regions in the feature space. The loss function is the L1 feature reconstruction error of the masked positions: $\mathcal{L}_{spatial} = \sum_{c \in M} \|g_\phi(f_\theta(x); M)_c - f'_{\theta'}(y)_c\|_1$.
    - **Design Motivation**: By reconstructing occluded anatomical regions, the model must learn the spatial relationships and visual features among different cardiac structures, which is exactly the anatomical knowledge sonographers need. Unlike pixel space reconstruction (MAE), feature space reconstruction focuses more on semantic-level understanding.

2. **Motion World Modeling**:

    - **Function**: Encode the causal relationship between probe motion and visual changes.
    - **Mechanism**: Given two frames $I_a, I_b$ from the same scan and the relative pose $p_{a\to b}$ between them, the motion encoder $A_\psi$ encodes the pose difference into motion features $z_{a\to b}$. The predictor predicts the average-pooled features of $I_b$, conditioned on the context features of $I_a$ and the motion features. An InfoNCE contrastive loss is used instead of an L1 reconstruction loss to make the model more sensitive to the direction and magnitude of the probe motion. The formula is: $\mathcal{L}_{motion} = -\frac{1}{B}\sum_i \log \frac{\exp(\hat{h}_{y_i}^\top h_{y_i}/\tau)}{\sum_j \exp(\hat{h}_{y_i}^\top h_{y_j}/\tau)}$.
    - **Design Motivation**: A core skill of a sonographer is knowing "how the image will change if I rotate the probe left by 10 degrees". Contrastive learning allows the model to precisely differentiate distinct visual outcomes resulting from different motions. Joint pre-training of both tasks ($\mathcal{L}_{total} = \mathcal{L}_{spatial} + 0.1 \cdot \mathcal{L}_{motion}$) forms a unified representation that understands both anatomy and motion.

3. **Motion-Aware Attention**:

    - **Function**: Explicitly integrate 3D relative pose information between frames into feature interactions during guidance inference.
    - **Mechanism**: In standard attention, Keys and Values are linear projections of frame features, independent of inter-frame relationships. Motion-aware attention redefines Keys and Values as joint functions of frame features and relative poses: $K_j^{(i)} = \text{MLP}(h_j, z_{i\to j})$, $V_j^{(i)} = \text{MLP}(h_j, z_{i\to j})$, where $z_{i\to j}$ is the relative motion encoding between frame $i$ and frame $j$. This ensures that the interaction of each query token $i$ with another frame $j$ contains information about their spatial relationship.
    - **Design Motivation**: Standard attention is permutation-invariant and does not differentiate which frame the information is coming from. In probe guidance, however, a frame at 5cm distance from the current position and a frame at 1cm distance have entirely different reference values. Embedding the pose difference into K/V allows the model to weigh and aggregate historical information according to spatial relationships.

### Loss & Training

Pre-training utilizes over 1 million ultrasound images from 200+ routine scans, jointly optimizing the spatial and motion world modeling losses. The fine-tuning stage uses an L1 guidance loss $\mathcal{L}_{guide} = \|a_t - \hat{a}_t\|_1$. The probe pose is represented with 6 degrees of freedom (3 translation + 3 rotation). The target encoder is updated via an EMA of the context encoder.

## Key Experimental Results

### Main Results (Probe guidance error across 10 standard views)

| Method | Pre-training | Mean Translation Error (mm) | Mean Rotation Error (°) |
|------|--------|-----------------|-----------------|
| Scratch | None | ~8.86 | ~9.07 |
| DINOv2 | ImageNet | ~8.62 | ~8.52 |
| EchoCLIP | Ultrasound | ~8.35 | ~8.37 |
| USFM | Ultrasound | ~8.39 | ~8.42 |
| US-MAE | Ultrasound | ~8.45 | ~8.46 |
| **EchoWorld** | **Ultrasound (World Modeling)** | **~7.85** | **~7.95** |

EchoWorld achieves the lowest average guidance error across all 10 standard views, reducing the translation error by approximately 0.5 mm and the rotation error by approximately 0.4° compared to the best ultrasound pre-training method, EchoCLIP.

### Ablation Study

| Configuration | Average Error | Description |
|------|---------|------|
| Spatial World Modeling Only | Moderate | Lacks motion knowledge |
| Motion World Modeling Only | Moderate | Lacks anatomical knowledge |
| Spatial + Motion (Full Pre-training) | Optimal | Complementary knowledge types |
| Standard Attention (No Motion-Awareness) | Worse | Unable to explicitly utilize pose relationships |
| Motion-Aware Attention | Optimal | Frame relative pose embedding in K/V is effective |

### Key Findings
- World model pre-training significantly outperforms all general pre-training (ImageNet) and ultrasound-specific pre-training (USFM, EchoCLIP, etc.), proving that domain-specific world modeling is more effective than general self-supervised learning.
- Joint pre-training of spatial and motion modeling performs better than either alone, demonstrating that the two types of knowledge are complementary.
- Motion-aware attention shows a clear advantage over standard attention, validating the effectiveness of embedding inter-frame pose differences in K/V.
- From scratch to DINOv2 and then to EchoWorld, the improvement in pre-training quality directly translates to an increase in guidance accuracy.
- Qualitative analysis indicates that the pre-trained predictor of EchoWorld can serve as an ultrasound simulator: given a frame and a probe motion, it can reasonably predict what will be seen after movement (when paired with a diffusion model to map onto pixel space).

## Highlights & Insights
- **Applying the world model perspective to medical imaging** is highly inspiring—drawing an analogy from ultrasound scanning to an agent navigating a world, treating anatomical knowledge as a "world map", and treating motion prediction as anticipating the effects of movement, makes for a highly natural conceptual transfer.
- The design of **motion-aware attention** embeds 3D pose differences into the attention K/V, providing a general spatiotemporal relationship modeling approach that can be transferred to any visual sequence task involving camera/sensor pose changes (e.g., robotic navigation, autonomous driving).
- Contrastive learning for motion world modeling is more effective than regression loss, as contrastive loss is better at distinguishing subtle motion differences.

## Limitations & Future Work
- The dataset is derived from robotic-assisted scanning in a single center, where the probe is controlled by a robotic arm, which may exhibit different motion patterns compared to actual hand-held scanning.
- The computational complexity of calculating inter-frame K/V in motion-aware attention is $O(N^2)$ (where N is sequence length), which may require more efficient approximations for long sequences.
- Validation is currently restricted to 10 standard views; non-standard views or pathological deformities are not yet addressed.
- Pre-training requires a massive amount of ultrasound video data annotated with time-synchronized poses, the acquisition of which is highly expensive.

## Related Work & Insights
- **vs US-GuideNet**: US-GuideNet provides rotational guidance but uses a general vision backbone; EchoWorld achieves more precise guidance through domain-specific world model pre-training.
- **vs USFM/EchoCLIP**: These ultrasound foundation models employ general self-supervised/multimodal learning but do not model visual-motion dynamics; EchoWorld's motion world modeling fills this gap.
- **vs JEPA/I-JEPA**: EchoWorld is based on the JEPA framework but introduces motion-conditional prediction, extending it from passive observation to an active interaction world model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The world model perspective is a brand-new attempt in the field of ultrasound guidance, and the motion-aware attention design is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 standard views, large-scale data, extensive baseline comparisons, and comprehensive qualitative visualization analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow, highly natural analogy from world models to ultrasound guidance, and excellent illustrations.
- Value: ⭐⭐⭐⭐⭐ Directly advances the automation of ultrasound, and the framework can be generalized to other image-guided tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2025\] EchoONE: Segmenting Multiple Echocardiography Planes in One Model](echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[CVPR 2026\] OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement](../../CVPR2026/medical_imaging/osa_echocardiography_video_segmentation_via_orthogonalized_state_update_and_anat.md)
- [\[CVPR 2026\] SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking](../../CVPR2026/medical_imaging/simspine_a_biomechanics-aware_simulation_framework_for_3d_spine_motion_annotatio.md)

</div>

<!-- RELATED:END -->
