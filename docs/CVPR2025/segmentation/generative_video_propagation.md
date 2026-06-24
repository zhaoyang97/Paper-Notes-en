---
title: >-
  [Paper Note] Generative Video Propagation
description: >-
  [CVPR 2025][Segmentation][Video Propagation] The GenProp framework is proposed, which cooperates a selective content encoder (SCE) with an I2V generative model to uniformly propagate first-frame edits to the entire video, simultaneously supporting multiple video tasks such as video editing, object removal, object insertion, and object tracking in a single model.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video Propagation"
  - "Image-to-Video Generation"
  - "Video Editing"
  - "Object Removal"
  - "Selective Content Encoding"
date: 2026-05-08
content_hash: 8c6aa4d6725bd015
---

# Generative Video Propagation

**Conference**: CVPR 2025  
**arXiv**: [2412.19761](https://arxiv.org/abs/2412.19761)  
**Code**: [https://genprop.github.io/](https://genprop.github.io/)  
**Area**: Image Segmentation  
**Keywords**: Video Propagation, Image-to-Video Generation, Video Editing, Object Removal, Selective Content Encoding

## TL;DR

The GenProp framework is proposed, which cooperates a selective content encoder (SCE) with an I2V generative model to uniformly propagate first-frame edits to the entire video, simultaneously supporting multiple video tasks such as video editing, object removal, object insertion, and object tracking in a single model.

## Background & Motivation

**Background**: Large-scale video generation models (e.g., SORA, SVD) have demonstrated powerful real-world modeling capabilities, but downstream video editing tasks remain fragmented—video inpainting, appearance editing, and object insertion each require tailored method designs. Traditional video propagation methods rely on intermediate representations like optical flow, depth maps, or radiation fields to propagate sparse-frame edits to other frames, which are prone to error accumulation and have limited generalization ability.

**Limitations of Prior Work**: Existing diffusion-based video editing methods (such as InsV2V, AnyV2V, etc.) primarily perform appearance-level modifications through text control, making them struggle to handle large deformations in object shapes. Object removal typically requires dense mask annotations for every frame (e.g., Propainter), which is highly user-unfriendly. Furthermore, many methods require video-specific LoRA fine-tuning, which is computationally expensive and prone to overfitting to the original video.

**Key Challenge**: How can a unified model accurately propagate various edits of the first frame (editing, removal, insertion) while maintaining consistency in the unmodified regions with the original content? The key challenge lies in making the content encoder "selectively" preserve unmodified regions while fully unleashing the generation capabilities of the I2V model on the modified regions.

**Goal**: To define the new problem of "Generative Video Propagation" and construct a general framework to propagate arbitrary edits from the first frame to the entire video while maintaining the consistency of other regions.

**Key Insight**: It is observed that many video tasks (editing, removal, insertion, tracking) can be modeled as a "first-frame modification + propagation" problem. The inherent real-world modeling capabilities of I2V generative models are leveraged to propagate edits, while a meticulously designed encoder separates the modified and unmodified regions.

**Core Idea**: Encode the unchanged parts of the original video with a selective content encoder and propagate the modified parts of the first frame using an I2V generative model, achieving effective decoupling between the two via a region-aware loss and a mask prediction decoder head.

## Method

### Overall Architecture

The input consists of the original video $V$ and the edited first frame $v'_1$, and the output is the complete video $V'$ propagated with the first-frame edits. The architecture contains two core components: a Selective Content Encoder (SCE) that encodes the unchanged parts of the original video, and an I2V generative model that propagates and generates conditioned on the edited first frame. Pairwise data synthesized from video instance segmentation datasets is used during training; synthesized data is only fed into the SCE, while the original video is fed into the I2V model, preventing the learning of synthesis artifacts.

### Key Designs

1. **Selective Content Encoder (SCE)**:

    - **Function**: Encodes information from unmodified regions in the original video to guide the I2V model in keeping these regions unchanged.
    - **Mechanism**: Duplicates the first N blocks of the I2V model as the encoder, similar to the ControlNet architecture. Features after each encoder block are added to the corresponding layers of the I2V model via a zero-initialized MLP injection layer. The key design is the bidirectional information exchange—features from the I2V model are also fed back into the input of the SCE, allowing the SCE to perceive which regions have been modified, thereby "selectively" encoding only the unmodified regions.
    - **Design Motivation**: If the SCE indiscriminately encodes all regions, it will also transmit the original content of the modified regions to the I2V model, suppressing its generation capability and causing the original objects to "reappear". The bidirectional information exchange informs the SCE of the modification scope, enabling selective encoding.

2. **Region-Aware Loss (RA Loss)**:

    - **Function**: Effectively decouples training signals between modified and unmodified regions.
    - **Mechanism**: Splits the loss into three parts—$\mathcal{L}_{mask}$ supervises the generation quality of modified regions, $\mathcal{L}_{non-mask}$ supervises the preservation quality of unmodified regions, and $\mathcal{L}_{grad}$ minimizes the gradient response of the SCE in modified regions via finite difference approximation. The total loss is $\mathcal{L} = \mathcal{L}_{non-mask} + \lambda \mathcal{L}_{mask} + \beta \mathcal{L}_{grad} + \gamma \mathcal{L}_{MPD}$, where $\lambda=2.0$, $\beta=1.0$, $\gamma=1.0$.
    - **Design Motivation**: When the modified region is very small, a standard global loss causes the SCE to ignore the modified region and directly reconstruct the original content. Separating the losses of the two regions ensures that even small-area edits receive sufficient supervision. The gradient loss further penalizes SCE responses in modified regions, reinforcing selective encoding.

3. **Mask Prediction Decoder (MPD) + Synthetic Data Strategy**:

    - **Function**: Assists the model in identifying spatial regions that require edit propagation, and covers multiple video tasks through synthetic data.
    - **Mechanism**: The MPD mirrors the last block of the I2V model, adding an MLP to output frame-by-frame modification masks under MSE loss supervision. Training data is generated via three synthetic augmentations: Copy-and-Paste to simulate insertion, Mask-and-Fill to simulate editing/removal, and Color Fill to simulate tracking. Each approximation corresponds to a task embedding injected into the model.
    - **Design Motivation**: Without the MPD, attention maps often degenerate, making the model uncertain about which region to modify, which leads to incomplete removal (the removed object reappearing in subsequent frames). Although the Color Fill augmentation is simple, it explicitly trains the model to maintain first-frame modifications consistently across the entire sequence, which is crucial for propagation under large shape changes.

### Loss & Training

The total loss is a region-aware weighted combination. SCE and MPD are trainable, while the I2V model is frozen. Experiments are conducted on both DiT and U-Net (SVD) architectures. The CFG scale is set to 20, and the data augmentation ratio is Copy-and-Paste/Mask-and-Fill/Color Fill = 0.5/0.375/0.125.

## Key Experimental Results

### Main Results

| Method | PSNR_m ↑ (Classic) | CLIP-T ↑ | CLIP-I ↑ | GenProp Preference % (Align/Quality) |
|------|-------------------|----------|----------|-------------------------------|
| **GenProp** | **33.837** | **0.3229** | 0.9825 | - |
| InsV2V | 28.999 | 0.3049 | 0.9737 | 60/60 |
| AnyV2V | 32.090 | 0.3050 | 0.9676 | 95.56/86.67 |
| Pika | 32.568 | 0.3226 | **0.9923** | 62.22/55.56 |
| ReVideo | 31.765 | 0.3196 | 0.9777 | 75.56/71.11 |

The advantage is more pronounced on the Challenging Test Set (PSNR_m 32.163 vs. the best baseline of 31.329), with user preference rates as high as 82-98%.

### Ablation Study

| Configuration | CLIP-T ↑ | CLIP-I ↑ | Description |
|------|---------|---------|------|
| Full model | 0.3316 | 0.9872 | Full model |
| w/o MPD | 0.3252 | 0.9834 | Remove mask prediction head, degradation in modified region recognition |
| w/o RA Loss | 0.3261 | 0.9825 | Remove region-aware loss, original objects may reappear |
| w/o Color Fill | - | - | Failure in propagation under large shape changes |

### Key Findings
- The effect of MPD is most prominent in object removal tasks: without the MPD, the removed object partially reappears in subsequent frames.
- RA Loss resolves the "over-encoding" issue of the SCE: without it, original objects gradually leak back into the edited regions.
- Color Fill augmentation is critical for large shape changes: extreme changes, such as turning a girl into a kitten, can only be successfully propagated with Color Fill.
- During object removal, GenProp automatically removes associated effects such as shadows and reflections, which is impossible for traditional mask-based methods (like SAM + Propainter).
- The quality of video generation from the DiT architecture outperforms that of the SVD architecture.

## Highlights & Insights
- **Unifying multiple video tasks into the abstraction of "first-frame edit propagation"**: This problem definition is highly elegant, processing editing, removal, insertion, and tracking in a single model, thereby avoiding the redundancy of designing separate pipelines for each task.
- **Ingenious design of the SCE bidirectional information exchange**: Letting the encoder "know" what has been modified to enable selective encoding; this feedback mechanism is simple yet effective and can be extended to any scenario requiring selective injection from a conditional encoder.
- **No need for dense mask annotations**: Traditional removal methods require frame-by-frame mask annotations, whereas GenProp only requires editing the first frame to automatically propagate, significantly simplifying user workflows.
- **Leveraging video generation pre-training to acquire physical rule understanding**: GenProp is capable of tracking object reflections and shadows, a capability stemming from physical commonsense learned during the pre-training of I2V models.

## Limitations & Future Work
- Currently supports only first-frame edit propagation and cannot handle propagation originating from middle frames of a video.
- Video length is limited by the frame count limit of the I2V model (32/64/128 frames), requiring long videos to be processed in segments.
- Tracking precision is inferior to specialized tracking models (such as SAM-V2), and inference speed is relatively slow.
- Synthetic training data may not cover all real editing scenarios, potentially leading to failures under extreme circumstances.
- Future directions could consider expanding to multi-keyframe editing and a wider range of downstream tasks.

## Related Work & Insights
- **vs AnyV2V**: Both are first-frame edit propagation methods, but AnyV2V is a training-free solution with limited generalization ability, whereas GenProp achieves stronger preservation capabilities and propagation quality through dedicated training of the SCE.
- **vs ReVideo**: ReVideo uses black box masks and motion trajectory control based on SVD, suffering from significant information loss and blurry boundaries, whereas GenProp implicitly preserves unmodified regions via the SCE without requiring explicit masking.
- **vs SAM+Propainter**: Traditional cascaded schemes require dense masks and fail to remove shadows/reflections, whereas GenProp achieves this in a single step leveraging generative priors.

## Rating
- Novelty: ⭐⭐⭐⭐ The definition uniting video tasks as a propagation problem is novel, and the design of SCE + RA Loss is effective, though the component designs are relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three sub-tasks, multiple test sets, user studies, and thorough ablations; the experiments are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, though the discussion of certain design choices could be more in-depth.
- Value: ⭐⭐⭐⭐⭐ A unified framework processing multiple video tasks, highly practical, with a direct impact on industry workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] Using Diffusion Priors for Video Amodal Segmentation](using_diffusion_priors_for_video_amodal_segmentation.md)
- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)

</div>

<!-- RELATED:END -->
