---
title: >-
  [Paper Note] Exploring Timeline Control for Facial Motion Generation
description: >-
  [CVPR 2025][Human Understanding][Facial Motion Generation] This paper introduces timeline control for facial motion generation for the first time, where users specify precise frame intervals for various facial actions on a multi-track timeline. Frame-level facial motion annotation is achieved with minimal effort through TICC temporal clustering, and a base-branch diffusion model is designed to decouple facial regions while preserving natural coupling…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Facial Motion Generation"
  - "Timeline Control"
  - "Facial Motion Annotation"
  - "Diffusion Models"
  - "TICC Temporal Clustering"
date: 2026-05-08
content_hash: 4751bc061ed266dc
---

# Exploring Timeline Control for Facial Motion Generation

**Conference**: CVPR 2025  
**arXiv**: [2505.20861](https://arxiv.org/abs/2505.20861)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Facial Motion Generation, Timeline Control, Facial Motion Annotation, Diffusion Models, TICC Temporal Clustering

## TL;DR

This paper introduces timeline control for facial motion generation for the first time, where users specify precise frame intervals for various facial actions on a multi-track timeline. Frame-level facial motion annotation is achieved with minimal effort through TICC temporal clustering, and a base-branch diffusion model is designed to decouple facial regions while preserving natural coupling, generating natural and smooth facial motions precisely aligned with the timeline.

## Background & Motivation

**Background**: Generating realistic facial motions is widely demanded in digital humans and film production. Existing methods mainly use audio or text as control signals—audio-driven methods can only generate motions synchronized with audio, while text-driven methods can only provide coarse-grained temporal descriptions using temporal adverbs (e.g., "then").

**Limitations of Prior Work**: Fine-grained control frequently required by users—such as "raise eyebrows in frames 10-30 while smiling in frames 14-43"—cannot be achieved by existing control modalities. Audio signals bind the tempo, whereas text signals lack frame-specific precision. While rule-based methods (e.g., directly adjusting blendshape curves) enable precise timing control, the generated motions are unnatural and deviate from the real motion distribution.

**Key Challenge**: Achieving frame-level temporal control requires frame-level facial motion annotations, which are extremely expensive to acquire. Existing methods use ChatGPT to summarize temporal sequences or rely on thresholds to determine blendshape values, but the former cannot establish precise start and end frames, while the latter struggles with setting reliable thresholds for complex actions (e.g., eyebrow movements) and ignores relationships between multiple motion descriptors.

**Goal**: (1) How to effortlessly acquire frame-level facial motion interval annotations? (2) How to design a generative model that achieves precise timeline alignment while maintaining motion naturalness? (3) How to balance the coupling and decoupling between different facial regions?

**Key Insight**: Leveraging TICC (Toeplitz Inverse Covariance-based Clustering) to automatically segment continuous facial motion time series into discrete action intervals and cluster similar motion patterns. Human annotators only need to inspect a few samples from each cluster to determine the action type, significantly reducing annotation overhead.

**Core Idea**: Automatically annotate facial motion intervals through temporal clustering, and then employ a base-branch diffusion model to generate natural and temporally precise facial movements from the timeline.

## Method

### Overall Architecture

The system consists of two core components: (1) A frame-level facial motion annotation pipeline: extracting ARKit blendshape time series $\to$ TICC segmentation and clustering $\to$ manual verification of cluster labels $\to$ obtaining frame-level annotations; (2) A timeline-driven generative model: the base network encodes global motion coupling $\to$ the branch networks independently generate decoupled motions for each facial region $\to$ combination yields the complete facial motion $\to$ a diffusion renderer performs photorealistic rendering. It supports converting natural language into timelines via ChatGPT to achieve text control.

### Key Designs

1. **TICC-based Frame-level Facial Motion Annotation**:

    - **Function**: Achieving frame-level facial motion interval annotations with extremely low manual labor cost.
    - **Mechanism**: Extracting ARKit blendshape coefficients of each video as facial motion descriptors (eyebrows: browDown/browInnerUp/browOuterUp; eyes: eyeBlink/eyeSquint/eyeWide; mouth: mouthSmile/mouthStretch/mouthFrown). The time series of multiple videos are concatenated into a single long sequence, separated by "null sequences" (sequences of length 100 with a value of -1), and then fed into the TICC algorithm. TICC simultaneously performs two tasks: (a) segmenting the time series into several action pattern intervals with explicit start and end frames; (b) clustering intervals with similar patterns. Human annotators only need to inspect a few representative samples of each cluster to determine its action category.
    - **Design Motivation**: Avoiding the massive workload of frame-by-frame manual annotation. Compared with threshold-based methods, TICC automatically considers the relationships between multiple descriptors without requiring manual threshold setup for complex movements. Experiments show that the macro-F1 for eyebrows, eyes, and mouth annotations reaches 0.90, 0.91, and 0.87, respectively.

2. **Base-Branch Diffusion Generative Model**:

    - **Function**: Maintaining natural coupling of facial expressions while precisely aligning with the timeline.
    - **Mechanism**: The generative model is divided into a base network and multiple branch networks. The base network receives timelines and noisy motions of all facial regions, encoding global motion coupling into base features via a Transformer encoder. The branch networks are split into three branches: upper face (eyes + eyebrows + gaze), lower face (mouth + jaw), and pose & others. Each branch only receives the timeline of its corresponding region and the base features (with the exception of the pose branch, which receives timelines of all regions because head pose is coupled with all facial movements). Timelines guide the motion generation via cross-attention, where the initial timeline tokens are re-introduced at each layer to prevent the temporal information from being altered.
    - **Design Motivation**: Movements of different facial regions are naturally coupled (e.g., squinting and slightly lowering eyebrows when smiling), which is crucial for naturalness. However, fully coupled generation degrades accuracy (e.g., being forced to lower the eyebrows when generating a smile, which conflicts with a user-specified eyebrow-raise). The Base-Branch design allows the base network to learn global coupling, while the branch networks decouple regions, striking a balance between precision and naturalness.

3. **Persistent Timeline Token Injection + Classifier-Free Guidance**:

    - **Function**: Enhancing the precise alignment of motion with the timeline and improving generalization.
    - **Mechanism**: In each Transformer encoder layer of the base and branch networks, the initial timeline tokens are always used instead of the output tokens of the previous layer. During training, classifier-free guidance is applied, where the conditioning timeline of each facial region is dropped independently with a probability of 0.5, with an additional 0.1 probability of dropping all conditions, and a 0.1 probability of keeping all conditions. When a condition is dropped, the timeline value for that region is set to -1.
    - **Design Motivation**: Iteratively updating timeline tokens across layers would gradually dilute the temporal precision. Persistent injection ensures that every layer accesses the original precise temporal information. Classifier-free guidance enhances the model's robustness and generalization to partially missing conditions, with 0.5 identified as the optimal drop probability.

### Loss & Training

Using the standard diffusion denoising loss $\mathcal{L}_{denoise} = \mathbb{E}_{t,M_{(0)},C}[\|M_{(0)} - \mathcal{G}(M_{(t)}, t, C)\|^2]$, which directly predicts the original signal instead of noise. FaceVerse 3DMM coefficients are adopted as the motion representation. The dataset is RealTalk (692 real conversation videos, approx. 600k frames). The network consists of an 8-layer Transformer, where the base and branch networks share the architecture but have independent parameters. The optimal configuration is 6 layers for the base and 2 layers for the branches.

## Key Experimental Results

### Main Results

| Method | Var→ | FID$_{fm}$↓ | FID$_{\Delta fm}$↓ | SND↓ | TAS↑ |
|------|------|------------|----------------|------|------|
| w/o branch | 0.68 | 7.39 | 0.14 | 7.53 | 0.66 |
| w/o base | 0.64 | 12.4 | 0.18 | 12.58 | 0.81 |
| all decoup. | 0.41 | 28.4 | 0.23 | 28.63 | 0.69 |
| **Ours** | **0.70** | **4.54** | **0.09** | **4.63** | **0.84** |

TAS (Timeline Alignment Score) evaluates timeline alignment accuracy, and SND evaluates motion naturalness. The Var of GT is 0.73.

### Ablation Study

| Configuration | TAS↑ | SND↓ | Description |
|------|------|------|------|
| w/o time con. (w/o persistent injection) | 0.79 | 5.48 | Degraded temporal precision |
| branchL1 (only 1 branch layer) | 0.76 | 6.36 | Branch too shallow, insufficient decoupling |
| branchL2 (Ours) | **0.84** | **4.63** | Optimal balance |
| branchL4 (4 branch layers) | 0.83 | 6.76 | Base too shallow, insufficient coupling |
| drop 0 (w/o CFG) | 0.78 | 7.01 | Poor generalization |
| drop 0.5 (Ours) | **0.84** | **4.63** | Optimal |
| drop 0.7 | 0.68 | 4.21 | Conditioning signal too weak, low precision |

### Key Findings

- TICC annotation quality is high: eyebrow macro-F1 is 0.90, eye is 0.91, and mouth is 0.87. When using AU instead of blendshapes, F1 drops to 0.73, indicating that descriptor resolution is critical.
- The Base-Branch design is indispensable: removing the branch (base only) drops TAS to 0.66 (poor accuracy), and removing the base (branch only) increases SND to 12.58 (unnatural). Completely decoupling all regions into independent branches leads to the worst FID (28.4) and unnatural movements.
- Persistent timeline injection improves TAS from 0.79 to 0.84.
- User study: 89% of responses consider the generated motions to be accurately aligned with the timeline, and 86% consider them natural.
- Supports ChatGPT for text-to-timeline conversion, enabling natural language-controlled facial motion generation.

## Highlights & Insights

- **Timeline control is a new paradigm for facial motion generation**: Compared to audio and text control, timelines offer frame-level temporal control capabilities. This granularity of control is highly practical in film production and animation.
- **TICC for facial motion annotation is highly ingenious**: Utilizing a temporal clustering algorithm to simultaneously achieve segmentation and clustering converts the frame-level annotation problem into a lightweight task of "inspecting representative samples from K clusters", significantly reducing annotation cost.
- **Base-Branch design precisely balances coupling and decoupling**: This architectural idea can be migrated to other multi-condition generation tasks that require partial decoupling, such as independent limb control and full-body coordination in human motion generation.

## Limitations & Future Work

- Annotations only cover limited action categories (3 for eyebrows, 4 for eyes, 4 for mouth, etc.), failing to describe finer-grained facial micro-expressions.
- The number of clusters and the beta parameter in TICC require manual tuning, with different optimal parameters for different facial regions.
- Training data originates from conversation scenarios, occasionally leading to uncontrolled speaking actions in the generated results.
- The quality and identity preservation of the renderer (diffusion rendering) limit the practical utility of the final video.
- Currently, only symmetric facial movements are processed (using only left blendshape coefficients); extending to asymmetric expressions (e.g., winking) is a promising future direction.

## Related Work & Insights

- **vs AgentAvatar/InstructAvatar**: These text-driven methods can only roughly describe action sequences using temporal adverbs, failing to achieve frame-level control. Timeline control fundamentally solves the issue of control granularity.
- **vs Human motion timeline control (TEACH, etc.)**: Human motion timeline control is achieved by generating segments first and then splicing them together. However, facial motion changes rapidly and frequently, so splicing strategies would produce unnatural transitions. This paper's base-branch diffusion model generates directly on the entire timeline at once, making it more suitable for facial scenarios.
- **vs Rule-based methods**: Rule-based methods can control timing precisely but produce unnatural movements. This method ensures naturalness while maintaining temporal precision by learning from the real data distribution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneering introduction of timeline control for facial motion, with an ingenious TICC annotation scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablation studies and comprehensive user studies, but lacks quantitative comparisons with other methods.
- Writing Quality: ⭐⭐⭐⭐ Generally clear, though the connection between the annotation and generation parts could be tighter.
- Value: ⭐⭐⭐⭐ The demand for fine-grained facial motion control is real and crucial, and the model holds strong potential for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[CVPR 2026\] PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation](../../CVPR2026/human_understanding/pc-talk_precise_facial_animation_control_for_audio-driven_talking_face_generatio.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)

</div>

<!-- RELATED:END -->
