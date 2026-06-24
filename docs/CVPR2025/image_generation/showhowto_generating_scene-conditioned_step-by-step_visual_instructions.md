---
title: >-
  [Paper Note] ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions
description: >-
  [CVPR 2025][Image Generation][Visual Instruction Generation] This paper proposes ShowHowTo, a video diffusion model capable of generating a sequence of step-by-step visual instructions consistent with a user-provided initial scene image and textual instructions. It also constructs a large-scale instructional dataset containing 578k sequences, collected from online instructional videos via a fully automated pipeline.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Visual Instruction Generation"
  - "Video Diffusion Models"
  - "Instructional Videos"
  - "Scene Consistency"
  - "Step-by-Step Generation"
date: 2026-05-08
content_hash: 36516ae9187bb544
---

# ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions

**Conference**: CVPR 2025  
**arXiv**: [2412.01987](https://arxiv.org/abs/2412.01987)  
**Code**: [https://soczech.github.io/showhowto/](https://soczech.github.io/showhowto/)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Visual Instruction Generation, Video Diffusion Models, Instructional Videos, Scene Consistency, Step-by-Step Generation  

## TL;DR

This paper proposes ShowHowTo, a video diffusion model capable of generating a sequence of step-by-step visual instructions consistent with a user-provided initial scene image and textual instructions. It also constructs a large-scale instructional dataset containing 578k sequences, collected from online instructional videos via a fully automated pipeline.

## Background & Motivation

**Background**: Current Large Language Models (LLMs) can reliably generate personalized step-by-step textual instructions, but translating text instructions into visual instructions remains highly challenging. Video generation models focus on short video clip generation, while image generation models produce only a single image at a time, neither of which can directly generate long-horizon multi-step visual instruction sequences.

**Limitations of Prior Work**: (1) Existing methods can only generate single-step visual instructions or are not linked to the user's scene—the generated images depict arbitrary environments rather than the user's actual surroundings. (2) Iterative generation methods (using the output of the previous step as input for the next step) suffer from accumulated errors and drift. (3) There is a lack of large-scale dataset for training step-by-step visual instructions.

**Key Challenge**: To simultaneously satisfy three requirements: (a) each step's image is faithful to the textual instruction, (b) the entire sequence remains consistent with the input scene, and (c) temporal coherence is maintained across steps—which existing methods fail to balance.

**Goal**: (1) Establish a scalable pipeline for collecting step-by-step visual instruction datasets. (2) Train a diffusion model that can generate the entire sequence in a single execution, conditioned on both the scene image and the textual instructions for each step.

**Key Insight**: Leverage the natural alignment between narrations and visual demonstrations in instructional videos to automatically extract high-quality sequences of image-text pairs as training data.

**Core Idea**: Formulate step-by-step visual instruction generation as a conditional video diffusion problem, where each frame receives independent textual conditioning while scene consistency is maintained through cross-frame attention.

## Method

### Overall Architecture

ShowHowTo is based on a latent Video Diffusion Model (SVD). The inputs are a user-provided scene image $I_0$ and $n$ step-by-step textual instructions $\{\tau_i\}_{i=0}^{n}$, and the outputs are $n$ corresponding instruction images. The scene image is encoded by a VAE and concatenated with the noise of each frame. During the denoising process, the spatial and temporal attention layers of the U-Net interact across frames to ensure consistency, while the cross-attention layer allows each frame to independently focus on its corresponding textual instruction.

### Key Designs

1. **Automated Dataset Construction Pipeline**:

    - **Function**: Automatically extract high-quality sequences of step-by-step image-text pairs from millions of online instructional videos.
    - **Mechanism**: A four-stage pipeline: (1) Use WhisperX to transcribe video narrations with high accuracy; (2) Use LLaMA-3.1 to filter out non-instructional videos (e.g., product reviews, vlogs); (3) Use an LLM to extract structured step-by-step instructions and their corresponding time intervals from the transcripts; (4) Use DFN-CLIP within each time interval to perform cross-modal alignment and select the best representative frame, while ensuring chronological order consistency. Ultimately, 578k sequences and 4.5M image-text pairs covering 25,026 types of "HowTo" tasks are acquired from millions of videos.
    - **Design Motivation**: Manual annotation is expensive and not scalable, and existing datasets (e.g., WikiHow hand-drawn illustrations, HowToStep coarse-grained segments) suffer from insufficient quality. A fully automated pipeline is key to scalability.

2. **Per-Frame Independent Text Conditioning**:

    - **Function**: Ensure each frame's image precisely matches its corresponding textual instruction.
    - **Mechanism**: Unlike standard video diffusion models that use a single global text prompt, ShowHowTo independently injects the corresponding textual instruction $\tau_i$ for each frame $i$ in the sequence via the cross-attention layers of the U-Net. Conditioning each frame independently allows the model to handle actions with large semantic shifts, such as "chopping vegetables" and "searing meat", within a single generation step.
    - **Design Motivation**: Ablation studies show that using a single prompt (either concatenated or summarized) yields a Step Faithfulness of only 0.20-0.21, whereas per-frame independent prompting achieves 0.52, showing a massive difference. The semantic variation between adjacent steps in a visual instruction sequence is much larger than the frame-to-frame variation in ordinary videos.

3. **Variable-Length Sequence Training**:

    - **Function**: Support generating instruction sequences of arbitrary lengths (ranging from 1 to 15 steps).
    - **Mechanism**: During training, different sequence lengths are used across batches (kept consistent within each batch for computational efficiency). If a dataset sequence is longer than the target length, a starting frame is randomly selected and $k$ consecutive frames are extracted. The maximum training length is set to 8 frames, balancing scene consistency and step accuracy.
    - **Design Motivation**: Ablational studies indicate that training with short sequences ($\le 4$ frames) improves scene consistency but degrades step accuracy, whereas training with long sequences ($\le 16$ frames) does the opposite. A maximum length of 8 frames achieves the best balance. Continuous sampling performs better than random sampling because it maintains continuous scene progression.

### Loss & Training

Based on the standard denoising diffusion loss of SVD. The entire U-Net is fine-tuned on the self-built dataset, initialized from a WebVid10M pre-trained checkpoint. The scene image $I_0$ is encoded by the VAE and concatenated with the noise along the channel dimension as input, while also being globally conditioned via independent cross-attention layers.

## Key Experimental Results

### Main Results

| Method | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| InstructPix2Pix | 0.25 | 0.17 | 0.25 |
| GenHowTo | 0.49 | 0.13 | 0.27 |
| StackedDiffusion | 0.43 | 0.02 | 0.42 |
| **ShowHowTo** | **0.52** | **0.34** | **0.42** |
| Original Video Sequence | 0.50 | 1.00 | 0.56 |

ShowHowTo exceeds the original video sequence in Step Faithfulness (0.52 vs 0.50), while significantly outperforming sequence generation methods in Scene Consistency (StackedDiffusion achieves only 0.02).

### Ablation Study

| Text Conditioning Type | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| Single Prompt (Concatenated) | 0.21 | 0.29 | 0.38 |
| Single Prompt (Summarized) | 0.20 | 0.30 | 0.40 |
| **Per-Frame Independent Prompting** | **0.52** | **0.34** | **0.42** |

| Training Data | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| WikiHow-VGSI | 0.55 | 0.12 | 0.30 |
| HowToStep | 0.39 | 0.33 | 0.29 |
| **ShowHowTo Dataset** | **0.52** | **0.34** | **0.42** |

### Key Findings

- Per-frame independent text conditioning is the most critical design for performance, boosting Step Faithfulness by over 150%.
- The ShowHowTo dataset significantly outperforms WikiHow and HowToStep, primarily due to more precise frame selection and cleaner instruction extraction.
- In the human evaluation, ShowHowTo outperforms the original video sequences in 42% of cases in terms of steps and scene dimensions.
- ShowHowTo demonstrates robust generalization capability in zero-shot tests on WikiHow.

## Highlights & Insights

1. **The dataset construction pipeline is the most significant contribution**: The fully automated, scalable, and high-quality dataset of 578k sequences lays the foundation for this research direction.
2. **Insight into per-frame independent conditioning**: Visual instruction sequences are fundamentally different from ordinary videos; they exhibit large semantic jumps between frames, requiring independent guidance.
3. **Generation quality exceeds real videos**: It outperforms original source videos in step faithfulness, as actions in real videos may be occluded or off-camera.
4. **Broad application value**: It benefits not only human users but can also generate intermediate goal images for robot policy learning.

## Limitations & Future Work

- The model struggles to maintain object state consistency across long sequences (e.g., already cooked -> reverting to raw).
- It may generate physically implausible configurations for rare objects (e.g., electronic components).
- It inherits the inherent limitations of the underlying video diffusion model (such as occasional blurriness or artifacts).
- Future directions: Introduce object state tracking and extend to more domains (e.g., assembly, repair).

## Related Work & Insights

- Relationship with GenHowTo: GenHowTo also extracts data from instructional videos but generates iteratively using an image-to-image approach, which accumulates errors.
- Relationship with StackedDiffusion: StackedDiffusion is trained on WikiHow hand-drawn illustrations, resulting in extremely poor scene consistency (0.02).
- AURORA performs local editing, which preserves the scene but loses step information.
- Insight: Simultaneously satisfying precise conditioning and global consistency remains the core challenge of visual generation.

## Rating

- **Novelty**: 8/10 — The automated design of the dataset pipeline and the per-frame independent conditioning are both innovative.
- **Experimental Thoroughness**: 9/10 — Evaluation on multiple datasets, human studies, and detailed ablations.
- **Writing Quality**: 9/10 — Clear problem formulation and well-structured contributions.
- **Value**: 8/10 — Highly insightful for both visual instruction generation and robotic planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ACL 2025\] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](../../ACL2025/image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

</div>

<!-- RELATED:END -->
