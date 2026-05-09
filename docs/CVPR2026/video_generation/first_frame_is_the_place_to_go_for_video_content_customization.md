---
title: >-
  [Paper Note] First Frame Is the Place to Go for Video Content Customization
description: >-
  [CVPR 2026][Video Generation][Video content customization] This paper identifies an intrinsic capability of video generation models to implicitly use the first frame as a "conceptual memory buffer" for storing and reusing multiple visual entities. Building on this observation, the authors propose FFGo—a lightweight LoRA adaptation method requiring only 20–50 training samples—that activates this capability without any architectural modification, enabling multi-reference video content customization. FFGo is rated best in 81.2% of cases in user studies.
tags:
  - CVPR 2026
  - Video Generation
  - Video content customization
  - first-frame conceptual buffer
  - multi-reference video generation
  - LoRA fine-tuning
  - vision-language model
date: 2026-05-08
content_hash: a2529b901294b1a0
---

# First Frame Is the Place to Go for Video Content Customization

**Conference**: CVPR 2026
**arXiv**: [2511.15700](https://arxiv.org/abs/2511.15700)
**Code**: [http://firstframego.github.io](http://firstframego.github.io)
**Area**: Video Understanding / Video Generation
**Keywords**: Video content customization, first-frame conceptual buffer, multi-reference video generation, LoRA fine-tuning, vision-language model

## TL;DR

This paper identifies an intrinsic capability of video generation models to implicitly use the first frame as a "conceptual memory buffer" for storing and reusing multiple visual entities. Building on this observation, the authors propose FFGo—a lightweight LoRA adaptation method requiring only 20–50 training samples—that activates this capability without any architectural modification, enabling multi-reference video content customization. FFGo is rated best in 81.2% of cases in user studies.

## Background & Motivation

1. **Background**: Video generation models (Wan2.2, Stable Video Diffusion, etc.) can already produce high-quality videos. Multi-reference video generation—composing objects and scenes from multiple reference images into a single video—is a key application direction. Representative methods such as VACE and SkyReels-A2 achieve this through architectural modifications and training on millions of samples.
2. **Limitations of Prior Work**: (a) Architecture-modification approaches alter the structure of pretrained models, compromising compatibility and efficiency; (b) large-scale task-specific fine-tuning causes models to overfit to particular scenarios (predominantly human–object interaction), losing the broad generative priors acquired during pretraining; (c) existing methods typically cap the number of references at three (person, object, scene) and cannot handle more.
3. **Key Challenge**: How can multi-reference video content customization be achieved without architectural modification and without relying on large-scale custom datasets?
4. **Goal**: (a) Understand the intrinsic capabilities of video generation models—can the first frame serve as a concept store? (b) How can this capability be reliably activated? (c) How can multi-scenario generalization be achieved while preserving pretrained knowledge?
5. **Key Insight**: The authors identify a previously overlooked phenomenon: pretrained image-to-video (I2V) models already possess the latent ability to extract visual concepts from a collage first frame and fuse them coherently in subsequent frames. However, this ability is difficult to trigger reliably via prompt engineering alone (instability, loss of object identity). A lightweight LoRA adaptation on a minimal number of training samples suffices to activate it reliably.
6. **Core Idea**: The first frame of a video model functions as a "conceptual buffer" rather than merely a spatiotemporal starting point. LoRA fine-tuning on 20–50 samples activates this intrinsic capability, enabling multi-reference video customization without architectural modification while preserving pretrained knowledge.

## Method

### Overall Architecture

FFGo comprises three components: (1) **Dataset curation**: a vision-language model (Gemini-2.5-Pro) and SAM 2 are used to extract elements and backgrounds from existing videos, producing high-quality composite-image–video paired training data; (2) **Few-shot LoRA adaptation**: LoRA is trained on the pretrained I2V model (Wan2.2-I2V-A14B) to learn scene transition and subject fusion from a composite first frame; (3) **Clean video generation at inference**: given a composite image and a text prompt containing a transition trigger phrase, the first $F_c$ compressed frames are discarded after generation, yielding a clean customized video.

### Key Designs

1. **First Frame as Conceptual Memory Buffer**

    - **Function**: Reveals and exploits an intrinsic capability of video generation models—treating the first frame as a spatial buffer that stores multiple visual concepts.
    - **Mechanism**: Standard I2V models already possess the ability to fuse collage elements in the first frame into coherent scenes in subsequent frames. Direct use, however, suffers from three problems: (a) transition prompts require laborious manual tuning and vary across models and videos; (b) scene transitions are unstable; (c) reference object identity is frequently lost. FFGo learns to trigger this capability reliably from a small number of training samples.
    - **Design Motivation**: This constitutes a fundamental insight into the intrinsic capabilities of pretrained models. Rather than modifying the architecture to introduce multi-reference input support, FFGo leverages existing model capabilities, thereby preserving pretrained knowledge and avoiding overfitting.

2. **VLM-Assisted Data Curation Pipeline**

    - **Function**: Automatically generates high-quality training triples (composite image + video + text prompt).
    - **Mechanism**: 50 videos with clear multi-element interactions are curated from a pool of 2,000 videos. For each video: (a) Gemini-2.5-Pro identifies and extracts individual elements from the first frame and generates a complete background with elements removed; (b) SAM 2 removes white backgrounds to produce RGBA elements; (c) elements are placed on the left and the background on the right to form a composite image $I_{mix}$; (d) Gemini-2.5-Pro generates text prompts describing element interactions and video content. The result is 50 (composite image, text, video) triples.
    - **Design Motivation**: Using a VLM for data curation rather than manual annotation ensures element extraction quality and textual precision. Fifty samples suffice to cover diverse scenario types including human–object, human–human, element insertion, and robotic manipulation.

3. **Few-Shot LoRA Adaptation and Scene Transition Mechanism**

    - **Function**: Reliably activates the model's subject fusion and scene transition capabilities with minimal training data.
    - **Mechanism**: LoRA (rank=128) is trained on Wan2.2-I2V-A14B. A unique transition trigger phrase (e.g., "ad23r2 the camera view suddenly changes."—analogous to a DreamBooth rare identifier) is introduced. The generated video frame structure is $F = \{F_c, F_g\}$, where $F_c=4$ temporally compressed frames retain the composite image and $F_g$ frames contain the generated fused content. Wan2.2 uses two separate denoising Transformers (low-noise and high-noise domains), which are trained independently. At inference, the first $F_c$ frames are discarded to obtain a clean video.
    - **Design Motivation**: The low-rank nature of LoRA ensures minimal deviation from pretrained weights, preserving broad generative priors. The unique trigger phrase avoids interference with normal text. Only 50 samples and 5 hours of training on 2 H200 GPUs are required, making the approach highly efficient.

### Loss & Training

Standard diffusion model denoising loss is used. LoRA rank=128. Training data covers four categories: human–object interaction (60%), human–human interaction (14%), element insertion (20%), and robotic manipulation (6%). Training is conducted on 2 NVIDIA H200 GPUs with batch size 4 for 5 hours.

## Key Experimental Results

### Main Results

User study (200 annotations, 40 users):

| Model | Overall Quality↑ | Object Identity↑ | Scene Identity↑ | Avg. Rank↓ | Ranked 1st↑ |
|-------|-----------------|-----------------|-----------------|-----------|------------|
| Wan2.2-I2V-A14B | 2.09 | 3.32 | 3.01 | 3.27 | 3.4% |
| SkyReels-A2 | 2.34 | 2.89 | 3.43 | 3.02 | 4.3% |
| VACE | 3.00 | 3.50 | 3.66 | 2.50 | 11.1% |
| **FFGo (Ours)** | **4.28** | **4.53** | **4.58** | **1.21** | **81.2%** |

FFGo comprehensively outperforms all baselines across every dimension, with 81.2% of users rating its results as best.

### Ablation Study

Qualitative comparison with the base model:

| Configuration | Behavior |
|--------------|----------|
| Wan2.2 + best hand-crafted transition prompt | Frequently animates elements independently; objects disappear after transition |
| Wan2.2 + no transition prompt | Subject fusion is rarely achieved |
| **FFGo (after LoRA adaptation)** | Consistently preserves object identity with coherent scene transitions |

Key finding: FFGo transforms Wan2.2—the worst-performing baseline—into the best-performing model in the evaluation.

### Key Findings

- **Validation of intrinsic capability**: In the rare cases where the base model successfully preserves all object identities and performs coherent scene transitions, FFGo's output closely resembles that of the base model, demonstrating that FFGo activates an existing capability rather than learning a new one.
- **Generalization far exceeds million-scale trained methods**: VACE and SkyReels-A2 are trained on millions of samples but primarily target human–object scenarios, performing poorly on robotic manipulation, driving simulation, and underwater scenes. FFGo, trained on only 50 samples but retaining pretrained priors, substantially outperforms both in generalization.
- **Advantage in reference count**: The architectures of VACE and SkyReels-A2 limit input to at most three references; FFGo imposes no such constraint due to the first-frame conceptual buffer design. Experiments validate performance with up to five references (four objects + one scene).
- Preserving pretrained knowledge is critical—post-training data quality and diversity are far lower than pretraining data, and excessive fine-tuning leads to model degradation.

## Highlights & Insights

- **The insight into video models' intrinsic capabilities** is highly inspiring: the first frame is not merely a spatiotemporal starting point but a conceptual buffer. This finding reframes our understanding of I2V models and suggests that other overlooked intrinsic capabilities may remain latent in pretrained models.
- **The "activate rather than train" paradigm** is elegantly conceived: instead of training a model to acquire new capabilities, the approach identifies and activates capabilities the model already possesses but cannot reliably trigger. With a cost of only 20–50 samples, the results surpass million-scale training, demonstrating that understanding a model's intrinsic mechanisms is more valuable than brute-force data scaling.
- **High practical value**: As a lightweight plugin, FFGo is compatible with existing I2V models, requires no architectural changes, trains quickly, preserves pretrained capabilities, and can be applied as a drop-in enhancement to multi-reference customization for any I2V model.

## Limitations & Future Work

- As the number of reference objects increases, the per-object resolution within the first frame decreases, making identity preservation more difficult. The practical upper limit is approximately 4–5 references.
- Selective text-based control over individual objects becomes challenging as the number of references grows, since textual descriptions may lack sufficient precision.
- Reliance on Gemini-2.5-Pro for data curation introduces cost and dependency on a closed-source API.
- The authors suggest using multiple starting frames as an extended conceptual buffer to overcome capacity limitations—a promising direction for future work.
- The approach is currently validated only on Wan2.2; whether other I2V models (e.g., CogVideoX, Sora) possess similar intrinsic capabilities remains an open question.

## Related Work & Insights

- **vs. VACE**: VACE modifies the architecture to accept multi-reference inputs and trains on millions of samples, primarily excelling at human–object scenarios. It performs poorly on multi-object interactions or scenarios requiring more than three references. FFGo comprehensively surpasses it using only 50 samples without architectural changes.
- **vs. SkyReels-A2**: SkyReels-A2 likewise combines architectural modification with large-scale training and is architecturally limited to three references. It is rated best in only 4.3% of user study cases, compared to FFGo's 81.2%.
- **vs. Wan2.2 base model**: FFGo demonstrates that the base model already possesses subject fusion capability, albeit unreliably. LoRA adaptation converts this unstable latent capability into a reliable, practical one.
- This work extends the research line on "exploring intrinsic capabilities of pretrained models" (e.g., in-context LoRA for DiT, I2V models for perception tasks), representing a direction worthy of deeper investigation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The reinterpretation of the first frame's role is highly insightful; the "activate rather than train" paradigm is genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Includes user studies and multiple qualitative comparisons, but lacks automated quantitative metrics (e.g., CLIP score).
- Writing Quality: ⭐⭐⭐⭐ — The narrative is coherent, the observations are convincing, and the figures are abundant.
- Value: ⭐⭐⭐⭐⭐ — Provides a highly efficient video customization solution with broad implications for the understanding of pretrained model capabilities.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](../../ICCV2025/video_generation/dreamrelation_relation-centric_video_customization.md)
- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)

<!-- RELATED:END -->
