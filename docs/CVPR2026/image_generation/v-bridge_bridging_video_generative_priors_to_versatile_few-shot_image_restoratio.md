---
title: >-
  [Paper Note] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration
description: >-
  [CVPR 2026][Image Generation][Video generative priors] This paper reformulates image restoration as a progressive video generation process. By leveraging the rich visual priors of a pretrained video model (Wan2.2-TI2V-5B), the proposed method achieves versatile all-in-one restoration across multiple degradation types using only 1,000 multi-task training samples (less than 2% of existing methods), surpassing specialized architectures trained on million-scale datasets.
tags:
  - CVPR 2026
  - Image Generation
  - Video generative priors
  - image restoration
  - few-shot learning
  - progressive restoration
  - drift correction
date: 2026-05-08
content_hash: 31b5cc497339ea4d
---

# V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2603.13089](https://arxiv.org/abs/2603.13089)
**Code**: Available (open-source)
**Area**: Image Generation
**Keywords**: Video generative priors, image restoration, few-shot learning, progressive restoration, drift correction

## TL;DR
This paper reformulates image restoration as a progressive video generation process. By leveraging the rich visual priors of a pretrained video model (Wan2.2-TI2V-5B), the proposed method achieves versatile all-in-one restoration across multiple degradation types using only 1,000 multi-task training samples (less than 2% of existing methods), surpassing specialized architectures trained on million-scale datasets.

## Background & Motivation
**State of the Field**: Image restoration tasks (denoising, deblurring, dehazing, etc.) predominantly rely on task-specific models that require large-scale supervised data (>1M samples) to learn restoration mappings from scratch. All-in-One approaches (AirNet, PromptIR, FoundIR) unify multiple tasks but still demand massive datasets.

**Limitations of Prior Work**: Existing methods are decoupled from advances in large-scale generative models. Video generation models (Wan, HunyuanVideo) have acquired rich structural, semantic, and dynamic priors from vast training data, yet these priors remain unexploited for low-level vision tasks.

**Root Cause**: Video generation models have learned powerful visual world models, while restoration methods continue to be trained from scratch — a substantial gap in prior utilization persists.

**Starting Point**: The paper observes that the restoration process is intrinsically a "progressive evolution from low quality to high quality," which naturally corresponds to the temporal evolution of video frames — enabling restoration to be modeled as a video generation problem.

**Core Idea**: Each (LQ, HQ) pair is constructed as a pseudo-temporal sequence, allowing the video generation model to learn restoration trajectories rather than performing single-step regression.

## Method

### Overall Architecture
V-Bridge formulates image restoration as a "quality evolution trajectory" from low quality to high quality. The degraded image $\mathbf{I}_{\text{LQ}}$ serves as the first frame and the clean image $\mathbf{I}_{\text{HQ}}$ as the last frame, with intermediate frames generated via linear interpolation. Built upon the Wan2.2-TI2V-5B video generation backbone, training proceeds in three stages: pseudo-temporal data construction → progressive curriculum training → drift correction.

### Key Designs

1. **Pseudo-Temporal Data Construction**:

    - Function: Converts static (LQ, HQ) pairs into pseudo-video sequences of length $T+1$
    - Mechanism: $\mathbf{I}_t = (1-\alpha_t)\mathbf{I}_{\text{LQ}} + \alpha_t \mathbf{I}_{\text{HQ}}$, where $\alpha_t = t/T$, forming a monotonically increasing quality trajectory
    - Design Motivation: Video models are inherently designed to model temporally consistent continuous changes. The pseudo-temporal sequence provides temporally coherent supervision signals, yielding more stable training than single-step regression.

2. **Progressive Curriculum Training**:

    - Function: Bridges the gap between the video pretraining resolution (720p) and the high-resolution restoration target (4K) through a multi-stage resolution escalation strategy
    - Mechanism: A resolution curriculum $\{r_t\}$ ($r_1 < r_2 < \cdots < r_T$) is defined; at each stage, training data undergoes DownUp downsampling $v_i^{(t)} = \text{DownUp}(v_i, r_t)$, progressively transitioning from low to high resolution
    - Design Motivation: The model first learns global structural restoration before gradually refining high-frequency details. Direct high-resolution training is computationally expensive and introduces a large resolution gap from pretraining, reducing learning efficiency.
    - Unified training objective: $\mathcal{L}(\theta) = \mathbb{E}[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)]$

3. **Drift Correction**:

    - Function: Corrects high-frequency detail deviations caused by the limited pretraining resolution of the video backbone
    - Mechanism: The base model output $\hat{x}$ is treated as a sample from the low-fidelity distribution $p_\theta^{\text{LR}}(x)$. An auxiliary correction model $g_\phi: p_\theta^{\text{LR}} \to p_{\text{HR}}$ is trained on short interpolated sequences constructed between $\hat{x}$ and $x^{\text{HR}}$.
    - Design Motivation: Curriculum training reduces but cannot fully eliminate the resolution gap. Drift correction compensates for residual deviation using a lightweight short-sequence refinement module with minimal computational overhead. It essentially models the systematic bias induced by resolution constraints as a novel form of degradation.

### Loss & Training
- Both models use Wan2.2-TI2V-5B as the backbone network
- Unified training objective: $\mathcal{L}(\theta) = \mathbb{E}[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)]$, essentially supervised fine-tuning for conditional video generation
- Only 50 samples per degradation type are selected (from FoundIR and RealCE), totaling ~1,000 training samples
- The drift correction model likewise uses 50 samples per type and also adopts the Wan2.2 backbone

## Key Experimental Results

### Main Results — FoundIR Test Set (PSNR/SSIM, higher is better)

| Degradation | V-Bridge (1K data) | FoundIR-G (1M data) | Best Specialized Method |
|-------------|-------------------|---------------------|------------------------|
| Blur | 24.92 / 0.781 | 24.34 / 0.786 | 25.31 (DiffUIR) |
| Lowlight | **26.94 / 0.894** | 12.35 / 0.719 | 21.90 (InstructIR) |
| B+N (mixed) | **27.31 / 0.847** | 22.53 / 0.765 | 24.44 (DiffUIR) |
| B+J (mixed) | **25.33 / 0.802** | 28.33 / 0.849 | 32.99 (InstructIR) |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| w/o Drift Correction | Low-light PSNR drops from 26.94 to 19.18 (−7.76 dB) |
| 50 → fewer samples per type | Performance degrades gradually, demonstrating strong few-shot capability |
| Video prior vs. image-based methods | V-Bridge with 1K data surpasses FoundIR with 1M data on low-light and mixed degradations |

### Key Findings
- The most significant advantages are observed on low-light enhancement and mixed degradations (+7.76 dB / +4.78 dB), indicating that video priors generalize better to complex degradations.
- On simple single-type degradations (noise, JPEG), the method slightly underperforms approaches trained on million-scale data, yet achieves a ~1000× improvement in data efficiency.
- Drift correction contributes the largest individual gain, particularly on tasks requiring high-frequency detail reconstruction (nearly 8 dB PSNR improvement on low-light).
- The method demonstrates strong out-of-distribution generalization, maintaining competitive performance on unseen degradation types and external benchmarks.
- Training requires only 50 samples per degradation type (~1K total), compared to 1M samples for FoundIR — approximately a 1000× improvement in data efficiency.
- The progressive curriculum training strategy effectively bridges the gap between the pretraining resolution (720p) and the restoration target resolution.

## Highlights & Insights
- **Paradigm Innovation**: This work is the first to reformulate image restoration as a video generation task, replacing single-step regression with a "quality evolution trajectory" — a fundamentally new task definition that breaks the conventional input→output paradigm in restoration.
- **1000× Data Efficiency**: Achieving competitive or superior performance with 1K versus 1M samples demonstrates that video generation models have implicitly encoded powerful restoration priors — priors acquired "for free" during video training.
- **Extension of Chain-of-Frames**: The Chain-of-Frames reasoning paradigm is extended from high-level semantic inference to low-level pixel restoration, demonstrating the generality of video models.
- **Engineering Value of Progressive Curriculum**: The coarse-to-fine strategy not only improves performance but also serves as a general engineering methodology for adapting large models to new tasks.

## Limitations & Future Work
- Performance on simple degradations such as denoising and JPEG artifact removal falls short of specialized methods, possibly because these degradations do not require complex structural priors.
- Inference cost is high (5B parameter video generation model), making the method unsuitable for real-time applications.
- Only Wan2.2 is validated as the backbone; other video models (e.g., CogVideoX, HunyuanVideo) remain unexplored.
- Linear interpolation for intermediate frame construction is overly simplistic and may fail to capture non-linear quality evolution dynamics.
- The effect of the pseudo-temporal sequence length $T$ on performance is not sufficiently ablated.
- The drift correction model also employs the full Wan2.2-5B backbone, resulting in significant parameter redundancy — it remains an open question whether a lightweight model could achieve equivalent refinement.

## Related Work & Insights
- **vs. FoundIR**: FoundIR is the current all-in-one restoration state of the art but requires 1M training samples; V-Bridge achieves competitive performance with 0.1% of that data and even surpasses it on low-light and mixed degradations.
- **vs. DiffBIR/DiffUIR**: These methods leverage 2D diffusion priors, whereas V-Bridge leverages 3D video priors. The additional temporal modeling capacity yields better global consistency, especially in generalization across degradation types.
- **vs. Chain-of-Frames**: CoF reasoning has primarily been applied to high-level semantic tasks; V-Bridge is the first to demonstrate its feasibility for pixel-level restoration.
- **vs. All-in-One Methods (AirNet/PromptIR)**: These methods rely on degradation-type-specific prompts or contrastive learning; V-Bridge requires no degradation type prior.
- **Insight**: Video foundation models may serve as the backbone for the next generation of unified vision models — applicable not only to generation but also to understanding and restoration. This opens a new dimension for foundation model applications.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — A genuinely new paradigm: modeling restoration as video generation, with a fresh perspective strongly supported by experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple degradation types and benchmarks with thorough OOD testing, though comparisons with more all-in-one methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear and architecture diagrams are intuitive, though some experimental tables lack polish.
- Value: ⭐⭐⭐⭐⭐ — Opens a new research direction for applying video priors to low-level vision, with broad potential impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](asbridge_a_bidirectional_generative_framework_brid.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[ICLR 2026\] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](../../ICLR2026/image_generation/semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)

<!-- RELATED:END -->
