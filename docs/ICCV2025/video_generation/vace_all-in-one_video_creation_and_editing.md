---
title: >-
  [Paper Note] VACE: All-in-One Video Creation and Editing
description: >-
  [ICCV 2025][Video Generation][Video Editing] This paper proposes VACE, an all-in-one video creation and editing framework built on Diffusion Transformer. Through a unified Video Condition Unit (VCU) interface and a plugg…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Video Editing"
  - "Diffusion Transformer"
  - "Unified Framework"
  - "Video Condition Unit"
date: 2026-05-08
content_hash: 3c1566c0da40d532
---

# VACE: All-in-One Video Creation and Editing

**Conference**: ICCV 2025
**arXiv**: [2503.07598](https://arxiv.org/abs/2503.07598)  
**Code**: [https://ali-vilab.github.io/VACE-Page/](https://ali-vilab.github.io/VACE-Page/)  
**Area**: Video Understanding / Video Generation / Video Editing
**Keywords**: Video Generation, Video Editing, Diffusion Transformer, Unified Framework, Video Condition Unit

## TL;DR

This paper proposes VACE, an all-in-one video creation and editing framework built on Diffusion Transformer. Through a unified Video Condition Unit (VCU) interface and a pluggable Context Adapter architecture, a single model covers 12+ video tasks including reference-based generation, video editing, and mask-based editing, achieving performance on par with task-specific models.

## Background & Motivation

**Background**: The video generation field encompasses a rich set of downstream tasks — inpainting, editing, controllable generation, reference-based generation, identity-preserving generation, etc. Unified frameworks (e.g., ACE, OmniGen) have emerged in the image domain, but the video domain still predominantly follows a "one model per task" paradigm due to the greater difficulty of maintaining spatiotemporal consistency.

**Limitations of Prior Work**: (1) Deploying multiple specialized models is costly (separate models for I2V, inpainting, controllable generation, etc.); (2) No single model supports complex compositional tasks (e.g., "reference + inpainting", "sketch + video extension"); (3) A unified multi-task video evaluation benchmark is lacking.

**Key Challenge**: Video task inputs are highly heterogeneous (text, images, video, masks) and must preserve spatiotemporal consistency. Handling these diverse requirements through a unified interface is the central challenge.

**Goal**: Construct an all-in-one video generation and editing framework supporting T2V, R2V (reference-based generation), V2V (video-to-video editing), MV2V (mask-based video editing), and their free combinations.

**Key Insight**: Decompose all video task inputs into a unified triplet of "text + frame sequence + mask sequence."

**Core Idea**: Use VCU to unify the multimodal inputs of diverse video tasks into a standardized representation of frame sequences and mask sequences. Task conditioning information is injected via the Context Adapter, enabling a single model to handle all video creation and editing tasks.

## Method

### Overall Architecture

VACE is built on a Diffusion Transformer (DiT) backbone. The input is a VCU (text $T$ + frame sequence $F$ + mask sequence $M$). The pipeline proceeds through Concept Decoupling (separating content to be edited from content to be preserved), Context Latent Encoding (projecting to latent space), Context Embedder (generating conditioning tokens), and finally Context Adapter injection into the DiT backbone. Two model scales are supported: LTX-Video-2B and Wan-T2V-14B.

### Key Designs

1. **Video Condition Unit (VCU)**:

    - **Function**: Unifies all video task inputs into the format $V = [T; F; M]$.
    - **Mechanism**: Frame sequence $F$ and mask sequence $M$ are spatiotemporally aligned. Different tasks are represented by different assignments:
        - **T2V**: $F = \{0\} \times n,\ M = \{1\} \times n$ (all-zero frames + all-one mask = full generation)
        - **R2V**: $F = \{r_1,\ldots,r_l\} + \{0\} \times n,\ M = \{0\} \times l + \{1\} \times n$ (reference frames preserved + subsequent frames generated)
        - **V2V**: $F = \{u_1,\ldots,u_n\},\ M = \{1\} \times n$ (input video + full replacement)
        - **MV2V**: $F = \{u_1,\ldots,u_n\},\ M = \{m_1,\ldots,m_n\}$ (input video + local mask editing)
    - **Task Composition**: Naturally supported — e.g., reference + inpainting simply requires concatenating reference frames with masked video frames.
    - **Design Motivation**: A mathematically unified representation eliminates interface discrepancies across tasks, freeing the model from needing to perceive the specific task type.

2. **Concept Decoupling**:

    - **Function**: Decomposes frame sequence $F$ via the mask into reactive frames $F_c = F \times M$ (regions to be modified) and inactive frames $F_k = F \times (1 - M)$ (regions to be preserved).
    - **Mechanism**: Different visual concepts (natural video vs. control signals such as depth/pose) follow different distributions; explicit separation facilitates model convergence.
    - **Design Motivation**: Editing tasks require distinguishing "what to change" from "what to keep"; mixing them in the input increases learning difficulty.

3. **Context Adapter Tuning**:

    - **Function**: Selects and copies a subset of Transformer Blocks from the DiT to form Context Blocks, which process context tokens and inject an additive signal into the main branch.
    - **Mechanism**: Follows a Res-Tuning approach — the main DiT branch is frozen; only the Context Embedder and Context Blocks are trained.
    - **Comparison with Full Fine-tuning**: Achieves comparable performance with faster convergence and supports plug-and-play behavior (the adapter can be unloaded at any time to restore the original T2V model).
    - **Block Distribution Strategy**: Distributed placement outperforms consecutive shallow placement under equal block count; the final design uses partial distributed placement.
    - **Design Motivation**: (a) Avoids catastrophic forgetting of pretrained capabilities; (b) Enables plug-and-play composition with the base model.

4. **Context Latent Encoding**:

    - **Function**: Encodes the decoupled $F_c$ and $F_k$ via a Video VAE into a latent space of the same dimensionality as the noisy latent $X$; $M$ is directly reshaped and interpolated.
    - Reference images are encoded separately and concatenated along the temporal dimension (to avoid mixed encoding artifacts between images and video).
    - The Context Embedder concatenates $F_c$, $F_k$, and $M$ along the channel dimension before tokenization. Weights corresponding to $F_c$ and $F_k$ are initialized by copying from the original video embedder; weights for $M$ are zero-initialized.

### Loss & Training

- **Staged training**: The model first learns basic tasks (inpainting, extension, and other modality-complementary tasks), then progressively expands to single-reference → multi-reference → compositional tasks, and finally undergoes quality fine-tuning with high-quality data and longer sequences.
- Supports arbitrary resolution, dynamic duration, and variable frame rate.
- Different tasks are randomly combined during training to support compositional scenarios.
- All mask-related operations are augmented to accommodate requirements at various granularities.

## Key Experimental Results

### Main Results

Automated scoring and user study on VACE-Benchmark (480 evaluation samples, 12 tasks):

| Task | Method | Normalized Avg. (Auto) | User Avg. (1–5) |
|------|--------|------------------------|-----------------|
| I2V | CogVideoX-I2V | 73.66% | 2.92 |
| I2V | **VACE** | **74.38%** | **3.24** |
| Outpainting | M3DDM | 73.16% | 3.29 |
| Outpainting | **VACE** | **74.25%** | **3.80** |
| Depth Control | ControlVideo | 70.07% | 2.29 |
| Depth Control | **VACE** | **74.99%** | **3.23** |
| Pose Control | Follow-Your-Pose | 66.43% | 2.06 |
| Pose Control | **VACE** | **76.13%** | **3.18** |
| R2V | Kling 1.6 (commercial) | 78.81% | 4.04 |
| R2V | VACE | 76.76% | 3.40 |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Full Fine-tuning vs. Context Adapter | Comparable performance; Adapter converges faster |
| Consecutive shallow blocks vs. distributed blocks (equal count) | Distributed placement significantly better |
| With Concept Decoupling vs. without | With decoupling yields more pronounced loss decrease |
| 1/4 blocks vs. 1/2 blocks vs. all blocks | More blocks is better, with diminishing returns |

### Key Findings

- **Unified model vs. specialized models**: VACE outperforms open-source specialized methods on I2V, outpainting, depth, pose, and flow tasks, and matches ProPainter on inpainting.
- **Gap between R2V and commercial models**: On reference-based generation, VACE (based on LTX-Video-2B) still lags behind commercial models (Kling, Pika, Vidu), though its metrics are close to Vidu 2.0.
- **Unique value of compositional tasks**: VACE enables compositional tasks such as "Move Anything," "Swap Anything," and "Expand Anything," which existing single-model or multi-model approaches cannot achieve.
- **Distributed block placement outperforms consecutive placement**: Diversity in injection positions matters more than depth.
- **Concept Decoupling is effective**: Explicitly separating regions to be modified from regions to be preserved accelerates model convergence.

## Highlights & Insights

- **The VCU unified representation is remarkably elegant**: Representing all video tasks as a (frame sequence, mask sequence) pair yields a concise and general mathematical formulation. The semantics of the binary mask are clear — 1 denotes regions to be generated, 0 denotes regions to be preserved. This design is transferable to any multi-task visual generation framework.
- **Engineering value of the plug-and-play design**: The Context Adapter can be attached to or detached from the base T2V model on demand, substantially reducing deployment overhead. A single base model paired with multiple adapters can serve diverse use cases.
- **Emergent compositional capabilities**: By simply concatenating VCU representations of different tasks, new capabilities such as "Move Anything" and "reference-based inpainting" naturally emerge — a core advantage of the unified framework.
- **VACE-Benchmark fills a gap**: It is the first unified evaluation benchmark covering 12 video tasks, incorporating both automatic metrics and user studies.

## Limitations & Future Work

- A notable gap remains between VACE and commercial models on R2V (user score 3.40 vs. 4.04), likely attributable to model scale.
- VACE-Benchmark contains only ~20 samples per task, making the evaluation scale relatively small.
- The paper does not discuss inference efficiency in detail — the computational overhead introduced by the Context Adapter is not quantified.
- Constructing compositional tasks requires users to manually assemble VCU inputs, which imposes a non-trivial usage burden.
- Data construction relies on large-scale automated pipelines (SAM2, RAM, Grounding DINO, etc.), and annotation quality may be inconsistent.
- Full ablation studies are conducted only on LTX-Video-2B; ablation results for the 14B variant are absent.

## Related Work & Insights

- **vs. ACE [2024]**: A unified generation and editing framework in the image domain that unifies different tasks via condition tokens. VACE extends this to the video domain; its core innovations lie in the spatiotemporal unified representation of VCU and the Context Adapter.
- **vs. OmniGen [2024]**: Another unified image generation method. VACE addresses the harder problem of spatiotemporal consistency in video.
- **vs. InstructPix2Pix [2023]**: An instruction-based image editing method. VACE requires no natural language instructions; editing regions and references are directly specified via VCU.
- **vs. ControlNet [2023]**: Enables single-condition controllable generation. VACE unifies all condition types and extends the paradigm to video.
- The VCU design philosophy of VACE is transferable to other multimodal conditional generation tasks, including 3D generation and audio generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The VCU unified representation and Context Adapter are elegant designs, though the underlying concept is a natural extension of existing image unification frameworks to video.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons across 12 tasks, ablation studies, user studies, and compositional task visualizations are provided, though VACE-Benchmark remains small in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The formal definition of VCU is clear and concise; the architectural description is well-organized.
- **Value**: ⭐⭐⭐⭐⭐ — The first unified all-task model on video DiT, filling the gap in unified video generation and editing; code is open-sourced from Alibaba Tongyi Lab.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](../../CVPR2026/video_generation/videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_generation/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)

</div>

<!-- RELATED:END -->
