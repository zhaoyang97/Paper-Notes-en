---
title: >-
  [Paper Note] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning
description: >-
  [CVPR 2026][Multimodal VLM][Long-range visual consistency] This paper proposes the Narrative Weaver framework, which combines narrative planning via MLLMs with fine-grained generation via diffusion models. Through learnable queries and a dynamic Memory Bank, the framework achieves long-range visually consistent generation under multi-modal conditioning. The authors also introduce EAVSD, the first e-commerce advertising video storyboard dataset, comprising 330K+ images.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Long-range visual consistency
  - narrative generation
  - AR+Diffusion
  - Memory Bank
  - e-commerce advertising
date: 2026-05-08
content_hash: 69d3360787cce1aa
---

# Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning

**Conference**: CVPR 2026
**arXiv**: [2603.06688](https://arxiv.org/abs/2603.06688)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Long-range visual consistency, narrative generation, AR+Diffusion, Memory Bank, e-commerce advertising

## TL;DR
This paper proposes the Narrative Weaver framework, which combines narrative planning via MLLMs with fine-grained generation via diffusion models. Through learnable queries and a dynamic Memory Bank, the framework achieves long-range visually consistent generation under multi-modal conditioning. The authors also introduce EAVSD, the first e-commerce advertising video storyboard dataset, comprising 330K+ images.

## Background & Motivation

**State of the Field**: Generative AI systems such as Sora, Veo, and Midjourney demonstrate strong performance on short-clip image/video generation, yet long-range narrative generation—maintaining character, background, and style consistency across frames—remains a major challenge.

**Limitations of Prior Work**: (1) Video generation models suffer rapid consistency degradation beyond short clips; (2) Image generation models operate on individual frames and cannot plan multi-frame narratives; (3) Existing planning methods rely on purely textual conditioning and cannot produce controllable, visually grounded outputs.

**Root Cause**: No unified framework exists that integrates narrative planning, fine-grained control, and long-range consistency into a single system. Large-scale multi-modal conditioned generation datasets are also lacking.

**Paper Goals**: Achieve multi-modal conditioned long-sequence consistent generation of the form (text, image) → (text, {Image_i}).

**Starting Point**: A hybrid architecture combining an AR model for planning and a diffusion model for generation, with a Memory Bank propagating consistency information across keyframes.

**Core Idea**: An MLLM acts as a "director" to plan narratives and compress context into learnable queries; a Memory Bank anchors the initial visual condition to prevent drift; three-stage progressive training enables data-efficient learning.

## Method

### Overall Architecture
A hybrid AR + Diffusion architecture: an MLLM (Qwen2.5-VL-3B) serves as the AR component responsible for textual narrative planning and historical information encoding, while Flux.1-Dev serves as the diffusion component responsible for image generation. The input consists of a conditioning image and a user instruction; the output is a multi-frame visual narrative sequence.

### Key Designs

1. **Multi-Modal Interaction and Learnable Queries**:

    - **Function**: The MLLM simultaneously performs narrative planning (text generation) and high-level visual content aggregation (query vector generation).
    - **Mechanism**: A dynamic causal attention mask is designed—text tokens attend only to preceding text tokens (standard causal attention), while learnable queries $q_n$ can attend to the full multi-modal context (input $\mathbf{I}$, all narrative texts $\{t_j\}$, and preceding queries $\{q_k\}$).
    - Special tokens `<img>` / `</img>` delimit query sequences, enabling the model to learn when to generate an image versus when to continue planning text.
    - **Design Motivation**: Prevents queries from interfering with original text generation while allowing queries to fully absorb multi-modal information.

2. **Dynamic Memory Bank**:

    - **Function**: Caches VAE features of previously generated images to prevent visual drift.
    - **Mechanism**: Features from the most recent $T$ frames are cached and compressed via geometrically decayed average pooling—the feature length of the $k$-th frame is $l/\lambda^{k-1}$, ensuring the total memory length is bounded by $L < l \cdot \lambda/(\lambda-1)$.
    - The final conditioning signal is: $\mathbf{C}_n = \text{Concat}(q_n, f^{cond}, \hat{f}_{n-1}, ..., \hat{f}_{n-T})$
    - **Design Motivation**: Recent frames retain more detail (high resolution) while distant frames provide coarse-grained context (compressed), decoupling the trade-off between consistency and efficiency.

3. **Three-Stage Progressive Training**:

    - Stage 1 (Narrative Planning): Trains the MLLM to learn textual narrative generation and image-timing prediction using a standard cross-entropy loss.
    - Stage 2 (Semantic Consistent Generation): Trains learnable queries and projectors; pre-trained on 30M low-resolution text–image pairs, then fine-tuned on 60K high-quality samples using a Flow Matching objective.
    - Stage 3 (Fine-Grained Consistent Alignment): Fully trains the diffusion model, incorporating VAE features of conditioning images and Memory Bank features, continuing with the Flow Matching objective.

### Efficiency Analysis
- DiT computational complexity is reduced from quadratic to linear growth with respect to the number of images.
- The bottleneck shifts to the highly optimizable MLLM component.
- Parallel planning and generation are supported at inference time.

## Key Experimental Results

### GPT-4o Evaluation (Consistent Visual Generation)

| Method | Text Control | ITC | RGC | MSSC | MSCC | IMQ |
|--------|-------------|-----|-----|------|------|-----|
| StoryDiffusion | ✗ | 6.54 | 5.86 | 7.48 | 6.00 | 6.80 |
| IP-Adapter | ✗ | 7.11 | 6.10 | 8.57 | 7.57 | 6.65 |
| Flux.1-kontext | ✗ | 7.06 | 9.41 | 8.11 | 7.28 | 6.94 |
| **Narrative Weaver** | **✓** | **7.54** | 8.86 | **8.67** | **7.91** | **7.35** |

### Automatic Evaluation (DreamSim↓ / CLIP Score↑)

| Method | DreamSim↓ (Avg) | Notes |
|--------|----------------|-------|
| StoryDiffusion | 56.33 | Multi-scene generation method |
| IP-Adapter | 33.30 | Reference image method |
| Flux.1-kontext | 3.71 | Editing method (but exhibits copy-paste artifacts) |
| Narrative Weaver | **12.18** | Best among multi-scene generation methods |

### User Study
- 180+ user preference surveys confirm the model's advantages.
- Although Flux.1-kontext achieves better automatic metrics, its "copy-paste" behavior is disfavored by users.

## Highlights & Insights
- The first generative framework to unify narrative planning, fine-grained control, and long-range consistency, filling an important gap in the field.
- The dynamic causal attention mask is elegantly designed; text planning can be learned with as few as ~5K samples.
- The geometrically decayed compression in the Memory Bank guarantees bounded memory while prioritizing recent frames.
- EAVSD fills the gap in e-commerce advertising storyboard datasets (330K+ images).
- The three-stage training strategy achieves state-of-the-art performance under limited computation and data, offering strong practical utility.
- Computational complexity is reduced from quadratic to linear growth, enabling generation of longer narrative sequences.

## Limitations & Future Work
- The current approach focuses primarily on keyframe generation; consistency of transitional video segments between keyframes remains unresolved.
- The planning capacity of Qwen2.5-VL-3B may limit narrative complexity; larger MLLMs could raise the performance ceiling.
- EAVSD dataset construction relies on commercial models (Qwen-Image, Flux.1-kontext), which may introduce generative bias.
- Dedicated modules for identity preservation (e.g., face ID embeddings) could be incorporated to further improve character consistency.
- The effect of the geometric decay rate $\lambda$ in the Memory Bank across different narrative lengths warrants further ablation.
- Stage 3 is trained for only 1–2 epochs; more thorough training may further improve fine-grained consistency.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_multimodal_moe_anomaly_detection.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

<!-- RELATED:END -->
