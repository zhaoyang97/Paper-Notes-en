---
title: >-
  [Paper Note] Infinite-Story: A Training-Free Consistent Text-to-Image Generation
description: >-
  [AAAI 2026][Image Generation][consistent text-to-image generation] Built upon a scale-wise autoregressive model (Infinity), this work introduces three training-free techniques—Identity Prompt Replacement (eliminating con…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "consistent text-to-image generation"
  - "visual storytelling"
  - "autoregressive generation"
  - "training-free"
  - "style consistency"
date: 2026-05-08
content_hash: 6941faa6a6b4b056
---

# Infinite-Story: A Training-Free Consistent Text-to-Image Generation

**Conference**: AAAI 2026
**arXiv**: [2511.13002](https://arxiv.org/abs/2511.13002)  
**Code**: N/A  
**Area**: Image Generation / Consistent Generation
**Keywords**: consistent text-to-image generation, visual storytelling, autoregressive generation, training-free, style consistency

## TL;DR
Built upon a scale-wise autoregressive model (Infinity), this work introduces three training-free techniques—Identity Prompt Replacement (eliminating contextual bias in the text encoder), Adaptive Style Injection (reference image feature injection), and Synchronized Guidance Adaptation (synchronizing both branches of CFG)—to achieve identity- and style-consistent multi-image generation at 6× the speed of diffusion-based methods (1.72 s/image).

## Background & Motivation
Consistent text-to-image generation is critical for visual storytelling, comics, and character-driven content creation. Existing methods suffer from two issues: (1) most are built on diffusion models with slow inference (typically >10 s/image), exceeding user interaction tolerance; (2) prior work focuses primarily on identity consistency while neglecting style consistency—the rendering style, color tone, and background aesthetics of the same character may vary drastically across scenes (e.g., 1Prompt1Story). Scale-wise autoregressive models (e.g., Infinity) offer faster inference via a next-scale prediction paradigm, yet face analogous consistency challenges.

## Core Problem
How can a scale-wise autoregressive T2I model generate a set of images that are consistent in both identity and style, without any additional training? The challenge stems from contextual bias in the text encoder (the same identity description yields different semantic representations under different prompts) and the lack of a cross-image visual feature alignment mechanism.

## Method

### Overall Architecture
Built on Infinity (a 2B-parameter scale-wise autoregressive model with a Flan-T5 text encoder), $N$ prompts are processed as a single batch in parallel. The first sample serves as the reference (anchor), and its identity and style features are propagated to the remaining samples. The three techniques operate at the text encoding layer and the early self-attention layers of the generation process.

### Key Designs
1. **Identity Prompt Replacement (IPR)**: The text encoder encodes "a dog" differently depending on context (e.g., "springing toward a frisbee" vs. "on a porch swing"), yielding distinct semantics (Corgi vs. Golden Retriever). IPR replaces the identity embeddings of all samples with the reference identity embedding $T_{iden}^1$, while normalizing the expression/scene embeddings to preserve their relative scale: $\hat{T}_{exp}^n = \frac{\|T_{iden}^1\|}{\|T_{iden}^n\|} \cdot T_{exp}^n$. This eliminates contextual bias at the encoding stage.

2. **Adaptive Style Injection (ASI)**: During early generation steps ($S_{\text{early}}=\{2,3\}$) in the self-attention layers, the Keys of all samples are replaced with those of the reference, and the Values are adaptively interpolated based on cosine similarity: $\bar{V}_s^n = \alpha_s^n V_s^n + (1-\alpha_s^n) V_s^1$, where $\alpha_s^n = \lambda \cdot \text{sim}(V_s^1, V_s^n)$. Regions with high similarity retain more of their original features, while low-similarity regions borrow more from the reference, enabling adaptive appearance and style alignment.

3. **Synchronized Guidance Adaptation (SGA)**: Applying ASI exclusively to the conditional branch of CFG disrupts the balance between the conditional and unconditional branches, degrading prompt fidelity. SGA synchronizes the same operation (using the same $\alpha$ weights computed from the conditional branch) to the unconditional branch, restoring CFG balance.

### Loss & Training
Entirely training-free; no training or fine-tuning is required. All parameters are frozen, and only the K/V features in the attention layers are modified at inference time.

## Key Experimental Results

| Method | CLIP-I↑ | DreamSim↓ | CLIP-T↑ | DINO↑ | Time (s/img) |
|---|---|---|---|---|---|
| Infinite-Story | **0.8089** | **0.1834** | 0.8732 | **0.9267** | **1.72** |
| 1Prompt1Story | 0.7687 | 0.1993 | 0.8942 | 0.9117 | 22.57 |
| IP-Adapter | 0.7834 | 0.2266 | 0.8661 | 0.9243 | 10.40 |
| ConsiStory | 0.6895 | 0.2787 | 0.9019 | 0.8954 | 37.76 |
| Vanilla Infinity | 0.6965 | 0.2780 | 0.8836 | 0.8955 | 1.71 |

User study: 58.4% of participants preferred Infinite-Story (vs. 18% for 1Prompt1Story, 16.4% for IP-Adapter, and 7.2% for OneActor).

### Ablation Study
- IPR alone: CLIP-I improves from 0.6965 to 0.7119; DreamSim decreases from 0.2780 to 0.2569.
- Adding ASI: DINO improves substantially to 0.9242 (significant style consistency gain); CLIP-I jumps to 0.8082.
- Adding SGA: CLIP-T recovers from 0.8625 to 0.8732 (prompt fidelity restored); overall $S_H$ reaches its optimum.
- Sensitivity to $\lambda$: $\lambda=0.85$ achieves the best trade-off between consistency and prompt fidelity.
- The method generalizes to Switti and HART, demonstrating transferability to other scale-wise autoregressive models.

## Highlights & Insights
- **6× inference speedup**: 1.72 s/image vs. 10–38 s/image for diffusion-based methods, reaching the practical threshold for interactive applications.
- **Discovery and resolution of contextual bias**: Identity Prompt Replacement elegantly addresses the problem of "identical descriptions yielding different semantics due to varying context" in the text encoder.
- **Adaptive interpolation weights**: ASI adaptively modulates injection strength via cosine similarity, avoiding detail loss caused by hard replacement.
- **Fully training-free**: All three techniques operate solely on attention features at inference time, incurring zero additional training cost.

## Limitations & Future Work
- Relies on a single reference image (anchor); poor anchor quality propagates to the entire batch.
- Identity consistency is achieved primarily through attention-layer manipulation, with limited control over highly structured or fine-grained details.
- Validated only on scale-wise autoregressive models; applicability to diffusion models remains unexplored.
- CLIP-T is slightly lower than some baselines, indicating a residual trade-off between consistency and prompt fidelity.
- Adaptive anchor selection or correction mechanisms are not explored.

## Related Work & Insights
- **vs. 1Prompt1Story**: Both are training-free, but 1Prompt1Story is diffusion-based (22.57 s/image) and addresses only identity consistency, not style consistency.
- **vs. ConsiStory/StoryDiffusion**: These methods achieve identity consistency by modifying attention weights but suffer from extremely slow inference (24–38 s/image) and poor style consistency.
- **vs. IP-Adapter**: IP-Adapter requires a reference image, has slower inference, and exhibits lower prompt fidelity due to over-reliance on the reference image.
- Synchronizing operations across both CFG branches is a general-purpose technique for controllable generation applicable to other training-free methods.
- Contextual bias is not unique to T2I; it is pervasive in VLMs, where the same visual concept is interpreted differently under different textual contexts.
- The inference speed advantage of scale-wise autoregressive models warrants attention as a strong alternative to diffusion models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First training-free method for consistent T2I on scale-wise autoregressive models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-metric evaluation, user study, complete ablations, and cross-model generalization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the visualization of contextual bias is intuitive and persuasive.
- **Value**: ⭐⭐⭐⭐ The substantial inference speedup brings consistent T2I to a practical level, with direct value for visual storytelling applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](../../ICLR2026/image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)
- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[AAAI 2026\] RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation](retrysql_text-to-sql_training_with_retry_data_for_self-correcting_query_generati.md)

</div>

<!-- RELATED:END -->
