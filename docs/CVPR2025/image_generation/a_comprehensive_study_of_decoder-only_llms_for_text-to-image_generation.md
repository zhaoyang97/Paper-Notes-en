---
title: >-
  [Paper Note] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation
description: >-
  [CVPR 2025][Image Generation][Text Encoder] This paper systematically investigates the effectiveness of using decoder-only LLMs as text encoders for text-to-image diffusion models. The authors find that while directly using the last-layer embeddings yields worse results than T5, aggregating embeddings across all layers via layer-normalized averaging significantly outperforms the T5 baseline.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text Encoder"
  - "Decoder-Only LLM"
  - "Text-to-Image Generation"
  - "Layer-Normalized Average"
  - "Vision-Language Reasoning"
date: 2026-05-08
content_hash: e174975a81e164d8
---

# A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2506.08210](https://arxiv.org/abs/2506.08210)  
**Code**: None  
**Area**: Image Generation / Multimodal  
**Keywords**: Text Encoder, Decoder-Only LLM, Text-to-Image Generation, Layer-Normalized Average, Vision-Language Reasoning

## TL;DR

This paper systematically investigates the effectiveness of using decoder-only LLMs as text encoders for text-to-image diffusion models. The authors find that while directly using the last-layer embeddings yields worse results than T5, aggregating embeddings across all layers via layer-normalized averaging significantly outperforms the T5 baseline.

## Background & Motivation

**Background**: Current text-to-image generation models (e.g., Stable Diffusion, DALL-E 3) commonly rely on T5 or CLIP as text encoders. However, T5 is an older encoder-decoder architecture, and CLIP has a small capacity (354M parameters) and a restrictive token length limit of 77, which constraints their expressiveness.

**Limitations of Prior Work**: Although decoder-only LLMs have comprehensively surpassed encoder-decoder architectures in NLP, their potential in text-to-image generation has not been systematically investigated. The few existing works that utilize LLMs (e.g., Lumina, Sana) directly employ the last-layer output and vary widely in training configurations, rendering fair comparisons difficult.

**Key Challenge**: Decoder-only LLMs use causal attention masks where information flows only from left to right, meaning the last layer may not provide the optimal representation. In contrast, encoder-decoder models (like T5) utilize bidirectional attention, leading to more complete information in the final layer.

**Goal**: (1) Can decoder-only LLMs replace T5 for text-to-image generation? (2) How can embeddings be optimally extracted from LLMs? (3) Do instruction-tuned/finetuned embedding models perform better? (4) Does scaling up the model size lead to continuous improvements?

**Core Idea**: Use layer-normalized averaging to aggregate embeddings across all layers of the LLM, enabling the complementary language features captured by different layers to form a richer composite text representation.

## Method

### Overall Architecture

Based on the U-Net architecture of Stable Diffusion v2, this work keeps the text encoder frozen and only replaces it, injecting text embeddings into the U-Net via cross-attention. A linear projection layer (with a 1024-dimensional output) is integrated to accommodate the varying embedding dimensions of different text encoders. The model is trained for 800K iterations on a dataset of 46M text-image pairs at 256×256 resolution using 32×A100 GPUs.

### Key Designs

1. **Embedding Extraction Strategy Comparison**:

    - Function: Compares four extraction strategies: last-layer, single intermediate layer, average, and layer-normalized average.
    - Key Findings: For decoder-only LLMs, last-layer embeddings perform the worst (VQAScore of 0.675 for Mistral-7B), lagging significantly behind T5's 0.741. An intermediate layer (e.g., the 15th layer) slightly improves results (0.725), while simple averaging (avg) boosts performance to 0.731. **Layer-normalized average (norm avg) achieves the best performance (0.769)** because embedding norms vary drastically across different layers; normalizing them before averaging is essential to fairly fuse features across all layers.
    - Design Motivation: Each LLM layer captures distinct linguistic features—lower layers identify lexical/syntactic structures, middle layers encode semantics, and top layers compress information geared toward the next-token prediction objective. Averaging across all layers comprehensively leverages these complementary features.

2. **Evaluation of Fine-Tuned LLM Embedding Models**:

    - Function: Evaluates top-ranking fine-tuned embedding models from the MTEB leaderboard (e.g., bge-Gemma2, sfr-Mistral, gte-Qwen2).
    - Key Findings: bge-Gemma2 (fine-tuned on Gemma2-9B) achieves peak performance using norm avg (VQAScore of 0.789), outperforming T5 (0.741) by a wide margin. In contrast, gte-Qwen2 performs poorly (0.482), likely because its fine-tuning objective overly emphasizes sentence-level semantics, which degrades token-level fine-grained information.
    - Design Motivation: Since embedding models are fine-tuned via contrastive learning to enhance semantic comprehension, they should theoretically excel at capturing the semantic alignment required for text-to-image integration.

3. **Model Scaling Effects**:

    - Function: Compares physical scales, such as Gemma2-2B vs. 9B and Qwen2-1.5B vs. 7B.
    - Key Findings: Scaling up model size consistently improves overall performance (Gemma2: 0.757 → 0.789, Qwen2: 0.740 → 0.769). However, improvements are uneven across categories: Counting and Comparison exhibit the largest gains, while Scene and Negation show limited improvements.
    - Design Motivation: Evaluates whether the scaling law of LLMs successfully transfers to the text-to-image generation domain.

### Loss & Training

- VFC (VisualFactChecker) is used for caption upsampling to enhance training text diversity.
- During inference, Gemma2-9B is leveraged for prompt upsampling to match the training distribution.
- CFG is fixed at 7.0 for a fair comparison.
- GenAI-Bench outputs are evaluated using VQAScore (implemented with GPT-4o), which reflects compositional text-to-image alignment more accurately than CLIPScore or FID.

## Key Experimental Results

### Main Results: Last-Layer Embedding Comparison (VQAScore on GenAI-Bench)

| Model | Parameters | Average | Counting | Comparison | Negation |
|------|--------|------|----------|------------|----------|
| CLIP-ViT-H/14 | 354M | 0.622 | 0.529 | 0.522 | 0.480 |
| T5-XXL | 4.7B | 0.741 | 0.677 | 0.717 | 0.599 |
| Mistral-7B | 7B | 0.675 | 0.576 | 0.556 | 0.524 |
| Gemma2-9B | 9B | 0.710 | 0.642 | 0.659 | 0.544 |
| bge-Gemma2 | 9B | 0.737 | 0.662 | 0.654 | 0.623 |

### Ablation Study: Different Embedding Strategies (VQAScore)

| Model | Strategy | Average | Counting | Comparison | Negation |
|------|------|------|----------|------------|----------|
| T5-XXL | last layer | 0.741 | 0.677 | 0.717 | 0.599 |
| T5-XXL | norm avg | 0.747 | 0.687 | 0.736 | 0.617 |
| Mistral-7B | last layer | 0.675 | 0.576 | 0.556 | 0.524 |
| Mistral-7B | norm avg | **0.769** | 0.699 | 0.716 | 0.630 |
| bge-Gemma2 | last layer | 0.737 | 0.662 | 0.654 | 0.623 |
| bge-Gemma2 | norm avg | **0.789** | **0.745** | **0.776** | **0.712** |

### Key Findings

- **Last-layer is a pitfall**: The last-layer embeddings of all decoder-only LLMs perform worse than T5, but notably surpass it after applying norm avg. This occurs because the final layer of the LLM is "polluted" by the next-token prediction objective, leading to compressed information.
- **Norm avg is key**: Mistral-7B improves from 0.675 → 0.769 (+13.9%), and bge-Gemma2 improves from 0.737 → 0.789 (+7.1%). Normalization resolves the massive discrepancy in embedding norms across different layers.
- **Best Model**: bge-Gemma2 + norm avg achieves 0.789, completely outperforming T5-XXL's 0.741 and leading across all 10 evaluated skill dimensions.
- **Significant improvement in Negation**: This is traditionally the weakest skill category for CLIP and T5 (which requires understanding negative semantics like "not"), where LLMs possess an inherent advantage.

## Highlights & Insights

- **Counter-intuitive discovery**: Directly utilizing the final layer of an LLM for text-to-image generation performs worse than T5, but changing the extraction strategy allows it to vastly outperform T5. This demonstrates that "how to use" is more critical than "what to use," which offers valuable guidance for the community using LLMs as text encoders.
- **Elegance of layer-normalized average**: No training is required; simply altering the embedding extraction method yields massive gains. This simple trick can be directly integrated into any system utilizing an LLM as a text encoder.
- **Systematic benchmark design**: The evaluations span 27 models under a unified training configuration across 10 skill categories. The rigorous control of variables ensures high credibility of the conclusions.

## Limitations & Future Work

- Validated only on the U-Net architecture at 256×256 resolution; DiT architectures and higher resolutions were not evaluated.
- High computational cost: Training each model takes 7 days on 32×A100 GPUs, resulting in a highly expensive systematic study over 27 models.
- More complex layer fusion strategies (such as learned layer weighting) have not yet been explored.
- Whether these findings generalize to autoregressive image-generation models remains unverified.

## Related Work & Insights

- **vs. Playground-v3**: Playground-v3 also utilizes LLaMA-3 as a text encoder, but adapts various intermediate layers for different DiT blocks via adapters. The proposed norm avg approach in this work is conceptually simpler and yields superior results.
- **vs. Lumina/Sana**: They directly employ the final layer representation of Gemma2, which, according to the findings here, constitutes a suboptimal configuration.
- **vs. T5 Baseline**: As a bidirectional encoder-decoder model, T5's final layer naturally integrates complete bidirectional context. However, LLMs can mitigate the informational bottleneck of causal attention through multi-layer aggregation.

## Rating

- Novelty: ⭐⭐⭐⭐ The systematic study perspective is novel. Although layer-normalized average is simple, it is highly effective, and the core findings hold high value for the research community.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The study evaluates 27 models under a unified training configuration with 10-dimensional skill decomposition, exhibiting extremely rigorous control of variables.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear and visualization is abundant, although the data density in some tables is quite high.
- Value: ⭐⭐⭐⭐⭐ Provides a clear, practical guide for the text-to-image community on employing LLM text encoders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders](randar_decoder-only_autoregressive_visual_generation_in_random_orders.md)
- [\[CVPR 2025\] VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness](verbdiff_text-only_diffusion_models_with_enhanced_interaction_awareness.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[CVPR 2025\] Make It Count: Text-to-Image Generation with an Accurate Number of Objects](make_it_count_text-to-image_generation_with_an_accurate_number_of_objects.md)

</div>

<!-- RELATED:END -->
