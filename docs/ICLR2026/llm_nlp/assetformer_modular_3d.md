---
title: >-
  [Paper Note] AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer
description: >-
  [ICLR 2026][LLM/NLP][modular 3D assets] This paper proposes AssetFormer, an autoregressive Transformer-based framework for modular 3D asset generation. By designing graph-traversal token ordering, token set modeling, and a SlowFast decoding strategy, it generates high-quality architectural assets composed of discrete primitives from text descriptions, and introduces the first large-scale real-world modular 3D dataset (16k real + 4k synthetic samples).
tags:
  - ICLR 2026
  - LLM/NLP
  - modular 3D assets
  - autoregressive Transformer
  - user-generated content
  - token ordering
  - speculative decoding
date: 2026-05-08
content_hash: d36a1a404c61218a
---

# AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer

**Conference**: ICLR 2026
**arXiv**: [2602.12100](https://arxiv.org/abs/2602.12100)
**Code**: [GitHub](https://github.com/Advocate99/AssetFormer)
**Area**: LLM/NLP
**Keywords**: modular 3D assets, autoregressive Transformer, user-generated content, token ordering, speculative decoding

## TL;DR
This paper proposes AssetFormer, an autoregressive Transformer-based framework for modular 3D asset generation. By designing graph-traversal token ordering, token set modeling, and a SlowFast decoding strategy, it generates high-quality architectural assets composed of discrete primitives from text descriptions, and introduces the first large-scale real-world modular 3D dataset (16k real + 4k synthetic samples).

## Background & Motivation

1. **Background**: Existing 3D generation methods adopt representations such as voxels, point clouds, neural fields, and meshes, yet face challenges in professional game production and UGC scenarios, including insufficient quality, large file sizes, and high barriers for non-expert users.

2. **Limitations of Prior Work**: Traditional 3D generation methods output dense meshes that are difficult to integrate directly into game engines; publicly available training data for modular 3D assets is absent; existing mesh generation methods such as MeshGPT require complex graph encoders.

3. **Key Challenge**: The game industry widely adopts modular design based on CSG principles, yet automated modular asset generation remains virtually unexplored.

4. **Goal**: Construct a framework capable of automatically generating modular 3D assets from text descriptions.

5. **Key Insight**: Modular assets are inherently sequences of discrete elements—each primitive carries category, rotation, and position attributes—making them well-suited for autoregressive modeling.

6. **Core Idea**: Treat 3D modular assets as ordered token sequences, determine the optimal ordering via graph traversal, and perform next-token prediction with a decoder-only Transformer.

## Method

### Overall Architecture
The input is a text description encoded by FLAN-T5 and projected into tokens. The model is based on the Llama architecture (312M parameters), with a joint vocabulary comprising 25 primitive categories + 4 rotation values + 3-dimensional positions = 214 tokens. The output is a token sequence decoded into 3D primitive parameters and rendered in a game engine.

### Key Designs

1. **Token Set Modeling**:
    - Function: Handles next-token prediction over a mixed vocabulary.
    - Mechanism: The finite discrete values of each primitive's five attributes $(c, r, x_0, x_1, x_2)$ are merged into a joint vocabulary $\mathcal{V}$. During inference, invalid logits are masked by attribute cycle and the remaining logits are renormalized.
    - Design Motivation: Using a joint vocabulary directly avoids multi-stage decoding and keeps the model simple.

2. **Token Re-Ordering**:
    - Function: Determines the optimal ordering of 3D primitives.
    - Mechanism: Starting from the bottom corner of an asset, DFS/BFS graph traversal visits all primitives to produce a permutation $\mathcal{A} = \{\tau_0, ..., \tau_{n-1}\}$. DFS marginally outperforms BFS and random ordering.
    - Design Motivation: Unlike text, 3D assets have no natural order; DFS guarantees local connectivity while maintaining a global bottom-to-top structure.

3. **SlowFast Decoding**:
    - Function: Accelerates inference without sacrificing quality.
    - Mechanism: A small model (AssetFormer-S, 87M) rapidly predicts simple tokens, while a large model (AssetFormer-B, 312M) handles complex tokens. The approach adapts the speculative decoding algorithm and incorporates token-type filtering.
    - Design Motivation: Many positional tokens in modular assets follow common patterns that a small model can predict efficiently.

### Loss & Training
- Standard cross-entropy loss with next-token prediction.
- Classifier-Free Guidance (CFG) scale = 2.0; conditions are randomly dropped with 10% probability during training.
- Top-$k$ sampling ($k=10$), temperature = 0.7.
- Dataset: 16k real samples (online UGC platform) + 4k PCG synthetic samples.

## Key Experimental Results

### Main Results

| Method | FID ↓ | CLIP ↑ |
|--------|-------|--------|
| PCG (algorithmic generation) | 108.476 | 0.319 |
| AssetFormer + Greedy | 63.351 | 0.319 |
| AssetFormer + Beam | 63.333 | 0.321 |
| AssetFormer + Top-K | **55.186** | 0.320 |
| Real data | / | 0.322 |

### Ablation Study

| Configuration | FID ↓ | Notes |
|---------------|-------|-------|
| Raw Order | 65.215 | No ordering leads to isolated components |
| RAR (random permutation) | 83.561 | Randomization strategy from image domain does not transfer to 3D |
| BFS | 61.620 | Effective but slightly inferior to DFS |
| DFS | **55.186** | Optimal ordering |
| Synthetic data only | 113.560 | Insufficient diversity |
| Real data only | 63.381 | Lacks structured foundation |
| Mixed data | **55.186** | Two data types are complementary |

### Key Findings
- Top-$k$ sampling achieves the best balance between quality and diversity.
- DFS ordering outperforms BFS and random ordering by preserving local connectivity.
- Synthetic and real data are complementary: synthetic data provides a structured foundation while real data contributes diversity.
- SlowFast decoding yields a 47% speedup (80.62 → 119.02 tokens/s) with negligible quality loss.

## Highlights & Insights
- First application of autoregressive Transformers to modular 3D asset generation.
- Key advantages of modular representation: lossless discretization, small file sizes, easy integration into game engines, and simplified texture mapping.
- Complements dense mesh methods such as MeshGPT: modular representation is particularly suited to regular, architectural assets.
- The data collection strategy is noteworthy: real UGC platform data + PCG synthesis + GPT-4o annotation.

## Limitations & Future Work
- Supports only text input; image-conditioned generation is not explored.
- The fixed discrete vocabulary is difficult to adapt to evolving design spaces.
- Validation is limited to architectural assets; extension to other modular categories such as furniture and vehicles is not addressed.
- Texture handling is delegated to post-processing rather than modeled end-to-end.

## Related Work & Insights
- **vs. MeshGPT**: MeshGPT generates dense meshes whereas AssetFormer generates modular assets; the two approaches are complementary.
- **vs. Hunyuan3D**: Native 3D generation methods perform poorly on interior architectural structures, and watertight preprocessing discards modular information.
- **vs. PCG**: PCG requires carefully handcrafted algorithms, whereas AssetFormer is data-driven and enables text-controlled generation.

## Rating
- Novelty: ⭐⭐⭐⭐ Autoregressive generation of modular 3D assets represents a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Practice-oriented with clearly articulated industrial application value.
- Value: ⭐⭐⭐⭐ Directly applicable to game UGC and 3D content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)
- [\[ICLR 2026\] Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)

</div>

<!-- RELATED:END -->
