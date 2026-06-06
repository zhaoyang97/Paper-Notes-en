---
title: >-
  [Paper Note] Evolve to Inspire: Novelty Search for Diverse Image Generation
description: >-
  [NeurIPS 2025][Image Generation][Novelty Search] This paper proposes Wander, a framework that leverages novelty search and LLM-driven prompt evolution to generate highly diverse image collections from a single text promp…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Novelty Search"
  - "Image Diversity"
  - "Evolutionary Strategy"
  - "Prompt Optimization"
  - "CLIP"
date: 2026-05-08
content_hash: 06ea04d3ba84e5f2
---

# Evolve to Inspire: Novelty Search for Diverse Image Generation

**Conference**: NeurIPS 2025
**arXiv**: [2511.00686](https://arxiv.org/abs/2511.00686)  
**Code**: Not released  
**Area**: Image Generation
**Keywords**: Novelty Search, Image Diversity, Evolutionary Strategy, Prompt Optimization, CLIP

## TL;DR

This paper proposes Wander, a framework that leverages novelty search and LLM-driven prompt evolution to generate highly diverse image collections from a single text prompt, surpassing existing evolutionary prompt optimization baselines on the Vendi Score metric.

## Background & Motivation

Text-to-image diffusion models (e.g., FLUX, Stable Diffusion) excel at generating high-fidelity images but exhibit significant limitations in diversity—repeated use of the same prompt tends to produce visually similar results, and manual prompt adjustment yields unpredictable outcomes. This constitutes a bottleneck for applications requiring rapid generation of diverse ideas, such as creative exploration and brainstorming. Existing prompt optimization techniques (e.g., QDAIF, PromptBreeder) primarily target quality or NLP task performance rather than systematically improving visual diversity.

## Core Problem

1. How to automatically generate a highly diverse set of images from a single text prompt?
2. VLM-scoring-based methods such as QDAIF fail in the image domain—VLMs cannot reliably identify visual novelty.
3. How to design effective mutation strategies to guide evolutionary search across diverse regions of the prompt space?

## Method

### Overall Architecture of Wander

The framework operates through three core components iterated in sequence: **Emitter Selection**, **Prompt Evolution**, and **Pool Update**.

### Problem Formulation

Given a pool $\mathbf{P}$ containing at most $N$ prompt-image pairs $x_i = (p_i, I_i)$, a novelty score $f(x_i, \mathbf{P}) \in [0, 1]$ is defined. The objective is to maximize the minimum novelty score across the pool:

$$\mathbf{P}^* = \max_{\mathbf{P}} \left( \min_i (f(x_i, \mathbf{P})) \right)$$

### Novelty Metric

The novelty score is defined as the average $k$-nearest-neighbor cosine distance in CLIP embedding space:

$$f(x_i, \mathbf{P}) = \frac{1}{k} \sum_{j=1}^{k} d(I_i, I_j)$$

where $d(I_i, I_j)$ denotes the cosine distance between CLIP image embeddings.

### Emitter Mechanism

Emitters are predefined mutation strategies (e.g., "change composition," "adjust lighting," "add elements") embedded in mutation prompts to guide the LLM in varying prompts along specific directions. Emitters are selected via a bandit strategy or random sampling.

### Prompt Evolution

Each mutation applies one of two operations (each with 50% probability):
- **Mutation**: The LLM modifies a single prompt according to the emitter instruction.
- **Crossover**: The LLM combines elements from two existing prompts to produce a new variant.

### Pool Update

Candidate prompts are passed to the diffusion model to generate images; CLIP embeddings are then computed. A candidate replaces the pool entry with the lowest novelty score if its own novelty score is higher.

### Vendi Score Evaluation

$$\text{VS}(K) = \exp\left(-\sum_{i=1}^{n} \lambda_i \log \lambda_i\right)$$

where $\lambda_i$ are the eigenvalues of the normalized diversity matrix $K/n$, reflecting the effective number of diverse samples in the pool.

## Key Experimental Results

### Main Results (10 prompts × 10 runs)

| Method | Vendi↑ | LPIPS↑ | Relevance↑ | Token Usage↓ |
|--------|--------|--------|------------|--------------|
| EvoPrompt-DE | 1.42±0.04 | 0.51±0.01 | 0.292±0.001 | 38,243 |
| QDAIF | 1.80±0.02 | 0.51±0.02 | 0.297±0.001 | 43,464 |
| Lluminate | 3.29±0.02 | 0.75±0.01 | 0.210±0.070 | 175,902 |
| Wander-NE (no emitter) | 2.61±0.10 | 0.79±0.01 | 0.279±0.004 | 23,884 |
| **Wander** | **3.60±0.09** | **0.80±0.01** | 0.272±0.003 | **24,347** |

- Wander achieves a Vendi Score 9.4% higher than Lluminate while consuming only **1/7** of its token budget.
- Introducing multiple emitters improves the Vendi Score from 2.61 to 3.60 (+38%).

### LLM Model Comparison (20 generations)

| Model | Vendi Score↑ | Token Usage↓ |
|-------|-------------|--------------|
| GPT-4o-mini | 4.2±0.1 | 61,402 |
| GPT-4o | 4.8±0.1 | 78,067 |
| o3 | **5.2±0.1** | 236,081 |

Stronger LLMs yield greater diversity but at the cost of substantially higher reasoning token consumption.

## Highlights & Insights

- ⭐ The combination of novelty search and CLIP embedding distance is elegant and effective, requiring no fine-tuning of any model.
- ⭐ The emitter mechanism is well-designed; random sampling over multiple emitters alone yields substantial diversity gains.
- ⭐ The fixed-size pool design reduces token consumption by 7× compared to Lluminate's unbounded pool.
- The method is fully model-agnostic and transferable to arbitrary diffusion models.
- UMAP visualizations clearly demonstrate the expansion of diversity in embedding space throughout the evolutionary process.

## Limitations & Future Work

- The novelty objective occasionally causes semantic drift (approximately once every five runs); a relevance penalty could be incorporated to mitigate this.
- Emitters require manual design, which may constrain or bias the upper bound of achievable diversity.
- Aesthetic quality of generated images is not evaluated (no issues were observed with FLUX-DEV in practice).
- Validation is conducted only on a single diffusion model (FLUX-DEV); cross-model generalization remains to be confirmed.
- The Relevance score is slightly lower than QDAIF (0.272 vs. 0.297), indicating a trade-off in semantic preservation.

## Related Work & Insights

| Method | Initial Prompt | Objective | Evolution Strategy |
|--------|---------------|-----------|-------------------|
| APE | Multiple | Fitness | Crossover |
| QDAIF | Single | Quality-Diversity | Directed Mutation |
| Lluminate | Single | Novelty | Creative Strategy Mutation |
| **Wander** | **Single** | **Novelty** | **Directed Mutation + Crossover** |

- The framework is extensible to other modalities with embedding distance metrics, such as text and audio.
- The emitter concept can be combined with automated prompt engineering, where LLMs self-generate emitters.
- The approach can serve as a data augmentation strategy—generating diverse training images to improve generalization in downstream visual tasks.
- Novelty search methods may be applicable to jailbreak detection and red-teaming.

## Rating

- Novelty: ⭐⭐⭐⭐ (Applying novelty search to image generation is a novel and promising direction.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive multi-baseline comparisons, ablation studies, and LLM comparisons.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-defined problem formulation.)
- Value: ⭐⭐⭐ (Workshop paper with limited immediate impact, but an interesting research direction.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[NeurIPS 2025\] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](../../ICML2026/image_generation/unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
