---
title: >-
  [Paper Note] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models
description: >-
  [Image Generation] This paper formulates the retrieval of relevant and diverse LoRA combinations from a library of 100K+ adapters as a combinatorial optimization problem. It proposes LoRAverse, a framework based on submodular function maximization, which achieves relevance- and diversity-aware LoRA selection through concept extraction followed by submodular retrieval.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 5c6196d31ec7fcdf
---

# LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2510.15022](https://arxiv.org/abs/2510.15022)
- **Code**: Not released
- **Area**: Image Generation
- **Keywords**: LoRA model retrieval, submodular optimization, diversity, concept extraction, CivitAI

## TL;DR
This paper formulates the retrieval of relevant and diverse LoRA combinations from a library of 100K+ adapters as a combinatorial optimization problem. It proposes LoRAverse, a framework based on submodular function maximization, which achieves relevance- and diversity-aware LoRA selection through concept extraction followed by submodular retrieval.

## Background & Motivation

Platforms such as CivitAI host over 100K LoRA models, each fine-tuned for a specific style, theme, or character. Users face three challenges:

**Scale explosion**: Manually browsing and evaluating massive LoRA libraries is infeasible.

**Redundant selection**: Naive cosine-similarity retrieval returns highly similar models with poor diversity.

**Concept alignment**: User prompts contain multiple concepts, requiring precise association between individual LoRAs and their corresponding concepts.

The existing method Stylus employs top-K ranking with LLM-based filtering; however, top-K retrieval inherently favors similar models, and the LLM introduces input bias, collectively imposing an upper bound on diversity.

**Mechanism**: LoRA selection is formulated as monotone submodular function maximization, which guarantees that a greedy algorithm achieves a $(1-1/e)$ approximation ratio.

## Method

### Overall Architecture: Concept Extractor + Submodular Retriever

### 1. Concept Extractor

An LLM decomposes the user prompt into non-overlapping concept spans:

$$\mathcal{C}(s) = \{t_i | t_i \in \mathcal{T}(s), t_i \subseteq s\}$$

For example, "a British Shorthair cat playing in a cherry blossom garden" → ["British Shorthair cat", "cherry blossom garden"].

### 2. Submodular Retriever

**Relevance objective** (modular function, hence submodular):
$$\mathcal{F}_{\text{relevance}}(\mathcal{P}) = \sum_{a_i \in \mathcal{P}} \mathcal{F}_{\text{sim}}(\phi(a_i), \phi(s))$$

**Diversity objective** (submodular function exploiting clustering):
$$\mathcal{F}_{\text{diversity}}(\mathcal{P}) = \sum_{k=1}^{K} \log\left(1 + \sum_{a_i \in \mathcal{C}_k \cap \mathcal{P}} \mathcal{F}_{\text{reward}}(\phi(a_i))\right)$$

The concavity of $\log(1+\cdot)$ enforces diminishing returns: the marginal gain from selecting additional models within an already-covered cluster decreases monotonically.

**Joint objective**:
$$\mathcal{F}(\mathcal{P}) = \lambda_1 \mathcal{F}_{\text{relevance}}(\mathcal{P}) + \lambda_2 \mathcal{F}_{\text{diversity}}(\mathcal{P})$$

The paper proves the submodularity of this objective, and the greedy algorithm provides a $(1-1/e) \approx 0.63$ approximation guarantee.

### Safety Check

GPT-4o is employed as an adapter safety checker to filter LoRAs containing inappropriate content.

## Key Experimental Results

### Main Results (CFG=7, Realistic-Vision-v6 checkpoint)

| Method | CLIP↑ | TCE↑ | TIE↑ | I2I↓ | User Preference↑ |
|--------|-------|------|------|------|-----------------|
| SD v1.5 | 25.88 | 19.43 | 38.12 | 0.846 | 22.55% |
| Stylus | 25.41 | 20.30 | 38.53 | 0.825 | 29.90% |
| **LoRAverse** | 25.07 | **22.63** | **40.06** | **0.784** | **47.55%** |

- TCE (Truncated CLIP Entropy) improves by 16.5%: substantial gain in semantic diversity.
- TIE (Truncated Inception Entropy) improves by 5.1%: increased visual diversity.
- I2I decreases by 7.3%: higher inter-image dissimilarity.
- CLIP drops by only 3.1%: text alignment is largely preserved while diversity improves.

### Ablation Study: Retrieval Algorithm Comparison

| Method | CLIP↑ | TCE↑ | TIE↑ | I2I↓ |
|--------|-------|------|------|------|
| Cosine Similarity | **24.67** | 23.47 | 41.99 | 0.781 |
| **Submodular Retrieval** | 24.50 | **23.97** | **42.43** | **0.762** |

The submodular method outperforms across all diversity metrics, trading only a 0.7% CLIP reduction for comprehensive diversity gains.

### Inference Time Analysis

LoRAverse incurs approximately 26 s of additional overhead, predominantly from clustering (23.2 s). This cost is fixed and does not scale with batch size, making the marginal cost negligible in large-batch scenarios.

## Highlights & Insights

1. **Elegant mathematical formulation**: The intuitive goal of "relevant yet diverse" selection is precisely cast as submodular optimization with theoretical guarantees.
2. **Concept-level retrieval**: Rather than performing a single retrieval over the entire prompt, the framework decomposes concepts individually and merges the results.
3. **VLM-as-Judge**: GPT-4o is used to evaluate three-dimensional metrics covering diversity, quality, and text alignment.
4. **Robustness**: Results are insensitive to the number of clusters and concepts; the hyperparameters $\lambda_1=7.0$ and $\lambda_2=1.0$ suffice.

## Limitations & Future Work

- Clustering quality affects final results: if similar LoRAs are assigned to different clusters, effective diversity may be reduced.
- Combining multiple LoRAs may cause style drift, which requires debias prompts to mitigate.
- Retrieved LoRAs may amplify social biases present in training data.

## Related Work & Insights

- **LoRA retrieval**: Stylus, RAG-based methods
- **Submodular optimization**: Classic diversity-promoting subset selection
- **Diffusion model personalization**: DreamBooth, LoRA

## Rating
- **Novelty**: ★★★★☆ — The combination of submodular framework and concept extraction is original.
- **Technical Depth**: ★★★★☆ — A complete proof of submodularity is provided.
- **Practicality**: ★★★★☆ — Directly addresses real pain points faced by CivitAI users.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)

</div>

<!-- RELATED:END -->
