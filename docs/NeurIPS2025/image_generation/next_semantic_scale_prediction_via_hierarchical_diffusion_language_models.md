---
title: >-
  [Paper Note] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion model] This paper proposes HDLM (Hierarchical Diffusion Language Model), which introduces cluster tokens with coarse-grained semantics as an intermediate hierarchy betw…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "discrete diffusion model"
  - "hierarchical vocabulary"
  - "semantic scale prediction"
  - "language modeling"
  - "CTMC"
date: 2026-05-08
content_hash: fff3559160fa8d8e
---

# Next Semantic Scale Prediction via Hierarchical Diffusion Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.08632](https://arxiv.org/abs/2510.08632)  
**Code**: [https://github.com/zhouc20/HDLM](https://github.com/zhouc20/HDLM)  
**Area**: Image Generation / Diffusion Language Model
**Keywords**: discrete diffusion model, hierarchical vocabulary, semantic scale prediction, language modeling, CTMC

## TL;DR

This paper proposes HDLM (Hierarchical Diffusion Language Model), which introduces cluster tokens with coarse-grained semantics as an intermediate hierarchy between clean tokens and mask tokens, enabling "next semantic scale prediction" in discrete diffusion language modeling. The method derives a closed-form ELBO, achieves consistently lower perplexity than MDLM/GIDD on OpenWebText, and reduces generation perplexity by 62% after stochastic perturbation.

## Background & Motivation

Autoregressive language models represent the current SOTA, yet the "next token prediction" paradigm cannot correct previously generated tokens. Discrete diffusion models have attracted attention for their progressive denoising and correction capabilities, and fall into two main categories:

**Masked discrete diffusion** (MDLM): all masked tokens share a single embedding, lacking rich semantics; decoded tokens cannot self-correct.

**Uniform discrete diffusion** (SEDD): uniform perturbation to random tokens leads to semantic inconsistency and weaker performance than masked diffusion.

GIDD provides a unified framework combining masked and uniform noise, but noisy tokens still lack rich semantics, and self-correction ability stems only from uniform noise, which in practice degrades performance.

**Key Challenge**: Masked diffusion lacks self-correction and produces semantically impoverished intermediate states; uniform diffusion suffers from semantic inconsistency and poor performance.

**Core Idea**: Inspired by the "next scale prediction" paradigm in visual VAR, HDLM introduces a semantic hierarchy into language tokens — cluster tokens (obtained by clustering pretrained embeddings) are inserted between word tokens and mask tokens. The forward process progressively maps tokens to more abstract ancestors, while the reverse process progressively predicts finer-grained semantics.

## Method

### Overall Architecture

HDLM is built on the CTMC framework. The hierarchical vocabulary consists of: word tokens → cluster tokens → mask token. The forward process follows $x \to c \to m$; the reverse process restores fine-grained tokens from abstract representations step by step.

### Key Designs

1. **Hierarchical Vocabulary Construction**:

    - Function: Establishes a semantic intermediate layer between standard vocabulary tokens and the mask token.
    - Mechanism: K-means clustering is applied over pretrained model embeddings to construct a surjective mapping. The optimal number of clusters is approximately the square root of the vocabulary size.
    - Design Motivation: Cluster tokens serve as "partially masked tokens with high-level semantics" — more informative than pure masks, more semantically consistent than random tokens, and uncertainty provides room for self-correction.

2. **Hierarchical CTMC Process**:

    - Function: Defines the forward and reverse processes for hierarchical diffusion.
    - Mechanism: The marginal distribution is $\text{Cat}(z_t;\, \alpha_t x + \beta_{t,c}\, c(x) + \beta_{t,m}\, m)$. The transition rate matrix has a block upper-triangular structure (word → cluster → mask), with mask as an absorbing state.
    - Design Motivation: The block structure ensures tokens only transition toward higher hierarchy levels; the reverse process achieves hierarchical decoding via Bayesian posteriors.

3. **Closed-Form ELBO (Theorem 3)**:

    - Function: Derives the training objective.
    - Mechanism: The ELBO decomposes into two cross-entropy losses — cluster tokens perform within-cluster word classification, and mask tokens perform cluster-level classification. The expectation of both weights is identically 1 (Theorem 4).
    - Design Motivation: This naturally induces curriculum learning from easy to hard. MDLM is a special case with $n=1$.

4. **Stochastic Perturbation ($\xi < 1$)**:

    - Function: During training, tokens are perturbed to an incorrect cluster with probability $1 - \xi$.
    - Design Motivation: Trains the model to recover correct tokens from inaccurate context, mitigating the train-test gap. At $\xi = 0.8$, generation perplexity drops by 62%.

### Practical Techniques

- **Force transition decoding**: Constrains cluster tokens to decode only into word tokens within the corresponding cluster.
- **Flexible weight clipping**: Clips extreme weights to stabilize optimization.
- **Hard training mode**: Replaces cluster-level CE with token-level CE; the slight performance drop validates the advantage of progressive denoising.

## Key Experimental Results

### Main Results

OpenWebText (DiT architecture, GPT-2 tokenizer, 131B tokens):

| Model | Params | Valid PPL | Gen PPL |
|-------|--------|-----------|---------|
| MDLM-small | 170M | 27.39 | 163.7 |
| GIDD+-small | 170M | 25.82 | 170.2 |
| **HDLM-small-64** | 170M | **23.36** | **144.2** |
| **HDLM-small-128** | 170M | **23.25** | 148.0 |
| **HDLM-base-128** | 425M | **19.22** | **139.9** |
| GPT-2 | 117M | 23.40 | - |

### Ablation Study

Effect of number of clusters:

| $n$ | Valid PPL | Gen PPL | Note |
|-----|-----------|---------|------|
| 1 (=MDLM) | 25.72 | 163.9 | Degenerates to MDLM |
| 64 | **23.36** | **144.2** | Best range |
| 128 | 23.25 | 148.0 | Best Valid PPL |
| 256 | 23.65 | 150.4 | Performance drops with too many clusters |

Effect of stochastic perturbation (HDLM-64):

| $\xi$ | Valid PPL | Gen PPL | Note |
|-------|-----------|---------|------|
| 1.0 | 23.36 | 144.2 | Standard |
| 0.9 | 23.54 | **69.76** | Gen PPL ↓ 51% |
| 0.8 | 25.93 | **54.15** | Gen PPL ↓ **62%** |

### Key Findings

- HDLM consistently outperforms MDLM/GIDD; the small model's Valid PPL drops from 27.39/25.82 to 23.25.
- The base model achieves Valid PPL of 19.22, competitive with autoregressive models.
- The optimal cluster count is approximately $\sqrt{|V|}$, dividing generation into two stages of roughly equal complexity.
- Stochastic perturbation is highly effective: Gen PPL drops dramatically from 144.2 to 54.15.
- MDLM is a special case of HDLM ($n=1$), verified both theoretically and empirically.
- Force transition remains effective under $\xi < 1$; contextual robustness proves more important than in-place error correction.

## Highlights & Insights

- "Next semantic scale prediction" pioneering transfers the multi-scale idea from visual VAR to the language domain.
- Theoretically rigorous: closed-form ELBO, weight invariance, and MDLM as a special case are all formally established.
- Stochastic perturbation yields a striking improvement (Gen PPL ↓ 62%) through a simple and effective mechanism.
- The curriculum learning interpretation of the ELBO is natural: cluster tokens perform within-cluster classification, while mask tokens perform cluster-level classification.

## Limitations & Future Work

- Experiments are conducted at a relatively small scale (170M/425M); performance at 7B+ remains to be verified.
- Only one intermediate hierarchy level is explored; multi-level experiments are left for future work.
- The static clustering strategy could potentially be replaced by a learnable mapping.
- Evaluation is limited to language modeling; effectiveness on downstream tasks is unknown.

## Related Work & Insights

- HDLM forms an evolutionary chain with MDLM and GIDD: masked → masked+uniform → hierarchical.
- Stochastic perturbation for mitigating the train-test gap parallels noise perturbation techniques in continuous diffusion models.
- The hierarchical vocabulary and curriculum learning design are transferable to other discrete generative tasks.
- Large-scale diffusion language models such as LLaDA and Dream suggest significant scaling potential for HDLM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](../../ICCV2025/image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[ICCV 2025\] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction](../../ICCV2025/image_generation/animegamer_infinite_anime_life_simulation_with_next_game_state_prediction.md)

</div>

<!-- RELATED:END -->
