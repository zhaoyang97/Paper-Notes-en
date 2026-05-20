---
title: >-
  [Paper Note] MMaDA: Multimodal Large Diffusion Language Models
description: >-
  [NeurIPS 2025][Image Generation][Diffusion language models] This paper presents MMaDA, the first multimodal foundation model that simultaneously achieves text reasoning, multimodal understanding…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion language models"
  - "unified multimodal architecture"
  - "UniGRPO"
  - "mixed long chain-of-thought"
  - "discrete diffusion"
date: 2026-05-08
content_hash: ff0cbd069ce051a1
---

# MMaDA: Multimodal Large Diffusion Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.15809](https://arxiv.org/abs/2505.15809)  
**Code**: [GitHub](https://github.com/Gen-Verse/MMaDA)  
**Area**: Diffusion Models / Multimodal Foundation Models
**Keywords**: Diffusion language models, unified multimodal architecture, UniGRPO, mixed long chain-of-thought, discrete diffusion

## TL;DR

This paper presents MMaDA, the first multimodal foundation model that simultaneously achieves text reasoning, multimodal understanding, and text-to-image generation within a unified discrete diffusion architecture. MMaDA bridges the gap between diffusion model pre-training and post-training through mixed long chain-of-thought (CoT) fine-tuning and the UniGRPO reinforcement learning algorithm.

## Background & Motivation

Multimodal large models are undergoing an architectural paradigm shift:

**Pure Autoregressive (AR)**: Models such as Emu3 and Janus unify all modalities via next-token prediction. The architecture is simple, but visual generation quality is limited.

**AR + Diffusion (two models)**: Models such as DreamLLM use AR for text and continuous diffusion for vision, requiring two separate models.

**AR + Diffusion (one model)**: Models such as Show-o and Transfusion mix AR and diffusion objectives within a single model, introducing complex hybrid mechanisms.

Nevertheless, existing unified multimodal models **severely lack exploration of post-training**—they are deployed directly after pre-training without CoT fine-tuning or reinforcement learning. This gap is especially pronounced in non-autoregressive architectures, as directly adapting AR-paradigm GRPO to diffusion models faces three major technical obstacles:
1. Token-level log-likelihood is valid only over masked regions.
2. The policy distribution is conditioned on the sampled mask rate.
3. Sequence-level likelihood cannot be accumulated via the autoregressive chain rule.

MMaDA's core positioning: unify all modalities with **pure (discrete) diffusion**, and for the first time systematically design a complete post-training pipeline for a diffusion foundation model.

## Method

### Overall Architecture

MMaDA's training pipeline consists of three stages:
- **Stage 1 (Pre-training)**: Unified diffusion objective with joint training on text generation, class-conditional image generation, and image–text understanding (600K steps).
- **Stage 2 (Mixed Long-CoT Fine-tuning)**: Fine-tuning on cross-modal CoT data to establish reasoning capabilities (50K steps).
- **Stage 3 (UniGRPO Reinforcement Learning)**: RL training with diverse reward signals (50K steps).

### Key Designs

1. **Unified Discrete Diffusion Architecture**: Built on LLaDA-8B as the backbone, MMaDA treats both text and images as masked prediction tasks over discrete tokens. Images are quantized into $32 \times 32 = 1024$ discrete tokens by MAGVIT-v2 (codebook size 8192). The unified training objective is:
    $$\mathcal{L}_{\text{unify}}(\theta) = -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^L \mathbf{I}[x_t^i=\texttt{[MASK]}] \log p_\theta(x_0^i | x_t)\right]$$
   The core advantage is that language and vision share exactly the same probabilistic formulation and architecture, with no modality-specific components.

2. **Mixed Long-CoT Fine-tuning**: A unified CoT format is designed as `|<special_token>|<reasoning_process>|<special_token>|<result>` to align reasoning processes across tasks. Data sources include:

    - Text reasoning: ReasonFlux, LIMO, OpenThoughts, etc.
    - Multimodal reasoning: correct responses from LMM-R1 on GeoQA/CLEVR.
    - Knowledge-aware image generation: scientific, cultural, and landmark description pairs synthesized by GPT-4.1.

   A key design choice is **transferring text reasoning capability to visual generation**—the model first performs textual reasoning (analyzing object attributes and spatial relationships) and then generates the image.

3. **UniGRPO**: A policy-gradient RL algorithm specifically designed for diffusion models, addressing three key challenges:

    - **Structured noise policy**: For each response $o_i$, a mask rate $p_i \in [0,1]$ is uniformly sampled with random seeds varied across gradient steps. This exposes the model to a wide range of denoising stages, from nearly fully masked to nearly fully unmasked.
    - **Efficient log-likelihood approximation**:
    $$\pi'_\theta = \frac{1}{M}\sum_{o_{i,t} \in M} \log p_\theta(o_{i,t} | q)$$
      Averaging over masked tokens avoids the 128-sample Monte Carlo estimation used in LLaDA.
    - **Uniform random masking**: Starting steps are sampled randomly, and the remaining denoising steps are distributed uniformly (rather than fully at random), approximating the Monte Carlo average and yielding more stable training.

### Loss & Training

**Diverse reward modeling**:
- Text reasoning: correctness reward 2.0 + format reward 0.5 (following `<think>...</think>` format).
- Multimodal reasoning: correctness + format + CLIP reward $0.1 \cdot \text{CLIP}(\text{image}, \text{text})$.
- Image generation: CLIP reward + ImageReward (human preference score), both scaled by 0.1.

**Sampling strategies**:
- Text generation: semi-autoregressive denoising (64-token blocks; 2 lowest-confidence tokens unmasked per step).
- Image generation: parallel non-autoregressive with cosine mask schedule, 50 denoising steps, CFG = 3.5.

## Key Experimental Results

### Main Results

**Multimodal Understanding Benchmarks**

| Model | POPE | MME | VQAv2 | GQA | MMMU | MMB | SEED |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 85.9 | 1510.7 | 78.5 | 62.0 | 35.4 | 64.3 | 58.6 |
| Show-o | 80.0 | 1097.2 | 69.4 | 58.0 | 26.7 | - | - |
| **MMaDA** | **86.1** | **1410.7** | **76.7** | **61.3** | **30.2** | **68.5** | **64.2** |

**Text-to-Image Generation**

| Model | WISE Cultural↑ | ImageReward↑ | CLIP Score↑ | GenEval Overall↑ |
|------|:---:|:---:|:---:|:---:|
| SDXL | 0.43 | 1.13 | 32.12 | 0.55 |
| Janus | 0.16 | 1.03 | 29.45 | 0.61 |
| Show-o | 0.28 | 0.92 | 28.94 | 0.53 |
| **MMaDA** | **0.67** | **1.15** | **32.46** | **0.63** |

**Text Reasoning**

| Model | Architecture | MMLU | GSM8K | MATH | ARC-C |
|------|------|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | AR | 64.5 | 53.1 | 15.1 | 53.1 |
| Qwen2-7B | AR | 70.3 | 80.2 | 43.5 | 60.6 |
| LLaDA-8B | Diffusion | 65.9 | 70.7 | 27.3 | 47.9 |
| **MMaDA-8B** | Diffusion | **68.4** | **73.4** | **36.0** | **57.4** |

### Ablation Study

| Stage | GSM8K | MATH500 | GeoQA | CLEVR | CLIP Score | ImageReward |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage 1 (Pre-training) | 17.4 | 4.2 | 8.3 | 10.3 | 23.1 | 0.69 |
| + Mixed Long-CoT | 65.2 | 26.5 | 15.9 | 27.5 | 29.4 | 0.84 |
| + UniGRPO | **73.4** | **36.0** | **21.0** | **34.5** | **32.5** | **1.15** |

| Masking Strategy | Performance | Notes |
|-------------|------|------|
| d1 (question masked + answer fully masked) | Slow convergence, low reward | Ignores the multi-step nature of diffusion models |
| Fully random mask rate | Unstable training | High reward variance |
| **UniGRPO (uniform random)** | **Fast convergence, high reward** | Approximates Monte Carlo average |

### Key Findings

- **Cross-modal synergy**: All metrics across the three tasks (text / understanding / generation) improve simultaneously during Stage 2 training; text reasoning capability directly improves the semantic accuracy of image generation.
- **Sampling efficiency**: Image generation maintains strong performance with as few as 15 denoising steps (CLIP 31.7 vs. 32.8 with the full 1024 steps); text generation converges at 256 steps.
- **Native support for inpainting tasks**: The diffusion model can be directly applied to text span prediction, VQA answer completion, and image inpainting without additional fine-tuning.

## Highlights & Insights

- **The first fully diffusion-based multimodal foundation model** to undergo systematic post-training, demonstrating that diffusion models can perform not only generation but also understanding and reasoning.
- UniGRPO's uniform random masking strategy is both efficient and stable, resolving the key technical obstacles in adapting GRPO to diffusion models.
- A substantial lead on the WISE Cultural benchmark (0.67 vs. 0.43 for SDXL) demonstrates that reasoning enhancement genuinely benefits knowledge-intensive image generation.
- The inherent parallel decoding and inpainting capabilities of diffusion models constitute structural advantages over AR models.

## Limitations & Future Work

- A performance gap relative to leading AR models on pure-text tasks remains (MMLU 68.4 vs. 70.3 for Qwen2-7B; GSM8K 73.4 vs. 80.2).
- Image resolution is limited to 512×512; high-resolution generation has not been explored.
- Multimodal understanding falls short of LLaVA-v1.5 on certain benchmarks (e.g., MME 1410.7 vs. 1510.7).
- Training cost is high (64 A100 GPUs), limiting scalability.
- Video generation and the unification of additional modalities remain unexplored.

## Related Work & Insights

- MMaDA forms a direct contrast with Show-o (AR + diffusion hybrid), demonstrating that a pure diffusion approach can match or surpass hybrid architectures.
- LLaDA lays the foundation for text diffusion models; MMaDA extends it to the multimodal domain and adds a complete post-training pipeline for the first time.
- UniGRPO complements the GRPO approach of RLVR-World (another paper in this batch)—the former targets discrete diffusion models, while the latter targets autoregressive video models.
- This work may inspire exploration of post-training paradigms for larger-scale diffusion foundation models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first to achieve a three-in-one system (text + understanding + generation) under a pure diffusion architecture with systematic post-training; a pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple benchmarks across three task categories with thorough ablations, though comparisons with larger models (e.g., 13B/70B) are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is presented clearly and systematically, though the overall paper structure is somewhat verbose.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for diffusion models as general-purpose foundation models; UniGRPO has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/image_generation/diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
