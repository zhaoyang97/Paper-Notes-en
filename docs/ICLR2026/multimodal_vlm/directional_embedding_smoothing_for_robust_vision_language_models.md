---
title: >-
  [Paper Note] Directional Embedding Smoothing for Robust Vision Language Models
description: >-
  [ICLR2026][Multimodal VLM][VLM safety] This paper extends RESTA (Randomized Embedding Smoothing and Token Aggregation) from LLMs to VLMs…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "VLM safety"
  - "jailbreak defense"
  - "randomized smoothing"
  - "embedding perturbation"
  - "directional noise"
date: 2026-05-08
content_hash: 380b69b97b88621d
---

# Directional Embedding Smoothing for Robust Vision Language Models

**Conference**: ICLR2026
**arXiv**: [2603.15259](https://arxiv.org/abs/2603.15259)  
**Code**: Not open-sourced  
**Area**: Multimodal VLM
**Keywords**: VLM safety, jailbreak defense, randomized smoothing, embedding perturbation, directional noise

## TL;DR
This paper extends RESTA (Randomized Embedding Smoothing and Token Aggregation) from LLMs to VLMs, demonstrating that directional embedding noise significantly outperforms isotropic noise in the safety-utility tradeoff, serving as a lightweight inference-time defense layer against multimodal jailbreak attacks.

## Background & Motivation
- The widespread deployment of vision-language models (VLMs) in agentic AI systems makes their safety and reliability increasingly critical.
- Despite safety alignment training, VLMs remain vulnerable to jailbreaking attacks, where adversaries bypass safety alignment through carefully crafted text-image inputs.
- Various defense strategies have been proposed, including perplexity filtering, repetition consistency detection, auxiliary guard models, and chain-of-thought safety reasoning, yet many purportedly strong defenses have subsequently been circumvented.
- RESTA, originally designed for LLMs (Hase et al., 2024) and inspired by randomized smoothing, enhances robustness by injecting noise into the embedding space and aggregating predictions via majority voting over multiple samples.
- This paper is motivated by naturally extending RESTA to the VLM setting and systematically evaluating the effect of different noise types.

## Core Problem
1. Can the RESTA defense mechanism be effectively transferred to multimodal VLM settings?
2. How much does the directionality of embedding noise (directional vs. isotropic) affect defense performance?
3. Is it possible to identify a reasonable operating point in the tradeoff between safety improvement and utility preservation?

## Method

### Extending RESTA to VLMs
- In many VLMs (e.g., LLaVA, Gemma), visual content is first processed by a vision backbone to extract patch-level features, then projected into the LLM input embedding space, forming a unified embedding sequence with text tokens: $\bm{e} = (e_1, \ldots, e_n) \in \mathbb{R}^{d \times *}$.
- This shared embedding space architecture allows RESTA to be directly extended by applying noise perturbations to the unified embedding sequence.

### Core Algorithm
- **Sampling stage**: Generate $k$ noisy copies of the input embedding sequence as $\tilde{\bm{e}}^i = H_\sigma(\bm{e})$.
- **Decoding stage**: At each step, perform greedy decoding on each of the $k$ copies to obtain candidate tokens, then select the final token via **majority vote**.
- **Selective perturbation**: Noise is applied only to token embeddings corresponding to user content; tokens from the system prompt and dialogue format templates are left unperturbed.

### Two Noise Types
1. **Isotropic (Normal) noise**: Adds independent Gaussian noise $\mathcal{N}(0, \sigma^2)$ to each dimension of the embedding vector.
2. **Hard directional noise**: Adds noise along the direction of the embedding vector, i.e., $e + \frac{ze}{\|e\|_2}$, where $z \sim \mathcal{N}(0, \sigma^2 d)$, aligning the noise with the original embedding vector.

### Intuition Behind Directional Noise
- The semantic information of an embedding vector is primarily encoded in its direction rather than its magnitude.
- Directional noise perturbs only the vector's norm without changing its direction, thereby better preserving semantic content.
- The normalization factor $\sqrt{d}$ is used to align the effective noise power between the two noise types.

## Key Experimental Results

### Experimental Setup
- **Models**: LLaVA-1.5-7B and Gemma-3-4B
- **Sample count**: $k=10$ perturbed embedding samples
- **Safety evaluation**: JailBreakV-28K benchmark (28K multimodal jailbreak attacks, 14 attack strategies × 2000 harmful queries)
- **Utility evaluation**: ScienceQA benchmark (4,241 multimodal multiple-choice questions)
- **Jailbreak judgment**: Automated ASR evaluation via Llama-Guard-3-8B

### Main Results

| Model | Noise Type | ASR (↓) | ScienceQA Acc (↑) | Notes |
|------|----------|---------|-------------------|------|
| LLaVA-1.5-7B | No defense | 50.13% | 64.07% | baseline |
| LLaVA-1.5-7B | Hard directional | **25.93%** | 61.42% | ASR halved, accuracy drops only 2.65% |
| LLaVA-1.5-7B | Isotropic | Worse | Worse | Tradeoff approaches trivial diagonal |
| Gemma-3-4B | Hard directional | Significantly reduced | Moderately preserved | Consistently outperforms isotropic |

### Key Findings
- **Directional noise uniformly outperforms isotropic noise**: The safety-utility tradeoff curve of directional noise is significantly superior to that of isotropic noise on both models.
- **Isotropic noise approaches trivial tradeoff**: The performance of isotropic noise is close to or worse than a naive random-refusal strategy (diagonal baseline).
- The importance of directionality is more pronounced than what Hase et al. (2024) observed for LLMs.

## Highlights & Insights
1. **Lightweight inference-time defense**: No model retraining is required; noise injection and majority voting at inference time alone achieve effective defense.
2. **Key insight on directional noise**: Reveals the importance of directional information in the embedding space for semantic preservation, providing a valuable guiding principle for future defense designs.
3. **Natural extension from LLM to VLM**: Leverages the shared embedding space for text and visual tokens in VLMs to seamlessly transfer RESTA.
4. **Large-scale and diverse evaluation**: Results are evaluated across 28K attack samples and 14 attack strategies, lending credibility to the findings.

## Limitations & Future Work
1. **Absence of adaptive attack evaluation**: Testing is conducted only on static benchmarks; adaptive attacks specifically designed against RESTA are not evaluated, leaving the true robustness uncertain.
2. **Weak theoretical foundations**: Although inspired by randomized smoothing, jailbreak attacks differ fundamentally from conventional adversarial examples (not restricted to small perturbations, with complex output spaces), and rigorous theoretical guarantees are lacking.
3. **Limited model coverage**: Only two relatively small models (7B and 4B) are tested; effectiveness on larger or proprietary VLMs remains unknown.
4. **Inference overhead**: Sampling $k=10$ copies implies approximately 10× the computational cost per inference.
5. **Only greedy decoding + majority vote explored**: Alternative aggregation strategies (e.g., logit averaging) are not investigated.

## Related Work & Insights

| Method | Type | Scope | Characteristics |
|------|------|----------|------|
| **RESTA (Ours)** | Inference-time embedding perturbation | VLM/LLM | Lightweight, training-free, directional noise is key |
| SmoothLLM (Robey et al., 2023) | Input-level character perturbation | LLM | Random token-level substitution/insertion/deletion |
| Llama Guard (Inan et al., 2023) | Auxiliary guard model | LLM | Requires extra model; filters inputs and outputs |
| Perplexity filtering (Alon et al., 2023) | Attack detection | LLM | Detects anomalous inputs without modifying model behavior |
| Safety reasoning (Rashid et al., 2025) | Chain-of-thought reasoning | LLM | Leverages CoT for safety-aware reasoning |
| Activation intervention (Zou et al., 2025) | Intermediate layer intervention | VLM | Modifies intermediate activations of the model |

RESTA's advantage lies in its implementation simplicity and independence from auxiliary models, though it lacks theoretical guarantees and adaptive attack validation compared to other approaches.

The finding that directional noise is effective while isotropic noise is not reinforces the hypothesis that embedding vector directions encode semantics, offering guidance for understanding and leveraging embedding spaces. The paper's conjecture that jailbreak attacks may exploit fragile "narrow pathways" in activations—thus susceptible to noise perturbation—if formalized theoretically, would carry significant implications for understanding VLM safety. The authors' pragmatic framing of RESTA as one layer within a broader security framework is a commendable stance, and the growing integration of VLMs into autonomous agentic systems will only increase the importance of inference-time defenses.

## Rating
- Novelty: ⭐⭐⭐⭐ (The extension of RESTA to VLMs is relatively straightforward, though the directional noise finding is valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Large-scale benchmarks, but lacks adaptive attack evaluation and broader model coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear exposition with candid and thorough discussion of limitations)
- Value: ⭐⭐⭐⭐ (Practical inference-time defense paradigm; the directional noise insight contributes to the field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](../../AAAI2026/multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](../../ICML2026/multimodal_vlm/circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)

</div>

<!-- RELATED:END -->
