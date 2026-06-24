---
title: >-
  [Paper Note] De-mark: Watermark Removal in Large Language Models
description: >-
  [ICML2025][LLM Safety][Watermark Removal] The De-mark framework is proposed, which estimates the n-gram watermark strength and reconstructs red-green lists through a random selection probing strategy. It enables watermark removal without requiring knowledge of the hash function, while providing theoretical guarantees on the distribution gap between the post-removal LM distribution and the original distribution.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Watermark Removal"
  - "LLM Watermarking"
  - "n-gram Watermark"
  - "Red-Green List"
  - "Watermark Spoofing"
date: 2026-05-08
content_hash: 43d9fc2a0b92a581
---

# De-mark: Watermark Removal in Large Language Models

**Conference**: ICML2025  
**arXiv**: [2410.13808](https://arxiv.org/abs/2410.13808)  
**Code**: [GitHub - De-mark](https://github.com/RayRuiboChen/De-mark)  
**Area**: AI Safety  
**Keywords**: Watermark Removal, LLM Watermarking, n-gram Watermark, Red-Green List, Watermark Spoofing

## TL;DR
The De-mark framework is proposed, which estimates the n-gram watermark strength and reconstructs red-green lists through a random selection probing strategy. It enables watermark removal without requiring knowledge of the hash function, while providing theoretical guarantees on the distribution gap between the post-removal LM distribution and the original distribution.

## Background & Motivation

### Key Challenge

**Key Challenge**: The robustness of n-gram watermarks (Kirchenbauer et al.), which embed detectable signals by biasing towards green tokens, is significantly overestimated. If adversaries can reverse-engineer the watermarking rules, the watermark can be effectively removed.

### Goal

**Goal**: Pioneering work by Jovanovic et al. requires knowledge of the underlying hash function and relies on paraphrasing tools to remove watermarks (which fails to preserve the original LM distribution).

### Improvements of De-mark

It completely eliminates the need for prior knowledge of the hash function, provides distribution-level theoretical guarantees, and enables watermark spoofing (making other models generate watermarked content).

## Method

### Random Selection Probing
By sending meticulously designed query sequences to the watermarked LM, the red-green list partition and the watermark strength delta for each n-gram context are reconstructed through token frequency statistics.

### Watermark Strength Estimation
An unbiased estimator is proposed to accurately evaluate the watermark offset delta, with theoretical guarantees for estimation consistency.

### Watermark Removal
Based on the estimated red-green list and delta, the token probability distribution is corrected during inference to restore the logit shift of green tokens.

### Theoretical Guarantees
It is proved that the KL divergence between the post-removal LM distribution and the original unwatermarked distribution is bounded.

### Watermark Spoofing
Reversely applying the estimated watermarking rules to another LM to generate watermarked content, demonstrating the vulnerability of watermarking schemes.

## Key Experimental Results

### Watermark Removal Performance (Llama3/Mistral)


### Main Results

| Model | Pre-removal z-score | Post-removal z-score | Text Quality Preservation |
|------|-------------|-------------|-----------|
| Llama3-8B | 4.2 | 0.3 | High |
| Mistral-7B | 3.8 | 0.4 | High |

### Comparison with Existing Methods


### Ablation Study

| Method | Requires Hash Function | Preserves Distribution | Effectiveness |
|------|-----------|---------|------|
| Jovanovic et al. | Yes | No | Medium |
| **De-mark** | **No** | **Yes (with theoretical guarantees)** | **High** |

### ChatGPT Case Study
Watermark removal is also successfully achieved on commercial-grade LLMs, validating the practical threat of the proposed method.

### Key Findings
1. Random selection probing is more efficient than frequency analysis.
2. Theoretical distribution guarantees ensure the naturalness of the post-removal text.
3. The capability of watermark spoofing reveals a severe security threat.
4. The proposed method remains effective across different n-gram lengths (1-4).

## Highlights & Insights

1. Unveils the fundamental vulnerabilities of watermarking schemes from an adversary's perspective.
2. Does not rely on any prior knowledge (such as hash functions or watermarking parameters).
3. Offers theoretical guarantees (bounded KL divergence), making the method trustworthy.
4. The watermark spoofing capability exacerbates the issue — adversaries can forge watermarks.
5. ChatGPT experiments demonstrate that commercial systems are also vulnerable.

## Limitations & Future Work

1. A relatively large number of queries are required to estimate the red-green list.
2. Applicability to language-aware or semantic watermarks has not been verified.
3. Ethical risks associated with watermark spoofing require community discussion.
4. Defense countermeasures (how to make watermarks more robust) have not been deeply explored.
5. Scenarios with overlapping multiple watermarks are not considered.

## Related Work & Insights

- Relationship with distortion-free watermarks: De-mark targets bias-based watermarks.
- Comparison with semantic watermarking: Semantic watermarks might be harder to remove.
- Insights: Watermark designs should consider robustness against adversarial probing.

## Rating
- Novelty: 4.5/5 — Watermark removal without relying on prior knowledge + theoretical guarantees
- Experimental Thoroughness: 4.5/5 — Evaluation on multiple models and commercial LLMs
- Writing Quality: 4.5/5
- Value: 5.0/5 — Crucial security warnings for LLM watermarking

## Supplementary Technical Details

### How Random Selection Probing Works
Sends carefully designed prefix sequences to the LM and observes which tokens are systematically preferred. By accumulating statistical signals through multiple probings, the red-green partition for each n-gram context is reconstructed.

### Theoretical Guarantees for KL Divergence
The KL divergence between the post-removal distribution and the original LM distribution shrinks as the delta estimation error decreases and the number of queries increases.

### Implications for Defense
It is recommended that watermark designs should consider robustness to random probing, such as using randomized hashing or multi-level watermarking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improved Unbiased Watermark for Large Language Models](../../ACL2025/llm_safety/improved_unbiased_watermark_for_large_language.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2025\] MorphMark: Flexible Adaptive Watermarking for Large Language Models](../../ACL2025/llm_safety/morphmark_adaptive_watermarking.md)
- [\[ICML 2025\] Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)
- [\[ICML 2025\] Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models](ferret_federated_full-parameter_tuning_at_scale_for_large_language_models.md)

</div>

<!-- RELATED:END -->
