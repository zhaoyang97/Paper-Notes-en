---
title: >-
  [Paper Note] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)
description: >-
  [ICLR 2026][LLM Safety][Membership Inference Attack] This paper presents the first systematic study of membership inference attack (MIA) vulnerabilities in diffusion language models (DLMs), proposing SAMA: a method that exploits DLMs' bidirectional masking structure to generate exponentially many probing opportunities, and handles sparse, heavy-tailed membership signals via progressive masking, sign voting, and adaptive weighting. SAMA achieves AUC of 0.81 across 9 datasets, outperforming the best baseline by 30%.
tags:
  - ICLR 2026
  - LLM Safety
  - Membership Inference Attack
  - Diffusion Language Model
  - Privacy Leakage
  - Robust Subset Aggregation
  - Progressive Masking
date: 2026-05-08
content_hash: 4bf10626bb5e5d9c
---

# Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)

**Conference**: ICLR 2026
**arXiv**: [2601.20125](https://arxiv.org/abs/2601.20125)
**Code**: [https://github.com/Stry233/SAMA](https://github.com/Stry233/SAMA)
**Area**: AI Security / Privacy Attacks
**Keywords**: Membership Inference Attack, Diffusion Language Model, Privacy Leakage, Robust Subset Aggregation, Progressive Masking

## TL;DR
This paper presents the first systematic study of membership inference attack (MIA) vulnerabilities in diffusion language models (DLMs), proposing SAMA: a method that exploits DLMs' bidirectional masking structure to generate exponentially many probing opportunities, and handles sparse, heavy-tailed membership signals via progressive masking, sign voting, and adaptive weighting. SAMA achieves AUC of 0.81 across 9 datasets, outperforming the best baseline by 30%.

## Background & Motivation

**Background**: Diffusion language models (DLMs, e.g., LLaDA/Dream) are an emerging alternative to autoregressive models (ARMs), using bidirectional masked token prediction. Existing MIA methods are designed for ARMs, leaving the privacy risks of DLMs completely uncharacterized.

**Limitations of Prior Work**:
   - ARM-based MIA methods (Loss/Min-K%/ReCall, etc.) applied directly to DLMs perform near-randomly (AUC ≈ 0.5)
   - Image diffusion MIA methods (SecMI/PIA) are also inapplicable (AUC ≤ 0.52)
   - Membership signals in DLMs are configuration-dependent — signals fluctuate drastically across masking configurations, with intra-sample variance (σ ≈ 0.10) exceeding the member/non-member margin (δ ≈ 0.06)
   - Domain adaptation effects introduce heavy-tailed noise, causing mean aggregation to collapse under extreme values

**Key Challenge**: DLMs' bidirectional structure provides exponentially many probing opportunities, yet the membership signal is extremely sparse and corrupted by heavy-tailed noise.

**Core Idea**: Progressive multi-density mask probing + sign voting to suppress heavy-tailed noise + adaptive weighting = robust MIA.

## Method

### Overall Architecture
Given a fine-tuned DLM $\mathcal{M}^T$ and a pretrained reference model $\mathcal{M}^R$, for a target text $\mathbf{x}$: (1) progressively increase masking density (5%→50%), sampling multiple mask configurations at each step; (2) compute local subset loss differences under each configuration and apply sign voting; (3) aggregate across densities with adaptive weighting to produce a final membership score $\phi \in [0,1]$.

### Key Designs

1. **Membership Signal Difference Between DLMs and ARMs**:

    - ARMs have a single fixed left-to-right prediction pattern → only one attack point
    - Each masking configuration $\mathcal{S}$ in a DLM constitutes an independent probe: $\Delta_{DF}(\mathbf{x};\mathcal{S}) = \ell_{DF}(\mathbf{x};\mathcal{S},\mathcal{M}^R) - \ell_{DF}(\mathbf{x};\mathcal{S},\mathcal{M}^T)$
    - Bidirectional context further enables probing of inter-token memorization relationships (e.g., jointly masking $x_i, x_j$ to test pair-level memorization)

2. **Robust Subset Aggregation (Core Contribution)**:

    - **Function**: Converts sparse noisy signals into robust votes
    - **Mechanism**: Randomly sample $N$ local subsets (each of $m=10$ tokens) from the masked positions, compute the loss difference $\Delta^n$ for each subset, binarize as $B^n = \mathbf{1}[\Delta^n > 0]$, and average over $N$ votes
    - **Theoretical Guarantee** (Hodges–Lehmann theorem): For non-members, the probability of $B^n=1$ is exactly 0.5 regardless of the noise distribution; true member signals consistently push votes toward 1. The sign test remains reliable even under infinite variance
    - This component accounts for the primary AUC improvement of 20–30%

3. **Progressive Masking**:

    - **Function**: Probes signals across multiple masking density levels
    - Masking density increases linearly: $\alpha_t = \alpha_{\min} + \frac{t-1}{T-1}(\alpha_{\max} - \alpha_{\min})$
    - Sparse masking: rich context → strong signal but few aggregation points; dense masking: many aggregation points but weaker individual signals and larger domain noise
    - Default: $T=16$ steps, $\alpha \in [5\%, 50\%]$

4. **Adaptive Weighting**:

    - $\text{Sama}(\mathbf{x}) = \sum_t w_t \hat{\beta}_t$, $w_t = \frac{1/t}{\sum_i 1/i}$
    - Earlier steps (sparse masking) receive higher weights as their signals are cleaner

### Loss & Training
- No training required — purely an inference-time attack method
- 16 queries per sample (aligned with baselines), $N=128$ subsets, $m=10$ tokens/subset

## Key Experimental Results

### Main Results: MIMIR Benchmark, 9 Datasets

| Dataset | SAMA AUC | Best Baseline AUC | TPR@1%FPR (SAMA) | TPR@1%FPR (Baseline) |
|--------|----------|-----------------|-----------------|-------------------|
| ArXiv | **0.850** | 0.597 | **0.178** | 0.023 |
| GitHub | **0.876** | 0.743 | **0.259** | 0.154 |
| HackerNews | **0.657** | 0.575 | **0.027** | 0.013 |
| PubMed | **0.814** | 0.555 | — | — |
| Wikipedia | **0.790** | 0.653 | — | — |
| Average | **~0.81** | ~0.62 | — | — |

### Ablation Study: Component Contributions

| Component | AUC Gain | Notes |
|------|---------|------|
| Baseline (Loss) | ~0.50 | Random-level |
| + Reference model calibration | +0.09~0.19 | Isolates fine-tuning-specific memorization |
| + Progressive masking | +2~3% | Multi-scale signal coverage |
| **+ Robust subset aggregation** | **+20~30%** | **Key: sign voting handles heavy-tailed noise** |
| + Adaptive weighting | +3~5% | Final refinement |

### Key Findings
- **Existing ARM MIA methods completely fail on DLMs**: AUC ≈ 0.50, confirming that DLMs require dedicated attack methods
- **Sign voting is the critical component**: Contributing 20–30% AUC improvement, with theoretical grounding via the Hodges–Lehmann theorem guaranteeing robustness to heavy-tailed noise
- **Advantage is more pronounced at low FPR**: TPR@0.1%FPR improves by up to 14×, which is highly significant for real-world deployment scenarios
- **Effective on both LLaDA-8B and Dream-7B**: Cross-architecture generalization demonstrated

## Highlights & Insights
- **First privacy attack study on DLMs**: Fills an important gap — as DLMs (LLaDA/Dream) grow in popularity, their privacy risks require systematic evaluation
- **Elegant use of sign voting to handle heavy-tailed noise**: Transforming continuous noisy signals into binary votes leverages the distribution-free robustness of sign statistics — a technique transferable to any heavy-tailed noise setting
- **DLMs' bidirectional structure is a double-edged sword**: It enables stronger language modeling, but also creates an exponentially large attack surface — every masking configuration is an independent privacy probing channel

## Limitations & Future Work
- **Gray-box assumption**: Requires access to logits from both the target and reference models; not applicable in black-box settings
- **Query overhead**: 16 queries per sample, which may be costly for large-scale auditing
- **Only fine-tuning scenarios tested**: Membership inference during pretraining remains unexplored
- **Defense directions**: "Masking configuration randomization" defenses could be designed — deliberately injecting configuration noise across queries to obscure membership signals

## Related Work & Insights
- **vs. Min-K%/ReCall (ARM MIA)**: These methods rely on a single left-to-right prediction pattern; DLMs' bidirectional structure renders them ineffective
- **vs. SecMI (image diffusion MIA)**: The continuous denoising process of image diffusion is fundamentally different from the discrete masking mechanism of text diffusion
- **vs. Purifying LLMs (same conference)**: That paper finds backdoors redundantly encoded in MLPs; SAMA finds privacy signals sparsely distributed across masking configurations — the two works reveal parameter-level characteristics along different security dimensions

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First DLM MIA study + innovative combination of sign voting to handle heavy-tailed noise
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets × 2 models × 10+ baselines × comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear logical chain from theoretical motivation to method to experiments
- Value: ⭐⭐⭐⭐⭐ Directly informs DLM privacy risk assessment and defense design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](../../NeurIPS2025/llm_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](../../CVPR2026/llm_safety/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](../../ACL2026/llm_safety/jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)

</div>

<!-- RELATED:END -->
