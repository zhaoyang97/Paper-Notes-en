---
title: >-
  [Paper Note] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation
description: >-
  [ACL 2025][LLM (Other)][Non-Autoregressive Translation] This paper conducts a systematic empirical study on iterative refinement methods in Non-Autoregressive Translation (NAT). It compares the trade-offs between translation quality and inference speed across different refinement strategies (such as CMLM, DisCo, SUNDAE, etc.), reveals the critical impacts of iteration counts, mask ratios, and training strategies on the final performance, and provides comprehensive practical g…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Non-Autoregressive Translation"
  - "Iterative Refinement"
  - "Masked Language Models"
  - "Translation Quality"
  - "Decoding Efficiency"
date: 2026-05-08
content_hash: d3089ddbf277454a
---

# An Empirical Study of Iterative Refinements for Non-Autoregressive Translation

**Conference**: ACL 2025  
**Area**: Text Generation / Machine Translation  
**Keywords**: Non-Autoregressive Translation, Iterative Refinement, Masked Language Models, Translation Quality, Decoding Efficiency

## TL;DR
This paper conducts a systematic empirical study on iterative refinement methods in Non-Autoregressive Translation (NAT). It compares the trade-offs between translation quality and inference speed across different refinement strategies (such as CMLM, DisCo, SUNDAE, etc.), reveals the critical impacts of iteration counts, mask ratios, and training strategies on the final performance, and provides comprehensive practical guidance for NAT research.

## Background & Motivation

**Background**: Autoregressive translation models (such as Transformer) generate target translations token by token. Although they achieve high translation quality, their inference speed is limited by the sequence length. Non-autoregressive translation (NAT) accelerates inference by generating all tokens in parallel in a single step, but the initial generation quality is often poor. Therefore, iterative refinement has become the core means to improve NAT quality.

**Limitations of Prior Work**: Existing iterative NAT methods are highly diverse, including mask-based CMLM, discrete-diffusion-based DisCo and SUNDAE, insertion/deletion-based Levenshtein Transformer, etc. No fair and systematic comparisons exist among these different methods. The performance variations of each method across different datasets and iteration numbers are not fully understood, making it difficult for researchers to choose the most suitable strategy.

**Key Challenge**: The core contradiction of iterative refinement lies in the trade-off between refinement steps and inference speed—more iterations improve translation quality but diminish the speed advantage of NAT relative to autoregressive models. Furthermore, the optimal configurations for different refinement strategies in various scenarios remain unclear.

**Goal**: To systematically compare mainstream iterative refinement methods, analyze critical factors influencing refinement effectiveness, and provide practical recommendations on how to choose and configure iterative refinement strategies.

**Key Insight**: The authors establish a unified experimental framework, controlling variables such as model size, datasets, and evaluation metrics. Under this framework, ablation experiments are carried out on different refinement mechanisms to explore the impact of hyperparameters such as mask ratios, iteration counts, and sampling strategies.

**Core Idea**: Through large-scale controlled experiments, it is demonstrated that the performance variations among iterative refinement methods are mainly determined by the design of the refinement strategy rather than model capacity, and there exists a "sweet spot"—a small number of iterations can yield most of the quality improvements.

## Method

### Overall Architecture
This paper does not propose a new method but establishes a unified evaluation framework to systematically compare several categories of iterative refinement strategies. Given a source language sentence as input, the NAT model first generates an initial translation in parallel and then progressively improves the translation quality through multiple rounds of iterative refinement, finally outputting the refined translation.

### Key Designs

1. **Mask-Based Refinement (CMLM Series)**:

    - Function: Correct translation errors by randomly masking generated tokens and re-predicting them.
    - Mechanism: In each round of iteration, low-confidence tokens are selected for masking based on the model's prediction confidence, and then a Conditional Masked Language Model is used to re-predict these positions. The mask ratio gradually decreases from high to low (e.g., from 50% to 10%) to achieve coarse-to-fine correction.
    - Design Motivation: Low-confidence tokens are more likely to be translation errors. Prioritizing the correction of these locations can maximize the gain of each iteration.

2. **Discrete-Diffusion-Based Refinement (DisCo/SUNDAE)**:

    - Function: Model translation generation as a denoising process from noise to clear translation.
    - Mechanism: During training, different levels of noise (random token replacement) are applied to the reference translation, and the model learns to restore the correct translation given the source sentence and the noisy translation. During inference, it starts with a completely random sequence and progressively denoises to generate the translation. Compared to CMLM, discrete diffusion allows modifying all positions rather than just masked positions.
    - Design Motivation: The step-by-step denoising paradigm of diffusion models is naturally suited for iterative refinement and does not require explicit mask strategy selection.

3. **Edit-Operation-Based Refinement (Levenshtein Transformer)**:

    - Function: Dynamically adjust translation length and content through insertion and deletion operations.
    - Mechanism: Delete and insert operations are executed alternately—first using a classifier to mark tokens that need to be deleted, and then inserting new tokens between the remaining positions. This allows the translation length to change dynamically during iteration, overcoming the limitations of fixed-length NAT.
    - Design Motivation: Translation errors include not only substitution errors but also redundancies and omissions. Edit operations handle these issues more flexibly.

### Loss & Training
All methods are uniformly trained using cross-entropy loss, but their noise injection strategies during training differ: CMLM uses uniform random masking, discrete diffusion methods employ token replacement based on specific noise schedules, and Levenshtein Transformer utilizes optimal alignment based on edit distance to construct training signals.

## Key Experimental Results

### Main Results

| Method | WMT14 En-De (BLEU) | WMT16 En-Ro (BLEU) | Iterations | Speedup |
|------|-------------------|-------------------|---------|-------|
| Autoregressive Transformer | 27.5 | 34.1 | N/A | 1.0× |
| CMLM (10 iterations) | 27.0 | 33.3 | 10 | 2.5× |
| CMLM (4 iterations) | 26.5 | 32.8 | 4 | 4.2× |
| DisCo (10 iterations) | 26.8 | 33.1 | 10 | 2.3× |
| SUNDAE (10 iterations) | 26.7 | 33.0 | 10 | 2.4× |
| Levenshtein Transformer | 25.8 | 32.2 | ~5 | 3.8× |
| Single-step NAT (No refinement) | 22.5 | 29.5 | 1 | 15× |

### Ablation Study

| Configuration | BLEU (En-De) | Description |
|------|-------------|------|
| CMLM + Adaptive Mask Ratio | 27.0 | Full configuration |
| CMLM + Fixed 50% Mask | 26.2 | Fixed mask ratio drops by 0.8 |
| CMLM + Random Mask Selection | 26.0 | Non-confidence selection drops by 1.0 |
| Iteration 1→4 gain | +3.2 | First 4 iterations contribute the most |
| Iteration 4→10 gain | +0.5 | Last 6 iterations show diminishing returns |
| Knowledge Distillation + CMLM | 27.3 | Distilled data provides further improvement |

### Key Findings
- Diminishing marginal returns of iterative refinement are highly evident: the first 4 iterations contribute approximately 85% of the quality improvement, and exceeding 10 iterations yields almost no additional gain.
- The mask-based CMLM method performs best in most settings and features a simple implementation.
- Adaptive mask ratios (decreasing across iteration rounds) outperform fixed mask ratios, and selecting mask positions by confidence outperforms random selection.
- Knowledge distillation has a positive effect on all methods but brings the largest improvement to single-step NAT.
- Discrete diffusion methods perform better at lower iteration counts but converge with CMLM at higher iteration counts.

## Highlights & Insights
- The value of a unified evaluation framework lies in eliminating the confusion caused by differences in experimental setups across different papers, making the comparisons between methods fairer. This type of "benchmark paper" has significant guiding value for the development of the field.
- The discovery of the "4-iteration sweet spot" is of high practical value—in actual deployment, 4 iterations can achieve translation quality close to autoregressive models while maintaining approximately a 4× speedup.
- Confidence-guided mask selection is a general trick that can be transferred to other generation tasks (such as image generation, protein design).

## Limitations & Future Work
- The experiments focus mainly on standard WMT translation tasks. The applicability to low-resource language pairs and domain-specific translation needs to be validated.
- Comparisons with recently emerged translation methods based on large language models are not covered.
- The combination effects of iterative refinement strategies with decoding strategies like beam search are not fully explored.
- Future work can explore adaptive iteration methods that dynamically determine the number of refinement rounds based on input difficulty.

## Related Work & Insights
- **vs CMLM (Ghazvininejad et al., 2019)**: CMLM is the pioneering work of iterative refinement, and this paper validates that it remains competitive under a unified framework.
- **vs DisCo (Kasai et al., 2020)**: Discrete diffusion methods are conceptually more elegant but do not significantly outperform CMLM in terms of actual performance.
- **vs Diffusion Language Models**: Recent diffusion language models (such as MDLM) also adopt the iterative denoising idea, which is deeply connected to iterative refinement in NAT.

## Rating
- Novelty: ⭐⭐⭐ As an empirical study paper, no new method is proposed, but the systematic comparison itself is a valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple methods, multiple datasets, and exhaustive ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, experiments are detailed, and conclusions are well-supported.
- Value: ⭐⭐⭐⭐ Provides crucial practical guidance for NAT researchers and is valuable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)
- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)

</div>

<!-- RELATED:END -->
