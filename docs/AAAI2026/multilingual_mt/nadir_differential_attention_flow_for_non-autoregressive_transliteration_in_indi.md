---
title: >-
  [Paper Note] NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages
description: >-
  [AAAI 2026][Multilingual & Machine Translation][Non-autoregressive models] This paper proposes NADIR, a non-autoregressive (NAR) multilingual transliteration architecture combining a differential Transformer with a Mixtu…
tags:
  - "AAAI 2026"
  - "Multilingual & Machine Translation"
  - "Non-autoregressive models"
  - "differential attention mechanism"
  - "mixture of experts"
  - "transliteration"
  - "Indic languages"
date: 2026-05-08
content_hash: d5934c488304acac
---

# NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages

**Conference**: AAAI 2026
**arXiv**: [2601.12389](https://arxiv.org/abs/2601.12389)
**Code**: None
**Area**: Natural Language Processing / Multilingual Transliteration
**Keywords**: Non-autoregressive models, differential attention mechanism, mixture of experts, transliteration, Indic languages

## TL;DR

This paper proposes NADIR, a non-autoregressive (NAR) multilingual transliteration architecture combining a differential Transformer with a Mixture-of-Experts (MoE) module. NADIR achieves over 13× inference speedup on Indic language transliteration tasks while substantially reducing hallucination errors common in NAR models (repetition, substitution, omission, and insertion), narrowing the accuracy gap with autoregressive counterparts.

## Background & Motivation

Transliteration converts text from one writing system to another while preserving pronunciation. Unlike translation, it maps phonetics rather than semantics. The Indic language family encompasses diverse scripts such as Devanagari (Hindi, Marathi), Bengali, and Punjabi, with a combined speaker population exceeding 1.6 billion. The task presents three key challenges: (a) character mapping ambiguity (many-to-one, one-to-many, and many-to-many), (b) phonetic variation — different words may transliterate to identical romanized forms, and (c) homophone and phoneme constraints — similar pronunciations correspond to different characters in different contexts.

Current state-of-the-art methods (e.g., IndicXLIT) employ autoregressive (AR) models, which, despite high accuracy, suffer from slow inference speeds (~77 words/sec), making large-scale real-time deployment impractical. NAR models generate all output tokens in parallel but suffer from severe hallucination problems in transliteration — including token repetition, substitution, omission, and insertion. Existing approaches to mitigating NAR quality degradation (knowledge distillation, iterative refinement, CTC loss) have not been applied to transliteration tasks.

The central research question is: **Can reducing attention noise and incorporating MoE help NAR models effectively capture context without autoregression?** The answer is affirmative.

## Method

### Overall Architecture

The NADIR (Non-Autoregressive Differential Intelligent Router) pipeline proceeds as follows:

1. **Preprocessing**: Input sequences are tokenized, combined with learnable token embeddings and RoPE (Rotary Position Encoding).
2. **Stacked Encoder**: Multiple encoder blocks, each consisting of a differential Transformer layer and an MoE routing module.
3. **Lightweight NAR Decoder**: An MLP-based non-autoregressive decoder that leverages the refined encoder representations to generate target script characters in parallel.

### Key Designs

1. **Multi-head Differential Attention**: In NAR settings, the absence of sequential inductive bias causes standard attention to struggle focusing on the most relevant input tokens, resulting in noisy attention maps. Differential attention mitigates this by computing the difference between two sets of normalized softmax attention scores:

$$\text{DiffAttn}(X) = \left( \text{S}\left(\frac{Q_1 K_1^\top}{\sqrt{d}}\right) - \lambda \text{S}\left(\frac{Q_2 K_2^\top}{\sqrt{d}}\right) \right) V$$

where $Q_1, Q_2$ and $K_1, K_2$ are two partitions of the query/key projections, and $\lambda$ is a learnable modulation parameter defined as $\lambda = \exp(\boldsymbol{\lambda}_{q_1} \cdot \boldsymbol{\lambda}_{k_1}) - \exp(\boldsymbol{\lambda}_{q_2} \cdot \boldsymbol{\lambda}_{k_2}) + \lambda_{\text{init}}$. The subtraction operation suppresses attention noise and enables the model to focus more precisely on relevant local context. Empirically, RMSNorm outperforms GroupNorm within the differential attention blocks.

2. **Mixture-of-Experts (MoE) Module**: Preliminary analysis of the differential Transformer revealed that languages with more training data perform better, indicating that a single shared FFN is insufficient to capture the diversity across languages and scripts. NADIR adopts a learnable-routing MoE framework with $M$ expert FFNs per layer, using Top-2 routing to select the two highest-scoring experts:

$$\text{MoE}(x) = p_i \cdot E_i(x) + p_j \cdot E_j(x)$$

Routing probabilities $p_i$ are computed via softmax over a trainable gating network $G(x)$. This design enables token-level dynamic computation and demonstrates improved robustness in multilingual settings.

3. **Implicit Sequence Termination**: NAR models cannot naturally predict EOS tokens as AR models do. NADIR appends an EOS token to each target sequence during training, computing loss only up to the first predicted EOS, allowing the model to implicitly learn sequence boundaries without a dedicated length prediction network.

### Loss & Training

The total training objective is a weighted sum of two terms:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{token}} + \beta \mathcal{L}_{\text{load}}$$

- **Token-level cross-entropy loss** $\mathcal{L}_{\text{token}}$: ensures local prediction accuracy.
- **Load balancing loss** $\mathcal{L}_{\text{load}}$: promotes uniform utilization across MoE experts and prevents routing collapse.

Optimal hyperparameters are $\alpha=0.8, \beta=0.2$. The model is trained with AdamW (learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-3}$), a linear learning rate scheduler (15% warmup), dropout 0.1, capacity factor 1.25, for 100 epochs.

## Key Experimental Results

### Main Results

Evaluation is conducted on the Aksharantar dataset, which covers 21 Indic languages with 24.8M training, 129.6K validation, and 180.1K test samples.

| Direction | Metric | NADIR | IndicXLIT (SOTA) | Difference |
|-----------|--------|-------|-----------------|------------|
| Roman→Indic | mean CER ↓ | 15.78% | 14.44% | +1.34% |
| Roman→Indic | mean WAcc ↑ | 50.13% | 51.23% | −1.10% |
| Roman→Indic | mean InfT ↓ | 8.95s | 116.48s | **13× speedup** |
| Indic→Roman | mean CER ↓ | 17.56% | 16.59% | +0.97% |
| Indic→Roman | mean WAcc ↑ | 34.50% | 36.29% | −1.79% |
| Indic→Roman | mean InfT ↓ | 9.07s | 124.18s | **13.7× speedup** |

NADIR outperforms IndicXLIT on both CER and WAcc for Telugu, Malayalam, Tamil, Kannada, and Sanskrit.

### Ablation Study

| Model Variant | mean CER ↓ | mean WAcc ↑ | Notes |
|--------------|-----------|------------|-------|
| Standard NAR | 21.88 | 38.98 | Baseline NAR model |
| Diff NAR | 16.12 | 46.89 | With differential attention |
| Diff MoE NAR (NADIR) | 15.78 | 50.13 | With differential attention + MoE |

Hallucination error breakdown (Roman→Indic direction):

| Error Type | Standard NAR | NADIR | Reduction |
|-----------|-------------|-------|-----------|
| Insertion | 28,454 | 23,654 | 16.87% |
| Substitution | 72,127 | 54,494 | 24.45% |
| Omission | 37,769 | 25,334 | 32.92% |
| Repetition | 6,313 | 3,186 | 49.53% |

### Key Findings

- Differential attention is the **primary contributor** to performance gains, reducing CER from 21.88 to 16.12 and substantially decreasing substitution, omission, and repetition errors.
- The MoE module addresses edge cases not covered by differential attention alone, particularly insertion errors (−14.55%) and repetition errors (further −22.78%), though it introduces approximately 8% more omission errors.
- NADIR maintains low latency across varying batch sizes, whereas IndicXLIT achieves peak performance only within a narrow batch size range.

## Highlights & Insights

1. **Precise Problem Formulation**: The paper explicitly defines "NAR hallucination" and categorizes it into four types (insertion, substitution, omission, repetition), providing a clear framework for systematically addressing NAR quality degradation.
2. **First Application of Differential Attention to NAR**: A mechanism originally developed to improve AR Transformer efficiency is repurposed here to address attention noise in NAR models. The subtraction operation effectively "carves out" ambiguous features, preserving sharp and precise representations.
3. **Linguistically Motivated MoE Design**: Starting from the observation that different languages require different processing, the paper first explores hard-coded routing before naturally transitioning to learnable MoE routing — a persuasive and principled design progression.
4. **Implicit Length Prediction**: The use of EOS tokens with truncated loss elegantly avoids an explicit length prediction network, eliminating a major source of instability.
5. **Strong Practical Value**: The 13× inference speedup makes large-scale multilingual transliteration deployment feasible (~1,005 words/sec vs. 77 words/sec).

## Limitations & Future Work

1. **Remaining Accuracy Gap**: NADIR's CER remains approximately 1–1.3 percentage points above IndicXLIT, requiring further improvement for precision-critical applications.
2. **Weak Performance on Low-Resource Languages**: Kashmiri (only 46K training samples) achieves a CER of 34.32%, far above the mean, indicating that MoE dynamic routing has not fully resolved data imbalance issues.
3. **MoE-Induced Omission Errors**: Although the MoE module reduces insertion and repetition errors, it increases omission errors by ~8%, potentially requiring finer-grained expert design.
4. **Validation Limited to Transliteration**: While the authors claim NADIR generalizes to tasks such as code refactoring and grammatical error correction, no experimental evidence is provided for these settings.
5. **Lack of Comparison with Other NAR Improvement Methods**: Iterative refinement approaches such as Mask-Predict and the Levenshtein Transformer are not included as baselines.

## Related Work & Insights

- **Differential Transformer (Ye et al. 2025)**: The primary inspiration for NADIR, reducing noise via dual-path attention subtraction.
- **MoE (Shazeer et al. 2017; Fedus et al. 2022)**: The Top-2 routing strategy from Switch Transformer is adopted in this work.
- **IndicXLIT (Madhani et al. 2023)**: The current SOTA autoregressive Indic transliteration model, serving as the primary baseline.
- **Broader Implication**: The combination of differential attention and MoE can potentially be generalized to other locally-dependent sequence tasks requiring high throughput.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First combination of differential attention and MoE for NAR transliteration, with clear problem definition and solution.
- **Technical Depth**: ⭐⭐⭐⭐ — Architecture design is linguistically motivated; ablation analysis is thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 20 languages with multi-dimensional error analysis.
- **Value**: ⭐⭐⭐⭐⭐ — The 13× speedup carries significant practical implications for real-world deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear motivation, though some sections are slightly verbose.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](../../ACL2026/multilingual_mt/beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](../../ACL2026/multilingual_mt/prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)

</div>

<!-- RELATED:END -->
