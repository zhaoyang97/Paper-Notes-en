---
title: >-
  [Paper Note] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale
description: >-
  [ACL 2025][Rationale reduction] The VARR framework is proposed to identify and remove redundant sentences in reasoning paths on a sentence-by-sentence basis using a likelihood-based "verbosity" criterion, achieving an average accuracy improvement of 7.71% while reducing token generation by 19.87% across various reasoning tasks.
tags:
  - "ACL 2025"
  - "Rationale reduction"
  - "redundant sentences"
  - "verbosity"
  - "CoT fine-tuning"
  - "token saving"
date: 2026-05-08
content_hash: b50071dfd185688b
---

# Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale

**Conference**: ACL 2025  
**arXiv**: [2412.21006](https://arxiv.org/abs/2412.21006)  
**Code**: None  
**Area**: Others  
**Keywords**: Rationale reduction, redundant sentences, verbosity, CoT fine-tuning, token saving

## TL;DR

The VARR framework is proposed to identify and remove redundant sentences in reasoning paths on a sentence-by-sentence basis using a likelihood-based "verbosity" criterion, achieving an average accuracy improvement of 7.71% while reducing token generation by 19.87% across various reasoning tasks.

## Background & Motivation

LLMs improve the quality of final answers by generating lengthy intermediate reasoning steps, which inevitably increases inference costs and latency. Crucially, fine-tuning LLMs with complete reasoning paths does not guarantee performance gains, as some reasoning sentences may be redundant or even detrimental.

Existing rationale reduction methods suffer from two key limitations: (1) They perform reduction at the token level (e.g., ICoT-SI), which lacks linguistic rationality and can disrupt sentence semantics. (2) They lack principled criteria to determine what should be removed, relying mostly on heuristics. Furthermore, these methods are primarily validated on simple arithmetic tasks (e.g., multi-digit multiplication), lacking generalizability.

## Method

### Overall Architecture

VARR comprises three stages: (1) a warm-up stage using normal CoT fine-tuning with complete reasoning paths; (2) a verbosity evaluation stage that examines each reasoning sentence from front to back; and (3) a continued training stage after removing sentences based on the verbosity criterion.

### Key Designs

1. **Empirical finding of higher redundancy in early reasoning sentences**: By calculating the change in answer NLL after removing sentences at different positions, the authors discover that removing prior sentences has the minimal impact (marginal NLL difference), indicating early sentences contribute the least to generating the correct answer. This provides empirical justification for the "front-to-back removal" strategy.

2. **Definition of Verbosity**: $verbosity(y_g) = \log(p(y_g|R',x) / p(y_g|R,x))$. In essence, it is the difference in KL divergence, measuring whether the probability of the correct answer increases after removing sentence $r_i$. A $verbosity \ge 0$ indicates that the correct answer probability does not decrease after removal, meaning the sentence can be safely removed.

3. **Enhancement with Wrong Answers (VARR+)**: $verbosity(y_w) = \frac{1}{K} \sum \log(p(y_w^k|R',x) / p(y_w^k|R,x))$, where $K$ wrong answers are sampled using in-batch negative samples. When $verbosity(y_w) - verbosity(y_g) \le 0$, it indicates that removing the sentence yields a greater probability gain for the correct answer than for the wrong answers, further confirming that safe removal is warranted.

4. **Linear Removal Schedule**: $r(t) = \lfloor N_t \cdot (t/T) \rfloor$, which gradually increases the upper limit of removable sentences as training progresses, though actual removal remains constrained by the verbosity criterion (no forced removal).

### Loss & Training

The standard CoT training loss is $-\log p(y_g, R|x)$. The warm-up stage accounts for 10% of total training steps. The optimizer is re-initialized at the start of each epoch to stabilize training.

## Key Experimental Results

### Main Results — Mistral 7B

| Method | MathQA | GSM8K | CommonQA | TriviaQA | StrategyQA |
|------|--------|-------|----------|----------|------------|
| Explicit-CoT | 55.84 | 55.26 | 84.33 | 82.94 | 74.70 |
| ICoT-SI | 35.84 | 28.27 | 67.82 | 77.09 | 61.33 |
| Coconut | - | - | - | - | - |
| **VARR+** | **56.95** | **54.98** | **89.56** | **83.45** | **78.19** |

(VARR+ achieves an average gain of 7.71% and a token reduction of 19.87%)

### Ablation Study — Comparison of Reduction Units

| Method | Average Accuracy | Average Tokens |
|------|----------|----------|
| ICoT-SI (token, no criterion) | Lowest | Low |
| VARR-Tok (token + verbosity criterion) | Medium (+24.74% vs ICoT-SI) | Medium |
| **VARR-Sent (sentence + verbosity criterion)** | **Highest (+15.98% vs VARR-Tok)** | Lower |

### Ablation Study — Removal Positions

| Removal Position | Average Accuracy |
|---------|----------|
| No Rule (random uncontrolled) | Lowest |
| Random (random + verbosity) | Medium |
| Back (backward + verbosity) | Medium |
| **Front (forward + verbosity)** | **Highest** |

### Key Findings

1. **Sentence >> Token as the Reduction Unit**: Sentence-level removal yields a 15.98% performance improvement over token-level removal because token-level removal can truncate semantic meaning.
2. **Verbosity Criterion is Crucial**: Even at the token level, introducing the verbosity criterion brings a 24.74% improvement over ICoT-SI.
3. **Severe Performance Degradation of ICoT-SI and Coconut**: Average performance drops by 21.98% and 25.20%, respectively, indicating that heuristic removal methods damage reasoning capabilities.
4. **Contrast with Wrong Answers in VARR+ Further Enhances Robustness**: VARR+ outperforms VARR on most datasets.
5. **Adaptive Actual Removal Rate**: Most redundant sentences are removed during early training stages, and the removal automatically stabilizes in the later phase, illustrating the self-regulating capability of the framework.

## Highlights & Insights

- Core Insight: Redundancy in reasoning paths is not randomly distributed but is concentrated in the early stages. This might be because LLMs generate a large volume of "contextual paving" sentences in the initial stages of reasoning, while the actual reasoning is completed in the subsequent steps.
- Formalization of the verbosity concept (KL divergence difference $\rightarrow$ likelihood ratio) is elegant and highly interpretable.
- Overcomes the limitations of prior methods being validated only on simple arithmetic tasks, covering various tasks such as mathematical reasoning, commonsense reasoning, and reading comprehension.

## Limitations & Future Work

- Experiments are only conducted on 7B models with small batch sizes (single A100 GPU).
- Evaluation on long-sequence reasoning tasks is not investigated.
- The computation of verbosity requires extra forward passes (once each for $R$ and $R'$), adding to the training time complexity.
- The linear removal schedule might not be optimal; adaptive scheduling could be explored.

## Related Work & Insights

- Complements methods that "improve performance through more reasoning" (such as Self-Consistency and Tree of Thoughts)—this work demonstrates that "reducing reasoning can also improve performance."
- The token-level removal concept of ICoT-SI is systematically proven to be sub-optimal by this study.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of the verbosity concept and sentence-level reduction represents a novel methodological contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across 5 datasets, multiple ablation studies, and cross-model scenarios (Mistral + Llama3.2).
- **Writing Quality**: ⭐⭐⭐⭐ — Delivers clear theoretical derivations and highly rigorous experimental logic.
- **Value**: ⭐⭐⭐⭐⭐ — Methods that simultaneously improve performance and efficiency are highly attractive in practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](../../ICML2025/others/randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](../../ECCV2024/others/domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ACL 2025\] HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)
- [\[ICLR 2026\] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](../../ICLR2026/others/revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)

</div>

<!-- RELATED:END -->
