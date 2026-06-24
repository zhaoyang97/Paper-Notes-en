---
title: >-
  [Paper Note] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning
description: >-
  [ACL 2025][LLM Alignment] > PCPO introduces a token-level probability consistency metric into the preference pair selection stage. By selecting pairs where the answer is correct and the reasoning process is most "similar" to that of the incorrect response for DPO training, the model is forced to focus on key reasoning differences. This approach consistently outperforms IRPO/ScPO across multiple mathematical reasoning benchmarks.
tags:
  - "ACL 2025"
  - "LLM Alignment"
date: 2026-05-08
content_hash: 3e5eb9e622705e95
---

# Probability-Consistent Preference Optimization for Enhanced LLM Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.23540](https://arxiv.org/abs/2505.23540)  
**Code**: [https://github.com/YunqiaoYang/PCPO](https://github.com/YunqiaoYang/PCPO)  
**Institution**: CUHK MMLab, SenseTime Research, Shanghai AI Lab  

## TL;DR

> PCPO introduces a token-level probability consistency metric into the preference pair selection stage. By selecting pairs where the answer is correct and the reasoning process is most "similar" to that of the incorrect response for DPO training, the model is forced to focus on key reasoning differences. This approach consistently outperforms IRPO/ScPO across multiple mathematical reasoning benchmarks.

## Background & Motivation

Preference optimization (DPO and its variants) has become a mainstream method to enhance the mathematical reasoning capabilities of LLMs. The core step involves constructing chosen/rejected preference pairs:

- **IRPO**: Evaluates correctness using gold labels: correct $\rightarrow$ chosen, incorrect $\rightarrow$ rejected.
- **ScPO**: Uses self-consistency voting to select the majority answers as chosen and minority answers as rejected.

**Limitations of Prior Work**: Both methods focus solely on the final outcome (outcome-level), completely ignoring the **reasoning consistency** within the responses. Even with the same final answer, the reasoning paths can be entirely different. If randomly paired, it becomes difficult for the model to capture the key discrepancies between chosen and rejected responses.

**Design Motivation**: Can token-level conditional probability information be leveraged to measure the "intrinsic consistency" between two responses, thereby selecting the most informative preference pairs?

## Method

### Overall Architecture

PCPO is an iterative training framework, wherein each round consists of three steps:

1. **Generation & Candidate Pairing**: Use the current model to generate $N=16$ answers for each question, categorize them into $Y_w$ (correct) and $Y_l$ (incorrect) based on correctness, and filter out similar candidate preference pairs using Levenshtein distance.
2. **Probability Consistency Scoring & Preference Pair Selection**: Calculate a token-level probability consistency weighted score $s_w$ for candidate pairs, and select the pair with the highest score as the training pair.
3. **PCPO Loss Training**: Train the next-round model using a weighted DPO + NLL loss.

### Key Designs

#### Key Design 1: Token Probability Consistency Scoring

For a candidate preference pair $(y_w, y_l)$, a matching function (based on the Longest Common Subsequence of SequenceMatcher) is first used to align the common tokens of the two responses. For each matched token $y_t$, the consistency score is calculated as:

$$c_t(y_w | y_l) = \exp(-|\log P_w(y_t|x, y_{<t}) - \log P_l(y_t|x, y_{<t})|)$$

- $c_t \in [0, 1]$, where a value closer to 1 indicates that the conditional probability of the token in both responses is more similar.
- Intuition: High consistency implies that the reasoning paths of the two responses are highly similar "before making a mistake", indicating that differences are concentrated on critical decision points.

Then, aggregate into a pair-weighted score: $s_w(y_w|y_l) = \sum_t c_t(y_w|y_l) / l_{y_l}$

**Selection Strategy**: For each rejected response $y_l$, select the paired chosen response $y_w$ with the highest $s_w$ $\rightarrow$ this ensures that the reasoning paths between preference pairs are highly similar, keeping differences minimal yet critical.

#### Key Design 2: Levenshtein Distance Pre-screening

Directly calculating the probability consistency for all $p \times q$ candidate pairs is computationally expensive. PCPO first utilizes Levenshtein edit distance to select the top-k (k=8) most similar chosen answers for each rejected answer, significantly reducing computation. Experiments show that ranks 1-5 cover 95.4% of the final selected pairs.

#### Key Design 3: PCPO Loss Function

$$\mathcal{L}_{PCPO} = \underbrace{-s_w \cdot \log\sigma\left(\beta\log\frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta\log\frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} \underbrace{- \frac{\alpha \cdot s_w}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- Uses $s_w$ to dynamically weight each sample, allowing highly consistent pairs (which are more informative) to generate larger gradients.
- The NLL term prevents the model from deviating from overall language modeling capabilities ($\alpha=1, \beta=0.5$).

## Key Experimental Results

### Main Results: Multi-Model Multi-Benchmark Comparison

| Method | GSM8K Pass@1 | MATH-500 Pass@1 | OlympiadBench Pass@1 | AMC23 Pass@1 |
|:---|:---:|:---:|:---:|:---:|
| Llama3-8B Seed | 71.3 | 30.8 | 8.1 | 10.0 |
| IRPO M2 | 81.1 | 30.6 | 6.7 | 0 |
| ScPO M2 | 81.6 | 32.2 | 7.9 | 5.0 |
| **PCPO M2** | **82.8** | **33.2** | **9.5** | **10.0** |
| Mathstral-7B Seed | 84.3 | 57.2 | 21.8 | 25.0 |
| IRPO M2 | 87.7 | 58.4 | 24.6 | 20.0 |
| ScPO M2 | 87.6 | 60.4 | 24.1 | 27.5 |
| **PCPO M2** | **89.0** | **61.8** | **25.2** | **32.5** |

### Generalization: PCPO Data + Different DPO Variants

| Method | MATH-500 Pass@1 | OlympiadBench Pass@1 | AMC23 Pass@1 |
|:---|:---:|:---:|:---:|
| IPO M1 | 24.4 | 8.1 | 10.0 |
| **PCPO+IPO M1** | **32.2** | **9.9** | **15.0** |
| ORPO M1 | 27.0 | 8.0 | 10.0 |
| **PCPO+ORPO M1** | **29.0** | **8.6** | **10.0** |
| TDPO M1 | 29.8 | 7.7 | 5.0 |
| **PCPO+TDPO M1** | **30.4** | **8.4** | **5.0** |

PCPO's preference pair selection strategy can be applied in a plug-and-play manner to improve various DPO variants such as IPO, ORPO, and TDPO.

## Highlights & Insights

- **Token-level probability consistency is introduced into the preference pair selection stage for the first time**, departing from relying solely on outcome-level signals. This successfully extracts "hard" sample pairs with highly similar reasoning paths but different outcomes.
- **Universal Framework**: PCPO's data selection strategy can be combined with any DPO variant; experiments validate its generalizability across 5 different variants.
- **Intuitive Case Studies**: Specific cases demonstrate that PCPO can precisely distinguish and pair different reasoning patterns, whereas outcome-only methods pair randomly.

## Limitations & Future Work

- **Reliance on Gold Labels**: Constructing preference pairs requires knowing the correct answers, limiting its applicability to unlabeled data scenarios.
- **Additional Computational Overhead**: Calculating token probabilities increases total training time by approximately 15% (8.9 vs. 7.7 GPU-hours/iteration).
- **Validation Limited to Mathematical Reasoning**: The generalizability has not yet been verified on other reasoning tasks such as code generation or commonsense reasoning.
- **Mainly Evaluated on 7B Models**: No experiments were conducted on larger models (70B+), and the improvement on already highly-optimized models (like the Qwen2.5 series) is limited.

## Related Work & Insights

- **Preference Optimization for Mathematical Reasoning**: DPO $\rightarrow$ IRPO (iterative + answer correctness) $\rightarrow$ ScPO (self-consistency voting) $\rightarrow$ PCPO (token probability consistency)
- **Token-level Preference Optimization**: TDPO (token-level KL constraint), SparsePO (sparse token mask), cDPO (critical token identification) — these methods manipulate tokens during the optimization phase, whereas PCPO exploits token information during the data selection phase.
- **DPO Variants**: IPO (preventing overfitting), ORPO (reference-model-free), RLCD (contrastive distillation)

## Rating

⭐⭐⭐⭐ — Re-evaluating preference pair selection from the perspective of token probability is a novel and intuitive entry point. The experiments thoroughly cover 4 models $\times$ 4 benchmarks $\times$ 5 DPO variants. However, the method relies on gold labels and is restricted to mathematical reasoning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)
- [\[ACL 2025\] Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](teaching_an_old_llm_secure_coding.md)

</div>

<!-- RELATED:END -->
