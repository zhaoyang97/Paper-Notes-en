---
title: >-
  [Paper Note] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation
description: >-
  [ACL 2025][Text Generation][Sampling decoding strategy] This paper proposes a systematic evaluation framework based on Context-Preserving Prefix Trees (CP-Trie) to evaluate the intrinsic adaptability of truncation sampling methods between diversity and risk using probability-free and tuning-free metrics, providing practical guidance for parameter selection in real-world applications.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Sampling decoding strategy"
  - "Truncation sampling"
  - "Diversity-risk trade-off"
  - "Prefix tree"
  - "Parameter selection"
date: 2026-05-08
content_hash: d3dca7facaa212b5
---

# Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation

**Conference**: ACL 2025  
**arXiv**: [2408.13586](https://arxiv.org/abs/2408.13586)  
**Code**: [https://github.com/ZhouYuxuanYX/Benchmarking-and-Guiding-Adaptive-Sampling-Decoding-for-LLMs](https://github.com/ZhouYuxuanYX/Benchmarking-and-Guiding-Adaptive-Sampling-Decoding-for-LLMs)  
**Area**: Text Generation  
**Keywords**: Sampling decoding strategy, Truncation sampling, Diversity-risk trade-off, Prefix tree, Parameter selection

## TL;DR
This paper proposes a systematic evaluation framework based on Context-Preserving Prefix Trees (CP-Trie) to evaluate the intrinsic adaptability of truncation sampling methods between diversity and risk using probability-free and tuning-free metrics, providing practical guidance for parameter selection in real-world applications.

## Background & Motivation

**Background**: Large Language Models (LLMs) widely adopt sampling decoding strategies in open-ended text generation. These strategies, such as the truncation sampling methods Top-k and Top-p, balance the diversity and quality of the generated text through temperature adjustment and tail truncation. Recently, adaptive truncation methods such as Mirostat, η-sampling, and Adaptive Decoding have been proposed sequentially.

**Limitations of Prior Work**: Existing evaluations of sampling methods suffer from two fundamental limitations: (1) performance differences between methods may stem solely from the granularity of parameter tuning rather than the intrinsic capability of the methods; (2) users cannot determine optimal parameters in practical applications, and existing evaluations fail to address the practical demand of "which method to choose and which parameters to use."

**Key Challenge**: Evaluation heavily relies on specific parameter settings and limited exemplar texts, making it impossible to fairly compare the intrinsic capabilities of different sampling methods. Furthermore, n-gram models overestimate the data support size under a given prefix, which further degrades evaluation accuracy.

**Goal**: (1) To design a parameter-tuning-independent evaluation protocol to estimate the theoretical capability of truncation sampling methods; (2) to comprehensively compare existing sampling methods and provide a practical guide for parameter selection.

**Key Insight**: The authors observe that the optimal truncation position varies dramatically across different prefixes. Therefore, the key to adaptive truncation lies in adapting to variations in data support size. By constructing a prefix tree that preserves the full sentence context, the optimal truncation set can be estimated more accurately.

**Core Idea**: Utilizing the data support information provided by the Context-Preserving Prefix Tree (CP-Trie), this work defines probability-independent Recall and Risk metrics. Under fixed average Risk levels, the diversity (AR) and stability (RSE) of different methods are compared, thereby achieving a parameter-agnostic, fair evaluation.

## Method

### Overall Architecture
The input is the Wikipedia English corpus, and the output is the evaluation results of diversity and stability for different truncation sampling methods under various risk levels. The workflow includes: (1) converting the dataset into a sentence-level Context-Preserving Prefix Tree (CP-Trie); (2) defining an approximate optimal truncation set based on the CP-Trie; (3) designing probability-independent Recall and Risk metrics; and (4) evaluating each method under a fixed average Risk.

### Key Designs

1. **Context-Preserving Prefix Tree (CP-Trie)**:

    - **Function**: To construct a prefix tree that preserves the full sentence context for estimating the set of plausible tokens (data support) under a given prefix.
    - **Mechanism**: Constructing the prefix tree using sentences as the basic unit rather than traditional n-grams. The tree recursively collects child nodes starting from the "beginning of a sentence" while filtering out sentences containing rare proper nouns or invalid words. Ultimately, a CP-Trie containing 31.55 million leaf nodes is built from 6.45 million Wikipedia articles.
    - **Design Motivation**: Due to limited context windows, n-gram models severely overestimate the data support size, whereas CP-Trie preserves sentence-level full context, providing a tighter lower bound for data support.

2. **Probability-Independent Recall/Risk Metrics**:

    - **Function**: To quantify the diversity and quality of sampling methods on a single prefix node.
    - **Mechanism**: Recall is defined as the ratio of the truncated allowed set size to the approximate optimal set size (capped at 1), and Risk is defined as the proportion of the truncated set that lies outside the optimal set. This only checks whether tokens are within the data support and does not use probability values, thereby avoiding empirical probability bias and the interference of temperature parameters.
    - **Design Motivation**: Probability is unreliable as a quality metric—higher likelihood does not necessarily imply better quality, and empirical distributions suffer from sparsity issues.

3. **Parameter-Agnostic Evaluation Protocol (AR/RSE at fixed Risk)**:

    - **Function**: To compare the Average Recall (diversity) and Risk Standard Error (stability) of different methods under fixed average Risk levels.
    - **Mechanism**: Given a target average Risk (e.g., 1, 5, 15), a coarse-to-fine grid search is performed to determine the corresponding parameters for each method, after which AR and RSE are calculated under these parameters. Since the parameters are uniquely determined by the Risk level, the evaluation results reflect the intrinsic capabilities of the methods.
    - **Design Motivation**: The parameter ranges of different methods vary drastically (e.g., Top-p uses two decimal places, whereas η-sampling uses four decimal places), making direct comparison unfair.

### Loss & Training
This paper presents an evaluation framework rather than a training method; thus, it does not involve loss function design. The evaluation utilizes 8 models in total, including GPT-2-XL, the Llama-2/3 families, and the Mistral/Mixtral families.

## Key Experimental Results

### Main Results

| Method | AR@Risk=1 | RSE@Risk=1 | AR@Risk=5 | RSE@Risk=5 | AR@Risk=15 | RSE@Risk=15 |
|------|-----------|------------|-----------|------------|------------|-------------|
| Adaptive | **0.167** | 0.260 | **0.787** | 0.343 | **2.685** | 0.418 |
| Mirostat | 0.139 | **0.230** | 0.804 | **0.318** | 2.630 | 0.393 |
| Top-k | 0.128 | 0.228 | 0.576 | 0.290 | 1.701 | **0.346** |
| η-sampling | 0.445 | 0.181 | 2.112 | 0.271 | 6.009 | 0.373 |
| Top-p | 0.451 | 0.154 | 2.061 | 0.224 | 5.770 | 0.326 |

*Taking Llama-3-8B as an example, Bold indicates the best performance.*

### TruthfulQA Verification

| Method | Avg. Risk 1 | Avg. Risk 5 | Avg. Risk 15 |
|------|-------------|-------------|--------------|
| Greedy | 0.338 | — | — |
| Naive | 0.421(0.004) | — | — |
| Mirostat | **0.413**(0.010) | **0.425**(0.013) | **0.425**(0.009) |
| Adaptive | 0.395(0.012) | 0.424(0.011) | 0.421(0.009) |
| Top-k | 0.401(0.010) | 0.436(0.008) | 0.421(0.010) |
| Top-p | 0.355(0.013) | 0.378(0.011) | 0.389(0.012) |

### Key Findings
- Adaptive Sampling and Mirostat are the two best-performing methods in terms of balancing diversity and stability, while Top-p performs the worst.
- On TruthfulQA, the correlation between RSE and accuracy reaches up to -0.92, verifying the effectiveness of the stability metric.
- Sampling methods are undervalued in existing literature: in HumanEval, all sampling methods outperform greedy decoding, and Top-p and η-sampling also outperform greedy on GSM8K under low Risk settings.
- Larger models exhibit higher Recall under the same Risk level, though direct comparison across different model families is discouraged due to differing vocabulary sizes.

## Highlights & Insights
- **Ingenious CP-Trie Design**: Replacing the n-gram Trie with a sentence-level prefix tree preserves context integrity while controlling data volume demands. This "context-preserving" philosophy can be transferred to other evaluation scenarios requiring long-dependency modeling.
- **Evaluation as a Parameter Guide**: Serving simultaneously as a practical parameter recommendation system, Table 1 directly provides recommended parameter values for each method under different Risk levels, which holds high practical value.
- **Revealing the Undervaluation of Sampling**: Through controlled variables (fixed Risk levels), the work reveals that sampling methods are not inherently inferior to deterministic ones as claimed by Shi et al. (2024a); rather, the issue lies in sub-optimal parameter selection.

## Limitations & Future Work
- The work is solely based on English Wikipedia data and does not cover other languages.
- It does not include all sampling methods (such as Locally Typical Sampling or Min-P Sampling).
- The data support estimation of CP-Trie is still a lower bound, which may be insufficient for rare but plausible continuations.
- The framework can be extended to multilingual scenarios or combined with risk preferences of downstream tasks for automated parameter recommendation.

## Related Work & Insights
- **vs. LESS/LIMA (Data Selection Works)**: Although this paper does not perform data selection, its philosophy of "evaluating the intrinsic capability of a method rather than its performance under specific parameters" can be transferred to the evaluation of data selection methods.
- **vs. Shi et al. (2024a)**: While that study suggests deterministic decoding outperforms sampling, this work reveals that such a conclusion likely stems from improper parameter selection by controlling Risk levels.

## Rating
- Novelty: ⭐⭐⭐⭐ The evaluation framework is designed ingeniously but does not introduce a brand-new method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models, 5 methods, 3 Risk levels, TruthfulQA verification, and re-evaluation on 3 downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, solid motivation, and rich figures/tables.
- Value: ⭐⭐⭐⭐ Highly instructive for parameter selection in sampling decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)
- [\[ACL 2025\] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)
- [\[ICLR 2026\] p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding](../../ICLR2026/nlp_generation/p-less_sampling_a_robust_hyperparameter-free_approach_for_llm_decoding.md)

</div>

<!-- RELATED:END -->
