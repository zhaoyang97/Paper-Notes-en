---
title: >-
  [Paper Note] Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points
description: >-
  [ACL 2025][LLM Alignment][DPO] Observing that errors in code generation models are highly concentrated on specific "error-prone points" where prefixes/suffixes remain almost identical while the middle segment determines correctness, this study proposes Focused-DPO. It ranks and locates key middle segments via PageRank on a code-test bipartite graph and magnifies their weights in the DPO loss ($w_{focused}=2$). With only 5,000 samples, it improves HumanEval+ by 4.41% and relat…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "DPO"
  - "Code Generation"
  - "Error-Prone Points"
  - "PageRank"
  - "Weighted Preference Optimization"
  - "Prefix-Suffix Matching"
date: 2026-05-08
content_hash: beb72a12b6581205
---

# Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points

**Conference**: ACL 2025  
**arXiv**: [2502.11475](https://arxiv.org/abs/2502.11475)  
**Code**: None  
**Area**: LLM Alignment / Code Generation  
**Keywords**: DPO, Code Generation, Error-Prone Points, PageRank, Weighted Preference Optimization, Prefix-Suffix Matching

## TL;DR

Observing that errors in code generation models are highly concentrated on specific "error-prone points" where prefixes/suffixes remain almost identical while the middle segment determines correctness, this study proposes Focused-DPO. It ranks and locates key middle segments via PageRank on a code-test bipartite graph and magnifies their weights in the DPO loss ($w_{focused}=2$). With only 5,000 samples, it improves HumanEval+ by 4.41% and relatively boosts LiveCodeBench-Hard by 42.86%.

## Background & Motivation

**Background**: Preference optimization methods such as DPO and RLHF have been widely applied to the post-training alignment of code generation models. Models like Qwen2.5-Coder and DeepSeekCoder achieve pass rates >90% on benchmarks like HumanEval through training with millions of SFT + DPO data points.

**Limitations of Prior Work**: Standard DPO treats all tokens with equal weight. However, the contributions of different code parts to correctness are highly uneven—function signatures, import statements, and return statements are almost always correct, whereas actual errors concentrate in the core area of algorithmic logic (the middle segment). Consequently, massive gradient signals are wasted on already correct tokens, failing to effectively improve the generation quality of key locations.

**Key Insight**: By sampling Qwen2.5-Coder-7B 20 times, the authors observed that the Phi correlation coefficient between the common prefix/suffix of the code and correctness is only 0.07–0.08 (almost uncorrelated), whereas the Phi coefficient of the error-prone middle segment reaches 0.57–0.61 (strongly correlated). Continuing generation from a correct middle segment yields a pass@1 of 90.02%, whereas continuing from an incorrect one yields only 3.17%.

**Key Challenge**: Error-prone points are crucial for final correctness, but methods like standard DPO and Step-DPO cannot distinguish the structural importance of different code positions, leading to low optimization efficiency.

**Mechanism**: Automatically locate error-prone points in code and scale up their weights within the preference optimization loss, enabling the model to focus on learning the most critical code decisions.

## Method

### Overall Architecture

A three-phase pipeline: (1) extracting programming concepts from real open-source repositories and synthesizing problem prompts; (2) using the policy model to generate both code and test cases, ranking via PageRank, and locating error-prone points through common prefix/suffix matching to construct a fine-grained preference dataset; (3) utilizing a modified weighted DPO loss to perform focused preference optimization training.

### Key Designs

1. **Error-Point Identification (Error-Prone Point Dataset Construction)**: For each problem prompt, the policy model samples $k=10$ code candidates and test cases with temperature=1.5. A code-test bipartite graph is constructed, where an edge is drawn if code $c_i$ passes test $t_j$. PageRank is iteratively applied to update the scores of code and test nodes until the rankings stabilize, and the tests passed by the highest-scoring code are selected as the ground truth. After classifying code into correct/incorrect sets, the common prefix and suffix of each (chosen, rejected) pair are matched to extract the middle differing segments (mid_chosen, mid_rej) as the error-prone points. The most discriminative pairs are selected by maximizing a Diff function (ranking difference + $\lambda$ × length of common segments), ultimately filtering and yielding 5,000 training and 1,000 validation samples.

2. **Weighted Loss Design for Focused-DPO**: The code is segmented into three parts: prefix, mid (focus), and suffix. In the reward function of the chosen sample, the mid segment is scaled by a weight $w_{focused}=2$, while prefix and suffix weights remain 1. For the rejected sample, the contribution of the suffix segment is discarded (as empirical findings indicate suffix is almost irrelevant to correctness). The finalized loss function simplifies to $\mathcal{L} = -\mathbb{E}[\log\sigma(\Delta_{mid} + \Delta_{suffix})]$, where $\Delta_{mid}$ is amplified by $w_{focused}$, concentrating gradient signals on the error-prone points.

3. **Self-Generation & Self-Verification PageRank Mechanism**: Unlike methods like Magicoder that directly treat all generated test cases as the ground truth, this work employs PageRank to iteratively filter out lower-quality test cases (where low-scoring test cases are automatically down-weighted). Consequently, even if the policy model's generation quality is inferior to GPT-4, a high-quality preference dataset can still be produced. Experiments show a 32% overlap in error-prone points among different models on the same problems, suggesting cross-model generalizability of these error-prone points.

## Key Experimental Results

### Main Results: HumanEval(+) / MBPP(+)

| Model | Method | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|------|-----------|------------|------|-------|
| Qwen2.5-Coder-Instruct-7B | Baseline | 91.5% | 84.1% | 82.8% | 71.4% |
| | +Focused-DPO | 92.7% | **87.8%** | 84.7% | **76.2%** |
| | +DPO/Step-DPO | 92.1% | 85.4% | 84.1% | 74.3% |
| | +Token-DPO | 92.7% | 87.2% | 83.3% | 75.1% |
| MagiCoder-S-DS-6.7B | Baseline | 73.2% | 68.3% | 76.7% | 66.7% |
| | +Focused-DPO | **82.3%** | **74.4%** | **79.4%** | **69.8%** |
| DeepSeekCoder-Instruct-6.7B | Baseline | 77.4% | 70.1% | 75.1% | 65.9% |
| | +Focused-DPO | **82.3%** | **73.2%** | **76.5%** | **66.9%** |

### LiveCodeBench Results by Difficulty

| Model | Method | Easy | Medium | Hard | Avg |
|------|------|------|--------|------|-----|
| Qwen2.5-Coder-Instruct-7B | Baseline | 69.2% | 22.0% | 3.4% | 31.2% |
| | +Focused-DPO | 73.5% | 24.2% | **4.8%** | 33.9% |
| | Gain | +6.2% | +10.0% | **+42.9%** | +8.4% |
| MagiCoder-S-DS-6.7B | Baseline | 48.1% | 10.7% | 0.1% | 19.3% |
| | +Focused-DPO | 51.3% | 11.8% | **1.9%** | 21.3% |
| | Gain | +6.6% | +10.1% | **+1752%** | +10.1% |

### Correlation Analysis Between Error-Prone Points and Correctness

| Code Segment | Frequency in Correct Code | Frequency in Incorrect Code | Phi Coefficient |
|--------|----------------|----------------|----------|
| Common Prefix | 0.791 | 0.733 | 0.068 |
| Common Suffix | 0.848 | 0.786 | 0.080 |
| Prefix + Chosen Mid | 0.637 | 0.091 | **0.565** |
| Prefix + Reject Mid | 0.012 | 0.558 | **-0.609** |

### Multi-Stage Training Compounding Effect

| Training Stage | HumanEval | HumanEval+ | MBPP | MBPP+ |
|----------|-----------|------------|------|-------|
| DeepSeekCoder-base-6.7B | 47.6% | 39.6% | 70.2% | 56.6% |
| + SFT (MagiCoder) | 73.2% | 68.3% | 76.7% | 66.7% |
| + First DPO (CodeDPO) | 83.5% | 76.2% | 80.7% | 70.9% |
| + Focused-DPO | **87.2%** | **79.3%** | **82.3%** | **72.8%** |

### Key Findings

- **Greater improvements on more rigorous benchmarks**: HumanEval+ (+4.41%) > HumanEval (+1.29%), and MBPP+ (+6.71%) > MBPP (+2.24%), indicating that the method truly improves code correctness rather than surface-level matching.
- **Most significant improvements on Hard problems**: LiveCodeBench-Hard relatively improved by 42.86% (Qwen2.5) and 1752% (MagiCoder), showing that the effect of error-prone points is more pronounced as difficulty increases.
- **Still effective for strong models aligned with large-scale SFT+DPO**: After Qwen2.5-Coder was aligned on millions of data points, Focused-DPO using only 5,000 samples yielded further improvements.
- **Compounding multi-stage gains are effective**: Applying Focused-DPO after the base $\rightarrow$ SFT $\rightarrow$ DPO pipeline consistently brings positive gains at each stage, displaying no diminishing returns.
- **Ablation Study**: $w_{focused}=2$ is optimal; too large (5) or too small (1) values lead to performance degradation. Discarding the suffix term in rejected samples shows positive effects.

## Highlights & Insights

- **Empirical observation of "errors concentrated in the middle" is simple yet powerful**: The pass@1 gap of 90% vs 3% quantifies the decisive role of error-prone points, offering a brand-new perspective for code alignment research.
- **PageRank self-verification mechanism**: Autonomously overcomes the data quality bottleneck without human annotations, enabling the policy model to self-bootstrap and generate high-quality preference datasets.
- **Exceptional data efficiency**: Using only 5,000 samples achieves significant improvements on Qwen2.5, which had already undergone million-scale alignment training, reflecting high practical utility.

## Limitations & Future Work

- Extends validation only to Python code generation. Other programming languages (such as C++, Java, Rust) may exhibit different error distribution patterns.
- $w_{focused}=2$ is a hardcoded value; future research can explore adaptive weight learning or dynamic adjustments according to problem difficulty.
- The prefix/suffix matching mechanism is relatively simplistic; more refined difference localization methods like git-diff, although showing slightly lower performance currently, have room for improvement.
- Generating the dataset requires multiple samplings ($k=10$) and test case executions, resulting in relatively high computational overhead during data construction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of quantitative observation of error concentration, PageRank self-verification, and position-weighted DPO is both novel and sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Encompasses 5 models × 5 benchmarks, correlation analyses, dual ablations on datasets/loss functions, and multi-stage training experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structured progression across three research questions, logically flowing from motivation proof to main results to ablation studies.
- **Value**: ⭐⭐⭐⭐ Significant gains on already-strong aligned models using just 5,000 samples. The method is simple, highly reproducible, and of direct reference value to code alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](rise_error_inject_preference.md)
- [\[ACL 2025\] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings](focalpo_enhancing_preference_optimizing_by_focusing_on_correct_preference_rankin.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)

</div>

<!-- RELATED:END -->
