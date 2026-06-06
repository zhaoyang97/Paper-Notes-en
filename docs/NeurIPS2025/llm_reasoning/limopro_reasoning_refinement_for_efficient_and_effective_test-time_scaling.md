---
title: >-
  [Paper Note] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling
description: >-
  [NeurIPS 2025][LLM Reasoning][reasoning refinement] This paper proposes PIR (Perplexity-based Importance Refinement), a framework that categorizes reasoning chains distilled from LRMs into "progressive reasoning" and "fu…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "reasoning refinement"
  - "chain-of-thought"
  - "perplexity-based pruning"
  - "test-time scaling"
  - "efficient reasoning"
  - "PIR"
date: 2026-05-08
content_hash: 5843fb3192eb594f
---

# LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2505.19187](https://arxiv.org/abs/2505.19187)  
**Code**: [LIMOPro](https://github.com/) (mentioned in the paper)  
**Area**: LLM Reasoning / Efficient Inference
**Keywords**: reasoning refinement, chain-of-thought, perplexity-based pruning, test-time scaling, efficient reasoning, PIR

## TL;DR
This paper proposes PIR (Perplexity-based Importance Refinement), a framework that categorizes reasoning chains distilled from LRMs into "progressive reasoning" and "functional steps" (verification / multi-method validation / error correction), and prunes only functional steps with low PIR scores while preserving the progressive reasoning backbone intact. Fine-tuning on the refined data improves accuracy by 0.9%–6.6% on AIME/AMC/GPQA while reducing token usage by 3%–41%, yielding up to 71% efficiency gain.

## Background & Motivation
- **Background**: Large reasoning models such as DeepSeek-R1 and QwQ generate CoT chains containing extensive functional steps — verification, error correction, and multi-method validation — that simulate human problem-solving but substantially increase inference overhead.
- **Limitations of Prior Work**: Using such verbose reasoning chains for SFT transfers the same redundant reasoning behavior to student models, significantly raising inference time and compute cost. Existing approaches (e.g., SPIRIT) apply uniform perplexity-based pruning without distinguishing step types, risking the removal of critical progressive reasoning steps and degrading accuracy.
- **Key Challenge**: The heterogeneous importance of functional steps is ignored: different instances of the same step type (e.g., verification) contribute very differently to the final answer, requiring quantitative assessment rather than heuristic deletion.
- **Goal**: Achieve a favorable efficiency–quality trade-off for practical test-time scaling deployment, and provide a generalizable framework across diverse distillation sources (Gemini: 71.4% progressive reasoning vs. DeepSeek-R1: 59.7%).

## Method
The PIR framework consists of a four-stage pipeline:

### 1. Reasoning Chain Segmentation and Classification
- Claude 3.7 Sonnet segments reasoning chains into logical steps, each comprising multiple coherent sentences.
- A two-stage classification is applied: rule-based matching of linguistic markers ("Let me check" → verification; "I made a mistake" → error correction), followed by Claude-based contextual analysis for steps lacking explicit markers.
- Four reasoning patterns are defined: **progressive reasoning** (forward chained inference, always retained), **verification** (checking existing computations), **multi-method validation** (re-solving via an alternative approach), and **error correction** (fixing identified mistakes).
- Human validation: 5% of steps are randomly sampled and independently evaluated by four graduate students, yielding 93.4% classification accuracy.

### 2. PIR Score Computation
- **Core Idea**: The greater the increase in answer perplexity upon removing a step, the more important that step is.
- $\text{PIR}(x_i) = \log\!\left(\frac{\text{PPL}(R \setminus \{x_i\})}{\text{PPL}(R)}\right)$
- Perplexity is computed using Qwen2.5-32B-Instruct to measure the change in model confidence over the correct answer after removing step $i$.
- A higher PIR value indicates greater importance (removal causes a sharp drop in answer confidence).

### 3. Selective Pruning
- **Core Principle**: All progressive reasoning steps are retained in full; only functional steps are ranked by PIR score and pruned.
- Functional steps with the lowest PIR values are removed according to a preset ratio threshold.
- The optimal pruning ratio lies in the range 0.2–0.3; excessively high ratios (e.g., 0.8) achieve the greatest length reduction but incur accuracy degradation.

### 4. Dataset Construction and Fine-tuning
- PIR is applied to three datasets: LIMO (distilled from DeepSeek-R1), S1K (distilled from Gemini), and LIMO-V2 (distilled from QwQ).
- Refined variants LIMO-P, S1K-P, and LIMO-V2-P are constructed for SFT.

## Key Experimental Results

| Model | AIME ACC↑ | AIME TOK↓ | AIME EFF↑ | AMC ACC↑ | GPQA ACC↑ | GPQA TOK↓ |
|-------|-----------|-----------|-----------|----------|-----------|-----------|
| S1-32B | 37.9 | 6646 | 5.71E-5 | 80.9 | 60.7 | 4172 |
| **S1-32B-P** | **42.1**(+4.2) | **4716**(-29%) | **8.92E-5**(+56%) | **83.1**(+2.2) | **61.6**(+0.9) | **2472**(-41%) |
| LIMO | 56.7 | 12497 | 4.53E-5 | 91.9 | 67.2 | 7173 |
| **LIMO-P** | **63.3**(+6.6) | **10588**(-15%) | **5.98E-5**(+32%) | **93.8**(+1.9) | **71.2**(+4.0) | **6969**(-3%) |
| LIMO-V2 | 66.3 | 13896 | − | 94.4 | 70.2 | 8035 |
| **LIMO-V2-P** | **71.2**(+4.9) | **12163**(-12%) | − | **96.6**(+2.2) | **74.2**(+3.0) | **6968**(-13%) |

| Method Comparison (S1K) | AIME ACC | AIME TOK | GPQA ACC | GPQA EFF |
|-------------------------|----------|----------|----------|----------|
| S1-32B (baseline) | 37.9 | 6646 | 60.7 | 1.46E-4 |
| S1-PROMPT | 36.7 | 8013 | 58.0 | 2.03E-4 |
| S1-SPIRIT (all-step pruning) | 37.1 | 4906 | 60.1 | 2.13E-4 |
| S1-RULE (random functional pruning) | 36.7 | 4807 | 58.1 | 1.51E-4 |
| **S1-32B-P (PIR)** | **42.1** | **4716** | **61.6** | **2.49E-4** |

## Highlights & Insights
- **Counterintuitive "less is more" finding**: Removing low-value reasoning steps actually improves accuracy, suggesting that redundant functional steps may interfere with model learning.
- **Step-type distinction is critical**: By explicitly separating progressive reasoning from functional steps, PIR outperforms SPIRIT's undifferentiated pruning by 5 points on AIME.
- **Remarkable efficiency gains**: S1-32B-P achieves a 71% efficiency improvement on GPQA, with accuracy increasing while token count is nearly halved.
- **Cross-source generalization**: PIR proves effective across data distilled from Gemini, DeepSeek-R1, and QwQ, indicating that it captures universal properties of reasoning.
- **Cross-scale generalization**: Models ranging from 3B to 32B parameters all benefit, with the most pronounced gain on AIME (+11.8% accuracy) observed at the 32B scale.

## Limitations & Future Work
- Validation is limited to mathematical and scientific reasoning tasks; broader domains such as logical and commonsense reasoning remain unexplored.
- Perplexity may not fully capture the semantic contribution of certain steps, introducing the risk of information loss.
- The optimal pruning ratio varies across tasks and models; an adaptive strategy is lacking.
- The method relies on model perplexity outputs and is therefore not applicable to closed-source models.
- The classification stage depends on Claude 3.7 Sonnet, introducing additional cost and potential classifier bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The PIR metric and the design principle of retaining progressive reasoning while pruning functional steps are clear and novel, though the overall framework builds incrementally on prior work (SPIRIT).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three distillation sources × three benchmarks × multiple model scales × ablation studies — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The problem is clearly defined, experimental results are presented systematically, and tables are highly informative.
- **Value**: ⭐⭐⭐⭐ Offers direct practical value for LLM inference efficiency optimization; the method is simple and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
