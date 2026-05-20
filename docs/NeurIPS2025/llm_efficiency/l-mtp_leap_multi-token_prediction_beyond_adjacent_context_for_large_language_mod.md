---
title: >-
  [Paper Note] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models
description: >-
  [NeurIPS 2025][LLM Efficiency][Leap Prediction] L-MTP introduces a leap mechanism into multi-token prediction (MTP) by predicting tokens at non-adjacent positions (e.g., positions 1, 3, 5, 7 instead of 1, 2, 3…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Leap Prediction"
  - "Inference Efficiency"
  - "Speculative Decoding"
  - "Multi-Head Prediction"
  - "Tree Attention"
date: 2026-05-08
content_hash: 14de28b39dc45f05
---

# L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.17505](https://arxiv.org/abs/2505.17505)  
**Code**: [https://github.com/Xiaohao-Liu/L-MTP](https://github.com/Xiaohao-Liu/L-MTP)  
**Area**: LLM Inference Acceleration, Multi-Token Prediction
**Keywords**: Leap Prediction, Inference Efficiency, Speculative Decoding, Multi-Head Prediction, Tree Attention

## TL;DR

L-MTP introduces a leap mechanism into multi-token prediction (MTP) by predicting tokens at non-adjacent positions (e.g., positions 1, 3, 5, 7 instead of 1, 2, 3, 4). A "looking backward" decoding strategy reuses prior predictions to fill the gaps, achieving a 22% inference speedup on 3B–12B models while maintaining or improving task performance.

## Background & Motivation

- **Background**: Standard next-token prediction (NTP) generates tokens autoregressively one at a time, resulting in low inference efficiency; the model is also confined to a myopic adjacent context, making it prone to overlooking "hard decisions."
- **Limitations of Prior Work**: MTP extends training signals and accelerates inference by predicting $n$ adjacent tokens in parallel via multiple output heads (e.g., positions $t+1$ to $t+n$), but the prediction scope remains restricted to adjacent positions.
- **Key Challenge**: Human reasoning rarely proceeds strictly in order; people commonly skip intermediate steps to reason efficiently (fuzzy-trace theory), motivating leap-style token prediction. Recent work (Rho, TokenSkip, etc.) further shows that many intermediate reasoning steps can be compressed or abstracted without sacrificing correctness.
- **Goal**: L-MTP achieves a similar effect by altering the prediction pattern rather than explicitly modeling token importance.

## Method

### Core Idea

L-MTP retains the multi-head parallel architecture of MTP but redistributes each head's prediction target. A leap interval parameter $k$ is introduced so that $n$ heads predict positions $[t+1,\ t+k+1,\ t+2k+1,\ \ldots,\ t+k(n-1)+1]$ instead of the standard MTP targets $[t+1, t+2, \ldots, t+n]$. For example, with $k=2$ and $n=4$, the predicted positions are $[t+1, t+3, t+5, t+7]$, expanding the coverage from $n=4$ to $k(n-1)+1=7$.

### Two-Stage Training

**Stage 1: Head Warm-up**
- The LLM backbone and original output head are frozen; only the newly added prediction heads are trained.
- Self-distillation data is used (the unmodified LLM is queried with input questions to collect outputs).
- Supervision signals are assigned to new heads according to the leap pattern: the $i$-th head predicts position $t+k(i-1)+1$.
- Learning rate: $1\times10^{-3}$; 5 epochs; cosine scheduler; warmup ratio 0.1.

**Stage 2: Full Model Tuning**
- All components are unfrozen; LoRA (rank=32, alpha=16) is applied for fine-tuning.
- The loss consists of two components: the original NTP loss plus a $\beta$-weighted leap head loss.
- Learning rate: $1\times10^{-5}$; 3 epochs.
- Default hyperparameters: $k=2$, $n=4$.

### Looking Backward Decoding

Each forward pass of L-MTP predicts only an incomplete sequence (leap positions), but the gap tokens have already been predicted in **prior steps**:
- Given $x_{\leq t}$, the model predicts $\{x_{t+1}, x_{t+3}, x_{t+5}, x_{t+7}\}$.
- Given $x_{\leq t-1}$, the model has already predicted $\{x_t, x_{t+2}, x_{t+4}, x_{t+6}\}$.
- Interleaving the two sets yields the complete contiguous sequence $x_{t+1}$ through $x_{t+7}$.
- Key advantage: no additional inference is required; prior predictions are retrieved directly from cache.

Formally, for $i \in \{1, \ldots, k(n-1)+1\}$, token $x_{t+i}$ is predicted conditioned on $x_{\leq t - (i-1) \bmod k}$, where $\bmod k$ switches between predictions from different steps.

### Tree Attention Integration

L-MTP is seamlessly compatible with speculative decoding:
- A hierarchical tree structure is constructed, with the $i$-th layer containing candidate tokens from the $i$-th head.
- Tree attention masks restrict each hidden state to attend only to its ancestor nodes.
- The looking-backward decoding strategy supplies contiguous sequences for verification.
- The approach can be directly grafted onto existing methods such as Medusa.

## Theoretical Analysis

**Attenuation**: In multi-token prediction, the marginal probability of predicting the $i$-th future token decreases as the prediction distance grows: $p(x_{t+1}|x_{\leq t}) > p(x_{t+2}|x_{\leq t}) > \cdots > p(x_{t+n}|x_{\leq t})$. This is an intrinsic property of LLM predictive capability.

**Consistency**: The expected marginal probability of predicting $x_{t+i}$ is stable across different inputs, following a predictable decay function $f(i)$.

**Acceleration Theorem (Theorem 3)**: Let the decay coefficient be $\gamma$ with decay function $f(i) = \exp[-\gamma \cdot (i-1)]$. When $\gamma n^2 \leq C$ (i.e., $\gamma = O(1/n^2)$), the expected acceptance length of L-MTP is strictly greater than that of standard MTP. Intuitively, the smaller the decay (i.e., the stronger the model's ability to predict distant tokens), the greater the acceleration advantage of L-MTP.

## Key Experimental Results

### Main Results (6 models × 8 datasets)

| Model | Scale | NTP Avg. | MTP Avg. | L-MTP Avg. |
|-------|-------|----------|----------|------------|
| Llama 3.2 | 3B | 25.52 | 25.46 | **26.68** |
| Llama 3.1 | 8B | 36.40 | 35.87 | **36.80** |
| Qwen 2.5 | 3B | 53.43 | 52.79 | **53.75** |
| Qwen 2.5 | 7B | 63.86 | 63.34 | **64.16** |
| Gemma 3 | 4B | 37.41 | 36.59 | 37.01 |
| Gemma 3 | 12B | 46.87 | 45.17 | **49.58** |

L-MTP outperforms MTP in the vast majority of settings, with particularly notable gains on Gemma 3 12B (+4.41 over MTP).

### Inference Speedup

- L-MTP achieves approximately 22% additional speedup over MTP given the same number of heads.
- When grafted onto Medusa, the gains are more pronounced: on Vicuna 7B, MTP 1.83× → L-MTP 2.32×; on Vicuna 13B, MTP 2.24× → L-MTP 2.43×.
- Speedup advantages are most evident on GSM8K.

### Ablation Study

- Per-position prediction accuracy experiments validate the attenuation and consistency hypotheses.
- Larger models exhibit more severe myopia — accuracy at distant positions degrades faster in larger models, highlighting an inherent limitation of NTP pretraining.
- Increasing training data improves per-head prediction accuracy, but the gains are nonlinear.

## Highlights & Insights

1. **Conceptual simplicity**: Only the target positions of prediction heads are modified (from adjacent to leap); no architectural changes or additional computation are required.
2. **Looking backward decoding**: Gap tokens are filled by cleverly reusing predictions from prior steps, incurring zero additional inference overhead.
3. **Plug-and-play**: The decoding strategy can directly replace that of existing MTP models to obtain a 22% speedup without retraining.
4. **Theory–experiment alignment**: The decay condition in Theorem 3 is precisely validated through experiments.
5. **Broad validation**: Experiments span three model families (Llama, Qwen, Gemma) at scales from 3B to 12B.

## Limitations & Future Work

- In some settings, NTP fine-tuning itself leads to performance degradation; while L-MTP outperforms MTP, it cannot fully compensate, likely due to training data quality.
- The theoretical proof relies on the exponential decay assumption $f(i) = \exp[-\gamma \cdot (i-1)]$; the true distribution may be more complex.
- Validation is limited to 3B–12B scales; larger models (70B+) and pretraining-from-scratch scenarios remain unexplored.
- The values of $k$ and $n$ are currently fixed; the paper mentions adaptive selection based on local entropy as future work.

## Related Work & Insights

- **MTP Foundations**: Qi et al. (ProphetNet) proposed $n$-step-ahead prediction; Gloeckle et al. demonstrated that pretraining with additional prediction heads substantially improves performance on code tasks.
- **Industry Deployment**: DeepSeek-V2 and MiMo adopt MTP to improve training efficiency and planning capability.
- **Speculative Decoding**: Medusa adds FFN heads for self-speculative decoding; EAGLE performs feature-level speculation; SpecInfer and SpecTr use tree-structured parallel verification.
- **Paper Positioning**: This is the first work to introduce a leap strategy into MTP, simultaneously broadening training signals and accelerating inference.

## Rating
⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention](dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)
- [\[NeurIPS 2025\] SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[NeurIPS 2025\] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_efficiency/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)

</div>

<!-- RELATED:END -->
