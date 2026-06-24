---
title: >-
  [Paper Note] SSA: Improving Performance With a Better Scoring Function
description: >-
  [ACL 2026][Interpretability][Softmax saturation] This paper identifies that Softmax attention collapses into an approximate hardmax under distribution shifts due to high-magnitude tokens. It proposes Scaled Signed Averaging (SSA) as a trainable alternative scoring function, which demonstrates superior generalization performance over Softmax across synthetic ICL tasks, a 114M decoder-only language model, and BabyBERTa encoder probes.
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Softmax saturation"
  - "attention scoring function"
  - "SSA"
  - "distribution shift"
  - "ICL generalization"
date: 2026-05-08
content_hash: 5f16d5750d11200f
---

# SSA: Improving Performance With a Better Scoring Function

**Conference**: ACL 2026  
**arXiv**: [2508.14685](https://arxiv.org/abs/2508.14685)  
**Code**: https://github.com/omyokun/SSA/  
**Area**: Interpretability / Attention Mechanisms / In-Context Learning  
**Keywords**: Softmax saturation, attention scoring function, SSA, distribution shift, ICL generalization

## TL;DR
This paper identifies that Softmax attention collapses into an approximate hardmax under distribution shifts due to high-magnitude tokens. It proposes Scaled Signed Averaging (SSA) as a trainable alternative scoring function, which demonstrates superior generalization performance over Softmax across synthetic ICL tasks, a 114M decoder-only language model, and BabyBERTa encoder probes.

## Background & Motivation
**Background**: The default attention mechanism in Transformers uses Softmax to normalize query-key scores into weights. The success of Softmax has made it a standard configuration; however, increasing research in In-Context Learning (ICL) shows that models performing well near the training distribution often fail when encountering simple distribution shifts.

**Limitations of Prior Work**: Many failure analyses of ICL remain at the level of data scale, pretraining corpora, or task distribution, making it difficult to determine whether the model failed to learn the rules or if the architecture itself prevented the model from aggregating context under specific inputs. The authors constructed small models trained from scratch to exclude interference from pretraining corpora and prompt engineering.

**Key Challenge**: ICL requires the model to integrate multiple context examples, but Softmax concentrates exponentially on the maximum item when logit gaps are large. An abnormally high-magnitude token, even if irrelevant to the task, can absorb nearly all attention weights.

**Goal**: To prove that Softmax saturation is an architectural source of failure in ICL out-of-distribution generalization and to propose an alternative scoring function that delays attention collapse and retains more contextual quality.

**Key Insight**: The paper uses two transparent synthetic tasks to locate the problem: a quantifier judgment task requiring the model to observe the entire sequence, and a linear function task requiring the model to infer $f(x)=ax+b$ from context examples. Both clearly expose the phenomenon where a "deviant value" disrupts overall reasoning.

**Core Idea**: Replace exponential growth in the scoring function with a trainable polynomial-type signed scaling (SSA). This ensures that attention does not degrade into hardmax as quickly under high-magnitude inputs while maintaining the ability to select important tokens.

## Method

### Overall Architecture

The paper argues that ICL generalization failure under distribution shift stems partly from architectural flaws in the Softmax scoring function itself, rather than just data or prompt issues. It uses an elimination strategy to narrow the problem down to attention: the model can identify the sign of individual numbers, attention-only models can learn ICL while FF-only models cannot, failures occur in both full Transformers and attention-only models, and extreme tokens in the attention map absorb nearly all weights. Based on this, the authors propose SSA, an alternative scoring function that delays collapse, and validate it across three levels: synthetic ICL, a 114M decoder-only LM, and an encoder-only BabyBERTa. These three levels move from controllable toy tasks to real-world language modeling and linguistic probing.

### Key Designs

**1. Mechanism Diagnosis of Softmax Saturation: Explaining how a deviant token ruins multi-token aggregation tasks**

The issue lies in the exponential normalization of Softmax: if the gap between the maximum logit and other logits is $\Delta$, the weights of non-maximum items decay rapidly at a rate of $e^{-\Delta}$. When input values maintain their magnitude order after linear embedding, an abnormally high-magnitude token naturally generates a large embedding norm, pushing the attention towards an approximate hardmax. Since many ICL rules depend on multiple context examples to determine the answer, hardmax-style attention confuses "large magnitude" with "relevance," causing even a single irrelevant extreme token to break quantifier judgments and linear function predictions.

**2. Scaled Signed Averaging Scoring Function: Replacing exponential collapse with trainable polynomial collapse**

SSA applies a transformation $(1+b|x|)^{sgn(x)n}$ to each logit before normalization, where $b>0$ and $n\geq1$ are trainable parameters. Positive values grow at a polynomial rate, while negative values decay towards 0. As $b=1/m, n=m, m\to\infty$, it can approximate the exponential function, making Softmax a limiting special case. The key difference is that when faced with global amplification or a single dominant token, Softmax concentrates exponentially whereas SSA concentrates only at a polynomial rate. This allows the model more opportunity to retain secondary but relevant context tokens, thereby mitigating the saturation diagnosed above.

### Loss & Training

In synthetic ICL, MSE loss is used for the linear function task and Cross-Entropy for the quantifier task, training for 500,000 steps with a batch size of 64. The decoder-only experiments use a 114M Nemotron-style model (12 layers, 24 heads, hidden size 768), trained on 10B tokens of FineWeb for 22k steps. BabyBERTa experiments are trained from scratch on AO-CHILDES, comparing SSA versions with fixed exponents $n=1.5$ and $n=2$.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Softmax | SSA | Gain Observation |
|--------|------|------|------|------|
| arc_challenge | acc_norm | 0.2398 | 0.2713 | Improvement in scientific knowledge |
| arc_easy | acc_norm | 0.2934 | 0.5387 | One of the largest gains |
| boolq | acc | 0.3783 | 0.5618 | Significant improvement in binary comprehension |
| cb | acc / f1 | 0.1429 / 0.1310 | 0.4643 / 0.2663 | Significant gain on small-scale NLI |
| copa | acc | 0.5900 | 0.6400 | Improvement in causal selection |
| hellaswag | acc_norm | 0.2550 | 0.3283 | Improvement in commonsense continuation |
| record | f1 / em | 0.1983 / 0.1932 | 0.2482 / 0.2427 | Improvement in reading comprehension metrics |
| winogrande | acc | 0.4972 | 0.5178 | Slight improvement in pronoun disambiguation |

### Ablation Study

| Analysis Item | Softmax | SSA | Description |
|------|------|------|------|
| FineWeb perplexity ↓ | 21.86 | 19.73 | Lower perplexity on training distribution |
| Wikipedia perplexity ↓ | 24.58 | 22.07 | Better performance on OOD text |
| BabyBERTa subject-verb across PP | 56.00 | 65.95 (SSA-2) | Improved long-distance agreement |
| BabyBERTa swapped arguments | 83.30 | 92.00 (SSA-1.5) | More sensitive to argument structure |
| BabyBERTa binding principle A | 78.25 | 87.90 (SSA-1.5) | Gain in binding relationship probes |
| BabyBERTa quantifier superlative | 71.20 | 83.95 (SSA-1.5) | Improvement in quantifier-related grammar |

### Key Findings
- In synthetic ICL, Softmax models perform well within the training distribution, but attention concentrates on a single token as soon as a high-magnitude deviant input appears, causing significant degradation in quantifier and linear function predictions.
- SSA is not merely Softmax with "added temperature." The authors tested temperature scaling, Sparsemax, Entmax, mixed scoring heads, linear attention, CosFormer, and SA-Softmax; none of these alternatives consistently outperformed Softmax across tasks.
- SSA is effective in real-world language modeling: the 114M decoder-only model, trained for only 22k steps, systematically outperformed Softmax across multiple zero-shot benchmarks and in perplexity.

## Highlights & Insights
- The paper clarifies a common but easily overlooked issue: "large" in attention weights does not always equal "relevant." The exponential amplification of Softmax makes this confusion severe under distribution shifts.
- The design of SSA is concise, adding only a few learnable scaling parameters per head. This changes the inductive bias from exponential collapse to polynomial collapse in a mathematically clean way.
- Validation across three levels—synthetic tasks, decoder-only LMs, and encoder-only BabyBERTa—makes the work more convincing than typical "activation function replacement" papers, as it both explains the failure and verifies the transferability.

## Limitations & Future Work
- Decoder-only experiments were only scaled to 114M parameters and trained for 10B tokens over 22k steps; whether the advantages hold for 7B+ models or modern training recipes remains unverified.
- SSA mitigates but does not fully solve strong distribution shifts, particularly when both input and function distributions shift significantly.
- The paper notes that a more fundamental issue is the tendency of the attention structure to confuse token representation magnitude with task relevance; SSA is a local correction rather than a complete solution.
- The new scoring function may impact existing efficient attention kernels and inference deployment, requiring subsequent engineering evaluations for speed, numerical stability, and hardware friendliness.

## Related Work & Insights
- **vs Temperature Softmax**: Temperature scales the distribution globally but cannot change the fundamental nature of exponential collapse; SSA retains polynomial context quality during extreme amplification.
- **vs Sparsemax / Entmax**: These methods control sparsity but yielded no stable gains in the ICL distribution shift tasks studied; SSA specifically targets saturation caused by high-magnitude tokens.
- **vs CosFormer / SA-Softmax**: These alternative attention forms also attempt to change normalization or kernels but failed to resolve the failure mode identified by the authors in experiments.
- **Insight**: For interpretability research, architectural diagnosis is best started from controllable tasks. For model design, the attention scoring function remains a space worth systematic searching rather than defaulting to Softmax.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explaining ICL generalization failure through Softmax saturation and proposing a theoretically distinct scoring function is a clear and solid approach.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic tasks, real LMs, encoder probes, and comparisons with multiple alternatives; lacks large-scale model validation.
- Writing Quality: ⭐⭐⭐⭐☆ Problem positioning, mathematical explanation, and the experimental chain are coherent, though some formulas are long.
- Value: ⭐⭐⭐⭐☆ Highly insightful for attention mechanism modification and ICL generalization research; practical large-scale deployment value requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Improving Sparse Autoencoder with Dynamic Attention](../../CVPR2026/interpretability/improving_sparse_autoencoder_with_dynamic_attention.md)
- [\[NeurIPS 2025\] Monte Carlo Expected Threat (MOCET) Scoring](../../NeurIPS2025/interpretability/monte_carlo_expected_threat_mocet_scoring.md)
- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)
- [\[ICLR 2026\] Causality ≠ Invariance: Function and Concept Vectors in LLMs](../../ICLR2026/interpretability/causality_invariance_function_and_concept_vectors_in_llms.md)
- [\[ICLR 2026\] Dissecting Representation Misalignment in Contrastive Learning via Influence Function](../../ICLR2026/interpretability/dissecting_representation_misalignment_in_contrastive_learning_via_influence_fun.md)

</div>

<!-- RELATED:END -->
