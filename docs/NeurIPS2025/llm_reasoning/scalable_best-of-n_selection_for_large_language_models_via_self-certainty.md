---
title: >-
  [Paper Note] Scalable Best-of-N Selection for Large Language Models via Self-Certainty
description: >-
  [NeurIPS 2025][LLM Reasoning][Best-of-N] This paper proposes Self-Certainty, a metric that quantifies model confidence via the token probability distribution of LLM outputs…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Best-of-N"
  - "self-certainty"
  - "distributional quantification"
  - "reward-model-free"
  - "inference scaling"
date: 2026-05-08
content_hash: 121f4aa01f045611
---

# Scalable Best-of-N Selection for Large Language Models via Self-Certainty

**Conference**: NeurIPS 2025
**arXiv**: [2502.18581](https://arxiv.org/abs/2502.18581)  
**Code**: [GitHub](https://github.com/backprop07/Self-Certainty)  
**Area**: LLM Inference & Selection Strategies
**Keywords**: Best-of-N, self-certainty, distributional quantification, reward-model-free, inference scaling

## TL;DR
This paper proposes Self-Certainty, a metric that quantifies model confidence via the token probability distribution of LLM outputs, enabling scalable Best-of-N selection without any auxiliary reward model. The approach achieves performance comparable to or exceeding reward-model-based methods.

## Background & Motivation
**Cost**: Existing Best-of-N methods rely on reward models (ORM/PRM), incurring high training and inference costs.

**Distribution Shift**: Reward models are susceptible to distributional shift and prone to reward hacking.

**Limitations of Prior Work**: Self-consistency is restricted to tasks with deterministic answers, limiting its generality.

**Opportunity**: The token probability distribution of LLMs inherently encodes model confidence and can be leveraged directly.

## Method

### Overall Architecture
Self-Certainty is grounded in distributional confidence quantification. The core idea is:
- A more concentrated token probability distribution → higher model confidence.
- The distance between the output distribution and the uniform distribution reflects token-level confidence.

### Key Designs
**1. Self-Certainty Metric**:
Based on KL divergence (equivalently, cross-entropy):

$$\text{Self-Certainty} = -\frac{1}{nV} \sum\sum \log\bigl(V \cdot p(j \mid x, y_{\leq i})\bigr)$$

where $n$ is the response length, $V$ is the vocabulary size, and $p$ is the token probability.

**2. Baseline Metrics for Comparison**:
- AvgLogP: direct average of log-probabilities
- Perplexity: exponentiated negative log-probability
- Entropy: information-theoretic entropy
- Gini Impurity: decision-tree-derived measure
- DP: distributional perplexity

**3. Borda Voting Fusion**:
- Rank $N$ samples by confidence score
- Assign weighted votes: $v(r) = (N - r + 1)^p$
- Aggregate answers by accumulating votes

## Key Experimental Results

### Confidence Metric Comparison (LiveBench-Math)

| Metric | N=8 | N=16 | N=32 | N=64 | Trend |
|---|---|---|---|---|---|
| AvgLogP | 17.66% | 17.5% | 18.2% | 18.3% | Flat |
| Perplexity | 20.44% | 18.3% | 16.5% | 15.8% | Declining |
| Entropy | — | — | — | — | Complementary |
| KL-Divergence | 22.1% | 25.2% | 28.1% | **29.8%** | Rising ✓ |
| Self-Certainty | 20.87% | 22.01% | 27.5% | 28.5% | Strongly Rising ✓ |

### Main Results (Table 1)

| Method | LiveBench-Math | GSM8K | MATH | CRUXEval-O | Avg. |
|---|---|---|---|---|---|
| Greedy | 12.23% | 47.96% | 46.02% | 39.88% | 36.5% |
| Self-Consistency | 22.50% | 89.42% | 58.60% | 47.58% | 56.15% |
| Self-Certainty | 20.87% | 87.32% | 54.63% | 45.38% | 52.71% |
| Borda Voting ($p{=}1.2$) | **23.21%** | **89.51%** | **59.04%** | **47.93%** | **56.51%** |

### Ablation Study: Effect of $p$ in Borda Voting

| $p$ | N=8 | N=16 | N=32 | N=64 | Characteristics |
|---|---|---|---|---|---|
| 0.0 | 23.02% | 22.5% | 22.5% | 26.25% | Majority voting |
| 0.3 | 23.69% | 26.5% | 26.5% | 26.47% | Recommended for small $N$ |
| 0.7–1.2 | 23.21% | **26.69%** | **26.69%** | 26.41% | **Optimal range** |
| 2.0+ | 22.45% | 26.41% | 24.1% | 18.2% | Degradation |

### Open-Ended Task Performance (LiveCodeBench)

| Model | Greedy | USC | Self-Certainty | Borda Voting |
|---|---|---|---|---|
| Llama-8B | 42.93% | 43.78% | **45.83%** | **50.85%** |
| Qwen-32B | 78.6% | 76.8% | **79.5%** | **81.2%** |

## Highlights & Insights
1. **Intrinsic Signal Theory**: LLM token distributions inherently encode confidence, requiring no external annotation.
2. **No Distributional Bias**: Self-Certainty relies solely on native probabilities, avoiding the biases of learned reward models.
3. **Length Robustness**: Unlike negative perplexity, Self-Certainty does not inflate scores for longer sequences.
4. **Open-Ended Generalization**: Overcomes the deterministic-answer constraint of self-consistency, extending applicability to tasks such as code generation.

## Limitations & Future Work
1. **Spurious Confidence**: Certain generations may receive high scores due to superficial confidence despite being factually incorrect.
2. **Performance Gap**: Self-consistency retains a marginal ~0.5% advantage on closed-form MATH tasks.
3. **Hyperparameter Sensitivity**: The Borda exponent $p$ requires tuning with respect to $N$ and task type.
4. **Theoretical Gap**: The deeper theoretical explanation for why KL divergence outperforms other metrics remains incomplete.

## Related Work & Insights
- **Best-of-N**: Self-consistency, USC, ORMs/PRMs
- **Confidence Estimation**: BSDetector, TrustScore, self-evaluation methods
- **Inference Scaling**: Test-time compute scaling, multi-path reasoning

## Rating
⭐⭐⭐⭐⭐

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)

</div>

<!-- RELATED:END -->
