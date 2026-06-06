---
title: >-
  [Paper Note] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum
description: >-
  [ICML 2026][LLM Evaluation][Supervised Fine-Tuning] This paper systematically investigates the behavior of probability-based objective functions in SFT…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Supervised Fine-Tuning"
  - "Loss Functions"
  - "Model-Capability Continuum"
  - "Probability Objectives"
  - "Prior Bias"
date: 2026-05-08
content_hash: 4e9a38e6b373951b
---

# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

**Conference**: ICML 2026  
**arXiv**: [2510.00526](https://arxiv.org/abs/2510.00526)  
**Code**: https://github.com/GaotangLi/Beyond-Log-Likelihood  
**Area**: LLM Training  
**Keywords**: Supervised Fine-Tuning, Loss Functions, Model-Capability Continuum, Probability Objectives, Prior Bias  

## TL;DR
This paper systematically investigates the behavior of probability-based objective functions in SFT, finding that standard NLL is not universally optimal. On tasks where the model has a strong prior, prior-leaning objectives such as $-p$ significantly outperform NLL (up to 16% improvement). Conversely, NLL remains superior for tasks with weak priors. This reveals a selection principle for objective functions driven by the model-capability continuum.

## Background & Motivation

**Background**: Supervised Fine-Tuning (SFT) is the standard paradigm for LLM post-training, with Negative Log-Likelihood (NLL, $-\log p$) as the default objective. NLL has been proven optimal in classical learning-from-scratch theories.

**Limitations of Prior Work**: Extensive practice has found that the generalization capability of SFT is limited. This deficiency may not stem from the imitation learning paradigm itself, but rather from the NLL objective. Post-training differs fundamentally from learning from scratch: models already encode substantial task-related priors, and supervision sequences often span thousands of tokens and may contain noise. NLL forces the model to replicate reference answers token-by-token and applies excessive gradient signals to low-probability tokens, potentially harming generalization.

**Key Challenge**: The optimality assumption of NLL (IID, learning from scratch) is violated in post-training scenarios, where models possess existing priors whose strength varies by task domain. By extending NLL to a parameterized family of objectives $f^\alpha(p) = (1-p^\alpha)/\alpha$ (recovering NLL as $\alpha \to 0$), it is observed that $\alpha=1$ (i.e., $-p$) and $\alpha=10$ significantly outperform NLL in mathematical reasoning.

**Goal**: The objective is not to promote a "universal" loss, but to systematically characterize under what conditions specific objectives are optimal, establishing an analytical framework based on the "model-capability continuum."

**Key Insight**: The authors observe that the effectiveness of an objective function is closely related to the strength of the base model's prior for the task. When the prior is strong (e.g., mathematics, where 25% of pre-training tokens are math-related), prior-leaning objectives perform better. When the prior is weak (e.g., figfont anagrams, entirely unseen during pre-training), NLL remains superior.

**Core Idea**: The selection of the SFT objective should match the model capability—replacing the "one-size-fits-all" NLL with the model-capability continuum.

## Method

### Overall Architecture
Given a pre-trained model $p_\theta$ and an SFT dataset $T$, standard SFT minimizes the NLL $\mathcal{L}_{-\log p}$. This work extends it to a general probability objective $\mathcal{L}_{f(p)}(\theta) = \mathbb{E}_{(x,\tilde{y})\sim T}[\sum_t f(p_\theta(y_t|y_{<t},x))]$, where $f:[0,1]\to\mathbb{R}$ is a non-increasing differentiable function. By analyzing the shape of the gradient weights of different functions $f$, objectives are categorized into prior-leaning and prior-averse types, with their performance validated across different positions on the model-capability continuum.

### Key Designs

1.  **Unified Probability Objective Family $f^\alpha(p)$**:
    - **Function**: Provides a continuously adjustable family of objectives covering the spectrum from NLL to $-p$.
    - **Mechanism**: Defined as $f^\alpha(p) = (1-p^\alpha)/\alpha$. As $\alpha \to 0$, it converges to $-\log p$ (NLL); at $\alpha=1$, it becomes $1-p$ (maximizing mean predicted probability). The function is concave for $\alpha \geq 1$ and convex for $0 \leq \alpha \leq 1$. The gradient weight for the correct logit is $W_f(p) = p^\alpha(1-p)$: larger $\alpha$ values lead to faster decay of gradient signals for low-probability tokens, thus "respecting" the model prior more.
    - **Design Motivation**: Uses a single parameter $\alpha$ to unify the degree of prior-leaning, making convexity/concavity a proxy for analyzing prior utilization and supporting continuous interpolation.

2.  **Model-Capability Continuum**:
    - **Function**: Establishes a diagnostic framework for "when to use which objective."
    - **Mechanism**: Divides tasks into three intervals based on the base model's prior strength. **Model-Strong (MS)**: Domains well-covered by pre-training (e.g., math), where the average prediction probability is high (0.81 for Qwen2.5-Math-7B); prior-leaning objectives are optimal. **Model-Weak (MW)**: Domains uncovered by pre-training (e.g., figfont anagrams), where the average prediction probability is extremely low (~0.01); NLL is optimal. **Model-Intermediate (MI)**: Partially covered domains (e.g., medical reasoning, ~0.50), where both types of objectives perform similarly.
    - **Design Motivation**: Quantitatively explains why the same loss function behaves oppositely across different tasks and provides a practical diagnostic method based on prediction probabilities.

3.  **Gradient Weights and Prior-Leaning Classification**:
    - **Function**: Explains differences in learning behavior from a gradient perspective.
    - **Mechanism**: Lemma 3.1 derives the gradient weight $W_f(p) = -f'(p) \cdot p(1-p)$ for any objective $f$ relative to the correct token logit. Proposition 3.2 proves that for convex functions, the peak of $W_f$ is in $[0, 0.5]$ (prior-averse, emphasizing low-probability tokens), while for concave functions, the peak is in $[0.5, 1]$ (prior-leaning, emphasizing high-probability tokens). Accordingly, $-\log p$ (convex) is classified as prior-averse, while $-p$ and $-p^{10}$ (concave) are prior-leaning.
    - **Design Motivation**: Quantifies the intuition that "NLL focuses too much on low-probability tokens" into a rigorous mathematical characterization. Theorem 6.4 proves via gradient flow theory that the risk of prior-leaning objectives decreases faster on the MS end, while NLL decreases faster on the MW end.

## Key Experimental Results

### Main Results: Model-Strong (Mathematical Reasoning)

| Model | Objective | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Mean |
|-------|-----------|---------|---------|---------------|--------|-------|------|
| Qwen2.5-Math-1.5B | Base | 30.71 | 8.81 | 14.88 | 2.49 | 17.97 | 14.97 |
| | $-\log p$ (NLL) | 42.52 | 12.71 | 12.09 | 0.62 | 17.03 | 17.00 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | 63.95 | 24.79 | 26.08 | 7.09 | 38.28 | **32.04** |
| | $-p$ | **65.27** | **26.18** | **26.66** | 6.88 | 38.13 | **32.75** |
| Qwen2.5-Math-7B | Base | 40.38 | 13.66 | 16.36 | 6.04 | 24.69 | 20.23 |
| | $-\log p$ (NLL) | 51.90 | 18.88 | 17.37 | 2.70 | 22.50 | 22.67 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | **67.85** | **32.47** | **33.90** | **8.76** | **47.81** | **38.16** |
| | $-p$ | 68.47 | 31.99 | 32.26 | 8.75 | 41.09 | 36.51 |

### Model-Weak and Continuum Summary

| Dimension | Model-Strong (MS) | Model-Intermediate (MI) | Model-Weak (MW) |
|-----------|-------------------|------------------------|-----------------|
| Representative Domain | Math Reasoning | Medical Reasoning | figfont Anagrams |
| Prediction Prob. | 0.76–0.81 | ~0.50 | ~0.01 |
| Optimal Objective | $-p$ / Threshold NLL | Similar | NLL ($-\log p$) |
| $-p$ vs NLL Gap | $-p$ wins by up to +16% | Gap <2% | NLL wins ($-p$ near 0) |
| Root Cause | NLL over-focuses on noise | Prior neither strong nor weak | $-p$ reinforces wrong high-prob |

### Key Findings
- On the MS end, training only on the top 10% high-probability tokens exceeds full-token NLL, proving low-probability tokens are essentially noise when priors are strong.
- As $\alpha$ increases from 0.1 to 10.0, accuracy monotonically increases on the MS end and decreases on the MW end, forming a perfect "X-shaped" crossover.
- In general instruction tuning (Qwen2.5-3B/7B/14B), the advantage of $-p$ gradually emerges as model size increases, validating that the continuum holds within the same data distribution.
- Coding tasks also belong to the MS end ($-p$ > NLL), while low-resource multilingual tasks belong to the MW end (NLL > $-p$), confirming the continuum's cross-domain generalization.

## Highlights & Insights
- The core insight is remarkably simple: a single scalar parameter $\alpha$ controls prior utilization, and the optimal $\alpha$ is determined by the model's prior strength for the task. This is more elegant than designing complex adaptive losses.
- Thresholding NLL ($-\log p \cdot \mathbf{1}_{p \geq 0.2}$) serves as a practical intermediate solution; it achieves performance close to $-p$ without changing the loss shape, merely by filtering low-probability tokens, making the implementation cost near-zero.
- The average predicted probability on the training set serves as a quantitative diagnostic for objective selection—use prior-leaning objectives above 0.5, and NLL below 0.1. This rule can be directly migrated to any SFT scenario.

## Limitations & Future Work
- Objective selection remains static: while model capability changes during training, this study uses fixed objectives throughout, leaving adaptive scheduling of $\alpha$ unexplored.
- The measure of model capability relies on prediction probabilities, which may mislead objective selection if the model is severely miscalibrated or generates confident hallucinations.
- In the MI region (e.g., medical reasoning), neither objective type shows a clear advantage; improvements may require directions beyond loss functions, such as data quality or domain knowledge injection.
- Theoretical analysis is based on simplified assumptions (e.g., uniform distribution predictions on the MW end); real-world scenarios are more complex.

## Related Work & Insights
- The uniform gradient reweighting proposed by Wu et al. (2026) is essentially equivalent to the $-p$ objective; this paper incorporates it into a unified framework and identifies its failure conditions.
- Focal loss is categorized under this framework as a "more prior-averse than NLL" objective, while Huber-like probability losses are prior-leaning.
- Insight: Post-training should not blindly apply NLL. Evaluating the base model's prior strength before choosing a matching objective may be the simplest and most effective way to improve SFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_evaluation/hyperbolic_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
