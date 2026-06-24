---
title: >-
  [Paper Note] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum
description: >-
  [ICML 2026][LLM Evaluation][Supervised Fine-Tuning] This paper systematically investigates the behavior of probability-based objective functions in SFT, discovering that the standard NLL is not universally optimal: on tasks where the model has a strong prior, prior-leaning objectives like $-p$ significantly outperform NLL (with gains up to 16%). Conversely, NLL remains superior on tasks with weak priors, revealing an objective selection principle governed by the model-capabil…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Supervised Fine-Tuning"
  - "Loss Functions"
  - "Model Capability Continuum"
  - "Probability Objectives"
  - "Prior Leaning"
date: 2026-05-08
content_hash: ad547bf7468718b9
---

# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

**Conference**: ICML 2026  
**arXiv**: [2510.00526](https://arxiv.org/abs/2510.00526)  
**Code**: https://github.com/GaotangLi/Beyond-Log-Likelihood  
**Area**: LLM Training  
**Keywords**: Supervised Fine-Tuning, Loss Functions, Model Capability Continuum, Probability Objectives, Prior Leaning  

## TL;DR
This paper systematically investigates the behavior of probability-based objective functions in SFT, discovering that the standard NLL is not universally optimal: on tasks where the model has a strong prior, prior-leaning objectives like $-p$ significantly outperform NLL (with gains up to 16%). Conversely, NLL remains superior on tasks with weak priors, revealing an objective selection principle governed by the model-capability continuum.

## Background & Motivation

**Background**: Supervised Fine-Tuning (SFT) is the standard paradigm for LLM post-training, where the default training objective is Negative Log-Likelihood (NLL, $-\log p$). NLL has been proven optimal in the classic theory of training from scratch.

**Limitations of Prior Work**: Extensive practice has found that SFT generalization remains limited. This deficiency may not stem from the imitation learning paradigm itself but rather from the NLL objective function. Post-training differs fundamentally from training from scratch: the model has already encoded substantial task-related priors, and supervision sequences often reach thousands of tokens and may contain noise. NLL forces the model to copy reference answers token-by-token, tokens imposing excessive gradient signals on low-probability tokens, which can harm generalization.

**Key Challenge**: The optimality assumptions of NLL (IID, learning from scratch) are violated in post-training scenarios—the model possesses existing priors, and the prior strength varies across task domains. By generalizing NLL to a parametric family of objectives $f^\alpha(p) = (1-p^\alpha)/\alpha$ (which degrades to NLL as $\alpha \to 0$), it is observed that $\alpha=1$ (i.e., $-p$) and $\alpha=10$ significantly surpass NLL in mathematical reasoning.

**Goal**: Instead of promoting a single "universal" loss, this study aims to systematically characterize under what conditions each objective is optimal, establishing an analytical framework for the "model-capability continuum."

**Key Insight**: The effectiveness of an objective function is closely related to the base model's prior strength for the task. When the prior is strong (e.g., mathematics, where 25% of tokens in pre-training are math-related), prior-leaning objectives perform better; when the prior is weak (e.g., figfont riddles, entirely unseen during pre-training), NLL dominates.

**Core Idea**: The selection of SFT objectives should match model capabilities—replacing the "one-size-fits-all" NLL with a model-capability continuum.

## Method

### Overall Architecture
Given a pre-trained model $p_\theta$ and an SFT dataset $T$, standard SFT minimizes the NLL $\mathcal{L}_{-\log p}$. This work generalizes it into a universal probability objective $\mathcal{L}_{f(p)}(\theta) = \mathbb{E}_{(x,\tilde{y})\sim T}[\sum_t f(p_\theta(y_t|y_{<t},x))]$ (where $f:[0,1]\to\mathbb{R}$ is non-increasing and differentiable). Different $f$ are categorized into "prior-leaning" and "prior-averse" classes based on the shape of their gradient weights, and the sweet spot for each category is identified along the "model-capability continuum."

### Key Designs

**1. Unified Probability Objective Family $f^\alpha(p)$: Integrating NLL and $-p$ into a Single Knob**

NLL is the default choice because classic theory for training from scratch deems it optimal; however, in post-training where models have priors and sequences are long and noisy, this optimality assumption is violated. Instead of proposing a separate loss, this work defines a family of objectives $f^\alpha(p) = (1-p^\alpha)/\alpha$, where a single scalar parameter $\alpha$ continuously controls "how much to respect the prior." As $\alpha \to 0$, it converges to $-\log p$ (NLL); $\alpha=1$ yields $1-p$ (equivalent to maximizing average prediction accuracy). The function is concave for $\alpha \geq 1$ and convex for $0 \leq \alpha \leq 1$. Crucially, the gradient weight for the correct logit is $W_f(p) = p^\alpha(1-p)$: as $\alpha$ increases, gradient signals for low-probability tokens decay faster—meaning the model is not forced to rigidly replicate tokens it already deems unlikely. Thus, convexity/concavity serves as a natural proxy for "prior utilization."

**2. Model Capability Continuum: Determining Objectives via Prediction Probability**

The variable determining why a single loss succeeds in math but fails in riddles is the "prior strength of the base model for the task." This work arrays tasks along a continuum based on the average prediction probability on the training set, divided into three diagnostic segments. **Model-Strong (MS)** includes domains well-covered in pre-training (e.g., math, where Qwen2.5-Math-7B has an average probability of $0.81$). Here, prior-leaning objectives are optimal because low-probability tokens are mostly noise. **Model-Weak (MW)** includes domains almost unseen (e.g., figfont riddles, average prob $\approx 0.01$), where NLL is optimal as the prior cannot assist. **Model-Intermediate (MI)** covers partially represented domains (e.g., medical reasoning, $\approx 0.50$), where both types of objectives perform similarly. The average prediction probability serves as a direct diagnostic metric: if $> 0.5$, consider prior-leaning; if $< 0.1$, stick with NLL.

**3. Gradient Weights and Prior-Leaning Classification: Formalizing NLL's Focus on Low-Probability Tokens**

Lemma 3.1 provides the gradient weight of any objective $f$ with respect to the correct token logit:

$$W_f(p) = -f'(p) \cdot p(1-p),$$

Proposition 3.2 proves that the peak of $W_f$ for convex functions falls in $[0, 0.5]$, implying gradients are primarily directed at low-probability tokens ("prior-averse"). For concave functions, the peak falls in $[0.5, 1]$, concentrating gradients on already high-probability tokens ("prior-leaning"). Consequently, $-\log p$ (convex) is prior-averse, while $-p$ and $-p^{10}$ (concave) are prior-leaning. Finally, Theorem 6.4 uses gradient flow theory to show that risk decreases faster for prior-leaning objectives in the MS regime and faster for NLL in the MW regime.

## Key Experimental Results

### Main Results: Model-Strong (Mathematical Reasoning)

| Model | Objective | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Mean |
|------|------|---------|---------|---------------|--------|-------|------|
| Qwen2.5-Math-1.5B | Base | 30.71 | 8.81 | 14.88 | 2.49 | 17.97 | 14.97 |
| | $-\log p$ (NLL) | 42.52 | 12.71 | 12.09 | 0.62 | 17.03 | 17.00 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | 63.95 | 24.79 | 26.08 | 7.09 | 38.28 | **32.04** |
| | $-p$ | **65.27** | **26.18** | **26.66** | 6.88 | 38.13 | **32.75** |
| Qwen2.5-Math-7B | Base | 40.38 | 13.66 | 16.36 | 6.04 | 24.69 | 20.23 |
| | $-\log p$ (NLL) | 51.90 | 18.88 | 17.37 | 2.70 | 22.50 | 22.67 |
| | $-\log p \cdot \mathbf{1}_{p \geq 0.2}$ | **67.85** | **32.47** | **33.90** | **8.76** | **47.81** | **38.16** |
| | $-p$ | 68.47 | 31.99 | 32.26 | 8.75 | 41.09 | 36.51 |

### Model-Weak End and Continuum Summary

| Dimension | Model-Strong (MS) | Model-Intermediate (MI) | Model-Weak (MW) |
|------|-------------------|------------------------|-----------------|
| Representative Domains | Math Reasoning | Medical Reasoning | Figfont Riddles |
| Model Prediction Prob | 0.76–0.81 | ~0.50 | ~0.01 |
| Optimal Objective | $-p$ / Thresholded NLL | Similar performance | NLL ($-\log p$) |
| $-p$ vs NLL Gain | $-p$ wins by up to +16% | Difference < 2% | NLL wins ($-p$ near 0) |
| Root Cause | NLL over-fits low-prob noise | Mixed prior strength | $-p$ reinforces error |

### Key Findings
- In the MS regime, training only on the top 10% high-probability tokens outperforms full-token NLL, proving that low-probability tokens are essentially noise when the prior is strong.
- As $\alpha$ increases from 0.1 to 10.0, accuracy increases monotonically in the MS regime and decreases monotonically in the MW regime, forming a perfect "X-shaped" crossover.
- In general instruction tuning (Qwen2.5-3B/7B/14B), the advantage of $-p$ gradually emerges as model size increases, verifying that the continuum holds even within the same data distribution.
- Coding tasks also belong to the MS regime ($-p$ > NLL), while low-resource multilingual tasks belong to the MW regime (NLL > $-p$), demonstrating cross-domain generalization of the continuum.

## Highlights & Insights
- The core insight is remarkably concise: a single scalar $\alpha$ controls the degree of prior utilization, and the optimal $\alpha$ is determined by the model's task-specific prior strength. This is more elegant than designing complex adaptive losses.
- Thresholded NLL ($-\log p \cdot \mathbf{1}_{p \geq 0.2}$) serves as a practical intermediate solution; it gains benefits close to $-p$ by simply filtering low-probability tokens without changing the loss shape, incurring nearly zero cost.
- The average prediction probability on the training set is a quantitative diagnostic tool—use prior-leaning objectives above 0.5 and NLL below 0.1. This rule is directly transferable to any SFT scenario.

## Limitations & Future Work
- Objective selection remains static: model capability changes during training, but this work uses a fixed objective throughout and does not explore adaptive scheduling of $\alpha$.
- The measure of model capability depends on prediction probabilities, which may mislead objective selection if the model is severely miscalibrated or exhibits confident hallucinations.
- In the MI region (e.g., medical reasoning), neither class of objectives shows a distinct advantage; improvements here may require approaches beyond loss functions (e.g., data quality or domain knowledge injection).
- Theoretical analysis is based on simplified assumptions (e.g., uniform distributions for MW models), which are more complex in real-world scenarios.

## Related Work & Insights
- Wu et al. (2026) proposed uniform gradient reweighting, which is essentially equivalent to the $-p$ objective; this work integrates it into a unified framework and identifies its failure conditions.
- Focal loss is categorized as "more prior-averse than NLL" within this framework, while Huber-like probability losses are prior-leaning.
- Insight: Post-training should not blindly adopt NLL. Evaluating the base model's prior strength to select a matching objective may be the simplest and most effective method for SFT improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](../../ICLR2026/llm_evaluation/log_probability_tracking_of_llm_apis.md)
- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)
- [\[ICLR 2026\] Sci2Pol: Evaluating and Fine-tuning LLMs' "Science-to-Policy Brief" Generation Capabilities](../../ICLR2026/llm_evaluation/sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->
