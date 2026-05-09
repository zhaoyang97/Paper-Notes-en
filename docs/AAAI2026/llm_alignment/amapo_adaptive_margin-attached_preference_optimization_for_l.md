---
title: >-
  [Paper Note] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment
description: >-
  [AAAI 2026][LLM Alignment][Preference optimization] This paper proposes AMaPO, an algorithm that dynamically modulates gradient magnitudes via instance-level adaptive margins (combining Z-normalization and exponential scaling) to address the core overfitting-underfitting dilemma in offline preference optimization methods such as DPO, thereby substantially improving ranking accuracy and downstream alignment performance.
tags:
  - AAAI 2026
  - LLM Alignment
  - Preference optimization
  - adaptive margin
  - ranking accuracy
  - gradient dynamics analysis
date: 2026-05-08
content_hash: a6ebb168db24b79b
---

# AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.09385v2](https://arxiv.org/abs/2511.09385v2)
**Code**: None
**Area**: LLM Alignment
**Keywords**: Preference optimization, LLM alignment, adaptive margin, ranking accuracy, gradient dynamics analysis

## TL;DR

This paper proposes AMaPO, an algorithm that dynamically modulates gradient magnitudes via instance-level adaptive margins (combining Z-normalization and exponential scaling) to address the core overfitting-underfitting dilemma in offline preference optimization methods such as DPO, thereby substantially improving ranking accuracy and downstream alignment performance.

## Background & Motivation

Offline preference optimization methods (e.g., DPO, SimPO) are simpler and more stable than RLHF and have become the dominant paradigm for LLM alignment. The key to these methods lies in the **ranking accuracy** of the implicit reward model — i.e., whether the model can correctly distinguish preferred from dispreferred responses. Although prior work has introduced fixed or dynamic margins to improve ranking accuracy, a unified theoretical framework for analyzing how different margin designs dynamically affect ranking accuracy has been lacking. The authors identify a fundamental flaw in existing margin designs: they impose excessively large gradients on already correctly ranked samples (overfitting) while providing insufficient correction signals for incorrectly ranked samples (underfitting).

## Core Problem

**Overfitting-Underfitting Dilemma**: Within a unified margin framework, DPO's margin is derived from the reference model ($\gamma = \beta(\log\pi_{\text{ref}}(y_w|x) - \log\pi_{\text{ref}}(y_l|x))$), which neither adapts dynamically to instance-level ranking correctness nor guarantees non-negativity, leading to unnecessarily large gradients on correctly ranked samples (overfitting) and insufficient correction on incorrectly ranked samples (underfitting). Although SimPO ensures $\gamma = C > 0$, its fixed margin similarly ignores differences in ranking difficulty across samples.

## Method

### Overall Architecture

The authors first establish a **unified margin-based objective framework**:
$$\mathcal{L}_{\text{unified}}(\theta) = -m(h_w(\log\pi_w) - h_l(\log\pi_l) - \gamma) + \Lambda(\log\pi_w)$$

Under this framework, methods such as DPO and SimPO can be uniformly expressed via different definitions of $h_w$, $h_l$, $m$, and $\gamma$. Gradient analysis reveals that the margin $\gamma$ is the central lever controlling the learning rate.

AMaPO builds on the SimPO framework, replacing its static margin with an **instance-level adaptive margin** $\gamma(x, y_w, y_l)$, with the objective:
$$\mathcal{L}_{\text{AMaPO}} = -\mathbb{E}[\log\sigma(r_{\pi_\theta}(x,y_w,y_l) - h_\gamma(\text{sg}[\gamma(x,y_w,y_l)]))]$$

### Key Designs

1. **Oracle Ranking Margin and Ideal Adaptive Margin**: An Oracle Ranking Margin $\gamma^*$ is defined as the ideal instance-level non-negative threshold. For incorrectly ranked samples ($r_{\pi_\theta} \leq 0 < \gamma^*$), the margin is positive and large to amplify the correction gradient; for already correctly ranked samples ($r_{\pi_\theta} > \gamma^* \geq 0$), the margin is zero to suppress the gradient. The ideal adaptive margin is formalized as:
$$\gamma^*(x,y_w,y_l) = \mathbb{I}[(\gamma^* - r_{\pi_\theta}) > 0] \cdot \gamma^*$$

2. **Z-normalization for Oracle Margin Estimation**: Since the true $\gamma^*$ is inaccessible, the mean implicit margin $\mu_r$ within the current training batch is used as a surrogate estimate of $\gamma^*$. Z-score normalization is applied to achieve stable estimation and appropriate scaling:
$$\gamma(x,y_w,y_l) = \max\left(\frac{\mu_r - r_{\pi_\theta}(x,y_w,y_l)}{\sigma_r} \cdot \mu_r, \ 0\right)$$
where $\mu_r$ and $\sigma_r$ are the mean and standard deviation of $r_{\pi_\theta}$ within the batch. The normalization term $(\mu_r - r_{\pi_\theta})/\sigma_r$ measures the relative "difficulty" of each sample.

3. **Exponential Scaling Function**: Since log-probabilities may not faithfully reflect the quality of generated sequences, and motivated by the strong correlation between perplexity (PPL) and generation quality, exponential scaling is introduced to better represent quality gaps and accelerate training on difficult samples:
$$h_\gamma(\gamma) = \begin{cases} 0 & \text{if } \gamma = 0 \\ \beta \cdot e^\gamma & \text{if } \gamma > 0 \end{cases}$$
Theoretically, this exponential scaling is equivalent to a power of the geometric mean of the PPL ratio between losing and winning responses within the batch.

### Loss & Training

The final objective applies a **stop-gradient** ($\text{sg}[\cdot]$) to the adaptive margin to prevent gradients from flowing back into the margin computation, treating it as a fixed target at each optimization step:
$$\mathcal{L}_{\text{AMaPO}}(\pi_\theta;\mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(r_{\pi_\theta}(x,y_w,y_l) - h_\gamma(\text{sg}[\gamma(x,y_w,y_l)])\right)\right]$$

where $r_{\pi_\theta}(x,y_w,y_l) = \frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)$ is the implicit ranking score. Training follows the standard SimPO setup with Adam optimizer and cosine learning rate scheduling; the only hyperparameter to tune is $\beta$ (recommended default around 2).

## Key Experimental Results

| Dataset / Benchmark | Metric | AMaPO | SimPO | DPO | Gain (vs. SimPO) |
|---|---|---|---|---|---|
| AlpacaEval2 (Llama3-8B-Base) | LC Win Rate | 26.4% | 22.0% | 18.2% | +4.4% |
| AlpacaEval2 (Mistral-7B-Base) | LC Win Rate | 24.3% | 21.5% | 15.1% | +2.8% |
| AlpacaEval2 (Llama3-8B-Instruct) | LC Win Rate | 46.1% | 44.7% | 40.3% | +1.4% |
| MT-Bench (Llama3-8B-Instruct) | GPT-4 Turbo | 7.2 | 7.0 | 7.0 | +0.2 |
| RM-Bench (Llama3-8B-Base) | Avg. | 58.6 | 56.9 | 54.6 | +1.7 |
| RM-Bench (Llama3-8B-Base) | Hard | 25.4 | 23.3 | 15.0 | +2.1 |
| OOD Generalization (Mistral-7B-Base) | Response-OOD | 83.62 | 76.08 | 68.41 | +7.54 |
| OOD Generalization (Mistral-7B-Instruct) | Mutual-OOD | 90.91 | 85.64 | 72.89 | +5.27 |
| Open LLM Leaderboard (Mistral-7B-Instruct) | Avg. Rank | 1.9 | 2.9 | 3.6 | Better |

### Ablation Study

Ablation results on Llama3-8B-Base (AlpacaEval2 LC / WR / MT-Bench):
- **Full AMaPO**: 26.3% / 21.4% / 6.5
- **w/o Z-normalization**: 24.8% / 20.4% / 6.3 → Z-norm plays an important role in stabilizing margin estimation
- **w/o exponential scaling**: 24.0% / 17.6% / 6.1 → exponential scaling significantly affects WR
- **w/o adaptive (removing both Z-norm and exp)**: 20.7% / 16.4% / 6.3 → substantial degradation
- **w/o zero-margin setting**: 22.3% / 21.1% / 6.6 → retaining a positive margin for correctly ranked samples leads to a marked increase in generation length

The effect of $\beta$ follows an inverted-U shape: too small a $\beta$ yields insufficient correction, while too large a value produces overly sharp distributions; the optimal value is approximately 3.

## Highlights & Insights

- **Solid theoretical contribution**: The paper unifies the margin framework of DPO-family methods and exposes the overfitting-underfitting dilemma from a gradient-dynamics perspective, providing an analytical tool for future work.
- **Simple and elegant design**: No additional hyperparameters are introduced (only $\beta$ is inherited); adaptivity is achieved via batch statistics (mean and standard deviation) at minimal implementation cost.
- **Strong OOD generalization**: Improvements are especially pronounced under Response-OOD and Mutual-OOD settings (up to +7.5%), validating that the adaptive margin genuinely alleviates overfitting.
- **Improved generation quality**: AMaPO raises the win rate while reducing generation length (200+ tokens shorter than SimPO), indicating that the model learns more concise expression.

## Limitations & Future Work

- **Limited scale validation**: Experiments are conducted only on 7B–8B scale models; overfitting/underfitting behavior may differ on larger models (70B+).
- **Generality of the oracle margin estimator**: Using the batch mean as a surrogate for $\gamma^*$ may not be optimal and is susceptible to SFT model quality and noise in preference data.
- **Performance degradation on mathematical tasks**: AMaPO consistently underperforms SimPO on the MATH benchmark; the authors conjecture that assigning a positive margin to correctly ranked samples may benefit mathematical generation tasks.
- **Scaling function selection**: Only the exponential function combined with Z-normalization is explored; alternative scaling functions are not systematically evaluated.
- **Scope of analysis**: The gradient analysis provides static snapshots rather than capturing the full dynamics of the training trajectory.

## Related Work & Insights

| Method | Margin Design | Adaptive? | Core Issue |
|---|---|---|---|
| DPO | Derived from reference model; can be positive or negative | No | Overfitting + underfitting |
| SimPO | Fixed constant $C > 0$ | No | Overfitting (ignores sample difficulty variation) |
| IPO | Derived from reference model + calibration term | No | Similar to DPO |
| ODPO | Derived from reference model + externally annotated margin | Partially (relies on external annotation) | Similar to DPO |
| α-DPO | Policy-driven + tunable target margin | Partially | Multiple hyperparameters; similar to DPO |
| FocalPO | Non-monotonic gradient magnitude | No | Prioritizes correctly ranked samples (opposite direction to AMaPO) |
| **AMaPO** | **Instance-level adaptive, based on batch statistics** | **Yes** | **Directly resolves the dilemma** |

The concept of adaptive learning signals is broadly applicable: analogous "hard example mining" ideas appear in contrastive learning and object detection (focal loss), and AMaPO's Z-norm + exponential scaling scheme may transfer to those domains. Using batch statistics as surrogate targets is a lightweight and efficient normalization strategy that avoids maintaining an additional reference model or reward model, offering advantages in resource-constrained settings. The overfitting-underfitting analysis framework can serve as a standard tool for evaluating new preference optimization algorithms. The performance degradation on mathematical tasks merits further investigation and may suggest that different task types require different margin strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified framework and formal definition of the overfitting-underfitting dilemma are reasonably novel, though the idea of instance-level adaptive margins is not entirely new (α-DPO has explored similar directions).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four model configurations, multiple benchmarks, OOD generalization tests, comprehensive ablation studies, and case studies — highly thorough.
- **Writing Quality**: ⭐⭐⭐⭐ The logic is clear; the derivation from analysis to method design flows smoothly; the unified framework is presented rigorously.
- **Value**: ⭐⭐⭐⭐ The method is simple, effective, and introduces no additional hyperparameters; the theoretical analysis of DPO-family methods has reference value, though the degradation on mathematical tasks limits its universality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](../../ICLR2026/llm_alignment/towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ICLR 2026\] From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](../../ICLR2026/llm_alignment/from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)

</div>

<!-- RELATED:END -->
