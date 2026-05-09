---
title: >-
  [Paper Note] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
description: >-
  [NeurIPS 2025][LLM Alignment][LLM unlearning] This paper identifies that reference model bias in NPO (Negative Preference Optimization) leads to uneven optimization power allocation across forget data and early-stage gradient weight smoothing failure. The proposed SimNPO eliminates reference model dependency and adopts length-normalized rewards, improving FQ from 0.79 to 0.99 on TOFU and consistently outperforming NPO across all benchmarks.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - LLM unlearning
  - negative preference optimization
  - SimNPO
  - reference model bias
  - length normalization
date: 2026-05-08
content_hash: 3fe1e95a0e9f3fd8
---

# Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

**Conference**: NeurIPS 2025
**arXiv**: [2410.07163](https://arxiv.org/abs/2410.07163)
**Code**: [GitHub](https://github.com/OPTML-Group/Unlearn-Simple)
**Area**: LLM Alignment
**Keywords**: LLM unlearning, negative preference optimization, SimNPO, reference model bias, length normalization

## TL;DR
This paper identifies that reference model bias in NPO (Negative Preference Optimization) leads to uneven optimization power allocation across forget data and early-stage gradient weight smoothing failure. The proposed SimNPO eliminates reference model dependency and adopts length-normalized rewards, improving FQ from 0.79 to 0.99 on TOFU and consistently outperforming NPO across all benchmarks.

## Background & Motivation

**Motivation for LLM Unlearning**: Removing the influence of copyrighted, private, or harmful content from LLMs without costly retraining.

**Limitations of GA (Gradient Ascent)**: Lacks divergence control, often causing model collapse.

**Progress and Limitations of NPO**:
- NPO treats forget data as negative samples in DPO, providing a bounded unlearning objective and adaptive gradient weight smoothing.
- However, the authors are the first to identify **reference model bias** in NPO — reliance on the reference model to assess unlearning effectiveness may mislead optimization.

## Method

### Analysis of Reference Model Bias in NPO

**Limitation L1: Uneven Optimization Power Allocation**

The NPO gradient weight is $w_\theta(x,y) = \frac{2\pi_\theta(y|x)^\beta}{\pi_\theta(y|x)^\beta + \pi_{\text{ref}}(y|x)^\beta}$.

For strongly memorized data (high $\pi_{\text{ref}}$), the weight is paradoxically smaller, allocating less optimization power. Yet strongly memorized data is harder to forget and should receive more power.

Weakly memorized data receives excessive power → potential over-forgetting → wasted optimization budget.

**Empirical Validation**:
- Strongly vs. weakly memorized data: NPO yields FQ near 0 on strongly memorized data.
- Short vs. long response data: NPO performs poorly on short responses (FQ=0.58) and better on long ones (FQ=0.81).

**Limitation L2: Early-Stage Gradient Weight Smoothing Failure**

At initialization $\theta \approx \theta_{\text{ref}}$, so $w_\theta(x,y) \approx 1$, making NPO equivalent to GA in early training and potentially causing significant utility degradation.

### SimNPO

SimNPO replaces NPO's reference model comparison with the length-normalized reward from SimPO (reference-free preference optimization):

$$\ell_{\text{SimNPO}}(\theta) = \mathbb{E}_{(x,y)\in\mathcal{D}_f}\left[-\frac{2}{\beta}\log\sigma\left(-\frac{\beta}{|y|}\log\pi_\theta(y|x) - \gamma\right)\right]$$

where $|y|$ is the response length and $\gamma \geq 0$ is the reward margin parameter (default $\gamma=0$).

### Gradient Analysis of SimNPO

$$\nabla_\theta \ell_{\text{SimNPO}} = \mathbb{E}\left[\frac{2(\pi_\theta(y|x))^{\beta/|y|}}{1+(\pi_\theta(y|x))^{\beta/|y|}} \cdot \frac{1}{|y|} \cdot \nabla_\theta \log\pi_\theta(y|x)\right]$$

**Advantage (a)**: Length normalization $1/|y|$ reduces weight for longer responses, avoiding uneven allocation. In the limit $\beta \to 0$, SimNPO reduces to weighted GA: $\mathbb{E}[1/|y| \cdot \nabla_\theta \log\pi_\theta]$, rather than NPO's plain GA.

**Advantage (b)**: The weight $w'_\theta(x,y) < 2/|y|$ depends on data characteristics rather than the reference model, eliminating NPO's early-stage $w \approx 1$ issue.

### Loss & Training

The full SimNPO objective: $\min_\theta \ell_{\text{SimNPO}}(\theta) + \lambda \mathbb{E}_{(x,y) \in \mathcal{D}_r}[-\log\pi_\theta(y|x)]$

Forget loss + cross-entropy regularization on the retain set.

## Key Experimental Results

### TOFU Forget05 (LLaMA2-7B-chat)

| Method | FQ↑ | MU↑ | Note |
|--------|-----|-----|------|
| Original | 0.00 | 0.62 | No unlearning |
| Retrain | 1.00 | 0.62 | Gold standard |
| GA | ~0 | 0.00 | Model collapse |
| GradDiff | ~0 | 0.56 | Insufficient unlearning |
| IDK | ~0 | 0.57 | Insufficient unlearning |
| NPO | 0.79 | 0.57 | Baseline |
| **SimNPO** | **0.99** | **0.58** | Best |

### MUSE News (LLaMA2-7B)

| Method | PrivLeak (→0) | KnowMem $\mathcal{D}_r$↑ |
|--------|---------------|--------------------------|
| NPO | 108.91 | 37.58 |
| **SimNPO** | **72.93** | **39.65** |
| Retrain | 0.00 | 53.79 |

SimNPO more closely approximates Retrain across all metrics.

### Strongly vs. Weakly Memorized Data

| Data Type | NPO FQ | SimNPO FQ | Retrain FQ |
|-----------|--------|-----------|------------|
| Strongly memorized | ≈0 | Significant improvement | Reference |
| Weakly memorized | Over-forgotten | Moderately forgotten | Reference |

SimNPO's distribution more closely matches Retrain, validating the reference model bias hypothesis.

### Gradient Weight Analysis

| Stage | NPO w | SimNPO w' |
|-------|-------|-----------|
| Epoch 1 | ≈1 (uniform) | Modulated by $1/|y|$ |
| Epoch 2–3 | Begins to differentiate | Prioritizes short responses |
| Epoch 10 | Fully differentiated | Approaches uniform |

SimNPO allocates different weights based on data difficulty from the very beginning.

### Relearning Attack Robustness

- SimNPO maintains higher FQ under both random and shortest-response relearning attacks.
- NPO is particularly vulnerable to shortest-response relearning attacks.
- SimNPO exhibits slower FQ degradation.

### Markov Chain Synthetic Experiments

Two core advantages validated:
1. SimNPO achieves more balanced forgetting across data of varying lengths.
2. SimNPO achieves more balanced forgetting across data of varying memorization degrees.

NPO over-forgets weakly memorized data and under-forgets strongly memorized data.

## Highlights & Insights

1. **Discovery of Reference Model Bias**: The first work to identify this fundamental problem in NPO, validated through reference model perturbation experiments and stratified data analysis.
2. **Simplification as Improvement**: Removing reference model dependency yields better results; length normalization provides more principled data-aware modulation.
3. **Theoretical and Synthetic Experimental Support**: Markov Chain synthetic experiments precisely control forgetting difficulty, cleanly validating the hypotheses.
4. **Relearning Attack Robustness**: SimNPO's robustness to short-response relearning attacks explains the value of length normalization.

## Limitations & Future Work

1. SimNPO still relies on promoting divergence for unlearning, inevitably incurring some utility loss.
2. Balancing unlearning effectiveness and utility retention in knowledge unlearning settings (e.g., WMDP) remains challenging.
3. Theoretical guarantees for SimNPO have yet to be established.
4. The choice of $\gamma$ affects the strictness of the forgetting condition and requires task-specific tuning.
5. Whether length normalization is optimal across all settings remains to be verified.

## Related Work & Insights

- **NPO**: The direct improvement target; SimNPO preserves the bounded loss advantage while removing reference model dependency.
- **SimPO**: A reference-free method for preference optimization; SimNPO transfers its core idea to the unlearning setting.
- **DPO/GA/GradDiff**: Other unlearning baselines; GA lacks divergence control, GradDiff yields insufficient unlearning.
- **Insight**: The connection between preference optimization and unlearning optimization merits deeper exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ — The design insight (reference model bias) is profound, though the method itself is a transfer of SimPO to the unlearning setting.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks (TOFU, MUSE, WMDP), synthetic experiments, relearning attacks, and gradient analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem analysis progresses rigorously from intuition to mathematics to empirical validation.
- Value: ⭐⭐⭐⭐⭐ — Provides direct practical guidance for LLM unlearning; SimNPO is simple and easy to apply.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](../../ICLR2026/llm_alignment/reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[NeurIPS 2025\] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)

<!-- RELATED:END -->
