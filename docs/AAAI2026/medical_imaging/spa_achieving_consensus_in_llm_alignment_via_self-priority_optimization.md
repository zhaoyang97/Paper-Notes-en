---
title: >-
  [Paper Note] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization
description: >-
  [AAAI 2026][Medical Imaging][LLM alignment] This paper proposes Self-Priority Alignment (SPA), a fully unsupervised framework that enforces a strict "trustworthiness before helpfulness" priority ordering via lexicographic optimization. The model self-generates diverse responses, self-evaluates, and self-improves; dual-criterion denoising constructs preference pairs; and an uncertainty-weighted SimPO loss fine-tunes the model, simultaneously improving safety and helpfulness across multiple benchmarks.
tags:
  - AAAI 2026
  - Medical Imaging
  - LLM alignment
  - priority alignment
  - lexicographic optimization
  - unsupervised
  - preference learning
  - safety
date: 2026-05-08
content_hash: 7b9379330520d21a
---

# SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization

**Conference**: AAAI 2026
**arXiv**: [2511.06222](https://arxiv.org/abs/2511.06222)
**Code**: None
**Area**: Medical Imaging
**Keywords**: LLM alignment, priority alignment, lexicographic optimization, unsupervised, preference learning, safety

## TL;DR

This paper proposes Self-Priority Alignment (SPA), a fully unsupervised framework that enforces a strict "trustworthiness before helpfulness" priority ordering via lexicographic optimization. The model self-generates diverse responses, self-evaluates, and self-improves; dual-criterion denoising constructs preference pairs; and an uncertainty-weighted SimPO loss fine-tunes the model, simultaneously improving safety and helpfulness across multiple benchmarks.

## Background & Motivation

- **Root Cause in High-Stakes Scenarios**: In medical, legal, and self-harm contexts, LLMs must be both trustworthy (harmless/honest) and helpful, yet these objectives frequently conflict.
    - Example: "What should I do if I have thoughts of self-harm?" — safety must take precedence, but a blanket refusal may leave users feeling dismissed.
- **Three Limitations of Existing Multi-Objective Alignment Methods**:
  1. **Context-agnostic weights**: Static or heuristic weights cannot adapt to dynamic user intent and risk profiles.
  2. **Lack of safety-aware optimization**: Compromise-based balancing may undermine safety in pursuit of helpfulness.
  3. **Data scarcity**: High-quality annotated data capturing genuine trustworthiness–helpfulness trade-offs is scarce.
- **Core Idea**: Introduces a new paradigm of **priority alignment** — primary objectives (e.g., harmlessness) must be satisfied before secondary objectives (e.g., helpfulness) are optimized.

## Method

### Theoretical Foundation: Lexicographic Optimization

Priority alignment is formalized as a lexicographic optimization problem: multiple objectives are optimized in strict priority order.

**Challenge**: The high-dimensional, non-convex parameter space of LLMs renders traditional lexicographic methods infeasible (one cannot fully optimize $G_a$ before $G_b$).

**Solution**: Pareto frontier enumeration is combined with preference optimization (PO) — preference pairs implicitly encode Pareto dominance relations:

- If $G_a(y) \geq G_a(y^-)$ and $G_b(y) > G_b(y^-)$, then $y$ Pareto-dominates $y^-$.
- Collecting a large number of such preference pairs implicitly characterizes the Pareto frontier between $G_a$ and $G_b$.
- DPO/SimPO fine-tuning internalizes this priority structure into the model.

### SPA Framework: Three Components

#### 1. Diverse Sampling + Self-Improvement

**Diverse Sampling**: For each prompt $x_j$, $n$ diverse candidate responses are generated via:
- High-temperature sampling ($\tau > 1$)
- System prompt variants (to increase diversity)

**Self-Evaluation**: Each response is self-scored along two dimensions:
- $s_{a,j}^{(i)} = S_a(x_j, y_j^{(i)})$ (primary objective, e.g., harmlessness)
- $s_{b,j}^{(i)} = S_b(x_j, y_j^{(i)})$ (secondary objective, e.g., helpfulness)
- Scoring functions are grounded in an AI constitution $\mathcal{C}$ defining HHH principles.

**Self-Improvement**: Based on all candidates and their scores, an improved response $\tilde{y}_j$ is generated:

$$\tilde{y}_j \sim \pi_\theta(\cdot | x_j, \{y_j^{(i)}, s_{a,j}^{(i)}, s_{b,j}^{(i)}\}_{i=1}^n, \mathcal{C})$$

#### 2. Dual-Criterion Denoising

**Motivation**: Self-evaluations by weaker models may be biased, inconsistent, and noisy.

**Consistency-Driven Denoising**: Only samples where the improved response strictly outperforms all candidates on both dimensions are retained:

$$\mathcal{Y}_{\text{perf}} = \{(y, s_a, s_b) \in \mathcal{Y} \mid \tilde{s}_a > \max_i s_{a}^{(i)} \text{ and } \tilde{s}_b > \max_i s_{b}^{(i)}\}$$

**Informativeness-Driven Denoising**: RV coefficient analysis shows that high-variance samples degrade weak-to-strong model alignment. Samples with excessively large score variance are filtered:

$$\mathcal{Y}_{\text{final}} = \{(y, s_a, s_b) \in \mathcal{Y}_{\text{perf}} \mid 0 < \det(\Sigma_j) \leq \tau\}$$

where $\Sigma_j$ is the covariance matrix of the two-dimensional scores.

**Empirical Support**: When more than 32% of samples (ranked by variance) are included, the RV coefficient between weak and strong models drops sharply.

#### 3. Preference Dataset Construction and Priority Optimization

**Preference Pair Construction** (lexicographic):

$$(G_a(y), G_b(y)) >_{\text{lex}} (G_a(y^-), G_b(y^-))$$

i.e., $G_a(y) > G_a(y^-)$, or $G_a(y) = G_a(y^-)$ and $G_b(y) > G_b(y^-)$.
A margin constraint $\Delta(y, y^-) \geq \delta$ ensures that preference differences are meaningful.

### Loss Function: Uncertainty-Guided SimPO

$$\mathcal{L}_{\text{SPA}}(\theta) = -\mathbb{E}_{(x,y,y^-) \in \mathcal{D}_{\text{pref}}} \left[ w_i \cdot \log \sigma\left(\frac{\beta}{|y|} \log \pi_\theta(y|x) - \frac{\beta}{|y^-|} \log \pi_\theta(y^-|x) - \gamma\right) \right]$$

- Built on SimPO (length normalization prevents preference for verbose outputs).
- **Uncertainty weighting** $w_i = (\Delta_i / \bar{\Delta})^\alpha$: pairs with larger score margins receive stronger gradients.

## Key Experimental Results

### Setup

- **Models**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct
- **Datasets**: SafeRLHF (harmlessness + helpfulness), WildGuard (same), HoneSet (honesty + helpfulness)
- **Training Scale**: 300–400 prompts (very lightweight)
- **Evaluation**: LLM-as-Judge (GPT-4o) + human evaluation
- **Baselines**: Reward Soups (multi-weight interpolation), Self-Criticism, SFT, SPADPO, SPASimPO

### Main Results (Score-Based Evaluation)

| Method | SafeRLHF Harm.↑ | SafeRLHF Help.↑ | WildGuard Harm.↑ | WildGuard Help.↑ | HoneSet Hon.↑ | HoneSet Help.↑ |
|------|---------|---------|---------|---------|--------|--------|
| Llama Vanilla | 9.62 | 5.23 | 8.22 | 6.09 | 6.30 | 7.75 |
| Llama SFT | 9.68 | 5.57 | 9.79 | 3.20 | 6.11 | 7.66 |
| **Llama SPA** | **9.90** | **7.14** | **8.85** | **6.22** | **7.75** | **7.83** |
| Mistral Vanilla | 8.83 | 7.53 | 6.83 | 7.15 | 5.81 | 7.62 |
| **Mistral SPA** | **9.76** | **8.39** | **7.27** | **7.44** | **7.18** | **7.82** |

Key findings:
- SPA outperforms Vanilla and SFT on **all metrics**, with particularly comprehensive gains on Mistral.
- Helpfulness improves substantially (Llama: 5.23→7.14; Mistral: 7.53→8.39) without sacrificing safety.
- In **pairwise comparisons**, SPA achieves a win rate as high as 86% (HoneSet helpfulness).

### Comparison with Multi-Objective Baselines

| Method | Harm. | Help. | HH₅ | HH₁₀ | HH₂₀ |
|------|-------|-------|------|------|------|
| Self-Criticism | 9.65 | **7.68** | 9.32 | 9.47 | 9.56 |
| RS 9:1 | 9.90 | 6.17 | 9.28 | 9.56 | 9.72 |
| **SPA** | **9.90** | 7.14 | **9.44** | **9.65** | **9.77** |

SPA leads comprehensively on the weighted composite metric $HH_\lambda$ ($\lambda=5,10,20$); the advantage grows as harmlessness priority increases.

### General Capability Retention

- MTBench: SPA outperforms Vanilla in 3 out of 4 configurations (maximum gain +2.52%).
- MMLU: Fluctuation within ±2%, indicating that alignment improvements **do not come at the cost of general capability**.

### Generalization

SPA trained on SafeRLHF transfers directly to unseen datasets:
- JailbreakTrigger: Harm. 9.80 / Help. 6.35 (far exceeding Vanilla and SFT).
- WildGuard: Harm. 9.29 (among the strongest).

### Ablation Study

- **Denoising components**: Removing denoising causes harmlessness and helpfulness to drop by more than 0.1 each.
- **Iteration count**: A second iteration yields improvement (especially when the same prompts are reused); returns diminish at the third iteration.
- **Human evaluation**: Agreement between GPT-4o as judge and human annotations: harmlessness 91–94%, helpfulness 89–92%.

## Highlights & Insights

1. **Priority alignment is a paradigm innovation in alignment research**: Rather than seeking "balance," it establishes a "priority ordering," which better suits high-stakes scenarios.
2. **Fully unsupervised**: No human-annotated data required; the model self-generates, self-evaluates, and self-improves — highly scalable.
3. **Lexicographic preference pairs implicitly encode Pareto dominance**: An elegant theoretical approximation of otherwise intractable lexicographic optimization.
4. **Dual-criterion denoising** effectively mitigates noise in weak-model self-evaluation, with RV coefficient analysis providing theoretical grounding.
5. **Minimal training data** (only 300 prompts) demonstrates the efficiency and practicality of the approach.

## Limitations & Future Work

- Self-evaluation quality is bounded by the model's own capabilities; weaker models may introduce systematic biases.
- Validation is limited to 8B-scale models; larger and smaller models have not been tested.
- The definition and boundaries of high-stakes scenarios still require human judgment and cannot be fully automated.
- Theoretical guarantees for lexicographic optimization are approximate in non-convex settings.
- Evaluation relies primarily on LLM-as-Judge, which may introduce evaluation bias.
- Whether 300 training prompts provide sufficient diversity to cover a broad range of scenarios remains open to question.

## Related Work & Insights

- **Alignment algorithms**: PPO/RLHF, DPO, SimPO, KTO, IPO, and other preference learning methods.
- **Multi-objective alignment**: Reward Soups (linear interpolation of model weights), MetaAligner, RiC (Rewards-in-Context).
- **Self-alignment**: Self-Criticism (HHH principles), Meta-self-alignment.
- **Safety alignment**: SafeDPO, TISDPO.

## Rating ⭐⭐⭐⭐

The method is theoretically elegant (lexicographic optimization + Pareto frontier) and practically efficient (unsupervised + minimal data), with significant empirical gains (simultaneous improvements in safety and helpfulness). Dual-criterion denoising and uncertainty-weighted loss constitute valuable technical contributions. The main limitation is that experiments are confined to 8B models with limited coverage of high-stakes scenarios. Overall, this represents a meaningful advance in LLM safety alignment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[AAAI 2026\] An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](an_llm-based_simulation_framework_for_embodied_conversationa.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning](taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)

<!-- RELATED:END -->
