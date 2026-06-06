---
title: >-
  [Paper Note] Gemstones: A Model Suite for Multi-Faceted Scaling Laws
description: >-
  [NeurIPS 2025][LLM Pretraining][scaling laws] This work releases the Gemstones model suite — an open-source collection of over 4,000 checkpoints spanning 50M–2B parameters and diverse width-depth ratios. Through systemat…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "scaling laws"
  - "width-depth ratio"
  - "compute-optimality"
  - "model design"
  - "convex hull fitting"
date: 2026-05-08
content_hash: 3c256327f39844cc
---

# Gemstones: A Model Suite for Multi-Faceted Scaling Laws

**Conference**: NeurIPS 2025
**arXiv**: [2502.06857](https://arxiv.org/abs/2502.06857)  
**Code**: [GitHub](https://github.com/mcleish7/gemstone-scaling-laws)  
**Area**: Scaling Laws / Model Architecture
**Keywords**: scaling laws, width-depth ratio, compute-optimality, model design, convex hull fitting

## TL;DR

This work releases the Gemstones model suite — an open-source collection of over 4,000 checkpoints spanning 50M–2B parameters and diverse width-depth ratios. Through systematic experimentation, the paper demonstrates that scaling laws are highly sensitive to design choices such as model selection, learning rate scheduling, and cooldown strategies, and proposes a convex-hull-based fitting method to improve scaling law stability under sparse sampling.

## Background & Motivation

- **Background**: Scaling laws serve as a critical guide for LLM training budget allocation, aiming to identify the optimal balance between parameter count and training token count under a given FLOP budget. Kaplan et al. (2020) and Hoffmann et al. (2022, Chinchilla) yield substantially different prescriptions, and the community has long debated the sources of this discrepancy.
- **Limitations of Prior Work**: Existing scaling law studies typically reduce model design to a single dimension (parameter count), fix the width-depth ratio within a narrow range, adopt a single learning rate schedule, and train on limited datasets for a restricted number of tokens. These restrictive design choices may fundamentally distort the resulting scaling law coefficients.
- **Key Challenge**: If scaling laws are highly sensitive to experimental design, then laws fitted under one set of conditions may offer limited guidance to practitioners operating under entirely different architectural and hyperparameter configurations.
- **Goal**: By constructing the most comprehensive open-source model checkpoint suite to date, this work systematically quantifies the degree to which model shape (width vs. depth), learning rate, and cooldown strategy affect scaling laws, and proposes a more robust fitting methodology.
- **Key Insight**: Scaled-down variants of the Gemma architecture are used to conduct large-scale experiments in a two-dimensional space (width × depth) rather than the conventional one-dimensional space (parameter count).
- **Core Idea**: The prescriptions derived from scaling laws depend critically on how experiments are designed and how models are selected — this sensitivity must be quantified and understood rather than ignored.

## Method

### Overall Architecture

The Gemstones project comprises three core components: (1) designing and training a large-scale model suite covering diverse width-depth ratios; (2) proposing a convex-hull-based fitting method to improve upon traditional binning approaches; and (3) systematically ablating design choices to quantify their impact on scaling law coefficients. All 22 distinct model configurations are trained on the Dolma 1.7 dataset for 350B tokens, with checkpoints saved every 2B tokens, along with cooldown and learning rate ablation experiments.

### Key Designs

1. **Multi-Dimensional Model Suite Design**:

    - *Function*: Construct model families covering five parameter scales — 50M, 100M, 500M, 1B, and 2B — each with 3–5 distinct width-depth configurations.
    - *Mechanism*: Model widths range from 256 to 3072 and depths from 3 to 80 layers, yielding 2,222 configurations across 11 width values and 18 depth values. Feasible models within a ±5% parameter tolerance are identified, distributing models across the two-dimensional width-depth space rather than along a conventional one-dimensional parameter-count line.
    - *Design Motivation*: Prior work — including Chinchilla and Pythia — largely confines models to a fixed width-depth line (see Figure 2), making it impossible to disentangle the individual contributions of width and depth. By sampling across the 2D space, Gemstones enables, for the first time, independent analysis of model shape.

2. **Convex Hull Fitting Approach**:

    - *Function*: Replace Chinchilla's binning method with a fitting scheme better suited to sparse, multi-dimensional model distributions.
    - *Mechanism*: The lower convex hull is computed over all models' FLOP–loss curves, $\mathcal{H} = \text{ConvexHull}(\{(\text{FLOP}_i, L_i)\})$, and scaling law lines $\text{parameters} = c \cdot \text{compute}^{\text{exponent}}$ are fitted using only the hull vertices. The convex hull naturally excludes suboptimal models (points above the hull), preventing noise from these points from distorting the fit.
    - *Design Motivation*: Chinchilla's Approach 1 places 250 log-equally-spaced FLOP bins per order of magnitude and selects the best model in each bin. When models are sampled across the 2D width-depth space, however, the "best" model within many bins may in fact be globally suboptimal. The convex hull method focuses exclusively on the true Pareto frontier, yielding greater stability with respect to changes in model selection strategy.

3. **Parameterized Scaling Law Fitting (Enhanced Approach 3)**:

    - *Function*: Fit the parameterized formula $L(p,T) = A/p^{\alpha} + B/T^{\beta} + \varepsilon$ to predict the optimal parameter-to-token ratio.
    - *Mechanism*: L-BFGS is used to minimize a Huber loss ($\delta=10^{-4}$) between empirical log-loss values and model predictions, with multiple random initializations (following Besiroglu et al., 2024) to avoid local optima.
    - *Design Motivation*: Approach 3 is better suited for extrapolation and less sensitive to individual data points than Approach 1. Fitting on different subsets and comparing coefficient stability provides a quantitative measure of the impact of design choices.

### Loss & Training

All models are trained with the AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, weight decay=0.1), using a linear warmup over 80M tokens followed by a constant learning rate. Learning rates are adjusted by model size according to a scalable scheme to support hyperparameter transfer across scales. Batch size is fixed at 4M tokens with a context length of 2048. Cooldown ablations are performed from checkpoints every 10B tokens, with the learning rate linearly decayed to zero over an additional 10% of tokens seen. All models are trained on AMD MI250X GPUs via tensor parallelism.

## Key Experimental Results

### Main Results

Sensitivity of the scaling law slope (parameters $\propto$ compute$^{\text{exponent}}$) to design choices:

| Fitting Configuration | Token Range | Cooldown | LR Ablation | Embeddings | Slope | Δ (vs. Baseline) |
|---|---|---|---|---|---|---|
| Hoffmann original | — | ✗ | ✗ | ✓ | 0.513 | — |
| Approach 1 (baseline) | All | ✗ | ✗ | ✓ | 0.458 | — |
| Approach 1 | ≤100B | ✗ | ✗ | ✓ | 0.499 | +0.041 |
| Approach 1 | >120B | ✗ | ✗ | ✓ | **0.799** | **+0.341** |
| Approach 1 | All | ✗ | ✓ | ✓ | 0.513 | +0.055 |
| Approach 1 | All | ✓ | ✗ | ✓ | 0.597 | +0.139 |
| Approach 3 (baseline) | All | ✗ | ✗ | ✓ | 0.697 | — |
| Approach 3 | ≤100B | ✗ | ✗ | ✓ | 0.699 | +0.002 |
| Approach 3 | >120B | ✗ | ✗ | ✓ | 0.752 | +0.055 |
| Approach 3 (Chinchilla sampling) | All | ✗ | ✗ | ✓ | 0.632 | −0.065 |

### Ablation Study

Effect of width-depth configuration on validation loss and benchmark accuracy (1B-scale models, comparable FLOP budget):

| Model Config (Width×Depth) | Type | Accuracy at 200B Steps | Accuracy at 350B Steps | Trend |
|---|---|---|---|---|
| 2560×8 | Wide & Shallow | Lower | Lower | High wall-clock efficiency |
| 1792×18 | Balanced | Medium | Medium | Balanced |
| 1280×36 | Deep & Narrow | **Highest** | **Highest** | High FLOP efficiency |

Benchmark prediction model fitting quality:

| Prediction Method | Formula | Fit Quality |
|---|---|---|
| Error prediction | $\text{Err}(L)=\epsilon - k\cdot\exp(-\gamma L)$ | Tight |
| Accuracy prediction | $\text{Acc}(L)=\frac{a}{1+e^{-k(L-L_0)}}+b$ | Noisy (due to shape variation) |
| Most predictable benchmarks | ARC, HellaSwag, MMLU | Most stable at small compute scales |

### Key Findings

1. **Scaling laws are highly fragile**: Changing as few as 5 model selections (Chinchilla subsampling) shifts the slope by 0.065–0.100; restricting the token range (≤100B vs. >120B) under Approach 1 produces a slope change as large as 0.341.
2. **A clear width-depth trade-off exists**: Deeper models are superior in terms of FLOPs (lower loss), while wider models are superior in terms of GPU wall-clock time (lower communication overhead under tensor parallelism).
3. **Whether embedding parameters are counted is the primary source of the Kaplan–Hoffmann discrepancy**: This is consistent with the findings of Pearce and Song (2024) and Porian et al. (2024).
4. **Validation set distribution does not affect model ranking**: Losses computed on Dolma, FineWeb, FineWeb-Edu, and DCLM differ only as a vertical shift; relative rankings are entirely consistent.
5. **Approach 3 is more stable than Approach 1**: Slope variation under Approach 3 (Δ < 0.07) is substantially smaller than under Approach 1 (Δ up to 0.34).

## Highlights & Insights

- **Convex hull fitting is an elegant solution to the sparse sampling problem** — when models are no longer densely distributed along a one-dimensional parameter-count line, the "optimal" model selected by traditional binning may be globally suboptimal; the convex hull resolves this issue naturally.
- **The finding that "deep models win on FLOPs, wide models win on time" has significant practical implications** — practitioners should select the optimal model shape based on their own parallelism strategy (with or without pipeline parallelism) rather than blindly following FLOP-based scaling law prescriptions.
- **Changing as few as 5 models can substantially shift the scaling law slope** — this directly challenges the reliability of scaling laws fitted from small model sets, even when the empirical rule of "55 models is sufficient" is satisfied.
- **The infrastructure value of 4,000+ open-source checkpoints is substantial** — practitioners need only record wall-clock time for a small number of steps on their own hardware to translate Gemstones' GPU-hours analysis into results applicable to their own systems.

## Limitations & Future Work

- Transformer expansion factors, vocabulary size, and batch size are held fixed; the impact of varying these dimensions is unexplored.
- Only tensor parallelism is used (no pipeline parallelism); the observed time disadvantage of deeper models may not hold under other parallelism strategies.
- Training is conducted exclusively on Dolma data; the effect of data distribution is not quantified.
- The largest model is 2B parameters; the extrapolation capacity to larger scales requires validation.
- Sparse architectures such as MoE and hybrid architectures (e.g., Mamba+Transformer) are not explored.

## Related Work & Insights

- **vs. Chinchilla (Hoffmann et al., 2022)**: Chinchilla trains approximately 400 models along a fixed width-depth line — Gemstones trains 2,222 configurations in 2D space and quantitatively demonstrates that Chinchilla's prescriptions are highly sensitive to model selection.
- **vs. Pythia (Biderman et al., 2023)**: Pythia is also an open-source model suite but is trained on the now-retired Pile and adopts a single width-depth configuration — Gemstones trains on a currently available dataset and covers diverse model shapes.
- **vs. Gemma 2 (Team et al., 2024b)**: Gemma 2 reports that deeper 9B-scale models outperform wider counterparts but provides sparse details — Gemstones systematically validates this finding in an open, reproducible manner.
- **vs. Alabdulmohsin et al. (2024)**: They study width-depth scaling laws for encoder-decoder ViTs — Gemstones extends analogous analysis to decoder-only LLMs.
- **Insight**: Scaling laws should be regarded as conditional empirical observations rather than universal laws; practitioners must consider the degree to which their own settings match the conditions under which a given scaling law was derived.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The convex hull fitting method is novel and the multi-dimensional scaling law analysis is the first systematic treatment of its kind, though the core contribution remains empirical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4,000+ checkpoints, multi-dimensional ablations, multiple validation sets, and benchmark prediction analyses constitute an exceptionally comprehensive study.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, and intuitive presentation of key findings.
- **Value**: ⭐⭐⭐⭐⭐ — The open-source infrastructure is of immense value to the community, and the "fragility of scaling laws" finding carries an important cautionary message.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/llm_pretraining/model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](../../ICML2026/llm_pretraining/dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)

</div>

<!-- RELATED:END -->
