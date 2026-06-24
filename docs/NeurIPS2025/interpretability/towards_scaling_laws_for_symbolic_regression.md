---
title: >-
  [Paper Note] Towards Scaling Laws for Symbolic Regression
description: >-
  [NeurIPS 2025][Interpretability][Symbolic Regression] This work presents the first systematic study of scaling laws for symbolic regression (SR), demonstrating that end-to-end Transformer-based SR follows power-law scaling trends across three orders of magnitude of compute, and derives empirical rules for the optimal token-to-parameter ratio ($\approx 15$), as well as batch size and learning rate scaling with model size.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Symbolic Regression"
  - "Scaling Laws"
  - "Transformer"
  - "Power Law"
  - "Compute-Optimal"
date: 2026-05-08
content_hash: 422e2e1a66f0da9a
---

# Towards Scaling Laws for Symbolic Regression

**Conference**: NeurIPS 2025
**arXiv**: [2510.26064](https://arxiv.org/abs/2510.26064)  
**Code**: None  
**Area**: Interpretability / Symbolic Regression
**Keywords**: Symbolic Regression, Scaling Laws, Transformer, Power Law, Compute-Optimal

## TL;DR

This work presents the first systematic study of scaling laws for symbolic regression (SR), demonstrating that end-to-end Transformer-based SR follows power-law scaling trends across three orders of magnitude of compute, and derives empirical rules for the optimal token-to-parameter ratio ($\approx 15$), as well as batch size and learning rate scaling with model size.

## Background & Motivation

- **Symbolic regression** aims to discover underlying mathematical expressions from observational data, offering both interpretability and generalization ability.
- Pretrained Transformer-based SR methods have recently approached the performance of genetic programming, yet **scaling effects** remain almost entirely unstudied — no existing work exceeds $\sim 100$M parameters.
- Motivated by LLM scaling laws (Kaplan et al., Hoffmann et al.), the authors raise the central question: **Do analogous scaling laws exist for SR?** If so, can they guide the design of next-generation SR models?
- Prior work primarily adjusts training details at a fixed scale, lacking systematic analysis of the scale–performance relationship.

## Method

### Overall Architecture

An end-to-end encoder-decoder Transformer architecture is adopted. The input is tabular data (numerical pairs) and the output is a mathematical expression in LaTeX format. The overall pipeline is:

1. **Data generation**: Recursively generate a base expression set → insert random constants + sample datasets.
2. **Encoding**: Each numerical value in the table is split into mantissa and exponent, then projected into an embedding space.
3. **Model inference**: A table-aware encoder (row/column bidirectional attention) feeds into a standard autoregressive decoder to generate expressions.
4. **Evaluation**: Sample 128 expressions and select the one with the highest $R^2$ as the prediction.

### Key Designs

1. **Two-step data generation**:
    - Step 1: Starting from variables $\{x_1, x_2\}$, recursively apply unary operators (exp, sin, neg, sqrt) and binary operators (+, −, ·, ÷) to generate all expression trees with depth ≤ 3; apply SymPy for canonicalization and deduplication, yielding $|E| = 100{,}000$ base expressions.
    - Step 2: For each base expression, sample $k = 3{,}600$ (expression, dataset) pairs — randomly insert integer constants (range −9 to 9, probability $p = 0.2$) and sample 64 data points from a Gaussian mixture distribution.
    - Advantage: Avoids over-sampling bias present in traditional approaches, yielding cleaner training data.

2. **Table-aware encoder architecture**:
    - Traditional methods merge each input point into a single embedding; this work generates independent embeddings for each **cell** in the table.
    - Mantissa and exponent are each up-projected to the embedding dimension and summed.
    - Inspired by tabular foundation models (TabPFN, etc.), each layer performs both **row attention** (across variables) and **column attention** (across data points).
    - The decoder cross-attends only to the updated embeddings of target cells.

3. **End-to-end training pipeline**:
    - Expressions including constants are predicted directly, without BFGS post-processing.
    - Target expressions are represented as LaTeX strings, with constants tokenized digit by digit.
    - All model scales share the same data generation and evaluation protocol, ensuring fair scaling analysis.

### Loss & Training

- **Loss function**: Standard cross-entropy loss between predicted tokens and ground-truth expression tokens.
- **Optimizer**: AdamCPR ($\beta_1 = 0.9$, $\beta_2 = 0.98$) with linear warmup (first 5% of steps) followed by cosine annealing.
- **FLOPs estimation**: $\text{FLOPs} \approx 6 \cdot (N_{enc} \cdot D_{in} + N_{dec} \cdot D_{out})$, where $N = N_{enc} + N_{dec}$ denotes the number of feed-forward parameters.
- **Hyperparameter search strategy**: For each model scale, batch size and learning rate are grid-searched at a token-to-parameter ratio of 20; the optimal configuration is then used to sweep ratios from 5 to 80.

## Key Experimental Results

### Main Results

Detailed architectures and best performance across five model scales (6.5M–93M):

| Model | Dimension | Encoder Layers | Decoder Layers | Attention Heads | Parameters |
|-------|-----------|---------------|---------------|----------------|------------|
| XS | 256 | 3 | 3 | 4 | 6.48M |
| S | 320 | 4 | 4 | 5 | 13.40M |
| M | 384 | 5 | 5 | 6 | 24.01M |
| L | 448 | 7 | 7 | 7 | 45.53M |
| XL | 512 | 11 | 11 | 8 | 93.08M |

Best performance of each model at maximum compute budget:

| Model | Max FLOPs | $\text{Acc}_{\text{solved}}$ | $\text{Acc}_{R^2>0.99}$ | Val. Loss |
|-------|-----------|------------------------------|--------------------------|-----------|
| 6.5M | 7.20e+16 | 0.149 | 0.526 | 0.424 |
| 13.5M | 2.88e+17 | 0.271 | 0.667 | 0.312 |
| 24M | 9.81e+17 | 0.378 | 0.762 | 0.240 |
| 45.5M | 3.53e+18 | 0.519 | 0.835 | 0.168 |
| 93M | 1.47e+19 | 0.597 | 0.883 | 0.105 |

### Ablation Study

- **Token-to-parameter ratio sweep**: Ratios from 5 to 80 are evaluated; the optimal value is approximately $\approx 15$, with a slight upward trend as the compute budget increases, suggesting that data volume should grow slightly faster than model parameters.
- **Batch size scaling**: The optimal batch size increases with model scale — 32 for 6.5M, 128 for 13.5M, and 256 for 93M.
- **Learning rate scaling**: The optimal learning rate increases with compute budget (4.6e-4 for 6.5M and 24M; 1.0e-3 for 93M), which is **opposite** to the trend observed in LLMs where learning rate decreases with scale.

### Key Findings

1. **Power-law scaling**: $\text{Acc}_{\text{solved}}$ grows from $\sim 0.03$ at the lowest compute budget to $\sim 0.60$ at the highest, following a clear power-law trend; extrapolation predicts 0.8 accuracy at $3.8 \times 10^{21}$ FLOPs.
2. **$\text{Acc}_{R^2>0.99}$ improves faster**: Approximate matching is considerably easier than exact matching; the 93M model already achieves 0.883.
3. **No sign of saturation**: The largest model continues to improve at the largest compute budget, suggesting that further scaling can yield additional gains.

## Highlights & Insights

- **First scaling laws for SR**: This work demonstrates that symbolic regression obeys power-law scaling analogous to LLMs, providing a new design principle for the field — systematic performance improvement through scale rather than elaborate tricks.
- **Learning rate trend opposite to LLMs**: The optimal learning rate increases with scale in SR, revealing fundamental differences in training dynamics across task types.
- **Direct practical value of token-to-parameter ratio**: The optimal ratio of $\approx 15$ provides practitioners with an actionable heuristic for resource allocation.
- **Data generation methodology**: Recursive generation combined with canonicalization and deduplication outperforms traditional random sampling, ensuring uniform coverage of expression space.
- **Table-aware encoder**: The row/column bidirectional attention design represents an effective cross-domain transfer from tabular foundation models.

## Limitations & Future Work

- **Limited expression complexity**: Only expressions with ≤ 2 variables and small integer constants are considered; real-world SR tasks typically involve more variables and floating-point constants.
- **Single-seed training**: Due to compute constraints, each configuration is trained with a single seed, introducing result variance.
- **Limited compute range**: Three orders of magnitude provide limited reliability for extrapolation predictions.
- **No comparison with existing SR methods**: The focus is on scaling insights without verifying whether the approach surpasses GP or other deep SR methods.
- **Directions for improvement**:
    - Extend to more variables, floating-point constants, and more complex operator sets.
    - Verify whether end-to-end SR + improved data generation + scaling comprehensively outperforms alternative methods.
    - Train at larger scale (> 100M parameters) to validate extrapolation predictions.

## Related Work & Insights

- **E2E** (Kamienny et al., NeurIPS 2022): The closest predecessor, the first end-to-end Transformer for SR; this work shifts focus from loss engineering to scaling effects.
- **Biggio et al.**: First proposed pretraining Transformers for SR, using a set encoder with BFGS constant optimization.
- **Chinchilla** (Hoffmann et al.): Compute-optimal scaling laws for language models; the methodology is directly borrowed by this work.
- **TabPFN** (Hollmann et al., Nature 2025): A tabular foundation model whose row/column attention architecture is adopted here.
- Core insight: **Symbolic regression should follow the same paradigm as LLMs — systematically leveraging the benefits of scale rather than pursuing elaborate training tricks.**

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce scaling law analysis into symbolic regression, filling an important gap.
- **Technical Depth**: ⭐⭐⭐ — The methodology is relatively standard (Transformer + synthetic data); the primary contribution lies in experimental design and analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five model scales, three orders of magnitude of compute, and systematic hyperparameter sweeps, though single-seed training is a weakness.
- **Value**: ⭐⭐⭐⭐ — Token-to-parameter ratio and batch size/learning rate scaling trends are directly useful to SR practitioners.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, well-designed figures, and prominent presentation of key findings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[ICML 2025\] Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p](../../ICML2025/interpretability/ab_initio_nonparametric_variable_selection_for_scalable_symbolic_regression_with.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)

</div>

<!-- RELATED:END -->
