---
title: >-
  [Paper Note] PT2-LLM: Post-Training Ternarization for Large Language Models
description: >-
  [ICLR 2026][LLM/NLP][Ternarization] This paper proposes PT2-LLM, the first post-training ternarization framework for LLMs. Through an asymmetric ternary quantizer (featuring iterative ternary fitting and activation-aware…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "Ternarization"
  - "Post-Training Quantization"
  - "Ultra-Low Bit-Width"
  - "LLM Compression"
  - "Column Reordering"
date: 2026-05-08
content_hash: b9444eb6bf602fc2
---

# PT2-LLM: Post-Training Ternarization for Large Language Models

**Conference**: ICLR 2026  
**arXiv**: [2510.03267](https://arxiv.org/abs/2510.03267)  
**Code**: [GitHub](https://github.com/XIANGLONGYAN/PT2-LLM)  
**Area**: LLM/NLP  
**Keywords**: Ternarization, Post-Training Quantization, Ultra-Low Bit-Width, LLM Compression, Column Reordering

## TL;DR

This paper proposes PT2-LLM, the first post-training ternarization framework for LLMs. Through an asymmetric ternary quantizer (featuring iterative ternary fitting and activation-aware grid alignment) and a structural similarity reordering strategy, it achieves superior performance over 2-bit PTQ methods at 1.58-bit precision.

## Background & Motivation

Ternarization (constraining weights to $\{-1, 0, +1\}$) represents an extreme compression scheme:
- Compared to low-bit quantization (2–4 bit), ternarization eliminates most floating-point multiplications, requiring only addition operations.
- Compared to binarization, ternarization better matches the unimodal weight distributions of LLMs and offers stronger representational capacity.

Existing ternarization methods (BitNet b1.58, TernaryLLM) rely on QAT, which is impractical for LLMs. PTQ-based ternarization faces two key challenges:
1. Ternary parameters cannot be optimized via gradients — training-free parameter optimization is required.
2. Weight distributions are dispersed and contain outliers — quantization error is amplified at ultra-low bit-widths.

## Method

### Overall Architecture

PT2-LLM consists of two core components: an Asymmetric Ternary Quantizer (ATQ) and Structural Similarity Reordering (SSR), applied block-by-block within the GPTQ framework.

### Key Designs

1. **Asymmetric Ternary Quantizer (ATQ)**:

    - Introduces a row-wise offset $\mu$: $\hat{\mathbf{W}} = \alpha \mathbf{T} + \mu$, adapting to non-zero-mean weight distributions.
    - **Iterative Ternary Fitting (ITF)**: Alternately optimizes the ternary grid and the ternary matrix.
        - Optimal grid (closed-form solution): $\alpha^* = \frac{m \cdot (\mathbf{W} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1}) \circ (\mathbf{W}\mathbf{1})}{m \cdot (\mathbf{T} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})^2}$
        - Flexible rounding: $\mathbf{T}_{ij}^* = \arg\min_{t \in \{-1,0,1\}} |Z_{ij} - t|$, where $Z_{ij} = (W_{ij} - \mu_i^*) / \alpha_i^*$
        - Convergence within approximately 10 iterations.
    - **Activation-aware Grid Alignment (AGA)**: Uses calibration data to minimize output error $\mathcal{E}_x = \|\mathbf{WX} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$.

2. **Structural Similarity Reordering (SSR)**:

    - Motivation: In naive block-wise ternarization, weights within a block exhibit high variance and contain outlier columns.
    - Computes inter-column cosine similarity: $S_{ij} = \frac{\mathbf{W}_{:,i}^\top \mathbf{W}_{:,j}}{\|\mathbf{W}_{:,i}\|_2 \|\mathbf{W}_{:,j}\|_2}$
    - Clusters structurally similar columns within the same block, resulting in more compact intra-block distributions.
    - Lightweight strategy: at each step, selects the top-$k$ columns most similar to a mean reference to form the next quantization block.

### Loss & Training

- The ITF stage minimizes weight quantization error $\mathcal{E}_w = \|\mathbf{W} - \hat{\mathbf{W}}\|_F^2$.
- The AGA stage minimizes output error $\mathcal{E}_x = \|\mathbf{WX} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$.
- AGA updates $(\alpha, \mu)$ only once (with $\mathbf{T}$ frozen) to avoid overfitting on the calibration set.
- Quantization block size is set to 128, integrated within the GPTQ framework.

## Key Experimental Results

### Main Results (LLaMA-7B Zero-Shot QA)

| Method | #W (bit) | Wiki2 PPL ↓ | C4 PPL ↓ | 7-Task Avg. Acc ↑ |
|--------|---------|------------|---------|-----------------|
| FP16 | 16 | 5.68 | 7.34 | 61.73% |
| AWQ 2-bit | 2 | 2.60e5 | 2.86e5 | 32.50% |
| GPTQ 2-bit | 2 | 129.19 | 79.06 | 34.35% |
| Slim-LLM 2-bit | 2 | 14.58 | 30.71 | 39.74% |
| PB-LLM 1.7-bit | 1.7 | 82.76 | 76.63 | 33.44% |
| **PT2-LLM 1.58-bit** | 1.58 | **11.39** | **24.55** | **45.07%** |

### LLaMA-13B Results

| Method | #W (bit) | Wiki2 PPL ↓ | 7-Task Avg. Acc ↑ |
|--------|---------|------------|-----------------|
| FP16 | 16 | 5.09 | 63.81% |
| GPTQ 2-bit | 2 | 20.46 | 41.00% |
| **PT2-LLM 1.58-bit** | 1.58 | **8.93** | **49.14%** |

### Key Findings

- PT2-LLM surpasses all 2-bit PTQ methods at 1.58-bit with lower memory footprint.
- The two-stage optimization of ITF and AGA reduces weight error and output error respectively.
- SSR effectively reduces intra-block variance; clustering outlier columns together renders them no longer anomalous.
- Inference acceleration is achieved end-to-end in both prefill and decode stages.

## Highlights & Insights

- First work to achieve LLM ternarization under the PTQ setting, filling an important gap.
- The alternating optimization strategy of ITF is elegant — each step admits a closed-form optimal solution without gradient-based optimization.
- A critical design decision in AGA: freezing $\mathbf{T}$ and updating only the grid parameters effectively prevents overfitting.
- The intuition behind SSR is incisive: "outliers are no longer outliers among themselves."

## Limitations & Future Work

- The 1.58-bit accuracy still lags significantly behind FP16 (e.g., LLaMA-7B average accuracy 45% vs. 62%).
- SSR incurs additional overhead from recomputing similarities at each step.
- No direct comparison with QAT-based ternarization methods (e.g., BitNet b1.58).
- Evaluation is limited to the LLaMA family; models such as Qwen and Mistral are not covered.

## Related Work & Insights

- Relationship to GPTQ: PT2-LLM performs ternarization within the GPTQ framework, inheriting its block-wise error compensation.
- Distinction from BitNet b1.58: PT2-LLM is a PTQ approach that requires no training from scratch.
- Takeaway: Substantial room remains for ultra-low-bit PTQ; asymmetric quantization and structure-aware reordering are effective directions.

## Rating

- Novelty: ⭐⭐⭐⭐ PTQ-based ternarization is an unexplored setting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-task validation with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and visualizations are intuitive.
- Value: ⭐⭐⭐⭐ Provides a new option for ultra-low-bit LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] Neural Synchrony Between Socially Interacting Language Models](neural_synchrony_between_socially_interacting_language_models.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)

</div>

<!-- RELATED:END -->
