---
title: >-
  [Paper Note] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs
description: >-
  [ICLR2026][Model Compression][SFT] This paper systematically revisits the impact of domain-specific SFT on the general capabilities of LLMs…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "SFT"
  - "domain fine-tuning"
  - "general capability degradation"
  - "learning rate"
  - "token-adaptive reweighting"
  - "continual learning"
  - "LLM"
date: 2026-05-08
content_hash: 7f3b288af5b9f6f6
---

# SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs

**Conference**: ICLR2026
**arXiv**: [2509.20758](https://arxiv.org/abs/2509.20758)
**Code**: Not released
**Area**: Model Compression
**Keywords**: SFT, domain fine-tuning, general capability degradation, learning rate, token-adaptive reweighting, continual learning, LLM

## TL;DR

This paper systematically revisits the impact of domain-specific SFT on the general capabilities of LLMs, demonstrating that **using a smaller learning rate can substantially mitigate general capability degradation**, and proposes Token-Adaptive Loss Reweighting (TALR), which further optimizes the trade-off between domain adaptation and general capability retention by adaptively down-weighting the loss of low-probability tokens.

## Background & Motivation

1. **Domain SFT is a standard paradigm**: LLMs perform well on general tasks but still require SFT to inject domain knowledge for specialized fields such as medicine and e-commerce.
2. **General capability degradation is widely reported**: Multiple studies have noted that SFT on domain data severely impairs general capabilities such as mathematical reasoning, code generation, and instruction following, raising doubts about the practicality of SFT.
3. **Prior work uses excessively large learning rates**: Existing studies commonly adopt learning rates such as 5e-6 or 2e-5, which may be a contributing factor to the exaggeration of degradation phenomena.
4. **Data-oblivious settings are more realistic**: Pre-training data is typically unavailable in practice, making mitigation strategies that do not rely on such data more valuable.
5. **Token-level analysis is lacking**: Prior research has primarily analyzed degradation at the sample or benchmark level, lacking fine-grained understanding of the learning difficulty of individual tokens in training data.
6. **Theoretical justification is absent**: A formal information-theoretic analysis of why learning rate magnitude affects the degree of general capability degradation has not been established.

## Method

### Core Finding: Small Learning Rates Achieve a Favorable Trade-off

The authors conduct systematic experiments on two datasets—MedCalc (medical calculation) and ESCI (e-commerce classification)—and find:

- **Finding 1**: Using a smaller learning rate (e.g., 1e-6) substantially reduces general capability degradation while achieving domain performance comparable to larger learning rates. This stands in sharp contrast to the conventional wisdom in deep learning that larger learning rates yield better downstream performance.
- **Finding 2**: When the training objective covers only labels (without chain-of-thought reasoning), the range of learning rates that achieve Pareto optimality is wider, and 5e-6 also performs well.

### Theoretical Analysis from an Information-Theoretic Perspective

Treating the LLM as a data compressor and leveraging a token tree and arithmetic coding framework, the authors derive:

- **Proposition 3.1**: The expected change in encoding length from model $\theta_1$ to $\theta_2$ equals the difference in KL divergences, which can quantify changes in general capability.
- **Theorem 3.1**: A smaller distribution update step $\lambda$ (corresponding to a smaller learning rate) reduces the upper bound on general capability degradation.
- **Theorem 3.2**: As the number of hard tokens decreases, the safe step-size range expands—explaining why label-only training tolerates larger learning rates.

### TALR: Token-Adaptive Loss Reweighting

The theoretical analysis identifies the gradient contribution of hard tokens (low-probability tokens) as the primary driver of general capability degradation, motivating the proposed TALR method:

1. **Constrained optimization**: Minimizing weighted loss with entropy regularization over the simplex yields the closed-form solution $w_i^* \propto p_\theta(x_i)^{1/\tau}$
2. **Adaptive weights**: High-probability (easy) tokens receive larger weights, while low-probability (hard) tokens are down-weighted
3. **Dynamic $\tau$ parameter**: $\tau$ is set to the median of token losses within the batch and decays automatically during training
4. **Curriculum learning effect**: Training initially focuses on easy tokens and progressively incorporates formerly hard tokens as the model improves
5. **Stop-gradient**: Weight computation is excluded from backpropagation to ensure optimization stability

## Key Experimental Results

### Table 1: Domain/General Performance Comparison on MedCalc Benchmark at Learning Rate 1e-6

| Method | Qwen2.5-3B Domain | Qwen2.5-3B General | Qwen3-4B Domain | Qwen3-4B General | Avg. Domain | Avg. General |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard (Ours) | 0.495 | 0.620 | 0.548 | 0.784 | 0.534 | 0.692 |
| L2-Reg | 0.490 | 0.621 | 0.469 | 0.796 | 0.506 | 0.697 |
| LoRA | 0.126 | 0.583 | 0.195 | 0.764 | 0.181 | 0.490 |
| Wise-FT | 0.195 | 0.629 | 0.143 | 0.788 | 0.198 | 0.727 |
| FLOW | 0.364 | 0.597 | 0.477 | 0.787 | 0.469 | 0.692 |
| **TALR (Ours)** | **0.481** | **0.648** | **0.489** | **0.788** | **0.501** | **0.717** |

At a small learning rate, performance gaps across methods are modest; TALR achieves the best general capability retention.

### Table 2: Domain/General Performance Comparison on MedCalc Benchmark at Learning Rate 5e-6

| Method | Avg. Domain | Avg. General |
|------|:---:|:---:|
| Standard | 0.558 | 0.381 |
| L2-Reg | 0.555 | 0.395 |
| FLOW | 0.553 | 0.450 |
| **TALR (Ours)** | **0.542** | **0.502** |

At a larger learning rate, general capability degradation intensifies; TALR demonstrates the most pronounced advantage—outperforming Standard by 12 percentage points on general performance.

### Token-Level Analysis

- The vast majority of SFT training tokens are of low learning difficulty for LLMs (median probability close to 1.0), even when the model exhibits poor zero-shot performance on domain tasks.
- A small number of hard tokens appear primarily at domain-specific concepts (e.g., clinical conversion factors) and constitute the performance bottleneck.
- During TALR training, the proportion of tokens with $p > 0.2$ increases steadily from Epoch 1 to Epoch 2, exhibiting curriculum learning dynamics.

## Highlights & Insights

- **Challenging prevailing assumptions**: The paper systematically demonstrates that SFT does not always significantly harm general capabilities, and that the exaggerated conclusions in prior literature are partly attributable to inappropriate learning rate selection.
- **Theory and practice unified**: The information-theoretic analysis not only explains empirical observations but also directly informs the design of TALR.
- **Elegant design of TALR**: The method features a closed-form solution, no additional hyperparameter search (with adaptive $\tau$), and stop-gradient for stability—resulting in a clean implementation.
- **Clear practical guidelines**: (1) Prioritize small learning rates; (2) Apply TALR when a stronger balance is required.

## Limitations & Future Work

- **Degradation not fully eliminated**: No method, including TALR, can entirely prevent general capability degradation at large learning rates.
- **Limited dataset coverage**: Validation is conducted only on MedCalc and ESCI, without extending to a broader range of domains.
- **Restricted model scale**: Experiments are limited to models with 3B–4B parameters; applicability to larger models or MoE architectures has not been verified.
- **Optimal learning rate selection**: The theoretical analysis does not provide practical criteria for automatically selecting the optimal learning rate.
- **Computational resource constraints**: The authors acknowledge that resource limitations precluded more extensive experimental validation.

## Related Work & Insights

| Method Category | Representative Work | Relationship to This Paper |
|----------|---------|-----------|
| L2 Regularization | EWC, L2-Reg | Constrains parameter drift, but with limited effectiveness |
| Model Merging | Wise-FT | Causes substantial domain performance drops; unsuitable when the domain gap is large |
| LoRA | Hu et al. 2022 | Low-rank constraints lead to severely insufficient domain performance |
| Data Reweighting | FLOW | Based on sample-level difficulty distinction; this paper proposes a finer-grained token-level scheme |
| Continual Learning | Data-dependent methods | Require pre-training data, which is unavailable in practical settings |

TALR achieves the best Pareto trade-off under a data-oblivious setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Revisits an overlooked learning rate factor, combines information-theoretic analysis with token-level adaptive reweighting
- Experimental Thoroughness: ⭐⭐⭐ — Validated across multiple models and settings, but the variety of datasets is limited
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure with tight integration of theory and experiments
- Value: ⭐⭐⭐⭐ — Provides direct practical guidance for domain fine-tuning of LLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)

</div>

<!-- RELATED:END -->
