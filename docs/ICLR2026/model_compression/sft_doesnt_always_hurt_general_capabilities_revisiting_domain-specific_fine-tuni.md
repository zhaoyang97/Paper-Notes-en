---
title: >-
  [Paper Note] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs
description: >-
  [ICLR 2026][Model Compression][SFT] This paper systematically revisits the impact of domain-specific SFT on the general capabilities of LLMs. It finds that **using a smaller learning rate can significantly mitigate general capability degradation**, and proposes the Token-Adaptive Loss Reweighting (TALR) method to further optimize the trade-off between do
tags:
  - ICLR 2026
  - Model Compression
  - SFT
date: 2026-05-08
content_hash: a6edccfa392133bb
---
# SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs

**Conference**: ICLR2026  
**arXiv**: [2509.20758](https://arxiv.org/abs/2509.20758)  
**Code**: Not open-sourced  
**Area**: Model Compression  
**Keywords**: SFT, domain fine-tuning, general capability degradation, learning rate, token-adaptive reweighting, continual learning, LLM

## TL;DR

This paper systematically revisits the impact of domain-specific SFT on the general capabilities of LLMs. It finds that **using a smaller learning rate can significantly mitigate general capability degradation**, and proposes the Token-Adaptive Loss Reweighting (TALR) method to further optimize the trade-off between domain adaptation and general capabilities by adaptively down-weighting the loss of low-probability tokens.

## Background & Motivation

1. **Domain SFT is a standard paradigm**: While large language models perform excellently on general tasks, they still require SFT to inject domain-specific knowledge for specialized fields such as healthcare and e-commerce.
2. **General capability degradation is widely reported**: Numerous studies suggest that SFT on domain data severely harms general capabilities like mathematical reasoning, code generation, and instruction following, raising doubts about the practicality of SFT.
3. **Prior studies used excessively large learning rates**: Most existing works employ relatively large learning rates such as 5e-6 or 2e-5, which may be one reason why the degradation phenomenon has been overstated.
4. **Data-oblivious settings are more realistic**: In practical scenarios, pre-training data is often inaccessible. Therefore, mitigation strategies that do not rely on original data are of higher value.
5. **Lack of token-level analysis**: Previous research primarily analyzed degradation at the sample or benchmark level, lacking a fine-grained understanding of the learning difficulty of individual tokens within the training data.
6. **Lack of theoretical support**: There is a deficiency in formal analysis from an information-theoretic perspective regarding why the learning rate affects the degree of general capability degradation.

## Method

### Overall Architecture

The paper establishes its argument with a counter-intuitive empirical finding: by reducing the learning rate of domain SFT (e.g., to 1e-6), general capability degradation can be significantly mitigated while domain performance remains almost unchanged. It then provides a formal explanation for this phenomenon using information theory, pinpointing the root cause of degradation to the gradient contributions of a small number of "hard tokens" in the training data. Finally, following this insight, it proposes TALR (Token-Adaptive Loss Reweighting). In a realistic data-oblivious setting (where pre-training data is unavailable), TALR uses adaptive weights to suppress the loss of hard tokens, further improving the trade-off between domain adaptation and general capabilities.

### Key Designs

**1. Small Learning Rate for a Good Trade-off: Debunking the Convention that "Large Learning Rates are Better"**

The authors systematically scanned learning rates on two datasets, MedCalc (medical calculation) and ESCI (e-commerce classification), arriving at a conclusion contrary to traditional deep learning experience: using a smaller learning rate (e.g., 1e-6) can significantly reduce general capability degradation while achieving domain performance nearly equal to that of larger learning rates. This suggests that the conclusion in prior literature—"SFT severely harms general capabilities"—partially stems from the use of larger learning rates like 5e-6 or 2e-5. It was further discovered that the composition of the training objective also matters: when supervision signals contain only labels without CoT (Chain of Thought) reasoning, the learning rate range for reaching the Pareto optimum is broader, and 5e-6 can also perform well. These two points together elevate "learning rate" from an overlooked hyperparameter to a core position in mitigating degradation.

**2. Degradation Upper Bound from an Information-Theoretic Perspective: Quantifying General Capability Change as Code Length**

To explain why small learning rates preserve general capabilities, the authors view the LLM as a data compressor and perform formal analysis using token trees and arithmetic coding frameworks. The core conclusion is that when a model is updated from parameter $\theta_1$ to $\theta_2$, the change in the expected code length for general data is exactly equal to the difference in their KL divergence; thus, the loss of general capability can be precisely quantified. On this basis, it is proved that a smaller distribution update step $\lambda$ (corresponding to a small learning rate) can lower the upper bound of general performance degradation, providing theoretical support for Finding 1. This analysis also explains Finding 2: when the number of hard tokens decreases, the range of the "safe step size" expands, allowing label-only training to tolerate larger learning rates.

**3. TALR: Adaptive Reweighting by Token Probability to Suppress the True Source of Degradation**

Since theory points to hard tokens (tokens with low current probability in the model) as the main driver of degradation, TALR directly reduces the weights of these tokens at the loss level. This avoids manual challenges such as selecting hard tokens, determining thresholds, or deciding the degree of weight reduction. It is formulated as a constrained optimization on a simplex: minimizing the weighted loss plus an entropy regularization term,

$$\min_{\mathbf{w}\in\Delta_n}\ \sum_i w_i\,\ell_i(\theta)+\tau\sum_i w_i\log w_i,$$

where the first term favors low-loss tokens and entropy regularization prevents excessive weight concentration. This problem has a closed-form solution $w_i^* \propto p_\theta(x_i)^{1/\tau}$—simple tokens with high probability receive larger weights, while low-probability hard tokens are automatically suppressed, requiring no additional hyperparameter search. The temperature $\tau$ is not manually tuned; instead, it matches the median token loss within each batch and automatically decays as training progresses. Consequently, the weight distribution changes dynamically: it focuses on easy-to-learn tokens in early training and gradually incorporates the original hard tokens as the model improves, reflecting curriculum learning dynamics (verified by the steady increase in the proportion of tokens with $p>0.2$ from Epoch 1 to Epoch 2). Two engineering details ensure stability: weights $w_i$ are calculated via stop-gradient and do not participate in backpropagation to avoid coupling and oscillation between weights and losses; meanwhile, a lower bound $w_{\min}$ is set for weights ($w_i\leftarrow\max(w_i,w_{\min})$) to prevent hard tokens from being assigned zero weight, ensuring they eventually learn domain knowledge.

## Key Experimental Results

### Table 1: MedCalc Benchmark - Domain/General Performance Comparison at Learning Rate 1e-6

| Method | Qwen2.5-3B Domain | Qwen2.5-3B General | Qwen3-4B Domain | Qwen3-4B General | Avg. Domain | Avg. General |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard (Ours) | 0.495 | 0.620 | 0.548 | 0.784 | 0.534 | 0.692 |
| L2-Reg | 0.490 | 0.621 | 0.469 | 0.796 | 0.506 | 0.697 |
| LoRA | 0.126 | 0.583 | 0.195 | 0.764 | 0.181 | 0.490 |
| Wise-FT | 0.195 | 0.629 | 0.143 | 0.788 | 0.198 | 0.727 |
| FLOW | 0.364 | 0.597 | 0.477 | 0.787 | 0.469 | 0.692 |
| **TALR (Ours)** | **0.481** | **0.648** | **0.489** | **0.788** | **0.501** | **0.717** |

Under a small learning rate, the gap between methods is small; TALR is optimal in maintaining general capabilities.

### Table 2: MedCalc Benchmark - Domain/General Performance Comparison at Learning Rate 5e-6

| Method | Avg. Domain | Avg. General |
|------|:---:|:---:|
| Standard | 0.558 | 0.381 |
| L2-Reg | 0.555 | 0.395 |
| FLOW | 0.553 | 0.450 |
| **TALR (Ours)** | **0.542** | **0.502** |

At a larger learning rate, general capability degradation intensifies. TALR shows the most significant advantage—general performance is 12 percentage points higher than Standard.

### Key Findings at Token Level

- Most SFT training tokens have low learning difficulty for LLMs (median probability close to 1.0), even if the model's zero-shot performance on the domain task is poor.
- A small number of hard tokens mainly appear at domain-specific concepts (e.g., clinical conversion factors), which are the performance bottlenecks.
- During TALR training, the proportion of tokens with $p>0.2$ steadily grows from Epoch 1 to Epoch 2, exhibiting curriculum learning dynamics.

## Highlights & Insights

- **Challenging Mainstream Perceptions**: Systematically proved that SFT does not always significantly harm general capabilities; overstated conclusions in prior literature partly stem from improper learning rate selection.
- **Unification of Theory and Practice**: The information-theoretic analysis not only explains empirical phenomena but also directly guides the design of the TALR method.
- **Elegant Design of TALR**: Features a closed-form solution, no extra hyperparameter search ($\tau$ is adaptive), and stop-gradient to ensure stability, making the implementation concise.
- **Clear Practical Guidelines**: (1) Prioritize small learning rates; (2) Employ TALR when a stronger balance is required.

## Limitations & Future Work

- **Degradation Not Fully Eliminated**: No method, including TALR, can completely avoid general capability degradation at larger learning rates.
- **Limited Datasets**: Validated only on MedCalc and ESCI datasets, without covering more domains.
- **Scale Constraints**: Experiments involve only 3B-4B parameter models, leaving applicability to larger models or MoE architectures unverified.
- **Optimal Learning Rate Selection**: Theoretical analysis does not provide a practical rule for automatically selecting the optimal learning rate.
- **Computational Resource Constraints**: The authors acknowledge that a wider range of experimental validation was not conducted due to limited resources.

## Related Work & Insights

| Category | Representative Work | Relation to Ours |
|----------|---------|-----------|
| L2 Regularization | EWC, L2-Reg | Constrains parameter drift, but effects are limited |
| Model Merging | Wise-FT | Domain performance drops significantly; unsuitable for large domain gaps |
| LoRA | Hu et al. 2022 | Low-rank constraints lead to insufficient domain performance |
| Data Reweighting | FLOW | Sample-level easy/hard distinction; Ours proposes a more granular token-level scheme |
| Continual Learning | data-dependent methods | Requires pre-training data, which is infeasible in practical scenarios |

TALR achieves the best Pareto trade-off under the data-oblivious setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Revisiting the overlooked learning rate factor + information-theoretic analysis + token-level adaptive reweighting
- Experimental Thoroughness: ⭐⭐⭐ — Sufficient validation across models and settings, but limited dataset diversity
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, tight integration of theory and experiments
- Value: ⭐⭐⭐⭐ — Directly instructive for LLM domain fine-tuning practices

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](sculpting_subspaces_constrained_full_fine-tuning_in_llms_for_continual_learning.md)
- [\[CVPR 2026\] S2FT: Parameter-Efficient Fine-Tuning in Sparse Spectrum Domain](../../CVPR2026/model_compression/s2ft_parameter-efficient_fine-tuning_in_sparse_spectrum_domain.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/model_compression/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[ICLR 2026\] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs](gradpruner_gradient-guided_layer_pruning_enabling_efficient_fine-tuning_and_infe.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](../../ACL2025/model_compression/domix_an_efficient_framework_for_exploiting.md)

</div>

<!-- RELATED:END -->
