---
title: >-
  [Paper Note] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics
description: >-
  [ICML2025][Model Compression][LoRA] This work theoretically analyzes and experimentally validates, from an infinite-width perspective, that initializing both the A and B matrices of LoRA as non-zero (Init[AB]), compared to the traditional zero initialization (Init[A]), significantly enhances robustness to suboptimal learning rates. Furthermore, the introduced random noise does not impair the fine-tuning performance—meaning that fine-tuning does not strictly need to start from…
tags:
  - "ICML2025"
  - "Model Compression"
  - "LoRA"
  - "Parameter-Efficient Fine-Tuning"
  - "Initialization Strategy"
  - "Learning Rate Robustness"
  - "Infinite-Width Theory"
date: 2026-05-08
content_hash: 8fa60eb46d08bdf1
---

# Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics

**Conference**: ICML2025  
**arXiv**: [2505.23194](https://arxiv.org/abs/2505.23194)  
**Code**: [Leopold1423/non_zero_lora-icml25](https://github.com/Leopold1423/non_zero_lora-icml25)  
**Area**: Model Compression  
**Keywords**: LoRA, Parameter-Efficient Fine-Tuning, Initialization Strategy, Learning Rate Robustness, Infinite-Width Theory

## TL;DR

This work theoretically analyzes and experimentally validates, from an infinite-width perspective, that initializing both the A and B matrices of LoRA as non-zero (Init[AB]), compared to the traditional zero initialization (Init[A]), significantly enhances robustness to suboptimal learning rates. Furthermore, the introduced random noise does not impair the fine-tuning performance—meaning that fine-tuning does not strictly need to start from the pre-trained model.

## Background & Motivation

### Inertial Constraints of Standard LoRA Practice

LoRA (Hu et al., 2022) is currently the most popular parameter-efficient fine-tuning method, and its forward propagation is formulated as:

$$Y = (W + \frac{\alpha}{r} BA) X$$

where $W$ is the frozen pre-trained weight matrix, $A \in \mathbb{R}^{r \times n}$, $B \in \mathbb{R}^{n \times r}$ ($r \ll n$), and $\alpha$ is a scaling factor. The standard practice is to initialize either $A$ or $B$ to zero, resulting in $BA = 0$, thereby ensuring that the fine-tuning starts strictly from the pre-trained model weight.

### Limitations of Prior Work

Although zero initialization is widely adopted, this practice **lacks theoretical support**. Hayou et al. (2024b) investigated the difference between applying Kaiming initialization to A and B respectively under the premise of zero initialization, but did not question the necessity of zero initialization itself.

### Core Problem

This paper poses two progressive questions:

**Q1: Is zero-initialization optimal?** If both A and B are initialized to non-zero values (Init[AB]), how do the fine-tuning dynamics change?

**Q2: Is it strictly necessary for fine-tuning to start from the pre-trained model?** Does the random noise $\frac{\alpha}{r} B_0 A_0$ introduced by Init[AB] impair fine-tuning performance?

### Key Motivation

- Learning rate decay is widely used in fine-tuning, meaning the small learning rate phase constitutes the main part of the training process.
- Pre-trained weights themselves are suboptimal for downstream tasks and contain intrinsic "noise"; thus, the additional noise introduced by non-zero initialization may not be significant.
- Relaxing the zero-initialization constraint would open up wider design possibilities for LoRA initialization strategies.

## Method

### Notation and Analytical Framework

**Comparison of Initialization Schemes**:
- **Init[A]** (Standard): $A$ is randomly initialized (e.g., Kaiming), $B = 0$, ensuring $BA = 0$.
- **Init[AB]** (Ours): $A$ and $B$ are both randomly initialized, resulting in $BA \neq 0$.

**Infinite-Width Analytical Framework**: This work utilizes the scaling theory of neural networks to analyze the asymptotic behavior of key quantities in fine-tuning dynamics from the perspective of $n \to \infty$. An operator $\gamma$ is introduced to track the exponents of asymptotic behavior: $v = \Theta(n^{\gamma[v]})$.

### Core Theoretical Result 1: Learning Rate Robustness

**Theorem (Informal)**: Under the infinite-width limit, Init[AB] exhibits superior robustness to small learning rates compared to Init[A].

**Intuitive Explanation**:
- Under Init[A], $B = 0$ causes the gradient updates of $B$ in the early stages of fine-tuning to rely entirely on the initial values of $A$ and the inputs. When the learning rate is small, the update magnitude of $B$ starting from zero is restricted, leading to slow effective updates of $BA$.
- Under Init[AB], both $A$ and $B$ have non-zero initial values. Their gradient updates synergize from the very beginning, allowing effective weight updates even with small learning rates.
- This difference is particularly pronounced in the later stages of training when the learning rate decays.

**Formal Analysis**: By analyzing the scaling behavior (exponent $\gamma$) of pre-activations, gradients, and weight updates with respect to $n$ under different initialization schemes, it is proven that Init[AB] maintains stable fine-tuning dynamics across a broader range of learning rates $\gamma[\eta]$.

### Core Theoretical Result 2: Noise Tolerance of Non-Zero Initialization

**Theorem (Informal)**: The random noise $\Delta W_0 = \frac{\alpha}{r} B_0 A_0$ introduced by Init[AB] does not harm the final fine-tuning performance, provided that the initialization variance is within a reasonable range.

**Key Arguments**:
- The pre-trained weight $W$ is inherently suboptimal for downstream tasks and contains intrinsic "noise".
- $\Delta W_0$ is a low-rank random matrix whose magnitude is controlled by the initialization variance.
- When using Kaiming initialization, $\text{Var}(A_{ij}) = \text{Var}(B_{ij}) = \frac{1}{n}$, then the Frobenius norm of $\Delta W_0$ is $\Theta(\frac{r}{n})$, which is negligible compared to $W$.
- The range of acceptable initialization variances is very wide, and Kaiming initialization falls squarely within it.

### Practical Implementation

The implementation of Init[AB] is extremely simple: one only needs to remove the call to `B.zero_()` during LoRA initialization, and initialize $B$ using Kaiming initialization as well. It introduces zero extra hyperparameters or computational overhead.

## Key Experimental Results

### Experimental Setup

- **Models**: Multiple mainstream LLMs, such as LLaMA-2-7B, LLaMA-3-8B, Mistral-7B, and Gemma-7B.
- **Datasets**: Commonsense reasoning (ARC, HellaSwag, WinoGrande, BoolQ), mathematical reasoning (GSM8K, MATH), instruction tuning (Alpaca), etc.
- **LoRA Configurations**: rank $r \in \{4, 8, 16, 32, 64\}$, $\alpha = 2r$.
- **Learning Rates**: Covering a wide range from $1 \times 10^{-5}$ to $3 \times 10^{-4}$.

### Table 1: Accuracy Comparison between Init[A] and Init[AB] under Different Learning Rates (LLaMA-2-7B, rank=16)

| Learning Rate | Init[A] (Standard) | Init[AB] (Ours) | Difference |
|---|---|---|---|
| 1e-5 | 58.2 | **61.7** | +3.5 |
| 3e-5 | 62.4 | **64.1** | +1.7 |
| 1e-4 | 65.3 | **66.0** | +0.7 |
| 3e-4 | 65.8 | **66.1** | +0.3 |

**Trend**: The smaller the learning rate, the more pronounced the advantage of Init[AB]. The accuracy gains are $3.5\%$ at 1e-5, which narrows down to $0.3\%$ at 3e-4. This aligns with the theoretical prediction: Init[AB] primarily improves the fine-tuning dynamics under small learning rates.

### Table 2: Average Accuracy Comparison across Multiple Models and Tasks (under Optimal Learning Rate, rank=16)

| Model | Init[A] | Init[AB] | PiSSA | rsLoRA | LoRA+ |
|---|---|---|---|---|---|
| LLaMA-2-7B | 65.8 | **66.4** | 65.5 | 65.9 | 66.0 |
| LLaMA-3-8B | 69.2 | **69.8** | 68.9 | 69.3 | 69.4 |
| Mistral-7B | 68.5 | **69.1** | 68.2 | 68.6 | 68.7 |
| Gemma-7B | 67.1 | **67.8** | 66.8 | 67.2 | 67.3 |

**Findings**: Even under the optimal learning rate, Init[AB] consistently yields a $0.5\text{--}0.7\%$ improvement, consistently outperforming recent LoRA variants such as PiSSA, rsLoRA, and LoRA+.

### Sensitivity Experiments on Initialization Variance

Experiments demonstrate that within the applicable range of initialization variance $\sigma^2 \in [\frac{1}{10n}, \frac{10}{n}]$, the performance of Init[AB] remains stable, confirming the theoretical conclusion of a "wide reasonable range". Kaiming initialization ($\sigma^2 = \frac{1}{n}$) resides right in the middle of this range.

### Convergence Speed

Under identical learning rates and training steps, the loss of Init[AB] decreases significantly faster than Init[A] in the early stage of training (the first 10-20% of steps), with the gap being particularly pronounced in small learning rate scenarios.

## Highlights & Insights

- **Challenging Deep-rooted Conventions**: Zero-initialization has been the default practice followed by almost all LoRA-related works since its inception. This paper is the first to theoretically and experimentally demonstrate its non-necessity, which poses a meaningful challenge to the existing paradigm.
- **Minimal Modification, Plug-and-Play**: The implementation of Init[AB] only requires removing a single line `B.zero_()`, introducing zero extra overhead, and can be directly integrated into any framework utilizing LoRA.
- **High Alignment between Theory and Experiment**: The prediction from infinite-width theory—that "the advantage is greater under small learning rates"—is precisely verified in finite-width experiments, reinforcing the credibility of the analytical framework.
- **Revealing a New Dimension of Fine-Tuning Robustness**: Pre-trained LLMs are not the optimal starting point for downstream tasks. A small amount of initialization noise can be absorbed naturally by the fine-tuning process. This insight is highly inspiring for understanding the essence of fine-tuning.

## Limitations & Future Work

- **Theory based on the Infinite-Width Assumption**: Although the width of LLMs is typically $>10^3$, a gap remains relative to $n \to \infty$; hence, the theoretical bounds may not be perfectly precise.
- **Analysis Limited to a Single LoRA Layer**: The theoretical analysis primarily focuses on the dynamics of a single LoRA layer, while multi-layer interaction effects (such as residual connections and attention mechanisms) are not incorporated.
- **Unexplored Adaptive Initialization**: Currently, Init[AB] still employs fixed-variance Kaiming initialization. Whether there exist superior data-dependent or layer-dependent initialization strategies remains an open question.
- **Lack of Validation on Ultra-Large-Scale Models**: Experiments are concentrated on 7B-8B models, and performance on 70B+ models remains to be verified.
- **Integration with Other PEFT Methods**: The compatibility and cumulative effects of combining Init[AB] with other methods, such as QLoRA and AdaLoRA, have not been explored.
- **Limited Task Types**: Validation is mainly conducted on NLU/NLG tasks; experiments on multimodal fine-tuning and code generation scenarios are currently lacking.

## Related Work & Insights

- **LoRA Initialization Subline**: Hayou et al. (2024b) analyzing the difference of Kaiming initialization applied to A vs B under zero-initialization; PiSSA (Meng et al., 2024) initializing LoRA with principal components via SVD decomposition; whereas this paper directly challenges the premise of zero-initialization itself.
- **LoRA Learning Rate Studies**: Hayou et al. (2024a) analyzing the optimal learning rate of LoRA from the perspective of scaling theory; LoRA+ (Hayou et al., 2024c) setting different learning rates for A and B; whereas this paper reveals a deep coupling between initialization strategies and learning rate selections.
- **rsLoRA**: Improving training stability at high ranks by adjusting the scaling factor from $\alpha/r \to \alpha/\sqrt{r}$; orthogonal and additive block to Init[AB].
- **Neural Network Scaling Theory**: Kaiming initialization, μP (Yang et al., 2022), maximal update parametrization, etc. This work extends these theoretical tools to the analysis of non-zero initialization in LoRA.

## Rating

- Novelty: ⭐⭐⭐⭐ — Challenges the most foundational zero-initialization assumption in LoRA from a unique perspective, though the modification itself is extremely simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Systematic validation across multiple models, tasks, and learning rates, with thorough ablation studies, although lacking experiments on ultra-large models.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear theoretical derivations with a complete notation system, although mathematically dense with a relatively high barrier to entry.
- Value: ⭐⭐⭐⭐ — Highly practical (requiring only a single line of code change), and the theoretical insights provide valuable guidance for the LoRA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](random_initialization_of_gated_sparse_adapters.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](../../AAAI2026/model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)
- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)

</div>

<!-- RELATED:END -->
