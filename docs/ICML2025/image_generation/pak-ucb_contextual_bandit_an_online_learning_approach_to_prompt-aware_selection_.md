---
title: >-
  [Paper Note] PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs
description: >-
  [ICML2025][Image Generation][contextual bandit] This work proposes the PAK-UCB contextual bandit algorithm, which learns independent kernel functions for each generative model to predict the optimal model for a given prompt online, achieving prompt-level generative model/LLM selection and utilizing Random Fourier Features (RFF) to reduce computational overhead.
tags:
  - "ICML2025"
  - "Image Generation"
  - "contextual bandit"
  - "generative model selection"
  - "kernel methods"
  - "UCB"
  - "random Fourier features"
  - "prompt-aware"
date: 2026-05-08
content_hash: 7c7baefd25518e05
---

# PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs

**Conference**: ICML2025  
**arXiv**: [2410.13287](https://arxiv.org/abs/2410.13287)  
**Code**: [github.com/yannxiaoyanhu/dgm-online-select](https://github.com/yannxiaoyanhu/dgm-online-select)  
**Area**: Online Learning / LLM Selection  
**Keywords**: contextual bandit, generative model selection, kernel methods, UCB, random Fourier features, prompt-aware

## TL;DR

This work proposes the PAK-UCB contextual bandit algorithm, which learns independent kernel functions for each generative model to predict the optimal model for a given prompt online, achieving prompt-level generative model/LLM selection and utilizing Random Fourier Features (RFF) to reduce computational overhead.

## Background & Motivation

**Core Problem**: Current selection of generative models typically relies on choosing a single optimal model based on average scores across all prompts. However, different models exhibit significantly different performance on diverse types of prompts. For instance:

- Gemini-2.5-Flash achieves a 92.3% pass rate on Python code completion but only 48.8% on Java→C++ translation, whereas Qwen-Plus shows the opposite trend (86.5% vs 82.3%).
- Stable Diffusion v1.5 achieves a higher CLIPScore on "car" prompts (36.10 vs 35.68) but performs worse on "dog" prompts compared to PixArt-α (36.37 vs 37.24).

**Motivation**: There is a need for an online learning approach to dynamically select the optimal generative model based on the features of the input prompt, rather than relying on global average scores. This is naturally formulated as a **contextual bandit** problem, where the prompt acts as the context, models serve as the arms, and generation quality scores represent the reward.

**Differences from Existing CB Methods**: Traditional LinUCB/KernelUCB assume that all arms share the same weight vector. However, in the context of model selection, feeding the same prompt into different models yields distinct scores, necessitating **per-arm** independent prediction functions.

## Method

### Problem Formulation

Consider $G$ generative models $\mathcal{G} = [G]$. Over $T$ rounds of iteration:

1. At each round $t$, a prompt $y_t \sim \rho$ is received.
2. The algorithm selects a model $g_t$ and samples a response $x_t \sim P_{g_t}(\cdot|y_t)$.
3. A score $s_t = s(y_t, x_t) \in [-1, 1]$ is received.

Goal: Minimize the cumulative regret:

$$\text{Regret}(T) = \sum_{t=1}^{T} \left( s_\star(y_t) - s_{g_t}(y_t) \right)$$

where $s_\star(y) = \max_{g \in \mathcal{G}} s_g(y)$ is the optimal expected score under the given prompt.

### PAK-UCB Algorithm

**Core Assumption (Realizability)**: There exists an RKHS mapping $\phi$ and **per-model independent** weights $w_g^\star$ such that $s_g(y) = \langle \phi(y), w_g^\star \rangle$.

**Algorithm Flow**:

1. Maintain an **independent** observation set $\Psi_g$ for each model $g$ (comprising only the rounds where model $g$ was selected).
2. Use Kernel Ridge Regression (KRR) to estimate the mean $\hat{\mu}_g$ and uncertainty $\hat{\sigma}_g$:
    - $\hat{\mu}_g = k_y^\top (K + \alpha I)^{-1} v$
    - $\hat{\sigma}_g = \alpha^{-1/2} \sqrt{k(y,y) - k_y^\top (K + \alpha I)^{-1} k_y}$
3. Compute the UCB score: $\hat{s}_g = \hat{\mu}_g + \eta \hat{\sigma}_g$
4. Select the model with the highest UCB score: $g_t = \arg\max_g \hat{s}_g$

**Key Designs**: Per-arm kernel function—each model utilizes an independent kernel matrix and weight vector to prevent biases caused by shared weights.

### RFF-UCB: Reducing Computational Complexity

The computational complexity of PAK-UCB at each round is $O(t^3/G^2)$ due to kernel matrix inversion scaling with the amount of data. This is approximated using **Random Fourier Features (RFF)**:

- Projects the prompt into a $2D$-dimensional random feature space: $\varphi(y) = \frac{1}{\sqrt{D}}[\cos(w_1^\top y), \sin(w_1^\top y), \ldots]$
- Replaces kernel ridge regression with linear ridge regression.
- Reduces the computational complexity to $O(tD^2)$ (scaling linearly over time).

**Theoretical Guarantees**:

| Algorithm | Time Complexity Per Round | Space Complexity | Regret Bound |
|------|---------------|-----------|------------|
| PAK-UCB | $O(t^3/G^2)$ | $O(t^2/G)$ | $\tilde{O}(\sqrt{GT})$ |
| RFF-UCB | $O(tD^2)$ | $O(tD)$ | $\tilde{O}(\sqrt{GT})$ |

Both achieve the same regret bound, but RFF-UCB is significantly more efficient computationally.

## Key Experimental Results

### Experimental Setup

- **T2I Models**: Stable Diffusion v1.5, PixArt-α, UniDiffuser, DeepFloyd IF
- **LLMs**: Gemini-2.5-Flash, o3-mini, Deepseek-Chat, Qwen-Plus
- **Evaluation Metrics**: CLIPScore (T2I), pass rate (code generation)
- **Performance Indicators**: Outscore-the-Best (OtB) and Optimal Pick Ratio (OPR)
- **Baselines**: KernelUCB, LinUCB, One-arm Oracle, Random, Naive-KRR

### Main Results

| Scenario | Method | Key Performance |
|------|------|---------|
| T2I (dog/car) | PAK-UCB-poly3 | CLIPScore exceeds One-arm Oracle, high OPR |
| T2I (dog/car) | RFF-UCB-RBF | Close to PAK-UCB with faster computation |
| Code Generation | RFF-UCB-RBF | Dynamically allocates to superior LLMs based on task type |
| Sudoku | RFF-UCB-RBF | Outperforms all baseline methods |

### Key Findings

- PAK-UCB/RFF-UCB **outperforms One-arm Oracle** (which knows the globally optimal single model), validating the advantage of prompt-level selection.
- Significantly outperforms shared-weight LinUCB and KernelUCB, demonstrating the necessity of the per-arm design.
- RFF-UCB dramatically reduces computational costs while maintaining comparable performance.
- The algorithms can adapt to **newly introduced models** and **novel prompt types** (Setup 3).

## Highlights & Insights

1. **Precise Problem Modeling**: Translates the observation that "different models excel at different prompts" into a per-arm contextual bandit framework, overcoming the limitations of shared weights in traditional CB.
2. **Solid Theory**: Both PAK-UCB and RFF-UCB provide a regret bound of $\tilde{O}(\sqrt{GT})$, showing that RFF does not sacrifice theoretical guarantees.
3. **Strong Practicality**: Covers diverse scenarios including T2I, image captioning, LLM code generation, and Sudoku, verifying generalizability.
4. **Dynamic Adaptation**: Capable of handling active model additions and prompt distribution drifts, making it suitable for real-world deployment.
5. **Computational Efficiency**: RFF reduces the cubic complexity to linear, enabling large-scale online deployment.

## Limitations & Future Work

1. **Single Evaluation Metric**: Currently only optimizes a single score (e.g., CLIPScore), without considering multi-objective goals such as generation diversity or novelty.
2. **Kernel Function Selection**: RBF and polynomial kernels require manual hyperparameter tuning ($\sigma$, $\gamma$), lacking an adaptive kernel selection mechanism.
3. **Cold-Start Problem**: Newly added models require an initial exploration period, which can be costly in practice.
4. **Dependence on Prompt Representations**: Relies on embeddings from pre-trained models (e.g., CLIP/RoBERTa), meaning representation quality directly affects selection efficacy.
5. **Unmodeled Query Costs**: Query costs (API pricing, latency) of different models can vary significantly, which has not been structured into cost-aware selection.
6. **Limited Experimental Scale**: T2I experiments mainly focus on selection between two classes in MS-COCO, representing a relatively simple scenario.

## Related Work & Insights

- **FID-UCB** (Hu et al., 2025a): Focuses on MAB-based selection for unconditional generative models; this work extends it to conditional generation (with prompts).
- **Mixture-UCB** (Rezaei et al., 2025): Mixes multiple generative models to enhance diversity.
- **KernelUCB** (Valko et al., 2013): A foundational work in kernelized CB, but assumes shared weights.
- **Offline Routing Methods** (Luo et al., 2024; Qin et al., 2024): Train neural networks for prompt-to-model routing, but require substantial offline data.
- **Insights**: The per-arm kernel method paradigm can be generalized to other online selection scenarios featuring "the same input, different backends" (e.g., API routing, model ensembles).

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying per-arm kernelized CB to prompt-aware model selection represents a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple scenarios including T2I, LLMs, and image captioning, and includes adaptivity experiments.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clear, and the theoretical and empirical elements are well-integrated.
- Value: ⭐⭐⭐⭐ — Addresses practical model routing problems, balancing theory and application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](../../ICCV2025/image_generation/pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICML 2025\] Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction](sample_complexity_of_distributionally_robust_off-dynamics_reinforcement_learning.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](../../ICML2026/image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICLR 2026\] FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference](../../ICLR2026/image_generation/fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference.md)
- [\[ICML 2025\] Stealix: Model Stealing via Prompt Evolution](stealix_model_stealing_via_prompt_evolution.md)

</div>

<!-- RELATED:END -->
