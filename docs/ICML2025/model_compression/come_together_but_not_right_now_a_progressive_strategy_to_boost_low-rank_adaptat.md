---
title: >-
  [Paper Note] Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation
description: >-
  [Model Compression] > Proposes CoTo (Come Together), a progressive training strategy: randomly deactivates LoRA adapters during the early phase of fine-tuning, with the activation probability linearly increasing from 0 to 1, encouraging even gradient distribution across all layers. It theoretically guarantees dropout stability and linear mode connectivity (LMC). Empirical results demonstrate simultaneous improvements in single-task generalization, multi-task merging…
tags:
  - "Model Compression"
date: 2026-05-08
content_hash: f0c5a4ae5f9d7d46
---

# Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation

| Information | Content |
|------|------|
| **Conference** | ICML 2025 |
| **arXiv** | [2506.05713](https://arxiv.org/abs/2506.05713) |
| **Code** | [zwebzone/coto](https://github.com/zwebzone/coto) |
| **Area** | LoRA/Model Fine-tuning |
| **Keywords** | LoRA, Parameter-Efficient Fine-Tuning, Progressive Training, Adapter Dropout, Linear Mode Connectivity, Model Merging, Pruning |

## TL;DR

> Proposes CoTo (Come Together), a progressive training strategy: randomly deactivates LoRA adapters during the early phase of fine-tuning, with the activation probability linearly increasing from 0 to 1, encouraging even gradient distribution across all layers. It theoretically guarantees dropout stability and linear mode connectivity (LMC). Empirical results demonstrate simultaneous improvements in single-task generalization, multi-task merging, and pruning robustness while reducing training overhead.

---

## Background & Motivation

### Existing Issues

LoRA, as the primary parameter-efficient fine-tuning (PEFT) method, reduces the number of trainable parameters by decomposing the weight increment into low-rank matrices $\Delta W = \alpha BA$. However, vanilla LoRA suffers from two key limitations:

**"Lazy Training" Phenomenon**: The lazy dynamics of standard gradient optimization cause the adapter to converge to a sub-optimal local minimum near initialization, which limits the generalization capability of the model.

**Inter-layer Gradient Imbalance**: Adapters in higher layers receive the vast majority of gradient signals and dominate the task performance, while adapters in lower layers are severely underutilized. This not only impairs single-task generalization but also hurts downstream operations like adapter merging and pruning.

### Limitations of Existing Approaches

- **Element-wise/Column-wise Dropout** (e.g., dropout inside the LoRA matrix): Fails to consider the sequential computation of adapters across layers, thus failing to correct the disproportionate gradient updates received by higher-layer adapters.
- **Adaptive Rank Approaches** (e.g., AdaLoRA, ALoRA, LoRA-drop): Automatically adjust the rank of each layer but do not explicitly address the inter-layer optimization imbalance.
- **Initialization/Optimization Improvements** (e.g., PiSSA, LoRA-GA, rsLoRA, LoRA+, LoRA-Pro): Accelerate convergence speed or improve final performance, but similarly do not directly address the inter-layer imbalance.

### Core Motivation

Can we design a **training strategy that requires no architectural changes**, directly intervening in the optimization process to enable balanced utilization across all adapter layers, while simultaneously improving generalization, mergeability, and pruneability?

---

## Method

### Overall Architecture

The core idea of CoTo is extremely simple: **progressively increase the activation probability of each adapter** during fine-tuning.

- **First 75% of Training**: Each adapter is randomly activated with a time-varying probability $p(t)$, where $p(t)$ increases linearly from 0 to 1.
- **Final 25% of Training**: All adapters are fully activated ($p(t)=1$), degenerating into standard LoRA fine-tuning.

This "sparse-to-full" curriculum-like scheduling encourages the model to perform broader exploration within the loss landscape.

### Activation Probability Scheduling

For a total of $T$ training steps, at step $t$, the activation indicator variable for each adapter $i$ is defined as:

$$\delta_i \sim \text{Bernoulli}(p(t))$$

where the scheduling function of the activation probability is:

$$p(t) = \begin{cases} \frac{4t}{3T} & t < \frac{3T}{4} \\ 1 & t \geq \frac{3T}{4} \end{cases}$$

The model output is adjusted accordingly as:

$$\hat{y} = f\left(x_0; \left\{W_i + \delta_i \mathbf{1} \odot \Delta W_i\right\}_{i=1}^{L}\right)$$

The training objective is to minimize the expected loss:

$$\min_{\{\Delta W_i\}} \mathbb{E}_{\boldsymbol{\delta}}\left[\ell(\hat{y}, y)\right]$$

### Key Design 1: Progressive Optimization Perspective

CoTo can be viewed as training a **weighted ensemble of partial LoRA subnetworks**. Defining the expected model prediction when exactly $j$ adapters are activated as $\tilde{y}_j$, the probability of exactly $j$ adapters being activated at step $t$ corresponds to the binomial distribution weights:

$$w_j(p(t)) = \binom{L}{j} p(t)^j (1-p(t))^{L-j}$$

**Theorem 3.1**: When the loss function $\ell(\cdot, y)$ is convex, the expected objective of CoTo is an upper bound on the weighted sum of the individual subnetwork losses:

$$\min_{\{\Delta W_i\}} \mathbb{E}_{\boldsymbol{\delta}}[\ell(\hat{y}, y)] \geq \min_{\{\Delta W_i\}} \sum_{j=1}^{L} w_j(p) \ell(\tilde{y}_j, y)$$

This ensures that CoTo implicitly optimizes the model to perform well **regardless of which subset of adapters is deactivated**, thereby facilitating dropout stability and linear mode connectivity (LMC).

### Key Design 2: Cooperative Game Perspective

By treating each adapter as a "player" in a cooperative game, the **Shapley value** is utilized to quantify the marginal contribution of each adapter. For a subset of adapters $\mathcal{R}$, the value function is defined as:

$$v(\mathcal{R}) = \mathbb{E}_x\left[\ell\left(f\left(x; \{W_i + \delta_i \mathbf{1} \odot \Delta W_i\}_{i=1}^L\right), y\right)\right]$$

The Shapley value is approximated via Multilinear Extension:

$$\varphi_i(v) = \int_0^1 c_i(p) dp, \quad c_i(p) = \mathbb{E}[v(\mathcal{R}_i \cup \{i\}) - v(\mathcal{R}_i)]$$

Experiments show that the Shapley values of vanilla LoRA are highly concentrated in the higher layers (69% concentrated in the top 4/12 layers), whereas CoTo yields more balanced contributions across all layers (deviation of ±8%, and only ±3% for the early-stopped version).

### Computational Savings

When $\delta_i = 0$, adapter $i$ is fully bypassed, eliminating the need to execute matrix multiplications for $A_i$ and $B_i$. Consequently, CoTo reduces both forward and backward computation during the early stage of training.

---

## Key Experimental Results

### Main Results 1: Vision Benchmark (11 Image Classification Tasks, ViT-B/16, rank=2)

| Method | Average Accuracy |
|------|-----------|
| LoRA | 82.95% |
| LoRA-CoTo | **83.48%** (+0.53) |
| DoRA | 83.45% |
| DoRA-CoTo | **83.93%** (+0.48) |
| HiRA | 83.98% |
| HiRA-CoTo | **84.34%** (+0.36) |

CoTo achieves consistent improvements across all LoRA variants.

### Main Results 2: Commonsense Reasoning (8 Tasks, LLaMA-3-8B, rank=32)

| Method | Average Accuracy |
|------|-----------|
| LoRA | 80.79% |
| LoRA-CoTo | **85.02%** (+4.23) |
| DoRA | 85.20% |
| DoRA-CoTo | **85.49%** (+0.29) |
| HiRA | 86.72% |
| HiRA-CoTo | **87.00%** (+0.28) |

On LLaMA-2-7B, LoRA-CoTo improves by 3.02% compared to LoRA (77.61%→80.63%).

### Main Results 3: Mathematical Reasoning (GSM8K, LLaMA-2-7B, rank=8)

| Method | Without CoTo | With CoTo | Gain |
|------|----------|---------|------|
| LoRA | 42.08 | 55.85 | **+13.77** |
| DoRA | 53.07 | 56.56 | +3.49 |
| HiRA | 54.51 | 56.68 | +2.17 |
| rsLoRA | 45.62 | 56.99 | +11.37 |
| LoRA-Pro | 54.23 | 57.16 | +2.93 |

The improvement of CoTo on vanilla LoRA is particularly striking (+13.77), indicating that the base method is most severely affected by lazy training.

### Multi-task Merging (LLaMA-2-7B, 9 Language Understanding Tasks)

| Merging Strategy | Without CoTo | With CoTo | Gain |
|---------|----------|---------|------|
| Fusion | 47.17% | 58.53% | **+11.36** |
| Ensemble | 56.84% | 56.66% | -0.18 |
| LoRA-LEGO | 62.21% | 67.19% | **+4.98** |

Linear Mode Connectivity (LMC) experiments: At the interpolation point $\lambda=0.5$, LoRA-CoTo maintains 79% accuracy, while vanilla LoRA drops to 39%.

### Pruning Robustness

- **Structured Pruning**: CoTo consistently and significantly outperforms LoRA under all pruning modes (alternate layers, lower layers, middle layers, and higher layers).
- **Unstructured Pruning**: At 50% sparsity, LoRA-CoTo achieves 10% higher accuracy than LoRA.

### Ablation Study

| Ablation Item | Key Findings |
|-------|---------|
| First-stage Ratio | 75% is the optimal trade-off point |
| Dropout Strategy | Uniform dropout (CoTo) and high-to-low dropout (CoTo-H) both outperform low-to-high dropout (CoTo-L) |
| Rank Sensitivity | Consistent improvements across rank=8/32/128 |
| Learning Rate Sensitivity | Improvements are observed across 5e-5/1e-4/2e-4 |
| Inserted Modules | Effective when applied to Attention, Projection, and Gating layers |

### Training Time (Single A6000 GPU, Mathematical Reasoning Task)

| Method | Without CoTo | With CoTo | Speedup |
|------|----------|---------|--------|
| LoRA | 7h38min | 7h05min | 7.20% |
| DoRA | 19h00min | 14h30min | **23.69%** |
| HiRA | 11h39min | 8h50min | **24.21%** |

Variants with larger adapter footprints (DoRA, HiRA) yield greater computational savings.

---

## Highlights & Insights

1. **Utterly Simple**: CoTo introduces no structural alterations, securing comprehensive performance gains using only a linearly increasing activation probability schedule—a classic "one-line code improvement."
2. **Dual Support of Theory and Practice**: The progressive optimization perspective (convex loss upper bound) and the cooperative game perspective (Shapley value equalization) explain the efficacy of the method from different angles.
3. **"Four Birds with One Stone"**: Simultaneously improves single-task generalization, multi-task merging, pruning robustness, and training efficiency—benefiting all four dimensions.
4. **Universal Compatibility**: Highly effective across diverse variants including LoRA, DoRA, HiRA, PiSSA, rsLoRA, LoRA+, and LoRA-Pro, acting as an orthogonal training strategy.
5. **Substantial Improvement in Linear Mode Connectivity**: Increases LMC from 39% to 79% at $\lambda=0.5$, which represents a significant milestone for model merging in practical deployments.
6. **Computational "Free Lunch"**: Bypassing some adapters in the early stage accelerates training, yielding a 24% speedup for DoRA.

---

## Limitations & Future Work

1. **Hyperparameter Selection**: Although the 75%/25% split ratio demonstrated optimal experimental performance, whether it requires adjustment for different tasks and model scales remains unexplored.
2. **Optimality of Linear Scheduling**: Is the linear increase of $p(t)$ optimal? Other scheduling schemes like cosine or step functions have not been systematically compared.
3. **Validation on Ultra-Large Models**: Experiments were conducted on models up to LLaMA-2-13B; its efficacy on 70B+ models is yet to be validated.
4. **Joint Optimization Potential**: The paper identifies the joint application of CoTo with adaptive rank schemes (AdaLoRA) and quantization (QLoRA) as future directions, but these combinations have not yet been evaluated.
5. **Theoretical Assumptions**: Theorem 3.1 relies on the convexity assumption of the loss function, whereas loss functions in practical deep networks are typically non-convex.

---

## Related Work & Insights

- **PEFT Methods**: Adapter (Houlsby et al., 2019), Prompt-Tuning (Lester et al., 2021), Prefix-Tuning (Li & Liang, 2021)
- **LoRA Variants**: DoRA (Liu et al., 2024a), HiRA (Huang et al., 2025), LoRA-FA (Zhang et al., 2023a), FourierFT (Gao et al., 2024b), AdaLoRA (Zhang et al., 2023b)
- **LoRA Initialization and Optimization**: PiSSA (Meng et al., 2024), LoRA-GA (Wang et al., 2024b), rsLoRA (Kalajdzievski, 2023), LoRA+ (Hayou et al., 2024), LoRA-Pro (Wang et al., 2024c)
- **Model Merging**: LoraHub (Huang et al., 2023), LoRA-LEGO (Zhao et al., 2024b), ZipLoRA (Shah et al., 2025)
- **Stochastic Regularization**: Dropout (Srivastava et al., 2014), Stochastic Depth (Huang et al., 2016), LayerDrop (Fan et al., 2020)

---

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| **Overall Score** | **⭐⭐⭐⭐** |

The method itself is extremely simple (yielding moderate novelty), but the theoretical analysis is solid, the experimental coverage is exceptionally broad (spanning vision, language, and diffusion models), and the practical value is highly significant—integrating seamlessly into almost any LoRA workflow with near-zero cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ACL 2025\] Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)

</div>

<!-- RELATED:END -->
