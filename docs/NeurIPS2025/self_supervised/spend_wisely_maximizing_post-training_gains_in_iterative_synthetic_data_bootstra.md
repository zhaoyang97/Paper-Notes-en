---
title: >-
  [Paper Note] Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping
description: >-
  [NeurIPS 2025][Self-Supervised Learning][synthetic data] This paper provides the first theoretical analysis of the budget allocation problem in iterative synthetic data bootstrapping…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "synthetic data"
  - "iterative bootstrapping"
  - "budget allocation strategy"
  - "exponential growth strategy"
  - "diffusion models"
  - "large language models"
  - "post-training optimization"
date: 2026-05-08
content_hash: c4f21da6cd6b6010
---

# Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2501.18962](https://arxiv.org/abs/2501.18962)  
**Authors**: Pu Yang (Peking University), Yunzhen Feng (NYU), Ziyuan Chen (Peking University), Yuhang Wu (UC Berkeley), Zhuoyuan Li (NUS)  
**Code**: Not released  
**Area**: Image Restoration  
**Keywords**: synthetic data, iterative bootstrapping, budget allocation strategy, exponential growth strategy, diffusion models, large language models, post-training optimization  

## TL;DR

This paper provides the first theoretical analysis of the budget allocation problem in iterative synthetic data bootstrapping, proving that constant strategies fail to converge with high probability, that exponential growth strategies outperform polynomial strategies in the worst case, and validating these findings empirically on image denoising (DPM) and mathematical reasoning (LLM) tasks.

## Background & Motivation

### State of the Field
Modern foundation models commonly adopt an iterative "bootstrapping" paradigm during post-training: the model generates synthetic data → an external verifier filters low-quality samples → the high-quality subset is used for further fine-tuning. Repeating this process over multiple rounds yields continuous performance gains. This paradigm is widely applied to improve LLM reasoning (e.g., STaR), instruction following, alignment training, and visual tasks such as image segmentation and denoising.

### Limitations of Prior Work
- In existing practice, the amount of synthetic data generated per iteration is typically kept **fixed** (constant strategy), with no theoretical justification.
- Ferbach et al. (2024) provide the only theoretical analysis, but prove convergence only under an **infinite-sample assumption**, offering no guidance for finite-sample budget allocation.
- Research on model collapse addresses the risks of synthetic data but does not consider how to optimally allocate the generation budget across iterations.
- For practitioners, how to decide how much synthetic data to generate and select per round under a fixed total compute budget remains an open question.

### Root Cause
The key question is: given a fixed total compute budget, how should the generation and training volume of synthetic data be allocated across iterations to maximize final model performance?

## Method

### Problem Formulation
- **Generative model**: a parameterized model $f(\cdot;\theta)$ generating data $x \sim \mathcal{P}_\theta$
- **Reward model**: $\mathcal{R}(x) \in [0,1]$ evaluates the quality of generated data
- **Objective**: maximize expected reward $r(\theta) = \mathbb{E}_{x \sim \mathcal{P}_\theta}[\mathcal{R}(x)]$
- **Iterative process**: at each round $t$, generate $N_t$ samples, select $n_t$ high-quality samples with probability proportional to $\mathcal{R}(x)$, and update the model via gradient descent
- **Strategy definition**: $\pi: \{n_t\}_{t=0,1,\cdots}$ controls the number of selected samples per round
- **Total compute cost**: $C(\pi, T) = \sum_{t=0}^{T-1} c_g N_t + c_t n_t$, where $c_g$ and $c_t$ are the per-sample costs of generation and training, respectively

### Gaussian Case Analysis (Warm-up)
Under a Gaussian generator $\mathcal{P}_\theta = \mathcal{N}(\theta, \sigma^2)$ and exponential reward $\mathcal{R}(x) = \exp(-x^2/(2\kappa^2))$ with MLE updates, the optimal strategy admits a closed-form solution:

$$n_t \propto \left(1 + \frac{\sigma^2}{\kappa^2}\right)^t$$

**Theorem 3.1** establishes that the optimal strategy is exponential growth — intuitively, the contribution of early updates to the final error decays exponentially with the total number of iterations $T$, so later iterations require exponentially more samples to ensure accuracy.

### Main Theoretical Results (General Setting)

**Theorem 4.1 (Constant strategies do not converge)**: For any constant strategy $\pi_{\text{const}}$ (i.e., $n_t = n_0$), with probability at least $1/4$,
$$r^* - r(\theta_{\pi_{\text{const}}}^{(T)}) \geq c \cdot n_0^{-1/2}$$
Under a constant strategy, the stochastic noise term in the gradient is proportional to $n_t^{-1}$ and does not decay, resulting in an irreducible performance gap.

**Theorem 4.2 (Increasing strategies converge)**: For any increasing strategy $\pi$ (i.e., $n_t$ is monotonically increasing), for any $\varepsilon > 0$, there exists a sufficiently large $T$ such that with high probability $r^* - r(\theta_\pi^{(T)}) \leq \varepsilon$.

**Theorem 4.3 (Exponential strategies converge exponentially)**: There exists an exponential growth strategy $\pi_{\text{exp}}^*$ such that with high probability:
$$r^* - r(\theta_{\pi_{\text{exp}}^*}^{(T)}) = \mathcal{O}((1+\zeta)^{-2T})$$

**Theorem 4.4 (Worst-case optimality of exponential strategies)**: For any constant or polynomial growth strategy $\pi$, in the worst case, the total compute cost required by the exponential strategy to achieve equivalent performance is no greater than that of $\pi$.

### Hierarchy of Results
1. Constant strategy → cannot converge to optimal reward
2. Polynomial growth strategy → convergent but at a slower rate
3. **Exponential growth strategy → exponential convergence and worst-case optimal**

## Key Experimental Results

### Experiment 1: Image Denoising (Diffusion Model)

A pre-trained DDPM is fine-tuned on MNIST for denoising; PSNR serves as the reward signal with training batch size $B=640$.

| Strategy | $n_t$ Setting | s=10 PSNR | s=20 PSNR |
|----------|--------------|-----------|-----------|
| Pre-trained | N/A | 40.92 | 36.96 |
| constant | $1 \cdot B$ | 42.75 | 38.91 |
| constant | $10 \cdot B$ | 43.15 | 39.45 |
| constant | $100 \cdot B$ | 44.18 | 39.91 |
| linear | $t \cdot B$ | 43.90 | 39.90 |
| linear | $3t \cdot B$ | 44.23 | 40.08 |
| linear | $10t \cdot B$ | 44.64 | 40.26 |
| **exponential** | $1.1^t \cdot B$ / $1.05^t \cdot B$ | **44.84** | **40.14** |

- The exponential strategy achieves the best PSNR at $s=10$ (44.84) and is competitive with the best linear strategy at $s=20$.
- Constant strategies initially match the exponential strategy but quickly plateau and degrade due to overfitting.
- Linear strategies rank second overall.

### Experiment 2: Mathematical Reasoning (LLM)

Llama-3-8B-Base is fully fine-tuned and evaluated on three difficulty levels of GSM-Symbolic, with training batch size $B=256$.

| Strategy | $n_t$ Setting | GSM_symbolic | GSM_p1 | GSM_p2 |
|----------|--------------|-------------|--------|--------|
| Pre-trained | N/A | 55.60 | 34.98 | 15.72 |
| constant | $10 \cdot B$ | 60.28 | 38.50 | 16.93 |
| constant | $30 \cdot B$ | 60.16 | 39.68 | 17.73 |
| constant | $100 \cdot B$ | 64.04 | 41.66 | 18.81 |
| linear | $3t \cdot B$ | 62.54 | 38.88 | 17.01 |
| linear | $10t \cdot B$ | 61.96 | 39.94 | 17.29 |
| linear | $30t \cdot B$ | 64.24 | 40.84 | 19.17 |
| **exponential** | $10 \cdot 2^t \cdot B$ | **65.66** | **47.26** | **20.65** |

- The exponential strategy achieves the best results across all three difficulty levels, with a particularly notable gain on GSM_p1 (from 35% to 47%, +12.28 pp).
- Linear strategies rank second overall but cannot match the exponential strategy under large budgets.
- The exponential strategy yields the most consistent improvements without the overfitting observed under constant strategies.

## Highlights & Insights

- **Pioneer theoretical framework**: This work is the first to formally analyze the budget allocation problem in iterative synthetic data training, establishing a complete convergence hierarchy (constant → polynomial → exponential).
- **Practical guidance**: The paper directly informs practice — later iterations should use exponentially more data — challenging the prevailing norm of constant-budget strategies.
- **Cross-modal validation**: Theoretical predictions are consistently validated in both image (DPM denoising) and text (LLM mathematical reasoning) domains.
- **Rigorous theoretical guarantees**: The exponential strategy not only achieves the fastest convergence rate (exponential) but also incurs no greater compute cost than any polynomial strategy in the worst case.
- **Elegant theoretical intuition**: The conclusion that later iterations require exponentially more samples is intuitively compelling, as the influence of early update errors on final performance decays exponentially with the number of iterations.

## Limitations & Future Work

- **Strong theoretical assumptions**: The analysis requires regularity assumptions on the loss function (Assumptions B.1–B.3) and assumptions linking rewards to the loss, which may not fully hold in practical settings.
- **SFT-only framework**: Reinforcement learning-based post-training paradigms such as RLHF and DPO are not considered; the authors identify this as a direction for future work.
- **Oracle reward model**: The image denoising experiments use clean ground-truth images to compute PSNR as the reward, which is typically unavailable in practice.
- **Limited compute scale**: LLM experiments are conducted only on an 8B model and are not validated at larger scales (70B+).
- **Hyperparameter selection**: The exponential strategy requires selecting the base $u$ (e.g., 1.05, 1.1, 2); while ablations are provided, no adaptive selection method is proposed.
- **Data diversity not considered**: The analysis focuses solely on data volume allocation without examining how data diversity evolves across iterations and its impact on strategy design.
- **Single-step gradient descent assumption**: The theoretical analysis assumes single-step gradient descent updates, which differs from multi-step SGD training used in practice.

## Related Work & Insights

- **Ferbach et al. (2024)**: The only existing theoretical analysis, but convergence is proven only under an infinite-sample assumption; this paper is the first to derive optimal budget allocation strategies under finite samples.
- **STaR (Zelikman et al., 2022)**: A classic iterative synthetic data training method that uses a constant strategy; both the theory and experiments in this paper demonstrate that constant strategies are suboptimal.
- **Model Collapse (Shumailov et al., 2024)**: Focuses on the degradation risks of training on synthetic data; this paper avoids degradation via verifier-based filtering and instead studies optimal allocation.
- **Singh et al. (2025), Guan et al. (2025)**: Iterative methods that leverage search algorithms to generate diverse reasoning paths, but do not study budget allocation.
- **El Firdoussi et al. (2025)**: Analyzes the potential of synthetic data from a random matrix theory perspective; complementary in focus.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to formalize and solve the budget allocation problem in iterative synthetic data training; a pioneering theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both image and text domains with comprehensive ablations and comparisons, though the compute scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The progressive structure from Gaussian warm-up to general theory to experiments is clear and elegant; theorem statements are precise.
- Value: ⭐⭐⭐⭐ — Provides clear strategic guidance for practitioners, though the scope of direct generalization is constrained by the theoretical assumptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)

</div>

<!-- RELATED:END -->
