---
title: >-
  [Paper Note] BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization
description: >-
  [ICML 2025][Optimization][preference optimization] This paper introduces preference optimization into Neural Combinatorial Optimization (NCO) and proposes BOPO. Through (1) best-anchored preference pair construction (hybrid rollout + uniform filtering + best-anchored pairing) and (2) an objective-guided adaptively scaled loss function ($\beta = g(y_l)/g(y_w)$), BOPO comprehensively outperforms state-of-the-art (SOTA) approaches on three classical combinatorial optimization pr…
tags:
  - "ICML 2025"
  - "Optimization"
  - "preference optimization"
  - "neural combinatorial optimization"
  - "job-shop scheduling"
  - "TSP"
  - "DPO"
  - "SimPO"
  - "sample efficiency"
date: 2026-05-08
content_hash: 72f7eea5f7466c83
---

# BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization

**Conference**: ICML 2025  
**arXiv**: [2503.07580](https://arxiv.org/abs/2503.07580)  
**Code**: [https://github.com/L-Z-7/BOPO](https://github.com/L-Z-7/BOPO)  
**Area**: Alignment RLHF  
**Keywords**: preference optimization, neural combinatorial optimization, job-shop scheduling, TSP, DPO, SimPO, sample efficiency

## TL;DR

This paper introduces preference optimization into Neural Combinatorial Optimization (NCO) and proposes BOPO. Through (1) best-anchored preference pair construction (hybrid rollout + uniform filtering + best-anchored pairing) and (2) an objective-guided adaptively scaled loss function ($\beta = g(y_l)/g(y_w)$), BOPO comprehensively outperforms state-of-the-art (SOTA) approaches on three classical combinatorial optimization problems (JSP, TSP, and FJSP) without requiring a reward model or a reference policy.

## Background & Motivation

### Evolution of NCO Training Paradigms

**Supervised Learning (SL)**: Requires expensive (near-)optimal solutions as labels, which limits its practical utility.

**Reinforcement Learning (RL)**: The dominant paradigm (such as POMO using REINFORCE), but faces challenges of **sparse rewards** and **low sample efficiency**.

**Self-Labeled Learning (SLL)**: Such as SLIM, which uses the best self-sampled solution as a pseudo-label for cross-entropy training. While it improves upon RL, it discards all non-optimal sampled solutions, leaving sample efficiency relatively low.

### Core Case

NCO exhibits two unique characteristics that can be leveraged:
1. Generative models can produce **multiple distinct solutions** for the same instance.
2. Objective values in combinatorial optimization problems (e.g., makespan, tour length) can be **precisely computed at low cost**.

$\rightarrow$ A **natural preference relation** exists among different solutions, allowing the direct utilization of objective values to construct preference pairs for training without requiring human annotations or extra reward models.

### Comparison with LLM Preference Optimization

- **DPO**: Requires a reference model to prevent the policy from shifting too far $\rightarrow$ introduces extra computational overhead.
- **SimPO**: Eliminates the reference model but introduces additional hyperparameters ($\beta, \gamma$) that require tuning.
- **BOPO**: Uses the objective value ratio $g(y_l)/g(y_w)$ as an adaptive scaling factor, requiring no extra hyperparameters.

## Method

### Overall Architecture

The workflow of BOPO (as shown in Figure 1 of the paper):

1. **Hybrid Rollout**: Generates $B$ solutions for each instance ($B-1$ sampled solutions + 1 greedy solution).
2. **Uniform Filtering**: Uniformly selects $K$ representative solutions from the sorted list of $B$ solutions.
3. **Best-anchored Pairing**: Anchors the best solution and pairs it with the remaining $K-1$ solutions to construct $K-1$ preference pairs.
4. **Objective-guided Loss**: Trains the model using the adaptively scaled preference optimization loss.

### Key Designs

#### 1. Best-anchored Preference Pair Construction

**Hybrid Rollout**: Balances exploration ($B-1$ sampled solutions) and exploitation (1 greedy solution). The greedy solution guarantees that high-quality candidates are always present in the preference pairs. Experimental results show that hybrid rollout is roughly equivalent to doubling $B$.

**Uniform Filtering**: Uniformly samples $K$ solutions from the sorted solutions ($y_k = y'_{\lfloor B/K \rfloor \cdot (k-1) + 1}$), which avoids overfitting to clusters of similar solutions. In baseline comparisons, top-$K$ filtering performed the worst, because filtering out poor solutions reduced the preference discrepancy.

**Best-anchored Pairing**: Generates only $K-1$ preference pairs (the best vs. each sub-optimal solution) instead of $\binom{K}{2}$ fully permuted pairs. The rationale is that COPs focus strictly on finding the optimal solution, making the learning signals anchored to the best solution more focused.

#### 2. Objective-guided Preference Optimization Loss

**Explicit Preference**: $f^*(y, x) = -g(y)$ (negating the objective value to convert the minimization problem into a preference).

**Implicit Preference**: $f_\theta(y, x) = \frac{1}{|y|} \log \pi_\theta(y|x)$ (average log-likelihood).

The preference distribution is modeled using the Bradley-Terry model:

$$p_\theta(y_w \succ y_l | x) = \sigma\left(\beta(x, y_w, y_l) \cdot (f_\theta(y_w, x) - f_\theta(y_l, x))\right)$$

where $\beta(x, y_w, y_l) = \frac{f^*(y_l, x)}{f^*(y_w, x)} = \frac{g(y_l)}{g(y_w)}$ is the **adaptive scaling factor**.

### Loss & Training

The BOPO loss function is:

$$\mathcal{L}_{\text{BOPO}}(\pi_\theta, x, y_w, y_l) = -\log \sigma\left(\underbrace{\frac{g(y_l)}{g(y_w)}}_{\text{Adaptive Scaling}} \cdot \left(\underbrace{\frac{\log \pi_\theta(y_w|x)}{|y_w|} - \frac{\log \pi_\theta(y_l|x)}{|y_l|}}_{\text{Average Log-likelihood Difference}}\right)\right)$$

**Gradient Analysis**:

$$\nabla_\theta \mathcal{L}_{\text{BOPO}} = -\underbrace{\frac{g(y_l)}{g(y_w)}}_{\text{Adaptive Scaling}} \cdot \underbrace{(1 - \sigma(z))}_{\text{Confidence Weight}} \cdot \left(\frac{1}{|y_l|}\nabla_\theta \log \pi_\theta(y_l|x) - \frac{1}{|y_w|}\nabla_\theta \log \pi_\theta(y_w|x)\right)$$

Three key attributes:
- **Adaptive Scaling**: $g(y_l)/g(y_w) > 1$ (for minimization problems); the larger the gap between poor and good solutions, the larger the magnitude of the gradient.
- **Confidence Weight**: Pairs that the model has already mastered contribute small gradients, preventing overfitting.
- **Length Normalization**: $1/|y|$ ensures equal contribution from solution sequences of different lengths.

The training algorithm utilizes the Adam optimizer, with a straightforward workflow: sampling $\rightarrow$ filtering $\rightarrow$ pairing $\rightarrow$ computing BOPO loss $\rightarrow$ updating.

## Key Experimental Results

### Main Results

**JSP (Job-shop Scheduling Problem) — TA Benchmark:**

| Method | Type | Average Gap (B'=512) |
|------|------|----------|
| OR-Tools | Exact Solver | 2.0% |
| L2S 5k | RL Improvement | 7.4% |
| SLIM (GAT-MHA) | SLL-greedy | 13.4% |
| SLIM (B'=512) | SLL-sampling | 7.8% |
| **BOPO (greedy)** | **PO** | **12.9%** |
| **BOPO (B'=512)** | **PO** | **7.5%** |

BOPO achieves overall superiority among constructive methods. It is nearly on par with L2S 5k's 7.4% (at 7.5%), yet with an inference time of 4.8 minutes versus 4 hours.

**TSP — 1000 Random Instances:**

| Method | TSP20 Gap | TSP50 Gap | TSP100 Gap |
|------|-----------|-----------|------------|
| POMO | 0.04% | 0.21% | 0.46% |
| DABL (SL) | 0.01% | **0.04%** | 0.29% |
| SLIM | 0.22% | 1.55% | 3.22% |
| **BOPO** | **0.00%** | 0.07% | **0.12%** |

On TSP100, BOPO outperforms POMO by approximately 4 times (0.12% vs. 0.46%) with fewer training epochs (700 vs. 2000).

**FJSP — LA Benchmark (Sampling B'=100):**

| Dataset | HG | RS | **BOPO** |
|--------|-----|-----|------|
| LA(e-data) | 8.2% | 6.9% | **6.1%** |
| LA(r-data) | 5.8% | 4.7% | **4.0%** |
| LA(v-data) | 1.4% | 0.8% | **0.6%** |

### Ablation Study

**Ablation of Preference Pair Construction (DMU benchmark, Average Gap):**

| Ablation | Avg Gap |
|------|---------|
| w/o Hybrid Rollout (pure sampling) | 14.4% |
| w/o Uniform Filtering (random) | 14.0% |
| w/o Uniform Filtering (top-K) | 15.6% |
| w/o Best-anchored (full permutation) | 14.1% |
| **BOPO (Full)** | **13.3%** |

Top-$K$ filtering performs the worst (15.6%), indicating the necessity of retaining poor solutions to create sufficient preference discrepancy.

**Ablation of Loss Functions (TA benchmark, B'=512):**

| Loss | TA Gap |
|------|--------|
| DPO | 7.7% |
| SimPO | 7.8% |
| BOPO (w/o scaling) | 7.6% |
| **BOPO** | **7.5%** |

The performance gap is more significant on the DMU benchmark: DPO 14.1% $\rightarrow$ SimPO 13.6% $\rightarrow$ BOPO 12.9%. SimPO performs poorly in out-of-distribution (OOD) scenarios (DMU), suggesting that its target margin $\gamma$ causes overfitting.

### Key Findings

1. **Training Curve Comparison** (Figure 3): With the same number of training instances, BOPO's gap is significantly lower than that of RL (PPO/POMO) and SLL (SLIM), demonstrating a greater advantage particularly when data is limited.
2. **$B=256, K=16$ is the optimal hyperparameter combination**: An excessively large $B$ yields diminishing returns while increasing the memory footprint, whereas an excessively large $K$ leads to high similarity among selected solutions.
3. **Model-agnostic Nature of BOPO**: It has been successfully applied to two distinct architectures, POMO (Transformer) and MGL (GAT+LSTM), as well as INViT.
4. **MGL vs. GAT-MHA** (the model used in SLIM): MGL requires only 1/27 of the backpropagation memory of GAT-MHA, making training much more efficient.

## Highlights & Insights

1. **Clever Cross-Domain Transfer**: Novel adaptation of preference optimization from LLM alignment to combinatorial optimization.
2. **No Reward Model or Reference Policy Required**: Simpler than DPO and requires fewer hyperparameters than SimPO.
3. **Elegant Adaptive $\beta$ Design**: $g(y_l)/g(y_w)$ is instantly available, tuning-free, and holds clear mathematical meaning—the ratio of objective values serves as an excellent metric for preference strength.
4. **Simple and Effective Best-Anchored Pairing**: Utilizing only $K-1$ pairs (compared to $\binom{K}{2}$) ensures high computational efficiency and a focused learning signal on searching for the optimum.
5. **Architecture Agnostic**: Same training paradigm can be seamlessly integrated with different NCO models, demonstrating high generalizability.
6. **Implicit Curriculum of Hybrid Rollout**: In early training stages, greedy solutions dominate to provide strong gradients; in later stages, sampled solutions catch up to provide diversity.

## Limitations & Future Work

1. **Substantial Rollout Required**: The sampling cost of $B=256$ is comparable to that of SLIM, which is non-negligible.
2. **Validated Only on Constructive Methods**: The performance of BOPO-infused training has not been directly compared with improvement-based methods (e.g., L2S) using the same underlying models.
3. **Large-Scale Problems**: Has not been tested on TSP with $n > 100$ or JSP with $n > 50$; generalization capabilities require further validation.
4. **Lack of Theoretical Proof**: Short of theoretical analysis regarding the convergence of BOPO or sample complexity guarantees compared to RL.
5. **Domain Classification**: Although categorized under alignment/RLHF, it is fundamentally a combinatorial optimization method where preference optimization serves merely as a technical tool.

## Related Work & Insights

- **Evolutionary Trajectory of DPO $\rightarrow$ SimPO $\rightarrow$ BOPO**: DPO requires a reference model, SimPO eliminates it but introduces hyperparameters, while BOPO relies on objective-based adaptation, serving as a natural extension.
- **POMO** (Kwon et al. 2020): A multi-start RL baseline leveraging solution symmetries; the TSP experiments in BOPO are built on the POMO model.
- **SLIM** (Corsini et al. 2024): An SLL SOTA method that uses self-labeled optimal solutions for cross-entropy training. BOPO builds upon this by utilizing sub-optimal solution data.
- **Insight**: Preference optimization is not restricted to LLMs—any sequence generation problem with a well-defined objective function can potentially benefit from it. Future avenues could explore applications in molecular generation, protein design, and similar domains.

## Rating
- Novelty: ⭐⭐⭐⭐ (Adapting preference optimization to NCO is a novel combination, though the individual components are relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Evaluation on three problem classes + three benchmarks + extensive ablation studies + hyperparameter sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, thorough gradient analysis, though the paper is quite long)
- Value: ⭐⭐⭐⭐ (Establishes a preference optimization training paradigm for NCO with broad applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICLR 2026\] Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems](../../ICLR2026/optimization/neural_multi-objective_combinatorial_optimization_for_flexible_job_shop_scheduli.md)
- [\[ICLR 2026\] Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery](../../ICLR2026/optimization/unleashing_llms_in_bayesian_optimization_preference-guided_framework_for_scienti.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](../../NeurIPS2025/optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)

</div>

<!-- RELATED:END -->
