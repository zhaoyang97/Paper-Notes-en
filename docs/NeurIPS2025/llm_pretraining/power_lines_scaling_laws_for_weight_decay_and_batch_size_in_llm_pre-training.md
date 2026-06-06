---
title: >-
  [Paper Note] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training
description: >-
  [NeurIPS 2025][LLM Pretraining][scaling laws] This paper proposes a set of power-law scaling relations for weight decay $\lambda$ and batch size $B$ in LLM pre-training. By introducing the concept of an AdamW timescale $…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "scaling laws"
  - "weight decay"
  - "batch size"
  - "LLM pre-training"
  - "AdamW"
date: 2026-05-08
content_hash: a93d3c46365352f2
---

# Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training

**Conference**: NeurIPS 2025
**arXiv**: [2505.13738](https://arxiv.org/abs/2505.13738)  
**Code**: To be confirmed  
**Area**: LLM Pre-training
**Keywords**: scaling laws, weight decay, batch size, LLM pre-training, AdamW

## TL;DR

This paper proposes a set of power-law scaling relations for weight decay $\lambda$ and batch size $B$ in LLM pre-training. By introducing the concept of an AdamW timescale $\tau$, it unifies hyperparameter scaling relationships, enabling accurate prediction of optimal hyperparameters prior to large-scale training.

## Background & Motivation

The pre-training performance of large language models (LLMs) is highly sensitive to hyperparameter tuning, including learning rate $\eta$, weight decay $\lambda$, and batch size $B$. Yet at frontier training scales, there is essentially no budget for hyperparameter search. Most existing work focuses on the scaling behavior of learning rate and batch size:

- **DeepSeek LLM** fits power laws for $B_{\text{opt}}$ and $\eta_{\text{opt}}$ as functions of total compute $C$
- **μP (Maximal Update Parameterization)** enables transfer of the optimal base learning rate across model widths
- **Wang et al.** propose the AdamW epoch timescale $\tau_{\text{epoch}}$, which remains stable on image tasks

Several critical gaps remain:

1. **The scaling of weight decay $\lambda$ is almost entirely unstudied** — in practice, $\lambda = 0.1$ is typically used without further justification
2. Relying on a unique optimal $B_{\text{opt}}$ lacks flexibility, failing to accommodate hardware constraints and time/compute trade-offs
3. It remains unclear whether $B_{\text{opt}}$ and $B_{\text{crit}}$ are best characterized in terms of $C$, $L$, $N$, or $D$

This paper aims to answer these questions through a unified power-law framework, covering both compute-optimal and overtrained regimes.

## Method

### Overall Architecture

The central idea is to unify the weight decay and learning rate of AdamW into a single **AdamW timescale $\tau$**, and then study how $\tau$ scales with model size and data volume.

**EMA perspective on AdamW**: The AdamW update can be written in the form of an exponential moving average:

$$\theta_t = (1 - \eta\lambda)\theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where the parameter $\theta_t$ can be viewed as an EMA of weight updates with smoothing coefficient $\alpha = \eta\lambda$, corresponding to an iterative timescale $\tau_{\text{iter}} = 1/(\eta\lambda)$.

**Normalized timescale $\tau$**: Since LLM pre-training makes a single pass over the data, the timescale is normalized as:

$$\tau = \frac{B}{\eta\lambda D}$$

where $S = D/B$ denotes the total number of optimization steps. $\tau$ reflects the fraction of past-iteration information incorporated into the final weights.

### Key Designs

**Finding 1: $\tau_{\text{opt}}$ follows a power law in tokens-per-parameter (TPP)**

$$\tau_{\text{opt}}(\text{TPP}) = c_\tau \cdot \text{TPP}^{m_\tau}, \quad \text{TPP} = D/N$$

The fit achieves $R^2 = 0.975$, with bootstrap 10th/90th percentiles for $m_\tau$ of $(-0.529, -0.507)$, indicating exceptional stability. This implies that $\tau_{\text{opt}}$ decreases from $\sim$1.0 to $\sim$0.01 as training ranges from compute-optimal ($\sim$20 TPP) to heavily overtrained ($1000+$ TPP).

**Derivation of $\lambda_{\text{opt}}$**:

$$\lambda_{\text{opt}} = \frac{B \cdot \text{TPP}^{-m_\tau}}{c_\tau \cdot \eta \cdot D}$$

Given any $N$, $D$, and $B$, the optimal weight decay can be accurately predicted in advance.

**Finding 2: $\lambda_{\text{opt}}$ scales linearly with $B$ (when $B \leq B_{\text{crit}}$)**

Validated on a 610M model at 20 TPP: for $B \in [63, 2016]$, $\tau_{\text{opt}}$ remains stable at $\sim$0.21, confirming that $\lambda_{\text{opt}}$ scales linearly with $B$. This relationship begins to drift once $B > B_{\text{crit}}$.

**Finding 3: Both $B_{\text{opt}}$ and $B_{\text{crit}}$ follow power laws in $D$**

$$B_{\text{crit}}(D_{\min}) = c_{B_{\text{crit}}} \cdot D_{\min}^{m_{B_{\text{crit}}}}$$

This contrasts with prior work that fits these quantities against $C$ or $L$ — the authors demonstrate that such relationships hold only at fixed TPP, and that $D$ is the fundamental variable.

### A New Method for Estimating $B_{\text{crit}}$

A key methodological contribution is a general procedure that is independent of any specific LR schedule or optimizer:

1. For each $B$, train across multiple values of $D$ and fit a batch-specific power law $L_B(D) = E_N + D_{\text{const}} \cdot D^{-\beta}$
2. Interpolate from the fitted curve to obtain $D_B$, the data volume required to reach a target loss $\hat{L}$
3. Fit the McCandlish formula to the $\langle D_B,\ S = D/B \rangle$ pairs to obtain $B_{\text{crit}} = D_{\min}/S_{\min}$

### Loss & Training

- GPT2-like architecture with ALiBi positional encoding and SwiGLU activations
- Training data: SlimPajama; validation set of 1.1B tokens
- Optimizer: AdamW with μP; linear LR schedule (10% warmup followed by linear decay to zero)
- $\lambda$ is swept at $2\times$ intervals for each $(N, D, B)$ configuration

## Key Experimental Results

### Main Results

| Experimental Setting | Key Finding |
|---|---|
| $\tau_{\text{opt}}$ vs. TPP power law | $R^2 = 0.975$; generalizes well across 3 orders of magnitude in compute |
| $\lambda$ vs. $B$ linear relationship (610M, 20 TPP) | $\tau_{\text{opt}} \approx 0.21$ for $B \in [63, 2016]$ |
| $B_{\text{opt}}$ vs. $D$ | Power-law fit substantially better than vs. $C$ (the latter exhibits spurious $N$ dependence) |
| $B_{\text{crit}}$ vs. $D$ | High-quality power-law fit; consistent with Zhang et al. on 302M models |

**Comparison of tuning $\lambda$ vs. tuning $\eta$ (610M, 20 TPP)**:

| $B$ | Fixed $\eta=0.016$, tune $\lambda$ | Fixed $\lambda=0.1$, tune $\eta$ |
|---|---|---|
| 63 | 2.583 | 2.579 |
| 126 | 2.570 | 2.565 |
| 252 | **2.563** | **2.563** |
| 504 | **2.571** | 2.570 |
| 2016 | **2.625** | 2.637 |
| 4032 | **2.754** | 2.733 |

Tuning $\lambda$ strictly outperforms tuning $\eta$ on 6 out of 8 batch size values. The key reason is that $\eta$ has an upper bound beyond which training becomes unstable, limiting its flexibility relative to $\lambda$.

### Ablation Study

- **111M model, 200 TPP**: default HP loss = 2.810; tuning $\eta$ = 2.808; tuning $\lambda$ = 2.805 — differences are small but tuning $\lambda$ consistently wins
- **Generalization check**: 4 "blind" evaluation points (not used in fitting, spanning a $1000\times$ scale difference) fall precisely on the power-law curve
- **$\tau$ is not constant in $D$**: This refutes the conclusion of Wang et al. that $\tau$ remains constant in multi-epoch training — in single-epoch LLM pre-training, $\tau_{\text{opt}}$ varies with TPP

### Key Findings

1. Weight decay is more suitable than learning rate as the hyperparameter to scale — the linear relationship is clean and does not induce training instability
2. The relationship between $B_{\text{opt}}$ / $B_{\text{crit}}$ and $D$ is fundamental; previously reported relationships with $C$ or $L$ are derivative
3. Pareto frontier analysis suggests that **small models with extensive overtraining** may simultaneously offer faster per-step throughput and greater parallelism

## Highlights & Insights

1. **Unified perspective**: The AdamW timescale $\tau$ unifies $\eta$, $\lambda$, $B$, and $D$ within a single framework, greatly simplifying hyperparameter selection
2. **High practical value**: The paper provides a directly usable formula for predicting $\lambda_{\text{opt}}$, eliminating the need for hyperparameter search at large scale
3. **Counterintuitive finding**: $\lambda$, rather than $\eta$, is the hyperparameter that should be scaled with $B$ and $D$
4. **New paradigm for estimating $B_{\text{crit}}$**: Schedule- and optimizer-agnostic, applicable to arbitrary optimizers and schedules
5. **All experiments conducted on Cerebras CS-3**: Large-scale validation across hundreds of models ensures the reliability of the conclusions

## Limitations & Future Work

1. Only a GPT2-like architecture and the SlimPajama dataset are used; applicability to other architectures (e.g., MoE) or datasets is not verified
2. Conclusions depend on μP parameterization — power-law coefficients may differ for training setups that do not use μP
3. Only linear LR schedules are studied; commonly used schedules such as cosine decay are not covered
4. The drift in $\lambda$ observed when $B > B_{\text{crit}}$ lacks in-depth analysis and modeling
5. The effect of late-stage training phases (e.g., cooldown) on $\tau$ is not addressed

## Related Work & Insights

- **Chinchilla (Hoffmann et al.)**: Establishes the compute-optimal $N$-$D$ relationship; this paper builds upon it to derive scaling laws for hyperparameters
- **DeepSeek Scaling**: Fits $B_{\text{opt}}$ and $\eta_{\text{opt}}$ as power laws in $C$; this paper argues that $D$ is the more fundamental variable
- **μP (Yang et al.)**: Provides theoretical grounding for LR transfer across model widths; this paper extends the principle to $\lambda$ transfer across $D$ and $B$
- **Wang et al. (EMA perspective)**: First proposes the EMA interpretation of AdamW and the $\tau_{\text{epoch}}$ concept; this paper finds that $\tau$ is not constant in LLM pre-training

**Implications for future work**: This power-law framework can be expected to extend to MoE models, multimodal models, and combinations with alternative schedules (WSD, trapezoidal).

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of weight decay scaling; the EMA-based unification via $\tau$ is both novel and practically useful
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Hundreds of models trained; validated across 3 orders of magnitude in FLOPs; rigorous bootstrap statistical analysis
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete derivations, and highly informative figures
- Value: ⭐⭐⭐⭐⭐ Provides a directly usable $\lambda_{\text{opt}}$ prediction formula with immediate practical value for industrial-scale training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [\[ICML 2026\] On the Expressive Power of Permutation-Equivariant Weight-Space Networks](../../ICML2026/llm_pretraining/on_the_expressive_power_of_permutation-equivariant_weight-space_networks.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](../../ICML2026/llm_pretraining/dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)

</div>

<!-- RELATED:END -->
