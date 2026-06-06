---
title: >-
  [Paper Note] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization
description: >-
  [ICML 2026][Model Compression][Weight-only Quantization] OSAQ leverages the consistent low-rank null space of the Hessian across different inputs for each LLM layer. It linearly combines null space vectors into an additi…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Weight-only Quantization"
  - "Outlier Suppression"
  - "Hessian Null Space"
  - "Additive Transformation"
  - "Closed-form Solution"
date: 2026-05-08
content_hash: 1d9170585275cb8c
---

# OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

**Conference**: ICML 2026  
**arXiv**: [2605.04738](https://arxiv.org/abs/2605.04738)  
**Code**: None  
**Area**: Model Compression / LLM Weight-only Quantization  
**Keywords**: Weight-only Quantization, Outlier Suppression, Hessian Null Space, Additive Transformation, Closed-form Solution  

## TL;DR
OSAQ leverages the consistent low-rank null space of the Hessian across different inputs for each LLM layer. It linearly combines null space vectors into an additive weight perturbation $\Delta W$, allowing outlier weights to be "self-absorbed" without changing the second-order task loss. This reduces the perplexity of 2-bit weight-only quantization by over 40% compared to vanilla GPTQ.

## Background & Motivation

**Background**: The bottleneck in LLM deployment is the memory bandwidth during the decoding stage (memory wall), making weight-only quantization (W4/W3/W2A16) a mainstream compression approach. Representative methods include GPTQ, which uses approximate Hessians for error compensation; AWQ, which utilizes activation distributions for per-channel scaling; and QuIP/QuaRot/SpinQuant, which use orthogonal rotations to "flatten" outliers across other dimensions.

**Limitations of Prior Work**: All these methods essentially rely on **multiplicative equivalent transformations between adjacent layers**: $(XW_1)W_2 = (XW_1T^{-1})(TW_2)$. In extreme low-bit scenarios like 2-bit, multiplicative transformations alone fail to suppress outlier peaks into the range covered by the quantization grid, resulting in severe perplexity degradation.

**Key Challenge**: The multiplicative paradigm must "transfer" the transformation to adjacent layers. This is limited by network topology (unabsorbable paths across residuals and LayerNorm) and numerical range coupling. Meanwhile, a large number of "loss-insensitive" directions within a single layer remain unutilized.

**Goal**: To find a **purely additive** outlier suppression method that **acts only on the current layer's weights**, **strictly does not affect second-order task loss**, and is **absorbed offline in a single pass**.

**Key Insight**: The authors' empirical measurements show that while activation covariance structures vary significantly across different inputs, the **Hessian of the task loss with respect to weights exhibits a consistent low-rank structure across samples**. A group of tail eigenvalues collectively accounts for only 0.01% of the energy, and these null space directions remain stable across different samples. This implies the existence of a set of directions along which modifying the weight barely affects the loss.

**Core Idea**: By weighted summing these Hessian null space vectors, an additive perturbation $\Delta W = \beta \mathcal{N}$ is constructed to minimize $\|W + \Delta W\|_\infty$ and flatten outliers, while ensuring $\Delta w^\top H^w \Delta w \approx 0$ so the loss remains nearly unchanged. Using a Softmax-$\infty$ approximation, the non-differentiable $\ell_\infty$ is converted into a weighted $\ell_2$ with a temperature coefficient, leading to a closed-form solution for $\beta$ without requiring training or iteration.

## Method

### Overall Architecture
OSAQ is a plug-and-play PTQ pre-processing step. Given a pre-trained LLM and a small amount of calibration data (128 sequences, length 2048), it processes each linear weight layer $W \in \mathbb{R}^{M\times N}$ independently in four steps: (1) Estimate the approximate Hessian $H^w$ for that layer; (2) Perform eigendecomposition on $H^w$ and extract the null space matrix $\mathcal{N} \in \mathbb{R}^{K\times N}$ based on a tail energy threshold $\gamma$; (3) Solve for the closed-form coefficient vector $b_i$ for each output channel to form $\beta \in \mathbb{R}^{M\times K}$; (4) Apply $W \leftarrow W + \beta\mathcal{N}$, absorbing the transformation into weights offline before finishing with existing quantizers like GPTQ, AWQ, or QuIP. The process adds zero inference overhead and does not modify adjacent layers.

### Key Designs

1.  **Hessian Null Space Extraction (loss-invariant degrees of freedom)**:
    - **Function**: Identifies directions where weights can be modified with zero second-order loss increment, serving as the basis for the additive perturbation.
    - **Mechanism**: Retains the Hessian term $\frac{1}{2}\Delta w^\top H^w \Delta w$ from the second-order Taylor expansion. Perform eigendecomposition $H^w = V\,\mathrm{diag}(\lambda_1,\dots,\lambda_N)V^\top$. Accumulate eigenvectors in ascending order of $|\lambda|$ until the tail energy reaches threshold $\gamma \in (0,1)$ ($K = \min_k\{\sum_{i=1}^k|\lambda_i| \ge \gamma\sum_{i=1}^N|\lambda_i|\}$), taking the first $K$ vectors as $\mathcal{N}$. This adaptive threshold avoids severe imbalance in null space dimensions across layers.
    - **Design Motivation**: A fixed numerical threshold would cause some layers to have empty null spaces and others to explode in dimensionality. The tail energy strategy ensures each layer receives roughly equivalent "available degrees of freedom" while ensuring the perturbation curvature is approximately zero and won't cause loss spikes.

2.  **Softmax-$\infty$ Objective Approximation (making $\ell_\infty$ differentiable)**:
    - **Function**: Converts the non-smooth discrete objective of "minimizing the maximum absolute value of perturbed weights" into a quadratic objective with a closed-form solution.
    - **Mechanism**: The original objective is $\min_\beta \|W + \beta\mathcal{N}\|_\infty$, but $\ell_\infty$ is non-differentiable. Borrowing the log-sum-exp/softmax trick from convex optimization (Boyd & Vandenberghe), absolute values are normalized via temperature: $s_{ij} = \exp(|W_{ij}|/\tau) / \sum_t \exp(|W_{it}|/\tau)$. As temperature $\tau \to 0^+$, $s_{ij}$ concentrates on the "maximum element," thus $\sum_j s_{ij}(W_{ij}+\cdot)^2$ approximately penalizes only peaks, transforming $\ell_\infty$ into $\ell_2$ weights focused on outliers.
    - **Design Motivation**: Directly optimizing $\ell_\infty$ requires iterative methods (e.g., MagR) and lacks a closed-form solution. Softmax-$\infty$ preserves the "outlier-targeting" semantics while allowing for a direct solve via normal equations due to the squared loss.

3.  **Closed-form Normal Equation for $\beta$**:
    - **Function**: Independently solves a $K\times K$ symmetric positive definite linear system for each output channel to obtain global optimal coefficients.
    - **Mechanism**: For the $i$-th output channel, the objective is $\min_{b_i} \tfrac{1}{2}\sum_j s_{ij}(W_{ij}+b_i^\top n_j)^2 + \tfrac{\mu_1}{2}\|b_i\|_2^2 + \tfrac{\mu_2}{2}(b_i^\top v)^2$ (representing peak-weighted fitting, $\ell_2$ regularization for coefficient scale, and anti-translation regularization to prevent uniform channel shifts). The first-order optimality condition yields $A_i b_i = -\rho_i$, where $A_i = \sum_j s_{ij}n_j n_j^\top + \mu_1 I_K + \mu_2 v v^\top$. Since $A_i$ is strictly positive definite ($A_i \succeq \mu_1 I_K \succ 0$), the solution $b_i^\ast = -A_i^{-1}\rho_i$ is unique. Stacking solutions for $M$ channels forms $\beta^\ast$.
    - **Design Motivation**: A closed-form solution implies zero hyperparameter search and no convergence issues without GPU training. The end-to-end process requires only one eigendecomposition and one small linear inversion per channel, processing all layers of a 70B model in minutes.

### Loss & Training
OSAQ involves no training loss. The flow is PTQ-style calibration: estimate $H^w$ using 128 sequences of length 2048. Hyperparameters include tail energy threshold $\gamma$, temperature $\tau$, and regularization coefficients $\mu_1, \mu_2$. The authors demonstrate robustness to these choices via grid search. OSAQ is orthogonal to and can be used with quantizers like GPTQ, AWQ, and QuIP. In 2-bit settings, coordinate descent iteration (denoted as $\dagger$) can be added for further gains.

## Key Experimental Results

### Main Results
Models include LLaMA2-{7B, 13B, 70B}, LLaMA3-{8B, 70B}, Mistral-Large-123B-Instruct, and Llama-3.1-405B-Instruct. Evaluation metrics cover language generation (WikiText2 / C4 PPL), common sense QA (PIQA / ARC / WinoGrande zero-shot accuracy), MMLU, and MT-Bench. Baselines include GPTQ, AWQ, QuIP, MagR, and OmniQuant.

| Model / Setting | Metric | FP16 | GPTQ | OSAQ+GPTQ | Gain |
|---|---|---|---|---|---|
| LLaMA2-7B W4A16 | WikiText2 PPL | 5.47 | 5.83 | **5.73** | -0.10 |
| LLaMA2-13B W4A16 | WikiText2 PPL | 4.88 | 5.13 | **5.04** | -0.09 |
| LLaMA3-70B W4A16 | WikiText2 PPL | 2.90 | 3.60 | **3.42** | -0.18 |
| LLaMA3-70B W4A16 | C4 PPL | 6.90 | 7.40 | **7.24** | -0.16 |
| LLaMA2-7B W4A16 | C4 PPL | 6.97 | 7.37 | **7.34** | -- |

Combinations with AWQ show similar trends: for LLaMA3-8B, OSAQ+AWQ reduces WikiText2 PPL from 7.10 to 6.82 and C4 PPL from 10.1 to 9.93. The "40% 2-bit perplexity reduction" mentioned in the abstract refers to OSAQ$^\dagger$+GPTQ versus vanilla GPTQ in extreme low-bit (W2A16) scenarios.

### Ablation Study

| Configuration | LLaMA2-7B WikiText2 W4A16 PPL | Explanation |
|---|---|---|
| Vanilla GPTQ | 5.83 | Only GPTQ |
| OSAQ+GPTQ | 5.73 | With null space additive transformation |
| OSAQ+AWQ | 5.99 | Equally effective when switched to AWQ |
| OSAQ+GPTQ (varying $\gamma$) | Insensitive to $\gamma$ | Grid search (Fig. 5) confirms robustness |
| Fixed null space threshold | Imbalanced layer dimensions | Energy strategy is necessary |

### Key Findings
- Hessian low-rank structure is highly consistent across inputs: null spaces calculated from different batches nearly overlap when projected onto a 2D plane (Figure 1 right), whereas input null spaces diverge. This is the experimental foundation of OSAQ.
- Larger models and lower bit-widths yield more significant relative gains with OSAQ—aligning with observations that outliers exacerbate as model scale increases.
- Orthogonal to all multiplicative methods (scaling/rotation); combining OSAQ with any of them provides stable improvements, showing it covers "intra-layer degrees of freedom" missed by the multiplicative paradigm.

## Highlights & Insights
- The perspective of "loss-invariant perturbation directions" transforms the quantization outlier problem into an **optimization within the $H^w$ null space**, a elegant transfer from the traditional OBS (Optimal Brain Surgeon) to the LLM era.
- The Softmax-$\infty$ approximation converts non-differentiable $\ell_\infty$ into weighted $\ell_2$, enabling closed-form solutions and avoiding the expensive iterations of MagR. This trick is applicable to many "minimax with closed-form" scenarios.
- The orthogonality between "additive vs. multiplicative" transformations identifies a new research axis: future PTQ designs can simultaneously consider multiplicative topological constraints and additive null space utilization.

## Limitations & Future Work
- Calibration depends on approximate Hessians (often using Fisher or empirical second-order estimates), making it sensitive to calibration data distribution. Null space stability under distribution shift needs further verification.
- Currently processes each layer independently without considering cumulative cross-layer perturbations; the second-order approximation of the total loss might become inaccurate after stacking many layers.
- Code is not public, posing a reproduction barrier. Engineering costs for Hessian estimation and decomposition on massive models (405B) are not discussed in detail despite the results being provided.

## Related Work & Insights
- **vs GPTQ**: GPTQ uses Hessians for error compensation during quantization ("post-quantization fix"); OSAQ uses the Hessian null space to flatten weights before quantization ("pre-processing"). They are naturally complementary.
- **vs AWQ / QuIP / SpinQuant**: These rely on multiplicative transformations between layers (scaling or rotation) and are limited by network topology. OSAQ uses intra-layer additive transformations, filling the blind spots of the multiplicative paradigm.
- **vs MagR**: MagR also minimizes $\ell_\infty$ but uses iterative subgradient methods. OSAQ uses Softmax-$\infty$ for a closed-form solution, being an order of magnitude more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Additive transformation + Hessian null space" is a rare and fresh perspective in LLM quantization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7B-405B spectrum and multiples baselines, though detailed W2A16 ablations are not fully public.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivation and smooth narrative; null space consistency visualizations are convincing.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero overhead, and compatible with all existing PTQ methods, making it industry-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs](lfq_logit-aware_final-block_quantization_for_boosting_the_generation_quality_of_.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[ICML 2026\] LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection](liftquant_continuous_bit-width_llm_via_dimensional_lifting_and_projection.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
