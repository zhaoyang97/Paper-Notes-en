---
title: >-
  [Paper Note] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization
description: >-
  [ICML 2026][Model Compression][Weight-only Quantization] OSAQ leverages the observation that the Hessian of each LLM layer maintains a consistent low-rank null space across different inputs. By linearly combining the nul…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Weight-only Quantization"
  - "Outlier Suppression"
  - "Hessian Null Space"
  - "Additive Transformation"
  - "Closed-form Solution"
date: 2026-05-08
content_hash: 6afc77922c054fa7
---

# OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

**Conference**: ICML 2026  
**arXiv**: [2605.04738](https://arxiv.org/abs/2605.04738)  
**Code**: None  
**Area**: Model Compression / LLM Weight-only Quantization  
**Keywords**: Weight-only Quantization, Outlier Suppression, Hessian Null Space, Additive Transformation, Closed-form Solution  

## TL;DR
OSAQ leverages the observation that the Hessian of each LLM layer maintains a consistent low-rank null space across different inputs. By linearly combining the null space vectors into an additive weight perturbation $\Delta W$, OSAQ "self-absorbs" outlier weights without altering the second-order task loss, reducing the perplexity of 2-bit weight-only quantization by over 40% compared to naive GPTQ.

## Background & Motivation

**Background**: The main bottleneck in LLM deployment lies in memory bandwidth during decoding (the "memory wall"), making weight-only quantization (W4/W3/W2A16) the mainstream compression approach. Representative methods include GPTQ, which uses approximate Hessians for error compensation; AWQ, which applies per-channel scaling based on activation distributions; and QuIP/QuaRot/SpinQuant, which use orthogonal rotations to "spread" outliers across other dimensions.

**Limitations of Prior Work**: All these methods fundamentally rely on **multiplicative equivalence transformations between adjacent layers** $(XW_1)W_2 = (XW_1T^{-1})(TW_2)$. In extremely low-bit scenarios such as 2-bit, multiplicative transformations alone cannot sufficiently suppress outlier spikes in weights to fit within the quantization grid, leading to severe perplexity degradation.

**Key Challenge**: The multiplicative paradigm inevitably "shifts" the transformation to adjacent layers, but due to network topology (e.g., residuals, LayerNorm skip paths) and numerical coupling, the degrees of freedom for suppressing outliers are limited. However, within a single layer, there remain many "loss-insensitive" directions that have not been exploited.

**Goal**: To find a **purely additive**, **layer-local**, **strictly loss-invariant (second-order)**, and **offline one-shot** method for outlier suppression.

**Key Insight**: Empirical results show that, although activation covariance structures vary greatly across inputs, **the Hessian of the task loss with respect to weights exhibits a consistent low-rank structure across samples**—the tail eigenvalues collectively account for only 0.01% of the energy, and these null space directions remain stable across samples. This suggests the existence of directions along which weight modifications have negligible impact on the loss.

**Core Idea**: Aggregate these Hessian null space vectors to construct an additive perturbation $\Delta W = \beta \mathcal{N}$, minimizing $\|W + \Delta W\|_\infty$ to suppress outliers, while ensuring $\Delta w^\top H^w \Delta w \approx 0$ so the loss remains nearly unchanged. The non-differentiable $\ell_\infty$ is approximated by a temperature-weighted $\ell_2$ using Softmax-$\infty$, yielding a closed-form solution for $\beta$ without any training or iteration.

## Method

### Overall Architecture
OSAQ is a pluggable PTQ pre-processing step: Given a pretrained LLM and a small calibration set (128 samples, sequence length 2048), each layer's linear weights $W \in \mathbb{R}^{M\times N}$ are processed independently in four steps—(1) estimate the approximate Hessian $H^w$ for the layer; (2) perform eigendecomposition on $H^w$ and extract the null space matrix $\mathcal{N} \in \mathbb{R}^{K\times N}$ based on a tail energy threshold $\gamma$; (3) solve for the closed-form coefficient vector $b_i$ for each output channel and stack them into $\beta \in \mathbb{R}^{M\times K}$; (4) update $W \leftarrow W + \beta\mathcal{N}$, absorbing the transformation offline into the weights, then hand off to existing quantizers such as GPTQ / AWQ / QuIP. This process incurs no inference overhead and does not modify adjacent layers.

### Key Designs

1. **Hessian Null Space Extraction (loss-invariant degrees of freedom)**:

    - **Function**: Identify a set of directions along which weight modifications yield zero second-order loss increment, serving as the basis for constructing additive perturbations.
    - **Mechanism**: Perform a second-order Taylor expansion of the task loss with respect to weights, retaining the Hessian term $\frac{1}{2}\Delta w^\top H^w \Delta w$. Eigendecompose $H^w = V\,\mathrm{diag}(\lambda_1,\dots,\lambda_N)V^\top$, and accumulate the smallest $|\lambda|$ values until the tail energy threshold $\gamma \in (0,1)$ is reached (formula $K = \min_k\{\sum_{i=1}^k|\lambda_i| \ge \gamma\sum_{i=1}^N|\lambda_i|\}$), taking the first $K$ eigenvectors as $\mathcal{N}$. This adaptive threshold avoids severe imbalance in null space dimensions across layers.
    - **Design Motivation**: Fixed numerical thresholds may result in empty null spaces for some layers and explosion for others; the tail energy strategy ensures each layer obtains a roughly equal amount of "usable degrees of freedom" while guaranteeing the perturbation directions are nearly zero-curvature, preventing loss explosion.

2. **Softmax-$\infty$ Objective Approximation (making $\ell_\infty$ differentiable)**:

    - **Function**: Convert the non-smooth, discrete objective of "minimizing the maximum absolute value of perturbed weights" into a quadratic objective with a closed-form solution.
    - **Mechanism**: The original objective is $\min_\beta \|W + \beta\mathcal{N}\|_\infty$, but $\ell_\infty$ is non-differentiable. Using the log-sum-exp / softmax trick from convex optimization (Boyd & Vandenberghe), the absolute value is temperature-normalized: $s_{ij} = \exp(|W_{ij}|/\tau) / \sum_t \exp(|W_{it}|/\tau)$; as $\tau \to 0^+$, $s_{ij}$ concentrates on the "maximum element", so $\sum_j s_{ij}(W_{ij}+\cdot)^2$ penalizes only the peaks, effectively turning $\ell_\infty$ into a peak-weighted $\ell_2$.
    - **Design Motivation**: Directly optimizing $\ell_\infty$ requires iterative methods (e.g., MagR) and cannot be solved in closed form; Softmax-$\infty$ preserves the "targeting outliers" semantics while enabling immediate solution via squared loss.

3. **Closed-form Normal Equation for β**:

    - **Function**: For each output channel, independently solve a $K\times K$ symmetric positive definite linear system to obtain the globally optimal coefficients.
    - **Mechanism**: For the $i$-th output channel, the objective is $\min_{b_i} \tfrac{1}{2}\sum_j s_{ij}(W_{ij}+b_i^\top n_j)^2 + \tfrac{\mu_1}{2}\|b_i\|_2^2 + \tfrac{\mu_2}{2}(b_i^\top v)^2$ (three terms: peak-weighted fit, $\ell_2$ regularization to prevent large coefficients, and anti-translation regularization to avoid uniform channel shifts). The first-order optimality condition yields $A_i b_i = -\rho_i$, where $A_i = \sum_j s_{ij}n_j n_j^\top + \mu_1 I_K + \mu_2 v v^\top$. $A_i$ is strictly positive definite ($A_i \succeq \mu_1 I_K \succ 0$), so $b_i^\ast = -A_i^{-1}\rho_i$ exists and is unique. Stacking the $M$ channel solutions yields $\beta^\ast$.
    - **Design Motivation**: The closed-form solution eliminates hyperparameter search and convergence issues, and does not require GPU training; the entire process involves only one eigendecomposition and one small linear inversion per channel, enabling all layers of a 70B model to be processed in minutes.

### Loss & Training
OSAQ involves no training loss. The entire process is PTQ calibration-style: 128 samples of sequence length 2048 are used to estimate $H^w$. Hyperparameters include tail energy threshold $\gamma$, temperature $\tau$, and regularization coefficients $\mu_1, \mu_2$; simple grid search demonstrates robustness to these choices. OSAQ is orthogonal to downstream quantizers (GPTQ/AWQ/QuIP) and can be used as a plug-and-play module. In 2-bit settings, coordinate descent iteration (denoted $\dagger$) can be stacked for further improvement.

## Key Experimental Results

### Main Results
Models include LLaMA2-{7B,13B,70B}, LLaMA3-{8B,70B}, Mistral-Large-123B-Instruct, and Llama-3.1-405B-Instruct. Evaluation covers language generation (WikiText2 / C4 perplexity), commonsense QA (PIQA / ARC / WinoGrande zero-shot accuracy), MMLU, and MT-Bench. Baselines include GPTQ, AWQ, QuIP, MagR, OmniQuant, etc.

| Model / Setting | Metric | FP16 | GPTQ | OSAQ+GPTQ | Gain |
|---|---|---|---|---|---|
| LLaMA2-7B W4A16 | WikiText2 PPL | 5.47 | 5.83 | **5.73** | Decrease 0.10 |
| LLaMA2-13B W4A16 | WikiText2 PPL | 4.88 | 5.13 | **5.04** | Decrease 0.09 |
| LLaMA3-70B W4A16 | WikiText2 PPL | 2.90 | 3.60 | **3.42** | Decrease 0.18 |
| LLaMA3-70B W4A16 | C4 PPL | 6.90 | 7.40 | **7.24** | Decrease 0.16 |
| LLaMA2-7B W4A16 | C4 PPL | 6.97 | 7.37 | **7.34** | -- |

Similar results are observed when combined with AWQ: On LLaMA3-8B, OSAQ+AWQ reduces WikiText2 PPL from 7.10 to 6.82 and C4 from 10.1 to 9.93. The "2-bit achieves 40% lower perplexity than naive GPTQ" highlighted in the abstract refers to OSAQ$^\dagger$+GPTQ under W2A16 in extremely low-bit scenarios.

### Ablation Study

| Configuration | LLaMA2-7B WikiText2 W4A16 PPL | Notes |
|---|---|---|
| Vanilla GPTQ | 5.83 | GPTQ only |
| OSAQ+GPTQ | 5.73 | With null space additive transformation |
| OSAQ+AWQ | 5.99 | Also effective with AWQ |
| OSAQ+GPTQ (varying $\gamma$) | Insensitive to $\gamma$ | Grid search (Fig.5) confirms hyperparameter robustness |
| Fixed threshold for null space | Severe imbalance in layer dimensions | Text explains necessity of tail energy strategy |

### Key Findings
- The low-rank structure of the Hessian is highly consistent across different inputs: projecting null spaces computed from different batches onto a 2D plane yields near overlap (Figure 1 right), while input null spaces diverge. This underpins OSAQ's experimental foundation.
- The larger the model and the lower the bit-width, the more significant the relative gain from OSAQ—consistent with the empirical observation that outliers worsen with scale.
- OSAQ is orthogonal to all multiplicative transformation methods (scaling/rotation); stacking with any of them yields stable improvements, indicating it indeed covers "intra-layer degrees of freedom" missed by the multiplicative paradigm.

## Highlights & Insights
- The perspective of "loss-invariant perturbation directions" reframes the outlier quantization problem as an **optimization within the null space of $H^w$**, representing a successful transfer of the approximate OBS (Optimal Brain Surgeon) tradition to the LLM era.
- The Softmax-$\infty$ approximation converts the non-differentiable $\ell_\infty$ into a temperature-weighted $\ell_2$, enabling a closed-form solution and avoiding the expensive iterations of MagR—this trick is broadly applicable to scenarios seeking both minimax and closed-form solutions.
- The orthogonality of "additive vs multiplicative" equivalence transformations points to a new axis: future PTQ designs can simultaneously consider the topological constraints of multiplicative transformations and the null space utilization of additive ones.

## Limitations & Future Work
- Calibration relies on approximate Hessians (often using Fisher/empirical second-order estimates), making it sensitive to calibration data distribution; the stability of the null space under distribution shift requires further validation.
- Currently, each layer is processed independently, without considering cumulative perturbations across layers; after stacking, the global second-order loss approximation may no longer hold.
- The code is not publicly available, raising the bar for reproducibility. While results for ultra-large models (405B) are provided, the engineering cost of Hessian estimation and eigendecomposition is not discussed in detail.

## Related Work & Insights
- **vs GPTQ**: GPTQ uses the Hessian for error compensation during quantization ("post-hoc correction"); OSAQ uses the Hessian null space to flatten weights before quantization ("pre-processing"), making them naturally complementary.
- **vs AWQ / QuIP / SpinQuant**: These methods rely on multiplicative transformations (scaling or rotation) between adjacent layers, limited by network topology; OSAQ is an intra-layer additive transformation, independent of adjacent layers, filling the blind spot of the multiplicative paradigm.
- **vs MagR**: MagR also minimizes $\ell_\infty$, but uses iterative subgradient methods; OSAQ leverages the Softmax-$\infty$ approximation for a closed-form solution, achieving an order-of-magnitude efficiency improvement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Additive transformation + Hessian null space" is a rare and novel perspective in LLM quantization
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers the full spectrum from 7B to 405B models and stacks with multiple baselines, but lacks detailed public ablation for W2A16
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, motivation is clearly narrated, and the null space consistency illustrations are convincing
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero overhead, compatible with all existing PTQ methods, and industry-friendly for deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](../../NeurIPS2025/model_compression/littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](../../NeurIPS2025/model_compression/learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](../../NeurIPS2025/model_compression/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)

</div>

<!-- RELATED:END -->
