---
title: >-
  [Paper Note] Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations
description: >-
  [ICML2025][LLM Pretraining][Implicit Neural Representations] Starting from the Neural Tangent Kernel (NTK) linear dynamics model, this paper proposes the Inductive Gradient Adjustment (IGA) method. By inductively generalizing the eNTK gradient transformation matrix, it **purposefully** mitigates the spectral bias of MLPs, enabling INRs to efficiently learn high-frequency details even on million-scale data points.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Implicit Neural Representations"
  - "Spectral Bias"
  - "Neural Tangent Kernel"
  - "Gradient Adjustment"
  - "Training Dynamics"
date: 2026-05-08
content_hash: 5cbbe6c52416c2e1
---

# Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations

**Conference**: ICML2025  
**arXiv**: [2410.13271](https://arxiv.org/abs/2410.13271)  
**Code**: [LabShuHangGU/IGA-INR](https://github.com/LabShuHangGU/IGA-INR)  
**Area**: Implicit Neural Representations (INR) / Spectral Bias  
**Keywords**: Implicit Neural Representations, Spectral Bias, Neural Tangent Kernel, Gradient Adjustment, Training Dynamics

## TL;DR

Starting from the Neural Tangent Kernel (NTK) linear dynamics model, this paper proposes the Inductive Gradient Adjustment (IGA) method. By inductively generalizing the eNTK gradient transformation matrix, it **purposefully** mitigates the spectral bias of MLPs, enabling INRs to efficiently learn high-frequency details even on million-scale data points.

## Background & Motivation

- **Implicit Neural Representations (INR)** parameterize discrete signals as continuous functions using MLPs, which are widely applied in tasks such as image fitting, 3D reconstruction, and novel view synthesis.
- **Spectral Bias**: Vanilla ReLU-MLPs tend to learn low-frequency components first and converge extremely slowly on high-frequency details, leading to blurry textures and lost edges.
- Existing mitigation schemes fall into two categories:
    - **Structural architectural modifications**: Positional Encoding (PE), periodic activation (SIREN), Gabor wavelet activation, etc.—which require introducing complex inference structures.
    - **Training dynamics adjustment**: Fourier Reparameterized Training (FR), Batch Normalization (BN)—which do not alter the inference structure but lack theoretical guidance, leading to unstable performance.
- **Core Problem**: How to **purposefully** adjust training dynamics under theoretical guidance to overcome spectral bias? The NTK matrix is a key bridge connecting training dynamics and spectral bias, but it faces two major obstacles:
    1. The NTK matrix has **no analytical expression** in deep networks.
    2. The size of the NTK matrix grows **quadratically** with the volume of data $N$ (e.g., Kodak images require >8192 GiB of memory).

## Method

### 3.1 Connection between Spectral Bias and Training Dynamics

For a scalar signal $\bm{y} \in \mathbb{R}^N$, MLP parameters $\Theta$, and residual $\bm{r}_t = f(\bm{X};\Theta_t) - \bm{y}$, under the conditions of a wide network and a small learning rate, the training dynamics can be approximated as a linear model:

$$\bm{r}_t = (\bm{I} - \eta \bm{K}) \bm{r}_{t-1}$$

Performing eigenvalue decomposition on the NTK matrix $\bm{K} = \sum_i \lambda_i \bm{v}_i \bm{v}_i^\top$ yields:

$$\|\bm{r}_t\|_2 = \sqrt{\sum_{i=1}^{N} (1-\eta\lambda_i)^{2t} (\bm{v}_i^\top \bm{y})^2}$$

**Key Insight**: Larger eigenvalues $\lambda_i$ correspond to faster convergence in those directions. In vanilla MLPs, high-frequency directions correspond to small eigenvalues, leading to extremely slow convergence of high frequencies, which causes spectral bias. Making the spectrum of $\bm{K}$ more uniform can mitigate this bias.

### 3.2 Inductive Gradient Adjustment (IGA)

**Step 1 — NTK Gradient Adjustment Framework**: Introduce a transformation matrix $\bm{S}$ to adjust the gradient:

$$\Theta_{t+1} = \Theta_t - \eta \nabla_\Theta f(\bm{X};\Theta) \bm{S} \bm{r}_t$$

where $\bm{S} = \sum_i (g_i(\lambda_i)/\lambda_i) \bm{v}_i \bm{v}_i^\top$, and the modified convergence rates are $\{g_i(\lambda_i)\}$.

**Step 2 — Substituting NTK with eNTK** (Addressing the lack of analytical expression): Use the empirical NTK (eNTK) $\tilde{\bm{K}} = \nabla_{\Theta_t} f^\top \nabla_{\Theta_t} f$ to replace the theoretical NTK. Theorem 3.1 proves that as the network width $m$ increases, the eigenvalues/eigenvectors of the eNTK converge one-to-one to those of the NTK, making the eNTK-based adjustment asymptotically equivalent to the NTK-based adjustment.

**Step 3 — Inductive Generalization** (Addressing the curse of dimensionality): Divide $N$ data points into $n$ groups (each group has $p$ points, such that $N=np$). Sample 1 point from each group to form $\bm{X}_e$ ($|\bm{X}_e|=n \ll N$), compute a small-scale eNTK $\tilde{\bm{K}}_e \in \mathbb{R}^{n \times n}$ on $\bm{X}_e$, construct the transformation matrix $\tilde{\bm{S}}_e$, and then **inductively generalize** it to the gradients of the full dataset:

$$\Theta_{t+1} = \Theta_t - \eta \sum_{i=1}^{p} \nabla_{\Theta_t} f(\bm{X}_i, \Theta_t) \tilde{\bm{S}}_e \bm{r}_t^i$$

Theorem 3.2 guarantees that the generalization error $\epsilon_1 + \epsilon_2$ decreases as the width $m$ increases.

### Transformation Matrix Construction

After performing eigenvalue decomposition on $\tilde{\bm{K}}_e$, equalize the first $\text{end}$ eigenvalues to $\tilde{\tilde{\lambda}}_{\text{start}}$:

$$\tilde{\bm{S}_e} = \sum_{i=\text{start}}^{\text{end}} \frac{\tilde{\tilde{\lambda}}_{\text{start}}}{\tilde{\tilde{\lambda}}_i} \tilde{\tilde{\bm{v}}}_i \tilde{\tilde{\bm{v}}}_i^\top + \sum_{i \notin [\text{start},\text{end}]} \tilde{\tilde{\bm{v}}}_i \tilde{\tilde{\bm{v}}}_i^\top$$

- Larger $\text{end}$ $\rightarrow$ more uniform spectrum $\rightarrow$ stronger enhancement on high frequencies (controllable adjustment).
- Under the Adam optimizer, $\tilde{\tilde{\lambda}}_{\text{end}+1}$ is used to replace $\tilde{\tilde{\lambda}}_{\text{start}}$ to ensure convergence stability.

### Sampling Strategy

- 1D/2D signals: Group by adjacent coordinates (non-overlapping segments/patches).
- High-dimensional signals: Flatten and group along the first dimension.
- Select the point with the largest residual in each group to form $\bm{X}_e$.

## Key Experimental Results

### 2D Color Image Fitting (Kodak Dataset)

| Method | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS ↓|
|------|--------|--------|-----------|---------|
| ReLU (Vanilla) | 21.78 | 0.4833 | 0.6521 | 0.6302 |
| ReLU + FR | 22.14 | 0.4919 | 0.6800 | 0.6315 |
| ReLU + BN | 22.55 | 0.5004 | 0.7090 | 0.6182 |
| **ReLU + IGA** | **23.00** | **0.5126** | **0.7383** | **0.5549** |
| PE (Vanilla) | 28.64 | 0.7832 | 0.9466 | 0.2223 |
| PE + FR | 31.65 | 0.8167 | 0.9564 | 0.1869 |
| PE + BN | 28.78 | 0.8030 | 0.9554 | 0.2346 |
| **PE + IGA** | **32.46** | **0.8822** | **0.9752** | **0.0938** |
| SIREN (Vanilla) | 32.65 | 0.8975 | 0.9818 | 0.0807 |
| SIREN + FR | 32.61 | 0.8991 | 0.9820 | 0.0813 |
| **SIREN + IGA** | **33.48** | **0.9121** | **0.9847** | **0.0668** |

### 3D Shape Representation

| Method | IOU ↑ | Chamfer Distance ↓ |
|------|-------|---------------------|
| ReLU (Vanilla) | 9.647e-1 | 5.936e-6 |
| **ReLU + IGA** | **9.733e-1** | **5.487e-6** |
| PE (Vanilla) | 9.942e-1 | 5.123e-6 |
| **PE + IGA** | **9.970e-1** | **5.108e-6** |
| SIREN (Vanilla) | 9.889e-1 | 5.688e-6 |
| **SIREN + IGA** | **9.897e-1** | **5.157e-6** |

### Key Findings

- IGA consistently outperforms FR and BN across **all architectures** (ReLU / PE / SIREN) and **all metrics**.
- Increasing the number of equalized eigenvalues $\rightarrow$ monotonically enhances high-frequency detail learning (validating controllable adjustment).
- As the group size $p$ increases, IGA's performance decreases slightly but consistently outperforms the baseline, validating the robustness of inductive generalization.

## Highlights & Insights

1. **Theory-driven**: For the first time, starting from the NTK linear dynamics, a **quantitative strategy** is provided to regulate spectral bias, in contrast to empirical observations like FR/BN.
2. **Controllable adjustment**: The $\text{end}$ parameter precisely controls the strength of high-frequency enhancement, avoiding over-correction.
3. **Architecture-agnostic**: IGA operates as a training-time gradient transformation without altering the inference structure, allowing it to be freely combined with any INR architecture (ReLU / PE / SIREN).
4. **Efficient sampling**: Through inductive generalization, the computational cost of eNTK is reduced from $O(N^2)$ to $O(n^2)$ ($n \ll N$), making application to million-scale data points feasible.
5. **Two theorems** provide rigorous asymptotic guarantees: eNTK $\rightarrow$ NTK equivalence (Thm 3.1); Inductive generalization error bound (Thm 3.2).

## Limitations & Future Work

1. **Additional computational overhead**: Calculating the eNTK and its eigenvalue decomposition on a sampled subset at each step, although $n \ll N$, still introduces a non-zero overhead.
2. **Theoretical analysis limited to two-layer networks**: Theorems 3.1 and 3.2 are analyzed based on two-layer networks; rigorous guarantees for deep networks have not yet been provided.
3. **Simple grouping strategy**: Currently, grouping is based on adjacent coordinates; optimal grouping strategies for irregular sampling (e.g., NeRF rays) have not been explored.
4. **Hyperparameter selection**: The optimal values for $n$, $p$, and $\text{end}$ require task-specific tuning, and a fully automatic selection mechanism is lacking.
5. **Incompatibility between SIREN and BN** is pointed out but remains unresolved.

## Related Work & Insights

- **Fourier Reparameterized Training (FR)** [Shi et al., 2024a]: Learning parameters in the Fourier domain can improve INRs, but lacks theoretical guidance.
- **BN for INR** [Cai et al., 2024]: Classic BN can also improve INR performance, but is incompatible with SIREN.
- **NTK Spectrum Analysis** [Ronen et al., 2019; Tancik et al., 2020]: Explaining spectral bias using the NTK spectrum; this work goes further to convert it into an operational training strategy.
- **Geifman et al., 2023**: First attempt to modify the NTK spectrum to accelerate convergence, but limited to toy setups; this work extends it to practical scales through inductive generalization.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to convert eNTK spectral regulation into a practical training-time gradient transformation with a complete theoretical chain.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 2D images, 3D shapes, and synthetic data, with comprehensive comparisons against FR and BN.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical derivation is clear, and the transition from formulas to the algorithm is smooth.
- Value: ⭐⭐⭐⭐ — Has practical application value in the field of INR; controllable adjustment of spectral bias is a significant contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](../../NeurIPS2025/llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)

</div>

<!-- RELATED:END -->
