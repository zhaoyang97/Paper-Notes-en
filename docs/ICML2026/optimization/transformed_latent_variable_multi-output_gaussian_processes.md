---
title: >-
  [Paper Note] Transformed Latent Variable Multi-Output Gaussian Processes
description: >-
  [ICML 2026][Optimization][Multi-output Gaussian Process] This paper proposes T-LVMOGP: it transforms the core modeling challenge of multi-output GPs—constructing cross-output covariance $k_{p,p'}(x…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Multi-output Gaussian Process"
  - "Deep Kernel"
  - "Lipschitz Regularization"
  - "SVGP"
  - "Spectral Normalization"
date: 2026-05-08
content_hash: 69e82e8e1b7aa375
---

# Transformed Latent Variable Multi-Output Gaussian Processes

**Conference**: ICML 2026  
**arXiv**: [2605.05133](https://arxiv.org/abs/2605.05133)  
**Code**: No explicit repository address provided in the paper  
**Area**: Optimization / Gaussian Processes / Bayesian Nonparametrics / Probabilistic Machine Learning  
**Keywords**: Multi-output Gaussian Process, Deep Kernel, Lipschitz Regularization, SVGP, Spectral Normalization

## TL;DR
This paper proposes T-LVMOGP: it transforms the core modeling challenge of multi-output GPs—constructing cross-output covariance $k_{p,p'}(x, x')$—into "computing inner products with a single scalar base kernel in a Lipschitz-regularized RCNN embedding space," and fully embeds this into the SVGP framework. For the first time, MOGP can scalably and expressively handle $P > 10000$ outputs (including ZINB-likelihood spatial transcriptomics data), comprehensively outperforming baselines such as SV-LMC, OILMM, and GS-LVMOGP.

## Background & Motivation
**Background**: Multi-output GP (MOGP) generalizes single-output GP to vector-valued observations, with broad applications in medical time series, climate modeling, spatial transcriptomics, and robotic inverse dynamics. The classic LMC approach expresses each output $f_p$ as a linear combination of shared latent GPs $f_p = \sum_{q,r} \alpha_{p,r}^{(q)} g_r^{(q)}$, making cross-output covariance equivalent to a linear kernel over latent output embeddings, structurally low-rank. LV-MOGP further assigns each output a latent variable $h_p$ and applies any valid kernel over $\{h_p\}$, extending to GS-LVMOGP's sum-of-separable kernels.

**Limitations of Prior Work**: Standard MOGP complexity is $O(P^3)$ in the number of outputs $P$, which is prohibitive for high-dimensional output scenarios like climate ($P \sim 10^4$) and spatial transcriptomics ($P \sim 5000$ genes). Existing scalable approaches either enforce rigid Kronecker/low-rank/sum-of-separable structures, or use deep kernel with naive neural embeddings, which suffer from feature collapse, loss of distance awareness, and overconfident predictions.

**Key Challenge**: Scalability, structural flexibility, and reliable uncertainty are hard to achieve simultaneously—LMC/OILMM trade expressiveness for scalability, naive deep kernel GP sacrifices uncertainty for expressiveness, and GS-LVMOGP remains limited by fixed kernel structures.

**Goal**: Construct an MOGP framework that (i) is scalable in $P$ (mini-batch over both inputs & outputs); (ii) requires no structural assumptions for cross-output covariance; (iii) retains GP's distance awareness and credible uncertainty; (iv) naturally supports non-Gaussian likelihoods and recent tighter variational bounds.

**Key Insight**: Decouple the two aspects of MOGP—"assigning embeddings to each output" and "computing covariance over embeddings." The former is handled by learnable latent variables $h_p$ plus a neural mapping, the latter by standard single-output SVGP inference; as long as the embedding space is Lipschitz-continuous, deep kernel pathologies are suppressed.

**Core Idea**: Concatenate $(x, h_p)$ and map through a Lipschitz-RCNN $\Phi_\theta$ into the embedding space; cross-output covariance is $\text{cov}[f_p(x), f_{p'}(x')] = k_{\text{base}}(\Phi_\theta(x, h_p), \Phi_\theta(x', h_{p'}))$, reducing MOGP to a scalar GP over embedding space inducing points, directly compatible with SVGP mini-batch training.

## Method

### Overall Architecture
The framework consists of three layers: (1) Latent variable layer—each output $p$ is assigned a Gaussian prior $p(h_p) = \mathcal{N}(0, I)$, approximated by variational distribution $q(h_p) = \mathcal{N}(m_p, \Sigma_p)$; (2) Embedding layer—a Lipschitz-regularized residual network $\Phi_\theta : \mathbb{R}^{D_X} \times \mathbb{R}^{D_H} \to \mathbb{R}^{D_T}$ encodes $(x_n, h_p)$ as $\tilde{x}_{n,p}$; (3) GP layer—$M$ inducing points $Z$ are placed in the embedding space, and standard SVGP computes $q(f_p(x_n)) = \int q(u) p(f_p(x_n) | u) du$. Reparameterization $h_p^{(j)} = m_p + \Sigma_p^{1/2} \epsilon^{(j)}$ makes the ELBO differentiable; mini-batch samples both input $\mathcal{B}_N$ and output $\mathcal{B}_P$ to estimate $\tilde{\mathcal{L}}_3$.

### Key Designs

1. **Learnable latent variables $h_p$ + neural embedding constructed multi-output deep kernel**:

    - **Function**: Expresses any MOGP cross-output covariance $k_{p,p'}(x, x')$ as $k_{\text{base}}(\Phi_\theta(x, h_p), \Phi_\theta(x', h_{p'}))$, without requiring low-rank/Kronecker/sum-of-separable structure.
    - **Mechanism**: Concatenate $(x_n, h_p)$ and input to $\Phi_\theta$ to obtain $\tilde{x}_{n,p}$, then compute similarity in embedding space using any stationary base kernel (default ARD-RBF); Appendix D proves this strictly includes LV-MOGP's separable and sum-of-separable kernels as special cases.
    - **Design Motivation**: Unifies "output ID + input" as a point in embedding space, reducing multi-output GP to a scalar GP in embedding space—enabling seamless SVGP integration and fundamentally resolving the $O(P^3)$ issue; Bayesian treatment of $h_p$ preserves uncertainty modeling over output relationships, avoiding overfitting from point estimates.

2. **Lipschitz-regularized RCNN: residual connections + spectral normalization**:

    - **Function**: Uses a network with controllable Lipschitz constant as $\Phi_\theta$, preventing deep kernel pathologies such as feature collapse, loss of distance awareness, and overconfident OOD predictions.
    - **Mechanism**: Employs residual connections; each layer's weight matrix is spectrally normalized (power iteration for largest singular value) to control spectral norm upper bound SN-UB; $L$-layer network's Lipschitz constant is bounded by $(1 + \text{SN-UB})^L$. Bartlett et al. guarantee this parameterization still represents a broad class of smooth Lipschitz mappings.
    - **Design Motivation**: To ensure GP's "nearby inputs are similar, distant inputs are different" property holds in embedding space, $\Phi_\theta$ must not arbitrarily "flatten" distant points; ablation shows removing spectral normalization increases EEG NLL from 0.814 to 4.109, demonstrating its necessity.

3. **SVGP + dual mini-batch + plug-and-play tighter variational bound**:

    - **Function**: Enables inference scalable in both input size $N$ and output size $P$, and naturally supports non-Gaussian likelihoods such as ZINB.
    - **Mechanism**: Place $M$ inducing points $Z$ in embedding space; ELBO $\mathcal{L}_3 = \sum_n \sum_p \mathbb{E}_{q(h_p) q(f_p(x_n))}[\log p(y_{n,p}|f_p(x_n))] - \mathrm{KL}[q(u)\|p(u)] - \sum_p \mathrm{KL}[q(h_p)\|p(h_p)]$; mini-batch samples both $\mathcal{B}_N$ and $\mathcal{B}_P$ to estimate $\tilde{\mathcal{L}}_3$; analytic expectation for Gaussian likelihood, Gauss-Hermite or MC for non-Gaussian (e.g., ZINB); Titsias 2025 and Bui 2025 tighter bounds are incorporated by adding $\Delta = \frac{1}{2} \sum_n [d_n / \sigma_y^2 - \log(1 + d_n/\sigma_y^2)]$.
    - **Design Motivation**: Treating the $P$-dimensional output space as a "sampling dimension" is key—prior MOGPs mostly mini-batch only over $N$; dual mini-batch enables training with $P > 10^4$. Overall complexity reduces to $O(N_b P_b M^2 + M^3)$ plus spectral normalization $O(Tmn)$ (the latter negligible as RCNN width $\sim 10$, depth $\sim 5$).

### Loss & Training
Negative ELBO $-\mathcal{L}_3$ is used as the loss; analytic expectation for Gaussian likelihood, Gauss-Hermite quadrature or MC + reparameterization for non-Gaussian likelihoods; mini-batch over both inputs and outputs. Main hyperparameters are number of inducing points $M$, SN-UB, latent dimension $D_H$, and embedding dimension $D_T$; optimal SN-UB is $\approx 0.005$ for EEG, $\approx 1.0$ for SARCOS, showing a trade-off curve (too tight loses expressiveness, too loose overfits).

## Key Experimental Results

### Main Results

| Dataset | Metric | T-LVMOGP | Next Best Baseline | Note |
|---|---|---|---|---|
| EEG ($P=7$) | MSE / NLL | **0.115 / 0.814** | SV-LMC 0.282 / 0.857 | Electrode voltage prediction under visual stimulus |
| SARCOS ($P=7$, $N \approx 5 \times 10^4$) | MSE / NLL / Training Time | **0.022 / -0.485 / 5.26 s** | G-MOGP 0.023 / -0.483 / 5.89 s | Robotic arm inverse dynamics |
| ERA5 ($P=3395$) | MSE / NLL | **0.002 / -1.564** | GS-LVMOGP 0.014 / -0.699 | UK 2 m air temperature, 30 months |
| Copernicus Marine ($P=21679$) | MSE / NLL / Time | **0.029 / -0.439 / 1.23 s** | GS-LVMOGP($Q=3$) 0.035 / 4.975 / 2.08 s | Sea surface temperature, output extrapolation |
| Spatial transcriptomics ($P=5000$, ZINB likelihood) | MSE / NLL | **9.189 / 0.674** | GS-LVMOGP($Q=3$) 11.024 / 0.674 | $\approx 2.18 \times 10^7$ observations |

### Ablation Study

| Configuration | EEG NLL | SARCOS NLL | ERA5(random) NLL |
|---|---|---|---|
| Full T-LVMOGP | **0.814** | **-0.485** | **-1.564** |
| w/o Spectral Normalization (w/o SN) | 4.109 | 0.112 | -1.401 |
| w/o Neural Network (identity mapping) | 1.153 | -0.336 | -1.554 |
| SN-UB set to 0.001 (EEG) / 0.1 (SARCOS) | 1.371 | -0.363 | — |
| Tighter variational bound | — | -0.502 | — |

### Key Findings
- Spectral normalization is an indispensable "safety catch" for deep kernel GPs: on EEG, NLL drops from 4.109 to 0.814, the largest effect among all ablations; on larger datasets like ERA5, the effect is smaller but consistent, indicating that the smaller the data and the higher the overfitting risk, the more critical the Lipschitz constraint.
- SN-UB shows a "middle optimum" curve: too tight (0.001) leads to lack of expressiveness, too loose (no SN) degenerates to ordinary deep kernel; must be tuned per dataset, one of the few practical limitations.
- On Copernicus Marine output extrapolation, T-LVMOGP's NLL drops from GS-LVMOGP's 4.975 to -0.439 (a difference of nearly 5.4 nats), showing that deep kernel flexibility is most advantageous for generalizing to new outputs.
- The combination of single-layer GP + complex embedding outperforms multi-kernel GP in wall-clock time on large-scale problems (SARCOS 5.26 s/epoch vs G-MOGP 5.89 s), indicating that shifting complexity from kernel stacking to embedding networks is a cost-effective design.

## Highlights & Insights
- The abstraction of "expressing any MOGP as a scalar GP in embedding space" is elegant—it liberates MOGP from the constraints of Kronecker/low-rank to "geometric deep kernel GP + output embedding," sharing methodology with metric learning and CLIP.
- Imposing a Lipschitz constraint on deep kernels is a known trick (DUE/SNGP), but its application to MOGP is particularly apt: MOGP inherently requires "output-to-output" distance, and spectral normalization preserves this property, yielding greater benefit than in single-output GPs.
- Dual mini-batch (sampling both $N$ and $P$) is the key engineering point enabling MOGP to scale to $P > 10^4$; previous SVGP-on-MOGP approaches mostly mini-batched only over inputs.
- The framework directly supports non-Gaussian likelihoods such as ZINB, enabling unified modeling of zero-inflated count data in spatial transcriptomics—extending MOGP from "Gaussian regression only" to biomedical scenarios.

## Limitations & Future Work
- The latent variable posterior uses mean-field factorization $q(H) = \prod_p q(h_p)$, unable to capture posterior coupling between outputs; the authors acknowledge that structured variational or amortized inference could address this in future work.
- SN-UB requires dataset-specific tuning (EEG 0.005 vs SARCOS 1.0), lacking an automatic selection strategy; this weakens "out-of-the-box" usability.
- There is no theoretical guidance for choosing embedding dimension $D_T$ and latent dimension $D_H$; only empirical values are provided.
- Lipschitz constraint ensures distance awareness but does not directly guarantee calibration, especially for uncertainty reliability under severe OOD inputs, which is not systematically evaluated.
- When outputs have strong non-smooth structure (e.g., abrupt changes in time series), a single stationary base kernel may be insufficient, requiring the embedding layer to absorb all non-stationarity, possibly demanding higher $\Phi_\theta$ capacity and looser SN-UB.

## Related Work & Insights
- **vs LMC / OILMM / SV-LMC**: All structure cross-output covariance as linear combinations of low-rank matrices; this work uses deep kernel to completely remove the low-rank assumption; MSE on EEG/ERA5 is significantly better than these linear methods.
- **vs LV-MOGP / GS-LVMOGP (Dai 2017 / Jiang 2025)**: Direct predecessors, mapping latent variables to deep kernel embeddings; Appendix D proves this kernel class strictly includes sum-of-separable as a special case, and GS-LVMOGP is outperformed on multiple datasets.
- **vs G-MOGP (Dai 2024)**: G-MOGP uses attention-based graph models for expressive priors; T-LVMOGP achieves similar goals with deep kernel embedding but shorter training time (SARCOS 5.26 vs 5.89 s/epoch).
- **vs DUE / SNGP (Van Amersfoort 2021 / Liu 2020)**: Directly borrows the core idea of Lipschitz-regularized deep kernel, extending it from single-output GP to MOGP and fully integrating with SVGP.
- **vs Tighter Variational Bounds (Titsias 2025 / Bui 2025)**: The authors show these bounds can be plug-and-play incorporated, yielding a small improvement in SARCOS NLL from -0.485 to -0.502, demonstrating framework extensibility.

## Rating
- Novelty: ⭐⭐⭐⭐ The abstraction of "expressing any MOGP as a scalar GP in embedding space" plus the introduction of Lipschitz deep kernel is a clean and original combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Wide range from $P=7$ EEG to $P > 21000$ marine temperature and ZINB-likelihood spatial transcriptomics; ablations cover SN, NN, SN-UB, bound, etc.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and Figures 1/2, strong structural clarity in the main text despite theorems/proofs being in the appendix.
- Value: ⭐⭐⭐⭐ Frees MOGP from the "must be low-rank/Kronecker" shackle, practical for large-scale multi-output modeling in climate, biology, robotics, etc.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICLR 2026\] Learning to Solve Orienteering Problem with Time Windows and Variable Profits](../../ICLR2026/optimization/learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](../../ICLR2026/optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[AAAI 2026\] Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering](../../AAAI2026/optimization/instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](../../AAAI2026/optimization/motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)

</div>

<!-- RELATED:END -->
