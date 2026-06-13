---
title: >-
  [Paper Note] Transformed Latent Variable Multi-Output Gaussian Processes
description: >-
  [ICML 2026][Computational Biology][Multi-output Gaussian Processes] This paper proposes T-LVMOGP: transforming the core modeling problem of Multi-Output GPs (MOGPs)—the construction of cross-output covariance $k_{p,p'}(x…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Multi-output Gaussian Processes"
  - "Deep Kernels"
  - "Lipschitz Regularization"
  - "SVGP"
  - "Spectral Normalization"
date: 2026-05-08
content_hash: 778b62926ef80632
---

# Transformed Latent Variable Multi-Output Gaussian Processes

**Conference**: ICML 2026  
**arXiv**: [2605.05133](https://arxiv.org/abs/2605.05133)  
**Code**: No explicit repository address provided in the text  
**Area**: Computational Biology  
**Keywords**: Multi-output Gaussian Processes, Deep Kernels, Lipschitz Regularization, SVGP, Spectral Normalization

## TL;DR
This paper proposes T-LVMOGP: transforming the core modeling problem of Multi-Output GPs (MOGPs)—the construction of cross-output covariance $k_{p,p'}(x, x')$—into "inner products using a single scalar base kernel in a Lipschitz-regularized RCNN embedding space." By fully integrating this into the SVGP framework, the method enables MOGPs to handle high-dimensional outputs ($P > 10000$, including spatial transcriptomics with ZINB likelihood) with high expressivity and scalability, outperforming baselines such as SV-LMC, OILMM, and GS-LVMOGP.

## Background & Motivation
**Background**: Multi-Output GPs (MOGPs) generalize single-output GPs to vector-valued observations and are widely used in clinical time series, climate modeling, spatial transcriptomics, and robot inverse dynamics. The classic LMC scheme represents each output $f_p$ as a linear combination of shared latent GPs $f_p = \sum_{q,r} \alpha_{p,r}^{(q)} g_r^{(q)}$, where the cross-output covariance is equivalent to a linear kernel on latent output embeddings, resulting in a low-rank structure. LV-MOGP further assigns a latent variable $h_p$ to each output and applies any valid kernel over $\{h_p\}$, extending to the sum-of-separable kernels in GS-LVMOGP.

**Limitations of Prior Work**: The complexity of standard MOGPs is $O(P^3)$ relative to the number of outputs $P$, which is prohibitive in high-dimensional scenarios like climate ($P \sim 10^4$) and spatial transcriptomics ($P \sim 5000$ genes). Existing scalable solutions either impose rigid structural assumptions (Kronecker, low-rank, or sum-of-separable) or utilize deep kernels with simple neural embeddings, which often suffer from feature collapse, loss of distance awareness, and overconfident predictions.

**Key Challenge**: It is difficult to simultaneously satisfy scalability, structural flexibility, and uncertainty reliability. LMC/OILMM sacrifice expressivity for scalability; naive deep kernel GPs sacrifice uncertainty for expressivity; and GS-LVMOGP is still limited by fixed kernel-like structures using sum-of-separable forms.

**Goal**: To construct an MOGP framework that is: (i) scalable with respect to $P$ (mini-batching over both inputs and outputs); (ii) free of structural assumptions for cross-output covariance; (iii) maintains distance awareness and credible uncertainty of GPs; and (iv) naturally compatible with non-Gaussian likelihoods and recent tighter variational bounds.

**Key Insight**: Separate the two tasks of MOGP—assigning embeddings to each output and calculating covariance over these embeddings. The former is handled by learnable latent variables $h_p$ and neural mappings, while the latter follows a standard single-output SVGP inference pipeline. As long as the embedding space satisfies Lipschitz continuity, the pathologies of deep kernels can be mitigated.

**Core Idea**: Concatenate $(x, h_p)$ and map them to an embedding space using a Lipschitz-RCNN $\Phi_\theta$. The cross-output covariance is defined as $\text{cov}[f_p(x), f_{p'}(x')] = k_{\text{base}}(\Phi_\theta(x, h_p), \Phi_\theta(x', h_{p'}))$. This reduces the MOGP to a scalar GP where inducing points reside in the embedding space, allowing for direct application of SVGP with mini-batch training.

## Method

### Overall Architecture
The framework consists of three layers: (1) Latent Variable Layer—each output $p$ is assigned a Gaussian prior $p(h_p) = \mathcal{N}(0, I)$, approximated by a variational distribution $q(h_p) = \mathcal{N}(m_p, \Sigma_p)$; (2) Embedding Layer—a Lipschitz-regularized residual network $\Phi_\theta : \mathbb{R}^{D_X} \times \mathbb{R}^{D_H} \to \mathbb{R}^{D_T}$ encodes $(x_n, h_p)$ into $\tilde{x}_{n,p}$; (3) GP Layer—$M$ inducing points $Z$ are placed in the embedding space, and $q(f_p(x_n)) = \int q(u) p(f_p(x_n) | u) du$ is calculated using standard SVGP. Reparameterization $h_p^{(j)} = m_p + \Sigma_p^{1/2} \epsilon^{(j)}$ makes the ELBO differentiable, and mini-batches are sampled over both inputs $\mathcal{B}_N$ and outputs $\mathcal{B}_P$ to estimate $\tilde{\mathcal{L}}_3$.

### Key Designs

1. **Multi-output deep kernels via learnable latent variables $h_p$ + neural embeddings**:
    - **Function**: Represents any MOGP cross-output covariance $k_{p,p'}(x, x')$ uniformly as $k_{\text{base}}(\Phi_\theta(x, h_p), \Phi_\theta(x', h_{p'}))$ without low-rank, Kronecker, or sum-of-separable assumptions.
    - **Mechanism**: $(x_n, h_p)$ are concatenated and passed through $\Phi_\theta$ to obtain $\tilde{x}_{n,p}$. Similarity in the embedding space is then calculated using any stationary base kernel (defaulting to ARD-RBF). Appendix D proves this form strictly contains the separable and sum-of-separable kernels of LV-MOGP as special cases.
    - **Design Motivation**: Treating "output ID + input" as points in an embedding space reduces the multi-output GP to a scalar GP in that space. This allows seamless integration with SVGP, fundamentally solving the $O(P^3)$ scaling issue, while Bayesian treatment of $h_p$ preserves uncertainty modeling of output relationships to avoid overfitting.

2. **Lipschitz-regularized RCNN: Residual connections + Spectral Normalization**:
    - **Function**: Uses a network with a controllable Lipschitz constant as $\Phi_\theta$ to avoid feature collapse, loss of distance awareness, and overconfidence on OOD inputs.
    - **Mechanism**: The architecture utilizes residual connections. Weight matrices of each layer are constrained via spectral normalization (computing the largest singular value through power iteration) to control the spectral norm upper bound (SN-UB). The Lipschitz constant of an $L$-layer network is bounded by $(1 + \text{SN-UB})^L$. Results from Bartlett et al. ensure this remains expressive enough to represent a large class of smooth Lipschitz mappings.
    - **Design Motivation**: To ensure the GP's distance awareness ("similar for close inputs, different for far ones") holds in the embedding space, $\Phi_\theta$ must not arbitrarily collapse distant points. Ablations show that removing spectral normalization causes the NLL on the EEG dataset to jump from 0.814 to 4.109, proving its necessity.

3. **SVGP + Double mini-batch + Plug-and-play tighter variational bounds**:
    - **Function**: Provides scalability for both input size $N$ and output size $P$, and remains compatible with non-Gaussian likelihoods like ZINB.
    - **Mechanism**: $M$ inducing points $Z$ are placed in the embedding space. The ELBO is defined as $\mathcal{L}_3 = \sum_n \sum_p \mathbb{E}_{q(h_p) q(f_p(x_n))}[\log p(y_{n,p}|f_p(x_n))] - \mathrm{KL}[q(u)\|p(u)] - \sum_p \mathrm{KL}[q(h_p)\|p(h_p)]$. Mini-batches $\mathcal{B}_N$ and $\mathcal{B}_P$ are sampled simultaneously to estimate $\tilde{\mathcal{L}}_3$. Gaussian likelihood expectations are analytical; non-Gaussian cases (e.g., ZINB) use Gauss-Hermite or MC. Tighter bounds (Titsias 2025, Bui 2025) are supported by adding $\Delta = \frac{1}{2} \sum_n [d_n / \sigma_y^2 - \log(1 + d_n/\sigma_y^2)]$.
    - **Design Motivation**: Treating the $P$-dimensional output space as a sampling dimension is crucial—prior MOGPs mostly mini-batched over $N$ only. Double mini-batching enables training with $P > 10^4$. The total complexity is reduced to $O(N_b P_b M^2 + M^3)$ plus spectral normalization $O(Tmn)$, where the latter is negligible given typical RCNN dimensions (width $\sim 10$, depth $\sim 5$).

### Loss & Training
The negative ELBO $-\mathcal{L}_3$ serves as the loss function. Gaussian likelihoods provide analytical expectations, while non-Gaussian likelihoods use Gauss-Hermite quadrature or MC with reparameterization. Input and output mini-batches are sampled concurrently. Primary hyperparameters include the number of inducing points $M$, SN-UB, latent variable dimension $D_H$, and embedding dimension $D_T$. On the EEG dataset, the optimal SN-UB value is $\approx 0.005$, while on SARCOS it is $\approx 1.0$, illustrating a trade-off curve (too strict limits expressivity, too loose leads to overfitting).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Sub-optimal Baseline | Remarks |
|---|---|---|---|---|
| EEG ($P=7$) | MSE / NLL | **0.115 / 0.814** | SV-LMC 0.282 / 0.857 | Electrode voltage prediction under visual stimuli |
| SARCOS ($P=7$, $N \approx 5 \times 10^4$) | MSE / NLL / Time | **0.022 / -0.485 / 5.26 s** | G-MOGP 0.023 / -0.483 / 5.89 s | Robot arm inverse dynamics |
| ERA5 ($P=3395$) | MSE / NLL | **0.002 / -1.564** | GS-LVMOGP 0.014 / -0.699 | UK 2m temperature for 30 months |
| Copernicus Marine ($P=21679$) | MSE / NLL / Time | **0.029 / -0.439 / 1.23 s** | GS-LVMOGP($Q=3$) 0.035 / 4.975 / 2.08 s | Sea surface temperature, output extrapolation |
| Spatial Trans. ($P=5000$, ZINB Likelihood) | MSE / NLL | **9.189 / 0.674** | GS-LVMOGP($Q=3$) 11.024 / 0.674 | $\approx 2.18 \times 10^7$ observations |

### Ablation Study

| Configuration | EEG NLL | SARCOS NLL | ERA5(random) NLL |
|---|---|---|---|
| Full T-LVMOGP | **0.814** | **-0.485** | **-1.564** |
| w/o Spectral Norm (w/o SN) | 4.109 | 0.112 | -1.401 |
| w/o Neural Network (Identity) | 1.153 | -0.336 | -1.554 |
| SN-UB set to 0.001 (EEG) / 0.1 (SARCOS) | 1.371 | -0.363 | — |
| Tighter variational bound | — | -0.502 | — |

### Key Findings
- Spectral normalization is an indispensable "safety harness" for deep kernel GPs: on EEG, it reduces NLL from 4.109 to 0.814, the most significant impact among all ablations. On larger datasets like ERA5, the impact is smaller but consistent, indicating that smaller data with higher overfitting risk benefit more from Lipschitz constraints.
- SN-UB exhibits a "sweet spot": values that are too strict (0.001) lack expressivity, while no SN reduces the model to a standard deep kernel. Tuning this per dataset is one of the few practical limitations.
- In output extrapolation tasks (Copernicus Marine), T-LVMOGP's NLL drops significantly to -0.439 compared to GS-LVMOGP's 4.975 (a gap of nearly 5.4 nats), demonstrating the advantage of deep kernel flexibility when generalizing to new outputs.
- The combination of a single-layer GP with complex embeddings is more cost-effective than multi-kernel GPs in large-scale problems; wall-clock time on SARCOS was 5.26 s/epoch vs G-MOGP’s 5.89 s.

## Highlights & Insights
- The abstraction of "using a single scalar GP in embedding space to represent any MOGP" is elegant. It liberates MOGPs from low-rank/Kronecker constraints and aligns them with methodologies like metric learning and CLIP.
- Applying Lipschitz constraints to deep kernels is an established technique (DUE/SNGP), but the author's application to MOGPs targets a specific pain point: MOGPs naturally require "output-to-output" distance preservation, which spectral normalization maintains effectively.
- Double mini-batching (simultaneous $N$ and $P$) is the key engineering factor pushing MOGPs to $P > 10^4$. Most prior SVGP-on-MOGP methods only mini-batched the input side.
- Broad compatibility with non-Gaussian likelihoods (like ZINB) allows spatial transcriptomics data (zero-inflated counts) to be processed within the same framework, extending MOGPs from Gaussian regression to biomedical applications.

## Limitations & Future Work
- The latent variable posterior uses a mean-field factorization $q(H) = \prod_p q(h_p)$, failing to capture posterior coupling between outputs. The authors acknowledge that structured variational or amortized inference could address this.
- SN-UB requires per-dataset tuning (EEG 0.005 vs SARCOS 1.0), lacking an automated selection strategy, which limits "out-of-the-box" usability.
- Theoretical guidance for choosing dimensions $D_T$ and $D_H$ is missing; only empirical values are provided.
- Lipschitz constraints guarantee distance awareness but not necessarily calibration, particularly for OOD inputs, which requires systematic evaluation.
- For outputs with highly non-smooth structures (e.g., jumps in time series), a single stationary base kernel might be insufficient, placing a heavy burden on the embedding layer $\Phi_\theta$ and potentially requiring higher capacity or looser SN-UB.

## Related Work & Insights
- **vs LMC / OILMM / SV-LMC**: These methods structure cross-output covariance as low-rank matrices via linear combinations. This paper uses deep kernels to completely bypass low-rank assumptions, achieving significantly better MSE on EEG/ERA5.
- **vs LV-MOGP / GS-LVMOGP (Dai 2017 / Jiang 2025)**: Direct predecessors that map latent variables to kernels. Appendix D proves the proposed kernel class strictly contains sum-of-separable kernels as special cases, and experiments confirm superiority over GS-LVMOGP across datasets.
- **vs G-MOGP (Dai 2024)**: G-MOGP uses attention-based graph models for expressive priors; T-LVMOGP achieves similar goals via deep kernel embeddings with faster training (SARCOS 5.26 vs 5.89 s/epoch).
- **vs DUE / SNGP (Van Amersfoort 2021 / Liu 2020)**: Borrowed the core idea of Lipschitz-regularized deep kernels, extending it from single-output GPs to the MOGP setting with full SVGP integration.
- **vs Tighter Variational Bounds (Titsias 2025 / Bui 2025)**: The framework demonstrates plug-and-play capability for these bounds, yielding small improvements in SARCOS NLL (from -0.485 to -0.502).

## Rating
- Novelty: ⭐⭐⭐⭐ The abstraction of MOGPs as scalar GPs in embedding space + Lipschitz deep kernels is a clean and original combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage from $P=7$ (EEG) to $P > 21000$ (Marine Temp) plus ZINB likelihoods; ablations cover SN, NN, SN-UB, and variational bounds.
- Writing Quality: ⭐⭐⭐⭐ Clear alignment between formulas and Figures 1/2; strong structural logic despite theorem proofs being relegated to the appendix.
- Value: ⭐⭐⭐⭐ Breaks the "mandatory low-rank/Kronecker" constraints for MOGPs, providing practical utility for large-scale modeling in climate, biology, and robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICML 2026\] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](routing_by_reaching_composition_of_pre-trained_gflownets_for_multi-objective_gen.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[ICML 2026\] iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiome Diagnosis](ilora_bayesian_low-rank_adaptation_with_latent_interaction_graphs_for_microbiome.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)

</div>

<!-- RELATED:END -->
