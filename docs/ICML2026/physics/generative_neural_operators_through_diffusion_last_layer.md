---
title: >-
  [Paper Note] Generative Neural Operators Through Diffusion Last Layer
description: >-
  [ICML 2026][Physics & Scientific Computing][Neural Operator] Append a "Diffusion Last Layer" (DLL) after any neural operator backbone (FNO/DeepONet): compress the target field into an $r$-dimensional coefficient vector u…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Neural Operator"
  - "Diffusion Model"
  - "Karhunen–Loève"
  - "Flow Matching"
  - "Probabilistic Surrogate"
date: 2026-05-08
content_hash: 2d50f1f7efed878d
---

# Generative Neural Operators Through Diffusion Last Layer

**Conference**: ICML 2026  
**arXiv**: [2602.04139](https://arxiv.org/abs/2602.04139)  
**Code**: https://github.com/sungwpark/dll-no  
**Area**: Scientific Computing / Neural Operator / Diffusion Model / Uncertainty Quantification  
**Keywords**: Neural Operator, Diffusion Model, Karhunen–Loève, Flow Matching, Probabilistic Surrogate  

## TL;DR
Append a "Diffusion Last Layer" (DLL) after any neural operator backbone (FNO/DeepONet): compress the target field into an $r$-dimensional coefficient vector using input-dependent bases $\Phi_a$, then perform conditional flow matching with a small MLP velocity field in the coefficient space. This upgrades deterministic operators to generative ones capable of sampling stochastic solutions and providing rollout uncertainty.

## Background & Motivation
**Background**: Neural operators (FNO, DeepONet, etc.) have become the mainstream framework for solving parametric PDEs, learning function-to-function mappings with discretization invariance. However, most operators are deterministic—outputting a single solution $u$ for a given input $a$.

**Limitations of Prior Work**: Real-world scientific problems are often stochastic: random forcing, unresolved subgrid physics, and the sensitivity of chaotic systems to initial conditions cause "the same input" to correspond to a distribution rather than a point. Deterministic operators cannot represent this aleatoric uncertainty and fail to provide uncertainty estimates for error accumulation during long-term autoregressive rollouts. Existing probabilistic schemes have flaws: Bayesian neural operators only characterize epistemic uncertainty and are constrained by priors/likelihoods; pixel-space diffusion models (DM) are computationally expensive; Latent Diffusion Models (LDM) lose the geometric structural advantages of operators by using generic autoencoders for compression.

**Key Challenge**: To achieve both "distribution modeling capability" and the "discretization invariance and geometric awareness of operator learning." Direct diffusion in function or pixel space is too heavy, while generic LDM discards the structural inductive bias of operator backbones.

**Goal**: Design a lightweight, modular probabilistic output head applicable to any operator backbone that (i) preserves discretization invariance; (ii) models the conditional distribution $p(u\mid a)$ at a controllable cost; and (iii) provides rollout uncertainty even for deterministic problems.

**Key Insight**: The authors observe that the target solution field $u$ given $a$ usually possesses a low-dimensional structure—an intuition from the classic Karhunen–Loève expansion. By projecting $u$ into low-dimensional coefficients $\xi\in\mathbb{R}^r$ using **input-dependent** bases $\Phi_a$, diffusion modeling can be conducted entirely in $r$-dimensional Euclidean space, with the backbone's geometric/spectral structure carried by the basis functions.

**Core Idea**: Use a neural operator to learn a set of low-rank basis functions adapted to input $a$, reducing the distribution modeling problem to the coefficient space, and then learn this distribution using a conditional flow matching diffusion model—essentially treating diffusion as the "last layer" of the neural operator.

## Method
DLL training consists of two stages: first, train the operator encoder to learn input-dependent low-rank bases and coefficients; once frozen, train the conditional diffusion model in the coefficient space. During inference, sample coefficients $\theta\sim p(\theta\mid \mathcal{D},a)$ for each input $a$, then reconstruct the function $\hat u=\theta^\top \Phi_a$.

### Overall Architecture
- **Input**: Parameter function $a\in\mathcal{A}$ (e.g., PDE coefficients, initial conditions, geometry).
- **Operator Encoder** $\mathtt{NO}_\psi+\mathtt{NF}_\varphi$: The backbone $\mathtt{NO}_\psi$ maps $a$ to $r$ basis functions $\Phi_a=(\phi_1(a)$, $\dots$, $\phi_r(a))$; the neural functional encoder $\mathtt{NF}_\varphi$ (FNO + global average pooling) maps target $u$ to coefficients $\xi=\mathtt{NF}_\varphi(u)\in\mathbb{R}^r$. Reconstruction is $\hat u=\xi^\top\Phi_a$.
- **Diffusion Last Layer**: Learns the conditional flow $p(\xi\mid a)$ on $\mathbb{R}^r$ using an MLP velocity field $v_\phi$ on top of the frozen encoder.
- **Inference**: For a new input $a$, the backbone first provides $\Phi_a$; $\theta$ is recovered from noise by solving the probability flow ODE in the coefficient space; finally, decode $\hat u=\theta^\top\Phi_a$.
- **Output**: The entire predictive distribution over $\hat u$; repeatable sampling yields mean, variance, or rollout ensembles.

### Key Designs

1.  **Operator Encoder with Input-Dependent Bases**:

    - **Function**: Compresses the high-dimensional solution field $u$ into an $r$-dimensional coefficient vector $\xi$ while preserving geometric/spectral info of $a$, providing an "information-dense, low-dimensional" latent space for diffusion.
    - **Mechanism**: Use an FNO as the basis generator so that bases $\Phi_a=\mathtt{NO}_\psi(a)$ vary with input; use another FNO+GAP as the coefficient encoder $\mathtt{NF}_\varphi(u)$. Reconstruction uses rank-$r$ expansion $\hat u=\sum_k\xi_k\phi_k(a)=\xi^\top\Phi_a$. The training objective is simple reconstruction mean squared error $\mathcal{L}_{\mathrm{OE}}=\mathbb{E}\|u-\mathtt{NF}_\varphi(u)^\top\mathtt{NO}_\psi(a)\|_2^2$. Proposition 4.1 proves that under ideal conditions, minimizing this objective is equivalent to finding the leading $r$-dimensional subspace of the conditional second-moment operator, i.e., learning the input-dependent (uncentered) KL expansion.
    - **Design Motivation**: Unlike generic autoencoders that cram all structure into the latent, input-dependent bases allow the "reconstruction subspace to vary with the condition." This allows the latent space to carry more output structure for the same dimension, letting $\xi$ only handle "instance-level deviations." This is the fundamental reason why reducing diffusion to $r$ dimensions remains effective.

2.  **Diffusion Last Layer in Coefficient Space**:

    - **Function**: Learns a conditional generative model $p(\xi\mid a)$ in the $r$-dimensional coefficient space, replacing direct diffusion in pixel or function space.
    - **Mechanism**: Construct a forward noise process $x_t=(1-t)x+t\epsilon$ (linear schedule) and use an MLP velocity field $v_\phi(x_t,t,c)$ to minimize the conditional velocity matching loss $\mathcal{L}_V(c)=\mathbb{E}\big[\|v_\phi(x_t,t,c)-(\dot a_t x+\dot b_t\epsilon)\|_2^2\big]$, where condition $c$ comes from features of $a$. Inference involves backward integration of the probability flow ODE. NFE can be as low as 10. Proposition 2.3 provides an end-to-end Wasserstein bound $\mathcal W_2(p,\rho_0)\le C\sqrt{\mathcal L_V(c)}$, so reducing velocity matching loss simultaneously controls the output distribution distance.
    - **Design Motivation**: Direct diffusion on $128\times128$ fields requires heavy U-Nets and many NFEs; if the latent space is an $r=64$ dimensional vector, an MLP is sufficient, enabling millisecond sampling. Furthermore, since basis functions are generated by neural operators, decoding $\hat u=\theta^\top\Phi_a$ naturally inherits discretization invariance, allowing inference at new resolutions without retraining.

3.  **Unified Probabilistic View for Deterministic/Stochastic Problems**:

    - **Function**: Allows the same framework to handle both "inherently stochastic" problems (SPDEs) and "deterministic but uncertainty-required" problems (chaotic systems).
    - **Mechanism**: The target operator is unified as $\mathcal{G}^\ddagger:\mathcal A\to\mathcal P(\mathcal U)$ (deterministic operators are a degenerate case of Dirac measures). It is noted that $\mathcal{W}_2$ is a reasonable metric in both cases, so the same $\mathcal L_V$ loss applies to both. In deterministic cases, residual $\mathcal L_V(c)$ is interpreted as a "qualitative indicator of epistemic uncertainty arising from finite data/model error"—the diffusion underfit can be read as rollout uncertainty.
    - **Design Motivation**: Prior works either designed specialized probabilistic operators for SPDEs or added ensembles for rollout stability. DLL unifies both using one loss, linking the "training objective" with the "uncertainty source," making it easy to deploy on any existing FNO/DeepONet backbone.

### Loss & Training
Two-stage training: Stage one minimizes $\mathcal L_{\mathrm{OE}}$ to learn $(\mathtt{NO}_\psi,\mathtt{NF}_\varphi)$; stage two freezes the encoder and minimizes conditional velocity matching $\mathcal L_V$ on $(a,\xi)$ pairs. Default configuration: FNO backbone, $r=64$, NFE=10, linear noise schedule.

## Key Experimental Results

### Main Results (Stochastic PDE Distribution Fitting)
Evaluation metrics include Energy Distance (ED), Sliced Wasserstein Distance (SWD), and mean/std NRMSE. 64 ground-truth realizations are sampled for each input to compare the learned distribution with the truth.

| Dataset | Metric | FNO | PNO | DM | LDM | **DLL** |
|--------|------|-----|-----|----|-----|---------|
| Stochastic Burgers (1D, $u\in\mathbb R^{256}$) | ED ↓ | 6.491 | 1.766 | 1.355 | 1.373 | **1.285** |
| Stochastic Burgers | SWD ↓ | 0.426 | 0.253 | 0.239 | 0.249 | **0.213** |
| Stochastic Burgers | $\mathrm{NRMSE}_s$ ↓ | 1.000 | 0.457 | 0.323 | 0.297 | **0.289** |
| Stochastic Darcy (2D, $128^2$) | ED ↓ | 1.463 | 0.305 | 0.269 | 0.368 | **0.227** |
| Stochastic Darcy | $\mathrm{NRMSE}_s$ ↓ | 1.000 | 0.285 | 0.360 | 0.268 | 0.357 |

DLL achieves the lowest ED on both SPDEs and the lowest SWD on Burgers, overall stronger than pixel/generic latent diffusion. For the deterministic baseline FNO, $\mathrm{NRMSE}_s$ is 1.0, indicating it fails to model variance.

### Long-term Autoregressive Rollout Stability (Deterministic PDE)
Trained on step length 50, evaluated on rollout 100. Lower CRPS is better; SSR (spread-skill ratio) closer to 1 is better.

| Dataset | Metric | FNO | FNO-d | PNO | DM | LDM | **DLL** |
|--------|------|-----|-------|-----|----|-----|---------|
| Kuramoto–Sivashinsky (1D) | NRMSE ↓ | 0.404 | 0.384 | 0.354 | 0.395 | 0.576 | **0.343** |
| KS | CRPS ↓ | – | 0.523 | 0.514 | 0.545 | 0.878 | **0.470** |
| KS | SSR | – | 0.975 | 0.550 | 0.961 | 0.802 | 0.949 |
| Kolmogorov Flow (2D, $128^2$) | NRMSE ↓ | 0.528 | 0.463 | 0.492 | **0.369** | 0.615 | 0.426 |
| Kolmogorov | SSR | – | 0.546 | 0.167 | 0.601 | 0.548 | **0.620** |

DLL achieves optimal NRMSE and CRPS on KS with SSR near 1, indicating improved accuracy and well-calibrated uncertainty. On Kolmogorov, pixel diffusion achieves the best NRMSE due to stronger U-Net spatial priors, but DLL still outperforms the FNO backbone and maintains the best SSR.

### Key Findings
- The Operator Encoder (OE) outperforms autoencoders (AE) on deterministic PDEs (KS, Kolmogorov) even at high compression ratios of 4× / 256× (e.g., KS NRMSE $2.45\times10^{-4}$ vs $7.75\times10^{-4}$), supporting that "input-dependent bases" capture the low-dimensional structure of the solution manifold.
- On stochastic PDEs (Burgers, Darcy), OE reconstruction error is slightly behind AE, but DLL distribution metrics still lead, indicating that "structured latent info" is more important than "sample-wise reconstruction accuracy" for distribution modeling tasks.
- The performance ceiling of DLL is clearly limited by backbone quality: on Kolmogorov, pixel DM is stronger due to U-Net, suggesting further gains by using stronger FNO/Transformer backbones.

## Highlights & Insights
- **Reducing diffusion to the "coefficient space" is a clever decomposition**: Let the backbone handle geometric/spectral inductive bias and let the small MLP handle distribution modeling. Responsibilities are clear, training and inference are cheap, and integration into existing FNO/DeepONet models is non-intrusive.
- **Using "diffusion underfit" as epistemic uncertainty**: In deterministic problems where aleatoric distributions cannot be directly defined, the authors interpret residual $\mathcal L_V(c)$ as "implicit ensemble width under finite data," providing a free source of rollout uncertainty that could migrate to any conditional diffusion surrogate.
- **Proposition 4.1 links "input-dependent bases" to KL expansion**: This gives a clear theoretical explanation for an seemingly ad-hoc design—it is essentially learning a condition-dependent KL subspace, which explains why OE performs better than generic AE on deterministic fields.

## Limitations & Future Work
- For 2D chaotic systems with strong spatial structure (Kolmogorov flow), NRMSE lags behind pixel DM, indicating that DLL's ceiling is constrained when the backbone's expressiveness is insufficient; stronger backbones or higher $r$ are needed.
- $r$, NFE, and backbone architecture are critical hyperparameters; verification was primarily on FNO/DeepONet, with complex irregular geometries (GNO/GINO) not yet fully covered.
- Uncertainty is currently limited to distribution fitting and heuristic SSR; no coverage guarantees like conformal prediction are provided; rigorous UQ requires post-processing calibration.
- The coefficient dimension $r$ is a fixed hyperparameter; there is a lack of adaptive selection or hierarchical basis mechanisms, which may lead to over-compression for highly complex fields.

## Related Work & Insights
- **vs Pixel-space Diffusion (DM)**: DM learns diffusion directly on $128^2$ grids using strong U-Net spatial bias—high accuracy but high cost and slow sampling. DLL learns in $r=64$ coefficient space, reflecting lower costs but a ceiling limited by the backbone.
- **vs Generic Latent Diffusion (LDM)**: LDM uses operator-agnostic autoencoders for compression, losing the backbone's geometric/spectral structure. DLL generates input-dependent bases using the operator itself, keeping geometric info in the latent space.
- **vs Bayesian Neural Operators / FNO-d**: BNO and MC-dropout only characterize parameter posterior (epistemic), limited by priors/likelihoods. DLL learns the output conditional distribution, representing both aleatoric and epistemic uncertainty.
- **vs Probabilistic Neural Operator (PNO)**: PNO provides pointwise Gaussian predictions; DLL provides full-field joint non-Gaussian distributions, capturing spatial correlations and achieving significantly better distance metrics.
- **vs Function Space Diffusion (Lim et al., Shi et al.)**: That line of work defines diffusion directly in infinite-dimensional function space; DLL bypasses this complexity using finite-dimensional coefficients, making it more engineering-friendly but requiring well-learned bases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/physics/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)
- [\[ICML 2026\] Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](iterative_refinement_neural_operators_are_learned_fixed-point_solvers_a_principl.md)
- [\[ICML 2026\] PINNfluence: Interpreting PINNs Through Influence Functions](pinnfluence_interpreting_pinns_through_influence_functions.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)

</div>

<!-- RELATED:END -->
