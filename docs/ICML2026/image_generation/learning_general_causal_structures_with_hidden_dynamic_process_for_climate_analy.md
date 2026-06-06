---
title: >-
  [Paper Note] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis
description: >-
  [ICML 2026][Image Generation][Causal Discovery] This paper proposes CaDRe, which utilizes a structurally constrained sequential VAE to jointly identify the "causal graph between observed variables" and the "latent dynami…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Causal Discovery"
  - "Causal Representation Learning"
  - "Latent Dynamic System"
  - "Nonlinear ICA"
  - "VAE"
date: 2026-05-08
content_hash: 3a291eae9a6a71ea
---

# Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis

**Conference**: ICML 2026  
**arXiv**: [2501.12500](https://arxiv.org/abs/2501.12500)  
**Code**: https://github.com/MinghaoFu/CaDRe (Available)  
**Area**: Causal Inference / Causal Representation Learning / Sequential Generative Models / Climate Modeling  
**Keywords**: Causal Discovery, Causal Representation Learning, Latent Dynamic System, Nonlinear ICA, VAE

## TL;DR
This paper proposes CaDRe, which utilizes a structurally constrained sequential VAE to jointly identify the "causal graph between observed variables" and the "latent dynamic processes driving observations" within a single non-parametric framework. It provides identifiability theorems for simultaneously recovering both from sequential data, validates the theory on synthetic data, and achieves causal graphs consistent with domain experts along with competitive temperature prediction accuracy on CESM2 climate data.

## Background & Motivation
**Background**: Understanding the causal structure of climate systems is both the foundation of scientific reasoning and a prerequisite for reliable prediction. Current mainstream approaches follow two paths: one is Causal Representation Learning (CRL), which typically recovers "latent drivers" $\mathbf{z}_t$ from observations based on nonlinear ICA using auxiliary variables, sparsity, or restricted generative functions; the other is classical Causal Discovery (CD), such as LiNGAM, additive noise models, and PC/FCI, which construct DAGs among observed variables.

**Limitations of Prior Work**: Both paths have blind spots. CRL generally assumes no direct causal links between observed variables and that the generation $\mathbf{z}_t \to \mathbf{x}_t$ is noise-free and invertible—conditions that climate data fails to meet (e.g., temperatures in adjacent regions influence each other via heat conduction, and observations contain strong stochastic perturbations). The CD branch can handle graph structures among observations but is either parametric (LiNGAM assumes linear non-Gaussianity) or can only provide PAG equivalence classes under non-parametric settings with latent confounders (e.g., FCI), identifying neither the latent variables themselves nor the causal relationships between them.

**Key Challenge**: In climate systems, "invisible atmospheric/radiative dynamics" and "visible spatial coupling between regions" are coupled—latent processes drive observations, while observations have their own causal graphs. Neither can be handled individually by existing methods.

**Goal**: To simultaneously identify (i) the causal graph $\mathbf{J}_g(\mathbf{x}_t)$ between observed variables $\mathbf{x}_t$, and (ii) the latent variables $\mathbf{z}_t$ along with their instantaneous and lagged dependencies, within a single sequential non-parametric framework, integrating both into an estimable generative model.

**Key Insight**: The authors provide two key observations. First, the DAG structure ensures that the distribution transformation from noise sources $\mathbf{s}_t$ to observations $\mathbf{x}_t$ is injective—causal propagation between observations does not destroy the recoverability of the latent space. Second, the original SEM is equivalent to a nonlinear ICA with latent variables in the sense of data generation, allowing the use of ICA identifiability tools for CD.

**Core Idea**: Use context information from three consecutive observations $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ to recover the latent space to the value level $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$. Then, leverage the SEM $\leftrightarrow$ ICA equivalence to represent the observational causal graph as $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$. Finally, perform unified estimation on a VAE with sparsity and DAG constraints.

## Method

### Overall Architecture
The data generation process is formulated as a sequential SEM with latent variables: observations $x_{t,i}=g_i(\mathbf{pa}_O(x_{t,i}),\mathbf{pa}_L(x_{t,i}),s_{t,i})$ are simultaneously influenced by other observed components $\mathbf{x}_{t,\setminus i}$, current and historical latent variables $\mathbf{z}_t,\mathbf{z}_{t-1}$, and endogenous noise $s_{t,i}=g_{s_i}(\mathbf{z}_t,\epsilon^x_{t,i})$ that depends on $\mathbf{z}_t$. Latent variables $z_{t,j}=f_j(\mathbf{pa}_L(z_{t,j}),\epsilon^z_{t,j})$ have both instantaneous and lagged dependencies.

CaDRe is a state-space VAE: the input is a sequence $\mathbf{x}_{1:T}$, a z-encoder $\phi$ infers $\hat{\mathbf{z}}_t$ from observations, an s-encoder $\eta$ infers non-stationary noise $\hat{\mathbf{s}}_t$, and a decoder $\psi$ reconstructs $\hat{\mathbf{x}}_t$ from $(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$. Two prior networks (normalizing flows) learn the temporal prior of $\hat{\mathbf{z}}_t$ and the conditional prior of $\hat{\mathbf{s}}_t$, respectively, with causal graphs extracted from their Jacobians. The loss involves ELBO + sparsity penalty $\mathcal{L}_s$ + DAG penalty $\mathcal{L}_d$.

### Key Designs

1. **Non-parametric Value-level Identifiability of Latent Space (Theorem 3.2)**:
    - **Function**: Recovers latent variables to the level of $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$ ($h_z$ is invertible and differentiable) from a mixture of observational causal graphs and stochastic noise, without assuming noise-free invertibility of $\mathbf{z}_t \to \mathbf{x}_t$.
    - **Mechanism**: Lemma 3.1 first proves that $L_{\mathbf{x}_t\mid \mathbf{s}_t}$ is injective—the DAG ensures information flows unidirectionally along causal paths, so the graph between observations does not block distribution-level recovery. Then, through the injectivity conditions (A2) of operators $L_{\mathbf{x}_{t+1}\mid \mathbf{z}_t}$ and $L_{\mathbf{x}_{t-1}\mid \mathbf{x}_{t+1}}$ over three-step contexts $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$, combined with latent drift (A3, where different $\mathbf{z}_t$ yield different conditional distributions $p(\mathbf{x}_t\mid\mathbf{z}_t)$) and differentiability (A4), the uniqueness of the posterior set $\{p(\mathbf{x}_t\mid\hat{\mathbf{z}}_t)\}$ is proven, followed by the extraction of the value-level mapping.
    - **Design Motivation**: Previous non-parametric identification based on eidendecomposition (Hu & Schennach type) could only recover to the distributional level $p_{\hat{\mathbf{z}}_t}=p_{\mathbf{z}_t}$, which cannot support downstream component-wise CD. Furthermore, they required partially known generative functions, which is unfriendly to black-box systems like climate. This paper elevates the conclusion to the value level using A4 (requiring only a differentiable encoder), which is a prerequisite for Theorems 3.5/3.6.

2. **SEM $\leftrightarrow$ Nonlinear ICA Equivalence and Closed-form Expression of Observational Causal Graph (Lemma 3.3 + Theorem 3.5)**:
    - **Function**: Transforms "non-parametric causal discovery between observations" into "mixing structure identification of nonlinear ICA with latent variables," thus bypassing the non-parametric CD dilemma where methods like FCI/CDNOD only provide equivalence classes.
    - **Mechanism**: It is proven that for each $i$, there exists $m_i$ such that $x_{t,i}=g_i(\mathbf{pa}_O,\mathbf{pa}_L,s_{t,i})=m_i(\mathbf{z}_t,\mathbf{s}_t)$ describes the same data generation process—an edge $x_{t,2}\to x_{t,1}$ in the SEM is equivalent to an "indirect path" $s_{t,2}\to x_{t,2}\to x_{t,1}$ from the ICA perspective. By defining Jacobians $[\mathbf{J}_g(\mathbf{x}_t)]_{i,j}=\partial x_{t,i}/\partial x_{t,j}$, $[\mathbf{J}_m(\mathbf{s}_t)]_{i,j}=\partial x_{t,i}/\partial s_{t,j}$ and the diagonal block $\mathbf{D}_m(\mathbf{s}_t)$, it follows that $\mathbf{J}_g(\mathbf{x}_t)\mathbf{J}_m(\mathbf{s}_t)=\mathbf{J}_m(\mathbf{s}_t)-\mathbf{D}_m(\mathbf{s}_t)$, leading to $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$. Under A5 (generation variability, requiring the $2d_x$ Jacobian vectors $\mathbf{V}(t,k),\mathbf{U}(t,k)$ to be linearly independent), Theorem 3.6 further establishes that the support set $\text{supp}(\mathbf{J}_g(\mathbf{x}_t))$ of the observational causal graph is identifiable.
    - **Design Motivation**: Previous ICA-based CD (Monti 2020, Reizinger 2023) required no latent confounding. This paper incorporates the latent process $\mathbf{z}_t$ as a "continuous conditional prior" into ICA. The DAG structure simultaneously helps remove the invertibility assumption for ICA—this is key to obtaining "both latent and observational causal graphs non-parametrically" in climate systems with latent dynamics (Table 2).

3. **Dual Prior Estimation Architecture based on Normalizing Flow + Jacobian Graph Extraction**:
    - **Function**: Implements the three theorems into a trainable VAE and directly extracts three types of structures from the Jacobians of the two prior networks: "latent instantaneous graph $\mathbf{J}_r(\hat{\mathbf{z}}_t)$, latent lagged graph $\mathbf{J}_r(\hat{\mathbf{z}}_{t-1})$, and observational causal graph $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)$."
    - **Mechanism**: The z-prior uses a conditional normalizing flow to learn the inverse transformation $\hat{\epsilon}^z_{t,i}=r_i(\hat{\mathbf{z}}_{t-1},\hat{\mathbf{z}}_t)$. The latent instantaneous and lagged structures are read directly from the block Jacobian of the transformation $\kappa$; the s-prior similarly learns $\hat{\epsilon}^x_{t,i}=w_i(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$. The observational causal graph is calculated via $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)=\mathbf{I}-\mathbf{D}_{\hat m}\mathbf{J}_{\hat m}^{-1}$ using the decoder's $\mathbf{J}_{\hat m}(\hat{\mathbf{s}}_t)$ per Corollary 2; thus, the DAG is time-varying with $\hat{\mathbf{z}}_t$. To avoid redundant edges and cycles, sparsity penalties $\mathcal{L}_s=\|\mathcal{M}(\mathbf{J}_r(\hat{\mathbf{z}}_t))\|_1+\|\mathcal{M}(\mathbf{J}_r(\hat{\mathbf{z}}_{t-1}))\|_1+\|\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)\|_1$ are added, where $\mathcal{M}(\mathbf{J})=(\mathbf{I}+\mathbf{J})^\top(\mathbf{I}+\mathbf{J})-\mathbf{I}$ represents the Markov network structure; the DAG constraint is $\mathcal{L}_d=\mathcal{D}_g(\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t))+\mathcal{D}_g(\mathbf{J}_r(\hat{\mathbf{z}}_t))$ with $\mathcal{D}_g(A)=\mathrm{tr}[(\mathbf{I}+\tfrac1d A\circ A)^d]-d$ (following Yu et al. 2019).
    - **Design Motivation**: After removing the invertibility requirement of ICA via the DAG, the mixing structure $\mathbf{J}_m$ remains an invertible square matrix (Corollary 1). Thus, $\mathbf{J}_g$ is obtained via an analytical closed-form rather than learning another graph network, resulting in fast inference (only 1.1 ms on CESM2) rooted in theory. Flows are used instead of Gaussian priors to ensure the "independent components" assumption holds, allowing the KL term to effectively drive $\hat{\bm{\epsilon}}^z, \hat{\bm{\epsilon}}^x \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, satisfying the conditional independence prerequisite in Theorem 3.6.

### Loss & Training
Total loss: $\mathcal{L}_{ALL}=\mathcal{L}_{ELBO}+\alpha\mathcal{L}_s+\beta\mathcal{L}_d$. ELBO's two KL coefficients are $\lambda_1=4\times10^{-3}$ and $\lambda_2=1.0\times10^{-2}$ (controlling the prior matching strength for s and z); structural loss coefficients are $\alpha=1.0\times10^{-4}$ and $\beta=5.0\times10^{-5}$. z/s encoders, decoder, and prior networks are all MLPs, with normalizing flows in the prior approximating the posterior to $\mathcal{N}(\mathbf{0},\mathbf{I})$. A three-step window is used as context under the first-order Markov assumption (generalizations beyond second-order are in Appendix E.3).

## Key Experimental Results

### Main Results
Evaluations include "latent representation recovery" and "observational causal graph recovery" on synthetic data, and "causal graph quality" and "temperature prediction" on climate data.

| Dataset | Metric | CaDRe | Prev. SOTA | Gain / Difference |
| :--- | :--- | :--- | :--- | :--- |
| Synthetic Independent ($d_z=3,d_x=10$) | MCC | **0.9811** | TDRL 0.9106 | +0.07, latent factors almost perfectly recovered |
| Synthetic Sparse | MCC | **0.9306** | G-CaRL 0.7701 | +0.16, closest to actual climate sparsity |
| Synthetic Dense | MCC | **0.6750** | G-CaRL 0.6714 | Parity, all methods degrade under dense latent graphs |
| CESM2 (CD) | WSHD ↓ | **0.012** | LPCMCI 0.019 | Lower than all PC/FCI/CDNOD/PCMCI/LPCMCI/TCDF/TDRL/IDOL |
| CESM2 (CD) | WTPR ↑ | **0.532** | TCDF 0.327 | 60% higher true edge recall than second best |
| CESM2 (CD) | Latency (ms) ↓ | **1.095** | TDRL 0.974 | Same tier as fastest baseline, ~3000x faster than PCMCI/LPCMCI |
| CESM2 Forecast (H=96) | MSE | **0.410** | CARD 0.409 | Parity with SOTA, but provides interpretable causal graphs |
| CESM2 Forecast (H=192) | MSE | **0.412** | CARD 0.422 | Slightly superior for long-range prediction |

### Ablation Study
The paper uses three levels of synthetic structures (Independent / Sparse / Dense) as structural ablations. Representative comparisons are shown below:

| Configuration | MCC (Sparse) | $R^2$ (Sparse) | Description |
| :--- | :--- | :--- | :--- |
| CaDRe (Full) | **0.9306** | **0.9102** | z/s dual encoder + flow prior + sparsity + DAG |
| iCITRIS (No explicit $\mathbf{s}_t$) | 0.4531 | 0.6326 | Fails to separate endogenous noise from latent factors $\to$ entanglement |
| TDRL / LEAP (No obs. DAG) | 0.6628 / 0.6453 | 0.6953 / 0.4637 | Treats obs. causal links as independent noise $\to$ severe drop in sparse scenarios |
| G-CaRL (No flow + weak prior) | 0.7701 | 0.5443 | Insufficient prior expressivity, latent component independence fails |

### Key Findings
- In sparse scenarios (most realistic for climate), CaDRe outperforms the second-best method by 0.16 in MCC and 0.22 in $R^2$—demonstrating that "explicit $\mathbf{s}_t$ modeling + DAG-constrained observational graphs" contribute most when structures are sparse. Performance degradation for all methods in Dense settings confirms that the sparsity assumption (Appendix A.3) is an essential prerequisite for component-wise identifiability.
- Inference latency on CESM2 is approximately three orders of magnitude faster than classical CD methods like PCMCI/LPCMCI (1.1 ms vs 3000+ ms), as $\mathbf{J}_g$ is obtained in a single closed-form step from the decoder’s $\mathbf{J}_m$ rather than running conditional independence tests every time.
- On the temperature prediction task, CaDRe is nearly on par with pure prediction models like CARD and iTransformer, while delivering a causal graph consistent with expert knowledge for free—indicating that incorporating causal structure does not incur significant predictive accuracy costs.

## Highlights & Insights
- **DAG Rescues ICA Invertibility**: Traditional nonlinear ICA must assume $\mathbf{z}\to\mathbf{x}$ is invertible. This paper uses the SEM's DAG to prove $\mathbf{J}_m$ is invertible (Corollary 1). This implies that as long as the data generation satisfies the acyclicity assumption, one can align the noise dimension with the observation dimension, allowing ICA to function in realistic scenarios with latent confounding and observational coupling.
- **Formulating "Observational Causal Graphs" as Analytical Functions of Decoder Jacobians**: $\mathbf{J}_g=\mathbf{I}-\mathbf{D}_m\mathbf{J}_m^{-1}$ is the most practical conclusion of this theory—once the VAE is trained, the causal graph is generated for free, varies over time, and supports millisecond-level inference. This is transferable to any task requiring time-varying causal graphs on top of generative models (e.g., fMRI in brain science, macro-economic time series).
- **Applying "Context" to Distribution-level ICA**: Using the $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ three-window to construct operator injectivity conditions (A2) essentially replaces "exogenous labels" in traditional auxiliary variable ICA with "temporal neighborhoods," making it applicable to any continuous temporal systems without explicit domain labels.

## Limitations & Future Work
- First-order Markov + sparse latent structures are default assumptions. Although high-order Markov generalizations are provided in Appendix E.3, whether multi-scale coupling in actual climate systems (daily, seasonal, annual) can be identified in a single flow lacks empirical validation.
- Assumption A5 (generation variability) requires $2d_x$ Jacobian vectors to be linearly independent, a strong non-degereacy condition. As the observation dimension $d_x$ grows and variables become highly collinear, the verifiability of this assumption on actual climate grids requires more careful discussion.
- Experiments are primarily conducted on the single CESM2 climate dataset, lacking systematic validation on more diverse physical scenarios (e.g., ocean thermohaline circulation, ecosystems); forecasting comparisons are limited to relatively short horizons of 96/192.
- The numerical stability of the decoder's $\mathbf{J}_m^{-1}$ in high dimensions may become a bottleneck; the paper does not provide an analysis of stability under large-scale observation dimensions.

## Related Work & Insights
- **vs iCITRIS / TDRL / LEAP / G-CaRL (CRL line)**: These methods only identify latent factors and latent causes, treating dependencies between observations as noise. Ours simultaneously identifies "observational graphs + latent graphs + cross-layer effects," with MCC gains of 0.16–0.5 in sparse settings. The advantage lies in explicitly separating endogenous noise $\mathbf{s}_t$; the cost is higher model complexity requiring two prior networks.
- **vs FCI / LPCMCI (CD line)**: FCI/LPCMCI only provide PAG equivalence classes under latent confounding, and non-parametric CI tests are slow on high-dimensional time series. This paper converts CD into ICA + Jacobian graph extraction, identifying "specific directed graphs" while compressing inference cost from seconds to milliseconds.
- **vs CDSD (Brouillard et al. 2024)**: CDSD also attempts to unify CRL and CD but does not identify causal graphs between observations. This work is the only method in Table 2 that ticks all five attributes (Non-parametric / Latent variables / Latent causal graph / Observational causal graph / No equivalence class).
- **Insight**: This "trio" of using DAGs to drop ICA invertibility, flow priors to ensure independence, and Jacobians for graph extraction may serve as a general paradigm for upgrading any VAE-style generative model with causal interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified framework providing component-wise identifiability amidst non-parametric settings, latent dynamics, and observational causal graphs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across three synthetic structure levels, CESM2 causal graphs, and temperature prediction tasks, though verified on only one real climate dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear transition from theory to implementation; dependencies between the three theorems are well-indicated, though minor symbols (e.g., $\mathbf{J}_d$ vs $\mathbf{J}_r$) show slight inconsistency in formulas.
- Value: ⭐⭐⭐⭐⭐ Addresses long-standing pain points where CRL and CD were treated separately, providing a general "VAE + causal graph" template reusable in climate, brain science, and economics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICLR 2026\] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning](../../ICLR2026/image_generation/embracing_discrete_search_a_reasonable_approach_to_causal_structure_learning.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)
- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](../../CVPR2026/image_generation/dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)

</div>

<!-- RELATED:END -->
