---
title: >-
  [Paper Note] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules
description: >-
  [ICML 2026][Image Restoration][Diffusion Posterior Sampling] This paper systematically treats the three forces in diffusion posterior sampling—Data Consistency (DC) guidance, Classifier-Free Guidance (CFG)…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Diffusion Posterior Sampling"
  - "CFG Scheduling"
  - "Stochasticity Regularization"
  - "GRPO"
  - "Inverse Problems"
date: 2026-05-08
content_hash: c9f1d126c87230fc
---

# Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules

**Conference**: ICML 2026  
**arXiv**: [2605.26470](https://arxiv.org/abs/2605.26470)  
**Code**: To be confirmed  
**Area**: Image Restoration / Diffusion Posterior Sampling / Inverse Problems  
**Keywords**: Diffusion Posterior Sampling, CFG Scheduling, Stochasticity Regularization, GRPO, Inverse Problems

## TL;DR
This paper systematically treats the three forces in diffusion posterior sampling—Data Consistency (DC) guidance, Classifier-Free Guidance (CFG), and stochasticity—hitherto treated as constants, as a **coupled time-varying triadic system**. Theoretical and empirical evidence shows that early CFG and DC directions conflict, while stochasticity pulls the trajectory back toward the high-probability manifold. Accordingly, a monotonic triadic scheduling trend of "DC↓, CFG↑, η↓" is proposed. Two methods, "Template Search + GRPO Reinforcement Learning," are used to find optimal curves, simultaneously setting new state-of-the-art records for both distortion and perceptual metrics in super-resolution and deblurring on FFHQ / DIV2K.

## Background & Motivation

**Background**: Diffusion/Flow Matching models have become the mainstream prior for imaging inverse problems (super-resolution, deblurring, inpainting, etc.). A standard posterior sampler performs three actions at each timestep: (1) pulls $\hat{x}_{0|t}$ toward the measurement subspace satisfying $y = \mathcal{A}x_0 + n$ using DC guidance; (2) extrapolates the score toward the text-conditional direction using CFG; (3) injects Gaussian noise $\eta(t)\epsilon$ to maintain stochasticity. These three forces are controlled by scalars $\beta(t)$, $\lambda(t)$, and $\eta(t)$, respectively.

**Limitations of Prior Work**: Almost all existing works treat these three scalars as **time-independent constants or only adjust them locally**. FlowChef sets $\lambda(t)=\lambda, \beta(t)=\beta, \eta(t)=0$; PiGDM and Fast Samplers only tune $\beta(t)$ to prevent over-saturation; ReSample/DDPG/FlowDPS jointly tune $\beta$ and $\eta$ but do not touch $\lambda$. While adjusting CFG over time has been proven to improve quality in text-to-image generation, it remains largely unexplored in the context of inverse problems.

**Key Challenge**: The authors argue that these three forces are **not independently tunable**. They interfere with each other on the same sampling trajectory: CFG aims to push samples toward the semantic manifold, while DC aims to pull samples toward the measurement manifold, and their directions are inherently inconsistent. Whether stochasticity can remedy this conflict had not been quantitatively characterized. Consequently, "optimal individual forces ≠ optimal collective force," leaving significant room for performance gains.

**Goal**: (i) Formulate the directional conflict of triadic coupling as computable geometric quantities; (ii) derive a **universal, data-driven validated scheduling trend**; (iii) provide two practical curve optimization frameworks covering both "interpretable baseline" and "performance maximization" needs.

**Key Insight**: Rewrite posterior sampling as a **time-varying optimal control problem**, where the state is $x_t$, the control is $(\beta(t), \lambda(t), \eta(t))$, and the objective is to maximize a joint perceptual-distortion reward. Once the control is time-varying, the coupling between the directions of the three forces can be observed through first-order derivative analysis (Proposition 1) and cosine similarity visualization.

**Core Idea**: The early high-noise phase must employ "strong DC + weak CFG + strong stochasticity" to establish global structure, suppress CFG-DC conflict, and pull the trajectory back to high-probability regions. The late low-noise phase should employ "weak DC + strong CFG + weak stochasticity" to refine semantics and avoid noise leakage. This is condensed into the **monotonic triadic scheduling trend**: $\beta(t)\downarrow, \lambda(t)\uparrow, \eta(t)\downarrow$.

## Method

### Overall Architecture
The backbone of TriPS is a standard Flow Matching posterior sampler (based on SD3.5-M or SD1.5), but it elevates $\beta, \lambda, \eta$ from constants to learnable/searchable time-varying functions. The algorithm operates on two layers:

1.  **Bottom-level sampler**: At each timestep, the velocity field after CFG enhancement is first calculated via $v_t(x_t)=v_\theta(x_t,\varnothing)+\lambda(t)(v_\theta(x_t,c)-v_\theta(x_t,\varnothing))$. The Flow Tweedie formula yields $\hat{x}_{0|t}$ and $\hat{x}_{1|t}$, followed by a DC gradient update $\tilde{x}_{0|t}=\hat{x}_{0|t}-\beta(t)\nabla\mathcal{L}(\mathcal{A}\hat{x}_{0|t},y)$ (where DC loss uses a hybrid form of Back-Projection and Least-Squares). Finally, stochasticity is injected via $\tilde{x}_{1|t}=\sqrt{1-\eta^2(t)}\hat{x}_{1|t}+\eta(t)\epsilon$, and the sample progresses to $x_{t+\Delta t}$ via Euler integration.

2.  **Top-level schedule optimizer**: Two complementary paradigms generate the $(\beta(t), \lambda(t), \eta(t))$ curves: $\text{TriPS}_\text{T}$ uses an analytical template family for coarse searching, and $\text{TriPS}_\text{G}$ uses GRPO reinforcement learning for fine-tuning, with $\text{TriPS}_\text{T}$ serving as a warm-start for $\text{TriPS}_\text{G}$.

To quantify triadic coupling, the authors define two types of **cosine similarity diagnostic quantities**: $\text{COS-SIM}_1(x_t)=\langle \tilde{b}_\text{dc},\tilde{b}_\text{cfg}\rangle/(\|\tilde{b}_\text{dc}\|\|\tilde{b}_\text{cfg}\|)$ measures the directional conflict between DC and CFG; $\text{COS-SIM}_2(x_t)=\langle b_\text{det},\nabla_{x_t}\log p_t(x_t)\rangle/(\cdots)$ measures the alignment between the total drift and the unconditional score. Empirical findings show that at early stages $t\simeq 1$, $\text{COS-SIM}_1$ is significantly negative (CFG opposes DC), and larger CFG leads to more severe conflict and slower reduction of the residual norm $\mathcal{R}(\hat{x}_{0|t})=\|y-\mathcal{A}\hat{x}_{0|t}\|^2$. While increasing both $\beta$ and $\lambda$ causes $\text{COS-SIM}_2$ to drop (trajectory deviates from the manifold), increasing $\eta$ uniquely pulls $\text{COS-SIM}_2$ back to a positive direction. This provides the empirical basis for the "triadic scheduling trend."

### Key Designs

1.  **Triadic Coupling Diagnosis + Proposition 1 (First-order derivative of residual norm w.r.t. CFG)**:
    - **Function**: Transforms the intuition of "whether CFG hinders DC" into a monitorable scalar signal.
    - **Mechanism**: Taking the first-order derivative of the expected next-step residual norm with respect to the CFG scale yields $\partial_{\lambda(t)}\mathbb{E}[\mathcal{R}(\hat{x}_{0|t+\Delta t})|x_t]=-\Delta t\langle\tilde{b}_\text{dc},\tilde{b}_\text{cfg}\rangle+o(\Delta t)$. When the inner product is negative (common in early stages), increasing $\lambda$ actually slows down residual reduction, providing rigid mathematical evidence that "early $\lambda$ must be low." Combined with $\text{COS-SIM}_2$, it empirically shows that stochasticity is the only variable that pulls $b_\text{det}$ back to the score direction, leading to the synergistic conclusion that "early $\eta$ and $\beta$ must be high."
    - **Design Motivation**: Previous works provided only empirical schedules and lacked theoretical explanation for why CFG cannot be fully applied from the start or why noise is needed to "feed" the sample back to the manifold. This diagnosis ensures the triadic trend $\beta\downarrow, \lambda\uparrow, \eta\downarrow$ is not just heuristic but mathematically and geometrically validated.

2.  **$\text{TriPS}_\text{T}$: Discrete Schedule Search Based on Function Templates**:
    - **Function**: Produces a robust and interpretable baseline schedule with minimal degrees of freedom.
    - **Mechanism**: Each curve is selected from three function families $\mathcal{T} = \{\text{linear, exp, log}\}$, with the constraint that the direction must satisfy the triadic monotonic trend. Magnitudes are truncated via $\lambda \in [1, 6]$, $\eta \in [0, 1]$, and $\beta \in [\beta_\min^T, \beta_\max^T]$. Thus, the search space for each task is only $|\mathcal{T}|^3 = 27$ combinations (plus a small grid for magnitudes). Grid search identifies $\tau^\star = \arg\max_\tau \mathcal{U}(\tau; \mathcal{D}_\text{cal})$ on a small calibration set $\mathcal{D}_\text{cal}$ based on a multi-objective utility $\mathcal{U}$ (composite of PSNR and LPIPS).
    - **Design Motivation**: Collapses high-dimensional optimization of independent per-timestep scalars into low-dimensional template selection, avoiding large-scale numerical optimization on non-differentiable samplers. The resulting curves naturally fall within the physically feasible domain and serve as a warm-start for GRPO, eliminating the risk of collapse during the RL cold-start phase.

3.  **$\text{TriPS}_\text{G}$: Bernstein-Beta Parameterization + GRPO Schedule Reinforcement Learning**:
    - **Function**: Goes beyond fixed function families to capture complex time-varying curves that templates cannot fit, pushing the perception-distortion trade-off to the limit.
    - **Mechanism**: Each curve is expressed as a $d$-th order Bernstein polynomial $\tilde{s}(t)=\sum_{k=0}^d w_k^{(s)}B_{k,d}(t)$ ($s\in\{\lambda,\beta,\eta\}$), with coefficients $w_k^{(s)}\sim\text{Beta}(a_k^{(s)},b_k^{(s)})$. Beta samples naturally reside in $(0,1)$, and the partition-of-unity property of Bernstein bases ensures $\tilde{s}(t)$ stays in $(0,1)$, which is then affinely mapped to the physical interval $[s_\min, s_\max]$. This structural parameterization guarantees exploration within legal boundaries. Policy parameters $\theta=\{a_k^{(s)},b_k^{(s)}\}$ are trained using GRPO: in each round, $G$ sets of coefficients $\{\mathbf{w}_i\}$ are sampled, the full sampler is run to generate reconstructed images, and the group-standardized advantage $\hat{A}_i$ is calculated based on a hybrid reward $R = w_\text{dist}R_\text{dist} + w_\text{perc}R_\text{perc}$ (PSNR + LPIPS + CLIP-IQA+ + Q-Align). The policy is updated using a PPO-style clipped objective $\max_\theta\mathbb{E}_i[\min(r_i\hat{A}_i,\text{clip}(r_i,1\pm\epsilon)\hat{A}_i)]-\beta_\text{KL}D_\text{KL}(\pi_\theta\|\pi_\text{ref})$, where the reference policy $\pi_\text{ref}$ is initialized as $\mathbf{S}_\text{T}$ found by $\text{TriPS}_\text{T}$.
    - **Design Motivation**: GRPO requires neither a value network nor a differentiable sampler (traditional actor-critic is unfeasible here). Group-standardization for baseline estimation is ideal for settings where rewards require full sampling runs. The Bernstein-Beta parameterization ensures structural stability during RL exploration, which is more robust than simple penalty terms.

### Loss & Training
The $\text{TriPS}_\text{T}$ phase involves no gradient training, only grid search on a calibration set. The $\text{TriPS}_\text{G}$ phase uses a weighted sum of PSNR, LPIPS, CLIP-IQA+, and Q-Align as rewards. Hyperparameters for group size $G$, KL coefficient $\beta_\text{KL}$, and PPO clip $\epsilon$ are provided in Appendix E.2. The reference policy is fixed to the optimal curve from $\text{TriPS}_\text{T}$ to provide a warm-start and restrict policy drift.

## Key Experimental Results

### Main Results
FFHQ ($768^2$, 1000 images) + DIV2K ($768^2$, 800 images), backbone SD3.5-M, NFE=28, measurement noise $\sigma_n=0.03$.

| Task / Dataset | Metric | FlowChef | FlowDPS | FLAIR | $\text{TriPS}_\text{T}$ | $\text{TriPS}_\text{G}$ |
|---|---|---|---|---|---|---|
| FFHQ SR×8 | PSNR↑ / LPIPS↓ | 27.53 / 0.147 | 27.92 / 0.120 | 28.88 / 0.123 | **29.03** / 0.113 | 28.55 / **0.107** |
| FFHQ Motion Deblur | PSNR↑ / FID↓ | 24.88 / 63.48 | 25.15 / 43.18 | 28.80 / 21.57 | **31.20** / 17.28 | **31.20** / **15.89** |
| FFHQ Gaussian Deblur | PSNR↑ / LPIPS↓ | 27.30 / 0.152 | 26.02 / 0.204 | 28.60 / 0.090 | **29.95** / 0.084 | 29.60 / **0.074** |
| DIV2K SR×8 | PSNR↑ / FID↓ | 22.08 / 47.47 | 22.14 / 35.18 | 22.90 / 41.23 | **23.05** / 31.80 | 22.78 / **27.84** |
| DIV2K Motion Deblur | PSNR↑ / LPIPS↓ | 19.62 / 0.366 | 19.88 / 0.322 | 23.90 / 0.129 | **26.29** / 0.066 | 26.19 / **0.066** |

$\text{TriPS}_\text{T}$ generally performs best in distortion metrics, while $\text{TriPS}_\text{G}$ excels in perceptual metrics. On motion deblurring, the gain over FLAIR exceeds 2 dB in PSNR, while KID/LPIPS are nearly halved.

### Schedule Migration and Backbone Validation
| Setting | Method | PSNR↑ | LPIPS↓ | KID↓ |
|---|---|---|---|---|
| FFHQ Gaussian Deblur (Direct migration of schedule learned on SR×8) | FLAIR | 27.74 | 0.109 | 0.012 |
| Same as above | $\text{TriPS}_\text{G}$ on SR×8 | **28.90** | **0.089** | 0.014 |
| FFHQ SR×12 (Cross-degradation migration) | FLAIR | 27.51 | 0.148 | 0.017 |
| Same as above | $\text{TriPS}_\text{G}$ on SR×8 | **28.80** | **0.099** | **0.012** |

Schedules learned by GRPO still outperform baselines on unseen degradation operators, indicating that the triadic trend captures structural laws weakly related to specific $\mathcal{A}$. Table 3 shows consistent advantages on SD1.5 across PSLD/DDPG/P2L/TReg.

### Key Findings
- **Early CFG and DC are truly directionally opposed**: The $\text{COS-SIM}_1$ in Figure 1 is negative at $t\simeq 1$, with larger $\lambda$ exacerbating the conflict. High $\lambda$ directly produces "tiger-stripe hallucinations," destroying measurement consistency, for which Proposition 1 provides an analytical explanation.
- **Stochasticity is a hidden early regularizer**: Figure 2 shows that increasing $\beta$ or $\lambda$ monotonically decreases $\text{COS-SIM}_2$ (deviating from the manifold), while only increasing $\eta$ steadily pulls the total drift back to the score direction. KID experiments also prove that appropriate early noise reduces the gap between generated and real distributions.
- **GRPO is more aggressive but more sensitive than templates**: In settings favoring perceptual metrics, $\text{TriPS}_\text{G}$ wins comprehensively, though its PSNR is sometimes lower than $\text{TriPS}_\text{T}$, suggesting RL exploration biases toward reward-dominant directions. Bernstein-Beta + KL constraints ensure it stays within physical feasibility.

## Highlights & Insights
- **Shift from "tuning a parameter" to "controlling a trajectory"**: The primary perspective shift is treating posterior sampling as a time-varying optimal control problem. Explicitly modeling the coupling of CFG-DC-stochasticity makes the monotonic triadic trend almost an inevitable conclusion. This idea can be transferred to any diffusion control scenario involving multiple guiding forces (e.g., human feedback alignment, controllable generation, video posterior).
- **Bernstein-Beta parameterization is an elegant trick for "hard constraints" in RL**: Moving the feasible domain from the reward function into the network structure using basis function partition of unity + bounded distributions is far more stable than simply increasing KL or penalty terms. This "guaranteed safe exploration via parameterization" is reusable in any RL scheduling scenario.
- **Diagnosis-first methodology**: Defining quantifiable diagnostics ($\text{COS-SIM}_1$ for conflict, $\text{COS-SIM}_2$ for manifold deviation), then deriving scheduling trends from these results, and finally providing optimization frameworks—this "observation → law → engineering" approach is more convincing and reproducible than direct NAS/RL search.

## Limitations & Future Work
- The "hard monotonicity" constraint of the triadic trend ($\beta\downarrow, \lambda\uparrow, \eta\downarrow$) might not be optimal for certain extreme degradations (strong non-linearity, extreme brightness/darkness), which the paper does not discuss.
- $\text{TriPS}_\text{G}$ requires repeated full sampling runs to collect rewards, causing training costs to grow linearly with group size $G$ and NFE. While NFE is set to 28, costs may explode for larger models or higher resolutions.
- Text prompts were limited to fixed templates (e.g., "A high quality photo of a face" for FFHQ). The sensitivity of CFG scheduling to prompt quality is not fully analyzed, which might affect real-world reproducibility.
- While cross-degradation migration yields good results, it was only validated on a 100-image subset of FFHQ; robustness across data domains (e.g., natural images to medical/satellite) remains to be investigated.

## Related Work & Insights
- **vs FlowChef / FlowDPS**: These works treat $\beta, \lambda, \eta$ as constants or only tune them locally. This paper systemizes them as time-varying controls, with performance gains primarily stemming from the "low CFG + high stochasticity" combination in early stages.
- **vs FLAIR**: FLAIR is a strong flow-matching baseline. TriPS matches or slightly exceeds its PSNR via $\text{TriPS}_\text{T}$ but significantly outperforms it in perceptual metrics (LPIPS / FID / KID) via $\text{TriPS}_\text{G}$, with the gap coming from non-trivial time-varying curves.
- **vs Limited Interval CFG (Generation tasks)**: That work found CFG is only useful in middle intervals for text-to-image generation. This paper effectively migrates and geometrically explains the "interval CFG" concept within the context of inverse problems and coupling with DC.
- **vs Restart Sampling / DDPM Stochasticity Studies**: Where those works view stochasticity as a "perturbation source for restarting," this paper assigns it a new role as an "early manifold stabilizer," consistent with KID evidence.

## Rating
- Novelty: ⭐⭐⭐⭐ First to explicitly model three scalars in diffusion posterior sampling as a time-varying coupled system. Analysing CFG-DC conflict via first-order derivatives and interpreting stochasticity as regularization is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers FFHQ/DIV2K across SR and deblurring using both flow and diffusion backbones, plus migration experiments. Limited mostly to faces and single resolutions.
- Writing Quality: ⭐⭐⭐⭐ The diagnosis-first narrative is clear. Proposition 1 provides mathematical support for key intuitions. The two-track approach ($\text{TriPS}_\text{T}/\text{TriPS}_\text{G}$) is well-defined.
- Value: ⭐⭐⭐⭐ The triadic trend + Bernstein-Beta + GRPO framework can be directly applied to other diffusion sampling problems requiring time-varying multi-force scheduling. High engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] UOTIP: Unbalanced Optimal Transport Mapping for Unpaired Inverse Problems](uotip_unbalanced_optimal_transport_map_for_unpaired_inverse_problems.md)
- [\[ICML 2026\] Learning Normalized Energy Models for Linear Inverse Problems](learning_normalized_energy_models_for_linear_inverse_problems.md)
- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](../../CVPR2026/image_restoration/variational_garrote_for_sparse_inverse_problems.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)

</div>

<!-- RELATED:END -->
