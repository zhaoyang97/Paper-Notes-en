---
title: >-
  [Paper Note] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges
description: >-
  [ICML 2026][Image Restoration][Latent diffusion bridge] SDB reformulates modality translation as "selecting one coupling from the set $\mathcal{P}$ of all couplings satisfying marginal constraints." Built upon LDDBM…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Latent diffusion bridge"
  - "Marginal matching"
  - "Cycle consistency"
  - "Winner-takes-all"
  - "Semi-paired training"
date: 2026-05-08
content_hash: a729ce7622b4c3b3
---

# Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges

**Conference**: ICML 2026  
**arXiv**: [2605.02973](https://arxiv.org/abs/2605.02973)  
**Code**: None  
**Area**: Diffusion Models / Modality Translation / Image Super-resolution / Unpaired Learning  
**Keywords**: Latent diffusion bridge, Marginal matching, Cycle consistency, Winner-takes-all, Semi-paired training  

## TL;DR
SDB reformulates modality translation as "selecting one coupling from the set $\mathcal{P}$ of all couplings satisfying marginal constraints." Built upon LDDBM, it incorporates marginal matching (WTA + capacity constraints) and dual-layer cycle consistency (endpoint and trajectory levels). By treating paired supervision as an optional heuristic, the model is trainable under zero, semi, and full paired supervision budgets, outperforming paired-only baselines even in fully supervised settings (FFHQ→CelebA-HQ PSNR increased from 25.6 to 25.9).

## Background & Motivation

**Background**: Diffusion bridges (DDBM, LDDBM, etc.) have emerged as a powerful paradigm for inter-distribution translation. LDDBM addresses dimensionality inconsistency via a shared latent space and achieves SOTA in tasks like image super-resolution and shape↔voxel translation. however, most bridge methods require **full paired supervision**, where training samples must be paired $(x,y)$ such as LR-HR images or multi-view image-voxel pairs.

**Limitations of Prior Work**: Paired data implicitly carries three independent constraints: (i) semantic correspondence (correct mapping from source to target), (ii) distributional validity (outputs falling on the target marginal), and (iii) geometric consistency (reversibility). Forcing a single paired loss to handle all three is suboptimal; even with sufficient data, the model might minimize reconstruction error while deviating from the true manifold. Worse, in many scenarios (medical imaging, artistic style transfer), paired data is scarce or non-existent, rendering standard bridge methods unusable.

**Key Challenge**: Given only marginals $p_\mathcal{X}$ and $p_\mathcal{Y}$, there are **infinitely many** possible joint distributions $\mathcal{P}=\{p(x,y)|p(x)=p_\mathcal{X},p(y)=p_\mathcal{Y}\}$. Marginal information alone cannot uniquely determine a coupling. Existing methods use "paired samples" as an implicit constraint for Doob’s h-transform, effectively attempting to select a $p$ through data exhaustion, which is both inefficient and fragile.

**Goal**: (i) Explicitly restate modality translation as a geometric problem of "selecting a coupling within $\mathcal{P}$"; (ii) introduce composable structural constraints (marginal matching + cycle consistency) to allow the paired loss to degrade into just one of many heuristics; (iii) maintain graceful degradation across three paired ratios $\rho\in\{0,0.5,1\}$; (iv) achieve performance gains even under full paired supervision.

**Key Insight**: Classical unpaired translation (CycleGAN, CUT) utilizes cycle consistency; the Schrödinger bridge line of work (UNSB) uses adversarial training and entropy regularization. The authors **transplant these experiences into the latent diffusion bridge framework**—benefiting from the dimension-agnostic convenience of LDDBM while using the intermediate states $z_t$ exposed by the diffusion process to apply cycle consistency over the **entire trajectory** rather than just at endpoints.

**Core Idea**: Reformulate "training a diffusion bridge" as a weighted combination of several independent heuristics: marginal matching (ensuring end states land on the target marginal) + endpoint-level cycle ($y\to x\to\hat y\approx y$) + trajectory-level cycle (consistency between latent states of forward and backward trajectories at times $t$ and $T-t$) + optional paired supervision.

## Method

### Overall Architecture
The model follows the bidirectional diffusion bridge architecture of LDDBM: modality-specific encoders $E_\mathcal{X}, E_\mathcal{Y}$ map $x\in\mathcal{X}, y\in\mathcal{Y}$ into a shared latent space. The bridge is learned in the latent space, with forward score $s_{\mathcal{X}\to\mathcal{Y}}(z,t)$ and backward score $s_{\mathcal{Y}\to\mathcal{X}}(z,t)$. Each training step simultaneously optimizes a weighted sum of four objectives (Eq. 10): $\mathcal{L}_{total}=\mathcal{L}_{DSM}+\lambda_{end}\mathcal{L}_{cycle}^{end}+\lambda_{traj}\mathcal{L}_{cycle}^{traj}+\lambda_{pair}\mathbf{1}_{(x,y)\in\mathcal{D}_{pair}}\mathcal{L}_{pair}$, with all $\lambda=1$. The fourth term is active only for the subset when $\rho>0$, while the purely heuristic approach works when $\rho=0$.

### Key Designs

1. **Marginal Matching with Winner-Takes-All (WTA)**:
    - **Function**: Guarantees the bridge endpoint lands on the target marginal $p_\mathcal{X}$ in zero/semi-paired settings, while alleviating the degenerate coupling problem where "any pairing can optimize DSM."
    - **Mechanism**: Independently sample $x\sim p_\mathcal{X}$ and $y\sim p_\mathcal{Y}$. For each target $z_0=E_\mathcal{X}(x)$, draw $K$ conditional candidates $\{y^{(k)}\}_{k=1}^K\sim p_\mathcal{Y}$ and calculate the DSM loss $\mathcal{L}_{DSM}=\mathbb{E}\|s_\theta(z_t,t|y)-\nabla_{z_t}\log q(z_t|z_0)\|_2^2$. Backpropagate only for $k^\star=\arg\min_k \mathcal{L}_{DSM}(z_0,y^{(k)})$—selecting the condition that the current bridge explains best. To prevent "condition dominance" (where a few low-information $y$ are repeatedly selected), a capacity constraint $C_y=2$ is added, meaning each candidate $y^{(i)}$ can be selected at most twice per epoch.
    - **Design Motivation**: DSM with pure random pairing under $\rho=0$ learns an arbitrary mixed coupling (mode mixing). WTA is a classic optimization heuristic to select "locally most compatible" candidates to reduce coupling ambiguity. The capacity constraint prevents WTA from degrading into a state where "a few $y$ explain all $x$." Note that the authors state WTA is an optimization trick rather than an identifiability guarantee.

2. **Dual-level Cycle Consistency (Endpoint + Trajectory)**:
    - **Function**: Constrains the bridge to be approximately reversible at both the endpoints and along the entire trajectory, punishing irreversible mode dropping and arbitrary source/target mixing.
    - **Mechanism**: Let $\Phi_{\mathcal{X}\to\mathcal{Y}}$ be the forward stochastic flow and $\Phi_{\mathcal{Y}\to\mathcal{X}}$ be the backward flow. Endpoint-level $\mathcal{L}_{cycle}^{end}=\mathbb{E}\|\hat z_0-z_0\|_2^2$ where $\hat z_0=\Phi_{\mathcal{Y}\to\mathcal{X}}\circ\Phi_{\mathcal{X}\to\mathcal{Y}}(z_0)$. Trajectory-level pairs times $t$ and $T-t$ on the forward path $\{z_t^{X\to Y}\}$ and backward path $\{z_{T-t}^{Y\to X}\}$ to compute $\mathcal{L}_{cycle}^{traj}=\mathbb{E}[w(t)\|z_t^{X\to Y}-z_{T-t}^{Y\to X}\|_2^2]$, with weights $w(t)=1/(\sigma_t^2+\epsilon)$ for scale normalization.
    - **Design Motivation**: Endpoint cycles in CycleGAN are deterministic mappings. Since diffusion bridges are stochastic, endpoint cycles alone are insufficient. Trajectory-level consistency extends the constraint to the entire stochastic path, forcing the forward and backward directions to "travel through the same tunnel," equivalent to trajectory-level identifiability regularization. Empirical results show the trajectory term improves unpaired content accuracy from 16% to 87%.

3. **Unified Objective = Composable Heuristics + Paired as an Option**:
    - **Function**: Downgrades paired supervision from a "necessity" to one of four parallel heuristics, allowing the same code to be trained at any $\rho\in[0,1]$ with graceful degradation.
    - **Mechanism**: $\mathcal{L}_{total}=\mathcal{L}_{DSM}+\lambda_{end}\mathcal{L}_{cycle}^{end}+\lambda_{traj}\mathcal{L}_{cycle}^{traj}+\lambda_{pair}\mathbf{1}_{(x,y)\in\mathcal{D}_{pair}}\mathcal{L}_{pair}$. The paired loss uses an indicator to activate only for the paired subset. All $\lambda=1$ without fine-tuning. This additive combination is equivalent to adding multiple soft constraint surfaces in $\mathcal{P}$, gradually compressing the feasible set toward couplings that are reversible and condition-preserving.
    - **Design Motivation**: Geometrizing the training objective as a "union of heuristics" makes ablations clear—each term's role can be analyzed individually. The indicator-gated paired term simplifies engineering, as the dataloader does not need to strictly separate paired and unpaired samples.

### Loss & Training
Details provided in the third key design above. Specifically, $K$ WTA candidates and a capacity $C_y=2$ are used, with all $\lambda=1$. Bidirectional bridges are trained simultaneously (required for the cycle terms). In semi-paired settings, the size of the paired subset is determined by $\rho$, while the total number of endpoint samples remains constant.

## Key Experimental Results

### Main Results
FFHQ→CelebA-HQ Super-Resolution (Zero-shot SR, scanning $\rho$):

| Method | $\rho=0$ | $\rho=0.5$ | $\rho=1.0$ |
|---|---|---|---|
| **SDB PSNR↑** | **19.0 ± 0.6** | **25.2 ± 0.3** | **25.9 ± 0.3** |
| DiWa PSNR | n/a | 22.6 ± 0.2 | 23.3 |
| LDDBM PSNR | n/a | 24.9 ± 0.3 | 25.6 ± 0.4 |
| SDB SSIM↑ | 0.54 | 0.68 | 0.69 |
| SDB LPIPS↓ | 0.37 | 0.32 | 0.31 |

Impact of structural constraints on coupling quality on synthetic benchmarks ($\rho=0$):

| Method | SWD ↓ | MMD² ↓ | Content Acc. ↑ | Cycle MSE ↓ |
|---|---|---|---|---|
| Marginal matching only | 0.02021 | $-1.03\times10^{-4}$ | 0.162 | 0.972 |
| + Endpoint cycle | 0.01891 | $-1.69\times10^{-4}$ | 0.662 | 0.831 |
| + Trajectory cycle | 0.01968 | $-1.11\times10^{-4}$ | **0.868** | **0.680** |

### Ablation Study

| Config ($\rho$) | Key Change | Conclusion |
|---|---|---|
| MM only ($\rho=0$) | Content Acc 0.162 | Marginal matching aligns distributions but does not learn coupling |
| + Endpoint cycle | Acc 0.662 | Endpoint reversibility significantly restores semantic correspondence |
| + Trajectory cycle | Acc 0.868 | Trajectory constraints further reduce coupling ambiguity |
| Paired-only ($\rho=0.5$) | Acc 0.641 | Semi-paired paired loss is inferior to SDB heuristic combination |
| **SDB Semi-paired ($\rho=0.5$)** | **Acc 0.955** | Heuristics + paired data synergy is the strongest |
| Paired-only ($\rho=1.0$) | Acc 0.887 | Pure paired supervision even at $\rho=1$ is inferior to additional structural constraints |
| **SDB ($\rho=1.0$)** | **Acc 0.965** | Full paired + structural constraints simultaneously improve performance |

### Key Findings
- Semi-paired SDB at $\rho=0.5$ already reaches or exceeds the performance of Paired-only at $\rho=1.0$, indicating that structural constraints successfully take over two of the three tasks usually handled by paired data.
- Even at $\rho=1$, SDB outperforms the pure paired baseline (PSNR 25.9 vs 25.6, Content Acc 0.965 vs 0.887), confirming that structural constraints are **complementary** rather than redundant.
- At $\rho=0$, the purely heuristic PSNR of 19.0 is meaningful (exceeding random baselines), making the LDDBM framework trainable in zero-paired settings for the first time.
- In Multi-view→3D Voxel (ShapeNet) experiments, SDB comprehensively outperforms EDM and LDDBM at $\rho\in\{0.5,1.0\}$ and remains trainable at $\rho=0$ where baselines are unusable.

## Highlights & Insights
- **Heuristic combination perspective**: Previous bridge methods treated objectives as unified optimizations; this work decomposes them into four geometrically interpretable constraints. Each disabled term corresponds to relaxing a specific type of reversibility.
- **Trajectory-level cycle consistency**: Implementing trajectory consistency in a stochastic diffusion framework is stricter than CycleGAN's endpoint cycle. It constrains the **symmetry of the entire SDE path**, representing an elegant adaptation of reversibility intuitions from OT/Schrödinger bridges to DDBM.
- **WTA + Capacity constraints**: A simple implementation for "selecting couplings without correspondence supervision." It is easier than adversarial training or InfoNCE and acts as a nearly zero-cost plugin with significant gains in Content Accuracy.
- **Graceful degradation**: The same objective works across different paired budgets, making SDB a unified framework for real-world scenarios where data is often partially paired rather than either 0% or 100%.

## Limitations & Future Work
- Using $\lambda=1$ for all terms is elegant but might require weight tuning in more difficult high-resolution or 3D tasks; the paper does not provide a systematic study on weight tuning.
- The number of WTA candidates $K$ and capacity $C_y=2$ are empirical values sensitive to computation budgets; adaptive $K$ is a natural extension.
- Cycle consistency requires simultaneous training of bidirectional bridges, doubling parameter count and training costs.
- Evaluation is limited to established benchmarks like image SR and 3D voxel translation; performance on "semantically misaligned" open-domain translation (e.g., sketch↔photo) remains to be verified.
- The theoretical connection to OT-style work could be further explored, such as aligning trajectory cycles with Schrödinger bridge marginals + cost structures.

## Related Work & Insights
- **vs LDDBM (Berman 2026)**: Direct base model; SDB adds structural constraints and makes paired data optional.
- **vs DDBM (Zhou 2024)**: Uses Doob h-transform to implicitly constrain the bridge via paired data; SDB makes these constraints **explicit and decomposable**.
- **vs CycleGAN / CUT**: These use deterministic endpoint cycles; SDB extends cycle consistency to stochastic diffusion trajectories.
- **vs UNSB (Adversarial Schrödinger bridge)**: UNSB uses GANs for training and suffers from mode collapse; SDB uses score matching and cycles for more stable training.
- **vs LADB**: LADB reduces paired data requirements by reusing pre-trained source latent diffusion; SDB adds first-order heuristics directly into the bridge, offering an orthogonal and composable approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending cycle consistency to stochastic diffusion trajectories is an elegant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Real SR + 3D voxel tasks across three $\rho$ values with full ablation matrices.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear geometric framing of "translation = selecting a coupling in $\mathcal{P}$."
- Value: ⭐⭐⭐⭐ Unlocks zero/semi-paired applications for LDDBM, significantly impacting fields where paired data is scarce.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)
- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](../../ICCV2025/image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)

</div>

<!-- RELATED:END -->
