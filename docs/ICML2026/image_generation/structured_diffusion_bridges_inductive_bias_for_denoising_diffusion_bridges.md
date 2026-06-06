---
title: >-
  [Paper Note] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges
description: >-
  [ICML 2026][Image Generation][Latent diffusion bridge] SDB reframes modality translation as "selecting a coupling from all joint distributions $\mathcal{P}$ satisfying marginal constraints…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Latent diffusion bridge"
  - "Marginal matching"
  - "Cycle consistency"
  - "Winner-takes-all"
  - "Semi-paired training"
date: 2026-05-08
content_hash: 22688d57eb128ddc
---

# Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges

**Conference**: ICML 2026  
**arXiv**: [2605.02973](https://arxiv.org/abs/2605.02973)  
**Code**: None  
**Area**: Diffusion Models / Modality Translation / Image Super-Resolution / Unpaired Learning  
**Keywords**: Latent diffusion bridge, Marginal matching, Cycle consistency, Winner-takes-all, Semi-paired training

## TL;DR
SDB reframes modality translation as "selecting a coupling from all joint distributions $\mathcal{P}$ satisfying marginal constraints," stacking marginal matching (WTA + capacity constraint) and both endpoint-level and trajectory-level cycle consistency on top of LDDBM. Paired supervision becomes merely an optional heuristic, enabling training under zero-paired, semi-paired, and fully-paired regimes. Even with full pairing, SDB outperforms paired-only baselines (e.g., FFHQ→CelebA-HQ PSNR improves from 25.6 to 25.9).

## Background & Motivation

**Background**: Diffusion bridges (DDBM, LDDBM, etc.) have become powerful paradigms for distribution translation. LDDBM addresses endpoint dimensionality mismatch via a shared latent space, achieving SOTA on tasks like image super-resolution and shape↔voxel translation. However, nearly all bridge methods require **fully paired supervision**—training samples must be $(x, y)$ pairs such as LR-HR or multi-view-image–voxel.

**Limitations of Prior Work**: Paired data implicitly enforces three independent constraints: (i) semantic correspondence (correct source-to-target mapping), (ii) distribution validity (outputs lie on the target marginal), and (iii) geometric consistency (invertibility). Relying on a single paired loss for all three is suboptimal: even with abundant data, models may minimize reconstruction error while drifting from the true manifold. Worse, paired data is scarce or nonexistent in many scenarios (e.g., medical imaging, artistic style transfer), severely limiting bridge methods.

**Key Challenge**: Given only marginals $p_\mathcal{X}, p_\mathcal{Y}$, the feasible set of joint distributions $\mathcal{P} = \{p(x, y) | p(x) = p_\mathcal{X}, p(y) = p_\mathcal{Y}\}$ is **infinite**; marginal information alone cannot uniquely determine the coupling. Existing methods use "paired samples" as an implicit Doob h-transform constraint, effectively enumerating $p$ via data, which is inefficient and brittle.

**Goal**: (i) Explicitly reframe modality translation as a geometric problem of "selecting a coupling in $\mathcal{P}$"; (ii) introduce composable structural constraints (marginal matching + cycle consistency), relegating paired loss to just one of many heuristics; (iii) ensure graceful degradation across $\rho \in \{0, 0.5, 1\}$ paired ratios; (iv) achieve gains even under full pairing.

**Key Insight**: Classic unpaired translation (CycleGAN, CUT) leverages cycle consistency; Schrödinger bridge approaches (UNSB) use adversarial training + entropy regularization. The authors **transplant these insights into the latent diffusion bridge framework**—retaining LDDBM's dimension-agnostic benefits, while leveraging intermediate diffusion states $z_t$ to enforce cycle consistency along the **entire trajectory**, not just endpoints.

**Core Idea**: Reconstruct "training a diffusion bridge" as a weighted combination of independent heuristics: marginal matching (ensuring terminal states land on the target marginal) + endpoint-level cycle ($y \to x \to \hat y \approx y$) + trajectory-level cycle (forward and reverse trajectories match latent states at $t$ and $T-t$) + optional paired supervision.

## Method

### Overall Architecture
The bidirectional diffusion bridge architecture from LDDBM is adopted: modality-specific encoders $E_\mathcal{X}, E_\mathcal{Y}$ map $x \in \mathcal{X}, y \in \mathcal{Y}$ to a shared latent space, where the bridge is learned. The forward $\mathcal{X} \to \mathcal{Y}$ score is $s_{\mathcal{X}\to\mathcal{Y}}(z, t)$, and the reverse is $s_{\mathcal{Y}\to\mathcal{X}}(z, t)$. Each training step jointly optimizes a weighted sum of four objectives (Eq. 10): $\mathcal{L}_{total} = \mathcal{L}_{DSM} + \lambda_{end}\mathcal{L}_{cycle}^{end} + \lambda_{traj}\mathcal{L}_{cycle}^{traj} + \lambda_{pair}\mathbf{1}_{(x, y) \in \mathcal{D}_{pair}}\mathcal{L}_{pair}$, with all $\lambda = 1$. The fourth term is active only for the paired subset when $\rho > 0$; for $\rho = 0$, the method runs purely on heuristics.

### Key Designs

1. **Marginal Matching with Winner-Takes-All (WTA) Allocation**:

    - **Function**: Ensures the bridge's terminal state lands on the target marginal $p_\mathcal{X}$ in un/semi-paired settings, while mitigating the degenerate coupling issue where "any pairing can optimize DSM."
    - **Mechanism**: Independently sample $x \sim p_\mathcal{X}, y \sim p_\mathcal{Y}$; for each target $z_0 = E_\mathcal{X}(x)$, draw $K$ candidate conditions $\{y^{(k)}\}_{k=1}^K \sim p_\mathcal{Y}$, compute DSM loss $\mathcal{L}_{DSM} = \mathbb{E}\|s_\theta(z_t, t | y) - \nabla_{z_t}\log q(z_t | z_0)\|_2^2$, and backpropagate only for $k^\star = \arg\min_k \mathcal{L}_{DSM}(z_0, y^{(k)})$—selecting the condition best explained by the current bridge. To prevent "a few low-information $y$ being repeatedly chosen" (condition dominance), a capacity constraint $C_y = 2$ is imposed: each candidate $y^{(i)}$ can be selected at most twice per epoch.
    - **Design Motivation**: Random pairing in DSM at $\rho = 0$ can learn arbitrary mixed couplings (mode mixing); WTA is a classic optimization heuristic that narrows coupling ambiguity by selecting "locally most compatible" candidates. The capacity constraint prevents WTA from degenerating into "a few $y$ explaining all $x$." Notably, the authors clarify that WTA is not a guarantee of identifiability, but an optimization trick to reduce coupling ambiguity.

2. **Dual-Level Cycle Consistency (Endpoint + Trajectory)**:

    - **Function**: Constrains the bridge to be approximately invertible at both endpoints and along the entire trajectory, penalizing irreversible information loss patterns (mode dropping, arbitrary source/target mixing).
    - **Mechanism**: Let the forward stochastic flow be $\Phi_{\mathcal{X}\to\mathcal{Y}}$, reverse as $\Phi_{\mathcal{Y}\to\mathcal{X}}$. Endpoint-level: $\mathcal{L}_{cycle}^{end} = \mathbb{E}\|\hat z_0 - z_0\|_2^2$, where $\hat z_0 = \Phi_{\mathcal{Y}\to\mathcal{X}} \circ \Phi_{\mathcal{X}\to\mathcal{Y}}(z_0)$. Trajectory-level: for forward trajectory $\{z_t^{X\to Y}\}$ and reverse $\{z_{T-t}^{Y\to X}\}$, compute $\mathcal{L}_{cycle}^{traj} = \mathbb{E}[w(t)\|z_t^{X\to Y} - z_{T-t}^{Y\to X}\|_2^2]$, with weight $w(t) = 1/(\sigma_t^2 + \epsilon)$ to normalize for scale changes.
    - **Design Motivation**: CycleGAN's endpoint cycle is deterministic; diffusion bridges are stochastic, so endpoint cycle alone is insufficient. Trajectory-level cycle extends the constraint to the entire stochastic path, enforcing "the same tunnel is traversed in both directions," equivalent to a trajectory-level identifiability regularizer. The two-level combination constrains invertibility at both coarse (endpoint) and fine (path) granularity; empirically, the trajectory term boosts unpaired content accuracy from 16% to 87%.

3. **Unified Objective = Composable Heuristics + Paired as Optional**:

    - **Function**: Demotes paired supervision from "necessity" to one of four parallel heuristics, enabling the same codebase to train at any $\rho \in [0, 1]$ with graceful degradation.
    - **Mechanism**: $\mathcal{L}_{total} = \mathcal{L}_{DSM} + \lambda_{end}\mathcal{L}_{cycle}^{end} + \lambda_{traj}\mathcal{L}_{cycle}^{traj} + \lambda_{pair}\mathbf{1}_{(x, y) \in \mathcal{D}_{pair}}\mathcal{L}_{pair}$; the paired loss is activated only for the paired subset via an indicator. All $\lambda = 1$; no fine-tuned weighting (the authors observed no significant benefit from tuning). This "additive combination + indicator gating" is equivalent to imposing multiple soft constraint surfaces in $\mathcal{P}$, progressively shrinking the feasible set toward reversible, condition-preserving couplings.
    - **Design Motivation**: Geometrizing the training objective as a "union of heuristics" clarifies ablation—each term can be toggled to observe its effect (see Table 1 for the full ablation matrix). The indicator for the paired term allows the dataloader to avoid strict separation of paired/unpaired samples, simplifying engineering.

### Loss & Training
See the third key design above for details; specifically, $K$ WTA candidates, capacity $C_y = 2$, all $\lambda = 1$; bidirectional bridges are trained jointly (cycle terms required). For semi-paired settings, the size of the paired subset is determined by $\rho$, with the total number of endpoint samples fixed (only the label availability ratio changes).

## Key Experimental Results

### Main Results
FFHQ→CelebA-HQ Super-Resolution (Zero-shot SR, $\rho$ sweep):

| Method | $\rho=0$ | $\rho=0.5$ | $\rho=1.0$ |
|---|---|---|---|
| **SDB PSNR↑** | **19.0 ± 0.6** | **25.2 ± 0.3** | **25.9 ± 0.3** |
| DiWa PSNR | n/a | 22.6 ± 0.2 | 23.3 |
| LDDBM PSNR | n/a | 24.9 ± 0.3 | 25.6 ± 0.4 |
| SDB SSIM↑ | 0.54 | 0.68 | 0.69 |
| SDB LPIPS↓ | 0.37 | 0.32 | 0.31 |

On synthetic benchmarks, effect of structural constraints on coupling quality ($\rho=0$ slice):

| Method | SWD ↓ | MMD² ↓ | Content Acc. ↑ | Cycle MSE ↓ |
|---|---|---|---|---|
| Marginal matching only | 0.02021 | $-1.03\times10^{-4}$ | 0.162 | 0.972 |
| + Endpoint cycle | 0.01891 | $-1.69\times10^{-4}$ | 0.662 | 0.831 |
| + Trajectory cycle | 0.01968 | $-1.11\times10^{-4}$ | **0.868** | **0.680** |

### Ablation Study

| Config ($\rho$) | Key Change | Conclusion |
|---|---|---|
| MM only ($\rho=0$) | Content Acc 0.162 | Marginal matching aligns distributions but does not learn coupling |
| + Endpoint cycle | Acc 0.662 | Endpoint invertibility significantly restores semantic correspondence |
| + Trajectory cycle | Acc 0.868 | Trajectory constraint further compresses coupling ambiguity |
| Paired-only ($\rho=0.5$) | Acc 0.641 | Semi-paired paired loss is inferior to SDB's heuristic combination |
| **SDB Semi-paired ($\rho=0.5$)** | **Acc 0.955** | Heuristics + paired are maximally synergistic |
| Paired-only ($\rho=1.0$) | Acc 0.887 | Pure paired is inferior to structural constraints even with full pairing |
| **SDB ($\rho=1.0$)** | **Acc 0.965** | Full pairing + structural constraints yields further gains |

### Key Findings
- Semi-paired SDB at $\rho=0.5$ already matches or surpasses Paired-only ($\rho=1.0$), indicating that structural constraints indeed take over two of the three roles of paired data (correspondence/validity/invertibility).
- Even at $\rho=1$ (fully paired), SDB outperforms the pure paired baseline (PSNR 25.9 vs 25.6, Content Acc 0.965 vs 0.887), confirming that structural constraints are **complementary** rather than substitutes.
- At $\rho=0$, the purely heuristic PSNR of 19.0 is still meaningful (exceeding random baseline), marking the first time the LDDBM framework is trainable under zero-paired settings.
- In Multi-view→3D Voxel (ShapeNet) experiments, SDB outperforms EDM and LDDBM across $\rho \in \{0.5, 1.0\}$, and remains trainable at $\rho=0$ (baselines are unusable).

## Highlights & Insights
- **Reframing the training objective as a "combination of heuristics"**: Previous bridge methods treated the objective as a monolithic optimization; the authors decompose it into four independently togglable geometric constraints, making ablation geometrically interpretable—each disabled term corresponds to relaxing a type of invertibility.
- **Trajectory-level cycle consistency**: Enforcing trajectory consistency in a stochastic diffusion framework is much stricter than CycleGAN's endpoint cycle, equivalent to constraining the **symmetry of the entire SDE path**—a brilliant adaptation of OT/Schrödinger bridge invertibility intuition to DDBM.
- **WTA + capacity constraint**: A lightweight implementation for "selecting couplings without correspondence supervision," simpler than adversarial training or InfoNCE, nearly zero-cost, yet empirically yields large Content Acc gains.
- **Graceful degradation**: The unified objective works seamlessly across all three pairing budgets, making SDB a truly practical framework for real-world scenarios—many medical/scientific imaging tasks are partially paired, rarely at the 0/1 extremes.

## Limitations & Future Work
- All $\lambda=1$ appears elegant, but more challenging high-resolution/3D tasks may require tuning; the paper does not provide a systematic study of weighting.
- WTA candidate number $K$ and capacity $C_y=2$ are empirical values and sensitive to computational budget; adaptive $K$ is a natural extension.
- Cycle consistency requires joint training of bidirectional bridges, doubling parameter count and training cost.
- Evaluation is limited to image super-resolution and 3D voxel translation on established benchmarks; true "semantically unaligned" open-domain translation (e.g., sketch↔photo) remains to be validated.
- Connections to OT-style work could be further explored, e.g., aligning trajectory cycle with Schrödinger bridge marginals + cost structure for theoretical analysis.

## Related Work & Insights
- **vs LDDBM (Berman 2026)**: Direct foundation; SDB adds structural constraints and demotes paired supervision to optional.
- **vs DDBM (Zhou 2024)**: Uses Doob h-transform with implicit paired data constraints; SDB makes these constraints **explicit and decomposable**.
- **vs CycleGAN / CUT**: They use deterministic endpoint cycles; SDB generalizes cycle to stochastic diffusion trajectories.
- **vs UNSB (Schrödinger bridge adversarial version)**: UNSB uses GAN training, prone to mode collapse; SDB employs score matching + cycle, yielding more stable training.
- **vs LADB**: LADB reduces pairing needs by reusing pretrained source latent diffusion; SDB directly adds first-order heuristics as constraints within the bridge, orthogonal and composable.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending cycle consistency to stochastic diffusion trajectories is an elegant innovation; the overall framework is a beautiful "combination"
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + real SR + 3D voxel tasks + three $\rho$ regimes + complete ablation matrix
- Writing Quality: ⭐⭐⭐⭐⭐ The geometric framing of "translation = selecting a coupling in $\mathcal{P}$" is exceptionally clear
- Value: ⭐⭐⭐⭐ Unlocks zero/semi-paired LDDBM applications, highly valuable for domains with scarce paired data

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)
- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](../../ICLR2026/image_generation/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[ACL 2026\] Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding](../../ACL2026/image_generation/think_bright_diffuse_nice_enhancing_t2i-icl_via_inductive-bias_hint_instruction_.md)

</div>

<!-- RELATED:END -->
