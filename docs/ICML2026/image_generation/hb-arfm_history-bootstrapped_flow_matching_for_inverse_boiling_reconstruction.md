---
title: >-
  [Paper Note] (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction
description: >-
  [ICML2026][Image Generation][Flow Matching] HB-ARFM addresses the inverse reconstruction of multiphase boiling flow fields via "history-observation-guided" conditional flow matching. It first bootstraps an initial latent…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Flow Matching"
  - "Autoregression"
  - "Boiling Flow Fields"
  - "Partial Observation"
  - "Spatio-temporal Inverse Problem"
date: 2026-05-08
content_hash: 1167c0c182d8c03c
---

# (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction

**Conference**: ICML2026  
**arXiv**: [2606.00349](https://arxiv.org/abs/2606.00349)  
**Code**: TBD  
**Area**: Scientific Machine Learning / Inverse Problems / Flow Matching / Multiphase Flow  
**Keywords**: Flow Matching, Autoregression, Boiling Flow Fields, Partial Observation, Spatio-temporal Inverse Problem

## TL;DR
HB-ARFM addresses the inverse reconstruction of multiphase boiling flow fields via "history-observation-guided" conditional flow matching. It first bootstraps an initial latent state from a historical observation window and then autoregressively advances the reconstruction using the same conditional velocity field. This approach achieves the first consistent spatio-temporal reconstruction of complete temperature and velocity fields while observing only interface geometry and interface velocity.

## Background & Motivation
**Background**: Two-phase boiling is one of the most efficient forms of heat transfer, yet key variables such as temperature fields, velocity fields, and interphase mass transfer are nearly impossible to measure directly in experiments. Previous learning methods (e.g., neural operators, Bubbleformer, FFNO) mostly assume the availability of full simulation data for "forward prediction" or surrogate modeling.

**Limitations of Prior Work**: Applying these forward models to real-world scenarios with only images (interface segmentation + optical flow) immediately encounters the **cold start** problem—there is no ground-truth initial state to feed into the model. Generative inverse methods like DiffusionPDE or FunDPS only perform single-frame reconstruction and cannot guarantee temporal consistency; models like VE-SDE may even collapse into near-uniform fields.

**Key Challenge**: Although the two-phase Navier-Stokes equations are Markovian regarding the **complete state**, observing only partial variables such as the interface geometry $\phi$ introduces a **non-local memory kernel** into the effective dynamics of the observable variables, according to the Mori-Zwanzig (MZ) formalism. The influence of the "missing latent variables" in a single-moment observation must be recovered through **history**. In other words, partial observation transforms a Markovian forward problem into a non-Markovian inverse problem.

**Goal**: Reconstruct the complete temperature $\tau$ and velocity $\mathbf{u}$ fields for both liquid and gas phases given only (i) interface geometry $\phi$ and (ii) interface normal velocity $\mathbf{u}_\Gamma$, ensuring: observational consistency, physical plausibility, temporal coherence, and long-term rollout stability.

**Key Insight**: Since MZ tells us that the memory kernel originates from latent variables, a **finite-length observation history window** can be used to approximately recover this information—provided the history length covers characteristic time scales such as bubble rise and condensation times.

**Core Idea**: Use historical observations to bootstrap the first latent state, and then treat the same conditional flow matching model as an "autoregressive propagator with built-in data assimilation," where each step consumes both the current observation and the previous reconstruction result.

## Method

### Overall Architecture
The input is an observation sequence $y_{0:T}$ starting from $t=0$ with a window length $w$ (each frame consists of the interface $\phi$ in SDF form and the interface normal velocity $\mathbf{u}_\Gamma$); the output is the reconstructed complete temperature and velocity fields $\{\hat{X}_t\}$ for $t=w,\dots,T$. The pipeline consists of two stages sharing a single conditional velocity network $v_\theta$:

1. **Bootstrap Phase**: A history encoder $\zeta_\phi$ compresses $y_{0:w}$ into an initial state estimate $\hat{X}_w$, which is then fed into $v_\theta$ to generate the first frame $\hat{X}_w$ from Gaussian noise via flow matching.
2. **Autoregressive Phase**: For $t>w$, the previous prediction $\hat{X}_{t-1}$ and the current observation $y_t$ are concatenated into the condition $\mathbf{c}_t=[y_t,\hat{X}_{t-1}]$. The same $v_\theta$ is used to solve the ODE and sample $\hat{X}_t$, rolling forward in this manner.

The combination of the bootstrap path and the autoregressive feedback loop (AR + data assimilation) allows the model to unify "inverse reconstruction" and "data assimilation" within a conditional transport framework.

### Key Designs

1. **History Bootstrap Initialization (Solving Cold Start)**:
    - **Function**: In inverse problems where no ground-truth initial state is available, a historical observation window $y_{0:w}$ of length $w$ is used to generate the first-frame latent state estimate $\hat{X}_w$, replacing the ground-truth $X_0$ typically required by AR methods.
    - **Mechanism**: During training, two losses are jointly optimized: the regression loss of the history encoder $\mathcal{L}_{\text{boot}}=\|\zeta_\phi(y_{t_0-w:t_0-1})-X_{t_0}\|^2$, and the flow matching velocity field loss conditioned on $\mathbf{c}_{t_0}=[y_{t_0},\hat{X}_{t_0}]$, defined as $\|X_{t_0}-\mathbf{x}^0-v_\theta(\mathbf{x}^s,\mathbf{c}_{t_0},s)\|^2$, where $\mathbf{x}^s=(1-s)\mathbf{x}^0+sX_{t_0}$ is a linear interpolation on the OT path.
    - **Design Motivation**: The MZ formalism guarantees that the memory kernel decays exponentially beyond characteristic time scales (bubble rise and condensation times in boiling); thus, a **fixed-length** window is sufficient. This step relaxes the ill-posed mapping from "instantaneous observation $\rightarrow$ full state" into a well-constrained problem of "historical observation $\rightarrow$ full state."

2. **Shared Conditional Flow Matching Velocity Field + Autoregressive Propagation**:
    - **Function**: The same network $v_\theta(\mathbf{x}^s,\mathbf{c}_t,s)$ performs both bootstrapping and AR propagation. Each step absorbs new observations while assimilating the previous state, effectively performing **inverse reconstruction + data assimilation** in a unified framework.
    - **Mechanism**: In the AR phase, the condition is updated to $\mathbf{c}_t=[y_t,\hat{X}_{t-1}]$, and the flow matching objective becomes $\|X_{t_0+k}-\mathbf{x}^0-v_\theta(\mathbf{x}^s,\mathbf{c}_{t_0+k},s)\|^2$. During inference, a 4th-order Runge-Kutta method is used to solve $d\mathbf{x}_s/ds=v_\theta(\mathbf{x}_s,\mathbf{c}_t,s)$ to generate $\hat{X}_t$ from noise. Since bootstrap and AR share parameters, a single trajectory contributes both losses $\mathcal{L}=\mathcal{L}_{\text{boot}}+\mathcal{L}_{\text{AR}}/(K-1)$ during training, avoiding distribution shift caused by two-stage splicing.
    - **Design Motivation**: Schemes like HistoryFM that "sample each frame independently" do not guarantee that $\hat{X}_{t+1}$ is physically reachable from $\hat{X}_t$, leading to flickering wakes. By explicitly including the previous state in the condition, this model acts as an implicit data assimilator with "intrinsic physical manifold constraints."

3. **Factored Spatio-Temporal Transformer History Encoder**:
    - **Function**: Compresses $w$ observation frames $\{y_{t-w},\dots,y_{t-1}\}$ into the first-frame latent state estimate while avoiding the quadratic cost of joint space-time attention.
    - **Mechanism**: Each frame uses a $p\times p$ convolutional patch embedding to obtain $N=(H/p)(W/p)$ tokens, supplemented with 2D sinusoidal spatial positional encoding and learnable temporal positional encoding. $L$ layers alternate between (spatial self-attention within each frame) and (causal temporal self-attention across frames at each patch location). The final layer retains only the tokens of the last frame, which are restored to pixel space via linear unpatchify with a learnable scale/bias.
    - **Design Motivation**: Decoupling spatial and temporal attention allows each pixel to integrate the entire history at a cost of only $O(w+N^2)$. The causal temporal mask ensures no "peeking" into the future during bootstrap. Taking only the last frame's tokens aligns with the semantics of "nowcasting the current state using history."

### Loss & Training
The joint loss for a single trajectory is $\mathcal{L}=\mathcal{L}_{\text{boot}}+\mathcal{L}_{\text{AR}}/(K-1)$, where the bootstrap loss includes both history regression MSE and flow matching velocity field MSE, and the AR loss is the average of $K-1$ steps of conditional flow matching. A Residual U-Net is used for velocity field parameterization, with flow time $s\in[0,1]$ injected via sinusoidal embeddings into residual blocks. ODEs are solved using RK4. During training, the start point $t_0\sim\mathrm{Uniform}(w,T-K)$ is randomly sampled, and bootstrap and AR segments are jointly optimized on the same trajectory to force parameter sharing of $v_\theta$.

## Key Experimental Results

### Main Results
The dataset used is BubbleML, categorized into subcooled pool boiling and flow boiling settings. Two inverse tasks: (T1) reconstructing temperature from $\phi$ only, and (T2) jointly reconstructing temperature and velocity from $\phi+\mathbf{u}_\Gamma$. Baselines include DDPM, VE-SDE, standard FM, DiffusionPDE, PDEDiff, Bubbleformer, FFNO, UNet, and HistoryFM.

| Dimension | HB-ARFM Performance | Comparison Baselines | Key Observation |
|--------|------|------|----------|
| Near-interface Reconstruction | Sharp gradients + consistent wake | Most models perform decently | Strong geometric constraints near interface, low discriminability |
| Bulk Temperature/Velocity | Preserves fine scales + high-freq energy | FFNO/UNet over-smooth; VE-SDE collapses to near-uniform | The bulk region is the true test |
| HF Energy Ratio | Highest (comparable to HistoryFM) | DiffusionPDE/DDPM fragmentation | History conditioning preserves high frequencies |
| Wall Heat Flux Error | Lowest overall | — | Most critical engineering metric for boiling heat transfer |
| Long-term Rollout (300 steps) | Stable error, variance across seeds decreases over time | Bubbleformer/UNet collapse immediately; HistoryFM drifts; PDEDiff diverges | Both cold start and AR are indispensable |
| Flow Boiling Generalization | Velocity magnitude ratio $\approx 1$, wall heat flux error $< 0.2\%$ | — | Holds even in settings with completely different dynamics |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Full HB-ARFM | Consistent reconstruction + long-term stability | Joint optimization of bootstrap + AR |
| History window $w$ from 1 $\rightarrow$ 64 | Error decreases monotonically with $w$ | Velocity is more sensitive to history, aligning with MZ: memory kernels contribute more to hidden velocity fields |
| HistoryFM (Independent sampling, no AR) | Visuals look good per frame but wakes flicker | Proves AR feedback is key to temporal coherence |
| PDEDiff (Joint window + random masking) | Immediate divergence | Joint space-time modeling is unstable under multiphase/sharp interface conditions |
| Forward SOTA (Bubbleformer/UNet) | Fails at cold start | Proves history bootstrap is essential |

### Key Findings
- Larger history length $w$ leads to smaller reconstruction errors, with the **improvement in velocity significantly exceeding that in temperature**. This is consistent with MZ formalism: the velocity field is latent-dominated and requires longer memory kernels for recovery.
- While both HistoryFM and HB-ARFM utilize history, only HB-ARFM avoids wake "flickering." This indicates that **AR feedback explicitly encodes "physical reachability between adjacent frames" into the condition**, which is more critical than simply "seeing more history."
- In measurement-space consistency tests, the cosine similarity between the observed operator $H$ applied to predicted velocity and the input $y_{\text{obs}}$ remains $> 0.92$ after 300 rollout steps, proving the model does not generate self-contradictory fields.
- The model maintains $< 2\%$ wall heat flux error even at an OOD $T_{\text{wall}}=117$°C, indicating it has **learned the functional relationship "boundary conditions $\rightarrow$ phase-change heat transfer"** rather than simple interpolation.

## Highlights & Insights
- **Mori-Zwanzig Formalism as a Design Principle**: The authors do not add history based on intuition; instead, they use MZ to prove that "partial observation inevitably brings non-Markovianity" and justify the "finite window sufficiency" using the exponential decay of the MZ memory kernel. Theoretical motivation and engineering design are perfectly aligned.
- **Shared $v_\theta$ Merging Cold Start and Data Assimilation**: The bootstrap phase transforms a "no-initial-state" inverse problem into a "with-initial-state" problem, while the AR phase transforms "continuous observation" into implicit data assimilation. Both are accomplished within a single conditional velocity field, ensuring parameter efficiency and distributional alignment.
- **Boiling Flow as a SciML Inverse Problem Benchmark**: Characterized by sharp interfaces, multiphysics coupling, and hidden transport, this scenario simultaneously stress-tests generative, AR, and PINN methods. The paper effectively maps the failure modes of existing methods, providing high reference value for future work.

## Limitations & Future Work
- The model is entirely data-driven and **does not explicitly include conservation constraints** during training; divergence errors suggest that mass conservation is not perfect. Future work could incorporate projection or divergence penalties.
- The history encoder is a **lossy bottleneck**. Stronger history encoding, such as learnable memory kernels or state-space models (SSMs), presents an opportunity for improvement.
- All evaluations are on simulation data; the **sim-to-real gap remains an open question**: real boiling images contain noise and sensor drift, and there is no ground-truth field for supervision.
- Observation: While bootstrap and AR share the same $v_\theta$, their conditional structures differ significantly (one takes $\hat{X}_w=\zeta_\phi(y_{0:w})$, the other $\hat{X}_{t-1}$). Whether parameter sharing is optimal is worth further investigation—especially for long history windows where the bootstrap branch might require different capacity.
- In flow boiling tasks, temperature reconstruction remains relatively weak due to streamwise advection dominance. The model does not specifically handle long-range dependencies of streamwise advection; advection-aware modules could be considered.

## Related Work & Insights
- **vs. DiffusionPDE / FunDPS**: These methods add observational guidance during sampling for single-frame reconstruction. This work uses the conditional transport of flow matching to treat observations as explicit conditions and adds AR feedback to solve the "visually good but temporally disconnected" issue.
- **vs. S3GM**: S3GM jointly models the entire spatio-temporal volume $p(X_{0:T})$, requiring sampling in a massive space. This work **factorizes it over time** into conditional transport, offering better efficiency, scalability, and built-in AR data assimilation.
- **vs. HistoryFM**: Both use sliding window history, but HistoryFM samples each frame independently without feedback, which is equivalent to discarding the state constraint in MZ, leading to wake flickering. HB-ARFM includes the previous prediction in the condition, explicitly restoring the "trajectory continuity" constraint.
- **vs. Bubbleformer / FFNO / UNet**: These forward predictors assume a complete initial state and collapse immediately after a cold start. HB-ARFM uses history bootstrap for initialization while retaining the expressive diversity of generative models.

## Rating
- Novelty: ⭐⭐⭐⭐ Using MZ formalism as a design principle + sharing $v_\theta$ between bootstrap/AR is a fairly novel combination, though AR flow matching and history conditioning alone are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 2 tasks $\times$ 2 settings $\times$ 10 baselines, including history-length ablation, 300-step rollout stability, OOD boundary condition extrapolation, and measurement-space consistency.
- Writing Quality: ⭐⭐⭐⭐ Rigorous problem formulation (clear MZ argument), complete methodology, and pseudo-code. Minor drawback: figure references are less friendly for plain-text readers.
- Value: ⭐⭐⭐⭐⭐ First to achieve "reconstructing complete boiling thermal-fluid fields from imaging observations alone." This has direct industrial significance for thermal management, data center cooling, and safety (CHF estimation) and establishes boiling as a standard stress test for SciML inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICML 2026\] Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching](bootstrap_your_generator_unpaired_visual_editing_with_flow_matching.md)
- [\[ICML 2026\] Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)
- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)

</div>

<!-- RELATED:END -->
