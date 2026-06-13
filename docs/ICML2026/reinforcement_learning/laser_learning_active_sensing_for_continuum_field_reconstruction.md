---
title: >-
  [Paper Note] LASER: Learning Active Sensing for Continuum Field Reconstruction
description: >-
  [ICML 2026][Reinforcement Learning][Active Sensing] The problem of "where to place sparse sensors" is modeled as a POMDP. A "continuum field latent world model" consisting of an encoder, GRU, diffusion dynamics predictor…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Active Sensing"
  - "World Models"
  - "GRPO"
  - "POMDP"
  - "Continuum Field Reconstruction"
date: 2026-05-08
content_hash: 1bcdb619bb55065a
---

# LASER: Learning Active Sensing for Continuum Field Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2604.19355](https://arxiv.org/abs/2604.19355)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / World Models / Active Sensing  
**Keywords**: Active Sensing, World Models, GRPO, POMDP, Continuum Field Reconstruction

## TL;DR
The problem of "where to place sparse sensors" is modeled as a POMDP. A "continuum field latent world model" consisting of an encoder, GRU, diffusion dynamics predictor, and implicit neural field decoder provides imagined future latent states as policy conditions. The policy is trained using GRPO with dynamic group filtering and multi-step lookahead rewards, consistently outperforming fixed and offline-optimized layouts on Navier-Stokes, Shallow-Water, and Sea Surface Temperature (SST) datasets.

## Background & Motivation

**Background**: Recovering continuous physical fields (turbulence, stress fields, temperature fields, etc.) from sparse discrete sensor measurements is a core problem in scientific computing and engineering. Recent mainstream approaches use neural operators, INRs, or transformer operators for reconstruction, treating sensor positions either as fixed inputs (AROMA, DiffusionPDE) or as **globally static** layouts generated via offline optimization (PhySense).

**Limitations of Prior Work**: Fixed or globally optimized layouts ignore the **non-stationary** nature of physical fields—the information content of a specific set of sensor locations varies significantly across different time steps and initial conditions. Literature reports that reconstruction accuracy can vary by several factors with different layouts. However, current research lacks instance-specific sensor adaptation in a closed-loop.

**Key Challenge**: To enable online adaptive sensor positioning, an environment model capable of "what-if" simulation is required. One needs to know "how reconstruction error changes in the next step if sensors move 0.1 north now," but real physical systems cannot be repeatedly rolled out. Additionally, active sensing involves high-dimensional continuous action spaces and sparse, delayed feedback, making standard RL unstable.

**Goal**: (i) Construct a latent world model capable of forward prediction and reconstruction reward calculation as a differentiable environment proxy; (ii) Train an RL policy to **proactively** determine sensor displacements within this latent space; (iii) Stabilize RL training under sparse rewards.

**Key Insight**: Leveraging the World Model paradigm (Ha & Schmidhuber 2018)—decoupling "environment simulation" from "planning in latent imagination." However, a world model for continuum fields must handle **arbitrary numbers and positions** of sparse observations, perform forward rolls, and output continuous fields as differentiable reward sources. On the policy side, group-relative advantage estimation from the DeepSeek-R1 series (GRPO) is adapted for continuous control.

**Core Idea**: Use "world-model imagined next-step latent states" as the query context for the policy, making sensor decisions **proactive** rather than reactive, while stabilizing training with GRPO and dynamic filtering.

## Method

### Overall Architecture
LASER models active sensing as a POMDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{E},\mathcal{T}_\phi,\mathcal{R}_\phi,\gamma)$, where the latent state $\bm s_t=[\bm z_t,\bm h_t]$ consists of the current observation latent code $\bm z_t$ and GRU history $\bm h_t$. The action $\bm a_t=\Delta\bm X_t$ represents sensor displacement, and the reward $r_t=-\mathcal{L}(\bm u_{t+1},\hat{\bm u}_{t+1})$ is the negative MSE of the reconstruction decoded by the world model. Training consists of two stages: (1) **Offline** pre-training of the world model $\phi$ (encoder/dynamics/decoder joint ELBO + diffusion denoising), where sensor layouts are randomly re-sampled at each step to learn invariance; (2) **Online** policy $\pi_\theta$ training using GRPO, sampling $G$ action groups from the current $\hat{\bm z}_{t+1}$ and $\bm o_t$. The environment provides rewards based on ground truth from the training dataset, eliminating the need for a real-time physical simulator.

### Key Designs

1.  **Continuum Field Latent World Model**:
    - **Function**: Acts as a high-fidelity differentiable proxy for the physical environment while outputting forward-predicted latents and reconstruction-based rewards.
    - **Mechanism**: Three modules integrated as $p_\phi^{enc}\to p_\phi^{dyn}\to p_\phi^{dec}$. The encoder uses $M$ learnable latent queries to perform cross-attention on $(\bm x_t^{(i)},\bm u_t(\bm x_t^{(i)}))$ to obtain $\bm z_t\sim\mathcal{N}(\bm\mu_\phi, \bm\sigma_\phi^2)$, which is naturally permutation-invariant and accommodates any number of sensors/positions. The dynamics predictor is a **conditional diffusion** model conditioned on $\bm z_t$ and GRU history $\bm h_t=\mathrm{GRU}_\phi(\bm h_{t-1}, \bm z_t)$, performing $K$-step denoising on $\tilde{\bm z}_{t+1}$ to output $\hat{\bm z}_{t+1}$. The decoder is an Implicit Neural Field (INR), taking $(\bm z_t, \bm x)$ as input to output field values $\hat{\bm u}_t(\bm x)$ at any spatial coordinate. Training objective: $\mathcal{L}_{world}=\mathcal{L}_{recon}+\beta\mathcal{D}_{KL}+\lambda\mathcal{L}_{diffusion}$.
    - **Design Motivation**: (i) Randomly re-sampling sensor positions during encoder training forces the world model to learn layout-invariant representations; (ii) Using diffusion rather than deterministic MLPs for dynamics captures multi-modal future distributions of turbulent/non-stationary fields; (iii) The INR decoder allows reward calculation over the **entire** $\Omega$ rather than just grid points, providing continuous differentiable feedback.

2.  **Proactive Policy and Multi-scale Cross-Attention**:
    - **Function**: Decides continuous displacement for each sensor using the "imagined next-step latent state" as context.
    - **Mechanism**: The policy $\pi_\theta(\bm a_t|\hat{\bm z}_{t+1}, \bm o_t)$ is a Transformer. Sensor-side queries are formed by concatenating position and value embeddings $\mathbf q^{(i)}=[\gamma_{pos}(\bm x_t^{(i)});\text{Embed}(\bm u_t(\bm x_t^{(i)}))]$, where $\gamma_{pos}$ uses multi-scale Fourier features $\gamma^s(\bm x)=[\sin(\bm x\bm\omega^s), \cos(\bm x\bm\omega^s)]$. Queries and imagined latents $\hat{\bm z}_{t+1}$ interact via **multi-scale** cross-attention $\mathbf f=\bigoplus_{s}\text{softmax}(\mathbf q^{(s)}(\mathbf k^{(s)})^\top/\sqrt{c_s})\mathbf v^{(s)}$, followed by self-attention for sensor coordination. Finally, an MLP head outputs Gaussian displacements $(\bm\mu_\theta^{(i)}, \log\bm\sigma_\theta^{(i)})$, with actions clipped to $[-a_{max}, a_{max}]$.
    - **Design Motivation**: Sensor decisions must anticipate rather than react—by using the "imagined" $\hat{\bm z}_{t+1}$ as key/value, the policy evaluates the optimal movement for future field states. Multi-scale Fourier features and attention capture both local details and global structures, while self-attention prevents sensors from clustering in the same region.

3.  **GRPO Training: Dynamic Group Filtering + Multi-step Lookahead Reward**:
    - **Function**: Stabilizes policy training under sparse rewards and high-dimensional continuous action spaces.
    - **Mechanism**: For each $t$, $G$ action groups are sampled to obtain rewards $\{r_t^g\}$. Group-relative advantages $A_{g,t}=(r_t^g-\text{mean})/\text{std}$ are further normalized within the batch as $\hat A_{g,t}$. The objective $\mathcal{J}_{GRPO}=\mathbb E[\min(s_{g,t}(\theta)\hat A_{g,t}, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat A_{g,t})]$ follows the PPO clip. Two key augmentations: (a) **Dynamic Group Filtering**—maintaining a running mean $\tau$ of $\min_g r_t^g$ and discarding low-quality samples where the entire group reward $<\tau$ (e.g., sensors clustering in low-variance areas); (b) **Multi-step Lookahead Reward**—freezing the layout after executing $\bm a_t$ and performing an autoregressive rollout of $p_\phi^{dyn}$ for $H=3$ steps, aggregating discounted rewards $r_t^{look}=\sum_{h=1}^H\gamma^{h-1}r_{t+h}/\sum\gamma^{h-1}$.
    - **Design Motivation**: (a) In active sensing, many action configurations are systematically uninformative; filtering prevents them from polluting advantage estimation. (b) Single-step rewards encourage short-sighted layouts; lookahead rewards link decisions to future reconstruction quality, crucial for rapidly evolving turbulent scenarios.

### Loss & Training
The world model is pre-trained offline with $\mathcal{L}_{world}$ and frozen. The policy is trained online via GRPO. Parameters such as $H=3$, diffusion denoising steps $K$, $G$, $\epsilon$, $\beta$, and $\lambda$ are provided in the appendix. Each episode randomly selects trajectories and start times $t_0$, initializing sensors in a uniform distribution to prevent overfitting to initial conditions.

## Key Experimental Results

### Main Results
Benchmarks include NS-1e-3 / NS-1e-5 (2D Navier-Stokes in vorticity form), Shallow-Water (3D Shallow Water equations), and SST (Real Sea Surface Temperature). Reconstruction error $\mathrm{MSE}_{recon}$ ($\times 10^{-3}$) is shown below (Avg is the average of In-time + Out-time):

| #Obs | Dataset | AROMA (Fixed) | DiffusionPDE | PhySense (Offline Opt) | LASER-PPO | **Ours (LASER)** |
|------|---------|---------------|--------------|-------------------------|-----------|------------------|
| 256  | NS-1e-3 | 2.720         | 1.344        | 0.376                   | 0.304     | **0.302**        |
| 128  | NS-1e-3 | 5.816         | 6.609        | 0.370                   | 0.353     | **0.321**        |
| 64   | NS-1e-3 | 20.27         | 6.543        | 0.466                   | 0.396     | **0.434**        |
| 256  | Shallow | 12.59         | 3.175        | 0.355                   | 0.326     | **0.257**        |
| 100  | SST     | 1.0586        | 3.4626       | 0.7059                  | —         | **0.6932**       |

LASER achieves the lowest error in 11/12 (dataset × #Obs) combinations. The relative gain over fixed layouts increases as sensing becomes sparser (47x improvement on NS-1e-3 @64). Comparison with LASER-PPO confirms that GRPO + dynamic filtering further reduces error.

### Ablation Study

| Configuration | NS-1e-3 @256 Avg | Description |
|---------------|------------------|-------------|
| LASER (Full)  | 0.302            | Complete model |
| LASER† (w/o Dynamic Filtering) | 0.391 | Out-time error 0.685 vs 0.483; filtering critical for long-range prediction |
| LASER($\phi$) (World Model Only) | 0.359 | Active sensing provides ~16% Gain |
| Lookahead $H=1$ | Out-t 0.6136     | Single-step reward is short-sighted |
| Lookahead $H=5$ | Out-t 0.3380     | Higher $H$ improves Out-time performance (~45% gain) |

GRU history ablation (Table 6): Stronger turbulence (NS-1e-5, Shallow-Water) favors shorter history (3 steps), as excessive context introduces outdated information.

### Key Findings
- **Active Sensing > Offline Optimized Layouts**: Even against the strongest offline baseline (PhySense), LASER consistently wins, demonstrating that instance-specific adaptation is irreplaceable.
- **Lookahead Reward is Critical for Out-time Generalization**: Increasing $H=1\to 5$ cuts Out-time error by 45% while In-time error remains stable, suggesting lookahead addresses future step prediction outside the training distribution.
- **Sparsity Magnifies Adaptive Advantages**: AROMA's performance degrades by over 10x when reducing sensors from $N=256$ to $N=64$, whereas LASER degrades by less than 2x.
- **GRPO Outperforms PPO**: LASER consistently outperforms LASER-PPO across all datasets, confirming the effectiveness of group-relative advantage and dynamic filtering.

## Highlights & Insights
- **World Model as Differentiable Environment**: This concept is highly suitable for scientific computing where real physical simulators are expensive or non-differentiable. The world model enables both forward prediction and gradient-based rewards.
- **Proactive Paradigm**: Using $\hat{\bm z}_{t+1}$ (future latent) instead of $\bm z_t$ (current latent) as the policy condition is a simple yet profound choice that allows the policy's cognition to stay "one step ahead."
- **GRPO for Continuous Control**: While most GRPO work focuses on LLM reasoning (discrete tokens), this work adapts it for continuous high-dimensional actions with a dynamic group filtering trick.
- **Multi-scale Physics Modeling**: The use of multi-scale Fourier encoding and attention aligns with the multi-scale structures inherent in physical fields (e.g., large and small eddies in turbulence).

## Limitations & Future Work
- The world model is **frozen** after pre-training; if the policy discovers layouts outside the training distribution, the world model might become unreliable (model exploitation risk).
- Experiments are restricted to simulated datasets and historical SST data, lacking **real-world closed-loop hardware experiments** with movement latency and noise.
- Rollouts of $H=3$ could benefit from larger $H$ for Out-time performance, but diffusion dynamics steps are computationally expensive.
- Cross-physical domain transfer (e.g., training on turbulence and transferring to temperature fields) remains unvalidated.

## Related Work & Insights
- **vs AROMA (Serrano et al., 2024)**: AROMA serves as the backbone for the LASER encoder but uses fixed sensor positions. LASER promotes "position" to a controllable action.
- **vs DiffusionPDE (Huang et al., 2024)**: DiffusionPDE performs sampling at test time with high overhead; LASER limits diffusion to latent space dynamics for efficiency and autoregressive rollouts.
- **vs PhySense (Ma et al., 2025)**: PhySense optimizes globally static layouts; LASER reduces average error by 30%+ in the sparsest settings, proving the value of instance-specific adaptation.
- **vs DreamerV3**: While sharing the latent imagination paradigm, LASER adapts it for scientific ML with PDE continuum fields, Transformer encoders, and diffusion dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combines world-model RL, GRPO, and multi-scale scientific ML encoders for active sensing.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive datasets and ablations; missing hardware validation.)
- Writing Quality: ⭐⭐⭐⭐ (Clear POMDP formulation and topology diagrams.)
- Value: ⭐⭐⭐⭐ (Provides a clear "active sensing = world-model RL" paradigm for the scientific ML community.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)
- [\[ICLR 2026\] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](../../ICLR2026/reinforcement_learning/cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)
- [\[ICML 2026\] Mind Dreamer: Untethering Imagination via Active Causal Intervention on Latent Manifolds](mind_dreamer_untethering_imagination_via_active_causal_intervention_on_latent_ma.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
