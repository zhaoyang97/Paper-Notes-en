---
title: >-
  [Paper Note] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL
description: >-
  [ICML 2026][Reinforcement Learning][Offline GCRL] QHyer replaces the trajectory-dependent RTG in Decision Transformer with state-dependent Q-values estimated by Normalizing Flows…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline GCRL"
  - "Decision Transformer"
  - "Normalizing Flows"
  - "Mamba"
  - "Trajectory Stitching"
date: 2026-05-08
content_hash: 31073f0d916e7aa7
---

# QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL

**Conference**: ICML 2026  
**arXiv**: [2605.01862](https://arxiv.org/abs/2605.01862)  
**Code**: Not released  
**Area**: Reinforcement Learning / Sequence Modeling / Offline Goal-conditioned RL  
**Keywords**: Offline GCRL, Decision Transformer, Normalizing Flows, Mamba, Trajectory Stitching

## TL;DR
QHyer replaces the trajectory-dependent RTG in Decision Transformer with state-dependent Q-values estimated by Normalizing Flows, and stacks a gated Hybrid Attention-Mamba backbone to achieve content-adaptive historical compression. It sets new SOTA on both non-Markovian and Markovian offline goal-conditioned RL datasets (OGBench/D4RL).

## Background & Motivation

**Background**: Offline goal-conditioned reinforcement learning (Offline GCRL) learns "goal-reaching" policies from static datasets. Two main approaches: value-based methods using Bellman backup (IQL/HIQL, etc.) and Decision Transformer (DT) series that treat decision-making as sequence modeling. The latter naturally handles historical dependencies, making it more suitable for real-world datasets with non-Markovian behavior policies (e.g., OGBench play).

**Limitations of Prior Work**: Directly applying DT to Offline GCRL faces two major obstacles. First, DT uses RTG (Return-to-Go) as the conditioning signal, but under sparse goal rewards, RTG degenerates into a near-binary signal indicating only "trajectory success"—the same state gets 1 in successful trajectories and 0 in failures, making it **impossible to compare state quality across trajectories**. As a result, "locally useful segments" from failed demonstrations cannot be stitched into new policies, collapsing the stitching capability. Second, pure attention is insensitive to temporal structure; LSDT/DMixer use fixed-window causal convolutions to supplement "local branching," but play data requires long memory, noisy data only needs short memory, and fixed receptive fields either waste capacity or truncate key dependencies.

**Key Challenge**: These two limitations are **coupled**. Simply replacing RTG with Q-values while keeping the RTG-style fixed window still suffers from convolutional issues on non-Markovian play; only changing the backbone but keeping RTG does not solve the stitching bottleneck under sparse rewards. Both must be addressed—requiring both "state-dependent value signals" and "content-adaptive effective memory."

**Goal**: (i) Find a conditioning signal for DT that can distinguish state quality under sparse goal rewards; (ii) Design a temporal module for the backbone that can dynamically adjust memory length per token.

**Key Insight**: The authors observe that the goal-reaching Q-function $Q^\beta(s,a,g)=p^\beta_+(g\mid s,a)$ represents the probability of reaching goal $g$ from $(s,a)$, **independent of trajectory**—precisely the "trajectory-independent value metric" needed for stitching. Meanwhile, Mamba's selective SSM makes the discretization step $\Delta_t$ an input-dependent function, allowing effective memory to drift per token without changing the structure. These two observations directly address the two limitations.

**Core Idea**: **Use Normalizing Flows to estimate MC Q-value as the conditioning token to replace RTG**, and **replace pure attention with a hybrid backbone that learns gated fusion of Attention+Mamba**, enabling sequence modeling to truly fit Offline GCRL.

## Method

### Overall Architecture
QHyer represents each timestep as a $(Q_t, [s_t;g], a_t)$ triplet: $Q_t=\log p_\theta(g\mid s_t,a_t)$ is the log-probability of "reaching the goal" given by NFs, $[s_t;g]$ is the state-goal concatenation token (ensuring goal signal is visible at every step without increasing sequence length from $3T$ to $4T$). This sequence is fed into $L$ layers of Hybrid Attention-Mamba blocks, each with two parallel branches (attention for global goal planning, Mamba for temporal compression), with outputs fused by a scalar gate $\alpha=\sigma(\mathbf{w}^\top x + b)$. Training jointly optimizes NFs likelihood, Q expectile regression, and behavior cloning. Inference is two-stage autoregressive: first predict the maximal Q, then generate the action conditioned on the maximal Q.

### Key Designs

1. **NFs-based Q-value replaces RTG**:

    - **Function**: Provides DT with a trajectory-independent state-action-goal value signal, enabling the model to find "high-Q segments" in failed demonstrations for stitching.
    - **Mechanism**: Uses coupling-layer NFs to model the conditional density $p_\theta(g\mid s,a)$, obtaining exact log-likelihood $Q^\beta_\theta(s,a,g)=\log p_0(f_\theta(g;z))+\log\bigl|\det\partial f_\theta(g;z)/\partial g\bigr|$ via invertible mapping $f_\theta(\cdot;z)$ and change-of-variable formula. Expectile regression $L^2_\tau(u)=|\tau-\mathds{1}(u<0)|\cdot u^2$ ($\tau\in(0.5,1)$) is used to train the transformer's own $\hat Q_\phi(s,g)$ from behavior $Q^\beta$, converging to the **in-distribution maximal Q** (Theorem 3.1 shows bias $\epsilon_\tau$ decreases as $\tau$ increases).
    - **Design Motivation**: The authors argue why CVAE (only provides ELBO lower bound), Contrastive RL (density ratio has goal-dependent bias), and Diffusion (likelihood requires ODE+Hutchinson estimation, introducing variance) are unsuitable—they are either unnormalized or distort the "Q-token sequence across multiple goals." NFs' triangular Jacobian enables **precise and efficient** log-density, exactly what transformer cross-goal conditioning requires; empirical results show NFs have the lowest estimation error (Appendix G.4). RTG covers only 25% under sparse rewards, while NFs Q-value conditioning reaches 92%.

2. **Hybrid Attention-Mamba Backbone**:

    - **Function**: Uses an attention branch for global goal-directed reasoning and a Mamba branch for content-adaptive historical compression, with learnable gating for fusion.
    - **Mechanism**: The Mamba branch uses causal convolution to extract local features $x'_t$, then passes through selective SSM $h_t=\bar A h_{t-1}+\bar B x'_t,\ y_t=Ch_t$, where $\bar A_t=\exp(\Delta_t\cdot A)$ and $\Delta_t=\mathrm{softplus}(\mathrm{Linear}_\Delta(x'_t))$. When $\Delta_t$ is small, $\bar A_t\approx 1$ retains long history (suitable for play); when $\Delta_t$ is large, $\bar A_t\approx 0$ focuses on local (suitable for noisy). The gate $\alpha=\sigma(\mathbf w^\top x+b)$ dynamically allocates capacity between the two branches.
    - **Design Motivation**: LSDT/DMixer's convolutional local branch is limited by fixed kernels—convolution's effect for $j<k$ is fixed weight $w_j$, and is hard-truncated beyond that; Mamba provides "input-dependent smooth forgetting," automatically adjusting effective memory across datasets without manual tuning, which fixed-window structures cannot achieve.

3. **Concatenated State-Goal Token + End-to-End Triple Loss**:

    - **Function**: Embeds goal information into each timestep token, keeping sequence length at $3T$ and avoiding the quadratic attention cost of extra tokens.
    - **Mechanism**: Each step's token sequence is $(Q_t,[s_t;g],a_t)$ instead of $(Q_t,s_t,g,a_t)$. Training loss $\mathcal L_{\text{QHyer}}=\lambda_{\text{critic}}\mathcal L_{\text{NFs}}+\lambda_{\text{BC}}\mathcal L_{\text{BC}}+\lambda_Q \mathcal L_Q$ corresponds to NFs maximum likelihood, Q-conditioned behavior cloning, and transformer-side Q expectile regression.
    - **Design Motivation**: Concatenation rather than separation maintains goal visibility while controlling computational cost, a key engineering trick for seamlessly integrating NFs Q signals into the DT pipeline.

### Loss & Training

NFs are trained with hindsight relabeling and $-\log p_\theta(g\mid s_t,a_t)$ for maximum likelihood; transformer-side BC loss is $\mathcal L_{\text{BC}}=-\mathbb E[\log\pi_\theta(a_t\mid Q_t,[s_t;g])]$; expectile $\tau=0.9$ is used for low-coverage play, $\tau=0.95$ for high-coverage noisy data. Inference is two-stage: first generate $\hat Q(s_t,g)$, then generate $a_t$ conditioned on it.

## Key Experimental Results

### Main Results
OGBench manipulation (5 test goals, average success rate %) and D4RL Maze (normalized score).

| Dataset | Task | Prev. SOTA | Ours (QHyer) | Gain |
|---|---|---|---|---|
| OGBench cube-play | single | GCIQL 68 | **84** | +16 |
| OGBench cube-play | double | GCIQL 40 | **56** | +16 |
| OGBench cube-noisy | double | GCIQL 23 | **30** | +7 |
| OGBench puzzle-play | 4x5 | GCIQL 14 | **31** | +17 |
| D4RL AntMaze-v2 | large-play | IQL 39.6 | **44.2** | +4.6 |
| D4RL AntMaze-v2 | medium-diverse | LSDT 75.8 | **94.0** | +18.2 |
| D4RL Maze2d | medium | QT 172.0 | **173.0** | +1.0 |

Total: OGBench cube-play 24→152 (HIQL baseline), AntMaze total 303.6→483.4, Maze2d total 136.5→291.5. On large mazes where RTG-based methods (DT/EDT/DC) nearly collapse, QHyer breaks through directly.

### Ablation Study

| Configuration | cube-single-play | cube-single-noisy | Conclusion |
|---|---|---|---|
| RTG + Attention (≈DT) | Low | Low | RTG fails |
| NFs Q + Attention only | 74 | 60 | Lacks temporal adaptivity |
| NFs Q + Mamba only | 80 | 91 | Lacks global reasoning |
| NFs Q + Hybrid (QHyer) | **84** | **95** | Complementary gating |
| Hybrid + No Q | -- | -- | Degrades to BC |
| Hybrid + CVAE Q | < CRL | < CRL | ELBO lower bound distortion |
| Hybrid + CRL Q | < NFs | < NFs | Negative sampling bias |

Expectile $\tau$ increases monotonically from 0.5 to optimal at 0.9; above 0.95, performance degrades due to insufficient coverage.

### Key Findings
- **Both innovations are necessary and optimal in combination**: Independent ablations (fixing NFs and changing backbone, fixing RTG and changing backbone, fixing backbone and changing Q estimator) all show that QHyer's two modifications are additive, not redundant.
- **Mamba's $\Delta_t$ truly "drifts with data shape"**: On play, mean $\Delta_t=0.38$, $\bar A_t=0.92$, effective memory about 12 steps, gate allocates 0.57 capacity to attention; on noisy, $\Delta_t=1.05$, $\bar A_t=0.61$, effective memory about 3 steps, gate allocates 0.58 to Mamba.
- **NFs > CRL > CVAE > No-Q**: Accurate normalized log-density is the key bottleneck for sequence modeling stitching.

## Highlights & Insights
- **The argument that "the two limitations are coupled" is very clear**: The authors explicitly point out the failure modes of only addressing one side (retaining convolution's "non-Markovian disease" or RTG's "trajectory dependence bottleneck"), providing strong motivation for simultaneous changes—more convincing than the common "we added A and also B" narrative.
- **"Structural argument" for NFs selection**: The analysis of why CVAE/CRL/Diffusion are unsuitable is elevated to the specific scenario of transformer cross-goal Q-token reading and the need for normalization, offering design principles beyond experimental numbers—this kind of analysis on when density model properties are decisive is highly transferable.
- **Gating + Mamba adaptive Δ as "dual-level adaptivity"**: Coarse-grained gating allocates capacity between branches, fine-grained $\Delta_t$ adjusts memory length per token. This "hierarchical adaptivity" is a good paradigm for heterogeneous temporal structure datasets, transferable to robotics multi-tasking, dialogue history compression, etc.

## Limitations & Future Work
- Still limited on visual-noisy: pixel-level NFs density estimation is the main error source, and Markovian behavior offsets the non-Markovian modeling advantage.
- Theoretical analysis is based on deterministic transition assumption (inherited from R2CSL); extension to stochastic environments remains open.
- Training cost is higher than pure DT: NFs critic + Mamba SSM + expectile, with three components stacked; the paper does not provide detailed wall-clock comparison.
- Expectile $\tau$ is strongly coupled with coverage $\tilde c$, still requiring manual selection of $\tau\in\{0.9,0.95\}$ across datasets.

## Related Work & Insights
- **vs DT/EDT/DC/DMamba**: All use RTG as conditioning, which degenerates to binary signals under sparse goal rewards; QHyer replaces RTG with NFs Q, fundamentally improving stitching.
- **vs QDT/CGDT/QT/Reinformer/VDT**: Still retain RTG, using Q as auxiliary loss or regularization; QHyer directly replaces RTG with Q-token, more thorough for sparse goal rewards.
- **vs LSDT/DMixer**: Use fixed-kernel convolution for local supplementation, constrained by receptive field; QHyer uses Mamba selective SSM for "content-adaptive" memory, requiring no manual tuning across play/noisy.
- **vs HIQL/SAW/OTA**: Hierarchical methods assume Markovian transitions between subgoals, which do not hold for play data; QHyer's direct sequence modeling naturally handles non-Markovian cases.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use NFs Q + Hybrid Attention-Mamba for Offline GCRL, with a thorough argument for the coupling of the two limitations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual benchmarks (OGBench + D4RL), ablations for 3 Q estimators, 3 backbones, $\tau$ sensitivity, $\Delta_t$/gate weight visualization, closed-loop validation of both innovations.
- Writing Quality: ⭐⭐⭐⭐⭐ Stepwise reasoning from limitation → root cause → choice, textbook-level comparative argument for NFs selection.
- Value: ⭐⭐⭐⭐ Provides a viable "sequence modeling + precise density Q" route for Offline GCRL, directly transferable to robotics, long-horizon navigation, and other downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] InFOM: Intention-Conditioned Flow Occupancy Models](../../ICLR2026/reinforcement_learning/infom_intention_flow_occupancy.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)

</div>

<!-- RELATED:END -->
