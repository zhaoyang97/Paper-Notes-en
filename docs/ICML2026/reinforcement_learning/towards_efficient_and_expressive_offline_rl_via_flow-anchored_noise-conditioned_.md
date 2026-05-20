---
title: >-
  [Paper Note] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline RL] This paper proposes FAN: compressing "expensive generative policy + distributional critic" into "single-step flow anchoring + single noise-sample critic"—using Flow Anchori…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Flow Matching Policy"
  - "Distributional Critic"
  - "Noise-conditioned Q-learning"
  - "Behavior Regularization"
date: 2026-05-08
content_hash: 16186feafb11c04f
---

# Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning

**Conference**: ICML 2026  
**arXiv**: [2605.01663](https://arxiv.org/abs/2605.01663)  
**Code**: https://github.com/brianlsy98/FAN (available)  
**Area**: Reinforcement Learning / Offline RL / Generative Policy  
**Keywords**: Offline RL, Flow Matching Policy, Distributional Critic, Noise-conditioned Q-learning, Behavior Regularization

## TL;DR
This paper proposes FAN: compressing "expensive generative policy + distributional critic" into "single-step flow anchoring + single noise-sample critic"—using Flow Anchoring to complete behavior regularization within one flow evaluation, and replacing quantile multi-sample with a single Gaussian noise sample in the noise-conditioned critic. Achieves SOTA performance on D4RL/OGBench while training 5-14× faster than comparable distributional methods.

## Background & Motivation

**Background**: The core challenge in offline RL is to constrain the policy within the dataset’s behavior distribution to avoid OOD overestimation. Two highly expressive tools are widely adopted recently: (1) **Flow/diffusion policies** use flow matching to model multi-modal behavior distributions, offering stronger expressiveness than Gaussian policies (e.g., FQL, IDQL, Diffusion-QL); (2) **Distributional critics** learn the entire return distribution via quantile mechanisms rather than just the expectation (e.g., IQN, CODAC, Value Flows). Combining both achieves SOTA, but at high computational cost.

**Limitations of Prior Work**: (i) Flow policies require solving an ODE for each action generated, so 10 steps = 10× single-step forward cost; using flow for behavior regularization during training (e.g., FQL’s $\mathcal{L}_P$) requires solving the ODE to obtain $a_\theta$ before computing $\|a_\omega-a_\theta\|^2$, multiplying the flow steps into training cost. (ii) Distributional critics typically compute loss over 16-32 quantiles, and additional max-over-samples steps for ess sup further increase computation and variance.

**Key Challenge**: Expressiveness (multi-modal behavior + full return distribution) and efficiency (single forward pass + single-sample estimation) are inherently at odds; prior work sacrifices several times to an order of magnitude in training/inference speed for expressiveness.

**Goal**: Retain the expressiveness of flow policy + distributional critic while answering two technical questions—(1) Can flow policy perform behavior regularization with just a single iteration? (2) Can distributional critic be trained with only a single Gaussian noise sample?

**Key Insight**: Behavior regularization essentially requires the policy distribution to match the behavior distribution, not necessarily sampling real behavior actions—an equivalent goal is to constrain the policy to "lie on the velocity field trajectory of the behavior flow," which only needs single-step flow evaluation. Similarly, distributional information can be encoded with a continuous noise variable $\epsilon$ (instead of discrete quantile $\tau$), so the critic as $Q(s,a,\epsilon)$ can be learned with a single noise sample.

**Core Idea**: Replace ODE solving with Flow Anchoring—using the flow matching loss $\|(\pi_\omega(s,\epsilon)-\epsilon)-v_\theta(s,t,a_{t,\omega})\|^2$ to constrain the "displacement" of the one-step policy by the behavior flow’s velocity field; use a noise-conditioned critic + upper expectile regression to compress distributional information into a single Gaussian noise sample, with $\kappa\approx 1$ asymmetric expectile estimating $\mathrm{ess\,sup}$.

## Method

### Overall Architecture
FAN is a behavior-regularized actor-critic framework with four networks:

- One-step policy $\pi_\omega(s,\epsilon)$: takes state and noise as input, outputs action directly;
- Behavior flow policy $v_\theta(s,t,a_t)$: uses flow matching to fit the dataset $(s,a)$ distribution;
- Noise-conditioned critic $Q_\phi(s,a,\epsilon)$: evaluates Q for a single Gaussian noise sample;
- Upper quantile estimator $Z_\psi(s,a)$: uses expectile regression with $\kappa=0.9$ to estimate $\mathrm{ess\,sup}_\epsilon Q_\phi(s,a,\epsilon)$.

The actor-critic loop: behavior flow is maintained with BC loss $\mathcal{L}_F$; critic is trained with TD loss and includes the Flow Anchoring regularization term $\alpha_2 R$ in the target; policy update is constrained by both $-Q_\phi-Z_\psi$ (maximizing return) and $\alpha_1\mathcal{L}_B$ (Flow Anchoring behavior regularization).

### Key Designs

1. **Flow Anchoring: Single-step Flow for Behavior Regularization Instead of ODE**:

    - **Function**: Converts the constraint "policy output close to behavior flow endpoint" into "policy displacement close to behavior flow velocity field," eliminating the ODE solving cost.
    - **Mechanism**: Behavior flow $v_\theta$ is trained with standard CFM loss $\mathcal{L}_F(\theta)=\mathbb{E}[\|v_\theta(s,t,a_t)-(a-\epsilon)\|^2]$ ($a_t=(1-t)\epsilon+ta$). The actor’s Flow Anchoring loss is $\mathcal{L}_B(\omega)=\mathbb{E}[\|(\pi_\omega(s,\epsilon)-\epsilon)-v_\theta(s,t,a_{t,\omega})\|^2]$, where $a_{t,\omega}=(1-t)\epsilon+t\pi_\omega(s,\epsilon)$; the same anchoring term $-\alpha_2\mathbb{E}_t[\|\cdot\|^2]$ is also added to the critic target $q_\psi^{\pi_\omega,v_\theta}$. Theoretically (Theorem B.3), this loss upper bounds the Wasserstein-2 distance between policy and behavior distributions; minimizing it minimizes the distributional distance.
    - **Design Motivation**: FQL’s $\mathcal{L}_P=-Q+\alpha\|a_\omega-a_\theta\|^2$ requires ODE solving for $a_\theta$, with $N$ forward steps per gradient update; Flow Anchoring only evaluates $v_\theta$ once at $(s,t,a_{t,\omega})$, reducing training cost from $O(N_\text{flow})$ to $O(1)$, with theoretical guarantees intact. This is a classic "replace integral with its upper bound" trick.

2. **Noise-conditioned Critic + Operator $\mathcal{T}_n^\pi$**:

    - **Function**: Encodes distributional information into the noise variable $\epsilon$, enabling distributional critic training with a single noise sample and supporting Q-learning-style greedy max selection.
    - **Mechanism**: Defines a new operator $\mathcal{T}_n^\pi Q(s,a,\epsilon'):\overset{d}{=} r+\gamma\,\mathrm{ess\,sup}_{\epsilon\sim\mathcal{N}(0,I_d)}Q(s',\pi(s',\epsilon'),\epsilon)$. Theorem 4.1 proves it is a $\gamma$-contraction under $d_\infty$, so Banach fixed point exists and is unique. Critic is trained with TD: $\mathcal{L}_Q(\phi)=\mathbb{E}[(Q_\phi(s,a,\epsilon')-(r+\gamma q_\psi^{\pi_\omega,v_\theta}(s',\epsilon')))^2]$, where target $q$ uses $Z_\psi$ to estimate the ess sup part.
    - **Design Motivation**: Standard distributional critics (IQN/CODAC) require loss computation over 16-32 quantiles, and ess sup needs max-over-samples, further increasing variance; replacing quantile index with $\epsilon$ mathematically encodes the full distribution (since $\epsilon$ is continuous), and single-sample training is unbiased in expectation. Retaining ess sup instead of mean continues the Q-learning greedy philosophy, avoiding the underestimation issues of expected SARSA in OOD.

3. **Upper Expectile Regression for ess sup Estimation**:

    - **Function**: Uses asymmetric expectile loss with $\kappa\approx 1$ to estimate $Z_\psi\approx\mathrm{ess\,sup}_\epsilon Q_\phi$, avoiding explicit max-over-samples.
    - **Mechanism**: Expectile loss $\mathcal{L}_2^\kappa(\hat x-x)=|\kappa-\mathbb{1}((\hat x-x)<0)|(\hat x-x)^2$ converges to ess sup as $\kappa\to 1^-$ (Theorem 4.2); $Z_\psi(s,a)$ is trained with $\mathcal{L}_Z(\psi)=\mathbb{E}_{(s,a)\sim\mathcal{D},\epsilon}[\mathcal{L}_2^\kappa(Q_{\hat\phi}(s,a,\epsilon)-Z_\psi(s,a))]$, with fixed $\kappa=0.9$. The value-maximizing actor loss $\mathcal{L}_P(\omega)=\mathbb{E}[-Q_\phi(s,a_\omega,\epsilon')-Z_\psi(s,a_\omega)]$ leverages both noise-conditioned Q and upper expectile.
    - **Design Motivation**: Direct Monte Carlo estimation of ess sup requires sampling multiple $\epsilon$ and taking the maximum, increasing overestimation; expectile regression fits quantile-like values with a single sample, offering more controllable variance and bias; here, IQL’s in-sample max (for value function) is extended from "max over action" to "max over noise."

### Loss & Training

- Jointly optimize $\mathcal{L}_F(\theta)+\alpha_1\mathcal{L}_B(\omega)+\mathcal{L}_P(\omega)+\mathcal{L}_Q(\phi)+\mathcal{L}_Z(\psi)$, alternating actor/value updates.
- $\kappa=0.9$, $\tau=0.995$ (target network soft update), $\alpha_1,\alpha_2$ tune behavior regularization strength (separately for OGBench/D4RL).
- Inference uses only one-step $\pi_\omega(s,\epsilon)$ sampling, no ODE solving.

## Key Experimental Results

### Main Results
D4RL (4 antmaze + 12 adroit) and OGBench (25 state + 4 pixel) for a total of 9 task groups:

| Benchmark | Task Group | ReBRAC | IDQL | FQL | IQN | CODAC | Value Flows | **FAN** |
|-----------|-----------|--------|------|-----|-----|-------|-------------|---------|
| D4RL | antmaze (4) | 73 | 75 | **79±8** | 46±4 | 46±3 | 17±4 | **76±4** |
| D4RL | adroit (12) | 59 | 52±4 | 52±3 | 50±3 | 52±1 | 50±2 | **53±4** |
| OGBench | antsoccer (5) | 16±1 | 33±6 | **60±2** | 24±7 | 33±14 | 27±7 | **60±8** |
| OGBench | puzzle-3x3 (5) | 22±2 | 19±1 | 30±4 | 15±1 | 20±5 | 87±13 | **100±1** |
| OGBench | puzzle-4x4 (5) | 14±3 | 25±8 | 17±5 | 27±4 | 20±18 | 27±4 | **42±10** |
| OGBench | cube-double (5) | 15±6 | 14±5 | 29±6 | 42±8 | 61±6 | 69±4 | 46±11 |
| OGBench | scene (5) | 45±5 | 30±4 | 56±2 | 40±1 | 55±1 | **59±4** | 58±1 |
| OGBench | vis-locomotion (2) | 28±11 | 44±4 | 17±2 | 32±4 | **49±2** | 44±4 | **49±4** |
| OGBench | vis-manipulation (2) | 16±4 | 8±11 | 28±5 | 6±3 | 2±1 | 30±4 | **33±16** |

FAN achieves SOTA in 7 out of 9 task groups (within 95% optimal range), especially outperforming all baselines on complex multi-modal behavior distributions such as puzzle-3x3 with 100% success rate.

### Ablation Study

| Configuration | 5 OGBench Tasks Avg | Description |
|---------------|--------------------|-------------|
| Full FAN | Best | Flow Anchoring + $\mathcal{T}_n^\pi$ |
| NBRAC (ReBRAC’s standard BC instead of Flow Anchoring) | Loses 4/5 tasks | No flow for multi-modal behavior |
| NFQL (FQL’s flow ODE BC instead of Flow Anchoring) | Loses 4/5 tasks | Comparable expressiveness but higher computation |
| FAQL (Flow Anchoring, but non-distributional Bellman) | Loses 4/5 tasks | Lacks distributional information |
| Value Flows / CODAC (distributional critic) | 5-14× slower training | quantile multi-sample |

### Key Findings
- **Flow Anchoring vs Standard BC**: On tasks with multi-modal behavior distributions (OGBench puzzle/cube), flow-based behavior constraints significantly outperform Gaussian BC, as Gaussian fitting averages out multi-modality, producing OOD actions in the middle.
- **$\mathcal{T}_n^\pi$ vs Non-distributional Bellman**: FAN outperforms FAQL (distributional component removed) in 4/5 tasks, showing the utility of noise-conditioned critic’s distributional information, not just Flow Anchoring.
- **Training Efficiency**: FAN is 5-14× faster in training than IQN/CODAC/Value Flows (measured on cube-double-play); inference is even faster than all non-distributional baselines (since $\pi_\omega$ is single-step and $Z_\psi$ is not used in inference).
- **Offline-to-Online**: Reducing $\alpha_1,\alpha_2$ after offline training for online fine-tuning, FAN achieves SOTA in 4/5 OGBench tasks (puzzle-3x3 99→100, puzzle-4x4 17→100), indicating Flow Anchoring naturally supports online exploration—unlike directly sampling behavior actions, which limits exploration.
- **Theory + Experiment Loop**: Theorem 4.1 ($\mathcal{T}_n^\pi$ is a $\gamma$-contraction under $d_\infty$), Theorem 4.2 (expectile converges to ess sup as $\kappa\to 1$), and Theorem B.3 (Flow Anchoring controls Wasserstein-2 distance) together guarantee "simplification without loss of correctness."

## Highlights & Insights
- **"Replacing Integral with Its Upper Bound" is a Noteworthy Meta-trick**: FQL solves ODEs to obtain $a_\theta$ for BC distance; Flow Anchoring directly constrains the policy’s displacement to the velocity field—bypassing ODE solving, yet the theoretical upper bound still holds. This idea can transfer to other scenarios requiring forward simulation for loss computation (e.g., reverse SDE training, ODE-based generative model training).
- **Noise Variable as a Continuous Alternative to Quantile**: Replacing discrete quantile indices in distributional critics with continuous Gaussian noise enables unbiased single-sample training in expectation; this is the key to shifting "distributional RL" from the quantile paradigm to the "noise-conditioned" paradigm. Expectile estimation of ess sup is an elegant reuse of IQL’s philosophy.
- **All Three Components Have Theoretical Support**: Flow Anchoring (Theorem B.3), $\mathcal{T}_n^\pi$ contraction (Theorem 4.1), upper expectile convergence (Theorem 4.2)—rarely do all "compression tricks" in an efficiency-driven design have strict proofs, clearly written to preempt reviewer skepticism about "simplification by luck."
- **Offline-to-Online Friendly**: FAN does not directly sample dataset actions (unlike IDQL/FQL), but constrains the policy space, so reducing $\alpha$ in the online phase naturally restores exploration, making it highly compatible with online RL.
- **Engineering-driven Design**: The entire design is reverse-engineered from "training efficiency" and "inference efficiency," the two metrics users care about most, ultimately achieving SOTA + 5-14× speedup + ultra-fast inference. This "engineering-driven + theory-backed" research paradigm is highly instructive.

## Limitations & Future Work
- Assumes deterministic transition/reward for $\mathcal{T}_n^\pi$ derivation; more complex noise + state-transition decoupling is needed for stochastic environments, which is not discussed.
- The sensitivity of ess sup + $\kappa=0.9$ asymmetric expectile to reward scale variation is not analyzed; in some task groups (e.g., D4RL adroit), FAN only matches baselines, possibly related to reward shaping/dimension.
- The Wasserstein-2 upper bound for Flow Anchoring requires "all flow trajectories are straight lines" + Lipschitz conditions; in practice, $v_\theta$’s flow trajectories may not be straight. The paper assumes Lipschitz but does not quantify deviation from straightness.
- No experiments on large-scale or long-horizon tasks (e.g., Atari/Procgen), only on relatively simple robotics scenarios like D4RL/OGBench; whether SOTA holds in more complex environments remains to be seen.
- Inference still uses only a single noise sample; multi-sample policy improvement is unexplored—achieving Pareto optimality at inference may require further trade-offs between quality and latency.

## Related Work & Insights
- **vs FQL (Park et al. 2025c)**: FQL uses flow ODE for BC distance → $N$ flow solves per training step; FAN uses Flow Anchoring single-step evaluation → 5-14× faster training, and outperforms FQL on several OGBench tasks.
- **vs IDQL/Diffusion-QL**: Uses diffusion policy + Q-weighted sampling, requiring multi-step reverse diffusion; FAN uses one-step $\pi_\omega$ + behavior constraint, much faster inference.
- **vs IQN/CODAC (quantile distributional)**: Computes loss on fixed quantile grid → multi-sample overhead; FAN uses continuous noise + expectile → single-sample, and ess sup is more suitable for Q-learning’s greedy philosophy than mean-based.
- **vs Value Flows (Dong et al. 2025)**: Also distributional + flow, but Value Flows requires Jacobian-vector products during training, making it slow; FAN’s more direct noise-conditioned design is much more efficient.
- **vs IQL (Kostrikov et al. 2021)**: IQL’s in-sample max is borrowed by FAN for the noise dimension—replacing "max over OOD action" with "max over noise," following the same logic.

## Rating
- Novelty: ⭐⭐⭐⭐ Both Flow Anchoring and noise-conditioned $\mathcal{T}_n^\pi$ are original simplifications, but are compositional innovations built on FQL/IQL/IQN.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 29 tasks across D4RL/OGBench state-based/pixel-based, ablations for both Flow Anchoring and $\mathcal{T}_n^\pi$, offline-to-online validation, dual FLOPs + wall-clock measurement—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation → operator design → theoretical guarantee → experimental validation in a clear line, three core theorems for three core tricks; clear pseudocode, complete appendix derivations.
- Value: ⭐⭐⭐⭐⭐ Sets a new balance of "expressiveness + efficiency" for offline RL, highly useful for production deployment (robotics, autonomous driving); offline-to-online friendliness also opens a promising direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](../../ICLR2026/reinforcement_learning/reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ICLR 2026\] InFOM: Intention-Conditioned Flow Occupancy Models](../../ICLR2026/reinforcement_learning/infom_intention_flow_occupancy.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](perceptual_flow_network_for_visually_grounded_reasoning.md)

</div>

<!-- RELATED:END -->
