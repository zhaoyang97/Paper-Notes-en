---
title: >-
  [Paper Note] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning
description: >-
  [ICML 2026][Robotics][Offline RL] This paper proposes FAN: compressing "expensive generative policies + distributional critics" into "one-step flow anchoring + single-noise sample critic." By using Flow Anchoring to perf…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Offline RL"
  - "Flow Matching Policy"
  - "Distributional Critic"
  - "Noise-conditioned Q-Learning"
  - "Behavior Regularization"
date: 2026-05-08
content_hash: eccacabfd966e02f
---

# Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning

**Conference**: ICML 2026  
**arXiv**: [2605.01663](https://arxiv.org/abs/2605.01663)  
**Code**: https://github.com/brianlsy98/FAN (Available)  
**Area**: Reinforcement Learning / Offline RL / Generative Policy  
**Keywords**: Offline RL, Flow Matching Policy, Distributional Critic, Noise-conditioned Q-Learning, Behavior Regularization

## TL;DR
This paper proposes FAN: compressing "expensive generative policies + distributional critics" into "one-step flow anchoring + single-noise sample critic." By using Flow Anchoring to perform behavior regularization within a single flow evaluation and a noise-conditioned critic to replace multi-sample quantiles with a single Gaussian noise sample, FAN achieves SOTA performance on D4RL/OGBench while training 5-14× faster than similar distributional methods.

## Background & Motivation

**Background**: The core challenge of offline RL is constraining the policy within the behavior distribution of the dataset to avoid OOD overestimation. Recently, two types of highly expressive tools have been widely adopted: (1) **flow/diffusion policies** use flow matching to model multi-modal behavior distributions, offering stronger expressivity than Gaussian policies (e.g., FQL, IDQL, Diffusion-QL); (2) **distributional critics** learn the entire return distribution rather than just the expected value through mechanisms like quantiles (e.g., IQN, CODAC, Value Flows). Combining the two can achieve SOTA results, but at a significant computational cost.

**Limitations of Prior Work**: (i) Flow policies require solving an ODE to generate an action; 10 iterations equals 10× the overhead of a single forward pass. Using flow for behavior regularization during training (like $\mathcal{L}_P$ in FQL) requires solving the ODE to obtain $a_\theta$ and then calculating $\|a_\omega - a_\theta\|^2$, multiplying the flow step count into the training cost. (ii) Distributional critics typically need to compute losses across 16-32 quantiles simultaneously. When performing operations like ess sup, they introduce additional max-over-samples steps, increasing both computation and variance.

**Key Challenge**: There is a natural conflict between expressivity (multi-modal behavior + complete return distribution) and efficiency (single forward pass + single-sample estimation). Prior works have sacrificed training and inference speeds by several to dozen times to achieve high expressivity.

**Goal**: While retaining the expressivity of flow policies and distributional critics, this paper seeks to answer two specific technical questions: (1) Can flow policies use only a single iteration for behavior regularization? (2) Can distributional critics be trained using only a single Gaussian noise sample?

**Key Insight**: Behavior regularization essentially constrains the policy distribution to be close to the behavior distribution, which does not necessarily require sampling real behavior actions. An equivalent goal is to constrain the policy to "fall onto the velocity field trajectory of the behavior flow," which only requires a single-step flow evaluation. Similarly, distributional information can be encoded using a continuous noise variable $\epsilon$ (instead of discrete quantiles $\tau$). If the critic is written as $Q(s, a, \epsilon)$, it can be learned with a single noise sample.

**Core Idea**: Replace ODE solving with Flow Anchoring—constraining the "displacement" of the one-step policy by the behavior flow velocity field using the flow matching loss $\|(\pi_\omega(s, \epsilon) - \epsilon) - v_\theta(s, t, a_{t, \omega})\|^2$. Use a noise-conditioned critic + upper expectile regression to compress distributional information into a single Gaussian noise sample, employing asymmetric expectile estimation with $\kappa \approx 1$ to estimate $\mathrm{ess\,sup}$.

## Method

### Overall Architecture
FAN is a behavior-regularized actor-critic framework consisting of four networks:

- One-step policy $\pi_\omega(s, \epsilon)$: Directly outputs an action given state and noise.
- Behavior flow policy $v_\theta(s, t, a_t)$: Fits the dataset $(s, a)$ distribution via flow matching.
- Noise-conditioned critic $Q_\phi(s, a, \epsilon)$: Evaluates Q for a Gaussian noise sample.
- Upper value estimator $Z_\psi(s, a)$: Estimates $\mathrm{ess\,sup}_\epsilon Q_\phi(s, a, \epsilon)$ using expectile regression with $\kappa=0.9$.

The complete actor-critic loop: The behavior flow is maintained with a BC loss $\mathcal{L}_F$; the critic is trained with TD loss and incorporates a Flow Anchoring regularization term $\alpha_2 R$ into the target; the policy update is constrained by both $-Q_\phi - Z_\psi$ (maximizing returns) and $\alpha_1 \mathcal{L}_B$ (Flow Anchoring behavior regularization).

### Key Designs

1. **Flow Anchoring: Replacing ODE with Single-Step Flow for Behavior Regularization**:

    - **Function**: Transitions the constraint from "policy output close to behavior flow terminal state" to "policy displacement close to behavior flow velocity field," eliminating the cost of ODE solving.
    - **Mechanism**: The behavior flow $v_\theta$ is trained with the standard CFM loss $\mathcal{L}_F(\theta) = \mathbb{E}[\|v_\theta(s, t, a_t) - (a - \epsilon)\|^2]$ where $a_t = (1-t)\epsilon + ta$. The Actor's Flow Anchoring loss is $\mathcal{L}_B(\omega) = \mathbb{E}[\|(\pi_\omega(s, \epsilon) - \epsilon) - v_\theta(s, t, a_{t, \omega})\|^2]$, where $a_{t, \omega} = (1-t)\epsilon + t\pi_\omega(s, \epsilon)$. The same anchoring term $-\alpha_2 \mathbb{E}_t[\|\cdot\|^2]$ is added to the critic target $q_\psi^{\pi_\omega, v_\theta}$. Theoretically (Theorem B.3), this loss is an upper bound on the Wasserstein-2 distance between the policy and behavior distributions; minimizing it minimizes the distributional distance.
    - **Design Motivation**: FQL's $\mathcal{L}_P = -Q + \alpha\|a_\omega - a_\theta\|^2$ requires solving an ODE for $a_\theta$, necessitating $N$ forward passes per gradient update. Flow Anchoring evaluates $v_\theta$ only once at $(s, t, a_{t, \omega})$, reducing training costs from $O(N_\text{flow})$ to $O(1)$ while maintaining theoretical guarantees. This is a classic trick of "replacing the integral with its upper bound."

2. **Noise-conditioned Critic + Operator $\mathcal{T}_n^\pi$**:

    - **Function**: Encodes distributional information into the noise variable $\epsilon$, allowing the distributional critic to be trained with a single noise sample and supporting greedy max selection like Q-learning.
    - **Mechanism**: A new operator is defined as $\mathcal{T}_n^\pi Q(s, a, \epsilon') \overset{d}{=} r + \gamma \mathrm{ess\,sup}_{\epsilon \sim \mathcal{N}(0, I_d)} Q(s', \pi(s', \epsilon'), \epsilon)$. Theorem 4.1 proves it is a $\gamma$-contraction under the $d_\infty$ metric, ensuring a unique Banach fixed point. The critic learns via TD: $\mathcal{L}_Q(\phi) = \mathbb{E}[(Q_\phi(s, a, \epsilon') - (r + \gamma q_\psi^{\pi_\omega, v_\theta}(s', \epsilon')))^2]$, where the target $q$ uses $Z_\psi$ to estimate the ess sup component.
    - **Design Motivation**: Standard distributional critics (IQN/CODAC) must compute losses over 16-32 quantiles simultaneously, and ess sup requires max-over-samples, further increasing variance. By replacing quantile indices with $\epsilon$, $\mathcal{T}_n^\pi$ mathematically encodes full distributional information (as $\epsilon$ is continuous), and single-sample training is unbiased in expectation. Retaining ess sup instead of the mean follows the greedy philosophy of Q-learning, avoiding underestimation issues OOD associated with expected SARSA.

3. **Upper Expectile Regression for ess sup Estimation**:

    - **Function**: Uses asymmetric expectile loss with $\kappa \approx 1$ to estimate $Z_\psi \approx \mathrm{ess\,sup}_\epsilon Q_\phi$, avoiding explicit max-over-samples.
    - **Mechanism**: The expectile loss $\mathcal{L}_2^\kappa(\hat x - x) = |\kappa - \mathbb{1}((\hat x - x) < 0)|(\hat x - x)^2$ converges to the ess sup as $\kappa \to 1^-$ (Theorem 4.2). $Z_\psi(s, a)$ is trained using $\mathcal{L}_Z(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}, \epsilon}[\mathcal{L}_2^\kappa(Q_{\hat \phi}(s, a, \epsilon) - Z_\psi(s, a))]$ with $\kappa=0.9$. The actor loss for value maximization $\mathcal{L}_P(\omega) = \mathbb{E}[-Q_\phi(s, a_\omega, \epsilon') - Z_\psi(s, a_\omega)]$ utilizes both the noise-conditioned Q and the upper expectile.
    - **Design Motivation**: Direct Monte Carlo estimation of ess sup requires multiple samples of $\epsilon$ to take the maximum, which increases overestimation. Expectile regression fits the quantile equivalent with a single sample, providing more controllable variance and bias. This extends the in-sample max idea of IQL from "maximizing over actions" to "maximizing over noise."

### Loss & Training
- Five terms—$\mathcal{L}_F(\theta) + \alpha_1 \mathcal{L}_B(\omega) + \mathcal{L}_P(\omega) + \mathcal{L}_Q(\phi) + \mathcal{L}_Z(\psi)$—are jointly optimized with alternating actor/value updates.
- Hyperparameters: $\kappa=0.9$, $\tau=0.995$ for soft target updates; $\alpha_1, \alpha_2$ adjust behavior regularization strength.
- Inference uses only a one-step $\pi_\omega(s, \epsilon)$ sampling without ODE resolution.

## Key Experimental Results

### Main Results
Testing across D4RL (4 antmaze + 12 adroit) and OGBench (25 state + 4 pixel), totaling 9 task categories:

| Benchmark | Task Group | ReBRAC | IDQL | FQL | IQN | CODAC | Value Flows | **FAN (Ours)** |
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

FAN reaches SOTA (within 95% of the best) in 7 out of 9 categories, with a notable 100% success rate on puzzle-3x3, significantly outperforming all baselines in complex multi-modal tasks.

### Ablation Study

| Configuration | 5 OGBench Task Avg | Note |
|------|-------------------|------|
| FAN Full | Best | Flow Anchoring + $\mathcal{T}_n^\pi$ |
| NBRAC (Standard BC) | Lost 4/5 tasks | Failed to express multi-modal behavior |
| NFQL (Flow ODE BC) | Lost 4/5 tasks | Comparable expression but computationally expensive |
| FAQL (No distribution) | Lost 4/5 tasks | Loss of distributional information |
| Value Flows / CODAC | 5-14× Slower | Due to multi-sample quantiles |

### Key Findings
- **Flow Anchoring vs. Standard BC**: In tasks with multi-modal behavior distributions (OGBench puzzle/cube), flow-based behavior constraints significantly outperform Gaussian BC, as Gaussian fitting forces averaging that creates OOD actions in intermediate regions.
- **$\mathcal{T}_n^\pi$ vs. Non-distributional Bellman**: FAN outperforms FAQL (which removes distributional components) in 4/5 tasks, proving that noise-conditioned critic information is useful beyond just Flow Anchoring.
- **Efficiency**: FAN trains 5-14× faster than IQN/CODAC/Value Flows (measured on cube-double-play). Inference speed exceeds all non-distributional baselines because $\pi_\omega$ is one-step and $Z_\psi$ is not used during inference.
- **Offline-to-Online**: When transitioning from offline training to online fine-tuning by reducing $\alpha_1, \alpha_2$, FAN achieves SOTA on 4/5 OGBench tasks (puzzle-4x4 17→100), showing that Flow Anchoring is naturally compatible with online exploration.
- **Theory-Experiment Loop**: Theorem 4.1 ($\gamma$-contraction), Theorem 4.2 (expectile convergence), and Theorem B.3 (Wasserstein-2 bound) provide guarantees that "simplification does not sacrifice correctness."

## Highlights & Insights
- **"Replacing an integral with its upper bound" is a noteworthy meta-trick**: FQL solves an ODE to get $a_\theta$ for BC; Flow Anchoring constrains displacements to fall on the velocity field—bypassing the intermediate ODE product while maintaining a theoretical upper bound. This can be transferred to other scenarios requiring forward dynamics simulation to calculate loss.
- **Noise variables as continuous replacements for quantiles**: Switching distributional RL indices from discrete quantiles to continuous Gaussian noise allows single-sample training to remain unbiased in expectation. This is a key leap from quantile-based distributional RL to a noise-conditioned paradigm.
- **Theoretical support for the "compressed" trio**: Every core trick—Flow Anchoring (Thm B.3), $\mathcal{T}_n^\pi$ contraction (Thm 4.1), and upper expectile convergence (Thm 4.2)—is strictly proven, clearly intended to prevent concerns that the simplifications are merely heuristic.
- **Offline-to-Online Friendly**: Unlike IDQL/FQL, FAN does not sample dataset actions directly but constrains the policy space; thus, exploration capability is naturally released in the online stage after reducing $\alpha$.
- **Engineering-Oriented Design**: Designing the algorithm backwards from metrics users actually care about ("training efficiency" and "inference efficiency") resulted in SOTA performance + 5-14× speedup.

## Limitations & Future Work
- The derivation of $\mathcal{T}_n^\pi$ assumes deterministic transitions/rewards. Stochastic environments might require complex noise and state-transition decoupling, which is not discussed.
- Sensitivity of asymmetric expectiles ($Z_\psi$) to tasks with large reward scale variations is not analyzed. FAN only matched baselines in some groups (e.g., D4RL adroit).
- The equality of the Wasserstein-2 bound for Flow Anchoring requires "all flow trajectories to be straight" plus Lipschitz conditions. In practice, $v_\theta$ trajectories may not be straight; deviation analysis is missing.
- Lack of experiments on large-scale or long-horizon tasks (e.g., Atari/Procgen); SOTA status in environments beyond robotics remains to be verified.
- Inference still uses a single noise sample; multi-sample policy improvement paths were not explored for Pareto optimality in quality vs. latency.

## Related Work & Insights
- **vs. FQL (Park et al. 2025c)**: FQL uses flow ODE for BC distance → needs N flow evaluations per step. FAN uses Flow Anchoring → 5-14× training speedup and higher OGBench performance.
- **vs. IDQL/Diffusion-QL**: These use multiple steps of reverse diffusion. FAN's one-step $\pi_\omega$ is significantly faster at inference.
- **vs. IQN/CODAC**: These calculate losses over fixed quantile grids. FAN uses continuous noise + expectiles to achieve single-sample efficiency, and ess sup fits the Q-learning philosophy better than mean-based approaches.
- **vs. Value Flows (Dong et al. 2025)**: Both are distributional + flow, but Value Flows requires Jacobian-vector products, making it slower in wall-clock time.
- **vs. IQL (Kostrikov et al. 2021)**: FAN borrows it in-sample max philosophy and extends it to the noise dimension—replacing "max over OOD action" with "max over noise."

## Rating
- Novelty: ⭐⭐⭐⭐ Flow Anchoring and the noise-conditioned $\mathcal{T}_n^\pi$ are original designs, though built on the combination of previous works like FQL/IQL/IQN.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 29 tasks across D4RL/OGBench, ablation of each component, offline-to-online verification, and measurement of both FLOPs and wall-clock time.
- Writing Quality: ⭐⭐⭐⭐⭐ A clear path from motivation to operator design to theory and experimental validation.
- Value: ⭐⭐⭐⭐⭐ Successfully balances expressivity and efficiency in offline RL, which is highly useful for production deployment (robotics, autonomous driving).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DiBO: Offline Black-box Optimization with Diffusion Language Models (DNA + Robot Morphology)](training_diffusion_language_models_for_black-box_optimization.md)
- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](../../CVPR2026/robotics/rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)
- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](../../ICLR2026/robotics/on_entropy_control_in_llm-rl_algorithms.md)
- [\[ICML 2026\] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)

</div>

<!-- RELATED:END -->
