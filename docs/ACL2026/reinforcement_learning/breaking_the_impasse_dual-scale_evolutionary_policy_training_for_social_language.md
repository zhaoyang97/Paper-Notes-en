---
title: >-
  [Paper Note] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents
description: >-
  [ACL 2026][Reinforcement Learning][self-play RLVR] To address the "evolution impasse" in self-play RLVR within open-ended social language games (Negotiation / Don't Say It / Two Dollar Game)—where agent behavior homogeni…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "self-play RLVR"
  - "evolution impasse"
  - "dual-timescale baseline"
  - "asymmetric advantage"
  - "social games"
date: 2026-05-08
content_hash: 46c72ea2b7f61db5
---

# Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents

**Conference**: ACL 2026  
**arXiv**: [2605.08721](https://arxiv.org/abs/2605.08721)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Self-play RLVR / Social Language Agents  
**Keywords**: self-play RLVR, evolution impasse, dual-timescale baseline, asymmetric advantage, social games

## TL;DR
To address the "evolution impasse" in self-play RLVR within open-ended social language games (Negotiation / Don't Say It / Two Dollar Game)—where agent behavior homogenization leads to deterministic outcome distributions and vanishing gradient signals—this paper proposes DEPT. It utilizes a fast/slow dual-timescale EMA baseline to detect stagnation and asymmetric advantage reshaping to suppress dominant outcomes while amplifying rare trajectories. On Qwen3-4B/8B-Base, it increases the negotiation win rate from 16-20% to 32% and yields simultaneous benefits on OOD math/reasoning benchmarks.

## Background & Motivation

**Background**: RLVR has been validated as effective in closed-ended tasks like math and code (e.g., DeepSeek-R1 / Kimi K1.5). Extending it to multi-agent self-play to train social language capabilities is a natural direction (e.g., SPIRAL / MARS). The appeal of self-play lies in its automatic curriculum and the lack of need for human-labeled data, theoretically allowing for infinite scaling in zero-shot settings.

**Limitations of Prior Work**: The authors observed a failure mode when running standard self-play RLVR on Qwen3-4B-Base: while training rewards and game lengths increase (indicating mastery of game mechanics), the win rate against a fixed Gemini-2 opponent actually drops and plateaus at a suboptimal level. Diagnosis reveals that the match outcome distribution rapidly collapses into determinism (e.g., always draw or always lose), with match entropy $H_{match}^{(t)} \to 0$.

**Key Challenge**: When $H_{match}^{(t)} \to 0$, the value baseline $b_p$ converges to a constant $R_t$, causing the advantage $A_p(\tau) = R_p(\tau) - b_p \to 0$, which leads to the disappearance of the policy gradient. This mechanism locks the agent in a suboptimal state. Given the vast strategy space of open-ended social language games (much larger than Tic-Tac-Toe or Kuhn Poker), agents cannot escape local optima through random exploration alone.

**Goal**: (1) Design a metric capable of "detecting evolution impasse in real-time" to distinguish "true convergence to optimum" from "stagnation at sub-optimum"; (2) Design an intervention mechanism to "restore gradient signals" and restart exploration without destroying normal training dynamics.

**Key Insight**: Standard baselines (single-timescale EMA) are naturally reactive—they only track current expected returns and cannot distinguish between "stable optimum" and "stagnation." The authors introduce the divergence of dual-timescale baselines as a differential indicator of "training velocity," combined with match entropy to jointly determine if an intervention is required.

**Core Idea**: The divergence between a fast EMA and a slow EMA, multiplied by $(1 - \text{match entropy})$, is used as an intervention coefficient $\lambda^{(t)}$ to dynamically switch baselines. During normal training, a slow baseline (standard advantage) is used; when an impasse occurs, the system switches to asymmetric value (suppressing dominant outcomes with $V_{max}$ and elevating rare outcomes with $V_{min}$), artificially injecting synthetic variance to restore the gradient.

## Method

### Overall Architecture
DEPT maintains two fast/slow EMA baselines for roles $p \in \{0,1\}$ plus global $V_{max}/V_{min}$ historical bounds on top of GRPO/role-conditioned advantage estimation. Each training step includes: (1) self-play to collect a batch of trajectories; (2) calculating the match outcome distribution $P = \{p_{win}, p_{draw}, p_{loss}\}$ and match entropy $H_{match}^{(t)}$ to identify dominant outcomes; (3) updating both baselines and calculating the stagnation coefficient $\sigma^{(t)}$ and intervention coefficient $\lambda^{(t)}$; (4) using $\lambda^{(t)}$ to linearly fuse the slow baseline with asymmetric value into a reshaped baseline $\tilde{b}_p(\tau)$; (5) performing policy gradient updates using the reshaped advantage.

### Key Designs

1.  **Dual-timescale EMA + Intervention Coefficient $\lambda^{(t)}$**:
    - **Function**: Identify the specific moments when training has entered an impasse in real-time, avoiding sustained unnecessary interventions that might disrupt normal learning.
    - **Mechanism**: Maintain two baselines $b_p^{fast,t} = \alpha_{fast} b_p^{fast,t-1} + (1-\alpha_{fast}) R_p(\tau)$ and $b_p^{slow,t}$ (with $\alpha_{fast} = 0.5 < \alpha_{slow} = 0.95$). Theoretically, $\mathbb{E}[|b_p^{fast} - b_p^{slow}|] \propto |d\mu/dt|$ (proven in Prop A.1), so the divergence reflects "training velocity." The stagnation coefficient is defined as $\sigma^{(t)} = 1 - \tanh(|b_p^{fast} - b_p^{slow}|)$, combined with match entropy as $\lambda^{(t)} = \sigma^{(t)} \cdot \sqrt{1 - H_{match}^{(t)}}$.
    - **Design Motivation**: A single baseline will stabilize in both "stable optimum" and "stagnant sub-optimum" states, making them indistinguishable. The divergence of dual baselines can distinguish between "baseline stable but still progressing fast (large divergence)" and "baseline stable and no longer changing (small divergence)." Adding entropy information allows for precise locking onto impasses.

2.  **Asymmetric Advantage Reshaping + Global Historical Bounds**:
    - **Function**: Apply asymmetric pressure to dominant and rare outcomes when an impasse is triggered ($\lambda^{(t)} \to 1$), suppressing over-represented behaviors and amplifying under-represented ones.
    - **Mechanism**: Maintain historical value bounds $V_{max}^{(t)} = \max_{i \leq t} b_p^{fast,i}$ and $V_{min}^{(t)} = \min_{i \leq t} b_p^{fast,i}$ (using the fast baseline to retain sensitivity to extremes). Define an asymmetric target value $V_{asym}(\tau, b_p^{fast}) = \mathbb{I}(o_\tau = o_{dom}) \cdot V_{max}^{(t)} + \mathbb{I}(o_\tau \neq o_{dom}) \cdot V_{min}^{(t)}$. For dominant trajectories, the advantage $R - V_{max}$ is always non-positive (strong suppression); for rare trajectories, the advantage $R - V_{min}$ is maximized (strong amplification).
    - **Design Motivation**: In standard methods, the baseline equals the batch average dominant return, making the advantage of dominant samples near zero. Since rare samples are too few, the total gradient vanishes. Asymmetric reshaping uses a push-pull mechanism to inject synthetic variance $\nu_{syn} \propto (V_{max} - V_{min})^2$, ensuring $\|\nabla J\| > 0$ (proven in Thm A.3).

3.  **Adaptive Baseline Fusion for Smooth Transitions**:
    - **Function**: Avoid "hard switches" between intervention and non-intervention phases that could damage training stability.
    - **Mechanism**: The final baseline is $\tilde{b}_p(\tau) = (1 - \lambda^{(t)}) \cdot b_p^{slow,(t)} + \lambda^{(t)} \cdot V_{asym}(\tau, b_p^{fast})$. When $\lambda^{(t)}$ is small, it reverts to the standard slow baseline; as $\lambda^{(t)} \to 1$, it is dominated by the asymmetric value.
    - **Design Motivation**: Treating intervention as a continuous scalar rather than a binary switch allows policy updates to smoothly increase intervention intensity at the edge of an impasse, preventing jitter from minor fluctuations in $\sigma$.

### Loss & Training
The base loss is a self-play GRPO-style policy gradient $\nabla_\theta J(\theta) = \mathbb{E}_\tau [\sum_{p,t} \tilde{A}_p(\tau) \cdot \nabla_\theta \log \pi_\theta(y_t^{(p)} | o_t, p)]$, with the only modification being the replacement of $A_p$ with the reshaped $\tilde{A}_p = R_p - \tilde{b}_p$. Format rewards enforce a reasoning-then-acting structure ($\langle \text{think} \rangle ... \langle \text{act} \rangle ...$); rewards are $\{+1, -1, 0\}$ for win/lose/draw, with a format error penalty of -1.5. Parameters $\alpha_{fast}=0.5, \alpha_{slow}=0.95$. Training lasted 400 steps with a batch size of 128 and an lr of 1e-6, taking approximately 30 GPU-hours on 8×A800.

## Key Experimental Results

### Main Results: Success Rates across Three Social Games (vs. Average Win Rate of Three Evaluator Opponents %)

| Backbone | Method | Don't Say It AVG | Negotiation AVG | Two Dollar AVG |
|----------|--------|------------------|------------------|----------------|
| Qwen3-4B | VANILLA | 3.39 | 1.04 | 1.56 |
| Qwen3-4B | SPAG | 26.17 | 16.76 | 25.52 |
| Qwen3-4B | GRPO | 42.01 | 19.21 | 27.91 |
| Qwen3-4B | MARS | 40.89 | 20.73 | 27.69 |
| Qwen3-4B | SPIRAL | 45.88 | 16.84 | 27.65 |
| Qwen3-4B | **DEPT (Ours)** | **56.73** | **32.35** | **34.07** |
| Qwen3-8B | VANILLA | 15.17 | 5.69 | 2.13 |
| Qwen3-8B | SPIRAL | 37.89 | 17.30 | 26.22 |
| Qwen3-8B | MARS | 40.62 | 16.10 | 29.04 |
| Qwen3-8B | **DEPT (Ours)** | **54.56** | **31.88** | **36.50** |

On the Negotiation task, DEPT nearly doubled performance—rising from 16-17% (SPIRAL) to 32%, demonstrating the power of breaking the impasse.

### Main Results: OOD Reasoning Benchmarks (Zero-shot Transfer, AVG@16-32)

| Backbone | Method | Math500 | AIME24 | AIME25 | Olympiad | GPQA-D | MMLU-Pro | Average |
|----------|--------|---------|--------|--------|----------|--------|----------|---------|
| Qwen3-4B | VANILLA | 65.80 | 9.58 | 6.88 | 34.52 | 28.79 | 39.36 | 31.07 |
| Qwen3-4B | MARS | 69.25 | 9.41 | 8.47 | 35.18 | 34.01 | 53.34 | 35.59 |
| Qwen3-4B | SPIRAL | 67.95 | 9.55 | 7.81 | 34.15 | 34.18 | 51.32 | 34.73 |
| Qwen3-4B | **DEPT** | **74.64** | **11.22** | **10.03** | **38.79** | **37.04** | **56.45** | **38.68** |
| Qwen3-8B | **DEPT** | 74.98 | 13.06 | 12.43 | 40.83 | 38.72 | 57.60 | **41.21** |

OOD gains prove that DEPT learns genuine transferable reasoning capabilities rather than task-specific shortcuts.

### Ablation Study

| Configuration | Description |
|------|------|
| Full DEPT | Complete method |
| w/o Stagnation Coefficient | Continuous intervention during non-stagnant periods disrupts advantage estimation |
| w/o Match Entropy gating | Imposes penalties even with diverse outcomes, forcing ineffective exploration |
| w/o Asymmetric value | Fails to selectively suppress over-represented outcomes; stays at low entropy sub-optimum |
| w/o Dual-baseline | Fast-only or slow-only both perform worse than dual, proving synergy is necessary |

All ablations showed performance drops, proving the necessity of all four components.

### Key Findings
- **Impasse is a failure mode unique to open-ended social games**: In Negotiation, the win rates for SPIRAL/MARS actually decreased compared to early training, showing that better baselines can lead to stagnation—a phenomenon not seen in closed-ended RL.
- **Intervention Coefficient is an interpretable dynamic signal**: In early training, $\lambda$ is near 0 (healthy exploration); in later stages, it rises to 0.5+ (active intervention), adapting to match entropy.
- **Hyperparameters are insensitive to $\alpha_{fast}$**: Macro-F1 remained nearly constant within the [0.4, 0.6] range, indicating DEPT does not require fine-tuning.
- **Negligible additional overhead**: The combined cost of dual-baselines and reshaping accounts for < 0.0016% of per-iteration training time.

## Highlights & Insights
- **Theoretical analysis of "dual-timescale divergence $\approx$ training velocity differential"**: The authors provide a rigorous proof in Appendix A.1 showing $\mathbb{E}[|b_p^{fast} - b_p^{slow}|] \propto (\mathcal{T}_{slow} - \mathcal{T}_{fast}) |\dot\mu(t)|$. This gives a mathematical foundation to the intuition that "divergence equals speed," which can be applied to other RL scenarios requiring the detection of stagnation.
- **Asymmetric Push-Pull from a synthetic variance perspective**: Reshaping is interpreted as injecting synthetic variance $\nu_{syn} \propto (V_{max} - V_{min})^2$ when natural variance $\nu(t) \to 0$. This "signal engineering" perspective is transferable to any sparse reward or mode collapse problem in RL.
- **OOD Math/Reasoning Gains**: Capabilities learned through game-based self-play transfer to AIME / MATH500, further supporting the hypothesis that "game-based self-play elicits reasoning." This is critical evidence that DEPT is not just a task-specific method.

## Limitations & Future Work
- **30 GPU-hour training cost**: 30 hours on 8×A800 is still relatively expensive for academic researchers.
- **Reasoning-then-acting template increases inference latency**: Each action requires generating a chain-of-thought first, which significantly slows down long-horizon tasks.
- **Only validated on zero-sum two-player games**: N-player coordination or mixed-motivation scenarios have not been tested; whether the intervention coefficient is well-defined in non-zero-sum settings remains an open question.
- **More complex timescales beyond fast/slow EMA**: Could multi-scale wavelet or attention-based baselines further improve detection accuracy?
- **Lack of opponent diversification**: Self-play is always conducted against the current self, without introducing historical checkpoints or populations, which may limit the performance ceiling.

## Related Work & Insights
- **vs SPIRAL (Liu et al. 2025a)**: Both use self-play RLVR, but SPIRAL assumes role asymmetry is the primary issue; DEPT reveals the more fundamental problem of "outcome distribution collapse."
- **vs MARS (Yuan et al. 2025)**: MARS improves GRPO with turn-level advantages and role-specific normalization; DEPT goes further at the advantage level by introducing dual-timescale and asymmetric reshaping, which are orthogonal and additive.
- **vs SPAG (Cheng et al. 2024)**: SPAG uses offline RL and discounted rewards; DEPT uses online training and dynamic intervention, making it more adaptable to open-ended games.
- **vs common exploration techniques like entropy regularization**: Traditional entropy regularization adds bonuses to token-level entropy but struggles with "outcome-level collapse." DEPT directly diagnoses and intervenes in the outcome distribution.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ "Evolution impasse" is formally characterized as a unique failure mode of open-ended self-play RLVR; the dual-timescale + asymmetric reshaping combination is a novel diagnostic-intervention loop.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 3 games × 2 backbones × 3 evaluators + 6 OOD reasoning benchmarks + 3 OOD games + full ablations + theoretical analysis + significance testing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative (impasse → diagnosis → intervention → experiment → theory) with a tight coupling between mathematical formulas and empirical phenomena.
- **Value**: ⭐⭐⭐⭐⭐ Beyond solving social language game issues, the "dual-timescale divergence + asymmetric reshaping" approach is a general tool transferable to all self-play RLVR tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ICML 2026\] ALSO: Adversarial Online Strategy Optimization for Social Agents](../../ICML2026/reinforcement_learning/also_adversarial_online_strategy_optimization_for_social_agents.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](../../ICLR2026/reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](../../ICLR2026/reinforcement_learning/how_far_can_unsupervised_rlvr_scale_llm_training.md)

</div>

<!-- RELATED:END -->
