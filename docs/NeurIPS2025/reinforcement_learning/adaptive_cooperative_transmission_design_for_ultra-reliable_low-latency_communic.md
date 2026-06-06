---
title: >-
  [Paper Note] Adaptive Cooperative Transmission Design for URLLC via Deep RL
description: >-
  [NeurIPS 2025][Reinforcement Learning][URLLC] This paper proposes DRL-CoLA, a dual-agent DQN algorithm that adaptively configures 5G NR transmission parameters (numerology, mini-slot…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "URLLC"
  - "cooperative transmission"
  - "deep reinforcement learning"
  - "5G NR"
  - "dual-agent DQN"
date: 2026-05-08
content_hash: fd101ad6730c87de
---

# Adaptive Cooperative Transmission Design for URLLC via Deep RL

**Conference**: NeurIPS 2025
**arXiv**: [2511.02216](https://arxiv.org/abs/2511.02216)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: URLLC, cooperative transmission, deep reinforcement learning, 5G NR, dual-agent DQN

## TL;DR
This paper proposes DRL-CoLA, a dual-agent DQN algorithm that adaptively configures 5G NR transmission parameters (numerology, mini-slot, MCS) at the source and relay nodes respectively. Operating over a two-hop relay system with only local CSI, DRL-CoLA achieves URLLC reliability close to the optimum attained under full global CSI.

## Background & Motivation

1. **Background**: Next-generation wireless communications must support mission-critical applications such as remote surgery and autonomous driving, requiring packet error rates as low as $10^{-5}$–$10^{-7}$ and end-to-end latency on the order of milliseconds. Cooperative relay communication (two-hop transmission) is a key technique for improving reliability.
2. **Limitations of Prior Work**:
    - Existing two-hop transmission schemes are predominantly one-shot: a decoding failure at either hop results in packet loss, and they assume globally known CSI for both hops—an overhead incompatible with URLLC latency budgets.
    - ARQ retransmission protocols improve reliability at the cost of increased latency; 5G NR features such as AMC, flexible numerology, and mini-slots have previously been optimized in isolation rather than jointly exploited.
    - No prior work has studied the impact of ARQ retransmission on reliability in two-hop relay systems under strict latency constraints.
3. **Key Challenge**: In two-hop transmission, the total end-to-end delay $\mathcal{T}$ is a random variable depending on channel fading and the number of retransmissions. Its distribution is analytically intractable, making it infeasible for conventional optimization methods to enforce the constraint $\mathcal{T} \le T_{\text{th}}$.
4. **Goal**:
    - Jointly optimize numerology $\mu$, mini-slot size $N_{\text{sym}}$, and MCS index $I_{\text{MCS}}$ at each (re)transmission attempt for both hops.
    - Maximize the end-to-end successful delivery probability while satisfying strict latency constraints.
    - Rely solely on local CSI and ARQ feedback, without requiring global CSI.
5. **Key Insight**: The adaptive two-hop transmission problem is formulated as an MDP, with the source node and relay node acting as two independent agents that each learn a delay-aware transmission policy.
6. **Core Idea**: A dual-agent DQN framework learns distributed per-hop transmission parameter configuration policies, using the delay outage rate (DOR) as a cross-hop coordination signal to achieve URLLC without global CSI.

## Method

### Overall Architecture

The system consists of a half-duplex S → R → D two-hop relay:
- **Input**: Each agent observes local SNR $\gamma_i$, the average SNR of the next hop $\bar{\gamma}_{i+1}$, packet size $H$, and remaining delay budget $\tau_n$.
- **Output**: A transmission parameter tuple $(μ, N_{\text{sym}}, I_{\text{MCS}})$.
- **Procedure**: S transmits first; upon successful decoding, R forwards to D. A decoding failure at either hop triggers ARQ retransmission. This process repeats until D receives the packet or the delay budget is exhausted.

### Key Designs

1. **MDP Formulation**:
    - *Function*: Models the sequential parameter selection across both hops as an MDP.
    - *Mechanism*: The state $s_n^{(i)} = (\gamma_i, \bar{\gamma}_{i+1}, H, \tau_n)$ is 4-dimensional. The action space $\mathcal{A} = \{(\mu, N_{\text{sym}}, I_{\text{MCS}})\}$ contains $5 \times 4 \times 15 = 300$ discrete actions. State transitions are governed by the decoding error rate $\varepsilon_i$ and the remaining budget, with two absorbing terminal states: Success and Failure.
    - *Design Motivation*: MDPs naturally accommodate sequential decision-making, and RL circumvents the need for analytical modeling of the distribution of $\mathcal{T}$.

2. **DOR Reward Design**:
    - *Function*: Uses the delay outage rate as a cross-hop coordination signal.
    - *Mechanism*: The reward for S accounts not only for success at the current hop but also for the estimated probability that the next hop succeeds within the remaining budget $\tau_{n+1}$. DOR is defined as $\mathcal{P}_{\text{DOR}}(\bar{\gamma}_i, \tau) = 1 - \exp\!\left(-\frac{1}{\bar{\gamma}_i}(2^{H/(W\tau)} - 1)\right)$. On success, the reward is $1 - \mathcal{P}_{\text{DOR}}$; on failure, $-1$; on retransmission, $-0.1$.
    - *Design Motivation*: Since S cannot directly observe second-hop outcomes, DOR leverages the next-hop average SNR and remaining budget to indirectly estimate the success probability, enabling distributed coordination.

3. **Dual-Agent DQN Architecture**:
    - *Function*: S and R each maintain an independent DQN network and learn separately.
    - *Mechanism*: Each agent employs $\epsilon$-greedy exploration, experience replay for training, and a target network for stability. The decoding error probability is computed using the finite blocklength formula: $\varepsilon_i = Q\!\left(\ln 2 \sqrt{m_i/V_i} \left(\log_2(1+\gamma_i) - H/m_i\right)\right)$.
    - *Design Motivation*: DQN is well-suited to discrete action spaces (300 actions). The decoupled dual-agent design eliminates the need for global CSI, requiring only ARQ feedback for coordination.

### Loss & Training

- Standard DQN MSE loss: $\mathcal{L}_i(\theta_i) = \mathbb{E}\!\left[(y_n^{(i)} - Q_i(s_n^{(i)}, a_n^{(i)}; \theta_i))^2\right]$
- Target value: $y_n^{(i)} = \mathcal{R}_{n+1}^{(i)} + \gamma \max_{a'} Q_i(s_{n+1}^{(i)}, a'; \theta_i^-)$
- The target network is synchronized every $E'$ episodes.
- The number of retransmissions is implicitly optimized by the policy, with the small negative reward discouraging unnecessary retransmissions.

## Key Experimental Results

### Main Results — End-to-End Reliability

| Configuration | Packet Loss Rate | Comparison |
|---|---|---|
| DRL-CoLA (local CSI) | Near-optimal | Nearly overlaps with global-CSI one-shot |
| One-shot (global CSI) | Theoretical lower bound | Requires perfect global CSI |

When the relay is positioned along a path of total length $d_1 + d_2 = 1000$ m, the packet loss rate curve is V-shaped, reaching its minimum at the symmetric placement $d_1 = d_2$. The configuration $d_1 > d_2$ slightly outperforms $d_1 < d_2$, as a tighter latency budget on the second hop makes it advantageous for R to be closer to D.

### Ablation Study — RL Algorithm Selection

| RL Algorithm | Convergence Speed | Cumulative Reward | Notes |
|---|---|---|---|
| DQN | Fastest | Highest | Best choice for discrete action spaces |
| A2C | Slower | Lower | Policy-gradient less efficient in this setting |
| PPO | Intermediate | Intermediate | Same as above |

### Key Findings

- **Near-optimal performance without global CSI**: DRL-CoLA achieves packet loss rates nearly identical to the global-CSI one-shot scheme using only local CSI and ARQ feedback, demonstrating the effectiveness of the distributed learning approach.
- **DQN outperforms A2C/PPO**: In the 300-action discrete setting, value-based DQN converges faster and achieves superior final performance compared to policy-gradient methods.
- **Coordination role of DOR reward**: DOR enables S to "account for" the second-hop success probability when selecting transmission parameters, preventing S from exhausting too much of the delay budget and leaving R unable to complete its transmission.

## Highlights & Insights

- **DOR as a cross-agent coordination signal**: The use of the delay outage rate elegantly resolves the implicit coordination problem between two independent agents—S does not need to know R's specific decisions, but only needs to estimate, via DOR, whether sufficient time remains for R. This idea generalizes naturally to multi-hop networks.
- **Joint optimization of 5G NR features**: This is the first work to jointly optimize numerology, mini-slot size, and MCS simultaneously in a two-hop relay system; prior work optimized these dimensions individually.
- **Finite blocklength error probability modeling**: In the URLLC short-packet regime, the Polyanskiy finite blocklength formula replaces the Shannon capacity assumption, yielding a more practically accurate model.
- **Practical distributed architecture**: Each agent maintains an independent DQN, making deployment straightforward with no need for centralized training or inter-agent communication overhead.

## Limitations & Future Work

- **Rayleigh fading only**: The channel model is simplified (single-path Rayleigh fading), without considering multipath, Rician fading, or frequency-selective fading.
- **Perfect ARQ assumption**: ARQ requests are assumed to be always received correctly; in practice, ARQ signaling itself may be erroneous.
- **Two-hop only**: Scalability to multi-hop scenarios has not been validated.
- **Static channel model**: The channel is assumed quasi-static throughout the latency budget, which does not hold in high-mobility scenarios.
- **Future Directions**:
    - Extension to multi-hop multi-relay scenarios using a multi-agent RL framework.
    - Incorporation of channel estimation errors and imperfect ARQ.
    - Joint optimization of resource allocation across multiple users and carriers.

## Related Work & Insights

- **vs. Saatchi et al. (2023)**: That work jointly optimizes numerology, mini-slot, and MCS in a single-hop point-to-point setting; this paper extends the problem to two-hop relaying and incorporates ARQ retransmission.
- **vs. traditional one-shot schemes**: One-shot approaches require global CSI and offer no retransmission opportunity; the proposed scheme is more practical while achieving comparable performance.
- **vs. multi-agent RL (MARL)**: The dual-agent design relies on independent learning with implicit coordination via DOR, rather than CTDE-based MARL frameworks, and proves sufficient for the two-agent scenario considered.

## Rating

- Novelty: ⭐⭐⭐ Applying DRL to communication parameter optimization is precedented; the DOR reward design is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐ Comparisons with one-shot baselines and alternative RL algorithms are adequate, though comparisons with other MARL approaches are absent.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and mathematical derivations are complete.
- Value: ⭐⭐⭐ Offers practical value for 5G URLLC relay systems; academic novelty is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[NeurIPS 2025\] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
