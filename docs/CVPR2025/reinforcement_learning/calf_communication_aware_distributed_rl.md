---
title: >-
  [Paper Note] CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning
description: >-
  [CVPR 2025][Reinforcement Learning][Distributed RL] This paper proposes the CALF framework, which injects configurable network delay, jitter, and packet loss models into RL training. This reduces policy performance degradation by approximately 3-4 times when deployed on real distributed edge devices, revealing that network conditions represent an important but overlooked dimension in the sim-to-real gap.
tags:
  - "CVPR 2025"
  - "Reinforcement Learning"
  - "Distributed RL"
  - "network-aware training"
  - "sim-to-real"
  - "latency robustness"
  - "edge deployment"
date: 2026-05-08
content_hash: 310a8157b2215cb8
---

# CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning

**Conference**: CVPR 2025  
**arXiv**: [2603.12543](https://arxiv.org/abs/2603.12543)  
**Code**: Yes (released with paper)  
**Area**: Reinforcement Learning / Distributed Systems  
**Keywords**: Distributed RL, network-aware training, sim-to-real, latency robustness, edge deployment

## TL;DR

This paper proposes the CALF framework, which injects configurable network delay, jitter, and packet loss models into RL training. This reduces policy performance degradation by approximately 3-4 times when deployed on real distributed edge devices, revealing that network conditions represent an important but overlooked dimension in the sim-to-real gap.

## Background & Motivation

**Background**: RL policies are typically trained under the assumption of zero-latency synchronous interactions, whereas sim-to-real research primarily focuses on physical and visual domain randomization. Large-scale distributed RL frameworks (such as IMPALA, SEED RL) optimize the communication of the training infrastructure rather than the network latency in the control loop.

**Limitations of Prior Work**: (1) When policies are deployed to edge devices (e.g., Raspberry Pi + cloud servers), Wi-Fi latency (30-80ms), jitter, and packet loss cause a 40-80% performance drop; (2) Existing latency-aware methods (like DCAC and delayed MDPs) either assume fixed latency or require modifying the RL algorithm itself; (3) There is a lack of reproducible, cross-hardware, network-aware RL training infrastructure.

**Key Challenge**: RL training assumes perfect communication, whereas real-world deployment faces imperfect networks. Network conditions constitute a sim-to-real gap dimension that is orthogonal to physical and visual domains.

**Goal**: (1) Quantify the impact of networks on distributed RL; (2) Verify whether network-aware training can narrow this gap; (3) Provide reproducible infrastructure.

**Key Insight**: Instead of modifying the RL algorithm, modify the training environment to transparently inject network impairments into the communication link. This is an algorithm-agnostic infrastructure solution.

**Core Idea**: Use NetworkShim middleware to transparently inject latency, jitter, and packet loss into agent-environment communication, allowing the training process to "experience" deployment-time network conditions.

## Method

### Overall Architecture

CALF implements the policy and environment as networked services communicating via message passing. It supports three progressive deployment modes: Mode 1 (local simulation, zero latency) $\to$ Mode 2 (simulation + simulated network) $\to$ Mode 3 (real edge hardware + real network). NetworkShim middleware sits on the communication link to transparently inject network impairments.

### Key Designs

1. **NetworkShim Network Impairment Injector**:

    - **Function**: Transparently injects latency, jitter, and packet loss into agent-environment communication.
    - **Mechanism**: For each packet, packet loss is first sampled according to Bernoulli($p_{loss}$), and then latency is sampled from $\max(0, \mathcal{N}(\mu_{latency}, \sigma_{jitter}^2))$. It supports two types of models: synthetic models (Ethernet 2ms±0.5ms, Wi-Fi-normal 30ms±10ms/2% packet loss, Wi-Fi-degraded 80ms±40ms/10% packet loss) and replay models based on real Wi-Fi trace collection.
    - **Design Motivation**: The transparent design makes the agent and environment network-unaware, requiring no modifications to any RL algorithm; the synthetic + trace models cover both controllable experiments and real-world scenarios.

2. **Progressive Deployment Modes**:

    - **Function**: Enables progressive-stage verification from pure simulation to real hardware.
    - **Mechanism**: Mode 1 has zero network overhead (~100K steps/hr CartPole) for rapid iteration; Mode 2 injects a simulated network (~50K steps/hr) for network-aware training; Mode 3 runs the environment on a Raspberry Pi and the policy on a Desktop (~20K steps/hr) for real hardware validation. The exact same policy code runs across all three modes.
    - **Design Motivation**: Deployment parity—ensuring that training and deployment share the exact same code path.

3. **Latency-Robust State Representation**:

    - **Function**: Enables the policy to infer the current state from delayed observations.
    - **Mechanism**: CartPole uses frame stacking ($k=d+1$ frames for a latency of $d$ steps) to infer velocity; MiniGrid uses LSTM to maintain a belief state. Optionally, action history tracking can be added for in-flight actions.
    - **Design Motivation**: Under network latency, the policy can only observe outdated states, necessitating temporal information to infer the current state.

### Loss & Training

Standard PPO is employed (via Stable-Baselines3) under three training regimes: Baseline (Mode 1, no network), Delay-Only (Mode 2, fixed 50ms), and Full Net-Aware (Mode 2, latency + jitter + loss). 10 random seeds are used, and each policy is evaluated under 5 deployment conditions for 50 episodes.

## Key Experimental Results

### Main Results

| Training Regime | Sim-Clean | Wi-Fi-Normal | Wi-Fi-Degraded | Sim-to-Real Gap |
|---------|-----------|-------------|---------------|----------------|
| Baseline (CartPole) | 500±0 | 285±45 | 120±60 | 76% |
| Delay-Only | 498±3 | 410±25 | 280±40 | 44% |
| Full Net-Aware | 495±5 | 465±15 | 420±30 | **15%** |

| Training Regime | Sim-Clean | Wi-Fi-Normal | Wi-Fi-Degraded | Sim-to-Real Gap |
|---------|-----------|-------------|---------------|----------------|
| Baseline (MiniGrid) | 0.92±0.05 | 0.55±0.12 | 0.28±0.15 | 70% |
| Full Net-Aware | 0.88±0.06 | 0.78±0.08 | 0.72±0.10 | **18%** |

### Ablation Study

| Network Phenomenon | Impact on CartPole | Description |
|---------|-------------------|------|
| Fixed delay 50ms | Moderate degradation (~15%) | Can be compensated via frame stacking |
| Random jitter 10ms | Significant degradation (~25%) | Unpredictability is more destructive |
| 2% packet loss | Severe degradation (~35%) | Missing observations lead to control failure |
| Jitter + packet loss | Maximum degradation (~55%) | Combined effect far outweighs individual impacts |

### Key Findings

- Network-aware training reduces the sim-to-real gap of CartPole from 76% to 15% (an approximate 4x improvement).
- Random jitter and packet loss are more destructive than fixed delay—modeling fixed delay alone is insufficient.
- Policies trained with Full Net-Aware exhibit only a ~1% performance degradation on Sim-Clean (retaining near-optimal performance under ideal conditions).
- Hierarchical policy graphs (with distributed deployment of multiple policy units) can also run successfully under the CALF framework.

## Highlights & Insights

- **Reveals an overlooked dimension of the sim-to-real gap**: Network conditions are orthogonal to physical and visual domains. Even with perfect physical modeling, a 100ms latency can cause control failure. This observation is simple yet crucial.
- **Algorithm-agnostic infrastructure solution**: Modifying the training environment rather than the RL algorithm allows any RL method to be plug-and-play. This software design philosophy is highly valuable.
- **Quantitative analysis of individual network phenomena**: The finding that jitter > packet loss > fixed latency provides practical guidelines for system design.

## Limitations & Future Work

- Validations were only conducted on CartPole and MiniGrid, which are relatively simple tasks.
- WAN or adversarial network conditions were not tested.
- Real-world hardware experiments were limited to a Raspberry Pi + Desktop setup, leaving more complex edge devices (such as Jetson, mobile phones) untested.
- Throughput drops significantly (Mode 3 is only 20% of Mode 1), meaning the feasibility of large-scale training remains to be verified.
- The framework could be combined with domain randomization strategies to simultaneously randomize both physical and network parameters.

## Related Work & Insights

- **vs DCAC**: Modifies the TD-learning algorithm to handle delay, requiring an algorithm redesign. CALF achieves algorithm agnosticism via environment-level injection.
- **vs IMPALA/SEED RL**: Optimizes worker-learner communication of the training infrastructure. CALF focuses instead on the agent-environment communication in the control loop.
- **vs Domain Randomization**: Randomizes physical/visual parameters. CALF complements this by adding network conditions into the randomization distribution.

## Rating

- Novelty: ⭐⭐⭐⭐ Pointing out networks as an independent dimension of the sim-to-real gap is an important insight.
- Experimental Thoroughness: ⭐⭐⭐ Simple environments and limited quantitative results.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and rigorous experimental design.
- Value: ⭐⭐⭐⭐ High practical engineering value for edge-deployed RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](../../ICML2026/reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICLR 2026\] Asynchronous Policy Gradient Aggregation for Efficient Distributed Reinforcement Learning](../../ICLR2026/reinforcement_learning/asynchronous_policy_gradient_aggregation_for_efficient_distributed_reinforcement.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](../../AAAI2026/reinforcement_learning/mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](../../ICML2026/reinforcement_learning/llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/reward-aware_proto-representations_in_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
