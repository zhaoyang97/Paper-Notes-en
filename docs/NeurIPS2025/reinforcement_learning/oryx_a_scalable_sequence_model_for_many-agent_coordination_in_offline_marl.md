---
title: >-
  [Paper Note] Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline Multi-Agent Reinforcement Learning] This paper proposes Oryx, a scalable sequence model algorithm for offline cooperative MARL that integrates the Retention-based Sable arch…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Offline Multi-Agent Reinforcement Learning"
  - "Sequence Modeling"
  - "Autoregressive Policy"
  - "Multi-Agent Coordination"
  - "Retention Mechanism"
date: 2026-05-08
content_hash: cc1a2c52bd2bfa8f
---

# Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL

**Conference**: NeurIPS 2025
**arXiv**: [2505.22151](https://arxiv.org/abs/2505.22151)
**Code**: [https://github.com/instadeepai/og-marl](https://github.com/instadeepai/og-marl)
**Area**: Reinforcement Learning
**Keywords**: Offline Multi-Agent Reinforcement Learning, Sequence Modeling, Autoregressive Policy, Multi-Agent Coordination, Retention Mechanism

## TL;DR
This paper proposes Oryx, a scalable sequence model algorithm for offline cooperative MARL that integrates the Retention-based Sable architecture with an autoregressive formulation of ICQ offline regularization. Through a dual-decoder that jointly outputs policies and Q-values, combined with counterfactual advantage estimation, Oryx achieves state-of-the-art performance on more than 80% of 65 datasets and demonstrates robust scalability to 50-agent scenarios.

## Background & Motivation
Offline Multi-Agent Reinforcement Learning (Offline MARL) aims to train multi-agent policies solely from pre-collected data without further environment interaction. This setting is critical for safety-sensitive and resource-constrained domains such as autonomous driving, warehouse logistics, and railway scheduling, where large amounts of historical log data exist but live experimentation is prohibitively costly.

Offline MARL faces two core challenges:

**Extrapolation Error**: Agents select out-of-distribution actions during training, and the resulting error grows exponentially with the joint action space. Prior works (ICQ, OMAR, CFCQL, etc.) mitigate this via policy constraints or conservative Q-value estimation, but are typically evaluated only in simple scenarios with a small number of agents.

**Miscoordination**: Without active interaction during offline training, agents must rely on behaviors produced by other—typically suboptimal—policies in the historical data, potentially developing mutually incompatible strategies. This problem is exacerbated with long temporal dependencies and large numbers of agents.

Existing methods either address only extrapolation error (e.g., ICQ, CFCQL) or lack validation in large-scale, long-horizon settings. The core idea is to combine a Retention-based sequence model capable of long-context modeling with the autoregressive ICQ offline constraint, explicitly addressing miscoordination through sequential policy updates.

## Method

### Overall Architecture
Oryx consists of an encoder and a dual-head decoder. The encoder applies Retention blocks to process each agent's observation sequence (timesteps $t$ to $t+k$), jointly reasoning over both agents $(a_1, \ldots, a_n)$ and temporal context $(t, \ldots, t+k)$ within each block. The encoded representations, together with dataset actions, are fed into the decoder, which produces Q-values and policy distributions via two separate heads.

### Key Designs
1. **Retention-Based Encoder-Decoder Architecture**:

    - Adopts the Retention mechanism from Sable (Mahjoub et al., 2025) in place of standard softmax attention, using decay matrices: chunkwise mode for efficient parallel computation during training and recurrent mode for maintaining hidden states during inference.
    - Compared to MAT (Transformer-based), Sable achieves superior efficiency and stability in large-agent scenarios.
    - Dual-output decoder: simultaneously outputs policy logits (action probabilities) and Q-value estimates.
    - Unlike the original Sable, the Value head on the encoder side is removed, and the encoder and decoder are trained end-to-end.
    - **Design Motivation**: The linear complexity of Retention ensures scalability to large numbers of agents and long sequences.

2. **Autoregressive ICQ Loss**:

    - The joint policy is factorized autoregressively: $\pi(\boldsymbol{a}|\boldsymbol{\tau}) = \prod_{j=1}^n \pi^{i_j}(a^{i_j} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}})$
    - Applies the multi-agent advantage decomposition theorem (Kuba et al., 2021): $A(\boldsymbol{\tau}, \mathbf{a}) = \sum_{j=1}^n A^{i_j}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}}, a^{i_j})$
    - Each agent's policy update is performed sequentially under the ICQ framework, guaranteeing monotonic improvement.
    - **Core Theorem (Theorem 1)**: The autoregressive policy can be optimized agent-by-agent under ICQ regularization, with each policy update given by:
    $\pi_*^{i_j} = \arg\max_{\pi^{i_j}} \mathbb{E}\left[-\frac{1}{Z^{i_{1:j}}} \log(\pi^{i_j}(a^{i_j} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}})) \exp\left(\frac{A^{i_{1:j}}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j}})}{\alpha}\right)\right]$
    - **Design Motivation**: Sequential updates condition each agent's policy improvement on the actions already selected by other agents, directly mitigating miscoordination.

3. **Counterfactual Advantage Estimation**:

    - The gradient variance upper bound under standard centralized advantage estimation contains an $(n-1)$ factor that grows linearly with the number of agents.
    - A COMA-style counterfactual baseline is adopted to eliminate the $(n-1)$ term:
    $A^{i_{1:j}}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j}}) = \sum_{m=1}^j \left[Q(\boldsymbol{\tau}, \mathbf{a}^{i_{1:m}}) - \sum_{a^{i_m}} \pi^{i_m}(a^{i_m} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:m-1}}) Q(\boldsymbol{\tau}, \mathbf{a}^{i_{1:m}})\right]$
    - **Design Motivation**: Reducing gradient estimation variance to ensure stable training as the number of agents scales.

### Loss & Training
- **Critic Loss**: SARSA-like ICQ update in autoregressive form, sampling target actions from the dataset with implicit importance weights:
$$J_Q(\phi) = \mathbb{E}_\mathcal{B}\left[\left(r + \gamma \frac{\exp(Q_{\phi^-}^{i_j}/\alpha)}{Z(\boldsymbol{\tau}')} Q_{\phi^-}^{i_j} - Q_\phi^{i_j}\right)^2\right]$$
- **Policy Loss**: Minimizes the KL divergence between the ICQ-optimal policy and the current policy.
- Agent orderings $i_{1:n}$ are randomly sampled at each step to avoid fixed-order bias.
- A target network $\phi^-$ is used to stabilize training.

## Key Experimental Results

### Main Results

| Environment | # Datasets | Oryx ≥ SOTA | Key Comparison |
|-------------|-----------|-------------|----------------|
| SMAC | 43 | 34/43 (79%) | vs CFCQL, OMIGA, ICQ-MA |
| MAMuJoCo | 16 | 14/16 (88%) | vs CFCQL, OMIGA |
| RWARE | 6 | 6/6 (100%) | ~20% gain on multiple scenarios |
| **Total** | **65** | **54/65 (83%)** | |

| Architecture Comparison (Oryx vs MAT+ICQ) | SMAC Median↑ | SMAC IQM↑ | RWARE Median↑ |
|-------------------------------------------|-------------|-----------|--------------|
| MAT+ICQ | 0.71 | 0.67 | 0.85 |
| **Oryx** | **0.91** | **0.87** | **0.89** |

### Ablation Study

| Configuration | T-Maze Replay | T-Maze Expert | Note |
|---------------|---------------|---------------|------|
| I-ICQ | 0.0±0.0 | 0.0±0.0 | Independent learning completely fails |
| MAICQ | 0.0±0.0 | 0.0±0.0 | CTDE also fails |
| Oryx w/o autoregressive actions | 0.0±0.0 | 0.0±0.0 | Autoregression is critical |
| Oryx w/o memory | 0.58±0.04 | 0.63±0.04 | Memory important for long-horizon coordination |
| Oryx w/o ICQ | 0.0±0.0 | 0.0±0.0 | Offline constraint is indispensable |
| **Oryx (full)** | **0.99±0.01** | **0.94±0.03** | All three components are necessary |

### Key Findings
- All three core components of Oryx (autoregressive action selection, memory mechanism, ICQ offline regularization) are individually indispensable; removing any one causes complete failure on the T-Maze task.
- In the Connector environment with 23 to 50 agents, Oryx maintains near-expert performance, while MAICQ degrades sharply (reaching only ~25% of expert level at 50 agents).
- Oryx performs particularly well in RWARE's long-horizon (500-step) sparse-reward settings, with gains of nearly 20% on multiple datasets.

## Highlights & Insights
- **Unified treatment of both core challenges in offline MARL**: Miscoordination is addressed through autoregressive policy updates, and extrapolation error through ICQ constraints—both are naturally integrated within the sequence modeling framework. The elegance of this design lies in the autoregressive structure simultaneously serving action generation and policy regularization.
- **Systematic validation at 50-agent scale**: Most existing offline MARL work is evaluated on only 3–10 agents; Oryx demonstrates robust performance at 50 agents via the Connector environment, substantially advancing the frontier of scalability.

## Limitations & Future Work
- The sequential dependency of autoregressive policy updates introduces additional computational overhead, with decoding time growing linearly with the number of agents.
- The advantage over baselines is smaller on continuous action spaces (MAMuJoCo) than on discrete ones (SMAC/RWARE), possibly due to lower importance sampling efficiency for ICQ in continuous spaces.
- Dataset quality and coverage significantly affect performance, yet the paper lacks a systematic analysis of dataset quality effects.
- Random agent permutations avoid fixed-order bias but may introduce additional variance.

## Related Work & Insights
- **vs MAT (Multi-Agent Transformer)**: MAT employs Transformer-based autoregressive action selection in the online setting; Oryx extends this to the offline regime and replaces attention with Retention for improved scalability.
- **vs MAICQ**: MAICQ combines RNN memory, CTDE value decomposition, and non-autoregressive ICQ; Oryx improves along all three dimensions (Retention memory, autoregressive ICQ, counterfactual advantage).
- **vs CFCQL/OMIGA**: These methods address extrapolation error via conservative regularization or distributional constraints but lack explicit mechanisms for large-scale agent coordination.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Retention-based sequence modeling with autoregressive ICQ is innovative, and Theorem 1 provides theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 65 datasets, T-Maze validation, 50-agent scalability tests, and architectural ablations constitute an exceptionally comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Methodological derivations are clear, though notation is occasionally inconsistent.
- Value: ⭐⭐⭐⭐⭐ Establishes a new SOTA in offline MARL with open-sourced code and datasets, providing direct impetus for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](../../AAAI2026/reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)

</div>

<!-- RELATED:END -->
