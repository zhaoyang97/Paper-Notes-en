---
title: >-
  [Paper Note] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration
description: >-
  [ICML 2025][Reinforcement Learning][Multi-Agent Reinforcement Learning] Proposed the SMPE² algorithm, which learns meaningful state belief representations through variational inference and integrates adversarial intrinsic exploration. It significantly enhances coordination in partially observable cooperative multi-agent environments, outperforming SOTA on three benchmarks: MPE, LBF, and RWARE.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Multi-Agent Reinforcement Learning"
  - "Partial Observability"
  - "State Modelling"
  - "Adversarial Exploration"
  - "Intrinsic Reward"
date: 2026-05-08
content_hash: 4d48cabc780d45a4
---

# Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration

**Conference**: ICML 2025  
**arXiv**: [2505.05262](https://arxiv.org/abs/2505.05262)  
**Code**: [github.com/ddaedalus/smpe](https://github.com/ddaedalus/smpe)  
**Area**: Reinforcement Learning  
**Keywords**: Multi-Agent Reinforcement Learning, Partial Observability, State Modelling, Adversarial Exploration, Intrinsic Reward

## TL;DR

Proposed the SMPE² algorithm, which learns meaningful state belief representations through variational inference and integrates adversarial intrinsic exploration. It significantly enhances coordination in partially observable cooperative multi-agent environments, outperforming SOTA on three benchmarks: MPE, LBF, and RWARE.

## Background & Motivation

In cooperative multi-agent deep reinforcement learning (MARL), agents need to learn collaborative policies under distributed partially observable environments. Real-world applications are widespread, such as multi-robot coordination, wireless network optimization, autonomous driving, and air traffic control.

**Key Challenge**: Under the CTDE (Centralized Training with Decentralized Execution) framework, agents can share information during training but must rely solely on their local observations during execution, without explicit communication channels.

**Limitations of Prior Work**:

**Disconnection between Agent Modelling (AM) representations and policies**: Belief representations learned by standard AM are not optimized for maximizing the value function, resulting in suboptimality (e.g., LIAM, SoMM).

**Unfiltered redundant state information**: Directly using compressed embeddings of global states can actually harm performance, as joint states contain redundant features that are uninformative for individual agents.

**Failure of using AM for improving exploration**: Existing AM methods cannot enhance initial random exploration policies, making them particularly inefficient in sparse-reward environments.

**Practical limitations**: Many methods assume a single controller, require prior knowledge, or need to access other agents' actions during execution.

**Ours focuses on two core questions**:
- Q1: Can agents infer meaningful state representations solely from their own observations to enhance coordination?
- Q2: Can these representations be leveraged to efficiently explore the state space to discover superior policies?

## Method

### Overall Architecture

SMPE² (State Modelling for Policy Enhancement through Exploration) is built upon MAA2C and consists of two core modules:

**1. Self-supervised state modelling**: Each agent $i$ infers a belief embedding $z^i$ from its own observation $o^i$ using a variational encoder-decoder (ED) to reconstruct the (filtered) observations of other agents. An AM filter $w^i$ is introduced to automatically filter out uninformative features.

**2. Adversarial count-based intrinsic exploration**: Intrinsic rewards are calculated using the SimHash of $z^i$ to encourage agents to discover novel states. This establishes an adversarial framework—new observations discovered by agent $i$ simultaneously present an "adversarial target" to the ED of other agents, thereby enhancing mutual state inference capabilities.

**Architectural Components** (see Figure 1):
- **Actor (blue)**: Partially observable, with policy $\pi_\psi(a_t^i \mid h_t^i, z_t^i)$ that explicitly incorporates the belief embedding into the policy network.
- **Critic (red)**: Both critics possess global observability.
    - $V_\xi(s)$: Standard MAA2C critic.
    - $V_k(\hat{s})$: Critic trained on the state purified by the AM filter $\hat{s} = o^i \oplus (w^i \cdot o^{-i})$.

### Key Designs

#### 1. Variational Encoder-Decoder (ED) and AM Filter

The ED of each agent $i$ includes:
- **Encoder** $q_{\omega^i}(z^i \mid o^i)$: Infers a Gaussian latent variable $z^i$ from its own observation.
- **Decoder** $p_{\phi^i}(w^i \cdot o^{-i} \mid z^i)$: Reconstructs the filtered observations of other agents.

The design of the **AM Filter** is crucial. $w^i_j = \sigma(\phi^i_w(o^j))$ is a learnable sigmoid MLP with output values in $[0,1]$ per dimension, controlling the information contribution of each observation feature of other agents $j$ to agent $i$.

**Definition of Uninformative Features** (which must simultaneously satisfy):
- Irrelevant to agent $i$'s future reward maximization.
- Infeasible to infer via $z^i$ due to partial observability and non-stationarity.

**Why is the ED conditioned only on $o^i$ (rather than history $h^i$)?** Because $z^i$ is utilized for intrinsic exploration, it should reward "novel observations" rather than "novel trajectories." Conditioning on $h^i$ would over-compress a novel high-value observation in an otherwise thoroughly explored trajectory, making it fail to receive sufficient intrinsic motivation. The policy network itself is already conditioned on $h_t^i$ and implicitly leverages beliefs $z_t^i$ across all timesteps.

#### 2. Adversarial Count-Based Exploration

A SimHash-based count method is adopted, but the hash domain is $z^i \in \mathcal{Z}$ instead of the raw observations:

$$\hat{r}^i = \frac{1}{n(SH(z^i))}$$

where $n(SH(z^i))$ is the visitation count of the hash value of $z^i$. SimHash is chosen because it maps neighboring $z^i$ to neighboring hash values at a low computational cost.

**Adversarial Aspect**: Agent $i$ is motivated to discover novel $o^i$ (leading to novel $z^i$). These new observations simultaneously serve as unseen reconstruction targets for the ED of other agents—effectively acting as "adversarial samples" for other agents to help them learn stronger state inference capabilities.

**Stability Guarantee**: A periodic hard update (every $N_{ED}=2000$ steps) is applied to the ED parameters $(\omega^i, \phi^i)$, preventing severe fluctuations of $z^i$ across continuous training epochs and ensuring the intrinsic reward stably decays to a minimum plateau.

#### 3. Dual-Critic Design

- $V_\xi(s)$: Standard critic, providing advantage estimation for the actor.
- $V_k(\hat{s})$: Critic trained on the filtered state $\hat{s}$, allowing $w^i$ to learn a filtering policy customized for policy optimization.

Ablation studies show that a single critic leads to high variance and divergence.

### Loss & Training

**Total Loss for State Representation Learning**:

$$L_\text{encodings} = L_\text{critic}^w + \lambda_\text{rec} \cdot L_\text{rec} + \lambda_\text{norm} \cdot L_\text{norm} + \lambda_\text{KL} \cdot L_\text{KL}$$

Descriptions of each loss term:

| Loss Term | Formula | Function |
|-------|------|------|
| $L_\text{rec}$ | $\|\tilde{w}^i \cdot o^{-i} - w^i \cdot \hat{o}^{-i}\|^2$ | Self-supervised reconstruction loss (with target network stabilization) |
| $L_\text{norm}$ | $-\|w^i\|_2^2$ | Prevents the AM filter from degenerating to zero |
| $L_\text{KL}$ | $\text{KL}(q_{\omega^i}(z^i\|o^i) \| \mathcal{N}(0,I))$ | Variationalizes belief representations and promotes belief consistency among agents |
| $L_\text{critic}^w$ | $[r_t^i + V_{k'}^\pi(\hat{s}_{t+1}) - V_k^\pi(\hat{s}_t)]^2$ | Trains the AM filter for policy optimization |

**Actor Loss**:

$$L_\text{actor}(\psi^i) = -\beta_H \cdot H(\pi_{\psi^i}(a_t^i \mid h_t^i, z_t^i)) - \log \pi_{\psi^i}(a_t^i \mid h_t^i, z_t^i) \cdot (r_t^i + V_{\xi'}^\pi(s_{t+1}) - V_\xi^\pi(s_t))$$

**Total Loss**: $L_\text{SMPE} = L_\text{actor} + L_\text{critic} + L_\text{encodings}$

**Key Hyperparameter Settings** (using LBF as an example): $\lambda_\text{rec}=1$, $\lambda_\text{norm}=0.1$, $\lambda_\text{KL}=1$, intrinsic reward coefficient $=0.1$. The ED uses a 3-layer MLP (non-RNN), and the latent variable dimension $=32$.

## Key Experimental Results

### Main Results

Evaluated on three major benchmarks and compared against methods such as MAA2C, COMA, MAPPO, ATM, EOI, EMC, and MASER.

| Benchmark/Task | Metric | SMPE² | Strongest Baseline | Performance |
|-----------|------|-------|---------|------|
| MPE Spread-3/4/5/8 | Average Episode Reward | Best | MAA2C/ATM | Superiority becomes more pronounced as active agent count increases |
| LBF 2s-9x9-3p-2f | Average Episode Reward | **Fully solved** | Other methods are 0 | Solves open challenge |
| LBF 4s-11x11-3p-2f | Average Episode Reward | **Fully solved** | Other methods are 0 | Solves open challenge |
| LBF 2s-12x12-2p-2f | Average Episode Reward | Best | EOI second best | Faster convergence |
| LBF 7s-20x20-5p-3f | Average Episode Reward | Consistently leading | MAA2C | Sustained advantage in large grids |
| RWARE tiny-2ag-hard | Average Episode Reward | Best | EOI | Significant lead |
| RWARE tiny-4ag-hard | Average Episode Reward | Best | EOI | Significant lead |
| RWARE small-4ag-hard | Average Episode Reward | Best | MAA2C | EOI completely fails here |

Runtime efficiency comparison (LBF 2s-12x12-2p-2f):

| Method | Runtime | Relative to SMPE² |
|------|---------|-----------|
| MAA2C | 37min | 0.5x |
| SMPE² | **1h13min** | 1x |
| ATM | 2h32min | 2x |
| EOI | 17h11min | 17x |
| MASER | 25h8min | 25x |
| EMC | 29h34min | 30x |

### Ablation Study

| Configuration | Key Performance | Note |
|------|---------|------|
| SMPE² (Full) | Best | All components cooperate synergistically |
| no_intr (No intrinsic reward) | Complete failure | State modeling alone is insufficient to discover high-value states under sparse rewards |
| no_filters (No AM filters) | Significant drop | Redundant information interferes with belief learning |
| no_kl (No KL regularization) | Obvious drop | Inconsistent beliefs, lower exploration efficiency |
| no_L2_norm (No norm regularization) | Decayed | $w^i$ may degenerate to zero |
| obs_rew (Standard SimHash on raw obs) | Inferior to SMPE² | Hashing $z^i$ outperforms hashing raw observations |
| no_critic_w (No second critic) | Decayed | $w^i$ cannot learn targeting policy optimization |
| SMPE_PPO (MAPPO backbone) | Outperforms MAPPO | Flexible framework, transferable to other backbones |

### Key Findings

1. **Exploration and coordination complement each other**: Neither state modeling alone (no_intr) nor exploration alone (standard methods) is sufficient; the success of SMPE² lies in the tight coupling of both.
2. **t-SNE Visualization**: When $L_\text{KL}$ is enabled, the beliefs of three agents overlap over a wide area (57.5% consistency). When disabled, they are completely separated (99.3%), demonstrating that KL regularization promotes belief consistency.
3. **Interpretability of the AM Filter**: In LBF tasks, the filter automatically learns to mask invisible and irrelevant features (such as the position of food out of view) while retaining information relevant to coordination.
4. **Failure Reasons of EMC and MASER**: Heavily reliant on the initial random policy, generating misleading intrinsic rewards or subgoals based on low-value data.

## Highlights & Insights

1. **Unification of State Modelling and Policy Optimization**: Proved that $V_{SM}^* = V^*$ (Proposition 2.1). The state modeling objectives do not constrain the optimal policy space, theoretically guaranteeing the optimality of the framework.
2. **Implicit Emergence of Adversarial Exploration**: Without explicitly designed adversarial mechanisms, encouraging the discovery of novel $z^i$ naturally forms a mutually beneficial "adversaria" between agents—your new discovery serves as my training fuel.
3. **Elegant Design of Self-Supervised Filtering**: The AM filter appears simultaneously in the target and prediction of the loss function, forming a self-consistent self-supervised learning loop.
4. **Extremely High Computational Efficiency**: 25x faster than MASER and 30x faster than EMC, with an additional overhead only about 2x that of MAA2C.
5. **Solves Multiple Recognized Open Challenges**: Tasks 2s-9x9-3p-2f and 4s-11x11-3p-2f in LBF, which were flagged as open challenges by prior works, are successfully resolved.

## Limitations & Future Work

1. **Architectural Extension**: Currently, the ED uses an MLP. The authors suggest that integrating Transformers into the architecture could further enhance state modeling capabilities.
2. **Scalability**: The reconstruction target grows linearly with the number of agents. Although the AM filter mitigates parts of this issue, further verification is still needed in larger-scale scenarios.
3. **Adaptability to Stochastic Environments**: Current experiments are based on deterministic dynamics. The paper does not evaluate performance under noisy observations or complex dynamics.
4. **Benchmark Coverage**: Evaluation is only performed on MPE, LBF, and RWARE, lacking more complex benchmarks such as SMAC (StarCraft Multi-Agent Challenge).
5. **Communication Setup**: Operates under a strict zero-communication assumption without comparing against hybrid methods that incorporate lightweight communication.

## Related Work & Insights

- **LIAM** (Papoudakis et al., 2021): Reconstructs other agents' information via ED, but its representation is disconnected from the policy. Ours resolves this through joint optimization.
- **SIDE** (Xu et al., 2022): Infers states but does not utilize $z^i$ during execution. SMPE² explicitly leverages belief embeddings to enhance policies.
- **EMC** (Zheng et al., 2021): Curiosity-based exploration reliant on QMIX, which performs very poorly on LBF.
- **MASER** (Jeon et al., 2022): Subgoal generation depends on the quality of the initial policy, yielding misleading subgoals in sparse-reward environments.
- **EOI** (Jiang & Lu, 2021): Encourages exploration via diversity, which is inapplicable to homogeneous agent scenarios (LBF).
- **MAVEN** (Mahajan et al., 2019): Explores joint action spaces but lacks encouragement for discovering novel states.

**Insights**: The concept of utilizing belief representations to simultaneously serve policy enhancement and exploration-driven mechanisms can be extended to other scenarios requiring "inference of unobserved information," such as single-agent tasks in POMDPs, decentralized communication networks, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of the state modeling framework and adversarial exploration is highly creative, though individual components (VAE, SimHash, AM) themselves are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely thorough, featuring three major benchmarks, comprehensive ablation studies, t-SNE visualizations, interpretability analysis of the AM filter, and runtime performance comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with a good balance of theory and intuition, though heavy mathematical notation slightly hinders the readability of certain derivations.
- Value: ⭐⭐⭐⭐⭐ — Successfully resolves several recognized open challenges. The proposed method is highly efficient and scalable, with open-sourced code, holding high value for the MARL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TaskForce: Cooperative Multi-agent Reinforcement Learning for Multi-task Optimization](../../CVPR2026/reinforcement_learning/taskforce_cooperative_multi-agent_reinforcement_learning_for_multi-task_optimiza.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets](adversarial_cooperative_rationalization_the_risk_of_spurious_correlations_in_eve.md)
- [\[ICLR 2026\] When Is Diversity Rewarded in Cooperative Multi-Agent Learning?](../../ICLR2026/reinforcement_learning/when_is_diversity_rewarded_in_cooperative_multi-agent_learning.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)

</div>

<!-- RELATED:END -->
