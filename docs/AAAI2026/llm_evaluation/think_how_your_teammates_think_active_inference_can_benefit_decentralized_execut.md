---
title: >-
  [Paper Note] Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution
description: >-
  [AAAI 2026][LLM Evaluation][Multi-agent cooperation] This paper proposes AIM (Active Inference Modeling), a framework for decentralized multi-agent reinforcement learning that models teammates' active inference processes — as perception–belief–action triple portraits — based solely on local observations without any communication. A dual filtering mechanism based on accuracy and relevance selectively integrates teammate belief portraits to assist decision-making. AIM achieves state-of-the-art or near-state-of-the-art performance across four benchmarks: SMAC, SMACv2, MPE, and GRF.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Multi-agent cooperation
  - active inference
  - teammate modeling
  - communication-free framework
  - decentralized execution
date: 2026-05-08
content_hash: 4e15a824aff6a384
---

# Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution

**Conference**: AAAI 2026
**arXiv**: [2511.18761](https://arxiv.org/abs/2511.18761)
**Area**: LLM Evaluation
**Keywords**: Multi-agent cooperation, active inference, teammate modeling, communication-free framework, decentralized execution

## TL;DR

This paper proposes AIM (Active Inference Modeling), a framework for decentralized multi-agent reinforcement learning that models teammates' active inference processes — as perception–belief–action triple portraits — based solely on local observations without any communication. A dual filtering mechanism based on accuracy and relevance selectively integrates teammate belief portraits to assist decision-making. AIM achieves state-of-the-art or near-state-of-the-art performance across four benchmarks: SMAC, SMACv2, MPE, and GRF.

---

## Background & Motivation

**Coordination challenges in decentralized settings**: In decentralized multi-agent systems, agents lack awareness of teammates' decision logic, which easily leads to miscoordination and suboptimal policies.

**Limitations of communication-based methods**: Exchanging decision information (Tell) is an intuitive solution, but is infeasible or unreliable in many real-world scenarios due to limited bandwidth, high latency, noise, and communication attacks.

**Shortcomings of existing agent modeling methods**:
- Some methods require access to other agents' full trajectories (e.g., ToMnet), which is unavailable during decentralized execution.
- Others (e.g., OMG) can only model agents with fixed parameters, capping the upper bound of team policy performance.
- Existing methods model only partial decision components (behavior or intent), ignoring the complete decision process and creating a gap between the model and reality.

**Core insight**: Rather than telling agents what their teammates do (Tell), agents should think about how their teammates reason (Think). Inspired by human cognitive decision-making and active inference theory, this work models the teammate decision process as a perception–belief–action active inference pipeline.

---

## Method

### Overall Architecture

AIM consists of two main modules: (1) a teammate triple portrait modeling module based on active inference, which generates perception–belief–action portraits for each teammate using only local observations; and (2) a dual filtering module that selectively integrates teammate belief portraits based on accuracy and relevance to assist decision-making. Training follows the CTDE paradigm (QMIX) and is compatible with value decomposition methods such as VDN and QPLEX.

### Key Designs 1: Perception Portrait

- **Function**: Constructs teammate $j$'s perspective observation $\hat{o}_{ij}^t$ from agent $i$'s local observation $o_i^t$.
- **Mechanism**: A perspective transformation operation — using teammate $j$'s position as the origin, recomputing the relative positions of other agents and taking the intersection with agent $i$'s observation as the perception portrait.
- **Implementation**: The perception portrait $\hat{o}_{ij}^t$ is fed into a GRU network to obtain teammate $j$'s historical trajectory information $\hat{h}_{ij}^t$.
- **Design Motivation**: To understand a teammate's behavior, one must first understand what the teammate "sees."

### Key Designs 2: Belief Portrait

- **Function**: Constructs the high-level decision basis — belief representation $z_{-i}^t$ — for teammates.
- **Distinction from perception**: Perception is objective (depending on environmental state and position), whereas belief is subjective (highly variable due to limited observation) and is thus modeled from agent $i$'s own perspective rather than the teammate's.
- **Generation**: Agent $i$'s trajectory $h_i^t$ and teammate index $id_{-i}$ are fed into a belief encoder, which outputs a Gaussian distribution $\mathcal{N}(\mu_i^t, \delta_i^t)$; the belief representation is obtained via reparameterization.
- **Two constraints**:
    - **Decision support**: Maximize mutual information between belief $z_{-i}^t$ and teammates' actual actions: $\mathcal{L}_{mi} = \mathbb{E}[\mathcal{D}_{KL}(p(z_{-i}^t | h_i^t, id_{-i}) \| q_\xi(z_{-i}^t | h_i^t, a_{-i}^t, id_{-i}))]$
    - **Short-term stability**: Cosine similarity loss between beliefs at adjacent timesteps: $\mathcal{L}_{cn} = \mathbb{E}[-\frac{z_{-i}^{t-1} \cdot z_{-i}^t}{\|z_{-i}^{t-1}\| \|z_{-i}^t\|}]$

### Key Designs 3: Action Portrait

- **Function**: Predicts teammates' actual actions as posterior feedback on modeling accuracy.
- **Input**: Concatenation of belief portrait $z_{-i}^t$ and historical perception information $\hat{h}_{-i}^t$.
- **Loss**: Cross-entropy between predicted and ground-truth actions: $\mathcal{L}_{ce} = -\sum_i a_{-i}^{true} \log \hat{a}_{-i}$.
- **Joint optimization**: Backpropagation of action prediction errors jointly optimizes perception and belief portraits, forming a closed loop.
- **Combined triple portrait loss**: $\mathcal{L}_{MD} = \lambda_{mi}\mathcal{L}_{mi} + \lambda_{cn}\mathcal{L}_{cn} + \lambda_{ce}\mathcal{L}_{ce}$

### Key Designs 4: Accuracy Filter

- **Problem**: Due to limited local observations, perception portraits inevitably contain errors; blindly using inaccurate portraits distorts decision-making.
- **Approach**: Learn a mapping $f: \mathbb{R}^h \mapsto \mathbb{R}$ that maps perception portraits to accuracy scores $c_{ij}^t = \text{softmax}(f(\hat{h}_{ij}^t))$, constructing an $N \times N$ evaluation matrix $\mathcal{C}^t$.
- **Three property constraints**:
    - **Mutual evaluation symmetry**: Symmetry loss $\mathcal{L}_{sy} = \|\mathcal{C} - \mathcal{C}^T\|_\mathcal{F}$
    - **Self-evaluation supremacy**: Diagonal loss $\mathcal{L}_{se} = -\sum_i c_{ii}$
    - **Similarity yields high scores**: Automatically satisfied by the neural network's property of producing similar outputs for similar inputs
- **Selection**: The $top\_k$ teammates with the highest accuracy scores proceed to the next stage.

### Key Designs 5: Relevance Filter

- **Problem**: Multi-agent cooperation is typically local, and integrating information from all teammates is unnecessary.
- **Approach**: An attention mechanism is used with agent $i$'s own perception history $h_i^t$ as the Query, the perception histories $\hat{h}_k^t$ of the $k$ filtered teammates as Keys, and their belief portraits $z_k^t$ as Values.
- **Attention score**: $\alpha_{i,k} = \frac{\exp(\frac{1}{\sqrt{d_{key}}}(h_i^t W_Q) \cdot (\hat{h}_k^t W_K)^T)}{\sum_{j=1}^k \exp(\frac{1}{\sqrt{d_{key}}}(h_i^t W_Q) \cdot (\hat{h}_j^t W_K)^T)}$
- **Fusion result**: $e_i^t = \sum_{j=1}^k \alpha_{i,j} \cdot z_j^t$, concatenated with $h_i^t$ and passed through a linear layer to compute local Q-values.
- **Design Motivation**: Belief portraits are fused rather than action portraits, as the high-level belief representation can dilute the impact of single-step modeling errors.

### Overall Training Objective

$$\mathcal{L}_{tot} = \mathcal{L}_{TD} + \mathcal{L}_{MD} + \mathcal{L}_{DF}$$

where $\mathcal{L}_{TD}$ is the TD loss from QMIX, $\mathcal{L}_{MD}$ is the triple portrait loss, and $\mathcal{L}_{DF} = \lambda_{sy}\mathcal{L}_{sy} + \lambda_{se}\mathcal{L}_{se}$ is the dual filtering loss.

---

## Key Experimental Results

### Experimental Setup

- **Benchmarks**: SMAC (6 maps), SMACv2 (6 tasks), MPE (3 tasks), GRF (Google Research Football)
- **Baselines**: QMIX, QPLEX, RODE, COLA, SIRD (without communication); MAIC, T2MAC (with communication); OMG (agent modeling)
- **Evaluation**: Average over 5 random seeds

### Main Results

1. **SMAC**: AIM outperforms all baselines on maps requiring explicit role assignment and cooperative partner selection (3s5z_vs_3s6z, corridor, 6h_vs_8z), **even surpassing communication-based methods**, suggesting that modeling is more effective than communication at close range.
2. **SMACv2**: Under additional challenges including randomized initial positions and unit types, baseline methods experience significant performance drops, while AIM remains competitive, validating the environmental adaptability of the perception portrait module.
3. **MPE**: AIM achieves state-of-the-art or near-state-of-the-art performance on locally observable Predator-Prey tasks.
4. **Ablation Study**: Removing any single component — belief portrait, action portrait, or either filtering module — leads to performance degradation, confirming the necessity of each component.

---

## Highlights & Insights

### Strengths

- The paradigm shift from "Tell" to "Think" is novel; incorporating active inference theory into teammate modeling in MARL represents a creative contribution.
- The perception–belief–action triple portrait design is systematic and complete, with clear correspondence to cognitive science concepts.
- The dual filtering mechanism (accuracy + relevance) addresses the inevitable noise in local observation-based modeling.
- Experiments span four major benchmarks with comprehensive baselines, including both communication-based and communication-free comparisons.

### Limitations & Future Work

- The perspective transformation operation (perception portrait construction) relies on known teammate positions, making it inapplicable in scenarios where positional information is unavailable.
- The selection of $top\_k$ requires tuning across different scenarios; no adaptive strategy is proposed.
- Validation is limited to discrete action spaces; extension to continuous action spaces is not discussed.

---

## Related Work & Insights

- **MAIC**: A communication-based MARL method that achieves teammate awareness through message exchange; AIM outperforms MAIC even at close range, demonstrating that modeling can substitute for communication.
- **OMG**: An agent modeling method based solely on local observations, but with fixed teammate policies; AIM allows all agents to train simultaneously, breaking through the upper bound on cooperative efficiency.
- **Theory of Mind (ToMnet)**: Models agents' mental states based on theory of mind, but requires access to the modeled agent's trajectory; AIM operates entirely from local observations, better satisfying decentralized requirements.
- **Active Inference**: A decision-making framework derived from Friston's free energy principle; AIM borrows its perception–belief–action structure for teammate modeling, representing its first systematic application in MARL.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation](../../ACL2026/llm_evaluation/think_in_latent_thoughts_a_new_paradigm_for_gloss-free_sign_language_translation.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)
- [\[AAAI 2026\] Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)

<!-- RELATED:END -->
