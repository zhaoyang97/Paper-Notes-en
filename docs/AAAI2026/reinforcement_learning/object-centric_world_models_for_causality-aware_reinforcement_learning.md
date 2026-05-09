---
title: >-
  [Paper Note] Object-Centric World Models for Causality-Aware Reinforcement Learning
description: >-
  [AAAI 2026][Reinforcement Learning][Object-centric world models] This paper proposes STICA, a framework that implements the world model, policy network, and value network through a unified object-centric Transformer architecture. The world model decomposes observations into independent per-object latent states for token-level dynamics prediction, while the policy and value networks estimate token-level causal relationships via a causal attention mechanism to enable causality-aware decision-making. STICA significantly outperforms DreamerV3 and other state-of-the-art methods on the Safety Gym and OCVRL benchmarks.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Object-centric world models
  - causal attention
  - model-based reinforcement learning
  - Slot Attention
  - Transformer
date: 2026-05-08
content_hash: e66f649612ddf6de
---

# Object-Centric World Models for Causality-Aware Reinforcement Learning

**Conference**: AAAI 2026
**arXiv**: [2511.14262](https://arxiv.org/abs/2511.14262)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Object-centric world models, causal attention, model-based reinforcement learning, Slot Attention, Transformer

## TL;DR

This paper proposes STICA, a framework that implements the world model, policy network, and value network through a unified object-centric Transformer architecture. The world model decomposes observations into independent per-object latent states for token-level dynamics prediction, while the policy and value networks estimate token-level causal relationships via a causal attention mechanism to enable causality-aware decision-making. STICA significantly outperforms DreamerV3 and other state-of-the-art methods on the Safety Gym and OCVRL benchmarks.

## Background & Motivation

Deep reinforcement learning (RL) has achieved success across multiple domains, yet still requires extensive environment interactions, which is prohibitively costly in real-world tasks involving real-time operation and physical hardware failures. Model-based RL (MBRL) improves sample efficiency by learning a world model to optimize policies within an "imagined" environment.

Nevertheless, existing world models face the following key challenges:

**Limitations of holistic representations**: From the Dreamer family to recent Transformer-based world models such as STORM and IRIS, existing approaches learn holistic representations of the environment. When the environment is high-dimensional, non-stationary, and involves multiple objects with complex interactions, holistic representations struggle to capture the important relationships and interactions among individual objects.

**Insights from human cognition**: Humans perceive the environment by decomposing it into discrete concepts such as objects and events, enabling more efficient and causality-aware decision-making. Incorporating this cognitive mechanism into world models is expected to allow RL agents to operate more effectively in complex scenarios.

**Shortcomings of existing object-centric MBRL**: Prior object-centric MBRL methods either require random-exploration episodes (OODP, COBRA), rely on supervised learning (FOCUS, OC-STORM), or depend on pre-training with external datasets (SOLD); moreover, most are not applicable to non-stationary environments or partially observable settings such as first-person perspectives. **STICA is the first MBRL agent that extracts object-centric representations directly from observations without requiring random exploration, supervision, or pre-training.**

## Method

### Overall Architecture

STICA consists of three components:
1. **Object-centric world model**: Slot-based autoencoder + Transformer dynamics model
2. **Causal policy network**: Transformer + causal attention
3. **Causal value network**: Transformer + causal attention

The training pipeline follows the standard MBRL paradigm: collect experience from the real environment → train the world model → train the policy and value networks on imagined trajectories.

### Key Designs

1. **Slot-based autoencoder**: Decomposes observation $o_t$ into object-centric representations and a background representation.

   **Encoder**: Slot Attention (Locatello et al. 2020) is applied to obtain $n$ slots $(s_t^1, \ldots, s_t^n)$ from the observation. Each slot is represented as 128-dimensional logits defining 16 categorical distributions over 8 categories, from which latent states $z_t^i$ are sampled. A learnable, **time-independent** background latent state $z_{BG}$ is additionally defined to represent static background information that is invariant to time (e.g., environment layout, agent body), enabling the encoder to disentangle dynamic objects from the static background.

   **Decoder**: A Spatial Broadcast decoder reconstructs individual RGB images $\hat{o}_t^k$ and unnormalized masks $m_t^k$ from the latent states; the mask for the background $\hat{o}_{BG}$ is filled with 0. After softmax normalization of the masks, the reconstruction is obtained by mixing:
   $$\hat{o}_t = \sum_{k=1}^{n} M_t^k \odot \hat{o}_t^k + M_{BG} \odot \hat{o}_{BG}$$

   **Loss function**:
   $$\mathcal{L}_\phi^{ae} = \mathbb{E}_B\!\left[\frac{1}{T}\sum_{t=1}^{T}\!\left(\mathcal{J}_{rec.}^t + \alpha_1 \mathcal{J}_{ent.}^t + \alpha_2 \mathcal{J}_{cross}^t\right)\right]$$
   comprising a reconstruction error, an entropy regularization term (to prevent the distribution from collapsing to a deterministic one), and a cross-entropy term (to align extracted latent states with dynamics model predictions). Since Slot Attention extracts slots in a random order, the cross-entropy term requires index permutation to minimize L1 distance before computation.

2. **Transformer dynamics model**: A recurrent model based on Transformer-XL that takes historical rewards $r_{1:t-1}$, latent states $(z_{1:t}^1, \ldots, z_{1:t}^n)$, and actions $a_{1:t}$ as tokens, with a causal mask to prevent access to future time steps. The key innovation is that positional encodings **depend only on time $t$, not on object indices $1, \ldots, n$**, ensuring equivariance of the output with respect to latent state ordering.

   The predictors include:
   - **Latent state predictor**: $p_\psi^{\hat{z}}(\hat{z}_{t+1}^k \mid h_t^k)$, categorical distribution
   - **Reward predictor**: $p_\psi^{\hat{r}}(\hat{r}_t \mid h_t')$, Gaussian distribution
   - **Discount factor predictor**: $p_\psi^{\hat{\gamma}}(\hat{\gamma}_t \mid h_t')$, Bernoulli variable

   Loss function:
   $$\mathcal{L}_\psi^{dyn} = \mathbb{E}_B\!\left[\frac{1}{T}\sum_{t=1}^{T}\!\left(\mathcal{J}_{cross}^{t+1} + \beta_1 \mathcal{J}_{rew.}^t + \beta_2 \mathcal{J}_{dis.}^t\right)\right]$$

3. **Causal Attention mechanism**: This is the core innovation of STICA.

   **Design Motivation**: The environment contains causal objects (relevant to action selection and value estimation, e.g., goals and obstacles) and non-causal objects (irrelevant, e.g., floor tiles). The policy should attend to reward-relevant objects, while the value estimator can entirely ignore non-causal objects.

   **Causal graph matrix $G$**: Defines causal relationships among objects, where $G_{i,j}$ indicates the existence of a causal relationship from object $j$ to object $i$ (1 for policy/value, 2 for causal objects, 3 for non-causal objects).

   **Causal score $p_t^k \in [0,1]$**: An MLP estimates the probability that each latent state $z_t^k$ represents a causal object. A weight matrix $W_t$ is constructed accordingly, and causal relationships between tokens are then represented by $W_t G W_t^\top$.

   **Causal attention computation**:
   $$\text{CA}_t = \text{Norm}\!\left(\text{softmax}\!\left(\frac{Q_t K_t^\top}{\sqrt{d}}\right) \odot W_t G W_t^\top\right) V_t$$

   This scales standard attention weights by the causal relationship matrix, enabling every attention layer in the policy and value networks to incorporate causal structure. All latent states share identical positional encodings to ensure output equivariance with respect to ordering.

### Loss & Training

- Policy learning employs A2C + GAE.
- The policy and value networks each have independent Transformers.
- Each Transformer processes $n$ latent state tokens plus one learnable goal token.
- End-to-end learning with no pre-training or supervision required.

## Key Experimental Results

### Main Results

**Safety Gym benchmark** (8 3D tasks, first-person view, non-stationary goals, multi-object interactions):

| Method | PointGoal1 | PointGoal2 | PointBtn1 | PointBtn2 | CarGoal1 | CarGoal2 | CarBtn1 | CarBtn2 | Mean | Norm. Mean |
|--------|-----------|-----------|----------|----------|---------|---------|--------|--------|------|-----------|
| PPO | 5.00 | 4.43 | 4.94 | 3.78 | 3.15 | 1.46 | 0.95 | 1.51 | 3.15 | 1.00 |
| TWM | 9.89 | 8.87 | 2.79 | 0.23 | 16.89 | 17.25 | 3.79 | 7.40 | 8.39 | 3.83 |
| DreamerV3 | 19.13 | 13.64 | 4.01 | 4.16 | 16.32 | 15.30 | 5.20 | 3.87 | 10.20 | 4.06 |
| TD-MPC2 | 8.00 | 6.50 | 4.31 | 4.84 | 2.57 | 1.24 | 1.90 | 0.48 | 3.73 | 1.15 |
| **STICA** | 13.63 | **13.64** | **11.52** | **5.97** | **17.09** | **18.27** | **9.65** | **9.25** | **11.90** | **5.49** |

**OCVRL benchmark** (success rate):

| Method | Obj. Goal | Obj. Interaction | Obj. Reaching |
|--------|-----------|-----------------|---------------|
| TWM | 0.727 | 0.080 | 0.772 |
| DreamerV3 | 0.677 | 0.156 | 0.697 |
| **STICA** | **0.737** | **0.333** | **0.867** |

### Ablation Study

| Configuration | PointButton1 | CarButton1 | Note |
|---------------|-------------|-----------|------|
| STICA (full) | Best | Best | All components |
| STICA w/o CA | Significant drop | Significant drop | Causal attention removed |
| STICA w/o BR | Slight drop | Slight drop | Background separation removed |
| STICA w/o CA+BR | Moderate drop | Moderate drop | Both removed |
| STICA w/o CA+BR+TP+TV | Close to TWM | Close to TWM | Transformer policy/value also removed |
| TWM | Baseline | Baseline | No object-centric representation |

Priority ranking of ablation findings:
1. **Causal attention**: Largest gain (from "w/o CA" to full STICA)
2. **Transformer policy/value networks**: Substantial gain
3. **Background separation**: Moderate gain
4. **Object-centric world model alone**: Only marginal gain

### Key Findings

1. **Consistent superiority on Safety Gym**: STICA achieves the best performance on 7 out of 8 tasks, with a normalized mean score of 5.49 vs. DreamerV3's 4.06 (+35.2%).
2. **Large advantage on Button tasks**: STICA scores 11.52 on PointButton1 compared to DreamerV3's 4.01, as Button tasks require identifying the target button among multiple dynamic obstacles—precisely the setting where object-centric representations and causal attention are most beneficial.
3. **Obj. Interaction task**: STICA achieves a success rate of 0.333 vs. DreamerV3's 0.156 (+113.5%), as this task requires the agent to directly interact with objects, benefiting from the synergy between the object-centric world model and the causal value network.
4. **PointGoal1 exception**: This task involves few and simple objects, making holistic representations sufficient; STICA shows no notable advantage here.
5. **Causal attention visualization**: The value network attends almost exclusively to reward-relevant goal objects; the policy network primarily attends to goals but also appropriately attends to other objects—consistent with intuition.

## Highlights & Insights

- **Unified framework**: The world model, policy, and value networks all employ object-centric Transformers, resulting in a highly unified and elegant architecture.
- **Explicit causal modeling**: Beyond using object-centric representations, STICA explicitly characterizes "which objects have a causal influence on decision-making" through causal scores, while providing interpretability via attention weight visualization.
- **End-to-end background separation**: Through a learnable, time-independent background latent state $z_{BG}$, foreground–background separation is achieved without pre-training.
- **Positional encoding design**: The dynamics model uses only temporal positional encodings (not object indices), ensuring equivariance with respect to slot ordering—critical given the random extraction order of Slot Attention.
- **First object-centric MBRL without additional supervision**: No random exploration, labeled data, or pre-training is required.

## Limitations & Future Work

1. The causal graph structure $G$ in causal attention is predefined (3 categories: goal, causal objects, non-causal objects), limiting flexibility.
2. The fixed number of slots $n$ may not adapt well to scenarios with a varying number of objects.
3. Evaluation is limited to Safety Gym (3D first-person) and OCVRL (2D/3D fixed viewpoint); more complex real-world environments have not been tested.
4. The random extraction order of Slot Attention necessitates index permutation for loss computation, introducing additional computational overhead.
5. The reliability of the causal score $p_t^k$ estimated by the MLP in more complex scenarios remains to be verified.

## Related Work & Insights

- **Dreamer family** (Hafner et al.): Representative RSSM-based world models; STICA directly benchmarks against DreamerV3.
- **STORM** (Zhang et al. 2023): Transformer world model that learns holistic representations.
- **TWISTER** (Burchi & Timofte 2025): Transformer + contrastive predictive coding.
- **OCRL** (Yoon et al. 2023): Model-free object-centric RL using Transformers for policy and value networks but lacking a world model.
- **EIT** (Haramati et al. 2024): Entity Interaction Transformer; similar object-centric policy without a world model.
- **SlotFormer** (Wu et al. 2023): Object-centric video prediction model; serves as a reference for the STICA dynamics model design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Causal attention + object-centric world model + background separation in a unified framework; highly elegant design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Dual-benchmark evaluation on Safety Gym and OCVRL with comprehensive ablations and convincing visualizations, though real-world environments are absent)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous formulations, and an intuitive architecture diagram in Figure 1)
- Value: ⭐⭐⭐⭐⭐ (A significant advance in object-centric MBRL; causal attention has broad applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](../../ICLR2026/reinforcement_learning/from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
