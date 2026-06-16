---
title: >-
  [Paper Note] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition
description: >-
  [AAAI 2026][Image Generation][Task Decomposition] This paper proposes CD3T, a two-level hierarchical MARL framework that employs a conditional diffusion model to learn action semantic representations $z_a^i$ (conditioned…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Task Decomposition"
  - "Conditional Diffusion Model"
  - "Action Semantic Representation"
  - "Value Decomposition"
  - "CTDE"
date: 2026-05-08
content_hash: 83cf9c98a8d3dd15
---

# Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition

**Conference**: AAAI 2026
**arXiv**: [2511.13137](https://arxiv.org/abs/2511.13137)  
**Code**: None (based on PyMARL)  
**Area**: Image Generation
**Keywords**: Task Decomposition, Conditional Diffusion Model, Action Semantic Representation, Value Decomposition, CTDE

## TL;DR
This paper proposes CD3T, a two-level hierarchical MARL framework that employs a conditional diffusion model to learn action semantic representations $z_a^i$ (conditioned on observations and other agents' actions to predict next observations and rewards), obtains subtask partitions via k-means clustering, and uses a high-level subtask selector combined with a low-level policy operating over a restricted action space. CD3T significantly outperforms all baselines on Super Hard scenarios in SMAC.

## Background & Motivation

**Background**: In cooperative MARL, CTDE (Centralized Training with Decentralized Execution) addresses partial observability through value decomposition methods (VDN, QMIX, etc.). However, as the number of agents grows, the joint action space expands exponentially, making exploration of high-value states increasingly sparse.

**Limitations of Prior Work**: Task decomposition—breaking complex tasks into subtasks—is a natural remedy, yet existing approaches (RODE using MLP-based action representations, GoMARL using grouping) suffer from insufficient representational capacity. Simple network architectures struggle to learn sufficiently discriminative subtask latent representations in high-dimensional continuous spaces.

**Key Challenge**: Subtask representations must simultaneously satisfy two requirements: (a) temporal stability (avoiding frequent switching); and (b) sufficient diversity (distinct subtasks must be meaningfully separable). Simple networks cannot satisfy both simultaneously.

**Key Insight**: Diffusion models are inherently suited for modeling stochastic processes via iterative denoising and possess strong representational capacity in high-dimensional continuous spaces, enabling the capture of multimodal distributions—which naturally corresponds to the distinct "behavioral modes" of different subtasks.

**Core Idea**: A conditional diffusion model is used as a flexible feature extractor to learn semantic representations capturing the effect of actions on the environment; clustering then yields subtask assignments, and subtask representations are used to enhance credit assignment in value decomposition.

## Method

### Overall Architecture
CD3T is a two-level hierarchical MARL framework: (1) a conditional diffusion model pre-trains action semantic representations $z_a^i$ during the first 50K steps → k-means clustering produces subtask partitions → (2) a high-level subtask selector assigns subtasks every $\Delta T$ steps, while a low-level policy executes within a restricted action space. Both levels employ a multi-head attention mixing network conditioned on subtask/action representations for value decomposition.

### Key Designs

1. **Learning Action Representations via Conditional Diffusion Model**:

    - Function: Encodes agent $i$'s one-hot action $a_i$ into a $d$-dimensional representation $z_a^i$, conditioned on local observation $o_i$ and other agents' actions $a_{-i}$.
    - Mechanism: A UNet with cross-attention denoising network $\epsilon_{\theta_d}(z_k, k, o_i, a_{-i})$ recovers $z_a^i$ from noise. Concurrently, $z_a^i$ is used to predict the next observation $o_i'$ and global reward $r$, grounding the representation in the action's effect on the environment. Total loss: $\mathcal{L} = \mathcal{L}_p + \eta_d \mathcal{L}_d$
    - Design Motivation: The multimodal generative capacity of diffusion models naturally induces diverse representations for distinct subtasks without requiring explicit diversity regularization.

2. **Dynamic Task Decomposition into Subtasks**:

    - Function: Clusters learned action representations into $g$ subtasks, each corresponding to a restricted action space.
    - Mechanism: K-means clustering is applied once over all action representations after 50K steps. The subtask representation $z_{\phi_j}$ is the centroid of cluster $j$. The high-level selector estimates agent $i$'s expected return for executing subtask $\phi_j$ as $Q_i^\phi(\tau_i, \phi_j) = z_{\tau_i}^T z_{\phi_j}$.
    - Design Motivation: K-means is simple and efficient; performing clustering only once (fixed after 50K steps) avoids the per-step clustering overhead of ACORM.

3. **Subtask-Aware Value Decomposition (Subtask-based Credit Assignment)**:

    - Function: Incorporates subtask/action representations into the mixing network to improve credit assignment.
    - Mechanism: Multi-head dot-product attention computes per-agent credit weights: $\lambda_{h,i}^\phi = \text{softmax}((W_{z_\phi} z_\phi)^\top \text{ReLU}(W_s s))$. The joint Q-value is: $Q_{tot}^\Phi = c_\phi(s) + \sum_h w_h^\phi \sum_i \lambda_{h,i}^\phi Q_i^\phi$. Theorem 1 proves that this formulation satisfies the IGM principle.
    - Design Motivation: Standard QMIX mixes solely based on the global state, potentially introducing spurious correlations; incorporating subtask semantic information enables more accurate estimation of each agent's contribution to the team.

## Key Experimental Results

### Main Results (SMAC Win Rate)

| Scenario | Difficulty | VDN | QMIX | RODE | GoMARL | CD3T |
|---|---|---|---|---|---|---|
| 8m | Easy | ~95% | ~97% | ~85% | ~97% | **~98%** |
| 3s5z_vs_3s6z | Super Hard | ~20% | ~30% | ~10% | ~65% | **~80%** |
| corridor | Super Hard | ~40% | ~45% | ~15% | ~70% | **~90%** |
| 6h_vs_8z | Super Hard | ~10% | ~15% | ~5% | ~45% | **~70%** |

### Ablation Study

| Configuration | corridor Win Rate | 3s5z_vs_3s6z | Note |
|---|---|---|---|
| CD3T (full) | ~90% | ~80% | Best |
| CD3T w/o diffusion | ~40% | ~35% | MLP replaces diffusion model; sharp drop |
| CD3T w/o Subtask-Attention | ~75% | ~65% | Subtask attention removed |
| CD3T (subtask=3) | ~85% | ~75% | 3 subtasks (vs. default 5) |

### Key Findings
- **The diffusion model is the primary performance driver**: Removing it causes a win rate drop of 50%+, confirming that the multimodal representations learned by diffusion models are far superior to those from MLPs.
- **Advantage is most pronounced on Super Hard scenarios**: CD3T achieves ~90% win rate on corridor (a 6-vs-24 numerical disadvantage), where all other methods remain below 70%.
- **Learned subtask semantics are interpretable**: PCA visualization shows that the diffusion model naturally clusters actions such as "attack," "move toward enemy," and "retreat" into well-separated groups.
- **Dynamic subtask switching carries tactical significance**: Visualizations show agents first assigned a "decoy" subtask to draw enemy attention, then switching to "focus fire" and "kiting" subtasks—tactical strategies learned entirely without supervision.
- **Fixing subtasks after 50K steps is sufficient**: Continuous clustering is unnecessary; a single clustering result provides adequate subtask assignments while substantially reducing computational overhead.

## Highlights & Insights
- **Using a conditional diffusion model for action semantic extraction** is a novel cross-domain application—the multimodal generative capacity of diffusion models naturally encourages subtask diversity, which is more elegant than explicit diversity regularization.
- **"Action's effect on the environment" as a representation learning objective** (predicting next observation + reward) yields richer semantic content than straightforward action encoding.
- **One-time clustering with fixed subtasks** vs. ACORM's per-step clustering—trading a marginal performance cost for a substantial gain in computational efficiency.
- **The "decoy-and-divide" tactic in the corridor scenario** is learned entirely automatically, demonstrating the potential of hierarchical MARL for complex cooperative behaviors.

## Limitations & Future Work
- The number of subtasks $g$ is a manually specified hyperparameter (3–5), with no mechanism for automatic determination.
- The diffusion model is trained only during the first 50K steps; if environmental dynamics change significantly thereafter, the fixed subtasks may no longer be appropriate.
- Evaluation is limited to cooperative game benchmarks such as SMAC and LBF, lacking assessment in more diverse real-world application scenarios.
- Inference latency of the diffusion model may become a bottleneck in large-scale multi-agent systems (though the paper notes it is not used after 50K steps).

## Related Work & Insights
- **vs. RODE**: RODE extracts role semantics using a simple MLP, limiting its expressive power; CD3T's use of a diffusion model yields a clear advantage on Super Hard scenarios.
- **vs. GoMARL**: GoMARL emphasizes inter-group contributions but neglects intra-group interactions; CD3T's subtask representations are more fine-grained.
- **vs. QMIX**: QMIX performs simple monotonic mixing; CD3T augments credit assignment with subtask/action semantic attention mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying diffusion models to subtask discovery in MARL is an innovative cross-domain contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers LBF + SMAC (8 scenarios) + SMACv2 (3 scenarios) + ablations + visualizations; reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly described, theoretical derivations are rigorous (Theorem 1 proving IGM), and visualizations are excellent.
- Value: ⭐⭐⭐⭐ Introduces a new technical direction (diffusion models) for subtask discovery in MARL, with significant performance gains on Super Hard scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](../../ICML2026/image_generation/offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)
- [\[AAAI 2026\] Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model](virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)
- [\[ICML 2026\] Bayesian Tensor Decomposition with Diffusion Model Prior](../../ICML2026/image_generation/bayesian_tensor_decomposition_with_diffusion_model_prior.md)
- [\[AAAI 2026\] Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines](backdoors_in_conditional_diffusion_threats_to_responsible_synthetic_data_pipelin.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](../../ICLR2026/image_generation/jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)

</div>

<!-- RELATED:END -->
