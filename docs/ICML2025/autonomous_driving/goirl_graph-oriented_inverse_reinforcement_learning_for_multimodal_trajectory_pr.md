---
title: >-
  [Paper Note] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction
description: >-
  [ICML 2025][Autonomous Driving][Inverse Reinforcement Learning] This work integrates the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) framework with vectorized scene representations for the first time, proposing the GoIRL trajectory prediction framework. Utilizing a learnable Feature Adaptor, it aggregates graph features into a grid space to accommodate IRL. It then employs a hierarchical parameterized trajectory generator (Bézier curves + refinement module) al…
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "Inverse Reinforcement Learning"
  - "MaxEnt IRL"
  - "Vectorized Representation"
  - "Multimodal Trajectory Prediction"
  - "Bézier Curves"
  - "MCMC Sampling"
date: 2026-05-08
content_hash: 8267f3e4c8c6a02d
---

# GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction

**Conference**: ICML 2025  
**arXiv**: [2506.21121](https://arxiv.org/abs/2506.21121)  
**Code**: None  
**Area**: Autonomous Driving / Trajectory Prediction  
**Keywords**: Inverse Reinforcement Learning, MaxEnt IRL, Vectorized Representation, Multimodal Trajectory Prediction, Bézier Curves, MCMC Sampling

## TL;DR

This work integrates the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) framework with vectorized scene representations for the first time, proposing the GoIRL trajectory prediction framework. Utilizing a learnable Feature Adaptor, it aggregates graph features into a grid space to accommodate IRL. It then employs a hierarchical parameterized trajectory generator (Bézier curves + refinement module) along with an MCMC probability fusion mechanism for multimodal trajectory prediction. GoIRL achieves state-of-the-art (SOTA) performance on Argoverse and nuScenes, demonstrating significantly stronger generalization capabilities compared to supervised models.

## Background & Motivation

**Background**: Prevailing trajectory prediction methods typically adopt the Behavior Cloning (BC) paradigm, which directly fits trajectory distributions via supervised learning on large-scale driving datasets. In recent years, vectorized representations (such as lane graph convolutions in LaneGCN) have replaced rasterized BEV images, significantly improving prediction accuracy.

**Limitations of Prior Work**: The BC paradigm faces two key issues: (1) Covariate shift: when test scenarios deviate significantly from the training distribution (e.g., temporary road hazards changing the drivable area), supervised models fail to adapt, resulting in severe compounding errors. (2) Mode collapse: during training, only a single ground-truth (GT) trajectory is provided, making it difficult for models to learn one-to-many mapping relationships.

**Key Challenge**: In theory, the IRL paradigm can resolve both issues—reward-driven and interactive learning naturally handles distribution shifts, and the maximum entropy principle naturally captures multimodality. However, existing IRL approaches rely on rasterized representations, with feature quality severely lagging behind modern vectorized supervised models.

**Goal**: (1) Introduce vectorized scene representations to IRL to mitigate information loss; (2) Alleviate the computational overhead of IRL in large state spaces; (3) Convert coarse-grained policies inferred by IRL into fine-grained continuous trajectories.

**Key Insight**: It is observed that the gap between cross-grid operations of IRL and vectorized features can be bridged via a learnable adaptor, which maps graph features into grid cells based on physical locations, followed by CNN-based dimensionality reduction.

**Core Idea**: Use a Feature Adaptor to inject vectorized features into the MaxEnt IRL grid space, allowing an IRL-based predictor to achieve equivalent scene-understanding capabilities to vectorized supervised models for the first time.

## Method

### Overall Architecture

GoIRL adopts a two-stage architecture: **Policy Inference $\rightarrow$ Trajectory Generation**.

The input is the driving context $\mathcal{C} = \{\mathcal{X}, \mathcal{O}, \mathcal{M}\}$ (historical trajectory of the target agent, trajectories of surrounding agents, and HD map information). In the first stage, a graph encoder extracts vectorized features, which are transformed into a grid space by the Feature Adaptor; MaxEnt IRL is then executed to obtain the reward distribution and policy. In the second stage, continuous trajectories are conditionally generated along sampled coarse-grained paths, which are further refined by a refinement module to yield final predictions. The entire process decomposes the multimodal trajectory distribution as:

$$P(\hat{\mathcal{Y}}|\mathcal{C}) = \sum_{\hat{\tau} \in \mathbb{S}(\mathcal{C})} P(\hat{\mathcal{Y}}|\hat{\tau}, \mathcal{C}) P(\hat{\tau}|\mathcal{C})$$

### Key Designs

1. **Graph-Oriented Context Encoder**

    - Function: Encodes the driving scene in a vectorized manner, extracting rich topological and semantic features.
    - Mechanism: Utilizes a two-layer graph structure: a lane graph (extracting lane node features via LaneGCN's dilated LaneConv) and a drivable area graph (extracting occupancy node features using a PointNet-like network). Agent motions are encoded via a 1D CNN + FPN. All features are ultimately fused using multi-layer graph attention.
    - Design Motivation: Compared to rasterization, vectorized representations better capture complex topological connections and long-range interactions, which serves as the foundation for GoIRL's performance improvement.

2. **Feature Adaptor**

    - Function: Seamlessly maps graph features into the grid space required by IRL.
    - Mechanism: Constructs a uniform grid centered around the target agent, allocating fused features from drivable areas to corresponding grid cells based on physical locations, while padding non-drivable areas with zeros. A strided CNN is used to downsample the high-resolution $\mathcal{C}_{fine}$ to a lower-resolution $\mathcal{C}_{coarse}$. A $1 \times 1$ convolution is then applied to obtain the reward distribution $\mathcal{R}$.
    - Design Motivation: IRL depends on grid-shaped inputs, whereas vectorized features represent unordered graph nodes. The bridge provided by the Feature Adaptor makes them compatible for the first time, and the CNN downsampling effectively mitigates the computational overhead of IRL caused by a large state space.

3. **MaxEnt IRL Policy Inference + Hierarchical Trajectory Generation**

    - Function: Infers a multimodal policy from the reward distribution and converts discrete paths into continuous trajectories.
    - Mechanism: Models the problem as a finite MDP $\{\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}\}$ with 9 discrete actions (8 directions + stop). A soft-optimal policy is solved using approximate value iteration: $\pi(a|s) = \exp(Q(s,a) - V(s))$, where $V(s) = \log \sum_a \exp(Q(s,a))$. Multiple distinct paths are sampled via MCMC. Conditioned on the sampled paths, continuous trajectories are represented by recursively predicting control points using Bézier curves (to ensure smoothness). Then, a refinement module performs position correction using the full trajectory to retrieve local context features. Probability fusion combines the MCMC distribution and classification probabilities with weighted aggregation.
    - Design Motivation: The MaxEnt principle encourages policies toward high-entropy distributions, naturally capturing multimodality. Bézier parameterization guarantees continuity and smoothness, while the refinement module couples history and predictions to achieve temporal consistency. MCMC fusion compensates for the limitations of classification probabilities in capturing uncertainty.

### Loss & Training

During the IRL stage, negative log-likelihood of MaxEnt IRL is used to supervise reward-distribution learning. In the trajectory generation stage, a Winner-Takes-All (WTA) strategy is applied to supervise only the best prediction with a smooth L1 regression loss. Both stages are trained jointly end-to-end.

## Key Experimental Results

### Main Results

| Dataset | Metric | GoIRL | QCNet | MTR | Gain |
|--------|------|-------|-------|-----|------|
| Argoverse 1 | minADE₆↓ | **Best** | Second | Third | Significant |
| Argoverse 1 | minFDE₆↓ | **Best** | Second | Third | Significant |
| Argoverse 1 | MR₆↓ | **Best** | Second | Third | Significant |
| nuScenes | minADE₅↓ | **Best** | - | Second | Significant |
| nuScenes | minFDE₅↓ | **Best** | - | Second | Significant |

### Ablation Study

| Configuration | Impact | Explanation |
|------|------|------|
| W/o Feature Adaptor | minFDE degraded by ~14% | Most critical component, proving the necessity of vectorized injection |
| W/o MCMC Probability Fusion | MR degraded by ~9% | Probability fusion enhances multimodal coverage |
| W/o Refinement Module | ADE/FDE degraded | Local context retrieval improves accuracy |
| W/o Bézier Parameterization | Degradation in trajectory smoothness | Parameterization guarantees continuity |

### Key Findings

- The Feature Adaptor is the most critical component, confirming that vectorized feature injection is the backbone of GoIRL's performance.
- In out-of-distribution (OOD) scenarios involving drivable area changes, GoIRL significantly outperforms supervised models, validating the generalization advantages of the reward-driven IRL paradigm.
- The reward distribution inferred by IRL possesses great interpretability, with high-reward areas tightly aligning with feasible driving paths.

## Highlights & Insights

- **First MaxEnt IRL + Vectorized Representation**: The Feature Adaptor breaks the barrier that historically restricted IRL to rasterized inputs. This enables an IRL-based predictor to outperform supervised methods on large-scale benchmarks for the first time, serving as a critical bridge between these two technical pathways.
- **Quantitative Verification of OOD Generalization**: Experiments with changes in drivable areas demonstrate the fundamental advantage of the IRL paradigm over BC. Since the reward function encodes environment constraints rather than behavior distributions, it shows inherent robustness to environmental shifts.

## Limitations & Future Work

- The approximate value iteration in the inner loop of IRL still introduces extra computational overhead, resulting in a slower inference speed than purely supervised methods.
- The discrete 9-directional action space has limited granularity, which might restrict precise representation for high-curvature turns.
- The nearest-neighbor assignment of the Feature Adaptor may not be precise enough for sparse drivable areas; an attention mechanism could be considered.
- The model is only verified in single-agent prediction scenarios; performance under multi-agent interactions remains unexplored.

## Related Work & Insights

- **vs. PGP (Guo et al., 2022)**: Both are IRL-based predictors, but PGP is limited by its rasterized representation. GoIRL resolves this fundamental limitation via the Feature Adaptor.
- **vs. QCNet/MTR++**: Prevailing supervised methods exhibit high accuracy but weak generalization. GoIRL outperforms them in standard scenarios while showcasing more outstanding generalization capability.
- **vs. GAIL/AIRL**: Adversarial IRL methods are computationally more expensive and unstable to train, whereas GoIRL's MaxEnt IRL is simpler and more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ The Feature Adaptor elegantly bridges the two paradigms, demonstrating for the first time on large-scale benchmarks that IRL can outperform supervised methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two large-scale benchmarks along with generalization experiments and detailed ablation studies, though it lacks an inference speed comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed methodological description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ A renaissance work of IRL in the field of trajectory prediction; the insights on OOD generalization offer practical value to autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning](../../CVPR2026/autonomous_driving/w2w_language-model-based_trajectory_prediction_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](../../ICCV2025/autonomous_driving/foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)

</div>

<!-- RELATED:END -->
