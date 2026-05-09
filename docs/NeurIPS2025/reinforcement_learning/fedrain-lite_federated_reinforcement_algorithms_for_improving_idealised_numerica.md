---
title: >-
  [Paper Note] FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][Federated Reinforcement Learning] This paper proposes FedRAIN-Lite, a federated reinforcement learning framework that assigns RL agents to individual latitude bands to learn local climate parameterization policies with periodic global aggregation. Evaluated on a hierarchical idealized energy balance model (EBM), DDPG with this framework reduces area-weighted RMSE by over 50% in tropical and mid-latitude regions, providing a viable pathway for scaling RL to full-scale GCMs.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Federated Reinforcement Learning
  - Climate Modeling
  - Parameterization
  - Energy Balance Model
  - DDPG
date: 2026-05-08
content_hash: ab228a4f81f8a4e2
---

# FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models

**Conference**: NeurIPS 2025
**arXiv**: [2508.14315](https://arxiv.org/abs/2508.14315)
**Code**: [GitHub](https://github.com/p3jitnath/climate-rl-fedrl)
**Area**: Reinforcement Learning
**Keywords**: Federated Reinforcement Learning, Climate Modeling, Parameterization, Energy Balance Model, DDPG

## TL;DR
This paper proposes FedRAIN-Lite, a federated reinforcement learning framework that assigns RL agents to individual latitude bands to learn local climate parameterization policies with periodic global aggregation. Evaluated on a hierarchical idealized energy balance model (EBM), DDPG with this framework reduces area-weighted RMSE by over 50% in tropical and mid-latitude regions, providing a viable pathway for scaling RL to full-scale GCMs.

## Background & Motivation

**Background**: Subgrid parameterization in climate models (e.g., radiation, convection, turbulence) has traditionally relied on static parameters tuned offline against observational data. This approach is costly, inflexible, and poorly suited to the dynamic variability of climate states.

**Limitations of Prior Work**: The RAIN framework (Nath et al., 2024) demonstrated the feasibility of RL-based parameterization in idealized climate models, but it treats the entire system as a single agent, lacking spatial decomposition and regional adaptability, and cannot scale to the spatial grid structure of realistic GCMs.

**Key Challenge**: Realistic GCMs inherently possess a spatially decomposed structure—physical modules operate per latitude band—yet existing RL methods model the globe as a monolithic entity, failing to exploit regional specificity or enable parallelization.

**Goal**: How can RL parameterization policies mirror the spatially decomposed structure of GCMs? How should agents across latitude bands coordinate their learning? Which RL algorithm is best suited to this setting?

**Key Insight**: A federated learning paradigm is adopted—each latitude band corresponds to a local agent, and policy network parameters are periodically aggregated via FedAvg, preserving regional specialization while maintaining global consistency.

**Core Idea**: Map GCM architecture through spatially decomposed federated RL policies to enable regionally adaptive climate parameter learning.

## Method

### Overall Architecture
A three-tier progressively complex climateRL environment is constructed on the Budyko–Sellers energy balance model (EBM): ebm-v1 (single agent, global input) → ebm-v2 (multi-agent, global input + local reward) → ebm-v3 (multi-agent, local input + local reward, simulating GCM structure). Federated RL is implemented via the Flower framework on ebm-v2/v3, with synchronous aggregation every $K$ episodes.

### Key Designs

1. **Budyko–Sellers EBM and Its RL Formulation**:

    - **Function**: Transforms the latitude-resolved energy balance equation into an RL environment.
    - **Mechanism**: In the EBM equation $C(\phi)\frac{\partial T_s}{\partial t} = (1-\alpha(\phi))Q(\phi) - (A + BT_s) + \frac{D}{\cos\phi}\frac{\partial}{\partial\phi}(\cos\phi \frac{\partial T_s}{\partial\phi})$, the OLR coefficients $A$ and $B$ serve as policy outputs of the RL agent; the state is the latitude-resolved temperature profile $T_s(\phi)$; and the reward is the MSE against a target climatology.
    - **Design Motivation**: $A$ and $B$ are the physically most critical and hardest-to-calibrate parameters statically; dynamizing them is a natural entry point for improving model accuracy.

2. **Three-Tier Progressive Environment Design**:

    - **ebm-v1**: Single agent observing all 96 latitude-band temperatures and outputting global $\{A_\phi, B_\phi\}$—centralized baseline.
    - **ebm-v2**: Multi-agent (2 or 6 regions), each agent receiving the full temperature profile but optimizing only a regional reward, with periodic FedRL aggregation.
    - **ebm-v3**: Each agent receives only the temperature slice of its own region—fully simulating the decentralized, partially observable structure of a GCM.
    - **Design Motivation**: Progressively increasing spatial complexity validates the robustness and scalability of FedRL.

3. **Federated Aggregation Strategy**:

    - **Function**: Controls the frequency of local adaptation versus global synchronization.
    - **Mechanism**: Synchronous FedAvg is implemented via Flower, testing three schemes: fed05 (aggregation every 5 episodes), fed10 (every 10 episodes), and nofed (no aggregation).
    - **Design Motivation**: Frequent aggregation (fed05) promotes global stability but may suppress regional specialization; sparse aggregation (fed10) grants more regional freedom but risks divergence.

4. **RL Algorithm Selection**:

    - Three continuous control algorithms are compared: DDPG, TD3, and TQC.
    - DDPG consistently outperforms: it exhibits the greatest robustness to hyperparameters, the lowest computational cost, and the best cross-scenario generalization.
    - TD3 and TQC are occasionally competitive but exhibit higher variance, particularly instability in equatorial and polar regions.

### Loss & Training
Each configuration is trained with 10 random seeds, and 95% confidence intervals are reported. Policy network architecture and hyperparameters are held constant across ebm-v1/v2/v3 to ensure comparability.

## Key Experimental Results

### Main Results

| Environment | Algorithm | Aggregation | Convergence Speed | Tropical RMSE Improvement | Overall Performance |
|-------------|-----------|-------------|-------------------|--------------------------|---------------------|
| ebm-v1 | DDPG | — | >10K steps | Baseline | Single-agent baseline |
| ebm-v2 | DDPG | fed05 | 2.5K–5K steps | >50% | **Best** |
| ebm-v2 | DDPG | fed10 | 2.5K–5K steps | 30–40% | Higher variance |
| ebm-v2 | DDPG | nofed | 2.5K–5K steps | 20–30% | Unstable |
| ebm-v3 | DDPG | fed05 | 2.5K–5K steps | >50% | Comparable to ebm-v2 |
| ebm-v3 | TD3 | fed05 | Unstable | High variance | High variance |
| ebm-v3 | TQC | fed05 | Unstable | High variance | Unstable in equatorial region |

### Ablation Study

| Design Dimension | Finding | Impact Level |
|------------------|---------|--------------|
| Spatial decomposition granularity (2-agent vs. 6-agent) | Both effective; 6-agent shows more pronounced tropical improvement | Moderate |
| Aggregation frequency (fed05 vs. fed10 vs. nofed) | fed05 >> fed10 > nofed | Critical factor |
| Input scope (global vs. local slice) | ebm-v3 local input yields stronger regional specialization | Positive |
| Static baseline vs. RL | RL (especially DDPG+fed05) uniformly outperforms static climlab | Significant |

### Key Findings
- **DDPG is the most robust choice**: high hyperparameter robustness, good computational efficiency, and stable convergence across all spatial configurations.
- **Frequent aggregation (fed05) is critical**: areaWRMSE in the tropical bands 30°S–0° and 0°–30°N is reduced by over 50%.
- **Regional input in ebm-v3 promotes specialization**: local observations enable agents to learn better regional policies.
- **Polar regions** show comparable performance across all configurations—potentially requiring a distinct modeling strategy.
- **Convergence is substantially accelerated**: FedRL stabilizes within 2.5K–5K steps, while the single-agent baseline requires >10K steps.

## Highlights & Insights
- **Elegant mapping between GCM physical structure and federated learning architecture**: latitude bands → local agents; global climate consistency → FedAvg aggregation. This structural alignment enables natural integration of ML and domain knowledge.
- **Hierarchical experimental design**: the progressive v1→v2→v3 validation establishes a solid baseline and ultimately bridges the path to GCM-scale deployment.
- **The "back-to-basics" advantage of DDPG**: for the smooth, low-dimensional continuous control task of climate parameterization, the simplest DDPG proves more reliable than the more complex TD3/TQC, underscoring the importance of matching algorithm choice to problem characteristics.

## Limitations & Future Work
- **Validation is limited to an idealized EBM** without integration into a realistic GCM—the transition from EBM to GCM remains a substantial engineering challenge.
- **Limited physical complexity**: the EBM includes only radiative and diffusive transport, omitting critical processes such as convection and cloud microphysics.
- **The RL action space is restricted to only two parameters, $A$ and $B$**; realistic models require tuning tens of parameters.
- Only simple FedAvg aggregation is tested; heterogeneous methods (e.g., FedProx) may be better suited to settings with large polar–tropical discrepancies.
- Non-stationary climate (e.g., how parameters should evolve over time under global warming) is not addressed.

## Related Work & Insights
- **vs. RAIN (Nath et al., 2024)**: RAIN is the predecessor of this work, applying single-agent RL to global parameterization; the present work extends it to a multi-agent federated setting, introducing a spatial decomposition dimension.
- **vs. EnKI (Dunbar et al., 2021)**: Ensemble Kalman Inversion performs Bayesian parameter calibration but is a batch offline method lacking online adaptive capability; the proposed RL approach supports real-time interactive learning.
- This work demonstrates the potential of RL+FL in scientific computing; analogous ideas are transferable to ocean models, atmospheric chemistry models, and other domains with spatially decomposed structures.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of FedRL to climate parameterization; the physics–ML architectural mapping is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-tier progressive environments × three algorithms × three aggregation strategies × 10 seeds—systematic and rigorous.
- Writing Quality: ⭐⭐⭐⭐ Physical background is well-motivated, experimental visualizations are clear, and the appendix is comprehensive.
- Value: ⭐⭐⭐⭐ Paves the way for practical deployment of RL in climate science; directional significance outweighs the immediate technical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Improving Planning and MBRL with Temporally-Extended Actions](improving_planning_and_mbrl_with_temporally-extended_actions.md)

</div>

<!-- RELATED:END -->
