---
title: >-
  [Paper Note] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Paper Note] By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$ and combining it with continuity regularization and a HIQL hierarchical structure, LAVL achieves SOTA in 20 out of 22 OGBench datasets. It pushe
tags:
  - ICML 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: eda3fab1e84cdd99
---
# Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.25740](https://arxiv.org/abs/2605.25740)  
**Code**: https://github.com/oh-lab/LAVL.git (Available)  
**Area**: Reinforcement Learning / Offline Goal-Conditioned RL / Representation Learning  
**Keywords**: Offline GCRL, Value Function Architecture, Latent Representation Alignment, Quasimetric, Hierarchical Policy

## TL;DR
By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$ and combining it with continuity regularization and a HIQL hierarchical structure, LAVL achieves SOTA in 20 out of 22 OGBench datasets. It pushes the success rate of long-horizon tasks such as giant maze and stitch datasets from nearly zero in baselines to over 80%.

## Background & Motivation

**Background**: Offline goal-conditioned RL (GCRL) aims to learn a policy "capable of reaching any specified goal" from fixed trajectory data. A mainstream approach involves first learning a goal-conditioned value function $V(s,g)$ and then extracting the policy using advantage-weighted regression or quasimetric constrained optimization. Recent works like HIQL, GCIVL, QRL, CGCIVL, and OTA all focus on how to estimate this value function.

**Limitations of Prior Work**: In long-horizon sparse reward scenarios, the value functions learned via TD are highly unreliable: (i) success rates plummet as the horizon increases (e.g., antmaze-giant, humanoidmaze-giant); (ii) stitch-type datasets—which contain only short trajectory fragments and rely on rewards propagating through stitching—are particularly challenging; (iii) some methods require learning an additional high-level value network, increasing computation and tuning complexity.

**Key Challenge**: The authors identify the failure mode of "why $V$ is hard to learn" as **overgeneralization**. MLP-parameterized $V(s,g)$ tends to assign high values to states that are close to $g$ in Euclidean distance. Consequently, states on the "other side of a wall" are assigned high values despite having a large temporal distance, causing value "leakage" through walls. The root cause is the **inductive bias of the value function architecture**, not the learning objective itself. Existing quasimetric architectures (MRN, IQE) partially alleviate this but often fail on robotic manipulation tasks where behavior becomes highly unstable.

**Goal**: To find a value function architecture that both suppresses overgeneralization and remains stable across diverse tasks (maze and manipulation), while seamlessly integrating into a hierarchical policy to handle long horizons.

**Key Insight**: Through visualization and cross-ablation (e.g., "GCIVL × IQE" vs. "QRL × MLP"), the authors attribute performance disparities to architecture rather than objectives. They further discover that the strict metric constraints of quasimetrics are too strong an inductive bias, hurting tasks with inconsistent geometric structures like manipulation. A **weakened, learnable** latent space distance should be more universal than a rigid quasimetric.

**Core Idea**: Define $V(s,g)$ as the latent space Euclidean distance between state and goal embeddings, but **deliberately keep the state and goal encoders separate**. This asymmetry breaks the strict metric property in exchange for cross-task stability.

## Method

### Overall Architecture
LAVL is an IVL-style (implicit V-learning) offline GCRL algorithm consisting of three components: (1) The LAN value network, which parameterizes $V(s,g)$ as the negative Euclidean distance between the outputs of two asymmetric encoders; (2) A joint loss of TD and local continuity regularization to ensure global Bellman consistency while suppressing local oscillations under long horizons; (3) A HIQL-style hierarchical policy where the subgoal representation depends only on the goal itself $w=\phi(g)$, reusing the same LAN value function to provide advantages for both high-level and low-level policies without an additional high-level value head. The pipeline takes offline trajectories $\mathcal{D}=\{(s_t,a_t,s_{t+1})\}$ as input and outputs a hierarchical policy pair $(\pi^h, \pi^l)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["Offline Trajectories D = {(s, a, s′)}"]
    subgraph LAN["LAN Value Network (Asymmetric Dual Encoders)"]
        direction TB
        ES["State Encoder φ_S(s)"]
        EG["Goal Encoder φ_G(g)<br/>Independent, non-shared"]
        ES --> V["V(s,g) = −‖φ_S(s) − φ_G(g)‖<br/>Latent Negative Euclidean Distance"]
        EG --> V
    end
    IN --> LAN
    LAN --> REG["TD + Local Continuity Regularization<br/>Expectile TD for global Bellman consistency<br/>Finite-difference suppresses local spikes"]
    subgraph HIER["HIQL Hierarchical Policy + Shared Value"]
        direction TB
        PH["High-level π^h → Subgoal w = φ(g)<br/>Goal-only information bottleneck"]
        PL["Low-level π^l(a | s, w)"]
        PH --> PL
    end
    REG -->|Same LAN value for high/low-level advantage (AWR)| HIER
    HIER --> OUT["Hierarchical Policy (π^h, π^l)"]
```

### Key Designs

**1. Latent Alignment Network (LAN): Blocking overgeneralization via architecture**

The research found that MLP-parameterized $V(s,g)$ generalizes according to the original state space Euclidean distance, leading to value leakage through walls. LAN's counter-strategy is to change the generalization method: using two **independent, non-shared** networks $\varphi_S:\mathcal{S}\to\mathbb{R}^d$ and $\varphi_G:\mathcal{G}\to\mathbb{R}^d$ to embed states and goals separately, and defining value as the negative latent Euclidean distance:

$$V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2.$$

Value no longer diffuses along the original geometry but generalizes based on "latent alignment." The intentional lack of weight sharing and strict metric constraints is key: while strict quasimetrics (MRN/IQE) perform well on mazes, they fail on manipulation tasks with inconsistent geometry. The asymmetric dual-encoder is a middle ground—retaining the benefits of latent distance induction without the rigidity of the triangle inequality.

**2. TD + Local Continuity Regularization: Balancing global consistency and local smoothness**

While LAN influences generalization, sparse rewards and long horizons can still lead to sharp oscillations in the value function due to under-constrained local geometry. The TD part uses an expectile loss $\mathcal{L}_{TD}(V)=\mathbb{E}[\ell_2^\kappa(r(s,g)+\gamma\tilde V(s',g)-V(s,g))]$ for global Bellman consistency, supplemented by a finite-difference regularization:

$$\mathcal{L}_{Reg}(V)=\mathbb{E}\big[\big((V(s,g)-V(s',g))^2-\delta^2\big)_+\big],$$

which penalizes value differences between adjacent states only when they **exceed a threshold** $\delta$ (adaptively set as $\delta=1+(1-\gamma)|\bar V|$). The total loss is $\mathcal{L}(V)=\mathcal{L}_{TD}+w_c\mathcal{L}_{Reg}$. Unlike gradient norm regularization, finite-difference is computationally cheap yet effective—ablation shows it alone increases the success rate in pointmaze-giant from 35% to 95%.

**3. HIQL Hierarchy + Goal-driven Subgoal + Shared Value: Efficient hierarchy without extra heads**

To handle long horizons, hierarchical policies are employed. The high-level $\pi^h(w|s,g)$ generates a subgoal representation $w=\phi(g)$ (**depending only on the goal**, unlike HIQL's $\phi([g,s])$, effectively inserting an information bottleneck at the goal side). The low-level $\pi^l(a|s,w)$ executes actions. Both are trained using AWR. Crucially, advantages are calculated using the **same** LAN value function: $A^h=V(s_{t+k},g)-V(s_t,g)$ and $A^l=V(s_{t+1},s_{t+k})-V(s_t,s_{t+k})$. This eliminates the need for a separate high-level value network and out-performs independent heads.

### Loss & Training
The value phase jointly optimizes $\mathcal{L}_{TD}+w_c\mathcal{L}_{Reg}$. The policy phase uses AWR for both layers. The latent dimension is fixed at $d=64$, with task-adaptive $w_c$. Goal sampling follows a mixture of future and random distributions.

## Key Experimental Results

### Main Results
Average success rate (%) across 8 seeds for 22 OGBench datasets:

| Task Group | Dataset | HIQL | OTA | CGCIVL | QRL | **LAVL** |
|------------|---------|------|-----|--------|-----|----------|
| Pointmaze | giant-navigate | 46 | 72 | 65 | 68 | **91** |
| Antmaze | giant-stitch | 2 | 37 | 8 | 0 | **82** |
| Humanoidmaze| large-stitch | 28 | 57 | 20 | 3 | **72** |
| Cube | single-play | 15 | 9 | 23 | 5 | **83** |
| Scene | play | 38 | 30 | 56 | 5 | **88** |

LAVL achieves SOTA in 20 out of 22 datasets. The advantage over HIQL and OTA in manipulation tasks (Cube/Scene) is particularly significant (83 vs 15, 88 vs 30), validating that LAN does not suffer from the "incompatibility" issues of quasimetrics in manipulation.

### Ablation Study

| Configuration | pointmaze-giant Success Rate | Description |
|---------------|-----------------------------|-------------|
| LAVL Full | 95 | Complete model |
| LAVL w/o Continuity Reg | 35 | Long-horizon value oscillation leads to policy collapse |
| LAVL with IQE Value | Similar to LAN on antmaze-giant, scene <20 | IQE/MRN/Hilbert all <20% on manipulation |
| LAVL-HV (Indep. Value) | 80+ on maze, but weaker than LAVL | Unified LAN value is superior |

### Key Findings
- **Architecture is the bottleneck, not the objective**: GCIVL+IQE mitigates overgeneralization, while QRL+MLP is unstable. LAN is the only architecture that doesn't collapse on manipulation tasks (>80% vs <40% for others).
- **Horizon Robustness**: The relative performance drop from medium to giant mazes is only 9.6% for LAVL, compared to 23% for OTA, 58% for CGCIVL, and 75% for HIQL.
- **Stitching Robustness**: The average relative drop from navigate to stitch is 1.1% for LAVL, while others exceed 18%.
- **Hyperparameter Insensitivity**: Performance is stable across latent dimensions (16 to 256).

## Highlights & Insights
- Framed GCRL as a battleground for **inductive bias in value function architectures** rather than just learning objectives.
- The **asymmetric dual-encoder** is a simple yet powerful design: sacrificing metric properties for task robustness.
- **Finite-difference local regularization** is significantly cheaper than gradient norm regularization and acts as a potent stabilizer for long-horizon TD learning.
- **Unified value for both hierarchical layers** concludes the debate on independent high-level value heads: if the value function is sufficiently informative, unification is better and simpler.

## Limitations & Future Work
- Validated only on OGBench simulations; real-world robot control and more complex simulations remain to be explored.
- LAN discards the theoretical guarantees of quasimetrics. The gap between quasimetric properties and LAN's expressivity is not formally characterized.
- Hierarchical policy extraction remains a bottleneck in high-dimensional action spaces and long horizons.
- The adaptive threshold $\delta$ for continuity regularization is heuristic and may fail in online settings with rapid distribution shifts.

## Related Work & Insights
- **vs QRL (Wang+2023)**: QRL relies on the IQE architecture; LAVL shows QRL's maze performance stems from its architecture, but LAN's weaker inductive bias is more universal.
- **vs HIQL (Park+2023)**: Shares a hierarchical skeleton but replaces MLP value with LAN and simplifies subgoal dependency. Improves giant-stitch from 2% to 82%+.
- **vs OTA (Ahn+2025) / CGCIVL (Ke+2025)**: These improve value estimation through options or conservatism. LAVL's robustness suggests architectural changes yield larger gains.
- **vs Hilbert (Park+2024)**: Hilbert uses a single metric embedding; LAN defines the embedding as the value function itself and breaks symmetry.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](compositional_transduction_with_latent_analogies_for_offline_goal-conditioned_re.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[CVPR 2026\] MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning](../../CVPR2026/reinforcement_learning/mangobench_a_benchmark_for_multi-agent_goal-conditioned_offline_reinforcement_le.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
