---
title: >-
  [Paper Note] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline GCRL] By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$, combined with continuity regularization and a HIQL-style hierarchical structure, LAVL achieves SOTA on 20 out of 22 OGBench datasets. It increases the success rate on long-range tasks like giant maze and stitch datasets from nearly zero to over 80%.
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline GCRL"
  - "Value Function Architecture"
  - "Latent Representation Alignment"
  - "Quasimetric"
  - "Hierarchical Policy"
date: 2026-05-08
content_hash: d5e16826fbe7999c
---

# Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.25740](https://arxiv.org/abs/2605.25740)  
**Code**: https://github.com/oh-lab/LAVL.git (Available)  
**Area**: Reinforcement Learning / Offline Goal-Conditioned RL / Representation Learning  
**Keywords**: Offline GCRL, Value Function Architecture, Latent Representation Alignment, Quasimetric, Hierarchical Policy

## TL;DR
By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$, combined with continuity regularization and a HIQL-style hierarchical structure, LAVL achieves SOTA on 20 out of 22 OGBench datasets. It increases the success rate on long-range tasks like giant maze and stitch datasets from nearly zero to over 80%.

## Background & Motivation

**Background**: Offline goal-conditioned RL (GCRL) aims to learn policies that can reach arbitrary targets from fixed trajectory data. A primary approach involves learning a goal-conditioned value function $V(s,g)$ and extracting the policy using advantage-weighted regression or quasimetric constrained optimization. Recent works like HIQL, GCIVL, QRL, CGCIVL, and OTA focus on improving value function estimation.

**Limitations of Prior Work**: In long-horizon sparse reward scenarios, value functions learned via TD are highly unreliable: (i) success rates drop sharply as the horizon increases (e.g., antmaze-giant, humanoidmaze-giant); (ii) stitch-type datasets (requiring reward propagation across short trajectory segments) are particularly difficult; (iii) some solutions require an additional high-level value network, increasing computational and tuning costs.

**Key Challenge**: The authors identify the failure mode of V-learning as **overgeneralization**. MLP-parameterized $V(s,g)$ tends to assign high values to states near $g$ in Euclidean distance, leading to "value leakage" across boundaries (e.g., the other side of a wall) despite large temporal distances. This stems from the **inductive bias of the value function architecture** rather than the learning objective itself. Existing quasimetric architectures (MRN, IQE) partially alleviate this but often fail and exhibit instability in robotic manipulation tasks.

**Goal**: To find a value function architecture that suppresses overgeneralization while remaining stable across distinct task types (maze and manipulation), seamlessly integrating into a hierarchical policy to handle long horizons.

**Key Insight**: Through visualizations and cross-ablations (e.g., "GCIVL × IQE" vs. "QRL × MLP"), the authors attribute performance gaps to architecture rather than objectives. They further find that strict quasimetric constraints impose an overly strong inductive bias that hurts performance in tasks with inconsistent geometric structures like manipulation. A **weakened, learnable** latent space distance is more universal than rigid quasimetrics.

**Core Idea**: Parameterize $V(s,g)$ as the latent space Euclidean distance between state and goal embeddings, but **deliberately keep state and goal encoders unshared**. This asymmetry breaks strict metricity in exchange for cross-task stability.

## Method

### Overall Architecture
LAVL is an IVL (implicit V-learning) style offline GCRL algorithm composed of three components: (1) The LAN value network, which parameterizes $V(s,g)$ as the negative Euclidean distance between outputs of two asymmetric encoders; (2) A joint loss of TD and local continuity regularization to ensure global Bellman consistency while suppressing local oscillations under long horizons; (3) A HIQL-style hierarchical policy where the sub-goal representation $w=\phi(g)$ depends only on the goal itself, reusing the same LAN value function to provide advantages for both high-level and low-level policies. The pipeline takes offline trajectories $\mathcal{D}=\{(s_t,a_t,s_{t+1})\}$ and outputs a hierarchical policy pair $(\pi^h, \pi^l)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["Offline Trajectories D = {(s, a, s′)}"]
    subgraph LAN["LAN Value Network (Asymmetric Dual-Encoder)"]
        direction TB
        ES["State Encoder φ_S(s)"]
        EG["Goal Encoder φ_G(g)<br/>Independent, not shared with φ_S"]
        ES --> V["V(s,g) = −‖φ_S(s) − φ_G(g)‖<br/>Latent Negative Euclidean Distance"]
        EG --> V
    end
    IN --> LAN
    LAN --> REG["TD + Local Continuity Regularization<br/>Expectile TD for Global Bellman Consistency<br/>Finite-difference for Local Spike Suppression"]
    subgraph HIER["HIQL Hierarchical Policy + Shared Value"]
        direction TB
        PH["High-level π^h → Sub-goal w = φ(g)<br/>Goal-only Information Bottleneck"]
        PL["Low-level π^l(a | s, w)"]
        PH --> PL
    end
    REG -->|Same LAN value used for high/low level advantage (AWR)| HIER
    HIER --> OUT["Hierarchical Policy (π^h, π^l)"]
```

### Key Designs

**1. Latent Alignment Network (LAN): Blocking overgeneralization via architecture**

LAN addresses architectural overgeneralization where MLP-parameterized $V(s,g)$ generalizes based on raw state-space Euclidean distance. LAN uses two **independent, non-shared** networks $\varphi_S:\mathcal{S}\to\mathbb{R}^d$ and $\varphi_G:\mathcal{G}\to\mathbb{R}^d$ to embed states and goals, defining value as:

$$V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2.$$

Generalization follows "latent space alignment" rather than raw geometry. The asymmetry (not sharing encoders or enforcing strict quasimetrics) provides a middle ground: it retains the benefits of latent distance induction without the rigid triangle inequality constraints that cause failure in manipulation tasks.

**2. TD + Local Continuity Regularization: Balancing global consistency and local smoothness**

While LAN handles the generalization mode, sparse rewards and long horizons can still lead to sharp value oscillations. LAVL uses an expectile loss $\mathcal{L}_{TD}(V)=\mathbb{E}[\ell_2^\kappa(r(s,g)+\gamma\tilde V(s',g)-V(s,g))]$ for global consistency, coupled with a finite-difference regularizer:

$$\mathcal{L}_{Reg}(V)=\mathbb{E}\big[\big((V(s,g)-V(s',g))^2-\delta^2\big)_+\big],$$

which penalizes value differences between adjacent states only when they exceed a threshold $\delta$ (adaptively set as $\delta=1+(1-\gamma)|\bar V|$). Total loss: $\mathcal{L}(V)=\mathcal{L}_{TD}+w_c\mathcal{L}_{Reg}$. This stabilizer significantly improves performance in long-horizon tasks like pointmaze-giant.

**3. HIQL Hierarchy + Goal-driven Sub-goals + Shared Value**

To handle long horizons, LAVL employs a hierarchy. The high-level $\pi^h(w|s,g)$ generates a sub-goal representation $w=\phi(g)$ (depending **only on the goal**, unlike the $\phi([g,s])$ in the original HIQL). The low-level $\pi^l(a|s,w)$ executes actions. Crucially, advantages for both levels are derived from the **same** LAN value function: $A^h=V(s_{t+k},g)-V(s_t,g)$ and $A^l=V(s_{t+1},s_{t+k})-V(s_t,s_{t+k})$. This eliminates the need for an independent high-level value head.

## Key Experimental Results

### Main Results
Average success rates (%) across 8 seeds on 22 OGBench datasets:

| Task Family | Dataset | HIQL | OTA | CGCIVL | QRL | **LAVL** |
|-------------|---------|------|-----|--------|-----|----------|
| Pointmaze   | giant-navigate | 46 | 72 | 65 | 68 | **91** |
| Antmaze     | giant-stitch   | 2 | 37 | 8 | 0 | **82** |
| Humanoidmaze| large-stitch   | 28 | 57 | 20 | 3 | **72** |
| Cube        | single-play    | 15 | 9 | 23 | 5 | **83** |
| Scene       | play           | 38 | 30 | 56 | 5 | **88** |

LAVL achieved the best performance in 20 out of 22 datasets. Its advantage in manipulation tasks (Cube/Scene) is particularly significant compared to HIQL and OTA.

### Ablation Study

| Configuration | pointmaze-giant Success Rate | Note |
|---------------|------------------------------|------|
| LAVL Full     | 95 | Complete model |
| LAVL w/o Reg  | 35 | Long horizon value oscillations lead to policy collapse |
| LAVL with IQE | Similar in antmaze, <20 in scene | IQE/MRN/Hilbert fail in manipulation tasks (<20%) |
| LAVL-HV (Indep. Value) | 80+ in maze | Proves that a unified LAN value is superior |

### Key Findings
- **Architecture is the bottleneck, not the objective**: Replacing IQE/MRN/Hilbert/MLP with LAN was the only configuration consistently successful across both maze and manipulation tasks.
- **Horizon Robustness**: The relative drop in success rate from medium to giant mazes was only 9.6% for LAVL, compared to 23% for OTA and 75% for HIQL.
- **Stitching Robustness**: The drop from navigate to stitch datasets was 1.1% for LAVL, while others dropped by 18%+.
- **Hyperparameter Insensitivity**: Performance remained stable across latent dimensions from 16 to 256.

## Highlights & Insights
- Framed the value function architecture as the primary site of **inductive bias in GCRL**, demonstrating via cross-ablations that architecture choices outweigh objective functions.
- The **asymmetric dual-encoder** design is simple yet effective, challenging the intuition that strict quasimetricity is necessary for optimal value functions.
- The **finite-difference local regularization** is computationally cheaper than gradient-norm based methods and acts as a potent stabilizer for long-horizon TD learning.
- Unified value for both hierarchical levels proves that if the value representation is sufficiently informative, a single network can effectively serve as both a high-level signal and a low-level sub-goal coupler.

## Limitations & Future Work
- Validated only on OGBench simulations; real-world robot control and more complex simulations are not yet covered.
- LAN lacks the theoretical guarantees of quasimetrics; the gap between the quasimetric nature of the optimal $V$ and LAN's expressivity is not formally characterized.
- Policy extraction remains a bottleneck in high-dimensional action spaces under long horizons.
- The adaptive threshold $\delta$ for continuity regularization is empirical and may fail in online settings with rapid distribution shifts.

## Related Work & Insights
- **vs QRL (Wang et al., 2023)**: QRL relies on IQE and constrained optimization. LAVL shows that QRL's maze performance stems from its architecture, and LAN's weaker bias is more generalizable.
- **vs HIQL (Park et al., 2023)**: LAVL improves the same hierarchical backbone by replacing MLP values with LAN and simplifying the sub-goal bottleneck.
- **vs OTA (Ahn et al., 2025) / CGCIVL (Ke et al., 2025)**: These works focus on options or conservatism. LAVL's robustness suggests that architectural changes yield higher returns than objective modifications.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](compositional_transduction_with_latent_analogies_for_offline_goal-conditioned_re.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](../../ICLR2026/reinforcement_learning/occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)
- [\[ICML 2026\] Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning](direction-conditioned_policies_via_compositional_subgoal_scoring_for_online_goal.md)
- [\[ICML 2026\] Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning](unifying_value_alignment_and_assignment_in_cross-domain_offline_reinforcement_le.md)

</div>

<!-- RELATED:END -->
