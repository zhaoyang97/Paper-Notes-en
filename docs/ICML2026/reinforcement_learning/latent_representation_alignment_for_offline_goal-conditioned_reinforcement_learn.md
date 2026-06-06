---
title: >-
  [Paper Note] Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline GCRL] By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline GCRL"
  - "Value Function Architecture"
  - "Latent Representation Alignment"
  - "Quasimetric"
  - "Hierarchical Policy"
date: 2026-05-08
content_hash: 3544f5aa64a1a4ca
---

# Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.25740](https://arxiv.org/abs/2605.25740)  
**Code**: https://github.com/oh-lab/LAVL.git (Available)  
**Area**: Reinforcement Learning / Offline Goal-Conditioned RL / Representation Learning  
**Keywords**: Offline GCRL, Value Function Architecture, Latent Representation Alignment, Quasimetric, Hierarchical Policy

## TL;DR
By explicitly parameterizing the goal-conditioned value function as the **negative Euclidean distance in an asymmetric latent space** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$ and combining it with continuity regularization and a HIQL hierarchical structure, LAVL achieves SOTA results on 20 out of 22 OGBench datasets, increasing success rates for long-horizon tasks like giant mazes and stitch datasets from nearly zero to 80%+.

## Background & Motivation

**Background**: Offline goal-conditioned RL (GCRL) aims to learn a strategy that can reach any specified goal from fixed trajectory data. A mainstream approach involves learning a goal-conditioned value function $V(s,g)$ first, then extracting the policy via advantage-weighted regression or quasimetric constrained optimization. Recent works like HIQL, GCIVL, QRL, CGCIVL, and OTA focus on how to estimate this value function.

**Limitations of Prior Work**: In long-horizon sparse reward scenarios, value functions learned via TD are highly unreliable: (i) success rates drop sharply as the horizon extends (e.g., antmaze-giant, humanoidmaze-giant); (ii) stitch-type datasets (requiring the connection of short trajectory fragments) are particularly difficult; (iii) some solutions require learning an additional high-level value network, increasing computation and tuning effort.

**Key Challenge**: The authors identify the failure mode of "why V is poorly learned" as **overgeneralization**. MLP-parameterized $V(s,g)$ tends to assign high values to states that are Euclidean-close to $g$, causing value "leakage" to the other side of a wall where temporal distance is actually large. The root cause is the **inductive bias of the value function architecture**, not the learning objective itself. Existing quasimetric architectures (MRN, IQE) partially alleviate this but fail on robotic manipulation tasks due to unstable behavior.

**Goal**: To find a value function architecture that suppresses overgeneralization while remaining stable across both maze and manipulation tasks, and seamlessly integrate it into a hierarchical policy to handle long horizons.

**Key Insight**: Through visualizations and cross-ablations of "GCIVL × IQE / QRL × MLP," the authors attribute performance differences to architecture rather than objectives. They further find that strict quasimetric constraints are too strong an inductive bias, hurting tasks with inconsistent geometric structures like manipulation. A **weakened, learnable** latent space distance is more universal than hard quasimetrics.

**Core Idea**: Define $V(s,g)$ as the latent space Euclidean distance from state embedding to goal embedding, but **intentionally keep the state and goal encoders unshared**. This asymmetry breaks the strict metric property in exchange for cross-task stability.

## Method

### Overall Architecture
LAVL is an IVL (implicit V-learning) style offline GCRL algorithm consisting of three parts: (1) The value network **LAN**, which parameterizes $V(s,g)$ as the negative Euclidean distance between two asymmetric encoder outputs; (2) A joint loss of **TD + local continuity regularization** to ensure global Bellman consistency while suppressing local oscillations in long horizons; (3) A **HIQL-style hierarchical policy** where sub-goal representations depend only on the goal itself $w=\phi(g)$, reusing the same LAN value function for both high and low-level advantages without an extra high-level value head. The pipeline takes offline trajectories $\mathcal{D}=\{(s_t,a_t,s_{t+1})\}$ and outputs a hierarchical policy $(\pi^h, \pi^l)$.

### Key Designs

1.  **Latent Alignment Network (LAN)**:
    - **Function**: Acts as a new goal-conditioned value function architecture to fundamentally suppress overgeneralization.
    - **Mechanism**: Uses two independent networks $\varphi_S:\mathcal{S}\to\mathbb{R}^d$ and $\varphi_G:\mathcal{G}\to\mathbb{R}^d$ to embed states and goals into a latent space, defining $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$. Values generalize based on "latent space alignment" rather than "raw state space Euclidean distance."
    - **Design Motivation**: Pure MLPs cause value leakage across walls; strict quasimetrics (MRN/IQE) satisfy triangle inequalities which work for mazes but fail on manipulation. Asymmetric dual encoders retain the benefits of latent distance induction without imposing triangle inequalities, ensuring stability across diverse tasks. Ablations show LAN consistently outperforms or matches MLP/MRN/IQE/Hilbert parameterizations.

2.  **TD + Local Continuity Regularization**:
    - **Function**: Stabilizes value learning over long horizons and prevents sharp local oscillations.
    - **Mechanism**: The TD part uses an expectile loss $\mathcal{L}_{TD}(V)=\mathbb{E}[\ell_2^\kappa(r(s,g)+\gamma\tilde V(s',g)-V(s,g))]$, plus a finite-difference regularizer $\mathcal{L}_{Reg}(V)=\mathbb{E}[((V(s,g)-V(s',g))^2-\delta^2)_+]$. This punishes value differences between adjacent states only when they **exceed a threshold** $\delta$. Total loss: $\mathcal{L}(V)=\mathcal{L}_{TD}+w_c\mathcal{L}_{Reg}$, with $\delta=1+(1-\gamma)|\bar V|$.
    - **Design Motivation**: Sparse rewards and long horizons leave the Bellman term under-constrained locally, leading to value spikes. Unlike gradient norm regularization which requires $\nabla_s V$, finite-difference is computationally cheap. This contributes significantly to success rate gains in giant mazes.

3.  **HIQL Hierarchy + Goal-Driven Sub-goals + Shared Value**:
    - **Function**: Integrates LAN into a hierarchical framework for long horizons while eliminating the high-level value head.
    - **Mechanism**: The high-level $\pi^h(w|s,g)$ generates a sub-goal representation $w=\phi(g)$ (**depending only on the goal**, unlike HIQL's $\phi([g,s])$), and the low-level $\pi^l(a|s,w)$ executes actions. In LAN terms, $V(s,g)=-\|\varphi_S(s)-\varphi_G(\phi(g))\|_2$. Both policy levels are trained via AWR, calculating advantages from the **same** $V$: $A^h=V(s_{t+k},g)-V(s_t,g)$ and $A^l=V(s_{t+1},s_{t+k})-V(s_t,s_{t+k})$.
    - **Design Motivation**: Using a unified value function performs better than independent high-level values, suggesting LAN provides both informative signals for the high-level and proper coupling for the low-level. This also reduces hyperparameter tuning.

### Loss & Training
The value phase jointly optimizes $\mathcal{L}_{TD}+w_c\mathcal{L}_{Reg}$. Both policy levels use AWR. Latent dimension is fixed at $d=64$ across all experiments. Goal sampling uses a mix of future and random distributions.

## Key Experimental Results

### Main Results
Average success rate (%) across 8 seeds for 22 OGBench datasets:

| Task Group | Dataset | HIQL | OTA | CGCIVL | QRL | **LAVL** |
|------------|---------|------|-----|--------|-----|----------|
| Pointmaze  | giant-navigate | 46 | 72 | 65 | 68 | **91** |
| Antmaze    | giant-stitch | 2 | 37 | 8 | 0 | **82** |
| Humanoidmaze| large-stitch | 28 | 57 | 20 | 3 | **72** |
| Cube       | single-play | 15 | 9 | 23 | 5 | **83** |
| Scene      | play | 38 | 30 | 56 | 5 | **88** |

LAVL achieves the best performance in 20 out of 22 datasets. The advantage over HIQL and OTA is particularly massive in manipulation tasks (Cube/Scene), confirming that LAN avoids the "incompatibility" issues of quasimetrics.

### Ablation Study

| Configuration | pointmaze-giant Success Rate | Description |
|---------------|-----------------------------|-------------|
| Full LAVL | 95 | Full model |
| LAVL w/o Continuity Reg | 35 | Value oscillations lead to policy collapse |
| LAVL with IQE Value | antmaze-giant similar to LAN, scene <20 | IQE/MRN/Hilbert all <20% on manipulation |
| LAVL-HV (Indep. High-level V) | 80+ on maze | Unified LAN value is superior |

### Key Findings
- **Architecture is the bottleneck, not the objective**: Replacing IQE/MRN/Hilbert/MLP with LAN is the only way to avoid collapse on manipulation tasks (>80% vs <40%).
- **Horizon Robustness**: Success rate drop from medium to giant mazes is only 9.6% for LAVL, compared to 23% for OTA and 75% for HIQL.
- **Stitch Robustness**: Average drop from navigate to stitch is only 1.1% for LAVL, whereas others drop by 18%+.
- **Hyperparameter Insensitivity**: Performance is stable for latent dimensions from 16 to 256.

## Highlights & Insights
- **The "Value function architecture is the primary battleground for GCRL inductive bias" framing** is impactful—shifting the focus from objectives (TD vs contrastive) to architecture.
- **Asymmetric dual encoders** are a simple yet overlooked design: giving up metric properties allows for cross-task robustness, challenging the intuition that quasimetrics are necessary.
- **Finite-difference local regularization** is much cheaper than gradient norm regularization and provides a massive single-point contribution (35% $\to$ 95% on giant mazes).
- **Unified value for both levels** settles the debate on whether high-level values should be independent; if the value is informative enough, unification is better.

## Limitations & Future Work
- Verified only on OGBench simulations; real-world robot control is not covered.
- LAN discards quasimetric theoretical guarantees; the gap between LAN expressivity and optimal quasimetric properties is not formally characterized.
- Hierarchical policy extraction remains a bottleneck in long-horizon, high-dimensional action spaces.
- The adaptive threshold $\delta$ for continuity regularization is somewhat heuristic and might fail in online settings with rapid distribution shifts.

## Related Work & Insights
- **vs QRL (Wang+2023)**: QRL relies on IQE + constrained optimization. LAVL shows QRL's maze advantage comes from IQE, but LAN's weaker bias is more universal.
- **vs HIQL (Park+2023)**: Uses a similar hierarchical backbone but replaces MLP with LAN and simplifies the sub-goal representation.
- **vs OTA (Ahn+2025) / CGCIVL (Ke+2025)**: These improve value estimation via options or conservatism; LAVL’s robustness suggests changing architecture yields higher returns than changing objectives.
- **vs Hilbert Representations (Park+2024)**: While Hilbert uses a metric embedding as a downstream representation, LAN defines the embedding as the value function itself and breaks symmetry.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICML 2026\] Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](compositional_transduction_with_latent_analogies_for_offline_goal-conditioned_re.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICML 2026\] Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning](unifying_value_alignment_and_assignment_in_cross-domain_offline_reinforcement_le.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)

</div>

<!-- RELATED:END -->
