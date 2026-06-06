---
title: >-
  [Paper Note] Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM
description: >-
  [ICML 2026][Multi-Agent][ADMM unrolling] Sheaf-ADMM formulates the multi-agent coordination problem as an end-to-end differentiable ADMM unrolling—each agent observes only a local patch…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "ADMM unrolling"
  - "cellular sheaf"
  - "multi-agent consensus"
  - "sheaf Laplacian"
  - "local view fusion"
date: 2026-05-08
content_hash: c79630afbd8956a5
---

# Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM

**Conference**: ICML 2026  
**arXiv**: [2605.31005](https://arxiv.org/abs/2605.31005)  
**Code**: To be confirmed  
**Area**: Multi-Agent / Differentiable Optimization / Geometric Deep Learning  
**Keywords**: ADMM unrolling, cellular sheaf, multi-agent consensus, sheaf Laplacian, local view fusion

## TL;DR
Sheaf-ADMM formulates the multi-agent coordination problem as an end-to-end differentiable ADMM unrolling—each agent observes only a local patch, independently solves an ADMM subproblem ($\bm x$-update), negotiates consensus via "edge space projections" defined by a cellular sheaf ($\bm z$-update), and accumulates divergence using dual variables $\bm u$. Agents collaboratively reach correct global solutions on maze pathfinding, MNIST, and Sudoku, where the inference path features analyzable primal/consensus/dual states—making it more controllable than MPNNs.

## Background & Motivation

**Background**: Standard neural architectures are monolithic—a single large network processes the entire input. However, intelligence in nature is often collective—a group of agents with only local views collaboratively solve global tasks (e.g., ant colonies, neuronal clusters). Related existing architectures include sheaf neural networks (Bodnar 2022), neural cellular automata (Mordvintsev 2020), and recurrent MPNNs (Gilmer 2017).

**Limitations of Prior Work**: (1) MPNN-style architectures assign a single hidden state to each agent, mixing "local decisions" with "external negotiation"; (2) message passing consists of arbitrarily learned nonlinear functions, making behavior uninterpretable; (3) they typically require agents to reach consensus over the entire state vector, which is too rigid (e.g., adjacent maze regions only need boundary consistency, not internal consistency); (4) existing sheaf-constrained ADMM (Hanks 2025b) uses fixed manual sheaves for multi-agent linear control and lacks differentiable learning.

**Key Challenge**: To enable a group of "information-incomplete" agents to truly collaborate on global tasks, a framework must: (a) explicitly separate local decisions from global negotiation; (b) provide flexible "local consensus" semantics (different agents agree on different aspects); and (c) be end-to-end learnable. Existing architectures fail to achieve all three.

**Goal**: (1) Develop an end-to-end differentiable multi-agent coordination framework; (2) use cellular sheaves to define flexible semantics for "what needs to be consistent"; (3) maintain primal/consensus/dual states for each agent to ensure interpretability; (4) validate on standard DL tasks.

**Key Insight**: Leverage the natural decomposition of ADMM—consensus form ADMM splits a global problem into $N$ local subproblems plus a consensus projection. Replace the "full-state consensus" constraint with a cellular sheaf, requiring agents to reach consensus only after projecting onto the edge stalk space—providing a mathematically rigorous and geometrically intuitive "flexible consensus" semantic.

**Core Idea**: Unrolled ADMM + learnable cellular sheaf—each agent solves an $\bm x$-update (a convex subproblem parameterized by a neural encoder) $\rightarrow$ performs $\bm z$-update via sheaf-Laplacian diffusion (projection onto ker($\bm F$)) $\rightarrow$ executes $\bm u$-update to accumulate divergence. The entire pipeline is differentiable and trained via end-to-end backpropagation.

## Method

### Overall Architecture

The input $\bm D \in \mathbb{R}^{H \times W \times C_{in}}$ is divided into $N$ overlapping patches (agents):
1. **Encoding**: Each patch $\bm d_i$ passes through a shared encoder to produce $\bm Q_i, \bm q_i$, parameterizing the convex subproblem for agent $i$.
2. **ADMM Unrolling ($K$ steps)**:
   - $\bm x$-update: Agent $i$ solves $\arg\min_{\bm x_i} \tfrac{1}{2}\bm x_i^\top \bm Q_i \bm x_i + \bm q_i^\top \bm x_i + \tfrac{\rho}{2}\|\bm x_i - \bm z_i + \bm u_i\|^2$ (closed-form).
   - $\bm z$-update: Sheaf-Laplacian diffusion $\bm z^{t+1} = \bm z^t - \eta \bm L_\mathcal{F} \bm z^t$ projects onto ker($\bm F$).
   - $\bm u$-update: $\bm u^{k+1} = \bm u^k + \bm x^{k+1} - \bm z^{k+1}$.
3. **Decoding**: The final $\bm x$ combined with local patches passes through a decoder to obtain local predictions, which are aggregated into the global output.

### Key Designs

1. **Three-State Separation (primal $\bm x$ / consensus $\bm z$ / dual $\bm u$):**
    - **Function**: Clearly distinguishes between an agent's local decision, the negotiation target, and historical divergence.
    - **Mechanism**: $\bm x_i$ is the agent's local optimal solution (pulled toward $\bm z - \bm u$ by the augmented Lagrangian), $\bm z_i$ is the current consensus target (satisfying sheaf constraints), and $\bm u_i$ accumulates past divergence. The augmented Lagrangian term $\|\bm x - \bm z + \bm u\|^2$ links these three.
    - **Design Motivation**: Architectures like MPNN collapse all information into a single hidden state, effectively squeezing "what I think," "what we agreed on," and "where we disagreed" together. Separating these states in ADMM allows for analyzable inference dynamics (e.g., visualizing $\bm z$ to see consensus convergence or $\bm u$ to identify conflict zones).

2. **Cellular Sheaf for Flexible Consensus Semantics:**
    - **Function**: Allows agents to reach consensus only on low-dimensional edge stalks rather than the full state.
    - **Mechanism**: Each edge $e = (i, j)$ has an edge stalk $\mathbb{R}^{d_e}$ ($d_e < d_v$). Restriction maps $\bm F_{i \to e}, \bm F_{j \to e} \in \mathbb{R}^{d_e \times d_v}$ project agent states onto the edge stalk. Global consensus is defined as $\bm F_{i \to e} \bm x_i = \bm F_{j \to e} \bm x_j$. The sheaf Laplacian $\bm L_\mathcal{F} = \bm F^\top \bm F$ measures total divergence: $\bm x^\top \bm L_\mathcal{F} \bm x = \sum_e \|\bm F_{i \to e} \bm x_i - \bm F_{j \to e} \bm x_j\|^2$.
    - **Design Motivation**: Adjacent maze regions only need boundary consistency, not internal consistency—forcing full consensus wastes model capacity. Sheaves ensure consistency occurs only in low-dimensional subspaces actually required by the task; restriction maps learn "which aspects we should agree on."

3. **Unrolled ADMM + Inexact $\bm z$-update:**
    - **Function**: Treats ADMM as a differentiable recurrent layer for end-to-end learning.
    - **Mechanism**: ADMM is fixed to $K$ steps. Each $\bm z$-update uses a few steps of sheaf-Laplacian diffusion (inexact), and gradients are backpropagated through the whole sequence. Incomplete diffusion acts as a smoother—quickly removing high-frequency local divergence while preserving low-frequency global structure.
    - **Design Motivation**: In large sparse systems, the condition number of $\bm L_\mathcal{F}$ can be high, making perfect convergence expensive. A few diffusion steps suffice to align high-frequency components; this "few-step high-frequency elimination" inductive bias is a specific advantage of ADMM in multi-agent settings.

## Key Experimental Results

### Sudoku (Core Reasoning Task)

| Method | Solve Rate | Parameters |
|--------|------------|------------|
| MPNN (parameter-matched) | 32% | ~500K |
| Recurrent Transformer | 41% | ~500K |
| **Sheaf-ADMM** | **78%** | ~500K |

On Sudoku, a task with global logic constraints, Sheaf-ADMM significantly outperforms MPNN because the sheaf's local consistency constraints naturally match the row/column/block constraint structure of Sudoku.

### Maze Pathfinding

| Difficulty | MPNN | **Sheaf-ADMM** |
|------------|------|----------------|
| 8×8 | 89% | **96%** |
| 16×16 | 67% | **88%** |
| 32×32 | 23% | **64%** |

As maze size increases, MPNN performance degrades faster, while Sheaf-ADMM generalizes better due to the global convergence properties of ADMM.

### MNIST Robustness (Key Findings)

| Test Distribution | CNN baseline | **Sheaf-ADMM** |
|-------------------|--------------|----------------|
| Standard MNIST | 99.1 | 98.8 |
| Rotated MNIST | 73.2 | **89.4** |
| Translated MNIST | 81.7 | **93.1** |
| Noisy MNIST | 85.5 | **91.8** |

Performance is slightly lower on clean tests (−0.3), but significantly more robust under distribution shifts (+15% on rotation)—proving that local-view decomposition + sheaf consensus provides a stronger inductive bias.

### Interpretability (ADMM Three-State)

Figure 3 of the paper shows that in the maze task:
- $\bm x$ (primal): Early on, each agent independently proposes a local path.
- $\bm z$ (consensus): Through ADMM iterations, these merge into a coherent global path.
- $\bm u$ (dual): Identifies regions with "high historical divergence"—often the turning points of the maze.

This visualization is impossible with MPNNs (which have only one hidden state).

### Key Findings
- **The more global coordination a task requires, the greater the advantage of Sheaf-ADMM**: Sudoku (strong constraints +46%) > Maze 32×32 (+41%) > MNIST clean (−0.3). This indicates the framework provides significant benefits primarily for true coordination tasks.
- **Robustness to distribution shift is a natural byproduct**: Local-view decomposition makes the model independent of global positional priors, enabling generalization to rotation/translation.
- **Inexact ADMM is sufficient**: A few diffusion steps are enough to eliminate high-frequency divergence; perfect convergence is not required, saving significant computation.
- **Restriction maps learned by the sheaf are interpretable**: Visualizations show agents learn "which dimensions to negotiate on."

## Highlights & Insights
- **Three-state separation + sheaf consensus is a truly novel inductive bias**: Previous MPNNs mixed all information, and full-state consensus was a crude prior. This work separates "my decision," "our agreement," and "our disagreement," and formalizes "what to agree on" via sheaves—a structural prior with fewer degrees of freedom that matches the structure of coordination tasks.
- **Optimization-derived updates vs. arbitrary learned updates**: Sheaf-Laplacian diffusion and ADMM proximal updates are derived from optimization, not arbitrary functions. This provides a mathematical explanation for "why the update happens," allowing for analysis and intervention.
- **Interpretability + Controllability**: After training, $\bm x, \bm z, \bm u$ can be independently visualized and perturbed—MPNNs lack this level of analyzability, which is vital for safety-critical multi-agent applications.
- **Sheaf as a general framework**: Cellular sheaves are far more flexible than graph Laplacians, capable of expressing heterogeneous consensus semantics. This paper demonstrates a complete pipeline for applying sheaves to DL coordination.

## Limitations & Future Work
- The number of ADMM unrolling steps $K$ is fixed and does not adapt to sample difficulty; adaptive iterations or early termination could be considered.
- The condition number of $\bm L_\mathcal{F}$ remains an issue in large sparse systems; few-step diffusion is a workaround, but preconditioning might be better.
- Global sharing of restriction maps may be insufficient for highly heterogeneous agents (e.g., mixed modalities).
- Validation is limited to structured prediction tasks (grid-organized agents); expansion to arbitrary graph topologies needs further testing.
- Training cost—unrolling $K$ steps leads to memory bloat during backpropagation; implicit differentiation might be a better choice but requires redesign.

## Related Work & Insights
- **vs. MPNN / GNN**: MPNNs use arbitrary learned message functions with uninterpretable behavior; Sheaf-ADMM uses optimization-derived updates with three-state separation.
- **vs. Sheaf Neural Networks (Bodnar 2022)**: That work uses the sheaf Laplacian for diffusion-based message passing; Sheaf-ADMM goes further by using the sheaf to constrain the ADMM consensus with a primal-dual structure.
- **vs. Hanks 2025b**: That work uses fixed manual sheaves for multi-agent linear control; this work learns the sheaf and arbitrary convex subproblems in an end-to-end differentiable manner.
- **vs. Neural Cellular Automata**: NCA uses arbitrary learned updates for emergence; Sheaf-ADMM uses optimization structures to guarantee convergence.
- **Insight**: Turning any "distributed optimization algorithm" into a differentiable neural layer via unrolling is a fertile direction; ADMM, PGD, and Frank-Wolfe can all be approached this way.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ End-to-end learned sheaf-constrained ADMM is a truly novel multi-agent coordination architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sudoku + Maze + MNIST robustness cover reasoning and classification, though scenarios are still grid-structured.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous (complete derivation of sheaf + ADMM), with intuitive figures; the argument for three-state interpretability is clear.
- Value: ⭐⭐⭐⭐ Provides theoretical and practical insights for multi-agent RL, robot coordination, and distributed inference; interpretability is significant for safety-critical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoOT: Learning to Coordinate In-Context with Coordination Transformers](coot_learning_to_coordinate_in-context_with_coordination_transformers.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](../../ACL2026/multi_agent/explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](../../ACL2026/multi_agent/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)

</div>

<!-- RELATED:END -->
