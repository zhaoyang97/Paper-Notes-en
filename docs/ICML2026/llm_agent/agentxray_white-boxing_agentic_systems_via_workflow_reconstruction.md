---
title: >-
  [Paper Note] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction
description: >-
  [ICML 2026][LLM Agent][Agentic Workflow Reconstruction] The authors propose a new task, AWR, which aims to reconstruct an equivalent white-box workflow from a black-box agent system. They use MCTS to search the agent pri…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Agentic Workflow Reconstruction"
  - "MCTS"
  - "Red-Black Pruning"
  - "Black-Box Explanation"
  - "Multi-Agent"
date: 2026-05-08
content_hash: 1e1bd6eca05c082b
---

# AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2602.05353](https://arxiv.org/abs/2602.05353)  
**Code**: Not explicitly provided in the paper (no clear link)  
**Area**: LLM Agent / Interpretability / Combinatorial Optimization  
**Keywords**: Agentic Workflow Reconstruction, MCTS, Red-Black Pruning, Black-Box Explanation, Multi-Agent

## TL;DR
The authors propose a new task, AWR, which aims to reconstruct an equivalent white-box workflow from a black-box agent system. They use MCTS to search the agent primitive sequence space, combined with a Red-Black pruning method based on dynamic score coloring to balance depth and breadth, achieving interpretable white-box reconstruction in five real-world domains.

## Background & Motivation

**Background**: LLM agent/multi-agent systems (MAS) solve complex tasks via role specialization and tool use (e.g., ChatDev, MetaGPT). However, high-performance agents in deployment are often black boxes—their prompts, agent topology, and toolchains are not visible.

**Limitations of Prior Work**: Users can only observe inputs and outputs, with no insight into the decision process; debugging, modification, and security auditing are hindered. Existing agent interpretability research either targets single-step LLM reasoning or requires white-box access (e.g., model distillation), and cannot handle pure black-box APIs.

**Key Challenge**: The internal state space of black-box systems is enormous (agent roles × models × thought patterns × toolsets × order). Even if input-output pairs can be sampled, exhaustive search is infeasible; classic distillation requires model parameters, which is not applicable.

**Goal**: Define a new task, Agentic Workflow Reconstruction (AWR): synthesize an explicit, interpretable, and editable white-box workflow using only $(\tau, o^\ast)$ input-output pairs, such that its execution output matches the black box as closely as possible.

**Key Insight**: (1) Linearity Hypothesis—most real-world agent systems, when executed, are serialized into an action-observation sequence (even if designed as a graph), so the search space can be limited to primitive chains of length $\le L_{\max}$. (2) Output similarity is used as a proxy metric, circumventing the challenge of determining true functional equivalence.

**Core Idea**: Formulate AWR as a combinatorial optimization over discrete primitive sequence space, and use MCTS + Red-Black pruning to efficiently approximate the optimal workflow under a token budget.

## Method

### Overall Architecture
Input is a dataset $\mathcal{D}=\{(\tau_i, o_i^\ast)\}$, each from the black-box system $\mathcal{M}_{\text{black}}$. The unified primitive space $\Omega$ is defined: each primitive $p=\langle \rho, \mu, \pi, T_{\text{local}}\rangle$ (role, base model, thought pattern, toolset), covering both pure reasoning agents and tool-augmented agents. A workflow is represented as a linear sequence $\mathbf{s}=[s_1,\dots,s_L]$, $L \le L_{\max}$. The objective is
$$
\mathbf{s}^\ast = \arg\max_{\mathbf{s}} \mathbb{E}_{(\tau,o^\ast)}[\mathrm{Sim}(\Phi(\mathbf{s},\tau), o^\ast)]
$$
where $\mathrm{Sim}$ is a task-specific proxy metric (AST for code, cosine for text). AgentXRay uses MCTS: each node is a workflow prefix, each edge appends a primitive; Red-Black coloring decides whether a node prefers "deepening (refine)" or "branching (expand)".

### Key Designs

1. **Unified Primitive Space + Linearity Hypothesis**:

    - **Function**: Unifies heterogeneous agents/tools/single-agent and multi-agent systems under a single search unit, reducing search complexity from graph topology $O(2^{|\Omega|^2})$ to sequence $O(|\Omega|^{L_{\max}})$.
    - **Mechanism**: Each primitive is $\langle$role, model, thought pattern, local tools$\rangle$; pure reasoning agents have $T_{\text{local}}=\emptyset$, tool-augmented agents have $T_{\text{local}}\ne\emptyset$. Drawing on MacNet (Qian 2025), which shows multi-agent DAGs are topologically sorted at execution, and that ReAct/WebArena interactions are naturally ordered traces, the search space is restricted to linear sequences.
    - **Design Motivation**: Pure graph topology search is infeasible even for moderate $\Omega$; "behavioral fidelity" (input-output matching) only requires reproducing observable sequences, not internal topology, so linearization is a task-aligned pruning.

2. **MCTS Search Loop (with UCB + Early-Stopping Rollout)**:

    - **Function**: Handles $\mathrm{Sim}$, a sparse/delayed reward signal observable only near-complete workflows.
    - **Mechanism**: Each iteration samples a $(\tau, o^\ast)$; from the root, a path is selected, and at the expansion node, a sample-rollout is performed: the sequence is sampled to $L_{\max}$ and the workflow is executed to obtain output $o$; if execution fails, $r=0$, else $r=\mathrm{Sim}(o, o^\ast)$; backpropagate updates to $N(v), Q(v)$ along the path. Child selection at each node uses UCB to balance exploration/exploitation.
    - **Design Motivation**: Unlike "exhaustive $|\Omega|^{L_{\max}}$" search, MCTS amortizes search cost via statistical sampling; UCB is robust in heterogeneous action spaces (different roles, models, tools); early stopping prevents wasting tokens on invalid primitives.

3. **Red-Black Pruning (Score-Driven Dynamic Coloring)**:

    - **Function**: Under fixed iteration/token budget, automatically decides which nodes to deepen (depth refine) and which to branch (width expand), mitigating combinatorial explosion.
    - **Mechanism**: Before each iteration, ColorTree recolors the current tree: Red nodes indicate current choices are "stable" (high score + sufficient visits), so child selection uses UCB to go deeper; Black nodes are insufficiently explored, so new child nodes are prioritized for expansion. The search loop (Algorithm 1) consists of color-guided descent (Line 9), early-stopping rollout (Lines 11–13), and reward backpropagation (Line 22).
    - **Design Motivation**: Standard MCTS on large $\Omega$ often gets stuck in "too wide to go deep" or "deep in a bad branch"; Red-Black dynamically quantifies "confidence to refine current path" via scores, guiding search resources to subtrees with both potential and depth, enabling deeper and better search within the same iteration budget.

### Loss & Training
This is a non-gradient method with no training phase. The "loss" is negative proxy similarity $-\mathrm{Sim}(\Phi(\mathbf{s},\tau), o^\ast)$, and the "optimizer" is MCTS + Red-Black Pruning. Each workflow execution calls a real LLM API (e.g., GPT/Gemini), so budget is measured by iteration count $N$ and total tokens.

## Key Experimental Results

### Main Results
Five domains, five target systems: software development (ChatDev), data analysis (MetaGPT), education (TeachMaster), 3D modeling (ChatGPT GPT-5.2 API), scientific computing (Gemini 3 Pro). Proxy similarity uses Static Functional Equivalence (SFE).

| Area / Target System | Metric | AgentXRay Avg. SFE | Notes |
|---------------------|--------|--------------------|-------|
| Software Dev / ChatDev | AST-based | High SFE (overall mean 0.426) | Reconstructed executable dev workflow |
| Data Analysis / MetaGPT | AST + Text | Same | Multi-agent collaboration linearized and reproduced |
| Education / TeachMaster | Text Similarity | Same | Teaching process restored |
| 3D Modeling / ChatGPT | Output Comparison | Same | Single agent + tool call chain |
| Scientific Computing / Gemini 3 Pro | Output Comparison | Same | Long-chain scientific reasoning also approximated |
| Overall | — | 0.426 SFE | Significantly higher than baseline without pruning |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| Full AgentXRay (MCTS + Red-Black) | Best SFE, 8–22% fewer tokens | Pruning enables deeper search under same budget |
| No Red-Black Pruning (pure MCTS) | Lower SFE + more tokens | Node selection lacks score guidance, resources spread evenly |
| No Linearity Hypothesis (graph topology search) | Infeasible | $O(2^{|\Omega|^2})$ search explosion |
| Different $L_{\max}$ | Medium length optimal | Too short lacks expressiveness, too long increases rollout failure |
| Different scoring functions (Sim only vs Sim + depth) | Multi-dimensional scoring better | "Proxy quality + search depth" joint scoring makes Red-Black more sensitive |

### Key Findings
- Red-Black pruning is the key switch for token efficiency: with the same iteration budget, pruning enables deeper workflow levels, achieving better fidelity.
- The Linearity Hypothesis yields usable fidelity across five distinct domains (including true multi-agent ChatDev, MetaGPT), validating that "execution-time topological order" is the main observable signal in black boxes.
- Even for target systems like GPT-5.2 or Gemini 3 Pro (closed APIs), AgentXRay can approximate behavior with IO access only; this shows white-box reconstruction is effective for real "commercial black boxes".
- The reconstructed workflow is editable—users can replace roles/tools for downstream adaptation; this is fundamentally different from model distillation.

## Highlights & Insights
- Transforms interpretability into "behavioral equivalence + structural white-box" at the observable level, avoiding the impossible task of accessing model parameters; this is a pragmatic paradigm for "interpretability".
- Unified primitive definition covers both agents and tools, making the search space conceptually valid for single-agent + tool-use systems—broadening applicability beyond multi-agent systems.
- Red-Black pruning makes "prune or not" a node-level dynamic decision (score-dependent), more robust than static threshold pruning, and transferable to any sparse-reward LLM agent search.
- Using SFE as a proxy metric circumvents the undecidability of "true functional equivalence", a practical compromise for open-ended multi-file outputs; this approach is reusable for code synthesis/agent evaluation.

## Limitations & Future Work
- The Linearity Hypothesis is an upper bound: systems that truly depend on concurrency/asynchronous multi-agent (synchronous dialogue, cyclic feedback) may have essential behaviors missed by linear sequences.
- The evaluation metric SFE is a proxy; for some tasks, AST matching or text similarity cannot distinguish true functional differences—potentially misleading MCTS.
- Rollout requires real workflow execution, with each run incurring multiple LLM calls; after $N$ searches, token cost is substantial. The current 8–22% savings are relative; absolute cost remains high.
- The primitive space $\Omega$ requires pre-prepared role/model/pattern/tool candidates; if the black box uses tricks not in $\Omega$, they can never be reconstructed.

## Related Work & Insights
- **vs Model Distillation**: Distillation requires parameter access and produces a black-box small model; AWR only needs IO and produces an editable white-box workflow.
- **vs MacNet / Multi-Agent Graph Structures (Qian 2025)**: MacNet trains new agents with DAGs; this work does the opposite—infers an equivalent linear workflow from a black-box agent.
- **vs ReAct / WebArena and other interactive agents**: Those works design agents; this work reverses agents via observation, providing interpretable representations.
- **vs MCTS-for-LLM approaches (e.g., ToT, AgentTrek)**: They use MCTS to search a single reasoning path; this work uses MCTS to search the "agent construction graph" itself, a higher-level abstraction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The AWR task definition is new, and Red-Black score pruning is a substantive improvement to MCTS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five domains + real closed APIs covered, but details per domain could be expanded, and statistical significance could be strengthened.
- Writing Quality: ⭐⭐⭐⭐ Task motivation, unified primitives, and Linearity argument are all clear; Algorithm 1 is directly reproducible.
- Value: ⭐⭐⭐⭐ Directly aids interpretability/control/auditability of agent deployment, and could be a practical tool for "reverse engineering closed agent APIs".

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](../../ICML2025/optimization/metaagent_automatically_constructing_multi-agent_systems_based_on_finite_state_m.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
