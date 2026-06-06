---
title: >-
  [Paper Note] NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration
description: >-
  [ICML 2026][LLM Agent][Function Calling] NaviAgent decouples LLM tool invocation into two levels: "high-level four-way decision making + low-level path searching on a graph." A Tool World Navigation Model (TWNM)…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Function Calling"
  - "Tool Graph"
  - "Bilevel Planning"
  - "Heterogeneous Graph Transformer"
  - "Closed-loop Adaptation"
date: 2026-05-08
content_hash: 98509966b28f314e
---

# NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration

**Conference**: ICML 2026  
**arXiv**: [2506.19500](https://arxiv.org/abs/2506.19500)  
**Code**: None  
**Area**: LLM Agent / Tool Orchestration / Graph Representation Learning  
**Keywords**: Function Calling, Tool Graph, Bilevel Planning, Heterogeneous Graph Transformer, Closed-loop Adaptation  

## TL;DR
NaviAgent decouples LLM tool invocation into two levels: "high-level four-way decision making + low-level path searching on a graph." A Tool World Navigation Model (TWNM), trained with HGT, explicitly models structural and behavioral dependencies between tools. On ToolBench/API-Bank and 50 real-world RapidAPIs, it improves the Task Success Rate (TSR) by 4.3–18.2 percentage points over the strongest baselines while significantly reducing invocation steps.

## Background & Motivation
**Background**: Current mainstream function-calling agents (ReAct, ToolLLM, ToolNet, α-UMI, etc.) treat tools as a set of independent callable interfaces, where the LLM picks them one-by-one during reasoning. They either hardcode tool knowledge into model weights, pull a static graph from invocation logs, or rely on self-feedback strategies like ReAct/Reflexion.

**Limitations of Prior Work**: These solutions fail when tool scales expand to thousands and APIs change continuously. Chaining tools one-by-one leads to cumulative local errors. Static graphs cannot capture sparse multi-hop relationships. Dynamic strategies lack global structure, making it difficult to reuse tool chains for repetitive tasks.

**Key Challenge**: The difficulty of reconciling "structured but non-evolvable" (static dependency graphs) with "evolvable but unstructured" (self-feedback agents) leads to unreliability and lack of scalability in large-scale tool ecosystems.

**Goal**: Decomposed into two sub-problems: (1) Moving the planning layer back from "deciding the next specific API" to "deciding the next interaction action" to avoid overwhelming reasoning with tool combination complexity; (2) Creating an execution layer with a self-updating tool relationship graph based on real invocation feedback, which provides executable paths and reorganizes in real-time when APIs fail or semantic drift occurs.

**Key Insight**: The authors observe that real-world tools are not isolated but depend on each other through shared parameters and idiomatic invocation patterns. Explicitly encoding these dependencies into a heterogeneous graph transforms "picking the next tool" into "weighted path searching on a graph," while the graph itself is continuously updated by execution logs.

**Core Idea**: Isolate the LLM from tool combination complexity using a quaternary decision action space, offload combination difficulties to an evolvable tool graph, and refresh both the planning strategy and graph structure through an execution feedback loop.

## Method
The entire method centers on a quintuple $(\mathcal{H},\mathcal{O},\mathcal{G},\mathcal{A},F)$. History $\mathcal{H}$ consists of the last 3 observation-action pairs (found to be the trade-off between accuracy and efficiency). $\mathcal{O}$ is the current observation. $\mathcal{G}$ is the pruned tool subgraph. $\mathcal{A}$ comprises 4 optional actions. $F:\mathcal{H}\times\mathcal{O}\times\mathcal{G}\to\mathcal{A}$ is implemented by the LLM.

### Overall Architecture
NaviAgent features a dual-loop structure. The inner loop is the "planning-execution" cycle: upon receiving a user query, the LLM selects one of 4 actions (Direct Answer / Clarify Intent / Retrieve Toolchain / Execute Tool). For the latter two, it finds and executes an actionable path on the TWNM. The outer loop is the "graph-environment" loop: actual invocation results update the TWNM's edge weights and node states, which in turn affects future subgraph pruning. Consequently, the LLM always operates within a small action space, leaving combination challenges to the graph.

### Key Designs

1.  **Planning layer with four-dimensional decision-making**:
    - **Function**: Compresses "scheduling complex toolchains" into a lightweight 4-way decision, letting the LLM only judge whether to "speak, ask, retrieve, or execute."
    - **Mechanism**: History is represented by a sliding window $\mathcal{H}_t = \langle (o_{t-3},a_{t-3}),\dots,(o_{t-1},a_{t-1})\rangle$. The pruned tool subgraph $\mathcal{G}_{t-1}'$ from the previous step is serialized into a tree-like text and provided to the LLM. The decision function is $a_t = F(\mathcal{H}_t,\mathcal{O}_t,\mathcal{G}_{t-1}')$. During SFT, the loss is calculated only on the action generation segment: $\mathcal{L}_{\text{SFT}}=-\frac{1}{N}\sum_i \log p_\theta(a_t^*\mid \mathcal{H}_t,\mathcal{O}_t,\mathcal{G}_{\text{sub}})$.
    - **Design Motivation**: Traditional plan-then-execute requires the LLM to pre-arrange the entire sequence, which fails as tool count increases. Here, the LLM only selects the "next interaction mode," offloading combinatorial puzzles to graph search. Decoupling planning from execution allows for independent scaling.

2.  **TWNM: Heterogeneous Tool Graph + HGT Link Prediction**:
    - **Function**: Treats both APIs and parameters as nodes, uniformly modeling structural edges ("parameter → API / API → parameter") and behavioral edges ("API → API / parameter → parameter"). It infers future invocation pairs based on historical statistics.
    - **Mechanism**: Constructs a directed weighted graph $\mathcal{G}=(V,E,W)$ where edge weights $\tilde{w}_{ij} = N(v_i \to v_j)/N(v_j)$ reflect empirical invocation frequencies. A 2-layer multi-head HGT performs message passing. Attention scores $e_{uv}^{(k,r)} = (\mathbf{W}_Q^{(k,r)}\mathbf{h}_u')^\top(\mathbf{W}_K^{(k,r)}\mathbf{h}_v')/\sqrt{d_k} + \mathbf{b}_r^{(k)} + \tilde{w}_{uv}$ inject statistical weights directly into attention. The training objective is a curriculum-weighted sum $\mu_t = \mu_0 \gamma^t$ of cross-entropy $\mathcal{L}_{CE}$ with soft labels and an adaptive margin loss $\ell_{\text{margin}}(u,v)=\frac{1}{k}\sum_j [m_{uv}-s(u,v)^+ + s(u_j,v)^-]_+$, prioritizing accuracy first and then discriminative power.
    - **Design Motivation**: Parameter sharing is the strongest implicit dependency between tools. Explicitly modeling them as nodes is more reliable than "API similarity." HGT distinguishes node/edge types while utilizing statistical frequency as a structural prior, ensuring link prediction captures both semantic and behavioral information.

3.  **Closed-loop Evolution + Path Reorganization**:
    - **Function**: Allows the TWNM to follow changes in the real tool ecosystem (new tools, deprecated APIs, shifting patterns) and enables the agent to automatically find alternative paths upon execution failure.
    - **Mechanism**: Three graph maintenance mechanisms: incremental node access (new tools initialized with zero counts/weights); targeted subgraph pruning $\text{Prune}(v) \propto \lambda\sigma(f_{\text{fail}}(v)) + (1-\lambda)\sigma(f_{\text{freq}}(v)^{-1})$; and edge weight temporal propagation $\tilde{w}_{uv}^{(t)} = \eta \tilde{w}_{uv}^{(t-1)} + (1-\eta) N_{\text{succ}}^{\text{recent}}(u\to v)/N_{\text{succ}}^{\text{recent}}(v)$. Execution failures trigger three recovery modes: I/O equivalent replacement, upstream backtracking/rerouting, and subgraph switching. A theoretical result (Theorem 3.1) shows that in a fixed context, mechanism injection is equivalent to constraining the base policy to the feasible action set and normalizing: $\pi_{\text{inj}}(a\mid h)=\pi_0(a\mid h)\mathbf{1}\{a\in\mathcal{A}_{\text{feas}}(h)\}/\sum_{a'\in\mathcal{A}_{\text{feas}}(h)}\pi_0(a'\mid h)$, representing the minimal local correction in terms of KL projection.
    - **Design Motivation**: In reality, APIs change frequently; static graphs quickly become obsolete. Writing "failure/unreachability" back to the graph allows the LLM and the graph to evolve synchronously. This is equivalent to adding a "contextual feasibility" constraint to the policy at inference time, which is much cheaper than fine-tuning model weights.

### Loss & Training
Standard SFT is used for the LLM, backpropagating only through action generation. For the HGT, curriculum weighting (decay coefficient $\gamma\in(0,1)$) of cross-entropy and adaptive margin loss is used. TWNM updates and online inference are performed asynchronously to avoid blocking forward calls. Qwen2.5-14B was fine-tuned with 3,500+ carefully selected data points, with strict isolation between training and evaluation to prevent leakage.

## Key Experimental Results

### Main Results
Comparison of overall TSR / average steps for various backbone models on ToolBench (5k+ tools) (All column for Easy/Medium/Hard):

| Backbone | Method | TCR (%) | TSR (%) | Avg. Steps |
|----------|------|---------|---------|----------|
| Qwen2.5-14B | ToolNet | 49.7 | 28.0 | 6.53 |
| Qwen2.5-14B | NaviAgent | **61.6** | **35.8** | **4.38** |
| Qwen2.5-32B | α-UMI | 78.3 | 32.8 | 5.94 |
| Qwen2.5-32B | NaviAgent | **83.2** | **45.4** | **4.66** |
| DeepSeek-V3 | ToolNet | 76.6 | 44.9 | 6.02 |
| DeepSeek-V3 | NaviAgent | **97.0** | **55.2** | **4.60** |

Evaluation of 50 real-world RapidAPIs (7 domains, 303 queries):

| Backbone | Method | TSR (%) | Steps | Time (s) |
|----------|------|---------|------|----------|
| Qwen2.5-14B | ToolNet | 33.1 | 6.41 | 31 |
| Qwen2.5-14B | NaviAgent | **37.4** | 5.0 | 26 |
| Qwen2.5-32B | α-UMI | 42.4 | – | – |
| Qwen2.5-32B | NaviAgent | **54.4** | – | – |
| DeepSeek-V3 | NaviAgent | **64.6** | – | – |

### Ablation Study
| Configuration | TSR (Qwen2.5-14B, ToolBench All) | Description |
|------|----------------------------------|------|
| Full NaviAgent | 35.8 | Bilevel + TWNM + Closed-loop |
| Four-way decision only (No TWNM) | ~28 | ReAct with action constraints |
| Static Graph + Four-way decision | ~31 | No edge weight evolution |
| Full + SFT (14B) | 51.3 | Approaches 32B model (45.4) |

### Key Findings
- TWNM is the primary contributor for complex tasks: The authors report an average TSR improvement of 13.1 points on complex tasks. Moving from Easy to Hard, NaviAgent's relative decline is much smaller than baselines (37.5% on DeepSeek-V3 vs 57.1% for ToolNet and 50.6% for α-UMI).
- Injecting statistical invocation weights $\tilde{w}_{uv}$ into HGT attention recovers multi-hop dependencies better than just using semantic embeddings. Soft-label training is particularly important for sparse invocation logs.
- Closed-loop evolution allows small SFT-tuned models (14B) to approach the TCR/TSR of larger models (32B), indicating that tightening the action space reduces the marginal contribution of model size to success rates.

## Highlights & Insights
- Reframing "LLM selecting the next tool" into "LLM selecting the interaction mode + graph searching the path" is a useful decoupling: The action space drops from linear growth (tool scale) to a constant 4, which is crucial for scaling to tens of thousands of tools.
- Directly adding statistical edge weights to HGT attention logits is simpler than traditional "train-then-feed features" approaches, providing empirical priors to the model from the start.
- Three path recovery strategies (I/O equivalent replacement, upstream backtracking, subgraph switching) cover major real-world API failure modes, serving as an exemplar of substantiating "reflection" concepts through graph structures.
- The KL projection theorem, though characterizing only single-step reasoning, provides a clear inference-time explanation for "mechanism-injected" tool constraints: it is local normalization on the base policy rather than additional training.

## Limitations & Future Work
- The authors admit the theoretical results only cover single-step local corrections and do not provide global convergence proofs for processes like subgraph switching or rerouting that change the feasible set itself.
- While TWNM updates are asynchronous, HGT's 2-hop neighbor aggregation overhead and subgraph serialization length may become bottlenecks as tools scale from thousands to hundreds of thousands; strict complexity analysis is missing.
- The strongest gains in evaluations rely on large models like DeepSeek-V3; the 37.4% TSR of 14B on real APIs is still low for actual production deployment. The advantage of TWNM shrinks significantly in cold-start scenarios (nearly no behavioral edges), which is not fully discussed.
- Future directions: Introducing hierarchical graph abstraction (domain clustering) to reduce search costs; using RL instead of SFT for simultaneous online optimization of planning policies and TWNM edge weights.

## Related Work & Insights
- **vs ToolLLM**: ToolLLM uses DFSDT for plan-then-execute, where tool relationships remain implicit in the LLM's chain of thought. NaviAgent pulls tool relationships explicitly into a graph, freeing the planning layer from specific API combinations.
- **vs ToolNet**: ToolNet also maintains a dynamic invocation graph but only fits edge weights using invocation sequences, lacking parameter nodes and HGT attention. NaviAgent adds structural edges and couples the graph with four-way decisions, enabling more robust link prediction under sparse multi-hop data.
- **vs α-UMI**: α-UMI assigns planning/invocation/summarization to three lightweight LLMs. NaviAgent uses a single LLM for decisions in a smaller action space, offloading combination complexity to the graph, which is more engineering-efficient.
- **vs ControlLLM**: ControlLLM uses a static dependency graph for task decomposition, failing to adapt to API drift. NaviAgent updates the graph via execution feedback loops, unifying "structure" and "evolution."

## Rating
- Novelty: ⭐⭐⭐⭐ Decoupling the tool graph from four-way decisions plus closed-loop evolution is a clear system-level innovation. The TWNM structure is not aggressive, but the engineering integration is high.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five backbone models across two mainstream benchmarks and 50 real APIs, though lacks systematic analysis of graph scalability and cold-starts.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear, and theorems are formally presented. It is slightly regrettable that the core theory is hidden in the appendix, and descriptions of weight updates/recovery are somewhat brief.
- Value: ⭐⭐⭐⭐ Provides a reproducible engineering blueprint for "thousand-scale tool agents," with direct utility for production-side function calling stacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](../../ACL2026/llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](../../ACL2026/llm_agent/searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)
- [\[ICML 2026\] Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)

</div>

<!-- RELATED:END -->
