---
title: >-
  [Paper Note] GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving
description: >-
  [ICML 2026][LLM Efficiency][LLM Agent Serving] GraphFlow unifies multiple agent workflows into a single global operational DAG (wGraph). It generates task-specific subgraph workflows online using GNN+MLP and replaces tra…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "LLM Agent Serving"
  - "Workflow Graph"
  - "KV Cache Reuse"
  - "GNN Subgraph Generation"
  - "Topology-Aware State Management"
date: 2026-05-08
content_hash: 19f85d6ae14fbafd
---

# GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving

**Conference**: ICML 2026  
**arXiv**: [2605.22566](https://arxiv.org/abs/2605.22566)  
**Code**: To be confirmed  
**Area**: LLM Efficiency / Agent  
**Keywords**: LLM Agent Serving, Workflow Graph, KV Cache Reuse, GNN Subgraph Generation, Topology-Aware State Management

## TL;DR
GraphFlow unifies multiple agent workflows into a single global operational DAG (wGraph). It generates task-specific subgraph workflows online using GNN+MLP and replaces traditional independent workflow caching with a differential caching strategy ("Base KV + Sparse Prefix Residual + Path Pruning"). Across 5 reasoning/code/QA benchmarks, it achieves an average improvement of 4.95pp while compressing KV memory to approximately 1/4.

## Background & Motivation

**Background**: LLM agents increasingly rely on "workflows" for long-chain multi-step tasks—combining various atomic operations (tool calls, reasoning steps, verification modules) according to predefined sequences and control rules. Representative systems like MetaGPT, TaskWeaver, AFlow, and AgentKB typically maintain a workflow repository and retrieve the most similar template for execution based on the task description.

**Limitations of Prior Work**: The authors identify two significant engineering bottlenecks. First, template/retrieval-based construction is too "coarse-grained"—selecting the entire workflow as an indivisible unit fails to capture fine-grained correspondences between task requirements and internal process structures, leading to poor generalization for unseen or recombinable tasks. Second, during serving, KV cache is managed "independently per workflow." Since different workflows frequently reuse the same atomic operations (e.g., the same tool call or verification prompt), the KV states for the same operation are redundantly stored across multiple workflow copies, causing memory usage to grow linearly or even super-linearly with the number of workflows.

**Key Challenge**: The KV state of an operation depends on its prefix (stateful) to ensure attention context correctness. However, storing a separate copy for every (operation, prefix) pair leads to a "prefix combination explosion." Conversely, storing only the individual operation state (stateless) disrupts cross-step reasoning dependencies and causes significant performance degradation. Thus, a **trade-off exists between correctness (stateful) and scalability (sharing)**, and simple template assembly cannot amortize redundant storage within a shared structure like wGraph.

**Goal**: (1) Upgrade workflow construction from "template retrieval" to "task-driven subgraph selection on a shared operation graph." (2) Design a KV cache strategy on the shared operation graph that ensures both correctness and high reusability.

**Key Insight**: The authors observe that multiple workflows have significant overlap at the atomic operation level, and empirically, the KV matrices calculated for the same operation under different prefixes are highly similar—>75% of K terms and >70% of V terms differ only within a small threshold (Figure 3). This implies that KV states can be expressed as "Base KV + Sparse Residual."

**Core Idea**: Elevate both workflow "construction" and "state management" to a global operation graph (wGraph). For construction, use GNNs for conditional subgraph generation on the wGraph. For state management, eliminate redundant storage using "Base KV + Prefix Differential KV + High-frequency Path Pruning."

## Method

### Overall Architecture
In the offline phase, GraphFlow merges all existing workflows into a global directed acyclic graph $\mathcal{G}_{\text{op}}=(\mathcal{V}_{\text{op}},\mathcal{E}_{\text{op}})$, called **wGraph**. Each node $v_i$ represents an atomic operation (tool call / reasoning step / verification module), and each edge represents a valid execution dependency. Simultaneously, it precomputes the "prefix-free base KV cache" $\mathbf{KV}_{\text{base}}(v)$ for each node and trains the workflow generation model. In the online phase, for a user request $S$: a virtual task node $v_{\text{task}}$ is injected into the wGraph to form a task-conditioned graph. A task-specific subgraph $\mathcal{W}_c$ is generated as the workflow via GNN+MLP. Following the execution prefix of this subgraph, the base KV of the corresponding nodes is retrieved and merged with the sparse prefix residual $\Delta\mathbf{KV}$ to reconstruct the context-aware KV, which is then fed into the backbone LLM for execution.

### Key Designs

1.  **wGraph: Global Operation DAG with Virtual Task Node Conditioning**:
    - **Function**: Compress scattered workflow templates into a shared DAG, making "operation-level sharing" a first-class citizen for computation.
    - **Mechanism**: Merge identical atomic operations across workflows into the same node $v_i$ while preserving valid dependency edges to form the wGraph $\mathcal{G}_{\text{op}}$. For each new task, construct a task-conditioned graph $\mathcal{G}=(\mathcal{V}_{\text{op}}\cup\{v_{\text{task}}\},\,\mathcal{E}_{\text{op}}\cup\{(v_{\text{task}},v_i),(v_i,v_{\text{task}})\})$. Linking the task to all operation nodes via bidirectional edges allows task semantics to be injected into every candidate operation through message passing. Node features $\mathbf{x}_i\in\mathbb{R}^D$ encode functional semantics, language triggers, and internal execution schemas, while the task feature $\mathbf{x}_{\text{task}}\in\mathbb{R}^D$ is derived from the input query.
    - **Design Motivation**: Upgrade workflows from "indivisible retrieval units" to "subgraphs on wGraph." This explicitly represents cross-workflow sharing and unifies workflow generation and KV sharing within the same data structure, forming the core abstraction of the framework.

2.  **GNN+MLP Task-Adaptive Workflow Generation**:
    - **Function**: Given a task, "pick edges" from the wGraph to generate a task-specific subgraph $\mathcal{W}_c$, targeting $\mathcal{W}^*=\arg\max_{\mathcal{W}\subseteq\mathcal{G}_{\text{op}}}\mathbb{E}[f(S,\mathcal{W})]$.
    - **Mechanism**: First, a GNN layer learns node embeddings $\mathbf{H}=\mathrm{GNN}(\mathbf{X},\mathbf{A}|\Theta_{\text{GNN}})$ that fuse task context and structural dependencies. Then, for each candidate edge $(v_i,v_j)$ allowed in the wGraph, an MLP computes a task-aware compatibility score $s_{i,j}=\mathrm{MLP}(\mathrm{Concat}[\mathbf{h}_i,\mathbf{h}_j,\mathbf{h}_{\text{task}}]|\Theta_{\text{MLP}})\in[0,1]$, indicating the likelihood of adopting this dependency. Starting from $v_{\text{task}}$, high-scoring edges are selected greedily while enforcing structural validity (connectivity, DAG, reachable execution) to form an executable subgraph $\mathcal{W}_c$.
    - **Design Motivation**: Compared to "top-1 template retrieval," edge-level composition allows the model to reassemble operations based on the task. This enables the synthesis of reasonable flows for unseen tasks or those requiring branch reorganization, while unifying "what to do" and "in what order" into a conditional subgraph generation problem.

3.  **Differential KV cache + Effective Path Pruning (Topology-Aware State Management)**:
    - **Function**: Maintain stateful correctness while compressing the exponential overhead of "per-prefix independent KV storage" into "Base + Sparse Residual + High-frequency Paths."
    - **Mechanism**: Precompute $\mathbf{KV}_{\text{base}}(v)$ without prefixes for each operation $v$. For prefix paths $\mathcal{P}$ actually appearing in workflows, only sparse residuals $\Delta\mathbf{KV}(\mathcal{P},v)$ are stored. During execution, context-dependent KV is reconstructed via $\mathbf{KV}(\mathcal{P},v)=\mathbf{KV}_{\text{base}}(v)+\Delta\mathbf{KV}(\mathcal{P},v)$. An additional layer of **effective path pruning** uses execution statistics to identify high-frequency transitions in the wGraph, materializing residuals only for those. Rare or unreachable paths are not stored, falling back to on-the-fly computation if triggered.
    - **Design Motivation**: Empirical evidence (Figure 3) shows that KV states for the same operation under different prefixes are extremely similar (>75% K, >70% V terms near zero), making "Base + Sparse Differential" a nearly lossless compression. Since edge combinations in wGraph explode, path pruning converges storage size from "all potential paths" to the "actual working set," ensuring KV memory grows with effective execution trajectories rather than combinatorial complexity.

### Loss & Training
The main text provides the formal objective for inference: $\mathcal{W}^*=\arg\max_{\mathcal{W}}\mathbb{E}[f(S,\mathcal{W})]$, where $f$ is a task-level metric (success rate / accuracy) of the downstream agent. Specific training objectives, subgraph sampling (e.g., Gumbel-softmax), and GNN structural details are provided in Appendix B. Base KV is computed once offline; prefix residuals are driven by execution trajectory statistics, with path pruning determining which paths to materialize.

## Key Experimental Results

### Main Results

Setup: Three backbones (Qwen-2.5-7B, Llama-3.1-8B, Gemma-2-9B) across five benchmarks (GSM8K, MATH, HotpotQA, HumanEval, MBPP), compared with 7 baselines (Vanilla / MetaGPT / LLMCompiler / TaskWeaver / AgentKB / AutoFlow / AFlow). Metrics include Acc / F1 / pass@1 for tasks and P90 latency for efficiency.

| Backbone | Dataset | Metric | AFlow (SOTA baseline) | GraphFlow | Gain |
|----------|--------|------|------------------------|-----------|------|
| Qwen-2.5-7B | GSM8K | Acc | 89.2 | **92.1** | +2.9 |
| Qwen-2.5-7B | MATH | Acc | 72.1 | **76.4** | +4.3 |
| Qwen-2.5-7B | HumanEval | pass@1 | 78.1 | **86.2** | +8.1 |
| Qwen-2.5-7B | MBPP | pass@1 | 68.4 | **74.7** | +6.3 |
| Qwen-2.5-7B | HotpotQA | F1 | 67.5 | **70.4** | +2.9 |
| Llama-3.1-8B | HumanEval | pass@1 | 72.2 | **76.6** | +4.4 |
| Llama-3.1-8B | MATH | Acc | 47.5 | **52.6** | +5.1 |
| Gemma-2-9B | HumanEval | pass@1 | 75.4 | **82.5** | +7.1 |
| Gemma-2-9B | MBPP | pass@1 | 66.1 | **72.8** | +6.7 |

Regarding P90 latency, the authors report that the aggregated P90 across 5 benchmarks on Qwen-2.5-7B dropped from 14.06s (AFlow) to 12.25s, indicating that generated workflows are both more accurate and more streamlined.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Stateful KV (Upper Bound) | MATH Acc 53.8; GSM8K KV ≈ 50 GB; HotpotQA KV ≈ 85 GB | Independent caching per workflow; strongest correctness but memory explosion |
| **GraphFlow (Differential + Pruning)** | MATH Acc 52.6 (only 1.2pp drop); GSM8K KV ≈ 11 GB; HotpotQA KV ≈ 25 GB | ~1/4 memory; performance essentially aligned with stateful |
| Stateless KV | MATH Acc 39.4; HotpotQA F1 ≈ 58.6; KV 8–17 GB | Completely ignores prefix; significant drop in long-chain reasoning |
| GraphFlow w/o path pruning | GSM8K KV 15.0 → **11.5** GB; MBPP 9.9 → **7.2** GB; MATH/HotpotQA drops ≈ 5.3/4.2 GB | Differential alone is insufficient; pruning filters "potential but unused" edges |
| Concurrent Scaling (BS 10→50) | Stateful: 0.8 GB → > 2.4 GB; GraphFlow: Always < 0.5 GB | Base KV shared across concurrent requests; residuals are light, ensuring memory doesn't scale with concurrency |

### Key Findings
- **Feasibility of differential KV stems from structural observation**: Authors prove >75% of K and >70% of V prefix differences are near zero (Figure 3). This decouples "prefix dependency" from "memory redundancy"—prefixes only affect minimal KV terms, allowing precise compensation via sparse residuals with almost no performance loss.
- **Path pruning provides substantial gains**: For high-branching tasks like HotpotQA, pruning saves an additional ≈4.2 GB. This indicates that wGraph contains many "semantically reachable but practically unused" transitions; pruning ensures KV memory scales with the effective execution set rather than combinatorial complexity.
- **Structured generation + reuse provides simultaneous benefits**: On HumanEval, performance increased from 78.1% to 86.2% (+8.1pp) while latency decreased, proving task-adaptive subgraphs aren't "more accurate but more expensive" but rather "more accurate by trimming unnecessary operations."

## Highlights & Insights
- **Elevating "Workflow" from Template to Graph**: Unlike treating each workflow as a separate retrieval unit (MetaGPT/TaskWeaver), merging all workflows into an operation DAG allows the abstraction "workflow = wGraph subgraph." This enables GNN conditional subgraph generation for construction and cross-workflow KV sharing for state management. Addressing two problems with one abstraction is the most ingenious aspect.
- **Precise Targeting for Differential KV**: While much LLM serving work (PagedAttention, prefix caching) reduces KV redundancy at the token level, GraphFlow operates at the operation level and proves the resulting differentials are extremely sparse—transforming "theoretical differential potential" into "practical lossless engineering."
- **Transferable Design Thinking**: The combination of "Shared Base + Sparse Differential + Path Pruning" naturally transfers to any "highly overlapping multi-stage LLM call scenarios" like RAG pipelines or tool-use sequences. As long as units are nodes and combinations are graphs, differential KV can be directly reused.

## Limitations & Future Work
- **Maintenance Cost of wGraph**: The paper assumes a pre-existing set of atomic operations and dependencies. However, how these "primitives" are automatically extracted from historical workflows, the appropriate node granularity, and how wGraph expands for new scenarios remains unaddressed. Granularity that is too coarse loses reuse benefits; too fine makes wGraph huge and generation difficult to train.
- **Training Signals for GNN Generator**: The task-level reward $\arg\max\mathbb{E}[f]$ is naturally non-differentiable. The text defers training details to Appendix B, leaving the reader unable to judge whether RL, counterfactual supervision, or imitation learning was used, resulting in a high reproducibility barrier.
- **Accumulation of Differential KV Errors**: MATH shows a 1.2pp drop (53.8 → 52.6). It remains to be verified if small errors in residual sparsity accumulate in extremely long tool-use chains; currently, few of the 5 benchmarks represent "very long horizon" tasks.
- **Dependency on Execution Statistics for Pruning**: If statistics are sparse during early deployment, pruning may be too aggressive or conservative. Robustness during cold starts or distribution shifts was not discussed.

## Related Work & Insights
- **vs MetaGPT / TaskWeaver / AFlow**: These methods express workflows as SOPs or calculation graphs but treat each unit independently. GraphFlow merges them into a wGraph for subgraph generation, making structural "sharing" a first-class computable object.
- **vs LLMCompiler**: LLMCompiler uses DAGs for tool dependencies, but they are constructed per-task without cross-task sharing. GraphFlow's wGraph is global and cross-task, expanding the reuse scope from "within a request" to "across all requests."
- **vs PagedAttention / prompt cache / prefix caching**: Classic KV reuse assumes identical text prefixes at the token/page level. GraphFlow relaxes this to "prefix similarity ⇒ differential sparsity" at the operation level, better fitting the actual reuse patterns of agentic workflows.
- **vs AutoFlow / AgentKB**: These induce structural guidance from history but remain template-based. GraphFlow encodes this guidance into a graph and matches it with serving optimizations, bridging "structural understanding" and "structural acceleration."

## Rating
- Novelty: ⭐⭐⭐⭐ Elevating workflows to a global operation graph and optimizing serving via differential KV is a clean and rare combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backbones, five benchmarks, and three sets of ablations (differential/pruning/concurrency) covering both correctness and system performance.
- Writing Quality: ⭐⭐⭐⭐ Motivation, abstraction, and component integration are clear. Formulas and figures are well-coordinated, though Appendix reliance is heavy.
- Value: ⭐⭐⭐⭐ Highly relevant for industrial agent serving: 4× KV memory compression + average 4.95pp gain, with memory usage nearly independent of concurrency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)

</div>

<!-- RELATED:END -->
