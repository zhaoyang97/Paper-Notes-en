---
title: >-
  [Paper Note] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators
description: >-
  [AAAI 2026][LLM Agent][Workflow Automation] This paper proposes A2Flow, a framework that automatically extracts reusable abstract execution operators from expert data via a three-stage pipeline (case generation → functio…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Workflow Automation"
  - "Abstraction Operators"
  - "MCTS"
  - "Operator Memory Mechanism"
  - "Embodied Task Generalization"
date: 2026-05-08
content_hash: 87c12df6a8a80b4f
---

# A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators

**Conference**: AAAI 2026
**arXiv**: [2511.20693](https://arxiv.org/abs/2511.20693)  
**Code**: [https://github.com/pandawei-ele/A2FLOW](https://github.com/pandawei-ele/A2FLOW)  
**Area**: LLM Agent
**Keywords**: Workflow Automation, Abstraction Operators, MCTS, Operator Memory Mechanism, Embodied Task Generalization

## TL;DR

This paper proposes A2Flow, a framework that automatically extracts reusable abstract execution operators from expert data via a three-stage pipeline (case generation → functional clustering → deep extraction), replacing manually predefined operators. Combined with an operator memory mechanism that accumulates intermediate outputs to assist node decision-making, A2Flow outperforms AFLOW and other state-of-the-art methods across 8 benchmarks while reducing resource consumption by 37%.

## Background & Motivation

**LLM Agent workflow design relies on manual effort**: Current LLM Agents depend on carefully hand-crafted agentic workflows (structured sequences of LLM calls), whose design and iteration are costly and severely limit scalability.

**Existing automated methods still require predefined operators**: AFLOW formulates workflow optimization as a code search problem via MCTS, but its core building blocks (Ensemble, Review, Revise, etc.) must still be manually designed for each task.

**Predefined operators limit generalization**: Hand-crafted operators introduce domain bias, resulting in poor transferability to open-world tasks (embodied control, WebShop, etc.) and providing no guarantee of optimality in the search space.

**Subsequent work inherits the same limitations**: DebFlow introduces debate and reflection mechanisms but retains AFLOW's predefined operators, still suffering from redundant operators and additional overhead; MermaidFlow proposes domain-aware evolutionary operators but still depends on predefined initialization and generalizes poorly to embodied scenarios.

**Insufficient information flow between operators**: In AFLOW's sequential execution design, each operator $o_k$ depends only on the output of the preceding step $o_{k-1}$, lacking global context sharing, which limits reasoning quality on complex tasks.

**Core motivation**: A fully automated operator extraction approach is needed to adaptively discover compact, generalizable execution operators from raw expert data, while a memory mechanism is needed to compensate for insufficient information transfer between operators.

## Method

### Overall Architecture

A2Flow consists of two main modules: **Self-Adaptive Abstraction Operators** (three-stage operator extraction) and an **Operators Memory Mechanism**. After extraction, the adaptive operator set is injected into AFLOW's MCTS search framework for workflow optimization. The entire pipeline requires no manually predefined operators.

### Stage 1: Case-based Initial Operator Generation

- **Function**: For each sample in the validation set (20% of expert data), an LLM is prompted to generate a case-level initial operator set $O^{(e)} = \{o_{i,j} = E(C_i, P_e, M)\}$.
- **Mechanism**: An extraction prompt $P_e$ is designed for each case $C_i$, instructing the LLM to decompose the problem-solving process into reusable execution units in Python class form, where each operator is a basic block with single input/output and no intermediate jumps.
- **Design Motivation**: Inducing operators directly from actual task examples avoids manual prior assumptions and ensures semantic alignment between the operators and the target task.

### Stage 2: Operator Clustering and Preliminary Abstraction

- **Function**: An LLM performs functional clustering over the initial operator set $O^{(e)}$, merging semantically similar operators to yield a preliminary abstract operator set $O^{(a)} = \mathcal{C}(O^{(e)}, P_a, M)$.
- **Mechanism**: A clustering prompt $P_a$ instructs the LLM to identify functionally overlapping operators across cases, prune unnecessary operators, and ensure concise, non-redundant operator names.
- **Design Motivation**: The initial operator set is large and highly redundant; including it directly in search would cause search space explosion and inefficiency, necessitating prior aggregation.

### Stage 3: Deep Extraction for Abstract Execution Operators

- **Function**: Multi-path parallel generation combined with Long CoT prompting further abstracts the clustered operators into a final task-aware abstract execution operator set $O^{(t)}$.
- **Mechanism**: $m=6$ independent reasoning paths $\{\mathcal{P}_p\}_{p=1}^{6}$ are generated, each refined over three iterations (initial generation $o_1$ → CoT refinement $o_2$ → second CoT refinement $o_3$); an LLM aggregation function $\mathcal{A}_t$ then merges results across paths into the final operator set.
- **Design Motivation**: Drawing on the multi-path voting idea of self-consistency, temperature-controlled sampling produces diverse candidates; combined with deep reasoning, this ensures operators are both compact and generalizable (e.g., the final extraction yields three general-purpose operators: Planner, Executor, and Validator).

### Reflection Mechanism

- **Function**: After the LLM generates Python code for each operator, an executor performs syntax and executability checks; failures trigger LLM-based reflection and regeneration.
- **Mechanism**: A self-correction + error-feedback closed loop.
- **Design Motivation**: Operators in code form must be executable; iterative error correction improves extraction reliability.

### Operators Memory Mechanism

- **Function**: During workflow search, each node's execution is augmented with a memory space $\mathcal{M}_k$ that stores all historical operator outputs; the current operator is computed as $o_k = f_k(\text{input}_k, P_k, \mathcal{M}_{k-1})$ and the memory is updated as $\mathcal{M}_k = \mathcal{M}_{k-1} \cup \{o_k\}$.
- **Mechanism**: This breaks AFLOW's restriction where each operator only observes the immediately preceding output, allowing every operator to access the full historical context.
- **Design Motivation**: In complex tasks, later steps may require information from much earlier steps rather than just the previous one; global memory enables more accurate reasoning and better generalization.

### Automated Workflow Optimization

The AFLOW MCTS search framework (initialization → selection → expansion → evaluation → backpropagation) is retained, but the adaptive operator set $\{O^{(t)}\}$ and operator memory $\mathcal{M}$ are injected into the search process. The final optimization objective is $W^* = \mathcal{S}(W_0, \{O^{(t)}\}, G, D_V, \mathcal{M})$, simultaneously optimizing both workflow structure and operator representations.

## Loss & Training

- This work involves no conventional gradient-based training; optimization is performed via MCTS search, where the evaluation function $G$ averages the performance of candidate workflows over multiple executions on the validation set.
- The "training" in the operator extraction stage consists of prompt engineering combined with iterative LLM refinement (3-step CoT refinement × 6 parallel paths + reflection-based regeneration).
- Validation/test split: 20%/80%, with random seed fixed at 42.
- The optimizer uses Claude-3.5-sonnet; executors use GPT-4o-mini / GPT-4o / DeepSeek-v3; the operator generator uses DeepSeek-v3.

## Key Experimental Results

### Main Results: General Benchmarks (GPT-4o-mini as Executor)

| Method | HotpotQA | DROP | HumanEval | MBPP | GSM8K | MATH | Avg. |
|--------|----------|------|-----------|------|-------|------|------|
| IO | 68.1 | 68.3 | 87.0 | 71.8 | 92.7 | 48.6 | 72.8 |
| CoT | 67.9 | 78.5 | 88.6 | 71.8 | 92.4 | 48.8 | 74.7 |
| CoT SC (5-shot) | 68.9 | 78.8 | 91.6 | 73.6 | 92.7 | 50.4 | 76.0 |
| MultiPersona | 69.2 | 74.4 | 89.3 | 73.6 | 92.8 | 50.8 | 75.1 |
| ADAS | 64.5 | 76.6 | 82.4 | 53.4 | 90.8 | 35.4 | 67.2 |
| AFLOW | 73.5 | 80.6 | 90.9 | 83.4 | 93.5 | 56.2 | 79.6 |
| **A2Flow** | **74.1** | **85.1** | **92.4** | **85.0** | **93.8** | **58.5** | **81.5** |

A2Flow achieves the best performance on 5 of 6 benchmarks, outperforming AFLOW by an average of 1.9 percentage points; gains of +4.5% on DROP and +4.1% on MATH are particularly notable.

### Embodied / Game Benchmarks (DeepSeek-v3 as Executor)

| Method | ALFWorld Seen | ALFWorld UnSeen | TextCraft | Avg. |
|--------|---------------|-----------------|-----------|------|
| ReAct | 22.0 | 22.9 | 33.0 | 25.9 |
| AFLOW | 17.1 | 26.6 | 53.0 | 32.2 |
| **A2Flow** | **25.0** | **31.3** | **59.0** | **38.4** |

A2Flow achieves an average improvement of 19.3% over AFLOW on embodied and game tasks, validating the generalization advantage of adaptive operators in open-world settings.

### Ablation Study (MATH Benchmark)

| Variant | Score | ΔScore (%) |
|---------|-------|------------|
| w/o Abstraction Operators & Memory | 56.2 | — |
| w/o Operators Memory | 53.9 | -4.1 |
| **Full A2Flow** | **58.5** | **+4.1** |
| w/o Initial Operators | 49.6 | -11.7 |
| w/o Operator Clustering | 54.5 | -3.0 |
| w/o Deep Extraction | 51.6 | -8.2 |

Initial operator generation contributes the most (-11.7%), followed by deep extraction (-8.2%) and the memory mechanism (-4.1%).

## Key Findings

1. **Adaptive operators consistently outperform predefined ones**: Fully automatically extracted operators surpass manually designed operators (AFLOW) across all 8 benchmarks without requiring domain expert knowledge.
2. **Significant generalization to open-world tasks**: On embodied and game tasks such as ALFWorld and TextCraft, where AFLOW and ReAct previously generalize poorly, A2Flow achieves a 19.3% improvement, demonstrating that adaptive operators can discover task-appropriate workflow structures from limited training data.
3. **Pareto frontier advantage**: Cost analysis shows that workflows discovered by A2Flow enable weaker models to outperform stronger models on the performance-cost frontier, reducing resource consumption by 37%.
4. **All three stages are indispensable**: Ablation results show that initial generation (-11.7%), deep extraction (-8.2%), clustering (-3.0%), and memory (-4.1%) each make irreplaceable contributions.
5. **Limited gains on code execution tasks**: Improvements on HumanEval/MBPP are modest, as predefined operators already encode a strong prior in the form of Python interpreter invocation.

## Highlights & Insights

- **"Operators themselves can be automatically searched"**: A2Flow pushes the granularity of automation from workflow structure down to the operators themselves, achieving a more thorough end-to-end automation—not only is the optimal workflow topology discovered automatically, but the building blocks are as well.
- **Effective use of multi-path CoT + self-consistency**: Six independent reasoning paths × three rounds of iterative refinement, followed by cross-path aggregation, effectively performs an ensemble-like operation in the operator abstraction space, improving extraction robustness.
- **The memory mechanism is simple yet effective**: Accumulating historical outputs in a single set yields a 4.1% gain, confirming that propagating only the immediately preceding output in AFLOW is indeed an information bottleneck.
- **Intuitive case study**: The ALFWorld case study clearly illustrates the full extraction chain from raw task samples to ObserveEnvironment/CreatePlan → clustering → Planner/Executor/Validator.

## Limitations & Future Work

1. **Operator extraction depends on a strong LLM**: The extraction stage uses DeepSeek-v3; the quality of aggregation and refinement is bounded by LLM capability, and more complex open-domain tasks may require stronger reasoning models.
2. **The search framework still follows AFLOW's MCTS**: Innovation is concentrated in operator extraction; the search strategy itself is not improved, and more efficient search algorithms may exist.
3. **The memory mechanism is overly simplistic**: Simple set-based accumulation does not distinguish information importance; as the number of steps grows, context window pressure increases, and no forgetting or compression mechanism is in place.
4. **Validation set is only 20%**: For tasks with small sample sizes (e.g., ALFWorld with only 33 validation samples), extracted operators may overfit.
5. **Cross-modal and cross-task transfer not validated**: Whether operators extracted from one task can be directly transferred to another remains unexplored.
6. **Fixed 6 paths × 3 rounds**: The optimality of the number of reasoning paths and refinement iterations has not been subjected to sensitivity analysis.

## Related Work & Insights

- **AFLOW** (Zhang et al., 2025): The direct predecessor of A2Flow, which searches code-represented workflows via MCTS but relies on manually predefined operators—the core contribution of A2Flow is precisely to address this limitation.
- **DebFlow** (Su et al., 2025): Extends AFLOW with debate and reflection mechanisms but does not resolve the operator predefinition problem.
- **MermaidFlow** (Zheng et al., 2025): Proposes domain-aware evolutionary operators but still depends on predefined initialization and generalizes insufficiently to embodied scenarios.
- **ADAS** (Hu et al., 2024): Represents workflows in code form but relies on linear heuristic search with low efficiency.
- **DSPy** (Khattab et al., 2024): Formalizes LLM pipelines as learnable text transformation graphs, eliminating manual prompt templates, but does not address operator-level automation.
- **Self-Consistency** (Wang et al., 2022): The multi-path sampling and voting idea is adopted in A2Flow's deep extraction stage.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Extending automation from workflow structure to operator extraction is a meaningful advance, though each individual step in the three-stage pipeline (clustering, CoT refinement, self-consistency) is a combination of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 8 benchmarks across 5 domains with ablation and cost analysis; however, only two embodied task benchmarks are included, and comparisons with MermaidFlow and DebFlow are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and the case study is intuitive, but the Related Work section contains considerable repetition, and mathematical notation is occasionally inconsistent.
- **Value**: ⭐⭐⭐⭐⭐ The elimination of manually defined operators represents an important step forward for agentic workflow automation, and the generalization improvements on open-world tasks carry significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](../../ICML2026/llm_agent/agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)
- [\[AAAI 2026\] Automating Complex Document Workflows via Stepwise and Rollback-Enabled Operations](automating_complex_document_workflows_via_stepwise_and_rollback-enabled_operatio.md)
- [\[ACL 2026\] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](../../ACL2026/llm_agent/hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](../../ACL2026/llm_agent/supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[CVPR 2026\] Gen-n-Val: Agentic Image Data Generation and Validation](../../CVPR2026/llm_agent/gen_n_val_agentic_image_data_generation_and_validation.md)

</div>

<!-- RELATED:END -->
