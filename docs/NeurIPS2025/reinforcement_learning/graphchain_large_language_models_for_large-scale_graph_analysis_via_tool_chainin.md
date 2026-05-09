---
title: >-
  [Paper Note] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining
description: >-
  [NeurIPS 2025][Reinforcement Learning][large-scale graph analysis] This paper proposes GraphChain, a framework that enables LLMs to analyze large-scale graphs in a progressive, human-like exploratory manner through two key components: progressive graph distillation (RL-driven tool-chain sequence generation) and structure-aware test-time adaptation (lightweight adapters conditioned on graph topology fingerprints). GraphChain achieves an average accuracy of 84.7%, surpassing the best baseline by 20.7%, and scales to graphs with up to 200,000 nodes.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - large-scale graph analysis
  - tool chaining
  - information bottleneck
  - test-time adaptation
  - graph analysis
date: 2026-05-08
content_hash: 766e4adfa5bd3264
---

# GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining

**Conference**: NeurIPS 2025
**arXiv**: [2511.00457](https://arxiv.org/abs/2511.00457)
**Code**: [GitHub](https://github.com/wuanjunruc/GraphChain)
**Area**: Reinforcement Learning
**Keywords**: large-scale graph analysis, tool chaining, reinforcement learning, information bottleneck, test-time adaptation, graph analysis

## TL;DR
This paper proposes GraphChain, a framework that enables LLMs to analyze large-scale graphs in a progressive, human-like exploratory manner through two key components: progressive graph distillation (RL-driven tool-chain sequence generation) and structure-aware test-time adaptation (lightweight adapters conditioned on graph topology fingerprints). GraphChain achieves an average accuracy of 84.7%, surpassing the best baseline by 20.7%, and scales to graphs with up to 200,000 nodes.

## Background & Motivation

**Two core challenges for LLMs processing graph data**:
   - **Context Exhaustion**: Large-scale graphs (with millions of nodes/edges) cannot be compressed into the LLM context window.
   - **Reasoning Hallucination**: Single-step tool calls place excessive demands on individual tools and are insufficient for complex graph analysis.

**Limitations of prior work**:
   - Direct processing methods (text/visual graph descriptions, specialized tokenization) are constrained by context length.
   - Tool-augmented methods (Graph-ToolFormer, GraphForge) support only single-step tool calls and rely on textual descriptions of graph structure.

**Human cognition analogy**: Just as humans explore an unknown environment by first scanning broadly and then focusing on regions of interest, graph analysis requires progressive and sequential information gathering rather than a one-shot approach.

**Two technical challenges**: (a) How to generate optimal tool sequences within an exponentially large combinatorial tool space? (b) How to adapt to distribution shifts across graphs with heterogeneous topological structures?

## Method

### Overall Architecture
Graph analysis is formulated as a Markov Decision Process (MDP): $M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$
- **State** $s_t$: query $\mathcal{Q}$, graph reference, action history, and memory state $\mathbf{m}_{t-1}$
- **Action** $a_t = (T, \theta_T)$: selection of tool $T$ with parameters, or TERMINATE
- **Memory state**: $\mathbf{m} \approx (\mathbf{A}' \in \mathbb{R}^{n' \times n'}, \mathbf{X}' \in \mathbb{R}^{n' \times d}, \ldots)$
- Tool library: 45 curated functions based on NetworkX

### Progressive Graph Distillation

**Objective**: Train an agent via RL to generate tool sequences that progressively reduce the volume of the memory state while preserving task-relevant information.

**Graph Description Length (GDL)**: Quantifies the data volume of the memory state:

$$\text{GDL}(\mathbf{m}_t) = \alpha_s m_t' + \alpha_f n_t' d_f$$

where $m_t'$ is the number of edges, $n_t'$ is the number of nodes, and $d_f$ is the feature dimensionality.

**Task Relevance (Rel)**: An auxiliary LLM evaluates the usefulness of the current memory state with respect to the query:

$$\text{Rel}(\mathbf{m}_t, \mathcal{Q}) \approx \text{LLMScore}(\text{prompt}(\mathcal{Q}, H_t, d_t)) \in [0, 1]$$

**Distillation-aware reward function**:

$$R_t = \begin{cases} w_1 \cdot \hat{r}_t^{\text{Succ}} + w_2 \cdot \hat{r}_t^{\Delta\text{GDL}} + w_3 \cdot \hat{r}_t^{\Delta\text{Rel}} & t < N \\ w_{\text{solve}} \cdot \text{EvaluateTaskSuccess}(\mathcal{Q}, s_{N+1}) & t = N \end{cases}$$

where:
- $\hat{r}_t^{\text{Succ}}$: binary indicator of successful tool execution
- $\hat{r}_t^{\Delta\text{GDL}} = \tanh(\beta \frac{\text{GDL}(\mathbf{m}_{t-1}) - \text{GDL}(\mathbf{m}_t)}{\text{GDL}(\mathbf{m}_{t-1}) + \epsilon})$: rewards volume reduction
- $\hat{r}_t^{\Delta\text{Rel}} = \text{Rel}_t - \text{Rel}_{t-1}$: rewards relevance improvement

**Information bottleneck perspective**: The reward function is equivalent to maximizing $I(\mathbf{m}_t; Y)$ while minimizing $I(X; \mathbf{m}_t)$, i.e., retaining task-relevant information while compressing irrelevant information.

**Policy optimization**: PPO + GAE

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{N-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = R_{t+1} + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)$$

### Structure-Aware Test-Time Adaptation (STTA)

**Graph structure fingerprint**: The first $M+1$ smallest singular values of the normalized graph Laplacian $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ are computed as: $\mathbf{z}_G = (\sigma_0, \sigma_1, \ldots, \sigma_M)$

**Structure-conditioned soft prompt generation**: A lightweight adapter $\mathcal{A}_\psi$ maps the fingerprint to a soft prompt: $\mathbf{P}_G = \mathcal{A}_\psi(\mathbf{z}_G) \in \mathbb{R}^{L_p \times d_{emb}}$, prepended to the agent's input.

**Self-supervised adaptation objective**:

$$L_{\text{STTA}}(\psi) = \mathbb{E}\left[w_L N_{\tau_i} + w_{KL} \sum_{t=0}^{N_{\tau_i}-1} D_{KL}(\pi_\psi(\cdot|s_t; G_{\text{test}}) \| \pi_{\text{orig}}(\cdot|s_t))\right]$$

This balances chain length efficiency with policy regularization. Only the adapter parameters $\psi$ are updated; the base LLM remains frozen.

## Key Experimental Results

### Main Results (Accuracy %)

| Method | Params | Finance | Chemistry | Social | Citation | Traffic | Avg. |
|--------|--------|---------|-----------|--------|----------|---------|------|
| GPT-4o | ~200B | 57.5 | 62.7 | 65.2 | 71.5 | 43.4 | 59.4 |
| Claude-4-Sonnet | - | 58.2 | 62.9 | 61.7 | 77.5 | 32.8 | 58.6 |
| GPT-4.1 | - | 52.2 | 63.4 | 67.4 | 70.0 | 55.5 | 61.7 |
| GraphForge | 8B | 63.5 | 70.9 | 80.4 | 63.4 | 73.5 | 70.2 |
| ToolGen | 8B | 75.8 | 57.9 | 79.4 | 61.2 | 62.7 | 67.4 |
| **GraphChain** | **7B** | **81.5** | **81.1** | **89.6** | **83.6** | **84.1** | **84.7** |
| Relative Gain | - | +7.5% | +14.4% | +11.4% | +7.9% | +14.4% | **+20.7%** |

### Ablation Study
- Removing progressive graph distillation leads to the largest performance drop (most critical component).
- Removing test-time adaptation causes a smaller drop but limits cross-domain generalization.
- GraphChain without STTA still outperforms GraphForge, demonstrating the effectiveness of tool chaining itself.

### Transfer Learning (Trained on Financial Networks Only)

| Target Domain | In-Domain | w/ STTA | w/o STTA |
|---------------|-----------|---------|----------|
| Social Networks | 89.6 | 86.8 (−3.1%) | 84.5 (−5.7%) |
| Citation Graphs | 83.6 | 79.2 (−4.3%) | 75.1 (−10.2%) |
| Traffic Networks | 84.1 | 80.3 (−4.5%) | 77.4 (−8.0%) |

### Scalability
- GraphChain maintains consistent performance on graphs with up to 200,000 nodes.
- Baseline methods degrade significantly as graph size increases.
- GraphChain's advantage is more pronounced on 5-step tool-chain tasks.

### Robustness Across Base Models

| Base Model | Finance | Chemistry | Social | Citation | Traffic | Avg. |
|------------|---------|-----------|--------|----------|---------|------|
| Qwen2.5-7B | 70.5 | 81.1 | 90.4 | 79.0 | 82.0 | 80.6 |
| Llama3.1-8B | 69.3 | 81.7 | 93.7 | 82.5 | 81.7 | 81.8 |
| GLM4-9B | 70.2 | 78.9 | 93.8 | 79.7 | 79.9 | 80.5 |

### Tool Type Distribution
- Traffic networks are dominated by path-planning tools (33.8%).
- Social networks rely primarily on centrality measures (28.8%) and community detection (20.4%).
- Citation graphs show a more balanced distribution, with connectivity tools accounting for 18.9%.

## Highlights & Insights
- **Human-like exploratory cognition**: Graph analysis is reformulated as coarse-to-fine sequential exploration rather than a one-shot process.
- **Information bottleneck theoretical grounding**: Reducing GDL corresponds to compressing irrelevant information; increasing Rel corresponds to retaining task-relevant information.
- **Exceptional parameter efficiency**: A 7B model outperforms GPT-4o (~200B) by 25 percentage points.
- **Training-free test-time adaptation**: Lightweight adapters conditioned on graph topology fingerprints enable rapid adaptation to novel graph structures without retraining.
- **Informative tool-chain analysis**: Different domains self-organize into domain-specific tool usage distributions.

## Limitations & Future Work
- Only static graphs are supported; dynamic and temporal graphs are not addressed.
- The tool library is fixed at 45 NetworkX functions, potentially lacking domain-specific operations.
- Subgraphs are partitioned to fewer than 100 nodes for fair comparison with baselines; subgraph strategies for truly large graphs are not fully elaborated.
- RL training requires 3,000 expert-annotated (query, answer) pairs, incurring non-trivial annotation costs.
- Auxiliary query generation relies on an LLM, introducing circular dependency.
- Performance on more complex structures such as heterogeneous graphs and hypergraphs is not evaluated.
- Reducing the tool library by 50% still causes accuracy to drop from 84.7% to 79.8%, indicating residual dependence on tool library completeness.

## Related Work & Insights
- **vs. Graph-ToolFormer**: Single-step tool calls with textual graph descriptions; GraphChain employs multi-step chained calls with memory state management.
- **vs. GraphForge**: Supports external functions but performs single-step inference; GraphChain uses RL to train sequential decision-making.
- **vs. NLGraph/GraphWiz**: Text-instruction methods are constrained by context length; GraphChain bypasses this limitation via tool chaining.
- **vs. GNN-LLM methods**: GNN encoders require end-to-end training; GraphChain employs a plug-and-play tool library.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of progressive graph distillation and structure-aware TTA is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 domains, multiple baselines, ablation, transfer, scalability, and robustness.
- Writing Quality: ⭐⭐⭐⭐ Clear framework with well-motivated theoretical grounding.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for LLM-based large-scale graph analysis; 7B model surpasses 200B counterparts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/reinforcement_learning/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](../../ICLR2026/reinforcement_learning/graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [\[NeurIPS 2025\] Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification](kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)

</div>

<!-- RELATED:END -->
