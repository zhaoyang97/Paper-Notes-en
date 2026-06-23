---
title: >-
  [Paper Note] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models
description: >-
  [ACL 2026][LLM Agent][Paper Note] This paper proposes Guided Topology Diffusion (GTD), which models the communication topology generation of multi-LLM agents as a conditional graph diffusion process. It utilizes a proxy reward model to perform zero-order guidance at each denoising step, thereby generating task-adaptive collaboration networks that are s
tags:
  - ACL 2026
  - LLM Agent
date: 2026-05-08
content_hash: 3cf2f319f48aa175
---
# Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models

**Conference**: ACL 2026  
**arXiv**: [2510.07799](https://arxiv.org/abs/2510.07799)  
**Code**: code is available  
**Area**: LLM Agent / Multi-agent Collaboration  
**Keywords**: Multi-agent systems, Communication topology, Graph diffusion, Zero-order optimization, Proxy reward model

## TL;DR
This paper proposes Guided Topology Diffusion (GTD), which models the communication topology generation of multi-LLM agents as a conditional graph diffusion process. It utilizes a proxy reward model to perform zero-order guidance at each denoising step, thereby generating task-adaptive collaboration networks that are sparser, more token-efficient, and more robust.

## Background & Motivation
**Background**: LLM multi-agent systems often resolve complex tasks such as mathematical reasoning, code generation, and knowledge QA through structured communication. Existing systems commonly use chain, star, complete graph, layered workflows, or manual templates. Some recent methods have begun to use search, GNNs, or autoregressive models to learn task-relevant topologies.

**Limitations of Prior Work**: Fixed topologies struggle to adapt to varying tasks. Simple QA may only require minimal linear interactions, whereas software development or complex reasoning requires richer collaborative structures. Over-dense graphs waste tokens, while over-sparse graphs create bottlenecks. Optimizing solely for accuracy tends to neglect communication costs, sparsity, and robustness to failure.

**Key Challenge**: Topology quality requires a trade-off between utility, cost, robustness, and sparsity. However, true rewards can only be obtained by performing full multi-agent simulations, which are both expensive and non-differentiable. Generative models that only learn the training distribution cannot stay-adjust toward high-reward regions during sampling.

**Goal**: The authors aim to construct an end-to-end topology generation framework that dynamically generates agent communication graphs for each new task, optimizing task performance and communication costs in real-time without frequent, expensive simulations.

**Key Insight**: The paper treats topology synthesis as a conditional discrete graph generation problem. A diffusion model is responsible for progressively constructing the graph, a lightweight proxy model predicts the utility and cost of candidate graphs, and zero-order optimization selects the optimal direction from multiple candidates at each sampling step.

**Core Idea**: A proxy reward model is first trained using a small number of benchmark topology simulations. Then, during the reverse denoising phase of the graph diffusion, candidate topologies are repeatedly sampled, and the current optimal candidate is selected using $w_u\hat{u}-w_c\hat{c}$, directly injecting multi-objective preferences into the generation trajectory.

## Method

### Overall Architecture
The problem GTD solves is: given a task query and a set of available agents, how to generate a communication graph $A\in\{0,1\}^{N\times N}$ that ensures task accuracy, minimizes token waste, and withstands single-point failures. It splits this into two complementary models: the proxy reward model $\mathcal{P}_\phi$ cheaply predicts the utility and cost of a topology under the current task, while the conditional graph diffusion generator $\mathcal{G}_\theta$ learns the distribution of high-quality topologies. In the training phase, both models are fed with simulation results from a small number of baseline topologies. During inference, the diffusion starts from a Gaussian noise graph and takes shape over 50 denoising steps. At each step, multiple candidate graphs are generated, and the proxy model selects the one with the highest reward to guide the next move, incrementally injecting multi-objective preferences into the sampling trajectory.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["Task query + Available agents"] --> TRAIN
    subgraph TRAIN["Training Phase: Small-scale baseline topology simulation"]
        direction TB
        SIM["Simulation of 50 training samples<br/>to obtain true utility / cost"] --> PROXY["Proxy Reward Model P_φ<br/>Two-layer GAT + MLP, MSE fits [û, ĉ]"]
        SIM --> GEN["Conditional Graph Diffusion Generator G_θ<br/>Two-layer Graph Transformer, BCE learns high-quality distributions"]
    end
    TRAIN --> NOISE["Inference: Start from Gaussian noise graph A_T"]
    NOISE --> PRED
    subgraph LOOP["Zero-order Guided Sampling (Every denoising step t, 50 steps total)"]
        direction TB
        PRED["G_θ predicts clean graph Â₀"] --> CAND["Bernoulli(sigmoid Â₀) samples K=5 candidate graphs"]
        CAND --> SCORE["Proxy reward model P_φ scores each candidate [û, ĉ]"]
        SCORE --> SEL["Select candidate with argmax (w_u û − w_c ĉ)<br/>Used to calculate the next posterior"]
    end
    SEL -->|t ← t−1, until 0| PRED
    SEL -->|Denoising complete| OUT["Task-adaptive communication topology A"]
```

### Key Designs

**1. GAT Proxy Reward Model: Replacing expensive simulations with a single forward pass**

True rewards require running a full round of multi-agent simulation, which is expensive, non-differentiable, and impossible to invoke repeatedly within each diffusion timestep. GTD addresses this by training a lightweight proxy: $\mathcal{P}_\phi$ takes graph $A$ and task condition $C$, uses a two-layer Graph Attention Network to calculate node representations, performs mean pooling for a graph-level representation, concatenates this with the task vector, and outputs $[\hat{u}, \hat{c}]$ via an MLP. The training objective is to approximate the performance vector from real simulations using MSE. Crucially, the proxy model does not need to perfectly reconstruct absolute reward values—it only needs sufficient ranking fidelity among candidates to support selection, allowing it to be trained on very little simulation data.

**2. Conditional Graph Diffusion Generator: Preserving critical edges through iterative refinement**

Communication graphs are discrete structures; the presence or absence of a single edge can determine whether information flows or is redundantly broadcast. Single-step VAEs or Gumbel-Softmax models often miss critical edges in such spaces. GTD utilizes diffusion: the binary adjacency matrix is scaled to $\{-1, 1\}$, noise is added via a variance-preserving forward process, and a two-layer Graph Transformer predicts the clean graph in the reverse process. The global attention of the Graph Transformer ensures edge predictions depend on all other nodes and edges, capturing structural dependencies like cycles and hierarchies. The progressive refinement inherent in diffusion provides the interface for the proxy model to intervene at each step.

**3. Zero-order Proxy-guided Sampling: Reward guidance on non-differentiable graphs**

Standard classifier guidance relies on gradient backpropagation, but the sampling action from continuous prediction to discrete graphs breaks differentiability. Furthermore, objectives like token cost and robustness are black-box. GTD adopts a zero-order approach: at each timestep, it obtains the unguided clean graph prediction $\hat{A}_0^{(t)}$, samples $K$ candidate graphs from $Bernoulli(\mathrm{sigmoid}(\hat{A}_0^{(t)}))$, obtains $[\hat{u}_k, \hat{c}_k]$ from the proxy model, and selects the candidate $A_{0,best}^{(t)}$ that maximizes $w_u\hat{u}_k - w_c\hat{c}_k$. This selection process requires no gradients yet directly applies multi-objective trade-offs to the generation trajectory.

### Loss & Training
The proxy model is trained using MSE to predict utility and cost from simulations. The diffusion generator is trained on high-performance graph subsets using BCE to predict the original clean adjacency matrix. In main experiments, all agents use a GPT-4o-mini backbone. Math tasks use 4 MathSolvers, HumanEval uses 4 CodeSolvers, and MMLU uses 3 KnowledgeableAcademic agents. The proxy model is a two-layer GAT with a hidden dimension of 32, Adam optimizer with learning rate $1e^{-3}$, batch size 16, and 10 training epochs. The diffusion model is a two-layer Graph Transformer with 2 attention heads, learning rate $1e^{-4}$, and 50 diffusion timesteps. Training data involves evaluating baseline topologies on only 50 samples from the training set. During inference, $K=5$ candidate graphs are evaluated at each step.

## Key Experimental Results

### Main Results

| Benchmark | GTD | Strongest/Typical Comparison | Gain or Note |
|--------|------|----------|------|
| GSM8K | 94.14 | MaAS: 92.30, G-Designer: 92.09, Vanilla: 87.45 | Highest in math reasoning |
| MATH | 54.07 | MaAS: 51.82, AFlow: 51.28 | 2+ points higher than strongest baseline |
| MultiArith | 98.88 | MaAS: 98.80, G-Designer: 97.78 | Near saturation but still best |
| HumanEval | 91.46 | G-Designer: 91.11, AFlow: 90.93 | Effective for coding tasks |
| MMLU | 84.58 | G-Designer: 84.50, GPTSwarm: 83.98 | Slight lead in knowledge tasks |
| SVAMP | 91.33 | G-Designer: 90.00, LLM-Debate: 89.00 | Consistent lead |
| Avg. | 85.74 | MaAS: 84.49, G-Designer: 84.41, Vanilla: 81.75 | 3.99 avg. gain over Vanilla |

### Ablation Study

| Configuration | GSM8K | HumanEval | Note |
|------|---------|------|------|
| GTD | 94.14 | 91.43 | Full proxy-guided diffusion |
| w/o Guidance | 88.42 | 87.19 | GSM8K drops nearly 6 points without guidance |
| w/ Random | 89.65 | 88.32 | Random candidate selection yields minimal gain |
| Direct GNN pred. | 91.23 | N/A | One-step generation weaker than diffusion |
| MCMC 100 steps | 92.87 | N/A | Search-based methods lower than GTD |
| GTD, $K=5$ | 94.14 / 7.9s | N/A | Best accuracy/time trade-off |
| GTD, $K=10$ | 94.31 / 18.1s | N/A | Accuracy increases slightly; latency more than doubles |

### Key Findings
- GTD excels in token cost. On GSM8K, GTD reaches 94%+ accuracy using approx. $4.8\times10^6$ tokens. G-Designer requires 15% more tokens for lower accuracy, and LLM-Debate uses over 5x more tokens.
- On MultiArith, GTD nears 99% accuracy with $8.4\times10^4$ tokens. On SVAMP, it is the only method to exceed 91% accuracy while maintaining the lowest token usage ($1.4\times10^5$ tokens).
- Robustness experiments show GTD drops only 0.3 percentage points under 1-agent failure on GSM8K; 2.1% under 2-agent failure; and 1.4% with a noisy agent (50% error), outperforming MaAS and G-Designer.
- Proxy model ranking fidelity is sufficient: held-out Top-1 of 5 ranking accuracy is 78.4% for utility and 85.2% for cost. On OOD GTD topologies, Top-1 is 72.8%, and Top-2 of 5 is 89.3%.

## Highlights & Insights
- The paper identifies a core bottleneck in multi-agent systems: topology is not just an engineering detail but a control variable for performance, cost, and robustness.
- Using diffusion instead of one-shot generation is logical. In communication graphs, a single wrong edge can cause a disconnected flow or redundant broadcasts; progressive refinement is better suited for discrete structural optimization.
- The proxy model does not need perfect reward prediction; it only needs to rank candidates roughly, which lowers the threshold for training with expensive simulation data.
- Experiments on token cost and failure robustness move beyond small accuracy gains toward the requirements of real-world multi-agent deployment.

## Limitations & Future Work
- GTD requires pre-computing baseline topologies to train the proxy model. Although 50 samples are claimed to be sufficient, there is still a setup cost for new tasks or agent combinations.
- The topology is generated before the task starts and does not change dynamically as the dialogue progresses. Static graphs may no longer be optimal if requirements shift mid-task.
- In current benchmarks, performance gains saturate at around 4 agents. While the framework is memory-scalable for larger swarms, standard reasoning tasks might not reflect their value.
- Proxy reward design mainly covers utility and cost; complex goals like safety, role reliability, tool-call failure, and long-term memory consistency require additional modeling.

## Related Work & Insights
- **vs. Static Topology**: Chain, star, and complete graphs are controllable but cannot adjust communication density by task difficulty; GTD generates task-adaptive sparse graphs.
- **vs. GPTSwarm / G-Designer / MaAS**: These already focus on topology or collaboration optimization; GTD's distinction is its use of a diffusion process for progressive generation with multi-objective proxy guidance at each step.
- **vs. AFlow**: AFlow focuses more on workflow search and optimization, whereas GTD focuses on communication graph structure, suitable for modeling agent messaging as a graph.
- **Inspiration for Agent Systems**: Future multi-agent frameworks could treat "who communicates with whom" as a learnable control variable rather than a hardcoded prompt graph or role template.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using graph diffusion + proxy-guided zero-order optimization for LLM agent topology is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main tasks, tokens, robustness, ablations, and harder benchmarks, though complex real-world workflows are still limited.
- Writing Quality: ⭐⭐⭐⭐☆ Explanations and formulas are complete, and the narrative is clear.
- Value: ⭐⭐⭐⭐☆ Provides direct inspiration for reducing communication costs and improving robustness; suitable for combination with online topology adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ACL 2026\] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check](the_bitter_lesson_of_diffusion_language_models_for_agentic_workflows_a_comprehen.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)

</div>

<!-- RELATED:END -->
