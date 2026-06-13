---
title: >-
  [Paper Note] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models
description: >-
  [ACL 2026][LLM Agent][Multi-agent systems] This paper proposes Guided Topology Diffusion (GTD), which models the communication topology generation of multi-LLM agents as a conditional graph diffusion process. It utilizes…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Multi-agent systems"
  - "communication topology"
  - "graph diffusion"
  - "zero-order optimization"
  - "proxy reward model"
date: 2026-05-08
content_hash: aef027a6818c63d2
---

# Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models

**Conference**: ACL 2026  
**arXiv**: [2510.07799](https://arxiv.org/abs/2510.07799)  
**Code**: Specific URL not provided (text indicates code is available)  
**Area**: LLM Agent / Multi-Agent Collaboration  
**Keywords**: Multi-agent systems, communication topology, graph diffusion, zero-order optimization, proxy reward model

## TL;DR
This paper proposes Guided Topology Diffusion (GTD), which models the communication topology generation of multi-LLM agents as a conditional graph diffusion process. It utilizes a proxy reward model to perform zero-order guidance at each denoising step, thereby generating task-adaptive collaboration networks that are sparser, more token-efficient, and more robust.

## Background & Motivation
**Background**: LLM multi-agent systems often use structured communication to solve complex tasks such as mathematical reasoning, code generation, and knowledge QA. Existing systems frequently use chain, star, complete graph, layered workflows, or manual role templates. Some methods have begun using search, GNNs, or autoregressive models to learn task-relevant topologies.

**Limitations of Prior Work**: Fixed topologies struggle to adapt to task variations. Simple questions may only require a few linear interactions, while software development or complex reasoning requires richer collaborative structures. Overly dense graphs waste tokens, while overly sparse graphs create bottlenecks; optimizing solely for accuracy often ignores communication costs, sparsity, and failure robustness.

**Key Challenge**: Topology quality requires a trade-off between utility, cost, robustness, and sparsity. However, true rewards can only be obtained by executing a full multi-agent simulation, which is expensive and non-differentiable. If a generative model only learns the training distribution, it cannot gradually adjust toward high-reward regions during sampling.

**Goal**: The authors aim to construct an end-to-end topology generation framework that dynamically generates agent communication graphs for each new task and optimizes task performance and communication costs in real-time without frequently running expensive simulations.

**Key Insight**: The paper treats topology synthesis as a conditional discrete graph generation problem. A diffusion model is responsible for progressively constructing the graph, a lightweight surrogate model predicts the utility and cost of candidate graphs, and zero-order optimization selects the optimal direction from multiple candidates at each sampling step.

**Core Idea**: First, a proxy reward model is trained using a small number of benchmark topology simulations. Then, during the reverse denoising of the graph diffusion, candidate topologies are repeatedly sampled, and the current optimal candidate is selected using $w_u\hat{u}-w_c\hat{c}$, directly injecting multi-objective preferences into the generation trajectory.

## Method
GTD consists of two components: the proxy reward model $\mathcal{P}_\phi$ and the conditional graph diffusion generator $\mathcal{G}_\theta$. The former quickly predicts the utility and cost of a topology under the current task, while the latter learns the distribution of high-quality topologies. During inference, the diffusion model does not generate the graph in one step; instead, it refines it progressively over 50 denoising steps, with the proxy model evaluating multiple candidate graphs at each step.

### Overall Architecture
Given a task query and available agents, the system constructs a condition vector $C$ to generate an adjacency matrix $A\in\{0,1\}^{N\times N}$. The training phase evaluates a set of baseline topologies to obtain topologies, task conditions, and performance vectors to train a GAT-based surrogate model; meanwhile, high-performance graphs are filtered to train a Graph Transformer diffusion generator. During inference, sampling starts from a Gaussian noise graph and proceeds through denoising steps. Each step converts the continuous graph predicted by the generator into multiple discrete candidates, which are scored by the proxy model to select the candidate with the highest reward as the basis for posterior sampling.

### Key Designs
1. **GAT Proxy Reward Model**:
    - **Function**: Approximates expensive multi-agent simulation rewards in a low-cost manner.
    - **Mechanism**: $\mathcal{P}_\phi$ takes graph $A$ and task condition $C$ as input, generates node representations through two Graph Attention Network layers, followed by mean pooling into a graph representation. This is concatenated with the task vector and passed through an MLP to output $[\hat{u},\hat{c}]$. The training loss is the MSE between the predicted performance vector and the actual simulation results.
    - **Design Motivation**: Real rewards are non-differentiable and expensive to compute, making them unsuitable for invocation at every diffusion step. As long as the proxy model has sufficient ranking fidelity, it can facilitate fast selection among candidate graphs.

2. **Conditional Graph Diffusion Generator**:
    - **Function**: Generates high-quality communication topologies matching the task conditions.
    - **Mechanism**: Scales the binary adjacency matrix to $\{-1,1\}$ and adds noise via a variance-preserving forward process. The reverse process uses two Graph Transformer layers to predict the clean graph. The global attention of the Graph Transformer allows each edge prediction to depend on other nodes and edges, capturing structural dependencies like cycles and hierarchies.
    - **Design Motivation**: One-step VAEs or Gumbel-Softmax models tend to miss critical edges in discrete topology spaces. The iterative refinement of diffusion allows the proxy model to intervene at each step, gradually pushing the graph toward high-reward regions.

3. **Zero-order Proxy-guided Sampling**:
    - **Function**: Achieves reward-guided generation on non-differentiable discrete graphs.
    - **Mechanism**: At each timestep, a prediction for the unguided clean graph $\hat{A}_0^{(t)}$ is obtained, and $K$ candidate graphs are sampled from $Bernoulli(sigmoid(\hat{A}_0^{(t)}))$. The proxy model predicts $[\hat{u}_k,\hat{c}_k]$ for each, and the candidate $A_{0,best}^{(t)}$ that maximizes $w_u\hat{u}_k-w_c\hat{c}_k$ is chosen to compute the subsequent posterior.
    - **Design Motivation**: Standard classifier guidance requires gradients, but discrete sampling here breaks differentiability. Zero-order candidate selection requires no gradients and can directly handle black-box objectives like token cost and robustness.

### Loss & Training
The proxy model is trained using MSE to predict utility and cost from simulations. The diffusion generator is trained on a subset of high-performance graphs using BCE to predict the original clean adjacency matrix. In the main experiments, all agents use a GPT-4o-mini backbone. Math tasks use 4 MathSolvers, HumanEval uses 4 CodeSolvers, and MMLU uses 3 KnowledgeableAcademics. The proxy model is a two-layer GAT with a hidden dimension of 32, Adam optimizer with a learning rate of $1e^{-3}$, batch size of 16, and 10 training epochs. The diffusion model uses a two-layer Graph Transformer with 2 attention heads, a learning rate of $1e^{-4}$, and 50 diffusion timesteps. The training data uses only 50 training set samples for baseline topology evaluation. During inference, $K=5$ candidate graphs are evaluated at each step.

## Key Experimental Results

### Main Results
| Benchmark | GTD | Strongest/Typical Comparison | Gain or Description |
| :--- | :--- | :--- | :--- |
| GSM8K | 94.14 | MaAS: 92.30, G-Designer: 92.09, Vanilla: 87.45 | Highest in math reasoning |
| MATH | 54.07 | MaAS: 51.82, AFlow: 51.28 | 2+ points higher than strongest baseline |
| MultiArith | 98.88 | MaAS: 98.80, G-Designer: 97.78 | Near saturation but still best |
| HumanEval | 91.46 | G-Designer: 91.11, AFlow: 90.93 | Effective for coding tasks |
| MMLU | 84.58 | G-Designer: 84.50, GPTSwarm: 83.98 | Slight lead in knowledge tasks |
| SVAMP | 91.33 | G-Designer: 90.00, LLM-Debate: 89.00 | Consistent lead |
| Avg. | 85.74 | MaAS: 84.49, G-Designer: 84.41, Vanilla: 81.75 | 3.99 avg. gain over Vanilla |

### Ablation Study
| Configuration | GSM8K | HumanEval | Description |
| :--- | :--- | :--- | :--- |
| GTD | 94.14 | 91.43 | Full proxy-guided diffusion |
| w/o Guidance | 88.42 | 87.19 | GSM8K drops nearly 6 points without guidance |
| w/ Random | 89.65 | 88.32 | Random candidate selection offers minimal gain |
| Direct GNN pred. | 91.23 | N/A | One-step generation is weaker than diffusion |
| MCMC 100 steps | 92.87 | N/A | Search-based methods remain lower than GTD |
| GTD, $K=5$ | 94.14 / 7.9s | N/A | Best trade-off between accuracy and time |
| GTD, $K=10$ | 94.31 / 18.1s | N/A | Accuracy increases slightly while latency doubles |

### Key Findings
- GTD is superior in token cost. On GSM8K, GTD reaches 94%+ accuracy using approximately $4.8\times10^6$ tokens, while G-Designer requires 15% more tokens for lower accuracy, and LLM-Debate uses over 5x the tokens.
- On MultiArith, GTD approaches 99% accuracy with $8.4\times10^4$ tokens; on SVAMP, it is the only method exceeding 91% accuracy while maintaining the lowest token usage ($1.4\times10^5$ tokens).
- Robustness experiments show that GTD drops only 0.3 percentage points under a single-agent failure in GSM8K; it drops 2.1% under a two-agent failure and 1.4% with a noisy agent (50% error), both outperforming MaAS and G-Designer.
- Proxy model ranking fidelity supports guidance: held-out Top-1 of 5 ranking accuracy is 78.4% for utility and 85.2% for cost; on OOD GTD topologies, Top-1 is 72.8% and Top-2 of 5 is 89.3%.

## Highlights & Insights
- The paper identifies a core bottleneck in multi-agent systems: topology is not an engineering detail but a control variable for performance, cost, and robustness.
- Using diffusion instead of one-shot generation is rational. A single incorrect edge in a communication graph can cause information stagnation or redundant broadcasting; iterative correction is better suited for such discrete structural optimization.
- The proxy model does not need perfect reward prediction; it only needs to provide rough rankings among candidates, which lowers the barrier to training with expensive simulation data.
- Experiments on token cost and failure robustness ensure the paper does not just offer marginal accuracy improvements but addresses the needs of real-world multi-agent deployment.

## Limitations & Future Work
- GTD requires pre-calculating baseline topologies to train the proxy model. Although the authors claim 50 samples are sufficient, there is still a setup cost for new tasks or agent combinations.
- Topologies are currently generated before the task starts and do not change dynamically as the dialogue progresses. If requirements change mid-task, a static graph may no longer be optimal.
- In current benchmarks, performance gains saturate at around 4 agents; larger swarms might be memory-scalable, but standard reasoning tasks may not reflect their value.
- The proxy reward design mainly covers utility and cost; more complex objectives like safety, role reliability, tool call failures, and long-term memory consistency require additional modeling.

## Related Work & Insights
- **vs static topology**: Chain, star, and complete graphs are simple and controllable but cannot adjust communication density based on task difficulty; GTD generates task-adaptive sparse graphs directly.
- **vs GPTSwarm / G-Designer / MaAS**: These methods focus on topology or collaboration optimization; GTD differs by using a diffusion process for step-by-step generation with multi-objective proxy guidance.
- **vs AFlow**: AFlow focuses more on workflow search and optimization, while GTD focuses on communication graph structure generation, modeling message passing between agents as a graph.
- **Insights for Agent Systems**: Future multi-agent frameworks could treat "who communicates with whom" as a learnable control variable rather than a fixed component in a prompt graph or role template.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using graph diffusion + proxy-guided ZO for LLM agent topology generation is innovative, with clear generative modeling.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main tasks, tokens, robustness, ablations, open-source models, and hard benchmarks, though real complex agent workflows remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological explanations and formulas are complete; some tables in the cached text are fragmented, but the overall narrative is clear.
- Value: ⭐⭐⭐⭐☆ Offers direct insights into reducing multi-agent communication costs and improving robustness, suitable for future combination with online topology adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ACL 2026\] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check](the_bitter_lesson_of_diffusion_language_models_for_agentic_workflows_a_comprehen.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)

</div>

<!-- RELATED:END -->
