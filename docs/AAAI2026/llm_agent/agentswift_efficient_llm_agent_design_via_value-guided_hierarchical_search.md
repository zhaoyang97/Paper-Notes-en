---
title: >-
  [Paper Note] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search
description: >-
  [AAAI 2026][LLM Agent][Automated Agent Design] This paper proposes AgentSwift, a framework that automatically discovers high-performance LLM agent designs through a hierarchical search space (jointly optimizing agentic workflows and functional components), a lightweight value model for predicting agent performance, and an uncertainty-guided MCTS search strategy, achieving an average improvement of 8.34% across 7 benchmarks.
tags:
  - AAAI 2026
  - LLM Agent
  - Automated Agent Design
  - MCTS
  - Value Model
  - Hierarchical Search Space
  - Functional Components
date: 2026-05-08
content_hash: c5d8dcd4515778a7
---

# AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search

**Conference**: AAAI 2026  
**arXiv**: [2506.06017](https://arxiv.org/abs/2506.06017)  
**Code**: [https://github.com/Ericccc02/AgentSwift](https://github.com/Ericccc02/AgentSwift)  
**Area**: Agent  
**Keywords**: Automated Agent Design, MCTS, Value Model, Hierarchical Search Space, Functional Components

## TL;DR
This paper proposes AgentSwift, a framework that automatically discovers high-performance LLM agent designs through a hierarchical search space (jointly optimizing agentic workflows and functional components), a lightweight value model for predicting agent performance, and an uncertainty-guided MCTS search strategy, achieving an average improvement of 8.34% across 7 benchmarks.

## Background & Motivation
**Background**: LLM agents have demonstrated strong capabilities across diverse tasks, yet agent design remains heavily reliant on human expertise—from workflow organization to the selection and configuration of functional components such as memory, planning, and tool use, all of which require substantial domain knowledge and iterative tuning.

**Limitations of Prior Work**: Existing automated agent design methods suffer from three main problems:
   - **Restricted search space**: Methods such as AFlow and ADAS optimize only agentic workflow structures without considering functional components like memory, planning, or tool use, and thus cannot discover complete agent architectures.
   - **High evaluation cost**: Each new agent candidate requires a full benchmark run (e.g., evaluating a CoT agent on ALFWorld costs approximately \$60), causing significant resource waste on low-quality candidates.
   - **Low search efficiency**: Existing search strategies (e.g., AFlow's local optimization) are prone to getting trapped in local optima when navigating large design spaces.

**Key Challenge**: The fundamental tension between the combinatorial explosion of the design space (workflow × memory × tool × planning) and the high cost of individual evaluations makes exhaustive search infeasible, while local search tends to overlook high-quality designs.

**Goal**:
   - How to construct a unified search space that encompasses both workflows and functional components?
   - How to replace expensive real evaluations with low-cost surrogates?
   - How to efficiently navigate a vast search space?

**Key Insight**: Drawing inspiration from performance predictors in Neural Architecture Search (NAS)—where training a performance prediction model to substitute full training evaluations has proven highly effective—since agent design is structurally analogous to NAS.

**Core Idea**: Formalize agent design as a hierarchical search problem over workflows and functional components, employ a lightweight value model for low-cost evaluation, and use uncertainty-guided MCTS for efficient search.

## Method

### Overall Architecture
The AgentSwift pipeline consists of three core modules:
- **Input**: Task description $d$ and performance evaluation function $\text{Eval}_d(\cdot)$
- **Search Space**: Hierarchically defined as $\mathbf{A} = (\mathbf{W}, \mathbf{M}, \mathbf{T}, \mathbf{P})$, where $\mathbf{W}$ is the agentic workflow, $\mathbf{M}$ is memory, $\mathbf{T}$ is tool use, and $\mathbf{P}$ is planning
- **Search Engine**: Uncertainty-guided MCTS that generates new candidates via a three-level operation sequence: recombination → mutation → refinement
- **Evaluator**: A lightweight value model predicts candidate performance scores; only top candidates undergo real evaluation
- **Output**: Optimal agent design $\mathbf{A}^* = \arg\max_{\mathbf{A} \in \mathcal{S}_{\text{agent}}} \text{Eval}_d(\mathbf{A})$

### Key Designs

1. **Hierarchical Search Space**:

    - **Function**: Defines a unified joint search space over workflows and functional components.
    - **Mechanism**: The agentic workflow $\mathbf{W}=(N,E)$ comprises nodes (LLM invocation steps, each parameterized by model, prompt, temperature, and output format) and edges (execution order). Three functional components are attached in a plug-and-play manner: Memory $\mathbf{M}=(m,\tau,d)$ handles context retrieval and storage, Tool Use $\mathbf{T}=(t,\tau,u)$ connects to external APIs, and Planning $\mathbf{P}=(p,\tau)$ performs subgoal decomposition. The complete agent is defined as $\mathbf{A}=(\mathbf{W},\mathbf{M},\mathbf{T},\mathbf{P})$.
    - **Design Motivation**: Prior work such as AFlow searches only over workflows, while AgentSquare introduces components but searches within fixed workflow templates with a primary focus on prompt optimization. The proposed hierarchical space enables true joint optimization of workflows and components, yielding a substantially more expressive search space.

2. **Value Model (Performance Prediction Model)**:

    - **Function**: Given a candidate agent and task description, predict its performance score $\hat{v} = f_\theta(\mathbf{A}, d)$.
    - **Mechanism**: Built upon a 7B pretrained language model (Mistral-7B or Qwen2.5-7B) with a lightweight adapter, fine-tuned end-to-end using MSE loss. A key contribution lies in dataset construction: a $t=2$ covering array first ensures that all pairwise component interactions appear at least once (guaranteeing coverage), after which Balanced Bayesian Sampling explores both high-performance regions (UCB) and low-performance regions (LCB), formulated as $a_{\text{UCB}}(\mathbf{A}) = \mu(\mathbf{A}) + \kappa \cdot \sigma(\mathbf{A})$ and $a_{\text{LCB}}(\mathbf{A}) = -\mu(\mathbf{A}) + \kappa \cdot \sigma(\mathbf{A})$, yielding 220 annotated samples in total.
    - **Design Motivation**: Although GPT-4o in-context prediction is feasible, it is costly (requiring large model calls at each search step) and less accurate (Spearman correlation of 0.77 vs. 0.90 for the proposed method). Distilling into a 7B model enables fast inference at low cost; moreover, owing to the compositional nature of structured agent representations, the 7B model can learn effective component-to-performance mappings.

3. **Uncertainty-guided MCTS**:

    - **Function**: Efficiently searches for the optimal agent design.
    - **Mechanism**:
        - **Selection**: Employs a soft mixed probability strategy combining actual performance $s_i$ and uncertainty $u_i$: $P_{\text{mixed}}(i) = \lambda \cdot \frac{1}{n} + (1-\lambda) \cdot \frac{\exp(E(s_i, u_i))}{\sum_j \exp(E(s_j, u_j))}$, where $E(s_j, u_j) = \alpha((1-\beta) s_j + \beta u_j - s_{\max})$.
        - **Expansion**: Starting from the parent agent, three sequential operations are applied: (1) Recombination—replacing a subsystem by sampling from a component pool; (2) Mutation—an LLM generates new component implementations conditioned on the task and historical performance; (3) Refinement—fine-tuning prompts, temperature, and control flow based on failure cases. Each step uses the value model to score and select the best candidate.
        - **Evaluation**: The final candidate undergoes real evaluation; uncertainty is defined as $u = |s_{\text{real}} - \hat{s}|$.
        - **Backpropagation**: Real scores and uncertainties are propagated upward, updating visit counts.
    - **Design Motivation**: Pure exploitation risks local optima, while pure exploration wastes the search budget. The uncertainty mechanism directs search toward both high-performance regions and regions of high prediction uncertainty, which may conceal undiscovered high-quality candidates.

### Loss & Training
- The value model is trained with MSE loss: $\mathcal{L} = \frac{1}{N}\sum_i (v_i - f_\theta(\mathbf{A}_i, d_i))^2$
- Dataset of 220 samples, split 8:1:1 for training/validation/testing
- Search budget capped at 60 agents for fair comparison with baselines
- Value model trained on 3 A100 GPUs

## Key Experimental Results

### Main Results
Results on 7 benchmarks using GPT-4o-mini (averaged over 3 independent runs):

| Method | ALFWorld | SciWorld | MATH | WebShop | M3Tool | Travel | PDDL | Type |
|--------|----------|----------|------|---------|--------|--------|------|------|
| COT | 0.512 | 0.398 | 0.532 | 0.490 | 0.427 | 0.433 | 0.427 | Manual |
| FoA | 0.587 | 0.427 | 0.556 | 0.509 | 0.488 | 0.474 | 0.472 | Manual |
| AgentSquare | 0.701 | 0.475 | 0.556 | 0.520 | 0.561 | 0.553 | 0.577 | Search |
| AFlow | 0.619 | 0.452 | 0.562 | 0.497 | 0.524 | 0.497 | 0.528 | Search |
| **AgentSwift** | **0.806** | **0.509** | **0.628** | **0.562** | **0.634** | **0.573** | **0.614** | Search |

AgentSwift comprehensively outperforms both manually designed and automatically searched baselines across all 7 benchmarks, with an average improvement of 8.34%.

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| AgentSwift (full) | 0.806 (ALFWorld) | Complete model |
| w/o Uncertainty | Performance curve noticeably flatter | Search tends to exploit known regions |
| w/o MCTS | Flattest performance curve | Lacks hierarchical exploration; degenerates to local search |
| Full evaluation (replacing value model) | Slowest convergence | Budget heavily wasted on low-quality candidates |
| GPT-4o few-shot prediction | Intermediate | Lower accuracy than the value model |

Value model prediction quality comparison (Spearman correlation):

| Method | MSE | MAE | R² | Spearman |
|--------|-----|-----|----|----------|
| AgentSwift (Mistral) | 0.006 | 0.053 | 0.807 | **0.903** |
| AgentSwift (Qwen) | 0.005 | 0.055 | 0.828 | 0.899 |
| GPT-4o few-shot | 0.016 | 0.089 | 0.479 | 0.765 |
| GPT-4o zero-shot | 0.068 | 0.207 | -1.17 | 0.056 |

### Key Findings
- **MCTS combined with uncertainty guidance** is central to search efficiency—removing either component significantly flattens the search curve.
- The value model achieves near-oracle (full-training) MSE performance with only 30 labeled samples via few-shot adaptation to new tasks, suggesting that the structured representation of agent designs exhibits strong transferability.
- Discovered agent architectures generalize across models—agents found with GPT-4o-mini perform well when directly applied to other LLMs.
- Sensitivity analysis of hyperparameters $\alpha$, $\lambda$, and $\beta$ demonstrates robustness (ALFWorld performance varies within the range 0.768–0.813).

## Highlights & Insights
- **The paradigm transfer from NAS to Agent Search** is elegant: applying the performance predictor concept from NAS to agent design search yields a 7B surrogate evaluator that is both more accurate and cheaper than direct GPT-4o prediction. This idea generalizes to any search problem involving expensive evaluations.
- **Balanced Bayesian Sampling for dataset construction**: the strategy of sampling both high- and low-performance regions is well-conceived, ensuring that the value model can identify good agents while also distinguishing poor ones—resulting in greater discriminative power than datasets constructed from high-performance regions alone.
- **The three-level expansion operation (recombination → mutation → refinement)** constitutes a coarse-to-fine search strategy: recombination makes large jumps, mutation introduces moderate novelty, and refinement performs fine-grained adjustments—forming a well-structured hierarchy.

## Limitations & Future Work
- **Search space remains limited**: The definitions of the memory, tool use, and planning components are relatively fixed and do not cover more complex agent capabilities such as RAG, multi-agent collaboration, or self-play.
- **Evaluation limited to GPT-4o-mini**: The main experiments are conducted on a single model; although cross-model transfer experiments are included, whether the search process itself remains effective across different backbone models is not thoroughly validated.
- **220-sample training data for the value model**: Few-shot adaptation is required for new tasks; if the target task differs substantially from training tasks (e.g., cross-modal settings), transfer performance may be limited.
- **Single-agent design only**: The automated design of multi-agent systems is not addressed, representing an important direction for future extension.

## Related Work & Insights
- **vs. AFlow**: AFlow searches only over workflow structures without considering functional components. AgentSwift extends AFlow's search space and introduces a value model to reduce evaluation cost.
- **vs. AgentSquare**: Although AgentSquare incorporates memory, tool use, and planning, it searches within fixed workflow templates and relies on GPT-4o for in-context prediction (low accuracy, high cost). AgentSwift surpasses it on both dimensions through its hierarchical search space and dedicated value model.
- **vs. ADAS**: ADAS performs end-to-end workflow search without value model guidance, resulting in low search efficiency. AgentSwift's MCTS combined with uncertainty guidance is demonstrably more efficient.
- **Inspiration from NAS**: This paper establishes a clear analogy between NAS and agent design, providing a valuable framework reference for future research on automated agent design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of hierarchical search space, value model, and MCTS is well-integrated, though each individual component has precedents (NAS predictors, MCTS in LLM reasoning, etc.).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven benchmarks, comparisons with multiple baselines, detailed ablations, generalization analysis, and hyperparameter sensitivity—very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and formal definitions are complete, though some implementation details (e.g., component pool construction) are deferred to the appendix.
- **Value**: ⭐⭐⭐⭐ Provides a systematic framework for automated agent design that can practically accelerate the discovery of high-quality agent architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [\[ACL 2026\] What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](../../ACL2026/llm_agent/what_makes_an_llm_a_good_optimizer_a_trajectory_analysis_of_llm-guided_evolution.md)
- [\[AAAI 2026\] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_audiovisual_entity.md)

</div>

<!-- RELATED:END -->
