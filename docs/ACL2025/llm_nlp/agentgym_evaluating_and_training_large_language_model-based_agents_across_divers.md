---
title: >-
  [Paper Note] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
description: >-
  [ACL 2025][LLM (Other)][Generalist Agents] This paper proposes the AgentGym framework, which features 14 interactive environments, 89 task classes, standardized trajectory datasets, and evaluation benchmarks. It also introduces the AgentEvol self-evolution algorithm, enabling LLM agents to transition from imitation to autonomous evolution through cross-environment exploration and learning, achieving performance comparable to state-of-the-art models.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Generalist Agents"
  - "Self-Evolution"
  - "Multi-Environment Training"
  - "Behavior Cloning"
  - "AgentEvol"
date: 2026-05-08
content_hash: 71041381000fa0b0
---

# AgentGym: Evolving Large Language Model-based Agents across Diverse Environments

**Conference**: ACL 2025  
**arXiv**: [2406.04151](https://arxiv.org/abs/2406.04151)  
**Code**: [https://github.com/WooooDyy/AgentGym](https://github.com/WooooDyy/AgentGym)  
**Area**: LLM/NLP  
**Keywords**: Generalist Agents, Self-Evolution, Multi-Environment Training, Behavior Cloning, AgentEvol

## TL;DR
This paper proposes the AgentGym framework, which features 14 interactive environments, 89 task classes, standardized trajectory datasets, and evaluation benchmarks. It also introduces the AgentEvol self-evolution algorithm, enabling LLM agents to transition from imitation to autonomous evolution through cross-environment exploration and learning, achieving performance comparable to state-of-the-art models.

## Background & Motivation

**Background**: Building generalist agents capable of handling diverse tasks and self-evolving across different environments is a long-term goal of the AI community. LLMs are considered ideal foundations for building such agents due to their strong generalization capabilities. Current methods for constructing LLM agents primarily follow two lines of research.

**Limitations of Prior Work**: The first line of work leverages behavior cloning (BC) to make agents imitate expert trajectories, which requires extensive manual annotation, incurs high costs, scales poorly, and limits performance and generalization due to a lack of sufficient environmental exploration. The second line allows agents to autonomously explore and learn in environments, but is typically restricted to specific tasks within a single environment, resulting in "specialist agents" rather than "generalist agents."

**Key Challenge**: Training generalist agents requires a trinity of "diverse environments + high-quality trajectories + effective evolution methods." However, existing work lacks a unified multi-environment interaction platform and lacks effective cross-environment evolution methods.

**Goal**: To build a comprehensive framework that supports the training and evaluation of generalist LLM agents, and to explore the self-evolution capabilities of agents across tasks and environments.

**Key Insight**: Analogous to human learning—acquiring basic knowledge and skills first through imitation, and then autonomously learning and adapting to new tasks through interaction and exploration across diverse environments.

**Core Idea**: Propose the AgentGym framework (a multi-environment platform + a trajectory dataset + an evaluation benchmark) and the AgentEvol algorithm (cross-environment self-evolution) to realize the evolutionary process of LLM agents from behavior cloning to interactive learning.

## Method

### Overall Architecture
AgentGym consists of three core components: (1) An interaction platform—integrating 14 agent environments, providing a unified API via HTTP services to support real-time interaction, trajectory sampling, and online evaluation; (2) Data and benchmark—including an expanded instruction set, the AgentEval benchmark, and the AgentTraj high-quality trajectory dataset; (3) The AgentEvol evolution algorithm—where the base agent, after being trained via behavior cloning, explores and learns from experience across multiple environments.

### Key Designs

1. **Unified Interaction Platform**:

    - **Function**: Provides standardized interaction interfaces for 14 environments (web browsing, embodied tasks, scientific experiments, etc.).
    - **Mechanism**: Each environment is deployed as an independent HTTP service, and the client provides a wrapped, unified interface. All environments share the same observation/action space specifications, and the agent interacts with environments using the ReAct format (think-before-act). It supports concurrency and real-time feedback, enabling agents to explore multiple environments simultaneously.
    - **Design Motivation**: Existing agent frameworks either have a limited number of environments (e.g., 8 in AgentBench) or do not support interactive training. A unified platform serves as the fundamental infrastructure for enabling cross-environment evolution.

2. **AgentTraj Trajectory Dataset**:

    - **Function**: Provides high-quality expert trajectories for foundational behavior cloning training.
    - **Mechanism**: Uses crowd-sourcing and State-of-the-Art (SOTA) models (such as GPT-4) to collect trajectories across multiple environments. Instruction diversity is expanded via self-instruct and instruction evolution methods. Trajectories are organized in a unified format to form AgentTraj (base set, approx. 5,000 trajectories) and AgentTraj-L (expanded set, approx. 15,000 trajectories). Diverse and challenging subsets are selected to construct the AgentEval benchmark.
    - **Design Motivation**: It is extremely inefficient for an agent to learn from scratch in complex environments; it first needs to acquire basic instruction-following capabilities and prior knowledge through imitation.

3. **AgentEvol Self-Evolution Algorithm**:

    - **Function**: Enables the base agent to self-improve through environmental interaction, overcoming the performance bottlenecks of behavior cloning.
    - **Mechanism**: Split into three phases: (1) Exploration—the agent attempts new task instructions across multiple environments to collect interaction trajectories; (2) Filtering—successful trajectories are filtered using environmental reward signals; (3) Learning—supervised fine-tuning is performed on the filtered high-quality trajectories. A key innovation is the "dynamic sampling" strategy, which adaptively adjusts the exploration ratio of each environment based on environmental difficulty and the agent's current capability. Additionally, MCTS-inspired search is utilized to enhance trajectory diversity during the exploration phase.
    - **Design Motivation**: Behavior cloning is limited by the quality and quantity of expert data, whereas self-evolution through exploration can discover solution strategies not covered in the data, similar to off-policy learning in reinforcement learning.

### Loss & Training
In the behavior cloning stage, standard cross-entropy loss is used for training on AgentTraj. In the AgentEvol stage, an iterative DAgger-like strategy is used, alternating between exploration, filtering, and training cycles. Filtering uses binary rewards (success/failure) provided by the environment.

## Key Experimental Results

### Main Results

| Model | WebShop | ALFWorld | SciWorld | BabyAI | TextCraft | Average |
|------|---------|----------|----------|--------|-----------|------|
| GPT-4 | 52.3 | 78.0 | 43.2 | 90.0 | 18.0 | 56.3 |
| Lemur-70B-Chat | 38.5 | 18.0 | 19.5 | 81.1 | 6.0 | 32.6 |
| AgentGym-BC (Llama3-8B) | 45.2 | 62.0 | 34.8 | 88.9 | 12.0 | 48.6 |
| AgentGym-Evol (Llama3-8B) | **54.1** | **76.0** | **42.5** | **92.2** | **22.0** | **57.4** |

### Ablation Study

| Configuration | Average Performance | Description |
|------|---------|------|
| AgentEvol (Full) | 57.4 | Full evolution |
| BC only (AgentTraj) | 48.6 | Behavior cloning baseline |
| BC only (AgentTraj-L) | 53.8 | Upper bound of BC with larger dataset |
| AgentEvol Single-env evolution | 51.2 | Evolution restricted to a single environment |
| AgentEvol w/o Dynamic Sampling | 54.6 | Uniform exploration across environments |

### Key Findings
- AgentEvol, with only 8B parameters, outperforms GPT-4 on agent tasks, demonstrating the feasibility of the "small model + evolution" paradigm.
- Cross-environment evolution (57.4) significantly outperforms single-environment evolution (51.2), proving that environmental diversity is crucial for generalization.
- The dynamic sampling strategy yields an approximate 2.8-point gain, indicating that adjusting the exploration ratio according to environment difficulty is highly effective.
- The evolved AgentEvol (57.4) even outperforms the upper bound of BC using more data (AgentTraj-L at 53.8), proving that self-exploration can indeed discover strategies not covered in expert trajectories.

## Highlights & Insights
- The "platform + data + algorithm" trinity design of AgentGym provides a complete infrastructure for the agent community. Analogous to GLUE/SuperGLUE in the NLP domain, AgentEval is poised to become a standard benchmark for evaluating agent capability.
- The "imitation $\rightarrow$ exploration $\rightarrow$ learning" paradigm of AgentEvol closely mimics the human learning process, technically adapting the concepts of DAgger/RFT to agent training.
- The unified interface design for the 14 environments facilitates the easy integration of new environments, encouraging community contributions.

## Limitations & Future Work
- All environments are text-interactive, lacking visual or multimodal support.
- AgentEvol relies on binary reward signals provided by environments, which may have limited effectiveness in environments with sparse rewards (e.g., long-term planning tasks).
- Current evolution iterates for only 2-3 rounds; the effectiveness and stability of more evolutionary rounds have not been fully explored.
- The capabilities of the 8B-parameter agent remain limited, with gaps in complex reasoning and long-term memory compared to larger LLMs.

## Related Work & Insights
- **vs AgentBench**: AgentBench only offers evaluation and does not support training, whereas AgentGym supports both.
- **vs AgentOhana**: AgentOhana collects trajectories from multiple environments but lacks an interaction platform, while AgentGym provides comprehensive support for interactive training.
- **vs Pangu-Agent**: Pangu-Agent only supports single-environment evolution, whereas AgentGym's AgentEvol explores cross-environment evolution.

## Rating
- Novelty: ⭐⭐⭐⭐ Prominent framework-level contributions; while the AgentEvol algorithm is relatively conventional, its validation in the agent context is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 14 environments and 89 tasks, with thorough ablation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-organized with polished figures and complete technical details.
- Value: ⭐⭐⭐⭐⭐ Provides much-needed infrastructure and benchmarks to the agent community, with significant open-source contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Geometric Signatures of Compositionality Across a Language Model's Lifetime](geometric_compositionality_lifetime.md)
- [\[ACL 2025\] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](simulating_diverse_students.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[ACL 2025\] EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](escapebench_creative_agent.md)

</div>

<!-- RELATED:END -->
