---
title: >-
  [Paper Note] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration
description: >-
  [ICML 2026][Multi-Agent][Multi-Agent Systems] This paper formalizes the optimization space of multi-agent systems (MAS) into five dimensions (two functional and three structural). It utilizes a dual-actor algorithm—"Sema…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-Agent Systems"
  - "Collaborative Optimization"
  - "Contrastive Reasoning"
  - "Prompt Evolution"
  - "Supervised Optimization"
date: 2026-05-08
content_hash: 661d2fd78b697d91
---

# OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration

**Conference**: ICML 2026  
**arXiv**: [2505.11765](https://arxiv.org/abs/2505.11765)  
**Code**: https://github.com/xiwenchao/OMAC  
**Area**: LLM Agent / Multi-Agent Systems / Code Generation  
**Keywords**: Multi-Agent Systems, Collaborative Optimization, Contrastive Reasoning, Prompt Evolution, Supervised Optimization

## TL;DR
This paper formalizes the optimization space of multi-agent systems (MAS) into five dimensions (two functional and three structural). It utilizes a dual-actor algorithm—"Semantic Initializer" for generation and "Contrastive Comparator" for iterative improvement—to perform supervised optimization across each dimension. By iteratively and jointly optimizing multiple dimensions, OMAC consistently outperforms baselines such as DyLAN, ADAS, and AFlow on HumanEval, MMLU, and MATH.

## Background & Motivation

**Background**: Multi-agent systems (MAS) have demonstrated capabilities superior to single agents in tasks such as code generation (AgentVerse), reasoning (LLM Debate), and decision-making (Sun 2024). However, most existing MAS rely on manual design—agent roles are defined by human priors or direct LLM generation, and collaborative structures use fixed centralized, decentralized, or hierarchical topologies. A few automated works, such as DyLAN (dynamic agent team composition), ADAS (prompt evolution), AFlow (MCTS-based architecture search), and G-Designer/MaAS (architecture search), only optimize single aspects.

**Limitations of Prior Work**: (1) DyLAN uses an unsupervised "agent importance score," lacking supervised signals derived from training data; (2) ADAS, AFlow, G-Designer, and MaAS only optimize one specific aspect (either prompts *or* architecture); (3) None of these methods can simultaneously modify agent functions (prompts/few-shot) and collaborative structures (candidate selection, dynamic participation, and communication flow) within a unified framework.

**Key Challenge**: MAS is essentially an information flow graph where both nodes (agents) and edges (communication links) are optimizable. However, existing methods focus on either nodes or edges, and no universal algorithm exists to handle both types simultaneously.

**Goal**: Construct a unified framework capable of performing supervised optimization for any dimension of an MAS using a single algorithm while allowing for the iterative joint optimization of multiple dimensions.

**Key Insight**: After conceptualizing the MAS collaboration process as an information flow graph, the authors identify 5 core optimizable dimensions—2 regarding nodes (agents) and 3 regarding the graph (structure). They find that all 5 dimensions can be reduced to the problem of "optimizing a segment of LLM instruction prompts (plus optional few-shot examples)."

**Core Idea**: Employs a pair of LLM-driven actors—a Semantic Initializer to explore the semantic space for a diverse set of initial prompts, and a Contrastive Comparator to contrast high- and low-scoring pairs to generate superior prompts. This performs supervised contrastive reasoning optimization in each dimension, followed by an "one-dimension-at-a-time" iterative strategy to unify multiple dimensions.

## Method

### Overall Architecture
The input to OMAC is an existing MAS configuration (multiple agents and a collaborative topology) and a training set with supervised evaluation metrics (e.g., Pass@1 for HumanEval). The framework consists of two layers: (1) **Single-dimension optimization algorithm**—Select one of the 5 dimensions, use the Semantic Initializer to generate $n$ candidate agent/controller prompts, and run the full MAS collaboration on the training set to obtain performance scores. Sample a pair of positive and negative samples based on upper and lower thresholds $h, l$ and feed them into the Contrastive Comparator, which generates new candidates via contrastive reasoning. This loops until a preset iteration limit. (2) **Multi-dimension joint optimization**—Select multiple dimensions for joint optimization and iterate through them serially by "fixing others while optimizing one," preventing the Contrastive Comparator from struggling with attribution due to simultaneous changes.

### Key Designs

1.  **Division of 5D Optimization Space**:
    - **Function**: Systematically divides all optimizable parts of MAS into 5 non-overlapping dimensions—Fun-1 (optimizing existing agent prompts/few-shot), Fun-2 (constructing new agents), Str-1 (candidate agent selection controller), Str-2 (per-step dynamic participation controller), and Str-3 (inter-agent communication flow controller).
    - **Mechanism**: Views MAS as an information flow graph where nodes are agents and edges are communication links. Fun-1/2 optimize node capabilities (modifying existing nodes or adding new ones), while Str-1/2/3 optimize graph structure (global team selection, per-step local selection, and edge routing). Since all 5 dimensions involve "optimizing an LLM prompt," the same algorithm is applicable by merely swapping the context description.
    - **Design Motivation**: Previous works only covered specific dimensions (e.g., DyLAN for Str-1, ADAS/AFlow for combinations of Str-3 and Fun-2), lacking a unified coordinate system. By explicitly defining these dimensions, the authors found that all 5 have independent room for improvement and can be superimposed.

2.  **Semantic Initializer + Contrastive Comparator Dual-Actor Algorithm**:
    - **Function**: Uses two LLM-driven actors for supervised optimization in each dimension—the former for exploration (generating a diverse initial set) and the latter for exploitation (generating improvements by contrasting positive and negative differences).
    - **Mechanism**: The Semantic Initializer receives context (task description, current MAS configuration, dimension specification, and a one-shot example) and a quantity $n$, directing the LLM to perform diverse generation in the semantic space while maintaining functional consistency. Each candidate is evaluated on the training set. Pairs are sampled based on top-$\lfloor n h \rfloor$ and bottom-$\lfloor n l \rfloor$ thresholds for the Contrastive Comparator. The comparator performs reasoning: "Why is A better than B? Strengthen the advantages in A, eliminate the weaknesses in B, and generate a new version better than A."
    - **Design Motivation**: Purely letting an LLM generate prompts is unsupervised exploration that fails to leverage training scores. Contrastive reasoning utilizes "poor performance" as a supervised signal, letting the LLM perform attribution using its own reasoning—a lightweight, gradient-free "pseudo-RL" where contrastive samples change only one dimension at a time to avoid messy attribution.

3.  **Iterative Multi-Dimension Joint Optimization**:
    - **Function**: Rotates optimization across multiple dimensions, modifying one while fixing others to avoid combinatorial explosion and Comparator attribution failure across multiple variables.
    - **Mechanism**: Runs single-dimension optimization independently for each dimension to identify those with the highest gains. During joint optimization, it completes optimization for dim-1 and retains the best result → switches to dim-2 to optimize while dim-1 is fixed → returns to dim-1, and so on, until the iteration limit. Results show that joining the two strongest functional dimensions, or the strongest functional and strongest structural dimensions, yields the best results.
    - **Design Motivation**: An iterative rather than parallel approach ensures clear attribution for the Contrastive Comparator. In every contrastive pair, only one dimension changes while others remain at their current optimum, making the cause of difference interpretable. Ablation studies confirm that simultaneous reasoning over multiple dimensions leads to significantly smaller gains and higher variance.

### Loss & Training
OMAC does not update model parameters; all "optimization" is prompt engineering. Supervised signals are MAS task metrics on training sets (HumanEval Pass@1, MMLU Accuracy, MATH Accuracy). The backbone LLM is gpt-3.5-turbo-1106 with a temperature of 0.8. The Semantic Initializer generates 3 candidates per round, the contrastive reasoning limit is 3 iterations, and thresholds are $l=h=0.5$. Results are reported as mean ± standard deviation over 3 runs.

## Key Experimental Results

### Main Results
Benchmark tasks: HumanEval (code generation, Pass@1), MMLU (general reasoning, Accuracy), and MATH (arithmetic reasoning, Accuracy). Default MAS configurations are from DyLAN (7 agents for code, 7 for reasoning, 4 for math) with fully connected topologies.

| Task | Best Baseline | Baseline Score | OMAC Single-Dim Max | Dimension |
|------|---------|---------|--------------|------|
| HumanEval (Pass@1) | AFlow | 85.63 | 89.25 ± 1.30 | Fun-1.4 |
| MMLU (Accuracy) | ADAS | 69.02 | 74.22 ± 2.22 | Fun-1.4 |
| MATH (Accuracy) | AFlow | 32.49 | 35.17 ± 1.96 | Fun-1.1 |

Joint Optimization (3 iterations on MATH):

| Optimization Strategy | Performance Gain (vs. Default MAS) |
|---------|----------------------|
| Best Single Dimension | ~2.9% |
| Joint Iteration of Top 2 Dimensions (3 rounds) | ~9.6% |

### Ablation Study
OMAC-C: Removes the Contrastive Comparator and only uses the Semantic Initializer to select the highest-scoring candidate from the initial set.

| MATH Dimension | OMAC-C | OMAC | Comparator Gain |
|----------|--------|------|---------------|
| Str-1 | 32.64 | 33.34 | +0.70 |
| Str-2 | 32.67 | 33.41 | +0.74 |
| Str-3 | 32.76 | 33.70 | +0.94 |
| Fun-1.1 | 34.20 | 35.17 | +0.97 |
| Fun-2 | 32.71 | 33.95 | +1.24 |

### Key Findings
- **Independence of all 5 dimensions**: Single-dimension average gains are 3.6% for HumanEval, 2.8% for MMLU, and 4.9% for MATH, proving the 5D division captures orthogonal optimization directions.
- **Functional dimensions typically yield higher gains than structural dimensions**: Fun-1.x provided the highest single-dimension improvement across all three tasks, suggesting that modifying agent prompts is more direct than modifying collaboration topologies.
- **Synergistic gains from joint optimization**: On MATH, the gain rose from 2.9% (single-best) to 9.6% (joint), indicating synergy between structural and functional optimization. The "top-two strongest dimensions" strategy consistently outperforms random or weakest combinations.
- **Indispensability of the Comparator**: Removing contrastive reasoning (OMAC-C) caused performance drops across all dimensions, proving supervised contrastive reasoning is more efficient than pure exploration; however, OMAC-C still outperformed DyLAN, showing the inherent value of the Semantic Initializer’s exploration.
- **Reduced inference costs**: Dynamic agent selection and communication flow optimization allow OMAC to use fewer agents during inference compared to baselines, significantly reducing token consumption.

## Highlights & Insights
- **Formalizing MAS optimization into a 5D coordinate system**: While previous works defined "optimization" disparately, this paper provides a clear graph-theoretic framework (node capability × graph structure) that covers existing works and is easily extensible.
- **Contrastive reasoning as a lightweight supervised signal**: Without gradients or RL training, the framework uses the LLM's own reasoning to translate "poor performance" into "prompt improvement." This is a nearly universal paradigm applicable to any "natural language optimization" scenario.
- **Iterative "one-dimension-at-a-time" principle**: This engineering choice recognizes the limits of LLM attribution. Changing multiple variables simultaneously leads to incorrect causal attribution, as quantified by the ablation study.
- **Complementarity with any basic MAS design**: Since OMAC's input is an "existing MAS configuration," it can be used to further improve any manually designed or future complex MAS.

## Limitations & Future Work
- **Reliance on supervised signals**: Not directly applicable to open-ended tasks without clear metrics (e.g., open dialogue quality, long-range agent planning), though the authors performed minor supplementary experiments on GAIA.
- **Optimization quality capped by LLM reasoning**: The primary experiments used GPT-3.5; weaker LLMs as Comparators might lead to unstable contrastive reasoning. Appendix experiments with GPT-4 are present but not fully expanded.
- **Combinatorial explosion in joint optimization**: Fully joining all 5 dimensions requires $5^k$ times the computation of single-dimension optimization (where $k$ is iterations), limiting the framework to selecting the best 2 or 3 dimensions.
- **Unaddressed agent role conflicts**: Fun-2 (adding new agents) may introduce role overlap or contradictions within the MAS, which is not currently detected or handled.
- **Empirical subset sampling**: While subset evaluation saves compute, the optimal subset scale and sampling strategy lack deep theoretical analysis.

## Related Work & Insights
- **vs. DyLAN (2024)**: DyLAN uses unsupervised Agent Importance Scores to optimize only candidate selection (Str-1); OMAC covers 5 dimensions with supervised reasoning and outperforms DyLAN even on Str-1.
- **vs. ADAS / AFlow**: These focus on architecture search (prompt evolution / MCTS) optimizing Fun-2 and some Str-3; OMAC is consistently superior across broader dimensions using a lighter contrastive reasoning algorithm.
- **vs. G-Designer / MaAS**: Pure structural search methods; OMAC is competitive in structural dimensions (beating them in Str-1/2/3) while additionally optimizing functional dimensions.
- **vs. Gradient-based methods (SFT / prompt tuning)**: These are single-agent approaches that cannot handle MAS multi-step collaboration; OMAC replaces gradients with LLM reasoning.
- **vs. RL-based MAS optimization (Shao 2024, Liu 2025)**: Often limited to single-step interactions or shared policies; OMAC handles multi-step, role-specific MAS more naturally.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The 5D division and dual-actor contrastive reasoning provide clear conceptual contributions. While individual components are not revolutionary, their combination creates a new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes 3 classic benchmarks + 2 difficult benchmarks (MBPP/GAIA) + ablation + multiple baseline comparisons, with complete hyperparameter studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework and intuitive diagrams; logic for each design is well-explained despite minimal formulas.
- **Value**: ⭐⭐⭐⭐ Acts as a plug-and-play optimization suite for practical MAS, and the paradigm is transferable to any "natural language instruction optimization" scenario.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)
- [\[ICML 2026\] MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent](maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/multi_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)

</div>

<!-- RELATED:END -->
