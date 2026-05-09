---
title: >-
  [Paper Note] ChemAmp: Amplified Chemistry Tools via Composable Agents
description: >-
  [ACL 2026][Model Compression][Tool amplification] This paper proposes a novel "tool amplification" paradigm (distinct from conventional tool orchestration) and introduces the ChemAmp framework, which treats chemistry-specific tools (UniMol2, Chemformer, etc.) as composable building blocks to dynamically construct task-specialized super-agents. ChemAmp surpasses both domain-specific models and general-purpose LLMs on four core chemistry tasks—including molecular design and reaction prediction—while reducing inference token costs by 94%.
tags:
  - ACL 2026
  - Model Compression
  - Tool amplification
  - composable agents
  - chemistry AI
  - multi-agent systems
  - hierarchical composition
date: 2026-05-08
content_hash: 35feb38f49fde801
---

# ChemAmp: Amplified Chemistry Tools via Composable Agents

**Conference**: ACL 2026
**arXiv**: [2505.21569](https://arxiv.org/abs/2505.21569)
**Code**: [GitHub](https://github.com/Chang-pw/ChemAmp)
**Area**: Scientific AI / Chemistry
**Keywords**: Tool amplification, composable agents, chemistry AI, multi-agent systems, hierarchical composition

## TL;DR

This paper proposes a novel "tool amplification" paradigm (distinct from conventional tool orchestration) and introduces the ChemAmp framework, which treats chemistry-specific tools (UniMol2, Chemformer, etc.) as composable building blocks to dynamically construct task-specialized super-agents. ChemAmp surpasses both domain-specific models and general-purpose LLMs on four core chemistry tasks—including molecular design and reaction prediction—while reducing inference token costs by 94%.

## Background & Motivation

**Background**: LLM-based agents have demonstrated the ability to orchestrate multi-step tool-use workflows in the chemistry domain (e.g., ChemCrow, Coscientist), sequentially invoking tools such as RDKit and molecular generators to complete cross-task workflows.

**Limitations of Prior Work**: Existing approaches focus on "tool orchestration" (scheduling tool sequences across tasks), yet within-task performance remains bounded by the atomic capabilities of individual tools. Even state-of-the-art chemistry-specific tools (UniMol2, ChemDFM) achieve only 35% exact match in molecular description when used in isolation, and errors propagate through the reasoning chain.

**Key Challenge**: Tool orchestration optimizes inter-task tool scheduling, but the true bottleneck constraining agent performance is within-task tool capability limitations.

**Goal**: Shift from "tool orchestration" to "tool amplification"—enabling tools to exceed their individual atomic capabilities within a single task through dynamic composition.

**Key Insight**: Treat each tool as a composable building-block agent and construct higher-performing composite tools through hierarchical iterative encapsulation.

**Core Idea**: A two-stage amplification process—first encapsulating atomic tools into enhanced sub-agents (Stage 1), then composing sub-agents into a hierarchical network (Stage 2), with iterative refinement guided by adaptive scoring and automatic feedback.

## Method

### Overall Architecture

ChemAmp constructs an agent hierarchy through a two-stage bidirectional encapsulation engine. In **Stage 1** (atomic → composite amplification), each atomic tool is iteratively encapsulated into an Agent Composite Tool until performance ceases to improve, and all variants are registered in a tool library. In **Stage 2** (cross-composite collaboration), the best-performing tool from the library serves as a base and is combined with other top-$k$ tools to form higher-level composite tools, iterating until global performance stabilizes.

### Key Designs

1. **Dual Role of the Agent Composite Tool**:

    - **Function**: Serves simultaneously as a composable building block for higher-level agents and as an autonomous executor for chemistry sub-tasks.
    - **Mechanism**: Each $\mathcal{A}(t_1,\ldots,t_n)$ encapsulates multiple tools along with their coordination strategies, enabling both invocation by upper-level agents and independent execution. This duality allows ChemAmp to identify optimal enhancement points where tool coordination yields synergistic effects.
    - **Design Motivation**: Avoids naive stacking and enables genuine capability emergence.

2. **Two-Stage Iterative Encapsulation**:

    - **Function**: Automatically discovers optimal tool combinations.
    - **Mechanism**: Stage 1 iteratively encapsulates each atomic tool as $\mathcal{A}_i(t_k)$, scored by task metric $s_i$, continuing only when improvement exceeds threshold $\delta$. Stage 2 ranks the tool library, takes the top-1 tool as the base, and forms combinations $\{\mathcal{A}(t_1,t_2),\ldots,\mathcal{A}(t_1,t_k)\}$ with the top-$k$ tools, iterating until global performance no longer improves.
    - **Design Motivation**: Manual combination is infeasible and exhaustive search is prohibitively costly; iterative encapsulation with threshold control balances efficiency and effectiveness.

3. **Minimal Data Requirement (≤10 samples)**:

    - **Function**: Optimizes tool composition with extremely few validation samples.
    - **Mechanism**: Each task requires only ≤10 samples for composition scoring and selection. By leveraging the domain knowledge embedded in chemistry tools themselves, ChemAmp needs only a small number of examples to determine whether a combination yields improvement.
    - **Design Motivation**: Annotated data is scarce in the chemistry domain, necessitating a low-data-dependency approach.

## Key Experimental Results

### Main Results (Molecular Design — ChemLLMBench)

| Method | Exact Match | BLEU | FTS |
|--------|-------------|------|-----|
| ChemDFM-13B | 0.32 | 0.85 | 0.74 |
| Text+Chem T5 | 0.32 | 0.85 | 0.82 |
| GPT-4o | 0.01 | 0.57 | 0.54 |
| ChemAmp | **0.42** | **0.88** | **0.84** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| Stage 1 only | Improved | Single-tool enhancement is effective |
| Stage 1 + Stage 2 | Best | Cross-composite collaboration yields further gains |
| Vanilla multi-agent | Worse | Naive stacking underperforms structured composition |
| Token cost | 94% reduction | vs. vanilla multi-agent system |

### Key Findings
- ChemAmp comprehensively outperforms chemistry-specific models, general-purpose LLMs, and conventional agent orchestration systems across four core chemistry tasks.
- Inference token cost is only 6% of that of vanilla multi-agent systems, demonstrating exceptional efficiency.
- A bottom-up composition strategy outperforms top-down orchestration strategies.
- Exact match on molecular design improves from the prior SOTA of 0.32 to 0.42 (+31%), validating the practical effectiveness of tool amplification.

## Highlights & Insights
- **Paradigm Innovation**: The distinction between "tool amplification" and "tool orchestration" is clear and compelling, representing a shift from cross-task scheduling to within-task enhancement.
- **Efficiency and Effectiveness**: Surpassing SOTA while reducing inference token cost by 94% demonstrates that structured composition is far more efficient than brute-force stacking.
- **Generality**: Although applied to chemistry, the tool amplification paradigm is transferable to other scientific domains.
- **Low Data Requirement**: Composition optimization with ≤10 samples ensures strong practical utility.

## Limitations & Future Work
- **Dependence on GPT-4o as the core agent**: The effectiveness of the composition strategy may be constrained by the capabilities of the underlying LLM.
- **Evaluation limited to 100 instances on ChemLLMBench**: The test scale is relatively small.
- **Chemistry-domain specificity**: Applicability to other scientific domains requires further validation.
- Future directions include extending the framework to broader scientific domains, investigating the interpretability of composition strategies, and reducing reliance on closed-source LLMs.

## Related Work & Insights
- **vs. ChemCrow / Coscientist**: Representative tool orchestration systems effective at cross-task scheduling but incapable of enhancing within-task performance.
- **vs. ChemToolAgent**: Supports large tool sets and dynamic selection but remains within the orchestration paradigm.
- **vs. AgentPrune / GPTSwarm**: Automated workflow optimization without atomic tool-level enhancement.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "tool amplification" paradigm is novel and persuasive; the two-stage encapsulation engine is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across four chemistry tasks with ablation and efficiency analysis, though test scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ The orchestration-vs-amplification distinction figure is clear, and the algorithmic descriptions are complete.
- **Value**: ⭐⭐⭐⭐ Provides a new perspective for tool enhancement in scientific AI; the dual gains in efficiency and effectiveness carry practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)

</div>

<!-- RELATED:END -->
