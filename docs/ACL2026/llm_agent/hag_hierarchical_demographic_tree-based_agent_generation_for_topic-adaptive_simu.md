---
title: >-
  [Paper Note] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation
description: >-
  [ACL 2026][LLM Agent][Agent Generation] This paper proposes HAG, a framework that formalizes population-level agent generation as a two-stage hierarchical decision process — first constructing a topic-adaptive demographi…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agent Generation"
  - "Population Simulation"
  - "Hierarchical Decision-Making"
  - "Topic Adaptation"
  - "Agent-Based Modeling"
date: 2026-05-08
content_hash: 4c49228daa7c9796
---

# HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation

**Conference**: ACL 2026
**arXiv**: [2601.05656](https://arxiv.org/abs/2601.05656)  
**Code**: [https://github.com/Libra117/HAG](https://github.com/Libra117/HAG)  
**Area**: LLM Agent
**Keywords**: Agent Generation, Population Simulation, Hierarchical Decision-Making, Topic Adaptation, Agent-Based Modeling

## TL;DR
This paper proposes HAG, a framework that formalizes population-level agent generation as a two-stage hierarchical decision process — first constructing a topic-adaptive demographic distribution tree via a world knowledge model to achieve macro-level distributional alignment, then combining real-data retrieval with LLM-based agent augmentation to ensure micro-level individual consistency. HAG reduces population alignment error by an average of 37.7% and improves sociological consistency by 18.8% across multi-domain benchmarks.

## Background & Motivation

**Background**: Agent-Based Modeling (ABM) is increasingly important in computational social science, economic modeling, and personalized recommendation. These simulation systems rely heavily on user agents to simulate preferences and interaction behaviors, and agent quality directly determines the fidelity of the simulation system.

**Limitations of Prior Work**: Existing agent generation methods fall into two categories: (1) data retrieval-based methods that construct agent pools from real user logs, but are inherently static and cannot adapt to unseen or data-scarce topics; (2) LLM-based generation methods that produce agent personas via predefined schemas or textual reasoning, but lack explicit modeling of the joint distribution over multi-dimensional attributes — independently generating each agent causes the population distribution to deviate from reality.

**Key Challenge**: No existing method simultaneously achieves topic-adaptive macro-level population distribution modeling and sociologically valid micro-level individual attribute consistency. Independently generated agents may exhibit attribute contradictions (e.g., mismatched age and occupation), while static retrieval cannot cover emerging topics.

**Goal**: Design a population-level agent generation framework that simultaneously satisfies macro-level distributional alignment and micro-level individual consistency.

**Key Insight**: The authors observe that demographic structure is topic-dependent (e.g., user population distributions differ greatly between technology discussions and eldercare discussions), and therefore formulate population generation as a hierarchical conditional probability inference problem.

**Core Idea**: A world knowledge model (WKM) is used to construct a topic-adaptive demographic distribution tree in a top-down manner, capturing the joint distribution over multi-dimensional attributes via hierarchical conditional probabilities, followed by real-data instantiation combined with LLM augmentation to generate the final population.

## Method

### Overall Architecture
HAG operates in two stages: (1) Topic-adaptive distribution tree construction — given a target topic, the WKM infers hierarchical conditional probabilities over demographic attributes, generating a distribution tree from the topic down to complete personas; (2) Real-data instantiation and augmentation — real users are retrieved from the World Values Survey database according to the leaf-node distributions of the tree, with LLM-based constrained augmentation applied to nodes where data is insufficient.

### Key Designs

1. **Topic-Adaptive Distribution Tree Construction**:

    - Function: Transforms an abstract topic into a concrete joint distribution over multi-dimensional demographic attributes.
    - Mechanism: The WKM first identifies and ranks topic-relevant demographic dimensions (e.g., age > gender > education), establishing the hierarchical ordering of tree levels. The tree is then expanded layer by layer in a top-down fashion: node values and edge weights at each layer are determined by the WKM inferring the conditional probability $P(f^{(l)}=v^{(l)} | f^{(1:l-1)}=v^{(1:l-1)}, t)$. Each leaf node corresponds to a complete persona, with its target proportion equal to the product of all edge weights along the root-to-leaf path.
    - Design Motivation: Modeling attribute dependencies via hierarchical conditional probabilities rather than independent sampling ensures that the macro-level joint distribution aligns with the topic.

2. **Real-Data Instantiation and Agent Augmentation**:

    - Function: Converts the distribution tree into a concrete agent population, ensuring micro-level individual authenticity.
    - Mechanism: For each leaf-node persona, the required agent count is computed as $n(\mathbf{v}) = \text{Round}(N \cdot W(\mathbf{v}|t))$. Matching real users are retrieved from the World Values Survey database (HIT nodes are directly sampled); for MISS nodes with insufficient data, LLM-based augmentation is applied under the constraints of the corresponding tree path.
    - Design Motivation: Prioritizing real data ensures micro-level consistency, while LLM augmentation constrained by tree paths prevents incompatible attribute combinations.

3. **PACE Evaluation Framework**:

    - Function: Assesses generation quality along two complementary dimensions — population alignment and sociological consistency.
    - Mechanism: Population alignment is measured using JSD/KL divergence for distributional fidelity and the Gini-Simpson index for diversity error; sociological consistency extracts dominant prototypes via clustering to evaluate typicality, and performs internal self-consistency and contextual plausibility checks on each individual agent.
    - Design Motivation: Existing evaluation approaches lack a dedicated quantitative framework for agent population generation; both statistical alignment and semantic plausibility must be considered simultaneously.

### Loss & Training
HAG requires no training. It directly leverages a pretrained LLM as the WKM for inference, followed by retrieval from and augmentation of an existing database.

## Key Experimental Results

### Main Results
Evaluated across three domains — Bluesky (social simulation), Amazon (product recommendation), and IMDB (movie review):

| Method | Bluesky JSD↓ | Bluesky KL↓ | Bluesky ArchRel↑ | Bluesky IndCon↑ |
|--------|-------------|-------------|-------------------|-----------------|
| Random Select | 0.628 | 2.489 | 3.000 | 2.599 |
| Topic-Retrieval | 0.578 | 5.725 | 3.250 | 2.928 |
| LLM Generate | 0.539 | 2.487 | 3.063 | 3.197 |
| HAG-Flat | 0.401 | 2.436 | 3.750 | 3.324 |
| **HAG (Ours)** | **0.345** | **1.657** | **3.813** | **3.617** |

### Ablation Study

| Configuration | JSD↓ | KL↓ | Note |
|---------------|------|-----|------|
| HAG (Full) | 0.345 | 1.657 | Full model |
| HAG-Flat | 0.401 | 2.436 | Hierarchical tree removed; flat generation |
| LLM Generate | 0.539 | 2.487 | Direct LLM generation without tree structure |

### Key Findings
- HAG reduces population alignment error by an average of 37.7% and improves sociological consistency by 18.8% across all three domains.
- The hierarchical tree structure is critical: HAG-Flat (without hierarchical conditional probabilities) degrades JSD by approximately 16% relative to the full HAG model.
- The real-data retrieval combined with augmentation strategy effectively avoids the "Frankenstein Agent" problem (agents with contradictory, patchwork attributes).

## Highlights & Insights
- Formalizing population-level agent generation as a hierarchical decision process is an elegant modeling choice; combining chain-decomposed conditional probabilities with a tree structure achieves both interpretability and generation quality.
- The PACE evaluation framework fills a gap in agent population generation assessment, providing a systematic evaluation scheme from both statistical and semantic dimensions that can be reused for other population simulation tasks.
- Leveraging WKM world knowledge to infer topic-relevant demographic distributions avoids the bottleneck of relying on domain experts to manually design population schemas.

## Limitations & Future Work
- Tree construction depends on WKM quality; inference may be inaccurate for very rare or emerging topics.
- Only the World Values Survey is used as a real-data source, limiting cultural and geographic coverage.
- Dimension ordering affects results, but the optimality of the automated ordering lacks theoretical guarantees.
- Future work may explore dynamically updating the tree structure to adapt to real-time shifts in social trends.

## Related Work & Insights
- **vs. LLM Generate (direct generation)**: Direct LLM-based agent generation ignores the population-level joint distribution; HAG explicitly models attribute dependencies via tree structure.
- **vs. Topic-Retrieval**: Retrieval-based methods are constrained by existing data coverage; HAG achieves topic adaptation for data-absent topics through WKM inference and LLM augmentation.
- **vs. WorldValuesBench**: HAG inherits its attribute taxonomy but extends it with dynamic topic-adaptive capability.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical distribution tree modeling approach is novel, combining population generation with conditional probability inference.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across three domains is broad; the PACE evaluation framework is well-designed.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear; problem formulation and method description are logically coherent.
- Value: ⭐⭐⭐⭐ Practically valuable for the agent simulation field; the evaluation framework is broadly applicable.
- Overall: ⭐⭐⭐⭐ Clear problem definition, well-motivated design, and sufficient experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[AAAI 2026\] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](../../AAAI2026/llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)

</div>

<!-- RELATED:END -->
