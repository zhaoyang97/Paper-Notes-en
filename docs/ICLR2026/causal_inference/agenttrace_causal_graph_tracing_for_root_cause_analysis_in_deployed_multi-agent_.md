---
title: >-
  [Paper Note] AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems
description: >-
  [ICLR 2026][Causal Inference][multi-agent debugging] This paper proposes AgentTrace, a framework that constructs causal graphs from execution logs of multi-agent systems and localizes root cause nodes via backward tracin…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "multi-agent debugging"
  - "causal graph tracing"
  - "root cause analysis"
  - "execution log analysis"
  - "positional features"
date: 2026-05-08
content_hash: 50e36562ee2d25fa
---

# AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems

**Conference**: ICLR 2026
**arXiv**: [2603.14688](https://arxiv.org/abs/2603.14688)  
**Code**: [https://github.com/zwang000/AgentTrace](https://github.com/zwang000/AgentTrace)  
**Area**: Causal Inference
**Keywords**: multi-agent debugging, causal graph tracing, root cause analysis, execution log analysis, positional features

## TL;DR
This paper proposes AgentTrace, a framework that constructs causal graphs from execution logs of multi-agent systems and localizes root cause nodes via backward tracing combined with lightweight feature-based ranking (a weighted linear combination of five feature groups). On 550 synthetic fault scenarios, AgentTrace achieves Hit@1 of 94.9% with a latency of 0.12 seconds—69× faster than LLM-based analysis.

## Background & Motivation

**Background**: LLM-based multi-agent systems (e.g., AutoGen, MetaGPT) are increasingly deployed in customer support, DevOps, and research assistant scenarios, yet fault diagnosis remains extremely difficult—errors cascade across multiple agents, and the point of manifestation is often far removed from the root cause.

**Limitations of Prior Work**: (a) Traditional debugging methods inspect components individually and cannot capture cross-agent causal dependencies; (b) manual inspection of long execution traces is slow and unreliable; (c) LLM-based debugging approaches incur substantial inference cost and perform poorly on cross-agent issues.

**Key Challenge**: The distributed and emergent nature of multi-agent workflows makes root cause localization highly challenging, requiring an understanding of inter-agent information flow and causal relationships.

**Goal**: To design a lightweight, post-hoc fault diagnosis framework that does not rely on LLM inference and can rapidly localize root causes from execution logs.

**Key Insight**: Drawing inspiration from distributed systems tracing (e.g., Jaeger/Zipkin), but adapted to the LLM multi-agent setting—agent operations are modeled as nodes in a causal graph, with information flows as edges.

**Core Idea**: Reconstruct a causal graph from logs → perform backward tracing from error nodes → rank candidate root causes using interpretable structural and positional features.

## Method

### Overall Architecture
AgentTrace proceeds in three stages: (1) construct a causal DAG from execution logs; (2) collect candidate nodes via BFS backward traversal from error nodes; (3) rank candidates using a weighted linear combination of features. The entire pipeline requires no LLM inference and runs at CPU-level cost.

### Key Designs

1. **Causal Graph Construction**

    - Function: Identify three types of causal edges from execution logs.
    - Mechanism: **Sequential edges**—reasoning flow between consecutive operations within the same agent; **communication edges**—message send-receive events between agents; **data dependency edges**—data producer-consumer relationships identified via variable reference tracking.
    - Design Motivation: The three edge types cover the primary information flow patterns in multi-agent systems, and the resulting DAG fully captures the causal structure among operations.

2. **Backward Tracing Algorithm**

    - Function: Starting from the error node $v_{\text{error}}$, perform BFS backward traversal to collect all ancestor nodes within a depth limit as the candidate set.
    - Mechanism: Standard BFS with complexity $O(|V|+|E|)$; a depth parameter $d$ controls the search scope.
    - Design Motivation: The root cause must be a causal ancestor of the error node; backward tracing reduces the search space from all nodes to the relevant subgraph.

3. **Node Ranking Algorithm**

    - Function: Score and rank candidate nodes according to $\text{score}(v) = \sum_{i} w_i \cdot F_i(v)$.
    - Mechanism: A weighted linear combination of five feature groups:
        - **Positional features** ($w_p=0.70$): normalized position, distance to the error node, trace depth.
        - **Structural features** ($w_s=0.20$): out-degree, betweenness centrality, fan-out ratio.
        - **Content features** ($w_c=0.05$): error keywords, uncertainty markers, content length.
        - **Flow features** ($w_f=0.03$): whether the node involves cross-agent communication, role criticality.
        - **Confidence features** ($w_e=0.02$): explicit confidence scores, hedging language.
    - Design Motivation: Positional features receive the highest weight (0.70) because in hierarchical multi-agent workflows, early planning and routing decisions exert disproportionate influence on downstream behavior—an upstream error cascades and amplifies through the causal chain. Weights were determined via grid search over 50 validation scenarios.

### Runtime Characteristics
- Average processing time: 0.12 seconds, with no LLM inference overhead.
- Suitable for interactive debugging workflows.

## Key Experimental Results

### Main Results
Evaluated on 550 synthetic fault scenarios across 10 domains:

| Method | Hit@1 | Hit@3 | MRR |
|--------|-------|-------|-----|
| Random | 9.1% | 27.3% | 0.18 |
| First Node | 3.6% | 10.9% | 0.07 |
| Last Node | 12.7% | 38.2% | 0.25 |
| LLM Analysis (GPT-4) | 68.5% | 81.4% | 0.74 |
| **AgentTrace** | **94.9%** | **98.4%** | **0.97** |

McNemar's test confirms that AgentTrace significantly outperforms all baselines ($p < 0.001$); Cohen's $h = 0.77$ (large effect size) versus LLM Analysis.

### Ablation Study

| Feature Group | Hit@1 (Standalone) |
|--------------|-------------------|
| Position only | 87.3% |
| Structure only | 34.5% |
| Content only | 28.7% |
| Flow only | 15.2% |
| Confidence only | 12.1% |
| All features | **94.9%** |

### Key Findings
- Positional features alone account for 87.3% Hit@1, indicating that the "early error → late symptom" pattern is highly consistent in hierarchical workflows.
- Structural features contribute an additional 7.6% (87.3% → 94.9%), demonstrating independent value in topological information.
- Content, flow, and confidence features offer limited standalone performance but contribute positively at the margin when combined.
- AgentTrace latency of 0.12s versus 8.3s for LLM Analysis (69× speedup) makes it genuinely suitable for production environments.
- Performance is consistent across domains (Technical 96.4%, Knowledge 96.5%, Planning 91.3%).

## Highlights & Insights
- **Pragmatic design**: A simple weighted linear feature combination replaces LLM inference, lifting Hit@1 from GPT-4's 68.5% to 94.9%—on this specific task, structured features prove more effective than LLM comprehension.
- **Deeper implications of positional features**: The authors correctly note that this is not merely a benchmark artifact—in hierarchical multi-agent systems, upstream decisions constraining downstream solution spaces is a structural causal property. This finding also carries implications for agent system design: validation should be strengthened at early decision nodes.
- The transfer of distributed systems tracing concepts to the LLM agent domain offers a broadly inspiring methodological direction.

## Limitations & Future Work
- **Synthetic benchmark limitations**: All 550 scenarios involve artificially injected single root causes; real-world multi-agent failures typically involve multiple intertwined root causes.
- **Over-reliance on positional features**: A weight of 0.70 may be excessive in real-world settings—the assumption that root causes consistently appear early may not hold universally.
- The framework assumes complete and accurate execution logs; production environments may suffer from log loss or incompleteness.
- Scenario scale is modest (8–15 actions per trace); performance on large-scale agent workflows (hundreds of steps) remains unknown.
- All results are based on synthetic benchmarks; validation on real production systems is absent.
- As a single-author paper, the benchmark design may be subject to unintentional bias.

## Related Work & Insights
- **vs. LLM Self-Debug (Chen et al. 2024)**: LLM-based methods incur high inference cost and perform poorly on cross-agent issues (Hit@1 of only 68.5%); AgentTrace avoids LLM inference but relies on log structure.
- **vs. Jaeger/Zipkin**: Distributed tracing tools operate on RPC-level request metadata; AgentTrace handles agent messages carrying semantic content.
- **vs. Traditional RCA**: Conventional methods apply statistical techniques or PageRank; AgentTrace employs domain-specific features tailored to agent workflow topology.

## Rating
- Novelty: ⭐⭐⭐ The method itself is relatively straightforward (BFS + linear feature combination); the contribution lies in problem formulation and cross-domain conceptual transfer.
- Experimental Thoroughness: ⭐⭐⭐ Ablations are comprehensive, but all experiments rely on synthetic data; real-system validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Concise and clear; problem motivation and method description are well-presented.
- Value: ⭐⭐⭐⭐ Defines an important new problem (multi-agent RCA) and provides a practical baseline solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ACL 2026\] iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](../../ACL2026/causal_inference/itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)

</div>

<!-- RELATED:END -->
