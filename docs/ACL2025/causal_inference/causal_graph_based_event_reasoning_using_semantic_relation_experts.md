---
title: >-
  [Paper Note] Causal Graph based Event Reasoning using Semantic Relation Experts
description: >-
  [ACL 2025][Causal Inference][Causal Graph Generation] A causal event graph generation framework involving multi-round collaborative discussion among four types of semantic relation experts (Temporal, Discourse, Conditional, Commonsense) is proposed. Under a zero-shot setting, it achieves competitive results compared to fine-tuned models on multiple downstream tasks, such as event prediction and event forecasting, while providing explainable causal event chains.
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Causal Graph Generation"
  - "Event Reasoning"
  - "Multi-agent Collaboration"
  - "Semantic Relation Experts"
  - "Explainable Prediction"
date: 2026-05-08
content_hash: a103cb8303ba35dd
---

# Causal Graph based Event Reasoning using Semantic Relation Experts

**Conference**: ACL 2025  
**arXiv**: [2506.06910](https://arxiv.org/abs/2506.06910)  
**Code**: [github](https://github.com/StonyBrookNLP/causal-graphs)  
**Area**: Causal Reasoning  
**Keywords**: Causal Graph Generation, Event Reasoning, Multi-agent Collaboration, Semantic Relation Experts, Explainable Prediction

## TL;DR

A causal event graph generation framework involving multi-round collaborative discussion among four types of semantic relation experts (Temporal, Discourse, Conditional, Commonsense) is proposed. Under a zero-shot setting, it achieves competitive results compared to fine-tuned models on multiple downstream tasks, such as event prediction and event forecasting, while providing explainable causal event chains.

## Background & Motivation

- **Task Definition**: Given a set of events, construct a global causal event graph (nodes = events, directed edges = causal relationships) and utilize this graph to assist in reasoning tasks such as event prediction and forecasting.
- **Limitations of Prior Work**: Existing event reasoning methods mainly rely on distributional co-occurrence patterns of events, lacking explicit modeling of deep causal logic. Even SOTA LLMs still exhibit low accuracy in identifying causal relationships under standard in-context learning (ICL) settings.
- **Key Challenge**: Causal determination requires considering how events are embedded within the global context. For example, an earthquake itself is common, but "officials request aid" only occurs when both "city faces resource shortage" and "city suffers damage" hold simultaneously. Standard LLMs evaluating pairs in isolation easily overlook such subtle multi-event joint causalities.
- **Core Idea**: Four types of semantic relation experts are designed to generate a global causal graph through multi-round debate-like collaboration. This causal graph then drives downstream reasoning to achieve explainable event prediction.

## Method

### Overall Architecture

An LLM is used to simulate four "experts" focusing on different semantic dimensions. Through three stages—independent analysis, multi-round discussion, and judge consolidation—a global causal event graph is produced. Subsequently, the causal graph is applied to downstream tasks: Explainable Event Likelihood (EEL) prediction, Event Forecasting (ForecastQA), and Next Event Prediction (Narrative Cloze).

### Key Designs

**1. Four Types of Semantic Relation Experts**

Each expert is assigned a different perspective for causal determination:

| Expert | Dimension of Focus | Core Idea |
|------|---------|---------|
| Temporal Expert (Temporal) | Event temporal relationships | Temporal order is a necessary condition for causality. It narrows the search space by filtering temporally plausible event pairs. |
| Discourse Expert (Discourse) | Shared entity relationships | Event pairs sharing entities are more likely to have causal chains—operations on entities can trigger subsequent events. |
| Conditional Expert (Conditional) | Counterfactual preconditions | Determines via counterfactual reasoning: would event B still occur if event A were removed, identifying necessary preconditions. |
| Commonsense Expert (Commonsense) | Implicit background knowledge | Captures intermediate knowledge not explicitly mentioned in the text, bridging seemingly unrelated event pairs. |

**2. Multi-Round Collaborative Discussion Mechanism**

A "separation of concerns" strategy is adopted to avoid forcing the LLM to process all dimensions at once:

- **Initialization**: The four experts independently generate initial causal judgments and their reasoning paths.
- **Multi-round Discussion** (up to 3 rounds): In each round, each expert receives the responses of all other experts, analyzes them, revises their own causal links, and provides the rationale for changes. Experts can accept, refute, or supplement other experts' views.
- **Judge Consolidation**: A Causality Judge LLM summarizes all discussion results, resolves remaining disagreements, and outputs the final causal graph.

**3. Downstream Reasoning based on Causal Graph (CGEL)**

The causal graph is utilized for explainable event likelihood prediction: given a set of observed events and a query event, the system determines whether the query event can be inserted into the causal graph. If it can, the prediction is considered "likely," and a causal event chain is output as an explanation. This method is zero-shot and does not require fine-tuning on downstream tasks.

### Loss & Training

This method is a pure inference-time framework and requires no training or fine-tuning. GPT-4o and Llama-70B-instruct are used as the base LLMs, implementing the expert role assignment and discussion protocol through carefully designed prompts.

## Key Experimental Results

### Main Results: Causal Graph Generation Quality (CRAB Dataset, Graph-level Metrics)

| Method | LLM | BAcc | F1:Causal | F1:Non-Causal | Macro F1 |
|------|-----|------|-----------|---------------|----------|
| Direct (Zero-shot direct generation) | GPT-4o | 70.86 | 66.17 | 76.80 | 71.48 |
| Pairwise (Pairwise determination) | GPT-4o | 73.93 | 62.99 | 82.37 | 72.68 |
| Experts wo collab | GPT-4o | 74.92 | 70.21 | 78.23 | 74.22 |
| **Collab with experts** | **GPT-4o** | **79.27** | **75.62** | **82.80** | **79.21** |
| Direct | Llama-70B | 63.08 | 53.42 | 69.35 | 61.39 |
| **Collab with experts** | **Llama-70B** | **73.69** | **73.31** | **71.67** | **72.49** |

### Downstream Task Results

| Task | System | Accuracy |
|------|------|--------|
| Event Forecasting (ForecastQA) | GPT-4 baseline | 51.3% |
| | One-shot baseline | 50.0% |
| | **CGEL (Ours)** | **62.7%** |
| | BERT-large + MDS (Fine-tuned) | 67.4% |
| Next Event Prediction (NC) | ELM | 46.0% |
| | EGELM | 50.0% |
| | **CGEL with context** | **61.0%** |

In the EEL task, CGEL vs. One-shot baseline: wins by 41.6% in causality, 48.4% in informativeness, and 37.0% in coherence.

### Ablation Study

| Setting | BAcc | Macro F1 | Drop Relative to Full Method |
|------|------|----------|----------------|
| Collab with experts (Full) | 79.27 | 79.21 | — |
| Collab wo experts (No expert roles) | 75.39 | 75.51 | -3.70 |
| w/o Temporal Expert | 77.51 | 77.72 | -1.49 |
| w/o Precondition Expert | 77.48 | 77.26 | -1.95 |
| w/o Discourse Expert | 78.32 | 78.29 | -0.92 |
| w/o Commonsense Expert | 78.88 | 78.85 | -0.36 |

Removing any expert leads to a performance drop, with the precondition expert and temporal expert having the greatest impact.

### Debate Trajectory Analysis

| Expert | Initial Overlap with Gold | Post-Discussion Overlap with Gold | Contribution | Error Flip Rate |
|------|------------------|-------------------|--------|-----------|
| Temporal Expert | 13% | 33% | 64% | 0% |
| Discourse Expert | 17% | 24% | 64% | 0% |
| Precondition Expert | 17% | 22% | 46% | 67% |
| Commonsense Expert | 22% | 26% | 57% | 0% |

The temporal expert is initially the weakest but shows the greatest improvement after discussion; the precondition expert has the highest error flip rate.

## Highlights & Insights

1. **Heterogeneous Expert Collaboration Outperforms Homogeneous Debate**: Unlike approaches like ChatEval that let multiple identical-role LLMs debate, this work endows each agent with a different semantic relation expertise, achieving a true "separation of concerns." Experiments demonstrate that removing expert roles (Collab wo experts) causes a nearly 4-point drop in BAcc.

2. **Competitive Performance with Zero-Shot against Fine-tuned Models**: CGEL achieves 62.7% on ForecastQA, close to 67.4% from a fine-tuned BERT-large, without requiring any task-specific training data. It also outputs causal event chains as explanations, which is typically unachievable with fine-tuned models.

3. **Analyzable and Debuggable Debate Process**: The flipping, adding, and conflict patterns of each expert in the discussion are tracked in detail, forming a transparent decision path that facilitates future improvements.

## Limitations & Future Work

1. Reliance on the base LLM's causal understanding capability, which may deviate from human causal perception.
2. High computational cost of multi-round multi-expert discussion (requiring multiple LLM calls per scenario).
3. Lack of graded determination for causal strength, executing only binary causal/non-causal classification.
4. Using GPT-4 as an evaluator may introduce bias toward content generated by itself.
5. Potential to extend to more types of semantic relation experts and validation across more domains/languages.

## Rating

- **Novelty**: ★★★★☆ — The framework of heterogeneous semantic experts collaborating to generate a global causal graph is novel, differing fundamentally from existing multi-agent debate methods.
- **Value**: ★★★★☆ — Zero-shot and explainable, applicable to various scenarios like event prediction, forecasting, and explanation.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation dimensions including intrinsic evaluation, three extrinsic tasks, ablation, and debate trajectory analysis.
- **Writing Quality**: ★★★★☆ — Clear motivation and systematic framework description, though some symbol explanations are slightly redundant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[ICML 2026\] From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models](../../ICML2026/causal_inference/from_observation_to_intervention_a_causal_audit_of_expert_importance_in_mixture-.md)
- [\[ICLR 2026\] Query-Specific Causal Graph Pruning under Tiered Knowledge](../../ICLR2026/causal_inference/query-specific_causal_graph_pruning_under_tiered_knowledge.md)
- [\[ACL 2025\] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)
- [\[CVPR 2026\] CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization](../../CVPR2026/causal_inference/cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali.md)

</div>

<!-- RELATED:END -->
