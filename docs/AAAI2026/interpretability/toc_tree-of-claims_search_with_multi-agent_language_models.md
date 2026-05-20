---
title: >-
  [Paper Note] ToC: Tree-of-Claims Search with Multi-Agent Language Models
description: >-
  [AAAI 2026][Interpretability][Patent claim optimization] This paper proposes the Tree-of-Claims (ToC) framework, which models patent claim editing as a structured search problem. Through MCTS combined with EditorAgent/Ex…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Patent claim optimization"
  - "Monte Carlo Tree Search"
  - "multi-agent collaboration"
  - "uncertainty-awareness"
  - "structured editing"
date: 2026-05-08
content_hash: 89c1845bfd72f52a
---

# ToC: Tree-of-Claims Search with Multi-Agent Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.16972](https://arxiv.org/abs/2511.16972)  
**Code**: [ysy2003/ToC](https://github.com/ysy2003/ToC)  
**Area**: Interpretability
**Keywords**: Patent claim optimization, Monte Carlo Tree Search, multi-agent collaboration, uncertainty-awareness, structured editing

## TL;DR

This paper proposes the Tree-of-Claims (ToC) framework, which models patent claim editing as a structured search problem. Through MCTS combined with EditorAgent/ExaminerAgent multi-agent collaboration, ToC jointly optimizes novelty, scope preservation, and semantic consistency, achieving an average improvement of approximately 8% in overall score over zero/few-shot LLM baselines.

---

## Background & Motivation

**Task Importance**: Drafting and revising patent claims directly determines the legal scope, technical breadth, and commercial value of intellectual property, making it a high-stakes legal-technical task requiring high precision.

**Bottleneck of Manual Workflows**: Traditional practice relies on experienced patent attorneys performing repeated manual revisions—a process that is time-consuming, costly, inconsistent across practitioners, and difficult to scale.

**Limitations of Existing LLM Approaches**: Models such as GPT-4 can generate fluent text but operate in single-turn or few-shot modes, lacking iterative structured reasoning and producing uncontrolled outputs that may inadvertently broaden or narrow the claim scope.

**Lack of Controllability**: Existing multi-agent systems (e.g., AutoPatent, EvoPat) introduce collaborative mechanisms but offer opaque editing operations with high uncertainty, preventing patent practitioners from verifying each modification step by step.

**Potential of Search Methods**: MCTS has demonstrated value for exploring multiple reasoning paths in complex tasks (e.g., Tree-of-Thoughts), yet its direct application to legal text presents unique challenges around coherence, compliance, and scope preservation.

**Core Insight**: The authors observe that real-world patent revision workflows inherently involve two roles—"proposing legally valid edits" and "evaluating edits against novelty and examination standards"—which map naturally onto an Editor–Examiner dual-agent collaboration managed by MCTS over the large edit space.

---

## Method

### Overall Architecture

ToC models patent claim optimization as a **sequential decision problem**: given an initial claim $C_0$ and a set of prior art documents $P$, a sequence of atomic edit actions $a_t = (o_t, e_t, r_t, c_t)$ (operation type, target element, reasoning chain, confidence) generates a revised claim $C_T$, with the objective of maximizing cumulative reward $\mathcal{A}^* = \arg\max_{\mathcal{A}} \mathbb{E}[\sum_{t=0}^{T-1} R(s_t, a_t)]$. The search follows the four phases of MCTS (selection → expansion → simulation → backpropagation), augmented with uncertainty gating and progressive widening.

### Key Design 1: Ten Atomic Edit Operations

- **Function**: Defines 10 atomic operations—AddNovelFeature, ReplaceSynonym, ReframeViaFigure, DropElement, MergeElements, SplitElement, AddLimitation, ModifyRelationship, ChangeOrder, and AddDependency—along with a priority ordering among them (e.g., AddNovelFeature must precede ReplaceSynonym).
- **Mechanism**: Free-form text editing is discretized into composable atomic operations, so that each branch of the search tree corresponds to an explicit and interpretable modification.
- **Design Motivation**: Every edit step is made traceable and auditable to satisfy the transparency requirements of the legal domain, while constraining the action space to prevent combinatorial explosion of the search tree.

### Key Design 2: ExaminerAgent

- **Function**: Performs element-wise disclosure analysis of each claim element against prior art, producing structured assessments in strict JSON format—including status (Disclosed/NotDisclosed/PartiallyDisclosed), evidence citations, confidence $c_i \in [0,1]$, and epistemic uncertainty $\sigma_i$.
- **Mechanism**: Simulates the chain-of-thought reasoning of a real patent examiner, checking each technical feature for synonymous, functional, and structural equivalence.
- **Design Motivation**: Provides the EditorAgent with precise information about which elements have been disclosed, enabling targeted editing; uncertainty flags automatically identify boundary cases requiring human intervention.

### Key Design 3: EditorAgent

- **Function**: Based on ExaminerAgent feedback, selects the optimal edit from the 10 operations for each disclosed element, generating revised text along with justification.
- **Mechanism**: Favors minimal-change edits to overcome cited evidence while preserving legal language style and technical feasibility.
- **Design Motivation**: Avoids sweeping rewrites that could unnecessarily broaden or narrow scope, ensuring each edit is a precise and targeted adjustment.

### Key Design 4: Uncertainty-Aware MCTS Search

- **Function**: During selection, estimates epistemic variance $\sigma_{\text{epi}}(n)$ for each node; paths exceeding threshold $\sigma^{\text{epi}}_{\max} = 0.2$ are pruned or flagged for human review. Total variance is decomposed into epistemic and aleatoric components $\sigma^{\text{total}} = \sigma^{\text{epi}} + \sigma^{\text{ale}}$, with only the epistemic term entering the uncertainty penalty.
- **Mechanism**: The UCT selection formula $\text{UCT}(n) = Q(n)/N(n) + c\sqrt{\ln N(p)/N(n)}$ is augmented with $\sigma$-gating to implement self-auditing of what the model does not know.
- **Design Motivation**: Isolates model epistemic uncertainty from data noise, avoiding speculative edits in high-risk regions while retaining modification paths that are inherently sound despite high aleatoric uncertainty.

### Key Design 5: Progressive Widening

- **Function**: During expansion, only the top $K(n) = \lceil \alpha N(n)^{\delta} \rceil$ high-value child nodes are generated, with $(\alpha, \delta) = (2.0, 0.5)$.
- **Mechanism**: Encourages broad exploration early and focused exploitation later, adaptively controlling the branching factor.
- **Design Motivation**: Prevents exponential growth of the search tree while ensuring semantic validity, balancing search efficiency and output quality.

### Key Design 6: Multi-Objective Reward Function

- **Function**: Linearly combines five sub-objectives as $R(C_t) = w_1 R_{\text{cov}} - w_2 R_{\text{scope}} + w_3 R_{\text{novelty}} + w_4 R_{\text{cons}} - w_5 R_{\text{uncert}}$ with weights $(1.0, 0.5, 1.5, 0.8, 0.3)$.
- **Mechanism**: The coverage reward incentivizes converting "disclosed" elements to "not disclosed"; the scope penalty prevents unnecessary narrowing; novelty counts only changes the examiner identifies as inventive; consistency jointly measures legal readability and technical coherence; the uncertainty term suppresses speculative edits via epistemic variance.
- **Design Motivation**: Reflects the real-world trade-offs among competing objectives in patent revision; the highest novelty weight (1.5) reflects its primacy in patent examination.

---

## Key Experimental Results

### Dataset and Setup

- Data source: USPTO Office Actions dataset, wireless communication patents, comprising **1,145 patents** (106 granted / 1,039 rejected), **28,261 claims**, and **8,418 prior art citations**.
- Evaluation set: 500-sample hold-out, results averaged over 3 random seeds (mean ± SD).
- Models: Closed-source (OpenAI O1, GPT-4o, Claude-3.5 Sonnet) and open-source (Qwen2.5-VL-32B/72B), each serving dual roles as Examiner and Editor.

### Table 1: Core Reward Metrics Comparison (N=500, ±SD over 3 seeds)

| Model | Config | $R_{\text{cov}}$ | $R_{\text{scope}}$ | $R_{\text{novelty}}$ | $R_{\text{cons}}$ | Overall |
|------|------|:-:|:-:|:-:|:-:|:-:|
| GPT-4o | **+ToC** | **0.582** | 0.389 | **0.732** | **0.956** | **0.701** |
| GPT-4o | few-shot | 0.555 | 0.405 | 0.698 | 0.951 | 0.678 |
| GPT-4o | zero-shot | 0.520 | 0.417 | 0.659 | 0.947 | 0.647 |
| OpenAI O1 | +ToC | 0.560 | 0.374 | 0.712 | 0.942 | 0.680 |
| OpenAI O1 | few-shot | 0.525 | 0.388 | 0.685 | 0.937 | 0.658 |
| Claude-3.5 | +ToC | 0.548 | 0.370 | 0.703 | 0.945 | 0.675 |
| Qwen2.5-72B | +ToC | 0.534 | 0.361 | 0.682 | 0.930 | 0.658 |
| Qwen2.5-32B | +ToC | 0.507 | 0.351 | 0.665 | 0.924 | 0.639 |

### Table 2: Auxiliary Generation Quality Metrics

| Model | Config | JSON Compliance | PPL↓ | ROUGE-L | BLEU |
|------|------|:-:|:-:|:-:|:-:|
| GPT-4o | +ToC | 0.996 | **8.72** | **0.624** | **0.554** |
| GPT-4o | few-shot | 0.995 | 8.85 | 0.610 | 0.537 |
| OpenAI O1 | +ToC | 0.994 | 9.10 | 0.602 | 0.540 |
| Claude-3.5 | +ToC | 0.995 | 8.98 | 0.611 | 0.530 |
| Qwen2.5-72B | +ToC | 0.993 | 9.52 | 0.596 | 0.525 |
| Qwen2.5-32B | +ToC | 0.992 | 9.80 | 0.582 | 0.510 |

---

## Key Findings

1. **Consistent Gains from ToC**: ToC significantly outperforms zero-shot and few-shot baselines across all 5 models, with an average overall score improvement of approximately **8%**, peaking at **9%** (GPT-4o zero-shot → ToC: 0.647 → 0.701).
2. **GPT-4o Leads Across the Board**: Achieves best coverage, novelty, consistency, and perplexity, with an overall score of 0.701.
3. **Cross-Scale Transfer for Open-Source Models**: Qwen2.5-VL-72B with ToC approaches GPT-4o few-shot performance (0.658 vs. 0.678), with a notable gap between 32B and 72B, demonstrating the model-agnostic nature of the ToC framework.
4. **Ablation Study**: Removing any single module—uncertainty gating, progressive widening, or multi-agent interaction—degrades performance; uncertainty control and agent interaction have the largest impact on novelty and coverage.
5. **Hyperparameter Robustness**: Performance remains stable at 0.72–0.81 across the grid $\alpha \in [0.2, 0.8]$, $T_{\max} \in [5, 20]$, with a peak at $\alpha=0.6$, $T_{\max}=15$.
6. **Fast Search Convergence**: Approximately 70% of the final reward is realized within the first 6 iterations, with near-full convergence after 10 rounds.
7. **Expert Preference**: Five senior patent experts preferred ToC-generated revisions in approximately two-thirds of evaluations.

---

## Highlights & Insights

- **Elegant Problem Formulation**: Discretizing free-form text editing into 10 atomic operations with priority constraints makes MCTS search both tractable and interpretable—a modeling strategy transferable to contract revision, medical protocol optimization, and other structured text editing scenarios.
- **Uncertainty Decomposition**: Decomposing total variance into epistemic and aleatoric components and using only epistemic uncertainty for gating avoids spurious penalties from prior art phrasing variations, representing a practically sound engineering design.
- **Transparent and Auditable Edit Chains**: Unlike end-to-end generation, ToC produces a complete edit history with operation type, rationale, and confidence for each step, fully satisfying legal traceability requirements.
- **Reward Weight Design**: The novelty weight (1.5) substantially exceeds the scope penalty (0.5) and uncertainty penalty (0.3), reflecting the real-world patent prosecution strategy of prioritizing novelty before scope adjustment.

---

## Limitations & Future Work

1. **Single-Domain Dataset**: Validation is limited to USPTO wireless communication patents; generalization to highly specialized domains such as biopharmaceuticals or chemistry remains unclear.
2. **Computational Cost**: MCTS with up to 800 iterations and 3,600 seconds of search time, combined with dual LLM agent calls per step, poses significant API cost and latency challenges for deployment.
3. **Manual Reward Weights**: The five sub-objective weights are empirically tuned on a development set; automated weight learning or Pareto optimization is not explored.
4. **Error Analysis**: System control failures (12.2%) and unsupported novelty claims (9.9%) are the two dominant error types, indicating room for improvement in uncertainty calibration and verification mechanisms.
5. **Limited Multimodal Utilization**: Although the dataset contains figures, the practical effect of the ReframeViaFigure operation is not analyzed in depth.
6. **Human-in-the-Loop Workflow**: The specific interaction process triggered when $\sigma$-gating flags a case for human review is not sufficiently described.

---

## Related Work & Insights

- **Tree-of-Thoughts (Yao et al., 2023)**: ToC can be viewed as a domain-specialized extension of ToT for the legal domain, incorporating domain constraints and multi-agent evaluation.
- **MetaGPT / AutoGen**: General-purpose multi-agent frameworks; ToC achieves stronger task specificity by strictly binding roles to domain functions (examiner + editor).
- **ClaimBrush (Kawano et al.)**: Incorporates examiner feedback but performs only single-round revision; ToC's MCTS search enables multi-round iterative optimization.
- **Broader Implications**: The "atomic operations + search tree + dual-role evaluation" framework pattern is broadly applicable and can be directly transferred to structured text editing tasks requiring fine-grained control, such as contract clause optimization, clinical guideline revision, and scientific writing refinement.

---

## Rating

- Novelty: ⭐⭐⭐⭐ — Modeling patent editing as MCTS search with multi-agent collaboration; the uncertainty decomposition mechanism is creative
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 5 models × 3 configurations, ablations, sensitivity analysis, and expert evaluation
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear; prompt templates are fully disclosed
- Value: ⭐⭐⭐⭐ — The framework pattern is transferable to broader structured text editing scenarios, though the narrow domain limits direct impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [\[AAAI 2026\] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)
- [\[ICLR 2026\] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](../../ICLR2026/interpretability/mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](../../ICLR2026/interpretability/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)

</div>

<!-- RELATED:END -->
