---
title: >-
  [Paper Note] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond
description: >-
  [ACL 2025][LLM (Other)][Human-AI Cooperation] This paper presents the first systematic survey on the principles, formal taxonomy, and open challenges of Human-Model Cooperation. It proposes a taxonomy of three cooperation paradigms based on "who makes the final decision" (sequential, triage-based, and joint cooperation), and outlines the role frameworks and methodological roadmaps for each paradigm.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Human-AI Cooperation"
  - "Cooperation Paradigm Taxonomy"
  - "Sequential Cooperation"
  - "Triage-based Cooperation"
  - "Joint Cooperation"
  - "Cooperation Principles"
date: 2026-05-08
content_hash: 09811f70a6184687
---

# How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond

**Conference**: ACL 2025  
**arXiv**: [2501.05714](https://arxiv.org/abs/2501.05714)  
**Code**: None  
**Area**: LLM/NLP / Human-AI Cooperation  
**Keywords**: Human-AI Cooperation, Cooperation Paradigm Taxonomy, Sequential Cooperation, Triage-based Cooperation, Joint Cooperation, Cooperation Principles  

## TL;DR

This paper presents the first systematic survey on the principles, formal taxonomy, and open challenges of Human-Model Cooperation. It proposes a taxonomy of three cooperation paradigms based on "who makes the final decision" (sequential, triage-based, and joint cooperation), and outlines the role frameworks and methodological roadmaps for each paradigm.

## Background & Motivation

**Background**: With LLMs evolving from tools to agents with autonomous goals and strategies, human-model cooperation has emerged as a new paradigm in NLP. Significant progress has been made in various NLP tasks, including data annotation (Klie et al. 2020; Li et al. 2023a), information retrieval (Deng et al. 2023a), creative writing (Padmakumar & He 2022), and real-world problem-solving (Mehta et al. 2023).

**Limitations of Prior Work**: (1) Existing surveys (Wang et al. 2021a, 2023e; Wu et al. 2023; Gao et al. 2024) primarily focus on the "elements" of cooperation—such as user interfaces, message fusion, and evaluation methods—but lack a systematic analysis of how humans and models are formally organized into an effective cooperative team. (2) Different cooperative methods are scattered across specific applications, lacking a unified taxonomy to compare and comprehend their core differences. (3) Principles of cooperation (how to ensure rational behavior and reliable outputs from both parties) have not been systematically distilled or explicitly formulated.

**Key Challenge**: The number of human-AI cooperative methods is surging, yet they lack a unified analytical framework. Consequently, researchers struggle to answer the core question of "which cooperative paradigm is best suited for my scenario" because the applicability conditions, cost-benefits, and pros/cons of different paradigms (sequential, triage-based, and joint) have never been systematically analyzed and compared.

**Goal**: (1) Define the unified concepts and principles of human-model cooperation, distinguishing cooperation from collaboration and non-cooperation; (2) Propose a systematic taxonomy based on "who bears the responsibility for the final decision" to unify existing methods into three major cooperative paradigms; (3) Identify role frameworks, methodological roadmaps, and typical applications for each paradigm; (4) Identify key research frontiers and social impact issues.

**Key Insight**: Starting from Grice's conversational Cooperative Principle (sincerity, relation, manner, quantity), cooperation is formalized into three major types based on "who makes the final decision," which, supported by two role frameworks (helper-executor vs. equal partners), constitutes a complete analytical system.

**Core Idea**: Human-model cooperation can be systematically categorized into three paradigms based on decision-making responsibility: sequential cooperation (one party assists the other in decision-making), triage-based cooperation (tasks are allocated based on capability and completed independently), and joint cooperation (outputs from both parties are combined to produce the final decision).

## Method

### Overall Architecture

As a survey paper, the core contribution is the proposal of a unified human-model cooperation analytical framework consisting of three levels: (1) **Cooperative Principles**: reinterpreting Grice's cooperative principles into four maxims: Sincerity (no deception, supported by evidence), Relation (actions relevant to the task goal), Manner (clear and understandable expressions), and Quantity (sufficient but not redundant information); (2) **Role Frameworks**: defining two fundamental role relationships: the helper-executor framework (hierarchical, with one party leading decision-making and the other assisting) and the equal-partner framework (symmetric, with both sharing decision responsibility); (3) **Three Cooperation Paradigms**: categorizing all cooperation methods based on "who bears the final decision responsibility" into sequential cooperation (most common, one party assists the other step-by-step), triage-based cooperation (tasks assigned based on capability and completed independently), and joint cooperation (fusing output probabilities from both parties to generate the final result). Detailed methodological routes and typical application scenarios are further outlined for each paradigm.

### Key Designs

1. **Sequential Cooperation**:
    - Function: Two parties cooperate sequentially, where one party assists the other in making the final decision.
    - Mechanism: Divided into "Human-in-the-loop" (human-assisted model) and "Machine-in-the-loop" (model-assisted human) trajectories. In human-assisted models, the model makes the final decision but improves continuously through human feedback—categorized into training-based (RLHF, online learning) and training-free (ICL, model editing, rule learning) schemes. In model-assisted humans, the human makes the final decision while the model provides candidate solutions for selection or modification, where the key is to provide accurate and trustworthy advice.
    - Design Motivation: The most prevalent cooperative format, covering mainstream applications such as RLHF alignment training, AI-assisted writing, and code assistance.

2. **Triage-based Cooperation**:
    - Function: Allocates tasks to the most suitable party according to HABA-MABA (Humans-Are-Better-At / Machines-Are-Better-At) principles to complete tasks independently.
    - Mechanism: Evaluates model capability boundaries via internal allocators (modeling "hand over to human" as an extra output category using a triage-aware cross-entropy loss) or external allocators (filters based on prediction uncertainty, error rates, or data hardness estimations, such as an MLP predicting model error probability or ChatGPT estimating data difficulty).
    - Design Motivation: Highest efficiency—both parties work completely independently with zero interaction overhead, though the lack of feedback loops remains an inherent limitation.

3. **Joint Cooperation**:
    - Function: Probabilistically fuses outputs from both parties to generate a final result that surpasses either party alone.
    - Mechanism: Constructs a confusion matrix based on discrete human decisions to estimate human decision confidence, which is then Bayes-fused with model probability outputs. This can be implemented via supervised learning (pre-collected labeled data to estimate the confusion matrix) or unsupervised learning (EM algorithm estimation).
    - Design Motivation: Theoretically optimal (leveraging the complementarity of errors made by humans and models), but currently restricted to classification tasks. Joint cooperation for generative tasks remains a major research gap.

### Loss & Training

The survey summarizes the typical training methods for each paradigm:
- Human-assisted methods in sequential cooperation: RLHF/RLAIF alignment training (offline/online), training-free ICL schemes, model editing.
- Triage-based cooperation: triage-aware cross-entropy loss (utilizing an additional "hand over to human" category), dynamic threshold-based task allocation.
- Joint cooperation: EM estimation of human confusion matrices, clustering prototype-based enhancements to record historical human decisions.

## Key Experimental Results

### Systematic Comparison of Three Cooperation Paradigms

| Cooperation Paradigm | Final Decision Maker | Role Framework | Independent Decision | Human Cost | Information Utilization | Application Scenarios |
|---------|----------|---------|---------|---------|---------|---------|
| Sequential - Human-assisted | Model | Helper-Executor | No | Medium-High | Bidirectional Feedback | RLHF, Instruction Tuning |
| Sequential - Model-assisted | Human | Helper-Executor | No | Medium | Model → Human | AI Writing / Code Assistance |
| Triage | Respective responsibility | Equal Partners | Yes | Low | No Interaction | Data Annotation Triage |
| Joint | Jointly | Equal Partners | No | Medium | Output Fusion | Classification Decision Fusion |

### Coverage Comparison of Existing Surveys

| Survey | Elements (EC) | Formalization (FC) | Principles (PC) | Unique Contribution |
|------|---------|-----------|---------|---------|
| Wang et al. 2021a | ✓ | ✗ | ✗ | User interfaces, feedback types |
| Gao et al. 2024 | ✓ | ✗ | ✗ | Interaction mode classification |
| Wang et al. 2023e | ✓ | ✗ | ✗ | Message fusion, evaluation |
| **Ours** | ✓ | **✓** | **✓** | Cooperation principles + Three-paradigm taxonomy |

### Key Research Frontiers

| Frontier Direction | Core Challenge | Current Status |
|----------|---------|---------|
| Cross-paradigm standardized benchmarks | Lack of unified benchmarks to compare pros & cons of cooperative formats | **Completely Blank** |
| Human uncertainty estimation | Human decisions lack explicit uncertainty metrics | Existing methods are unreliable |
| Model alignment capability | LLMs tend to be 'one-size-fits-all', hard to adapt to different users | Preliminary exploration phase |
| LLM compliant behaviors | LLMs may exhibit deceptive or irrelevant content | Documented by existing cases |
| Joint cooperation → Generative tasks | Currently limited to classification tasks | **Core Research Gap** |

### Key Findings

- Sequential cooperation is the **most prevalent** cooperative form in NLP, yet it carries the highest human cost.
- Triage-based cooperation is highly efficient but restricted in information exchange—both parties work independently without a feedback loop.
- Joint cooperation is theoretically optimal but **limited to classification tasks**—how to fuse outputs from both parties in generative tasks remains an open problem.
- LLMs may violate the sincerity principle during cooperation, exhibiting deceptive behaviors (Huang et al. 2024d).
- Existing methods assume human feedback is always correct, whereas actual human decisions typically feature noise and bias.

## Highlights & Insights

- **Pioneering Unified Taxonomy**: The tripartite taxonomy based on "allocation of decision-making responsibility" is elegant and strong, unifying scattered cooperation methods under a single framework.
- **Principled Design**: Drawing from Grice's conversational maxims provides a solid theoretical foundation for compliant AI cooperation.
- **Identifying Key Blind Spots**: Spotlights the vacancy of joint cooperation in generative tasks, the difficulty of human uncertainty estimation, and potential violations of cooperation principles by models.
- **Discussion on Societal Impact**: Delves into real-world issues like trust calibration (over/under-reliance), regulatory accountability, and the "irony of automation."
- **Forward-Looking Perspective**: Proposes future directions such as self-evolving models, crowd-based training generalizations, and multi-party cooperation.

## Limitations & Future Work

- Focuses solely on single-human-single-model scenarios; complex dynamics of multi-party cooperation (multi-human/multi-model) are not covered.
- The distinction between cooperation and collaboration is somewhat blurred, as collaboration requires bidirectional communication and joint decision-making.
- Non-cooperative interactions (e.g., negotiation, persuasion) hold distinct research value but are excluded from the scope of this analysis.
- As a survey, this work cannot experimentally validate the relative superiority of the three paradigms, which necessitates future empirical validation.
- Combination of cooperative formats has not been explored—actual systems may integrate multiple hybrid paradigms.

## Related Work & Insights

- **Surveys on Human-AI Collaboration**: Wang et al. 2021a (human-computer interaction in NLP), Xi et al. 2023 (roles of agent cooperation).
- **Theoretical Foundations of Cooperation**: Grice 1975's Cooperative Principle → Reinterpreted in this paper within the human-AI context.
- **Representative Applications**: RLHF (Touvron et al. 2023), triage annotation (Li et al. 2023a), joint decision (Kerrigan et al. 2021).
- **Insights**: The three-paradigm taxonomy can serve as a decision-making guide for designing new AI collaborative systems. Practitioners can choose the most appropriate format based on task characteristics (e.g., triage for data annotation, sequential for writing assistance, and joint for high-risk decisions).

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](meraser_fingerprint_erasure.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)

</div>

<!-- RELATED:END -->
