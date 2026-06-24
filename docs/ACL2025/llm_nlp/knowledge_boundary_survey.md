---
title: >-
  [Paper Note] Knowledge Boundary of Large Language Models: A Survey
description: >-
  [ACL 2025][LLM (Other)][Knowledge Boundary] A formal defining framework for the knowledge boundary of LLMs is proposed, featuring three-tier nested boundaries (Outward⊂Parametric⊂Universal) and four categories of knowledge (PAK/PSK/MSU/MAU). The survey systematically reviews relevant research around three questions: "why, how to identify, and how to mitigate."
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Knowledge Boundary"
  - "Hallucination"
  - "Uncertainty Estimation"
  - "Calibration"
  - "Knowledge Categorization"
date: 2026-05-08
content_hash: 8eff54e081b7309a
---

# Knowledge Boundary of Large Language Models: A Survey

**Conference**: ACL 2025  
**arXiv**: [2412.12472](https://arxiv.org/abs/2412.12472)  
**Code**: [GitHub](https://github.com/li-moxin/knowledge_boundary_survey)  
**Area**: LLM NLP  
**Keywords**: Knowledge Boundary, Hallucination, Uncertainty Estimation, Calibration, Knowledge Categorization

## TL;DR

A formal defining framework for the knowledge boundary of LLMs is proposed, featuring three-tier nested boundaries (Outward⊂Parametric⊂Universal) and four categories of knowledge (PAK/PSK/MSU/MAU). The survey systematically reviews relevant research around three questions: "why, how to identify, and how to mitigate."

## Background & Motivation

### Background

**Background**: **Background**: LLMs store a vast amount of knowledge in their parameters, but limitations remain in memorizing and utilizing certain factual knowledge, leading to untruthful or inaccurate responses. **Limitations of Prior Work**: The concept of the Know-Unknown Quadrant is conceptual but lacks formalization; existing formal definitions only focus on specific LLMs. **Key Challenge**: The absence of a clear and unified definition for LLM knowledge boundaries hinders systematic identification and mitigation strategies. **Goal**: To provide a comprehensive, formal definition of knowledge boundaries and systematically review relevant research. **Key Insight**: Boundaries are defined across three dimensions: whether knowledge is known to humanity, whether it is embedded in parameters, and whether it is empirically verifiable. **Core Idea**: Knowledge is categorized into four classes: PAK (always answered correctly regardless of prompting), PSK (dependent on the prompt), MSU (unknown to the model but known to humanity), and MAU (unknown to humanity).

## Method

### Overall Architecture

The three-tier nested knowledge boundary is defined across three dimensions: (1) Outward (empirically verifiable); (2) Parametric (present in parameters); (3) Universal (the complete set of human knowledge). Based on this, four classes of knowledge are defined, and the survey is conducted around three research questions (RQs).

### Key Designs

1. **Formal Definition of Key Knowledge Classes**:
    - Function: Formalizes LLM knowledge into four classes: PAK, PSK, MSU, and MAU.
    - Mechanism: PAK: $K_{PAK}=\{k \in \mathcal{K} | \forall (q,a) \in \hat{Q}_k, P_\theta(a|q)>\epsilon\}$ — correct answers are generated regardless of the prompt formulation. PSK: Knowledge exists in parameters but is sensitive to prompts. MSU: Knowledge is not possessed by the model but is known to humanity. MAU: Knowledge is unknown to humanity.
    - Design Motivation: The distinction between PAK and PSK is highly insightful; the same piece of knowledge can be recognized as "known" or "unknown" depending on the prompt, directly guiding prompt engineering.

2. **Mapping Undesired Behaviors to Knowledge Types**:
    - Function: Maps three types of undesired LLM behaviors to specific knowledge types.
    - Mechanism: PSK $\rightarrow$ untruthful replies misled by context; MSU $\rightarrow$ factual hallucinations (insufficient domain knowledge, outdated knowledge, or overconfidence); MAU $\rightarrow$ random answers to ambiguous knowledge or biased replies to controversial topics.
    - Design Motivation: Provides a roadmap for targeted improvements—identifying the specific knowledge issue allows selecting the correct mitigation strategy.

3. **Classification of Identification and Mitigation Methods**:
    - Function: Systematically categorizes identification methods (uncertainty estimation, calibration, probing) and mitigation methods (prompt optimization, RAG, knowledge editing, abstention).
    - Mechanism: Identification methods are classified based on whether they require access to internal states. Mitigation methods are classified by knowledge type and the degree of parameter modification.
    - Design Motivation: Different knowledge types require distinct strategies—PAK/PSK utilizes prompt optimization, MSU utilizes RAG or editing, and MAU utilizes abstention or clarification.

### Loss & Training

As a survey paper, this work does not introduce new training procedures but systematically analyzes the training strategies of various mitigation methods: supervised fine-tuning (SFT) for abstention, reinforcement learning (RL) for honesty alignment, retrieval-augmented generation (RAG), and others.

## Key Experimental Results

### Main Results

The survey paper does not contain original experiments but systematically structures the key mappings:

| Knowledge Type | Undesired Behavior | Identification Method | Mitigation Strategy |
|---------|---------|---------|---------|
| PAK | None | High Probability Threshold Validation | — |
| PSK | Context Misdirection | Prompt Perturbation / Uncertainty Decomposition | Prompt Optimization / ICL / Reasoning / Decoding |
| MSU | Factual Hallucination | Semantic Consistency / Calibration / Probing | RAG / Knowledge Editing / Abstention |
| MAU | Bias / Randomness | Under-explored | Alignment Training / Clarifying Questions |

### Ablation Study

Key comparisons analyzed in the survey:

| Comparison Dimension | Finding |
|---------|------|
| Uncertainty Estimation vs. Calibration | The former focuses on the overall distribution, while the latter focuses on specific predictions—conceptually close but fundamentally different. |
| Epistemic Uncertainty vs. Aleatoric Uncertainty | Correspondence to the gaps between parametric boundaries and outward boundaries, respectively. |
| Abstention vs. Clarification | Abstention is effective for MAU but can result in over-abstention; clarification is more user-friendly but costlier. |

### Key Findings

1. **Most identification methods focus only on outward boundaries**—the identification of parametric boundaries remains an open question.
2. **LLMs suffer from severe overconfidence**—maintaining high confidence on unfamiliar topics while producing erroneous outputs.
3. **Abstention strategies do not distinguish between MSU and MAU**—causing answerable questions to be rejected, leading to poor user experience.
4. **Knowledge boundaries shift dynamically with training data cutoffs**—e.g., LLaMA-2 was trained on data up to 2022 but tends to heavily rely on 2019 data.

## Highlights & Insights

- **Formalized knowledge classification framework** is the core contribution—transforming the conceptual Known-Unknown Quadrant into actionable mathematical definitions.
- **The three-tier nested structure** ($Outward \subset Parametric \subset Universal$) provides a clear framework for locating problems.
- **The distinction between PAK and PSK** directly guides prompt engineering—many "unknown" problems are merely queried using suboptimal prompt formulations.
- **Summary Box designs** facilitate a quick grasp of the core concepts in each section.

## Limitations & Future Work

- Limited discussion on MAU (knowledge unknown to humanity), stemming from the overall scarcity of research in this area.
- The formalization of knowledge boundaries relies on the threshold $\epsilon$, for which optimal selection still lacks guidance.
- The survey's scope is current up to late 2024; the rapidly evolving LLM field may have advanced beyond.
- The knowledge boundary under multimodal contexts is not thoroughly explored.

## Related Work & Insights

- **vs. Know-Unknown Quadrant (Yin et al. 2023)**: A purely conceptual framework—this work provides a formalized definition.
- **vs. Hallucination Surveys (Ji et al. 2023)**: Lacked analysis from a knowledge boundary perspective—this work establishes structural mappings.
- **vs. Semantic Entropy (Kuhn et al. 2023)**: A specific methodology—this work provides a comprehensive categorization framework.
- **Insight**: Identifying knowledge boundaries is a crucial first step for reliable LLM deployment—knowing "what the model does not know" is more important than merely teaching it "to know more."

## Rating

- Novelty: ⭐⭐⭐⭐ The formalized defining framework presents original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematically comprehensive, spanning from motivation to identification and mitigation.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with helpful Summary Boxes.
- Value: ⭐⭐⭐⭐⭐ Offers direct guiding significance for improving LLM reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)

</div>

<!-- RELATED:END -->
