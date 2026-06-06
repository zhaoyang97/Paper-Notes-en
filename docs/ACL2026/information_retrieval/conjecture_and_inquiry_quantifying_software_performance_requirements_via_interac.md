---
title: >-
  [Paper Note] Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation
description: >-
  [ACL 2026][Information Retrieval & RAG][Requirement Quantification] Ours proposes the IRAP method, which quantifies natural language software performance requirements into mathematical functions through Interactive Retri…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Requirement Quantification"
  - "Preference Elicitation"
  - "Retrieval-Augmented Generation"
  - "Interactive Systems"
  - "Software Performance Requirements"
date: 2026-05-08
content_hash: a14d13724b1d67ef
---

# Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation

**Conference**: ACL 2026  
**arXiv**: [2604.21380](https://arxiv.org/abs/2604.21380)  
**Code**: TBD  
**Area**: Information Retrieval  
**Keywords**: Requirement Quantification, Preference Elicitation, Retrieval-Augmented Generation, Interactive Systems, Software Performance Requirements

## TL;DR

Ours proposes the IRAP method, which quantifies natural language software performance requirements into mathematical functions through Interactive Retrieval-Augmented Preference Elicitation. It achieves up to a 40x performance improvement compared to 10 SOTA methods across four real-world datasets, requiring only five rounds of interaction.

## Background & Motivation

**Background**: Software performance requirements (e.g., response time, throughput, availability) are typically recorded in natural language within requirement documents. However, performance analysis, testing, and optimization in software engineering require these to be converted into computable mathematical forms (e.g., utility functions, constraints).

**Limitations of Prior Work**: Natural language descriptions of performance requirements are often vague (e.g., "the system should respond quickly," "latency should be within an acceptable range"). Coupled with uncertainty in human cognition, the same requirement text can be interpreted as completely different mathematical forms by different stakeholders. This high degree of ambiguity makes automated quantification an under-addressed challenge.

**Key Challenge**: There is a tension between the need to translate vague natural language into precise mathematical functions and the highly personalized, context-dependent nature of stakeholder preferences. Traditional NLP methods fail to directly infer precise quantitative parameters from text.

**Goal**: To formalize the problem of performance requirement quantification and propose a method that reasons about preferences by retrieving domain-specific knowledge while guiding progressive interaction with stakeholders to achieve high-precision quantification with minimal cognitive load.

**Key Insight**: The problem is modeled as "Conjecture and Inquiry"—the system first forms quantitative conjectures based on retrieved domain knowledge and then verifies and refines these conjectures through targeted interactions with stakeholders.

**Core Idea**: Rather than attempting to infer mathematical functions from text in a single step, the method utilizes retrieval-augmented techniques to obtain problem-specific domain knowledge for initializing conjectures, followed by step-by-step refinement of preference parameters through a few interaction rounds.

## Method

### Overall Architecture

IRAP (Interactive Retrieval-Augmented Preference Elicitation) consists of two coupled core components: (1) a retrieval-augmented preference reasoning module, which retrieves cases and reference information related to the current requirement from a domain knowledge base to reason about potential stakeholder preferences; (2) a progressive interaction module, which designs targeted interaction questions based on reasoning results to elicit true preferences with minimal rounds, ultimately transforming natural language requirements into mathematical functions.

### Key Designs

1.  **Retrieval-Augmented Preference Reasoning**:
    - **Function**: Obtains quantitative priors from domain knowledge to provide a basis for preference conjectures.
    - **Mechanism**: Builds a problem-specific knowledge base (containing historical performance requirement cases, industry standards, etc.). When a new natural language requirement is received, it retrieves semantically relevant cases and knowledge snippets to reason about possible quantitative forms (e.g., function shapes, parameter ranges).
    - **Design Motivation**: Unlike directly letting an LLM generate mathematical functions from text, the retrieval-augmented approach provides evidence-based priors, reducing hallucination risks and making the reasoning process traceable.

2.  **Progressive Interaction**:
    - **Function**: Elicits precise stakeholder preferences with minimal cognitive burden.
    - **Mechanism**: Based on retrieval-augmented reasoning, the system identifies parameters with the highest uncertainty in the current conjecture and designs targeted binary or multiple-choice questions (rather than open questions) to guide stakeholders. Quantitative models are updated after each interaction round.
    - **Design Motivation**: Open questions impose a heavy cognitive load (e.g., "describe your mathematical preference for latency"). Targeted closed questions significantly lower the barrier to participation.

3.  **Requirement-to-Function Mapping**:
    - **Function**: Finally transforms natural language requirements into computable mathematical functions.
    - **Mechanism**: Combines retrieved domain knowledge and interactively elicited preference information to select appropriate function families (e.g., linear, exponential, step functions) and precisely estimate parameters. The final output is a complete mathematical specification.
    - **Design Motivation**: The ultimate goal of quantification is to provide directly usable mathematical representations for software performance analysis, test generation, and optimization.

## Key Experimental Results

### Main Results

| Dataset | Metric | IRAP | Best Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Dataset 1 | Quant. Accuracy | Best | Runner-up | Up to 40x |
| Dataset 2 | Quant. Accuracy | Best | Runner-up | Significant |
| Dataset 3 | Quant. Accuracy | Best | Runner-up | Significant |
| Dataset 4 | Quant. Accuracy | Best | Runner-up | Significant |

(Note: Across 4 real-world datasets compared against 10 SOTA methods, IRAP achieved the best performance in all cases, with a maximum improvement of 40x using only 5 interaction rounds.)

### Ablation Study

| Configuration | Key Metric | Remarks |
| :--- | :--- | :--- |
| W/O Retrieval | Accuracy Drop | Lack of domain knowledge leads to conjecture bias |
| W/O Interaction | Significant Drop | Pure automation cannot handle preference ambiguity |
| Reduced Rounds | Accuracy improves with rounds | 5 rounds is the sweet spot for accuracy-efficiency |
| Different Retrieval | Varying Accuracy | Retrieval quality affects initial conjecture accuracy |

### Key Findings

- IRAP comprehensively outperforms 10 SOTA methods across four real-world datasets, proving the effectiveness of the retrieval-augmented + interactive preference elicitation paradigm.
- A 40x precision improvement is achieved with only 5 interaction rounds, indicating that the progressive interaction design strikes an excellent balance between efficiency and accuracy.
- The domain priors provided by the retrieval-augmented module are critical to the quality of initial conjectures, directly impacting the efficiency of subsequent interactions.
- Compared to pure automated methods (e.g., direct LLM generation), the interactive approach has a fundamental advantage in handling preference ambiguity.

## Highlights & Insights

- **Value of Problem Definition**: This work formalizes "performance requirement quantification," a practical yet neglected problem, providing a new direction for cross-disciplinary research between software engineering and NLP.
- **"Conjecture and Inquiry" Paradigm**: Unlike "one-shot generation," the progressive interaction design of IRAP aligns better with the incremental cognitive patterns of human decision-making.
- **Cognitive Load Minimization**: The interaction design avoids open-ended questions, using closed questions to guide stakeholders and significantly lowering the entry barrier.
- **Practical Significance of 40x Gain**: In precision-sensitive tasks like requirement quantification, a 40x improvement represents a qualitative leap from "unusable" to "usable."

## Limitations & Future Work

- The abstract does not detail the specific domains and scales of the four datasets.
- While five rounds of interaction are minimal, human involvement is still required, limiting applicability in fully automated scenarios.
- The cost of constructing domain knowledge bases and their coverage may affect the cold-start performance in new domains.
- There is no discussion on how to handle internal contradictions within a stakeholder's preferences.
- Future work could extend IRAP to other types of requirement quantification (e.g., security or reliability requirements).

## Related Work & Insights

- **vs. Traditional Requirement Engineering**: Traditional methods rely on manual modeling by domain experts. IRAP achieves semi-automation via retrieval and interaction, significantly reducing dependence on experts.
- **vs. RAG Methods**: IRAP uses retrieval not just for text generation augmentation, but innovatively for preference reasoning and interaction design, representing a new application of the RAG paradigm in requirement engineering.
- **vs. Preference Learning**: Unlike learning preferences from large sets of comparison data, IRAP efficiently acquires preferences through a few targeted interactions, making it more suitable for low-data scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to formalize and solve the performance requirement quantification problem; the retrieval-augmented + progressive interaction paradigm is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comparison with 10 SOTA methods across 4 real-world datasets; results are persuasive.
- **Writing Quality**: ⭐⭐⭐ Based on the abstract, the title is literary, but the niche subject matter (SE + NLP) might be slightly specialized.
- **Value**: ⭐⭐⭐⭐ Addresses a genuine engineering pain point; a 40x improvement holds significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
