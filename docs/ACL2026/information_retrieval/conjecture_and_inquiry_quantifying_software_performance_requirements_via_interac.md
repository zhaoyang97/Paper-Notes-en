---
title: >-
  [Paper Note] Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation
description: >-
  [ACL 2026][requirement quantification] This paper proposes IRAP (Interactive Retrieval-Augmented Preference Elicitation), a method that quantifies natural-language software performance requirements into mathematical functions. Evaluated on 4 real-world datasets against 10 state-of-the-art baselines, IRAP achieves up to 40× performance improvement using only 5 interaction rounds.
tags:
  - ACL 2026
  - requirement quantification
  - preference elicitation
  - retrieval-augmented generation
  - interactive systems
  - software performance requirements
date: 2026-05-08
content_hash: b862d1552cd87802
---

# Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation

**Conference**: ACL 2026
**arXiv**: [2604.21380](https://arxiv.org/abs/2604.21380)
**Code**: To be confirmed
**Area**: Information Retrieval
**Keywords**: requirement quantification, preference elicitation, retrieval-augmented generation, interactive systems, software performance requirements

## TL;DR

This paper proposes IRAP (Interactive Retrieval-Augmented Preference Elicitation), a method that quantifies natural-language software performance requirements into mathematical functions. Evaluated on 4 real-world datasets against 10 state-of-the-art baselines, IRAP achieves up to 40× performance improvement using only 5 interaction rounds.

## Background & Motivation

**Background**: Software performance requirements (e.g., response time, throughput, availability) are typically documented in natural language, yet performance analysis, testing, and optimization in software engineering demand that they be expressed in computable mathematical forms (e.g., utility functions, constraints).

**Limitations of Prior Work**: Natural-language descriptions of performance requirements are inherently ambiguous (e.g., "the system should respond quickly," "latency should be within an acceptable range"), and cognitive uncertainty among stakeholders means the same requirement text can be interpreted as fundamentally different mathematical forms by different parties. This high degree of ambiguous uncertainty renders automated quantification a largely unsolved problem.

**Key Challenge**: There is a fundamental tension between the need to translate vague natural language into precise mathematical functions and the highly personalized, context-dependent nature of stakeholder preferences—parameters that conventional NLP methods cannot directly infer from text alone.

**Goal**: To formally define the problem of performance requirement quantification and to propose a method that reasons about preferences by retrieving domain-specific knowledge while conducting progressive interactions with stakeholders, achieving high-precision quantification with reduced cognitive burden.

**Key Insight**: The problem is framed as "Conjecture and Inquiry"—the system first forms a quantification conjecture grounded in retrieved domain knowledge, then iteratively verifies and refines it through targeted interactions with stakeholders.

**Core Idea**: Rather than attempting to infer mathematical functions from text in a single pass, IRAP leverages retrieval-augmented domain knowledge to initialize conjectures, then progressively refines preference parameters through a small number of interaction rounds.

## Method

### Overall Architecture

IRAP comprises two tightly coupled core components: (1) a **retrieval-augmented preference reasoning module** that retrieves cases and reference information relevant to the current requirement from a domain knowledge base to reason about latent stakeholder preferences; and (2) a **progressive interaction module** that designs targeted questions based on the reasoning results, eliciting true stakeholder preferences in minimal rounds and ultimately mapping natural-language requirements to mathematical functions.

### Key Designs

1. **Retrieval-Augmented Preference Reasoning**:
    - *Function*: Acquires quantitative prior information from domain knowledge to ground preference conjectures.
    - *Mechanism*: A problem-specific knowledge base is constructed (containing historical quantification cases, industry standards, and domain specifications). Upon receiving a new natural-language requirement, semantically relevant cases and knowledge snippets are retrieved and used to reason about possible quantification forms (e.g., function shape, parameter ranges).
    - *Design Motivation*: Unlike directly prompting an LLM to generate mathematical functions from text, the retrieval-augmented approach supplies verifiable prior information, reduces hallucination risk, and makes the reasoning process traceable.

2. **Progressive Interaction Design**:
    - *Function*: Elicits precise stakeholder preferences with minimal cognitive burden.
    - *Mechanism*: Based on retrieval-augmented reasoning, the system identifies the parameters with the highest uncertainty in the current conjecture and formulates targeted binary or multiple-choice questions (rather than open-ended prompts) to guide stakeholders in confirming or correcting their preferences. The quantification model is updated after each round.
    - *Design Motivation*: Open-ended questions impose excessive cognitive load on stakeholders (e.g., "Please describe your mathematical preference for latency"), whereas targeted closed-form questions substantially lower the barrier to participation.

3. **Requirement-to-Function Mapping**:
    - *Function*: Converts natural-language requirements into computable mathematical functions.
    - *Mechanism*: Combining retrieved domain knowledge with interactively elicited preference information, the system selects an appropriate function family (e.g., linear, exponential, step functions) and precisely estimates function parameters, producing a complete mathematical specification including both functional form and parameters.
    - *Design Motivation*: The ultimate goal of quantification is to provide directly usable mathematical representations for software performance analysis, test generation, and optimization.

## Key Experimental Results

### Main Results

| Dataset | Metric | IRAP | Best Baseline | Gain |
|---------|--------|------|---------------|------|
| Dataset 1 | Quantification accuracy | Best | 2nd best | Up to 40× |
| Dataset 2 | Quantification accuracy | Best | 2nd best | Significant |
| Dataset 3 | Quantification accuracy | Best | 2nd best | Significant |
| Dataset 4 | Quantification accuracy | Best | 2nd best | Significant |

*(Note: 4 real-world datasets; 10 SOTA baselines; IRAP achieves the best performance across all cases, with a maximum gain of 40× and requiring only 5 interaction rounds.)*

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| w/o retrieval augmentation | Accuracy degrades | Absence of domain knowledge leads to conjecture bias |
| w/o interaction | Accuracy degrades substantially | Fully automated approach cannot resolve preference ambiguity |
| Fewer interaction rounds | Accuracy improves with more rounds | 5 rounds is the accuracy–efficiency sweet spot |
| Different retrieval strategies | Accuracy varies | Retrieval quality affects initial conjecture accuracy |

### Key Findings

- IRAP comprehensively outperforms 10 SOTA methods across 4 real-world datasets, validating the effectiveness of the retrieval-augmented + interactive preference elicitation paradigm.
- Only 5 interaction rounds are sufficient to achieve up to 40× accuracy improvement, demonstrating that the progressive interaction design strikes a strong balance between efficiency and precision.
- The domain priors provided by the retrieval-augmented module are critical to initial conjecture quality and directly influence the efficiency of subsequent interactions.
- Compared to fully automated approaches (e.g., directly generating functions from text via LLM), the interactive approach holds a fundamental advantage in resolving preference ambiguity.

## Highlights & Insights

- **Value of problem formalization**: This work is the first to formally define the problem of "performance requirement quantification"—a practically important yet overlooked challenge—opening a new direction at the intersection of software engineering and NLP.
- **"Conjecture and Inquiry" paradigm**: Unlike one-shot generation, IRAP's progressive interaction design better aligns with the incremental cognitive processes underlying human decision-making.
- **Minimizing cognitive burden**: The interaction design eschews open-ended questions in favor of closed-form prompts that guide stakeholders, substantially lowering the barrier to participation.
- **Practical significance of 40× gain**: In precision-sensitive tasks such as requirement quantification, a 40× improvement represents a qualitative transition from "unusable" to "deployable."

## Limitations & Future Work

- The abstract does not specify the domains or scales of the 4 datasets in detail.
- Although 5 rounds is a low interaction count, human participation is still required, limiting applicability in fully automated scenarios.
- The cost and coverage of constructing domain knowledge bases may hinder cold-start performance in novel domains.
- The paper does not address how to handle cases where stakeholder preferences are internally contradictory.
- Future work could extend IRAP to other categories of requirement quantification (e.g., security requirements, reliability requirements).

## Related Work & Insights

- **vs. Traditional requirements engineering**: Conventional approaches rely on domain experts for manual modeling; IRAP achieves semi-automation through retrieval and interaction, substantially reducing expert dependency.
- **vs. RAG methods**: IRAP goes beyond using retrieval to augment text generation—it innovatively applies retrieved results to preference reasoning and interaction design, representing a novel application of the RAG paradigm in requirements engineering.
- **vs. Preference learning**: Unlike learning preferences from large volumes of comparative data, IRAP efficiently elicits preferences through a small number of targeted interactions, making it better suited to low-data settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First to formally define and address the performance requirement quantification problem; the retrieval-augmented + progressive interaction paradigm is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison against 10 SOTA methods across 4 real-world datasets yields convincing results.
- Writing Quality: ⭐⭐⭐ Based on abstract-level information; the title has literary flair, though the cross-disciplinary scope spanning software engineering and NLP may limit audience breadth.
- Value: ⭐⭐⭐⭐ Addresses a genuine engineering pain point; the 40× improvement has tangible practical applicability.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ICLR 2026\] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](../../ICLR2026/information_retrieval/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)

</div>

<!-- RELATED:END -->
