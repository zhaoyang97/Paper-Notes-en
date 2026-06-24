---
title: >-
  [Paper Note] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark
description: >-
  [ACL 2025][LLM Evaluation][Dialogue Synthesis] This paper proposes the CoPrUS framework, a consistency-preserving utterance synthesis method for dialogue benchmark construction. By explicitly maintaining consistency constraints across personas, knowledge, and dialogue history during dialogue data generation, it produces more realistic dialogue benchmark data than existing methods.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Dialogue Synthesis"
  - "Consistency Preservation"
  - "Benchmark Construction"
  - "Data Augmentation"
  - "Dialogue Benchmarking"
date: 2026-05-08
content_hash: 7f95a76a4816de63
---

# CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark

**Conference**: ACL 2025  
**Area**: LLM Evaluation  
**Keywords**: Dialogue Synthesis, Consistency Preservation, Benchmark Construction, Data Augmentation, Dialogue Benchmarking

## TL;DR
This paper proposes the CoPrUS framework, a consistency-preserving utterance synthesis method for dialogue benchmark construction. By explicitly maintaining consistency constraints across personas, knowledge, and dialogue history during dialogue data generation, it produces more realistic dialogue benchmark data than existing methods.

## Background & Motivation

**Background**: The evaluation and training of dialogue systems rely heavily on high-quality dialogue datasets. However, collecting real human-to-human or human-to-machine dialogues is costly, making synthetic dialogue data an important means to supplement or even replace real data. The emergence of LLMs has significantly lowered the barrier to data synthesis, but the quality of synthetic dialogues, especially consistency, remains a core challenge.

**Limitations of Prior Work**: Existing dialogue synthesis methods suffer from severe consistency issues, primarily manifested in three aspects: (1) Persona consistency—contradictions in the speaker's personality, background, and preferences across different turns (e.g., saying "I live in Beijing" in turn 3, and "the weather here in LA" in turn 7); (2) Knowledge consistency—contradictory factual knowledge cited in different turns of dialogues containing factual information; (3) Dialogue logical consistency—incoherence in topics, attitudes, and decisions across preceding and succeeding turns. These inconsistencies mislead evaluation when synthetic data is used for benchmarks, and teach models "inconsistent dialogue styles" when used for training.

**Key Challenge**: High-quality dialogue synthesis requires maintaining long-range consistency across multi-turn generation. However, the autoregressive generation nature of LLMs is fundamentally local—each generation turn mainly focuses on the recent context, making it difficult to track and maintain constraints established earlier in the dialogue.

**Goal**: To design a framework that explicitly tracks and maintains consistency constraints during dialogue synthesis, ensuring consistency across persona, knowledge, and logical dimensions, thereby generating more realistic dialogue benchmark data.

**Key Insight**: The authors formalize consistency maintenance as a constraint satisfaction problem—when generating each turn of a dialogue, it must not only be contextually coherent but also satisfy a set of explicit consistency constraints (persona, knowledge, and logical constraints).

**Core Idea**: Introduce a consistency constraint tracker and validator into the LLM-driven dialogue synthesis pipeline. By maintaining an explicit set of constraints and validating constraint satisfaction after each turn of generation, consistency-guaranteed dialogue synthesis is achieved.

## Method

### Overall Architecture
The synthesis pipeline of CoPrUS consists of: (1) Initialization phase—setting dialogue scenarios, participant personas, and knowledge bases to generate an initial set of constraints; (2) Iterative generation phase—generating dialogues turn-by-turn and checking consistency using a constraint validator after each turn; (3) Repair phase—performing targeted modification or regeneration for utterances that violate constraints. The final output consists of synthetic dialogues with verified consistency.

### Key Designs

1. **Explicit Constraint Manager**:

    - **Function**: Tracks and maintains all consistency constraints during the dialogue process.
    - **Mechanism**: Maintains a structured constraint base containing three types of constraints: (a) Persona constraints—attribute-value pairs extracted from persona descriptions (e.g., "Occupation = Teacher", "Age > 30"); (b) Knowledge constraints—factual triples extracted from the knowledge base (e.g., "Paris - Capital - France"); (c) Logical constraints—logical relationships inferred from the dialogue history (e.g., "User expresses dislike for spicy food in turn 3 $\rightarrow$ subsequent recommendations should not contain spicy dishes"). For each new generated dialogue turn, the constraint manager automatically extracts new constraints from the new utterance and appends them to the constraint base.
    - **Design Motivation**: Externalizes implicit consistency requirements to make them trackable and verifiable. The constraint base grows dynamically as the dialogue progresses, reflecting the natural process of information accumulation in a conversation.

2. **Constraint-Aware Generator**:

    - **Function**: Integrates relevant constraints into the generation process during each dialogue turn.
    - **Mechanism**: In addition to the dialogue history, the currently active constraint set is injected into the LLM prompt. Constraints are expressed in natural language (e.g., "Note: You are a teacher, you live in Beijing, and you previously mentioned that you do not like watching movies") as generation guidelines. A constraint relevance ranking is applied to inject only the constraints related to the current topic (rather than all of them) to avoid excessive prompt length. Relevance is calculated using the semantic similarity between the constraint content and the current dialogue topic.
    - **Design Motivation**: Explicitly telling the LLM which constraints to follow is more reliable than expecting the model to self-recognize and reason from dialogue history.

3. **Posterior Constraint Validation and Repair**:

    - **Function**: Detects and repairs consistency violations in the generated content.
    - **Mechanism**: After generating each turn of utterance, an NLI model is used to check the entailment/contradiction status between the new utterance and all relevant constraints in the constraint base. If a contradiction is detected, a targeted repair is triggered—the contradictory constraint and the violating content are fed together into the LLM, prompting it to modify the violating part while preserving the rest. A re-validation is performed after the repair, allowing up to 3 repair loops. If contradictions persist after 3 turns, the entire turn is regenerated.
    - **Design Motivation**: Constraint-aware generation cannot guarantee 100% consistency; posterior validation provides a safety net. Targeted repair is more efficient than complete regeneration and preserves more of the original content.

### Loss & Training
CoPrUS is an inference-time framework that requires no additional training. The NLI validator uses an existing DeBERTa-v3-large model fine-tuned on SNLI+MultiNLI. Constraint relevance ranking utilizes all-MiniLM-L6 to compute semantic similarity.

## Key Experimental Results

### Main Results

| Method | Persona Consistency↑ | Knowledge Consistency↑ | Logical Consistency↑ | Fluency↑ | Diversity↑ |
|---|---|---|---|---|---|
| CoPrUS | 92.4% | 89.7% | 86.3% | 4.32/5 | 0.78 |
| Direct LLM Gen | 76.1% | 72.3% | 68.5% | 4.41/5 | 0.81 |
| Self-Chat | 71.8% | 69.4% | 64.2% | 4.18/5 | 0.73 |
| SODA | 78.5% | 74.1% | 70.8% | 4.28/5 | 0.76 |
| Constraint Gen (w/o Validation) | 85.6% | 82.1% | 78.7% | 4.35/5 | 0.79 |

### Ablation Study

| Configuration | Persona Consistency | Knowledge Consistency | Logical Consistency | Description |
|---|---|---|---|---|
| Full CoPrUS | 92.4% | 89.7% | 86.3% | Complete system |
| w/o Posterior Validation | 85.6% | 82.1% | 78.7% | Validation & repair removed, drops by 6.8% |
| w/o Constraint Injection | 80.2% | 76.8% | 72.1% | Constraints not injected into prompt |
| w/o Logical Constraints | 91.8% | 88.9% | 77.4% | Only persona and knowledge constraints maintained |
| w/o Constraint Relevance Ranking | 89.1% | 86.3% | 83.5% | All constraints injected (inefficient) |

### Key Findings
- The posterior validation & repair mechanism contributes approximately 6-8 percentage points of consistency improvement, serving as a critical safety net for quality assurance.
- Logical consistency is the most difficult dimension to maintain (86.3%), as logical constraints must be inferred from the dialogue rather than directly extracted.
- Constraint relevance ranking significantly improves efficiency (reducing prompt length by approximately 40%) without compromising quality.
- The improvement in consistency does not significantly impair fluency and diversity, suggesting that the constraints do not excessively restrict generation freedom.

## Highlights & Insights
- Formalizing consistency maintenance as an explicit constraint satisfaction problem is a core innovation. It is far more reliable than expecting LLMs to "voluntarily remain consistent," and this methodology can be transferred to other generation tasks requiring long-range consistency.
- The design of the three constraint categories (persona, knowledge, and logic) systematically covers the main dimensions of dialogue consistency.
- The validation-repair loop is highly practical—rather than aiming for perfect single-pass generation, it allows for errors while incorporating detection and correction.

## Limitations & Future Work
- The constraint extraction of the constraint manager relies on NLI and information extraction models, whose own errors can propagate.
- As the number of dialogue turns increases, the constraint base grows linearly, making constraint management and validation efficiency for long dialogues a potential bottleneck.
- Current validation primarily target explicit consistency; the ability to handle situations that are "conceptually inconsistent but not literally contradictory" is limited.
- Future work can explore hierarchical organization and priority management of constraints.

## Related Work & Insights
- **vs SODA (Kim et al.)**: SODA uses social commonsense for conditional generation but does not explicitly track consistency; CoPrUS achieves explicit consistency guarantees through the constraint manager.
- **vs Faithful Dialogue Generation**: Faithful dialogue generation focuses on single-turn faithfulness, whereas this work addresses multi-turn, cross-turn consistency.
- **vs Self-Chat**: Self-Chat allows a model to play two roles in dialogue without consistency control; the constraint mechanism of CoPrUS substantially improves synthesis quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of formalizing consistency maintenance as a constraint satisfaction problem is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation, detailed ablation, and comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and intuitive, easy-to-understand constraint design.
- Value: ⭐⭐⭐⭐ Provides practical guidance for assuring consistency in dialogue data synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)
- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)
- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)
- [\[ICLR 2026\] DRBench: A Realistic Benchmark for Enterprise Deep Research](../../ICLR2026/llm_evaluation/drbench_a_realistic_benchmark_for_enterprise_deep_research.md)

</div>

<!-- RELATED:END -->
