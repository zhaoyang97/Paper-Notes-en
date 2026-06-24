---
title: >-
  [Paper Note] Writing Like the Best: Exemplar-Based Expository Text Generation
description: >-
  [ACL 2025][Text Generation][Exemplar-Driven Generation] Defines a new task, "Exemplar-Based Expository Text Generation"—generating an expository text about a target topic given an exemplar text about a source topic. It proposes the Recurrent Plan-then-Adapt (RePA) framework, which recurrently processes paragraph-level imitation planning, retrieval-augmented adaptive generation, and a dual-memory mechanism. RePA significantly outperforms GPT-4 and o1 baselines across three dat…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Exemplar-Driven Generation"
  - "Expository Text Generation"
  - "Adaptive Imitation"
  - "Long Text Generation"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 1c8e31a3128e3302
---

# Writing Like the Best: Exemplar-Based Expository Text Generation

**Conference**: ACL 2025  
**arXiv**: [2505.18859](https://arxiv.org/abs/2505.18859)  
**Code**: [https://github.com/liuyuxiang512/RePA.git](https://github.com/liuyuxiang512/RePA.git)  
**Area**: Text Generation  
**Keywords**: Exemplar-Driven Generation, Expository Text Generation, Adaptive Imitation, Long Text Generation, Retrieval-Augmented Generation  

## TL;DR

Defines a new task, "Exemplar-Based Expository Text Generation"—generating an expository text about a target topic given an exemplar text about a source topic. It proposes the Recurrent Plan-then-Adapt (RePA) framework, which recurrently processes paragraph-level imitation planning, retrieval-augmented adaptive generation, and a dual-memory mechanism. RePA significantly outperforms GPT-4 and o1 baselines across three datasets: Wikipedia, RoleEE, and USNews.

## Background & Motivation

**Background**: Large-scale expository text generation (e.g., university profiles, product descriptions, biographies) requires generating factually accurate content for different topics while maintaining structural consistency. Existing methods either require extensive domain corpora or tend to produce unconstrained, open-ended generations.

**Limitations of Prior Work**: (a) Directly prompting LLMs lacks structural consistency; (b) simply replacing topic names (Default) leads to factual errors; (c) long text generation is prone to inconsistencies and redundancies.

**Key Challenge**: Balancing the "correspondence" (structural similarity) and "variability" (different factual specifics) between source and target topics. For example, Wikipedia articles for two administrative districts share a similar structure but have entirely different population figures and historical events.

**Goal**: How to mimic the structure from a single exemplar while adaptively filling in the factual content of the target topic.

**Key Insight**: Borrowing from the human writing and learning process—studying excellent model essays to learn the writing structure while filling in one's own content. The concept of "Adaptive Imitation" is introduced.

**Core Idea**: Decompose the exemplar text into a question-based outline (Plan), answer each question using retrieval with confidence calibration (Adapt), and ensure long-text consistency via a dual-memory mechanism.

## Method

### Overall Architecture
The inputs are the source topic exemplar text $\mathbf{X}$, the target topic $\mathbf{t_y}$, and external knowledge $\mathbf{K}$. RePA segments the input text into paragraphs and processes each paragraph recursively: the Plan phase extracts outline questions, and the Adapt phase answers these questions to generate the output paragraph. Analogous to an LSTM-like recurrent structure, it utilizes short-term memory (to handle input anaphora) and long-term memory (to avoid output redundancy).

### Key Designs

1. **Plan Module (Imitation Planning)**:

    - **Clarify**: Resolves coreference ambiguity after segmentation using short-term memory $h_t$ (key information from the most recent input paragraphs), such as replacing pronouns like "it" with specific entity names.
    - **Outline**: Converts the clarified paragraph into an outline $q_t$ formatted as questions, and then transfers the questions from the source topic to the target topic via simple keyword replacement (e.g., changing "What is the population of Belebeyevsky District?" to "What is the population of Davlekanovsky District?").
    - Design Motivation: A question-formatted outline is both concise and highly transferable—maintaining structural consistency by simply replacing the topic name.

2. **Adapt Module (Adaptive Generation)**:

    - **Calibrated-QA**: Answers outline questions using a retrieval-augmented approach. A key innovation is the introduction of **confidence calibration**—prompting the LLM to attach confidence scores to its answers, whereby low-confidence answers are rejected (marked as "NA"). This resolves scenarios where source-specific questions have no corresponding answers for the target topic (e.g., the source topic contains a "Chuvash name" but the target topic does not).
    - **Write**: Generates the output paragraph based on the answered facts, while using long-term memory $c_t$ (a summary of all historical outputs) to eliminate redundancies.
    - Design Motivation: "An imperfect outline is acceptable as long as it is handled correctly"—fault tolerance is more practical than aiming for perfect planning.

3. **Dual-Memory Mechanism**:

    - **Short-term Memory** $h_t$: Stores information from the most recent input paragraphs to resolve coreferences in the Clarify phase.
    - **Long-term Memory** $c_t$: Stores the summary of all historical outputs to avoid repetition in the Write phase.
    - Design Motivation: Similar to LSTM's gated memory but operating in the text space, allowing the model to handle texts of arbitrary length.

### Loss & Training
- **Training-Free**: A pure prompt engineering approach where all components are implemented by prompting GPT-4 or LLaMA-3.
- Uses Wikipedia2Vec for topic pairing to ensure that the similarity between the source and target topics is $>0.95$.

## Key Experimental Results

### Main Results (GPT-4 on Wikipedia Dataset)

| Method | ROUGE-L | BERTScore | NLI-E(↑) | Halluc(↓) | Imitativeness(↑) | Adaptiveness(↑) |
|------|---------|----------|---------|----------|----------|----------|
| LLM (GPT-4) | 0.640 | 0.653 | 0.378 | 26.96 | 4.52 | 2.44 |
| o1+Retr | 0.871 | 0.867 | 0.781 | 9.02 | 4.32 | 3.02 |
| SR+Retr | 0.865 | 0.862 | 0.726 | 7.73 | 4.22 | 3.04 |
| **RePA** | **0.889** | **0.893** | **0.774** | **5.69** | 4.16 | **3.90** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Clarify | Decline in imitativeness and coherence | Anaphora ambiguity affects outline quality |
| w/o Calibrated-QA | Decline in factuality, increased hallucinations | Inability to reject unanswerable questions |
| w/o Short-term Memory | Decline in imitativeness | Loss of context during paragraph-by-paragraph processing |
| w/o Long-term Memory | Increased output redundancy | Inability to detect generated content |

### Key Findings
- RePA heavily outperforms all baselines in Adaptiveness—GPT-4's 2.44 vs. RePA's 3.90.
- Retrieval augmentation benefits all methods, but RePA's confidence calibration is more effective than simple retrieval.
- Performance remains stable when migrating from GPT-4 to LLaMA-3, demonstrating the framework's generalizability.
- Default (simply replacing topic names) yields the highest Imitativeness (5.00) but the lowest Adaptiveness (1.08), illustrating the trade-off between the two objectives.
- On the domain-specific USNews dataset, RePA's advantages are most prominent due to the greater cross-topic variability in this domain.

## Highlights & Insights

- **The concept of "Adaptive Imitation"** precisely captures the core of the task—neither blind copying nor complete rewriting, but performing factual adaptation while maintaining structural imitation. This approach is transferable to template-driven content production systems.
- **Using questions as outlines** is a clever design—questions are naturally transferable (requiring only keyword replacement) and provide clear query intents for retrieval-augmented generation.
- **The confidence calibration mechanism** addresses the "imperfect outline" issue—it acknowledges that certain talking points of the source topic may not exist in the target topic, gracefully skipping them instead of leveraging forced generation.
- The LSTM-like text-space recurrent architecture is an intriguing design paradigm—replacing vector operations in neural networks with LLM prompting.

## Limitations & Future Work

- Being a pure prompt-based method, it requires multiple LLM calls per paragraph, leading to high inference costs.
- The text length is constrained by the experimental setup; the performance on ultra-long texts (thousands of words) has not been verified.
- Evaluating imitativeness and adaptiveness relies solely on LLM-as-Judge, which carries inherent subjectivity.
- The source and target topics must be highly similar (cosine $>0.95$), making it unsuitable for cross-domain transfer.
- Confidence calibration relies on the LLM's self-calibration capacity, which might not be sufficiently accurate.

## Related Work & Insights

- **vs. Direct LLM Prompting**: LLMs struggle to maintain both structural consistency and factual accuracy simultaneously; RePA resolves this by decoupling the Plan and Adapt phases.
- **vs. Self-Refine**: Iterative refinement lacks clear directional guidance and often oscillates between structures and facts; RePA's modular design provides clearer objectives.
- **vs. Traditional Plan-and-Generate**: Traditional methods assume plans are perfectly executable; RePA's Calibrated-QA fault-tolerant mechanism is more practical.
- This framework can be applied to batch content generation scenarios (e.g., e-commerce product descriptions, university brochures).

## Rating

- Novelty: ⭐⭐⭐⭐ Defines a practical, high-demand new task ("exemplar-based expository text generation"). The concept of "Adaptive Imitation" is novel, and the LSTM-like text recurrent architecture is an intriguing design paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features three datasets (Wikipedia/RoleEE/USNews) + nine baselines (including GPT-4/o1) + ablation studies + human evaluation + LLM evaluation, making it highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation (balancing cross-topic consistency and cross-topic variability) and intuitive diagrams, though the method description is slightly wordy due to the multiple components.
- Value: ⭐⭐⭐⭐ The framework design is general and transferable to batch content generation scenarios (e.g., standardized template writing like e-commerce product descriptions, university enrollment brochures).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)
- [\[ACL 2025\] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](../../ACL2026/nlp_generation/can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)

</div>

<!-- RELATED:END -->
