---
title: >-
  [Paper Note] Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation
description: >-
  [ACL 2025][LLM (Other)][Instruction Synthesis] This paper proposes Self-Foveate, an approach inspired by the human visual foveation mechanism. Through a three-level foveation strategy ("micro-scatter-macro"), it systematically extracts multi-granularity information from unsupervised text to synthesize instruction data with higher diversity and difficulty for instruction tuning of LLMs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Instruction Synthesis"
  - "Unsupervised Text"
  - "Multi-Granularity Information Extraction"
  - "Data Diversity"
  - "Data Difficulty"
date: 2026-05-08
content_hash: 21bd9418efd9e621
---

# Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation

**Conference**: ACL 2025  
**arXiv**: [2507.23440](https://arxiv.org/abs/2507.23440)  
**Code**: [Yes](https://github.com/Mubuky/Self-Foveate)  
**Area**: Others  
**Keywords**: Instruction Synthesis, Unsupervised Text, Multi-Granularity Information Extraction, Data Diversity, Data Difficulty

## TL;DR

This paper proposes Self-Foveate, an approach inspired by the human visual foveation mechanism. Through a three-level foveation strategy ("micro-scatter-macro"), it systematically extracts multi-granularity information from unsupervised text to synthesize instruction data with higher diversity and difficulty for instruction tuning of LLMs.

## Background & Motivation

Instruction tuning of LLMs requires a large amount of high-quality SFT data, but human annotation is prohibitively expensive. Synthesizing instruction data from unsupervised text is a promising paradigm, as vast amounts of unsupervised text corpora are readily available. However, existing automatic synthesis methods suffer from two key limitations:

**Lack of diversity**: Synthesized instructions often exhibit repetitive patterns in structure and topic. For instance, Self-QA generates instructions via a single-step process, yielding monotonic structures and limited types of instructions.

**Lack of difficulty**: The lack of a control mechanism for instruction complexity results in mostly simple questions, failing to deeply explore the complex relationships among entities in the text.

Key insight: Unsupervised text naturally contains rich multi-granularity information—ranging from entity attributes and implicit cross-regional relationships to overall rhetorical devices—but existing methods fail to systematically utilize this information.

The multi-level processing of the human visual system (detail capture in the fovea $\rightarrow$ cross-regional integration via saccades $\rightarrow$ global perception in the peripheral vision) provides an analogy for multi-granularity text understanding.

## Method

### Overall Architecture

Self-Foveate takes unsupervised text as input, extracts information at different granularities through a three-level foveation mechanism, and cooperates with three synthesis paradigms to generate instruction data:

$$\mathcal{D}_{\text{gen}} = \mathcal{F}(\mathcal{D}) = \bigcup_{d_i \in \mathcal{D}} \bigcup_{\mathcal{F}_j \in \mathcal{F}} \mathcal{F}_j(d_i)$$

### Key Designs

1. **Micro-foveate Level**: Extracting fine-grained foveation elements $\rightarrow$ Backward Synthesis

    - Function: Extract all entities and their attributes (referred to as "foveation elements") from the text, focusing on minor entities and fine-grained attributes.
    - Mechanism: First extract as many foveation elements as possible using an LLM, and then filter them using embedding cosine similarity to retain elements semantically relevant to the entire text.
    - Backward Synthesis: Treat each foveation element as a potential answer and guide the LLM to backward-generate the corresponding question instruction.
    - Design Motivation: When directly synthesizing instructions, LLMs tend to focus on salient surface-level content, ignoring minor entities and fine-grained attributes.

2. **Scatter-foveate Level**: Extracting foveation elements and grouping them into foveation groups $\rightarrow$ Direct Synthesis

    - Function: Broadly extract foveation elements from the text and randomly organize them into foveation groups.
    - Mechanism: Group scattered information points in the text together to force the LLM to discover implicit semantic relationships among them (such as causal chains, comparison relationships, temporal dependencies, etc.).
    - Direct Synthesis: Treat each element in the foveation group as an indispensable part of the instruction, guiding the LLM to synthesize instructions requiring cross-entity reasoning.
    - Design Motivation: Implicit relationships between entities are scattered throughout unsupervised texts, which are difficult to capture via single-step generation.

3. **Macro-foveate Level**: Identifying text segments with rhetorical devices $\rightarrow$ Transcription Synthesis

    - Function: Identify paragraphs using rhetorical devices (foveation segments) such as metaphors, exaggerations, rhetorical questions, and citations.
    - Mechanism: Convert these declarative segments, which contain deep communicative intent, into questions or imperative sentences.
    - Design Motivation: The deep meanings of rhetorical devices go beyond literal content, but LLMs easily overlook them in the absence of explicit guidance.

4. **Re-synthesis Module**: Processing unanswerable instructions from the initial synthesis

    - One-shot Reference Synthesis: Process one failed instruction at a time, randomly selecting a successful sample as reference.
    - Highly Creative Hyperparameter Configuration: Adjust parameters such as temperature and top-p to increase the variation of synthesized instructions.
    - Execute iteratively over multiple rounds to gradually replace unanswerable instructions.

### Loss & Training

The synthesized instruction data is utilized for standard SFT training of downstream models. GPT-4o mini or DeepSeek-V3 is used as the teacher LLM for instruction synthesis.

## Key Experimental Results

### Diversity Analysis

| Dataset | Method | SelfBLEU Diversity ↑ | Embedding Diversity ↑ |
|--------|------|-------------------|-------------|
| SQuAD | Self-QA | 0.593 | 0.838 |
| | Self-Foveate | **0.665** | **0.851** |
| HotpotQA | Self-QA | 0.463| 0.823 |
| | Self-Foveate | **0.607** | **0.835** |
| FilmWiki | Self-QA | 0.406 | 0.687 |
| | Self-Foveate | **0.563** | **0.706** |

### Difficulty Comparison (Head-to-Head Win Rate)

| Dataset | Opponent | Self-Foveate Win Rate |
|--------|------|-------------------|
| SQuAD | Self-QA | 70.64% |
| SQuAD | Wiki2023 | 80.83% |
| SQuAD | Bonito | 99.96% |
| HotpotQA | Self-QA | 89.52% |
| FilmWiki | Self-QA | 85.12% |

### Downstream Task Performance (Synthesized with Llama-3.1-8B + GPT-4o mini)

| Method | SQuAD Recall | HotpotQA Recall | FilmWiki Recall |
|------|-------------|-----------------|-----------------|
| None | 0.309 | 0.244 | 0.212 |
| Self-QA | 0.367 | 0.372 | 0.328 |
| Self-Foveate | **0.484** | **0.507** | **0.512** |

### Ablation Study

| Setting | Recall | LLM Acc. |
|------|--------|----------|
| Full Self-Foveate | **0.484** | **0.490** |
| w/o Micro-Foveate | 0.283 | 0.277 |
| w/o Scatter-Foveate | 0.274 | 0.260 |
| w/o Macro-Foveate | 0.468 | 0.479 |

### Key Findings

1. Self-Foveate comprehensively outperforms all baselines in diversity and difficulty metrics, even approaching or exceeding the diversity level of human-constructed test questions.
2. All three levels of foveation mechanisms are indispensable, with the removal of Scatter-foveate and Micro-foveate causing the most significant performance drops.
3. As the scale of synthesized instructions increases, the performance gap between Self-Foveate and the baseline methods continues to widen.
4. Consistent conclusions are drawn across different teacher LLMs (GPT-4o mini vs. DeepSeek-V3).

## Highlights & Insights

1. **Bio-inspired Design**: The analogy from human visual foveation mechanisms to multi-granularity text understanding is highly natural and effective.
2. **Systematic Design**: The combination of three-level foveation and three synthesis paradigms achieves comprehensive extraction of text information.
3. **Innovation in Backward Synthesis**: The approach of "generating questions based on pre-defined answers" effectively prevents LLMs from ignoring fine-grained information.
4. **High Practical Value**: The method requires no human annotation and can be directly applied to any unsupervised text corpus.

## Limitations & Future Work

1. Dependence on teacher LLMs (GPT-4o mini / DeepSeek-V3) for instruction synthesis implies that synthesis quality is bounded by the capabilities of the teacher.
2. The filtering of foveation elements relies on simple embedding similarity, which might omit information that is semantically relevant but expressed differently.
3. Evaluation has not been conducted on a wider range of downstream task types, such as multi-turn dialogue or code generation.
4. The hyperparameter configuration of the Re-synthesis module requires additional tuning.

## Related Work & Insights

- Self-QA: A representative work of single-step unsupervised text $\rightarrow$ instruction synthesis, upon which Self-Foveate introduces multi-granularity information extraction.
- Bonito: Synthesizes instructions using a pre-trained, specialized 7B model. It does not rely on teacher LLM APIs but suffers from limited flexibility.
- Wiki2023: Another QA-pair extraction method based on unsupervised text.
- Self-Instruct: Seed-example-based guided instruction synthesis, which is complementary to the unsupervised text paradigm of this paper.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The design of the multi-level foveation mechanism is novel and intuitive, and the combination of the three synthesis paradigms is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation across three datasets, three base models, and three dimensions (diversity, difficulty, and downstream tasks), along with a complete ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Richly illustrated and well-structured, with clear methodology descriptions and appropriate analogies.
- **Value**: ⭐⭐⭐⭐ Provides a systematic framework for exploiting multi-granularity information in unsupervised instruction synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)
- [\[ACL 2025\] MExGen: Multi-Level Explanations for Generative Language Models](mexgen_multi_level_explanations.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)
- [\[ACL 2025\] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)

</div>

<!-- RELATED:END -->
