---
title: >-
  [Paper Note] Contradiction Detection in RAG-Based Chatbots
description: >-
  [ACL 2025][Information Retrieval & RAG][Contradiction Detection] This paper addresses the contradiction between retrieved documents and generated responses in RAG dialogue systems by proposing a multi-granularity contradiction detection framework that identifies explicit, implicit, and partial contradictions while providing interpretable contradiction localization.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Contradiction Detection"
  - "Retrieval-Augmented Generation"
  - "Dialogue Systems"
  - "Factual Consistency"
  - "RAG Hallucination"
date: 2026-05-08
content_hash: d89156a0a4aca8ca
---

# Contradiction Detection in RAG-Based Chatbots

**Conference**: ACL 2025  
**Area**: Information Retrieval  
**Keywords**: Contradiction Detection, Retrieval-Augmented Generation, Dialogue Systems, Factual Consistency, RAG Hallucination

## TL;DR
This paper addresses the contradiction between retrieved documents and generated responses in RAG dialogue systems by proposing a multi-granularity contradiction detection framework that identifies explicit, implicit, and partial contradictions while providing interpretable contradiction localization.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become the mainstream paradigm for improving the factual accuracy of LLMs by retrieving external documents to provide a factual foundation for generation. RAG systems are widely applied in question-answering, dialogue, and knowledge services, performing particularly well in scenarios requiring real-time knowledge updates.

**Limitations of Prior Work**: RAG systems face an underestimated but serious issue—the generated responses may contradict the retrieved documents. This contradiction stems from multiple factors: (1) internal knowledge conflicts within LLMs, where the model selectively ignores retrieved content; (2) conflicts among multiple retrieved documents, leading to inconsistencies during integration; (3) multi-turn dialogues where successive responses align with different documents, leading to internal contradictions. Existing hallucination detection methods mainly focus on "out-of-thin-air" errors and struggle with detailed contradiction categorization and detection.

**Key Challenge**: Simple NLI (Natural Language Inference) methods can judge whether two sentences contradict each other. However, in RAG scenarios, contradiction relationships are more complex—they can be partial contradictions (overall correct but containing incorrect details), implicit contradictions (no literal contradiction but contradictory after inference), or cross-turn contradictions. Binary classification fails to meet these complex demands.

**Goal**: To build a multi-granularity contradiction detection framework that can (1) distinguish between different types of contradictions; (2) locate the specific positions of contradictions; and (3) continuously monitor consistency in multi-turn dialogues.

**Key Insight**: The authors decompose contradiction detection into three subtasks: claim extraction (extracting factual claims from responses), evidence alignment (aligning claims with retrieved documents), and contradiction classification (determining whether aligned pairs are contradictory and their specific contradiction types), establishing an interpretable detection pipeline.

**Core Idea**: By decomposing responses into atomic claims and validating them one by one against the retrieved evidence, fine-grained contradiction detection and localization are achieved, while a cross-turn claim graph is maintained to capture dialogue-level contradictions.

## Method

### Overall Architecture
The inputs are the collection of retrieved documents and the generated response from the RAG system, and the outputs are the contradiction detection results (presence of contradiction, contradiction type, and position). The workflow involves: first extracting atomic factual claims from the response, then aligning each claim with the relevant passages in the retrieved documents, followed by contradiction classification on each claim-evidence pair, and finally aggregating these to obtain the overall contradiction judgment.

### Key Designs

1. **Atomic Claim Extractor**:

    - **Function**: Decomposing generated responses into indivisible factual claims
    - **Mechanism**: Using an instruction-tuned LLM to decompose complex sentences into simple "subject-verb-object" triplet claims. For instance, decomposing "Apple released the first iPhone in 2007, priced at $499 at the time" into two claims: (1) Apple released the first iPhone in 2007; (2) The first iPhone was priced at $499. This decomposition is guided by few-shot prompting.
    - **Design Motivation**: Coarse-grained sentence-level detection struggles to make accurate judgments when a sentence contains both correct and incorrect parts. Atomic decomposition enables precise contradiction localization.

2. **Evidence Alignment and Contradiction Classification Module**:

    - **Function**: Pairing each claim with the most relevant document evidence and determining the contradiction relationship
    - **Mechanism**: Utilizing a bi-encoder to calculate the semantic similarity between claims and document passages to select the top-k relevant evidence. A cross-encoder is then used to classify each claim-evidence pair into four categories: entailment, contradiction, partial contradiction, and neutral. The classifier is jointly trained on standard NLI data and self-constructed RAG contradiction data.
    - **Design Motivation**: Compared to traditional three-way classification, the four-way classification adds a "partial contradiction" category, which is highly common in RAG scenarios—the model's response might be generally correct but contain errors in specific numbers or dates.

3. **Cross-turn Consistency Graph**:

    - **Function**: Tracking consistency among claims across multi-turn dialogues
    - **Mechanism**: Maintaining a directed graph where nodes correspond to historical claims and edges denote entailment or contradiction relationships between them. Claims generated in each new response turn are consistency-checked against existing nodes in the graph. If a new claim contradicts historical claims, a cross-turn contradiction warning is triggered. The graph is updated incrementally, checking only the relationships between new nodes and relevant historical nodes in each turn.
    - **Design Motivation**: In RAG dialogue systems, because different documents are retrieved in each turn, successive responses may be based on conflicting factual bases, leading to inconsistencies. The graph structure systematically tracks this progressive contradiction.

### Loss & Training
The contradiction classifier is trained using a cross-entropy loss with label smoothing. The training data includes standard NLI datasets (SNLI, MultiNLI), a self-constructed RAG contradiction dataset (automatically generated through perturbations of correct responses), and a small set of human-annotated RAG dialogue contradiction samples. The automatic data generation strategy includes entity substitution, numeral perturbation, negation injection, and info omission.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | NLI-only | GPT-4 Judgment | Simple Rules |
|--------|------|---------|---------|----------|---------|
| RAG-Contradict (Ours) | F1 | 82.6 | 71.3 | 76.8 | 54.2 |
| RAG-Contradict (Ours) | Contradiction Localization Acc | 78.4 | - | 68.5 | - |
| BEGIN Benchmark | F1 | 79.1 | 73.5 | 75.2 | 51.8 |
| DialFact | F1 | 80.3 | 74.1 | 77.4 | 56.3 |
| Multi-turn Dialogue Set | Cross-turn Contradiction F1 | 74.8 | 62.1 | 70.3 | 43.7 |

### Ablation Study

| Configuration | F1 | Contradiction Localization Acc | Description |
|------|-----|-------------|------|
| Full model | 82.6 | 78.4 | Full model |
| w/o Atomic Claim Decomposition | 74.8 | - | Sentence-level detection, drops 7.8 F1 |
| w/o Partial Contradiction Category | 79.3 | 73.1 | Degraded to three-way classification, drops 3.3 F1 |
| w/o RAG-specific Training Data | 76.5 | 71.6 | Using only NLI data, drops 6.1 F1 |
| w/o Cross-turn Graph | 81.2 | 77.8 | Single-turn detection with simple historical comparison |

### Key Findings
- Atomic claim decomposition is the most critical design, contributing 7.8 F1 points, which proves that fine-grained analysis is vital for contradiction detection.
- Introducing the partial contradiction category brings a 3.3 F1 improvement, demonstrating that partial contradiction is indeed an important error pattern in RAG scenarios.
- RAG-specific training data is more critical than general-purpose NLI data, emphasizing the importance of domain adaptation.

## Highlights & Insights
- Performing contradiction detection at the atomic claim level is a key innovation. This not only improves detection accuracy but also provides interpretable error localization, which is highly valuable for building user trust.
- The design of the cross-turn consistency graph effectively addresses the unique challenge of RAG dialogue systems: progressive contradiction caused by retrieving different documents in different turns.
- Introducing the partial contradiction category fills a gap in existing work; in real-world scenarios, "generally correct but incorrect details" is more common and harder to detect than complete contradictions.

## Limitations & Future Work
- Atomic claim extraction relies on LLMs, and its potential errors can cascade to downstream detection.
- The inference latency of the framework is relatively high (requiring claim decomposition and sequential verification), which may become a bottleneck in real-time dialogue scenarios.
- The detection capability is still limited for implicit contradictions that require multi-step reasoning.
- Future work could study how to automatically correct contradictions once detected, forming a closed loop of detection and correction.

## Related Work & Insights
- **vs FActScore**: FActScore evaluates factual precision but only determines whether supporting evidence exists; this work further categorizes contradiction types and provides localization.
- **vs SelfCheckGPT**: SelfCheckGPT detects hallucinations based on the model's own sampling consistency without using retrieved evidence; this work directly compares retrieved documents with the response.
- **vs Q²**: Q² validates consistency through question generation and answer comparison; this work directly performs claim-level alignment, which is more efficient and accurate.
- **vs RARR**: RARR achieves factual enhancement through post-hoc editing; this work focuses on contradiction detection rather than automatic correction. The two can be combined to form a detection-correction loop.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-granularity contradiction detection framework and cross-turn graph design are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation, detailed ablation, and validation on real-world RAG systems.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and rigorous taxonomy.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for ensuring the reliability of RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Don't Lag, RAG: Training-Free Adversarial Detection Using RAG](../../ICML2025/information_retrieval/dont_lag_rag_training-free_adversarial_detection_using_rag.md)
- [\[ACL 2025\] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)
- [\[ACL 2025\] The Distracting Effect: Understanding Irrelevant Passages in RAG](the_distracting_effect_understanding_irrelevant_passages_in_rag.md)
- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)

</div>

<!-- RELATED:END -->
