---
title: >-
  [Paper Note] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation
description: >-
  [AAAI 2026][LLM/NLP][Competency Question Generation] This paper proposes the VSPO framework, which constructs a definition–axiom misalignment dataset and fine-tunes LLaMA-3.1-8B-Instruct to generate competency questions…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "Competency Question Generation"
  - "Semantic Pitfall Detection"
  - "Ontology Validation"
  - "LLM Fine-tuning"
  - "Misalignment Injection"
date: 2026-05-08
content_hash: 49a21b43fbe189ce
---

# VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation

**Conference**: AAAI 2026
**arXiv**: [2511.07991](https://arxiv.org/abs/2511.07991)
**Code**: [GitHub](https://github.com/Choi-Hyojun/VSPO)
**Area**: Ontology Engineering / Knowledge Representation
**Keywords**: Competency Question Generation, Semantic Pitfall Detection, Ontology Validation, LLM Fine-tuning, Misalignment Injection

## TL;DR

This paper proposes the VSPO framework, which constructs a definition–axiom misalignment dataset and fine-tunes LLaMA-3.1-8B-Instruct to generate competency questions (CQs) capable of validating semantic pitfalls in ontologies (e.g., misuse of `allValuesFrom`). The approach surpasses GPT-4.1 by 26% in precision and 28.2% in recall.

## Background & Motivation

**Background**: Competency Questions (CQs) are a central tool for ontology validation, used to define the knowledge scope of an ontology and verify whether it correctly encodes the intended knowledge. Prior work has explored using LLMs to automate CQ generation.

**Limitations of Prior Work**: Existing LLM-based approaches primarily evaluate generation quality based on similarity to existing CQ datasets, without genuinely assessing whether the generated CQs can detect semantic pitfalls in ontologies—such as confusion between `allValuesFrom` and `someValuesFrom`, or misuse of `union` instead of `intersection`.

**Key Challenge**: Semantic pitfalls (e.g., "P10. Missing disjointness" in the OOPS! catalog) cannot be detected by standard reasoners or rule-based methods and require manual expert review. Automated detection of such pitfalls remains an open challenge.

**Goal**: How can CQs that specifically target the detection of ontology semantic pitfalls be generated automatically?

**Key Insight**: The paper formalizes controlled "misalignments" between natural language definitions and axioms as a training signal, enabling an LLM to learn to identify inconsistencies and generate validation questions targeting them.

**Core Idea**: Semantic pitfalls are formalized as three types of misalignment (missing axiom, undefined axiom, misused axiom). An LLM generates definitions, misalignments are injected, template-based CQs serve as training labels, and LLaMA is fine-tuned to generate validating CQs.

## Method

### Overall Architecture

The VSPO pipeline consists of two main stages: **dataset construction** and **model fine-tuning**.

- Input: Each ontology term $T$ (class or property) and its associated axioms $A_T$
- Dataset construction: Template-based CQs are generated per axiom, then misalignment injection produces $(A_T, D_T, CQ_{sp})$ triples
- Model fine-tuning: LLaMA-3.1-8B-Instruct is fine-tuned on the constructed dataset to generate validating CQs conditioned on definition–axiom inconsistencies

### Key Designs

**Module 1: Template-Based CQ Generation**

- **Function**: For each axiom type (e.g., `inverseOf`, `EquivalentTo`), 3–7 templates are designed; GPT-4.1 generates $n=3$ CQs per axiom following the logical structure of each template.
- **Mechanism**: Templates provide a logical skeleton (e.g., "If A is connected to B via property X, does it imply B is connected to A via property Y?"), and GPT paraphrases rather than copies them, ensuring diversity.
- **Design Motivation**: Unconstrained LLM generation tends to produce repetitive outputs, leading to overfitting during fine-tuning; template guidance maintains logical accuracy while promoting expressive diversity.

**Module 2: Misalignment Injection**

- **Function**: Each ontology term is randomly assigned to one of four misalignment types, constructing semantic inconsistencies between definition $D_T$ and axiom $A_T$.
- **Mechanism**:
    - **Type 1 (Missing Axiom)**: A randomly selected axiom is removed from $A_T$, while $D_T$ is generated from the complete axiom set—simulating "an axiom that should exist is absent from the ontology."
    - **Type 2 (Undefined Axiom)**: $A_T$ remains intact, but a specific axiom is deliberately ignored when generating $D_T$—simulating "a required description is omitted from the definition."
    - **Type 3 (Misused Axiom)**: Logical constructors in an axiom are randomly substituted (e.g., `someValuesFrom` ↔ `allValuesFrom`, `intersection` ↔ `union`), while $D_T$ is generated from the original axiom—simulating "an incorrect logical operator is used in the axiom."
    - **Type 4 (Aligned)**: No modification; serves as a normal control case.
- **Design Motivation**: Collecting real-world semantic pitfall samples is extremely difficult; controlled misalignment injection allows the model to simultaneously learn "what is correct" and "what can go wrong."

**Module 3: LLM-Based Definition Generation**

- **Function**: GPT-4.1 generates natural language definitions for each ontology term from its axiom set.
- **Mechanism**: GPT is prompted to act as an ontology engineer, taking the term name and axiom set as input and producing a concise, accurate definition.
- **Design Motivation**: Existing ontologies provide very limited annotation information; automated definition generation is necessary to support large-scale dataset construction.

### Loss & Training

- **LoRA** (r=8, α=16, dropout=0.05) is applied for parameter-efficient fine-tuning of LLaMA-3.1-8B-Instruct.
- Training runs for 3 epochs (overfitting observed beyond 3 epochs), with effective batch size 4, learning rate $3 \times 10^{-4}$, and bf16 precision.
- Hardware: two NVIDIA RTX 3090 GPUs.
- For Type 4 samples (no semantic pitfall CQs), $n$ normal CQs ($CQ_{normal}$) are randomly sampled for training.

## Key Experimental Results

### Main Results

The dataset is drawn from 6 ontologies (AWO, Dem@Care, SWO, Stuff, OntoDT, Pizza), comprising 1,563 samples (1,368 train / 195 test), with similarity threshold $\tau = 0.7$.

| Model | CQ_sp P | CQ_sp R | CQ_sp F1 | CQ_sp CosSim | CQ_normal P | CQ_normal R | CQ_normal F1 |
|------|---------|---------|----------|--------------|-------------|-------------|--------------|
| GPT-4.1 | 49.0 | 34.0 | 40.1 | 0.659 | 82.1 | 27.1 | 40.7 |
| LLaMA-3.1-8B | 29.8 | 20.5 | 24.3 | 0.593 | 58.5 | 20.2 | 30.0 |
| **VSPO** | **75.0** | **62.2** | **68.0** | **0.795** | **95.9** | **35.8** | **52.1** |

### Ablation Study

Performance breakdown by misalignment type:

| Type | VSPO P | VSPO R | VSPO F1 | GPT-4.1 F1 |
|------|--------|--------|---------|------------|
| Type 1 Missing Axiom | 71.1 | 54.4 | 61.6 | 44.8 |
| Type 2 Undefined Axiom | 83.8 | 69.4 | 75.9 | 36.1 |
| Type 3 Misused Axiom | 69.0 | 63.2 | 66.0 | 39.0 |

Under the Type 4 aligned setting, VSPO achieves 96.7 precision and 80.4 F1, indicating that the model not only learns to detect pitfalls but also demonstrates strong general CQ generation capability.

### Key Findings

- VSPO outperforms baselines across all three semantic pitfall types; Type 2 (Undefined Axiom) F1 reaches 75.9, far exceeding GPT-4.1's 36.1.
- The threshold-based evaluation metrics are sensitive to $\tau$—LLaMA shows similar CosSim across types but large variance in P/R/F1, revealing limitations in prior evaluation paradigms.
- In generalization experiments on unseen ontologies, VSPO maintains competitive performance.

## Highlights & Insights

- **Novel problem formulation**: This work is the first to reframe the objective of CQ generation from "similarity to existing CQs" to "capability to detect semantic pitfalls," representing a substantive advance in ontology validation.
- The **controlled misalignment injection strategy** is elegant: by systematically deleting or substituting axioms to construct three types of semantic pitfall training data, it avoids the prohibitive cost of manual annotation.
- The fine-tuned 8B model substantially outperforms GPT-4.1, demonstrating that domain-specific fine-tuning retains advantages for tasks requiring deep logical reasoning.

## Limitations & Future Work

- The current work covers only TBox validation; ABox (instance-level) validation is not addressed.
- The misalignment injection strategy is relatively rule-based (random deletion/substitution); real-world semantic pitfalls may be considerably more complex.
- Evaluation relies on SentenceBERT cosine similarity with a fixed threshold, which may not fully reflect the actual validation effectiveness of the generated CQs.
- End-to-end validation via CQ-to-SPARQL-OWL query translation is a promising direction for future work.

## Related Work & Insights

- **OOPS!**: Establishes a pitfall catalog through empirical analysis of 693+ ontologies, but certain semantic pitfalls remain beyond the reach of rule-based detection.
- Prior CQ generation work (Rebboud 2024, Alharbi 2024, Pan 2024) all adopt similarity to existing CQs as the evaluation criterion, without addressing semantic pitfall detection.
- LoRA fine-tuning proves highly effective for niche, highly specialized tasks such as ontology engineering.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem formulation is novel and practically valuable; the misalignment injection strategy for data construction is elegant; and the experimental results are convincing. One point is deducted due to the relatively niche application domain (ontology engineering) and the absence of an end-to-end evaluation loop from CQ generation to actual ontology validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ACL 2026\] Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](../../ACL2026/llm_nlp/not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[AAAI 2026\] LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation](loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](../../ACL2026/llm_nlp/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)

</div>

<!-- RELATED:END -->
