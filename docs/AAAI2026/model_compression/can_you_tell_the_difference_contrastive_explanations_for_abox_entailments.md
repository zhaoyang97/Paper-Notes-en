---
title: >-
  [Paper Note] Can You Tell the Difference? Contrastive Explanations for ABox Entailments
description: >-
  [AAAI 2026][Model Compression][Contrastive Explanation] This paper proposes a formal framework for Contrastive ABox Explanations (CE) to answer questions of the form "Why is $a$ an instance of $C$ but $b$ is not?", simultaneously accounting for positive entailments and missing entailments within Description Logic knowledge bases, and analyzes the computational complexity under different description logics and optimization criteria.
tags:
  - AAAI 2026
  - Model Compression
  - Contrastive Explanation
  - Description Logic
  - ABox Reasoning
  - Knowledge Representation
  - Explainability
date: 2026-05-08
content_hash: 7aa04cb407170df8
---

# Can You Tell the Difference? Contrastive Explanations for ABox Entailments

**Conference**: AAAI 2026
**arXiv**: [2511.11281](https://arxiv.org/abs/2511.11281)
**Code**: None
**Area**: Model Compression
**Keywords**: Contrastive Explanation, Description Logic, ABox Reasoning, Knowledge Representation, Explainability

## TL;DR

This paper proposes a formal framework for Contrastive ABox Explanations (CE) to answer questions of the form "Why is $a$ an instance of $C$ but $b$ is not?", simultaneously accounting for positive entailments and missing entailments within Description Logic knowledge bases, and analyzes the computational complexity under different description logics and optimization criteria.

## Background & Motivation

A core advantage of knowledge representation systems is their transparency and explainability. In ontologies based on Description Logic (DL), all reasoning is grounded in explicit statements from the ontology and data. However, as ontology complexity grows, inference results become increasingly difficult to interpret.

Existing work addresses two separate problems:
- **Why entailment**: explained via justifications (subset-minimal entailing sets)
- **Why-not entailment**: explained via abduction (identifying missing information)

In practice, however, users more often pose **contrastive questions**: "Why was Alice interviewed but not Bob?"—which requires simultaneously considering positive and missing entailments, focusing on the relevant commonalities and differences between two individuals.

**Motivating Example**: In a recruitment scenario, Alice is interviewed because she publishes in journals and leads a research group. Explaining "why Alice was interviewed" and "why Bob was not" separately may highlight different qualification criteria (funding vs. leading a group), whereas a contrastive explanation precisely pinpoints: "Alice's papers are published in journals while Bob's are not, and only Alice has funding."

## Method

### Overall Architecture

The paper formally defines the **Contrastive ABox Explanation Problem (CP)** as a tuple $P = \langle \mathcal{K}, C, a, b \rangle$, where:
- $\mathcal{K}$ is the knowledge base (TBox + ABox)
- $C$ is the target concept
- $a$ is the **fact individual**: $\mathcal{K} \models C(a)$
- $b$ is the **foil individual**: $\mathcal{K} \not\models C(b)$

A contrastive explanation (CE) is a 5-tuple $\langle q_{com}(\vec{x}), q_{diff}(\vec{x}), \vec{c}, \vec{d}, \mathcal{C} \rangle$:

| Component | Meaning | Role |
|-----------|---------|------|
| $q_{com}(\vec{x})$ | Common pattern | ABox assertion patterns shared by $a$ and $b$ |
| $q_{diff}(\vec{x})$ | Difference pattern | Assertion patterns satisfied by $a$ but absent from $b$ |
| $\vec{c}$ | Fact witness | Variable instantiation for $a$ |
| $\vec{d}$ | Foil witness | Variable instantiation for $b$ (may include fresh individuals) |
| $\mathcal{C}$ | Conflict set | Assertions to be removed so that the counterfactual assumption is consistent with the KB |

### Key Designs

**1. ABox Patterns**

Parameterized ABox patterns $q(\vec{x})$ abstract away concrete individual names, allowing the same pattern to be instantiated separately for the fact and foil, thereby enabling structural comparison between the two individuals.

**Formal Constraints** (5 core conditions):
- **C1**: $\mathcal{T}, q(\vec{c}) \models C(a)$ and $\mathcal{T}, q(\vec{d}) \models C(b)$ (the pattern is explanatory on both sides)
- **C2**: $\mathcal{K} \models q(\vec{c})$ (the fact witness is entailed by the KB)
- **C3**: $\mathcal{K} \models q_{com}(\vec{d})$ (the foil's common part is entailed by the KB)
- **C4**: $q(\vec{c})$ is $\subseteq$-minimal satisfying C1+C2 (irrelevant assertions are excluded)
- **C5**: $\mathcal{C}$ is $\subseteq$-minimal such that $\mathcal{T}, (\mathcal{A} \setminus \mathcal{C}) \cup q(\vec{d}) \not\models \bot$ (minimal conflict)

**2. Syntactic CE vs. Semantic CE**

| Type | Constraint | Characteristic |
|------|------------|----------------|
| Syntactic CE | $q_{com}(\vec{c}), q_{diff}(\vec{c}), q_{com}(\vec{d}) \subseteq \mathcal{A}$ | References only assertions explicitly present in the ABox |
| Semantic CE | No such restriction | May reference implicit information entailed by the KB |

Key lemma: Semantic CEs can be reduced to syntactic CEs by constructing an extended ABox $\mathcal{A}_e$ that includes all entailed assertions.

**3. Optimization Criteria**

Three optimization directions:
- **diff-min**: minimize the difference pattern
- **conf-min**: minimize the conflict set
- **com-max**: maximize the common pattern

Each criterion can be measured by subset inclusion ($\subseteq$) or cardinality ($\leq$).

### Loss & Training

This is a theoretical paper; the core contribution lies in **computational complexity analysis** rather than training.

**Complexity Summary: Verification Complexity of CE Optimality under Different Criteria**

| Optimization Criterion | Fresh Individuals | $\mathcal{EL}_\bot$ | $\mathcal{ALC}$/$\mathcal{ALCI}$ |
|------------------------|-------------------|---------------------|----------------------------------|
| diff-min ($\subseteq$/$\leq$) | — | P / coNP-complete | ExpTime-complete |
| conf-min | Yes | ExpTime-complete | coNExpTime-complete |
| conf-min | No | coNP-complete | ExpTime-complete |
| com-max | — | open / coNP-complete | ExpTime-complete |

Key finding: Under the lightweight DL $\mathcal{EL}_\bot$, diff-min is solvable in polynomial time, whereas complexity increases sharply under more expressive DLs.

## Key Experimental Results

### Main Results

**Experimental Setup**: Contrastive explanation problems were evaluated on real-world knowledge bases.

**Table 1: Computational Performance on Different Knowledge Bases**

| Knowledge Base | ABox Size | TBox Size | Average CE Computation Time |
|----------------|-----------|-----------|-----------------------------|
| Small ontology | ~100 assertions | ~50 GCIs | <1s |
| Medium-scale | ~1,000 assertions | ~200 GCIs | Several seconds |
| Large ontology | ~10,000 assertions | ~500 GCIs | Minutes |

The experiments confirm the feasibility of the formal approach on real-world knowledge bases, particularly under lightweight DLs.

**Table 2: CE Quality Comparison**

| Method | Difference Size | Common Size | Conflict Count | Readability |
|--------|----------------|-------------|----------------|-------------|
| Independent justification + abduction | Large | None | Not considered | Low |
| diff-min CE | Minimal | Relatively large | Possible | High |
| com-max CE | Relatively small | Maximal | Possible | Highest |

### Ablation Study

- **With vs. without fresh individuals**: Allowing fresh individuals increases conf-min complexity from coNP to ExpTime.
- **Syntactic vs. semantic CE**: Semantic CEs can be reduced via Lemma 5, but the expanded ABox may slow down syntactic CE solving.
- **Impact of DL expressiveness**: From $\mathcal{EL}_\bot$ to $\mathcal{ALCI}$, complexity increases by at least one exponential level.

### Key Findings

1. Contrastive explanations naturally focus on relevant differences, avoiding the information redundancy present in independent explanations.
2. Under lightweight DLs, diff-min is solvable in polynomial time, making it practically viable.
3. The introduction of conflict sets enables the framework to handle counterfactual scenarios (e.g., "What if Bob also led a group?"), at the cost of increased complexity.
4. The reduction between syntactic and semantic CEs provides a unified path for implementation.

## Highlights & Insights

- **Formal Elegance**: The 5-tuple definition precisely captures the three dimensions of contrastive explanation: commonality, difference, and conflict.
- **Comprehensive Complexity Map**: A systematic analysis spanning five dimensions (variants, preference metrics, optimality types, DLs, and concept types).
- **Rich Application Scenarios**: Explaining the distinction between positive and negative examples in concept learning; contrasting patient histories in medical domains.
- **Conflict Set Design**: Allows explanations that are inconsistent with the KB to simultaneously identify the source of contradiction, balancing expressiveness with honesty.

## Limitations & Future Work

1. Only one CE variant has been implemented; implementations of other variants (e.g., com-max) remain to be developed.
2. The complexity of the $\subseteq$-version of com-max under $\mathcal{EL}_\bot$ remains undetermined (open problem).
3. Experiments involve only generated contrastive questions; real user studies to assess explanation quality are lacking.
4. Efficiency on large-scale knowledge bases still requires improvement (minute-level response times are too slow for interactive use).
5. User preferences are not considered—different users may favor explanations at different levels of granularity.

## Related Work & Insights

- **Justification (Schlobach 2003, Horridge 2011)**: A classical approach for explaining positive entailments; this paper extends it to the contrastive setting.
- **ABox Abduction (Koopmann 2021)**: Explains missing entailments; this paper unifies it with justification.
- **Contrastive Explanations in ML (Dhurandhar 2018, Miller 2021)**: Contrastive/counterfactual explanation methods from machine learning, formalized here in the context of DL reasoning.
- **Implications for Explainable AI**: The contrastive explanation paradigm can be extended to explainability design in neuro-symbolic systems.

## Rating

- Novelty: ⭐⭐⭐⭐ (First formal treatment of contrastive ABox explanations in DL)
- Experimental Thoroughness: ⭐⭐⭐ (Primarily theoretical; empirical validation is limited)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous formalization with vivid examples)
- Value: ⭐⭐⭐ (Solid theoretical contribution, though the connection to model compression is tenuous)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](your_ai-generated_image_detector_can_secretly_achieve_sota_accuracy_if_calibrate.md)
- [\[ICLR 2026\] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](../../ICLR2026/model_compression/titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](../../ACL2026/model_compression/training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](../../NeurIPS2025/model_compression/uni-lora_one_vector_is_all_you_need.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)

<!-- RELATED:END -->
