---
title: >-
  [Paper Note] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs
description: >-
  [ACL 2026][Content Moderation] This paper proposes ChAIRO, a Contextual Hierarchical Analogical Induction and Reasoning Optimization framework that employs a three-stage pipeline (analogical case generation → rule induction → rule-injected fine-tuning) to enable LLMs to autonomously generate analogical cases and induce explicit moderation rules for content moderation. ChAIRO achieves a 4.5% F1 improvement over single-instance rule generation and a 2.3% improvement over static RAG.
tags:
  - ACL 2026
  - Content Moderation
  - Rule Induction
  - Analogical Reasoning
  - Hierarchical Reasoning Chain
  - End-to-End Optimization
date: 2026-05-08
content_hash: 7f22b72fdf74e42c
---

# ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs

**Conference**: ACL 2026
**arXiv**: [2604.10502](https://arxiv.org/abs/2604.10502)
**Code**: None
**Area**: Information Retrieval
**Keywords**: Content Moderation, Rule Induction, Analogical Reasoning, Hierarchical Reasoning Chain, End-to-End Optimization

## TL;DR
This paper proposes ChAIRO, a Contextual Hierarchical Analogical Induction and Reasoning Optimization framework that employs a three-stage pipeline (analogical case generation → rule induction → rule-injected fine-tuning) to enable LLMs to autonomously generate analogical cases and induce explicit moderation rules for content moderation. ChAIRO achieves a 4.5% F1 improvement over single-instance rule generation and a 2.3% improvement over static RAG.

## Background & Motivation

**Background**: Leveraging LLMs for content moderation has become a promising direction, providing interpretable moderation decisions through chain-of-thought reasoning. However, even SOTA models frequently err in scenarios involving contextual ambiguity or underspecified moderation criteria.

**Limitations of Prior Work**: (1) CoT reasoning in content moderation lacks reference to precedents and relies solely on explicit criteria (e.g., presence of insults or incitement), failing to detect implicit discriminatory logic (e.g., metaphorical discrimination such as "low score equals low ability"); (2) manually defined high-level rules (e.g., "pornographic content") are too coarse to capture fine-grained distinctions; (3) LLM-driven adaptive rule discovery relies on general priors and ignores domain expertise accumulated by human moderation specialists.

**Key Challenge**: Accurate, context-sensitive moderation rules are required to handle ambiguous cases, yet constructing and discovering such rules is inherently difficult—manual enumeration is impractical, while automatic generation lacks sufficient precision.

**Goal**: To leverage analogical cases to improve rule induction quality, and to unify case retrieval, rule generation, and moderation decisions through end-to-end optimization.

**Key Insight**: Unlike CarO (a concurrent work from the same group, arXiv:2604.10504), ChAIRO does not employ DPO; instead, it introduces an explicit rule induction step—an auxiliary reasoning model first induces textual moderation rules from analogical cases, which are then injected into the reasoning chain for a second-round fine-tuning.

**Core Idea**: A three-stage hierarchical optimization process: (1) analogical chain SFT trains the model to generate analogical cases; (2) an auxiliary model induces explicit rules from analogical cases; (3) rules are injected into the reasoning chain for a second-round SFT that integrates "case + rule + reasoning" capabilities.

## Method

### Overall Architecture
**Stage 1**: For each training sample, retrieve semantically similar cases → generate reasoning chains containing analogies using an LLM → train via SFT to enable the model to autonomously generate analogical cases.
**Stage 2**: Use the Stage 1 model to generate virtual analogical cases for each sample → employ an auxiliary reasoning model (QwQ-32B) to induce explicit moderation rules from the analogical cases.
**Stage 3**: Structure analogical cases, rules, and reasoning chains into a hierarchical format (`<RULE>` + `<ANALOGY>` + `<REASONING>`) → perform the second-round SFT.

### Key Designs

1. **Self-Augmented Analogical Reasoning Chain Generation (Stage 1)**

    - **Function**: Internalizes analogical reasoning capability, enabling the model to autonomously generate relevant analogies for new samples.
    - **Mechanism**: BGE-M3 encodes all training samples; semantically similar cases are retrieved for each sample. The sample, retrieved cases, and labels are fed to an LLM to generate analogical reasoning chains, followed by SFT training. After training, the model can autonomously generate analogical cases without external retrieval.
    - **Design Motivation**: Cases retrieved by static RAG may not be optimal for a given sample; by internalizing this capability through SFT, the model can dynamically generate more relevant analogies.

2. **Auxiliary Model Rule Induction (Stage 2)**

    - **Function**: Distills explicit, interpretable moderation rules from analogical cases.
    - **Mechanism**: The Stage 1 model generates virtual analogical cases for each training sample; QwQ-32B then serves as the auxiliary reasoning model to induce textual moderation rules from the original sample and its analogical cases. Category descriptions within the induced rules are automatically validated against ground-truth labels, and inconsistent samples are discarded.
    - **Design Motivation**: Analogical cases provide contextual grounding, yielding more precise and targeted rules than those generated from single instances alone (+4.5% F1).

3. **Hierarchical Rule Injection and Final Fine-Tuning (Stage 3)**

    - **Function**: Integrates analogies, rules, and reasoning into a unified structured reasoning capability.
    - **Mechanism**: Special tokens structure the reasoning chain into three layers: `<RULE>` (induced rules), `<ANALOGY>` (analogical cases), and `<REASONING>` (holistic reasoning). A second-round SFT is performed on top of the Stage 1 parameters.
    - **Design Motivation**: The hierarchical structure explicitly guides the model on when to apply rules, when to reference analogies, and when to perform reasoning, thereby improving interpretability and consistency.

## Key Experimental Results

### Main Results (Chinese Content Moderation Dataset)

| Method | Avg. F1 | Political | Pornographic | Violent | Gambling | Bias | Harmless |
|--------|---------|-----------|--------------|---------|----------|------|----------|
| DeepSeek R1 | 77.1 | 72.7 | 91.4 | 86.1 | 94.3 | 64.6 | 59.7 |
| DeepSeek V3 | 80.3 | 79.0 | 90.3 | 89.8 | 95.0 | 70.5 | 62.5 |
| Naive SFT | ~85 | - | - | - | - | - | - |
| Rule-injected SFT (single-instance rules) | ~85.7 | - | - | - | - | - | - |
| Static RAG | ~87.9 | - | - | - | - | - | - |
| **ChAIRO (Ours)** | **~90.2** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Comparison | F1 Gain | Note |
|------------|---------|------|
| ChAIRO vs. Naive SFT | +5.3% | Value of explicit rule injection |
| ChAIRO vs. Single-instance Rule SFT | +4.5% | Analogical cases improve rule quality |
| ChAIRO vs. Static RAG | +2.3% | End-to-end optimization vs. staged pipeline |

### Key Findings
- **Explicit rule injection yields a 5.3% improvement**, demonstrating the critical role of rules in handling ambiguous moderation cases.
- **Analogy-driven rules outperform single-instance rules by 4.5%**, confirming that contextual analogies genuinely enhance rule quality.
- **End-to-end optimization outperforms static RAG by 2.3%**, as errors accumulate in staged pipelines.
- **Human evaluation confirms superior rule quality**: clarity, interpretability, and applicability all exceed baselines.
- **Cross-model generalization validated**: induced rules transfer effectively to other LLMs.

## Highlights & Insights
- The **three-layer cognitive architecture of "analogy → rule → reasoning"** simulates expert human decision-making. Compared to CarO's "analogy → reasoning" paradigm, ChAIRO adds an explicit knowledge abstraction layer, enhancing interpretability.
- The **hierarchical reasoning chain format** (`<RULE>` + `<ANALOGY>` + `<REASONING>`) provides structured audit trails, allowing each decision to be traced back to specific rules and analogical cases.
- **Complementary relationship with CarO**: CarO reinforces the consistency of analogical reasoning via DPO, while ChAIRO enhances reasoning interpretability through rule induction; the two approaches are mutually compatible.

## Limitations & Future Work
- Rule induction requires an auxiliary reasoning model (QwQ-32B), increasing training cost.
- Rules are expressed in natural language, offering no formal guarantees of consistency or non-contradiction.
- The two-round SFT training pipeline is relatively complex; simplification warrants investigation.
- The dataset is predominantly Chinese; validation on English and multilingual settings is insufficient.
- The rule repository lacks a continuous update mechanism, necessitating retraining when novel violation types emerge.

## Related Work & Insights
- **vs. CarO (2604.10504)**: A concurrent work from the same group. CarO reinforces analogical reasoning via DPO, while ChAIRO introduces explicit rule induction. ChAIRO places greater emphasis on interpretability.
- **vs. Rule-based Moderation**: Traditional rules are manually defined coarse-grained criteria, whereas ChAIRO's rules are fine-grained, context-sensitive rules automatically induced from analogical cases.
- **vs. Kumar et al. (2024)**: Also explores LLM-based rule discovery but relies on single-instance context; ChAIRO provides a richer inductive basis through analogical cases.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage hierarchical framework is well-conceived, though it is closely related to CarO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional ablations, human evaluation, and cross-model generalization testing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-designed RQ-driven experimental layout.
- Value: ⭐⭐⭐⭐ Explicit rule induction holds practical value for interpretable content moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/information_retrieval/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)

</div>

<!-- RELATED:END -->
