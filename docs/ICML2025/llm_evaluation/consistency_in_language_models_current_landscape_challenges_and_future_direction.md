---
title: >-
  [Paper Note] Consistency in Language Models: Current Landscape, Challenges, and Future Directions
description: >-
  [ICML 2025 (Workshop on Reliable and Responsible Foundation Models)][LLM Evaluation][LLM Consistency] This paper systematically surveys the landscape of LLM consistency research, proposing a taxonomy that comprises logical consistency (negation, symmetry, transitivity), semantic consistency, factual/informational consistency, and non-logical consistency (morality/norms). It analyzes the deficiencies of evaluation methods from 2019 to 2025 and calls for the establishment of st…
tags:
  - "ICML 2025 (Workshop on Reliable and Responsible Foundation Models)"
  - "LLM Evaluation"
  - "LLM Consistency"
  - "Behavioral Consistency"
  - "Evaluation Frameworks"
  - "Multilingual"
  - "Trustworthy AI"
date: 2026-05-08
content_hash: cec8c6c75b591b7d
---

# Consistency in Language Models: Current Landscape, Challenges, and Future Directions

**Conference**: ICML 2025 (Workshop on Reliable and Responsible Foundation Models)  
**arXiv**: [2505.00268](https://arxiv.org/abs/2505.00268)  
**Code**: None  
**Area**: LLM Reliability / Evaluation & Benchmarking  
**Keywords**: LLM Consistency, Behavioral Consistency, Evaluation Frameworks, Multilingual, Trustworthy AI

## TL;DR

This paper systematically surveys the landscape of LLM consistency research, proposing a taxonomy that comprises logical consistency (negation, symmetry, transitivity), semantic consistency, factual/informational consistency, and non-logical consistency (morality/norms). It analyzes the deficiencies of evaluation methods from 2019 to 2025 and calls for the establishment of standardized multilingual benchmarks and interdisciplinary approaches.

## Background & Motivation

**Consistency is a Prerequisite for Trust**: Consistency—producing similar outputs in similar scenarios and avoiding self-contradiction—is a fundamental user expectation for trusting AI systems, and a prerequisite for deployment in high-risk domains (medicine, law, finance). However, state-of-the-art LLMs frequently exhibit inconsistent behavior.

**Conceptual Confusion and Lack of Standards**: Existing studies lack a unified definition of "consistency"; authors often define it individually, contradict one another, or omit the definition entirely. Evaluation metrics are scattered, datasets are fragmented, and methods are non-reproducible, hindering the accurate estimation of model consistency levels and risking performance overestimation and risk underestimation.

**Core Motivation**: To map out the complete landscape of consistency research, establish a clear taxonomy, identify research gaps, and provide a roadmap of future directions for the community. This paper focuses strictly on the consistency analysis of text-only LLMs.

## Method

### Overall Architecture

As a survey/position paper, this work employs a systematic literature review methodology, focusing on peer-reviewed papers and influential preprints explicitly researching LLM consistency between 2019 and 2025. The dimensions of analysis include: (1) consistency terminology and taxonomy; (2) covered NLP tasks; (3) dataset size and availability; (4) evaluated model types; (5) evaluation methods and metrics; and (6) methods to improve consistency.

### Key Designs

1. **Two-tier Taxonomy of Behavioral Consistency**:
    - Function: Organizes fragmented consistency concepts into a structured taxonomy, distinguishing between logical/formal consistency and non-logical/informal consistency.
    - Core Taxonomy:
        - **Logical Consistency** (Jang et al., 2022): (a) Negation consistency, where $p \Leftrightarrow \neg p$ is false; (b) Symmetric consistency, where $f(x,y) = f(y,x)$; (c) Transitive consistency, where $X \to Y \land Y \to Z \Rightarrow X \to Z$; (d) Semantic consistency, where $f(X) = f(Y)$ when $X$ and $Y$ are semantically equivalent.
        - **Non-logical Consistency**: Moral consistency (maintaining non-contradictory moral stances across scenarios) and normative consistency (applying the same norms in similar situations).
        - **Factual/Informational Consistency**: Generated content does not contradict the source document (associated with hallucinations/faithfulness issues).
    - Design Motivation: The confusion of terminology in existing studies is the core issue hindering progress. Establishing a clear taxonomy is the first step toward standardized evaluation.

2. **Systematic Analysis and Critique of Evaluation Methods**:
    - Function: Analyzes the advantages and disadvantages of existing evaluation frameworks across four levels: input sampling, output sampling, base metrics, and aggregation methods.
    - Core Findings:
        - Input sampling (creating paraphrases/equivalent prompts) is more reliable than output sampling, as high-temperature sampling can artificially amplify inconsistency.
        - Base metrics have evolved from early token-matching (exact match rate) to semantic similarity (BERTScore, entailment/contradiction scores).
        - Aggregation methods are almost exclusively simple averages, with only Mündler et al. (2024) using sequential aggregation of contradiction scores, and Kuhn et al. (2023) using semantic entropy.
    - Design Motivation: To reveal the limitations of evaluation methods, providing a basis for designing more comprehensive benchmarks.

3. **Gap Analysis of Multilingual and Cross-lingual Consistency**:
    - Function: Identifies the most overlooked dimension in consistency research: cross-lingual consistency.
    - Core Evidence: Shen et al. (2024) found that safety guardrails are more easily bypassed in non-English; Xing et al. (2024) showed that querying the same knowledge in different languages yields inconsistent factual information; Jin et al. (2023) observed inconsistent medical advice across languages; and Zhou & Zhang (2024) found that bilingual models express different political orientations in different languages.
    - Design Motivation: 73% of LLM training data is in English (Longpre et al., 2023); the training data imbalance across languages inevitably leads to cross-lingual consistency issues, which is critical for global deployment.

### Existing Approaches for Improving Consistency

Existing approaches fall into two categories: (1) **Fine-tuning methods**—Elazar et al. (2021) designed custom loss functions, and Raj et al. (2025) utilized knowledge distillation and synthetic consistency datasets; (2) **Self-consistency methods**—self-consistency by Wang et al. (2023) and chain-of-thought by Wei et al. (2022) ensure alignment between the reasoning process and the final answer. The paper notes that these methods only address the symptoms rather than the root causes, lacking fundamental solutions at the levels of representation space, pre-training strategies, and architectural design.

## Key Experimental Results

### Main Results: Statistics of Current Consistency Research

| Analysis Dimension | Statistical Results | Description |
|---|---|---|
| Model Architecture | >2/3 use decoder-only/encoder-decoder (GPT/OPT/BART/T5) | Approximately 1/4 of studies involve BERT-like encoder-only models. |
| Proprietary Models | >50% of papers test closed-source models such as GPT-4 | Unreleased weights limit replication and root-cause analysis. |
| Task Types | QA > Summarization > NLI > Reasoning | About 1/3 of studies use non-standard custom tasks. |
| Dataset Sharing | Mostly public | Some only describe the creation process without sharing the data. |
| Evaluation Method | Dominated by input perturbation sampling | Output sampling (high temperature) may artificially amplify inconsistency. |

### Consistency Dimension Coverage

| Consistency Type | Number of Studies | Level of Standardization | Quality of Evaluation |
|---|---|---|---|
| Semantic Consistency | High (Most Common) | Medium | Medium |
| Logical Consistency (Negation/Symmetry/Transitivity) | Low | Low (Only Jang et al., 2022 is systematic) | Low |
| Factual Consistency | Medium | Medium | Medium |
| Cross-lingual Consistency | Extremely Low | No Standard | Low |
| Moral/Normative Consistency | Extremely Low | No Standard | Low |

### Statistics of Improvement Methods

| Approach Category | Representative Work | Effectiveness | Limitations |
|---|---|---|---|
| Custom Loss Fine-Tuning | Elazar et al. (2021) | Improves paraphrase consistency | May degrade performance on other tasks. |
| Knowledge Distillation | Raj et al. (2025) | Learns from a more consistent teacher model | Dependent on teacher model quality. |
| Synthetic Consistency Data | Raj et al. (2025); Zhao et al. (2024) | Constructs grouped consistent inputs and outputs | Limited coverage of synthetic data. |
| Self-Consistency Decoding | Wang et al. (2023) | Majority voting over multiple samplings | Increases inference cost; treats only symptoms. |

### Key Findings

- The lack of a unified terminology and definition system in consistency research is the foremost obstacle hindering progress.
- The vast majority of studies focus exclusively on English, leaving multilingual and cross-lingual consistency virtually unaddressed.
- Existing evaluations heavily rely on automated metrics and lack human evaluation baselines, particularly in culturally sensitive scenarios.
- There is tension between consistency and creativity/diversity—reducing inconsistency may sacrifice beneficial response diversity.
- The widespread use of closed-source models severely restricts root-cause analysis and reproducible research on inconsistency.

## Highlights & Insights

- **Consistency does not equal correctness**: A model can be consistently wrong, but inconsistency definitely implies that some outputs are incorrect. Consistency is a necessary but insufficient condition for trustworthiness.
- **Positive aspects of inconsistency**: Moderate inconsistency can foster beneficial diversity and creativity, and prompt users to critically examine AI outputs rather than trusting them blindly.
- **Value of the taxonomy**: Dividing consistency into logical and non-logical broad categories, further split into 5+ sub-types, provides a clear organizational framework for building comprehensive benchmarks.
- **Self-consistency vs. Faithfulness**: Self-consistency examines the stability of outputs under input perturbations, whereas faithfulness verifies whether explanations accurately reflect the reasoning process—the two are related but require entirely different evaluation methods.

## Limitations & Future Work

- As a survey/position paper, this work does not propose new consistency measurement metrics or evaluation benchmarks.
- Limited to text-only LLMs; multimodal consistency is only briefly mentioned in the appendix.
- Lacks in-depth analysis of hierarchical relationships and practical interactions between different consistency types.
- Does not provide quantitative comparisons or a unified evaluation framework scheme for different consistency types.
- Contains many recommendations but lacks clear concrete pathways for implementation, such as calling for "interdisciplinary approaches" without elaborating on execution.

## Related Work & Insights

- **BeCel Benchmark** (Jang et al., 2022): The first systematic LLM consistency benchmark, defining four logical consistency types: negation, symmetry, transitivity, and semantics. This paper extends non-logical consistency based on this.
- **SelfCheckGPT** (Manakul et al., 2023): Utilizes information consistency for zero-resource hallucination detection, demonstrating the practical value of consistency evaluations.
- **Semantic Entropy** (Kuhn et al., 2023): Quantifies output uncertainty through semantic entropy, acting as an aggregation method that goes beyond simple pairwise similarity.
- **Insights**: Consistency evaluation can serve as an auditing tool prior to LLM deployment; in safety-critical scenarios, consistency benchmarks may hold greater value than pure performance benchmarks; cross-lingual consistency issues may need to be addressed through representation alignment.

## Rating

- Novelty: ⭐⭐⭐ As a survey paper, it introduces no new methods, but the taxonomy and gap analysis make contributions.
- Experimental Thoroughness: ⭐⭐ No experiments; primarily literature review.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, in-depth analysis, and strong arguments.
- Value: ⭐⭐⭐⭐ Provides comprehensive directional guidance for consistency research, carrying significant reference value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](../../ACL2026/llm_evaluation/evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ACL 2025\] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](../../ACL2025/llm_evaluation/coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)
- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)

</div>

<!-- RELATED:END -->
