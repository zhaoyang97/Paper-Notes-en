---
title: >-
  [Paper Note] The Impossibility of Fair LLMs
description: >-
  [ACL 2025 (Long Paper)][LLM (Other)][LLM Fairness] This work systematically analyzes four mainstream technical fairness frameworks (FTU, multi-sided fairness, group fairness/fair representations, composability of fairness) and demonstrates that all of them face inherent and insurmountable challenges in general-purpose LLM scenarios. It argues that strictly fair LLMs are theoretically impossible and proposes three pragmatic future directions.
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM (Other)"
  - "LLM Fairness"
  - "Impossibility Result"
  - "Group Fairness"
  - "Fair Representations"
  - "Algorithmic Bias"
date: 2026-05-08
content_hash: d50f21d529e1dd98
---

# The Impossibility of Fair LLMs

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2406.03198](https://arxiv.org/abs/2406.03198)  
**Code**: None  
**Area**: AI Safety / Fairness / LLM Ethics  
**Keywords**: LLM Fairness, Impossibility Result, Group Fairness, Fair Representations, Algorithmic Bias

## TL;DR

This work systematically analyzes four mainstream technical fairness frameworks (FTU, multi-sided fairness, group fairness/fair representations, composability of fairness) and demonstrates that all of them face inherent and insurmountable challenges in general-purpose LLM scenarios. It argues that strictly fair LLMs are theoretically impossible and proposes three pragmatic future directions.

## Background & Motivation

**Background**: In traditional ML scenarios, robust technical frameworks for fairness exist. Group fairness requires conditional equivalence of classification outcomes across different demographic groups (e.g., demographic parity, equalized odds); fair representations require data representations to exclude identifiable information about sensitive attributes; and fairness through unawareness (FTU) requires that sensitive attributes are excluded from model inputs. These frameworks have been widely applied and verified in structured data and single-use case scenarios, such as loan approval and recidivism prediction.

**Limitations of Prior Work**: With the proliferation of general-purpose LLMs like ChatGPT and Claude, existing research on LLM fairness remains mostly at the level of "associative bias testing." For example, WinoBias evaluates gender bias in coreference resolution, and BBQ evaluates stereotypical associations in question answering. However, these benchmarks only capture statistical associations, which differ fundamentally from the fairness defined by stricter fairness frameworks. Passing WinoBias does not imply that a model is fair in the sense of group fairness.

**Key Challenge**: General-purpose LLMs differ fundamentally from traditional ML systems in three key dimensions: (1) training data consists of unstructured text rather than structured tables, (2) use cases involve infinitely many general-purpose tasks instead of a single predictive task, and (3) stakeholders expand from a binary "user-model" dynamic to a multi-party game involving "developers, users, content producers, and subjects." These differences invalidate the underlying assumptions of traditional fairness frameworks.

**Goal**: To systematically analyze whether mainstream fairness frameworks are viable in LLM scenarios, distinguish "inherent challenges" (insurmountable even with perfect technology) from "empirical challenges" (solvable through technical progress), and chart directions for future LLM fairness research.

**Key Insight**: A "framework-vs-challenge" comparative analysis is utilized. Instead of simply testing specific bias metrics, this work identifies the inherent properties of LLMs that render each fairness framework unviable.

**Core Idea**: General-purpose LLMs face inherent and insurmountable challenges under every strict technical fairness framework, rendering fair LLMs theoretically impossible.

## Method

### Overall Architecture

This work employs a systematic conceptual analysis method to cross-examine the core technical characteristics of LLMs (unstructured data, generality, multi-modality, and multi-stakeholder dynamics) against four major categories of fairness frameworks. For each intersection, it determines whether there exists an "inherent challenge"—an obstacle that cannot be overcome regardless of technological progress. The analytical workflow is: Framework Definition $\to$ LLM Character Conflict Point $\to$ Inherent Nature Argumentation $\to$ Impact Evaluation.

### Key Designs

1. **Impossibility Proof of Fairness Through Unawareness (FTU)**:

    - **Function**: Proves that sensitive attributes (such as gender, race, nationality, etc.) cannot be stripped from the unstructured training data of LLMs.
    - **Mechanism**: FTU requires model inputs to exclude sensitive attributes. While structured data allows direct deletion of the "gender" column, sensitive attributes in natural language permeate every level of language. At the explicit level, nationality information in sentences like "She grew up in Portugal" is deeply coupled with semantics, making the sentence meaningless once removed. At the implicit level, things like the frequency of first-person pronoun usage correlate with social status, as in $P(\text{lower status} | \text{high 1st-person pronoun rate}) > P(\text{higher status})$. In gendered languages (such as Spanish and German), gendered information is structurally embedded in the grammar and cannot be excised.
    - **Design Motivation**: Reveals that the unstructured nature of LLM training data makes FTU theoretically impossible, rather than just being a difficult engineering challenge.

2. **Invalidation of Producer-side Fairness Standards**:

    - **Function**: Proves that LLMs, acting as a new type of stakeholder, disrupt the traditional definition of producer-side fairness in multi-sided fairness frameworks.
    - **Mechanism**: The traditional multi-sided fairness framework in information retrieval requires content producers to receive fair exposure allocation. However, LLMs can bypass content producers entirely. When a user asks "how to roast coffee beans," the LLM provides the answer directly without linking back to the original source, resulting in zero exposure for the producer. The LLM system itself becomes a new stakeholder that extracts value from producers (e.g., SearchGPT integrating search), making the traditional $\text{Fairness}_{\text{producer}} = f(\text{exposure}_i / \text{relevance}_i)$ meaningless in LLM scenarios because $\text{exposure}_i \to 0$.
    - **Design Motivation**: Points out that LLMs are not just information retrieval tools, but hybrid engines of information production and consumption, fundamentally altering the benefit structures in multi-sided fairness frameworks.

3. **Impossibility of Cross-Context Fairness (Combinatorial Explosion)**:

    - **Function**: Proves that general-purpose LLMs cannot simultaneously maintain fairness across all combinations of demographics, use cases, and sensitive attributes.
    - **Mechanism**: Lechner et al. (2021) proved that non-trivial models cannot be simultaneously fair across all data distributions. General-purpose LLMs face a three-dimensional combinatorial explosion of demographics (global users), use cases (infinitely many tasks), and sensitive attributes (gender, race, age, nationality, etc., along with their intersectional combinations). Debiasing for one context may destroy necessary information for another; for instance, financial scenarios require removing gender information, whereas medical scenarios require gender information for precise diagnosis. The fair representation framework $Z = \text{Enc}(X)$ requires $I(Z; S) = 0$ (where $S$ is the sensitive attribute), but Gonen & Goldberg (2019) demonstrated that existing debiasing methods "only hide bias rather than removing it."
    - **Design Motivation**: Elevates the impossibility from a single framework to the systematic level—even if a framework is feasible within a single context, cross-context combinatorial explosion makes it unscalable.

4. **Impossibility of Composing Fairness**:

    - **Function**: Proves that modern LLM systems, as compositions of multiple models, cannot derive system-level fairness from component-level fairness.
    - **Mechanism**: Dwork & Ilvento (2019) showed that the combination of two individually fair models does not guarantee fairness. Modern LLM systems are inherently multi-model compositions (e.g., ChatGPT + DALL-E forming a multimodal system; RLHF/DPO viewed as a composition of an "ethics-aligned model" and a "base LLM"). Even if each component satisfies a certain fairness guarantee, the compiled system does not inherit these guarantees.
    - **Design Motivation**: Reveals that LLM alignment methods (RLHF, DPO, Constitutional AI) are themselves model compositions, and their fairness guarantees cannot be propagated to the final system.

### Framework for Future Directions

The authors propose three pragmatic future paths: (1) **Developer Responsibility Standards**: requiring LLM developers to provide training data transparency and real-world usage data to support third-party auditing; (2) **Context-Specific Evaluation**: abandoning the pursuit of general-purpose fairness in favor of tailoring fairness metrics and evaluation methodologies for specific application scenarios; (3) **Scalable AI-Assisted Evaluation**: utilizing techniques like LLM-as-a-judge and synthetic data simulation to scale up fairness evaluations across the diverse use cases of LLMs.

## Key Experimental Results

### Applicability Analysis of Fairness Frameworks

As this is a theoretical analysis paper, the core contribution is conceptual argument. The systematic analysis conclusions for each framework are synthesized below:

| Fairness Framework | Core Requirement | Inherent LLM Challenge | Overcomable? |
|-----------|---------|------------|-----------|
| Fairness Through Unawareness (FTU) | Model inputs contain no sensitive attributes | Sensitive attributes are ubiquitous in unstructured data | ❌ Inherently impossible |
| Group Fairness | Classification outcomes are equivalent across populations | Combinatorial explosion of populations/use cases/attributes | ❌ Inherently impossible |
| Fair Representations | Data representations contain no sensitive information | Debiasing only hides bias; conflicts across contexts | ❌ Inherently impossible |
| Producer-side Fairness | Content producers receive fair exposure | LLMs bypass producers with direct answers | ❌ Invalidation of standard |
| Counterfactual Fairness | Outputs remain unchanged under counterfactuals | Requires causal structure knowledge; infeasible in general scenarios | ❌ Inherently impossible |
| Individual Fairness | Similar inputs $\to$ Similar outputs | Similarity metrics cannot be defined across use cases | ❌ Inherently impossible |

### Gap Between Existing Evaluation Methods and Strict Fairness

| Evaluation Method | Test Content | Relation to Strict Fairness | Limitations |
|---------|---------|----------------|--------|
| WinoBias | Gender association in coreference resolution | Only measures association $\neq$ group fairness | Passing does not imply fairness |
| BBQ | Stereotypes in QA | Only measures associative bias | Only captures explicit bias |
| Input Perturbation (e.g., dialect switching) | Output consistency after input modifications | Crude counterfactual approximation | Does not address true counterfactuals |
| BOLD/RealToxicity | Toxicity/bias in text continuations | Statistical association rather than fairness | High score $\neq$ fairness |

### Key Findings

- **Existing LLM bias evaluation remains at the "association level" rather than the "fairness level"**: Benchmarks like WinoBias and BBQ measure statistical associations, rather than the conditional equivalence defined by frameworks like group fairness, leaving a fundamental semantic gap between them.
- **The distinction between "inherent challenges" and "empirical challenges" is a core contribution**: Empirical challenges are expected to be resolved through technical advances, whereas inherent challenges remain insurmountable even with perfect technology.
- **Fairness is non-transitive in multi-model compositions**: This raises fundamental questions about the fairness guarantees of alignment methods such as RLHF and DPO.
- **The "whack-a-mole" dilemma of debiasing**: Debiasing for one context may inadvertently destroy necessary information required for another context.

## Highlights & Insights

- **Elevating from "measuring bias" to "analyzing frameworks"**: Instead of merely pointing out that LLMs are biased, this work systematically asks whether a fair LLM is possible even with perfect bias measurements. This meta-level analysis establishes clear boundaries for fairness research.
- **The taxonomy of "inherent vs. empirical challenges" is highly valuable**: It helps researchers distinguish which issues warrant technical problem-solving (empirical challenges) and which require paradigm shifts (inherent challenges), preventing wasted research efforts in impossible directions.
- **The "new stakeholder" argument for producer-side fairness is forward-looking**: The integration of LLMs with search engines is turning this theoretical question into real-world business conflict. The paper's analysis provides a conceptual foundation for policy discussions.
- **The three future paths balance pragmatism and foresight**: Particularly, the direction of "AI-assisted evaluation"—using LLMs to evaluate LLM fairness—historically risks "bias all the way down," yet currently represents the only viable path to scale fairness evaluations to the sheer volume of LLM usage.

## Limitations & Future Work

- **Arguments are conceptual analyses rather than formal mathematical proofs**: Each impossibility claim relies on logical reasoning and counterexamples rather than strict mathematical proofs (such as Arrow's Impossibility Theorem). Formalizing these proofs would significantly enhance their rigor.
- **Omission of the feasibility and value of "partial fairness"**: While strict fairness is impossible, "good enough fairness" could still be valuable. Quantifying this feasible range remains an important open question.
- **Mainly focused on English LLMs**: Multilingual scenarios introduce additional complexities, including grammatical gender systems across different languages, cultural background variations, and amplified bias in low-resource languages.
- **Lack of concrete technical schemes for the three paths**: Issues such as operationalizing standards, standardizing evaluation pipelines, and avoiding circular bias remain unresolved and are left to future work.
- **Insufficient discussion on the policy implications of fairness impossibility**: What this implies for regulatory frameworks like the EU AI Act warrants deeper investigation.

## Related Work & Insights

- **vs. Gallegos et al. (2023) / Li et al. (2024) Surveys**: These surveys catalog bias metrics and debiasing methodologies under the assumption that fairness frameworks are applicable to LLMs. Conversely, this work asks a more fundamental question—whether the frameworks themselves are applicable.
- **vs. Lechner et al. (2021)**: The latter proves that fair representations cannot hold across all data distributions. This work generalizes this impossibility to all mainstream frameworks, offering broader coverage.
- **vs. Dwork & Ilvento (2019)**: The latter proves that fairness is not composable. This work applies this proof to RLHF/DPO alignment pipelines, demonstrating that alignment itself is a form of model composition.
- **vs. Gonen & Goldberg (2019)**: The latter experimentally shows that word embedding debiasing merely "hides bias." This work extends this insight to the entire fair representation framework within LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically argue the impossibility of fair LLMs from a framework perspective, offering a unique and in-depth view.
- Experimental Thoroughness: ⭐⭐⭐ As a theoretical analysis paper, it lacks experimental data, and its arguments rely on logical deduction rather than mathematical proof.
- Writing Quality: ⭐⭐⭐⭐⭐ Features a clear argumentative structure, a complete chain of logic, and a natural fusion of interdisciplinary perspectives.
- Value: ⭐⭐⭐⭐ Demarcates the boundaries for LLM fairness research, providing directional guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs + Persona-Plug = Personalized LLMs](llms_persona-plug_personalized_llms.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] LlamaDuo: LLMOps Pipeline for Seamless Migration from Service LLMs to Small-Scale Local LLMs](llamaduo_llmops_pipeline_for_seamless_migration_from_service_llms_to_small-scale.md)

</div>

<!-- RELATED:END -->
