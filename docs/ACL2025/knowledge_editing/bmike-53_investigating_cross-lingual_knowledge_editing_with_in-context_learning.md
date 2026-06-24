---
title: >-
  [Paper Note] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning
description: >-
  [ACL 2025][Knowledge Editing][cross-lingual knowledge editing] Proposes BMIKE-53, a cross-lingual benchmark covering 53 languages and integrating three knowledge editing datasets (zsRE, CounterFact, and WikiFactDiff). It systematically evaluates in-context knowledge editing methods from zero-shot to 8-shot settings, revealing that writing systems (Latin vs. non-Latin) are more decisive than language families for cross-lingual editing performance…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "cross-lingual knowledge editing"
  - "in-context learning"
  - "multilingual benchmark"
  - "script type"
  - "language confusion"
date: 2026-05-08
content_hash: d0efa079b0858ab5
---

# BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning

**Conference**: ACL 2025  
**arXiv**: [2406.17764](https://arxiv.org/abs/2406.17764)  
**Code**: [GitHub](https://github.com/ercong21/MultiKnow/)  
**Area**: Knowledge Editing  
**Keywords**: cross-lingual knowledge editing, in-context learning, multilingual benchmark, script type, language confusion

## TL;DR

Proposes BMIKE-53, a cross-lingual benchmark covering 53 languages and integrating three knowledge editing datasets (zsRE, CounterFact, and WikiFactDiff). It systematically evaluates in-context knowledge editing methods from zero-shot to 8-shot settings, revealing that writing systems (Latin vs. non-Latin) are more decisive than language families for cross-lingual editing performance, and that metric-specific exemplar strategies significantly outperform hybrid configurations.

## Background & Motivation

**Background**: LLMs encode a vast amount of knowledge during pre-training, but this knowledge is static and becomes outdated over time. Knowledge Editing (KE) techniques aim to selectively modify specific knowledge in LLMs while maintaining other knowledge unaffected. Gradient-free In-Context Learning (ICL) methods are particularly suitable for closed-source models as they do not require access to model parameters.

**Limitations of Prior Work**: Existing KE research predominantly focuses on monolingual (English) scenarios. Cross-lingual KE—where knowledge edited in a source language needs to generalize to equivalent queries in other target languages—presents greater challenges, yet systematic studies are scarce. Most existing cross-lingual KE works employ gradient-based methods (e.g., ROME/MEMIT), which incur high computational overhead and are inapplicable to closed-source models. More critically, there is a lack of a unified evaluation benchmark covering a wide range of languages.

**Key Challenge**: Can knowledge edited in a source language via ICL effectively migrate to semantically equivalent target multilingual queries? What factors determine the success or failure of such cross-lingual transfer?

**Key Insight**: Building the most comprehensive multilingual KE benchmark to date (53 languages $\times$ 3 datasets) to systematically analyze the capability boundaries of cross-lingual IKE from multiple dimensions, including model scale, exemplar strategies, query types, and linguistic attributes.

## Method

### Overall Architecture

Two major components:
1. **BMIKE-53 Benchmark Construction**: Unified format for three KE datasets $\rightarrow$ Structured translation via GPT-4o expanded to 52 target languages $\rightarrow$ Native speaker review + back-translation quality control.
2. **Cross-Lingual IKE Evaluation**: Define cross-lingual IKE tasks and four query types $\rightarrow$ Design 4 experimental settings (zero-shot / one-shot / 8-shot mixed / 8-shot metric-specific) $\rightarrow$ Multi-dimensional analysis.

### Key Designs

1. **三数据集整合与统一格式**:

    - **Function**: Integrates zsRE (standard factual modification), CounterFact (counterfactual knowledge updates), and WikiFactDiff (real-world temporal updates) into a unified benchmark.
    - **Mechanism**: Each data record uniformly contains the edited knowledge item + four types of evaluation queries (reliability, generality, locality, portability), stored in JSON format.
    - **Design Motivation**: The three datasets cover a complete spectrum of KE scenarios from standard to counterfactual to temporal real-world contexts. Portability queries are constructed via one-hop knowledge graph reasoning to test the logical reasoning capacity of edited knowledge.

2. **四种 IKE 实验设置**:

    - **Function**: Systematically controls the impact of exemplar quantity and quality on cross-lingual IKE.
    - **Mechanism**: zero-shot (no exemplar, depending entirely on pre-training) $\rightarrow$ one-shot (1 random exemplar to familiarize with format) $\rightarrow$ 8-shot mixed (8 mixed-type exemplars exposing various query query patterns) $\rightarrow$ 8-shot metric-specific (8 exemplars of the same type as the evaluation metric to provide targeted guidance).
    - **Design Motivation**: Experiments demonstrate that exemplar quality (type matching) is far more important than quantity—the performance gains of metric-specific configurations on locality and portability far exceed those from merely increasing the count of mixed exemplars.

3. **四类跨语言查询设计**:

    - **Function**: Evaluates the completeness of cross-lingual knowledge editing from different perspectives.
    - **Mechanism**: Reliability (precisely translated queries) evaluates basic editing capability $\rightarrow$ Generality (semantically equivalent but differently phrased queries) tests generalization $\rightarrow$ Locality (irrelevant queries) evaluates knowledge retention $\rightarrow$ Portability (one-hop inference queries) tests reasoning transfer.
    - **Design Motivation**: The four query categories progress in difficulty—while rel/gen reach similar levels, loc/port perform significantly worse, revealing the true bottlenecks of cross-lingual IKE.

### Loss & Training

This work does not involve model training. Evaluation metrics are F1 score and Exact Match (EM). Cross-lingual performance is normalized using English EM as the baseline.

## Key Experimental Results

### Main Results

Cross-lingual IKE performance of Llama3.1-8B across three datasets (average F1 of 52 languages):

| Setting | zsRE rel | zsRE port | CF rel | CF loc | WFD rel | WFD port |
|------|---------|----------|--------|--------|---------|---------|
| zero-shot | 65.53 | 10.05 | 63.01 | 18.68 | 67.84 | 4.15 |
| one-shot | 75.27 | 20.81 | 71.92 | 12.66 | 70.53 | 4.15 |
| 8-shot mixed | 74.29 | 25.18 | 75.15 | 11.40 | 68.57 | 8.87 |
| 8-shot metric-specific | **74.86** | **32.86** | **73.88** | **47.55** | **71.98** | **14.58** |

Llama3.2-3B shows the same trend but overall lower performance: CF loc for 8-shot metric-specific is 31.61 (vs. 47.55 for 8B).

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Model Scale (3B vs. 8B) | 8B comprehensively outperforms 3B, with larger gaps observed in loc/port queries. |
| Dataset Differences | WFD portability shows the lowest performance (involving second-order temporal knowledge chain reasoning). |
| Exemplar Strategy | 8-shot metric-specific > 8-shot mixed > one-shot > zero-shot |
| Script Type | Latin-script languages >> non-Latin-script languages (independent of linguistic family). |
| Linguistic Attribute Correlation | Syntactic similarity is positively correlated with $p < 0.05$, phonological similarity is positively correlated, and language family shows no significant correlation. |
| One-shot on Locality | **Intrinsically harmful**—random exemplars can mislead the model when they do not match the target query type. |

### Key Findings

- **Exemplar Quality >> Exemplar Quantity**: The improvement of 8-shot metric-specific on loc and port far exceeds that of 8-shot mixed, demonstrating that targeted matching is crucial.
- **Script Type is the Decisive Factor for Cross-Lingual KE**: Non-Latin-script languages (regardless of whether they belong to the Indo-European family) consistently perform worse than Latin-script languages, while the impact of language families is insignificant.
- **Language Confusion**: In non-Latin-script languages, the model frequently responds in English (even when instructions require the target language), which is the direct cause of the poor performance of non-Latin languages.
- **One-shot Can Be Harmful**: For locality queries, a single mismatched random exemplar degrades performance—exemplar strategies need alignment with query types.
- **Portability is the Biggest Bottleneck**: Port queries perform the poorest across all configurations, indicating that cross-lingual transfer of knowledge reasoning is a major vulnerability of current LLMs.

## Highlights & Insights

- Most comprehensive multilingual knowledge editing benchmark to date: 53 languages $\times$ 3 KE datasets in a unified format, covering a complete spectrum from standard to counterfactual and temporal scenarios.
- The "Script Type > Language Family" finding offers new linguistic insights—the weakness of non-Latin scripts is a representation issue rather than a language family issue.
- Systematic analysis of the language confusion phenomenon (models responding in English to non-Latin queries) provides a crucial reference for multilingual LLM research.
- The design of the metric-specific exemplar strategy is simple yet highly effective, offering a "quality-first" rule of thumb for ICL research.

## Limitations & Future Work

- Only two models (Llama3.2-3B and Llama3.1-8B) were evaluated, excluding larger $70\text{B}+$ scales or proprietary models like GPT-4.
- Multilingual translation relies on GPT-4o, which may introduce systematic translation bias (especially in low-resource languages).
- No direct comparison is made with gradient-based methods (e.g., ROME/MEMIT), making it difficult to evaluate the relative competitiveness of ICL-based methods.
- Portability queries in WikiFactDiff are constructed based on automated knowledge graph reasoning, which may introduce noise.
- Focusing solely on factual knowledge editing, the study does not cover the editing of reasoning rules or commonsense knowledge.

## Related Work & Insights

- **vs ROME/MEMIT**: Gradient-based methods modify specific parameters, which is computationally expensive and inapplicable to closed-source models. The ICL-based method in this work introduces zero parameter updates but exhibits limited cross-lingual transfer capabilities.
- **vs ReMaKE**: ReMaKE leverages retrieval-augmented generation for cross-lingual KE, but only targets batch editing scenarios, whereas this work covers a broader range of editing settings.
- **vs Beniwal et al. (EACL 2024)**: Prior cross-lingual KE works cover limited languages. This work scales to 53 languages and systematically analyzes the impact of legislative linguistic properties.
- **vs Multilingual ICL Studies**: Lai et al. (2023) study the multilingual capabilities of English-centric LLMs, whereas this work focuses on cross-lingual transfer specifically in knowledge-editing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The first multilingual KE benchmark of this scale; the insights on script types are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 53 languages, 3 datasets, 4 setups, and multi-dimensional analysis of linguistic attributes, though with a limited variety of model architectures.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with progressive multi-dimensional analyses and rich diagrams.
- Value: ⭐⭐⭐⭐ Directly contributes to the cross-lingual NLP and knowledge-editing communities, and the benchmark itself is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [\[CVPR 2025\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2025/knowledge_editing/mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)
- [\[ACL 2026\] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](../../ACL2026/knowledge_editing/hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)
- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](../../ICML2026/knowledge_editing/do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)

</div>

<!-- RELATED:END -->
