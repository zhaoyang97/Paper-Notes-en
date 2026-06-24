---
title: >-
  [Paper Note] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][event extraction] Introduces Lemonade—a large-scale multilingual expert-annotated event dataset based on ACLED conflict data (39,786 events, 20 languages, 171 countries, 10,707 entities). It proposes a new task paradigm, Abstractive Event Extraction (AEE), where event arguments are not limited to text spans but are normalized into numerical, categorical, or entity values. The accompanying zero-shot entity linking syste…
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "event extraction"
  - "multilingual"
  - "entity linking"
  - "abstractive"
  - "conflict data"
date: 2026-05-08
content_hash: 253b618796ba3363
---

# LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.00980](https://arxiv.org/abs/2506.00980)  
**Code**: [GitHub](https://github.com/stanford-oval/Lemonade)  
**Area**: Multilingual Translation  
**Keywords**: event extraction, multilingual, entity linking, abstractive, conflict data

## TL;DR
Introduces Lemonade—a large-scale multilingual expert-annotated event dataset based on ACLED conflict data (39,786 events, 20 languages, 171 countries, 10,707 entities). It proposes a new task paradigm, Abstractive Event Extraction (AEE), where event arguments are not limited to text spans but are normalized into numerical, categorical, or entity values. The accompanying zero-shot entity linking system, Zest, achieves an F1 score of 45.7% on the AEL subtask, significantly outperforming the baseline of 23.7%.

## Background & Motivation

**Background**: Event extraction (EE) extracts structured event information from unstructured texts and is a core task in NLP. Existing datasets (ACE05, DocEE) are primarily in English/Chinese, based on span annotations, with varying quality of crowd-sourced annotations.

**Limitations of Prior Work**: (a) Lack of multilingual coverage—global conflict analysis requires covering the Global South and multilingual sources; (b) Insufficient entity registry coverage—Wikipedia/Wikidata lack regional political entities; (c) Span-based EE is unsuitable for aggregate analysis—boolean or numerical information such as "Is violence targeted against women?" are not necessarily text spans; (d) High-stakes scenarios (humanitarian decision-making) require expert-level annotation quality.

**Key Challenge**: The design assumption of traditional span-based EE (arguments = text spans) limits the application of event data in global aggregate analysis.

**Goal**: Define the Abstractive EE task (normalizing event arguments into categorical/numerical/entity values) + construct the first large-scale multilingual expert-annotated dataset + establish baseline systems.

**Key Insight**: Utilize over a decade of expert-annotated global conflict data from ACLED, cleaning and re-annotating them into NLP-usable formats.

**Core Idea**: AEE removes the constraint that "arguments must be text spans," directly outputting normalized values: boolean (targeted at women), enumeration (event type), entity ID (linking participants to a database), and numerical (casualty count).

## Method

### AEE Task Formulation
Given a codebook $C = (T, \mathcal{D}, S)$ (event type set $T$, domain set $\mathcal{D}$, event signature $S$) and text $w$, extract $(t_i, v_1, \ldots, v_{n_i})$, where $v_j \in D_{i,j}$ can be an integer, string, boolean, or set of entities.

Three subtasks:
- **ED** (Event Detection): Identify event types.
- **AEAE** (Abstractive Event Argument Extraction): Extract non-entity arguments (numerical, boolean, enumeration).
- **AEL** (Abstractive Entity Linking): Link event participants in the text to an entity database.

### Dataset Construction
- Based on 344,116 events from ACLED (Jan 2024 to Jan 2025), yielding 39,786 events after filtering and cleaning.
- Multi-round review and annotation by 200+ regional experts (not crowd-sourced).
- Re-annotation: location parameters, entity description generation (writing retrieval descriptions for 10,707 entities).
- Final coverage of 25 event types (ranging from peaceful protests to chemical weapon deployment).

### Zest Zero-Shot Entity Linking System
- **Function**: Link event participants to a database of 10,707 entities without training data.
- **Mechanism**: Retrieve candidate entities (based on semantic similarity of entity descriptions) $\to$ LLM reranking/selection.
- vs OneNet (SOTA zero-shot EL): Zest F1=45.7% vs OneNet F1=23.7% (+22%).

## Key Experimental Results

### End-to-End AEE (Zero-shot)

| System | ED F1 | AEAE F1 | AEL F1 | End-to-End F1 |
|------|-------|---------|--------|---------------|
| GoLLIE | 45.2 | — | — | 41.6 |
| GPT-4o | 62.1 | — | — | 55.8 |
| **Best zero-shot** | — | — | — | **58.3** |
| Best supervised | — | — | — | **78.4** |

### AEL Subtask (Zero-shot)

| System | F1 |
|------|----|
| OneNet (SOTA baseline) | 23.7 |
| **Zest (Ours)** | **45.7** |

### Ablation: Language Coverage

| Language Group | Event Count | Description |
|--------|--------|------|
| English | ~15,000 | Most |
| Spanish | ~5,000 | |
| Arabic | ~4,000 | |
| Burmese/Somali/Nepali | ~500-1000 | First time included in an EE dataset |

### Key Findings
- **Huge gap between zero-shot and supervised**: End-to-end F1 differs by 20.1%, and AEL differs by 37.0%—indicating the task is extremely challenging.
- **LLMs outperform specialized EE models**: GPT-4o outperforms specialized EE models like GoLLIE in the zero-shot setting.
- **Entity linking is the biggest bottleneck**: The zero-shot performance of AEL is significantly lower than that of other subtasks—many of the 10,707 entities lack Wikipedia entries.
- **Zest's retrieve-and-rerank approach is effective**: Compared to OneNet's pipeline, Zest's retrieval strategy is more suitable for large-scale entity registries.
- **Multilingual challenges**: Performance on low-resource languages (Burmese, Somali) is significantly lower than on high-resource languages.

## Highlights & Insights
- **Paradigm shift from span-based to abstractive**: AEE removes the constraint of "arguments must be text fragments," making event data directly aggregatable and analyzable (e.g., "total violent casualties in 2024"), which is more practical for policy makers.
- **Quality gap between expert and crowd-sourced annotations**: Multi-round reviews by 200+ regional experts guarantee the annotation quality required for high-stakes scenarios, which crowd-sourcing cannot replace.
- **Coverage of 10,707 tail entities**: Includes many regional political entities without Wikipedia entries (such as Syrian militias), challenging the assumption that LLMs rely on memorized entities.
- **20 languages + 171 countries**: Far exceeds the language and geographic coverage of existing EE datasets.

## Limitations & Future Work
- **Single event / single document**: Only the main event is annotated per document; multi-event co-occurrence is not supported.
- **ACLED dependency**: Dataset quality is affected by ACLED's annotation strategies, potentially introducing systematic biases.
- **Large gap between zero-shot and supervised**: Suggests current LLMs lack understanding of domain-specific entities.
- **Event types limited to the conflict domain**: 25 types (related to violence/protests), not covering fields like economy or natural disasters.

## Related Work & Insights
- **vs ACE05 (Walker et al., 2006)**: ACE05 is the standard for span-based sentence-level EE, whereas Lemonade is abstractive document-level EE—a paradigm upgrade.
- **vs DocEE (Tong et al., 2022)**: DocEE scales to document-level but remains span-based, whereas Lemonade goes a step further towards abstractive.
- **vs ZESHEL (Logeswaran et al., 2019)**: ZESHEL is a zero-shot EL benchmark, but its entities still have Wikipedia descriptions; Lemonade entities are much further in the tail.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The AEE task paradigm is brand new, and the dataset size and coverage are unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Zero-shot + supervised + multi-system comparisons + subtask decomposition analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous task definitions (Definition 3.1/3.2), and intuitive examples in Figure 1.
- Value: ⭐⭐⭐⭐⭐ Directly contributes to global conflict analysis and humanitarian applications; high long-term value for the dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring In-Image Machine Translation with Real-World Background](exploring_in-image_machine_translation_with_real-world_background.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization](beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)

</div>

<!-- RELATED:END -->
