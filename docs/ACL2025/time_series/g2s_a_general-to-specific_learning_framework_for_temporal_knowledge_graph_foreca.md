---
title: >-
  [Paper Note] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models
description: >-
  [ACL 2025][Time Series][Temporal Knowledge Graphs] This paper proposes the G2S framework, which decouples general patterns (temporal structural regularities) from scenario-specific information (concrete entities/relations) in temporal knowledge graph (TKG) forecasting. By first learning general patterns on anonymized temporal structures and then injecting scenario-specific information, G2S effectively enhances the generalization capability of LLMs in TKG forecasting.
tags:
  - "ACL 2025"
  - "Time Series"
  - "Temporal Knowledge Graphs"
  - "LLM Fine-Tuning"
  - "Generalization"
  - "Anonymized Temporal Structure"
  - "Knowledge Decoupling"
date: 2026-05-08
content_hash: 3af09e179373c650
---

# G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.00445](https://arxiv.org/abs/2506.00445)  
**Code**: None  
**Area**: Time Series / Knowledge Graphs  
**Keywords**: Temporal Knowledge Graphs, LLM Fine-Tuning, Generalization, Anonymized Temporal Structure, Knowledge Decoupling

## TL;DR
This paper proposes the G2S framework, which decouples general patterns (temporal structural regularities) from scenario-specific information (concrete entities/relations) in temporal knowledge graph (TKG) forecasting. By first learning general patterns on anonymized temporal structures and then injecting scenario-specific information, G2S effectively enhances the generalization capability of LLMs in TKG forecasting.

## Background & Motivation
**Background**: TKG forecasting (predicting future events based on historical facts) is a crucial task in time-sensitive scenarios (finance, healthcare, politics). Traditional methods (e.g., RE-GCN, TLogic) are trained independently on each TKG, resulting in weak generalization capabilities. Recent LLM-based approaches (e.g., GenTKG) improve generalization through fine-tuning, but their effectiveness remains limited.

**Limitations of Prior Work**: Different TKGs describe diverse scenarios (e.g., GDELT for geopolitics, WIKI for sports/academia), characterized by entirely distinct entities and relations. Existing LLM-based methods attempt to learn general temporal patterns and scenario-specific information simultaneously. The interference between these two learning processes restricts generalization.

**Key Challenge**: The entanglement of general patterns (e.g., temporal regularities like "if A does X to B, A might do Y to C later") and scenario-specific information (e.g., concrete entity/relation names like "Trump" and "visit").

**Goal**: How can the learning processes of these two types of knowledge be decoupled?

**Key Insight**: Anonymization — replacing entities, relations, and timestamps with abstract IDs to construct scenario-agnostic temporal structures.

**Core Idea**: First learn general patterns on anonymized temporal structures $\rightarrow$ then inject scenario mappings to learn specific knowledge = a two-stage decoupled learning framework.

## Method

### Overall Architecture
G2S consists of two stages: (1) **General Learning Stage**: Anonymize entities, relations, and timestamps from multiple TKGs into abstract IDs, and fine-tune LLaMA3-8B on these anonymized temporal structures to learn cross-scenario general temporal patterns; (2) **Specific Learning Stage**: Keep the anonymized structures while providing ID-to-entity/relation name mapping tables on the target TKG, injecting scenario-specific information through In-Context Learning (ICL) or Supervised Fine-Tuning (SFT).

### Key Designs

1. **Anonymized Temporal Structure Transformation**:

    - **Function**: Replace all elements in the $(s, r, o, t)$ quadruplets with abstract IDs.
    - **Mechanism**: Timestamp ID = query time - fact time ($\mathcal{A}(t) = t_q - t$, with the query time set to 0). For entity/relation IDs, three strategies are introduced: FID (frequency-based ordering), GID (global original IDs), and RID (random assignment).
    - **Design Motivation**: Eliminate scenario-specific information, forcing the model to learn general patterns solely from temporal structures.

2. **Comparison of Three Anonymization Strategies**:

    - FID (Frequency ID): Assigns smaller IDs to high-frequency entities, preserving frequency signals.
    - GID (Global ID): Employs the original index from the dataset; indices are inconsistent across datasets.
    - RID (Random ID): Independently and randomly assigns IDs to entities for each query, providing the purest anonymization.
    - Experimental Finding: RID performs best in zero-shot settings (cleanest general pattern), while FID/GID perform better in supervised configurations.

3. **Two Specific Learning Modes**:

    - ICL Mode: No training data is allowed; scenario information is provided in-context solely via the mapping tables from IDs to entity/relation names.
    - SFT Mode: Allows further fine-tuning on the target dataset.
    - Design Motivation: ICL evaluates zero-shot generalization, while SFT tests the model's adaptation capability when training data is available.

### Loss & Training
Standard cross-entropy loss coupled with LoRA fine-tuning on LLaMA3-8B. The general stage is trained using 100k samples from GDELT and 30k samples from WIKI, while the specific stage is fine-tuned on the target dataset.

## Key Experimental Results

### Main Results (Standard Setup H@1)

| Dataset | RE-GCN | TLogic | GenTKG | G2S (SFT+RID) |
|--------|--------|--------|--------|----------------|
| ICEWS14 | 31.3 | 33.2 | 37.3 | **39.1** |
| ICEWS18 | 22.3 | 23.7 | 25.3 | **28.5** |
| YAGO | 78.7 | 82.3 | 83.1 | **86.2** |

### Ablation Study

| Setup (ICEWS14 H@1) | Without G2S | +General Stage | +Specific Stage |
|---------------------|-------|----------|----------|
| Zero-shot ICL | 15.2 | **22.8** | - |
| 5% Low-resource | 28.1 | 33.4 | **35.7** |
| Standard SFT | 37.3 | - | **39.1** |

### Key Findings
- **General stage significantly improves zero-shot performance**: H@1 increases from 15.2% to 22.8% (+7.6 pp), demonstrating that transferable general patterns indeed exist within anonymized structures.
- **RID achieves the best zero-shot performance**: Random numbering captures temporal structures in their purest form, avoiding frequency or indexing biases.
- **Low-resource scenarios benefit the most**: Under a 5% data constraint, G2S increases H@1 from 28.1% to 35.7% (+7.6 pp), approaching standard setup performance.
- **Decoupled learning is highly effective**: Compared with learning both types of knowledge simultaneously, the "general first, specific second" paradigm performs better across all settings.

## Highlights & Insights
- **Insight on Knowledge Decoupling**: The paper identifies the entanglement of general patterns and scenario-specific information as the generalization bottleneck in TKGs, offering anonymization as a simple yet effective decoupling strategy. This "structure first, concretization second" paradigm can be extended to other tasks requiring cross-domain generalization.
- **Relative Timestamp Encoding**: Encoding timestamps as offsets relative to the query time eliminates the bias of inconsistent training/testing horizons, serving as a simple but highly effective design.
- **Impact of Anonymization Strategies**: The comparative study of RID/FID/GID provides valuable guidance for practical applications of TKG with LLMs.

## Limitations & Future Work
- LLMs cannot generate comprehensive rankings for all entities, hindering the computation of certain metrics like MRR.
- Data for the general stage is sourced from only two datasets (GDELT and WIKI); incorporating more diverse TKGs could yield further improvements.
- Anonymization discards semantic information embedded in entity names (e.g., mapping "Trump" to "E5"), potentially losing helpful semantic associations.
- The optimal training configuration for the general stage (e.g., data volume, sampling ratios) remains unexplored.

## Related Work & Insights
- **vs GenTKG**: While GenTKG directly fine-tunes LLMs on the target TKG, G2S introduces an additional general pre-training stage, achieving H@1 of 39.1 vs 37.3 in the standard setup.
- **vs ICL (Lee et al.)**: Traditional ICL methods use large models like GPT-NeoX for zero-shot forecasting, whereas G2S improves performance by over 7 pp under the same setting.
- **vs RE-GCN/TLogic**: Conventional methods lack cross-domain generalization capabilities; G2S outperforms them even in standard setups.

## Rating
- Novelty: ⭐⭐⭐⭐ The decoupling of general patterns and scenario-specific information is a solid insight, supported by a systematic investigation of anonymization strategies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across three settings (zero-shot, low-resource, standard), five datasets, and three anonymization strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with intuitive framework diagrams.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm for TKG forecasting generalization, with decoupling principles that are broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)
- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](../../ACL2026/time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)
- [\[ICML 2026\] Building Social World Models with Large Language Models](../../ICML2026/time_series/building_social_world_models_with_large_language_models.md)
- [\[ICLR 2026\] Semantic-Enhanced Time-Series Forecasting via Large Language Models](../../ICLR2026/time_series/semantic-enhanced_time-series_forecasting_via_large_language_models.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)

</div>

<!-- RELATED:END -->
