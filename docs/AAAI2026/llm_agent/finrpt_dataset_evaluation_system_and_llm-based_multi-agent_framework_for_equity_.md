---
title: >-
  [Paper Note] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation
description: >-
  [AAAI 2026][LLM Agent][Equity research report generation] This paper is the first to systematically define the task of automated Equity Research Report (ERR) generation. It constructs the FinRpt dataset (6…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Equity research report generation"
  - "multi-agent framework"
  - "financial dataset"
  - "evaluation system"
  - "report enhancement"
date: 2026-05-08
content_hash: 62fc03adaddc1514
---

# FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation

**Conference**: AAAI 2026
**arXiv**: [2511.07322](https://arxiv.org/abs/2511.07322)  
**Code**: [https://github.com/jinsong8/FinRpt](https://github.com/jinsong8/FinRpt)  
**Area**: LLM Agent / Financial NLP
**Keywords**: Equity research report generation, multi-agent framework, financial dataset, evaluation system, report enhancement

## TL;DR
This paper is the first to systematically define the task of automated Equity Research Report (ERR) generation. It constructs the FinRpt dataset (6,825 high-quality bilingual reports integrating 7 categories of financial data), proposes an 11-metric evaluation framework, and designs the FinRpt-Gen generation framework with 9 collaborative agents featuring a three-stage enhancement pipeline (rating correction / expert review / language polishing). Human evaluation shows that generated reports approach expert-written quality.

## Background & Motivation

**Background**: LLMs have been successfully applied to financial sentiment analysis, question answering, and related tasks, but the automatic generation of complete equity research reports—structured, in-depth analytical documents spanning thousands of words—has not yet been systematically studied.

**Limitations of Prior Work**:
- Lack of high-quality ERR datasets — existing financial datasets focus on summaries or short texts and do not include structured long-form reports.
- Lack of ERR-specific evaluation metrics — general NLG metrics (BLEU/ROUGE) cannot measure the accuracy of financial analysis.
- ERRs comprise 6 key sections (financial analysis / news analysis / management assessment / risk analysis / investment evaluation / overall rating), and errors in any one section can mislead investors.

**Key Challenge**: ERR generation requires integrating multiple heterogeneous financial data sources (time-series stock prices, financial statements, announcements, news, market indices), and the output must be accurate, comprehensive, and actionable — far beyond standard text generation.

**Goal**: Define the ERR generation task + construct a dataset and evaluation framework + design a multi-agent generation framework.

**Key Insight**: Decompose report generation into a collaboration of 9 specialized agents, each responsible for a distinct data source or analytical dimension.

**Core Idea**: 7 categories of financial data + 9-agent division of labor + three-stage enhancement = automated ERR generation approaching human expert quality.

## Method

### Overall Architecture
Company ticker + research date → automatic collection of 7 data categories (company information / financial indicators / announcements / news / stock prices / market indices / macro data) → 9 agents process data independently → initial report → three-stage enhancement (rating correction → expert review → language polishing) → final research report.

### Key Designs

1. **Dataset Construction Pipeline**:

    - Function: Automatically generate high-quality ERR training data.
    - Mechanism: Collect data from public financial APIs → LLM initial generation → three-stage enhancement: rating correction (cross-referenced against market trends), expert-style review, and language polishing.
    - Scale: 6,825 bilingual reports (Chinese and English); human expert scores approach those of real research reports.

2. **11-Metric Evaluation Framework**:

    - Function: Comprehensively assess the quality of generated reports.
    - Coverage: Content accuracy, logical coherence, correctness of data citations, depth of industry analysis, comprehensiveness of risk assessment, actionability of investment recommendations, language professionalism, format compliance, information timeliness, peer comparison quality, and rating reasonableness.
    - Design Motivation: BLEU/ROUGE are nearly meaningless in financial contexts.

3. **9-Agent Collaborative Framework (FinRpt-Gen)**:

    - Function: Collaboratively generate complete research reports through division of labor.
    - Agent Roles: Data collection / financial analysis / news analysis / industry comparison / risk assessment / management evaluation / investment rating / report integration / quality review.
    - Design Motivation: A single LLM cannot simultaneously handle time-series data analysis, text comprehension, and investment reasoning.

### Loss & Training
- The dataset can be used for SFT fine-tuning.
- Evaluation employs LLM-as-judge combined with human financial analyst scoring.

## Key Experimental Results

### Main Results

| Metric | FinRpt Generated | Expert Written | Note |
|--------|-----------------|----------------|------|
| Content Accuracy | 4.1/5 | 4.3/5 | Approaches expert level |
| Logical Coherence | 4.0/5 | 4.2/5 | Approaches expert level |
| Fleiss Kappa | 0.67–0.82 | — | Good inter-annotator agreement |

### Ablation Study: Enhancement Stages

| Configuration | Quality Score |
|---------------|--------------|
| Initial generation | 3.5/5 |
| + Rating correction | 3.8/5 |
| + Expert review | 4.0/5 |
| **+ Polishing (full pipeline)** | **4.1/5** |

### Key Findings
- All three enhancement stages are indispensable; cumulatively improving scores from 3.5 to 4.1.
- Dataset quality approaches expert-written reports (Kappa 0.67–0.82).
- The financial analysis agent is the most critical component.

## Highlights & Insights
- First work to define a complete framework for the ERR generation task — covering input format, output format, and evaluation system.
- The three-stage enhancement pipeline is key to data quality.
- The 9-agent framework is generalizable to other structured long-form documents (audit reports, ESG reports).

## Limitations & Future Work
- Rating correction relies on ex-post market data (not causal prediction).
- The investment return value of generated reports has not been validated in real investment decision-making settings.
- Financial data timeliness — training data may become stale by deployment time.
- 6,825 reports may be insufficient for fine-tuning large models.
- Orchestration overhead and error cascade propagation across the 9-agent system are not analyzed.

## Related Work & Insights
- **vs. BloombergGPT**: Focuses on financial pre-training but does not perform report generation.
- **vs. Financial summarization**: Operates only at paragraph level. FinRpt targets complete multi-section reports.
- Makes a methodological contribution to financial AI.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define ERR generation + dataset + evaluation + method.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6,825 reports, 11 metrics, human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Task definition is systematic and complete.
- Value: ⭐⭐⭐⭐ Direct practical value for financial NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](../../ACL2026/llm_agent/yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[AAAI 2026\] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)

</div>

<!-- RELATED:END -->
