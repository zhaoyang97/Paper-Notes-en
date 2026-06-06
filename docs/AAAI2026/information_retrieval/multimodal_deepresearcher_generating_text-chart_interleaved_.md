---
title: >-
  [Paper Note] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework
description: >-
  [AAAI 2026 (Oral)][Information Retrieval & RAG][Deep Research Agent] This paper proposes Multimodal DeepResearcher, a four-stage agentic framework for generating text-chart interleaved research reports from scratch. It i…
tags:
  - "AAAI 2026 (Oral)"
  - "Information Retrieval & RAG"
  - "Deep Research Agent"
  - "Text-Chart Interleaved Reports"
  - "Formal Description of Visualization (FDV)"
  - "D3.js Chart Generation"
  - "Actor-Critic Chart Refinement"
date: 2026-05-08
content_hash: 4f91cc0c5193f593
---

# Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2506.02454](https://arxiv.org/abs/2506.02454)  
**Code**: [https://github.com/rickyang1114/multimodal-deepresearcher](https://github.com/rickyang1114/multimodal-deepresearcher)  
**Area**: Information Retrieval
**Keywords**: Deep Research Agent, Text-Chart Interleaved Reports, Formal Description of Visualization (FDV), D3.js Chart Generation, Actor-Critic Chart Refinement

## TL;DR
This paper proposes Multimodal DeepResearcher, a four-stage agentic framework for generating text-chart interleaved research reports from scratch. It introduces Formal Description of Visualization (FDV) to enable LLMs to learn and produce diverse charts, and employs an Actor-Critic iterative refinement mechanism (LLM generates D3.js code → browser rendering → multimodal LLM review). The system achieves an 82% overall win rate (Claude 3.7) on the newly constructed MultimodalReportBench and a 100% win rate in human evaluation.

## Background & Motivation

**Background**: Deep research frameworks (e.g., Deep Research) have enabled LLMs to generate comprehensive text reports through iterative retrieval and reasoning, but their outputs remain purely textual—lacking charts, visualizations, and other multimodal content.

**Limitations of Prior Work**: Real-world research reports extensively use charts to convey information efficiently. However, automatically generating text-chart interleaved reports faces two major challenges: (a) how to design informative and diverse visualizations—LLMs lack systematic understanding of chart design; and (b) how to organically integrate visualizations with text—charts should not be decorative but closely tied to the textual content.

**Key Challenge**: LLMs are inherently text-oriented, whereas visualizations are visual objects. A bridge is needed that allows LLMs both to "understand" existing high-quality charts and to "generate" new, diverse ones.

**Key Insight**: Inspired by Wilkinson's Grammar of Graphics, the authors design FDV (Formal Description of Visualization)—a structured textual format that represents charts along four dimensions: layout, scale, data, and visual marks—enabling LLMs to learn chart design patterns via in-context learning.

**Core Idea**: Textualize charts via FDV, combined with a four-stage agentic pipeline (search → exemplar textualization → planning → text-chart generation) and Actor-Critic rendering refinement, to generate high-quality text-chart interleaved research reports from scratch.

## Method

### Overall Architecture
A four-stage pipeline:
1. **Researching**: Iteratively retrieves and synthesizes relevant information.
2. **Exemplar Report Textualization**: Converts charts in human-authored exemplar reports into FDV textual representations.
3. **Planning**: Generates a content outline and a visualization style guide.
4. **Multimodal Report Generation**: Generates text, chart code, and iteratively refines the output.

### Key Designs

1. **FDV (Formal Description of Visualization)**:

    - **Function**: Represents visualization charts as structured textual descriptions along four dimensions.
    - **Mechanism**: **Layout** (subplot arrangement, spatial positioning, margins, title placement) + **Scale** (mapping logic from data to visual channels) + **Data** (numerical values, textual elements, labels, legends) + **Marks** (visual element design: fonts, colors, and interactions for bars/lines/points).
    - **Design Motivation**: Unstructured natural language cannot precisely describe all details of complex visualizations (e.g., multi-subplot dashboard layouts). FDV provides a structured specification that LLMs can learn from and generate.

2. **Stage 1: Researching**:

    - **Function**: Performs iterative web search and information synthesis for a given topic.
    - **Mechanism**: GPT-4o-mini generates keywords → searches → analyzes retrieved content → synthesizes structured "learning points" → generates new research questions for the next search iteration (progressive deepening).

3. **Stage 2: Exemplar Report Textualization**:

    - **Function**: Converts high-quality human-created multimodal reports into plain text, enabling LLMs to learn text-chart interleaving patterns via in-context learning.
    - **Mechanism**: A multimodal LLM (Claude 3.7) identifies charts in exemplar reports, extracts their FDV representations, and replaces images with textual descriptions. This teaches the LLM "what type of chart should follow what type of textual paragraph."

4. **Stage 4: Actor-Critic Chart Refinement**:

    - **Function**: Iteratively improves the quality of generated visualizations.
    - **Mechanism**: A three-step loop — (a) **Actor** (text LLM): generates D3.js code from FDV specifications; (b) **Browser Tool**: renders the code, captures console errors/warnings, and takes a screenshot; (c) **Critic** (multimodal LLM): inspects the rendered output for visual issues (element overlap, insufficient clarity) and console errors. The loop runs for at most 3 iterations or until the quality check passes.
    - **Design Motivation**: One-shot D3.js generation frequently produces rendering issues (overlap, incorrect proportions). The Actor-Critic loop corrects these through actual rendering feedback.

### Evaluation Framework
MultimodalReportBench: 100 diverse topics (technology, healthcare, education, climate, etc.) evaluated across five dimensions (informativeness, coherence, verifiability, visualization quality, and visualization consistency).

## Key Experimental Results

### Main Results

| Model | Overall Win Rate | Verifiability | Visualization Quality |
|-------|-----------------|---------------|-----------------------|
| Claude 3.7 Sonnet (MDR) | **82%** | 86% | 80% |
| Open-source Model (MDR) | 55% | — | — |
| Human Evaluation (20 topics) | **100%** | — | — |

### Chart Quality Comparison (10-point scale)

| Metric | MDR | Baseline (DataNarrative) |
|--------|-----|--------------------------|
| Layout | **9.23** | 8.48 |
| Aesthetics | **9.12** | 8.38 |

### Visualization Diversity

| Chart Type | MDR | Baseline |
|------------|-----|---------|
| Flowcharts | **15** | 2 |
| Dashboards | **18** | 1 |
| Complex Types Total | **280** | 96 |

### Ablation Study

| Removed Component | Performance Drop |
|-------------------|-----------------|
| Planning (Stage 3) | **85%** |
| Iterative Refinement (Stage 4 Critic) | 80% |
| Exemplar Learning (Stage 2) | 70% |

### Key Findings
- **82% automatic win rate + 100% human win rate**: A clear advantage over the DataNarrative baseline.
- **Substantially improved visualization diversity**: MDR generates 280 complex chart types vs. 96 for the baseline, with particularly large gains in flowcharts (15 vs. 2) and dashboards (18 vs. 1).
- **Planning is the most critical stage**: Removing it leads to an 85% performance drop, highlighting the importance of structured planning for text-chart coordination.
- **High agreement between human and automatic evaluation**: 80% of judgments are consistent, validating the reliability of the automatic evaluation.
- **Efficiency trade-off**: MDR averages 767s vs. 373s for the baseline (~2×), primarily due to the iterative rendering refinement loop.

## Highlights & Insights
- **FDV is the core innovation**: It addresses the fundamental problem of "how LLMs can understand and generate visualizations." By structuring visual design into four dimensions grounded in the Grammar of Graphics, it translates visual design into a text space operable by LLMs.
- **The Actor-Critic rendering loop is highly practical**: Rather than having LLMs generate code from imagination, the system actually renders the output, examines screenshots, and reads error logs to iteratively correct issues—representing genuinely grounded code generation.
- **The AAAI Oral recognition is well-deserved**: The problem addressed has substantial practical value. Research reports, business analyses, and news visualizations all require text-chart interleaved content, and the pure-text output of existing Deep Research systems is a clear limitation.
- **The in-context learning strategy via exemplar textualization is elegant**: No training is required; by converting high-quality examples into FDV text, LLMs learn text-chart interleaving patterns directly.

## Limitations & Future Work
- Element overlap in D3.js charts persists when spatial layouts are complex.
- When retrieved information is insufficient, the LLM may fabricate data for charts—a serious concern in research report generation.
- Generation is slow (~13 minutes per report), with the primary bottleneck in the browser rendering refinement loop.
- The core code is not fully open-sourced due to organizational constraints.
- Evaluation covers English reports only; multilingual support remains unvalidated.

## Related Work & Insights
- **vs. DataNarrative**: DataNarrative generates only simple chart types, whereas MDR produces complex and diverse visualizations (dashboards, mind maps, infographics) via FDV and exemplar learning.
- **vs. Deep Research (OpenAI)**: Deep Research outputs plain-text reports; MDR is the first system to achieve text-chart interleaving.
- **Implications for Agent research**: The combination of agents with browser tools (rendering, screenshot capture, log reading) is a transferable pattern applicable to all scenarios requiring "code execution verification" (e.g., web development, UI design).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ FDV combined with text-chart interleaved report generation constitutes an entirely new task—a pioneering contribution worthy of AAAI Oral recognition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automatic evaluation, human evaluation, ablation studies, and diversity analysis are all included, though only a single baseline is compared.
- Writing Quality: ⭐⭐⭐⭐ The four-stage framework is clearly described, and the FDV design is theoretically grounded.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core limitation of Deep Research's plain-text output, with immediate applicability to research assistants and business analytics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/information_retrieval/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/information_retrieval/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](../../ACL2026/information_retrieval/is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[NeurIPS 2025\] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](../../NeurIPS2025/information_retrieval/the_narrow_gate_localized_imagetext_communication_in_native.md)

</div>

<!-- RELATED:END -->
