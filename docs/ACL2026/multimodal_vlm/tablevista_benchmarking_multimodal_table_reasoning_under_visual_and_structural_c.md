---
title: >-
  [Paper Note] TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity
description: >-
  [ACL 2026 Findings][Multimodal VLM][Multimodal table reasoning] TableVista constructs a multimodal table benchmark consisting of 3,000 high-quality table reasoning questions expanded into 30…
tags:
  - "ACL 2026 Findings"
  - "Multimodal VLM"
  - "Multimodal table reasoning"
  - "visual robustness"
  - "structural complexity"
  - "Vision-only"
  - "CoT"
date: 2026-05-08
content_hash: ef6e43867be506ac
---

# TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.05955](https://arxiv.org/abs/2605.05955)  
**Code**: https://github.com/FlowRays/TableVista  
**Area**: Multimodal VLM / Table Reasoning  
**Keywords**: Multimodal table reasoning, visual robustness, structural complexity, Vision-only, CoT

## TL;DR
TableVista constructs a multimodal table benchmark consisting of 3,000 high-quality table reasoning questions expanded into 30,000 visual samples. Systematic evaluation of 29 foundation models reveals that while models are relatively stable to style changes, they degrade significantly under complex structures, cross-table reasoning, visual fragmentation, and vision-only inputs.

## Background & Motivation
**Background**: Table Question Answering (QA) has long relied on text serialization inputs like Markdown, HTML, or CSV, which are suitable for standard grids and simple lookups. however, real-world tables often appear as screenshots, web pages, paper PDFs, Excel sheets, or mobile photos, containing multi-level headers, merged cells, long tables, multiple related tables, and contextual descriptions.

**Limitations of Prior Work**: Text serialization flattens spatial structures into token sequences, leading to the loss of visual structural information during conversion. Existing multimodal table benchmarks often use single, idealized, or fixed rendering methods, failing to test whether models maintain consistent reasoning across variations in fonts, layouts, noise, truncation, omissions, and photographic artifacts.

**Key Challenge**: Multimodal models appear to possess OCR and image understanding capabilities, but true table reasoning requires the coupling of "visual localization + structural alignment + multi-step calculation." A model's ability to read cell text does not guarantee it can maintain row-column relationships, cross-table references, and multi-hop logic within complex visual layouts.

**Goal**: The authors aim to construct a table reasoning evaluation that incorporates both structural complexity and visual perturbations, forcing models to answer the same types of reasoning questions across diverse visual presentations to analyze whether current frontier models fail due to visual recognition, structural understanding, or reasoning calculation.

**Key Insight**: Instead of synthesizing simple tables from scratch, TableVista aggregates samples from 14 public table reasoning data sources, re-annotates them with expert and GPT-5 assistance, and expands each question into 10 visual versions using multi-style rendering and visual transformations.

**Core Idea**: Tables are redefined as "visual document objects" rather than "structured text inputs," using structural complexity and visual perturbations to simultaneously examine whether multimodal models truly understand tables.

## Method
The method section of TableVista primarily focuses on benchmark construction. It first establishes a high-quality textual base set to ensure the questions themselves possess sufficient reasoning difficulty; then, each table question is rendered into images across various real-world scenarios; finally, human audits ensure the questions remain answerable after visual transformations. This results in a multi-dimensional stress test matrix rather than a single dataset.

### Overall Architecture
The input consists of original records from 14 table datasets, including WTQ, HiTab, TabFact, MMQA, and FinQA. Each record is standardized into four parts: table, textual context, question, and answer, and tagged with structural attributes, information richness, reasoning skills, and reasoning steps. After filtering and enhancement, 3,000 high-quality QA pairs are retained.

The output comprises 30,000 multimodal samples. Each base sample is expanded into 10 visual versions: 4 scenario styles, 4 robustness perturbations, and 2 vision-only settings. Model evaluation considers both structural types/difficulty and performance fluctuations caused by changes in visual presentation.

### Key Designs
1. **Dual-Dimension Filtering for the Base Set**:
	- **Function**: Ensures each question covers realistic table structures and is not a simple lookup task.
	- **Mechanism**: The authors used GPT-5 to assist in labeling candidate samples with four types of attributes: table layout attributes, informativeness (1-5), scores for four skill categories (lookup, aggregation, numerical, logical), and reasoning steps. Quotas were set for five structural categories: Simple Structure, Text-Mixed, Complex Structure, Long Tables, and Multi-Table, prioritizing samples with high information content, high skill scores, and more reasoning steps.
	- **Design Motivation**: If only visual transformations are applied to simple questions, models might pass using only OCR and local lookups. Incorporating structural complexity and reasoning difficulty simultaneously allows for a true test of visual table reasoning capabilities.

2. **Multi-Style Visual Rendering and Robustness Perturbations**:
	- **Function**: Places the same table question into various realistic visual environments to test model stability.
	- **Mechanism**: Scenario styles include Web, LaTeX, Excel, and Customized. Web simulates Wikipedia/HTML tables, LaTeX simulates paper layouts, Excel simulates spreadsheet interfaces, and Customized samples fonts and color schemes from multiple themes. Robustness perturbations include Noise, Structural Noise, Partial, and Missing; "Partial" crops the table into discontinuous blocks along structural boundaries, while "Missing" masks arbitrary cells while ensuring remaining information is sufficient to answer.
	- **Design Motivation**: Real users do not always provide clean HTML tables. Mobile screenshots, partial occlusions, faded grid lines, and layout fragmentation destroy spatial cues, which are the signals table reasoning depends on most.

3. **Vision-only Settings and Human Quality Audit**:
	- **Function**: Tests whether models can simultaneously read the question, read the table, and complete reasoning from a single image while ensuring data validity.
	- **Mechanism**: "Screen Capture" renders the question, context, and table into a unified interface; "Simulated Photo" overlays camera artifacts like moiré patterns and perspective distortion on top of this. Twelve expert annotators participated in attribute labeling, QA enhancement, visual rendering audits, and quality checks. Each sample was manually inspected at least once, with 10% of samples undergoing double-blind cross-validation.
	- **Design Motivation**: Vision-only settings closer resemble user scenarios like taking a photo or screenshot of a question; however, visual transformations may render a question unanswerable, necessitating human audits and re-generation mechanisms to ensure the benchmark does not penalize models for data errors.

### Loss & Training
This paper does not involve model training but utilizes an evaluation protocol. Main experiments are conducted under a direct-output prompt without thinking mode enabled; metrics primarily use normalized exact match (EM), with GPT-5-mini serving as a secondary judge for answers that fail EM but are semantically equivalent. The authors also specifically compare direct-output with CoT: CoT uses step-by-step prompts or the model's native thinking mode, requiring the final output in the form of `<answer>...</answer>`.

## Key Experimental Results

### Main Results
The TableVista data scale includes 3,000 QA pairs, 4,449 tables, and 30,000 visual samples. The structural distribution is Simple (300), Text-Mixed (300), Complex (1,000), Long (700), and Multi-Table (700); questions average 26.2 words, answers average 1.4 words, and tables average 15.3 rows by 6.6 columns. Hard questions average 6.9 reasoning steps and a skill score of 15.1.

| Model | Simple | Text-Mixed | Complex | Long | Multi | Easy | Medium | Hard | Overall |
|------|--------|------------|---------|------|-------|------|--------|------|---------|
| GPT-5.4 | 73.0 | 86.7 | 81.7 | 68.9 | 61.3 | 93.6 | 80.1 | 47.0 | 73.6 |
| GPT-5.4-mini | 52.0 | 61.0 | 59.3 | 48.7 | 40.0 | 64.0 | 56.1 | 35.2 | 51.8 |
| Qwen2.5-VL-72B | 52.3 | 58.3 | 59.7 | 52.1 | 53.1 | 90.2 | 54.3 | 22.1 | 55.5 |
| Gemma-4-31B-it | 57.3 | 54.0 | 57.6 | 54.4 | 52.3 | 88.2 | 55.6 | 21.9 | 55.2 |
| Llama-4-Maverick | 55.3 | 55.7 | 55.9 | 52.3 | 52.4 | 84.4 | 53.7 | 24.4 | 54.2 |
| Qwen3-VL-8B | 40.7 | 44.0 | 44.1 | 41.9 | 39.9 | 76.7 | 37.3 | 12.7 | 42.2 |
| Table-LLaVA-v1.5-7B | 11.0 | 11.0 | 7.8 | 9.4 | 9.3 | 16.7 | 6.8 | 4.0 | 9.2 |

### Ablation Study
The paper lacks an ablation for training modules but provides critical comparisons for visual conditions and prompting methods. The table below shows overall model performance under different visual presentations.

| Model | Web | LaTeX | Excel | Custom | Noise | Structural | Partial | Missing | Screenshot | Photo | Avg. |
|------|-----|-------|-------|--------|-------|------------|---------|---------|------------|-------|------|
| GPT-5.4 | 73.6 | 72.2 | 71.9 | 72.0 | 70.8 | 70.4 | 68.8 | 84.8 | 69.4 | 67.3 | 72.1 |
| GPT-5.4-mini | 51.8 | 49.9 | 50.1 | 51.1 | 49.5 | 48.4 | 46.8 | 66.9 | 42.0 | 37.7 | 49.4 |
| Qwen2.5-VL-72B | 55.5 | 54.5 | 54.5 | 55.0 | 51.1 | 54.4 | 50.5 | 71.3 | 57.4 | 54.0 | 55.8 |
| Llama-4-Maverick | 54.2 | 53.2 | 53.9 | 52.7 | 53.5 | 54.4 | 52.9 | 66.9 | 53.4 | 51.8 | 54.7 |
| Qwen3-VL-8B | 42.2 | 41.8 | 41.7 | 42.0 | 41.4 | 41.9 | 39.2 | 57.7 | 45.3 | 44.6 | 43.8 |
| LLaVA-v1.5-7B | 6.4 | 6.2 | 6.2 | 7.0 | 6.9 | 6.7 | 6.4 | 10.4 | 0.5 | 0.4 | 5.7 |

| Model | Direct-output | CoT | Gain |
|------|---------------|-----|------|
| GPT-5.4 | 72.1 | 95.6 | +23.5 |
| GPT-5.4-mini | 49.4 | 91.5 | +42.1 |
| Qwen3.5-27B | 51.4 | 96.2 | +44.8 |
| Gemma-4-31B-it | 54.3 | 86.1 | +31.8 |
| Qwen3-VL-8B | 43.8 | 86.0 | +42.2 |

### Key Findings
- Style variation is not the primary bottleneck: scores across Web, LaTeX, Excel, and Custom are very close, indicating models have generalized well to fonts and themes.
- Partial and Photo are more difficult; the former destroys continuous spatial structures, while the latter introduces camera-style degradation. This suggests failures stem from spatial alignment rather than simple OCR.
- "Missing" often improves scores—for example, GPT-5.4 rose from Web 73.6 to Missing 84.8, and Qwen2.5-VL-72B from 55.5 to 71.3—likely because masking reduces noise and focuses the model on critical cells.
- CoT significantly narrows the gap between models; Qwen3-VL-8B jumped from 43.8 to 86.0, indicating many models possess reasoning capabilities but fail to internalize multi-step calculations into a single output under the direct-output setting.
- Among error types, Table Understanding accounts for 54%, Reasoning & Calculation 29%, and Visual Perception only 12%. Within these, Spatial Alignment (32%) and Structure Parsing (22%) dominate, reinforcing that the core bottleneck is structural alignment.

## Highlights & Insights
- The contribution of TableVista is not merely adding another table QA set, but centering evaluation on "semantic consistency across multiple visual forms." This exposes real-world deployment issues better than accuracy on a single clean table.
- The comparison between "Partial" and "Missing" is insightful: fragmentation causes models to lose spatial continuity, while omission/masking can actually improve performance. This suggests future training should focus on robust reasoning between information compression and spatial recovery rather than just more clear images.
- CoT results show that direct-output is a rigorous but meaningful stress test. It examines whether the model has internalized multi-step reasoning, whereas CoT acts like an external scratchpad; both should be reported simultaneously.
- This construction process can be transferred to evaluating financial reports, medical reports, experimental tables, and administrative forms: first control structural complexity, then systematically introduce realistic visual perturbations, and finally use error distributions to locate model weaknesses.

## Limitations & Future Work
- TableVista is an evaluation benchmark and does not provide training methods to directly improve model robustness; it identifies problems without providing model designs to solve spatial alignment failures.
- The data is centered on tables; real documents contain a mix of charts, natural images, flowcharts, footnotes, and formulas, requiring broader cross-modal document reasoning.
- Main experiments use GPT-5-mini as a semantic judge for EM correction; while reliable for numerical and short answers, it may be insufficient for open-ended table interpretations.
- "Simulated Photo" in vision-only settings uses synthetic artifacts, which still differ from real-world noise like actual mobile photography, compression, reflections, or handwritten annotations.

## Related Work & Insights
- **vs TableVQA-Bench / MMTabQA**: These benchmarks have introduced visual tables, but coverage of structural complexity and visual robustness is limited. TableVista covers hierarchy, long tables, multi-tables, scenario styles, perturbations, and vision-only settings.
- **vs MMTab / MMTBench**: These focus on multimodal table understanding and complex content. TableVista emphasizes multiple visual variants of the same base sample to assess consistency and robustness.
- **vs TABLET**: TABLET emphasizes large-scale robust tables rendered from raw web pages. TableVista systematically controls structural types and visual transformations for easier decomposition analysis.
- **Insights for VLM Training**: Future table models need to explicitly learn row-column alignment, cross-block relationship recovery, and sub-cell level numerical differentiation, rather than just expanding OCR data or performing general VQA instruction fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Integrates structural complexity and visual robustness into a complete table reasoning benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Analyzes 29 models across structure, difficulty, visual conditions, CoT, and error types; the data scale and dimensionality are solid.
- Writing Quality: ⭐⭐⭐⭐☆ The construction process is clear, and although the tables are dense, they are highly informative. Layout in the HTML version occasionally affects readability.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for multimodal document understanding, table VQA, visual RAG, and enterprise form automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[CVPR 2026\] Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality](../../CVPR2026/multimodal_vlm/benchmarking_vision-language_models_under_contradictory_virtual_content_attacks_.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)

</div>

<!-- RELATED:END -->
