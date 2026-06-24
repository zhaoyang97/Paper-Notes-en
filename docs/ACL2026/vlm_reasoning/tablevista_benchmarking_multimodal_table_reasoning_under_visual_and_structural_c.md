---
title: >-
  [Paper Note] TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity
description: >-
  [ACL 2026 Findings][VLM Reasoning][Multimodal Table Reasoning] TableVista constructs a multimodal table benchmark with 3,000 high-quality reasoning questions expanded into 30,000 visual samples. Systematic evaluation of 29 foundation models reveals that while models are relatively stable across style changes, they suffer significant degradation under complex structures, multi-table reasoning, visual fragmentation, and vision-only inputs.
tags:
  - "ACL 2026 Findings"
  - "VLM Reasoning"
  - "Multimodal Table Reasoning"
  - "Visual Robustness"
  - "Structural Complexity"
  - "Vision-only"
  - "CoT"
date: 2026-05-08
content_hash: 57f996ac16e673d9
---

# TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.05955](https://arxiv.org/abs/2605.05955)  
**Code**: https://github.com/FlowRays/TableVista  
**Area**: Multimodal VLM / Table Reasoning  
**Keywords**: Multimodal Table Reasoning, Visual Robustness, Structural Complexity, Vision-only, CoT

## TL;DR
TableVista constructs a multimodal table benchmark with 3,000 high-quality reasoning questions expanded into 30,000 visual samples. Systematic evaluation of 29 foundation models reveals that while models are relatively stable across style changes, they suffer significant degradation under complex structures, multi-table reasoning, visual fragmentation, and vision-only inputs.

## Background & Motivation
**Background**: Table Question Answering (TableQA) has long relied on text serialization such as Markdown, HTML, or CSV, which are suitable for standard grids and simple lookups. However, real-world tables often appear as screenshots, web pages, academic PDFs, Excel sheets, or mobile photos, containing multi-level headers, merged cells, long tables, multiple related tables, and contextual descriptions.

**Limitations of Prior Work**: Text serialization flattens spatial structures into token sequences, causing the loss of significant visual structural information during conversion. Furthermore, existing multimodal table benchmarks often use single, idealized, or fixed rendering methods, failing to test whether models maintain consistent reasoning under variations in font, layout, noise, truncation, missing data, and photographic artifacts.

**Key Challenge**: While multimodal models appear to possess OCR and image understanding capabilities, true table reasoning requires the coupling of "visual localization + structural alignment + multi-step calculation." A model's ability to read cell text does not guarantee it can maintain row-column relationships, cross-table references, and multi-hop logic within complex visual layouts.

**Goal**: The authors aim to construct a table reasoning evaluation that incorporates both structural complexity and visual perturbations, forcing models to answer the same reasoning questions under diverse visual presentations to analyze whether current frontier models fail due to visual recognition, structural understanding, or reasoning computation.

**Key Insight**: Instead of synthesizing simple tables from scratch, TableVista aggregates samples from 14 public table reasoning data sources, re-annotates them with expert and GPT-5 assistance, and expands each question into 10 visual versions using multi-style rendering and visual transformations.

**Core Idea**: Redefine tables from "structured text inputs" to "visual document objects," utilizing structural complexity and visual perturbations to verify if multimodal models truly understand tables.

## Method
The methodology of TableVista primarily focuses on benchmark construction. It first establishes a high-quality textual base set to ensure the questions themselves possess sufficient reasoning difficulty, then renders each table question into various real-world scenario images, and finally ensures the questions remain answerable after visual transformation through human auditing. This results in a multi-dimensional stress testing matrix rather than a single dataset.

### Overall Architecture
The input consists of raw records from 14 table datasets, including WTQ, HiTab, TabFact, MMQA, and FinQA. Each record is standardized into four parts: table, textual context, question, and answer, and labeled with structural attributes, information richness, reasoning skills, and reasoning steps. After filtering and enhancement, 3,000 high-quality QA pairs are retained.

The output comprises 30,000 multimodal samples. Each base sample is expanded into 10 visual versions: 4 scenario styles, 4 robustness perturbations, and 2 vision-only settings. Model evaluation considers performance fluctuations caused by changes in visual presentation as well as structural types and difficulty.

### Key Designs

**1. Base set filtering via structural and reasoning dimensions: Ensuring coverage of real table structures and non-trivial reasoning.**

If visual transformations are applied only to simple questions, models might bypass true reasoning through OCR and local lookups. Therefore, the authors used GPT-5 to label candidate samples across four categories: table layout attributes, information richness from $1$–$5$, four types of skill scores (lookup, aggregation, numerical, logical), and reasoning steps. Sampling was then performed according to quotas across five structures (Simple Structure, Text-Mixed, Complex Structure, Long Tables, Multi-Table).

Priority was given to samples with high information content, high skill scores, and more reasoning steps, resulting in 3,000 refined QA pairs. By embedding both "structural complexity" and "reasoning difficulty" into the base set, the benchmark forces models to expose their weaknesses in structural alignment and multi-step calculation under visual perturbations, rather than merely testing character recognition.

**2. Multi-style visual rendering and robustness perturbations: Testing stability across diversas real-world visual environments.**

Real users do not always provide clean HTML tables; mobile screenshots, partial occlusions, faded grid lines, and layout fragmentation can destroy spatial cues essential for table reasoning. TableVista applies four scenario styles to each base question: Web (simulating Wikipedia/HTML), LaTeX (simulating paper layouts), Excel (simulating spreadsheet interfaces), and Customized (sampling fonts and colors from various themes).

On top of these styles, four types of robustness perturbations are layered: Noise, Structural Noise, Partial, and Missing. Specifically, "Partial" cuts the table into discontinuous blocks along structural boundaries to attack spatial continuity, while "Missing" masks arbitrary cells while ensuring the remaining information is sufficient for the answer. Performance fluctuations across these visual forms directly indicate whether a model truly understands structure or merely reads text.

**3. Vision-only settings and human quality audit: Testing integrated perception and reasoning while ensuring data validity.**

The vision-only setting closely mirrors real-world scenarios like photo-based or screenshot-based Q&A. "Screen Capture" renders the question, context, and table into a unified interface, while "Simulated Photo" adds camera artifacts like moiré patterns and perspective distortion. Because realistic visual transformations risk making questions unanswerable, 12 expert annotators conducted attribute labeling, QA enhancement, visual rendering audits, and quality checks. Every sample was manually inspected at least once, with 10% undergoing double-blind cross-validation.

This audit and regeneration mechanism ensures that the benchmark does not unfairly penalize models for data errors caused by rendering failure, allowing low scores in vision-only settings to be accurately attributed to deficiencies in visual-structural-reasoning coupling.

### Loss & Training
This paper evaluates models using an evaluation protocol rather than training them. Main experiments use a direct-output prompt without thinking mode. The primary metric is normalized exact match (EM), with GPT-5-mini used as a secondary judge for EM failures that are semantically equivalent. The authors also compare direct-output with CoT, where CoT utilizes step-by-step prompts or native model thinking modes, requiring output in the `<answer>...</answer>` format.

## Key Experimental Results

### Main Results
The TableVista dataset features 3,000 QA pairs, 4,449 tables, and 30,000 visual samples. The structural distribution includes Simple (300), Text-Mixed (300), Complex (1,000), Long (700), and Multi-Table (700). Questions average 26.2 words, answers 1.4 words, and tables consist of 15.3 rows and 6.6 columns on average. "Hard" questions average 6.9 reasoning steps and a skill score of 15.1.

| Model | Simple | Text-Mixed | Complex | Long | Multi | Easy | Medium | Hard | Overall |
|-------|--------|------------|---------|------|-------|------|--------|------|---------|
| GPT-5.4 | 73.0 | 86.7 | 81.7 | 68.9 | 61.3 | 93.6 | 80.1 | 47.0 | 73.6 |
| GPT-5.4-mini | 52.0 | 61.0 | 59.3 | 48.7 | 40.0 | 64.0 | 56.1 | 35.2 | 51.8 |
| Qwen2.5-VL-72B | 52.3 | 58.3 | 59.7 | 52.1 | 53.1 | 90.2 | 54.3 | 22.1 | 55.5 |
| Gemma-4-31B-it | 57.3 | 54.0 | 57.6 | 54.4 | 52.3 | 88.2 | 55.6 | 21.9 | 55.2 |
| Llama-4-Maverick | 55.3 | 55.7 | 55.9 | 52.3 | 52.4 | 84.4 | 53.7 | 24.4 | 54.2 |
| Qwen3-VL-8B | 40.7 | 44.0 | 44.1 | 41.9 | 39.9 | 76.7 | 37.3 | 12.7 | 42.2 |
| Table-LLaVA-v1.5-7B | 11.0 | 11.0 | 7.8 | 9.4 | 9.3 | 16.7 | 6.8 | 4.0 | 9.2 |

### Ablation Study
The paper analyzes visual conditions and prompting methods. The following table shows overall performance across different visual presentations.

| Model | Web | LaTeX | Excel | Custom | Noise | Structural | Partial | Missing | Screenshot | Photo | Avg. |
|-------|-----|-------|-------|--------|-------|------------|---------|---------|------------|-------|------|
| GPT-5.4 | 73.6 | 72.2 | 71.9 | 72.0 | 70.8 | 70.4 | 68.8 | 84.8 | 69.4 | 67.3 | 72.1 |
| GPT-5.4-mini | 51.8 | 49.9 | 50.1 | 51.1 | 49.5 | 48.4 | 46.8 | 66.9 | 42.0 | 37.7 | 49.4 |
| Qwen2.5-VL-72B | 55.5 | 54.5 | 54.5 | 55.0 | 51.1 | 54.4 | 50.5 | 71.3 | 57.4 | 54.0 | 55.8 |
| Llama-4-Maverick | 54.2 | 53.2 | 53.9 | 52.7 | 53.5 | 54.4 | 52.9 | 66.9 | 53.4 | 51.8 | 54.7 |
| Qwen3-VL-8B | 42.2 | 41.8 | 41.7 | 42.0 | 41.4 | 41.9 | 39.2 | 57.7 | 45.3 | 44.6 | 43.8 |
| LLaVA-v1.5-7B | 6.4 | 6.2 | 6.2 | 7.0 | 6.9 | 6.7 | 6.4 | 10.4 | 0.5 | 0.4 | 5.7 |

| Model | Direct-output | CoT | Gain |
|-------|---------------|-----|------|
| GPT-5.4 | 72.1 | 95.6 | +23.5 |
| GPT-5.4-mini | 49.4 | 91.5 | +42.1 |
| Qwen3.5-27B | 51.4 | 96.2 | +44.8 |
| Gemma-4-31B-it | 54.3 | 86.1 | +31.8 |
| Qwen3-VL-8B | 43.8 | 86.0 | +42.2 |

### Key Findings
- Style variation is not the primary bottleneck: scores for Web, LaTeX, Excel, and Custom are very close, suggesting models generalize well across fonts and themes.
- Partial and Photo are more difficult; the former breaks spatial continuity, while the latter introduces camera-style degradation. This indicates failures stem from spatial alignment rather than simple OCR.
- "Missing" often improves scores (e.g., GPT-5.4 rose from 73.6 in Web to 84.8 in Missing). This likely occurs as masking reduces distractor information and focuses the model on critical cells.
- CoT significantly narrows the performance gap (e.g., Qwen3-VL-8B jumping from 43.8 to 86.0). This suggests many models possess reasoning capabilities but struggle to internalize multi-step calculations into a single output under direct-output settings.
- Error analysis shows Table Understanding accounts for 54%, Reasoning & Calculation for 29%, and Visual Perception for only 12%. Within Table Understanding, Spatial Alignment (32%) and Structure Parsing (22%) are the primary bottlenecks.

## Highlights & Insights
- TableVista's contribution lies in centering the evaluation on semantic consistency across multiple visual forms. This reveals real-world deployment issues more effectively than accuracy on a single clean table.
- The contrast between "Partial" and "Missing" is insightful: fragmentation causes score drops due to loss of spatial continuity, while occlusion might improve performance. This suggests that future training should focus on robust reasoning between information compression and spatial recovery.
- CoT results indicate that direct-output is a rigorous but meaningful stress test for internalizing multi-step reasoning, while CoT serves as an external scratchpad. Both should be reported.
- The construction workflow is transferable to benchmarks for financial reports, medical reports, and administrative forms: control structural complexity first, then systematically add visual perturbations, and finally use error distributions to locate model weaknesses.

## Limitations & Future Work
- TableVista is an evaluation benchmark and does not provide training methods to improve robustness. It identifies problems but lacks model designs to resolve spatial alignment failures.
- The data is centered on tables; real documents often mix charts, natural images, flowcharts, and formulas, requiring broader cross-modal document reasoning.
- Using GPT-5-mini as a judge for EM is reliable for numerical and short answers but may be insufficient for open-ended table interpretations.
- "Simulated Photo" in vision-only settings uses synthetic artifacts, which still differ from real-world noise like mobile compression, reflections, or handwritten annotations.

## Related Work & Insights
- **vs TableVQA-Bench / MMTabQA**: These benchmarks introduced visual tables but offer less coverage of structural complexity and visual robustness. TableVista covers hierarchy, long tables, multiple tables, scenarios, perturbations, and vision-only modes.
- **vs MMTab / MMTBench**: While these focus on multimodal table understanding and complex content, TableVista emphasizes multiple visual variants of the same base sample to assess consistency and robustness.
- **vs TABLET**: TABLET emphasizes large-scale robust tables rendered from raw web pages. TableVista provides more systematic control over structural types and visual transformations for decomposed analysis.
- **Insights for VLM Training**: Future table models need to explicitly learn row-column alignment, cross-block relationship recovery, and sub-cell level digit differentiation, rather than solely scaling OCR data or general VQA instruction tuning.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Integrating structural complexity and visual robustness into one table reasoning benchmark provides a comprehensive evaluation perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Analysis of 29 models across structure, difficulty, visual conditions, CoT, and error types is solid in both scale and dimension.
- Writing Quality: ⭐⭐⭐⭐☆ The construction process is clear and the tables are informative, though some HTML formatting for dense tables can be challenging to read quickly.
- Value: ⭐⭐⭐⭐⭐ Highly practical for multimodal document understanding, table VQA, visual RAG, and enterprise form automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity](../../CVPR2026/vlm_reasoning/infinibench_infinite_benchmarking_for_visual_spatial_reasoning_with_customizable.md)
- [\[CVPR 2026\] TableMix: Enhancing Multimodal Table Reasoning in MLLMs from a Data-Centric Perspective](../../CVPR2026/vlm_reasoning/tablemix_enhancing_multimodal_table_reasoning_in_mllms_from_a_data-centric_persp.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[CVPR 2026\] EmoThinker: Advancing Visual-Acoustic Emotion Analysis via Structural Token Selection and Chain-of-Thought Reasoning](../../CVPR2026/vlm_reasoning/emothinker_advancing_visual-acoustic_emotion_analysis_via_structural_token_selec.md)

</div>

<!-- RELATED:END -->
