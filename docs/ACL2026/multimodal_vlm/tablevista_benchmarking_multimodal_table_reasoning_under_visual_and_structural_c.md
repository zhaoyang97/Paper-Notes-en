---
title: >-
  [Paper Note] TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity
description: >-
  [ACL 2026][Multimodal VLM][Vision-only] TableVista constructs a multimodal table benchmark consisting of 3,000 high-quality table reasoning questions expanded into 30,000 visual samples. After systematically evaluating 29 foundation models, it was found that models are relatively stable to style changes but significantly degrade under complex structures, cro
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-only
  - CoT
date: 2026-05-08
content_hash: 1da2fa407ce42058
---
# TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.05955](https://arxiv.org/abs/2605.05955)  
**Code**: https://github.com/FlowRays/TableVista  
**Area**: Multimodal VLM / Table Reasoning  
**Keywords**: Multimodal Table Reasoning, Visual Robustness, Structural Complexity, Vision-only, CoT

## TL;DR
TableVista constructs a multimodal table benchmark consisting of 3,000 high-quality table reasoning questions expanded into 30,000 visual samples. After systematically evaluating 29 foundation models, it was found that models are relatively stable to style changes but significantly degrade under complex structures, cross-table reasoning, visual fragmentation, and vision-only input.

## Background & Motivation
**Background**: Table Question Answering (Table QA) has long relied on text serialization such as Markdown, HTML, or CSV, which is suitable for standard grids and simple lookups. However, real-world tables often appear as screenshots, web pages, paper PDFs, Excel files, or mobile photos, containing multi-level headers, merged cells, long tables, multiple related tables, and contextual descriptions.

**Limitations of Prior Work**: Text serialization flattens spatial structures into token sequences, causing the loss of much visual structural information. Existing multimodal table benchmarks often use single, idealized, or fixed rendering methods, failing to test whether models maintain consistent reasoning under changes in font, layout, noise, truncation, missing parts, and photographic artifacts.

**Key Challenge**: Multimodal models appear to possess OCR and image understanding capabilities, but table reasoning actually requires the coupled ability of "visual localization + structural alignment + multi-step calculation." A model's ability to read cell text does not imply it can maintain row-column relationships, cross-table references, and multi-hop logic within complex visual layouts.

**Goal**: The authors aim to construct a table reasoning evaluation that incorporates both structural complexity and visual perturbations, forcing models to answer the same types of reasoning questions across multiple visual presentations and analyzing whether current state-of-the-art models fail due to visual recognition, structural understanding, or reasoning calculation.

**Key Insight**: TableVista does not synthesize simple tables from scratch; instead, it aggregates samples from 14 public table reasoning data sources, re-annotates them with the assistance of experts and GPT-5, and then expands each question into 10 visual versions using multi-style rendering and visual transformations.

**Core Idea**: To redefine tables from "structured text inputs" to "visual document objects," using structural complexity and visual perturbations to simultaneously examine whether multimodal models truly understand tables.

## Method
The method section of TableVista primarily involves benchmark construction. It first establishes a high-quality textual base set to ensure the questions themselves possess sufficient reasoning difficulty; it then renders each table question into multiple images under real-world scenarios; finally, it ensures the questions remain answerable after visual transformation through human auditing. The resulting output is not a single dataset but a multi-dimensional stress test matrix.

### Overall Architecture
The input consists of original records from 14 table datasets, including WTQ, HiTab, TabFact, MMQA, and FinQA. Each record is standardized into four parts: table, textual context, question, and answer, and labeled with structural attributes, information richness, reasoning skills, and reasoning steps. After filtering and enhancement, the authors retained 3,000 high-quality QA pairs.

The output comprises 30,000 multimodal samples. Each base sample is expanded into 10 visual versions: 4 scene styles, 4 robustness perturbations, and 2 vision-only settings. Model evaluation considers both structural types and difficulty, as well as performance fluctuations caused by changes in visual presentation.

### Key Designs

**1. Structural and reasoning dual-dimension filtering for the base set: Ensuring each question covers real table structures and is not a simple lookup.**

If visual transformations are performed only on simple questions, models can pass using OCR combined with local lookups, failing to test true visual table reasoning capabilities. Therefore, the authors first used GPT-5 to assist in assigning four categories of labels to candidate samples: table layout attributes, information richness ranging from $1$–$5$, four types of skill scores (lookup / aggregation / numerical / logical), and reasoning steps. Sampling was then performed with quotas according to five structural categories (Simple Structure, Text-Mixed, Complex Structure, Long Tables, Multi-Table).

During filtering, priority was given to samples with high information volume, high skill scores, and more reasoning steps, ultimately refining 3,000 QA pairs from 14 data sources. By packing "structural complexity" and "reasoning difficulty" into the base set, the models are forced to expose real weaknesses in structural alignment and multi-step calculation under subsequent visual perturbations, rather than staying at the level of character recognition.

**2. Multi-style visual rendering and robustness perturbations: Placing the same question into multiple realistic visual environments to test stability.**

Real users do not always provide models with clean HTML tables; mobile screenshots, partial occlusions, faded grid lines, and layout fragmentation all destroy spatial cues, which are the signals table reasoning relies on most. TableVista applies four scene styles to each base question: Web (simulating Wikipedia/HTML), LaTeX (simulating paper typesetting), Excel (simulating spreadsheet interfaces), and Customized (sampling fonts and color schemes from multiple themes).

On top of these styles, four categories of robustness perturbations are overlaid: Noise, Structural Noise, Partial, and Missing. Among these, Partial cuts the table into discontinuous blocks along structural boundaries to specifically attack spatial continuity, while Missing covers arbitrary cells but ensures the remaining information is still sufficient for answering. After the same semantic question is spread across multiple visual forms, score fluctuations directly correspond to whether the "model truly understands the structure or can only read text."

**3. Vision-only settings and human quality review: Testing if models can solve the problem using only an image and ensuring data validity.**

The vision-only setting most closely resembles real-world scenarios of user photo-based or screenshot-based Q&A: Screen Capture renders the question, context, and table into a unified interface, while Simulated Photo further overlays camera artifacts like moiré patterns and perspective distortion. However, realistic visual transformations are more likely to make a question unanswerable; thus, human auditing is essential. 12 expert annotators participated in attribute labeling, QA enhancement, visual rendering review, and quality auditing. Each sample was manually checked at least once, with 10% of samples undergoing double-blind cross-validation.

It is this auditing and regeneration mechanism that ensures the benchmark does not "unfairly" penalize models for data errors caused by rendering destruction, allowing low scores in vision-only settings to be truly attributed to the model's insufficient visual-structural-reasoning coupling capability.

### Loss & Training
Ours does not train a model but uses an evaluation protocol. Main results are conducted under a direct-output prompt without turning on thinking mode. The primary metric is normalized exact match (EM), with GPT-5-mini used as a secondary judge for cases where EM fails but the answer is semantically equivalent. The authors also specifically compared direct-output with CoT: CoT uses step-by-step prompts or the model's native thinking mode, finally requiring output in the `<answer>...</answer>` format.

## Key Experimental Results

### Main Results
The TableVista data scale consists of 3,000 QA pairs, 4,449 tables, and 30,000 visual samples. The structural distribution is Simple 300, Text-Mixed 300, Complex 1,000, Long 700, and Multi-Table 700. The average question length is 26.2 words, the average answer length is 1.4 words, and tables average 15.3 rows and 6.6 columns. Hard questions average 6.9 reasoning steps and a skill score of 15.1.

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
There is no ablation of training modules in the paper, but key controls were performed for visual conditions and prompting methods. The table below shows the overall performance of models under different visual presentations.

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
- Style variation itself is not the biggest bottleneck: Scores across Web, LaTeX, Excel, and Custom are very close, indicating that models have already achieved a certain level of generalization regarding fonts and theme styles.
- Partial and Photo are more difficult; the former destroys the continuous spatial structure of the table, and the latter introduces camera-style degradation. This suggests that the primary failures stem from spatial alignment rather than pure OCR.
- Missing often actually improves scores; for example, GPT-5.4 increased from 73.6 (Web) to 84.8 (Missing), and Qwen2.5-VL-72B increased from 55.5 to 71.3. This may be because occlusion reduces interfering information and focuses attention on key cells.
- CoT significantly narrows the performance gap between models; Qwen3-VL-8B jumped from 43.8 to 86.0, showing that many models possess reasoning capabilities but cannot internalize multi-step calculations into a single output under the direct-output setting.
- In the error distribution, Table Understanding accounts for 54%, Reasoning & Calculation for 29%, and Visual Perception for only 12%. Within these, Spatial Alignment (32%) and Structure Parsing (22%) again indicate that the core bottleneck is structural alignment.

## Highlights & Insights
- The contribution of TableVista is not just creating another table QA set, but centering the evaluation on "whether semantic questions remain consistent across multiple visual forms." This exposes real deployment issues more effectively than the accuracy of a single clean table.
- The contrast between Partial and Missing is insightful: fragmentation causes models to lose spatial continuity, resulting in lower scores, whereas missing occlusions might actually improve scores. This suggests that future training should focus not just on adding more clear images, but on learning robust reasoning between information compression and spatial recovery.
- The CoT results indicate that direct-output is a rigorous but meaningful stress test. It examines whether the model has internalized multi-step table reasoning, whereas CoT acts more like an external scratchpad; both should be reported simultaneously.
- This construction process can be migrated to evaluations of financial reports, medical reports, experimental tables, and administrative forms: first controlling for structural complexity, then systematically adding real visual perturbations, and finally using error distributions to locate model weaknesses.

## Limitations & Future Work
- TableVista is an evaluation benchmark and does not provide training methods to directly improve model robustness; it identifies problems without offering model designs to solve spatial alignment failures.
- The data is centered around tables; real documents also contain a mix of charts, natural images, flowcharts, footnotes, and formulas, and cross-modal document reasoning has a broader scope.
- The main experiments use GPT-5-mini as a semantic judge to correct EM. Numerical and short answers are reliable, but it may still be insufficient for open-ended table explanations.
- Simulated photos in the vision-only setting are generated via synthetic artifacts and still differ from real-world noise such as actual mobile photography, compression, reflections, and handwritten annotations.

## Related Work & Insights
- **vs TableVQA-Bench / MMTabQA**: These benchmarks have introduced visual tables, but coverage of structural complexity and visual robustness is lower; TableVista covers hierarchy, long tables, multi-tables, scene styles, perturbations, and vision-only simultaneously.
- **vs MMTab / MMTBench**: These focus on multimodal table understanding and complex content. TableVista emphasizes multiple visual variants of the same base sample, enabling the assessment of consistency and robustness.
- **vs TABLET**: TABLET emphasizes large-scale robust tables rendered from raw web pages. TableVista more systematically controls for structural types and visual transformations, facilitating decomposition analysis.
- **Insights for VLM Training**: Future table models need to explicitly learn row-column alignment, cross-block relationship recovery, and sub-cell level digit differentiation, rather than just expanding OCR data or performing ordinary VQA instruction fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Intelligently synthesizes structural complexity and visual robustness into a table reasoning benchmark; the evaluation perspective is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Analyzes 29 models across structure, difficulty, visual conditions, CoT, and error types; the data scale and dimensions are solid.
- Writing Quality: ⭐⭐⭐⭐☆ The construction process is clear, and although tables are dense, they are high in information. Table typesetting in the HTML version slightly affects quick reading.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for multimodal document understanding, table VQA, visual RAG, and enterprise form automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity](../../CVPR2026/multimodal_vlm/infinibench_infinite_benchmarking_for_visual_spatial_reasoning_with_customizable.md)
- [\[CVPR 2026\] TableMix: Enhancing Multimodal Table Reasoning in MLLMs from a Data-Centric Perspective](../../CVPR2026/multimodal_vlm/tablemix_enhancing_multimodal_table_reasoning_in_mllms_from_a_data-centric_persp.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

</div>

<!-- RELATED:END -->
