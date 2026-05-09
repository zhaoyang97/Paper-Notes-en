---
title: >-
  [Paper Note] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective
description: >-
  [ICLR2026][LLM Evaluation][design aesthetics] This paper proposes AesEval-Bench, the first benchmark for systematically evaluating VLMs on graphic design aesthetics (4 dimensions × 12 indicators × 3 tasks). It finds that existing VLMs—including reasoning-augmented models—perform poorly on design aesthetics, and constructs training data via human-guided VLM labeling combined with indicator-grounded reasoning. Fine-tuning a 7B model with this data surpasses GPT-5 on the precise localization task.
tags:
  - ICLR2026
  - LLM Evaluation
  - design aesthetics
  - VLM evaluation
  - benchmark
  - indicator-grounded reasoning
  - graphic design
date: 2026-05-08
content_hash: c56fab561e61d4f0
---

# Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective

**Conference**: ICLR2026
**arXiv**: [2603.01083](https://arxiv.org/abs/2603.01083)
**Code**: [https://github.com/arctanxarc/AesEval-Bench](https://github.com/arctanxarc/AesEval-Bench)
**Area**: LLM Evaluation
**Keywords**: design aesthetics, VLM evaluation, benchmark, indicator-grounded reasoning, graphic design

## TL;DR
This paper proposes AesEval-Bench, the first benchmark for systematically evaluating VLMs on graphic design aesthetics (4 dimensions × 12 indicators × 3 tasks). It finds that existing VLMs—including reasoning-augmented models—perform poorly on design aesthetics, and constructs training data via human-guided VLM labeling combined with indicator-grounded reasoning. Fine-tuning a 7B model with this data surpasses GPT-5 on the precise localization task.

## Background & Motivation

**Background**: VLMs have achieved notable progress on tasks such as image captioning and VQA, yet their capability in graphic design aesthetics evaluation (assessing the visual appeal of posters, advertisements, and UIs) remains largely unexplored.

**Limitations of Prior Work**: (a) **Incomplete benchmarks**—existing design aesthetics benchmarks cover only a subset of dimensions (e.g., neglecting graphic quality or typography), and evaluation protocols are either coarse-grained scores (unable to localize problem regions) or open-ended descriptions (difficult to quantify); (b) **Lack of systematic comparison**—no comprehensive evaluation across open-source, closed-source, and reasoning-augmented VLMs exists; (c) **Scarcity of training data**—how to improve VLM performance in this domain has not been investigated.

**Key Challenge**: Design aesthetics is a multi-dimensional and highly subjective task (encompassing typography, layout, color, and graphics), and the general reasoning capabilities of existing VLMs are insufficient for the fine-grained, domain-knowledge-intensive evaluation it requires.

**Goal**: (a) Establish a quantitative benchmark covering comprehensive design dimensions; (b) Systematically evaluate the capability boundaries of diverse VLMs; (c) Construct training data that effectively improves VLM performance.

**Key Insight**: Design aesthetics is decomposed into 4 dimensions (typography, layout, color, graphics) × 12 indicators. Three tasks of increasing difficulty—judgment, region selection, and precise localization—are designed to evaluate VLMs from coarse to fine granularity. Indicator-grounded reasoning is then used to train VLMs to associate abstract aesthetic indicators with concrete design regions.

**Core Idea**: Establish the first systematic design aesthetics benchmark, reveal that reasoning-augmented VLMs offer no advantage in this domain, and significantly improve VLM aesthetic evaluation capability through indicator-grounded reasoning training data.

## Method

### Overall Architecture
The work consists of three components: (1) **AesEval-Bench construction**—professional designs are sampled from the Crello dataset, controlled perturbations are applied to generate defective designs, and 4,500 QA pairs are constructed after human annotation; (2) **Systematic evaluation**—10+ VLMs are tested across 3 task types; (3) **AesEval-Train construction**—labels are scaled up via human-guided VLM labeling, reasoning chains are generated through indicator-grounded reasoning, and VLMs are fine-tuned on the resulting data.

### Key Designs

1. **AesEval-Bench Design**:

    - **Function**: Provides a quantitative design aesthetics evaluation benchmark covering 4 dimensions and 12 indicators.
    - **Mechanism**: The 4 dimensions are typography (legibility, hierarchy), layout (balance, layering, whitespace, alignment), color (harmony, contrast, appeal, psychology), and graphics (quality, relevance). Three tasks of increasing difficulty are defined: aesthetic judgment (yes/no) → region selection (4-choice) → precise localization (bounding box coordinates).
    - **Design Motivation**: Existing benchmarks cover only partial dimensions and lack quantitative evaluation. The three-task progression from global perception to fine-grained localization enables comprehensive measurement of VLMs' depth of aesthetic understanding.

2. **Controlled Defect Design Generation**:

    - **Function**: Generates defective designs from professional originals via controlled perturbations.
    - **Mechanism**: The JSON metadata of the Crello dataset (containing element coordinates, fonts, and colors) is perturbed at the JSON level (repositioning elements, changing fonts, adjusting colors, etc.) and re-rendered into design images. Human annotators then verify whether each perturbation genuinely introduces an aesthetic problem.
    - **Design Motivation**: Directly collecting defective designs offers little control over defect type and location. Starting from professional designs and applying perturbations ensures both realism and precise control over ground truth.

3. **Human-guided VLM Labeling (Training Label Generation)**:

    - **Function**: Generates training labels at scale, avoiding the high cost of full manual annotation.
    - **Mechanism**: A small number of human annotations serve as in-context examples; bounding box coordinates of the perturbed regions are provided as prior information to guide a powerful VLM (e.g., GPT) in generating binary classification labels (whether the design has an aesthetic problem).
    - **Design Motivation**: Manual annotation is costly and unscalable. Although providing perturbation region coordinates is not available in real-world inference, it substantially improves label reliability during the annotation phase.

4. **Indicator-grounded Reasoning (Training Reasoning Chain Generation)**:

    - **Function**: Generates reasoning chains that anchor abstract aesthetic indicators to concrete design regions.
    - **Mechanism**: GPT is provided with bounding box coordinates of target regions and corresponding design layers, and is required to produce reasoning chains containing coordinates along with explanations of indicator relevance. Different strategies are applied per task: aesthetic judgment uses perturbed region bboxes; region selection provides both perturbed and non-perturbed regions; precise localization additionally emphasizes the relationship to the overall design.
    - **Design Motivation**: General reasoning (e.g., GPT-o1/o3) is found to be unhelpful for aesthetic evaluation, as such reasoning is generic analysis rather than anchored to specific regions. Indicator-grounded reasoning forces abstract concepts (e.g., "hierarchy") to be linked to concrete bounding boxes in the design, providing effective supervision signals.

### Loss & Training
Full-parameter fine-tuning is performed on Qwen2.5-VL-7B-Instruct with the visual encoder frozen and only the language model parameters updated. Training uses a learning rate of 1e-6, a cosine scheduler with 3% warmup, and bfloat16 precision with FlashAttention-2. The training set comprises 30k QA pairs; inputs consist of task descriptions, design images, and JSON metadata, while supervision signals include reasoning chains and task labels.

## Key Experimental Results

### Main Results (VLM Benchmark Evaluation)

| Model | Aesthetic Judgment Acc | Region Selection Acc | Precise Localization (choice) Acc | Precise Localization (bbox) IoU |
|-------|----------------------|---------------------|----------------------------------|--------------------------------|
| GPT-5 | **0.7252** | **0.6989** | **0.6090** | **0.1993** |
| GPT-4o | 0.7031 | 0.6745 | 0.5680 | 0.1712 |
| GPT-o3 | 0.7105 | 0.6581 | 0.5800 | 0.1418 |
| GPT-o1 | 0.6705 | 0.6347 | 0.5295 | 0.1286 |
| Gemini-2.5-Pro | 0.6368 | 0.6100 | 0.6047 | 0.0977 |
| Qwen-VL-72B | 0.6724 | 0.6626 | - | - |
| InternVL3-14B | 0.6883 | 0.6378 | - | - |
| AesExpert-7B | 0.4056 | 0.2883 | 0.3377 | 0.0327 |

### Ablation Study (Fine-tuning Effectiveness)

| Configuration | Aesthetic Judgment Acc | Region Selection Acc | Precise Localization (bbox) IoU |
|--------------|----------------------|---------------------|--------------------------------|
| Qwen-VL-7B (Base) | 0.6390 | 0.5795 | 0.0514 |
| + AesEval-Train | **0.6987 (+5.97%)** | **0.6065 (+2.70%)** | **0.2105 (+17.17%)** |
| − Reasoning Path | 0.6576 | 0.5795 | 0.1634 |
| − Positive Samples | 0.2072 | 0.2437 | 0.0012 |

### Key Findings
- **Reasoning-augmented VLMs offer no advantage**: GPT-o1/o3 do not outperform GPT-4o/GPT-5 on aesthetic judgment and region selection, indicating that general reasoning capability does not transfer directly to design aesthetics.
- **Image aesthetics specialist models perform poorly**: AesExpert and UNIAA-LLAVA score far below general-purpose VLMs, demonstrating a fundamental difference between natural image aesthetics and graphic design aesthetics.
- **Precise bbox localization is a hard problem**: Even GPT-5 achieves only an IoU of 0.1993 on precise localization, indicating that VLMs remain far from accurately understanding the spatial positions of design elements.
- **Indicator-grounded reasoning is critical**: Removing the reasoning path drops precise localization IoU from 0.2105 to 0.1634; removing positive samples causes near-complete collapse, confirming that domain-specific anchored reasoning is the primary source of improvement.
- **Fine-tuned 7B surpasses GPT-5**: On the precise localization task, fine-tuned Qwen-VL-7B (IoU 0.2105) outperforms GPT-5 (0.1993), demonstrating the value of domain-specific training data.

## Highlights & Insights
- **Elegant three-level task design**: The progression from judgment → selection → localization tests VLMs' depth of aesthetic understanding like a structured examination. This benchmark design paradigm is transferable to other subjective evaluation tasks (e.g., code quality assessment, writing quality evaluation).
- **Generality of indicator-grounded reasoning**: The idea of anchoring abstract concepts to concrete spatial regions is applicable not only to aesthetic evaluation but also to any task requiring the association of high-level concepts with low-level visual features (e.g., medical image anomaly localization, architectural design review).
- **Reasoning ≠ domain knowledge**: Reasoning-augmented VLMs excel at general tasks but do not necessarily hold an advantage in specialized domains—a finding with significant practical implications for VLM application selection.

## Limitations & Future Work
- **Single data source**: The benchmark relies solely on the Crello dataset, which primarily covers flat graphic design; UI design, web design, and packaging design are not included.
- **Limited perturbation types**: Perturbations are applied at the JSON level and do not address more complex design defects (e.g., semantic mismatch, cultural inappropriateness).
- **Simplistic evaluation metrics**: IoU may not be the optimal metric for aesthetic problem localization, as the boundaries of aesthetic problem regions are inherently ambiguous.
- **Absence of real designer feedback**: Reasoning chains in the training data are GPT-generated and have not been validated against the reasoning processes of professional designers.
- **Single model fine-tuning**: The training strategy is validated only on Qwen-VL-7B; its effectiveness on larger models remains unverified.

## Related Work & Insights
- **vs. AesBench/UNIAA-Bench (image aesthetics)**: These benchmarks target natural photographs and focus on factors such as exposure and composition. This paper focuses on graphic design and introduces typography and layout dimensions. The poor performance of design aesthetics specialist models on AesEval-Bench confirms the fundamental difference between the two domains.
- **vs. DesignProbe/GPT-Eval Bench (design aesthetics)**: These benchmarks cover fewer dimensions and employ a single evaluation format. AesEval-Bench is the first to simultaneously cover 4 dimensions, 12 indicators, and 3 quantitative task types.
- **vs. general grounded reasoning (e.g., SoM)**: General visual reasoning grounds semantic entities (cars, people), whereas this paper grounds aesthetic indicators (hierarchy, alignment) at a higher level of abstraction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic design aesthetics VLM benchmark combined with an indicator-grounded reasoning training method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison of 10+ VLMs, thorough ablation studies, and input component analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logical structure, well-defined problem formulation, and rich comparative tables.
- **Value**: ⭐⭐⭐⭐ — Establishes an evaluation foundation for VLM applications in design, with practically useful training strategies.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/llm_evaluation/vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[AAAI 2026\] Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](../../AAAI2026/llm_evaluation/perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](../../NeurIPS2025/llm_evaluation/can_large_language_models_master_complex_card_games.md)
- [\[ICLR 2026\] Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)

<!-- RELATED:END -->
