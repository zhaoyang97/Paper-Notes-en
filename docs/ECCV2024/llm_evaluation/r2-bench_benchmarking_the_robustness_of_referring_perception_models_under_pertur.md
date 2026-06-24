---
title: >-
  [Paper Note] R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations
description: >-
  [ECCV 2024][LLM Evaluation][Referring Perception] Proposes R²-Bench, a comprehensive benchmark to systematically evaluate the robustness of referring perception models (RPMs) under various perturbations. It features a complete perturbation taxonomy, a versatile perturbation synthesis toolbox, and an LLM-based automated evaluation agent (R²-Agent), covering five key tasks and revealing the vulnerability of current RPMs under noisy conditions.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Referring Perception"
  - "Robustness Evaluation"
  - "Perturbation Benchmark"
  - "Multimodal"
  - "LLM Agent"
date: 2026-05-08
content_hash: b7f8fcdbdbc8b934
---

# R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations

**Conference**: ECCV 2024  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Referring Perception, Robustness Evaluation, Perturbation Benchmark, Multimodal, LLM Agent

## TL;DR

Proposes R²-Bench, a comprehensive benchmark to systematically evaluate the robustness of referring perception models (RPMs) under various perturbations. It features a complete perturbation taxonomy, a versatile perturbation synthesis toolbox, and an LLM-based automated evaluation agent (R²-Agent), covering five key tasks and revealing the vulnerability of current RPMs under noisy conditions.

## Background & Motivation

**Background**: Referring perception aims to localize visual targets guided by multimodal referring expressions (such as text descriptions, clicks, etc.), which is a key technology connecting human instructions with environmental perception in intelligent systems. Significant progress has been made in various tasks such as referring expression comprehension (REC) and referring image segmentation (RIS).

**Limitations of Prior Work**: Although models achieve outstanding performance on standard benchmarks, their robustness lacks systematic evaluation when facing inevitable perturbations in real-world deployment (such as image noise, occlusion, text typos, missing modalities, etc.). Sensor noise, lighting variations, and user input errors in real-world scenarios can severely degrade model performance.

**Key Challenge**: Referring perception involves multimodal inputs. Perturbations can originate from the visual modality, linguistic modality, or their cross-modal interactions, creating a complexity that far exceeds single-modality robustness evaluation. The absence of a unified perturbation taxonomy and evaluation tools hinders fair comparison across different studies.

**Goal**: (1) Establish a perturbation taxonomy for the referring perception domain; (2) Develop a versatile perturbation synthesis and evaluation toolbox; (3) Construct R²-Bench, a robustness benchmark covering multiple tasks; (4) Provide an LLM-based automated evaluation agent to streamline the benchmarking process.

**Key Insight**: Starting from the multimodal nature of referring perception tasks, this paper systematically analyzes the types of potential perturbations affecting the models, ranging from general perturbations (e.g., image noise) to task-specific perturbations (e.g., referring ambiguity), thereby establishing a hierarchical perturbation taxonomy.

**Core Idea**: Build a comprehensive benchmark featuring a holistic perturbation taxonomy, a versatile toolbox, and an LLM agent to systematically evaluate the robustness of referring perception models under various noisy conditions.

## Method

### Overall Architecture

R²-Bench consists of three main components: (1) a perturbation taxonomy that categorizes perturbations affecting RPMs into general contextual perturbations and referring-specific perturbations; (2) a versatile perturbation synthesis and evaluation toolbox providing modular generation, combination, and impact assessment of perturbations; and (3) R²-Agent, an LLM-based autonomous agent that simplifies model evaluation processes via natural language instructions. The benchmark covers five key referring perception tasks.

### Key Designs

1. **Hierarchical Perturbation Taxonomy**:

    - **Function**: Provide a systematic perturbation classification framework for the referring perception field.
    - **Mechanism**: Categorize perturbations into two main groups: (a) general contextual perturbations, including visual perturbations (Gaussian noise, motion blur, brightness variations, occlusion, etc.) and textual perturbations (typos, synonym substitution, word order changes, etc.); (b) referring-specific perturbations, including spatial relationship ambiguity, attribute confusion, target quantity changes, etc. Several severity levels are defined for each perturbation type to support fine-grained evaluation.
    - **Design Motivation**: Existing robustness research mostly focuses on single-modality perturbations while ignoring the cross-modal perturbations unique to referring perception. A hierarchical taxonomy ensures the comprehensiveness of evaluation.

2. **Versatile Toolbox**:

    - **Function**: Support flexible synthesis, combination, and automated evaluation of perturbations.
    - **Mechanism**: The toolbox adopts a modular design, wrapping each perturbation type as an independent transformation module to support the generation of both single and composite perturbations. It provides unified APIs to easily combine different modalities and types of perturbations. The evaluation module supports diverse metrics (accuracy, IoU, precision/recall, etc.) and automatically generates comparison reports.
    - **Design Motivation**: A unified toolbox lowers the barrier to robustness evaluation, allowing researchers to quickly replicate and extend experiments, thereby facilitating fair comparisons.

3. **R²-Agent: LLM-based Automated Evaluation Agent**:

    - **Function**: Simplify and automate model robustness evaluation using natural language commands.
    - **Mechanism**: R²-Agent receives natural language evaluation queries from users (e.g., "evaluate a CLIP-based REC model under Gaussian noise"), automatically parses the intent, selects suitable perturbation configurations, executes the evaluation pipeline, and generates structured reports. The underlying LLM understands the perturbation taxonomy and coordinates modules within the toolbox.
    - **Design Motivation**: Manually configuring perturbation parameters and evaluation pipelines is tedious and error-prone. The LLM agent enables non-expert users to conduct systematic robustness evaluations.

### Loss & Training

R²-Bench serves as an evaluation benchmark rather than a training methodology and does not involve specific loss functions. The five tasks covered by the benchmark are Referring Expression Comprehension (REC), Referring Image Segmentation (RIS), Referring Video Object Segmentation (RVOS), Phrase Grounding, and Referring 3D Object Detection.

## Key Experimental Results

### Main Results

| Task | Perturbation Type | Model Category | Clean Performance | Perturbed Performance | Drop Ratio |
|------|---------|---------|-----------|-----------|---------|
| REC | Visual Noise | Task-Specific Model | ~85% | ~60-70% | 15-25% |
| REC | Textual Perturbation | Task-Specific Model | ~85% | ~55-65% | 20-30% |
| RIS | Visual Noise | General Model | ~70 mIoU | ~45-55 mIoU | 15-25 mIoU |
| RIS | Composite Perturbation | General Model | ~70 mIoU | ~35-45 mIoU | 25-35 mIoU |
| Cross-Modal | Combined Perturbation | Large Model | Baseline Level | Significant Drop | Max Drop |

### Ablation Study

| Perturbation Dimension | Degree of Impact | Description |
|---------|---------|------|
| Visual Perturbation Only | Moderate | Models exhibit some tolerance to image noise |
| Textual Perturbation Only | Large | Textual errors significantly impact referring disambiguation |
| Referring-Specific Perturbation | Maximum | Spatial relationships and attribute ambiguities are most challenging |
| Composite Perturbation | Extreme | Combining multimodal perturbations causes dramatic performance degradation |

### Key Findings

- Current RPMs are generally more sensitive to textual perturbations (particularly typos and synonym substitutions) than visual ones.
- Referring-specific perturbations (e.g., spatial relationship ambiguity) lead to more severe performance degradation than general perturbations.
- The impact of composite perturbations is far greater than the simple addition of single perturbations, indicating a deficiency in joint robustness.
- Although large general-purpose models (e.g., CLIP-based models) perform well on clean data, their robustness is not necessarily superior to smaller task-specific models.
- R²-Agent effectively reduces human labor costs in evaluation while producing results highly consistent with manual configurations.

## Highlights & Insights

- **Systematic Perturbation Taxonomy**: Establishes a hierarchical perturbation taxonomy for referring perception for the first time, distinguishing between general and task-specific perturbations. This classification framework can be transferred to robustness evaluations in other multimodal tasks (e.g., VQA, image captioners).
- **Automated Benchmarking with LLM Agent**: R²-Agent automates perturbation configuration and evaluation execution, demonstrating the potential of LLMs in automating ML evaluation pipelines. This approach is extensible to other automated benchmark testing.
- **Cross-Modal Composite Perturbation Analysis**: Reveals the limitations of single-modality robustness evaluation—models robust to a single modality may collapse under composite perturbations. This provides a crucial warning for real-world deployments.

## Limitations & Future Work

- Although comprehensive, the perturbation types are still predominantly synthetic, and discrepancies with real-world distributions are not fully analyzed.
- R²-Agent relies on the comprehension capability of LLMs, which might misunderstand complex or ambiguous evaluation requests.
- The benchmark mainly evaluates static robustness, neglecting the cumulative effects of perturbations in temporal sequences.
- There is a lack of systematic exploration of robustness enhancement methods (e.g., evaluating the efficacy of adversarial training or data augmentation strategies).
- Discrepancies in calibrating perturbation severities across the five tasks may exist, requiring caution when interpreting cross-task comparisons.

## Related Work & Insights

- **vs Robustness benchmarks**: Traditional robustness benchmarks (e.g., ImageNet-C) only focus on visual perturbations in image classification, whereas R²-Bench extends to multimodal referring perception, covering textual and cross-modal perturbations.
- **vs POPE/MMBench**: While these multimodal LLM benchmarks test capabilities, they do not systematically evaluate robustness. R²-Bench specifically targets performance degradation analysis under perturbed conditions.
- This benchmark provides important references for deploying RPMs in safety-critical scenarios (e.g., autonomous driving, medical assistance).

## Rating

- Novelty: ⭐⭐⭐⭐ The first benchmark to systematically evaluate the robustness of referring perception, featuring a well-designed perturbation taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Encompasses five key tasks and multiple perturbation types, offering rich analytical dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, and standard formatting of figures/tables.
- Value: ⭐⭐⭐⭐ Fills a crucial gap in referring perception robustness evaluation, and R²-Agent exhibits practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](../../ACL2026/llm_evaluation/stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)
- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ICLR 2026\] PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing](../../ICLR2026/llm_evaluation/pcb-bench_benchmarking_llms_for_printed_circuit_board_placement_and_routing.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)

</div>

<!-- RELATED:END -->
