---
title: >-
  [Paper Note] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits
description: >-
  [ACL 2025][LLM Evaluation][Text-guided image editing] This paper proposes EditInspector, a multi-dimensional text-guided image editing evaluation benchmark based on human annotations. It covers six dimensions: editing accuracy, artifact detection, visual quality, scene fusion, common sense consistency, and change description. It reveals the limitations of current VLMs in comprehensively evaluating editing quality, and proposes two new methods that outperform SOTA in artifact…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Text-guided image editing"
  - "editing quality evaluation"
  - "benchmark"
  - "artifact detection"
  - "difference description generation"
date: 2026-05-08
content_hash: 4042e894299c71bf
---

# EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits

**Conference**: ACL 2025  
**arXiv**: [2506.09988](https://arxiv.org/abs/2506.09988)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Text-guided image editing, editing quality evaluation, benchmark, artifact detection, difference description generation

## TL;DR

This paper proposes EditInspector, a multi-dimensional text-guided image editing evaluation benchmark based on human annotations. It covers six dimensions: editing accuracy, artifact detection, visual quality, scene fusion, common sense consistency, and change description. It reveals the limitations of current VLMs in comprehensively evaluating editing quality, and proposes two new methods that outperform SOTA in artifact detection and difference description generation.

## Background & Motivation

**Background**: With the rapid development of diffusion models and large-scale vision-language models, text-guided image editing technologies (such as InstructPix2Pix, MagicBrush, etc.) are becoming increasingly mature. Users can perform various editing operations (adding objects, modifying attributes, replacing backgrounds, etc.) on images using natural language instructions.

**Limitations of Prior Work**: Editing methods emerge endlessly, but there is a lack of a comprehensive and systematic evaluation framework. Existing evaluations typically focus solely on the "whether the edit conforms to the instruction" dimension, neglecting issues such as artifacts, degraded visual quality, mismatch with the scene, and violations of common sense that edits might introduce. Furthermore, existing automatic evaluation metrics (such as CLIPScore, LPIPS) have limited alignment with human perception.

**Key Challenge**: Editing quality is multi-dimensional—an edit might accurately execute the instruction but introduce obvious artifacts, or have high visual quality but deviate from the instruction's intent. There is a lack of a unified framework to simultaneously evaluate all these dimensions.

**Goal**: (1) Define comprehensive evaluation dimensions for text-guided image editing; (2) Build an evaluation benchmark based on human annotations; (3) Evaluate the performance of existing VLMs on editing evaluation; (4) Propose improved automatic evaluation methods.

**Key Insight**: Starting from the cognitive process of human evaluation of editing quality, the subjective question of "whether an edit is good" is decomposed into six measurable objective sub-problems.

**Core Idea**: Build an annotated dataset covering six evaluation dimensions to measure and improve the capability of VLMs in judging editing quality.

## Method

### Overall Architecture

The construction of EditInspector is divided into three stages. First stage: Collect outputs of various editing methods under different editing instructions to form a set of triplets (source image, editing instruction, edited result). Second stage: Design comprehensive annotation templates and invite human annotators to rate and annotate each edit across six dimensions. Third stage: Evaluate the editing judgment capabilities of existing VLMs using the collected annotation data, and train improved automatic evaluation models.

### Key Designs

1. **Six-Dimensional Editing Evaluation System**:

    - **Function**: Defines a complete framework for editing quality evaluation.
    - **Mechanism**: Decomposes editing quality into six independent dimensions: (a) **Editing Accuracy**—Is the text instruction correctly executed? (b) **Artifact Detection**—Does the edit introduce visible visual artifacts (e.g., edge discontinuities, texture distortion, etc.)? (c) **Visual Quality**—What is the aesthetic quality of the overall edited image? (d) **Scene Fusion**—Does the edited content fuse naturally with the original scene in terms of lighting, perspective, and style? (e) **Common Sense Consistency**—Does the edited result conform to physical common sense and semantic logic (e.g., human scale, physical properties, etc.)? (f) **Change Description**—Can the differences before and after editing be accurately described? Each dimension has a corresponding annotation template and grading standard.
    - **Design Motivation**: To cover all aspects that users actually care about when evaluating editing quality, avoiding partial or incomplete evaluation.

2. **Template-Based Human Annotation Protocol**:

    - **Function**: Collect high-quality, consistent human evaluation annotations.
    - **Mechanism**: Design detailed annotation templates for each evaluation dimension, containing positive/negative examples, rating scale definitions (1-5 points), and annotation guidelines. The annotation adopts a double-blind process: each sample is independently rated by at least 3 annotators, using Fleiss' Kappa to measure consistency. For artifact detection, annotators are also required to draw bounding boxes around artifact locations; for change descriptions, they are required to write concrete description texts of differences before and after editing.
    - **Design Motivation**: To ensure annotation quality and consistency, providing reliable ground truth for subsequent model evaluation.

3. **Improved Automatic Evaluation Methods**:

    - **Function**: Surpass existing VLMs in the two dimensions of artifact detection and difference description.
    - **Mechanism**: For artifact detection, a local region contrast method is proposed—splitting edited regions into small patches and comparing local feature changes before and after editing with a feature difference detector. Abnormally large feature changes are flagged as potential artifacts. For difference description generation, a cascaded method is proposed: first, a difference detection model is used to locate the changed areas, then a VLM is used to generate fine-grained descriptions of the changed areas, and finally, these are integrated into a complete narrative of differences. Both methods leverage task-specific inductive biases—artifacts are local phenomena, and change descriptions require localization before description.
    - **Design Motivation**: General-purpose VLMs perform poorly on these two tasks due to a lack of task-specific prior knowledge. Introducing local contrast and cascaded strategies overcomes the shortcomings of general-purpose models.

### Loss & Training

The artifact detector is trained using a binary classification loss (patch-level presence or absence of artifact labels). The difference description model is fine-tuned on the (edited pair, human description) data collected in EditInspector, using standard language model cross-entropy loss.

## Key Experimental Results

### Main Results

Performance of SOTA VLMs across the six dimensions of EditInspector (Accuracy / Correlation):

| Model | Editing Accuracy | Artifact Detection | Visual Quality | Scene Fusion | Common Sense Consistency | Change Description |
|------|---------|---------|---------|---------|---------|---------|
| GPT-4V | 72.3% | 41.5% | 58.7% | 54.2% | 61.8% | 38.5% |
| Gemini Pro Vision | 68.1% | 38.2% | 55.3% | 51.7% | 58.4% | 35.2% |
| LLaVA-1.5 | 59.4% | 32.8% | 48.6% | 45.3% | 52.1% | 28.7% |
| Ours-Artifact Detection | - | **58.3%** | - | - | - | - |
| Ours-Change Description | - | - | - | - | - | **52.1%** |

### Ablation Study

Common failure mode analysis of each model:

| Failure Type | GPT-4V | Gemini | LLaVA | Description |
|---------|--------|--------|-------|------|
| Missed Artifacts | 45.2% | 49.8% | 55.3% | Failed to detect obvious artifacts |
| False Positive Artifact Reports | 13.3% | 12.0% | 11.9% | Incorrectly reported non-existent artifacts |
| Change Hallucination | 31.5% | 35.7% | 42.8% | Described changes that did not occur |
| Missed Changes | 30.0% | 29.1% | 28.5% | Missed changes that actually occurred |
| Common Sense Misjudgment | 18.2% | 21.6% | 27.9% | Failed to discover edits that violate common sense |

### Key Findings

- **Artifact detection is the biggest weakness**: The accuracy of all VLMs in artifact detection is below 50%, indicating that current models lack sensitivity to image generation artifacts.
- **Change descriptions frequently suffer from hallucinations**: Models often "fabricate" non-existent changes when describing edits (30-43%), which is consistent with the general hallucination issues of VLMs.
- **Editing accuracy is relatively the best-performing dimension**: VLMs' judgments on "whether edits conform to instructions" are relatively reliable (GPT-4V reaches 72%), but they still struggle to evaluate quality-level aspects.
- **The proposed methods significantly lead in specific dimensions**: Detection of artifacts is improved by 16.8%, and change descriptions by 13.6%, proving the necessity of task-specific approaches.

## Highlights & Insights

- **Comprehensive and systematic evaluation dimension design**: The six-dimensional framework from "editing accuracy" to "common sense consistency" is a significant expansion of editing evaluation. This framework itself is a key contribution that can be widely adopted by subsequent work.
- **Revealing the "quality blind spot" of VLMs**: Models can judge "what the edit did" but are not good at judging "how well the edit was done." This finding offers important insights for the VLM evaluation system and the construction of training data.
- **An elegant and efficient approach of utilizing local contrast for artifact detection**: Utilizing the localized nature of edits to transform artifact detection into patch-level anomaly detection is a trick that can be transferred to fields such as image generation quality control.

## Limitations & Future Work

- **Limited annotation scale**: The cost of human annotation is high, and the benchmark dataset scale may not be sufficient for training large models.
- **Incomplete coverage of editing methods**: Rapidly evolving editing technologies might quickly make the benchmark outdated.
- **Doubt about the completeness of the six dimensions**: Other important evaluation dimensions (such as temporal consistency, cumulative errors in multi-turn editing, etc.) may not be covered yet.
- **Future directions**: Exploring targeted fine-tuning of VLMs using EditInspector's annotated data to improve their editing evaluation capabilities, or extending the framework to video editing evaluation.

## Related Work & Insights

- **vs TEdBench / MagicBrush evaluation**: These early benchmarks only focused on the single dimension of editing accuracy. EditInspector's six-dimensional framework is a significant expansion.
- **vs CLIPScore / LPIPS**: These automatic metrics lack the ability to evaluate artifacts and common sense. The human-annotated data in this work provides a foundation for training better automatic metrics.
- **vs General VLM Evaluation (such as POPE, MMBench)**: These benchmarks do not involve image-editing-specific capabilities. EditInspector fills the gap in the editing evaluation field.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-dimensional editing evaluation framework is novel, though the benchmark construction methodology itself is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of multiple SOTA VLMs with detailed failure mode analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, and the design of the evaluation dimensions is logically self-consistent.
- **Value**: ⭐⭐⭐⭐ Provides crucial infrastructure for image editing quality evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs](pap2pat_benchmarking_outline-guided_long-text_patent_generation_with_patent-pape.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](com2_causal_commonsense.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](../../ICCV2025/llm_evaluation/omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)

</div>

<!-- RELATED:END -->
