---
title: >-
  [Paper Note] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems
description: >-
  [ACL 2026][Multimodal VLM][Visual Mathematical Reasoning] The authors propose the FlowVerse benchmark (which decomposes mathematical problem information into four components—DI/EI/RP/OQ—to construct six variant versions)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Visual Mathematical Reasoning"
  - "Multimodal Large Language Models"
  - "Perception-Reasoning Decoupling"
  - "Mathematical Diagram Understanding"
  - "Benchmarking"
date: 2026-05-08
content_hash: cffa5f6a5bfc3c45
---

# MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems

**Conference**: ACL 2026  
**arXiv**: [2503.16549](https://arxiv.org/abs/2503.16549)  
**Code**: [GitHub](https://github.com/MathFlow-zju/MathFlow)  
**Area**: Multimodal VLM  
**Keywords**: Visual Mathematical Reasoning, Multimodal Large Language Models, Perception-Reasoning Decoupling, Mathematical Diagram Understanding, Benchmarking

## TL;DR

The authors propose the FlowVerse benchmark (which decomposes mathematical problem information into four components—DI/EI/RP/OQ—to construct six variant versions) and the MathFlow modular pipeline (which decouples perception and reasoning into independent stages). A specialized perception model, MathFlow-P-7B, is trained to extract key information from mathematical diagrams, significantly enhancing the visual mathematical problem-solving capabilities of various reasoning models.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) demonstrate excellent performance in tasks such as image captioning and visual question answering. However, they still exhibit significant deficiencies in solving visual mathematical problems, particularly in accurately perceiving and interpreting key information like geometric elements and numerical relationships within mathematical diagrams.

**Limitations of Prior Work**: Most existing methods focus on improving the reasoning process (e.g., CoT strategies, tool-assisted reasoning, reinforcement learning), overlooking a critical prerequisite—the quality of information extraction during the perception stage directly limits the upper bound of reasoning. Even powerful models like GPT-4V exhibit notable flaws when extracting key information from diagrams. Existing benchmarks (e.g., MathVista, MathVerse) also fail to finely distinguish between the respective contributions of perceptual and reasoning abilities.

**Key Challenge**: The perception and reasoning abilities of models are coupled during training and evaluation, making independent optimization impossible. Errors in the perception stage propagate through the reasoning stage, and existing frameworks cannot diagnose which stage constitutes the bottleneck.

**Goal**: (1) Construct a fine-grained benchmark to independently evaluate perception and reasoning abilities; (2) design a modular pipeline to decouple perception and reasoning; (3) train a specialized perception model to improve the quality of key information extraction.

**Key Insight**: Inspired by human problem-solving, where humans first extract key information from a diagram (perception) before performing mathematical derivations (reasoning). The authors decompose problem information into four independent components and build six variants by combining these components to pinpoint model weaknesses.

**Core Idea**: The information flow of visual mathematical problems is decomposed into Descriptive Information (DI), Essential Information (EI), Reasoning Properties (RP), and Original Question (OQ). Through controlled experiments, it is demonstrated that perception is the bottleneck, which is then addressed using a pipeline consisting of a specialized perception model and a general-purpose reasoning model.

## Method

### Overall Architecture

MathFlow is a two-stage modular pipeline: (1) Perception Stage: MathFlow-P-7B extracts EI (key information like angle values, side lengths) and RP (reasoning properties like inferred geometric relationships) from mathematical diagrams and converts them into text representations; (2) Reasoning Stage: The extracted textual information is concatenated with the original question and fed into any reasoning model (e.g., GPT-5, Claude) to generate a solution.

### Key Designs

1.  **Four-Component Information Decomposition in FlowVerse Benchmark**:
    - **Function**: Provides fine-grained diagnosis of perception-reasoning capabilities.
    - **Mechanism**: Decomposes information for each problem into DI (descriptive information, e.g., "Triangle ABC"), EI (essential information, e.g., "$\angle A=45^\circ$"), RP (reasoning properties needing inference from the diagram), and OQ (Question). Six variants are constructed: Text Centric (full text), Text Limited (no DI), Text Plus (no image), Vision Dense (no RP), Vision Centric (EI as image), and Vision Primary (EI+RP as image). Comparing the accuracy differences between variants allows for precise pinpointing of capability bottlenecks.
    - **Design Motivation**: Existing benchmarks mix the evaluation of perception and reasoning, making it impossible to identify the source of errors. RP serves as a controlled variable to independently test if the model can derive intermediate properties on its own.

2.  **MathFlow-P-7B Perception Model Training**:
    - **Function**: High-quality extraction of EI and RP from mathematical diagrams.
    - **Mechanism**: Two-stage training based on Qwen2-VL-7B. The multi-task pre-training stage includes EI description tasks (650k image-text pairs) and visual reasoning tasks (decomposing problem-solving steps into sequence prediction, 130k samples), with a 3:1 data ratio; the LLM is frozen while only the vision encoder and perceiver are trained. The supervised fine-tuning stage freezes the vision encoder and trains the perceiver and LLM using the meticulously annotated MathFlow-SFT dataset.
    - **Design Motivation**: EI description training teaches the model to "see" basic elements in diagrams; visual reasoning training teaches the model to "infer" implicit geometric relationships. The two stages optimize different modules separately to avoid mutual interference.

3.  **FlowVerse-CoT-E Evaluation Strategy**:
    - **Function**: Provides robust evaluation of the CoT process.
    - **Mechanism**: Experts pre-construct authoritative solution steps for each problem. These steps are added as hints to the prompt to evaluate model performance at different reasoning depths, with final scores aggregated as $$\text{Score}_{\text{final}} = 0.8 \cdot \frac{1}{N}\sum_{i=1}^N \text{Score}_i + 0.2 \cdot \text{Score}_0$$.
    - **Design Motivation**: MathVerse-CoT-E relies on GPT to decompose model answers into steps for evaluation, which introduces noise; FlowVerse-CoT-E uses human-annotated steps for more stable evaluation.

### Loss & Training

A learning rate of 1e-5 is used for the multi-task pre-training stage, and 5e-6 for the supervised fine-tuning stage, utilizing DeepSpeed Zero2. Different modules are frozen in each stage to ensure that perception-related components and language reasoning components are optimized independently.

## Key Experimental Results

### Main Results

On the FlowVerse benchmark (CoT-E metric):

| Model | All | Text Centric | Vision Dense | Vision Primary |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | 53.8 | 60.1 | 45.0 | 48.1 |
| MathFlow*_Qwen2.5-VL-7B | **57.0** | **62.0** | **49.0** | **52.0** |
| GPT-5 | 65.8 | 74.3 | 53.8 | 60.3 |
| MathFlow*_GPT-5 | **66.5** | **74.6** | **58.2** | **69.4** |

Structural description quality evaluation (F1):

| Model | EI (F1) | RP (F1) |
| :--- | :--- | :--- |
| GPT-4o | 87.7% | 70.1% |
| Claude-sonnet | 89.1% | 72.8% |
| MathFlow-P-7B | **97.2%** | **85.6%** |

### Ablation Study

| Configuration | FlowVerse† Accuracy | Note |
| :--- | :--- | :--- |
| Direct reasoning | Baseline | No perception enhancement |
| + MathFlow-P-7B (EI only) | +2-4% | Essential info extraction is effective |
| + MathFlow-P-7B (EI+RP) | +4-8% | RP inference provides extra gain |
| Various reasoning models | Consistent | Pipeline possesses strong generality |

### Key Findings

- Model accuracy actually improves after removing images (Qwen2-VL-72B +4.3%), indicating that images often act as a source of interference rather than information—perception is the true bottleneck.
- The Vision Primary version (both EI and RP in images) shows an average drop of over 10 percentage points compared to the Text Centric version (full text), proving models are "better at reading text than looking at images."
- MathFlow-P-7B achieves an EI F1 of 97.2%, far exceeding GPT-4o (87.7%), demonstrating the value of specialized perception models.
- The pipeline is effective for both open-source and closed-source reasoning models, indicating the universality of perception enhancement.

## Highlights & Insights

- The **evaluation design using information decomposition and controlled variables** is very ingenious. By combining DI/EI/RP/OQ differently, it accurately quantifies the contribution of each information type and modality to the final result. This methodology can be transferred to the evaluation design of other multimodal tasks.
- The conclusion that **"perception is the bottleneck"** is counter-intuitive but well-supported by data. Consistent observations across various models that removing images improves accuracy provide a strong experimental basis for the "perception then reasoning" separation strategy.
- The **7B perception model surpassing GPT-4o** illustrates that specialized training on specific sub-tasks can significantly outperform general-purpose large models. This idea is transferable to fields such as scientific diagram understanding and medical image analysis.

## Limitations & Future Work

- The manual annotation cost for RP is high, limiting data scale and domain expansion.
- The evaluation set mainly consists of Chinese and English exam questions, with limited coverage of university-level or applied mathematics.
- The two-stage pipeline introduces additional inference overhead (requiring two model calls), which needs to be considered for efficiency in real-time scenarios.
- MathFlow-P-7B is based on Qwen2-VL-7B; whether using a larger base model would provide further improvement is worth exploring.

## Related Work & Insights

- **vs MathVerse**: MathVerse primarily eliminates textual redundancy to ensure models must look at images but lacks fine-grained decoupling of perception vs. reasoning; FlowVerse achieves more detailed diagnosis through its four-component, six-variant design.
- **vs End-to-end Visual Math Models (e.g., VLM-R1)**: End-to-end methods are simple but suffer from perception-reasoning coupling, making them difficult to optimize independently; MathFlow's modular design allows perception and reasoning to iterate and upgrade separately.
- **vs Tool-assisted Reasoning**: Tool-assisted methods (e.g., AlphaGeometry) assume the input has been correctly parsed; this paper points out that parsing itself is the bottleneck.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The information decomposition and modular pipeline ideas are clear, though "separating perception and reasoning" is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The benchmark is well-designed, evaluations cover a large number of models and multiple datasets, and ablations are sufficient.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, though the high number of tables may increase the reading burden.
- **Value**: ⭐⭐⭐⭐⭐ Both the FlowVerse benchmark and the MathFlow pipeline provide practical value to the field of visual mathematical reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](../../CVPR2026/multimodal_vlm/virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)
- [\[ACL 2026\] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](do_mllms_understand_pointing_benchmarking_and_enhancing_referential_reasoning_in.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs](regate_learning_faster_and_better_with_fewer_tokens_in_mllms.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)

</div>

<!-- RELATED:END -->
