---
title: >-
  [Paper Note] DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation
description: >-
  [ECCV 2024][Code Intelligence][Synthetic Data Generation] Proposes using code generation to synthesize structured visual data (slides and UIs) to train understanding models, thereby reducing the need for manual annotation.
tags:
  - "ECCV 2024"
  - "Code Intelligence"
  - "Synthetic Data Generation"
  - "Slide Understanding"
  - "User Interface Understanding"
  - "Code Generation"
  - "Structured Vision"
date: 2026-05-08
content_hash: 8dab8edfb82832ac
---

# DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation

**Conference**: ECCV 2024  
**arXiv**: [2410.00201](https://arxiv.org/abs/2410.00201)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: Synthetic Data Generation, Slide Understanding, User Interface Understanding, Code Generation, Structured Vision

## TL;DR

Proposes using code generation to synthesize structured visual data (slides and UIs) to train understanding models, thereby reducing the need for manual annotation.

## Background & Motivation

Slides and user interfaces (UIs) are extremely common structured visual content in daily digital interactions. Machine understanding of such content is crucial for assisting individuals with disabilities in using digital tools. However, existing structured visual understanding methods typically rely on large-scale manually collected and annotated data, a process that is both time-consuming and labor-intensive.

The core limitations of prior work lie in: (1) Real slide and UI data involve privacy and copyright issues, making scale-up acquisition difficult; (2) Annotating structured visual content requires detailed element-level annotations (e.g., element type, position, hierarchical relationship, etc.), which is extremely costly; (3) Existing vision-language models have limited capability in understanding structured layouts.

The Key Insight of ours is highly elegant—generating synthetic structured visual data via code generation. Since slides and UIs are essentially defined by code or markup languages, a large amount of synthetic data with precise labels can be generated programmatically. This approach fundamentally resolves the data annotation bottleneck, as synthetic data naturally carries complete structured labels.

## Method

### Overall Architecture

The core pipeline of DreamStruct consists of three stages: (1) utilizing large language models (LLMs) to generate code describing slide or UI layouts; (2) executing the generated code to render synthetic structured visual images while automatically acquiring labels for all elements; (3) mixing synthetic data with a small amount of real annotated data to train downstream understanding models.

### Key Designs

1. **Code-Driven Synthetic Data Generation**:
    - **Function**: Automatically generate structured visual data with complete labels
    - **Mechanism**: Utilize LLMs to generate HTML/CSS or PPT script code. Executed code produces visual images, and structural information in the code is directly used as labels. By controlling code templates and parameters, diverse layout and style variations can be generated.
    - **Design Motivation**: Structured visual content is fundamentally defined by code, so the code itself is the annotation, eliminating the need for extra manual labeling.

2. **Few-Shot Guided Strategy**:
    - **Function**: Ensure the distribution of synthetic data aligns with real data
    - **Mechanism**: Use a small number of real annotated samples as references to guide the LLM to generate code similar in style and structure to real data. Through few-shot prompting, the generated synthetic data closely approaches real-world scenarios in visual appearance and structural complexity.
    - **Design Motivation**: Purely randomly generated synthetic data might have too large a domain gap from the real distribution. Guidance from a limited number of real samples significantly enhances the quality and utility of the synthetic data.

3. **Multi-Task Evaluation Framework**:
    - **Function**: Validate the method's effectiveness across three core tasks
    - **Mechanism**: Cover three tasks—element recognition, content description, and type classification—to comprehensively evaluate the benefits of synthetic data on downstream tasks.
    - **Design Motivation**: A single task may not fully reflect the value of synthetic data; multi-task evaluation better validates the generalizability of the method.

### Loss & Training

Regarding the training strategy, a hybrid training paradigm is adopted: a large volume of synthetic data is mixed with a small amount of real annotated data in a certain ratio for training. Synthetic data provides a high diversity of samples for feature learning, while real data helps the model calibrate to the real distribution. The specific ratio is adjusted based on validation set performance.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Slide Element Recognition | mAP | Significant improvement | Trained only on real data | +8-15% |
| UI Element Recognition | mAP | Significant improvement | Trained only on real data | +5-12% |
| Content Description | CIDEr | Clear improvement | Trained only on real data | +10-20% |
| Content Classification | Accuracy | Significant improvement | Trained only on real data | +3-8% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Real Data Only | Baseline | Limited data volume |
| Synthetic Data Only | Lower than mixed | Domain gap exists |
| Synthetic + Real Mix | Optimal | Complementary |
| Different Synthetic Data Volumes | Improves with volume | Larger synthetic data volume is better, but with a saturation point |

### Key Findings

- Code-generated synthetic data can effectively substitute for a large amount of manual annotation, with particularly significant improvements in low-annotation scenarios.
- Hybrid training with both synthetic and real data consistently outperforms using either data source alone.
- The method is effective in both the slide and UI domains, demonstrating excellent generalizability.
- The diversity of synthetic data is crucial for the generalization capability of the model.

## Highlights & Insights

- The core insight is highly elegant: structured visual content naturally corresponds to code, making the code the annotation itself.
- Using the code generation capabilities of LLMs to address data annotation issues in computer vision is an ingenious cross-modal methodological transfer.
- The method is simple and practical, requiring no complex model architecture innovations, instead addressing the problem from a data perspective.
- It holds direct practical value for the field of assistive technology.

## Limitations & Future Work

- There is still room for improvement in the visual realism of the synthetic data; the generated slides and UIs may not match real-world scenarios in visual complexity.
- The diversity of code generation is bounded by LLM capabilities and prompt design.
- The applicability of the method to more complex structured documents (e.g., academic papers, table-dense reports) has not been explored.
- Future work could consider introducing adversarial training or domain adaptation techniques to further narrow the synthetic-to-real domain gap.

## Related Work & Insights

- **Document Understanding**: The LayoutLM series learns document layout understanding through pre-training, but relies heavily on large amounts of annotated data.
- **Synthetic Data**: Works like SynthText have verified the effectiveness of synthetic data in the OCR domain; DreamStruct extends this idea to structured visual understanding.
- **LLM-Assisted Data Generation**: An increasing number of studies utilize LLMs to generate training data. DreamStruct provides a structured data-generation paradigm through code generation.
- **Insights**: The code generation paradigm can be extended to any visual content that can be defined by code (such as charts, flowcharts, web pages, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of "code as annotation" is simple and elegant, and the entry point is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ The evaluation across three tasks and two domains is relatively thorough, though it lacks details on quantitative ablated parameters.
- **Writing Quality**: ⭐⭐⭐⭐ The paper features clear logic and fully elaborated motivation.
- **Value**: ⭐⭐⭐⭐ It holds direct practical value for the assistive technology field, and the method exhibits good scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](../../ICLR2026/code_intelligence/diffucoder_understanding_and_improving_masked_diffusion_models_for_code_generati.md)
- [\[ICLR 2026\] TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning](../../ICLR2026/code_intelligence/tikzilla_scaling_text-to-tikz_with_high-quality_data_and_reinforcement_learning.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[ACL 2026\] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](../../ACL2026/code_intelligence/qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)
- [\[NeurIPS 2025\] Learning From Design Procedure To Generate CAD Programs for Data Augmentation](../../NeurIPS2025/code_intelligence/learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)

</div>

<!-- RELATED:END -->
