---
title: >-
  [Paper Note] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation
description: >-
  [ACL 2025][Multimodal VLM][Synthetic Data] The CoSyn framework is proposed to automatically create 400K text-rich images (charts, documents, diagrams, etc.) and 2.7M instruction-tuning data using the code generation capabilities of text-only LLMs. The trained 7B VLM achieves state-of-the-art (SOTA) performance across 7 benchmarks, outperforming GPT-4V and Gemini 1.5 Flash.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Synthetic Data"
  - "Code Guidance"
  - "Text-Rich Images"
  - "VLM Instruction Tuning"
  - "Chart & Document Understanding"
date: 2026-05-08
content_hash: 46b8566336fb9f18
---

# Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation

**Conference**: ACL 2025  
**arXiv**: [2502.14846](https://arxiv.org/abs/2502.14846)  
**Code**: [https://yueyang1996.github.io/cosyn](https://yueyang1996.github.io/cosyn)  
**Area**: Multimodal VLM  
**Keywords**: Synthetic Data, Code Guidance, Text-Rich Images, VLM Instruction Tuning, Chart & Document Understanding  

## TL;DR
The CoSyn framework is proposed to automatically create 400K text-rich images (charts, documents, diagrams, etc.) and 2.7M instruction-tuning data using the code generation capabilities of text-only LLMs. The trained 7B VLM achieves state-of-the-art (SOTA) performance across 7 benchmarks, outperforming GPT-4V and Gemini 1.5 Flash.

## Background & Motivation
**Background**: VLMs perform exceptionally well in natural image understanding but still exhibit significant shortcomings in text-rich images (charts, documents, labels, screenshots, etc.), which require both textual understanding and spatial reasoning.

**Limitations of Prior Work**: High-quality text-rich vision-language data is severely scarce. Existing datasets (e.g., ChartQA, FigureQA) are small, limited in type, and highly templated. Additionally, human annotation is costly, and open-source VLMs generalize poorly to new tasks (e.g., a model trained on millions of images performs poorly on nutrition label QA).

**Key Challenge**: Text-rich images are typically rendered by code (e.g., Matplotlib for charts, HTML for documents, LaTeX for equations). The generating code is inherently a perfect textual representation of the image, yet existing synthetic data methods fail to fully leverage this property.

**Goal**: (a) How to scale up the generation of diverse, high-quality text-rich images and instruction data? (b) Can synthetic data enable VLMs to generalize to completely unseen tasks? (c) How to alleviate annotation bias in current benchmarks?

**Key Insight**: Code is a natural bridge between text-rich images and text. Text-only LLMs can generate code to render images and then use the same code as context to generate QA data, entirely without requiring visual models in the data creation loop.

**Core Idea**: Using code as an intermediate representation, image synthesis and instruction data generation are unified into a code-understanding task for a text-only LLM: $P(I,T|q) = P_{LM}(C|q) \cdot P(I|C) \cdot P_{LM}(T|C)$.

## Method

### Overall Architecture
Input: A natural language query (e.g., "nutrition fact labels"), processed through one of CoSyn's 20 pipelines. Output: Synthetic images and corresponding QA instruction-tuning data. The process consists of 4 steps: (1) Topic Generation — sampling based on personas to determine themes $\rightarrow$ (2) Data Generation — populating specific content $\rightarrow$ (3) Code Generation — generating executable code to render images $\rightarrow$ (4) Instruction Generation — utilizing the code as context for the LLM to generate QA pairs (with CoT reasoning).

### Key Designs

1. **11 Rendering Tools × 20 Pipelines**:

    - **Function**: Covers the generation of 9 major categories of text-rich images.
    - **Mechanism**: Uses Matplotlib/Plotly/Vega-Lite to generate charts; LaTeX/HTML to handle documents and tables; Mermaid/Graphviz to generate diagrams; SVG/Asymptote for vector graphics; Lilypond to generate sheet music; RDKit to draw chemical structures. The same tool can be used for multiple pipelines (e.g., HTML for documents, tables, and charts).
    - **Design Motivation**: To cover the entire spectrum of real-world text-rich images—not limited to charts, but extended to documents, equations, circuit diagrams, chemical structures, etc.

2. **Persona-driven Diversity**:

    - **Function**: Promotes content diversity by introducing personas during the Topic Generation stage.
    - **Mechanism**: Samples from a library of 200K personas (e.g., "a sci-fi novelist who likes alien worlds") to condition the LLM's topic generation. Different personas guide the creation of topics with distinct styles and content.
    - **Design Motivation**: Relying solely on parameters like sampling temperature often leads to repetitive and monotonous synthetic data from LLMs. Personas inspire diverse outputs from varied perspectives.

3. **Code-as-a-Bridge Instruction Generation**:

    - **Function**: Uses code (instead of the image itself) as context to enable text-only LLMs to generate high-quality QA data.
    - **Mechanism**: $P_{LM}(T|C)$ — Code precisely describes all information within the image (data values, labels, layout). The LLM can generate accurate question-answer-explanation triplets based on this code without VLM involvement, avoiding visual hallucinations.
    - **Design Motivation**: Code is a lossless textual representation of the image content, which is more precise than image captions or descriptions. Using code as context allows generating questions that require numerical reasoning (e.g., "calculating the average").

### Loss & Training
Model Architecture: CLIP ViT-L/14 + MLP connector + Mistral-7B, following the Molmo architecture. Two-stage training: (1) Dense caption pre-training, (2) Supervised fine-tuning combining three categories of data (138K evaluation datasets + 1M auxiliary datasets + 400K CoSyn synthetic data). Trained for 60K steps on TPU v3-128 with a batch size of 32.

## Key Experimental Results

### Main Results
Comparison of average scores across 7 text-rich benchmarks:

| Model | ChartQA | DocVQA | InfoVQA | TableVQA | AI2D | TextVQA | ScreenQA | Avg |
|------|---------|--------|---------|----------|------|---------|----------|------|
| GPT-4V | 78.1 | 87.2 | 75.1 | 60.5 | 89.4 | 78.0 | 41.6 | 72.8 |
| Gemini 1.5 Flash | 85.4 | 89.9 | 75.3 | 72.6 | 91.7 | 78.7 | 40.1 | 76.2 |
| Llama 3.2 11B | 83.4 | 88.4 | 63.6 | 51.1 | 91.9 | 73.1 | 87.7 | 77.0 |
| **Ours (7B)** | **86.3** | **90.0** | **70.5** | **65.8** | **91.9** | **82.0** | **80.1** | **80.9** |
| Ours (7B zero-shot) | 80.8 | 82.9 | 59.8 | 64.9 | 83.9 | 72.7 | 78.1 | 74.7 |

### Ablation Study
Ablation of training data combinations (average score):

| Configuration | Avg | Description |
|------|------|------|
| Aux only (1M images) | 58.7 | Auxiliary data does not generalize to evaluation tasks |
| Syn only (400K) | ~72 | Synthetic-only data already approaches GPT-4V performance |
| Aux + Syn | 74.7 | Outperforms GPT-4V (72.8) |
| Eval + Aux | 77.3 | +Aux improves performance by only 1.4% |
| Eval + Syn | 80.3 | +Syn improves performance by 3.6% (much larger effect than Aux) |
| Eval + Aux + Syn (Full) | **80.9** | Best |

### Key Findings
- **400K Synthetic Data > 1M Auxiliary Data**: The marginal contribution of synthetic data (+3.6 pp) is significantly greater than that of auxiliary real data (+1.4 pp), demonstrating that quality and relevance are more important than quantity.
- **7K In-Domain Data Outperforms Million-Scale Training**: On the new NutritionQA task, fine-tuning on just 7K nutrition label samples generated via CoSyn surpasses most open-source VLMs trained on millions of images.
- **Alleviating Annotation Bias**: In ChartQA, the performance gap between machine-generated and human-annotated questions is reduced from 21.8% (without synthetic data) to 14.2%, suggesting that synthetic data helps the model avoid overfitting to templated questions.
- **Selective Benefits of CoT Reasoning**: Effective on tasks requiring multi-step reasoning like ChartQA (+2.5 pp) and TableVQA (+2.2 pp), but slightly degrades performance on DocVQA due to answer formatting bias.
- **More Diverse Synthetic Data**: CoSyn-400K exhibits significantly higher image and text diversity compared to existing datasets like FigureQA and ChartQA.

## Highlights & Insights
- **Elegant Code-as-a-Bridge Formulation**: Since text-rich images are inherently generated by code, leveraging code inversely for data synthesis ensures precise image-text alignment while allowing a text-only LLM to handle the entire pipeline. This concept can be transferred to any visual content rendered by code (e.g., slides, web pages, CAD drawings).
- **Zero-Shot Generalization to New Domains**: Targeted data generation using simple natural language queries addresses the practical "cold-start" problem of VLMs on new tasks. The NutritionQA experiment is particularly impressive.
- **Persona-driven Diversity Strategy**: Simple yet effective, providing a generic practice worth adopting in all synthetic data generation scenarios.

## Limitations & Future Work
- **Dependence on High-Quality Coding LLMs**: Data generation relies on Claude-3.5-Sonnet and GPT-4o-mini; the performance of open-source alternatives remains unknown, and the API cost is not negligible.
- **Unreported Rendering Failure Rate**: Code generated by LLMs does not always compile or render successfully. The success rate and error-handling strategies are not thoroughly discussed.
- **Exclusive Focus on Text-Rich Images**: The impact on other VLM tasks such as natural image or scene understanding remains unexplored.
- **Limitations of Evaluation Metrics**: The authors observe that strict string matching penalizes detailed but correct answers (e.g., marking "Tuesday to Thursday" as incorrect), yet do not propose an alternative evaluation framework.

## Related Work & Insights
- **vs Molmo/PixMo-docs**: Molmo also utilizes synthetic text-rich datasets, but they lack the scale and diversity of CoSyn-400K. Ours adopts Molmo's architecture and pre-training but significantly scales up the synthetic data.
- **vs ChartPali**: ChartPali exhibits a 32.8% gap in ChartQA performance between machine-generated and human-annotated questions, whereas ours reduces this gap to 14.2%, proving that highly diverse synthetic data effectively mitigates bias.
- **vs LLaVA-OneVision**: LLaVA-OneVision has a larger training set but a lower average score (72.4) compared to ours (80.9), indicating that targeted synthetic data is much more effective than raw volume scaling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The code-as-a-bridge perspective is elegant and simple, and persona-driven diversity is also a key highlight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough evaluation across 7 benchmarks, including ablations, new task generalization, bias analysis, diversity quantification, and CoT analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured with rich insights; the NutritionQA case study is highly compelling.
- **Value**: ⭐⭐⭐⭐⭐ Provides both a general framework and a practically usable 400K dataset, significantly advancing the field of text-rich image understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation](enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](../../CVPR2025/multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](../../CVPR2025/multimodal_vlm/synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
