---
title: >-
  [Paper Note] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning
description: >-
  [ACL 2025][VLM Reasoning][Multimodal Mathematical Reasoning] This paper proposes leveraging code as a supervisory signal for cross-modal alignment to construct the ImgCode-8.6M dataset consisting of 8.6 million image-code pairs, and the MM-MathInstruct-3M dataset containing 3 million multimodal mathematical instruction-tuning samples. The trained MathCoder-VL achieves State-of-the-Art (SOTA) performance in multimodal mathematical reasoning among open-source models…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal Mathematical Reasoning"
  - "Image-to-Code"
  - "Cross-Modal Alignment"
  - "Data Synthesis"
  - "Geometry Problem Solving"
date: 2026-05-08
content_hash: d1dec4dec48900b6
---

# MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.10557](https://arxiv.org/abs/2505.10557)  
**Authors**: Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, Hongsheng Li (MMLab, CUHK)
**Code**: [https://github.com/mathllm/MathCoder](https://github.com/mathllm/MathCoder)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Mathematical Reasoning, Image-to-Code, Cross-Modal Alignment, Data Synthesis, Geometry Problem Solving

## TL;DR

This paper proposes leveraging code as a supervisory signal for cross-modal alignment to construct the ImgCode-8.6M dataset consisting of 8.6 million image-code pairs, and the MM-MathInstruct-3M dataset containing 3 million multimodal mathematical instruction-tuning samples. The trained MathCoder-VL achieves State-of-the-Art (SOTA) performance in multimodal mathematical reasoning among open-source models, outperforming GPT-4o and Claude 3.5 Sonnet on geometry problems.

## Background & Motivation

### Background
Large Multimodal Models (LMMs) are still far from ideal performance in mathematical reasoning, particularly struggling with tasks that require extracting mathematical information from images for reasoning. The core bottlenecks lie in two aspects: (1) insufficient precision in math-related vision-text cross-modal alignment, and (2) a lack of diverse mathematical figure generation capabilities to scale up training data.

### Limitations of Prior Work
- Traditional image-text description datasets (such as ShareGPT4V, LAION-5B) are primarily oriented towards natural scenes, overlooking the fine-grained details in mathematical graphics.
- Natural language descriptions cannot fully convey all mathematical information in images (such as precise angles, line segment relationships, etc.), nor can they guarantee correctness.
- Existing multimodal math data synthesis methods (e.g., Math-LLaVA, MammoTH-VL) primarily focus on textual diversity, while image diversity lags significantly behind.
- Although works like MAVIS use code to generate geometric figures, they only include three human-designed types, lacking diversity.

### Design Motivation
Code naturally encodes all the information required to generate the corresponding image, establishing a strict one-to-one correspondence between code and images. Leveraging this property allows: (1) achieving precise cross-modal alignment through code-image pairs; (2) automatically synthesizing diverse new mathematical graphics by modifying code parameters.

## Method

### Key Design 1: Iterative Image-to-Code Model (FigCodifier) and ImgCode-8.6M

**Data Collection**: Collect 3 million math-related images, with sources including:
- DaTikZ training set: 119K image-TikZ code pairs (seed data)
- K12 question bank: 1.57 million mathematical question images, covering 19 subjects
- Mathematics textbooks: 202K images extracted from 8K PDFs
- arXiv: 45K images with TikZ code + 681K images without code
- Open-source datasets: MathV360K, MultiMath, etc.

**Iterative Training Process** (Model-in-the-Loop):
1. Train an initial image-to-code model on InternVL-Chat-V1-2-40B using 119K seed data.
2. Use this model to translate the 3 million collected images into code, and execute the code to render new images.
3. Retain only the successfully generated ⟨Image^C, Code⟩ pairs and add them to the dataset.
4. Retrain the model on the expanded dataset (subsequently switching to InternVL2-8B to balance performance and cost).
5. Iterate repeatedly to finally obtain the FigCodifier model.

**TikZ to Python Conversion**: Leverage GPT-4o mini to translate TikZ code into Python code, further expanding data diversity and generating an additional 3.1 million image-Python pairs.

**Data Cleaning**: A five-step cleaning pipeline—code validation (keeping only executable code), deduplication (removing 4.4%), quality filtering (removing 3.7% low-quality data), code length filtering, and image quality check (removing nearly all-white images, ~0.5%). This finally yields 4.3 million TikZ pairs + 4.3 million Python pairs = ImgCode-8.6M.

The code execution success rate increased from an initial 46.5% to a final 81.2% for TikZ and 84.5% for Python.

### Key Design 2: MM-MathInstruct-3M Dataset Construction

**K12-2M**:
- Distinguish mathematical diagrams and formulas (by size) from 4.6 million math questions, and convert formulas into LaTeX text.
- Use GPT-4o mini to expand simple answers into detailed Step-by-Step CoT (Chain-of-Thought) solutions.
- Finally obtain 2 million samples containing actual images.

**Mathematical Questions with Synthesized New Images (New-1M)**:
1. Use FigCodifier (temperature 0.7) to convert the 1.57 million original images in K12-2M to code, generating 1.1 million new image-code pairs.
2. Use Qwen2.5-72B-Instruct to generate K12-difficulty mathematical reasoning problems based on the new images.
3. Solve the problems independently using Qwen2.5-Math-72B-Instruct and Qwen2.5-72B-Instruct.
4. Retain only samples where the solutions from both models are consistent (pass rate of 51%) to ensure answer correctness.
5. Obtain 1 million new samples after cleaning and deduplication.

### Key Design 3: Two-Stage Training

**Stage 1 — Image-to-Code Intermediate Training**:
- Train the vision encoder and MLP projection layer using ImgCode-8.6M.
- Freeze the LLM backbone to preserve general language capabilities.
- Purpose: Enhance the vision encoder's capability to extract mathematical visual features.

**Stage 2 — Mathematical Instruction Tuning**:
- Perform full-parameter fine-tuning using MM-MathInstruct-3M (K12-2M + New-1M).
- Purpose: Enhance the capabilities of multimodal mathematical problem solving.

## Key Experimental Results

### Experiment 1: Main Results

| Model | Parameters | MATH-Vision | MathVerse | MathVista(GPS) | GAOKAO-MM | We-Math(S3) |
|------|--------|------------|-----------|---------------|-----------|-------------|
| GPT-4o | - | 30.4 | 50.8 | 64.7 | - | 43.6 |
| Claude 3.5 Sonnet | - | 37.9 | 49.0 | 64.4 | - | - |
| InternVL2-8B | 8B | 20.0 | 35.9 | 62.0 | 32.5 | 35.2 |
| InternVL2-76B | 76B | 23.6 | 42.8 | 67.8 | 41.2 | 49.1 |
| MAVIS-7B | 7B | 19.2 | 35.2 | 64.1 | - | 34.6 |
| MathGLM-Vision-9B | 9B | 19.2 | 44.2 | 64.4 | - | - |
| **MathCoder-VL-8B** | **8B** | **26.1** | **46.5** | **73.6** | **51.2** | **52.1** |
| Δ vs Base | | +6.1 | +10.6 | +11.6 | +18.7 | +16.9 |

MathCoder-VL-8B achieves SOTA performance among open-source models of the same scale across all 6 benchmarks. On MathVista(GPS), it surpasses GPT-4o by 8.9% and Claude 3.5 Sonnet by 9.2%.

### Experiment 2: Detailed Comparison on Geometry Subset

| Model | Angle | Area | Length | Average |
|------|-------|------|--------|------|
| GPT-4o | 17.3 | 29.8 | 30.1 | 25.7 |
| InternVL2-8B | 20.8 | 22.4 | 20.5 | 21.2 |
| Math-LLaVA-13B | 20.2 | 18.4 | 17.6 | 18.7 |
| **MathCoder-VL-8B** | **48.6** | **32.2** | **32.1** | **37.6** |

On the plane geometry subset of MATH-Vision, MathCoder-VL-8B averages 37.6%, outperforming GPT-4o's 25.7% by 11.9 percentage points, and reaches 48.6% on the Angle subset.

### Experiment 3: Ablation Study

| Image-to-Code Intermediate Training | Fine-Tuning Data | MATH-Vision | MathVerse | MathVista(GPS) | GAOKAO-MM |
|:--:|:--:|:--:|:--:|:--:|:--:|
| ✗ | K12-2M | 20.3 | 27.2 | 45.7 | 30.0 |
| ✓ | K12-2M | 22.0 | 33.0 | 64.4 | 33.8 |
| ✓ | K12-2M+New-1M | 21.7 | 35.4 | 66.4 | 37.5 |

The intermediate training brings a massive improvement of +18.7% on MathVista(GPS); the synthesized new image data further yields +2.4% on MathVerse and +3.7% on GAOKAO-MM.

## Key Findings

1. **Code is a better cross-modal alignment signal than natural language**: The correspondence between code and images is precise and complete, without losing details or introducing errors.
2. **Intermediate training has the most significant impact on visual understanding**: It improves performance on the Vision-Only subset of MathVerse by up to 11.0%, indicating that code-based training significantly enhances pure visual information processing.
3. **Outstanding multi-step reasoning capabilities**: It achieves 52.1% on We-Math's 3-step problems, surpassing GPT-4o's 43.6% and InternVL2-76B's 49.1%.
4. **Synthesized image data boosts generalization**: The diversity brought by new images significantly improves model performance even on tasks different from traditional math questions (with an overall gain of +5.0% on MathVista).
5. **Iterative training significantly improves code generation quality**: The success rate of TikZ code generation increases from 46.5% to 81.2%.

## Highlights & Insights

- **Innovative Cross-Modal Alignment Paradigm**: The first systematic attempt to utilize code as a bridge for image-text alignment, addressing the issue of insufficient precision in natural language descriptions for the mathematical domain.
- **Large-Scale Data Engine**: The 8.6 million image-code pairs form the largest image-to-code dataset to date, scaled up automatically via an iterative model-in-the-loop approach.
- **First Image Synthesis-Driven Math Dataset**: MM-MathInstruct-3M is the first high-quality multimodal mathematical dataset that contains both new questions and newly synthesized images.
- **Breakthrough in Geometric Understanding**: Outperforms GPT-4o and Claude 3.5 Sonnet by a large margin on geometry problems with only 8B parameters.
- **Practical Two-Stage Training**: The intermediate stage freezes the LLM to preserve language capabilities, while the fine-tuning stage performs full parameter training to boost reasoning, showcasing a well-rationalized design.

## Limitations & Future Work

- **Limited Subject Coverage**: Focuses solely on mathematics, without incorporating other STEM disciplines such as physics or chemistry.
- **Single Language**: The dataset is entirely in English, and although it performs well on the Chinese GAOKAO-MM, no Chinese data was explicitly constructed.
- **Restricted Model Scales**: Only 2B and 8B models were trained, leaving the potential of larger models unexplored.
- **No Reinforcement Learning**: Reinforcement learning methods like GRPO were not employed in the post-training phase, potentially leaving room for further improvement.
- **Dependency on Code Generation**: The core pipeline relies heavily on the quality of image-to-code translation, which may still have limitations for highly complex graphics.

## Related Work & Insights

- **Math-LLaVA (Shi et al., 2024)**: Enhances questions via image complexity classification but does not generate new images; MathCoder-VL improves by +10.4% on MATH-Vision.
- **MAVIS (Zhang et al., 2025)**: Generates geometric and function images using code but is limited to three human-designed types; MathCoder-VL automatically generates diverse images via FigCodifier, yielding +6.9% on MATH-Vision.
- **MathGLM-Vision (Yang et al., 2024)**: Reaches 44.2% on MathVerse with 9B parameters; MathCoder-VL-8B achieves 46.5% and comprehensively outperforms it across other benchmarks.
- **Multimath (Peng et al., 2024)**: Achieves 66.8% on MathVista GPS with a 7B model; MathCoder-VL-8B scores 73.6%.
- **Math-PUMA (Zhuang et al., 2024)**: Focuses on geometric graphic understanding but relies on external data; MathCoder-VL leads by a large margin on the geometry subset (average 37.6% vs 13.2%).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Utilizing code as a cross-modal alignment signal is a highly ingenious idea, and the iterative data engine is also an innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated extensively across six benchmarks with detailed ablation studies, though lacking failure case analyses.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with comprehensive explanations of methods and experiments, and intuitive chart designs.
- Value: ⭐⭐⭐⭐⭐ — With data, models, and code all open-sourced, the 8.6 million-scale dataset represents a massive contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](mammoth_vl_multimodal_reasoning.md)
- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](wemath_knowledge_reasoning.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/vlm_reasoning/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ACL 2026\] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning](../../ACL2026/vlm_reasoning/vl-calibration_decoupled_confidence_calibration_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
