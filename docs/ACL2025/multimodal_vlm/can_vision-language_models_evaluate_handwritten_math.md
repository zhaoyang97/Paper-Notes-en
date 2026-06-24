---
title: >-
  [Paper Note] Can Vision-Language Models Evaluate Handwritten Math?
description: >-
  [ACL 2025][Multimodal VLM][Handwritten Math Evaluation] This paper proposes the FERMAT benchmark to systematically evaluate the error detection, localization, and correction capabilities of 9 VLMs on handwritten mathematical content. Using 609 manually curated Grade 7-12 math problems alongside over 2,200 handwritten erroneous solutions (covering computation, conceptual, notation, and formatting errors), the evaluation reveals that Gemini-1.5-Pro achieves the highest correcti…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Handwritten Math Evaluation"
  - "Error Detection"
  - "Error Localization"
  - "Error Correction"
  - "VLM Benchmark"
date: 2026-05-08
content_hash: 0785e5852bf9c691
---

# Can Vision-Language Models Evaluate Handwritten Math?

**Conference**: ACL 2025  
**arXiv**: [2501.07244](https://arxiv.org/abs/2501.07244)  
**Code**: [AI4Bharat/FERMAT](https://github.com/AI4Bharat/FERMAT)  
**Area**: Multimodal VLM  
**Keywords**: Handwritten Math Evaluation, Error Detection, Error Localization, Error Correction, VLM Benchmark

## TL;DR
This paper proposes the FERMAT benchmark to systematically evaluate the error detection, localization, and correction capabilities of 9 VLMs on handwritten mathematical content. Using 609 manually curated Grade 7-12 math problems alongside over 2,200 handwritten erroneous solutions (covering computation, conceptual, notation, and formatting errors), the evaluation reveals that Gemini-1.5-Pro achieves the highest correction rate of 77%, though all models still face significant challenges when processing handwritten content.

## Background & Motivation
VLMs hold immense potential in education, particularly in automated grading of handwritten math assignments. OpenAI previously demonstrated a GPT-4 demo evaluating handwritten math, drawing widespread attention. However, several critical issues remain:

**Lack of Systematic Evaluation**: Although VLMs have made progress in mathematical reasoning, comprehensive research on their ability to evaluate handwritten mathematical content is still lacking.

**Limitations of Prior Work**: Existing multimodal evaluation benchmarks focus on simple scenarios with printed text and images, or only handle single-line mathematical expression OCR, failing to address multi-line handwritten derivations and complex mathematical notation.

**Specific Challenges of Handwritten Content**: Varied handwriting styles, inconsistent writing quality, and diverse image conditions pose additional challenges for VLMs.

Key Challenge: Although VLMs claim to possess visual understanding capabilities, how well do their reasoning and evaluation capabilities actually perform when faced with highly varied handwritten mathematical content in real-world educational scenarios?

Core Idea: **Build a handwritten math error evaluation benchmark based on educational scenarios to systematically test the "detect $\rightarrow$ localize $\rightarrow$ correct" error-capability chain of VLMs through controlled perturbation and manual handwritten transcription.**

## Method

### Overall Architecture
The construction of FERMAT consists of four stages:
1. Problem collection (math textbooks + competition problems)
2. Designing a perturbation taxonomy (5 error axes)
3. Human-AI collaborative perturbation generation (GPT-4o generation + manual verification)
4. Handwritten transcription (43 annotators + quality audit)

### Key Designs

1. **Problem Collection and Processing**:

    - Hand-collected approximately 850 mathematical problems with detailed step-by-step solutions from Grade 7-12 textbooks.
    - Covers 7 major areas (Arithmetic, Algebra, Geometry & Measurement, Geometry, Probability & Statistics, Trigonometry, and Calculus) and over 50 sub-topics.
    - Additionally collected competition MCQs focusing on practical mathematics (e.g., profit/loss, time/work, data interpretation).
    - Utilized GPT-4o to convert the problem images to LaTeX format, followed by manual validation, yielding 609 high-quality LaTeX pairs (Q, A_gold).

2. **Perturbation Taxonomy (5 Error Axes)**:

    - **Calculation Errors (CO, 611 cases)**: Final numerical error, intermediate calculation error, non-propagated step error, propagated step error, transcribing error.
    - **Conceptual Errors (CP, 609 cases)**: Theorem misuse, misinterpreting the question, invalid assumption, obviously incorrect facts, formula misuse.
    - **Notation Errors (NO, 255 cases)**: Notation errors (x²→x2), operator swap (+→×), misplaced brackets.
    - **Formatting Errors (PR, 429 cases)**: Ignoring formatting requirements, term swap, incorrect logical order, contextual substitution, variable naming error, units error.
    - **Surface Perturbations (SU, 340 cases)**: Modifications that do not affect correctness (variable name changes, omitted steps, irrelevant information addition), used to test whether VLMs false-positive correct solutions.

3. **Human-AI Collaborative Perturbation Generation**:

    - Used GPT-4o to perturb correct solutions based on perturbation types, instructions, and 3 in-context examples.
    - Manually verified all perturbed outputs: checked if the perturbation aligned with the specified category, if the reasoning was logical, and if the final answer was correctly altered.
    - Further classified perturbations into real errors or surface variations.

4. **Handwritten Transcription and Validation**:

    - Hand-transcribed by 43 annotators from diverse demographic backgrounds.
    - Utilized different paper types, pen colors, and ink types.
    - Captured photos via mobile phones and uploaded them to a centralized platform.
    - Recorded metadata: readability, image orientation, and overall quality.
    - Developed dedicated validation tools for quality auditing.

5. **Evaluation Task Design (Ascending Difficulty)**:

    - **Error Detection (ED)**: Determine if an error exists in the image (binary) and provide a reasoning process.
    - **Error Localization (EL)**: Identify the specific line where the error occurs (harder than ED).
    - **Error Correction (EC)**: Output the complete corrected LaTeX solution, formulation being the most challenging.
    - Two variants exist for each task: processing the handwritten image directly, or performing OCR first and then processing (+OCR variant).
    - **Cascade Setting**: Sequential execution of ED $\rightarrow$ EL $\rightarrow$ EC, where the output of the previous stage serves as input for the next.

### Loss & Training
As this work introduces an evaluation benchmark, it does not involve model training.
- ED uses Balanced Accuracy (BACC) to account for class imbalance (positive vs. negative samples).
- EL and EC employ GPT-4o as the evaluator, achieving a 94% agreement rate with human evaluation.
- All models use the same prompt and a temperature of 0 to ensure reproducibility.

## Key Experimental Results

### Main Results

| Model | ED(BACC) | ED+OCR | EL(ACC) | EL+OCR | EC(ACC) | EC+OCR | Cascade |
|--------|------|------|------|------|------|------|------|
| Gemini-1.5-Pro | 0.63 | 0.67 | 0.43 | 0.56 | 0.76 | **0.77** | 0.50 |
| GPT-4o | **0.65** | 0.64 | **0.45** | 0.50 | 0.66 | 0.71 | 0.45 |
| Llama-3.2-90B | 0.52 | 0.62 | 0.18 | 0.41 | 0.25 | 0.57 | 0.31 |
| Phi-3.5-VI | 0.52 | 0.51 | 0.06 | 0.09 | 0.15 | 0.12 | 0.11 |

### Ablation Study

| Setting | GPT-4o ED(BACC) | Description |
|------|---------|------|
| Base | 0.658 | Basic prompt |
| L1 | 0.670 | Add grade/area/sub-domain |
| L2 | 0.676 | L1 + all perturbation descriptions and examples |
| L3 | 0.691 | L1 + specific perturbation category |
| L4 | 0.702 | L3 + erroneous solution examples and explanations |

### Key Findings
- **Gemini-1.5-Pro is strongest in error correction (77%)**, while GPT-4o performs best in detection and localization.
- **The OCR step is generally beneficial**: Pixtral-124B and Llama-3.2-90B show significant improvement with OCR (strong OCR capability compensates for weaker multimodal reasoning), while GPT-4o and Gemini-1.5-Pro yield marginal gains (due to their already strong intrinsic multimodal comprehension).
- **The cascade setting unexpectedly leads to performance drops**: This is primarily because conservative detection behavior during the ED stage filters out a large number of images.
- **Additional information indeed aids VLMs**: From L1 to L4, the ED performance of GPT-4o increases from 0.658 to 0.702.
- **Handwritten content remains the core challenge**: Replacing handwritten images with printed LaTeX-rendered images or direct text inputs consistently boosts performance, with the most substantial gain occurring when switching from image to text input.

## Highlights & Insights
- Fills the gap in evaluating VLMs on handwritten mathematics, keeping closely aligned with real-world educational scenarios.
- Outlines a comprehensive perturbation taxonomy design, incorporating the key category of "surface perturbations" to test false positive rates.
- Features handwriting diversity from 43 annotators, ensuring the ecological validity of the benchmark.
- Proposes an ascending task difficulty design (ED $\rightarrow$ EL $\rightarrow$ EC) which clearly exposes model bottlenecks within the evaluation chain.
- Conducts comparative experiments with OCR variants, revealing intriguing differences in model behavior: stronger models rely more on end-to-end multimodal understanding, while weaker models benefit more from explicit OCR steps.

## Limitations & Future Work
- The perturbation categories may not be exhaustive, leaving more real-world student error patterns uncovered.
- Mainly focuses on school-level mathematics, without covering more advanced mathematical domains.
- Has not explored multi-agent approaches for error detection.
- Information propagation in cascade settings can introduce error accumulation; more robust multi-step evaluation pipelines warrant investigation.
- Future studies could examine performance disparities of VLMs across different handwriting styles (e.g., neat vs. messy) and varying image qualities.
- Personalized feedback generation (providing pedagogical explanations rather than just detecting errors) is an avenue worth exploring.

## Related Work & Insights
- Conceptual extension of the CheckList framework (Ribeiro et al., 2020): expanding model behavior testing from text models to multimodal math evaluation.
- Related LLM evaluation benchmarks such as FBI, MathCheck, and DUPE: FERMAT uniquely focuses on handwritten visual inputs.
- Multimodal error detection benchmarks like ErrorRadar: FERMAT features more fine-grained perturbations and a broader range of error categories.
- Studies on LLM error correction capabilities (e.g., Li et al., 2024): Text-based LLMs exhibit weak detection yet strong correction abilities; FERMAT observes a similar trend in VLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ Evaluating handwritten mathematics is a practical and under-explored direction, though the overall evaluation framework remains standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 9 VLMs with multiple evaluation strategies and ablation studies, but lacks a fine-grained analysis of different handwriting qualities.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and experimental setups, with highly informative figures/tables.
- Value: ⭐⭐⭐⭐ Holds direct reference value for educational technology applications; the perturbation taxonomy is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ICCV 2025\] Vision-Language Models Can't See the Obvious](../../ICCV2025/multimodal_vlm/vision-language_models_cant_see_the_obvious.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)

</div>

<!-- RELATED:END -->
