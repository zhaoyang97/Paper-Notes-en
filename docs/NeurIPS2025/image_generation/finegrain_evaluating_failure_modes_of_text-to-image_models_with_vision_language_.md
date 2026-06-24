---
title: >-
  [Paper Note] FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges
description: >-
  [NeurIPS 2025][Image Generation][T2I Evaluation] FineGRAIN proposes a structured joint evaluation framework that defines 27 fine-grained failure modes and utilizes a VLM+LLM agentic pipeline to simultaneously evaluate the prompt following capability of text-to-image models and the image understanding capability of vision-language models, revealing the systematic deficiencies of both types of models on specific tasks.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "T2I Evaluation"
  - "VLM Evaluation"
  - "Failure Mode Classification"
  - "Prompt Following"
  - "Joint Evaluation"
date: 2026-05-08
content_hash: fcc792a561ae6ee6
---

# FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges

**Conference**: NeurIPS 2025  
**arXiv**: [2512.02161](https://arxiv.org/abs/2512.02161)  
**Code**: [finegrainbench.ai](https://finegrainbench.ai)  
**Area**: Multimodal VLM  
**Keywords**: T2I Evaluation, VLM Evaluation, Failure Mode Classification, Prompt Following, Joint Evaluation  

## TL;DR
FineGRAIN proposes a structured joint evaluation framework that defines 27 fine-grained failure modes and utilizes a VLM+LLM agentic pipeline to simultaneously evaluate the prompt following capability of text-to-image models and the image understanding capability of vision-language models, revealing the systematic deficiencies of both types of models on specific tasks.

## Background & Motivation
Text-to-Image (T2I) models (such as Flux, Stable Diffusion 3.5) are capable of generating visually stunning images, but frequently fail to accurately capture specific attributes in user prompts (e.g., correct object counts, color binding). At the same time, when VLMs are used as judges for T2I models, they suffer from similar understanding deficiencies—especially in compositional reasoning.

Existing T2I evaluation benchmarks (such as PartiPrompts, DrawBench, T2I-CompBench++) suffer from two core problems:
* **Insufficient granularity**: They mix different types of failures together for evaluation (e.g., grouping counting errors and shape errors into the same "Attribute" category).
* **Lack of joint evaluation for T2I and VLM**: Evaluating either side in isolation fails to reveal common deficiencies in visual understanding.

**Design Motivation**: There is a need for a hierarchical, fine-grained evaluation framework that can diagnose the specific weaknesses of both T2Is and VLMs simultaneously.

## Method

### Overall Architecture
FineGRAIN is an agentic evaluation system with the following workflow:
1. **Failure Mode Ontology**: Defines 11 high-level categories (Scene, Attribute, Relation, Count, Negation...), subdivided into 27 specific failure modes.
2. **Meticulously Crafted Prompts**: 25-30 challenging prompts are hand-written for each failure mode, totaling 750+ prompts.
3. **T2I Generation**: 5 T2I models generate images for each prompt, resulting in 3750+ images (at 1360×768 resolution).
4. **VLM Judging**: VLMs answer questions customized for specific failure modes.
5. **LLM Verdict**: The LLM compares the VLM's answers with the original prompt to provide a Boolean score (indicating whether the failure mode is present), a raw score, and an explanation.

### Key Designs

**Definition of 27 Failure Modes (Partial Examples):**

| High-Level Category | Specific Failure Mode |
|---------|------------|
| Attribute | Color binding, Shape binding, Texture binding, Counts, Scaling, Perspective |
| Human | Action/Motion, Anatomical accuracy, Emotional conveyance, Social relations |
| Text | Text-based, Short text, Long text, Tense+Rendering+Style |
| Adversarial | Opposite relation, Surreal, Abstract concepts |
| Temporal | Human action, Cause-and-effect, Tense variation |

**Failure-mode-specific evaluation prompt design:**
- Specific question templates are designed for each failure mode.
- For example, the "Counts or Multiple Objects" template: `"Count how many [object] are there? Count how many [object] are there?"`
- The LLM automatically generates specific VLM evaluation questions based on the T2I prompt and the failure mode template.
- This hierarchical design endows the evaluation with **programmable difficulty adjustment capability**.

**Three New Capabilities:**
1. **Boolean Score**: Directly determines "whether the T2I model followed the instruction" (a capability VQAScore/CLIPScore lacks).
2. **Objective Human Annotation**: Focuses on prompts with explicit, objective answers (e.g., counts, text rendering) to avoid subjective aesthetic judgments.
3. **Interpretable Scores**: The LLM outputs the reasoning process behind the failure judgment.

**Key Difference from VQAScore:**
- VQAScore feeds the T2I prompt directly to the VLM for judgment, relying on the VLM to determine correctness on its own—this causes the VLM to tend to confirm the accuracy of the prompt.
- FineGRAIN designs customized questions tailored to specific failure modes, avoiding the bias introduced by prompt-guided suggestions in the VLM.

### Loss & Training
FineGRAIN does not involve model training; it is a pure evaluation framework. The model components used are:
- **LLM**: Llama3-70B (for generating evaluation questions + final verdict)
- **VLM**: Molmo-72B (primary), supplemented by InternVL-78B and Pixtral-124B (for comparison)
- **T2I**: Flux-dev, SD3-Medium, SD3-Large, SD3.5-Medium, SD3.5-Large
- Each prompt-image pair is annotated by humans with binary labels (whether the instruction was followed).

## Key Experimental Results

### Main Results—Success Rate of T2I Models Across Failure Modes (Human Evaluation)

| Failure Mode | Flux | SD3.5 | SD3.5-M | SD3-M | SD3-XL |
|---------|------|-------|---------|-------|--------|
| Color Binding | **93.3** | **96.7** | **93.3** | **96.7** | 40.0 |
| Abstract Concepts | **92.3** | 84.6 | 88.5 | 73.1 | 69.2 |
| BG-FG Mismatch | **76.0** | 69.2 | 73.1 | 53.9 | 53.9 |
| Human Action | **72.4** | 69.0 | 27.6 | 13.8 | 44.8 |
| **Counts or Multiple Objects** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| **Long Text Specific** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| Short Text Specific | **64.0** | 48.0 | 24.0 | 20.0 | 0.0 |
| **Average** | **51.0±1.8** | 40.1±1.8 | 30.6±1.7 | 24.3±1.6 | 21.1±1.5 |

**Surprising Discovery**: All T2I models achieve a success rate of **zero** on "Object Counting" and "Long Text Generation"!

### Text Generation Difficulty Gradient

| Model | 3 Tokens | 10 Tokens | 20 Tokens | 50 Tokens |
|------|----------|-----------|-----------|-----------|
| Flux | 0.84 | 0.40 | 0.04 | 0.00 |
| SD3.5-Large | **0.92** | 0.28 | 0.00 | 0.00 |
| SDXL | 0.00 | 0.00 | 0.00 | 0.00 |

### FineGRAIN vs VQAScore Human Agreement Rate

| Metric | Average Agreement Rate |
|------|----------|
| VQAScore-Human | 57.7% |
| **FineGRAIN-Human** | **67.4%** (+10%) |

- FineGRAIN approaches human performance on "Counts" and "Long text".
- VQAScore's agreement with humans is <30% on both short and long text.
- VQAScore achieves its highest agreement in Color binding (84%).

### Key Findings
- **Flux dominates across the board**: With an average success rate of 51.0%, it is significantly higher than the runner-up, SD3.5, which scores 40.1%.
- **Fine-grained analysis is crucial**: Previous work (e.g., GenAIBench) suggested SDXL performs reasonably well on counting tasks, but FineGRAIN reveals that all models fail completely on this task—because prior evaluations mixed counting with other attribute errors.
- **Adjustable difficulty**: Text generation success rate decreases linearly from 0.52 for 3 tokens to 0.00 for 50 tokens; object counting drops from 0.66 for 1 object to 0.03 for 3 objects.
- **VLM Bias**: When the original T2I prompt is displayed to the VLM, the VLM tends to confirm that the image is accurate—which is a fundamental flaw in the VQAScore approach.

## Highlights & Insights
1. **Pioneering Joint Evaluation Framework**: Evaluates both T2I and VLM simultaneously, leveraging the failure of VLMs to calibrate the reliability of T2I evaluations.
2. **27 Failure Mode Ontology**: In contrast to existing benchmarks that only cover four main categories (Scene/Attribute/Relation/Count), FineGRAIN introduces 7 new high-level categories, including Human/Text/Multi-Style/Adversarial/Temporal.
3. **Programmable Difficulty Adjustment**: Achieves fine-grained control over evaluation difficulty through parameterized prompts (such as adjusting object counts and text length).
4. **Boolean Score + Interpretable Reasoning**: Provides a foundation for T2I test-time scaling—generating continuously until FineGRAIN yields a favorable judgment.

## Limitations & Future Work
- Focuses only on open-source models (excluding closed-source SOTAs such as DALL-E 3 and Midjourney).
- The main pipeline only utilizes a single LLM (Llama3-70B); other LLMs might perform better.
- Human annotation agreement is lower for subjective failure modes such as "Surreal".
- The VLM+LLM pipeline itself exhibits failure modes, limiting the evaluation's overall reliability.
- High human annotation costs make it practically challenging to scale to more models and more prompts.

## Related Work & Insights
- Complements the deficiencies of existing benchmarks such as GenAIBench, TIFA, and DSG (which are too coarse-grained or assess only one side).
- Complements VLM compositional reasoning benchmarks such as ConMe—FineGRAIN focuses on the VLM's ability to judge T2I outputs.
- Directly applicable to T2I reward modeling: FineGRAIN's Boolean score can serve as an RLHF signal.
- Inspires the concept of "evaluating the evaluator": calibrating the reliable boundaries of automated evaluation by assessing the accuracy of VLM judgments.

## Rating
- Novelty: ⭐⭐⭐⭐ (The joint evaluation framework and the 27 failure mode ontology are the primary innovations)
- Experimental Thoroughness: ⭐⭐⭐⭐ (5 T2Is + 3 VLMs, human annotations for 3750+ images, multidimensional analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich cases, though some results are scattered in the appendix)
- Value: ⭐⭐⭐⭐ (Establishes a new standard for T2I/VLM evaluation and reveals critical systematic defects)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[CVPR 2025\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](../../CVPR2025/image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)

</div>

<!-- RELATED:END -->
