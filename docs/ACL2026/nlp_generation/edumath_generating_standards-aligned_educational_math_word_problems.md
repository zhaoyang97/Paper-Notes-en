---
title: >-
  [Paper Note] EDUMATH: Generating Standards-aligned Educational Math Word Problems
description: >-
  [ACL 2026][Text Generation][Math Word Problem Generation] The authors systematize the task of "generating math word problems (MWP) aligned with K-12 curriculum standards." They collected STEM, a training dataset of 11…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Math Word Problem Generation"
  - "Standards Alignment"
  - "Teacher Annotation"
  - "KTO"
  - "ModernBERT Filtering"
date: 2026-05-08
content_hash: 3f21da25ffa6ca30
---

# EDUMATH: Generating Standards-aligned Educational Math Word Problems

**Conference**: ACL 2026  
**arXiv**: [2510.06965](https://arxiv.org/abs/2510.06965)  
**Code**: https://github.com/bryanchrist/EDUMATH  
**Area**: Text Generation / Education  
**Keywords**: Math Word Problem Generation, Standards Alignment, Teacher Annotation, KTO, ModernBERT Filtering

## TL;DR
The authors systematize the task of "generating math word problems (MWP) aligned with K-12 curriculum standards." They collected STEM, a training dataset of 11,000+ MWPs annotated by real U.S. teachers, and trained two open-source SOTA generators, EDUMATH-12B/30B, using SFT + KTO + ModernBERT filtering. The first RCT conducted with real students in grades 3-5 revealed that while students had comparable accuracy on LLM-generated vs. human-written problems, they **almost unanimously preferred customized LLM problems**.

## Background & Motivation

**Background**: Math word problems (MWP) are core assessment tools in K-12 mathematics. Customizing problems based on student interest and ability has been widely proven to improve learning outcomes. However, teacher shortages and burnout prevent the creation of such problems for every student, forcing reliance on limited problem banks. While the linguistic capabilities of LLMs make automated MWP generation seem straightforward, prior works (Christ 2024, Ariyarathne 2025) found large gaps between LLMs and educational-grade MWPs.

**Limitations of Prior Work**: (1) **Lack of solutions**: Existing generation works (Ariyarathne 2025, Sun 2025) only produce problem statements without solution steps, making them unusable for teachers/students; (2) **Coarse granularity**: Alignment targets broad topics like "addition/subtraction" rather than specific standards like "single-step addition/subtraction within two digits," preventing fine-grained difficulty control; (3) **Lack of real evaluation**: Previous evaluations relied on LLM self-assessment or college students rather than practicing teachers; (4) **Lack of training data**: Datasets like GSM8K, ASDIV, and SVAMP are not annotated according to K-12 curriculum standards.

**Key Challenge**: Educational MWPs must simultaneously satisfy four conflicting hard constraints: solvability, accuracy, educational appropriateness, and strict standards alignment. Failing any single constraint results in rejection by teachers. Existing LLM training data typically only satisfies 1-2 of these.

**Goal**: (1) Define the "standards-aligned educational MWP generation" task and provide 4 evaluation dimensions; (2) Construct STEM, the first teacher-annotated training set with readable solutions; (3) Train small open-source generators that rival closed-source SOTA; (4) Conduct the first controlled user study on real elementary school students.

**Key Insight**: The authors selected the Virginia SOL (VA SOL) instead of the Common Core, as VA SOL specifies quantifiable difficulty constraints (e.g., numerical ranges, step counts) for each standard, making it better suited for "strict alignment."

**Core Idea**: A "dual teacher + LLM annotation" process was used to filter a subset of ASDIV and 3,012 LLM-synthetic MWPs into a "Meets all criteria" (MaC) set. Then, a three-stage pipeline (SFT $\rightarrow$ KTO $\rightarrow$ ModernBERT) was used to contribute "high-quality data" to small models and "classifier post-processing" to large models.

## Method

### Overall Architecture
A five-stage pipeline: (1) **Annotating ASDIV subset**: 1,025 problems for grades 3-5 were cross-verified across four rounds (education undergraduates $\rightarrow$ K-12 teachers $\rightarrow$ Llama-3.3-70B $\rightarrow$ Gemma-3-27B) to provide VA SOL labels and CoT solutions; (2) **Synthesis + Teacher Annotation**: Llama-3.3-70B generated 3,012 problems across 38 SOL combinations. 1,372 Prolific U.S. teachers annotated each problem across 4 dimensions (at least 2 annotators per problem). Majority voting determined the MaC label, followed by a "quality re-check" by Gemma-3-27B to flip questionable labels; (3) **Training EDUMATH-12B**: Gemma-3-12B-IT underwent SFT on STEM, followed by KTO to align binary preferences on all teacher annotations, and finally added a ModernBERT MaC classifier for rejection sampling; (4) **Training EDUMATH-30B**: The ModernBERT classifier was applied directly to Qwen-3-30B outputs for filtering **without training** the backbone; (5) **Evaluation**: Sampling 250-1000 problems from 8 baselines using Gemma-3-27B as an automated judge (consistent with teacher $\kappa$) and conducting a student RCT.

### Key Designs

1.  **Four-dimension MaC (Meets-All-Criteria) Evaluation Protocol**:
    - **Function**: Decomposes "educational MWP" into four binary judgments: Solvability, Accuracy, Educational Appropriateness, and Standards Alignment.
    - **Mechanism**: Solvability—Is there a unique solution? Accuracy—Is the CoT solution correct and the reasoning readable? Educational Appropriateness—Is it free of grammar errors, conflicting info, and classroom-appropriate for grades 3-5? Standards Alignment—Does it strictly follow VA SOL constraints? MaC is True only if all four are True. Teacher agreement on MaC was $65.5\pm1.7\%$ due to the stringent joint requirement.
    - **Design Motivation**: Traditional MWP evaluation ignores standard alignment or lengthy solutions; MaC pushes the strictness to match real teacher workflows where any single failure leads to rejection.

2.  **STEM Dataset + Teacher $\times$ LLM Dual Filtering**:
    - **Function**: Constructs the first MWP training set (2,577 items) combining teacher annotation, standards alignment, and readable solutions.
    - **Mechanism**: Merges the ASDIV subset (1,025) and teacher-approved MaC synthetic problems (1,552). Asymmetric flipping rules: if Gemma rejects a teacher-approved problem, it is flipped to non-MaC (correcting human oversight); if Gemma approves a teacher-rejected problem, the teacher's judgment stands (trusting human expertise). Token length and readability indices were tracked to match human writing styles.
    - **Design Motivation**: Pure teacher annotation is costly and prone to fatigue; pure LLM annotation misses subtle appropriateness issues. The asymmetric rule extracts the maximum value from both to create the "cleanest" educational subset.

3.  **SFT + KTO + ModernBERT Post-processing Generation Pipeline**:
    - **Function**: Squeezes maximum value from limited high-quality data, allowing a 12B model to rival a 27B model and a 30B model to outperform closed-source models.
    - **Mechanism**: SFT teaches basic "solvability + alignment + CoT" patterns. KTO uses binary MaC labels as samples without needing paired preferences. Finally, a ModernBERT binary classifier (AUC-ROC 0.861) performs rejection sampling during inference.
    - **Design Motivation**: KTO is more robust for binary signals compared to DPO/RLHF. The ModernBERT classifier decouples "training cost" from "quality thresholding," allowing any future LLM to achieve zero-shot improvement by simply applying the filter.

### Loss & Training
SFT: Gemma-3-12B-IT, 5 epochs, $lr=1\times10^{-6}$, batch size 1, 10% warm-up; KTO: 4,039 teacher-annotated samples, 5 epochs, $lr=5\times10^{-6}$, batch size 8, weighted samples based on inverse frequency; ModernBERT: 3,664 rows, 10 epochs, $lr=1\times10^{-5}$, weighted cross-entropy. Evaluation used 8-shot prompts for all models.

## Key Experimental Results

### Main Results
Comparison of 8 models on 4-dimension MaC (evaluated by Gemma-3-27B judge):

| Model | PPL $\downarrow$ | BF1 vs Self | ASDIV BF1 | Q Length | A Length | MaC $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o (API) | 16.1 | 75.2 | 74.2 | 63.5 | 152.3 | 92.8 |
| GPT-4.1 (API) | 16.3 | 75.3 | 74.3 | 64.5 | 150.2 | 92.8 |
| GPT-4.5 (API) | 15.7 | 75.0 | 74.1 | 61.8 | 150.6 | 92.0 |
| Gemma-3-12B-IT | 11.3 | 77.2 | 74.0 | 84.7 | 240.8 | 63.9 |
| Gemma-3-27B-IT | 12.2 | 77.1 | 74.5 | 76.1 | 215.1 | 75.4 |
| Qwen-3-30B-IT | 12.3 | 76.9 | 74.0 | 68.1 | 202.9 | 87.3 |
| Qwen-3-235B-IT | 12.5 | 76.1 | 74.1 | 66.5 | 186.5 | 89.0 |
| **EDUMATH-12B** | **9.5** | 74.5 | 73.8 | 54.9 | 166.7 | 85.9 |
| **EDUMATH-30B** | 12.0 | 76.1 | 73.8 | 60.4 | 163.5 | **94.6** |

**Key Points**: (1) EDUMATH-30B became the new SOTA with 94.6% MaC, surpassing GPT-4.5 (92.0%) and GPT-4o (92.8%); (2) EDUMATH-12B (85.9%) closely approached 30B-class models with only 12B parameters; (3) EDUMATH-12B achieved the lowest PPL (9.5) and a question length (54.9) closest to human writing (53.9); (4) BERTScore indicated that EDUMATH's diversity is comparable to human-written levels.

### Ablation Study (EDUMATH-12B Pipeline)

| Stage | MaC % |
| :--- | :--- |
| Gemma-3-12B-IT base | 63.9 ($\pm1.5$) |
| + SFT on STEM | 76.2* (+12.3) |
| + KTO | 81.0* (+4.8) |
| + ModernBERT Filtering (Final EDUMATH-12B) | **85.9*** (+4.9) |

Every stage showed significant improvement ($p < 0.01$). EDUMATH-30B was the only model to exceed 90% MaC across all 8 math topics.

### Key Findings
- **Three-stage pipeline is essential**: SFT provides basic imitation, KTO calibrates binary preferences, and ModernBERT provides a safety net, enabling the leap from 63.9% to 85.9%.
- **Filter utility for un-trained models**: Applying the ModernBERT filter to Qwen-3-30B directly achieved SOTA, proving the model-agnostic nature of the proposed annotation value.
- **Error Analysis**: The dominant failure mode for all models was Accuracy (lengthy or flawed reasoning), accounting for 30-60% of errors; Educational Appropriateness was nearly saturated.
- **Student RCT Conclusion**: 94 students across two schools tested EDUMATH vs. human problems. While accuracy was similar, **students significantly preferred customized LLM problems** (11/12 students in the School 2 personalized scenario chose LLM problems) because they "liked the topics." This provides controlled evidence for the "customization $\rightarrow$ engagement" link.

## Highlights & Insights
- **Standard-level Alignment**: Moving from "topic-level" to "standard-level" alignment using VA SOL makes the task quantifiable and scalable for future K-12 LLMs.
- **KTO + ModernBERT**: This combination is optimal for education where only binary preference data is available. It is stable to train and requires minimal compute.
- **First Student RCT for MWP Generation**: By conducting blind tests with actual students, the study proves LLMs can produce content that is both competent and more engaging than generic human-written prompts.
- **Asymmetric Flipping Rule**: Absorbing human and LLM strengths separately is a clever data-cleaning strategy that boosts label SNR without increasing cost.

## Limitations & Future Work
- Only covers grades 3-5 math and English SOL; expanding to K-2/6-12 or other languages requires new annotations.
- Restricted to text-only problems; multimodal MWP generation involving diagrams remains unexplored.
- Prompt engineering was not the focus, and closed-source models might perform better with their own specific prompts.
- Evaluation relies on a single model judge (Gemma-3-27B), which may introduce systematic preference bias.
- RCT sample size ($n=94$) is relatively small and limited to two schools.

## Related Work & Insights
- **vs. MATHWELL (Christ 2024)**: They aligned to interest but not standards; Ours adds the Standards Alignment dimension and expands data/model scale.
- **vs. Mathwizards / Sun 2025**: They generate topic-aligned problems without solutions; EDUMATH outputs problem-solution pairs.
- **vs. GSM8K / ASDIV**: These train solvers; Ours uses STEM to train generators—the other side of the educational LLM coin.
- **vs. OpenAI GPT**: While closed models are strong (92%+), EDUMATH-30B is more balanced across topics and better matches human writing style.
- **Insight**: Any domain requiring "strict standardized output + scarce labels" (e.g., medical QA, legal contracts) can adopt this "SFT+KTO+Classifier" paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Standard-level alignment plus the KTO+ModernBERT combination forms a robust, reproducible pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High-density experiments including 11k teacher labels, 8 model comparisons, and real-world student RCT.
- **Writing Quality**: ⭐⭐⭐⭐ Five-phase narrative is clear; Table 1-3 are information-dense.
- **Value**: ⭐⭐⭐⭐⭐ Open-sourcing the data, model, and evaluation tools provides immediate social value for reducing teacher burnout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)
- [\[ACL 2026\] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation](losses_that_cook_topological_optimal_transport_for_structured_recipe_generation.md)
- [\[ACL 2026\] Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models](investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu.md)
- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)

</div>

<!-- RELATED:END -->
