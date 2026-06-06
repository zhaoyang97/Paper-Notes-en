---
title: >-
  [Paper Note] GenExam: A Multidisciplinary Text-to-Image Exam
description: >-
  [ICML 2026][Image Generation][Multidisciplinary Exam] GenExam adopts "drawing exams" as the gold standard to measure the integrated reasoning-understanding-generation capabilities of T2I models. It provides 1…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multidisciplinary Exam"
  - "Text-to-Image Evaluation"
  - "Scoring Points"
  - "MLLM-as-judge"
  - "GPT-Image-1.5"
date: 2026-05-08
content_hash: c1148f809e3569ed
---

# GenExam: A Multidisciplinary Text-to-Image Exam

**Conference**: ICML 2026  
**arXiv**: [2509.14232](https://arxiv.org/abs/2509.14232)  
**Code**: https://github.com/OpenGVLab/GenExam (Yes)  
**Area**: Multimodal VLM / Evaluation Benchmark / Text-to-Image Generation  
**Keywords**: Multidisciplinary Exam, Text-to-Image Evaluation, Scoring Points, MLLM-as-judge, GPT-Image-1.5

## TL;DR
GenExam adopts "drawing exams" as the gold standard to measure the integrated reasoning-understanding-generation capabilities of T2I models. It provides 1,000 questions across 10 disciplines, each paired with ground-truth images and fine-grained scoring points. Results reveal that even the strongest closed-source model, Nano Banana Pro, achieves only a 70.2% strict score, while most open-source T2I and unified MLLMs score below 3%.

## Background & Motivation

**Background**: Multidisciplinary reasoning has been evaluated by benchmarks like MMLU, MMMU, and Humanity's Last Exam, but these are primarily "understanding" tasks. Existing T2I multidisciplinary benchmarks (MMMG, OneIG-Bench, SridBench) focus on "concept illustrations" with loose evaluation criteria, functioning more as "illustrating a concept" rather than "completing a drawing exam."

**Limitations of Prior Work**: Current T2I evaluations suffer from (i) short and vague prompts, (ii) a lack of reference images and scoring rubrics, (iii) shallow knowledge depth without hierarchical classification, and (iv) evaluation protocols relying either on CLIP/VQA scores (which miss subject-specific correctness) or single-sentence MLLM-as-judge instructions (which overlook details). Hard errors, such as incorrect numbers of chemical bonds or geometric tangency relationships, remain undetected.

**Key Challenge**: The essence of multidisciplinary images lies in semantic correctness rather than realism or aesthetics—a single misplaced atom or reversed arrow invalidates the entire image. However, general image evaluation metrics cannot capture such fine-grained accuracy.

**Goal**: (1) Construct a T2I benchmark resembling AP / A-level / IB drawing questions, complete with standard answers, scoring rubrics, and knowledge taxonomy; (2) Design a reliable automated evaluation protocol for semantic correctness and visual plausibility; (3) Systematically expose the performance gap of current T2I and unified MLLMs in disciplinary generation.

**Key Insight**: This work ports exam-grading logic to T2I evaluation. Each question includes not only a prompt and reference image but also a list of "scoring points" (e.g., "Does the molecule contain exactly 8 C atoms?") co-developed by humans and GPT-5. An MLLM treats each scoring point as a VQA task (Yes/No), and final scores are aggregated via weighted summation.

**Core Idea**: Evaluate T2I models as if grading a drawing exam—each image is first assessed for "semantic correctness" based on customized scoring points, then for "visual plausibility" across three 0-2 point categories (spelling, readability, logical consistency), ultimately yielding both strict and relaxed scores.

## Method

### Overall Architecture
GenExam consists of three components: (1) A test bank of 1,000 questions covering 10 primary disciplines (Math, Physics, Chemistry, Biology, CS, Geography, Economics, Music, History, Engineering), structured into a four-level hierarchy (10/40/132/236) based on ISCED-F; (2) Each question is paired with a ground-truth image, 3-14 scoring points (averaging 6.9, weights sum to 1), and an exam-style prompt (averaging 74.8 words); (3) A dual-dimension evaluation protocol—semantic correctness (0-1) and visual plausibility (spelling/logic/readability, 0-2 each), resulting in strict and relaxed final scores.

### Key Designs

1. **Scoring Points**:
    - **Function**: Reduces the vague question of "is the image correct" into a set of deterministic VQA judgments.
    - **Mechanism**: GPT-5 drafts 3-14 Yes/No scoring points per question (e.g., "Does the molecule contain exactly 8 carbons?"), followed by manual refinement. During evaluation, an MLLM judge compares the generated image with the reference and answers Yes/No for each point. Semantic correctness is calculated as $\sum_i s_i \cdot \mathbb{1}[\text{answer}_i=\text{Yes}]$, where the sum of weights $s_i$ is 1.
    - **Design Motivation**: Single-sentence MLLM instructions often miss details like bond counts, geometric relations, or musical notes. Explicitly decomposing key constraints into individual points allows for stable capture of subject-level errors.

2. **Dual-Score Evaluation Protocol (Strict + Relaxed)**:
    - **Function**: Uses two scales to characterize both the "perfect accuracy rate" and the "degree of closeness," avoiding a one-size-fits-all approach.
    - **Mechanism**: The strict score represents the proportion of images that satisfy all scoring points and receive a score of 2 for spelling, logic, and readability. The relaxed score is calculated as $0.7\cdot\text{semantic}+0.1\cdot\text{spell}+0.1\cdot\text{logic}+0.1\cdot\text{read}$ (weights determined by human preference alignment). Strict scores highlight the difficulty where "perfect scores are rare," while relaxed scores differentiate among lower-performing models.
    - **Design Motivation**: Purely strict metrics would result in 0% for most models, losing informative value; purely weighted averages would mask the "all-or-nothing" nature of disciplinary accuracy.

3. **Data Curation Pipeline**:
    - **Function**: Ensures question difficulty, disciplinary coverage, and scoring point quality.
    - **Mechanism**: Keywords are generated via the four-level hierarchy → Web crawling or filtering from MLLM datasets → GPT-5 filters based on text richness, disciplinary density, and complexity → GPT-5 drafts prompts and scoring points → PhD annotators perform manual review and revision. In the final set, Hard accounts for 38%, Medium 38%, and Easy 24%, with prompts ranging from 24-173 words.
    - **Design Motivation**: Web image quality is inconsistent, manual curation is costly, and pure GPT-5 generation can be repetitive. The dual-layer GPT-5 + human review balances scale and rigor.

### Loss & Training
This work presents an evaluation benchmark and does not involve training. The only adjustable component is the MLLM judge (default: GPT-5 with reasoning effort set to low; the appendix validates that alternatives like Gemini-3-Flash remain highly consistent with humans).

## Key Experimental Results

### Main Results
Strict and relaxed scores were tested across 17 models (abridged):

| Model | Type | Strict ↑ | Relaxed ↑ |
|------|------|---------:|----------:|
| Nano Banana Pro | Closed-source | **70.2** | **93.0** |
| GPT-Image-1.5 | Closed-source | 42.5 | 81.5 |
| GPT-Image-1 | Closed-source | 13.1 | 62.2 |
| Seedream 4.5 | Closed-source | 12.3 | 59.5 |
| FLUX.2 max | Closed-source | 8.6 | 61.6 |
| FLUX.2 dev | Open-source T2I | 2.4 | 42.3 |
| Qwen-Image-2512 | Open-source T2I | 1.5 | 35.3 |
| BAGEL (thinking) | Open-source unified MLLM | 0.0 | 12.9 |
| Janus-Pro | Open-source unified MLLM | 0.0 | 9.5 |

Even the strongest closed-source models fail to pass effectively, and most open-source T2I models are nearly decimated. Open-source unified MLLMs score 0 on the strict metric, performing worse than specialized T2I models.

### Ablation Study

| Evaluator | Kendall $\tau$ with Human | Pearson $r$ |
|--------|----------------------:|------------:|
| Relaxed by GPT-5 | **0.675** | **0.844** |
| Relaxed by Gemini-3-Flash | 0.661 | 0.826 |
| Semantic Correctness only | 0.633 | 0.806 |
| VQA Score | 0.145 | 0.179 |
| CLIP Score | 0.116 | 0.165 |

MAE across dimensions: semantic 0.10, spelling 0.11, readability 0.20, logic 0.28. These low values indicate stable evaluation.

### Key Findings
- **Unified MLLMs underperform specialized T2I models**: Open-source unified models like BAGEL and Show-o2 score 0 in strict and lower than FLUX.2 dev in relaxed scores, suggesting that the "single model for understanding + generation" paradigm is yet to succeed for disciplinary imagery.
- **The bottleneck is visual execution, not knowledge**: FLUX.2 dev can correctly identify geographic locations for History questions but fails to render corresponding graphical elements, indicating a lack of "translating knowledge into readable images."
- **CLIP / VQA scores fail completely**: Correlation with humans is near 0.1, showing that traditional T2I metrics cannot capture disciplinary correctness.
- **Open-source focuses on basics**: Open-source models decline sharply in spelling and logical consistency, suggesting that text rendering and coordinate alignment must be addressed before reasoning.

## Highlights & Insights
- **Explicit "Scoring Rubrics" are a generalizable paradigm for LLM/T2I evaluation**: By decomposing vague "correct/incorrect" judgments into structured Yes/No lists, the MLLM judge's MAE becomes controllable, with correlations far exceeding traditional metrics. This approach is applicable to Chart QA, code generation, and math evaluation.
- **The dual-metric design (Strict + Relaxed) is clever**: It highlights the difficulty ceiling (separating top closed-source models) while revealing underlying differences among the majority of models that would otherwise all score 0.
- **The "Exam Perspective" reframes T2I evaluation goals**: While previous T2I evaluations focused on fidelity, aesthetics, and alignment, this shifts the focus toward "correctness and readability," aligning closer to testing "expert-level intelligence" on the AGI path.
- **The data curation protocol is reusable**: The two-tier GPT-5 drafting + human refinement pipeline can be applied to any benchmark requiring scoring criteria.

## Limitations & Future Work
- 1,000 questions across 10 disciplines and 4 levels is still relatively small; some sub-fields (e.g., Music) have only dozens of samples, limiting statistical stability.
- Reliance on cutting-edge closed-source MLLMs like GPT-5 as judges raises concerns regarding long-term reproducibility and cost; although an open-source judge was tested, its correlation with humans decreased.
- Scoring point weights are equally distributed and sum to 1, failing to reflect hierarchical importance (e.g., main structure vs. details).
- The focus is limited to "drawing exams," leaving multidisciplinary visualization tasks in animation, video, and 3D unexplored.

## Related Work & Insights
- **vs MMMU / MMLU / Humanity's Last Exam**: While these are multidisciplinary exams, they only evaluate understanding; GenExam brings the same academic rigor to the generation side.
- **vs MMMG / OneIG-Bench / SridBench**: Compared to other disciplinary T2I benchmarks that emphasize "concept illustration," GenExam features longer prompts, stricter constraints, and finer grading.
- **vs RISEBench / WiScore**: This work adopts strict binary scoring and human-aligned weighting but is the first to extend "customized scoring points" to subject-level evaluation.
- **Transferable Insights**: Making "VQA-style scoring points" a universal interface for model evaluation is applicable to multimodal reasoning, agent benchmarks, and code generation. It also serves as a reminder to unified MLLM researchers: the current disadvantage of unified architectures in generation suggests that "sharing a backbone for understanding + generation" still needs refinement.

## Rating
- Novelty: ⭐⭐⭐⭐ The first multidisciplinary T2I exam benchmark; the scoring-points protocol is a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 17 models across 10 disciplines with dual metrics, includes alignment with 5 humans across 250 questions, and tests multiple evaluators for robustness.
- Writing Quality: ⭐⭐⭐⭐ Clear charts and thorough protocol explanation; the appendix is quite detailed, requiring occasional back-reference for token details.
- Value: ⭐⭐⭐⭐⭐ Provides the first "exam-level" evaluation for the T2I community; likely to become a long-term benchmark for disciplinary capabilities of unified MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)
- [\[ICML 2026\] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)

</div>

<!-- RELATED:END -->
