---
title: >-
  [Paper Note] GenExam: A Multidisciplinary Text-to-Image Exam
description: >-
  [ICML 2026][Image Generation][Multidisciplinary exam] GenExam treats "drawing exams" as the gold standard for evaluating the comprehensive reasoning-understanding-generation abilities of T2I models. It provides 1…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multidisciplinary exam"
  - "text-to-image evaluation"
  - "scoring points"
  - "MLLM-as-judge"
  - "GPT-Image-1.5"
date: 2026-05-08
content_hash: 2c179d29a6096a2b
---

# GenExam: A Multidisciplinary Text-to-Image Exam

**Conference**: ICML 2026  
**arXiv**: [2509.14232](https://arxiv.org/abs/2509.14232)  
**Code**: https://github.com/OpenGVLab/GenExam (available)  
**Area**: Multimodal VLM / Benchmarking / Text-to-Image Generation  
**Keywords**: Multidisciplinary exam, text-to-image evaluation, scoring points, MLLM-as-judge, GPT-Image-1.5

## TL;DR
GenExam treats "drawing exams" as the gold standard for evaluating the comprehensive reasoning-understanding-generation abilities of T2I models. It provides 1,000 questions across 10 disciplines, each with a ground-truth image and fine-grained scoring points. Even the strongest closed-source model, Nano Banana Pro, achieves only 70.2% strict score, while most open-source T2I/unified MLLMs score below 3%.

## Background & Motivation

**Background**: Multidisciplinary reasoning benchmarks such as MMLU, MMMU, and Humanity's Last Exam focus on understanding tasks ("reading comprehension"). On the T2I side, multidisciplinary benchmarks (MMMG, OneIG-Bench, SridBench) mainly assess "concept illustration" with loose criteria, akin to "illustrate a concept" rather than "complete a drawing exam question."

**Limitations of Prior Work**: Existing T2I evaluations (i) use short, broad prompts, (ii) lack reference images and detailed scoring rubrics, (iii) cover shallow knowledge without hierarchical categorization, and (iv) rely on CLIP/VQA scores (which miss subject correctness) or single-sentence MLLM-as-judge instructions (missing many details). As a result, hard errors like "drawing the wrong number of chemical bonds" or "incorrect circle-tangent relationships" are not captured.

**Key Challenge**: The crux of multidisciplinary images is not realism or aesthetics, but semantic correctness—one wrong atom or a reversed arrow renders the image invalid. Generic image evaluation metrics cannot capture such fine-grained correctness.

**Goal**: (1) Construct a T2I benchmark with standard answers, scoring rubrics, and knowledge categorization, akin to AP/A-level/IB drawing questions; (2) Design an automated evaluation protocol that reliably judges semantic correctness and visual plausibility; (3) Systematically reveal the real gaps in subject-specific generation abilities of current T2I/unified MLLMs.

**Key Insight**: Transfer exam grading logic to T2I evaluation—each question comes with a prompt, reference image, and a list of "scoring points" (e.g., "Does the molecule contain exactly 8 carbon atoms?") jointly crafted by humans and GPT-5. Each scoring point is treated as a VQA task for the MLLM to answer Yes/No, with final scores aggregated by weighted sum.

**Core Idea**: Evaluate T2I models as if grading drawing exams—each image is scored for "semantic correctness" via customized scoring points, and for "visual plausibility" via three 0-2 sub-scores (spelling, readability, logical consistency), yielding both strict and relaxed scores.

## Method

### Overall Architecture
GenExam consists of three main components: (1) a question bank of 1,000 items covering 10 primary disciplines (math, physics, chemistry, biology, computer science, geography, economics, music, history, engineering), organized into a four-level ISCED-F taxonomy (10/40/132/236); (2) each question is paired with a ground-truth image, 3-14 scoring points (average 6.9, weights sum to 1), and an exam-style prompt averaging 74.8 words; (3) a dual-dimension evaluation protocol—semantic correctness (0-1) and visual plausibility (spelling/logic/readability, each 0-2)—yielding strict and relaxed final scores.

### Key Designs

1. **Scoring Points Rubric**:

    - **Function**: Reduces the ambiguous "is the image correct" question to a set of definite VQA judgments.
    - **Mechanism**: For each question, GPT-5 drafts 3-14 yes/no scoring points (e.g., "Does the molecule contain exactly 8 carbons?"), which are refined by human annotators. During evaluation, the MLLM judge reviews the generated and reference images, answering Yes/No for each point. Semantic correctness $= \sum_i s_i \cdot \mathbb{1}[\text{answer}_i=\text{Yes}]$, with total weights summing to 1.
    - **Design Motivation**: Single-instruction MLLM evaluation misses details (e.g., bond counts, geometric relations, musical notes); explicitly decomposing key constraints ensures stable capture of subject-level errors.

2. **Dual-Score Evaluation Protocol (Strict + Relaxed)**:

    - **Function**: Simultaneously characterizes "perfect correctness" and "degree of correctness" to avoid all-or-nothing outcomes.
    - **Mechanism**: Strict score = proportion of images fully satisfying all scoring points and scoring 2 on spelling/logic/readability (any error yields 0); relaxed score = $0.7\cdot\text{semantic}+0.1\cdot\text{spell}+0.1\cdot\text{logic}+0.1\cdot\text{read}$ (weights aligned with human preferences). Strict highlights the difficulty ceiling ("almost no one can score perfectly"), while relaxed distinguishes among low-scoring models.
    - **Design Motivation**: Pure strict leads to most models scoring 0%, losing informativeness; pure weighted average masks the "almost correct is still wrong" nature of subject tasks, so both are reported in parallel.

3. **Data Curation Pipeline**:

    - **Function**: Ensures question difficulty, subject coverage, and scoring point quality.
    - **Mechanism**: Generate keywords by four-level taxonomy → web image retrieval + filtering from existing MLLM datasets → GPT-5 filters by textual richness, subject density, and complexity → GPT-5 drafts prompts and scoring points → PhD-level annotators manually review and revise. In the final 1,000 questions, 38% are hard, 38% medium, 24% easy; prompts range from 24-173 words.
    - **Design Motivation**: Web images vary in quality, pure manual curation is costly, and pure GPT-5 tends to "pad" content; the dual-layer GPT-5 + human review balances scale and rigor.

### Loss & Training
This work is a benchmark; no training is involved. The only tunable component is the evaluation-side MLLM judge (default: GPT-5, reasoning effort set to low; appendix shows Gemini-3-Flash and other alternatives remain highly consistent with human judgments).

## Key Experimental Results

### Main Results
Strict and relaxed scores for 17 models (excerpt):

| Model | Type | Strict ↑ | Relaxed ↑ |
|-------|------|---------:|----------:|
| Nano Banana Pro | Closed-source | **70.2** | **93.0** |
| GPT-Image-1.5 | Closed-source | 42.5 | 81.5 |
| GPT-Image-1 | Closed-source | 13.1 | 62.2 |
| Seedream 4.5 | Closed-source | 12.3 | 59.5 |
| FLUX.2 max | Closed-source | 8.6 | 61.6 |
| FLUX.2 dev | Open-source T2I | 2.4 | 42.3 |
| Qwen-Image-2512 | Open-source T2I | 1.5 | 35.3 |
| BAGEL (thinking) | Open-source unified MLLM | 0.0 | 12.9 |
| Janus-Pro | Open-source unified MLLM | 0.0 | 9.5 |

Even the strongest closed-source model fails to pass; most open-source T2I models perform near zero. All open-source unified MLLMs score 0 on strict, even worse than dedicated T2I models.

### Ablation Study

| Evaluator | Kendall $\tau$ vs. Human | Pearson $r$ |
|-----------|-------------------------:|------------:|
| Relaxed by GPT-5 | **0.675** | **0.844** |
| Relaxed by Gemini-3-Flash | 0.661 | 0.826 |
| Semantic Correctness Only | 0.633 | 0.806 |
| VQA Score | 0.145 | 0.179 |
| CLIP Score | 0.116 | 0.165 |

MAE for each dimension: semantic 0.10, spelling 0.11, readability 0.20, logic 0.28—all low, indicating stable evaluation.

### Key Findings
- **Unified MLLMs underperform dedicated T2I models**: Open-source unified models like BAGEL and Show-o2 score 0 on strict and lower relaxed scores than FLUX.2 dev/Qwen-Image-2512, indicating that "one model for understanding and generation" is far from solved for subject images.
- **The bottleneck is visual execution, not knowledge**: FLUX.2 dev can correctly identify locations of Egypt/Iran/India/China in history questions but fails to render the corresponding graphical elements—models lack the ability to "translate knowledge into readable images."
- **CLIP/VQA scores are ineffective**: Correlation with human judgment is near 0.1, showing that traditional T2I metrics fail to capture subject correctness.
- **Open-source models should focus on fundamentals**: Open-source models perform worst on spelling and logic consistency, suggesting that basic skills like text rendering and coordinate alignment should be prioritized before reasoning.

## Highlights & Insights
- **Explicit scoring rubrics are a generalizable paradigm for LLM/T2I evaluation**: Decomposing "correct/incorrect" into structured yes/no lists makes MLLM judge MAE controllable and correlations far exceed traditional metrics. This approach is also applicable to chart QA, code generation, and math answer evaluation.
- **Dual strict + relaxed metrics are cleverly designed**: One highlights the difficulty ceiling (separating top closed-source models), the other reveals differences among low-scoring models, avoiding "all perfect" or "all zero" compression.
- **The "exam perspective" reframes T2I evaluation goals**: Traditional T2I evaluation focuses on fidelity/aesthetics/alignment; here, the focus shifts to "correctness + readability," aligning better with AGI's need for "expert-level intelligence."
- **The data curation protocol is reusable**: The dual-layer GPT-5 drafting + human refinement pipeline can be directly applied to other benchmarks requiring scoring criteria.

## Limitations & Future Work
- 1,000 questions are still insufficient for covering 10 disciplines and 4-level taxonomy; some subfields (e.g., music) have only a few dozen samples, limiting statistical stability.
- Reliance on cutting-edge closed-source MLLMs (GPT-5, Gemini-3-Flash) as judges poses long-term reproducibility and cost concerns; the appendix shows open-source judges, but with lower correlation to humans.
- Scoring point weights are evenly distributed and sum to 1, without reflecting the hierarchical importance of "main structure vs. details."
- The benchmark focuses on "drawing exams" and does not yet cover animation, video, or 3D subject visualization tasks.

## Related Work & Insights
- **vs MMMU / MMLU / Humanity's Last Exam**: All are multidisciplinary exams, but only assess understanding; GenExam brings the same rigorous exam scale to the generation side.
- **vs MMMG / OneIG-Bench / SridBench**: Also evaluate subject image generation, but prior work emphasizes "concept illustration" with loose constraints; GenExam uses longer prompts, stricter constraints, and finer-grained scoring.
- **vs RISEBench / WiScore**: Draws on strict binary scoring and human-aligned weighting, but is the first to extend "customized scoring points" to subject-level evaluation.
- **Transferable insights**: Making "VQA-style scoring points" a general interface for model evaluation is applicable to multimodal reasoning, agent benchmarks, and code generation; it also signals to unified MLLM researchers that current unified architectures are still lacking in subject-specific generation, and the "shared backbone for understanding and generation" design needs further refinement.

## Rating
- Novelty: ⭐⭐⭐⭐ First subject-level T2I exam benchmark; the scoring-points protocol is a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 models × 10 disciplines × dual metrics + 5 human annotators on 250 questions for alignment + multi-evaluator robustness, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and tables, thorough protocol explanation; appendix is detailed, but main text requires cross-referencing tokens, which is less user-friendly.
- Value: ⭐⭐⭐⭐⭐ Provides the T2I community with the first "exam-level" evaluation, likely to become a long-term standard for unified MLLM subject competence.

## Related Papers

- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](../../CVPR2026/image_generation/agentic_retoucher_for_texttoimage_generation.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](../../CVPR2026/image_generation/resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](../../CVPR2026/image_generation/emf_meanflow_text_to_image.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](../../CVPR2025/image_generation/scaling_down_text_encoders_of_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](../../CVPR2026/image_generation/agentic_retoucher_for_texttoimage_generation.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](../../CVPR2026/image_generation/resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](../../CVPR2026/image_generation/emf_meanflow_text_to_image.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)

</div>

<!-- RELATED:END -->
