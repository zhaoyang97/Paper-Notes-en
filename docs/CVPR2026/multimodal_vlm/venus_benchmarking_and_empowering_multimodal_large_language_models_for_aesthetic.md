---
title: >-
  [Paper Note] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping
description: >-
  [CVPR 2026][Multimodal VLM][Aesthetic Guidance] This paper defines the novel task of Aesthetic Guidance (AG) and constructs the AesGuide benchmark (10,748 photos annotated with aesthetic scores, analyses, and guidance)…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Aesthetic Guidance"
  - "Image Cropping"
  - "MLLM"
  - "Aesthetic Assessment"
  - "CoT Reasoning"
date: 2026-05-08
content_hash: 3d1bed560eb0c166
---

# Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping

**Conference**: CVPR 2026
**arXiv**: [2602.23980](https://arxiv.org/abs/2602.23980)  
**Code**: [https://github.com/PKU-ICST-MIPL/Venus_CVPR2026](https://github.com/PKU-ICST-MIPL/Venus_CVPR2026)  
**Area**: Multimodal VLM
**Keywords**: Aesthetic Guidance, Image Cropping, MLLM, Aesthetic Assessment, CoT Reasoning

## TL;DR
This paper defines the novel task of Aesthetic Guidance (AG) and constructs the AesGuide benchmark (10,748 photos annotated with aesthetic scores, analyses, and guidance), then proposes Venus, a two-stage framework that first empowers MLLMs with aesthetic guidance capability via progressive aesthetic QA, and subsequently activates aesthetic cropping capability through CoT reasoning, achieving state-of-the-art performance on both tasks.

## Background & Motivation
**Background**: Computational aesthetics has evolved from perception-level tasks such as aesthetic scoring and quality description to higher-level understanding. However, "aesthetic guidance"—identifying aesthetic issues and providing actionable shooting suggestions—remains a critical yet systematically understudied capability.

**Limitations of Prior Work**: (a) General MLLMs (e.g., GPT-4o) and aesthetic-specialized MLLMs (e.g., AesExpert) tend to produce overly positive evaluations when presented with photos, failing to identify problems or provide actionable recommendations; (b) aesthetic cropping models, while capable of cropping, lack interpretability and interactivity, as they can neither explain their cropping rationale nor adapt to user preferences.

**Key Challenge**: Existing aesthetic datasets primarily annotate "what is good," lacking guidance-oriented annotations that capture "what is wrong" and "how to improve." Furthermore, MLLMs are misaligned with the human aesthetic reasoning process.

**Goal**: (a) Construct the first aesthetic guidance dataset and benchmark; (b) empower MLLMs with aesthetic guidance capability; (c) leverage aesthetic guidance capability to activate cropping capability.

**Key Insight**: Aesthetic guidance follows a human cognitive process of "overall impression → analysis of strengths and weaknesses → improvement suggestions." MLLMs are trained to simulate this process through progressive complexity QA.

**Core Idea**: Through a two-stage approach—aesthetic guidance empowerment (progressive QA) and aesthetic cropping activation (CoT reasoning rationale)—Venus achieves both aesthetic understanding and aesthetic creation in MLLMs.

## Method

### Overall Architecture
The framework proceeds in two stages: Stage 1 trains an MLLM on AesGuide to answer aesthetic questions of increasing complexity (scoring → analysis → guidance), establishing aesthetic guidance capability. Stage 2 trains on cropping data using CoT rationales that incorporate aesthetic reasoning (AR), activating cropping capability.

### Key Designs

1. **AesGuide Dataset Construction (AGGF Framework)**:

    - **Function**: Collects 10,748 photos from online platforms and professional photographers, annotated with aesthetic scores, analyses, and guidance.
    - **Mechanism**: Two-stage annotation pipeline—MLLM refinement (GPT-4o analyzes raw comments → Qwen2.5-VL-72B validates completeness) → expert annotation (20 photography experts review and revise, separating annotations into aesthetic analysis and aesthetic guidance).
    - **Design Motivation**: Raw online comments are noisy and inconsistent; the MLLM + expert two-stage pipeline ensures annotation quality and consistency.

2. **Progressive Aesthetic QA Training**:

    - **Function**: Trains the MLLM to answer aesthetic questions across three levels.
    - **Mechanism**: Level 1—overall impression ("How is this photo?"); Level 2—detailed analysis ("What compositional issues exist? Is the lighting appropriate?"); Level 3—improvement guidance ("How should it be improved? How to adjust the shooting angle or lighting?").
    - **Design Motivation**: Simulates the progressive nature of human aesthetic cognition—building intuitive perception before deepening into rational analysis, and finally producing actionable recommendations.

3. **Aesthetic Cropping CoT Activation**:

    - **Function**: Trains the model to jointly output crop bounding boxes and aesthetic reasoning.
    - **Mechanism**: Aesthetic rationales (AR) are generated for both good and poor crops; GPT-4o explains why a cropped region marked with a red bounding box exhibits good or poor composition, and Qwen2.5-VL-72B validates consistency.
    - **Design Motivation**: Learning crop coordinates alone lacks compositional understanding; CoT rationales compel the model to understand "why crop here," improving both interpretability and interactivity.

4. **AG Evaluation Benchmark Design**:

    - **Function**: Uses GPT as a judge to evaluate outputs across three dimensions: completeness, preciseness, and relevance.
    - **Mechanism**: Each dimension is scored 0–2 with golden annotations as references and GPT-4 as the evaluator. Manual evaluation by 10 experts on 100 samples is conducted to validate the reliability of GPT-based scoring.

### Loss & Training
Both stages employ standard instruction fine-tuning: $\mathcal{L} = -\mathbb{E}\sum_t \log\pi_\theta(y_t|x,q,y_{<t})$. In Stage 1, the visual encoder and connector are frozen, with only the LLM being trained. In Stage 2, the aesthetic guidance MLLM undergoes full-parameter fine-tuning.

## Key Experimental Results

### Aesthetic Guidance Evaluation (AesGuide Benchmark)

| Model | Completeness | Preciseness | Relevance | Mean | Expert |
|------|-------------|-------------|-----------|------|--------|
| GPT-4o | 0.84 | 1.09 | 1.01 | 0.98 | 1.15 |
| AesExpert-7B | 0.33 | 0.56 | 0.51 | 0.47 | 0.56 |
| UNIAA-7B | 1.03 | 1.02 | 1.23 | 1.09 | 1.01 |
| InternVL 2.5-7B | 0.83 | 1.01 | 1.02 | 0.95 | 0.99 |
| **Venus-I (ours)** | **1.27** | **1.33** | **1.81** | **1.47** | **1.50** |
| LLaVA-1.5-13B | 0.67 | 0.86 | 0.41 | 0.65 | 0.61 |
| **Venus-L-13B (ours)** | **1.28** | **1.35** | **1.83** | **1.49** | **1.53** |

### Aesthetic Cropping (FLMS Benchmark)

| Model | IoU%↑ | Disp↓ | Interpretable | Interactive |
|------|-------|-------|--------|--------|
| CACNet | 72.8 | 0.062 | ✗ | ✗ |
| TransView | 71.5 | 0.068 | ✗ | ✗ |
| GPT-4o | 58.3 | 0.105 | ✓ | ✓ |
| **Venus-Q (ours)** | **74.2** | **0.055** | ✓ | ✓ |

### Key Findings
- Venus outperforms GPT-4o by approximately 50% on the aesthetic guidance Mean score (1.47 vs. 0.98), with the largest gain observed in the Relevance dimension (+0.79).
- Aesthetic guidance capability directly benefits cropping performance—omitting Stage 1 and training on cropping directly leads to a significant performance drop.
- A user study involving 1,069 participants shows that 91% desire aesthetic guidance functionality, validating the practical relevance of the task definition.
- Venus achieves simultaneous state-of-the-art cropping performance, interpretability, and interactivity, being the only method to satisfy all three criteria.
- Training with rationales that include "poor crops" yields better results than using only "good crops."

## Highlights & Insights
- **Contribution of Task Definition**: The formal definition of the Aesthetic Guidance (AG) task fills a critical gap in computational aesthetics, with a 91% user survey endorsing its practical demand. This definition is expected to stimulate subsequent research.
- **Two-Stage Capability Transfer**: The transfer pathway from AG capability to cropping capability is elegant—first enabling the model to "appreciate aesthetics," then enabling it to "create aesthetics," with Stage 1 serving as the foundation for Stage 2. This capability-progressive training paradigm is transferable to other dual "understanding + creation" tasks.
- **AGGF Annotation Framework**: The annotation pipeline combining MLLM refinement with expert review balances efficiency and quality, offering a practical solution for large-scale subjective task annotation.

## Limitations & Future Work
- The AesGuide data primarily originates from online photography communities, which may introduce stylistic bias toward specific aesthetic preferences.
- Cropping is limited to two-dimensional recomposition and does not address richer aesthetic corrections such as 3D perspective adjustment or lighting modification.
- Evaluation relies on GPT as a judge, which may introduce bias for highly subjective aesthetic assessments.
- Personalization is not explored—different users apply different criteria to "good photographs."

## Related Work & Insights
- **vs. AesExpert**: AesExpert focuses on aesthetic perception and description (with a positive bias), whereas Venus targets aesthetic guidance (identifying problems and providing suggestions), representing a fundamentally different positioning.
- **vs. CACNet**: CACNet is a dedicated cropping model with high IoU but no interpretability; Venus achieves both cropping and explanation simultaneously through CoT rationales.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The AG task definition fills a gap, and the AesGuide dataset is the first of its kind.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five MLLMs × two tasks, with dual evaluation via GPT and human experts.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; the user study strengthens persuasiveness.
- Value: ⭐⭐⭐⭐⭐ The dataset and benchmark offer high community value, directly targeting practical photography guidance scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A3: Towards Advertising Aesthetic Assessment](a3_towards_advertising_aesthetic_assessment.md)
- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
