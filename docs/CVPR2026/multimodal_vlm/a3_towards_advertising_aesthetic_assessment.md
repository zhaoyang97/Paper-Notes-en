---
title: >-
  [Paper Note] A3: Towards Advertising Aesthetic Assessment
description: >-
  [CVPR 2026][Multimodal VLM][Advertising Aesthetic Assessment] This paper proposes the A3 framework, comprising a theory-driven three-stage advertising aesthetic assessment paradigm A3-Law (Perceptual Attention → Formal Interest → Desire Impact), a 120K-annotation dataset A3-Dataset, an SFT+GRPO aligned model A3-Align, and the evaluation benchmark A3-Bench. A3-Align surpasses existing MLLMs on automated advertising aesthetic assessment.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Advertising Aesthetic Assessment
  - Multimodal Large Language Models
  - AIDA Model
  - Chain-of-Thought
  - GRPO
date: 2026-05-08
content_hash: ed32e61c4b1c0676
---

# A3: Towards Advertising Aesthetic Assessment

**Conference**: CVPR 2026
**arXiv**: [2603.24037](https://arxiv.org/abs/2603.24037)
**Code**: [https://github.com/euleryuan/A3-Align](https://github.com/euleryuan/A3-Align)
**Area**: Multimodal VLM
**Keywords**: Advertising Aesthetic Assessment, Multimodal Large Language Models, AIDA Model, Chain-of-Thought, GRPO

## TL;DR
This paper proposes the A3 framework, comprising a theory-driven three-stage advertising aesthetic assessment paradigm A3-Law (Perceptual Attention → Formal Interest → Desire Impact), a 120K-annotation dataset A3-Dataset, an SFT+GRPO aligned model A3-Align, and the evaluation benchmark A3-Bench. A3-Align surpasses existing MLLMs on automated advertising aesthetic assessment.

## Background & Motivation

**Background**: Advertising images are critical to commercial conversion rates, yet current evaluation methods rely primarily on subjective human scoring, lacking scalability, standardized criteria, and interpretability. Automated systems mostly employ simple threshold-based filtering and cannot provide diagnostic feedback.

**Limitations of Prior Work**: Although MLLMs possess strong vision-language understanding capabilities, they exhibit three shortcomings in advertising aesthetic assessment: (1) they produce only a single holistic score, ignoring the progressive nature of human cognition; (2) their outputs are unstable and sensitive to prompt phrasing; and (3) their reasoning processes frequently contradict their final judgments.

**Key Challenge**: Advertising aesthetic assessment requires multi-level judgment spanning from low-level perception (image quality) to high-level cognition (emotional arousal and persuasiveness), yet existing methods lack a methodology for translating abstract theory into an executable evaluation framework.

**Key Insight**: The classical AIDA marketing model (Attention → Interest → Desire → Action) is leveraged to construct a staged advertising aesthetic assessment framework.

**Core Idea**: Advertising aesthetic assessment is decomposed into three hierarchical levels (Perceptual Attention → Formal Interest → Desire Impact), each with explicit theoretical grounding and actionable evaluation rules, complemented by CoT-guided dataset construction and GRPO alignment training.

## Method

### Overall Architecture
A3 consists of four components: (1) A3-Law, a theoretical paradigm defining three-stage evaluation rules; (2) A3-Dataset, containing 30K advertising images and 120K instruction-response pairs; (3) A3-Align, a model trained via SFT+GRPO alignment; and (4) A3-Bench, an evaluation benchmark.

### Key Designs

1. **A3-Law: Three-Stage Hierarchical Paradigm**:

    - **Perceptual Attention**: Assesses whether image signals can capture viewer attention. Grounded in signal detection theory, it encompasses three rules: image fidelity (clarity and freedom from distortion), integration realism (consistency of lighting, shadows, and perspective), and professional refinement (absence of artifacts and clarity of details). Theoretical basis: information must cross a physiological threshold before entering higher-level cognition.
    - **Formal Interest**: Evaluates whether color and spatial layout can arouse interest. Covers color construction (hue adaptability and color harmonization, quantified via the Hasler metric) and spatial construction (layout adaptability, hierarchy, focal points, and safe zones). Theoretical basis: perceptual grouping mechanisms from Gestalt psychology.
    - **Desire Impact**: Assesses the semantic and emotional value of the image. Encompasses copywriting tone, promotional icon recognition (via object detection), aesthetic attributes (intuitive visual pleasure), and advertising attributes (brand emotional connection and persuasiveness). Theoretical basis: semiotics and affective appraisal theory.

2. **A3-Dataset Construction**:

    - **Function**: Generates 120K instruction-response pairs from 30K advertising images.
    - **Mechanism**: A two-stage pipeline — a human annotation stage (image collection, A3-Law rule annotation, and quality review, with objective metric accuracy >0.93, IoU >0.92, and subjective SRCC >0.85) followed by a model-augmentation stage (MLLM-generated CoT reasoning chains, validated by majority vote from a five-expert panel, with an overall acceptance rate >85% after iterative refinement).
    - **Design Motivation**: Combines the reliability of human annotation with the scalability of LLM-generated CoT.

3. **A3-Align Training**:

    - **Function**: Enables MLLMs to learn A3-Law rules and produce structured outputs.
    - **Mechanism**: The **SFT stage** acquires rules, formats, tool usage, and CoT capabilities; the **GRPO stage** optimizes via multi-signal rewards — generic rewards (format reward $R_{format}$, non-repetition reward $R_{nonrep}$) and rule-specific rewards (accuracy $R_{acc}$, tool usage $R_{tool}$, IoU reward $R_{IoU}$, continuous score reward $R_{score} = \exp(-\frac{(s-\hat{s})^2}{2\sigma^2})$).
    - **Design Motivation**: SFT provides structural foundations, while GRPO further calibrates behavioral format, task accuracy, evidential grounding, and subjective value alignment.

4. **Tool-Calling Mechanism**:

    - Three lightweight analysis tools: a hue analysis tool (for hue adaptability judgment), a color harmonization quantifier (Hasler index), and DeepSeek-OCR (for copywriting tone assessment).
    - Tool outputs are integrated into the reasoning chain as auxiliary evidence; decisions are not mechanistically determined by tool outputs alone.

### Loss & Training
The total reward is computed as a normalized weighted sum: $R_{total} = \frac{\sum_{i \in \mathcal{A}} \alpha_i R_i}{\sum_{i \in \mathcal{A}} \alpha_i}$, where different reward subsets are activated based on the type of the current sample.

## Key Experimental Results

### Main Results (Per-Rule Accuracy on A3-Bench)

| Model | Image Fidelity | Integration Realism | Color Harmonization | Layout Adaptability | Aesthetic SRCC |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B | 0.454 | 0.491 | 0.444 | 0.472 | 0.564 |
| Gemma-3-27B | 0.648 | 0.574 | 0.583 | 0.694 | 0.677 |
| GPT-4o | - | - | - | - | - |
| **A3-Align** | **Best** | **Best** | **Best** | **Best** | **Best** |

(Across the full 10-dimension evaluation, A3-Align significantly outperforms both open-source and closed-source MLLMs on nearly all rules.)

### Ablation Study (Training Strategy)

| Configuration | Binary Rules Avg Acc | Aesthetic SRCC | Advertising SRCC |
|------|------|------|------|
| SFT only | Baseline | Baseline | Baseline |
| SFT + GRPO (w/o tools) | +Gain | +Gain | +Gain |
| SFT + GRPO (full) | **Best** | **Best** | **Best** |

### Key Findings
- Even the strongest closed-source models (e.g., GPT-4o-thinking) perform poorly on A3-Law's hierarchical evaluation, demonstrating the necessity of domain alignment.
- The multi-signal rewards in the GRPO stage yield significant performance improvements over SFT alone across all dimensions.
- The tool-calling mechanism provides measurable benefits for color and copywriting evaluation.
- A3-Align demonstrates strong practical utility on two downstream tasks: advertising selection and diagnostic critique generation.

## Highlights & Insights
- **Theory-Driven Evaluation Framework**: Translating AIDA marketing theory into an executable three-stage computational assessment paradigm is an exemplary instance of engineering cognitive-psychological theory into practice. The complete methodology of "theory → paradigm → data → model → benchmark" is transferable to other subjective evaluation tasks.
- **CoT + GRPO Alignment Strategy**: Using SFT to first learn structure and rules, then applying GRPO's multi-signal rewards for fine-grained calibration, offers a generalizable paradigm for aligning LLMs to domain-specific evaluation standards.
- **Tool-Augmented Reasoning**: Integrating quantitative tools (color analysis, OCR) into the reasoning chain grounds subjective judgments in objective measurements.

## Limitations & Future Work
- The Desire Impact stage of A3-Law is positioned as a culturally universal framework; however, advertising aesthetics are highly culture-dependent, and cross-cultural adaptation remains unexplored.
- The current work addresses only static advertising images; evaluation of video and interactive advertisements is not covered.
- The 30K-image dataset may still be limited in diversity within the advertising domain, and more fine-grained rules may be required for specific verticals (e.g., luxury goods, fast-moving consumer goods).
- The choice of $\sigma$ in the Gaussian reward function for continuous scores affects training stability and precision, warranting further analysis.

## Related Work & Insights
- **vs. AVA/AADB**: Traditional aesthetic datasets provide only single-dimensional scores, whereas A3-Dataset offers multi-level, multi-dimensional annotations with CoT reasoning chains.
- **vs. General MLLMs (e.g., GPT-4o)**: General-purpose models lack rule awareness for advertising aesthetics; A3-Align achieves substantial domain alignment through domain-specific data and GRPO training.
- **Application Insights**: The three-stage framework of A3-Law can inspire the design of other tasks requiring hierarchical evaluation, such as UI design assessment and interior design evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic advertising aesthetic assessment framework, fully connecting theory → data → model → benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparisons and downstream task validation; ablation studies could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, though the density of content means some details require consulting the appendix.
- Value: ⭐⭐⭐⭐ Practically applicable to the advertising industry, though the domain is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[CVPR 2026\] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](../../ICLR2026/multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)

</div>

<!-- RELATED:END -->
