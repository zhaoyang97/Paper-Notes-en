---
title: >-
  [Paper Note] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought
description: >-
  [CVPR 2026][LLM Reasoning][e-commerce poster evaluation] This work constructs E-comIQ-ZH, the first multi-dimensional quality assessment framework for Chinese e-commerce posters, comprising an 18K expert-annotated dataset with CoT reasoning chains, a dedicated evaluation model E-comIQ-M (trained via SFT+GRPO), and a standardized benchmark E-comIQ-Bench.
tags:
  - CVPR 2026
  - LLM Reasoning
  - e-commerce poster evaluation
  - image quality assessment
  - Chain-of-Thought
  - multi-dimensional scoring
  - Chinese text quality
date: 2026-05-08
content_hash: c9d36c34ba33cf09
---

# E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought

**Conference**: CVPR 2026
**arXiv**: [2602.21698](https://arxiv.org/abs/2602.21698)
**Code**: [GitHub](https://github.com/4mm7/E-comIQ-ZH)
**Area**: Image Quality Assessment / E-commerce AI
**Keywords**: e-commerce poster evaluation, image quality assessment, Chain-of-Thought, multi-dimensional scoring, Chinese text quality

## TL;DR
This work constructs E-comIQ-ZH, the first multi-dimensional quality assessment framework for Chinese e-commerce posters, comprising an 18K expert-annotated dataset with CoT reasoning chains, a dedicated evaluation model E-comIQ-M (trained via SFT+GRPO), and a standardized benchmark E-comIQ-Bench.

## Background & Motivation
**State of the Field**: Generative AI is widely adopted for e-commerce poster production, yet automated quality assessment lags far behind generative capability. Existing IQA methods focus on generic aesthetics or low-level distortions and cannot measure the functional criteria required in e-commerce contexts.

**Limitations of Prior Work**: Chinese e-commerce content is particularly challenging—complex Chinese character strokes frequently produce subtle yet critical text rendering errors, which existing methods, including strong models such as GPT-4o and Gemini 2.5 Pro, fail to detect. As shown in Fig. 1, both Gemini 2.5 Pro and Q-Insight fail to identify stroke-level character corruption.

**Root Cause**: The absence of formal multi-dimensional quality standards prevents systematic evaluation, which in turn prevents the construction of training data and the training of dedicated evaluators—forming a vicious cycle. Current workflows still rely on slow and unscalable human review.

**Paper Goals**: To establish multi-dimensional quality assessment standards for e-commerce posters and an automated evaluation toolchain.

**Starting Point**: In collaboration with senior e-commerce art directors, quality is decomposed into four dimensions—Object, Background, Text, and Layout—to build a large-scale expert-annotated dataset and a dedicated evaluation model.

**Core Idea**: Train a domain-specific evaluation model using expert annotations and CoT reasoning chains to align automatic evaluation with human expert judgment.

## Method

### Overall Architecture
E-comIQ-ZH consists of three components: (a) the E-comIQ-18k dataset (18K posters with multi-dimensional scores and CoT reasoning chains), (b) the E-comIQ-M evaluation model (two-stage training), and (c) the E-comIQ-Bench benchmark (a platform for evaluating generative models).

### Key Designs
1. **Multi-Dimensional Annotation Schema**: Four orthogonal dimensions—Object (product integrity/clarity), Background (background compatibility/visual appeal), Text (typographic readability/correctness), and Layout (overall composition/visual hierarchy). The average Pearson correlation across dimensions is only $\rho \approx 0.24$, confirming that a single holistic score is insufficient to characterize e-commerce poster quality. A "weakest link" analysis reveals that Text is the quality bottleneck in 44.8% of low-quality images and exhibits the highest correlation with overall quality ($\rho = 0.67$).
2. **Human-in-the-Loop CoT Generation**: Qwen-2.5-VL-Max first generates reasoning chains conditioned on expert scores and issue labels; original annotators then use an NER interface to remove hallucinated content, correct reasoning errors, and supplement domain knowledge. CoT chains average over 800 Chinese characters in length.
3. **Two-Stage Training of E-comIQ-M**: Stage 1—SFT on the 15K training set to acquire domain knowledge and learn output format; Stage 2—GRPO fine-tuning on a 3K "hard sample" subset. The reward function is $R(x,y) = R_{score}(x,y) + \lambda_{fmt} R_{fmt}(y)$, where $R_{fmt}$ enforces valid JSON output.

### Annotation Quality Assurance
Six domain experts first cross-annotate a 1,000-image calibration set, achieving Krippendorff's $\alpha = 0.858$ before dividing annotation duties. A 10% random re-sampling protocol is maintained to prevent annotation drift.

## Key Experimental Results

### Main Results: Correlation with SOTA Models (E-comIQ-18k Test Set)

| Model | Overall PLCC/SRCC | Text PLCC/SRCC | Layout PLCC/SRCC |
|---|---|---|---|
| GPT-4o | 0.242/0.219 | 0.126/0.148 | 0.297/0.282 |
| Gemini 2.5 Pro | 0.213/0.228 | 0.146/0.122 | 0.350/0.320 |
| Qwen2.5-VL-72B | 0.127/0.144 | 0.100/0.070 | 0.142/0.153 |
| Q-Insight | 0.183/0.152 | -0.024/-0.027 | 0.134/0.149 |
| Qwen2.5-VL-7B+SFT | 0.346/0.346 | 0.272/0.283 | 0.390/0.418 |
| **E-comIQ-M (Ours)** | **0.425/0.433** | **0.364/0.392** | **0.483/0.506** |

E-comIQ-M comprehensively outperforms both general-purpose models and dedicated evaluators across all dimensions, with a particularly notable advantage on the Text dimension.

### Inter-Annotator Agreement

| Dimension | Krippendorff's $\alpha$ | Loose-Criterion Accuracy |
|---|---|---|
| Overall | 0.858 | 96.4% |
| Object | 0.745 | 92.2% |
| Background | 0.721 | 94.6% |
| Text | 0.765 | 93.2% |
| Layout | 0.877 | 96.6% |

### Ablation Study: Effect of Training Strategy

| Method | Overall PLCC/SRCC | Background PLCC/SRCC |
|---|---|---|
| Q-Insight+GRPO | 0.265/0.235 | 0.312/0.312 |
| Q-Insight+SFT | 0.297/0.319 | 0.442/0.478 |
| Q-Insight+SFT+GRPO | 0.338/0.348 | 0.459/0.496 |
| **E-comIQ-M** | **0.425/0.433** | **0.496/0.520** |

The two-stage SFT+GRPO training outperforms either stage alone, and Qwen2.5-VL-7B as the backbone is superior to Q-Insight.

### Key Findings
- Conventional NR-IQA models (MUSIQ, SPAQ) are nearly ineffective in the e-commerce setting (correlation < 0.2 or even negative).
- Strong general-purpose MLLMs (GPT-4o, Gemini) achieve Overall PLCC of only approximately 0.2, demonstrating that domain adaptation is essential for e-commerce evaluation.
- The Text dimension is the critical quality bottleneck for Chinese e-commerce posters, yet existing methods perform worst on this dimension.

## Highlights & Insights
- **Pioneering Scope**: The first complete framework—dataset, model, and benchmark—for multi-dimensional IQA of Chinese e-commerce posters.
- **Elegant CoT Annotation Design**: The AI-generation followed by human-verification pipeline balances scale and quality; reasoning chains averaging 800+ characters provide rich diagnostic information.
- **Validation of Dimension Orthogonality**: The low inter-dimension correlation of $\rho \approx 0.24$ strongly supports the necessity of multi-dimensional evaluation.
- **High Annotation Consistency**: $\alpha = 0.858$ with a loose-criterion accuracy of 96.4%.

## Limitations & Future Work
- The dataset is primarily sourced from Taobao/Tmall; generalization to other e-commerce platforms (e.g., Pinduoduo, cross-border platforms) has not been validated.
- Annotators are six specialists working independently after calibration rather than through full cross-annotation; annotation bias is mitigated by 10% re-sampling but not fully eliminated.
- The model is based on Qwen2.5-VL-7B; whether larger models yield further gains remains unexplored.
- The hard-sample selection strategy for GRPO (top 3K by MSE) is relatively simple; more sophisticated curriculum learning approaches could be explored.
- Quality assessment of video advertisements and dynamic posters is not addressed; coverage is limited to static images.
- The quality of generated CoT chains is bounded by the capabilities of Qwen-2.5-VL-Max and may still miss extremely subtle stroke-level errors.
- No quantitative comparison with the speed or cost of human review workflows is provided, leaving the practical deployment value insufficiently demonstrated.

## Related Work & Insights
- Conventional IQA methods (SSIM, MUSIQ, etc.) address only low-level distortions and cannot assess e-commerce functional quality.
- AIGC quality datasets (ImageRewardDB, AGIQA-3K) capture general preferences but lack domain depth.
- AIGuard, the only prior e-commerce IQA dataset, provides 253K binary labels but no multi-dimensional scores or CoT chains.
- E-comIQ-ZH fills the gap in multi-dimensional and interpretable quality assessment for the e-commerce domain.
- MLLM-based evaluators (Q-Align, VQ-R1, DeQA, Q-Insight) target general-purpose scenarios and perform poorly on e-commerce functional dimensions.
- Preference alignment methods such as DPO/GRPO are effective in general domains; this work demonstrates that domain-specific training data is the critical bottleneck.

## Dataset Construction Details
- **6 Source Categories**: 5K high-quality merchant photographs + 5K low-quality merchant photographs + professional design images (quality upper bound) + AI-generated posters + AI-edited composites + template-based workflows.
- **Annotation Protocol**: Continuous scores anchored to three tiers—Excellent [4.0, 5.0], Good [3.0, 4.0), Poor [1.0, 3.0)—accompanied by per-dimension issue labels.
- **Data Split**: 15K training / 2K validation / 1K test, balanced across source categories and quality dimensions.
- **Backbone Selection**: Qwen-2.5-VL-7B is selected as the base model for its strong vision-language capability and native Chinese language support.
- **GRPO Hard Subset**: The worst 3K samples ranked by SFT model MSE constitute $\mathcal{D}_{hard}$.

## Rating ⭐
- **Novelty**: ⭐⭐⭐⭐ — First multi-dimensional IQA framework for Chinese e-commerce posters; defines a new research track.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparisons (conventional IQA / general MLLMs / dedicated evaluators) with clear ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem-to-solution logical chain; figures and tables are information-dense.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to quality control of AI-generated content in e-commerce.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[ACL 2026\] Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](../../ACL2026/llm_reasoning/decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[CVPR 2026\] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)

<!-- RELATED:END -->
