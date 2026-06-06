---
title: >-
  [Paper Note] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification
description: >-
  [CVPR 2026][LLM Safety][multimodal large language models] This paper presents the first systematic evaluation of demographic fairness in face verification across 9 open-source MLLMs…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "multimodal large language models"
  - "face verification"
  - "demographic fairness"
  - "bias benchmark"
  - "ethnicity and gender bias"
date: 2026-05-08
content_hash: f5903b43394d42e8
---

# Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification

**Conference**: CVPR 2026
**arXiv**: [2603.25613](https://arxiv.org/abs/2603.25613)  
**Code**: [Project Page](https://www.idiap.ch/paper/mllm-fairness)  
**Area**: Multimodal / VLM
**Keywords**: multimodal large language models, face verification, demographic fairness, bias benchmark, ethnicity and gender bias

## TL;DR
This paper presents the first systematic evaluation of demographic fairness in face verification across 9 open-source MLLMs, measuring gender and ethnicity bias on the IJB-C and RFW benchmarks using 4 FMR-based fairness metrics, and finds that bias patterns in MLLMs differ substantially from those in traditional face recognition systems.

## Background & Motivation
**Background**: Multimodal large language models (MLLMs) have recently been explored for face verification—determining whether two faces belong to the same person via visual question answering. Unlike traditional embedding-based systems, MLLMs rely on general visual reasoning rather than task-specific training.

**Limitations of Prior Work**: Demographic bias in traditional face recognition systems has been extensively studied (e.g., Buolamwini & Gebru reported higher error rates for darker-skinned women), yet the fairness of MLLMs as face verification systems remains **entirely unexplored**.

**Key Challenge**: MLLMs process faces in a fundamentally different manner from embedding-based systems (visual question answering vs. feature distance), making it unknown whether bias patterns from traditional systems carry over to MLLMs.

**Goal**: Establish an evaluation benchmark for fairness in MLLM-based face verification, and identify and analyze bias patterns.

**Key Insight**: Nine MLLMs are evaluated under standard face verification protocols (IJB-C, RFW) across 4 ethnicity groups and 2 gender groups using multi-dimensional fairness metrics.

**Core Idea**: Bias patterns in MLLMs differ from those in traditional systems—the most accurate model is not necessarily the fairest, and poorly performing models may appear fair due to uniformly high error rates across all groups.

## Method

### Overall Architecture
Given two face images and a text prompt, an MLLM outputs a similarity score $s_{ij} \in [0,1]$. For multi-image templates, the template-level score is the average over all $m \times n$ pairs. FMR, FNMR, and EER are computed at varying thresholds and evaluated separately per demographic group.

### Key Designs
1. **Evaluation Protocol Design**: 10,000 template pairs are sampled from IJB-C (each model requires approximately 20 days of H100 computation due to MLLM inference speed), stratified by 4 ethnicities (African, East Asian, South Asian, Caucasian) and 2 genders. RFW provides a balanced set of 4 ethnicities × 6,000 pairs. **Design Motivation**: Ensure sufficient statistical samples within each demographic group.

2. **Multi-Dimensional Fairness Metric Suite**: Four complementary metrics are used—maximum FMR difference $\Delta$ (absolute), maximum FMR ratio $R$ (relative), maximum ratio to geometric mean $M$ (stable across multiple groups), and Gini coefficient $G$ (global inequality measure)—supplemented by the decidability index $d'$ measuring score distribution separability. **Design Motivation**: A single metric can be misleading; multi-dimensional evaluation is more comprehensive—$R$ becomes unstable when the minimum FMR approaches zero, while $M$ mitigates this via the geometric mean.

3. **Zero-Shot Evaluation**: All MLLMs are evaluated in a zero-shot setting without any fine-tuning or in-context learning, reflecting out-of-the-box fairness.

### Loss & Training
This is an evaluation study and involves no training. All models use pretrained weights with a unified text prompt format.

## Key Experimental Results

### Main Results (IJB-C EER, % ↓)

| Model | Global | African | Caucasian | East Asian | South Asian | σ(Ethnicity) |
|------|--------|---------|-----------|------------|-------------|--------|
| FaceLLM-8B | **5.13** | 4.65 | 5.53 | 4.31 | **3.98** | **0.58** |
| Qwen2-VL-7B | 8.54 | 6.34 | 9.14 | 7.49 | 5.97 | 1.23 |
| Qwen2.5-VL-7B | 10.43 | 9.24 | 11.07 | 10.38 | 8.13 | 1.12 |
| Ovis1.5-Llama3-8B | 30.89 | 32.71 | 30.35 | 31.30 | 32.67 | 0.99 |

### Ablation Study (RFW Cross-Ethnicity Comparison)

| Model | Global EER | African | Caucasian | Asian | Indian | σ |
|------|-----------|---------|-----------|-------|--------|---|
| FaceLLM-8B | **29.46** | 35.25 | 27.87 | **21.23** | 29.13 | 4.98 |
| Qwen2-VL-7B | 34.39 | 39.63 | 34.57 | 24.40 | 34.13 | 5.51 |
| Valley2 | 39.67 | 45.85 | 40.35 | 30.33 | 39.92 | 5.58 |

### Key Findings
- **FaceLLM-8B stands out**: As the only face-specialized MLLM, it substantially outperforms general-purpose MLLMs on both benchmarks (EER 5.13% vs. 8.54% for the runner-up).
- **Bias patterns differ from traditional systems**: On RFW, all MLLMs achieve the lowest EER (best performance) on the Asian group, whereas traditional systems typically perform best on Caucasians.
- **Accuracy ≠ Fairness**: FaceLLM-8B achieves σ=0.58 (smallest ethnicity disparity) on IJB-C but σ=4.98 on RFW, making it not the fairest model overall.
- **Spurious fairness from low accuracy**: Ovis1.5-Llama3-8B exhibits an apparently low ethnicity disparity of σ=0.99, yet its EER reaches 30.89%—because all groups perform uniformly poorly.
- **Gender bias**: Most MLLMs yield higher EER for females than males (gap of 1–4%); FaceLLM-8B shows the smallest gap (0.55%).

## Highlights & Insights
- **Filling a gap**: This work is the first to extend demographic fairness analysis from traditional face recognition to MLLMs.
- **Multi-dimensional metric suite**: The combination of 4 FMR-based metrics and the decidability index avoids the pitfalls of relying on any single metric.
- **Discovery of "spurious fairness"**: Low-accuracy models can masquerade as fair—Ovis1.5 shows an inter-ethnicity EER spread of only 0.99, yet its overall EER is 30.89%.
- **Practical significance**: MLLMs are being considered as replacements for traditional face systems; understanding their fairness characteristics is critical for responsible deployment.
- **Novel bias pattern**: MLLMs perform best on the Asian group (traditional systems typically favor Caucasians), potentially reflecting distributional differences in MLLM pretraining data.
- **Gender consistency of Qwen2-VL-7B**: A gender gap of only 0.10% may suggest that certain architectural choices are inherently conducive to gender fairness.
- **Advantage of domain-specific specialization**: FaceLLM achieves EER 5.13% vs. 8.54% for the best general-purpose model, demonstrating the indispensability of face-specific fine-tuning.

## Limitations & Future Work
- Extremely high computational cost (approximately 20 H100 days per model) limits the number of evaluable models and the IJB-C sampling scale.
- Only open-source models are evaluated; closed-source MLLMs such as GPT-4V and Gemini are excluded.
- Fairness analysis is restricted to ethnicity and gender, without covering finer-grained attributes such as age or skin tone.
- The **sources** of bias (imbalanced pretraining data vs. model architecture) are not investigated.
- Overall MLLM face verification performance remains far below that of dedicated systems, limiting practical utility.
- The potential of in-context learning or prompt engineering for bias mitigation is not explored.
- Inconsistent bias patterns across datasets (IJB-C vs. RFW) call for a larger-scale unified evaluation.

## Related Work & Insights
- **Distinction from FaceRecBench and FaceXBench**: The latter two assess the accuracy of MLLM face capabilities, whereas this paper focuses on the **fairness** dimension.
- The NIST FRVT series established fairness evaluation standards for traditional systems; this paper extends that framework to MLLMs.
- Fairness metric selection follows the ISO/IEC 19795-10:2024 standard.
- FaceLLM's advantage confirms that general-purpose MLLMs genuinely require domain-specific fine-tuning for face tasks.

## Technical Details
- **Template-level aggregation**: $s(T_p, T_g) = \frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n} s_{ij}$
- **Threshold sweep**: $\tau \in [0,1]$, step size 0.005 (201 operating points)
- **Decidability**: $d' = \frac{\mu_{genuine} - \mu_{impostor}}{\sqrt{\frac{1}{2}(\sigma_{genuine}^2 + \sigma_{impostor}^2)}}$
- **IJB-C sampling**: 10,000 pairs; approximately 20 H100 days per model
- **LLaVA-NeXT failure**: Produces only 2 distinct similarity scores, rendering meaningful verification impossible
- **All MLLMs show significant performance degradation on RFW** (EER 29–50%), indicating this benchmark is more challenging
- **4 fairness metrics**: $\Delta$ (difference), $R$ (ratio), $M$ (ratio to geometric mean), $G$ (Gini coefficient)
- **Gini coefficient definition**: $G = \frac{\sum_i\sum_j |e_i - e_j|}{2K\sum_i e_i}$, where $G=0$ denotes perfect fairness
- **Instability of maximum ratio**: $R$ diverges when the minimum group FMR approaches zero; $M$ mitigates this via the geometric mean
- **Model family coverage**: 9 models from 6 families—Idefics3, Ovis, Qwen2-VL, Qwen2.5-VL, Valley, LLaVA-NeXT, and FaceLLM
- **IJB-C template structure**: Each template contains 1 to hundreds of images/video frames
- **Zero-shot evaluation protocol**: No fine-tuning or in-context examples are provided to any model
- **TMR metric**: True Match Rate reported at three security levels: FMR = 10% / 1% / 0.1%
- **Cross-benchmark discrepancy**: Bias patterns differ significantly between IJB-C and RFW—e.g., FaceLLM achieves $\sigma$=0.58 (lowest) on IJB-C but $\sigma$=4.98 on RFW
- **Model parameter range**: From 2B (Qwen2-VL-2B) to 8B (FaceLLM-8B), covering different scales

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation of MLLM face verification fairness, revealing bias patterns distinct from traditional systems
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 models × 2 benchmarks × multiple metrics, though sampling scale is constrained by computational cost
- Writing Quality: ⭐⭐⭐⭐ Metric definitions are rigorous and analysis is thorough
- Value: ⭐⭐⭐⭐ Important reference for the responsible deployment of MLLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](../../ACL2026/llm_safety/when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)

</div>

<!-- RELATED:END -->
