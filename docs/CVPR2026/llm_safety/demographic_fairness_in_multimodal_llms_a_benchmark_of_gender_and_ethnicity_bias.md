---
title: >-
  [Paper Note] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification
description: >-
  [CVPR 2026][LLM Safety][Paper Note] This work provides the first systematic evaluation of demographic fairness for 9 open-source MLLMs in face verification tasks. Using 4 FMR-based fairness metrics to measure gender and ethnic bias on IJB-C and RFW benchmarks, the study reveals that MLLM bias patterns differ significantly from traditional face recognitio
tags:
  - CVPR 2026
  - LLM Safety
date: 2026-05-08
content_hash: c78e4eca5a5a30e5
---
# Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification

**Conference**: CVPR 2026  
**arXiv**: [2603.25613](https://arxiv.org/abs/2603.25613)  
**Code**: [Project Page](https://www.idiap.ch/paper/mllm-fairness)  
**Area**: Multimodal / VLM  
**Keywords**: Multimodal Large Language Models, Face Verification, Demographic Fairness, Bias Benchmark, Ethnicity and Gender Bias

## TL;DR
This work provides the first systematic evaluation of demographic fairness for 9 open-source MLLMs in face verification tasks. Using 4 FMR-based fairness metrics to measure gender and ethnic bias on IJB-C and RFW benchmarks, the study reveals that MLLM bias patterns differ significantly from traditional face recognition systems.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) have recently been explored for face verification—judging whether two faces belong to the same person via Visual Question Answering (VQA). Unlike traditional embedding-based systems, MLLMs rely on general visual reasoning rather than specialized training.

**Limitations of Prior Work**: Demographic bias in traditional face recognition systems has been extensively studied (e.g., Buolamwini & Gebru found higher error rates for dark-skinned females). However, the fairness of MLLMs as face verification systems remains **completely unexplored**.

**Key Challenge**: MLLMs process faces fundamentally differently from embedding systems (VQA vs. feature distance). It is unknown whether traditional bias patterns persist in MLLMs.

**Goal**: Establish an evaluation benchmark for MLLM face verification fairness to identify and analyze bias patterns.

**Key Insight**: Evaluate 9 MLLMs across 4 ethnic groups and 2 gender groups on standard face verification protocols (IJB-C, RFW) using multi-dimensional fairness metrics.

**Core Idea**: MLLM bias patterns differ from traditional systems—the most accurate models are not necessarily the fairest, and low-performance models may appear fair due to "uniformly high error rates."

## Method

### Overall Architecture
This work addresses a previously unquantified question: how demographic bias shifts when face verification moves from traditional "feature vector distance" to MLLM "VQA-based identification." The pipeline is straightforward: image pairs and a fixed text prompt are fed into an MLLM, which outputs a similarity score $s_{ij} \in [0,1]$. When an identity corresponds to multiple images (i.e., a template), the arithmetic mean of all $m \times n$ pairwise scores is calculated to obtain a template-level score. After collecting all scores, standard metrics such as FMR, FNMR, and EER are calculated by scanning decision thresholds, and biases are analyzed by segmenting samples by ethnicity and gender.

### Key Designs

**1. Converting Conversational MLLMs into Quantifiable Face Verifiers**

MLLMs output free-form text, unlike embedding systems that calculate distances between feature vectors. The first challenge is extracting a comparable score from a "chat model." The authors provide the model with an image pair and a fixed text prompt, requiring it to output a similarity $s_{ij}$ normalized to $[0,1]$. Since face verification protocols often involve multiple images per identity (templates), the template-level score is calculated as the arithmetic mean of all $m \times n$ image pair scores:

$$s(T_p, T_g) = \frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n} s_{ij}$$

This averaging significantly reduces noise from individual pairs to provide more stable estimates, albeit at a quadratic increase in inference cost relative to image count. All evaluations are performed in a **zero-shot** setting—no fine-tuning, no in-context examples, and a unified prompt across all models—to measure the true "out-of-the-box" state rather than an idealized performance. To compute metrics, the decision threshold $\tau$ is scanned across $[0,1]$ with a step of 0.005 (201 operating points) to derive FMR, FNMR, and EER.

**2. Evaluation Protocol: Ensuring Statistical Significance**

Fairness metrics compare error rate differences between groups. If group sample sizes are too small, differences are masked by noise, leading to unreliable conclusions. Given that MLLM inference is extremely slow (requiring ~20 H100 days for one model on the full IJB-C set), the authors sample 10,000 template pairs from IJB-C. These are partitioned across 4 ethnic groups (African, East Asian, South Asian, Caucasian) and 2 genders to ensure statistically sufficient samples per intersectional group. Additionally, RFW (balanced with 6,000 pairs per 4 ethnicities) is used as a control benchmark to avoid bias introduced by sampling.

**3. Multi-dimensional Fairness Metrics: Cross-Verification to Avoid Misleading Results**

Single fairness values can be misleading. Therefore, the authors report four complementary FMR-based metrics: the maximum difference $\Delta$ measures absolute gaps, and the maximum ratio $R$ measures relative scale. To address the instability of $R$ when FMR approaches zero, metric $M$ (ratio of max to geometric mean) is introduced. Finally, the Gini coefficient $G$ characterizes global inequality ($G = 0$ indicates perfect fairness):

$$G = \frac{\sum_i\sum_j |e_i - e_j|}{2K\sum_i e_i}$$

The Decidability index $d' = (\mu_{genuine} - \mu_{impostor})/\sqrt{\tfrac12(\sigma_{genuine}^2 + \sigma_{impostor}^2)}$ is also reported to measure the separation between genuine and impostor score distributions. This multi-dimensional perspective allows the authors to identify "false fairness," where low-accuracy models appear fair simply because their error rates are uniformly high across all groups.

### Loss & Training
This is an evaluation study and does not involve training. All models use pre-trained weights with a unified text prompt format.

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
- **FaceLLM-8B Superiority**: As the only face-specific MLLM, it significantly outperforms general MLLMs on both benchmarks (EER 5.13% vs. 8.54% for the runner-up).
- **Different Bias Patterns**: On RFW, all MLLMs achieve the lowest (best) EER on the Asian group, whereas traditional systems typically perform best on Caucasians.
- **Most Accurate ≠ Fairest**: While FaceLLM-8B has the smallest ethnicity gap on IJB-C ($\sigma=0.58$), it is not the fairest on RFW ($\sigma=4.98$).
- **False Fairness in Low Accuracy**: Ovis1.5-Llama3-8B shows a deceptively low $\sigma=0.99$ in ethnicity difference, but its EER is 30.89%—all groups perform "uniformly poorly."
- **Gender Bias**: Most MLLMs show higher EER for females than males (1-4% gap); FaceLLM-8B has the smallest gap (0.55%).

## Highlights & Insights
- **Closing the Gap**: First attempt to extend demographic fairness analysis from traditional recognition to the MLLM domain.
- **Multi-dimensional Metrics**: The combination of 4 FMR-based metrics and decidability index avoids the pitfalls of single-metric evaluation.
- **Discovery of "False Fairness"**: Reveals that low-accuracy models can masquerade as fair models (e.g., Ovis1.5).
- **Practical Implications**: As MLLMs are considered replacements for traditional face systems, understanding their fairness profiles is critical for responsible deployment.
- **New Bias Patterns**: Better performance on Asian groups may reflect differences in MLLM pre-training data distributions compared to traditional systems.
- **Qwen2-VL-7B Gender Consistency**: A gender gap of only 0.10% suggests certain architectural designs might naturally favor gender fairness.
- **Domain-Specific Advantage**: FaceLLM's EER gap (5.13% vs 8.54%) underscores the necessity of specialized fine-tuning for facial tasks.

## Limitations & Future Work
- Extreme computational cost (approx. 20 H100 days per model) limited the number of evaluated models and the sampling scale on IJB-C.
- Evaluation was restricted to open-source models, excluding closed-source MLLMs like GPT-4V or Gemini.
- Fairness analysis was limited to ethnicity and gender, excluding finer-grained attributes like age or skin tone.
- Lack of deep analysis into the **sources** of bias (e.g., pre-training data imbalance vs. model architecture).
- Overall face verification performance of MLLMs still lags behind specialized systems, limiting practical utility.
- No exploration of whether in-context learning or prompt engineering can mitigate bias.
- Inconsistent bias patterns across datasets (IJB-C vs. RFW) necessitate larger-scale unified evaluations.

## Related Work & Insights
- Distinction from FaceRecBench and FaceXBench: The latter focus on MLLM face capability accuracy, while this work focuses on the **fairness** dimension.
- NIST FRVT series established fairness standards for traditional systems; this work extends those to MLLMs.
- Fairness metric selection follows the ISO/IEC 19795-10:2024 standard.
- FaceLLM's performance indicates that general-purpose MLLMs indeed require domain-specific fine-tuning for facial tasks.

## Technical Details Supplementary
- **Template-level Aggregation**: $s(T_p, T_g) = \frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n} s_{ij}$
- **Threshold Scanning**: $\tau \in [0,1]$ with a step of 0.005 (201 operating points).
- **Decidability**: $d' = \frac{\mu_{genuine} - \mu_{impostor}}{\sqrt{\frac{1}{2}(\sigma_{genuine}^2 + \sigma_{impostor}^2)}}$
- **IJB-C Sampling**: 10,000 pairs, requiring ~20 H100 days per model.
- **LLaVA-NeXT Failure**: Only generated 2 unique similarity scores, failing to provide meaningful verification.
- **Performance Drop on RFW**: All MLLMs showed significant EER increases (29-50%), indicating a more challenging benchmark.
- **4 Fairness Metrics**: $\Delta$ (Difference), $R$ (Ratio), $M$ (Ratio to Geometric Mean), $G$ (Gini Coefficient).
- **Gini Coefficient Definition**: $G = \frac{\sum_i\sum_j |e_i - e_j|}{2K\sum_i e_i}$, where $G=0$ is perfect fairness.
- **Ratio Instability**: $R$ approaches infinity as the minimum group FMR nears zero; $M$ mitigates this using the geometric mean.
- **Model Family Coverage**: 9 models from 6 families including Idefics3, Ovis, Qwen2-VL, Qwen2.5-VL, Valley, LLaVA-NeXT, and FaceLLM.
- **IJB-C Template Structure**: Each template contains between one and several hundred images or video frames.
- **Zero-shot Protocol**: No fine-tuning or in-context examples provided to any model.
- **TMR Metrics**: True Match Rate reported at FMR levels of 10%, 1%, and 0.1%.
- **Cross-Benchmark Variance**: Bias patterns shifted significantly between IJB-C and RFW (e.g., FaceLLM's $\sigma$ was lowest on IJB-C but higher on RFW).
- **Parameter Range**: Models range from 2B (Qwen2-VL-2B) up to 8B (FaceLLM-8B).

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation of MLLM face verification fairness; identified unique bias patterns.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 models across 2 benchmarks with multiple metrics, though sampling was limited by compute costs.
- Writing Quality: ⭐⭐⭐⭐ Rigorous metric definitions and thorough analysis.
- Value: ⭐⭐⭐⭐ Provides significant reference for the responsible deployment of MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[CVPR 2026\] Towards Robust Multimodal Large Language Models Against Jailbreak Attacks](towards_robust_multimodal_large_language_models_against_jailbreak_attacks.md)
- [\[CVPR 2026\] Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models](towards_reasoning-preserving_unlearning_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
