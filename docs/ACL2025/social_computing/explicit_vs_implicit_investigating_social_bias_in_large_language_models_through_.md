---
title: >-
  [Paper Note] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection
description: >-
  [ACL 2025][Social Computing][implicit bias] Drawing on the Implicit Association Test (IAT) and Self-Report Assessment (SRA) from social psychology, this paper proposes a self-reflection evaluation framework to systematically study the explicit and implicit biases of LLMs. It finds that LLMs, similar to humans, exhibit an inconsistency between explicit and implicit biases—mild explicit bias but strong implicit bias—and this inconsistency becomes more severe with larger model s…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "implicit bias"
  - "explicit bias"
  - "IAT"
  - "self-reflection"
  - "social psychology"
date: 2026-05-08
content_hash: 3d6b72374de7cb3e
---

# Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection

**Conference**: ACL 2025  
**arXiv**: [2501.02295](https://arxiv.org/abs/2501.02295)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: implicit bias, explicit bias, IAT, self-reflection, social psychology

## TL;DR
Drawing on the Implicit Association Test (IAT) and Self-Report Assessment (SRA) from social psychology, this paper proposes a self-reflection evaluation framework to systematically study the explicit and implicit biases of LLMs. It finds that LLMs, similar to humans, exhibit an inconsistency between explicit and implicit biases—mild explicit bias but strong implicit bias—and this inconsistency becomes more severe with larger model sizes and more alignment training.

## Background & Motivation

**Background**: LLM bias research is abundant, but most focus either only on explicit bias (via direct questioning) or only on implicit bias (via embedding association), lacking a connection between the two.

**Limitations of Prior Work**: Social psychology in humans has proven that explicit and implicit biases are often inconsistent (e.g., explicitly supporting gender equality yet implicitly associating male = career), but this inconsistency has not been systematically studied in LLMs.

**Key Challenge**: Alignment methods such as RLHF have successfully reduced explicit biases in LLMs, but has implicit bias also been mitigated? If not, what is the root cause of this inconsistency?

**Goal**: To systematically compare the explicit and implicit biases of LLMs, and analyze the impacts of three factors: training data, model scale, and alignment methods.

**Key Insight**: Mapping IAT and SRA to LLM evaluation through prompt engineering, with the key innovation being the use of LLM self-reflection as an explicit bias measurement (letting the LLM evaluate its own performance in implicit tests).

**Core Idea**: The explicit and implicit biases of LLMs are highly inconsistent—training data, model scale, and alignment training reduce explicit bias but instead increase implicit bias.

## Method

### Overall Architecture
Design an IAT template to measure implicit bias -> LLM self-reflection to evaluate explicit bias -> compare across 6 social dimensions × 6 LLMs -> analyze the impact of training data, model scale, and alignment training.

### Key Designs

1. **Implicit Bias Measurement (Based on IAT)**

    - Prompt Design: `"<mask;> is [Attribute X] as <mask;> is [Attribute Y]"`
    - The LLM selects two words from a candidate word set to fill in the masks.
    - If the selection matches stereotypes (e.g., male-career, female-family), it is counted as an implicit bias.
    - 10 template variants × 200 experiments = 2000 measurements per dimension.
    - Design Motivation: Indirectly measure associations without directly mentioning social groups.

2. **Explicit Bias Measurement (Based on SRA + Self-Reflection)**

    - Replace masks in IAT templates with specific social group words.
    - Ask the LLM to rate this stereotypical statement using a 5-point Likert scale.
    - Key Innovation: Explicit measurement is modeled as self-reflection on the implicit measurement results.
    - Design Motivation: Measure whether the LLM is "aware" of its own implicit bias.

3. **Three-Factor Analysis (Based on Llama Family)**

    - Training data size: Llama variants trained on different data scales.
    - Model scale: 8B / 70B / 405B.
    - Alignment training: Base vs. instruction-tuned models, varying RLHF steps.
    - Design Motivation: Decouple the different impacts of three factors on the two types of biases.

## Key Experimental Results

### Main Results -- Stereotype Score (SC, higher meaning greater bias)

| Model | Implicit SC (Gender-Career) | Explicit SC (Gender-Career) | Gap |
|------|-------------------|-------------------|------|
| GPT-4o | **72%** | 8% | 64% |
| Claude-3.5 | **68%** | 5% | 63% |
| Llama-3.1-405B | **75%** | 12% | 63% |
| Llama-3.1-8B | 55% | 20% | 35% |

### Three-Factor Analysis

| Factor | On Explicit Bias | On Implicit Bias | Explanation |
|------|----------|----------|------|
| Increased Training Data | ↓ Decrease | **↑ Increase** | More data leads to stronger implicit bias |
| Increased Model Scale | ↓ Decrease | **↑ Increase** | Larger models are better at "hiding" bias |
| Alignment Training | **↓ Drastic Decrease** | → Mostly Unchanged | Alignment only suppresses explicit bias |

### Cross-Dimension Comparison

| Social Dimension | Average Implicit SC | Average Explicit SC |
|---------|------------|------------|
| Gender-Career | **72%**| 10% |
| Race (Black-White) | **65%** | 8% |
| Age | **60%** | 12% |
| Career-Gender | **58%** | 15% |

### Key Findings
- **All models exhibit explicit-implicit bias inconsistency across all dimensions**: Implicit bias is high (55-75% SC) while explicit bias is low (5-20% SC).
- **Alignment training only suppresses explicit bias**: Instruction-tuned models show a significant reduction in explicit bias, but implicit bias remains virtually unchanged.
- **Increasing model scale exacerbates inconsistency**: Larger models are better at being superficially "politically correct" while harbor deeper internal biases.
- **Increasing training data exacerbates implicit bias**: More training data means more social biases are encoded.
- **Claude 3.5 is the most adept at "hiding" bias**: It shows the lowest explicit SC, yet its implicit bias remains high.

## Highlights & Insights
- **The self-reflection methodology** rigorously transfers psychological measurement paradigms to LLM evaluation—which possesses a stronger theoretical foundation than simple prompt bias tests.
- **"Alignment only suppresses explicit bias"** is a cautionary finding—indicating that current RLHF merely teaches models to "not express bias" rather than to "not hold bias," similar to "political correctness ≠ lack of bias" in human society.
- **The opposing effects of training data, model scale, and alignment training** clearly explain why LLMs are becoming "larger" and "more aligned" but the issue of bias has not been truly solved.

## Limitations & Future Work
- IAT itself is controversial in human psychology (does it measure bias or cultural knowledge?).
- Computational cost is high across 14,400 experiments.
- Debiasing methods were not explored in this work.
- Future directions: comparing the performance of debiasing training on both types of biases, and testing under other cultural backgrounds.

## Related Work & Insights
- **vs Bai et al. (2024)**: They used IAT to measure implicit bias in LLMs; this paper adds explicit bias comparison and factor analysis.
- **vs Ganguli et al. (2023)**: They studied the impact of RLHF on moral self-correction; this paper finds that self-correction only operates at the explicit level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically compare LLM explicit/implicit biases and analyze three factors.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 models × 6 dimensions × 14,400 experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Solid psychological theoretical foundation, rigorous methodology.
- Value: ⭐⭐⭐⭐⭐ Far-reaching impact on AI fairness and alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ICML 2026\] Self-Debias: Self-correcting for Debiasing Large Language Models](../../ICML2026/social_computing/self-debias_self-correcting_for_debiasing_large_language_models.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)

</div>

<!-- RELATED:END -->
