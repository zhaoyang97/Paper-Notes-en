---
title: >-
  [Paper Note] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models
description: >-
  [ACL 2025][Social Computing][Multimodal Safety] Proposes the concept of "dual-implicit toxicity" — bias and discrimination that can only be identified by combining both textual and visual modalities. It constructs the MDIT-Bench benchmark containing 317K questions across 12 categories and 23 subcategories, and reveals a substantial amount of activatable hidden toxicity in mainstream large multimodal models through long-context jailbreaking.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Multimodal Safety"
  - "Implicit Toxicity"
  - "Dual-Implicit Toxicity"
  - "Safety Evaluation of Large Models"
  - "Long-Context Jailbreaking"
date: 2026-05-08
content_hash: 149b4e020b699836
---

# MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models

**Conference**: ACL 2025  
**arXiv**: [2505.17144](https://arxiv.org/abs/2505.17144)  
**Authors**: Bohan Jin, Shuhan Qi, Kehai Chen, Xinyi Guo, Xuan Wang (HIT Shenzhen, Univ. Barcelona)  
**Code**: [nuo1nuo/MDIT-Bench](https://github.com/nuo1nuo/MDIT-Bench)  
**Area**: Social Computing  
**Keywords**: Multimodal Safety, Implicit Toxicity, Dual-Implicit Toxicity, Safety Evaluation of Large Models, Long-Context Jailbreaking  

## TL;DR

Proposes the concept of "dual-implicit toxicity" — bias and discrimination that can only be identified by combining both textual and visual modalities. It constructs the MDIT-Bench benchmark containing 317K questions across 12 categories and 23 subcategories, and reveals a substantial amount of activatable hidden toxicity in mainstream large multimodal models through long-context jailbreaking.

## Background & Motivation

### Background
Large Multimodal Models (LMMs) such as GPT-4o and Gemini are widely utilized, but their outputs may contain harmful and discriminatory content. Existing safety research primarily focuses on **explicit toxicity** (directly containing abusive language) and **unimodal implicit toxicity** (implicitly harmful content detectable via a single modality), while ignoring the more covert forms of cross-modal toxicity.

### Limitations of Prior Work
- Most toxicity benchmarks focus on explicit or unimodal implicit toxicity, lacking evaluations for toxicity that requires **cross-modal reasoning** to be detected.
- Much of the work is limited to the text-only domain, lacking multimodal safety evaluations.
- Existing benchmarks have limited data scales and mostly rely on judge models for scoring — however, in dual-implicit toxicity scenarios, even the strongest models perform poorly and are incapable of acting as judges.

### Core Motivation
To fill the blank in fine-grained safety evaluations of bias and discrimination in multimodal scenarios. Key Observation: When key information in the query is replaced by an image (e.g., "Is the person in this image suitable to be a leader?"), toxicity only surfaces when the text and the image are combined — looking at the text alone yields a neutral question, and looking at the image alone is harmless.

## Method

### Overall Architecture
The construction process is divided into four stages: **question generation → data cleaning → modality expansion → benchmark construction**. The core method is termed Multi-stage Human-in-the-loop In-context Generation. Human intervention is introduced in each stage to align with human values.

### Key Designs

#### Key Design 1: Definition and Classification of Dual-Implicit Toxicity

Toxicity is classified into three levels based on its stealthiness:
- **Explicit Toxicity**: Contains direct insults or discriminatory language, making it easy to detect.
- **Unimodal Implicit Toxicity**: Free of offensive vocabulary, but detectable within a single modality through metaphors, sarcasm, etc.
- **Dual-Implicit Toxicity**: Both text and image are individually harmless, but present bias/discrimination only when combined.

The dataset covers 12 toxicity categories (racism, sexism, classism, homophobia, nationalism, ageism, ableism, religious discrimination, looksism, subculture discrimination, neuro-discrimination, and others), 23 subcategories, and 780 specific topics.

#### Key Design 2: Multi-stage Human-in-loop Data Generation

1. **Question Generation**: Seed questions are collected from sources like CVALUES, and implicit toxicity questions are human-created. "Pseudo-multimodal" versions are then constructed by replacing key toxicity words with "the [] in the image". These are utilized as exemplars for ICL (In-Context Learning) expansion.
2. **Data Cleaning**: Filtering is performed using the distribution of Replaced Words to retain 780 high-quality keywords.
3. **Modality Expansion**: Web images are crawled using the replaced words as keywords, and blurry/irrelevant images are manually filtered, resulting in 29,097 images.
4. **Benchmark Construction**: For each question, 5 options are constructed — Ans1 (non-toxic correct answer), Ans2 (toxic answer), Ans3 (long answer with embedded toxicity), Ans4 (image description distractor), and Ans5 (confusing option with replaced keywords).

#### Key Design 3: Difficulty Grading and Hidden Toxicity Measurement

Three difficulty levels:
- **Easy**: Explicit/unimodal implicit toxicity based on MMHS150K, containing 91,892 questions.
- **Medium**: Dual-implicit toxicity in the MDIT-Dataset, containing 112,873 questions.
- **Hard**: Built on top of Medium by adding Long-Context Jailbreaking, injecting a large number of toxic exemplars (32/64/128-shot) before the prompt.

The Hidden Toxicity (HT) metric is proposed to quantify the toxicity increase of the model at the hard level compared to the medium level:

$$HT(\mathcal{G}) = \sum_{i \in N} \left(1 - \frac{Acc_{n=i}}{Acc_{n=0}}\right) \cdot \text{Norm}_N(i)$$

Where $N=\{32,64,128\}$, and the normalization factor follows a power-law decay. Higher HT indicates more hidden toxicity in the model.

## Key Experimental Results

### Experiment 1: Accuracy at Easy and Medium Levels

| Model | Acc (Medium) | Acc (Easy) |
|------|-------------|------------|
| Qwen2-VL-7B| **67.2%** | 85.9% |
| Qwen2-VL-72B-AWQ | 65.5% | 87.7% |
| LLaVA-NeXT | 42.3% | 79.7% |
| LLaVA-1.5-13B | 35.9% | 71.1% |
| LLaVA-1.5-7B | 27.2% | 67.1% |
| BLIP2 | 40.9% | 75.3% |
| CogVLM2 | 16.3% | 72.2% |
| InstructBLIP | 12.4% | 33.2% |
| Random Baseline | 20.0% | 20.0% |

The vast majority of models achieve far lower accuracy on the Medium level than the Easy level, with InstructBLIP and CogVLM2 even scoring below the random baseline.

### Experiment 2: Hard Level and Hidden Toxicity Metrics

| Model | Acc (Med.) | Acc (32-shot) | Acc (64-shot) | Acc (128-shot) | HT |
|------|-----------|---------------|---------------|----------------|------|
| Qwen2-VL-7B | 67.2% | 47.7% | 41.8% | 33.7% | 0.476 |
| Qwen2-VL-72B-AWQ | 65.5% | 37.2% | 32.3% | 30.8% | 0.496 |
| BLIP2 | 40.9% | 22.5% | 20.2% | 19.5% | **0.530** |
| LLaVA-NeXT | 42.3% | 35.1% | 32.9% | — | 0.298 |
| LLaVA-1.5-13B | 35.9% | 28.9% | 26.8% | — | 0.279 |

BLIP2 exhibits the highest hidden toxicity (0.530). Although Qwen2-VL-7B performs best on Medium (67.2%), its performance plummets to 33.7% after 128-shot jailbreaking, resulting in an HT of 0.476.

### Closed-source Model Results (Subset Evaluation)

| Model | Acc (Med.) | HT |
|------|-----------|------|
| Gemini-1.5-Pro | 65.65% | 0.296 |
| Claude-3.5-Sonnet | 53.37% | 0.261 |
| GPT-4o | 41.50% | 0.124 |
| GPT-4o-mini | 43.63% | 0.401 |

GPT-4o has the lowest hidden toxicity, but its accuracy on Medium is also low, indicating that its toxicity might have already "leaked" under standard conditions. While Gemini performs best on Medium, its HT remains high.

## Key Findings

1. **Dual-implicit toxicity is a universal blind spot for LMMs**: Even the strongest Qwen2-VL-7B achieves only 67.2% accuracy, leaving a prominent gap before reaching safety standards.
2. **Model scale is not always better**: Qwen2-VL-7B slightly outperforms the 72B version. Larger models may generate longer replies and fall into the Ans3 trap.
3. **Detection of different toxicity categories varies significantly**: Sexism and neurological discrimination are detected well, whereas classism and subcultural discrimination are difficult to detect, potentially due to fewer corresponding training instances.
4. **Hidden toxicity is widespread and can be progressively activated**: As the number of shots increases, the proportion of chosen toxic options rises almost linearly (on a logarithmic scale), following a power-law relationship.
5. **Performance on Medium and Hard is not strictly correlated**: Successful performance on Medium does not guarantee low HT, implying that explicit safety alignment does not eliminate deeper biases.

## Highlights & Insights

- **Significant conceptual contribution**: Clearly defines the three-tiered toxicity system of explicit → unimodal implicit → dual-implicit, filling a conceptual gap in cross-modal safety evaluation.
- **Practical data construction method**: The Multi-stage Human-in-the-loop ICL method balances automation and quality successfully, scaling up to 317K with minimal manual iterations.
- **Ingenious five-option design**: Ans3 (middle toxicity embedding) tests paragraph-level toxicity detection, Ans4 (image description) tests instruction understanding, and Ans5 (keyword substitution) tests the utilization of visual information. This evaluates model capabilities from multiple dimensions.
- **Inspiring HT metric**: Disentangles "manifested toxicity" and "hidden activatable toxicity" measurements, offering a more fine-grained safety alignment evaluation dimension.
- **Human evaluation verifies benchmark validity**: The second-stage human evaluation achieves an overall accuracy of 98%, confirming the reliability of toxicity labels.

## Limitations & Future Work

- **Covers only bias & discrimination toxicity**: Does not involve other safety dimensions such as privacy leaks or guidance on dangerous activities.
- **Limitations of the multiple-choice format**: Cannot evaluate free-form generation behaviors, nor does it enforce outputting reasoning chains, limiting the analysis of model decision mechanisms.
- **Data generation dependency on models**: Seed scaling relies on LLM ICL, which might introduce systematic bias.
- **Images sourced from web scraping**: Albeit undergoing anonymization and filtering, the image quality and representativeness still have limitations.
- **Lack of defense/detoxification solutions**: Only evaluates toxicity without proposing mitigation methods specifically tailored to dual-implicit toxicity.
- **Closed-source models evaluated only on subsets**: Cost limits prevented precise evaluations of models like GPT-4o on the entire dataset.

## Related Work & Insights

- **MLLMGUARD** (Gu et al. 2024): 12 classes of social media data + red-teaming evaluation, but focuses on explicit toxicity.
- **SafeBench** (Ying et al. 2024): 2,300 harmful queries labeled by LLM judge, limited in scale.
- **SALAD-Bench** (Li et al. 2024): Contains attack-enhanced / defense-enhanced subsets, but is restricted to text-only.
- **MM-SafetyBench** (Liu et al. 2025): A 4-step multimodal safety evaluation, but does not focus on implicit toxicity requiring cross-modal reasoning.
- **SIUO** (Wang et al. 2024b): A cross-modal safety alignment challenge set, complementary to this work.
- **Many-shot Jailbreaking** (Anil et al. 2024, NeurIPS): Inspired the hard-level long-context jailbreaking method used in this paper.

**Insight**: The key challenge of dual-implicit toxicity is that models must **simultaneously perform** cross-modal information integration and deep semantic understanding — existing models struggle to coordinate these two tasks. This suggests that future safety alignment needs to intervene at the multimodal fusion layer rather than merely during text decoding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Systematically defines and evaluates dual-implicit toxicity for the first time; the conceptual contribution is clear and significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experimental design covering 13 models, 317K data, and three difficulty levels with human verification. However, closed-source models were only tested on a subset, which is slightly regrettable.
- Writing Quality: ⭐⭐⭐⭐ — Structured and clear, with rich tables and a reasonable toxicity classification system.
- Value: ⭐⭐⭐⭐ — Highlights a crucial blind spot in multimodal safety alignment, although the absence of detoxification solutions slightly reduces its practical prescriptive value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

</div>

<!-- RELATED:END -->
