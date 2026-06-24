---
title: >-
  [Paper Note] Can Large Language Models Address Open-Target Stance Detection?
description: >-
  [ACL 2025][LLM (Other)][stance detection] This paper proposes the Open-Target Stance Detection (OTSD) task—where targets are unseen during training and not provided as input. It systematically evaluates the performance of 8 LLMs across four families (GPT, Gemini, LLaMA, Mistral) in both target generation and stance detection stages, finding that LLMs generally outperform existing TSE methods, but their performance drops significantly when targets are non-explicit.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "stance detection"
  - "open-target"
  - "zero-shot"
  - "LLM evaluation"
  - "target generation"
  - "NLP"
date: 2026-05-08
content_hash: cd4224ee976478a9
---

# Can Large Language Models Address Open-Target Stance Detection?

**Conference**: ACL 2025  
**arXiv**: [2409.00222](https://arxiv.org/abs/2409.00222)  
**Code**: [AbuUbaida/opentarget](https://github.com/AbuUbaida/opentarget)  
**Area**: LLM/NLP  
**Keywords**: stance detection, open-target, zero-shot, LLM evaluation, target generation, NLP

## TL;DR
This paper proposes the Open-Target Stance Detection (OTSD) task—where targets are unseen during training and not provided as input. It systematically evaluates the performance of 8 LLMs across four families (GPT, Gemini, LLaMA, Mistral) in both target generation and stance detection stages, finding that LLMs generally outperform existing TSE methods, but their performance drops significantly when targets are non-explicit.

## Background & Motivation
**Background**: Stance Detection aims to determine the attitude (Favor/Against/None) of a text towards a specific target, which is a core task in social media analysis and public opinion monitoring. Existing research primarily focuses on Zero-Shot Stance Detection (ZSSD), where targets are unseen during training but are provided as inputs during inference.

**Limitations of Prior Work**: Almost all ZSSD methods assume that targets are known or provided to the model after manual annotation. However, in real-world scenarios, targets are often unknown, uncommon, or not explicitly expressed in the text, and annotating every possible target is highly costly.

**Key Challenge**: The only method attempting to address this issue, TSE (Li et al., 2023), can generate targets from text but relies on mapping them to a predefined target list (e.g., mapping "Religious diversity" to the nearest "Atheism" in the list), which is impractical in open-world scenarios.

**Ours**: Defines the OTSD task—generating targets directly from the text and detecting stance based on the generated targets, completely independent of predefined target lists, formalized as $x \xrightarrow{\text{generate}} t', \; x + t' \rightarrow y$.

**Key Insight**: Leveraging the zero-shot generation capabilities of LLMs to replace the traditional keyphrase extraction + target list mapping workflow, essentially treating OTSD as a testbed for evaluating LLM capabilities.

**Core Idea**: The strong context-understanding capability of LLMs makes them naturally suited for the two-step process of OTSD (target generation + stance detection), and joint single-step prompting (TG&SD) outperforms two-step separated prompting (TG+SD).

## Method

### Overall Architecture
- **Function**: Utilizing LLMs to identify the topic of discussion and determine the text's stance (Favor/Against/None) toward that target without any target information provided as input.
- **Design Motivation**: In real-world applications, users do not pre-specify what the target is; the model must understand by itself what topic the text is discussing and output a stance judgment. The approach of TSE, which relies on predefined target lists, cannot scale to open-world domains.
- **Mechanism**: Adopting a "Task Definition" prompting strategy (naming the task, defining inputs/outputs, and then requesting execution), and designing two prompting schemes: (1) TG+SD two-step method—generating the target first and then detecting stance to let the LLM focus step-by-step; (2) TG&SD single-step joint method—completing target generation and stance detection simultaneously in one prompt, allowing the model to better understand the text-target-stance relationship.

### Key Designs
1. **OTSD Task Definition and Formalization**

    - **Function**: Defines open-target stance detection as a two-stage problem: target generation (TG) and stance detection (SD), removing the target list mapping step in TSE.
    - **Design Motivation**: The mapping step of TSE requires a complete list of all possible targets, which is highly unrealistic in scenarios with high target diversity such as news comments or social media (the VAST dataset has 2,145 unique targets, and EZSTANCE has 6,873).
    - **Mechanism**: Formalized as $x \xrightarrow{\text{generate}} t'$, $x + t' \rightarrow y$, where $t'$ is the directly generated target (not a mapped result), and the stance $y \in \{\text{favor, against, none}\}$ is predicted based on the generated targets.

2. **BTSD Target Quality Evaluation Metric**

    - **Function**: Proposing a BERTweet-based stance classifier as an automatic evaluation metric for target generation quality (BTSD score).
    - **Design Motivation**: Targets generated by OTSD may have different wording from the gold targets but are semantically related (e.g., "gun control" vs. "permit to carry gun"), making traditional exact match metrics inapplicable. Furthermore, semantic similarity (SemSim) correlation with human judgment is only 0.57-0.59, which is not reliable.
    - **Mechanism**: Training a BERTweet classifier using 4 classic stance detection datasets (SemEval, AM, COVID-19, P-Stance, containing 19 targets), inputting the generated target + text into the classifier, and using F1-macro as a proxy metric for target quality. The correlation of this metric with human judgment (Kendall's $\tau$) reaches 0.74-0.85, significantly outperforming SemSim.

3. **Explicit vs. Non-Explicit Target Scenario Distinction**

    - **Function**: Partitioning dataset samples into two cases: targets explicitly mentioned in the text (explicit) and targets not explicitly mentioned (non-explicit), to evaluate them separately.
    - **Design Motivation**: The original TSE work did not distinguish between these two cases, but their difficulty levels vary drastically. Non-explicit targets require depth of reasoning and implicit semantic understanding from the model.
    - **Mechanism**: Partitioning by checking if the target words appear in the text after stopword removal, special character cleaning, and lemmatization.

## Key Experimental Results

### Table 1: Statistics of Three Datasets

| Dataset | Source | # Samples | # Unique Targets | Explicit / Non-Explicit | Stance Classes |
|--------|------|--------|-----------|------------|---------|
| TSE | Tweets | 3,000 | 6 | 1,804 / 1,196 | 3 |
| VAST | News Comments | 5,100 | 2,145 | 3,120 / 1,980 | 3 |
| EZSTANCE | Tweets | 9,313 | 6,873 | 9,313 / 149 | 3 |

### Table 2: Target Generation and Stance Detection Performance on the TSE Dataset (Explicit vs. Non-Explicit)

| Model | Explicit BTSD↑ | Explicit SC↑ | Non-Explicit BTSD↑ | Non-Explicit SC↑ |
|------|-----------|---------|-------------|-----------|
| TSE-Mapped | 36.63 | 38.10 | 30.56 | 32.00 |
| TSE-BestGen | 35.80 | 37.81 | 29.32 | 31.00 |
| GPT-3.5 (TG&SD) | 39.60 | 47.61 | 31.32 | 33.94 |
| GPT-4o (TG&SD) | **41.92** | 46.83 | **36.12** | 37.50 |
| Gemini-pro (TG&SD) | 40.92 | 45.71 | 34.85 | 35.96 |
| Llama-3-70B (TG&SD) | 41.52 | **49.84** | 34.67 | **35.50** |
| Mistral-large (TG&SD) | 41.39 | 49.76 | 35.42 | 34.70 |

### Table 3: Target Generation and Stance Detection Performance on the VAST Dataset

| Model | Explicit BTSD↑ | Explicit SC↑ | Non-Explicit BTSD↑ | Non-Explicit SC↑ |
|------|-----------|---------|-------------|-----------|
| GPT-4o (TG&SD) | **44.25** | 49.38 | 39.84 | 43.84 |
| Gemini-pro (TG&SD) | 42.78 | **51.46** | **40.53** | **48.53** |
| Llama-3-70B (TG&SD) | 42.50 | 48.73 | 42.02 | 46.57 |
| Mistral-large (TG&SD) | 43.13 | 51.30 | 39.98 | 46.55 |

### Key Findings
- **LLMs Comprehensively Outperform TSE**: On both target generation and stance detection stages, all tested LLMs outperform the two variants of TSE (TSE-Mapped and TSE-BestGen) in both explicit and non-explicit scenarios.
- **Non-Explicit Scenarios are the Bottleneck**: The performance of all models drops significantly under non-explicit target scenarios (BTSD drops by about 5-10 percentage points), as the text lacks sufficient implicit clues for models to infer the correct target.
- **Joint Prompting Outperforms Step-by-Step Prompting**: TG&SD (single-step joint) outperforms TG+SD (two-step separated) across most models and datasets, indicating that modeling target-stance relationships simultaneously benefits both subtasks.
- **No Absolute Advantage for Closed-Source vs. Open-Source**: GPT-4o achieves the overall best performance in target generation, but Llama-3-70B and Mistral-large frequently outperform the GPT series in stance detection.
- **"Antonymous Target" Issue**: GPT-4o sometimes generates targets that are semantically highly relevant but have opposite stance directions (e.g., gold target "permit to carry gun" $\rightarrow$ generated "gun control"), leading to stance reversal.

## Highlights & Insights
- **Meaningful Task Definition**: OTSD is closer to real-world application scenarios than ZSSD. Removing the assumption of predefined target lists greatly increases both the difficulty and the realism of the problem.
- **Ingenious Design of the BTSD Evaluation Metric**: Using the stance classifier's F1 as a proxy metric for target quality yields a high correlation with human judgment (0.74-0.85), solving the pain point where open generation is difficult to evaluate using exact matching.
- **Explicit/Non-Explicit Analysis Reveals Blind Spots**: Distinguishing between explicit and non-explicit targets reveals a performance gap unreported in the original TSE paper. Non-explicit scenarios represent a critical direction for future breakthroughs.
- **Broad Experimental Coverage**: 8 LLMs $\times$ 3 datasets $\times$ 2 prompting strategies $\times$ 2 scenarios, featuring a systematic and comprehensive experimental matrix design.

## Limitations & Future Work
- Only focusing on single-target scenarios, whereas real-world texts often contain multiple discussion targets that may be interrelated.
- Only experimenting on English; cross-lingual OTSD remains unexplored.
- Risk of data leakage—LLM pre-training data might contain parts of the test sets.
- The "antonymous target" issue (generated targets that are semantically related but opposite in stance direction) is not sufficiently addressed at the evaluation metric level, as BTSD is not sensitive enough to this.
- Small scale of human evaluation (500 samples per dataset with 3 annotators), providing limited validation strength for the reliability of the evaluation metrics.

## Related Work & Insights
- **vs. TSE** (Li et al., 2023): TSE is the closest pioneering work, but its generated targets must be mapped to a predefined list (e.g., "Religious diversity" $\rightarrow$ finding the nearest "Atheism" in the list), which is essentially still a semi-open setting. OTSD completely removes predefined lists, making it more challenging and practical. Experiments show that LLMs outperform TSE in both target generation and stance detection under the OTSD setting.
- **vs. ZSSD Methods** (Zhang et al., 2023; Allaway et al., 2021, etc.): Traditional zero-shot stance detection assumes targets are known during inference (only unseen during training), whereas OTSD does not provide targets during inference either, which is equivalent to adding a target recognition precursor task on top of ZSSD. OTSD can be viewed as a natural generalization of ZSSD.
- **vs. Cross-target SD** (Zhang et al., 2020): Although targets in cross-target stance detection are unseen during training, they usually belong to similar domains, whereas targets in OTSD can come from entirely different domains, demanding higher generalization capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ The OTSD task definition is meaningful and the BTSD evaluation metric design is novel; however, the core method consists only of prompt engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 models $\times$ 3 datasets $\times$ 2 strategies, featuring meticulous explicit/non-explicit analysis and human evaluation to validate metric reliability.
- Writing Quality: ⭐⭐⭐⭐ Clear task motivation and definition; the distinction between TSE and OTSD is intuitively illustrated with concrete examples.
- Value: ⭐⭐⭐ Provides a more realistic evaluation framework for the stance detection field, although practical deployment remains constrained by performance bottlenecks in non-explicit scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](bilingual_zero-shot_stance_detection.md)
- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)

</div>

<!-- RELATED:END -->
