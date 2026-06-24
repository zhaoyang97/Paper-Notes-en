---
title: >-
  [Paper Note] Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation
description: >-
  [ACL 2025][Social Computing][disinformation] This study systematically evaluates the capabilities of 6 mainstream LLMs to generate personalized disinformation, finding that most LLMs can generate high-quality personalized fake news. Furthermore, personalization requests actually reduce the trigger rate of safety filters (acting as a form of jailbreak) and slightly decrease the detectability of machine-generated texts.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "disinformation"
  - "personalization"
  - "LLM safety"
  - "safety filter"
  - "machine-generated text detection"
date: 2026-05-08
content_hash: 4a42edefaa0253ca
---

# Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation

**Conference**: ACL 2025  
**arXiv**: [2412.13666](https://arxiv.org/abs/2412.13666)  
**Code**: [GitHub](https://github.com/kinit-sk/personalized-disinfo) (dataset, request required)  
**Area**: Social Computing  
**Keywords**: disinformation, personalization, LLM safety, safety filter, machine-generated text detection

## TL;DR
This study systematically evaluates the capabilities of 6 mainstream LLMs to generate personalized disinformation, finding that most LLMs can generate high-quality personalized fake news. Furthermore, personalization requests actually reduce the trigger rate of safety filters (acting as a form of jailbreak) and slightly decrease the detectability of machine-generated texts.

## Background & Motivation
**Background**: LLMs have been proven capable of generating high-quality disinformation articles while also demonstrating content personalization capabilities.

**Limitations of Prior Work**: The combination of disinformation generation and personalization capabilities has not been systematically investigated. Prior work mostly focused on OpenAI's proprietary models, lacked evaluation of open-source models, and mostly relied on qualitative/anecdotal evidence.

**Key Challenge**: Malicious actors might exploit LLMs to generate personalized disinformation targeting specific demographics at scale, but there is a lack of systematic evidence to evaluate the severity of this threat.

**Goal**: (a) Can LLMs generate high-quality personalized disinformation? (b) Can LLM meta-evaluation substitute human evaluation of personalization quality? (c) Does personalization affect the detectability of machine-generated text?

**Key Insight**: Build the PerDisNews dataset (6 LLMs $\times$ 6 false narratives $\times$ 7 target groups $\times$ 3 personalization levels $\times$ 3 repetitions = 2268 articles) to comprehensively evaluate across four dimensions: generation quality, safety filtering, personalization quality, and detectability.

**Core Idea**: Use large-scale controlled experiments to prove that personalization requests essentially act as a jailbreak, reducing the effectiveness of LLM safety mechanisms.

## Method

### Overall Architecture
Experimental workflow: Select 7 target groups (grouped by political orientation/residence/age) and 6 European false narratives (health + politics) $\rightarrow$ use prompts of 3 personalization levels (No/Simple/Detailed) to prompt 6 LLMs to generate 3 articles each $\rightarrow$ evaluate from four aspects: linguistic quality, narrative stance, personalization quality, and detectability.

### Key Designs
1. **Three-level personalization prompt design**:

    - **Function**: Set up three prompt variations: No (no personalization baseline), Simple (target group name only), and Detailed (group name + detailed description).
    - **Core Idea**: Simple relies on the LLM's internal knowledge to understand the target group, while Detailed provides external attribute descriptions to guide the generation.
    - **Design Motivation**: Compare the variation in LLM responses to personalization instructions, especially the behavior of safety filters under different levels.

2. **Multi-LLM meta-evaluation of personalization quality**:

    - **Function**: Use three models (GPT-4o, Gemma-2-27b-IT, Llama-3.1-70B) to score each article and assess personalization quality (0-3 scale).
    - **Core Idea**: Average multi-model evaluations to reduce single-model bias, validating the correlation with 5 human annotators on a 109-article subset (Spearman $\rho = 0.76$).
    - **Design Motivation**: Purely human evaluation is costly and exposes annotators to harmful content; LLM meta-evaluation is scalable and reproducible.

3. **Evaluation of machine-generated text detectability**:

    - **Function**: Detect personalized/non-personalized text using 3 SOTA detectors (finetuned Gemma-2-9b-IT, Detection-Longformer, Binoculars).
    - **Core Idea**: Compare the detection True Positive Rate (TPR) and average confidence under different personalization levels.
    - **Design Motivation**: Verify whether personalization makes generated text more difficult to be identified as machine-generated.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Key Finding | Concrete Evidence |
|----------|---------|---------|
| Safety Filtering | Gemma is the safest (65% trigger rate), while GPT-4o/Mistral rarely trigger | Comparison of 6 LLMs |
| Personalization Quality | All LLMs except Falcon can generate high-quality personalized disinformation | 2268 articles in PerDisNews |
| Personalization = Jailbreak | Personalization reduces safety filtering triggers (No: 5.2% $\rightarrow$ Detailed: 3.5%) | Statistically significant |
| Detectability | Personalization slightly reduces detection rates (average TPR drops from 0.91 to 0.88) | 3 detectors |

### Detection Experiments

| Detector | TPR (No) | TPR (Detailed) | Decrease |
|--------|----------|----------------|------|
| Gemma-2-9b-IT | 0.9960 | 0.9960 | 0.00 |
| Detection-Longformer | 0.8968 | 0.8333 | -0.063 |
| Binoculars | 0.8333 | 0.8029 | -0.030 |

### Key Findings
- Political orientation (especially European conservatives) is the easiest to personalize, while student and urban populations are the hardest.
- Meta-evaluation has a strong correlation with human evaluation ($\rho = 0.76$), but lower agreement on middle scores (1-2 points).
- Health narrative H2 (cannabis cures cancer) and political narrative P1 (EU insect food) are the easiest for LLMs to agree to generate.

## Highlights & Insights
- The finding that personalization requests act as a jailbreak is a significant discovery: safety teams usually do not protect against personalization as an attack vector, yet detailed descriptions of target audiences indeed cause models to lower safety constraints.
- The cross-meta-evaluation scheme using three LLMs is a reusable evaluation design pattern to mitigate self-preference bias.

## Limitations & Future Work
- Limited to English, multilingual scenarios are not validated.
- The number of 6 false narratives is limited and does not cover the latest current events.
- The persuasive effects of the generated content on real users were not evaluated (only generation quality and detectability were assessed).
- There may be confounding factors (such as prompt length) between personalization and the reduction in safety filtering.

## Related Work & Insights
- **vs Vykopal et al. (2024)**: Expands the personalization dimension based on their non-personalized disinformation evaluation.
- **vs Gabriel et al. (2024)**: Expands from evaluating headline-only personalization to full-text content personalization.
- **vs Buchanan et al. (2021)**: Expands from evaluating GPT-3 only to a systematic comparison of 6 open/closed-source models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to systematically study the combined impact of personalization and disinformation on safety filtering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 2268 articles, multi-dimensional evaluation, human validation, but limited narratives and languages.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and comprehensive ethical discussion.
- **Value**: ⭐⭐⭐⭐ Direct reference value for LLM safety teams, revealing a novel threat of personalization acting as a jailbreak.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](../../ICLR2026/social_computing/socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[ACL 2025\] K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean](kda_automated_data_generation_pipeline_for_detoxifying_implicitly_offensive_lang.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)
- [\[ACL 2025\] Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement](is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](../../AAAI2026/social_computing/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)

</div>

<!-- RELATED:END -->
