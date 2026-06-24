---
title: >-
  [Paper Note] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?
description: >-
  [ACL 2025][vague quantifiers] This paper introduces the VAQUUM dataset (20,300 human ratings, 1,089 images) to systematically evaluate the alignment between vision-language models (VLMs) and humans regarding the use of vague quantifiers (e.g., *few*, *many*). The findings show that while VLMs are influenced by object counts similarly to humans, model performance varies significantly across different evaluation paradigms, indicating that judging and generating vague quantifier…
tags:
  - "ACL 2025"
  - "vague quantifiers"
  - "visual grounding"
  - "VLM evaluation"
  - "human judgment"
  - "multimodal"
date: 2026-05-08
content_hash: a0e7a578e31ee98d
---

# VAQUUM: Are Vague Quantifiers Grounded in Visual Data?

**Conference**: ACL 2025  
**arXiv**: [2502.11874](https://arxiv.org/abs/2502.11874)  
**Code**: [https://github.com/hughmee/vaquum](https://github.com/hughmee/vaquum)  
**Area**: Others  
**Keywords**: vague quantifiers, visual grounding, VLM evaluation, human judgment, multimodal

## TL;DR

This paper introduces the VAQUUM dataset (20,300 human ratings, 1,089 images) to systematically evaluate the alignment between vision-language models (VLMs) and humans regarding the use of vague quantifiers (e.g., *few*, *many*). The findings show that while VLMs are influenced by object counts similarly to humans, model performance varies significantly across different evaluation paradigms, indicating that judging and generating vague quantifiers depend on distinct cognitive processes.

## Background & Motivation
1. **Background**: Vague quantifiers (e.g., *a few*, *many*) are widely used in daily conversations, and their meanings are influenced by contextual factors (e.g., object quantity, size, individual beliefs). In NLP and multimodal research, most quantifier studies focus on precise quantifiers (e.g., *all*, *none*), leaving vague quantifiers under-explored.
2. **Limitations of Prior Work**: Existing studies often define vague quantifiers as fixed proportions (e.g., *few* = <17%), which eliminates their "vagueness" nature. There is a lack of large-scale human judgment datasets to measure VLM understanding of vague quantifiers in visual scenes. Furthermore, evaluation methods remain limited, failing to comprehensively reflect model behavior.
3. **Key Challenge**: The use of vague quantifiers depends on complex visual and contextual factors (count, area, real-world size). It remains unknown whether VLMs can flexibly adjust quantifier usage based on these factors like humans do, and it is even less clear whether different evaluation methods yield consistent conclusions.
4. **Goal**: (1) Which factors in visual scenes influence human judgments of vague quantifiers? (2) Are VLMs aligned with human behavior regarding these factors? (3) Do different evaluation paradigms (generation probability, numerical rating, multiple-choice questions) provide consistent results?
5. **Key Insight**: Constructing a benchmark featuring rich visual attribute annotations (object count, segmented area, real-world size norm) and using three distinct evaluation methods (probability extraction, prompted rating, multiple-choice questions) to comprehensively evaluate VLM alignment with humans.
6. **Core Idea**: Revealing that VLMs are only partially aligned with humans in vague quantifier usage and are highly sensitive to the evaluation method.

## Method

### Overall Architecture
The input consists of natural images and quantified statements ("There are [QUANT] [OBJECT] in the image", where QUANT $\in$ {few, a few, some, many, a lot of}), and the output is the "appropriateness" rating of the statement. Ratings are collected from both humans and VLMs to compare their alignment. For VLMs, three evaluation methods are employed: generation probability extraction, LLM numerical rating, and multiple-choice questions.

### Key Designs

1. **VAQUUM Dataset Construction**:

    - **Function**: To provide a human quantifier judgment benchmark with multi-dimensional visual feature annotations.
    - **Mechanism**: Images are merged from FSC-133 (an object counting dataset containing 7-3731 objects) and TallyQA (containing 2-15 objects). The 99 different counts are divided into 33 bins, and 33 images are sampled per bin, resulting in 1,089 images. Three visual features are annotated: (a) object count (uniformly sampled across bins); (b) segmented area (estimating the proportion of the image occupied by the object using CLIPSeg); (c) real-world size norm (typical real-world object size rated by humans, retrieved from the THINGSplus database). 203 native English-speaking participants were recruited, with each participant rating 100 statements on appropriateness using a slider.
    - **Design Motivation**: Existing datasets either use synthetic images (unrealistic) or define quantifiers by fixed proportions (eliminating vagueness). This dataset uses natural images, continuous ratings, and multi-dimensional features.

2. **Linear Mixed-Effects Model Analysis of Human Judgments**:

    - **Function**: To quantify the impact of various visual factors on human quantifier judgments.
    - **Mechanism**: An LMM is fitted to predict human ratings with quantifier, count, segmented area, and size norm as fixed effects, and participants and object categories as random effects. The model explains 50.3% of the total variance ($R^2_c = 0.503$). Interaction effects show that *few/a few* negatively correlate with count ($\beta = -0.37/-0.38$), while *many/a lot of* positively correlate with count ($\beta = 0.38/0.42$). The effects of area and size norm share the same direction but are weaker.
    - **Design Motivation**: To establish a precise statistical model of human behavior as a reference baseline for VLM evaluation.

3. **Three VLM Evaluation Paradigms**:

    - **Function**: To comprehensively evaluate the quantifier usage capabilities of VLMs from different perspectives.
    - **Mechanism**: (a) Generation Probability (Experiment 1): Prompting the VLM with "How would you describe the amount of [OBJECT]?" and extracting the log probabilities of each quantified statement, normalized by token length; (b) Numerical Rating (Experiment 2): Prompting the VLM to directly output a numerical appropriateness score for a statement; (c) Multiple-Choice Questions (Experiment 3): Asking the VLM to select the most appropriate statement among six choices. The alignment with human judgments across these evaluation methods is compared for 5 VLMs (BLIP-2, InstructBLIP, LLaVA-NeXT, LLaVA-OneVision, Molmo).
    - **Design Motivation**: Different evaluation methods may measure distinct capabilities—probabilities reflect intrinsic language modeling preferences, numerical ratings require metacognitive capabilities, and multiple-choice questions require comparison and judgment capabilities.

## Key Experimental Results

### Main Results — Spearman Correlation between Generation Probability and Human Judgment

| Model | few | a few | some | many | a lot of |
|------|-----|-------|------|------|---------|
| BLIP-2 | -0.18 | -0.19 | -0.06 | 0.14 | 0.13 |
| InstructBLIP | 0.06 | 0.04 | -0.03 | -0.01 | -0.04 |
| LLaVA-NeXT | **0.34** | **0.39** | **0.21** | **0.43** | **0.52** |
| LLaVA-OneVision | 0.30 | 0.40 | 0.22 | 0.52 | - |
| Molmo | ~0 | ~0 | ~0 | ~0 | ~0 |

### Human Judgments: Influence of Count (LMM Interaction Effects $\beta$)

| Quantifier | Count Interaction $\beta$ | Area Interaction $\beta$ | Size Norm Interaction $\beta$ |
|------|----------------|----------------|-------------------|
| few | -0.37 | -0.07 | -0.13 |
| a few | -0.38 | -0.10 | -0.11 |
| some | -0.20 | -0.05 | -0.07 |
| many | +0.38 | +0.08 | +0.14 |
| a lot of | +0.42 | +0.06 | +0.17 |

### Key Findings
- LLaVA series models align best with human judgments in generation probabilities (*few/a few* probability decreases as quantity increases, *many/a lot of* probability increases), whereas InstructBLIP and Molmo fail completely to distinguish among different quantifiers.
- The three evaluation paradigms yield inconsistent conclusions: probability extraction and multiple-choice methods show stronger alignment with humans, while the numerical rating prompt method exhibits much poorer alignment. This suggests that the "production" and "judgment" capabilities of VLMs may be driven by different mechanisms.
- Object count is the strongest factor affecting human quantifier judgments ($|\beta|$ 0.20-0.42), while segmented area (0.05-0.10) and real-world size (0.07-0.17) have weaker effects.
- In human judgments, *few* and *a few* behave almost identically, while *many* and *a lot of* are also highly similar—indicating that the "vague" boundaries of quantifiers are more consistent than predicted by some linguistic theories.
- The LMM model explains 50.3% of the variance, with inter-participant variance (0.042) being much larger than inter-object category variance (0.002), indicating that differences in quantifier usage are mainly driven by individual differences rather than object-specific attributes.

## Highlights & Insights
- The inconsistency across the three evaluation paradigms (probability/rating/multiple-choice) is the most valuable finding of this paper—it warns that the choice of method in VLM evaluation systematically affects the conclusions, and a single paradigm may mislead judgments about model capabilities.
- Using slider-based continuous ratings instead of discrete choices for collecting human judgments is an excellent design that preserves the continuous nature of vague quantifiers.
- Integrating psycholinguistic theories (e.g., Approximate Number System, subitizing threshold) with computational models provides a deeper cognitive science foundation for VLM evaluation.

## Limitations & Future Work
- Only 5 VLMs were tested, and all of them are open-source, omitting closed-source models such as GPT-4V.
- Object types in images are restricted to a subset of FSC-133, leaving complex scenes (e.g., co-occurrence of multiple object types) unexplored.
- Crucially, only English quantifiers were tested, and vague quantifier systems may vary significantly across different languages.
- The study does not explore the conditions under which VLMs produce "un-human" judgments (e.g., calling 10 objects "few"); such failure mode analysis would help understand model deficiencies.
- Human ratings were collected only via the Prolific platform (UK/Ireland), where cultural and linguistic backgrounds may influence quantifier usage.
- Large closed-source VLMs (e.g., GPT-4V, Gemini) were not included in the evaluation, which may miss the behavioral patterns of stronger models.

## Related Work & Insights
- **vs Testoni et al. (2024)**: They tested 3 VLMs' quantifier choices using synthetic images. This work uses natural images, 5 VLMs, and three evaluation paradigms, offering a more comprehensive assessment.
- **vs Enyan et al. (2024)**: They compared LLMs' performance on precise vs. vague quantifiers in a text-only setting. This work extends the investigation to grounded understanding in visual scenes.
- **vs Sorodoc et al. (2016/2018)**: They used fixed proportions to define quantifiers for training. This work preserves the vagueness of quantifiers and establishes ground truth using continuous human ratings.
- This dataset can be further utilized to train VLMs' quantifier usage capabilities or serve as a diagnostic tool for probing VLM counting abilities.

## Rating
- **Overall Evaluation**: An exemplary interdisciplinary study that provides crucial methodological warnings for VLM evaluations.
- **Novelty**: ⭐⭐⭐⭐ The first comprehensive work evaluating the visual grounding capabilities of VLMs on vague quantifiers.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Highly thorough with three evaluation paradigms and statistical modeling, though the model coverage is slightly narrow.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Seamless and natural interdisciplinary integration (linguistics + psychology + NLP).
- **Value**: ⭐⭐⭐⭐ Unveils the critical issue of evaluation paradigm dependency in VLMs.

<!-- VAQUUM: 20,300 human ratings, 1,089 images, 5 quantifiers, 5 VLMs, 3 evaluation paradigms -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)

</div>

<!-- RELATED:END -->
