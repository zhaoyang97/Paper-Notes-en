---
title: >-
  [Paper Note] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla
description: >-
  [ACL 2025][Social Computing][Bias Detection] This paper introduces BanStereoSet, a Bangla stereotypical bias dataset comprising 1,194 fill-in-the-blank instances covering 9 bias categories (including race, gender, religion, profession, physical appearance, age, caste, and region). It evaluates social biases in multilingual LLMs for Bangla, revealing that GPT-4o exhibits the highest bias while Mistral displays the lowest.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Bias Detection"
  - "Bangla"
  - "Stereotypes"
  - "Multilingual LLMs"
  - "Bias Benchmark"
date: 2026-05-08
content_hash: 1a8188f9ff31b6ee
---

# BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla

**Conference**: ACL 2025  
**arXiv**: [2409.11638](https://arxiv.org/abs/2409.11638)  
**Code**: [https://github.com/kamruzzaman15/BanStereoSet](https://github.com/kamruzzaman15/BanStereoSet)  
**Area**: Social Computing  
**Keywords**: Bias Detection, Bangla, Stereotypes, Multilingual LLMs, Bias Benchmark

## TL;DR
This paper introduces BanStereoSet, a Bangla stereotypical bias dataset comprising 1,194 fill-in-the-blank instances covering 9 bias categories (including race, gender, religion, profession, physical appearance, age, caste, and region). It evaluates social biases in multilingual LLMs for Bangla, revealing that GPT-4o exhibits the highest bias while Mistral displays the lowest.

## Background & Motivation
**Background**: Prior research on LLM bias has predominantly focused on English, utilizing benchmarks such as StereoSet and CrowS-Pairs. A limited number of studies address languages like French and Hindi, but research for Bangla—the 7th most spoken language globally with 237 million native speakers—remains virtually unexplored.

**Limitations of Prior Work**: Existing research on Bangla bias is confined to only gender and religion, failing to cover critical bias categories such as race, profession, physical appearance, age, caste, and region. Moreover, direct translations of English datasets fail due to cultural mismatches.

**Key Challenge**: Significant disparities exist between Bangla and English regarding cultural context, pronoun systems (Bangla has gender-neutral pronouns), and social bias structures, necessitating a culturally adapted dataset.

**Goal**: To construct a culturally adapted bias evaluation dataset for the Bangla context, covering a broader spectrum of bias categories.

**Key Insight**: Leveraging three source English datasets (StereoSet, GenAssocBias, and IndiBias), the authors translate and culturally adapt them for the Bangla context while resolving known data quality issues and introducing a new regional bias category.

**Core Idea**: Constructing the first Bangla stereotype dataset covering 9 bias categories through translation, cultural adaptation, and quality mitigation.

## Method

### Overall Architecture
Input: English bias datasets → Translation → Cultural Adaptation → Quality Mitigation → Human Verification → BanStereoSet (Fill-in-the-blank format)  
Evaluation: Querying multilingual LLMs to complete the fill-in-the-blank sentences (selecting stereotype/anti-stereotype/unrelated options) and measuring the stereotype percentage.

### Key Designs

1. **GPT-4 Translation + Human Review**:

    - **Function**: Translating English datasets to Bangla using few-shot GPT-4, followed by reviews and revisions from four native speakers.
    - Addressal of Bangla-specific linguistic issues: (a) Absence of gendered pronouns—resolved by appending "ekjon purush" (a man) or "ekjon mohila" (a woman) to differentiate gender (affecting 76 sentences); (b) Replacing English names with common Bangla names (e.g., John → Mehedi, 12 sentences); (c) Replacing Anglo-American cities with local places (e.g., Boston → Dhaka, 6 sentences).

2. **Mitigation of StereoSet Limitations**:

    - Mitigating five types of defects: meaningless stereotypes (47 sentences), misattributed stereotypes (9 sentences), invalid perturbations (15 sentences), incomparable target groups/attributes (58 sentences), and unnatural text (35 sentences).
    - Retention of instances based on multi-annotator discussions requiring at least 3/4 consensus.

3. **Introduction of New Regional Bias Data**:

    - Selecting 17 major Bangladeshi administrative districts/cities, utilizing GPT-4 to generate stereotype, anti-stereotype, and unrelated attributes for each.
    - Verifying cultural relevance of attributes through human annotation → leveraging GPT-4 to generate fill-in-the-blank sentences → translating → verifying and retaining items with at least 3/4 annotator agreement.

### Evaluation Setup
- 4 multilingual LLMs: GPT-4o, Mistral-7B, Llama3.1-70B, and Gemma2-27B.
- 3 prompt templates (baseline + 2 paraphrasings), with final results averaged.
- Evaluating concurrently in both English and Bangla for comparative analysis.

## Key Experimental Results

### Main Results (Bangla Stereotype Selection %; Ideal Value ≈50% or Lower)

| Bias Category | GPT-4o | Mistral | Llama | Gemma |
|---------|--------|---------|-------|-------|
| Gender | 76.8 | **58.0** | 73.9 | 81.3 |
| Race | 68.9 | **38.9** | 59.8 | 65.6 |
| Profession | 72.5 | **60.8** | 72.0 | 75.6 |
| Religion | 55.6 | 59.5 | **45.1** | 53.7 |
| Caste | 76.3 | **48.3** | 61.1 | 67.8 |
| Physical Appearance | 92.4 | **53.0** | 69.3 | 73.5 |
| Region | 82.1 | **67.4** | 78.7 | 81.5 |
| **Average** | 73.8 | **55.4** | 64.1 | 69.7 |

### Ablation: Bangla vs. English

| Model | Bangla Avg | English Avg | More Biased? |
|------|------------|---------|----------|
| GPT-4o | 73.8 | 63.8 | Bangla |
| Gemma | 69.7 | 66.0 | Bangla |
| Mistral | 55.4 | 59.9 | English |
| Llama | 64.1 | 65.5 | English |

### Key Findings
- GPT-4o exhibits the highest propensity for bias in Bangla (averaging 73.8%), particularly peaking at 92.4% in the physical appearance category.
- Mistral demonstrates the overall highest fairness (55.4%) and performs best in 7 out of 9 categories, suggesting that smaller models do not necessarily exhibit more bias.
- Religious bias is relatively lower across all models (45%–60%), potentially due to more extensive RLHF alignment regarding sensitive religious content.
- Generally, bias patterns between Bangla and English are similar, but individual models display inconsistent bias directions across the two languages.
- Regional bias is uniquely pronounced and meaningful in Bangla (whereas all models exhibit notably low regional bias, 11%–21%, in English).

## Highlights & Insights
- **Thoroughness of Cultural Adaptation**: Going beyond direct translation, this work addresses grammatical differences (pronouns), name and locality replacements, and mitigates StereoSet defects. This systematic pipeline serves as a template for building bias benchmarks in other low-resource languages.
- **Comprehensive Coverage of 9 Bias Categories**: Significantly expanding beyond previous Bangla studies that focused only on gender and religion, this dataset introduces culturally specific biases like caste and region.
- **Unveiling Language-Bias Interactions**: The variance in bias profiles of the same model across different languages (e.g., Mistral showing less caste bias in Bangla but more in English) demonstrates that social bias expression is language-specific.

## Limitations & Future Work
- Most data is translated from English datasets, which might not exhaustively reflect purely native Bangla stereotypical structures.
- The evaluation is restricted to binary gender, omitting non-binary gender identities.
- Sample sizes for certain categories are relatively small (religion: 56, caste: 60, region: 63), which may limit comprehensive assessment.
- Important bias categories such as sexual orientation, socioeconomic status, and disability are not yet covered.
- The dataset is culturally centered on Bangladesh, not accounting for other Bengali-speaking regions such as West Bengal, India.

## Related Work & Insights
- **vs. StereoSet**: BanStereoSet resolves several known limitations of StereoSet while executing rigorous cultural adaptation.
- **vs. IndiBias**: While IndiBias covers Indic languages, it adopts a different template. This work adapts only the caste-related component and reformats it into a fill-in-the-blank design.
- **vs. Sadhu et al.**: Previous research on Bangla bias only focused on gender and religion, whereas BanStereoSet expands this to 9 distinct categories.
- Provides a methodological roadmap for establishing bias evaluation benchmarks in other low-resource languages.

## Rating
- Novelty: ⭐⭐⭐ Primarily focused on dataset construction; methodological novelty is modest, but the cultural adaptation is highly systematic.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated across 4 models, 3 templates, and 2 languages, though some categories have small sample sizes.
- Writing Quality: ⭐⭐⭐⭐ Detailed description of the dataset creation pipeline and clear handling of translation nuances.
- Value: ⭐⭐⭐⭐ Fills a crucial gap in Bangla bias evaluation, providing a transferrable methodology for other low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)
- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[ACL 2025\] Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](llm_label_propagation.md)

</div>

<!-- RELATED:END -->
