---
title: >-
  [Paper Note] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection
description: >-
  [ACL 2025][AIGC Detection][AI Text Detection] Reveals that authors' sociolinguistic attributes (gender, CEFR level, academic discipline, language environment) systematically affect the accuracy of AI-generated text detectors, with language proficiency and environment causing the most prominent and consistent biases. A multi-factor WLS+ANOVA bias quantification framework is proposed.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "AI Text Detection"
  - "Sociolinguistic Bias"
  - "Author Attributes"
  - "ANOVA"
  - "Fairness"
date: 2026-05-08
content_hash: 01ea1995c2a4e58f
---

# Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection

**Conference**: ACL 2025  
**arXiv**: [2502.12611](https://arxiv.org/abs/2502.12611)  
**Code**: [https://github.com/leejamesss/AuthorAwareDetection](https://github.com/leejamesss/AuthorAwareDetection)  
**Area**: AIGC Detection  
**Keywords**: AI Text Detection, Sociolinguistic Bias, Author Attributes, ANOVA, Fairness  

## TL;DR
Reveals that authors' sociolinguistic attributes (gender, CEFR level, academic discipline, language environment) systematically affect the accuracy of AI-generated text detectors, with language proficiency and environment causing the most prominent and consistent biases. A multi-factor WLS+ANOVA bias quantification framework is proposed.

## Background & Motivation
**Background**: AI-generated text detection has rich benchmarks (RAID, MAGE, M4GT-Bench, etc.) covering multiple dimensions such as multi-models, multi-languages, and adversarial perturbations.

**Limitations of Prior Work**: Existing detectors and benchmarks focus almost exclusively on the model and data levels (sampling strategies, prompt engineering, adversarial augmentation), ignoring the "human" factor behind the text—differences in writing styles across different author groups can lead to systematic detector bias.

**Key Challenge**: Sociolinguistic research has long demonstrated that writing style systematically varies based on gender, language proficiency, academic background, and cultural environment (vocabulary choice, syntactic complexity, rhetorical conventions). However, detector training data and evaluation benchmarks do not consider this diversity, potentially unfairly penalizing specific populations.

**Goal** (a) Do and how do author attributes affect AI text detection accuracy? (b) Which attributes have the greatest and most consistent impact? (c) How do different detectors vary in their sensitivity to attribute bias?

**Key Insight**: Leveraging the ICNALE learner corpus (which has rich metadata annotations) to construct parallel texts of human-written and AI-generated content, evaluating multiple detectors under out-of-domain conditions, and isolating the independent effects of each attribute using multi-factor statistical analysis.

**Core Idea**: Introducing the sociolinguistic dimension of "who is writing" into the evaluation of AI text detection and utilizing a multi-factor WLS+ANOVA framework to quantify the detection bias caused by author attributes.

## Method

### Overall Architecture
Input: 5,138 human-written essays from 2,569 learners in the ICNALE corpus (with detailed demographic metadata) + parallel generation of each essay by 12 LLMs $\rightarrow$ totaling 66,794 texts. Each text is annotated with 4-dimensional author attributes (gender, CEFR level, academic discipline, language environment). These texts are fed into 9 off-the-shelf detectors (4 classifiers + 5 metric-based detectors) to obtain detection results under out-of-domain conditions, and then a multi-factor statistical framework is applied to analyze the impact of each attribute on detection accuracy.

### Key Designs

1. **67K Parallel Dataset Construction**:

    - Function: To construct a dataset that contains both human-written and AI-generated texts, with detailed author attribute annotations for each text.
    - Mechanism: Extract 5,138 human-written essays from ICNALE (including metadata such as gender, CEFR level A2-B2+NS, 4 categories of academic disciplines, and language environments EFL/ESL/NS). Then use 12 LLMs with parameters ranging from 0.5B to 72B (7 from the Qwen2.5 series, 4 from the LLaMA3 series, and 1 Mistral) to generate parallel AI texts for each essay. The prompt simulates the corresponding author's persona during generation.
    - Design Motivation: Existing datasets (e.g., TuringBench, RAID) are large and cover multiple models, but none contain author demographic metadata (the "Persona" column is always ✗), making them unable to support the analysis of sociolinguistic bias.

2. **Multi-Detector Out-of-Domain Evaluation**:

    - Function: To perform zero-shot evaluation on the constructed dataset using 9 off-the-shelf detectors of different categories.
    - Mechanism: Adopt the RAID evaluation paradigm—each detector outputs a scalar score, a threshold is set to fix the false positive rate (FPR) of human-written texts at 5%, and the detection accuracy on AI texts is calculated at this threshold. This includes classifiers (fine-tuned RoBERTa-Base/Large, RADAR) and metric-based detectors (GLTR, Binoculars, Fast-DetectGPT, DetectGPT, LLMDet).
    - Design Motivation: Skip fine-tuning on the target data and directly use off-the-shelf models to simulate real-world deployment scenarios where detectors must generalize.

3. **Multi-Factor WLS + Type II ANOVA Bias Analysis Framework**:

    - Function: To quantify the independent impact of each author attribute on detector accuracy while controlling for confounding effects of other attributes.
    - Mechanism: Take detection accuracy as the dependent variable and the four author attributes as categorical independent variables. Fit the model using Weighted Least Squares (WLS), where the weight $w_i$ is the sample size of each group: $\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} w_i (\text{accuracy}_i - \beta_0 - \sum_{k=1}^{p} \beta_k x_{ik})^2$. Then, perform Type II ANOVA to assess the unique contribution of each attribute by removing it one by one. For attributes significant in ANOVA, further use Least Squares Means (LSMeans, adjusted means after controlling for other factors) with Wald tests and Holm correction for post-hoc pairwise comparisons.
    - Design Motivation: Single-factor tests (such as t-test or one-way ANOVA) cannot handle correlations between attributes (e.g., CEFR level and language environment may be correlated). A multi-factor framework isolates the true independent effects of each factor, preventing spurious conclusions.

### Loss & Training
This paper does not involve model training; its core contribution is the evaluation and statistical analysis framework. The significance threshold for statistical analysis is set at $\alpha=0.05$, and post-hoc comparisons use Holm correction to control for multiple comparison errors.

## Key Experimental Results

### Main Results
ANOVA results show the significance of each attribute across different detectors ($p<0.05$ is significant):

| Attribute | Significant Detectors / Total | Consistency | Conclusion |
|------|----------------|--------|------|
| CEFR Level | 10/10 | Significant across all detectors | **Strongest and most consistent source of bias** |
| Language Environment | 8/10 | Significant for most detectors | Important bias factor second only to CEFR |
| Academic Discipline | 5/10 | Detector-dependent | Bias exists in about half of the detectors |
| Gender | 0/10 | None are significant | No evidence of gender bias |

### Ablation Study (LSMeans Comparison)
Adjusted accuracy of CEFR level on the Binoculars detector:

| CEFR Level | Binoculars Accuracy | Difference from NS (XX_0) |
|----------|-----------------|----------------|
| A2_0 | 0.9482 | +5.0pp (Significant) |
| B1_1 | 0.9443 | +4.6pp (Significant) |
| B1_2 | 0.9475 | +4.9pp (Significant) |
| B2_0 | 0.9507 | +5.3pp (Significant) |
| XX_0 (NS) | 0.8981 | Baseline |

Impact of language environment on detection accuracy (Binoculars):

| Language Environment | Accuracy | Description |
|---------|--------|------|
| EFL | 0.9482 | Highest—detectors more easily identify AI texts from EFL authors |
| ESL | 0.9337 | Intermediate level |
| NS | 0.8981 | Lowest—native speakers' texts are harder to classify correctly |

### Key Findings
- **CEFR level is the strongest source of bias**: All 10 detectors exhibit significant bias. Texts by non-native speakers (especially those with lower proficiency) are more likely to be correctly classified as AI-generated (possibly because their writing patterns differ more from AI styles), while native speakers' texts are more easily misclassified.
- **Language environment has a widespread impact**: All pairwise comparisons among EFL/ESL/NS for Binoculars are significant, with EFL accuracy being the highest and NS the lowest. This indicates that writing under native-speaker environments is more likely to be misidentified as AI-generated text.
- **No significant impact from gender**: This is good news, showing that current detectors demonstrate no systematic bias on the gender dimension.
- **Academic discipline bias varies by detector**: DetectGPT distinguishes Humanities vs. STEM and Life Sciences; GPT2-base/large focuses on differences between STEM vs. Humanities/Social Sciences; LLMDet is most sensitive to Life Sciences.
- **Naive thresholds cause catastrophic false positive rates**: For instance, GLTR has an FPR of 100% at a threshold of 0.25, and LLMDet has an FPR of 75.3% at 0.95, showing that FPR-calibrated evaluation is crucial.

## Highlights & Insights
- **Introduction of sociolinguistic perspective to AI detection evaluation**: Treating "who is writing" as a key variable fills a blind spot in existing detection benchmarks. This is cleverly done by leveraging the rich metadata in the existing learner corpus (ICNALE), avoiding high annotation costs from scratch.
- **Generality of the multi-factor WLS+ANOVA framework**: This statistical framework is not only applicable to text detection bias analysis but can also be transferred to any scenario requiring quantification of the impact of multi-dimensional demographic attributes on model performance (e.g., fairness auditing of resume screening or content moderation systems).
- **Parallel text design**: Prompting LLMs to simulate specific personas to generate AI texts parallel to human-written texts allows comparing detection results under identical topics and personas, reducing confounding variables.

## Limitations & Future Work
- **Corpus bias towards Asian English learners**: ICNALE mainly covers Asia; whether the conclusions generalize to other linguistic backgrounds (e.g., European or African English learners) remains to be validated.
- **Exclusive evaluation on open-source detectors**: Commercial detection services (e.g., Turnitin AI Detection, GPTZero) were not tested, though these tools are more common in actual deployment.
- **Single text genre**: The corpus only includes short essays (two prompts) and does not cover various genres such as long forms, academic papers, or creative writing.
- **Realism of LLM-simulated personas**: Whether prompting LLMs to simulate specific CEFR levels or academic backgrounds genuinely captures the characteristic differences of human writing remains open to question.
- **Directions for improvement**: Mitigation strategies (such as attribute-aware post-processing calibration for detectors) or the development of attribute-aware detection benchmarks could be integrated.

## Related Work & Insights
- **vs. RAID (Dugan et al.)**: RAID provides a large-scale multi-model, multi-domain detection benchmark and unified evaluation framework. Ours reuses its FPR@5% evaluation paradigm but adds the demographic dimension missing from RAID.
- **vs. MAGE (Li et al.)**: MAGE performs systematic in-domain and out-of-domain evaluations. Ours only focuses on out-of-domain but introduces sociolinguistic analysis.
- **vs. HC3**: HC3 compares ChatGPT with human experts but lacks persona annotation, preventing the analysis of author attribute bias.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically introduce sociolinguistic attributes to AI text detection bias analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 detectors $\times$ 12 LLMs $\times$ 4-dimensional attributes, with rigorous statistical analysis (multi-factor control + post-hoc correction).
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed statistical methodology.
- Value: ⭐⭐⭐⭐ Provides important insights for AI detection fairness, though practical applications (how to debias) only suggest directions without proposing solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text](chatgpt_user_ai_text_detection.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICLR 2026\] Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection](../../ICLR2026/aigc_detection/unveiling_perceptual_artifacts_a_fine-grained_benchmark_for_interpretable_ai-gen.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2025\] Cognitive Framework for Detecting AI-Generated Fiction](cognitive_framework_for_detecting_ai-generated_fiction.md)

</div>

<!-- RELATED:END -->
