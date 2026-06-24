---
title: >-
  [Paper Note] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions
description: >-
  [ACL 2025][LLM (Other)][sociodemographic prompting] This paper systematically investigates whether LLMs can predict individual annotators' subjective text perceptions using sociodemographic attributes (age, gender, education, race/ethnicity). The authors find that performance improvements after fine-tuning primarily result from learning individual annotator behaviors rather than sociodemographic patterns, raising doubts about the feasibility of using LLMs to simulate sociodem…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "sociodemographic prompting"
  - "annotator modeling"
  - "subjective text perception"
  - "individual differences"
  - "LLM fine-tuning"
date: 2026-05-08
content_hash: c8eba97797e9df1c
---

# Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions

**Conference**: ACL 2025  
**arXiv**: [2502.20897](https://arxiv.org/abs/2502.20897)  
**Code**: [https://github.com/morlikowski/beyond-demographics](https://github.com/morlikowski/beyond-demographics)  
**Area**: LLM/NLP  
**Keywords**: sociodemographic prompting, annotator modeling, subjective text perception, individual differences, LLM fine-tuning

## TL;DR
This paper systematically investigates whether LLMs can predict individual annotators' subjective text perceptions using sociodemographic attributes (age, gender, education, race/ethnicity). The authors find that performance improvements after fine-tuning primarily result from learning individual annotator behaviors rather than sociodemographic patterns, raising doubts about the feasibility of using LLMs to simulate sociodemographic variations.

## Background & Motivation

**Background**: Natural variations among annotators in subjective NLP tasks (e.g., sentiment, offensiveness, intimacy judgments) correlate with their sociodemographic traits. Recent studies have attempted to simulate the annotation behavior of different groups by prompting LLMs with sociodemographic attributes.

**Limitations of Prior Work**: Several studies show that LLMs perform poorly under zero-shot sociodemographic prompting; using prompts like "You are a 30-year-old white woman" does not align model predictions with the annotation preferences of that group. However, no prior work has systematically investigated whether fine-tuning can improve this.

**Key Challenge**: While sociodemographic attributes do influence annotation behavior, LLMs seem unable to leverage this information. Is the root cause a lack of sociodemographic knowledge in LLMs, or is such demographic knowledge inherently insufficient for predicting individual behavior?

**Goal**: To answer four research questions: (RQ1) Can LLMs better model annotators through sociodemographic attributes or individual IDs? (RQ2) Can they generalize to unseen annotators? (RQ3) What representations are learned from sociodemographics? (RQ4) Does individual-level modeling improve disagreement prediction?

**Key Insight**: Building DeMo, a unified multi-task and multi-dataset evaluation framework, to systematically compare four input formats: content-only, +attributes, +ID, and +ID+attributes.

**Core Idea**: By contrasting the effects of fine-tuning LLMs with "sociodemographic attributes" versus "annotator IDs," this work reveals that LLMs primarily learn to treat attribute combinations as proxies for individual identities, rather than learning true sociodemographic-annotation correlations.

## Method

### Overall Architecture
The input consists of text (tweets, Reddit comments, emails, or dialogues) coupled with optional annotator information (sociodemographic attributes and/or a unique ID), and the output is a predicted rating of the text by the annotator (3- or 5-way classification). Llama 3 8B is used as the base model and fine-tuned with an added prediction head.

### Key Designs

1. **DeMo Unified Evaluation Dataset**:

    - **Function**: Provides a standardized evaluation framework across five subjective tasks.
    - **Mechanism**: Integrates five existing datasets (Intimacy MinT, Offensiveness Popquorn, Politeness Popquorn, Safety DICES-350, Sentiment Díaz et al.), unifying sociodemographic attributes into four dimensions (age, gender, race/ethnicity, education), totalizing 21,632 texts, 2,614 annotators, and 147,297 annotations.
    - **Design Motivation**: Prior works rely on disparate datasets and attribute definitions, leading to a lack of comparability.

2. **Comparative Experimental Design of Four Input Formats**:

    - **Function**: Disentangles the contributions of sociodemographic information versus individual identity.
    - **Mechanism**: Designs four input formats: Content-Only (text-only baseline), +Attributes (with sociodemographic attributes), +ID (with unique annotator ID), and +ID+Attributes (both). Minimalist templates are used, e.g., "Annotator: hispanic/latino, 40 to 44 years old, a woman, a college degree\n Text: ...".
    - **Design Motivation**: To precisely measure the contribution of each information source through controlled variables, specifically distinguishing between group-level information from attributes and individual-level information from IDs.

3. **Dual Evaluation Strategy: Instance Split vs. Annotator Split**:

    - **Function**: Separately evaluates "modeling of known annotators" and "generalization to unseen annotators."
    - **Mechanism**: In the Instance Split, annotators appear in both the training and test sets (on different texts), testing whether the model can learn individual preferences. In the Annotator Split, test set annotators never appear in training, verifying if the model can generalize from attributes to unseen individuals.
    - **Design Motivation**: The two scenarios correspond to different application needs: personalization for known users vs. group profiling for unfamiliar users.

### Loss & Training
Uses a prediction head (resembling reward model architectures) for classification with cross-entropy loss. Fine-tuning is conducted via LoRA. The learning rate is selected via grid search over 10 runs, with each configuration in the main experiments run across 30 different random seeds. Macro-average F1 is used as the evaluation metric.

## Key Experimental Results

### Main Results (Instance Split — RQ1)

| Input Format | Intimacy F1 | Offensiveness F1 | Politeness F1 | Safety F1 | Sentiment F1 |
|---------|-------------|-----------------|---------------|-----------|-------------|
| Zero-shot | ~0.22 | ~0.28 | ~0.24 | ~0.25 | ~0.26 |
| Content-Only | ~0.30 | ~0.25 | ~0.32 | ~0.38 | ~0.35 |
| +Attributes | ~0.33 | ~0.28 | ~0.35 | ~0.42 | ~0.38 |
| +ID | **~0.42** | **~0.35** | **~0.44** | **~0.48** | **~0.45** |
| +ID+Attr | ~0.42 | ~0.35 | ~0.44 | ~0.48 | ~0.45 |

### Annotator Split Results (RQ2)

| Input Format | Average Performance | Description |
|---------|---------|------|
| Content-Only | Baseline | Text-only |
| +Attributes | ≈Baseline | Attributes provide no help for unseen annotators |
| +ID | ≈Baseline | Unseen IDs are uninformative, as expected |
| +ID+Attributes | ≈Baseline | Shows no generalization capabilities |

### Key Findings
- **RQ1**: Fine-tuning with attributes indeed yields consistent improvements, but the improvement from ID is much greater. +ID+Attributes does not outperform +ID, showing that the information provided by attributes is entirely subsumed by ID.
- **RQ2**: For unseen annotators, no additional profiling information outperforms the text-only baseline, demonstrating that the model does not learn generalizable sociodemographic-annotation rules.
- **RQ3**: By analyzing variations between annotators with "unique attribute combinations" vs. those with "frequent attribute combinations," it is shown that improvements from attributes mostly occur for unique combinations (where attributes serve as an equivalent to ID), while frequent combinations show almost no gains. This confirms that the model treats attributes as proxies for IDs.
- **RQ4**: Models using ID show a significant improvement in Wasserstein distance on high-disagreement (high label entropy) samples, indicating that individual-level modeling helps capture annotation disagreements.

## Highlights & Insights
- **Ingenious Experimental Design**: The contrastive analysis of unique vs. frequent attribute combinations (RQ3) carefully reveals that what appears to be "sociodemographic modeling" is actually individual identity recognition. This analysis method can be transferred to other personalized modeling research to help distinguish between group-level and individual-level effects.
- **Warning for LLM-based Simulation of Human Behavior**: The findings raise a critical warning for social science simulations using LLMs (e.g., substituting survey respondents with LLM agents), indicating that LLMs do not genuinely understand how sociodemographics influence subjective judgments.

## Limitations & Future Work
- The dataset only contains US-based annotators; cross-cultural generalization remains unverified.
- Experiments primarily focus on the Llama 3 model family (though supplementary experiments in the appendix with Mistral 7B show similar trends).
- The four-dimensional sociodemographic attributes might not be exhaustive, but they represent the largest intersection across the five datasets.
- Future work can explore more granular individual information (e.g., historical annotation patterns) to replace simple ID embeddings.

## Related Work & Insights
- **vs Orlikowski et al. (2023)**: Prior work showed that sociodemographics do not outperform IDs (the ecological fallacy). This paper confirms this finding at a much larger scale.
- **vs Fleisig et al. (2023)**: This work found attributes to outperform IDs, which contradicts the findings here. This discrepancy may stem from differences in datasets and architectures.
- **vs Beck et al. (2024)**: While the fine-tuning results in this work vastly outperform zero-shot sociodemographic prompting, the gains primarily come from individual-level memorization.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study on the sociodemographic modeling capabilities of fine-tuned LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 seeds, 5 tasks, 4 formats, 2 split types, and exhaustive analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions unfold progressively, backed by deep analysis.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for LLM-based social simulation and annotator modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
