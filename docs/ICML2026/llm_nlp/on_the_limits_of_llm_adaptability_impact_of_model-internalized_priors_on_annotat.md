---
title: >-
  [Paper Note] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation
description: >-
  [ICML 2026][LLM (Other)][Paper Note] Through large-scale experiments on toxicity detection (9 models × 5 datasets), this paper finds that LLM annotation performance is primarily determined by **definition alignment** rather than text memorization. Model-internalized priors render the vast majority of zero-shot errors "resilient" to prompt-based correction
tags:
  - ICML 2026
  - LLM (Other)
date: 2026-05-08
content_hash: de6021ef2eb2400e
---
# On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2606.00467](https://arxiv.org/abs/2606.00467)  
**Code**: To be confirmed  
**Area**: Social Computing / LLM Reliability  
**Keywords**: LLM Annotation, Model-Internalized Priors, Decision Stickiness, Prompt Steerability, Confidence Calibration

## TL;DR
Through large-scale experiments on toxicity detection (9 models × 5 datasets), this paper finds that LLM annotation performance is primarily determined by **definition alignment** rather than text memorization. Model-internalized priors render the vast majority of zero-shot errors "resilient" to prompt-based correction—even when explicit definitions and examples are provided, **two-thirds of errors** remain uncorrectable (rescue rate of only 34.8%), and confidence scores cannot be used to detect definition errors.

## Background & Motivation

**Background**: LLMs are widely utilized for zero-shot annotation and "LLM-as-a-judge" tasks. The traditional assumption is that the definitions provided by users via prompts dominate model behavior, and that selecting larger or stronger models will improve annotation quality.

**Limitations of Prior Work**: LLMs are not a blank slate—they develop an implicit understanding of common concepts (e.g., "toxicity") through pre-training, instruction tuning, and RLHF. However, the definition of the same concept can vary greatly across application scenarios (social media personal attacks vs. gaming disruptive behavior vs. speech targeting protected groups), and model-internalized prior concepts may not align with user intent.

**Key Challenge**: The mismatch between user definitions and model-internalized concepts is a deep-seated alignment problem. Existing research provides detailed knowledge of textual memorization in models, but little is known about the extent to which models can override their internalized task understanding through prompts.

**Goal**: Systematically investigate the interaction across three dimensions: (1) how the model's "familiarity" with data and task definitions affects performance; (2) whether additional prompt information can correct zero-shot errors ("decision stickiness"); and (3) model susceptibility to misaligned definitions.

**Key Insight**: Rather than assuming user definitions are optimal, this study directly measures the degree of alignment between the model's internal concepts and definitions. Instead of looking only at textual overlap, it approaches the problem from the perspective of conceptual boundaries.

**Core Idea**: Introduce the **Definition-Specific Familiarity (DSF)** metric to quantify the semantic alignment between the model's internal understanding of the target phenomenon and the dataset definition. Reveal that high-confidence errors are nearly impossible to correct through prompts via the "decision stickiness" phenomenon. Discover through definition substitution experiments that models confidently apply misaligned definitions.

## Method

### Overall Architecture
This is a diagnostic study: the authors do not propose a new model but use three progressive research questions (RQs) to disassemble and measure the question "to what extent do LLMs follow your prompts during annotation." RQ1 asks how model familiarity with data and task definitions affects annotation quality (proposing the DSF metric and comparing it with traditional textual memorization metrics); RQ2 asks how many zero-shot errors can actually be corrected by additional definitions, examples, and expert prompts (proposing "rescue rate" to characterize decision stickiness); RQ3 asks how the model reacts when provided with a **wrong** definition (definition misalignment experiments). The entire set of experiments is conducted on toxicity detection, using 9 models across 5 datasets.

### Key Designs

**1. Definition-Specific Familiarity (DSF): Measuring the distance between the model's concept and your definition**

Traditional judgments of "whether a model understands this task" rely on textual memorization metrics (ROUGE-L, BERTScore), but these only measure whether the model has seen the original text and cannot explain why certain models are more accurate on certain tasks. DSF takes a different perspective—it does not ask "has the model memorized this," but "does the concept the model understands align with your definition." The method first asks the model to explain the target concept in its own words ("How do you understand what makes content toxic?"), then uses 6 different sentence encoders (MiniLM, MPNet, BGE-large, E5-large, Instructor-large, OpenAI text-embedding-3-small) to calculate the semantic similarity between this explanation and the full dataset definition, averaging them to obtain the "Consensus DSF":

$$\text{DSF} = \frac{1}{6} \sum_{i=1}^6 \text{sim}\big(e_i(\text{model\_explanation}),\, e_i(\text{dataset\_definition})\big)$$

Using 6 encoders for consensus dilutes the bias of any single embedding model. This metric requires no labeled data. In subsequent experiments, DSF was indeed the only familiarity metric that remained positively correlated with accuracy after controlling for dataset difficulty, proving that performance is driven by conceptual alignment rather than textual overlap.

**2. Rescue Rate and Decision Stickiness: How much can prompts actually correct**

It is often assumed that "using larger models or writing better prompts" can fix annotation errors. The authors use the rescue rate to directly test this assumption. Rescue rate is defined as the probability that a zero-shot incorrect sample is corrected after adding definitions, examples, or expert prompts:

$$\text{Rescue Rate} = P(\text{Correct} \mid \text{Prompted},\ \text{Zero-Shot Wrong})$$

The overall measured rate was only 34.8%—two-thirds of zero-shot errors could not be rescued regardless of prompting, a phenomenon termed "decision stickiness." Even more critically, it follows a U-shape relative to confidence: the rescue probability peaks at 51.8% for medium confidence (0.6–0.7), while for the most confident errors (confidence $>0.9$), the rescue rate plunges to 20.8%. The authors used mixed-effects logistic regression to confirm this trend—for every standard deviation increase in zero-shot confidence, the odds of being rescued decreased by 16% ($\text{OR}=0.84$). Across 9 models, larger models were not significantly easier to steer, indicating that model-internalized priors set a ceiling for steerability that requires retraining rather than prompt tuning to breach.

**3. Definition Misalignment and Calibration Failure: Models will confidently use incorrect definitions**

The previous points assumed the provided definition was correct; this point tests the inverse: what happens if the definition itself is inconsistent with the model's internalized concepts. The authors constructed 6 misalignment conditions, ranging from narrow definitions (e.g., "hate speech" requiring targeting of protected identities) to broad definitions (e.g., "gaming toxicity" including any disruptive behavior), measuring change rates in prediction, rescue rates, damage rates, and bias. The most critical finding was not in accuracy, but in confidence—there was **no significant difference** in model confidence between aligned definitions and incorrect definitions. This directly violates calibration assumptions: practitioners intend to use confidence as a quality control signal to filter suspicious annotations, but since the model remains equally confident when using the wrong definition, confidence thresholds cannot detect definition errors, representing a dangerous blind spot during deployment.

## Key Experimental Results

### Main Results

| Condition | Zero-Shot | Aligned Def | Few-Shot | FS+Def | DSPy | Misaligned Avg |
|-----------|-----------|-------------|----------|--------|------|----------------|
| Llama-3.1-70B | 79.8 | 82.1 | 81.2 | 82.1 | 79.9 | 78.1 |
| Mistral-Small-24B | 78.0 | 81.0 | 80.8 | 82.3 | 79.3 | 80.7 |
| DeepSeek-V3 | 81.3 | 83.0 | 82.6 | 83.8 | 80.9 | 80.7 |
| GPT-4o-mini | 81.6 | 83.3 | 84.1 | 83.3 | 84.4 | 81.1 |
| Qwen-2.5-72B | 83.3 | 82.2 | 83.8 | 83.2 | 82.5 | 81.2 |
| **Condition Avg** | 80.3 | 82.0 | 81.5 | 81.6 | 80.2 | 80.3 |

Aligned definitions only improved performance by +1.7%—prompt improvement space is limited.

### Ablation Study

| Familiarity Metric | Original Correlation r | Partial Correlation (Dataset Controlled) |
|--------------------|------------------------|------------------------------------------|
| Text Memory (ROUGE-L) | -0.80 | -0.19 |
| Memory (BERTScore) | -0.76 | -0.15 |
| **Definition Alignment (DSF)** | +0.74 | **+0.41** (p=0.003) |

DSF is the only metric that remains positively correlated with accuracy after controlling for dataset difficulty.

### Key Findings
- **Decision Stickiness Dominates**: The rescue rate is only 34.8%, and the rescue rate for high-confidence errors is below 21%.
- **Definition Impact > Model Selection**: Across all models and datasets, definition choice caused a 17% fluctuation in accuracy, while model choice caused only ~5%.
- **Familiarity vs. Memory**: The positive correlation of DSF (+0.41) confirms that **conceptual alignment, not textual alignment, drives performance**.
- **Confidence Calibration Failure**: Model confidence remains unchanged when applying misaligned definitions.

## Highlights & Insights
- **Innovation of the DSF Metric**: Approaching from "whether the model understands your definition" is profound; shifting from "defense" (checking for contamination) to "diagnosis" (measuring alignment).
- **Systematic Characterization of Decision Stickiness**: Not only identifying the phenomenon but also quantifying it through mixed-effects regression (OR = 0.84) and proving it is a systematic constraint of internalized priors rather than a single-turn prompting issue.
- **The "High-Confidence, High-Risk" Paradox of Definition Misalignment**: The ability of models to confidently apply incorrect definitions is a major pitfall—offering direct warning value for practical deployment.

## Limitations & Future Work
- Experiments are limited to binary classification of toxicity/hate speech; multi-class, span annotation, or open-ended judgment may have different failure modes.
- All models are instruction-tuned, making it impossible to distinguish whether low rescue rates are due to model capability limitations or intentional steerability constraints in safety alignment design.
- The findings are correlational rather than causal.
- Future Work: Test stronger forms of misalignment, multi-turn correction strategies; compare base models with fine-tuned versions to isolate the contributions of pre-training vs. instruction tuning vs. RLHF.

## Related Work & Insights
- **vs. LLM Steerability Work** (Chang et al. 2026): Researches steerability in generative tasks; this paper complements the classification task perspective and finds completely different failure modes.
- **vs. Contamination Detection Work** (Min-K% Prob, BERTScore): Traditional contamination detection assumes memorization is the culprit; this paper proves this is a misdiagnosis through the comparison of DSF and memorization, showing the true driver is conceptual alignment.
- **vs. Model Calibration Research**: The "high-confidence, high-risk" finding in the annotation context—where high confidence does not mean the correct definition is being applied—is new and has direct safety implications for deployment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The DSF metric and the systematic characterization of the "decision stickiness" phenomenon are original and challenge the "prompt engineering is a panacea" myth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models × 5 datasets × multiple conditions, plus replication in non-toxicity domains to confirm generalizability.
- Writing Quality: ⭐⭐⭐⭐⭐ The three RQs progress clearly, charts are intuitively designed, and practical implications are clear.
- Value: ⭐⭐⭐⭐⭐ Provides direct insights for the industrial deployment of LLM annotation, offering a feasible diagnostic method (DSF) and clear warnings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_nlp/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)

</div>

<!-- RELATED:END -->
