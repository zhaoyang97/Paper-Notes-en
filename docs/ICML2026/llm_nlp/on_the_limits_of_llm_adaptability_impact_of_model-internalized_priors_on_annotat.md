---
title: >-
  [Paper Note] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation
description: >-
  [ICML 2026][LLM (Other)][Paper Note] Through large-scale experiments on toxicity detection (9 models × 5 datasets), the paper finds that LLM annotation performance is primarily determined by **definition alignment** rather than text memorization; model-internalized priors render the vast majority of zero-shot errors "resilient" to prompt correction—even w
tags:
  - ICML 2026
  - LLM (Other)
date: 2026-05-08
content_hash: 2750cf4c3fb8ecb5
---
# On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2606.00467](https://arxiv.org/abs/2606.00467)  
**Code**: To be confirmed  
**Area**: Social Computing / LLM Reliability  
**Keywords**: LLM Annotation, Model-Internalized Priors, Decision Stickiness, Prompt Steerability, Confidence Calibration

## TL;DR
Through large-scale experiments on toxicity detection (9 models × 5 datasets), the paper finds that LLM annotation performance is primarily determined by **definition alignment** rather than text memorization; model-internalized priors render the vast majority of zero-shot errors "resilient" to prompt correction—even with explicit definitions and examples, **two thirds of errors** remain unfixable (rescue rate of only 34.8%), and confidence cannot be used to detect definition errors.

## Background & Motivation

**Background**: LLMs are widely used for zero-shot annotation and "LLM-as-a-judge" tasks. The traditional assumption is that definitions provided by users through prompts will dominate model behavior, and selecting larger or stronger models will improve annotation quality.

**Limitations of Prior Work**: LLMs are not a blank slate—they develop an implicit understanding of common concepts (e.g., "toxicity") through pre-training, instruction tuning, and RLHF. However, the definition of the same concept varies significantly across application scenarios (social media personal attacks vs. game disruptive behavior vs. speech against protected groups), and the priors internalized by the model may not align with user intent.

**Key Challenge**: The mismatch between user definitions and model-internalized concepts is a deep alignment issue. Existing research provides a detailed understanding of the model's text memorization but knows little about the extent to which the model can override its internalized task understanding through prompts.

**Goal**: Systematically study the interaction of three dimensions—(1) how the model's "familiarity" with data and task definitions affects performance; (2) whether additional prompt information can correct zero-shot errors ("decision stickiness"); (3) the model's susceptibility to misaligned definitions.

**Key Insight**: Rather than assuming user definitions are optimal, the study directly measures the degree of alignment between the model's internal concepts and the definitions; rather than looking only at text overlap, it examines the perspective of conceptual boundaries.

**Core Idea**: Introduce the **Definition-Specific Familiarity (DSF)** metric to quantify the semantic alignment between the model's internal understanding and the dataset definition; reveal that high-confidence errors are nearly impossible to correct through prompts via the "decision stickiness" phenomenon; discover that models confidently apply misaligned definitions through definition replacement experiments.

## Method

### Overall Architecture
This is a diagnostic study: the authors do not propose a new model but decompose "whether LLM annotation follows prompts" into three progressive Research Questions (RQ). RQ1 asks how the model's familiarity with data and task definitions affects annotation quality (proposing the DSF metric and comparing it with traditional text memorization metrics); RQ2 asks how many zero-shot errors can be corrected by additional definitions, examples, or expert prompts (proposing the "rescue rate" to characterize decision stickiness); RQ3 asks how the model reacts when provided with a **wrong** definition (definition misalignment experiments). The entire experimental suite is conducted on toxicity detection using 9 models across 5 datasets.

### Key Designs

**1. Definition-Specific Familiarity (DSF): Measuring the distance between the model's internal concept and the definition**

Traditional judgments of whether a "model understands a task" rely on text memorization metrics (ROUGE-L, BERTScore), but these only measure if the model has seen the original text and cannot explain why certain models are more accurate on certain tasks. DSF takes a different perspective—instead of asking "if the model memorized it," it asks "if the concept understood by the model aligns with the target definition." The approach involves asking the model to explain the target concept in its own words ("How do you understand what makes content toxic?"), then calculating the semantic similarity between this explanation and the full dataset definition using 6 different sentence encoders (MiniLM, MPNet, BGE-large, E5-large, Instructor-large, OpenAI text-embedding-3-small), and averaging them to obtain the "Consensus DSF":

$$\text{DSF} = \frac{1}{6} \sum_{i=1}^6 \text{sim}\big(e_i(\text{model\_explanation}),\, e_i(\text{dataset\_definition})\big)$$

Using 6 encoders to reach a consensus mitigates the bias of any single embedding model. This metric requires no labeled data. In subsequent experiments, DSF proved to be the only familiarity metric that remains positively correlated with accuracy after controlling for dataset difficulty, proving that performance is driven by conceptual alignment rather than text overlap.

**2. Rescue Rate and Decision Stickiness: How many errors prompts can actually fix**

It is often assumed that "using larger models or writing better prompts" can fix annotation errors. The authors use the rescue rate to directly test this assumption. The rescue rate is defined as the probability that an incorrect zero-shot sample is corrected after adding definitions, examples, or expert prompts:

$$\text{Rescue Rate} = P(\text{Correct} \mid \text{Prompted},\ \text{Zero-Shot Wrong})$$

The measurement shows an overall rate of only 34.8%—two-thirds of zero-shot errors cannot be rescued regardless of the prompt; this is "decision stickiness." Critically, it exhibits a U-shape relative to confidence: the rescue probability peaks at 51.8% for medium confidence (0.6–0.7), while for the most confident errors (confidence $>0.9$), the rescue rate plummets to 20.8%. The authors use mixed-effects logistic regression to confirm this trend—for every one standard deviation increase in zero-shot confidence, the odds of being rescued decrease by 16% ($\text{OR}=0.84$). Across 9 models, larger models are not significantly easier to steer, indicating that model-internalized priors set a ceiling for steerability, which may require re-training rather than prompt tuning to breach.

**3. Definition Misalignment and Calibration Failure: Models confidently use incorrect definitions**

The first two points assume the provided definition is correct; this point tests the reverse: what happens if the definition itself is inconsistent with the model's internalized concepts. The authors construct 6 misalignment conditions, ranging from narrow definitions (e.g., "hate speech" requiring targeting protected identities) to broad definitions (e.g., "gaming toxicity" including any disruptive behavior), measuring changes in prediction rates, rescue rates, destruction rates, and bias. The most critical finding is not in accuracy but in confidence—there is **no significant difference** in the model's confidence between aligned and misaligned definitions. This directly violates the calibration assumption: practitioners might want to use confidence as a quality signal to filter suspicious annotations, but since the model remains equally confident when using the wrong definition, confidence thresholds cannot detect definition errors, representing a dangerous blind spot during deployment.

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

| Familiarity Metric | Original Correlation r | Partial Correlation (Control Dataset) |
|--------------------|------------------------|------------------------------------|
| Text Memorization (ROUGE-L) | -0.80 | -0.19 |
| Memorization (BERTScore) | -0.76 | -0.15 |
| **Definition Alignment (DSF)** | +0.74 | **+0.41** (p=0.003) |

DSF is the only metric that remains positively correlated with accuracy after controlling for dataset difficulty.

### Key Findings
- **Decision Stickiness Dominates**: The rescue rate is only 34.8%, and the rescue rate for high-confidence errors is less than 21%.
- **Definition Influence > Model Choice**: Across all models and datasets, definition selection caused a 17% fluctuation in accuracy, while model selection caused only ~5%.
- **Familiarity vs. Memorization**: The positive correlation of DSF (+0.41) confirms that **concept alignment, rather than text overlap, drives performance**.
- **Confidence Calibration Failure**: The model's confidence remains unchanged when applying misaligned definitions.

## Highlights & Insights
- **Innovation of the DSF Metric**: Approaching the problem from "whether the model understands your definition" is profound, shifting from "defense" (checking contamination) to "diagnosis" (measuring alignment).
- **Systematic Characterization of Decision Stickiness**: Beyond identifying the phenomenon, it is quantified through mixed-effects regression (OR = 0.84), proving this is not an issue of single-turn prompting but a systemic constraint of internalized priors.
- **The "High Confidence, High Risk" Paradox of Definition Misalignment**: The model's ability to confidently apply incorrect definitions is its greatest pitfall—offering direct warning value for practical deployment.

## Limitations & Future Work
- Experiments are limited to binary classification of toxicity/hate speech; multi-class classification, span annotation, and open-ended judgment may have different failure modes.
- All models are instruction-tuned; it is impossible to distinguish if the low rescue rate is a model capability limit or an intentional steering restriction designed for safety alignment.
- Findings are correlational rather than causal.
- Future Work: Testing stronger forms of misalignment and multi-turn correction strategies; comparing base models with fine-tuned versions to isolate the contributions of pre-training vs. instruction tuning vs. RLHF.

## Related Work & Insights
- **vs. LLM Steerability Work** (Chang et al. 2026): Studies steerability in generation tasks; this paper supplements the classification task perspective and finds completely different failure modes.
- **vs. Contamination Detection** (Min-K% Prob, BERTScore): Traditional contamination detection assumes memorization is the culprit; this paper proves this is a misdiagnosis through the comparison of DSF and memorization, showing the true driver is conceptual alignment.
- **vs. Model Calibration Research**: The "High Confidence, High Risk" finding in annotation scenarios—where high confidence does not mean the correct definition is applied—is new and has direct safety implications for deployment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The DSF metric and the systematic characterization of the "decision stickiness" phenomenon are original and challenge the "prompt engineering is everything" myth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models × 5 datasets × multiple conditions, supplemented by replication in non-toxic domains to confirm generalizability.
- Writing Quality: ⭐⭐⭐⭐⭐ The progression of the three RQs is clear, chart design is intuitive, and practical implications are explicit.
- Value: ⭐⭐⭐⭐⭐ Provides direct insights for the industrial deployment of LLM annotation, offering a feasible diagnostic method (DSF) and clear warnings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_nlp/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)

</div>

<!-- RELATED:END -->
