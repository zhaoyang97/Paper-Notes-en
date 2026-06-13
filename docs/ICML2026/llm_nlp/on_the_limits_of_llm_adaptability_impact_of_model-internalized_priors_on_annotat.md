---
title: >-
  [Paper Note] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation
description: >-
  [ICML 2026][LLM/NLP][LLM Annotation] Through large-scale experiments on toxicity detection (9 models × 5 datasets), the paper finds that LLM annotation performance is primarily determined by **definition alignment** rath…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "LLM Annotation"
  - "Model-Internalized Priors"
  - "Decision Stickiness"
  - "Prompt Steerability"
  - "Confidence Calibration"
date: 2026-05-08
content_hash: 5763dc06bf599ca4
---

# On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2606.00467](https://arxiv.org/abs/2606.00467)  
**Code**: TBD  
**Area**: Social Computing / LLM Reliability  
**Keywords**: LLM Annotation, Model-Internalized Priors, Decision Stickiness, Prompt Steerability, Confidence Calibration

## TL;DR
Through large-scale experiments on toxicity detection (9 models × 5 datasets), the paper finds that LLM annotation performance is primarily determined by **definition alignment** rather than text memorization; model-internalized priors render the vast majority of zero-shot errors "resilient" to prompt-based correction—even when provided with explicit definitions and examples, **two-thirds of errors** remain unfixable (rescue rate of only 34.8%), and confidence scores cannot be used to detect definition-related errors.

## Background & Motivation

**Background**: LLMs are widely used for zero-shot annotation and "LLM-as-a-judge" tasks. The traditional assumption is that the definitions provided by users through prompts will dominate model behavior, and that selecting larger or stronger models will improve annotation quality.

**Limitations of Prior Work**: LLMs are not a blank slate—they develop an implicit understanding of common concepts (e.g., "toxicity") through pre-training, instruction tuning, and RLHF. However, the definition of the same concept can vary significantly across application scenarios (social media personal attacks vs. game-disruptive behavior vs. speech against protected groups), and the model's internalized prior concepts may not align with user intent.

**Key Challenge**: The mismatch between user definitions and model-internalized concepts is a deep-seated alignment problem. Existing research provides extensive knowledge of models' text memorization, but little is known about the extent to which models can override their internalized task understanding through prompts.

**Goal**: Systematically investigate the interaction of three dimensions: (1) how model "familiarity" with data and task definitions affects performance; (2) whether additional prompt information can correct zero-shot errors ("decision stickiness"); and (3) the susceptibility of models to misaligned definitions.

**Key Insight**: Instead of assuming user definitions are optimal, the study directly measures the degree of alignment between the model's internal concepts and the provided definitions; rather than focusing solely on text overlap, it examines conceptual boundaries.

## Method

### Overall Architecture
Three research questions (RQs) are addressed progressively: RQ1 examines how familiarity with data and task definitions impacts annotation performance (DSF vs. text familiarity comparison); RQ2 investigates the extent to which external knowledge and advanced prompting can correct initial zero-shot errors (decision stickiness); RQ3 explores how LLMs behave when provided with misaligned or incorrect task definitions.

### Key Designs

1. **Definition-Specific Familiarity (DSF) Metric**:

    - **Function**: Quantifies the degree of alignment between the model's internal understanding of a target concept and the dataset's definition.
    - **Mechanism**: The model is prompted to explain the target concept in its own words ("How do you understand what makes content toxic?"), then 6 different sentence encoders (MiniLM, MPNet, BGE-large, E5-large, Instructor-large, OpenAI text-embedding-3-small) are used to calculate the semantic similarity between the model's explanation and the dataset's full definition, with the average taken as the "consensus DSF". $$\text{DSF} = \frac{1}{6} \sum_{i=1}^6 \text{sim}(e_i(\text{model\_explanation}), e_i(\text{dataset\_definition}))$$.
    - **Design Motivation**: Existing text memorization metrics (ROUGE-L, BERTScore) cannot explain why certain models perform better on specific tasks; DSF approaches the problem from the perspective of "concept alignment". Using a consensus of 6 encoders reduces the bias of any single embedding model, and the metric requires no labeled data.

2. **Rescue Rate + Decision Stickiness**:

    - **Function**: Measures the ability of LLMs to correct their own errors via prompts, revealing the fundamental limitations of high-confidence errors.
    - **Mechanism**: $\text{Rescue Rate} = P(\text{Correct} \mid \text{Prompted, Zero-Shot Wrong})$. A U-shaped curve was observed—rescue probability peaks at 51.8% at medium confidence (0.6-0.7) and drops sharply to 20.8% for high-confidence errors (> 0.9). This is quantified via mixed-effects logistic regression: for every one standard deviation increase in zero-shot confidence, the odds of rescue decrease by 16% (OR = 0.84).
    - **Design Motivation**: While traditional views suggest that larger models or better prompt design can solve annotation issues, comparing 9 models reveals that larger models are not necessarily easier to correct—internalized priors set a fundamental ceiling on steerability, and exceeding this ceiling requires retraining rather than just prompt tuning.

3. **Definition Misalignment and Calibration Failure**:

    - **Function**: Tests model behavior when receiving misaligned definitions, revealing the critical weakness of "high-confidence application of incorrect definitions."
    - **Mechanism**: Six definition misalignment conditions were designed (e.g., narrow definitions like "hate speech" requiring identity verification vs. broad definitions like "gaming toxicity" including any disruptive behavior). The study measures change rates, rescue rates, sabotage rates, and prediction bias. A key finding is that model confidence shows **no significant difference** between aligned and misaligned conditions—violating the basic calibration assumption. **Practitioners cannot use confidence thresholds to detect definition errors.**
    - **Design Motivation**: Practitioners deploying LLM annotation often assume confidence can serve as a quality control signal, but this study demonstrates that such an assumption is flawed.

## Key Experimental Results

### Main Results

| Condition | Zero-Shot | Aligned Def | Few-Shot | FS+Def | DSPy | Misaligned Avg |
|------|--------|---------|--------|--------|------|----------|
| Llama-3.1-70B | 79.8 | 82.1 | 81.2 | 82.1 | 79.9 | 78.1 |
| Mistral-Small-24B | 78.0 | 81.0 | 80.8 | 82.3 | 79.3 | 80.7 |
| DeepSeek-V3 | 81.3 | 83.0 | 82.6 | 83.8 | 80.9 | 80.7 |
| GPT-4o-mini | 81.6 | 83.3 | 84.1 | 83.3 | 84.4 | 81.1 |
| Qwen-2.5-72B | 83.3 | 82.2 | 83.8 | 83.2 | 82.5 | 81.2 |
| **Condition Mean** | 80.3 | 82.0 | 81.5 | 81.6 | 80.2 | 80.3 |

Aligned definitions only provided a +1.7% improvement—indicating limited room for improvement through prompting.

### Ablation Study

| Familiarity Metric | Original Correlation r | Partial Correlation (Dataset Controlled) |
|-----------|-----------|-------------------|
| Text Memorization (ROUGE-L) | -0.80 | -0.19 |
| Memorization (BERTScore) | -0.76 | -0.15 |
| **Definition Alignment (DSF)** | +0.74 | **+0.41** (p=0.003) |

DSF is the only metric that remains positively correlated with accuracy after controlling for dataset difficulty.

### Key Findings
- **Decision Stickiness Dominant**: The rescue rate is only 34.8%, and for high-confidence errors, it is below 21%.
- **Definition Impact > Model Selection**: Across all models and datasets, definition choice caused a 17% variation in accuracy, whereas model choice accounted for only ~5%.
- **Familiarity vs. Memorization**: The positive correlation of DSF (+0.41) confirms that **concept alignment, rather than text alignment, drives performance**.
- **Calibration Failure**: Model confidence remains unchanged even when the model applies an incorrect definition.

## Highlights & Insights
- **Innovation of the DSF Metric**: Approaching the problem via "whether the model understands your definition" is profound; it shifts focus from "defense" (detecting contamination) to "diagnosis" (measuring alignment).
- **Systematic Characterization of Decision Stickiness**: The study not only identifies the phenomenon but also quantifies it via mixed-effects regression (OR = 0.84), proving it is a systematic internalized prior constraint rather than just a single-turn prompting issue.
- **The "High-Confidence, High-Risk" Paradox of Misalignment**: The ability of models to confidently apply incorrect definitions is a major pitfall, providing direct warning value for practical deployments.

## Limitations & Future Work
- Experiments were limited to binary toxicity/hate speech classification; multi-class, span annotation, or open-ended judgment may exhibit different failure modes.
- Since all models were instruction-tuned, it is difficult to determine whether the low rescue rate is a model capability limit or an intentional safety alignment design to limit steerability.
- The findings demonstrate correlation rather than causation.
- Future work: Test stronger forms of misalignment and multi-turn correction strategies; compare base models and fine-tuned versions to isolate the contributions of pre-training vs. instruction tuning vs. RLHF.

## Related Work & Insights
- **vs. LLM Steerability Work** (Chang et al. 2026): While prior work studied steerability in generative tasks, this paper adds a classification perspective and identifies entirely different failure modes.
- **vs. Contamination Detection Work** (Min-K% Prob, BERTScore): Traditional contamination detection assumes memorization is the primary cause of performance variance; this paper uses DSF to show this is a misdiagnosis, proving concept alignment is the true driver.
- **vs. Model Calibration Research**: The discovery of "high-confidence, high-risk" in annotation scenarios—where high confidence does not guarantee the correct definition is being applied—is novel and has direct safety implications for deployment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The DSF metric and the systematic characterization of "decision stickiness" are original, challenging the myth that prompt engineering is a universal solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprising 9 models × 5 datasets × multiple conditions, supplemented by replication in non-toxicity domains to confirm generalizability.
- Writing Quality: ⭐⭐⭐⭐⭐ The three RQs progress clearly, visualizations are intuitive, and practical implications are well-defined.
- Value: ⭐⭐⭐⭐⭐ Provides direct insights for industrial deployment of LLM annotation, offering both a feasible diagnostic method (DSF) and clear warnings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_nlp/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)

</div>

<!-- RELATED:END -->
