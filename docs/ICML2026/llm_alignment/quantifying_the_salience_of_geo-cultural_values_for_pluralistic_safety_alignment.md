---
title: >-
  [Paper Note] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment
description: >-
  [ICML 2026][LLM Alignment][Pluralistic Alignment] The authors re-stratify annotators into "cultural zones/quadrants" using the Inglehart-Welzel cultural map. Using multilevel modeling across 8 safety datasets…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Pluralistic Alignment"
  - "Cultural Values"
  - "Safety Annotation"
  - "Annotator Disagreement"
  - "Multilevel Modeling"
date: 2026-05-08
content_hash: 3521045efbd21f39
---

# Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment

**Conference**: ICML 2026  
**arXiv**: [2606.00369](https://arxiv.org/abs/2606.00369)  
**Code**: https://github.com/asaakyan/culture-safety  
**Area**: Alignment / AI Safety / Pluralistic Alignment  
**Keywords**: Pluralistic Alignment, Cultural Values, Safety Annotation, Annotator Disagreement, Multilevel Modeling

## TL;DR
The authors re-stratify annotators into "cultural zones/quadrants" using the Inglehart-Welzel cultural map. Using multilevel modeling across 8 safety datasets, they demonstrate that cultural zones significant explain safety score variance even after controlling for demographics (age/gender/ethnicity) ($p<0.05$ in 6/8 datasets). They propose a Bayesian "cultural sensitivity score" quantifying that approximately 10% of samples would be mislabeled as safe if a cultural quadrant were ignored; further experiments indicate that while LLMs are unreliable as rater surrogates, they are feasible as triage tools for "culturally sensitive samples."

## Background & Motivation

**Background**: Current AI safety alignment (RLHF, Constitutional AI, LLM-as-a-judge) relies heavily on human safety ratings. However, major datasets (Anthropic HH, Bai et al. 2022, Glaese et al. 2022, etc.) primarily utilize rater pools with highly concentrated geographic origins (mostly US/UK crowdsourced workers). While a few geo-diverse datasets (PRISM, DICES, CREHate) have begun incorporating "country" as an attribute, most analyses remain limited to descriptive visualizations.

**Limitations of Prior Work**: (i) Most safety datasets do not report geographic or cultural variables, making retrospective analysis impossible. (ii) Those that do report such data often analyze demographic variables (age/gender/ethnicity) and "country" separately without joint control, potentially misattributing cultural effects to demographic differences or vice versa. (iii) The industry is shifting toward using LLMs-as-a-Judge to replace human annotation at scale, yet the ability of LLMs to simulate the judgments of diverse cultural groups has not been validated.

**Key Challenge**: Pluralistic alignment requires covering cross-cultural differences, but it remains unclear whether existing "demographically stratified" annotation protocols are sufficient, or if culture must be stratified as a factor independent of demographics. No methodological framework has previously existed to rigorously address this question.

**Goal**: To quantitatively answer three sub-questions: (1) Why: Do geo-cultural factors remain significant after controlling for demographics? (2) Where: Which specific data samples are mislabeled due to a lack of cultural coverage? (3) How: Can LLMs mitigate the high costs of global annotation?

**Key Insight**: The authors leverage two primary axes proposed by Inglehart and Welzel based on the World Values Survey (WVS)—Traditional vs. Secular and Survival vs. Self-Expression. These axes explain over 70% of cross-national variance in the WVS and categorize countries into "cultural zones." This provides a theory-driven, low-dimensional cultural proxy that avoids the overfitting and sparsity issues associated with one-hot encoding over 200 countries.

**Core Idea**: "Cultural zones/quadrants" are treated as fixed effects in multilevel models, jointly fitted with demographic fixed effects and rater/item random effects. A likelihood ratio test (LRT) is used to strictly compare models with and without cultural factors. Furthermore, a Bayesian posterior is constructed at the quadrant level to quantify the proportion of unsafe samples missed when a specific quadrant is ignored.

## Method

### Overall Architecture
The methodology proceeds in three stages:

1.  **Meta-analysis (Sec. 3)**: Using snowballing and keyword searches (10+ top conferences, 1062 candidates), 8 safety datasets that report both demographic and geographic attributes were selected (DIVE, CulturalFrames, PRISM, DICES-990, NLPositionality, D3, CREHate, Severity). Their modalities, annotation tasks, attribute coverage, and existing analysis methods were cataloged.
2.  **Significance Testing (Sec. 4)**: Each rater is mapped to a cultural zone/quadrant based on the Inglehart-Welzel map (priority: CoLR/CoB > CoR > CoN). A series of multilevel regression models are fitted. Significance is tested using three metrics: LRT, $\Delta$AIC, and the rater variance reduction ratio, to sequentially examine "Demographics vs. Base," "Culture vs. Base," "Culture + Demographics vs. Demographics Only," and "Culture $\times$ Demographics Interaction."
3.  **Blind Spot Quantification and LLM Automation (Sec. 5 & 6)**: A Bayesian posterior estimates the unsafe probability for each item per cultural quadrant to identify "culturally sensitive items." Subsequently, fine-tuned models (DeBERTa-Large, Gemma-3-4B) and prompted reasoning models (Gemini-3 Flash, GPT-5 Nano) are used to verify if LLMs can (a) simulate ratings for a specific cultural quadrant or (b) at least identify cultural sensitivity.

### Key Designs

1.  **Multilevel Regression + LRT to Quantify Independent Contributions**:
    *   **Function**: Quantitatively determines if cultural zones possess explanatory power while strictly controlling for rater/item random effects and demographic fixed effects.
    *   **Mechanism**: Taking Likert scores $H_{ij}$ as an example, the base model is $H_{ij}=\beta_0+u_i+v_j+\epsilon_{ij}$, where $u_i\sim\mathcal{N}(0,\sigma_{\text{rater}}^2)$ and $v_j\sim\mathcal{N}(0,\sigma_{\text{item}}^2)$. Demographic vectors $\mathbf{E}_i, \mathbf{A}_i, \mathbf{G}_i$ (ethnicity/age/gender) and cultural zone vectors $\mathbf{C}_i$ are added to create nested models (D, CZ, D+CZ, D$\times$CZ). Models are compared via LRT with Benjamini-Hochberg correction, $\Delta\text{AIC}$, and rater variance reduction $\%\Delta\sigma_{\text{rater}}^2$.
    *   **Design Motivation**: Conventional IRR methods relying on bootstrapping and stratified sub-samples are infeasible for sparse "Culture $\times$ Demographic" grids. Multilevel models handle unbalanced data by modeling variation at item, rater, and group levels, cleanly decoupling cultural and demographic fixed effects.

2.  **Cultural Sensitivity Score: Bayesian Quantification of Ignoring a Quadrant**:
    *   **Function**: For each dataset sample, it outputs a score $S_{iq} \in [0, 1]$ representing the probability that only quadrant $q$ considers it unsafe while others consider it safe. Items with $S_{iq}>0.5$ are labeled culturally sensitive.
    *   **Mechanism**: For sample $i$ and quadrant $q$, with total votes $n_{iq}$ and unsafe votes $k_{iq}$, a posterior $\text{Beta}(1+k_{iq},1+n_{iq}-k_{iq})$ is derived using a Beta(1,1) prior. The probability that the majority in that quadrant deems it unsafe is $H_{iq}=P(\theta_{iq}>0.5)$. Defining $S_{iq}=H_{iq}\cdot\prod_{q'\neq q,\,q'\text{ valid}}(1-H_{iq'})$, assuming quadrant independence. A validity filter requires at least 3 votes per quadrant, not all from a single demographic group.
    *   **Design Motivation**: Point estimates from raw voting ratios are dominated by noise in small samples ($n_{iq}=3$). The Bayesian posterior provides natural smoothing and uncertainty quantification. The product form calculates the joint probability of "minority unsafe + majority safe," providing an interpretable metric for missed labels.

3.  **LLM Dual-Role Evaluation: Rater Surrogate vs. Sensitivity Triage**:
    *   **Function**: Tests whether LLMs can replace multi-cultural human annotation (predicting $H_{iq}>0.5$) and whether they can assist in triaging sensitive samples to save costs.
    *   **Mechanism**: (a) The surrogate task is modeled as 4-label multi-label classification using masked binary cross-entropy. (b) The triage task is binary: classifying "safe vs. sensitive" versus "safe vs. unsafe" on the D3 dataset.
    *   **Design Motivation**: While the industry assumes LLMs-as-a-Judge can simulate diverse human perspectives, this has not been validated culturally. The separate evaluation of "replacement" versus "assistance" allows for both negative (cannot replace) and constructive (can triage) conclusions.

### Loss & Training
- Multilevel regressions are fitted via maximum likelihood estimation using lme4/lmer. Cumulative link mixed models were used for validation on Likert scores.
- LLM Surrogate experiments: 10 random seeds × 5 epochs; checkpoints selected by average validation F1. Masked binary cross-entropy applied only to quadrants with existing annotations.
- Triage experiments: 970 samples split 65/15/20 with no item overlap.

## Key Experimental Results

### Main Results: Significance of Cultural Zones (Summary of Table 2 & 3)

| Dataset | D vs Base $p$ | CZ vs Base $p$ | D+CZ vs D $p$ | $\Delta$AIC (D+CZ vs D) |
| :--- | :--- | :--- | :--- | :--- |
| DIVE | $<0.001$ | $0.003$ | $0.581$ | $+8.35$ (No gain) |
| CulturalFrames | $0.134$ | $<0.001$ | $<0.001$ | $-41.19$ |
| PRISM | $<0.001$ | $0.003$ | $0.012$ | $-3.97$ |
| DICES-990 | $<0.001$ | $0.004$ | $0.005$ | $-5.86$ |
| D3 | $<0.001$ | $<0.001$ | $<0.001$ | $-179.88$ |
| CREHate | $<0.001$ | $0.203$ | $0.008$ | $-5.12$ |
| Severity | $0.001$ | $<0.001$ | $<0.001$ | $-40.95$ |
| NLPositionality | $0.069$ | $0.344$ | $0.271$ | $+3.62$ |

Ours: In 6/8 datasets, adding cultural zones significantly improved the fit even after controlling for demographics (average $\Delta$AIC of $-46$, additional 4.64% rater variance explained). Insignificance in the remaining two was due to cultural imbalance or extremely small sample sizes.

### Proportion of Culturally Sensitive Samples (Summary of Table 5)

| Dataset | Sensitive Samples | Ratio ($S_{iq}>0.5$) | Valid Samples |
| :--- | :--- | :--- | :--- |
| DIVE | 123 | 13.9% | 887 |
| DICES-990 | 130 | 13.1% | 990 |
| D3 | 485 | 10.9% | 4453 |
| CREHate | 174 | 11.1% | 1562 |
| Severity | 2 | 3.0% | 66 |

Applying a stricter threshold of $S_{iq}>0.7$ reduced the overall ratio to approximately 3%, demonstrating reasonable robustness to threshold selection.

### Ablation Study / LLM Automation

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Always-Unsafe baseline | F1 $\approx 2p/(p+1)$ | Lower bound for surrogate task |
| DeBERTa-Large FT (Surrogate) | $\le$ baseline on D3 Q II/Q IV | Unreliable cultural simulation |
| Gemma-3-4B FT (Surrogate) | Similar to DeBERTa | Scaling didn't yield stable gains |
| Reasoning models (Prompted) | Below baseline | Reasoning LMs did not resolve the issue |
| Gemma-3-4B (Triage, S vs sensitive) | F1 = 0.72, $p=0.044$ | Significantly better than baseline |
| DeBERTa-Large (Triage) | F1 = 0.71, $p=0.071$ | Consistent trend |

### Key Findings
- Cultural zones contribute significant predictive power in 6/8 datasets even after controlling for age/gender/ethnicity, invalidating the assumption that demographic stratification alone is sufficient. The largest effect in D3 is attributed to its unique demographic balancing within each cultural zone.
- D3 is the only dataset showing significant "Cultural Zone $\times$ Demographic" interactions, suggesting that detecting fine-grained differences (e.g., Elder+LatAm vs. Elder+Confucian) requires demographic balancing within zones.
- Approximately 10% of safety dataset samples are culturally sensitive, a stable conclusion across tasks. This implies current RLHF data might systematically label 1/10th of content dangerous to specific cultures as "safe."
- LLMs cannot replace culturally diverse annotators; however, fine-tuned LLMs can serve as triage tools for culturally sensitive items (0.72 F1). The ability for "safe-sensitive" training to transfer to "safe-unsafe" tasks (but not vice versa) suggests "sensitive samples" contain richer safety representations.

## Highlights & Insights
- Integrating the Inglehart-Welzel cultural map provides a theory-driven, cross-dataset proxy for "culture," avoiding the pitfalls of sparse one-hot encodings.
- The Bayesian cultural sensitivity score quantifies the "cost of ignoring a cultural group" as a computable probability, which is more robust than traditional IRR for small-sample cells.
- The conclusion that "LLMs cannot be surrogates but can be triage tools" provides a constructive path forward, cautioning against LLM-as-a-Judge while offering a method for prioritization.
- The recommended rater proxy hierarchy (CoLR/CoB > CoR > CoN) offers direct engineering value for global rater recruitment, identifying which variables are most representative of true cultural values.

## Limitations & Future Work
- Cultural zones are coarse simplifications; the IW map cannot capture nuanced regional differences within a quadrant (e.g., Confucian vs. Eastern Orthodox).
- The meta-analysis was limited to 8 predominantly English datasets, and WVS data coverage is not universal.
- LLM experiments focused on text; conclusions for multimodal (image-text) safety require validation.
- Future work could collect datasets fully balanced across "Cultural Quadrant $\times$ Demographics" and integrate cultural sensitivity scores into active learning pipelines to optimize annotation budgets.

## Related Work & Insights
- **vs DICES (Aroyo et al., 2023), PRISM (Kirk et al., 2024)**: While these introduced cross-national rater pools, their analyses were descriptive; this paper provides rigorous statistical evidence using multilevel modeling to decouple cultural and demographic contributions.
- **vs D3 (Davani et al., 2024)**: This paper extends the multilevel approach by jointly controlling for demographic factors and adding cultural zones and interactions.
- **vs LLM-as-a-Judge Paradigms (Bai 2022b, Yuan 2025)**: This work provides the first counter-evidence regarding cultural dimensions, serving as a warning for engineering practices like Constitutional AI.
- **vs Sorensen et al. (2024)**: While that work provided a conceptual taxonomy for pluralistic alignment, this paper implements it and provides quantitative evidence for "geo-cultural pluralism."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Strong methodological originality in combining the IW map, multilevel modeling, and Bayesian sensitivity scores for safety analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 8 datasets, multiple model types, and independent surrogate/triage experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed statistical reporting, though background in multilevel modeling is helpful for some technical sections.
- **Value**: ⭐⭐⭐⭐⭐ Directly challenges industry assumptions and provides actionable takeaways for rater recruitment and LLM triage in RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICML 2026\] MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
