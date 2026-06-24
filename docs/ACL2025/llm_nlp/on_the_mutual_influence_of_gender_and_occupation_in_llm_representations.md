---
title: >-
  [Paper Note] On the Mutual Influence of Gender and Occupation in LLM Representations
description: >-
  [ACL2025][LLM (Other)][gender bias] By approximating the gender direction in the LLM embedding space, this study systematically investigates the bidirectional influence between the gender representation of first names and occupational contexts: occupational contexts shift the gender representation of names, while the gender representation of names in turn affects the biased behaviors of LLMs in occupation prediction tasks, though the correlation between the two is only modera…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "gender bias"
  - "occupation stereotype"
  - "LLM embeddings"
  - "gender direction"
  - "first name representation"
date: 2026-05-08
content_hash: a6d5bfdbda59d715
---

# On the Mutual Influence of Gender and Occupation in LLM Representations

**Conference**: ACL2025  
**arXiv**: [2503.06792](https://arxiv.org/abs/2503.06792)  
**Code**: No public code  
**Area**: LLM/NLP  
**Keywords**: gender bias, occupation stereotype, LLM embeddings, gender direction, first name representation

## TL;DR
By approximating the gender direction in the LLM embedding space, this study systematically investigates the bidirectional influence between the gender representation of first names and occupational contexts: occupational contexts shift the gender representation of names, while the gender representation of names in turn affects the biased behaviors of LLMs in occupation prediction tasks, though the correlation between the two is only moderate.

## Background & Motivation
Names are frequently used as proxies for gender, and social science research has demonstrated that gender-associated name stereotypes lead to discriminatory treatment in education and employment. LLMs have been found to exhibit human-like gender bias in tasks like hiring decisions and recommendation letter writing. However, existing work primarily employs black-box approaches, failing to deeply explore the internal mechanisms of such bias.

The Core Problems of this paper:
- How do LLMs **internally** represent the gender of names? Are these representations consistent with real-world gender distributions?
- Do occupational contexts (e.g., "nurse" vs. "programmer") **alter** the gender representation of names?
- Can these internal gender representations **explain** the biased behavior of LLMs in downstream occupation prediction?

Difference from prior work: Previous research on embedding bias (Bolukbasi et al. 2016) was mostly conducted on static word embeddings or early contextualized models. This paper is the first to establish a connection between **internal gender representations** and **extrinsic biased behaviors** in modern LLMs (Llama-3.1, Mistral, etc.).

## Method

### Overall Architecture
The research progresses in three steps: (1) Approximating the gender direction in the LLM embedding space and validating its quality; (2) Analyzing the relationship between name gender representations, real-world statistics, and occupational contexts; (3) Studying the impact of gender representations on the model's biased behavior in downstream occupation prediction tasks.

### Key Designs

#### Key Design 1: Gender Direction Approximation and Validation
**Approximation Method**: PCA is used to decompose the embedding difference matrix of gender word pairs (e.g., 9 pairs such as she/he, woman/man, excluding Mary/John to avoid overfitting). Specific steps:
1. Extract 3,000 contextual sentences for each gender word from English Wikipedia, and generate paired sentences via counterfactual replacement.
2. Compute the average contextualized embedding difference for each pair of gender words.
3. Perform PCA on the difference matrix, where the first principal component is the approximated gender direction $\vec{g}$.

**Validation Method**: Design a binary classification task—predicting the associated gender of a name using its name embedding (or its dot product with the gender direction). If the dot product feature can maintain a classification accuracy comparable to the original high-dimensional embedding, it indicates that the gender direction effectively captures gender information.

PCA results show that the variance explained by the first principal component across the four models ranges from 32% to 42%, which is significantly higher than subsequent components. As the gender direction, the first principal component performs comparably to or even better than the full embedding in the classification task (e.g., OLMo-7B improves from 76.60% to 80.57%), while neither the second principal component nor the average direction can maintain the accuracy.

#### Key Design 2: Impact of Occupational Context on Gender Representation
Using the template sentence "{NAME} is a/an {OCC.}. {NAME} is " to construct contexts containing occupational information, the study analyzes changes in name embeddings before and after mentioning the occupation:
- Calculate the change in the dot product of the name embedding and the gender direction: $\Delta \text{DOT}(\vec{n}_{\text{temp}}, \vec{g})$
- Simultaneously obtain the probability change of the model outputting "female"/"male" tokens.

The 28 occupations are sourced from the Bias in Bios dataset, with varying gender ratios. "person" is also used as a baseline free of stereotypes.

#### Key Design 3: Bias Analysis in Downstream Occupation Prediction
Perform zero-shot occupation prediction on the Bias in Bios dataset:
- Sample 135 male and 135 female biographies for each of the 28 occupations.
- Replace the name placeholders in the biographies with 470 candidate names.
- Each LLM performs approximately 3.55 million inferences in total.
- Measure extrinsic bias using the **Bias Coefficient** (Pearson correlation between TPR and the degree of name feminization).
- Measure the predictive power of internal representations using the **Internal Coefficient** (Spearman correlation between embedded gender representation and occupation prediction probability).

## Key Experimental Results

### Models & Data
- **Models**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, OLMo-7B, Phi-3.5-mini (3.8B)
- **Names**: 470 names, grouped into 10 buckets based on female proportions in the SSA dataset, covering 4 races/ethnicities.
- **Total Inferences**: Over 12 million times.

### Table 1: Gender Direction Validation - Binary Classification Accuracy (%)

| Model | Original Embedding | 1st PC Dot Product | 2nd PC Dot Product | Random Direction Dot Product |
|---|---|---|---|---|
| Llama-3.1-8B | 75.46 | **75.18** | 50.78 | 47.09 |
| Mistral-7B | 74.04 | **67.80** | 58.44 | 55.18 |
| OLMo-7B | 76.60 | **80.57** | 55.46 | 56.03 |
| Phi-3.5-mini | 65.67 | **70.64** | 49.08 | 55.60 |

The dot product of the first principal component maintains or even exceeds the classification accuracy of the original embedding on most models, while the second principal component and the random direction are close to random chance, validating the effectiveness of the gender direction approximation.

### Table 2: Occupation Prediction Case Study - Impact of Name Gender on Prediction Results (Llama-3.1-8B)

| Biography Occupation | Name | Name Female % | Model Prediction | Correct? |
|---|---|---|---|---|
| pastor | Luis | 0.53% | pastor | ✓ |
| pastor | Logan | 7.37% | pastor | ✓ |
| pastor | Jerre | 43.70% | pastor | ✓ |
| pastor | Alejandra | 99.00% | journalist | ✗ |
| pastor | Khadijah | 99.90% | journalist | ✗ |
| dietitian | Duc | 0.00% | personal trainer | ✗ |
| dietitian | Hunter | 5.02% | personal trainer | ✗ |
| dietitian | Ivory | 59.32% | dietitian | ✓ |
| dietitian | Bonnie | 98.78% | dietitian | ✓ |

When only the name is changed in the same biography, the model is more prone to making errors when the name's gender does not match the occupational stereotype (e.g., female name + pastor, male name + dietitian).

### Figure 6: Comparison of Bias Coefficient and Internal Coefficient (Llama-3.1-8B)
- Spearman correlation between Bias Coefficient and Internal Coefficient: **0.61** (Llama-3.1-8B), **0.76** (Mistral-7B), both $p < 0.001$.
- Indicates that internal gender representations can **partially** explain extrinsic biased behavior, but they are not entirely consistent.
- Inconsistencies appear in certain occupations: for instance, "nurse" exhibits significant extrinsic bias but its internal coefficient is not significant, whereas "physician" has a significant internal coefficient but shows no extrinsic bias.

## Key Findings

1. **Gender representations align with the real world**: The projected values of name embeddings on the gender direction in LLMs show a strong linear correlation (significant Pearson correlation) with the actual female proportions in the SSA dataset, suggesting that the models learn name-gender associations from their training data that match reality.

2. **Occupational context shifts gender representation**: After mentioning female-dominated occupations (e.g., nurse, 90.9% female), name embeddings shift toward the female direction; mentioning male-dominated occupations (e.g., comedian, 21.1% female) shifts them toward the male direction. Strong gender-indicative names are less affected by occupations, whereas gender-ambiguous names are the most influenced.

3. **Limited internal explanation of biased behavior**: The correlation between internal gender representations and extrinsic biased behavior is moderate (0.61-0.76). There are cases of "false negatives" (extrinsic bias present but non-significant internal coefficient) and "false positives" (significant internal coefficient but no extrinsic bias), which corroborates previous findings that intrinsic and extrinsic bias metrics do not perfectly align.

4. **Cross-model consistency**: All four models with different architectures/training methods exhibit the above trends, indicating that these phenomena are not specific to any single model.

## Highlights & Insights

- **From Black-Box to White-Box**: This work for the first time systematically adapts the gender subspace method of Bolukbasi et al. to modern LLMs and establishes a complete analysis pipeline from internal representations to extrinsic behaviors, bridging the gap between embedding bias research and behavioral bias research.
- **Names as Continuous Variables**: Instead of only studying highly gender-indicative names, the study includes gender-ambiguous names and treats gender representation as a continuous spectrum rather than a binary classification, revealing that the model has different contextual sensitivities to names with varying degrees of gender certainty.
- **Exceptional Experimental Scale**: With 12 million+ prompts, 470 names, 28 occupations, and 4 LLMs, the conclusions are built upon a rigorous large-scale statistical basis.
- **Methodological Insight**: The dot product with the gender direction as a single-dimensional feature can match or even surpass the gender classification accuracy of high-dimensional embeddings, indicating that the concept of gender in LLMs is indeed highly concentrated on a single principal component.

## Limitations & Future Work

- **Binary Gender Framework**: The gender direction approximation is based on a female-male binary definition, which cannot cover non-binary gender identities.
- **Limited Demographic Coverage**: Names are drawn exclusively from US SSA and voter registration data, covering only 4 races/ethnicities, and lack cross-cultural and cross-lingual validation.
- **Restricted Model Sizes**: All experiments are conducted on smaller models with 4B-8B parameters (due to resource constraints); hence, trends in larger models (such as 70B, GPT-4) remain unknown.
- **Lack of Mitigation Strategies**: The paper focuses on the discovery and explanation of biases and does not propose concrete bias mitigation methods.
- **Decoupling of Intrinsic and Extrinsic Metrics**: The internal coefficient can only partially explain extrinsic bias, with multiple failure cases (e.g., nurse, physician), suggesting that gender representation at the embedding level alone is insufficient to fully capture the model's biased decision-making process.

## Related Work & Insights

- **Embedding Bias**: The Word2Vec gender subspace method by Bolukbasi et al. (2016) serves as the direct foundation for this work; Basta et al. (2019) extended it to contextualized embeddings; this paper further adapts it to instruction-tuned LLMs.
- **Names and Demographic Attributes**: Although using names as gender/race proxies has limitations (Gautam et al. 2024), it is widely adopted in fairness research. This work enhances the granularity of the analysis by explicitly introducing gender-ambiguous names.
- **Intrinsic vs. Extrinsic**: Goldfarb-Tarrant et al. (2021) and Cao et al. (2022) have pointed out the inconsistency between intrinsic and extrinsic bias metrics. This paper reconfirms this in the context of name-occupation scenarios in LLMs.
- **Insights**: If one expects to eliminate occupation prediction bias through embedding-level interventions (e.g., debiasing projections), solely addressing the gender direction may not suffice; higher-order interaction effects and information flow within the attention mechanisms must also be considered.

## Rating

- Novelty: ⭐⭐⭐ — The method (PCA-based gender direction) is not novel; the contribution primarily lies in systematically adapting existing methods to modern LLMs and establishing internal-external connections.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 12 million+ prompts, 4 models, 470 names, and 28 occupations. The statistical testing is rigorous and the ablation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and coherent logic with thorough ethical discussions, though the heavy use of mathematical notation somewhat increases the cognitive load.
- Value: ⭐⭐⭐ — The findings are meaningful but somewhat descriptive. The conclusion that internal representations only partially explain extrinsic bias somewhat dampens the practical value of the methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](../../AAAI2026/llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)

</div>

<!-- RELATED:END -->
