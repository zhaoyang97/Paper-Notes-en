---
title: >-
  [Paper Note] Entity Framing and Role Portrayal in the News
description: >-
  [ACL 2025][LLM (Other)][Entity framing] This paper constructs a multilingual hierarchical entity framing corpus containing 5 languages, 1,378 news articles, and over 5,800 annotated entities. It proposes a narrative role classification system comprising 22 fine-grained roles (under three main frames: protagonist, antagonist, and innocent) and establishes baselines on fine-tuning multilingual Transformers and hierarchical zero-shot learning with LLMs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Entity framing"
  - "narrative roles"
  - "multilingual annotation"
  - "hierarchical classification"
  - "zero-shot learning"
date: 2026-05-08
content_hash: f38255f79ad7119b
---

# Entity Framing and Role Portrayal in the News

**Conference**: ACL 2025  
**arXiv**: [2502.14718](https://arxiv.org/abs/2502.14718)  
**Code**: [Dataset Page](https://mbzuai-nlp.github.io/entity-framing/)  
**Area**: Other  
**Keywords**: Entity framing, narrative roles, multilingual annotation, hierarchical classification, zero-shot learning

## TL;DR

This paper constructs a multilingual hierarchical entity framing corpus containing 5 languages, 1,378 news articles, and over 5,800 annotated entities. It proposes a narrative role classification system comprising 22 fine-grained roles (under three main frames: protagonist, antagonist, and innocent) and establishes baselines on fine-tuning multilingual Transformers and hierarchical zero-shot learning with LLMs.

## Background & Motivation

In the information age, where social media and news outlets are highly developed, the "framed" representation of entities (individuals, organizations, groups) in news reports has a profound impact on public perception:

**The Power of Affective Framing**: Referring to the same group as "freedom fighters" vs. "terrorists" triggers completely different emotional responses, yet current NLP research majorly remains at the level of sentiment polarity.

**Limitations of Coarse-Grained Framing**: Existing research typically categorizes entities simply into heroes, villains, or victims. However, narrative roles in actual news coverage are far more complex, and the same entity might play different roles across different paragraphs.

**Lack of Multilingual and Domain Diversity**: Existing datasets are mostly monolingual and single-domain.

**Entity-level vs. Document-level Analysis**: Most framing analyses are performed at the document level, lacking fine-grained characterization of specific entities.

Core Innovation: Redefining framing categories based on narrative function rather than moral judgment, replacing hero/villain/victim with protagonist/antagonist/innocent, thereby focusing on the functional roles of entities within the narrative.

## Method

### Overall Architecture

#### Classification System Design

22 fine-grained roles are nested under three main narrative frames:
- **Protagonist**: Guardian, Martyr, Underdog, Peacemaker, Rebel, Virtuous, Unifier, etc.
- **Antagonist**: Tyrant, Deceiver, Bigot, Foreign Aggressor, Instigator, Corrupt, Incompetent, etc.
- **Innocent**: Victim, Scapegoat, Exploited, Forgotten

Task Formulation: Given an article $S$ and an entity mention span $[i,j]$, predict the set of roles $\{r_1, r_2, ..., r_k\} \subseteq R$.

### Key Designs

1. **Corpus Construction Pipeline**:

    - **Article Selection**: Candidate articles were collected from large-scale news aggregation tools $\rightarrow$ filtered by keywords (>250 words) $\rightarrow$ manually reviewed (Perfect Fit / Average Fit / Uncertain / Unfit) $\rightarrow$ further filtered using zero-shot classifiers and persuasiveness scores.
    - **Coverage**: 5 languages (Bulgarian, English, Hindi, Portuguese, Russian), across 2 domains (Russia-Ukraine War and Climate Change).
    - **Annotation Process**: Each article was annotated by 2 annotators $\rightarrow$ reviewed and consolidated by a curator $\rightarrow$ subjected to periodic random quality assurance checks.
    - **Annotation Tool**: INCEpTION.

2. **XLM-R Fine-Tuning Experimental Design**:

    - Input Format: `entity mention + [SEP] + title + [SEP] + context`
    - Three Context Granularities: Document (DOC), Paragraph (PAR), Sentence (SEN).
    - Handling Long Documents: Bypassing the 512-token limit by narrowing the context window to paragraph or sentence levels.
    - Multi-label Classification: Sigmoid activation + Binary Cross-Entropy loss.

3. **LLM Hierarchical Zero-Shot Learning**:

    - **Single-Step**: A single prompt predicts both the main frame and fine-grained roles simultaneously.
    - **Multi-Step**: First predicts the main frame (protagonist/antagonist/innocent), then targets the fine-grained roles based on it.
    - Using GPT-4o for zero-shot inference.

### Loss & Training

- XLM-R: Binary Cross-Entropy loss is used for multi-label classification.
- Split train/dev/test sets at the article level to prevent data leakage.
- Evaluation under two settings: multilingual joint training vs. monolingual training.

## Key Experimental Results

### Main Results: Performance of Different Context Granularities

| Context | Main Frame Accuracy | Main Frame Balanced Acc | Fine-Grained Micro F1 | Fine-Grained Macro F1 |
|-------|---------------|-------------------|-----------------|-----------------|
| DOC | 0.601 / 0.723* | 0.590 / 0.724* | 0.391 | 0.231 |
| PAR | **0.738** / **0.753*** | **0.739** / **0.755*** | 0.421 | 0.239 |
| SEN | 0.718 / 0.750* | 0.712 / 0.750* | **0.434** | **0.253** |

> M = trained on the main frame only, F = evaluated on the main frame after training on fine-grained roles. Paragraph level performs best on the main frame, while sentence level performs best on the fine-grained roles.

### Zero-Shot vs. Fine-Tuning Comparison

| Method | Main Frame Accuracy | Fine-Grained Micro F1 | Fine-Grained Macro F1 | Cost (USD) |
|------|---------------|-------------|-------------|----------|
| GPT-4o Single-Step | 0.703 | 0.382 | **0.310** | $5.32 |
| GPT-4o Multi-Step | 0.705 | 0.317 | 0.277 | $3.19 |
| XLM-R (PAR) | **0.753** | **0.421** | 0.239 | - |

> XLM-R wins on Micro F1, but zero-shot performs better on Macro F1—due to insufficient training data for XLM-R on rare roles.

### Key Findings

1. **Paragraph-Level Context is Optimal**: Full-text information introduces excessive noise and interference, while individual sentences are too short to provide adequate narrative context.
2. **Multilingual Training Consistently Outperforms Monolingual**: Cross-lingual transfer significantly boosts performance across all languages.
3. **Extreme Class Imbalance**: Within the innocent class, 83.6% are classified as victim, and 74% of the entities appear only once.
4. **Role Transitions are Rare but Exist**: Only 99 out of 1,378 articles exhibit transitions in the main framing role, but these transition sequences are highly informative.
5. **Multi-Step Method is More Cost-Effective**: It is 40% cheaper than the single-step method and achieves comparable performance on the main frame, but degrades on fine-grained roles due to error propagation.
6. **Consistently Low Macro F1**: All methods show poor performance on rare roles, highlighting the severe challenge of class imbalance.

## Highlights & Insights

- **Well-Designed Classification System**: The hierarchical taxonomy of 22 narrative roles is validated through extensive real-world annotation, providing much richer analytical dimensions than simple three-way classification.
- **"Narrative Function" Replaces "Moral Judgment"**: Using protagonist/antagonist/innocent is more objective than hero/villain/victim, reducing subjective bias during annotation.
- **Valuable Multilingual Coverage**: The combination of 5 languages across 2 geopolitically sensitive domains provides a unique resource for cross-cultural media analysis.
- **Role Co-occurrence and Transition Analysis**: The study finds that peacemakers often co-occur with guardians, and scapegoats with those who are exploited, reflecting the complexity of real-world narratives.
- **Complementarity of Zero-Shot vs. Fine-Tuning**: Zero-shot is superior at handling rare categories, whereas fine-tuning performs better on common classes, suggesting that combining the two approaches could yield optimal results.

## Limitations & Future Work

1. **Limited Domains**: The corpus covers only two domains: the Russia-Ukraine War and Climate Change.
2. **Subjectivity in Annotation**: Despite comprehensive guidelines, entity framing annotation remains highly subjective by nature.
3. **Moderate IAA**: Krippendorff's $\alpha$ ranges between 0.43 and 0.73, which is acceptable but not particularly high.
4. **Severe Class Imbalance**: Most fine-grained roles have very few samples, which limits model training.
5. **Zero-Shot Reliance on Closed-Source Models**: Dependence on GPT-4o, which may face deprecation, poses a risk to reproducibility.
6. **Unexplored Few-shot and Multi-task Learning**: These techniques could potentially improve performance on rare categories but remain uninvestigated.

## Related Work & Insights

- **Sharma et al. (2023)** identify heroes, villains, and victims in memes, but focus exclusively on the visual modality.
- **Bergstrand & Jasper (2018)** analyze the moral attributes of the hero/villain/victim triad; the functional role definition proposed in this paper serves as an improvement on their framework.
- **Relationship to ABSA**: The task here is related to but distinct from Aspect-Based Sentiment Analysis (ABSA)—instead of assigning sentiment polarities, it assigns narrative roles.
- Insights: Entity framing analysis can be integrated with tasks like disinformation detection and propaganda analysis, contributing to a more complete media literacy toolchain.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The hierarchical classification system of 22 roles and the multilingual corpus serve as entirely new resources. |
| Experimental Thoroughness | 4 | Broad coverage including fine-tuning, zero-shot learning, and corpus analysis. |
| Writing Quality | 4 | The paper is structured as a standard dataset paper with comprehensive statistical analyses. |
| Value | 4.5 | Highly valuable as a community resource with broad application prospects. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Entity Identification in Language Models](on_entity_identification_in_language_models.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)

</div>

<!-- RELATED:END -->
