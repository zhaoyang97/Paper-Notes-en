---
title: >-
  [Paper Note] When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Metaphor Detection] A computational framework combining LLM word-level metaphor detection with SBERT document-level semantic association is proposed. Evaluating on 400,000 tweets about US immigration, it reveals a complex landscape where conservatives use more dehumanizing metaphors, but biological metaphors exert a stronger effect on user engagement among liberals.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Metaphor Detection"
  - "Dehumanizing Language"
  - "Immigration Discourse"
  - "Political Ideology"
  - "Social Media Analysis"
  - "LLMs"
  - "Document Embeddings"
date: 2026-05-08
content_hash: 816e27b5854e4b54
---

# When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.13246](https://arxiv.org/abs/2502.13246)  
**Code**: [github.com/juliamendelsohn/when_people_are_floods](https://github.com/juliamendelsohn/when_people_are_floods)  
**Area**: LLM/NLP  
**Keywords**: Metaphor Detection, Dehumanizing Language, Immigration Discourse, Political Ideology, Social Media Analysis, LLMs, Document Embeddings  

## TL;DR

A computational framework combining LLM word-level metaphor detection with SBERT document-level semantic association is proposed. Evaluating on 400,000 tweets about US immigration, it reveals a complex landscape where conservatives use more dehumanizing metaphors, but biological metaphors exert a stronger effect on user engagement among liberals.

## Background & Motivation

**Metaphor as a core rhetorical tool in political discourse**: Conceptual Metaphor Theory (Lakoff & Johnson, 1980) posits that metaphors construct conceptual mappings that highlight certain aspects of an issue while hiding others, thereby influencing public attitudes and policy preferences. In immigration discourse, dehumanizing metaphors comparing immigrants to floods, vermin, and parasites are widespread.

**Abundant qualitative research but lacking large-scale quantitative methods**: Critical discourse analysis has extensively documented seven source domains of immigration metaphors (animal, vermin, parasite, physical pressure, water, commodity, war). However, these studies primarily rely on small-scale manual analysis, making systematic measurement at social media scale difficult.

**Contested relationship between metaphors and ideology**: While conservatives lean towards threat framing (Mendelsohn et al., 2021), differences in metaphor usage between left- and right-wing media are not always significant (Porto, 2022). The impact of ideological extremity on metaphor usage also remains under-explored.

**Inconsistent findings on the audience effects of metaphors**: Experimental evidence shows that animal metaphors increase support for immigration restrictions (Utych, 2018), but other studies find that conservatives resist extreme metaphors (Hart, 2021; Boeynaems et al., 2023), whereas liberals might be more susceptible to metaphoric framing.

**Existing NLP methods focus on word-level binary detection**: Dominant approaches treat metaphor detection as a word-level binary classification task (BERT-based), neglecting document-level conceptual associations, and few studies focus on the large-scale analysis of political metaphors.

**Four research questions proposed**: H1: Whether conservative ideology is associated with higher metaphor usage; RQ1: Whether extreme ideologues use more metaphors than moderates; H2: Whether metaphor usage is associated with higher user engagement; RQ2: How ideology moderates the relationship between metaphors and engagement.

## Method

### Overall Architecture

The system consists of three core components: (1) LLM-based **word-level metaphor detection**, identifying metaphorical expressions in text mapped to seven source domains; (2) SBERT-based **document-level metaphor association**, calculating semantic similarity between documents and source domain concepts; and (3) **composite scoring (SUM)**, which adds word-level and document-level scores to obtain an overall metaphor score. Linear regression is then utilized to analyze the relationships between metaphors, ideology, and user engagement.

### Key Designs

#### Module 1: LLM Word-Level Metaphor Detection

- **Function**: Prompt LLMs to identify metaphorical words in the text and map them to corresponding source domains (or label them as "none").
- **Mechanism**: Two zero-shot prompting strategies are designed—Simple (providing only concept names and basic instructions) and Descriptive (additionally providing metaphor definitions and conceptual descriptions). Three LLMs (Llama3.1-70B, GPT-4-Turbo, and GPT-4o) are evaluated.
- **Design Motivation**: Metaphor detection requires semantic comprehension; LLMs' contextual understanding capability is inherently suited for this task. Descriptive prompting helps the model distinguish literal from metaphorical meanings by providing extra contextual information.
- **Word-level score calculation**: $\text{LLM}_{\text{concept}} = \frac{C(\text{concept})}{\log(C(\text{words}) + 1)}$, where $C(\text{concept})$ is the number of detected metaphorical expressions, and $C(\text{words})$ is the word count of the document. Logarithmic normalization is employed to prevent linear distortion between short and long texts.

#### Module 2: SBERT Document-Level Semantic Association

- **Function**: Use SBERT (all-MiniLM-L6-v2) to calculate the cosine similarity between tweets and source domain "carrier sentences".
- **Mechanism**: Even if a text does not contain specific source domain vocabulary, its overall semantic logic may still implicitly evoke metaphorical concepts. Document embeddings are leveraged to capture this document-level metaphoric association.
- **Design Motivation**: Directly embedding source domain names (e.g., "water") over-matches literal usages (e.g., immigrants crossing the sea). Thus, 104 "carrier sentences" are manually constructed to represent metaphorical usage (e.g., "they flood in", "they hunt them down"), with 8-22 carrier sentences per source domain.
- **Document-level score**: $\text{EMB}_{\text{concept}} = \cos(\mathbf{e}_{\text{tweet}}, \bar{\mathbf{e}}_{\text{carriers}})$, representing the cosine similarity between the tweet embedding and the average embedding of the carrier sentences.

#### Module 3: Composite Scoring (SUM)

- **Function**: Sum the word-level and document-level scores directly to obtain a comprehensive metaphor score.
- **Mechanism**: $\text{SUM}_{\text{concept}} = \text{LLM}_{\text{concept}} + \text{EMB}_{\text{concept}}$. Word-level signals cover explicit metaphors while document-level signals cover implicit metaphors, providing complementary information.
- **Design Motivation**: SUM naturally biases toward word-level signals (due to the sparsity of metaphorical words) but can still detect metaphors through document-level signals in the absence of explicit metaphorical words. Simple additive combination already outperforms individual components, leaving more complex fusion strategies for future work.

### Loss & Training

This method **requires no training on annotated data**, needing only: (1) brief descriptions of concepts; (2) a few examples of carrier sentences. Evaluation is conducted on a crowdsourced dataset of 1,600 tweets (annotated by approximately 8 annotators each), where the proportion of annotator agreement serves as the continuous ground truth. Evaluation metrics include Spearman correlation and ROC-AUC at different thresholds. In the analysis phase, linear regression models are used to control for message, author, and time variables, applying Holm-Bonferroni correction for multiple comparisons with a significance level of $p=0.05$.

## Key Experimental Results

### Main Results: Metaphor Detection Model Comparison (ROC-AUC, 30% Threshold)

| Model Combination | Without SBERT | With SBERT |
|---|---|---|
| Llama3.1 + Simple | 0.661 | 0.702 |
| Llama3.1 + Descriptive | 0.512 | 0.635 |
| GPT-4o + Simple | 0.681 | 0.715 |
| **GPT-4o + Descriptive** | **0.684** | **0.731** |
| GPT-4-Turbo + Simple | 0.643 | 0.682 |
| **GPT-4-Turbo + Descriptive** | **0.702** | **0.746** |

### Ablation Study: Gain of SBERT Document Signals

| Model | Gain at 30% Threshold | Gain at 70% Threshold | Gain at 90% Threshold |
|---|---|---|---|
| GPT-4o + Descriptive | +0.047 | +0.053 | +0.017 |
| GPT-4-Turbo + Descriptive | +0.044 | +0.042 | +0.014 |
| Llama3.1 + Simple | +0.041 | +0.043 | +0.033 |

### Key Findings

- **H1 Supported**: Conservative ideology is significantly correlated with higher metaphor scores across all seven source domains, with war and water showing the strongest effects, and creature categories (parasite, vermin, animal) showing the weakest.
- **Complex outcomes for RQ1**: Among conservatives, higher ideological extremity correlates with more metaphor usage (all source domains); among liberals, extremity correlates negatively with water/commodity metaphors but positively with creature metaphors—indicating that both extreme left and right wings use biological metaphors more heavily.
- **H2 Partially Supported**: Creature-based metaphors (vermin, parasite, animal) are significantly associated with higher retweets, while commodity metaphors are associated with fewer favorites.
- **Engagement effects for RQ2**: The positive effect of creature metaphors on retweets is primarily driven by **liberals**; the direction of engagement for water metaphors is opposite between the left and right (positive for conservatives, negative for liberals).
- **Four patterns of liberal metaphor usage**: (1) Direct adoption of metaphors (e.g., referring to immigrants as a "wave"); (2) Sympathetic framing (e.g., "they cage them like animals"); (3) Critical quoting of opponents' rhetoric (retelling conservatives' "infestation"); and (4) Redirecting dehumanizing metaphors toward political opponents.
- **Best Performing Model**: GPT-4o + Descriptive + SBERT performs optimally at most thresholds, with only 1/4 of the inference cost of GPT-4-Turbo.
- **SBERT Consistently Beneficial**: Integrating document-level signals improves ROC-AUC across all LLM × prompt combinations, proving that word-level and document-level signals are complementary.

## Highlights & Insights

1. **Zero-annotation methodological innovation**: The entire metaphor detection process requires no human annotation, relying only on concept descriptions and carrier sentences, greatly lowering the barrier for cross-domain migration.
2. **Word-level + document-level dual-channel design**: Resolves the limitation of word-only detection omitting implicit metaphors, and document-only detection lacking precision; simple summation outperforms individual components.
3. **Continuous metaphor metrics instead of binary classification**: Recognizing the continuous nature of metaphors, the proportion of crowdsourced annotations is used as ground truth, aligning more closely with linguistic intuition.
4. **Revealing the ideology-metaphor-engagement triangle**: Obtains the counter-intuitive finding that liberals are more sensitive to creature-based metaphors (producing more retweets), suggesting that the effects of dehumanizing metaphors transcend left-right divides.
5. **Qualitative analysis complementing quantitative findings**: Identifies four patterns of liberal usage of dehumanizing metaphors, showing that even with a pro-immigrant stance, harmful conceptual mappings can still be implicitly reinforced.

## Limitations & Future Work

1. **Lack of causal inference**: Regression analysis only establishes correlation without testing causal hypotheses, leaving the attribution of user engagement ambiguous.
2. **Single embedding model**: Document-level association was only tested using one SBERT model (all-MiniLM-L6-v2), leaving larger or domain-adapted models unexplored.
3. **Manual construction of carrier sentences**: The 104 carrier sentences rely on manual design, which may introduce bias; methods for the automatic discovery of metaphorical frames warrant investigation.
4. **Simple combination strategy**: Direct summation (SUM) may not be optimal; learning weighted coefficients or using more complex fusion strategies remains to be explored.
5. **Limited to English, Twitter, and the US**: The generalizability of the method to cross-lingual, cross-platform, or cross-cultural scenarios has not been verified.
6. **Pragmatic intent of metaphors unconsidered**: The approach does not distinguish between sympathetic usage, ironic quoting, and direct dehumanization, although the social consequences of different intents may differ dramatically.
7. **Limited engagement data**: Only favorites/retweets counts are available, without information on who is interacting, restricting inference regarding audience susceptibility.

## Related Work & Insights

- **Conceptual Metaphor Theory (Lakoff & Johnson, 1980)**: The theoretical foundation of this study, treating metaphors as cognitive structures rather than rhetorical ornaments.
- **Card et al. (2022)**: Leverages BERT token probabilities to quantify dehumanizing associations in political speeches; this study extends the scope to social media and introduces document-level signals.
- **Mendelsohn et al. (2020, 2021)**: Explores embedding associations between group labels and "vermin" using word2vec, providing the immigration tweets dataset utilized in this paper.
- **MelBERT (Choi et al., 2021) / FrameBERT (Li et al., 2023)**: Representative works in BERT-based metaphor detection, though constrained to word-level binary classification and requiring extensive annotation.
- **Sengupta et al. (2024)**: Finds that liberals perceive highly metaphorical comments as more persuasive, echoing the engagement findings of this work.
- **Insights**: The framework can be migrated to analyze metaphors in other political topics (such as climate change, gun control); the zero-annotation concept association method (carrier sentences + SBERT) is worth extending to other NLP implicit semantic detection tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The word-level + document-level dual-channel zero-annotation framework shows clear innovation; however, the components themselves (LLM prompting, SBERT cosine similarity) are combinations of existing techniques.
- **Technical Depth**: ⭐⭐⭐ — The method is simple and effective but presents limited technical challenges, with no model training and a straightforward combination strategy (direct addition).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparison across three LLMs × two prompt strategies × with/without SBERT; the crowdsourced evaluation dataset is reasonably designed, and the regression analysis controls for multiple confounding variables.
- **Practical Value**: ⭐⭐⭐⭐⭐ — The method is transferable without annotations and has open-sourced code, holding direct application value for computational social science and political discourse analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues](enhancing_spoken_discourse_modeling_in_language_models_using_gestural_cues.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)
- [\[ACL 2025\] When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)

</div>

<!-- RELATED:END -->
