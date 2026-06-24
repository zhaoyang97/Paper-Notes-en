---
title: >-
  [Paper Note] Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes
description: >-
  [ACL2025][Text Generation][topic modeling] The authors propose the Retell method: leveraging small LMs to generate abstractive retellings of literary passages, converting "showing" sensory details in narratives into "telling" high-level concepts, and subsequently running LDA topic modeling on the retold texts. Under resource-constrained conditions, this approach significantly outperforms baselines using direct LDA and directly querying LMs for topic labels.
tags:
  - "ACL2025"
  - "Text Generation"
  - "topic modeling"
  - "literary analysis"
  - "abstractive retelling"
  - "LDA"
  - "cultural analytics"
date: 2026-05-08
content_hash: 6f8964807a9d2134
---

# Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes

**Conference**: ACL2025  
**arXiv**: [2505.23166](https://arxiv.org/abs/2505.23166)  
**Code**: [lucy3/tell_dont_show](https://github.com/lucy3/tell_dont_show)  
**Authors**: Li Lucy, Camilla Griffiths, Sarah Levine, Jennifer L. Eberhardt, Dorottya Demszky, David Bamman  
**Affiliations**: UC Berkeley, Stanford University  
**Area**: Text Generation  
**Keywords**: topic modeling, literary analysis, abstractive retelling, LDA, cultural analytics

## TL;DR

The authors propose the Retell method: leveraging small LMs to generate abstractive retellings of literary passages, converting "showing" sensory details in narratives into "telling" high-level concepts, and subsequently running LDA topic modeling on the retold texts. Under resource-constrained conditions, this approach significantly outperforms baselines using direct LDA and directly querying LMs for topic labels.

## Background & Motivation

Topic modeling in literary text analysis is a key task in cultural analytics. Traditional bag-of-words methods like LDA face unique challenges when processing literary texts. The golden rule of creative writing is **"show, don't tell"**—effective narrative conveys themes through low-level sensory details (character actions, dialogue, scene descriptions) rather than high-level abstract explanations. Consequently, word-level features relied on by LDA struggle to capture deep themes across documents.

For example, in a passage describing a character dragging their body slowly, LDA can only capture surface-level vocabulary like "sluggishly," "arms," and "legs," but fails to extract high-level themes such as "consequences of violence" or "physical trauma."

Furthermore, a practical challenge persists: although powerful LMs (such as GPT-4) open new possibilities, humanities researchers are often constrained by API costs and computational resources. Existing LM-based topic modeling methods (like TopicGPT) require complex multi-step prompt chains, and smaller LMs exhibit unstable performance when directly generating topic labels. For instance, within the TopicGPT framework, Llama 3.1 8B produced 486 topics for just 100 documents, and the generated labels tended to be overly broad (e.g., "life" covering 32.9% of the paragraphs).

**Key Insight**: Rather than forcing LMs to directly output topic labels, it is better to have LMs "tell" us what the literary passages are "showing"—that is, producing abstractive retellings that translate the surface forms of narratives into high-level concepts, and then applying traditional LDA to these retold texts.

## Method

### Retell Framework

The method consists of two steps:

**Step 1: Abstractive Retelling**

An instruction-tuned small LM is used to generate a retelling for each literary passage (up to 250 words). Using a short, single-turn prompt template, the core instruction is:
"In one paragraph, [VERB] the following book excerpt for a literary scholar analyzing narrative content."

Three verbs for `[VERB]` are experimented with:
- `describe`: encourages high-level abstraction
- `summarize`: encourages high-level abstraction
- `paraphrase`: preserves more low-level details of the original text

The average length of the retellings ranges from approximately 105 to 170 words.

**Step 2: LDA Topic Modeling**

Mallet LDA is run on the retold texts, with preprocessing steps including:
- Lowercasing and removing words with fewer than 3 characters
- Removing high-frequency words that appear in more than 25% of the documents
- Removing low-frequency words that appear in fewer than 5 retellings
- Using spaCy NER to remove character names (to prevent clustering by book)

### Evaluated Models

Four resource-efficient, small instruction-tuned LMs:
- GPT-4o mini (closed-source)
- Llama 3.1 8B (open-source)
- Phi-3.5-mini / 3.8B (open-source)
- Gemma 2 2B (open-source)

### Baseline Methods

1. **Default LDA**: Running LDA directly on the raw passages.
2. **TopicGPT-lite**: A two-stage scheme adapted from TopicGPT.
    - Stage 1 (Topic Generation): The LM proposes one topic document-by-document on a sample of $N=1000$ documents.
    - Stage 2 (Topic Assignment): The LM assigns topic labels to all documents.
    - Restricting to single-label generation is applied to mitigate the topic-explosion issue in small LMs.

### Practical Advantages of Retell

- Each passage requires only a single LM inference pass followed by LDA, making it more computationally efficient than TopicGPT-lite.
- The number of topics $k$ can be adjusted quickly without needing to re-run the LM.
- It operates with a single prompt, avoiding complex prompt engineering.

## Key Experimental Results

### Experiment 1: Passage-Level Label Relevance Evaluation (Table 2)

**Dataset Construction**:
- Collected 50.7k titles from Project Gutenberg and contemporary bestseller lists.
- Used Goodreads reader tags, SparkNotes, and LitCharts topic labels as gold standards.
- Manually grouped labels into 27 general themes (e.g., gender, race, war, love, etc.).
- Obtained 11.6k annotated passages in total (732 books, 21.1k topic-passage pairs).
- Appended an equal number of random passages to stabilize LDA estimation, totaling 5.02M words.

**Evaluation Method**:
Prolific crowdsourced annotators judged the semantic relevance between predicted topics and the gold standards. A 3-point scale was used with a wage of $\$16$/hour; inter-annotator agreement yielded a weighted Cohen's kappa of $0.70$.

| Method | Highly Relevant | Irrelevant |
|------|---------|--------|
| **Retell-describe** | **0.60** | **0.10** |
| **Retell-summarize** | **0.59** | 0.11 |
| Retell-paraphrase | 0.50 | 0.14 |
| Default LDA | 0.38 | 0.27 |
| TopicGPT-lite | 0.22-0.35 | 0.17-0.68 |

**Key Findings**:
- Retell-describe/summarize significantly outperforms all baselines.
- The abstractive verbs `describe`/`summarize` outperform `paraphrase` (0.60 vs 0.50), validating the hypothesised advantage of "telling" over "showing".
- Default LDA's topics are heavily populated with functional/filler words (e.g., "n't", "got", "say"), leading to vague semantics.
- TopicGPT-lite generates excessively broad labels (e.g., a "loneliness" label incorrectly covering the "education" theme).

### Experiment 2: Passage-Level Topic Intrusion Test (Table 3)

Evaluated by in-house annotators with experience in film and literary annotation (weighted Cohen's kappa = $0.66$). Intrusion topic ratings were assigned alongside the top-3 predicted topics for 50 text passages:

| Method | Top-1 | Top-2 | Top-3 | Intruder |
|------|-------|-------|-------|--------|
| Retell-desc (GPT-4o mini, $k=50$) | **2.81** | **2.51** | 2.23 | 1.63 |
| Retell-desc (GPT-4o mini, $k=89$) | 2.60 | 2.53 | **2.40** | 1.77 |
| TopicGPT-lite (GPT-4o mini, $k=89$) | 2.59 | 2.48 | 2.51 | 1.52 |
| Retell-summ (Llama 8B, $k=50$) | 2.36 | 2.30 | 2.12 | 1.67 |

The top topic scores of all methods are significantly higher than the intruder scores (Mann-Whitney U test, $p < 0.05$). Retell exhibits comparable performance to TopicGPT-lite at the passage level, while being much more lightweight and efficient.

### Case Study: Race Themes in ELA Textbooks (Table 4-5)

**Data**:
396 US high school English Language Arts (ELA) textbooks (including AP Literature curriculum books and teacher-recommended reading lists), and 1,645 manually annotated passages (401 "mention" + 198 "discuss" + remainder "neither"). The dataset was coded over four months by an undergraduate team led by social psychology experts.

**Findings**:
- Retell produces topic words highly relevant to racial identity, such as "black, racial, white, community, individuals."
- The probability of these topics is significantly higher in "discuss" passages than in "mention" passages (Mann-Whitney U test, $p < 0.001$).
- Combining two relevant topics from Retell improves recall without sacrificing precision.
- The corresponding top words in Default LDA ("black, people, white") show weaker discriminative power.
- The labels generated by TopicGPT-lite (e.g., "Identity," "Family," "Work") show no significant differences across the three categories of passages.

## Highlights & Insights

- **Conceptual Novelty**: Ingeniously translates the creative writing principle of "show vs. tell" into a computational methodology, allowing the LM to act as a "translation layer" bridging narrative details and abstract concepts.
- **Minimalist Design**: Implements a single-prompt approach combined with standard LDA, requiring no complex prompt chains or model fine-tuning. This is highly user-friendly for humanities scholars and resource-constrained environments.
- **Interdisciplinary Value**: Bridges the gap between NLP methodological innovation and applications in humanities education, with the case study demonstrating a practical capability to analyze racial themes.
- **Multi-dimensional Evaluation**: Integrates crowdsourced evaluation, expert annotation, topic intrusion tests, and a domain-specific case study, covering both topic-level and passage-level granularities.

## Limitations & Future Work

- LM-generated retellings represent only one potential interpretation, as literary reading is inherently a subjective and culturally constructed process.
- LMs might supplement information not present in the passage using book knowledge from prior pre-training, introducing contextual bias.
- The study focuses solely on explicit mentions of race; the identification of implicit racial cues requires deeper investigation.
- Retellings might omit crucial descriptive details (such as descriptions of racial stereotypes), making the content selection behavior in summaries an area worthy of study.
- The performance of larger models remains under-explored (preliminary results for GPT-4o show that stronger LMs can also be highly effective at direct label generation).
- Gold standard labels are derived from specific online resources, which may introduce content production and coverage biases.

## Related Work & Insights

- **LDA (Blei et al., 2003)**: A classic probabilistic topic model that relies on surface-level word frequencies. It serves as a foundational component in this work, operating on top of the LM retellings.
- **TopicGPT (Pham et al., 2024)**: Obtains topic labels directly from LMs via multi-step prompting. It is used as a baseline here but exhibits unstable performance on small LMs.
- **Embedding-based Topic Models (e.g., BERTopic)**: Clusters documents based on embeddings. This represents a different technical approach compared to Retell, which directly leverages the abstractive generation capabilities of LMs.
- **Latent Information Imputation (Zhong et al., 2022; Hoyle et al., 2023)**: Uses LMs to describe documents to extract latent information, standing as the closest pioneering work.
- **Computational Humanities (Piper, 2018; Underwood, 2019)**: Belongs to the tradition of "distant reading." Retell provides a new tool for distant reading analysis.

## Rating

- Novelty: ⭐⭐⭐⭐  
  The translation of the "show to tell" metaphor into a systematic methodology is highly ingenious; the combination of abstractive retelling with LDA is both simple and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐  
  Provides a systematic comparison across multiple models, verbs, and $k$ values, evaluated through multi-layered assessments combining crowdsourcing, expert annotations, and a case study.
- Writing Quality: ⭐⭐⭐⭐⭐  
  An exemplary piece of interdisciplinary paper writing with a clear motivation, smooth narrative structure, and comprehensive ethical discussion.
- Value: ⭐⭐⭐⭐  
  Offers direct application value to computational humanities and cultural analytics. The method is simple and easily generalizable to other narrative text analyses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Abstractive Snippet Generation](abstractive_snippet_generation.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2025\] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)

</div>

<!-- RELATED:END -->
