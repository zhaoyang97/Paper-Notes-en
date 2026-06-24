---
title: >-
  [Paper Note] Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis
description: >-
  [ICML 2025][Interpretability][LLM Personality Analysis] This paper proposes the Supernova Event Dataset (comprising Wikipedia articles of biographies, historical events, news, and scientific discoveries). By instructing LLMs to extract and rank key events from long texts, and utilizing another LLM as a judge to infer the target model's "personality traits," this work reveals differences in the consistent behavioral patterns of different LLMs during subjective decision-making.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "LLM Personality Analysis"
  - "Event Extraction and Ranking"
  - "LLM-as-Judge"
  - "Subjective Reasoning"
date: 2026-05-08
content_hash: 1ab95fababf7a2ea
---

# Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis

**Conference**: ICML 2025  
**arXiv**: [2506.12189](https://arxiv.org/abs/2506.12189)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: LLM Personality Analysis, Event Extraction and Ranking, Interpretability, LLM-as-Judge, Subjective Reasoning

## TL;DR

This paper proposes the Supernova Event Dataset (comprising Wikipedia articles of biographies, historical events, news, and scientific discoveries). By instructing LLMs to extract and rank key events from long texts, and utilizing another LLM as a judge to infer the target model's "personality traits," this work reveals differences in the consistent behavioral patterns of different LLMs during subjective decision-making.

## Background & Motivation

Modern LLM benchmarks primarily focus on tasks with objective ground-truth answers (e.g., question answering, reasoning). However, as LLMs are increasingly deployed in high-stakes fields such as healthcare, law, and finance, evaluating factual accuracy alone is no longer sufficient; understanding the models' **subjective judgments** and **value inclinations** has become crucial.

Prior work has shown that LLMs can simulate personality traits when explicitly prompted with specific personas. However, the core finding of this work is that **even without role-playing prompts, LLMs exhibit consistent behavioral patterns when handling complex subjective tasks**, and these patterns can be interpreted as "personalities."

Critical event identification and ranking is inherently a subjective task:
- It requires reasoning across long contexts.
- It requires modeling causal chains and non-linear interactions between events.
- Different individuals (and models) make distinct choices due to variations in underlying values.

This makes the task an ideal tool for probing the latent decision-making tendencies of LLMs.

## Method

### Overall Architecture

The framework consists of three stages:

1. **Dataset Construction**: Building the Supernova Event Dataset, which includes four categories of Wikipedia articles (biographies, historical events, news, and scientific discoveries).
2. **Event Extraction and Ranking**: The target LLMs receive articles via RAG, extract, and rank the five most critical events.
3. **Personality Judgment**: Another LLM (Judge) analyzes the target model's event selection and ranking to infer its personality type.

### Key Designs

#### Dataset Construction (Supernova Event Dataset)

| Category | Source | Min Word Count | Min Views | Additional Filtering | Article Count |
|------|--------|---------|-----------|---------|--------|
| Biography | English Wikipedia | 3000 | 50000 | Infobox template filtering | 150 |
| Historical Events | English Wikipedia | 500 | 5000 | ORES $\ge$ B + LLM verification + Year < 2000 | 150 |
| News Events | English Wikipedia | 500 | 5000 | ORES $\ge$ B + LLM verification + Year > 2000 | 150 |
| Scientific Discoveries | Gemini Deep Research | - | - | Nobel Prize API + Gemini expansion | 25 |

Highlights of dataset design:
- **Biography**: Requires $\ge 3000$ words to ensure coverage of the subject's entire life, using standardized infobox templates.
- **Historical/News Events**: Two-stage filtering—first applying heuristic rules to filter ambiguous pages, then using a local LLaMA-3-8B for semantic validation (confidence $> 0.9$).
- **Scientific Discoveries**: Extracted 384 award records (1901-2024) from the Nobel Prize REST API, expanded into encyclopedic articles using Gemini 2.5 Pro Deep Research.

#### RAG Pipeline and Event Extraction

Document processing pipeline:
1. **Chunking**: Segmenting documents into semantic chunks of 1000 tokens (with a 100-token overlap).
2. **Embedding**: Using the `nomic-embed-text-v1` model to generate high-dimensional vectors.
3. **Indexing**: Storing vectors in a FAISS vector database.
4. **Retrieval**: `MultiQueryRetriever` rewrites queries into multiple search queries to improve retrieval recall.

**Two-stage prompt strategy**:
- **First-stage prompt**: Directs the retriever to focus on critical event characteristics such as "turning points" and "cascading effects," rather than merely fetching topically relevant content.
- **Second-stage prompt**: Guides the LLM to perform structural analysis, requiring it to identify and rank the five most critical events.

For scientific discoveries, counterfactual testing ("would the result change if this event did not occur?") is additionally incorporated as a selection criterion.

#### Personality Judgment Framework

- **Judge Model**: Uses Qwen-2.5 14B as an external evaluator.
- **Evaluation Method**: The Judge receives the target LLM's complete event selection and ranking output to analyze its decision-making patterns.
- **Personality Encoding**: Employs `sentence-transformers` (`all-MiniLM-L6-v2`) to perform semantic embedding of the identified personality traits.
- **Visualization**: Applies PCA dimensionality reduction to the aggregated embeddings to visualize the model's personality location in a 2D space.
- **Similarity Measure**: Calculates cosine similarity to quantify personality similarity across different models.

### Loss & Training

This work **does not involve model training**; instead, it is an evaluation framework. Its core components include:

- **Inference-time strategy**: Structured prompt guidance combined with RAG retrieval augmentation.
- **Personality quantification**: Aggregation of trait embeddings weighted by frequency.
- **Analysis of scientific discoveries**: Combines keyword counting and open coding to converge into three types of decision-making principles:
    - **Causality-centric**: Focuses on mechanisms and causal pathways.
    - **Enablement-centric**: Focuses on foundations, barrier removal, and validation.
    - **Synthesis-centric**: Emphasizes conceptual integration and paradigm-level connections.

## Key Experimental Results

### Main Results

**Evaluated Models**:
- Small Models: Phi-4, Orca 2 (13B), Qwen 2.5 (14B)
- Large Models (for Scientific Discoveries): Claude Sonnet 3.7, Gemini 2.5 Pro, OpenAI o3

**Distribution of Personality Categories** (seven personality dimensions):

| Model | Strategic Achiever | Creative Innovator | Emotional | Community Support | Ideological | Observational | Influencer |
|------|-----------|-----------|-------|---------|---------|-------|-------|
| Phi-4 | **Highest** | **High** | Moderate | Low | Low | Low | Low |
| Orca 2 | Moderate | Low | **Highest** | Moderate | Low | Low | Low |
| Qwen 2.5 | **Highest** | High | Moderate | Moderate | Moderate | Moderate | Moderate |

**Distribution of Decision-Making Principles in Scientific Discoveries**:

| Model | Causality-centric | Enablement-centric | Synthesis-centric |
|------|-----------|-----------|-----------|
| o3 | **Dominant** | Moderate | Low |
| Gemini 2.5 Pro | Moderate | **Dominant** | Low |
| Claude 3.7 Sonnet | Low | Prominent | **Dominant** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Movie Script Dataset (1,172 scripts) | Consistent personality patterns | Verifies personality stability across different domains |
| Phi-4's performance in movies | Strategic/plot-oriented | Prefers "Jafar's schemes", "Zuckerberg's decision to launch Facebook" |
| Orca 2's performance in movies | Emotional/relationship-oriented | Prefers "Aladdin meeting Jasmine", "Mark and Eduardo's fallout" |
| Qwen 2.5's performance in movies | Milestone-oriented | Prefers "The creation of Facemash", "The final performance at Lincoln Center" |

### Key Findings

1. **Reproducible Model Personality**: Models exhibit consistent behavioral preferences across different domains (biographies, financial crises, movie scripts, and scientific discoveries).
2. **Significant Differences in Small Models**: Phi-4 leans toward "strategic achievement," Orca 2 leans toward "emotional reasoning," and Qwen 2.5 is the most balanced.
3. **Divergent Reasoning Styles in Large Models**: o3 shows causal reasoning (step-by-step), Gemini focuses on empirical validation, and Claude excels in conceptual integration.
4. **Clear Separation in Semantic Space**: PCA visualization shows that the three small models occupy distinctly separate personality regions.
5. **No Role-Playing Required**: Personality traits naturally emerge without explicit personality prompting.

## Highlights & Insights

- **Elegant Task Design**: Critical event ranking is an inherently subjective task with no single correct answer, thereby directly reflecting the model's value preferences. This probes model behavior at a deeper level than traditional benchmarks.
- **Prompt-agnostic**: The personality identification methodology in this work does not rely on specific prompt engineering; the behavioral patterns of the models remain consistent across different prompts.
- **Insightful Scientific Discovery Analysis**: The three types of reasoning principles (causality, enablement, and synthesis) provide a practical reference for selecting LLMs: use o3 when causal analysis is needed, Gemini for methodological foundation evaluation, and Claude for cross-domain conceptual integration.
- **Counterfactual Testing**: Using "would the outcome have changed if this event had not occurred?" to filter key events is methodologically rigorous.
- **Significance for AI-Assisted Research**: Understanding the reasoning personality of LLMs assists in designing better human-AI collaborative research workflows.

## Limitations & Future Work

1. **Data Bias**: Wikipedia naturally contains inherent editorial biases and Western-centrism, which may influence the inferred personality labels.
2. **LLM-as-Judge Bias**: The evaluator model possesses its own stylistic bias and lacks human validation.
3. **Non-standardized Personality Framework**: The personality categories are empirically derived rather than being grounded in established psychological frameworks like the Big Five.
4. **Small Sample Size for Scientific Discoveries**: Consisting of only 25 articles, the statistical significance is limited.
5. **Lack of Adversarial Testing**: Whether model personalities remain stable under adversarial prompting has not been verified.
6. **Exclusion of Inference Parameters**: The impact of inference parameters such as temperature was not analyzed; different sampling strategies might affect event selection.
7. **Single Evaluator**: Only Qwen 2.5 was used as the judge, without cross-validation from a multi-judge committee.

## Related Work & Insights

- **LLM Personality Research**: Jiang et al. (2023) and Bodroža et al. (2024) utilize psychometric tools like the Big Five to evaluate the behavioral traits of LLMs. This study extends this exploration to scenarios without explicit personality prompts.
- **Event Extraction**: DDEE (Liu & Luo, 2024), ULTRA (Zhang et al., 2024), and EventRL (Gao et al., 2024) focus on the accuracy of event extraction; this paper shifts the focus to the subjective dimension of event **importance ranking**.
- **Long-Context Reasoning**: NoLiMa (Modarressi et al., 2025) and BABILong (Kuratov et al., 2024) evaluate long-context capabilities; this work complements these evaluations from a personality perspective.
- **Inspirations for Future Ideas**: The personality analysis framework can be extended to more models and domains; combining it with mechanistic interpretability can probe how personality-related features are internally represented in models.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Novel task design mapping event ranking to personality inference |
| Technical Depth | 3 | The method itself is relatively straightforward (RAG + prompt + judge) without complex model designs |
| Experimental Thoroughness | 3 | Sufficient cross-domain validation, but sample sizes are limited and human evaluation is lacking |
| Value | 4 | Offers practical guidance for model selection and human-AI collaboration |
| Writing Quality | 4 | Clear structure, rich cases, and strong readability |
| **Overall** | **3.5** | Valuable concept, but requires a more rigorous validation framework |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[ICLR 2026\] Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility](../../ICLR2026/interpretability/is_this_just_fantasy_language_model_representations_reflect_human_judgments_of_e.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](../../ACL2026/interpretability/dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)

</div>

<!-- RELATED:END -->
