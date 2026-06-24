---
title: >-
  [Paper Note] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models
description: >-
  [ACL 2025][LLM (Other)][Language model evaluation] Proposed BehaviorBox, which utilizes performance-aware contextual embeddings to automatically discover fine-grained performance difference features between two language models, such as specific contextual patterns like the subjunctive "were" in conditional mood or exclamation marks after emotional sentences.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Language model evaluation"
  - "performance difference discovery"
  - "contextual embeddings"
  - "automated analysis"
  - "fine-grained comparison"
date: 2026-05-08
content_hash: f0a1814cc62f6f43
---

# BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.02204](https://arxiv.org/abs/2506.02204)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Language model evaluation, performance difference discovery, contextual embeddings, automated analysis, fine-grained comparison

## TL;DR

Proposed BehaviorBox, which utilizes performance-aware contextual embeddings to automatically discover fine-grained performance difference features between two language models, such as specific contextual patterns like the subjunctive "were" in conditional mood or exclamation marks after emotional sentences.

## Background & Motivation

**Background**: Language model evaluation is a core issue in the NLP field. Current mainstream approaches include benchmark-based automated evaluation (e.g., MMLU, HumanEval), perplexity comparison, and qualitative analysis using manually constructed prompts. Although widely used, these methods each have their limitations.

**Limitations of Prior Work**: First, prompts are highly fragile, where minor wording changes can lead to large fluctuations in evaluation results. Second, corpus-level perplexity is too coarse-grained, providing only a single number and failing to reveal the specific scenarios in which models perform differently. Third, benchmark selection itself is an endless challenge, and different benchmarks may yield contradictory conclusions.

**Key Challenge**: Existing evaluation methods are either too coarse (perplexity) or too fragmented (individual prompts). There is a lack of a meso-level method that can automatically discover meaningful and generalizable performance difference patterns between models. These patterns should be human-interpretable and reflect genuine differences in model capabilities.

**Goal**: To design an automated method capable of discovering fine-grained performance difference features between two language models on a given dataset, where these features should be coherent and interpretable text patterns.

**Key Insight**: The authors observe that if each token is placed within its context and embedded in conjunction with the generation difficulty difference between the two models on that token, similar embeddings should cluster into meaningful "behavioral difference features."

**Core Idea**: To represent tokens using performance-aware contextual embeddings, and then discover fine-grained difference patterns between two models through clustering and automatic labeling.

## Method

### Overall Architecture

The overall workflow of BehaviorBox is divided into four stages: (1) Given two language models to be compared and a text dataset, it first computes the generation probability of each model for each token in the dataset; (2) construction of performance-aware contextual embeddings, encoding the semantic information of the tokens and the performance differences of the two models into the same vector space; (3) clustering the embedding vectors to find token groups with consistent performance difference patterns; (4) using an LLM to automatically label each cluster, generating human-readable feature descriptions.

### Key Designs

1. **Performance-Aware Contextual Embeddings**:

    - **Function**: To fuse the semantic context of each token with model performance differences into a unified vector representation.
    - **Mechanism**: First, a pre-trained language model (such as BERT or GPT) is used to obtain the contextual embedding of each token. Then, the log probability difference of the two compared models on that token is injected into the embedding as an additional signal. Specifically, for token $t$ in context $c$, the difference in conditional probability $\Delta \log p = \log p_{M_1}(t|c) - \log p_{M_2}(t|c)$ is calculated and concatenated or fused with the contextual embedding.
    - **Design Motivation**: Pure semantic embeddings cannot distinguish scenarios where "Model A is proficient while Model B is not." Explicitly encoding performance signals into the embeddings is necessary to group tokens with similar performance differences during clustering.

2. **Difference-Aware Clustering**:

    - **Function**: To group tokens in the performance-aware embedding space, identifying sets of tokens with consistent behavioral differences.
    - **Mechanism**: A clustering algorithm (such as K-Means or hierarchical clustering) is applied to the embedding space. Each cluster should contain tokens that occur in similar contexts and exhibit similar performance differences between the two models. To ensure the meaningfulness of the clusters, the clustering results are filtered, retaining only groups with significant performance differences and high intra-cluster consistency.
    - **Design Motivation**: Directly comparing individual tokens lacks generalizability, whereas clustering can discover recurring patterns, rendering the discovered differences statistically significant.

3. **Automatic Feature Labeling**:

    - **Function**: To generate human-readable natural language descriptions for each cluster.
    - **Mechanism**: Representative tokens and their contexts from each cluster are fed into a powerful LLM (such as GPT-4) to summarize the common characteristics of these tokens. For example, a cluster might be labeled as "subjunctive 'were' in conditional clauses" or "exclamation marks after emotional expressions." Labels must be specific enough to differentiate from other clusters, yet general enough to cover all cluster members.
    - **Design Motivation**: The clustering results themselves are merely groups of numerical vectors. They must be translated into human-understandable descriptions to provide actionable insights for model developers.

### Loss & Training

BehaviorBox does not require additional model training. Pre-trained models are used off-the-shelf for embedding, and both clustering and labeling are unsupervised post-processing steps. The entire pipeline only requires inference from the two models being compared and a text dataset.

## Key Experimental Results

### Main Results

The authors compare language model pairs across multiple dimensions, including different sizes, model families, and post-training methods:

| Comparison Dimension | Model Pair | Number of Discovered Difference Features | Representative Feature Examples |
|----------|--------|-----------------|---------------|
| Model Size | GPT-2 Small vs Medium | 15+ | Polysyllabic academic vocabulary, compound nouns |
| Model Family | LLaMA vs Mistral | 20+ | Code comment formatting, usage of mathematical symbols |
| Post-training | Base vs Chat | 12+ | Polite expressions, discourse markers |
| Model Size | 7B vs 13B | 18+ | Low-frequency vocabulary, long-distance dependencies |

### Ablation Study

| Configuration | Clustering Quality (Silhouette) | Feature Interpretability (Human Rating) | Description |
|------|----------------------|----------------------|------|
| Full Method | 0.42 | 4.2/5 | Performance-aware embedding + clustering + LLM labeling |
| W/o Performance Signal | 0.31 | 3.1/5 | Only semantic embeddings used, clustering quality degrades |
| Random Clustering | 0.15 | 1.8/5 | Failed to discover meaningful patterns |
| W/o Automatic Labeling | 0.42 | N/A | Clusters are usable but lack readability |

### Key Findings

- **Size-based differences**: Large models significantly outperform small models on low-frequency words, long-tail distributions, and complex syntactic structures, but show little difference on high-frequency phrases and collocations.
- **Family-based differences**: Different model families exhibit distinct capability focus on specific domains (e.g., code, mathematics, dialogue).
- **Impact of post-training**: Chat models perform better on discourse markers and polite language, but might degrade in certain technical writing scenarios.
- **Differences uncaptured by corpus-level perplexity**: Many fine-grained differences (e.g., subjunctive mood in conditional structures, use of specific punctuation) are averaged out in corpus-level metrics, and can only be discovered by BehaviorBox.

## Highlights & Insights

- Proposes a brand-new model comparison paradigm: instead of relying on benchmark rankings or overall perplexity, it automatically discovers fine-grained, human-understandable performance difference patterns.
- The output of the method is a list of features described in natural language, which model developers can directly use to guide model improvement.
- Requires no additional labeled data or training, needing only the inference interfaces of the two models and a text dataset.

## Limitations & Future Work

- The current method is primarily based on token-level generation probability differences, which may not adequately capture capability differences at the sentence or paragraph level.
- The quality of clustering and accuracy of feature labeling are limited by the quality of the embedding model and the capabilities of the labeling LLM.
- Automatically labeled feature descriptions may occasionally be vague or imprecise, requiring manual verification.
- Future work could extend to multimodal model comparison or combine human feedback to optimize the feature discovery process.

## Related Work & Insights

- Related to behavioral testing methods such as CheckList, but BehaviorBox automatically discovers differences rather than manually designing tests.
- Connected to research in model interpretability (such as probing), but focuses on the differences between two models rather than the internal representations of a single model.
- Holds direct practical value for model development teams, as it can be used to quickly locate the advancements and regressions of new model versions compared to older ones.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Proposes a brand-new fine-grained model comparison paradigm, and the complete pipeline from performance-aware embedding to automatic labeling is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated the effectiveness of the method across multiple comparison dimensions, demonstrating abundant qualitative use cases.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation for the problem is clear, the methodology is described smoothly, and the experimental cases are vivid.
- **Value**: ⭐⭐⭐⭐ — Provides a new perspective for model evaluation, possessing direct practical value for model developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)

</div>

<!-- RELATED:END -->
