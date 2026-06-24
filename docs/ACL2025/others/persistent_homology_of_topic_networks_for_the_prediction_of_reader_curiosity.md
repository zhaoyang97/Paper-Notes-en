---
title: >-
  [Paper Note] Persistent Homology of Topic Networks for the Prediction of Reader Curiosity
description: >-
  > This paper quantifies the topic network structure of text into topological voids (connected components, loops, cavities) using persistent homology, serving as a proxy for "information gaps" to predict reader curiosity, achieving 73% explained deviance on the novel *The Hunger Games* (vs. 30% for the baseline).
tags:

date: 2026-05-08
content_hash: 069f7c35a12ccdf9
---

# Persistent Homology of Topic Networks for the Prediction of Reader Curiosity

| Info | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2506.11095](https://arxiv.org/abs/2506.11095) |
| Code | [GitHub](https://github.com/mds-hopp/pers_homol_data) |
| Area | others (Computational Linguistics × Topological Data Analysis × Motivational Psychology) |
| Keywords | persistent homology, topic modeling, reader curiosity, information gap, topological data analysis |

## TL;DR

> This paper quantifies the topic network structure of text into topological voids (connected components, loops, cavities) using persistent homology, serving as a proxy for "information gaps" to predict reader curiosity, achieving 73% explained deviance on the novel *The Hunger Games* (vs. 30% for the baseline).

## Background & Motivation

- **Modeling Reader Curiosity**: Curiosity is a key cognitive-affective factor driving readers to continue reading, yet computational modeling of it in NLP remains scarce.
- **Limitations of Prior Work**: Existing research on reader engagement prediction mostly relies on surface-level linguistic features (sentiment analysis, readability, bag-of-words representations), lacking modeling of macro-level semantic structures and information gaps. Although knowledge graph approaches provide richer semantic representations, they do not explicitly model the dynamic evolution of the reader's information state.
- **Information Gap Theory**: Loewenstein (1994) proposed that curiosity arises from the perception of a gap between "what one knows" and "what one wants to know." Under this theoretical framework, this paper quantifies the information gaps in textual semantic structures via Topological Data Analysis (TDA).
- **Research Hypothesis**: Topological voids (disconnected components, loops, cavities) within a text's topic network can serve as a proxy for information gaps; the dynamic changes in these gaps capture the elements of "surprise"—a key driver of curiosity.

## Method

### Overall Architecture

The paper proposes a three-stage pipeline from text preprocessing to topological feature extraction:

1. **Preprocessing**: The text is cleaned and segmented using a sliding window (5 sentences per window, 2-sentence overlap), generating 2,656 text fragments.
2. **Dynamic Topic Modeling**: A dynamic topic network is constructed to represent the evolution of the textual topic structure.
3. **Topological Feature Extraction via Persistent Homology**: Topological voids in the topic network are quantified using persistent homology.

### Key Designs

#### Topic Modeling (Inspired by BERTopic)

- **Text Embedding**: The text fragments are embedded into a 1024-dimensional vector space using Voyage-AI's voyage-3-large model.
- **Dimensionality Reduction**: UMAP is utilized to reduce the embeddings to 32 dimensions (cosine distance, 15 nearest neighbors) as a preprocessing step for clustering.
- **Clustering**: HDBSCAN (minimum cluster size of 3) is applied to automatically determine the number of clusters, ultimately identifying 302 topics. 27% of the embeddings are classified as noise and excluded.

#### Dynamic Topic Network Construction

- **Vertices**: One vertex for each topic.
- **Edges**: Connect topics that appear in consecutive text fragments, reflecting the narrative flow.
- **Edge Weights**: Computed using the cosine similarity of topic vectors in the original embedding space.
- **Network Sequence**: Segmented by chapter (the points of user curiosity ratings) to construct a cumulative dynamic network sequence—the $n$-th graph contains all topics and relationships from the first $n$ chapters.

#### Persistent Homology Metrics

Vietoris-Rips filtration is applied to each static graph to construct simplicial complexes, extracting the following topological features:

- **Betti Numbers**: $\beta_0$ (number of connected components), $\beta_1$ (number of 1D loops), $\beta_2$ (number of 2D cavities)—representing information gaps.
- **Bottleneck Distance**: The maximum difference between the persistence diagrams of adjacent chapters—capturing major singular structural shifts.
- **Wasserstein Distance**: The average difference between the persistence diagrams of adjacent chapters—capturing average structural changes.

All features are detrended using linear models and winsorized at the 2.5%–97.5% level.

### Statistical Modeling

Generalized Additive Models (GAM) are employed to predict reader curiosity:
- **Null Model**: Controls only for control variables (number of new topics per chapter, chapter index).
- **Full Model**: Control variables + topological features (Betti numbers, Wasserstein distance, Bottleneck distance).
- Cubic regression splines are used with a basis dimension of $k=4$, REML estimation, and an additional penalty term to prevent overfitting.
- Significance is evaluated using 1,000 permutation tests.

## Experiments

### Dataset

- **Text**: The novel *The Hunger Games* by S. Collins, 27 chapters.
- **Reader Data**: Participants ($n=76$) were recruited via the Prolific platform; a subset of readers unfamiliar with the book and movie ($n=49$) provided chapter-by-chapter curiosity ratings (0–100).
- **Inter-rater Reliability**: Calculated using a mixed-effects model, ICC = 0.71 (moderate), indicating reasonable consistency in curiosity patterns across chapters.

### Main Results

| Model | Explained Variance $R^2$ | Explained Deviance |
|------|------------|---------|
| Null Model (Controls Only) | 23.8% | 29.7% |
| Full Model (+ Topological Features) | 65.7% | 72.9% |

- Likelihood Ratio Test: $\chi^2 = 11.25$, $\text{df} = 4.7$, $p < .001$, indicating a significant improvement in model fit after incorporating topological features.
- Permutation tests also confirmed the significance of the full model ($R^2$: $p < .05$; deviance: $p < .06$).

### Topic Distribution Analysis

- Each chapter contains an average of 25 topics ($\text{SD}=5$, range 15–35).
- Chapter 11 introduces the most new topics (28), corresponding to the start of the Hunger Games.
- Chapter 25 introduces the fewest new topics (1), corresponding to the end of the games.
- Topic clustering shows significant topic shifts around Chapters 11 and 26, aligning with the novel's key narrative turning points.

### Network Characteristics

- The final network contains 302 vertices and 778 edges, with an average degree of 5.15.
- The small-world index is 3.40, indicating a small-world network structure.
- Walktrap community detection identifies two main communities, corresponding to the "Hunger Games Phase" (Chapters 11–25) and the remaining chapters, respectively.

### Key Findings

- The trends of Betti numbers, Bottleneck distance, and Wasserstein distance highly align with the macro narrative structure of *The Hunger Games*.
- Topological features independently contribute approximately 40% of additional explained variance (increasing from 24% to 66%), validating the computational operationalization of Information Gap Theory.

## Highlights & Insights

1. **Interdisciplinary Innovation**: This research extends Topological Data Analysis (TDA) from static text representations to dynamic topic networks, combining it with the Information Gap Theory of motivational psychology to provide a novel paradigm for reader engagement modeling.
2. **Computational Operationalization of Information Gaps**: The abstract concept of an "information gap" is mapped to concrete topological voids (connected components, loops, cavities), providing quantifiable metrics.
3. **Capturing Narrative Structure**: Topological features naturally capture macro narrative turning points (such as the start/end of the games) without requiring any manual narrative annotation.
4. **High Explanatory Power with Small Samples**: With only 27 observation points (chapters), the full model still explains 73% of the deviance.
5. **Open Source**: The code and data are publicly available.

## Limitations & Future Work

1. **Extremely Small Sample Size**: Only one novel is analyzed with 27 chapters as observation points, which limits the robustness of the conclusions.
2. **Single Genre of Text**: Tested solely on young adult fiction; generalization to other genres such as expository texts, news, or educational materials remains unknown.
3. **Demographic Limitations**: The participants are exclusively UK residents; cultural differences might affect the generalizability of curiosity patterns.
4. **Undirected Network Structure**: The topic network is undirected, ignoring the directionality of learning dependencies, which could be particularly important in educational texts.
5. **Topic Networks vs. Knowledge Graphs**: Topic networks may restrict semantic granularity; emerging paradigms like LightRAG/GraphRAG could provide richer representations.
6. **Lack of Linguistic Integration**: The study focuses only on semantic/topological features, without incorporating lexical or syntactic linguistic features.

## Related Work & Insights

- **Computational Models for Reader Engagement**: These range from surface features (sentiment, readability) to cognitive features (uncertainty, semantic coherence), yet none have succeeded in modeling the reader's dynamic information state.
- **Graph and Topological Approaches in NLP**: Applications of TDA in NLP are growing, but primarily focus on static text representation. For instance, Christianson et al. (2020) used TDA to identify knowledge gaps in mathematics textbooks, while Tymochko et al. (2021) captured logical holes in summarization.
- **Network Science and Curiosity**: Patankar et al. (2023) used persistent homology to track structural changes in time-varying graphs, but did not focus on reader engagement. This study bridges this gap.

## Rating ⭐⭐⭐⭐

Highly innovative with a meaningful interdisciplinary contribution. The main deduction is due to the evaluation being limited to a single novel with a very small sample size ($n=27$ chapters), casting significant uncertainty on its generalizability. However, as a proof of concept, the pipeline design is sound, the theoretical motivation is clear, and the results are encouraging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling](s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](../../AAAI2026/others/piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)

</div>

<!-- RELATED:END -->
