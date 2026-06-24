---
title: >-
  [Paper Note] Inducing Lexicons of In-Group Language with Socio-Temporal Context
description: >-
  [ACL 2025][Lexicon Induction] This paper proposes the LISTN (Lexicon Induction with Socio-Temporal Nuance) framework, which utilizes dynamic word and user embeddings to jointly model the social structure and temporal evolution of community language. On the task of in-group lexicon induction within the anti-feminist online community (manosphere), LISTN achieves an average precision of 0.77, significantly outperforming existing methods.
tags:
  - "ACL 2025"
  - "Lexicon Induction"
  - "In-group Language"
  - "Dynamic Embeddings"
  - "Socio-temporal Modeling"
  - "Manosphere"
date: 2026-05-08
content_hash: a8f110a496b259a8
---

# Inducing Lexicons of In-Group Language with Socio-Temporal Context

**Conference**: ACL 2025  
**arXiv**: [2409.19257](https://arxiv.org/abs/2409.19257)  
**Code**: [GitHub](https://github.com/unimelb-nlp/listn)  
**Area**: Others  
**Keywords**: Lexicon Induction, In-group Language, Dynamic Embeddings, Socio-temporal Modeling, Manosphere

## TL;DR

This paper proposes the LISTN (Lexicon Induction with Socio-Temporal Nuance) framework, which utilizes dynamic word and user embeddings to jointly model the social structure and temporal evolution of community language. On the task of in-group lexicon induction within the anti-feminist online community (manosphere), LISTN achieves an average precision of 0.77, significantly outperforming existing methods.

## Background & Motivation

**In-group language** serves as an important marker of social groups: on the one hand, it is used for external obfuscation (making it difficult for outside observers to understand); on the other hand, it facilitates internal cohesion (signaling group identity). Since such informal language evolves rapidly, adopting the latest lexical innovations acts as a strong signal of group belonging.

Existing computational methods for in-group lexicon induction suffer from two main limitations:

**Manual construction is expensive and prone to obsolescence**: For instance, Rowe & Saif (2016) used a dictionary that was 7 years old when studying ISIS language.

**Computational methods ignore social and temporal dimensions**: Lucy & Bamman (2021) used contextualized word embeddings, and Farrell et al. (2020) used topic models, but both relied solely on linguistic information, ignoring the dynamic social structure of the group.

The **manosphere** (anti-feminist online communities), the focus of this paper, is a highly suitable subject of study: (1) it exhibits extremely active linguistic innovation (e.g., *foid*, *AWALT*); (2) the subgroup structures are complex and dynamically evolving (Incels, MRA, MGTOW, PuA, TRP); and (3) it is linked to real-world violent events, presenting an urgent social concern.

## Method

### Overall Architecture

LISTN consists of two steps:
1. **Representation Learning**: Train dynamic word embeddings and user embeddings using the Cerberus architecture (jointly factorizing the user-content matrix and the user-user adjacency matrix).
2. **Lexicon Induction**: A scoring method based on low-rank reconstruction computes the relevance of each word to different subgroups at various timesteps.

### Key Designs

#### 1. Representation Learning: Cerberus Dynamic Matrix Factorization

**Function**: At each timestep $t$, jointly factorize the user-content matrix $C_t$ and the user-user adjacency matrix $A_t$ to obtain dynamic user embeddings $U_t$ and word embeddings $W_t$.

**Mechanism**:
- **Content Matrix** $C_t$: Constructed using PPMI (Positive Pointwise Mutual Information) to measure the extent to which user $i$'s frequency of using word $j$ deviates from the background corpus.
- **Adjacency Matrix** $A_t$: Captures the interaction frequency of two users within the same discussion thread.
- Joint Factorization: $C_t \approx U_t \cdot W_t^T$ and $A_t \approx U_t \cdot V_t^T$.
- Temporal Regularization: Penalizes major changes in embeddings between consecutive timesteps to ensure temporal alignment.

**Design Motivation**: Mapping users and words into the same space naturally integrates social structure (who interacts with whom) with linguistic content (who uses what words). Temporal regularization ensures that embedding changes reflect true socio-semantic evolution rather than noise.

**Implementation Details**: Reimplemented using PyTorch, adopting a Generalized Matrix Factorization (GMF) formulation to support batch updates and sparse matrix processing. The training data consists of over 4 million posts from 50 manosphere subreddits, spanning a 9-month period from April to December 2018.

#### 2. Lexicon Induction Methods: Six LISTN Variants

Given the user embedding $u_{i,t}$ and word embedding $w_{j,t}$, the relevance of word $j$ to user $i$ at time $t$ is $r(i,j,t) = u_{i,t} \cdot w_{j,t}^T$.

Six aggregation methods:

| Method | Calculation | Concept |
|------|---------|------|
| **Community centroid** | Mean of all users $\times$ word vector | Community-wide level |
| **Category centroid** | Max across means of each sub-community (Incel/MRA/...) | Accounting for subgroup specialization |
| **Subreddit centroid** | Max across means of each subreddit | Finer granularity |
| **Cluster (K=5/20/100)** | Max across cluster centroids after K-means clustering | Data-driven subgroup discovery |
| **Bootstrap** | Nearest neighbors of seed lexicon | Lexicon expansion |
| **Bias** | Word bias term of the factorization model | Global popularity |

**Temporal Aggregation**: Taking the maximum score of a word across all timesteps allows the model to capture words entering or exiting popularity phases.

**Design Motivation**: Utilizing max instead of mean aggregation is motivated by the idea that a word should be considered in-group language as long as it is highly relevant in at least one subgroup. Methods at different granularities explore the effect of finding the "optimal subgroup partitioning."

#### 3. Evaluation Framework

**Task Definition**: Lexicon induction is framed as a binary classification task—determining whether a word is an in-group lexical innovation.

**Test Set Construction**: 
1. Initial scoring: Seed words from 5 existing manosphere dictionaries (483 known words) are scored by all methods.
2. Select the top-1000 words from the best baseline and the best LISTN method.
3. Independently annotated by the author and an expert with a PhD in social psychology (Cohen's Kappa = 0.726).
4. Final test set: 1,803 words (944 positive / 859 negative).

**Evaluation Metrics**: Average Precision (AP) and AUROC, with a primary focus on AP.

### Baselines

- **word2vec bootstrap**: Training word2vec followed by nearest-neighbor expansion of the seed lexicon.
- **PMI variants**: PPMI/NPMI computed at various granularities (community, subreddit, category, month).

## Key Experimental Results

### Main Results: Lexicon Induction Performance

| Method | AP | AUROC |
|------|-----|-------|
| Random | 0.52 | 0.50 |
| word2vec bootstrap | 0.5563 | 0.5427 |
| NPMI-category (best baseline) | 0.6790 | 0.6647 |
| LISTN-CA Cluster-5 | 0.7620 | 0.7403 |
| **LISTN-C Cluster-5** | **0.7679** | **0.7363** |
| LISTN-C Category | 0.7272 | 0.6809 |

### Key Method Comparisons

| Aggregation Granularity | LISTN-CA AP | LISTN-C AP |
|----------|-----------|-----------|
| Community (Global) | 0.6228 | 0.6297 |
| Category (5 classes)| 0.7231 | 0.7272 |
| Subreddit (52 subs) | 0.5519 | 0.5891 |
| Cluster-5 | **0.7620** | **0.7679** |
| Cluster-20 | 0.7069 | 0.7554 |
| Cluster-100 | 0.6950 | 0.7040 |
| Bootstrap | 0.5349 | 0.5276 |
| Bias | 0.6190 | 0.6016 |

### Key Findings

1. **LISTN-C (Content only) $\ge$ LISTN-CA (Content + Adjacency)**: Surprisingly, incorporating user interaction information does not improve lexicon induction ($p=0.723$, not significant). However, LISTN-C still retains social information—users who employ the same vocabulary are represented similarly.
2. **Cluster-5 is Optimal**: Clustering with $K=5$ outperforms the category-level division (which also has roughly 5 categories). This indicates that data-driven subgroup partitioning is more effective than platform-defined ones.
3. **Overly Fine Granularity is Harmful**: Subreddit-level aggregation performs the worst, likely because users who occasionally post in a subreddit introduce noise.
4. **Bootstrap/Bias Methods Perform Worst**: Approaches that ignore community structure perform poorly, showing that subgroup-specific specialization is the key.
5. **NPMI > PPMI**: Consistent with Lucy & Bamman (2021). However, monthly NPMI does not outperform global NPMI, suggesting that simple temporal slicing is insufficient to capture temporal dynamics.

## Additional Findings from Embedding Analysis

### Temporal Stability of Word Representations

- The embeddings of in-group words are more stable (lower CEV) than general words of comparable frequencies, indicating that in-group words function as social symbols governed by usage norms, and members use them "correctly."
- Word frequency is strongly negatively correlated with CEV ($\rho = -0.77$), but the rate of change approaches zero beyond a frequency of 10,000.
- Low-frequency, highly stable words include pharmaceutical names (e.g., *lamictal*, *seroquel*), indicating that technical terminology remains unaffected by social dynamics.

### Subgroup Linguistic Specialization

| Group Pair | Spearman $\rho$ | Explanation |
|--------|-----------|------|
| PuA $\leftrightarrow$ TRP | 0.729 | Share focus on seduction, resulting in high terminology overlap |
| MGTOW $\leftrightarrow$ MRA | 0.654 | MGTOW was originally founded by members of MRA |
| MGTOW $\leftrightarrow$ PuA | 0.282 | Ideological conflict (pursuing vs. avoiding women) |
| Incels $\leftrightarrow$ All others | $<0$ (min -0.240) | Highly unique vocabulary, independent origin |

These findings closely align with literature from sociology, validating the effectiveness of the proposed approach.

## Highlights & Insights

- **First to integrate social and temporal dimensions into lexicon induction**: While existing methods only examine linguistic features, LISTN simultaneously models "who uses what words" and "who interacts with whom."
- **Highly interpretable embedding space**: In addition to inducing lexicons, it analyzes temporal stability of words and subgroup-specific specialization, delivering insights of sociological value.
- **Rigorous test set construction**: Combining existing lexicons with expert annotation yields a Cohen's Kappa of 0.726.
- **Directly applicable outputs**: Releases 455 new manosphere terms along with their sub-community relevance scores.

## Limitations & Future Work

1. **Single score per word**: Does not handle polysemy or dogwhistles (words with different meanings inside versus outside the group).
2. **Evaluation restricted to single tokens**: Excludes multi-word expressions (such as "all women are like that" $\rightarrow$ AWALT).
3. **2018 training data**: In-group language evolves rapidly, making the timeliness of the lexicon limited.
4. **Restricted access to Reddit data**: Hard to obtain updated data to validate generalizability.
5. **Scalable future directions**: Combining with LLMs for dogwhistle detection; studying multilingual in-group languages; integrating with NPMI methods (as both target different regions of the lexical spectrum).

## Related Work & Insights

- **Lucy & Bamman (2021)**: Use BERT embeddings and statistical features for lexicon induction, without considering temporal and social factors.
- **Farrell et al. (2020)**: Use topic models and word2vec, identifying differences in terminology usage across subgroups.
- **Danescu-Niculescu-Mizil et al. (2013)**: Uncover that failure to adopt lexicon innovations predicts a user leaving the community.
- **Stewart & Eisenstein (2018)**: Study the linguistic and social factors governing the spread of language innovation.
- **Insights**: In-group language offers a unique window into social dynamics—language patterns encode group structure, cohesion, and evolutionary direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to incorporate socio-temporal context into lexicon induction, with theoretically grounded system design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Investigates 6 method variants against multiple baselines, with an expert-annotated test set and embedding analysis; limited, however, to evaluation on a single community (manosphere).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clearly motivated, engages deeply with sociological literature, and handles ethical considerations responsibly.
- **Value**: ⭐⭐⭐⭐ — The methodology is generalizable (not limited to specific groups or languages), the resulting dictionary holds immediate research value, and the embedding analysis reveals meaningful sociolinguistic insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)

</div>

<!-- RELATED:END -->
