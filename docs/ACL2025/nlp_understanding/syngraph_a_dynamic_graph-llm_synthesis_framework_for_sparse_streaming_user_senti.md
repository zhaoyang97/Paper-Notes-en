---
title: >-
  [Paper Note] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis
description: >-
  [ACL 2025][NLP Understanding][Streaming Sentiment Analysis] This paper proposes the SynGraph framework, which categorizes sparse users on a continuous-time dynamic graph into three classes (mid-tail, long-tail, and extreme), and leverages LLMs to synthesize augmented data tailored to different sparsity levels (combining local-global graph understanding, high-order relationships, and profile generation) to effectively alleviate data sparsity issues in streaming review sentimen…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Streaming Sentiment Analysis"
  - "Data Sparsity"
  - "Dynamic Graphs"
  - "LLM Data Augmentation"
  - "User Modeling"
date: 2026-05-08
content_hash: 3537e01671373c50
---

# SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis

**Conference**: ACL 2025  
**arXiv**: [2503.04619](https://arxiv.org/abs/2503.04619)  
**Authors**: Xin Zhang (Manchester), Qiyu Wei (Manchester), Yingjie Zhu (HIT), Linhai Zhang (KCL), Deyu Zhou (SEU), Sophia Ananiadou (Manchester)
**Code**: Not released  
**Area**: NLP Understanding  
**Keywords**: Streaming Sentiment Analysis, Data Sparsity, Dynamic Graphs, LLM Data Augmentation, User Modeling

## TL;DR

This paper proposes the SynGraph framework, which categorizes sparse users on a continuous-time dynamic graph into three classes (mid-tail, long-tail, and extreme), and leverages LLMs to synthesize augmented data tailored to different sparsity levels (combining local-global graph understanding, high-order relationships, and profile generation) to effectively alleviate data sparsity issues in streaming review sentiment analysis.

## Background & Motivation

### Background
User reviews on e-commerce platforms contain rich information about sentiment dynamics—for instance, a user's sentiment score for a certain category of products may gradually shift from positive to neutral over time. Traditional sentiment analysis methods treat each review as an independent sample statically, ignoring sequential dependencies in the review stream. Streaming review sentiment analysis is proposed precisely to model such temporal evolution and predict future sentiment.

### Limitations of Prior Work
- **Static methods ignore temporal dynamics**: Traditional sentiment classification models do not consider the temporal dependencies between user reviews.
- **Severe data sparsity**: In streaming scenarios, sparsity manifests in three forms—temporal sparsity (large intervals between user reviews), spatial sparsity (few social connections for users), and complex composite sparsity of both.
- **Graph structures make sparsity more complex**: Prior works use graph structures to supplement sparse user information, but classifying and handling sparse users becomes more difficult in streaming scenarios.
- **Limited existing LLM augmentation methods**: Existing LLM-based augmentation methods for recommendation systems mostly rely only on first-order neighbor information and overlook higher-order relations and temporal dynamics.

### Design Motivation
To design a unified framework that adopts differentiated strategies for users with varying levels of sparsity, and jointly utilizes dynamic graph structures and LLM generation capabilities to synthesize high-quality supplementary data, thereby improving streaming review sentiment prediction.

## Method

### Overall Architecture
SynGraph is a three-stage framework:
1. **Sparse User Categorization**: Users are categorized into mid-tail, long-tail, and extreme categories based on their temporal and structural characteristics.
2. **LLM-Augmented Data Synthesis**: Tailored synthetic data is generated using LLMs combined with graph structural information for different sparsity categories.
3. **Category Interpolation**: Synthetic data is inserted into blank time intervals of users to ensure that each user has at least 10 interaction records.

The underlying modeling is based on Continuous-Time Dynamic Graphs (CTDG), where users and items are nodes, and reviews are timestamped directed edges. Each edge carries the review text and a sentiment score.

### Key Designs

#### 1. Three-Level Categorization of Sparse Users

Based on the temporal activity and graph structural connectivity of users, they are classified into three categories:

- **Mid-tail users**: Users who maintain some degree of activity within specific periods but exhibit highly fluctuating review frequencies. This instability is captured by calculating the interaction variance $\sigma^2_{I(u_m)}$, where high variance indicates that user engagement fluctuates over time.
- **Long-tail users**: Users with a small total volume of reviews and sparse first-order neighbors (direct connections) but who typically possess more second-order connections. The first-order connection strength $C_1(u_l)$ is defined as the sum of weights of direct neighbors, which is small, while the second-order connection strength $C_2(u_l)$ is relatively large.
- **Extreme users**: Users who are extremely sparse in terms of both review activity and graph connectivity, acting almost as isolated nodes. The total connectivity is $C(u_e) \approx 0$.

This categorization allows subsequent data synthesis to tailor strategies for different user types—mid-tail users have enough historical reviews for LLMs to generate user profiles, long-tail users require indirect information from second-order neighbors, and extreme users must rely on globally high-rated items as proxies.

#### 2. Three Components of LLM-Augmented Data Synthesis

The framework supports data synthesis through three core components:

**(1) Local-Global Graph Understanding**: Capturing recent direct interactions from local neighbors and long-term trends and overall network context from global neighbors. The LLM enriches the user profile based on information from both levels.

**(2) High-Order Relationship Understanding**: Instead of relying solely on first-order connections, the framework extracts second-order neighbor relations $\mathcal{N}^2(u) = (\bigcup_{v \in \mathcal{N}(u)} \mathcal{N}(v)) \setminus \{u\}$ to compensate for sparse direct connections via indirect information propagation.

**(3) Profile Generation**: Utilizing LLMs and specific prompts to generate synthetic profiles for users and items from reviews: $\mathcal{P}(x) = \text{LLM}(\mathcal{R}(x), P_x)$, where user prompt $P_u$, product prompt $P_{\text{prod}}$, and data synthesis prompt $P_{\text{data}}$ are applied respectively.

These three components are combined differentially across different sparsity categories:
- **Mid-tail Synthesis**: Direct user profile generation from historical reviews + second-order product profiles $\rightarrow$ LLM synthesizes new reviews.
- **Long-tail Synthesis**: Combining local and global second-order neighbor information + selected reviews to generate user profiles + second-order product profiles $\rightarrow$ LLM synthesis.
- **Extreme Synthesis**: Using predefined user profiles + high-rated popular products as proxies $\rightarrow$ LLM synthesizes pseudo-reviews.

#### 3. Interpolation Mechanism

The interpolation factor $I_f(u) = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}_{\{I_t(u)=0\}}$ is used to quantify the degree of missing user data across various time windows. Synthetic data is inserted into these missing positions to ensure that every user maintains a minimum data availability threshold of 10 records.

## Key Experimental Results

### Experimental Setup
- **Datasets**: Three small-scale subsets of the Amazon dataset—Magazine_Subscriptions, Appliances, and Gift_Cards.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, MSE, RMSE, MAE.
- **Baselines**: BiLSTM+Att, BERT-Sequence, CHIM, IUPC, NGSAM, DC-DGNN.

### Main Results

#### Table 1: Main Results of RMSE Reduction Rates (Asterisk * indicates enhancement with SynGraph)

| Model | Magazine | Appliances | Gift_Cards |
|------|----------|------------|------------|
| BiLSTM+Att → +SynGraph | ↓3.93% | ↓48.61% | ↓0.12% |
| BERT-Sequence → +SynGraph | ↓36.80% | ↓4.36% | ↓2.57% |
| NGSAM → +SynGraph | ↓14.38% | ↓14.94% | ↓21.73% |
| CHIM → +SynGraph | ↑10.52% | ↓23.92% | ↑40.89% |
| IUPC → +SynGraph | ↓22.16% | ↓4.36% | — |
| DC-DGNN → +SynGraph | **↓26.42%** | **↓20.22%** | **↓38.88%** |

In 15 out of 18 experimental groups, the performance is significantly improved after enhancement with SynGraph. DC-DGNN consistently performs the best before and after augmentation, highlighting the importance of dynamic user-item modeling.

### Ablation Study

#### Table 2: Ablation Study (RMSE of DC-DGNN on Three Datasets)

| Variant | Magazine | Appliances | Gift_Cards |
|------|----------|------------|------------|
| DC-DGNN* (Three-Class Joint) | **0.6485** | **0.8165** | **0.3383** |
| DC-DGNN-M (Only mid-tail) | 0.6551 | 1.0235 | 0.3805 |
| DC-DGNN-L (Only long-tail) | 0.7324 | 0.7868 | 0.5535 |
| DC-DGNN-E (Only extreme) | 0.6385 | 1.0235 | — |

Combining the synthetic data of all three categories generally yields the best results; using a single category in isolation can sometimes introduce noise or exacerbate label imbalance.

### Quality Evaluation of GPT-4 Synthetic Data
Evaluated across four dimensions—Language Style Similarity (LSS), Rating Habit Similarity (RHS), Sentiment Similarity (SS), and Aspect Similarity (AS): mid-tail users achieve the highest synthesis quality, while extreme users exhibit lower LSS and RHS but maintain stable SS and AS, demonstrating that LLMs can effectively maintain product-level semantic coherence.

## Key Findings

1. **Effective across 15/18 configurations**: SynGraph augmentation improves performance on the vast majority of model-dataset combinations, validating the generalizability of the framework.
2. **DC-DGNN as the best fit**: DC-DGNN, designed specifically for continuous dynamic graphs, benefits the most when paired with SynGraph (reducing RMSE by 26%, 20%, and 39% across the three datasets), underscoring that dynamic graph modeling is key for streaming sentiment analysis.
3. **Joint three-class synthesis outperforms single class**: Universally, combining synthetic data from all three sparsity categories yields the best performance, though it may introduce extra noise on extremely small-scale datasets.
4. **Inadequate lexical diversity in LLM synthesis**: The lexical richness of the synthesized texts is consistently lower than that of the original reviews, indicating a tendency toward lexical homogenization in LLMs.

## Highlights & Insights

- **The three-stage design of categorization-synthesis-interpolation is highly practical**: It decomposes the sparsity problem into actionable sub-problems, offering customized augmentation schemes for each user category rather than a one-size-fits-all approach.
- **The fusion idea of graph structure + LLM is inspiring**: Instead of simply asking the LLM to generate data, the framework guides the LLM to understand local-global graph structures and high-order relations, offering a new paradigm for graph-augmented learning.
- **The handling strategy for extremely sparse users is exemplary**: For users with almost zero data, using popular products as proxies to achieve a "cold start" is straightforward and effective.

## Limitations & Future Work

- **Dataset scale is too small**: The three Amazon sub-datasets are extremely small (fewer than 500 users in total), leaving the scalability in large-scale real-world scenarios unknown.
- **Coarse random neighbor sampling strategy**: Random sampling is adopted when selecting neighbors, and the sampling variance on large-scale graphs might significantly affect performance.
- **LLM hallucination risk**: Synthetic data may not faithfully reflect the ground-truth distribution, and the paper does not design a dedicated quality-filtering mechanism.
- **LLM's understanding of graph structures is opaque**: The paper does not analyze how the LLM distinguishes and utilizes local vs. global graph information.
- **Performance degradation in a few configurations**: CHIM's performance dropped after augmentation on Magazine and Gift_Cards (with RMSE increasing by 10.52% and 40.89%, respectively), exposing the vulnerability of the method to specific model-data combinations.
- **Severe label imbalance**: In the Gift_Cards dataset, 5-star ratings account for over 92% of the data; synthetic data may exacerbate this imbalance.

## Related Work & Insights

- **Mitigating Data Sparsity**: Traditional methods handle sparsity via user attributes, social networks, cross-domain transfer, or meta-learning, but all of these are static solutions.
- **LLM-Augmented Recommendation**: LLMRec (Wei et al. 2023) uses first-order neighbor information for augmentation, and Sun et al. 2023 infer user profiles from text; however, neither considers high-order relationships and temporal dynamics.
- **Streaming Sentiment Analysis**: SentiStream (Wu et al. 2023) processes data streams but does not address sparsity; DC-DGNN (Zhang et al. 2023) models continuous dynamic graphs but lacks data augmentation.
- **Insights**: Embedding LLMs as data augmentation engines within graph learning pipelines is a promising direction, especially in user cold-start and long-tail distribution scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The integration of three-level sparse user categorization with LLM-augmented dynamic graphs is a novel combination, though individual components are not entirely pioneering.
- Experimental Thoroughness: ⭐⭐⭐ — Evaluation against 6 baselines, along with ablation studies and quality assessments, is relatively comprehensive, but the datasets are too small (users < 500), limiting reliability.
- Writing Quality: ⭐⭐⭐⭐ — The framework description is clear, the mathematical formulations are complete, and the illustrations are intuitive.
- Value: ⭐⭐⭐ — The framework concept is inspiring, but small-dataset experiments and performance degradation in some configurations weaken its persuasiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)
- [\[ACL 2025\] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)

</div>

<!-- RELATED:END -->
