---
title: >-
  [Paper Note] Croppable Knowledge Graph Embedding
description: >-
  [ACL 2025][Graph Learning][knowledge graph embedding] Proposes the MED framework to train "croppable" knowledge graph embeddings—optimizing 64 sub-models of different dimensions (sharing embedding prefixes) simultaneously in a single training run. Through mutual learning, evolutionary improvement, and dynamic loss weights, sub-models of each dimension can be directly cropped and used, outperforming independent training and distillation methods while being 10 times faster to t…
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "knowledge graph embedding"
  - "croppable model"
  - "mutual learning"
  - "multi-dimensional"
  - "parameter sharing"
date: 2026-05-08
content_hash: 09a0944e5d722403
---

# Croppable Knowledge Graph Embedding

**Conference**: ACL 2025  
**arXiv**: [2407.02779](https://arxiv.org/abs/2407.02779)  
**Code**: Implemented based on the OpenKE framework  
**Area**: Graph Learning  
**Keywords**: knowledge graph embedding, croppable model, mutual learning, multi-dimensional, parameter sharing

## TL;DR
Proposes the MED framework to train "croppable" knowledge graph embeddings—optimizing 64 sub-models of different dimensions (sharing embedding prefixes) simultaneously in a single training run. Through mutual learning, evolutionary improvement, and dynamic loss weights, sub-models of each dimension can be directly cropped and used, outperforming independent training and distillation methods while being 10 times faster to train.

## Background & Motivation
**Background**: KGE maps entities/relations to vector spaces for tasks such as link prediction. Different deployment scenarios have varying requirements for embedding dimensions—a server can use 500 dimensions, while mobile devices can only use 10 dimensions.

**Limitations of Prior Work**: A new model must be trained from scratch or distilled from a high-dimensional teacher model for each required new dimension. Having 64 dimensional requirements equals 64 independent training runs, which is extremely costly (e.g., training 64 models directly on WN18RR takes 141 hours).

**Key Challenge**: The demand for both high-dimensional and low-dimensional models coexists. However, independent training does not share information, and distillation methods (e.g., DualDE), although guiding the low-dimensional model with the high-dimensional one, require even longer training times (240 hours).

**Goal**: To obtain a croppable model in a single training run, where cropping the first $d_i$ dimensions of the embedding allows it to be used directly as a $d_i$-dimensional KGE.

**Key Insight**: Drawing inspiration from slimmable networks in computer vision, the approach allows sub-models of different dimensions to share parameter prefixes, and optimizes each dimension through mutual learning and curriculum training.

**Core Idea**: The first $d_1$ dimensions of the embedding vector represent the smallest sub-model, the first $d_2$ dimensions represent the second smallest, ..., and the first $d_n$ dimensions represent the largest sub-model. A single set of parameters is jointly optimized through the three mechanisms of MED.

## Method

### Overall Architecture
The $n=64$ sub-models $M_1,...,M_n$ share embedding parameter prefixes ($d_1=10, d_2=20,...,d_{64}=640$). For each triplet $(h,r,t)$, the first $d_i$ dimensions of the embedding are used to compute the score $s^i_{(h,r,t)}$ in sub-model $M_i$. All sub-models are optimized simultaneously during training, with three mechanisms ensuring the performance of each dimension.

### Key Designs

1. **Mutual Learning**:

    - **Function**: Enables adjacent dimensional sub-models to act as teacher and student to each other, transferring knowledge bidirectionally.
    - **Mechanism**: $L_{ML}^{i-1,i} = \sum_{(h,r,t)} d_\delta(s^{i-1}_{(h,r,t)}, s^i_{(h,r,t)})$, using Huber loss ($\delta=1$) to measure the difference in scores between adjacent sub-models.
    - **Design Motivation**: The low-dimensional model learns better scoring patterns from the high-dimensional one (knowledge distillation effect), while the high-dimensional model learns to maintain the stability of low-dimensional features from the low-dimensional one (preventing performance degradation when high dimensions overwrite low dimensions).

2. **Evolutionary Improvement**:

    - **Function**: Focuses high-dimensional sub-models on hard samples poorly predicted by low-dimensional sub-models.
    - **Mechanism**: For positive samples, triplets with lower scores from the low-dimensional model receive higher weights $pos^i \propto \exp(w_1 / s^{i-1})$; for negative samples, those with higher scores (false negatives/hard negatives) receive higher weights $neg^i \propto \exp(w_2 \cdot s^{i-1})$.
    - **Design Motivation**: High-dimensional sub-models should not merely repeat what has already been well handled by low-dimensional ones, but should focus on hard triplets that are unsolvable by low-dimensional models, achieving "evolutionary" capacity enhancement.

3. **Dynamic Loss Weight**:

    - **Function**: Adaptively balances the training focus of sub-models of different dimensions.
    - **Mechanism**: Total loss $L = \sum_{i=2}^n L_{ML}^{i-1,i} + \sum_{i=1}^n \exp(w_3 \cdot d_i / d_n) \cdot L_{EI}^i$.
    - **Design Motivation**: Low-dimensional models rely more on learning from high-dimensional soft labels (dominated by mutual learning), whereas high-dimensional models need to focus more on hard labels and hard samples (dominated by evolutionary improvement).

### Loss & Training
Adam optimizer + linear learning rate decay, batch size 1024, 64 negative samples generated per positive sample, up to 3000 epochs + early stopping. $w_1, w_2, w_3$ are learnable parameters (initialized to 1). Applicable to any KGE method with a triplet scoring function.

## Key Experimental Results

### Main Results (RotatE on WN18RR)

| Dimension | MED MRR | MED H@10 | DualDE MRR | DualDE H@10 | DT MRR | DT H@10 |
|------|---------|----------|------------|-------------|--------|---------|
| 10d | **0.324** | **0.469** | 0.179 | 0.440 | 0.131 | 0.303 |
| 40d | **0.466** | **0.561** | 0.441 | 0.525 | 0.447 | 0.528 |
| 160d | **0.471** | **0.574** | 0.465 | 0.560 | 0.470 | 0.575 |
| 640d | 0.476 | 0.574 | 0.476 | 0.576 | **0.508** | **0.612** |

### Ablation Study (Training Efficiency)

| Method | WN18RR Training Time | Relative to MED | FB15K237 Training Time | Relative to MED |
|------|----------------|---------|------------------|---------|
| **MED** | **12.7h** | **1.0×** | **35.5h** | **1.0×** |
| Directly training 64 models | 141.0h | 11.1× | 381.0h | 10.7× |
| DualDE | 240.0h | 18.9× | - | - |
| BKD | 163.0h | 12.8× | - | - |

### Key Findings
- The MRR of low-dimensional (10d) MED is 81% higher than DualDE (RotatE WN18RR), because the information loss in the distillation process of DualDE is significant.
- High-dimensional (640d) MED performs slightly lower than independent training (0.476 vs 0.508), which is consistent with the trade-off expected in multi-objective optimization.
- Training speed is 10 times faster than direct training and 19 times faster than DualDE.
- Under a real e-commerce scenario (Taobao SKG, 7M users / 50.7M triplets): mobile-side 10d product recommendation ndcg@5=0.422 > DT 0.344 > DualDE 0.404.
- Extension to BERT: BERT-MED outperforms BERT-HAT on GLUE by an average of 16-22%, demonstrating the generalizability of the framework.
- When the number of sub-models is reduced from 64 to 4, the 10d MRR only drops from 0.324 to 0.319, but the training time decreases from 12.7h to 3.3h.

## Highlights & Insights
- **The concept of "croppable"** is highly practical: train once, crop the dimensions as needed for direct use, perfectly matching deployment requirements on heterogeneous devices. This idea can be transferred to any embedding-based model.
- **The "hard-sample focusing" design** of the evolutionary improvement mechanism is ingenious: high-dimensional models do not repeat what has been resolved by low-dimensional ones, but instead specialize in hard problems, making the increase in dimensions truly bring about incremental capabilities.
- **The bidirectionality of mutual learning** is key: it is not just high-to-low distillation; low-to-high feedback helps high-dimensional models maintain structural stability of low-dimensional features, which is why MED outperforms pure distillation methods.

## Limitations & Future Work
- Experiments only cover 4 KGE methods (TransE, SimplE, RotatE, PairRE), while stronger methods (such as CompGCN) remain unverified.
- The performance of the maximum dimension (640d) is slightly lower than independent training, which may be insufficient for scenarios requiring extremely high precision.
- Training overhead grows linearly as the number of sub-models increases, which still poses scalability issues when there is a massive range of dimensional requirements.
- The BERT extension experiments only evaluated GLUE, without validation on broader NLP tasks.

## Related Work & Insights
- **vs DualDE (Zhu et al. 2023)**: DualDE uses bidirectional knowledge distillation but requires training a teacher first before distilling the student, resulting in a training time 19 times that of MED and poor low-dimensional performance.
- **vs Slimmable Networks (Yu et al. 2019)**: Inspired by the slimmable network concept in CV, but specifically tailored for the triplet scoring of KGE.
- This "croppable training" paradigm can be migrated to word embeddings, sentence embeddings, and even elastic deployment of LLM embedding layers.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of "croppable KGE" is novel, and the three MED mechanisms are reasonably designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 KGE methods × 2 datasets + efficiency analysis + industrial application + BERT extension.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and comprehensive experiments.
- Value: ⭐⭐⭐⭐ Solves practical deployment pain points, presenting high engineering value.
   - Distillation occurs only between adjacent sub-models—a smaller dimension gap yields better distillation results.

2. **Evolutionary Improvement**:
    - High-dimensional sub-models need to master triplets that low-dimensional sub-models fail to predict correctly.
    - Positive examples: $M_{i-1}$ has low score $\rightarrow$ $M_i$ has high optimization weight ($pos \propto \exp(w_1 / s^{i-1})$).
    - Negative examples: $M_{i-1}$ has high score $\rightarrow$ $M_i$ has high optimization weight ($neg \propto \exp(w_2 \cdot s^{i-1})$).
    - Forces high-dimensional models to focus on the "blind spots" of low-dimensional models.

3. **Dynamic Loss Weight**:
    - Low-dimensional sub-models rely more on mutual learning (soft labels), whereas high-dimensional ones rely more on evolutionary improvement (hard labels).
    - Loss: $L = \sum L_{ML} + \sum \exp(w_3 \cdot d_i / d_n) \cdot L_{EI}^i$, where the evolutionary improvement weight for high-dimensional sub-models grows exponentially.

## Key Experimental Results

| Evaluation | Results | Description |
|------|------|------|
| 4 KGE × 4 datasets | MED sub-models across all dimensions perform well | Single training run, direct multi-dimensional usage |
| vs DualDE (SOTA distillation) | Low-dimensional sub-models outperform | Significantly higher training efficiency |
| vs Independent training | Comparable or superior performance | Drastically reduced training cost |
| Real-world large-scale KG | Effective | Practical validation |
| BERT + GLUE Extension | Effective | Also applicable to non-KGE |

### Key Findings
- The three mechanisms are indispensable—mutual learning improves low dimensions, evolutionary improvement targets blind spots, and dynamic weights balance the global objectives.
- Distilling between neighbors with smaller dimensional gaps yields better performance than distilling across far distances, validating the design of learning only between adjacent sub-models.

## Highlights & Insights
- **The concept of "croppable"** is extremely practical: train once $\rightarrow$ crop as needed $\rightarrow$ deploy directly, dramatically reducing maintenance costs for multi-device KGE.
- **The "focus on blind spots" mechanism** of evolutionary improvement is ingenious: high dimensions do not merely repeat what low dimensions can already do, but specialize in what low dimensions cannot.
- **BERT extension validation** suggests that the MED framework is not limited to KGE and possesses general applicability.

## Limitations & Future Work
- The number of sub-models $n$ and each dimension $d_i$ must be preset and cannot be dynamically adjusted.
- Only validated on medium-scale KGs; the effectiveness on ultra-large-scale KGs remains to be tested.

## Rating
- Novelty: ⭐⭐⭐⭐ "Croppable" KGE concept is novel; the three mechanisms are fully designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 4 datasets + real-world scenarios + BERT extension.
- Writing Quality: ⭐⭐⭐⭐ Formula derivations are clear.
- Value: ⭐⭐⭐⭐ Significantly reduces the training and deployment costs of multi-dimensional KGE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Mutual Information Perspective on Knowledge Graph Embedding](a_mutual_information_perspective_on_knowledge_graph_embedding.md)
- [\[ACL 2025\] RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](rscf_relationsemantics_consistent_filter_for_entity.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[ACL 2025\] Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)

</div>

<!-- RELATED:END -->
