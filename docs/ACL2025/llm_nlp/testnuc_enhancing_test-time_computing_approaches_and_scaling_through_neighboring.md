---
title: >-
  [Paper Note] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency
description: >-
  [ACL 2025][LLM (Other)][Test-time computing] TestNUC proposes a linearly scaling test-time inference enhancement method. By retrieving nearest neighbor unlabeled samples for a test instance, it prompts LLMs to predict both the test sample and its neighbors, then aggregates these predictions via weighted majority voting to consistently improve classification accuracy.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Test-time computing"
  - "neighborhood consistency"
  - "unlabeled data"
  - "majority voting"
  - "LLM inference scaling"
date: 2026-05-08
content_hash: f95c5f51fb0bda01
---

# TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency

**Conference**: ACL 2025  
**arXiv**: [2502.19163](https://arxiv.org/abs/2502.19163)  
**Code**: Yes ([https://github.com/HenryPengZou/TestNUC](https://github.com/HenryPengZou/TestNUC))  
**Area**: Others  
**Keywords**: Test-time computing, neighborhood consistency, unlabeled data, majority voting, LLM inference scaling

## TL;DR

TestNUC proposes a linearly scaling test-time inference enhancement method. By retrieving nearest neighbor unlabeled samples for a test instance, it prompts LLMs to predict both the test sample and its neighbors, then aggregates these predictions via weighted majority voting to consistently improve classification accuracy.

## Background & Motivation

Test-time computing has emerged as a popular direction by allocating more computational resources during the inference phase to boost LLM performance. Existing strategies are categorized into two types:

**Input-level** (e.g., few-shot ICL): increasing the prompt length $\to$ computational cost scales **quadratically** with the number of tokens.

**Output-level** (e.g., self-consistency, best-of-N): sampling multiple answers and aggregating them $\to$ ignoring the extensive availability of **unlabeled data** in real-world scenarios.

**Core Problem**: **How to efficiently leverage unlabeled data to enhance test-time inference?**

The authors identify a phenomenon of "local consistency in embedding space": semantically similar instances are highly likely to share the same label. Preliminary analysis demonstrates that among the $K=20$ nearest neighbors, even in the worst-case scenario (GoEmotion with 150 classes), the purity reaches $\sim 30\%$, while most datasets exhibit much higher purity. Aggregating ground-truth labels from the neighborhood via majority voting yields highly accurate and stable predictions.

## Method

### Overall Architecture

TestNUC consists of two steps:
1. **Neighbor Retrieval**: Retrieving top-$K$ nearest neighbor unlabeled samples of the test instance based on embedding similarity.
2. **Collaborative Prediction**: Prompts the LLM to make predictions for both the test sample and its $K$ neighbors, then derives the final answer through the designed aggregation strategies.

### Key Designs

1. **Preliminary Analysis**

    - Defining neighborhood purity as $\phi(\mathcal{N}) = \frac{1}{KN} \sum_{i=1}^N \sum_{j \in \mathcal{N}} \mathbf{1}(y_i = y_j)$
    - Empirical findings: nearest neighbor samples exhibit high label consistency, and majority voting accuracy remains stable as $K$ increases.
    - Weighted voting further improves stability under large $K$.

2. **Three Aggregation Strategies**

    - **Naive Majority Voting**: directly takes the most frequent class among the $K$ predictions.
    - **Weighted Majority Voting**: votes weighted by cosine similarity to mitigate the influence of noisy distantly-related neighbors.
    - **Filtered Weighted Majority Voting** (Full Version): additionally leverages the verbalized confidence of the LLM to filter out low-quality predictions.
        - For each neighbor, the LLM outputs both a prediction and a confidence score.
        - Only predictions with confidence $\ge \text{threshold } \theta$ participate in the voting.

3. **Seamless Integration with Existing Methods**

    - Integration with **Self-Consistency**: performs self-consistency on each neighbor before aggregation.
    - Integration with **TopK-ICL**: enhances individual neighbor predictions with ICL first, then aggregates.
    - Integration with **Best-of-N**: performs best-of-N on top of the aggregated results of TestNUC.
    - All integrations bring further performance gains.

4. **Computational Complexity Analysis**

    - Embedding precomputation cost is $\mathcal{O}(N)$ (done offline), and retrieval cost is $\mathcal{O}(N)$.
    - LLM inference cost is $\mathcal{O}(K)$, which is comparable to self-consistency ($\mathcal{O}(M)$).
    - Overall scaling is linear, which is significantly better than the quadratic scaling of ICL.

## Key Experimental Results

### Main Results — 4 LLMs × 8 Datasets (Table 1 Summary)

| Model | Method | Average Accuracy |
|------|------|-----------|
| GPT-4o-mini | Standard Prompting | 0.613 |
| GPT-4o-mini | Self-Consistency | 0.625 |
| GPT-4o-mini | TestNUC | 0.660 |
| GPT-4o-mini | TestNUC† (K=50) | **0.676** |
| Llama-3.1-8B | Standard Prompting | 0.572 |
| Llama-3.1-8B | TestNUC† | **0.652** |
| GPT-4o | Standard Prompting | 0.715 |
| GPT-4o | TestNUC† | **0.754** |

### Integration with Existing Test-Time Methods (Table 2 Summary)

| Base Method | +TestNUC Gain |
|----------|-------------|
| KNN-ICL | +7.51% |
| KNN-ICL-P | +5.98% |
| Self-Consistency | **+9.56%** |
| Best-of-N | +6.24% |

### Ablation Study

| Dimension | Conclusion |
|------|------|
| Sensitivity to $K$ | Performance improves steadily for $K=10\text{-}50$, demonstrating robustness to $K$. |
| Embedding Models | Consistent effectiveness across various embedding models (e.g., SFR, GTE, NV-Embed). |
| Size of Unlabeled Data | Performance improves monotonically as data volume increases; more data leads to better results. |
| Aggregation Strategy | Filtered Weighted Voting > Weighted Voting > Naive Voting |

### Key Findings

1. **TestNUC consistently outperforms baselines across all LLMs and datasets**, achieving an average improvement of 4-8 percentage points.
2. **Perfectly complementary to existing methods**: all integrations yield additional gains, with the Self-Consistency integration showing the largest improvement (+9.56%).
3. **Scales linearly with unlabeled data volume**: more data yields better performance, making it highly suitable for data-rich practical scenarios.
4. **Insensitive to embedding models**: embedding models of different sizes and sources all perform well.
5. The largest improvements are observed in intent detection (BANKING/CLINC), while the smallest are in emotion detection (GoEmotion)—consistent with the neighborhood purity analysis.

## Highlights & Insights

1. **Simple and elegant formulation**: The intuition that "similar samples should share the same label" is simple yet highly effective, backed by theoretical support (neighborhood purity analysis).
2. **Plug-and-play**: Orthogonal to existing test-time computing methods, allowing direct integration.
3. **Training-free**: Leverages off-the-shelf embedding models and LLMs without any extra training.
4. **High practicality**: Linear scaling + robustness to hyperparameters + cross-model generalizability = easy deployment.

## Limitations & Future Work

- Requires a certain amount of task-related unlabeled data, making it unsuitable for absolute cold-start scenarios.
- Introduces $K$ additional LLM call overheads, potentially impacting latency-sensitive scenarios when $K$ is large.
- Validation is restricted to classification tasks; its applicability to generative tasks (e.g., text summarization, translation) remains unexplored.
- Neighborhood purity drops rapidly in fine-grained tasks (e.g., 150 classes), limiting its efficacy in extremely fine-grained classification.
- More complex aggregation strategies (e.g., learning-based aggregation) have not yet been evaluated.

## Related Work & Insights

- kNN-LM (Khandelwal et al., 2020) utilizes neighboring samples to improve language model generalization. While sharing a similar conception, it operates at the generation level.
- Self-Consistency (Wang et al., 2023b) improves inference via output sampling and aggregation, whereas TestNUC complements it from the input space.
- The paradigm of leveraging unlabeled data in semi-supervised learning (e.g., FixMatch) aligns with the neighborhood consistency assumption in TestNUC.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically leverage the neighborhood consistency of unlabeled data within a test-time computing framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 LLMs $\times$ 8 datasets $\times$ multiple integration methods $\times$ extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Preliminary analyses (purity, majority voting accuracy) establish a solid motivational foundation for the method.
- **Value**: ⭐⭐⭐⭐ — Simple yet effective inference scaling method with significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)
- [\[ACL 2025\] Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)
- [\[CVPR 2025\] Test-Time Visual In-Context Tuning](../../CVPR2025/llm_nlp/test-time_visual_in-context_tuning.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)

</div>

<!-- RELATED:END -->
