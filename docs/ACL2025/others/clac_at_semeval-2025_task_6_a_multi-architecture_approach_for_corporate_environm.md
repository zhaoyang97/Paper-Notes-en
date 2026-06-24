---
title: >-
  [Paper Note] CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification
description: >-
  [ACL 2025 (SemEval Workshop)][ESG reporting] This paper explores three progressive model architectures for the SemEval-25 Task 6 (PromiseEval) on corporate ESG report promise verification: an ESG-BERT baseline, a linguistic feature-enhanced version, and a joint sub-task model integrating attention pooling and multi-objective learning. The proposed approach slightly outperforms the official baseline on the private leaderboard (0.5268 vs 0.5227)…
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "ESG reporting"
  - "promise verification"
  - "multi-task learning"
  - "attention pooling"
  - "linguistic feature engineering"
date: 2026-05-08
content_hash: fad2e3def877e8ec
---

# CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification

**Conference**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2505.23538](https://arxiv.org/abs/2505.23538)  
**Code**: [https://github.com/CLaC-Lab/SemEval-2025-Task6](https://github.com/CLaC-Lab/SemEval-2025-Task6)  
**Area**: Other  
**Keywords**: ESG reporting, promise verification, multi-task learning, attention pooling, linguistic feature engineering

## TL;DR

This paper explores three progressive model architectures for the SemEval-25 Task 6 (PromiseEval) on corporate ESG report promise verification: an ESG-BERT baseline, a linguistic feature-enhanced version, and a joint sub-task model integrating attention pooling and multi-objective learning. The proposed approach slightly outperforms the official baseline on the private leaderboard (0.5268 vs 0.5227), validating the effectiveness of linguistic feature engineering and multi-task learning in ESG promise verification.

## Background & Motivation

Verifying promises in corporate ESG (Environmental, Social, and Governance) reports is an increasingly important yet highly challenging task:

**Background**: ESG reports released by corporations contain numerous promises regarding environmental protection and social responsibility, which profoundly affect stakeholder trust and corporate reputation. However, the complexity and sheer volume of these promises make manual verification extremely difficult.

**Task Decomposition**: PromiseEval decomposes promise verification into four sub-tasks:
   - Promise identification (whether it contains a promise)
   - Supporting evidence evaluation (whether concrete evidence exists)
   - Clarity evaluation (clear / vague / misleading)
   - Verification time classification (within 2 years / 2-5 years / more than 5 years / other)

**Key Challenge**:
   - **Extremely scarce data**: The English dataset contains only about 400 training instances.
   - **Class imbalance**: The class distributions across the four sub-tasks are highly skewed.
   - **Domain specificity**: The linguistic patterns in ESG reports (e.g., promise formulations, evasive phrasing) require specialized handling.

**Design Motivation**: In scenarios of data scarcity and class imbalance, relying solely on pre-trained models is insufficient. Integrating feature engineering with domain knowledge and multi-task learning for cross-task knowledge sharing is required to enhance performance.

## Method

### Overall Architecture

Three progressive model architectures are explored, with each increasing in complexity over the previous:
- **Model 1** (Baseline): ESG-BERT + four independent classification heads
- **Model 2** (Feature-Enhanced): ESG-BERT + linguistic feature tags custom-designed for each sub-task
- **Model 3** (Combined Model): DeBERTa-v3-large + attention pooling + multi-objective learning (sub-tasks 1 and 2 only)

### Key Designs

1. **ESG-BERT Baseline Model (Model 1)**:

    - Utilizes a pre-trained ESG-BERT model to leverage domain-specific ESG knowledge.
    - Trains an independent model for each of the four sub-tasks.
    - **Freezing Strategy**: Fine-tunes only the last two transformer layers and the classification head, while freezing remaining parameters. With only 400 training samples, this effectively prevents overfitting.
    - Employs 4-fold stratified cross-validation alongside Optuna for hyperparameter optimization.

2. **Linguistic Feature Enhancement (Model 2)**:

    - **Mechanism**: Prepends task-specific linguistic features as labels to the input text, helping the model rapidly capture key patterns.
    - **Sub-task 1 (Promise Identification)**: Detects promise-related words (stem matching for "commit", "pledge", "goal", etc.) + sentiment polarity analysis (promises are typically positive formulations).
        - Example: `POSITIVE Sentiment. Contains Promise Word. We commit to...`
    - **Sub-task 2 (Evidence Detection)**: Counts quantitative metric words ("percentage", "dollars") and evidence support words ("document") + NER detection of numbers and dates.
        - Example: `Proof_Count_2. Has_Numbers. Our carbon emissions decreased by 15%...`
    - **Sub-task 3 (Clarity Evaluation)**: Counts the occurrences of vague terms and explicit terms.
        - Example: `Vague_Terms_2. Specific_Terms_0. We might consider...`
    - **Sub-task 4 (Time Classification)**: Extracts dates + identifies timeline indicators.

3. **Joint Sub-task Model (Model 3)**:

    - **Base Model**: DeBERTa-v3-large (more powerful than ESG-BERT).
    - **Attention Pooling**: Replaces the standard [CLS] token representation by dynamically weighting all token representations.
        - $\alpha_i = \text{softmax}(W_{\text{attn}} h_i)$, $r = \sum_{i=1}^{n} \alpha_i h_i$
        - Allows the model to focus on semantically relevant key tokens in the document.
    - **Dual-Task Classification Heads**: Sub-task 1 (promise) and sub-task 2 (evidence) share an encoder, with independent classification pathways (Linear $\rightarrow$ Dropout $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Linear).
    - **Context Enrichment**: Prepends page numbers and document type tags to the text.
        - $x_{\text{enriched}} = \text{"[PAGE } p \text{] [ESG REPORT] "} + x_{\text{raw}}$
    - **Multi-Objective Weighting**: $\mathcal{L} = 0.6 \cdot \mathcal{L}_{\text{promise}} + 0.4 \cdot \mathcal{L}_{\text{evidence}}$, assigning more weight to promise detection as it is more fundamental.
    - **Test-Time Augmentation (TTA)**: Employs 3 forward passes with random word dropout (10%) and metadata variations, averaging predictions prior to classification with calibrated thresholds.

### Loss & Training

- Models 1 & 2: Standard cross-entropy loss, 4-fold stratified cross-validation, and Optuna hyperparameter optimization.
- Model 3: Multi-objective weighted loss (0.6:0.4), cosine learning rate scheduler with 10% warmup, gradient accumulation (16 steps), and a sequence length of 256.
- Final Submission: Model 3 is used for predictions on sub-tasks 1 and 2; Model 2 is used for sub-tasks 3 and 4.

## Key Experimental Results

### Main Results

| Model | Public Score | Private Score | Description |
|------|---------|---------|------|
| Kaggle Baseline | 0.5523 | 0.5227 | Official baseline |
| Model 1 (Base) | 0.5082 | 0.4994 | ESG-BERT baseline |
| Model 2 (Feature-Enhanced) | 0.5137 | 0.5094 | + Linguistic features |
| **Model 3 (Combined)** | 0.5255 | **0.5268** | + Attention pooling + multi-task |

### Ablation Study

| Improvement | Gain | Description |
|------|---------|------|
| Base $\rightarrow$ Feature-Enhanced | +0.010 | Limited gain from linguistic features |
| Feature-Enhanced $\rightarrow$ Combined | +0.017 | Greater contribution from multi-task + attention pooling |
| Combined vs Kaggle Baseline | +0.004 | Slightly outperforms baseline |

### Key Findings

- **Linguistic feature enhancement yields limited gain** (only +0.010): This may be because ESG-BERT has already implicitly learned many of these patterns during pre-training, leading to redundancy.
- **Limitations of prepend tag approach**: Structural disconnection may occur between feature tags and relevant textual spans.
- **Multi-task learning is effective but yields modest improvements**: Shared representations, attention pooling, and test-time augmentation (TTA) collectively contribute to the performance.
- **Data scale is the primary bottleneck**: Having only 400 training instances severely limits the learning potential of complex models (such as Model 3).
- **Potential negative transfer**: The promise and evidence tasks might require focus on contradictory features in certain instances.

## Highlights & Insights

- **The progressive experimental design is exemplary**: Transitioning from a simple baseline to feature enhancement and finally to the combined model, each step has a clear motivation for improvement and comparison, demonstrating the marginal utility of each component.
- **Practical value in linguistic feature engineering for ESG promise language**: Resources such as promise word lists, vagueness detection, and timeline indicators can be directly reused.
- **Honest reporting of marginal improvements**: Although it only outperforms the baseline by 0.004, the authors explain the underlying causes (limited data, redundancy with ESG-BERT, negative transfer) through thorough analysis. This honest diagnostics carries stronger academic value than overhyping marginal gains.
- **Open-source code**: The source code is publicly available.

## Limitations & Future Work

- **Data volume remains the biggest constraint**: Having only 400 training samples is extremely limited for any deep learning approach. Data augmentation, few-shot learning, or pre-training on external ESG reports could be beneficial.
- **Model 3 only covers sub-tasks 1 and 2**: Sub-tasks 3 and 4 were not integrated into the joint model. Multi-task modeling across all four sub-tasks could better capture inter-task dependencies.
- **Lack of systematic ablation studies**: The individual contributions of several components (e.g., attention pooling, metadata, TTA, multi-task learning) were not quantitatively isolated.
- **Use of English data only**: The original task is multilingual; cross-lingual transfer learning or multilingual transformers might yield additional gains.
- **Advanced PLMs remain unexplored**: Large language models (such as in-context learning with the GPT series) or larger open-source models were not investigated.
- **Imbalance mitigations can be deeper**: While stratified sampling and focal loss were used, alternative methods like oversampling or contrastive learning could be further explored.

## Related Work & Insights

- ESG-BERT (Mukherjee & Pothireddi, 2021) provides a pre-training foundation in the ESG domain.
- DeBERTa-v3-large (He et al., 2021) has shown excellent performance in NLU tasks with its disentangled attention.
- The ML-Promise dataset (Seki et al., 2024) is the first multilingual corporate promise verification dataset.
- Insights for the ESG text analysis community: **In low-resource scenarios, linguistic feature engineering and pre-trained models must be emphasized equally**, while remaining cautious of potential redundancies between handcrafted features and the pre-trained model's internalized knowledge.

## Rating

- Novelty: ⭐⭐⭐ The combination of methodologies is relatively standard, with no major new techniques proposed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Clear progressive comparisons among the three models, though lacking component-level ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with comprehensive descriptions of the design motivations and implementation details of each model.
- Value: ⭐⭐⭐⭐ As a SemEval system description paper, it provides practical ESG text analysis solutions and valuable lessons.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Inter-Passage Verification for Multi-evidence Multi-answer QA](inter-passage_verification_for_multi-evidence_multi-answer_qa.md)
- [\[ICLR 2026\] Ensemble Prediction of Task Affinity for Efficient Multi-Task Learning](../../ICLR2026/others/ensemble_prediction_of_task_affinity_for_efficient_multi-task_learning.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ACL 2025\] RePanda: Pandas-powered Tabular Verification and Reasoning](repanda_pandas-powered_tabular_verification_and_reasoning.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)

</div>

<!-- RELATED:END -->
