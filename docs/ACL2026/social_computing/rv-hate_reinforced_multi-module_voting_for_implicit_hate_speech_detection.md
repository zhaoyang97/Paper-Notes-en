---
title: >-
  [Paper Note] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection
description: >-
  [ACL2026][Social Computing][Implicit Hate Speech Detection] RV-HATE decomposes implicit hate speech detection into four BERT contrastive learning modules targeting different data characteristics and uses PPO to learn dat…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Implicit Hate Speech Detection"
  - "Multi-module Ensemble"
  - "PPO"
  - "Soft Voting"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 465cf17aa5938d68
---

# RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection

**Conference**: ACL2026  
**arXiv**: [2510.10971](https://arxiv.org/abs/2510.10971)  
**Code**: https://github.com/leeyejin1231/RV-HATE  
**Area**: Reinforcement Learning / Content Safety / Implicit Hate Speech Detection  
**Keywords**: Implicit Hate Speech Detection, Multi-module Ensemble, PPO, Soft Voting, Contrastive Learning

## TL;DR
RV-HATE decomposes implicit hate speech detection into four BERT contrastive learning modules targeting different data characteristics and uses PPO to learn dataset-specific soft voting weights. It improves the average macro-F1 to 84.47% across five benchmarks, outperforming SharedCon by an average of 1.8 percentage points.

## Background & Motivation
**Background**: Implicit hate speech detection is more challenging than detecting explicit attacks because it often depends on context, target references, cultural background, and implied stances. Existing methods include cross-entropy classification, supervised contrastive learning, SharedCon's cluster-anchor contrastive learning, and LAHN which utilizes hard negatives.

**Limitations of Prior Work**: Different hate speech datasets originate from different platforms and annotation standards, varying in linguistic style, degree of implicitness, target boundaries, noise, and mislabeling ratios. Many methods adopt a fixed training strategy, assuming a single model can handle all dataset characteristics, which leads to limited gains on certain datasets.

**Key Challenge**: Detecting implicit hate requires simultaneous attention to contextual semantics, target entities, data noise, and boundary samples, but any single module may bias toward a specific category of features. Merging all modules into a single model can result in the loss of specialization.

**Goal**: The authors aim to preserve the complementarity of multiple specialized modules and enable the system to automatically determine the weight of each module in the final prediction for specific datasets.

**Key Insight**: RV-HATE treats module combination as a policy optimization problem. Four classifiers learn different data characteristics, a PPO policy generates module weights that are non-negative and sum to 1, and the validation set macro-F1 serves as the reward.

**Core Idea**: Instead of designing a fixed detector, it is better to train multiple biased detectors and let reinforcement learning learn how to weight them at the dataset level.

## Method
The basic units of RV-HATE are four BERT-base classifiers. Each module is based on contrastive learning but emphasizes different data characteristics. Final predictions are not directly averaged; instead, each module outputs binary logits, which are then integrated via soft voting according to weights obtained through reinforcement learning.

### Overall Architecture
The workflow is divided into three stages. In the first stage, four modules $M_0$ to $M_3$ are trained separately for each dataset. In the second stage, a lightweight PPO policy is trained on the validation set to generate module weights $w=[w_0,w_1,w_2,w_3]$. In the third stage, during inference, the four modules output logits, the system calculates the weighted average logits, and the class with the maximum value is selected as the final label.

The paper discusses hate speech identification only from the perspectives of detection and dataset analysis and does not provide operational guidance for generating, evading, or amplifying harmful content.

### Key Designs
1. **Four Data Characteristic Modules**:

    - **Function**: Respectively capture semantic clustering, target references, data anomalies, and hard negative samples in implicit hate detection.
    - **Mechanism**: $M_0$ is based on SharedCon but uses cosine similarity to select cluster anchors; $M_1$ labels target entities such as groups, organizations, and regions in the training data to help distinguish "offensiveness" from "group-targeted hate"; $M_2$ uses IQR to remove anomalous samples far from cluster centers; $M_3$ uses a hard negative queue to strengthen decision boundaries.
    - **Design Motivation**: Implicit hate datasets do not share a single source of error. IHC may rely more on target references, Hateval may be more affected by noise, and Toxigen may rely more on boundary samples, necessitating modularity rather than a singular approach.

2. **Reinforcement Learning Guided Soft Voting**:

    - **Function**: Automatically assigns module weights for different datasets, avoiding fixed averaging or manual tuning.
    - **Mechanism**: Each module outputs binary logits $z_{k,i}$, and the final logit is $Z_i^{(h)}=\sum_{k=0}^{3}w_k z_{k,i}^{(h)}$. The PPO policy generates a weight vector based on the current state; macro-F1 reward is calculated on the validation set after soft voting, and updates are stabilized using a clipped objective.
    - **Design Motivation**: Individual modules are not necessarily strong on their own, but their error patterns differ. RL weights combine complementary perspectives while preserving specialization.

3. **Explainable Dataset Characteristic Analysis**:

    - **Function**: Beyond pursuing F1 scores, it explains how different datasets rely on specific features through weights and ablation studies.
    - **Mechanism**: The paper compares single modules, module removals, equal weight voting, Euclidean distance versions, and PPO weight versions, while analyzing entity annotation ratios, outlier removal ratios, and error type distributions across datasets.
    - **Design Motivation**: In content safety scenarios, understanding why a model is effective on a certain dataset is equally important. Module weights provide a coarse-grained but readable diagnostic signal.

### Loss & Training
The four detection modules use BERT-base-uncased, with SimCSE-BERT as the text embedding model, trained for 6 epochs. Learning rates are selected from $2e^{-5}$ and $3e^{-5}$, the temperature is 0.3, and the number of clusters is chosen from 20, 75, or 125. The reinforcement learning phase runs for 10,000 steps with initial weights of `[0.25, 0.25, 0.25, 0.25]`, constrained to be positive and sum to 1. All experiments use 3 random seeds; macro-F1 is reported due to class imbalance.

## Key Experimental Results

### Main Results
The paper compares CE, SCL, SharedCon, LAHN, and RV-HATE on five datasets: IHC, SBIC, DYNA, Hateval, and Toxigen.

| Method | IHC | SBIC | DYNA | Hateval | Toxigen | Avg. macro-F1 |
|------|-----|------|------|---------|---------|---------------|
| CE | 77.70 | 83.80 | 78.80 | 81.11 | 90.06 | 82.29 |
| SCL | 77.81 | 82.92 | 80.39 | 81.28 | 90.75 | 82.63 |
| SharedCon | 78.50 | 84.30 | 79.10 | 80.24 | 91.21 | 82.67 |
| LAHN | 78.40 | 83.98 | 79.64 | 80.42 | 90.42 | 82.57 |
| RV-HATE | 79.07 | 84.62 | 81.82 | 83.44 | 93.41 | 84.47 |

Compared to SharedCon, RV-HATE achieves an average Gain of 1.8 percentage points; it is 2.33 percentage points higher than CE on Hateval and 2.2 percentage points higher than SharedCon on Toxigen. Given that performance on these tasks often plateaus around 80%, this margin is practically significant.

### Ablation Study

| Configuration | IHC | SBIC | DYNA | Hateval | Toxigen | Avg. | Description |
|------|-----|------|------|---------|---------|------|------|
| combined modules | 77.32 | 81.31 | 76.50 | 81.26 | 92.02 | 81.64 | Single model merging all modules; loses specialization |
| equal weights | 78.58 | 84.06 | 81.07 | 82.52 | 92.69 | 83.78 | Fixed 0.25 weights |
| Euclidean version | 78.90 | 82.95 | 81.64 | 83.19 | 93.36 | 84.01 | L2 instead of cosine |
| RV-HATE | 79.07 | 84.62 | 81.82 | 83.44 | 93.41 | 84.47 | PPO weights + cosine |

| Module Setting | Avg. macro-F1 | Key Insight |
|----------|---------------|----------|
| $M_0$ alone | 82.68 | Base module for cosine-based clustering contrastive learning |
| $M_1$ alone | 82.43 | Target entity labeling does not benefit all datasets |
| $M_2$ alone | 82.89 | Outlier processing helps with noisy data |
| $M_3$ alone | 83.00 | Strongest single module using hard negative boundary modeling |
| RV-HATE Full | 84.47 | Best performance through module complementarity |
| w/o $M_3$ | 83.99 | Largest average drop, indicating hard negatives are critical |

### Key Findings
- Training the four modules as a single "combined model" dropped the average to 81.64, indicating that "module specialization + voting ensemble" is more suitable for this task than "forcing everything into one model."
- PPO weights improved the average by 0.68 percentage points over equal weight voting, proving that different datasets indeed require different module combinations.
- Cosine similarity outperformed Euclidean distance by 0.46 percentage points, aligning with the intuition that high-dimensional semantic embeddings prioritize direction.
- Computational overhead primarily stems from four BERT forward passes. The PPO policy has only about 4.8K parameters, adding approximately 5-10 minutes to training; inference latency is a linear multiple of a single model.

## Highlights & Insights
- The paper does not treat "generalization across all datasets" as the sole goal but acknowledges that dataset variance itself is important. This perspective is realistic for content safety, as labeling standards and platform context often dictate model behavior.
- Reinforcement learning is used here to learn ensemble weights rather than generate text, resulting in low risk and clear objectives. The action space for PPO is small, and the reward directly corresponds to the validation set macro-F1.
- Module ablation provides interpretability: if a dataset relies heavily on target entities or outlier cleaning, the weights and ablation experiments will expose this dependency.
- For practical systems, content safety classifiers can be designed as "multi-expert + dataset/domain adaptive weights" rather than deploying the same static classifier across all platforms.

## Limitations & Future Work
- The $M_1$ target labeling module is unstable on machine-generated samples, suggesting that entity labeling strategies are sensitive to style and data distribution.
- Inference requires four BERT-base forward passes; while parallelizable, this remains an additional cost for low-latency content moderation systems.
- Weights are optimized at the dataset level, not through per-sample dynamic routing. Finer-grained routing might be needed for different sub-communities or topics within the same dataset.
- The datasets themselves contain annotation ambiguity and mislabels; improvements in macro-F1 do not fully resolve whether the labels are reasonable. Future work could incorporate uncertainty, human disagreement, and cross-cultural annotation differences into the training objectives.

## Related Work & Insights
- **vs SharedCon**: SharedCon learns shared semantic patterns through cluster anchors. RV-HATE inherits this direction but switches to cosine similarity and adds specialized modules for targets, outliers, and hard negatives.
- **vs LAHN**: LAHN emphasizes hard negatives. RV-HATE treats hard negatives as one expert module and complements it with other modules through voting.
- **vs Single-model multi-function training**: The poor performance of "combined modules" suggests that module specialization is crucial in implicit hate detection; unified training causes conflicting feature preferences.
- **vs Standard ensemble**: While equal voting is already effective, PPO weights further capture dataset variances, making the ensemble both more interpretable and powerful.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using PPO for module weight learning is not overly complex, but its integration with dataset characteristic analysis is highly fitting.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Five datasets, multiple baselines, variants, and module ablations provide a complete study, though cross-lingual and cross-cultural extrapolation requires more validation.
- Writing Quality: ⭐⭐⭐⭐☆ The methodology is clearly decomposed, and module contributions are well-explained.
- Value: ⭐⭐⭐⭐☆ Offers practical reference value for modular and domain-adaptive design in content safety detection systems, especially for implicit and boundary-vague classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)

</div>

<!-- RELATED:END -->
