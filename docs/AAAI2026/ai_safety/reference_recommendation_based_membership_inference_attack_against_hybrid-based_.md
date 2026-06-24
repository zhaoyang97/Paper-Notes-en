---
title: >-
  [Paper Note] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems
description: >-
  [AAAI 2026][AI Safety][Membership Inference Attack] A reference recommendation-based Membership Inference Attack (MIA) is proposed, introducing a relative membership metric $\rho(u) = d(v_t, v_h) / d(v_t, v_r)$. By leveraging the personalized features of hybrid-based recommender systems to obtain reference recommendations, this work achieves the first effective attack against hybrid-based recommender systems, achieving an attack success rate of up to 93.4% with a computation…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Membership Inference Attack"
  - "Hybrid-based Recommender Systems"
  - "Reference Recommendation"
  - "Relative Membership Metric"
  - "Privacy Attacks"
date: 2026-05-08
content_hash: 68133bf2feaee0cf
---

# Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems

**Conference**: AAAI 2026  
**arXiv**: [2512.09442](https://arxiv.org/abs/2512.09442)  
**Code**: None (implementation code is provided in the appendix)  
**Area**: AI Security  
**Keywords**: Membership Inference Attack, Hybrid-based Recommender Systems, Reference Recommendation, Relative Membership Metric, Privacy Attacks

## TL;DR

A reference recommendation-based Membership Inference Attack (MIA) is proposed, introducing a relative membership metric $\rho(u) = d(v_t, v_h) / d(v_t, v_r)$. By leveraging the personalized features of hybrid-based recommender systems to obtain reference recommendations, this work achieves the first effective attack against hybrid-based recommender systems, achieving an attack success rate of up to 93.4% with a computation cost of only 10 seconds.

## Background & Motivation

### Privacy Risks of Recommender Systems

Recommender systems are widely deployed in e-commerce, social media, and other fields, recommending items or friends based on user preferences and interaction histories. However, these interaction histories often contain privacy-sensitive information. Membership Inference Attacks (MIAs) aim to determine whether a specific user's data was used to train the target recommender system, the success of which directly violates privacy regulations such as GDPR and CCPA.

### Limitations of Prior Work

Existing MIA methods (such as ST-MIA, DL-MIA) face two key limitations:

**Unrealistic assumptions**: They assume that all users with interaction histories are members, while new users with no interactions are non-members. In reality, existing users can also be non-members (e.g., they opted out of data collection or joined the platform outside the training window).

**Only applicable to hybrid-component recommender systems**: Existing attacks exploit the behavioral discrepancy between two different algorithms—collaborative filtering for members versus popularity-based recommendation for non-members. When facing a **hybrid-based recommender system (Hybrid-based RS)**—where a single algorithm utilizes both interaction histories and user attributes to serve all users simultaneously—existing attacks fail completely (with accuracy close to the 50% random guess level).

### Core Problem

**How does personalization in hybrid-based recommender systems affect MIA?** This is a non-trivial question:
- On one hand, stronger personalization might imply more privacy exposure.
- On the other hand, hybrid-based recommender systems alleviate cold-start and overfitting problems, which theoretically should enhance defense against MIA.

Previous highly efficient shadow-free attack methods (such as chi2024shadow) are also inapplicable to this scenario—as new users no longer receive uniform popularity-based recommendations, but instead receive personalized recommendations based on their attributes.

## Method

### Overall Architecture

The attack workflow consists of three steps:
1. Query the recommender system using the target user's interaction history + attributes to obtain the **target recommendation** $\mathcal{Y}_{u\_target}$.
2. Query the recommender system using only the target user's attributes to obtain the **reference recommendation** $\mathcal{Y}_{u\_ref}$.
3. Infer the membership status by comparing the target recommendation, reference recommendation, and historical interactions through the **relative membership metric** $\rho(u)$.

### Key Designs

#### 1. **Reference Recommendation Acquisition**

**Key Insight**: The unique capability of hybrid-based recommender systems is that they can generate personalized recommendations based on user attributes even without interaction histories. The attacker cleverly exploits this feature: querying the system using only the attributes $\Phi_u$ to obtain a reference baseline that is "uninfluenced by training information".

$$\mathcal{Y}_{u\_ref} = [y_{r_1}, \cdots, y_{r_n}]$$

**Design Motivation**: The reference recommendation represents "what the recommender system would provide if this user's data were not used for training". Comparing it with the target recommendation amplifies the discrepancy between members and non-members.

#### 2. **Relative Membership Metric**

$$\rho(u) = \frac{\|v_t - v_h\|_2}{\|v_t - v_r\|_2}$$

Where $v_t$, $v_h$, and $v_r$ are the feature vectors of the target recommendation, historical interactions, and reference recommendation, respectively (calculated by taking the average of the item embeddings).

**Decision Rule**: If $\rho(u) < 1$, the user is inferred as a member; otherwise, as a non-member.

**Intuition**: If the target recommendation is closer to the historical interactions (rather than the reference recommendation), it indicates that the historical interactions were likely involved in model training.

#### 3. **Mathematical Advantage Analysis of the Metric**

Let $x = d(v_t, v_h)/M$ be a normalized variable; the proposed metric is equivalent to the function $f(x) = x/(1-x)$, whereas existing linear metrics are equivalent to $g(x) = cx$.

- $f'(x) = 1/(1-x)^2 > 0$ and $f''(x) = 2/(1-x)^3 > 0$: The metric values exhibit a **non-linearly increasing** variation between members and non-members.
- For non-members (where $x$ is larger), the metric changes progressively more rapidly, thereby magnifying the gap between members and non-members.
- The rate of change for the linear metric $g'(x) = c$ is constant, yielding insufficient discriminative power for samples near the boundary.

**Connection to Special Cases**: Previous highly efficient shadow-free methods can be viewed as a special case of the proposed metric—where the reference recommendation $v_r$ is a constant determined by item popularity and does not vary across users. In contrast, $v_r$ in this work is personalized.

#### 4. **Feature Vector Construction**

Extract item feature embeddings from publicly crawlable datasets:

$$\hat{C}^{p \times q} = H \cdot W^T$$

The user-item interaction matrix is decomposed via matrix factorization, where each row $w_i$ of $W$ represents the latent feature vector of the $i$-th item. The feature vectors are computed as the average of the item embeddings in the list:

$$v_h = \frac{1}{m}\sum_{i=1}^{m} w_{h_i}, \quad v_t = \frac{1}{n}\sum_{i=1}^{n} w_{y_{t_i}}, \quad v_r = \frac{1}{n}\sum_{i=1}^{n} w_{y_{r_i}}$$

### Loss & Training

The proposed method **requires no training**—it eliminates the need for shadow models or training an attack classifier. It only requires computing a single metric value and comparing it to the threshold of 1, resulting in a time complexity of only $O(l)$ (where $l$ is the length of the feature vector).

## Key Experimental Results

### Main Results

Target recommender systems: DropoutNet and Heater  
Target datasets: MovieLens-1M (ML-1M) and MovieLens-100K (ML-100K)  
Shadow dataset (for baselines): ACM RecSys 2017 Challenge

**Attack Success Rate (ASR)**:

| Target RS | Target Dataset | Ours | ST-MIA | DL-MIA |
|--------|-----------|---------|--------|--------|
| DropoutNet | ML-1M | **0.9340** | 0.4995 | 0.5139 |
| DropoutNet | ML-100K | **0.9098** | 0.5079 | 0.5011 |
| Heater | ML-1M | **0.8376** | 0.5536 | 0.4995 |
| Heater | ML-100K | **0.7519** | 0.4920 | 0.5000 |

The ASR of baseline methods is approximately 0.5, which is almost equivalent to random guessing, demonstrating that existing methods fail completely on hybrid-based recommender systems.

**TPR@1%FPR** (High-reliability metric):

| Target RS | Target Dataset | Ours | ST-MIA | DL-MIA |
|--------|-----------|---------|--------|--------|
| DropoutNet | ML-1M | **99.84%** | 24.61% | 21.15% |
| DropoutNet | ML-100K | **68.88%** | 21.26% | 11.82% |
| Heater | ML-1M | **97.83%** | 25.05% | 24.02% |
| Heater | ML-100K | **56.05%** | 3.18% | 1.32% |

### Ablation Study

**Computational Efficiency Comparison**:

| Method | Average Computation Time | Relative Speed |
|------|-------------|---------|
| Ours | **10.4 seconds** | 1× |
| ST-MIA | 973.3 seconds | 93.6× slower |
| DL-MIA | 38,550 seconds | 3706.7× slower |

**Impact of Recommendation Size $n$**: As $n$ increases from 10 to 100, the ASR remains stable and slightly improves (e.g., (Dro., 100K) increases from < 0.9 to > 0.9).

**Impact of Feature Vector Length $l$**: As $l$ varies from 10 to 100, there is no significant change in ASR, indicating that the method is insensitive to this parameter.

**Differential Privacy Defense Evaluation**:

| Setting (Dro., 100K) | ε=0.1 | ε=0.5 | ε=1.0 | No DP |
|-------------------|-------|-------|-------|------|
| ASR | 0.5101 | 0.7837 | 0.7996 | 0.9098 |

DP provides a certain level of privacy protection (ASR approaches 0.5 under $\epsilon = 0.1$), but the proposed attack remains effective under a moderate privacy budget.

### Key Findings

1. **First effective attack against hybrid-based recommender systems**: Existing shadow training-based methods fail completely on hybrid-based recommender systems (ASR $\approx$ 0.5).
2. **Extremely high efficiency**: Eliminates shadow model training, taking 10.4 seconds versus 38,550 seconds, which is over 3,700 times faster.
3. **High reliability**: Reaches a TPR@1%FPR of 99.84% on DropoutNet + ML-1M.
4. **Distribution Visualization**: The metric value distributions of members and non-members are clearly separated, and the threshold boundary of $\rho = 1$ partitions the two classes almost perfectly.

## Highlights & Insights

1. **Exploiting system features to attack the system**: Cleverly leveraging the capability of hybrid-based recommender systems to "recommend based solely on attributes" as an attacking tool—the model's capability to mitigate cold-start issues turns out to be its privacy vulnerability.
2. **Elegant metric design**: The non-linear form of $x/(1-x)$ is naturally suited for binary classification without requiring user-specified thresholds (it is constantly set to 1) and is insensitive to absolute values.
3. **Training-free paradigm**: Unlike methods requiring shadow-model training and attack classifier training, the proposed method only requires two black-box queries and simple arithmetic operations.
4. **Unified theory and empirical evidence**: The effectiveness of the metric is comprehensively validated from mathematical function analysis to distribution visualization.

## Limitations & Future Work

1. **Use of Euclidean distance only**: Other distance metrics (such as Jaccard similarity, KL divergence) are left for future exploration.
2. **Limited dataset scale**: The evaluation is only conducted on the MovieLens datasets.
3. **Fixed threshold of 1**: Although theoretically sound, a flexible threshold might further enhance performance.
4. **Limited defense analysis**: Only differential privacy is evaluated, while other defense mechanisms are not considered.

## Related Work & Insights

- **ST-MIA** (the first recommender system MIA) and **DL-MIA** (an improvement via de-biased learning) are both based on shadow training pipelines, making them computationally expensive and ineffective on hybrid-based recommender systems.
- The shadow-free method of **chi2024shadow** can be viewed as a special case of the proposed metric (where the reference recommendation is a constant), which inspired the idea of personalized reference recommendations in this work.
- Insights for privacy research: The stronger the personalization capability of a recommender system, the greater its potential privacy risk—representing a trade-off where one cannot have both.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First attack against hybrid-based recommender systems, featuring an ingenious reference recommendation design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on 2 Recommender Systems $\times$ 2 datasets, including parameter analysis and defense evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous mathematical analysis, though some notations appear slightly redundant.
- Value: ⭐⭐⭐⭐ — Reveals critical privacy vulnerabilities, serving as an alert for recommender system security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ACL 2025\] Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](../../ACL2025/ai_safety/crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICLR 2026\] Curation Leaks: Membership Inference Attacks against Data Curation for Machine Learning](../../ICLR2026/ai_safety/curation_leaks_membership_inference_attacks_against_data_curation_for_machine_le.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)

</div>

<!-- RELATED:END -->
