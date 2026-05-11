---
title: >-
  [Paper Note] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems
description: >-
  [AAAI 2026][AI Safety][Membership Inference Attack] This paper proposes a Reference Recommendation-based Membership Inference Attack (MIA), designing a relative membership metric $\rho(u) = d(v_t, v_h) / d(v_t…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Membership Inference Attack"
  - "Hybrid-based Recommender Systems"
  - "Reference Recommendation"
  - "Relative Membership Metric"
  - "Privacy Attack"
date: 2026-05-08
content_hash: 2a7777ae305151c1
---

# Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems

**Conference**: AAAI 2026
**arXiv**: [2512.09442](https://arxiv.org/abs/2512.09442)
**Code**: None (implementation code provided in the appendix)
**Area**: AI Security
**Keywords**: Membership Inference Attack, Hybrid-based Recommender Systems, Reference Recommendation, Relative Membership Metric, Privacy Attack

## TL;DR

This paper proposes a Reference Recommendation-based Membership Inference Attack (MIA), designing a relative membership metric $\rho(u) = d(v_t, v_h) / d(v_t, v_r)$ that exploits the personalization capability of hybrid-based recommender systems to obtain reference recommendations. It is the first method to effectively attack hybrid-based recommender systems, achieving an attack success rate of up to 93.4% with a computational cost of only 10 seconds.

## Background & Motivation

### Privacy Risks in Recommender Systems

Recommender systems are widely deployed in e-commerce, social media, and other domains, suggesting items or connections based on user preferences and interaction histories. However, these interaction histories often contain privacy-sensitive information. Membership Inference Attacks (MIA) aim to determine whether a specific user's data was used to train the target recommender system; a successful attack constitutes a violation of privacy regulations such as GDPR and CCPA.

### Limitations of Prior Work

Existing MIA methods (e.g., ST-MIA, DL-MIA) suffer from two critical issues:

**Unrealistic assumptions**: They assume that all users with interaction histories are members and that new users without interactions are non-members. In practice, existing users may also be non-members (e.g., users who opted out of data collection or joined the platform outside the training window).

**Applicability limited to hybrid-component recommender systems**: Existing attacks exploit behavioral differences between two distinct algorithms—collaborative filtering for members and popularity-based recommendations for non-members. When confronted with **hybrid-based recommender systems (Hybrid-based RS)**—where a single algorithm simultaneously leverages interaction histories and user attributes to serve all users—existing attacks fail entirely, with success rates approaching random guessing (50%).

### Core Research Question

**How does personalization in hybrid-based recommender systems affect MIA?** This is a non-trivial question:
- On one hand, stronger personalization may imply greater privacy exposure.
- On the other hand, hybrid-based recommender systems mitigate cold-start and overfitting issues, theoretically strengthening defenses against MIA.

Prior efficient shadow-free attack methods (chi2024shadow) are also inapplicable in this setting, since new users no longer receive uniform popularity-based recommendations but instead receive attribute-based personalized recommendations.

## Method

### Overall Architecture

The attack proceeds in three steps:
1. Query the recommender system using the target user's interaction history and attributes to obtain the **target recommendation** $\mathcal{Y}_{u\_target}$.
2. Query the recommender system using only the target user's attributes to obtain the **reference recommendation** $\mathcal{Y}_{u\_ref}$.
3. Infer membership status by comparing the target recommendation, reference recommendation, and interaction history via the **relative membership metric** $\rho(u)$.

### Key Designs

#### 1. **Reference Recommendation Acquisition**

Core insight: The unique capability of hybrid-based recommender systems—generating personalized recommendations based solely on user attributes, even without interaction histories—is exploited by the attacker. By querying the system with only attributes $\Phi_u$, a reference baseline unaffected by training information is obtained.

$$\mathcal{Y}_{u\_ref} = [y_{r_1}, \cdots, y_{r_n}]$$

Design Motivation: The reference recommendation represents "what the recommender system would suggest if this user's data had not been used for training." Comparing it against the actual recommendation amplifies the behavioral difference between members and non-members.

#### 2. **Relative Membership Metric**

$$\rho(u) = \frac{\|v_t - v_h\|_2}{\|v_t - v_r\|_2}$$

where $v_t$, $v_h$, and $v_r$ denote the feature vectors of the target recommendation, interaction history, and reference recommendation, respectively (computed as the mean of item embeddings).

**Decision rule**: $\rho(u) < 1$ indicates membership; otherwise, non-membership.

**Intuition**: If the target recommendation is closer to the interaction history than to the reference recommendation, the interaction history likely participated in model training.

#### 3. **Mathematical Advantages of the Metric**

Let $x = d(v_t, v_h)/M$ be a normalized variable; the metric is equivalent to $f(x) = x/(1-x)$, whereas existing linear metrics are equivalent to $g(x) = cx$.

- $f'(x) = 1/(1-x)^2 > 0$, $f''(x) = 2/(1-x)^3 > 0$: the metric variation between members and non-members is **nonlinearly increasing**.
- For non-members (larger $x$), the metric changes more dramatically, amplifying the gap between members and non-members.
- The linear metric $g'(x) = c$ has a constant rate of change, providing insufficient discriminative power near the decision boundary.

**Connection to a special case**: The prior efficient shadow-free method can be viewed as a special case of the proposed metric, where the reference recommendation $v_r$ is a constant determined by item popularity rather than being user-specific.

#### 4. **Feature Vector Construction**

Item feature embeddings are extracted from publicly crawlable datasets:

$$\hat{C}^{p \times q} = H \cdot W^T$$

The user–item interaction matrix is factorized via matrix decomposition, where each row $w_i$ of $W$ is the latent feature vector of item $i$. Feature vectors are computed as the mean of item embeddings in the respective lists:

$$v_h = \frac{1}{m}\sum_{i=1}^{m} w_{h_i}, \quad v_t = \frac{1}{n}\sum_{i=1}^{n} w_{y_{t_i}}, \quad v_r = \frac{1}{n}\sum_{i=1}^{n} w_{y_{r_i}}$$

### Loss & Training

The proposed method **requires no training**—no shadow models, no attack classifier training. It simply computes a metric value and compares it against the threshold of 1, with a time complexity of only $O(l)$ (where $l$ is the length of the feature vector).

## Key Experimental Results

### Main Results

Target recommender systems: DropoutNet and Heater
Target datasets: MovieLens-1M (ML-1M) and MovieLens-100K (ML-100K)
Shadow dataset (for baselines): ACM RecSys 2017 Challenge

**Attack Success Rate (ASR)**:

| Target RS | Target Dataset | Ours | ST-MIA | DL-MIA |
|-----------|---------------|------|--------|--------|
| DropoutNet | ML-1M | **0.9340** | 0.4995 | 0.5139 |
| DropoutNet | ML-100K | **0.9098** | 0.5079 | 0.5011 |
| Heater | ML-1M | **0.8376** | 0.5536 | 0.4995 |
| Heater | ML-100K | **0.7519** | 0.4920 | 0.5000 |

Baseline methods achieve ASR ≈ 0.5, nearly equivalent to random guessing, demonstrating that existing methods completely fail against hybrid-based recommender systems.

**TPR@1%FPR** (high-reliability metric):

| Target RS | Target Dataset | Ours | ST-MIA | DL-MIA |
|-----------|---------------|------|--------|--------|
| DropoutNet | ML-1M | **99.84%** | 24.61% | 21.15% |
| DropoutNet | ML-100K | **68.88%** | 21.26% | 11.82% |
| Heater | ML-1M | **97.83%** | 25.05% | 24.02% |
| Heater | ML-100K | **56.05%** | 3.18% | 1.32% |

### Ablation Study

**Computational Efficiency Comparison**:

| Method | Average Computation Time | Relative Speed |
|--------|------------------------|----------------|
| Ours | **10.4 seconds** | 1× |
| ST-MIA | 973.3 seconds | 93.6× slower |
| DL-MIA | 38,550 seconds | 3706.7× slower |

**Effect of Recommendation Count $n$**: As $n$ increases from 10 to 100, ASR remains stable with a slight improvement (e.g., (Dro., 100K) increases from < 0.9 to > 0.9).

**Effect of Feature Vector Length $l$**: As $l$ varies from 10 to 100, ASR shows no significant change, indicating that the method is insensitive to this parameter.

**Differential Privacy Defense Evaluation**:

| Setting (Dro., 100K) | ε=0.1 | ε=0.5 | ε=1.0 | No DP |
|----------------------|-------|-------|-------|------|
| ASR | 0.5101 | 0.7837 | 0.7996 | 0.9098 |

Differential privacy provides some degree of privacy protection (ASR approaches 0.5 at $\epsilon = 0.1$), yet the proposed attack remains effective under moderate privacy budgets.

### Key Findings

1. **First effective attack against hybrid-based recommender systems**: Existing shadow training-based methods completely fail on hybrid-based recommender systems (ASR ≈ 0.5).
2. **Extremely high efficiency**: No shadow model training required; 10.4 seconds vs. 38,550 seconds—approximately 3,700× faster.
3. **Strong reliability**: TPR@1%FPR reaches 99.84% on DropoutNet + ML-1M.
4. **Distribution visualization**: The metric value distributions of members and non-members are clearly separated, with the threshold boundary $\rho = 1$ almost perfectly partitioning the two classes.

## Highlights & Insights

1. **Exploiting system capabilities to attack the system**: The attack elegantly leverages the hybrid-based recommender system's ability to generate recommendations from attributes alone—the very capability designed to eliminate cold-start becomes its privacy vulnerability.
2. **Elegance of the metric design**: The nonlinear form $x/(1-x)$ is naturally suited for binary classification, requires no user-specified threshold (always 1), and is insensitive to absolute values.
3. **Training-free paradigm**: Unlike methods requiring shadow model training and attack classifier training, the proposed approach requires only two black-box queries and simple arithmetic operations.
4. **Theoretical and empirical consistency**: Functional analysis and distribution visualization jointly validate the effectiveness of the metric from multiple perspectives.

## Limitations & Future Work

1. **Exclusive use of Euclidean distance**: Alternative distance metrics (Jaccard, KL divergence, etc.) are left for future exploration.
2. **Limited dataset scale**: Validation is performed only on the MovieLens series.
3. **Fixed threshold at 1**: Although theoretically justified, a flexible threshold may further improve performance.
4. **Limited defense analysis**: Only differential privacy is evaluated; other defense mechanisms are not considered.

## Related Work & Insights

- **ST-MIA** (the first MIA for recommender systems) and **DL-MIA** (improved via debiased learning) both rely on shadow training pipelines, which are computationally expensive and ineffective against hybrid-based recommender systems.
- The shadow-free method of **chi2024shadow** is a special case of the proposed metric (with a constant reference recommendation), which inspired the personalized reference recommendation approach in this work.
- Implication for privacy research: The stronger the personalization capability of a recommender system, the greater the potential privacy risk—utility and privacy remain fundamentally at odds.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First attack targeting hybrid-based recommender systems; the reference recommendation idea is highly inventive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 2 RS × 2 datasets, with parameter analysis and defense evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical analysis is rigorous, though some notation is slightly redundant.
- Value: ⭐⭐⭐⭐ — Reveals an important privacy vulnerability with significant warning implications for recommender system security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

</div>

<!-- RELATED:END -->
