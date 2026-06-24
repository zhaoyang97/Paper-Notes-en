---
title: >-
  [Paper Note] Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback
description: >-
  [ICLR2026][LLM Alignment][Reward Model] This paper points out that existing reward models (RMs) only utilize binary preferences ("A is better than B"). When faced with human Likert-scale feedback ("significantly better/better/slightly better"), they rely on ad-hoc heuristic patches like manual margins or weighting factors. The authors reformulate reward modeling as a **discrete ordinal regression** problem. From the ordered logit model, they naturally derive two principled lo…
tags:
  - "ICLR2026"
  - "LLM Alignment"
  - "Reward Model"
  - "Ordinal Regression"
  - "Likert Preferences"
  - "Threshold Learning"
  - "RLHF"
date: 2026-05-08
content_hash: c85ecf9be44c8c1a
---

# Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mteZOi0xyu](https://openreview.net/forum?id=mteZOi0xyu)  
**Code**: To be confirmed (authors claim extension based on TRL library, to be open-sourced after acceptance)  
**Area**: Alignment RLHF / Reward Modeling  
**Keywords**: Reward Model, Ordinal Regression, Likert Preferences, Threshold Learning, RLHF

## TL;DR
This paper points out that existing reward models (RMs) only utilize binary preferences ("A is better than B"). When faced with human Likert-scale feedback ("significantly better/better/slightly better"), they rely on ad-hoc heuristic patches like manual margins or weighting factors. The authors reformulate reward modeling as a **discrete ordinal regression** problem. From the ordered logit model, they naturally derive two principled losses (NLL and All-Threshold), allowing "thresholds" that separate preference levels to be learned directly from data. This approach consistently matches or outperforms heuristic baselines on RewardBench / RM-Bench and reduces error severity by 87%.

## Background & Motivation

**Background**: Modern LLM alignment (RLHF, DPO) is essentially built on the Bradley-Terry (BT) model, which assumes that "the probability of preferring A over B grows monotonically with the reward difference": $p^\star(y \succ y' \mid x) = \sigma\big(r^\star(x,y) - r^\star(x,y')\big)$. Reward models are discriminators trained using this binary logistic loss.

**Limitations of Prior Work**: The BT model is inherently designed for **binary comparisons**—A is either preferred or not. However, increasing preference datasets (e.g., HelpSteer2/3) provide more than binary labels; annotators provide **ordinal levels** on Likert scales such as "significantly better," "better," "slightly better," or "about the same." While these signals are much richer, current methods are locked into the binary foundation and often discard them as noise or apply ad-hoc patches.

**Key Challenge**: Existing patches fall into three categories: **Margin BT** (Llama-2) subtracts a manual margin $m(z)$ from the loss; **Scaled BT** (HelpSteer2) multiplies the loss by a preference intensity $m(z)$; **Soft Label** (Gunter et al.) interprets ordinal levels as soft probabilities $p(z)$. All share two fundamental flaws: first, **there is no mathematical model describing how humans provide ordinal labels**, making all modifications intuitive but lack clear assumptions; second, **they rely entirely on hand-tuned hyperparameters**—how large should the margin be between "slightly better" and "significantly better"? These values lack clear interpretation and make the methods fragile and dataset-dependent.

**Goal**: To provide a **principled, unified mathematical framework** for ordinal preference reward modeling. The loss function should derive naturally from clear modeling assumptions, and all parameters (including boundaries between levels) should be learned from data rather than manually specified.

**Key Insight**: The authors observe that "ordered categorical prediction" has long-standing solutions in statistics known as **ordinal regression**, specifically designed for rating systems (1–5 stars), surveys, and severity levels. While the machine learning community has developed extensive methods for it (ordered logits, threshold learning, large-margin methods), these have not been rigorously applied to preference learning.

**Core Idea**: Use a set of **thresholds learned from data** to partition the continuous reward difference space into several ordered intervals, each corresponding to a preference level. Replace heuristic margins/weights with mature ordinal regression losses (probabilistic NLL or margin-based All-Threshold), turning the distance between levels into learnable parameters rather than hyperparameters.

## Method

### Overall Architecture

**Mechanism**: The input consists of pairwise preference data with Likert levels $D = \{(x_i, y_i, y'_i, z_i)\}$, where $z_i \in [-K] \cup \{0\} \cup [K]$ is the ordinal label. $z>0$ indicates $y$ is preferred at level $z$, $z<0$ indicates $y'$ is preferred, and $z=0$ denotes a tie. The output is a scalar reward model $r_\phi(x,y)$. It is required that the **reward difference** $s_\phi(x,y,y') = r_\phi(x,y) - r_\phi(x,y')$ has the correct sign and a magnitude reflecting preference intensity.

The framework transforms "binary discrimination" into "ordinal regression" via four steps: ① Utilize the BT reward difference as the continuous predictor $s_\phi$; ② Introduce $2K$ ordered thresholds $\zeta_{-K} < \dots < \zeta_{-1} < \zeta_1 < \dots < \zeta_K$ to partition the real line into $2K+1$ segments; ③ Derive two types of losses (Probabilistic NLL / Margin-based AT) from this structure and apply L2 regularization; ④ Jointly optimize reward parameters $\phi$ and thresholds $\zeta$ via reparameterization.

### Key Designs

**1. Reformulating RM as Discrete Ordinal Regression: Space Partitioning via Learned Thresholds**

Addressing the weakness that BT only understands binary intensity, the authors borrow the latent variable framework from ordinal regression. Standard ordinal regression maps inputs to a 1D latent space and uses ordered thresholds $-\infty = \zeta_0 < \zeta_1 < \dots < \zeta_K = +\infty$ to partition it. The latent variable here is the reward difference $s_\phi(x,y,y')$. Because preferences are directional, $2K$ thresholds are placed symmetrically around zero. The shift is fundamental: "distance between slightly better and much better" is now a **model parameter $\zeta$** learned from data.

**2. Two Losses Derived from Modeling Assumptions: NLL and All-Threshold**

**Probabilistic (NLL)** assumes annotators follow an **ordered logit model**: the probability of observing level $z$ is the difference between sigmoids of adjacent thresholds. For example, for $z\in[K]$, $p(y \succ_z y' \mid x) = \sigma(\zeta_{z+1} - s_\phi) - \sigma(\zeta_z - s_\phi)$. This ensures probabilities are non-negative and sum to 1. The negative log-likelihood is $L_{\text{NLL}} = -\log p(y \succ_z y' \mid x)$.

**Margin-based (AT, All-Threshold)** does not assume a specific probability model but is inspired by large-margin methods. It penalizes the reward difference for falling on the wrong side of every threshold:
$$L_{\text{AT}}(r_\phi, \zeta) = \sum_{l \in [-K]\cup[K]} -\log \sigma\big(\nu(l;z)\cdot(\zeta_l - s_\phi)\big)$$
where $\nu(l;z) = -1$ if $l<z$ and $+1$ otherwise. It requires $s_\phi$ to be greater than all thresholds $l<z$ and smaller than all $l \ge z$. By **summing penalties across all thresholds**, it penalizes severe misclassifications more heavily.

**3. Symmetric vs. Asymmetric Thresholds: Theorem 3.2**

**Symmetric models** assume the intensity of "$y$ better than $y'$ at level $k$" equals its inverse, i.e., $\zeta_{-k} = -\zeta_k$. **Theorem 3.2** provides theoretical backing: under an ordered logit model, if preference data satisfies symmetry $P(y \succ_k y' \mid s = r) = P(y' \succ_{-k} y \mid s = r)$, then thresholds **must** satisfy $\zeta_{-k} = -\zeta_k$. Symmetry reduces parameters and prevents overfitting.

**4. Regularization for Boundedness + Monotonic Reparameterization**

**Theorem 3.1** proves that without regularization, if a separable solution exists, the loss can be minimized to zero by scaling thresholds to infinity, leading to numerical instability. The authors apply L2 regularization to thresholds:
$$\min_{\phi, \zeta \in C} \sum_{(x,y,y',z)\in D} L(r_\phi, \zeta) + \lambda \lVert \zeta \rVert_2^2$$
To enforce the strictly increasing constraint $\zeta_{k-1} < \zeta_k$, they use **monotonic reparameterization**: $\zeta_{k} = \zeta_{k-1} + \exp(\alpha_k)$.

### Loss & Training

Three instances are implemented: **NLL-Symmetric**, **NLL-Asymmetric**, and **All-Threshold**. Backbones are Llama-3.1-8B, Mistral-7B, and Zephyr-7B SFT checkpoints. Training involves 8 epochs, effective batch size 64, using FSDP on H100s.

## Key Experimental Results

### Main Results

Trained on HelpSteer2/3 (7-level Likert scale) and evaluated on RewardBench and RM-Bench. **NLL-Symmetric is the most robust method**, achieving the highest average scores across most configurations, exceeding baselines by 2–5%.

| Model | Method | Chat-Hard | Safety | Reason | Avg |
|------|------|-----------|--------|--------|-----|
| Llama | Margin BT | 0.660 | 0.885 | 0.703 | 0.802 |
| Llama | Soft Label | 0.581 | 0.689 | 0.680 | 0.722 |
| Llama | **Ours (NLL-Sym)** | **0.728** | **0.897** | **0.804** | **0.843** |

### Key Findings
- **Reduced Error Severity**: NLL-Sym reduces the number of errors by 35%, but more importantly, it reduces the **average error margin by 87%** (from 3.827 to 0.501). When it fails, it does so with low confidence on ambiguous samples, which is critical for preventing reward hacking in RL.
- **Joint Training is Essential**: Fitting thresholds to a frozen RM (post-hoc) yields an MAE of 1.725; joint training achieves 1.060. The fine-grained ordinal structure must be learned alongside the rewards.
- **Noise Robustness**: Performance is nearly invariant to systematic shift noise (100% pollution of labels shifted by one level) because the learned thresholds absorb the bias.
- **Necessary Regularization**: Without L2 regularization, thresholds grow unboundedly as predicted by Theorem 3.1.

## Highlights & Insights
- **Value of Frameworks over Patches**: The paper's "Aha!" moment is the realization that this is a standard problem in statistics (ordinal regression). Heuristics like margins are merely approximations of this mature mathematical object.
- **Erasure of Hyperparameters**: By converting "margin $m$" into learnable parameter "threshold $\zeta$", the method eliminates the fragility of manual tuning and adapts automatically to different datasets.
- **Theoretical Closure**: Theorem 3.1 explains the need for regularization, and Theorem 3.2 explains why symmetry helps. Both are empirically validated.

## Limitations & Future Work
- **Dependency on Likert Data**: Gains rely on ordinal labels; for binary datasets, it degrades to the BT model.
- **Ordered Logit Assumption**: NLL assumes human labeling follows an ordered logit process, which may be an approximation.
- **Downstream RL Validation**: While the paper argues that low-severity errors benefit RL, it does not explicitly run end-to-end RLHF to measure final policy quality.

## Related Work & Insights
- **vs. Margin BT (Llama-2)**: Margin BT uses manual $m(z)$. Ours replaces it with learned thresholds backed by probabilistic models.
- **vs. Soft Label**: Soft labels treat levels as probabilities. Ours proves this is a heuristic lacking a generative model and shows that post-hoc soft labels cannot recover the fine-grained structure found via joint training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[ICLR 2026\] RLBFF: Binary Flexible Feedback to Bridge Between Human Feedback & Verifiable Rewards](rlbff_binary_flexible_feedback_to_bridge_between_human_feedback_verifiable_rewar.md)
- [\[ICLR 2026\] Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](omni-reward_towards_generalist_omni-modal_reward_modeling_with_free-form_prefere.md)
- [\[ICLR 2026\] Robust Reward Modeling via Causal Rubrics](robust_reward_modeling_via_causal_rubrics.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->
