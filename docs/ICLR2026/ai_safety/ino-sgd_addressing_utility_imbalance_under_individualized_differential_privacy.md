---
title: >-
  [Paper Note] INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy
description: >-
  [ICLR 2026][AI Safety][Individualized DP] This paper identifies that "Individualized Differential Privacy" (IDP) creates utility imbalances even when the training set itself is balanced—data with stricter privacy requirements becomes severely underrepresented. The authors propose INO-SGD: sorting gradients by loss within each batch and applying "continuous" do
tags:
  - ICLR 2026
  - AI Safety
  - Individualized DP
  - DP-SGD
date: 2026-05-08
content_hash: 012d0ede814446d5
---
# INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HMapYMkcrl](https://openreview.net/forum?id=HMapYMkcrl)  
**Code**: [https://github.com/snoidetx/ino-sgd](https://github.com/snoidetx/ino-sgd)  
**Area**: Differential Privacy / Individualized Privacy / Privacy Fairness  
**Keywords**: Individualized DP, DP-SGD, Utility Imbalance, Gradient Reweighting, Order Statistics  

## TL;DR
This paper identifies that "Individualized Differential Privacy" (IDP) creates utility imbalances even when the training set itself is balanced—data with stricter privacy requirements becomes severely underrepresented. The authors propose INO-SGD: sorting gradients by loss within each batch and applying "continuous" down-weighting to unimportant gradients. This compensates the utility of more private groups while strictly satisfying the IDP budget of every data owner.

## Background & Motivation
**Background**: Differential Privacy (DP) is the mainstream method for protecting training data in machine learning. DP-SGD achieves $(\epsilon,\delta)$-DP by clipping individual gradients and adding Gaussian noise at each step. As laws like GDPR and CCPA establish "personal data ownership," different data owners set their own privacy budgets $\epsilon_n$, giving rise to Individualized Differential Privacy (IDP). IDP-SGD, proposed by Boenisch et al., uses two variants to satisfy IDP: SAMPLE provides different owners with different sampling rates $q_n$, and SCALE provides different owners with different clipping thresholds $C_n$.

**Limitations of Prior Work**: The authors find an overlooked "utility imbalance" problem in IDP-SGD. When different data groups are held by owners with **different** privacy requirements (e.g., owners of positive cases for a stigmatized disease requiring stronger privacy), the trained model's utility on these "more private" groups is significantly lower. This imbalance harms model owners and subsequent similar users (like other patients) during deployment.

**Key Challenge**: This imbalance stems from two mechanisms—**Minority Initial Drop (MID)**: In early stages, the gradient sum and magnitude of more private groups are too small; the loss always descends first in "less private" majority directions, while minority group losses may even rise. **Biased Optimization Objective (BOO)**: Upon convergence, the model biases toward groups where reducing loss is more "cost-effective." Crucially, traditional methods for correcting data/class imbalance fail under IDP: upsampling or scaling up minority gradients **violates IDP** (by increasing module sensitivity), downsampling or scaling down the majority **wastes privacy budget** and reduces overall utility, and rescaling losses is offset by clipping.

**Goal**: Actively correct the utility imbalance induced by IDP without violating any owner's IDP budget or sacrificing overall utility.

**Core Idea**: **[Strategic Discarding of Low-Importance Information]** Since the gradients of more private groups cannot be scaled up, the approach instead discards unimportant gradients with smaller losses (mostly contributed by less private owners) within a batch. This makes "room" for important gradients while ensuring the total change remains bounded by the clipping threshold, thereby compensating the minority group while satisfying IDP.

## Method

### Overall Architecture
INO-SGD treats each data gradient as a collection of infinitely many "gradient slices" (each slice has a loss value equal to the data's loss). Within each batch, all slices are sorted in descending order of loss. An **importance function** assigns a score of 0–1 to each slice representing its retention ratio: slices at the "tail" with low loss receive low scores and are partially discarded, while other slices are fully retained. The discarded slices free up sensitivity budget for important gradients, ensuring the batch's module sensitivity still does not exceed the individual clipping thresholds. Thus, it strictly satisfies IDP, and the privacy consumption per step is identical to IDP-SGD.

```mermaid
flowchart LR
    A[Sampling batch $B_t$] --> B[Computing individual gradients $g_d$]
    B --> C[Sorting by loss in descending order<br/>to get permutation $\pi_t$]
    C --> D[Constructing Batch Importance Function BIF $f_t$<br/>from TIF $f_{\text{tail}}$]
    D --> E[Integrating to get average<br/>importance scores $\rho_k$]
    E --> F[Summing clipped gradients scaled by $\rho_k$<br/>$G_t = \sum \rho_k \bar{g}_{\pi(k)}$]
    F --> G[Adding Gaussian noise / Expected batch size $b$<br/>Update $\theta_t$ via SGD]
```

### Key Designs

**1. "Continuous" Importance Modeling of Ordered Gradients: Turning unusable hard selection into IDP-compliant soft weighting.** The most intuitive approach is to keep only the gradients with the largest losses and discard the smallest. However, this increases the algorithm's module sensitivity $\Delta_A^d := \sup_{D \triangle D_d=\{d\}}\|A(D)-A(D_d)\|$, violating IDP. The breakthrough of INO-SGD is treating each data $d$'s gradient as a set of infinitesimal "gradient slices" with the same loss. Each datum retains at most $C_{o(d)}$ amount of slices, so the total volume of slices in the batch $\Gamma_t := \sum_{d\in B_t} C_{o(d)}$ is controlled. All retained slices are arranged in descending order of loss over a continuous interval $[0,\Gamma_t]$, and "importance" is distributed via integration. This transforms discrete hard truncation into a continuous soft weighting with provably bounded sensitivity.

**2. Tail Importance Function (TIF) and Batch Importance Function (BIF): Expressing beliefs on "which data is important" with minimal parameters.** TIF is defined as a non-increasing, Riemann-integrable function $f_{\text{tail}}:[0,\gamma]\to[0,1]$ over a fixed length $\gamma$. $f_{\text{tail}}(c)$ represents the importance of the gradient slice at the $(c/\gamma)$ quantile of the tail. Only slices falling within the tail of length $\gamma$ are assigned scores $<1$; all others receive 1 to maintain a high signal-to-noise ratio (since noise is added after summation). Each batch constructs a unique BIF $f_t$ by shifting and stitching the TIF onto the batch's $[0,\Gamma_t]$ interval. In practice, the authors model this belief using a (horizontally flipped) Cumulative Distribution Function of $\mathrm{Beta}(\alpha,\beta)$: $f_{\text{tail}}(c):=\int_0^{1-c/\gamma} x^{\alpha-1}(1-x)^{\beta-1}\mathrm{d}x \big/ \int_0^1 x^{\alpha-1}(1-x)^{\beta-1}\mathrm{d}x$. A larger $\alpha$ increases the weight of relatively more important gradients in the tail, while a larger $\beta$ decreases the weight of unimportant ones. For the $k$-th ranked gradient, its average importance score is given by the integral mean of the BIF over its corresponding interval: $\rho_k = \int_{c_{k-1}}^{c_k} f_t \,\mathrm{d}c / (c_k-c_{k-1})$. Finally, the clipped gradients are scaled by $\rho_k$, summed, and noisy.

**3. Privacy Provability: Module sensitivity is caught by clipping thresholds; sorting costs no extra budget.** The core theorem proves that for any datum $d$, adding $d$ to a batch does not change the influence of data less important than $d$ because the tail length is fixed. While more important data are "pushed to the left" causing their scores to rise, the total change relative to the sum of $d$'s clipped gradients is still bounded by $C_{o(d)}$ (proven using the triangle inequality + $L_2$ norm bounds + telescoping sums from the BIF construction). Consequently, INO-SGD satisfies $(\alpha,\bar\epsilon)$-IRDP, where $\bar\epsilon_n = 2T\alpha_n C_n^2 q_n^2/\sigma^2$, which is **identical** to IDP-SGD—meaning it does not sacrifice iteration steps for better learning dynamics. It is worth noting that the loss sorting at step $t$ comes from the model at step $t-1$, which is already protected by previous (I)DP rounds; thus, "using loss order" through adaptive composition introduces no extra privacy cost. The authors further generalize the mechanism into a universal INO-SGM.

**4. Equivalent Optimization Objective: Minimizing "Ordered Weighted Loss" to automatically bias toward hard-to-learn private groups.** Since cumulative clipping thresholds $(c_k)$ fall at different points across iterations, direct analysis is difficult. The authors analyze a "Uniform Privacy Expansion (UPE)" dataset $\hat D$, proving INO-SGD is equivalent to minimizing $L_{f_{\text{tail}}}(\theta;\hat D) = \frac{1}{K}\sum_{k=1}^K w_k\,\ell(\theta; d_{\pi_K(k)})$, where $w$ is a non-increasing sequence of weights $\le 1$. Data with larger losses receive larger weights. This objective can be interpreted as a mixed CVaR (focusing on worst-case loss), a robust objective more corrective than average loss yet more outlier-resistant than max-loss, and an Ordered Weighted Average (OWA) satisfying the Pigou-Dalton principle. Specifically, replacing a larger loss from a more private owner with a smaller loss from a less private owner decreases the total objective. This theoretically suppresses cross-owner utility imbalance, mitigating both MID and BOO.

## Key Experimental Results
Evaluated on MNIST, CIFAR-10, and CIFAR-100 (including CIFAR-100-FV: FISH vs VEHICLE domains) using CNNs by Papernot et al. (including ResNet-18 extensions), comparing against IDP-SGD. Owners with smaller indices have stronger privacy requirements.

### Main Results (Correction of Utility Imbalance, Fig. 5/6)

| Dimension | Observation |
|------|------|
| More Private Group Utility | INO-SGD consistently superior: starts earlier and faster, finishing with accuracy gain of approx. **+10%**. |
| Less Private Group Utility | Hardly decreases (only moderate concessions, Fig. 6 right). |
| Recall Change vs. Privacy Budget | Statistically significant Pearson correlation (**p < 0.001**) on CIFAR-10/100; more private owners see larger gains. |
| Timing of MID | Occurs at approx. 1/4 of training for easy tasks (MNIST). For hard tasks (CIFAR-10), MID hasn't ended before the budget is exhausted, confirming that MID is more harmful under (I)DP. |

### Overall Utility (Whether Sacrificed, Fig. 7)

| Dataset | Overall Validation Loss/Acc |
|--------|-------------------|
| MNIST / CIFAR-10 / CIFAR-100-FV | INO-SGD overall utility is **comparable or slightly superior** to IDP-SGD. |

This indicates that what was discarded was information harmful to the model rather than a waste of privacy budget, suggesting that vanilla IDP-SGD sits at a suboptimal privacy-utility tradeoff.

### Key Findings
- **MID is more harmful than BOO**: While conventional wisdom suggests MID disappears with iterations and isn't serious, (I)DP allows for only limited iterations. Budgets are often exhausted before MID ends, leading to permanently lower utility for minority groups—increasingly evident as task difficulty rises.
- **The more private, the greater the gain**: The recall gain is significantly negatively correlated with the privacy budget, precisely targeting improvements where they are most needed rather than providing a uniform boost.
- **Improved overall utility**: This suggests that discarded low-loss gradient information acts as "interference" for the model, and there is room to improve the privacy-utility tradeoff of IDP-SGD.

## Highlights & Insights
- **The problem identification itself is a contribution**: First to identify and formalize the "IDP-balance-utility trilemma"—even with fully balanced training sets and no natural underrepresentation, individualized privacy preferences **forcefully create** underrepresented groups, orthogonal to traditional data imbalance or DP's disparate impact.
- **The "Continuous Gradient" trick is ingenious**: Reformulating hard filtering (which violates IDP) as an integral soft reweighting of infinitesimal gradient slices is the central technique for bypassing module sensitivity constraints and naturally yields a provable privacy bound.
- **A free lunch**: The privacy cost per step is identical to IDP-SGD, yet it improves both imbalance and overall utility, implying the original method was suboptimal on the tradeoff curve.
- **Multiple interpretations of the optimization objective** (CVaR / Robust Loss / OWA + Pigou-Dalton) connect "compensating private groups" to mature theories of fairness and robust optimization, providing a solid theoretical foundation.

## Limitations & Future Work
- **Balance-utility remains a tradeoff**: Improving the utility of more private owners may slightly lower the utility of less private ones. The authors admit they are still constrained by the IDP-balance-utility trilemma but are closer to its boundary; theoretical characterization of this Pareto frontier is needed.
- **TIF requires prior belief**: The importance function (Beta parameters, tail length $\gamma$) depends on the model owner's settings regarding "which data is important." While tuning guides are provided, it remains an additional manual choice.
- **Medium experimental scale**: Primarily validated on MNIST/CIFAR series with CNN/ResNet-18; has not yet reached large-scale models or more complex real-world deployment scenarios.

## Related Work & Insights
- **IDP / IDP-SGD** (Alaggan 2016; Boenisch 2024): Direct precursors; the SAMPLE/SCALE variants are the objects of diagnosis and improvement.
- **Data/Class Imbalance** (MID: Ye 2021; Cond.(1): Francazi 2023): Provided the analysis framework for MID/BOO, but their corrective measures fail under IDP, highlighting the necessity of this work.
- **Disparate Impact of DP** (Bagdasaryan 2019): Studied how DP magnifies existing imbalances, which is orthogonal and complementary to this work's "IDP creates imbalance out of nothing."
- **Robust/Fair Optimization** (CVaR: Ogryczak 2000; OWA + Pigou-Dalton: Weymark 1981): Provided the theoretical explanation for the equivalent objective of INO-SGD, inspiring the translation of privacy fairness into ordered weighted loss minimization.
- **Insight**: When an "individualized constraint" (privacy, resources, licensing) cannot be symmetrically relaxed, "down-weighting the unimportant" is often easier than "scaling up the disadvantaged" to satisfy hard constraints. This logic is transferable to Federated Learning, personalized resource allocation, and similar scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to formalize utility imbalance induced by IDP; the "continuous gradient soft reweighting" is clever and provably satisfies IDP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete across multiple datasets/models + privacy testing + ablation + Pareto analysis, but scale is small-to-medium, lacking large model and real deployment validation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivations are progressive and theoretical/graphical alignment is clear, but high density of symbols and "continuation" concepts poses a slight barrier to initial reading.
- **Value**: ⭐⭐⭐⭐ Addresses a key fairness blind spot in individualized privacy deployment; the method improves equity and overall utility without increasing privacy costs, offering high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Federated Learning of Quantile Inference under Local Differential Privacy](federated_learning_of_quantile_inference_under_local_differential_privacy.md)
- [\[ICLR 2026\] Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy](beyond_membership_limitations_of_addremove_adjacency_in_differential_privacy.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)

</div>

<!-- RELATED:END -->
