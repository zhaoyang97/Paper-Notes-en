---
title: >-
  [Paper Note] Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification
description: >-
  [ICLR 2026][AI Safety][Paper Note] This paper reformulates the optimal fairness-accuracy trade-off achievable under a given model architecture (model-specific Pareto frontier) as a convex optimization problem over confusion vectors. It proves that existing post-processing frontiers are suboptimal and proposes a last-layer retraining framework with group
tags:
  - ICLR 2026
  - AI Safety
date: 2026-05-08
content_hash: f6985dba77928e6e
---
# Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=L8pyycR4wW](https://openreview.net/forum?id=L8pyycR4wW)  
**Code**: https://github.com/cjy24/Pareto-frontier  
**Area**: AI Safety / Fairness  
**Keywords**: Fair Classification, Pareto Frontier, Fairness-Accuracy Trade-off, Convex Optimization, Last-layer Retraining

## TL;DR
This paper reformulates the optimal fairness-accuracy trade-off achievable under a given model architecture (model-specific Pareto frontier) as a convex optimization problem over confusion vectors. It proves that existing post-processing frontiers are suboptimal and proposes a last-layer retraining framework with group-dependent biases, theoretically demonstrating its strict superiority over post-processing baselines such as randomized flipping.

## Background & Motivation
**Background**: As machine learning is widely deployed for decision-making, fairness has become a primary requirement. The community has established various fairness concepts (demographic parity DP, equal opportunity EOp, equalized odds EOd) and intervention methods. Extensive empirical observations show that improving fairness often comes at the cost of accuracy (fairness-accuracy trade-off), and satisfying multiple fairness concepts simultaneously can lead to conflicts (the "impossibility theorem"). To quantify "how much accuracy can be preserved as fairness constraints tighten for a fixed network architecture," prior work uses the **model-specific (MS) Pareto frontier**—taking ResNet-50 as an example, "model-specific" refers to the best trade-off curve achievable by all fairness interventions applied to ResNet-50.

**Limitations of Prior Work**: The authors point out critical flaws in previous MS frontier characterizations. Kim et al. (2020) and Jang et al. (2022) use **post-processing (randomized flipping)** to approximate the frontier, which faces three issues: ① Post-processing itself is suboptimal, leading to a "frontier" that is too low, such that other methods' curves may even exceed it—rendering such a "frontier" invalid as an upper bound baseline; ② The feasible region for post-processing is empirically determined by confusion matrices of each sensitive group, but distribution shifts between training and test sets mean solutions viable on training data may fail on test data, making the frontier unreliable; ③ Even when replacing this with in-processing (Dehdashtian et al. 2024) or directly optimizing Bayesian optimal classifiers (Wang et al. 2024, corresponding to model-agnostic frontiers), the former is admitted to be suboptimal by its authors, while the latter relies on high-variance joint posterior estimation.

**Key Challenge**: The fundamental problem is that **previous approaches coupled the "search for the optimal frontier" with specific fairness intervention algorithms**, thus the quality of the frontier was limited by the suboptimality of those algorithms. Furthermore, almost all discussions only cover the fairness-accuracy trade-off, while the **Pareto trade-off between different fairness concepts has never been formally characterized**.

**Goal**: (1) Provide an MS Pareto frontier that is independent of specific intervention algorithms and can approximate the "supremum of the best reachable accuracy"; (2) Extend this characterization to the trade-off between DP and EOd and explain how it varies with accuracy; (3) Develop an intervention method based on the new formulation that truly approaches the frontier.

**Key Insight**: The authors follow the key observation of Kim et al. (2020)—**accuracy and various fairness concepts can be expressed as linear transformations of confusion matrices (confusion vectors)**. Consequently, the difficult optimization problem regarding network parameters $\theta$ can be reformulated as a **convex optimization** over low-dimensional confusion vectors $z$, bypassing the pitfalls of algorithmic suboptimality.

**Core Idea**: Reformulate the Pareto optimal trade-off as a "constrained convex optimization over the confusion vector $z$," using the accuracy of vanilla training as an upper bound to lock the feasible region, thereby directly approximating the supremum. Subsequently, utilize last-layer retraining with group-dependent biases to approach this frontier.

## Method

### Overall Architecture
The work follows two tracks. **The first is "Measurement"**: how to plot a truly credible MS Pareto frontier. The authors write "the highest accuracy achievable given a fairness violation $\le \epsilon$" as a convex optimization over the confusion vector $z$, then uniformly sample $\epsilon$ in $[0,1]$ and solve point-by-point to construct the frontier. The same formulation can be adapted to find the "optimal trade-off curve between two fairness concepts at a fixed accuracy." **The second is "Intervention"**: inspired by this formulation, a last-layer retraining framework is proposed. It freezes the encoder and projection head, updating only the final linear classification head with added **group-dependent biases** $b_a$. This directly optimizes for fairness-accuracy objectives and is theoretically proven to be strictly superior to randomized flipping post-processing.

The key to the measurement track lies in three linear representations. Let the confusion vector for group $a$ be $z=[\text{TPR}_0,\text{TNR}_0,\text{TPR}_1,\text{TNR}_1]^T$, with base rates $\alpha_a:=\Pr[Y=1\mid A=a]$ and the marginal distribution of sensitive attributes $\beta:=\Pr[A=1]$. Then:

$$\text{Acc}=A_c z,\quad A_c=[\alpha_0(1-\beta),\,(1-\alpha_0)(1-\beta),\,\alpha_1\beta,\,(1-\alpha_1)\beta]$$

$$\text{DP}=|A_{DP}z+A'_{DP}(1-z)|,\qquad \text{EOd}=\|A_{EOd}z\|_1$$

where $A_{DP}=[\alpha_0,0,-\alpha_1,0]$, $A'_{DP}=[0,(1-\alpha_0),0,-(1-\alpha_1)]$, and $A_{EOd}=\begin{bmatrix}1&0&-1&0\\0&1&0&-1\end{bmatrix}$. With these linear expressions, the insoluble problem regarding $\theta$ becomes a convex problem regarding $z$.

### Key Designs

**1. Reformulating MS Pareto Frontier as Convex Optimization over Confusion Vectors**

Addressing the pain point that "frontiers are tied to specific algorithms and thus suboptimal," the authors stop optimizing specific fairness interventions and instead solve for the supremum in the confusion vector space. The MS fairness-accuracy trade-off is written as:

$$\arg\max_{z\in K} A_c z,\quad \text{s.t. } \|A_{EOd}z\|_1\le \epsilon \quad(\text{or } |A_{DP}z+A'_{DP}(1-z)|\le\epsilon),$$

Since $A_c, A_{EOd}, A_{DP}$ are determined by the **test set** marginal distribution and $z$ is only 4-dimensional, this is a low-dimensional convex problem that can be solved instantly. By taking $T$ uniform samples of $\epsilon$ in $[0,1]$ and solving each, the resulting $(\epsilon_i, A_c z_i)$ pairs form the frontier. Because no specific fairness relaxation is imposed on the classifier $f$, this frontier naturally corresponds to the **supremum** of the Pareto optimal trade-offs, rather than a lower curve reachable by a specific algorithm.

The authors also prove that old frontiers are indeed suboptimal (Lemma 1): Under the assumption that the ROC curve is concave (Assumption 1), for any solution $\tilde z$ obtained from the relaxation (Eq. 4) in Kim et al. (2020) within the feasible region $\hat K$ constructed by combining randomized flipping and threshold adjustment, **there always exists a strictly better $\hat z=\tilde z+[0,\delta,0,\delta]^T$** such that $A_c\tilde z< A_c\hat z$ while EOd remains unchanged, where:

$$\delta=\min\{|\Phi_0^{-1}([1,0,0,0]\tilde z)-(1-[0,1,0,0]\tilde z)|,\ |\Phi_1^{-1}([0,0,1,0]\tilde z)-(1-[0,0,0,1]\tilde z)|\}.$$

This theoretically demonstrates that old frontiers can be strictly exceeded and thus should not serve as baselines for evaluating trade-offs.

**2. Utilizing Vanilla Training Accuracy Bounds to Lock the Feasible Region $K$**

For the convex optimization to be credible, the definition of the feasible region $K$ is critical. The authors use two "vanilla training optimal" observations to tighten the bound. The first is the **overall accuracy bound**: imposing non-trivial fairness constraints cannot result in lower classification loss than unconstrained training, hence $A_c z\le A_c z_b$, where $z_b$ is the confusion vector of the baseline model. The second is the **group-specific accuracy bound**: accuracy within each sensitive group should not increase due to fairness constraints, so $A^a_c z\le A^a_c z^a_b$, where $z^a_b$ is the maximum group-specific accuracy obtained from either "vanilla training on that group alone" or "vanilla training on the full dataset" (considering information overlap between groups). Collectively, $K$ is defined as:

$$K:=\{z\mid 0\le z\le 1;\ A_c z_b\ge A_c z;\ A^a_c z^a_b\ge A^a_c z\}.$$

To approximate the best test accuracy in the feasible hypothesis space $H'$ determined by the training data, $z_b$ and $z^a_b$ are estimated through multiple random initializations of $f$. This step is the foundation of the measurement's credibility: it aligns the frontier with the "upper limit allowed by training capability" rather than being artificially lowered by a feasible region on the test set.

**3. Characterizing Trade-offs Between Fairness Concepts and "Impossibility" Boundaries**

Addressing the gap where "the trade-off between DP and EOd has never been quantified," the authors apply the same formulation to fairness-fairness trade-offs:

$$\arg\min_{z\in K}|A_{DP}z+A'_{DP}(1-z)|,\quad \text{s.t. }\|A_{EOd}z\|_1\le\epsilon,\ A_c z\ge\eta.$$

By sweeping both $\epsilon$ and the accuracy lower bound $\eta$, one can plot a family of contours in the EOd-DP plane that change with accuracy. Each line contains a Pareto elbow (where no other point can simultaneously improve DP and EOd at that accuracy). More importantly, Lemma 2 provides an impossibility boundary: **If $\alpha_0\ne\alpha_1$, the optimal classifier that simultaneously achieves DP and EOd must have accuracy that degrades to that of a constant predictor.** This implies that when evaluating fairness interventions, one should not expect both DP and EOd to be minimal simultaneously; instead, one should check if the EOd-DP curve is close to its own Pareto optimum—providing a more reasonable evaluation standard for scenarios requiring both fairness concepts.

**4. Last-layer Retraining with Group-dependent Bias and its Theoretical Superiority**

The measurement provides the "location of the frontier," but a method is needed to reach it. The authors view the classifier as an encoder $g$ plus a classification head $h$, further splitting $h$ into a projection head $h_1$ (reducing to low dimensions) and a linear head $h_2$. **Only $h_2$ is updated, while $g$ and $h_1$ are frozen**, keeping the optimization linear and controllable. The core action is appending group indicator bits to the projected features:

$$\hat x_i=[\hat x^0_i,\hat x^1_i,\ \mathbb{1}[a_i=0],\ \mathbb{1}[a_i=1]],$$

Thus $b_a$ in $h_2$ parameters $[w,b_0,b_1]$ becomes a **group-dependent bias**, equivalent to setting different thresholds for each group but with higher degrees of freedom than pure threshold adjustment. The optimization objective is:

$$\arg\max_{w,b_0,b_1}\ \text{Acc}-\lambda\,\text{EOd}\quad(\text{or } \text{Acc}-\lambda\,\text{DP}).$$

When the weight vector is fixed to the baseline $w^*$, the framework degrades to the threshold method $h_2(x_i)=\sigma(h_2^*(x_i)+c_a)$ where $c_a=b_a-b^*$ is the group-specific threshold; allowing $w$ to vary provides greater flexibility. Theoretically, Theorem 1 proves that under Assumption 1, **the EOd optimal point (the intersection with highest accuracy when EOd=0) of this method is strictly superior to the EOd optimal point of randomized flipping.** Furthermore, under the assumption that sub-group logits follow equal-variance Gaussians $l_i\sim\mathcal N(\mu_{ya},s^2)$, Lemma 3 provides a closed-form for the group-specific threshold $c^*_a$ at EOd optimality, and Theorem 2 provides an **analytical estimate of the accuracy drop** at EOd optimality, allowing for the estimation of the accuracy loss supremum on a dataset using only latent representations and $w^*$ without actually training $h_2$.

### Loss & Training
The measurement side requires no training, directly solving a 4-dimensional convex optimization for each sampled $\epsilon_i$. The intervention side retrains only the last-layer classification head $h_2$ with the objective $\text{Acc}-\lambda\,\text{EOd}$ or $\text{Acc}-\lambda\,\text{DP}$ (modified to consider both in joint EOd-DP scenarios). $\lambda$ serves as a tunable weight; sweeping $\lambda$ yields the method's own trade-off curve.

## Key Experimental Results

### Main Results
Verified on four datasets: COMPAS, Adult, CelebA (binary), and Drug (multi-class), using accuracy as utility and DP/EOd as fairness metrics. Comparison baselines include FACT, Eq. Odds, DFR, SELF, G-STAR, FOC, and other post-processing/last-layer retraining methods.

| Trade-off Type | Ours Frontier/Method | Key Finding |
|----------------|----------------------|-------------|
| EOd-Accuracy (Fig 2) | Ours-EOd-frontier | The frontier is nearly **horizontal**, meaning tightening EOd barely drops accuracy; frontiers for FACT and FACT+G-STAR are crossed by multiple method curves, proving they are not true upper bounds. |
| DP-Accuracy (Fig 3) | Ours-DP-frontier | Accuracy **drops significantly** as DP constraints tighten, confirming the inherent DP-Accuracy trade-off. |
| EOd-DP (Fig 4) | Ours (Joint Objective) | Clear conflict between both fairness concepts when accuracy is higher than a constant predictor; as accuracy relaxes, the frontier approaches the EOd axis and the conflict disappears. |

### Ablation Study

| Configuration | Key Effect | Description |
|---------------|------------|-------------|
| Ours (Full group-dependent bias $b_a$) | Curve closest to MS Pareto frontier | Complete method. |
| Setting $w=w^*$ (Degrades to G-STAR thresholding) | EOd optimal point is worse or merely equal | The degrees of freedom from allowing $w$ to vary are the source of gain. |
| FACT / Eq.Odds / DFR / SELF | Further from the frontier | Restricted by the suboptimality of post-processing or retraining. |

(Note: The detailed numerical tables are in Appendices 9/10; the main text presents trends via the frontier curves in Figures 2-4.)

### Key Findings
- **EOd-Accuracy is not an inherent trade-off**: Since EOd measures differences in error rates rather than the predictions themselves, eliminating group differences does not necessarily lower accuracy—an anti-intuitive but theoretically supported conclusion.
- **DP-Accuracy is a true trade-off**: Consistent with Zhao & Gordon (2022), accuracy decreases as DP becomes stricter when $\alpha_0\ne\alpha_1$.
- **Group-dependent bias is the primary source of gain**: Allowing $w$ to vary is more flexible than fixing $w^*$ (pure thresholding), and the EOd optimal point is strictly superior (Theorem 1).
- **Old frontiers can indeed be surpassed**: Methods like FACT and FACT+G-STAR are crossed by empirical method curves, validating Lemma 1.

## Highlights & Insights
- **"Linearized Confusion Vector $\to$ Convex Optimization" is the core lever**: Compressing an insoluble problem over $\theta$ into a 4-dimensional $z$ allows for instantaneous, credible solutions, serving as the pivot for the entire work.
- **Using vanilla training as an upper bound for the feasible region** is clever: It avoids defining the frontier using a specific intervention algorithm, thereby bypassing algorithmic suboptimality and directly approaching the supremum.
- **Quantifying the impossibility theorem**: Lemma 2 does not just state that "DP and EOd are incompatible," but specifies that "simultaneous satisfaction results in accuracy degrading to a constant predictor," leading to the more reasonable evaluation standard of checking if a curve is close to its own Pareto optimum.
- **Transferability of group-dependent bias**: Appending group indicators to last-layer features to allow independent biases is a lightweight trick transferable to other last-layer retraining tasks requiring group-level calibration.

## Limitations & Future Work
- **Focus limited to model-specific, not model-agnostic**: The frontier is tied to a given network architecture; the authors list MA Pareto optimal trade-offs as a future direction.
- **Theoretical reliance on specific assumptions**: Concave ROC curves (Assumption 1) and equal-variance Gaussian logits—the latter is a strong assumption, though partially defended by observing variance convergence under vanilla training.
- **Limited coverage of fairness concepts**: Primarily discusses group fairness like DP and EOd; individual fairness and min-max fairness are not included in the framework.
- **Distribution shift not considered**: Test set marginal distributions must be estimable; "Fairness-Accuracy trade-offs under distribution shift" is reserved for future work.

## Related Work & Insights
- **vs FACT / G-STAR (Kim 2020 / Jang 2022)**: They use randomized flipping/threshold post-processing to approximate frontiers. This paper proves such frontiers are suboptimal (Lemma 1) and potentially infeasible on test sets. This paper calculates the supremum directly on confusion vectors, making the frontier tighter and more credible, while extending the trade-off to inter-fairness concept scenarios.
- **vs Wang et al. (2024) (model-agnostic frontier)**: They optimize confusion matrices for Bayesian optimal classifiers directly but rely on high-variance joint posterior estimation. This paper takes a model-specific route, using vanilla training bounds to avoid posterior estimation uncertainty.
- **vs DFR / SELF (Last-layer retraining)**: DFR relies on feature reweighting and SELF on selective fine-tuning, both focusing on EOd-Accuracy. This paper's group-dependent bias offers higher degrees of freedom, theoretical guarantees of superior EOd optimality (Theorem 1), and the ability to handle DP and joint EOd-DP trade-offs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulating the Pareto frontier as a confusion vector convex optimization and quantifying inter-fairness trade-offs is a fresh and self-consistent perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across four datasets including binary and multi-class tasks with clear frontier comparisons, though numerical tables are mostly relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐ Complete theoretical derivation chain and consistent notation, though high formula density might pose a steep curve for non-theoretical readers.
- Value: ⭐⭐⭐⭐⭐ Provides a credible upper bound baseline and a more reasonable evaluation method for fairness interventions, contributing significant methodology to fair ML evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](../../ICML2026/ai_safety/demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICLR 2026\] Fair Classification by Direct Intervention on Operating Characteristics](fair_classification_by_direct_intervention_on_operating_characteristics.md)
- [\[ICLR 2026\] Fair Conformal Classification via Learning Representation-Based Groups](fair_conformal_classification_via_learning_representation-based_groups.md)
- [\[ICML 2026\] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](../../ICML2026/ai_safety/fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)
- [\[ICLR 2026\] Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models](rethinking_lora_for_privacy-preserving_federated_learning_in_large_models.md)

</div>

<!-- RELATED:END -->
