---
title: >-
  [Paper Note] Demystifying the Optimal Fair Classifier in Multi-Class Classification
description: >-
  [ICML 2026][AI Safety][In-processing] This paper provides an analytically tractable form (a closed-form solution with entropy regularization) for the Bayes optimal classifier in multi-class fairness problems. Based on this, it derives a unified algorithmic framework called OptFair: during the training phase, it uses reduction to transform the problem into
tags:
  - ICML 2026
  - AI Safety
  - In-processing
  - Post-processing
date: 2026-05-08
content_hash: 3f8d63e2b9f70338
---
# Demystifying the Optimal Fair Classifier in Multi-Class Classification

**Conference**: ICML 2026  
**arXiv**: [2606.00656](https://arxiv.org/abs/2606.00656)  
**Code**: None  
**Area**: AI Safety / Fairness / Multi-class Classification  
**Keywords**: Fair classification, multi-class classification, Pareto frontier, In-processing, Post-processing

## TL;DR
This paper provides an analytically tractable form (a closed-form solution with entropy regularization) for the Bayes optimal classifier in multi-class fairness problems. Based on this, it derives a unified algorithmic framework called OptFair: during the training phase, it uses reduction to transform the problem into a saddle-point optimization of cost-sensitive cross-entropy; during the deployment phase, it uses a plug-in estimator to solve a convex proximal gradient problem. Both methods theoretically converge to the accuracy-fairness Pareto frontier.

## Background & Motivation

**Background**: Group fairness (DP, EOP, EO) has become a standard constraint in high-stakes decision-making (e.g., healthcare, credit, judiciary). Existing methods either modify the objective during training (in-processing) or adjust the output after inference (post-processing), and these are typically designed independently.

**Limitations of Prior Work**: (1) Fairness metrics are inherently **non-decomposable and non-differentiable**. In multi-class settings, the output shifts from a scalar to a vector on a simplex, making direct adaptations of binary classification approaches clumsy. (2) In-processing mostly relies on **surrogate metrics** (e.g., hinge or Adv loss), which suffer from uncontrollable surrogate gaps and unstable convergence. (3) Post-processing often serves only a single fairness criterion or lacks an explicit characterization of "what the optimal classifier looks like," leaving the performance upper bound unclear. (4) The multi-class fair learning field **lacks an analytical characterization of the Pareto frontier**, making it impossible to determine whether performance drops are due to algorithmic weakness or the inherent nature of the problem.

**Key Challenge**: To achieve a solution that is "general across multiple fairness criteria, compatible with both in/post stages, and capable of approaching the optimum," one must first establish a **Bayes optimal analytical form valid for multi-class classification and multiple DP/EOP/EO constraints**. Otherwise, various implementations can only perform local approximations in the dark.

**Goal**: The paper addresses this in two steps: first, it answers the theoretical question of what the form of the optimal multi-class fair classifier is; then, it provides corresponding in-processing and post-processing algorithms, proving that both converge to the aforementioned optimal solution.

**Key Insight**: DP, EOP, and EO are all formulated as **linear constraints on group-specific confusion matrices $C^a$**, expressed as $|\sum_a \langle D^{a,k}, C^a(h) \rangle| \le \xi$. The constraints are then incorporated into the objective using a Lagrangian. To address analytical intractability, the authors draw inspiration from entropic Optimal Transport (OT) and introduce **entropy regularization** $E(h) = -\mathbb{E}_X [\sum_i h_i \log h_i]$, convexifying the argmax into a softmax to obtain a closed-form solution.

**Core Idea**: A closed-form softmax solution $h^{\lambda^*}_i(x) \propto \exp(\beta^{\lambda^*}_i(x)/\tau)$ for the optimal multi-class fair classifier is derived using the entropy-regularized Lagrangian saddle-point formulation. "Training fitting" and "inference calibration" are reduced to cost-sensitive classification and convex proximal optimization, respectively, unified under the OptFair framework.

## Method

### Overall Architecture

The paper tackles the dual theoretical and algorithmic problem of how to find and approximate the accuracy-fairness optimal classifier in multi-class settings. The approach involves writing the original constrained optimization $\min_h R(h)$ s.t. $|D_k(h)| \le \xi$ as a unified Lagrangian saddle point $L(h, \lambda) = R(h) + \lambda^\top D(h) - \xi \|\lambda\|_1$. It first analytically characterizes the appearance of the optimal classifier and then pursues two paths—training (in-processing) and deployment (post-processing)—to approximate it, proving both converge to the same Pareto frontier. The input is a finite sample $(X, A, Y)$ and a fairness threshold $\xi$, and the output is an attribute-blind randomized classifier $h: \mathcal{X} \to \Delta_m$.

```mermaid
graph TD
    A["Input: Finite samples (X,A,Y) + fairness threshold ξ"] --> B["Unified Linear Constraints + Entropy Regularization<br/>DP/EOP/EO written as confusion matrix linear constraints,<br/>Lagrangian saddle point + entropy regularization → softmax closed-form h^λ"]
    B --> C["In-processing<br/>Cost-sensitive cross-entropy saddle point, primal-dual training fitting"]
    B --> D["Post-processing<br/>Plug-in estimation of η, q_a + convex proximal, calibrating pre-trained models"]
    C --> E["Convergence to the same accuracy-fairness Pareto frontier"]
    D --> E
```

### Key Designs

**1. Unified Linear Constraints + Entropy Regularization: Convexifying the $\arg\max$ optimal solution into a softmax closed-form**

The most difficult aspect of multi-class fairness is that metrics are non-decomposable and non-differentiable, and outputs are vectors on a simplex. This paper first unifies criteria like DP, EOP, and EO as linear constraints on group-specific confusion matrices $C^a$: $|\sum_a \langle D^{a,k}, C^a(h)\rangle| \le \xi$. This allows multi-class and multi-criterion problems to share a single theory. After dualization, Theorem 4.2 provides the optimal solution without regularization: $h^*(x) \in \mathrm{conv}\{e_y : y \in \arg\max_j \beta^{\lambda^*}_j(x)\}$, where the decision vector is $\beta^{\lambda}(x) = \sum_a p_a(x)\, M(a,\lambda)^\top \eta(x,a)$. Here, the reweighting matrix $M(a,\lambda) = I - \frac{1}{\omega_a}\sum_k \lambda_k D^{a,k}$ indicates how each true-prediction pair should be weighted to satisfy fairness constraints if a sample belongs to group $a$.

Since this solution contains an $\arg\max$, the dual optimization is non-differentiable. Borrowing from entropic OT, entropy regularization $-\tau E(h)$ is added to the primal problem, convexifying $\arg\max$ into softmax. Theorem 4.3 provides the closed-form solution $h^{\lambda^*}_i(x) = \exp(\beta^{\lambda^*}_i(x)/\tau) / \sum_j \exp(\beta^{\lambda^*}_j(x)/\tau)$. The dual objective then becomes a convex smooth + L1 structure: $\min_\lambda \tau \mathbb{E}_X [\log \sum_j \exp(\beta^\lambda_j(X)/\tau)] + \xi\|\lambda\|_1$, solvable via standard proximal methods. The temperature $\tau$ controls stochasticity: as $\tau \to 0$, it reverts to a hard $\arg\max$ (consistent with Theorem 4.2), while a moderate $\tau$ ensures near-deterministic inference and smooth training gradients.

**2. In-processing: Reducing fair training to a cost-sensitive cross-entropy saddle-point problem**

During training, $\eta$ and $p_a$ are unknown, so the step "minimize $L(h,\lambda)$" must be reduced to a differentiable classification problem with an explicit calibrated loss for SGD. The paper defines a cost-sensitive loss $\ell_{\mathrm{cal}}(y, f(x;\theta), a, \lambda) = -\sum_i [M'(a,\lambda)]_{y,i}\, \log \mathrm{softmax}_i(f(x;\theta))$, where $M'(a,\lambda) = M(a,\lambda) + \kappa \mathbf{1}_{m\times m}$ includes a constant term to ensure strictly positive entries, making it a valid cost matrix. Theorem 5.1 proves that $h^*(x;f)$ induced by $\arg\min_f \mathbb{E}[\ell_{\mathrm{cal}}]$ is equivalent to the optimal $h^*(x;\beta^\lambda)$, meaning the loss is calibrated for the inner minimization. This addresses the limitation of prior in-processing methods using hinge/adversary surrogates with uncontrollable gaps. Algorithm 1 uses standard primal-dual: $R$ steps of $\theta$ gradients followed by one proximal update for $\lambda$: $\lambda_{t+1} = \mathrm{prox}_{\eta_\lambda(\xi\|\cdot\|_1 + I_{\Lambda})}(\lambda_t + \eta_\lambda D(h_{t+1}))$. Convergence is guaranteed by mixed Nash analysis (Theorem 5.2).

**3. Post-processing: Plug-in estimation + convex proximal for calibrating any pre-trained model**

During deployment, pre-trained scores $\hat\eta$ are available, and the goal is to output a fairly calibrated probabilistic classifier without retraining. The paper trains an auxiliary model $\hat q_a(x) \approx P(A|X, Y)$ and substitutes the sample estimate $\hat\beta^\lambda(x) = [\sum_a \mathrm{Diag}(\hat q_a(x))\, \hat M(a, \lambda)]^\top \hat\eta(x)$ back into the closed-form softmax. The optimal $\hat\lambda^*$ is obtained by solving the empirical dual $\hat H(\lambda) = \hat f(\lambda) + \xi\|\lambda\|_1$. A key benefit is that $\hat q_a$ decouples "attribute-blindness"—traditional post-processing often requires sensitive attributes at inference or only serves a single criterion, whereas this approach does not require true attributes at inference. Proposition 5.5 proves $\hat f(\lambda)$ is convex and L-smooth, so Algorithm 2 converges quickly to the global optimum using proximal gradient descent.

### Loss & Training

In-processing uses $\ell_{\mathrm{cal}}$ (cost-sensitive cross-entropy) + primal-dual optimization. The inner loop updates $\theta$ with step $\eta_\theta$, and the outer loop updates $\lambda$ with $\eta_\lambda = B_\Lambda / (u\sqrt{KT})$ using a prox operator to satisfy $\|\lambda\|_1 \le B_\Lambda$. Post-processing uses proximal gradient to solve $\hat H(\lambda)$. For a deterministic classifier, one can either rerun $\ell_{\mathrm{cal}}$ until convergence with a fixed $\bar\lambda$ (in-processing) or simply take $\arg\max h(x)$ (post-processing). A low temperature $\tau$ ensures the softmax output is nearly one-hot.

## Key Experimental Results

### Main Results

Evaluations on four standard fairness benchmarks (Adult / ENEM / ACSIncome / CelebA; the latter three are multi-class with $\ge 4$ classes) were conducted, scanning $\xi$ for DP and EO criteria to plot accuracy-fairness Pareto curves.

| Phase | Dataset / Criterion | OptFair Performance | Primary Baselines |
|-------|----------------------|---------------------|-------------------|
| In-proc | ENEM / DP | Pareto frontier shifted significantly; DP ~30% lower than sub-optimal at same accuracy | ERM / AdvDebias / Weight-ERM / FairBatch / F-divergence |
| In-proc | ACSIncome / EO | Accuracy ~0.47 at EO ≈ 0.1, significantly higher than baselines (~0.42–0.44) | Same as above |
| Post-proc | CelebA / DP | Accuracy ~0.74–0.76 at same DP; outperforms FairProjection, LinearPost, FRAPPÉ | Same as above |
| Post-proc | Adult / EO | Stable on the outer edge of the frontier across trade-off intervals | Same as above |

Qualitative conclusions: (1) In-processing shows a more pronounced advantage as it directly approximates the theoretical Pareto frontier. (2) On Adult/EO, fairness constraints actually **improved** accuracy, likely by reducing inherent bias.

### Ablation Study

On ENEM/ACSIncome, in-processing was trained to a certain fairness threshold and then followed by post-processing (In-Post-1 / In-Post-2 with different thresholds):

| Configuration | Description | Result |
|---------------|-------------|--------|
| OptFair-in (only) | In-processing only | Upper bound, closest to Pareto frontier |
| OptFair-post (only) | Post-processing only | Close to in-only, slightly inferior |
| In-Post-1 / In-Post-2 | In-proc training followed by post-proc calibration | Falls between the two; **no additive gain** |

### Key Findings

- **Non-additivity of In + Post**: In-processing debiases at the representation level, while post-processing modifies the output distribution. Since they operate on different scopes, cascading them usually does not yield further improvements but rather interpolates the two curves.
- In scenarios like Adult/EO, adding fairness constraints **increases accuracy**, suggesting that data bias causes ERM to learn sub-optimal decision boundaries; fairness constraints act as a regularizer.
- A smaller entropy regularization temperature $\tau$ results in outputs closer to deterministic and higher accuracy ceilings, but at the cost of less stable gradients. Theorem 5.6 provides the optimal order for $\tau$ to balance the $\tau \log m$ and $1/\tau$ terms.

## Highlights & Insights

- **Dual Role of Entropy Regularization + Lagrangian**: This allows the "optimal fair classifier" to transition from a convex hull containing an $\arg\max$ to a closed-form softmax (analytical), and transforms the dual problem into a convex + L-smooth + L1 problem (convex optimization). This logic can be migrated to any discrete output problem involving linear constraints and non-differentiable decisions (e.g., ranking or segmentation fairness).
- **Unification via $\beta^\lambda$ and $M(a,\lambda)$**: Both in-processing and post-processing algorithms are linked by the same components. This means an in-processing warm-start $\bar\lambda$ can be used during training, followed by post-processing fine-tuning during deployment, which is architecturally elegant.
- **Cost-Sensitive Loss**: Converting fairness constraints into a calibrated cross-entropy provides a cleaner paradigm for the in-processing community than surrogate-based losses. The dual variable $\lambda$ naturally provides cost weights for each $(a, y, \hat y)$ triplet.

## Limitations & Future Work

- **Quality of the auxiliary model $\hat q_a$**: The performance of post-processing is limited by the error in $\hat q_a$. Theorem 5.6's $\epsilon_1$ term includes $\|q_a - \hat q_a\|_1$. When groups are imbalanced or attributes are hard to predict, this term dominates the worst-case bound.
- **Missing In + Post joint ablation on vision data**: The authors note that sensitive attributes are difficult to feed directly into image data; thus, ablation was only performed on tabular data.
- **Lack of automated temperature $\tau$ selection**: In experiments, $\tau$ was chosen empirically. There is no cross-validation procedure or analysis on whether the optimal $\tau$ varies across different fairness criteria.
- **Practical controllability with joint multiple criteria ($K \ge 2$)**: While theory supports multiple simultaneous constraints, experiments only demonstrate single-criterion trade-offs. The nature of the Pareto surface under simultaneous DP and EO constraints remains unexplored.

## Related Work & Insights

- **vs. Agarwal et al. 2018 (Reductions for binary fairness)**: This work is the multi-class extension. However, the inner loop's calibrated loss evolves from 0/1 importance-weighting to a **cost-sensitive softmax form** (Theorem 5.1), with entropy regularization added to ensure solvability.
- **vs. Xian & Zhao 2024 / Denis et al. 2024 (Multi-class post-processing)**: These assume continuous output distributions and are mostly attribute-aware. This paper uses entropic relaxation to remove the continuity assumption and achieves attribute-blind inference via the auxiliary model $\hat q_a$.
- **vs. FairProjection / LinearPost / FRAPPÉ**: These either target specific criteria or lack a characterization of the "optimal classifier." OptFair-post writes the optimal solution as a closed-form softmax, making its optimization goal "approaching the optimum" rather than "heuristic calibration."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing entropic regularization (OT-style) to characterize the multi-class fair Bayes optimum and unifying in/post stages fills a significant theoretical gap.
- **Experimental Thoroughness**: ⭐⭐⭐ Dataset and baseline selection are complete, but it lacks joint multi-criterion experiments and joint ablation on image data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear correspondence between Theorems, Algorithms, and Experiments; consistent notation.
- **Value**: ⭐⭐⭐⭐ The fairness ML community can directly reuse this calibrated loss and plug-in framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](../../CVPR2026/ai_safety/your_classifier_can_do_more_towards_balancing_the.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)

</div>

<!-- RELATED:END -->
