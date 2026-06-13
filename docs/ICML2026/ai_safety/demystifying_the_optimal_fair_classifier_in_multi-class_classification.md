---
title: >-
  [Paper Note] Demystifying the Optimal Fair Classifier in Multi-Class Classification
description: >-
  [ICML 2026][AI Safety][Fair classification] This paper provides an analytically tractable form (a closed-form solution with entropy regularization) of the Bayes optimal classifier for multi-class fair classification prob…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Fair classification"
  - "Multi-class"
  - "Pareto frontier"
  - "In-processing"
  - "Post-processing"
date: 2026-05-08
content_hash: 1af336f006c2e940
---

# Demystifying the Optimal Fair Classifier in Multi-Class Classification

**Conference**: ICML 2026  
**arXiv**: [2606.00656](https://arxiv.org/abs/2606.00656)  
**Code**: None  
**Area**: AI Safety / Fairness / Multi-Class Classification  
**Keywords**: Fair classification, Multi-class, Pareto frontier, In-processing, Post-processing

## TL;DR
This paper provides an analytically tractable form (a closed-form solution with entropy regularization) of the Bayes optimal classifier for multi-class fair classification problems. Based on this, it derives a unified pair of algorithms, OptFair: during the training phase, it uses a reduction to transform the problem into a saddle-point optimization of cost-sensitive cross-entropy; during the deployment phase, it uses plug-in estimation to solve a convex proximal gradient problem. Both are theoretically proven to converge to the accuracy-fairness Pareto frontier.

## Background & Motivation

**Background**: Group fairness (DP, EOP, EO) has become a standard constraint in high-stakes decision-making (healthcare, credit, judiciary). Existing methods either modify the objective during training (in-processing) or adjust the output after inference (post-processing), but these are typically designed independently.

**Limitations of Prior Work**: (1) Fairness metrics are inherently **non-decomposable and non-differentiable**; in multi-class settings, the output changes from a scalar to a vector on a simplex, making direct adaptations of binary classification approaches clumsy. (2) In-processing mostly relies on **surrogate metrics** (hinge/Adv loss), leading to uncontrollable surrogate gaps and unstable convergence. (3) Post-processing methods either serve a single fairness criterion or lack an explicit characterization of "what the optimal classifier looks like," leaving the performance upper bound unclear. (4) The entire field of multi-class fair learning **lacks an analytical characterization of the Pareto frontier**, making it impossible to determine whether a drop in performance is due to algorithmic weakness or the problem's inherent nature.

**Key Challenge**: To be "universal across multiple fairness criteria and both in/post phases, while approaching optimality," one must first have a **Bayes optimal analytical form that holds for multi-class and various DP/EOP/EO definitions**. Otherwise, various implementations can only perform local approximations blindly.

**Goal**: Solve this in two steps: first, answer the theoretical question "what is the form of the optimal multi-class fair classifier"; second, provide corresponding in-processing and post-processing algorithms and prove they both converge to the aforementioned optimal solution.

**Key Insight**: Express DP/EOP/EO as **linear constraints of the group-specific confusion matrix $C^a$**: $|\sum_a \langle D^{a,k}, C^a(h) \rangle| \le \xi$. Then, use a Lagrangian to incorporate constraints into the objective. To address analytical intractability, borrow insights from entropic OT to add **entropy regularization** $E(h) = -\mathbb{E}_X [\sum_i h_i \log h_i]$, convexifying the argmax into a softmax to obtain a closed-form expression.

**Core Idea**: Use the entropy-regularized Lagrangian saddle-point formulation to provide a softmax closed-from solution for the multi-class fair optimal classifier, $h^{\lambda^*}_i(x) \propto \exp(\beta^{\lambda^*}_i(x)/\tau)$. Further reduce "training fitting" and "inference calibration" to cost-sensitive classification and convex proximal optimization, respectively, unifying them under the OptFair framework.

## Method

### Overall Architecture

The entire pipeline revolves around a **unified Lagrangian saddle-point problem**:

The input consists of finite samples $(X, A, Y)$ and a fairness threshold $\xi$; the output is an attribute-blind randomized classifier $h: \mathcal{X} \to \Delta_m$.

Step 1 (Theoretical Characterization): Formulate the original problem $\min_h R(h)$ s.t. $|D_k(h)| \le \xi$ as a Lagrangian $L(h, \lambda) = R(h) + \lambda^\top D(h) - \xi \|\lambda\|_1$, and prove strong duality holds. Theorem 4.2 shows that the optimal solution without entropy regularization takes the form $h^*(x) \in \mathrm{conv}\{e_y : y \in \arg\max_j \beta^{\lambda^*}_j(x)\}$, where $\beta^{\lambda}(x) = \sum_a p_a(x)\, M(a,\lambda)^\top \eta(x,a)$ and $M(a,\lambda) = I - \frac{1}{\omega_a}\sum_k \lambda_k D^{a,k}$.

Step 2 (Analytical Formulation): Direct dual optimization on $\arg\max$ is non-differentiable, so entropy regularization $-\tau E(h)$ is added to the primal objective. Theorem 4.3 provides the closed-form softmax solution $h^{\lambda^*}_i(x) = \exp(\beta^{\lambda^*}_i(x)/\tau) / \sum_j \exp(\beta^{\lambda^*}_j(x)/\tau)$. The dual objective becomes $\min_\lambda \tau \mathbb{E}_X [\log \sum_j \exp(\beta^\lambda_j(X)/\tau)] + \xi\|\lambda\|_1$, which is a convex-smooth + L1 problem solvable by standard proximal methods.

Step 3 (Implementation): Since $\eta, p_a$ are unknown during training, reduction-based in-processing is used (Algorithm 1). During deployment, with pre-trained scores $\hat\eta$ available, plug-in estimation is used for post-processing (Algorithm 2). Both paths theoretically converge to the same Pareto frontier.

### Key Designs

1.  **Unified Characterization of Multi-class Fairness + Entropy-Regularized Closed-Form Bayes Optimal Classifier**:
    - **Function**: Unifies fairness criteria like DP/EOP/EO into linear constraints $|\sum_a \langle D^{a,k}, C^a(h)\rangle| \le \xi$ on group-specific confusion matrices $C^a$. It then uses entropy regularization to convexify the argmax-based optimal solution into a softmax, making it **analytically tractable**.
    - **Mechanism**: In the decision vector $\beta^\lambda(x) = \sum_a p_a(x)\, M(a,\lambda)^\top \eta(x,a)$, $M(a,\lambda)$ represents "how each ground truth-prediction pair should be reweighted for group $a$ to satisfy fairness constraints." The softmax temperature $\tau$ controls stochasticity; $\tau \to 0$ recovers the hard $\arg\max$ (Theorem 4.2), while a moderate $\tau$ ensures near-determinism during inference and smooth gradients during training.
    - **Design Motivation**: (1) Linear constraints allow multiple criteria to share a single theory. (2) Entropy regularization bypasses the non-differentiability of $\arg\max$ and makes $\hat H(\lambda) = \hat f(\lambda) + \xi\|\lambda\|_1$ naturally convex + L1, facilitating proximal gradient solving. (3) The form reduces to classical threshold rules in the binary case (Menon & Williamson 2018), ensuring theoretical consistency.

2.  **In-processing: Reducing Fair Training to a Cost-Sensitive Cross-Entropy Saddle-Point**:
    - **Function**: Reduces the "$\min_h L(h,\lambda)$" step into a **classification problem with an explicit calibrated loss** when $\eta, p_a$ are unknown, enabling direct training with SGD.
    - **Mechanism**: Defines a cost-sensitive loss $\ell_{\mathrm{cal}}(y, f(x;\theta), a, \lambda) = -\sum_i [M'(a,\lambda)]_{y,i}\, \log \mathrm{softmax}_i(f(x;\theta))$, where $M'(a,\lambda) = M(a,\lambda) + \kappa \mathbf{1}_{m\times m}$, with a constant term ensuring strictly positive entries. Theorem 5.1 proves that $h^*(x;f)$ from $\arg\min_f \mathbb{E}[\ell_{\mathrm{cal}}]$ is equivalent to the optimal $h^*(x;\beta^\lambda)$, meaning the loss is calibrated for the inner min. Algorithm 1 uses standard primal-dual: $R$ steps of $\theta$ gradients per round, followed by a proximal $\lambda$ update $\lambda_{t+1} = \mathrm{prox}_{\eta_\lambda(\xi\|\cdot\|_1 + I_{\Lambda})}(\lambda_t + \eta_\lambda D(h_{t+1}))$.
    - **Design Motivation**: Previous in-processing methods used surrogate objectives with uncontrollable gaps. This method provides a **calibrated loss for the original saddle problem**. Combined with mixed Nash convergence analysis (Theorem 5.2), the strategy $(\bar h_T, \bar \lambda_T)$ converges to an approximate equilibrium as $T \to \infty$. Theorem 5.3 further provides a generalization bound of $O(\gamma_d(N, m^2/\delta))$, quantifying the distance to the Pareto frontier relative to training and data scale.

3.  **Post-processing: Solving the Convex Proximal Problem via Plug-in Estimation**:
    - **Function**: Outputs a fairness-calibrated probabilistic classifier given any pre-trained model $\hat\eta$ and an auxiliary model $\hat q_a(x) \approx P(A|X, Y)$ without retraining.
    - **Mechanism**: Substitute the sample estimate $\hat\beta^\lambda(x) = [\sum_a \mathrm{Diag}(\hat q_a(x))\, \hat M(a, \lambda)]^\top \hat\eta(x)$ into the closed-form softmax (Eq. 15). The optimal $\hat\lambda^*$ is obtained by solving the empirical dual $\hat H(\lambda) = \hat f(\lambda) + \xi\|\lambda\|_1$. Proposition 5.5 proves $\hat f(\lambda)$ is **convex and L-smooth**, allowing Algorithm 2 to converge quickly via proximal gradient descent.
    - **Design Motivation**: Traditional post-processing is often attribute-aware (requiring sensitive attributes at inference) or only valid for a single criterion. The auxiliary model $\hat q_a$ decouples the "attribute-blind" requirement, while the convex + L1 structure guarantees global optimality. Theorem 5.6 decomposes error into three terms: $\epsilon_1$ (auxiliary model bias), $\epsilon_2$ (finite samples), and $\epsilon_3$ (frequency estimation bias), proving a worst-case bound of order $O(\sqrt\epsilon)$ with a tuned $\tau$.

### Loss & Training

In-processing uses $\ell_{\mathrm{cal}}$ (cost-sensitive cross-entropy) + primal-dual optimization: the inner loop learns $\theta$ with $\eta_\theta$, while the outer loop updates $\lambda$ with $\eta_\lambda = B_\Lambda / (u\sqrt{KT})$ using a prox move to satisfy $\|\lambda\|_1 \le B_\Lambda$. Post-processing solves $\hat H(\lambda)$ via proximal gradient. For a deterministic classifier: in-processing runs $\ell_{\mathrm{cal}}$ to convergence with fixed $\bar\lambda$; post-processing uses $\arg\max h(x)$ directly. A small temperature $\tau$ makes the softmax output nearly one-hot.

## Key Experimental Results

### Main Results

Evaluated on four standard fairness benchmarks (Adult / ENEM / ACSIncome / CelebA, where the latter three are multi-class with $\ge 4$ classes), scanning $\xi$ under DP and EO to plot accuracy-fairness Pareto curves (closer to top-left is better).

| Stage | Dataset / Criterion | OptFair Performance | Main Baselines |
| :--- | :--- | :--- | :--- |
| In-proc | ENEM / DP | Pareto frontier clearly shifted outward; DP ~30% lower than next-best at same accuracy | ERM / AdvDebias / Weight-ERM / FairBatch / F-divergence |
| In-proc | ACSIncome / EO | Significantly higher accuracy (~0.47) than baselines (~0.42–0.44) at EO ≈ 0.1 | Same as above |
| Post-proc | CelebA / DP | Accuracy ~0.74–0.76 at same DP, better than FairProjection, LinearPost, FRAPPÉ | Same as above |
| Post-proc | Adult / EO | Stays consistently on the outer edge of the frontier across trade-off regions | Same as above |

Qualitative Conclusion: (1) In-processing advantages are more pronounced as it directly approaches the theoretical Pareto frontier; (2) In Adult/EO, fairness constraints actually **improve** accuracy by reducing inherent bias.

### Ablation Study

On ENEM/ACSIncome, in-processing was trained to a fairness threshold, followed by post-processing (In-Post-1 / In-Post-2 with different thresholds) and compared to single-stage versions:

| Configuration | Description | Result |
| :--- | :--- | :--- |
| OptFair-in (only) | In-processing only | Upper bound, closest to Pareto frontier |
| OptFair-post (only) | Post-processing only | Close to in-only, slightly lower |
| In-Post-1 / In-Post-2 | In-processing to threshold then Post-calibration | Falls between the two; **no additive gain** |

### Key Findings

- In + Post are not additive: In-processing debiases at the representation layer, while post-processing modifies the output distribution. They act on **different domains**, and serializing them typically does not yield further improvements, merely interpolating between the two curves.
- In scenarios like Adult/EO, adding fairness constraints **actually improves accuracy**, suggesting ERM learns suboptimal boundaries due to data bias; fairness constraints act as a form of regularization.
- A smaller entropy regularization temperature $\tau$ results in more deterministic outputs and higher accuracy upper bounds but less stable gradients. Theorem 5.6 provides the optimal order for $\tau$ to balance the $\tau \log m$ and $1/\tau$ terms.

## Highlights & Insights

- **Dual Role of Entropy Regularization + Lagrangian**: This approach transforms the "optimal fair classifier" from a convex hull containing $\arg\max$ into a closed-form softmax (analytical) and makes the dual problem convex + L-smooth + L1 (optimizable). This framework can be migrated to any discrete output problem with linear constraints and non-differentiable decisions (e.g., ranking or segmentation fairness).
- **Unified $\beta^\lambda, M(a,\lambda)$ for both In/Post**: This means in deployment, one can use in-processing for a warm-start $\bar\lambda$ during training and then use post-processing for fine-tuning. This is engineering-elegant even if accuracy isn't strictly additive.
- **Cost-Sensitive Loss**: Transforming fairness constraints into calibrated cross-entropy offers a cleaner paradigm for in-processing than surrogate-based losses. The dual variable $\lambda$ naturally provides cost weights for $(a, y, \hat y)$, which is more intuitive than manual reweighting.

## Limitations & Future Work

- **Dependence on auxiliary model $\hat q_a$**: The post-processing upper bound depends on the quality of $\hat q_a$. Theorem 5.6's $\epsilon_1$ includes $\|q_a - \hat q_a\|_1$; when groups are imbalanced or attributes are hard to predict, the worst-case bound is dominated by this term. This was not specifically stress-tested.
- **Joint In + Post Ablation on Image Data**: The authors admit sensitive attributes are hard to feed into image data directly, so ablation was limited to tabular data. CelebA could have been used for this, but the paper avoided it.
- **Automation of Temperature $\tau$**: In experiments, $\tau$ was chosen empirically without a cross-validation process or analysis of whether the optimal $\tau$ differs across fairness criteria.
- **Practical Controllability for Multiple Criteria $K \ge 2$**: While theory supports $K$ simultaneous linear constraints, experiments only demonstrate single-criterion trade-offs. The Pareto surface under DP+EO constraints remains unexplored.

## Related Work & Insights

- **vs Agarwal et al. 2018 (Reductions for binary fairness)**: This is the multi-class version. The inner loop calibrated loss evolved from 0/1 importance-weighted to a **cost-sensitive softmax form** (Theorem 5.1), with entropy regularization added to ensure dual solvability.
- **vs Xian & Zhao 2024 / Denis et al. 2024 (Multi-class post-processing)**: Those assume **continuous** output distributions and are mostly attribute-aware. This paper uses entropic relaxation to remove the continuity assumption and achieves attribute-blind inference via $\hat q_a$.
- **vs FairProjection (Alghamdi et al. 2022) / LinearPost / FRAPPÉ**: These are either single-criterion or lack a characterization of the optimal classifier. OptFair-post expresses the "optimal solution" as a closed-form softmax, making the objective "approaching optimality" rather than "heuristic calibration."
- **Insight**: Using "linear-constrained Bayes optimal + entropic relaxation" as a general template can be applied to other scenarios like ranking fairness or retrieval fairness. The calibrated cost-sensitive loss could also replace reward shaping in LLM alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ Fills a theoretical gap by introducing entropy regularization (OT-style) to multi-class Bayes optimal fair characterizations and unifying in/post stages.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive datasets and baselines, but lacks joint multi-criteria experiments and avoids joint ablation on image data.
- Writing Quality: ⭐⭐⭐⭐ Clear Theorem-Algorithm-Experiment alignment; consistent notation; self-contained Appendix.
- Value: ⭐⭐⭐⭐ The fair ML community can directly reuse this calibrated loss and plug-in framework; very deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](../../CVPR2026/ai_safety/your_classifier_can_do_more_towards_balancing_the.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](../../NeurIPS2025/ai_safety/multi-class_support_vector_machine_with_differential_privacy.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)

</div>

<!-- RELATED:END -->
