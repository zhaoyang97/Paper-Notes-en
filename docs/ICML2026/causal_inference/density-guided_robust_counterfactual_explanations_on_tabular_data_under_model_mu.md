---
title: >-
  [Paper Note] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity
description: >-
  [ICML 2026][Causal Inference][Counterfactual Explanations] DensityFlow reformulates the generation of robust counterfactual explanations (RCE) under model multiplicity as a density-constrained optimal transport problem.…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Counterfactual Explanations"
  - "Model Multiplicity"
  - "Neural ODE"
  - "Noise Contrastive Estimation"
  - "Density Guidance"
date: 2026-05-08
content_hash: c0491d4fd5df69ff
---

# Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity

**Conference**: ICML 2026  
**arXiv**: [2605.30901](https://arxiv.org/abs/2605.30901)  
**Code**: https://github.com/G-AILab/DensityFlow (Available)  
**Area**: Explainability / XAI / Counterfactual Explanations / Tabular Data  
**Keywords**: Counterfactual Explanations, Model Multiplicity, Neural ODE, Noise Contrastive Estimation, Density Guidance  

## TL;DR
DensityFlow reformulates the generation of robust counterfactual explanations (RCE) under model multiplicity as a density-constrained optimal transport problem. It trains a $(K+1)$-class discriminator using NCE to simultaneously learn classification and class-conditional density. A Neural ODE then transports query samples along density gradients to the high-density manifold of the target class. In black-box scenarios, local distillation alignment is performed only on the generated trajectories, achieving higher cross-model validity with significantly fewer queries than ensemble baselines.

## Background & Motivation

**Background**: Counterfactual explanation (CE) aims to find a minimal-cost perturbation $x'$ for a given query sample such that $h(x')=y^*$, serving as a core tool for algorithmic recourse and high-stakes decision explainability. Recent trends have shifted from per-sample optimization to generative paradigms: VAEs, diffusion models, and normalizing flows learn a data manifold prior and then search for or generate CEs in the latent space to ensure feasibility and realism.

**Limitations of Prior Work**: Under Model Multiplicity (MM), multiple "reasonable" classifiers $\{h_j\}$ with similar performance but different decision boundaries can cause a "CE effective for one model" to fail on another—the classic Rashomon effect. Generative methods fail to explicitly distinguish between "core high-density regions" and "long-tail low-density regions" within a class. Distance minimization naturally pulls $x'$ toward sparse areas near the boundaries of two classes, precisely where model disagreement is highest. Conversely, methods utilizing explicit ensemble consensus (MILP, rule-based, random retraining) are query-intensive and lack scalability.

**Key Challenge**: Robustness requires $x'$ to fall into high-density regions of the target class (strong model consistency), while cost minimization pulls $x'$ toward decision boundaries (inevitably entering low-density tails). These objectives are directly opposed. Additionally, gradients are unavailable in black-box scenarios, and full-space surrogate alignment is impractical.

**Goal**: (i) Explicitly model and utilize class-conditional density $p(x|y^*)$ within a generative framework to "block" low-density regions; (ii) perform boundary alignment between the surrogate and target models using minimal queries under black-box heterogeneous ensembles.

**Key Insight**: Integrate "validity + density" into the same surrogate. By using $(K+1)$-way NCE, a single network performs both classification and density ratio estimation, preventing density estimators from overfitting to sparse outliers. The generation process is formulated as a Neural ODE where the density signal acts as a potential function in the flow dynamics. Due to the inherent smoothness of ODE trajectories, imposing density constraints at the endpoints is sufficient to guide the entire path.

**Core Idea**: Rewrite robust counterfactuals as density-constrained optimal transport: $\min c(x,x')\ \text{s.t.}\ \mathbb{E}_{\mathcal{M}}[h(x')]=y^*,\ p(x'|y^*)\ge\tau\cdot p_{\text{ref}}$. Use NCE density gradients to guide the Neural ODE along "high-density highways" and perform local distillation only in the trajectory neighborhood for black-boxes.

## Method

### Overall Architecture
DensityFlow decomposes the generation of robust counterfactuals $x'$ for a query $x$ against a target black-box ensemble $\mathcal{M}=\{h_j\}_{j=1}^m$ into two alternating networks: (1) A **surrogate network** $f_\phi:\mathcal{X}\to\mathbb{R}^{K+1}$ that performs two tasks—the first $K$ logits predict target classes, and the $(K+1)$-th logit corresponds to a "noise class." The difference between adjacent logits $S(x|y^*)=z_{y^*}(x)-z_{K+1}(x)$ provides a differentiable estimate of $\log p(x|y^*)$. (2) A **generator** $v_\theta(z,t)$ representing the velocity field of a Neural ODE, which continuously transports $x$ over $t\in[0,1]$ to $z(T)=x'$. The training objective couples validity loss, kinetic energy (transport cost), and density penalties. For heterogeneous black-box ensembles, a "trajectory-aware local distillation" step is added, using samples at $z(T)$ to query $\mathcal{M}$ and align the boundary of $f_\phi$ with the ensemble consensus. The final output is the ODE endpoint $x'=z(T)$.

### Key Designs

1.  **(K+1)-way NCE Density Surrogate (Core)**:
    - **Function**: Provides the generator with a differentiable class-conditional density signal $\log p(x|y^*)$ that shares representations with the classifier, without training an independent density estimator.
    - **Mechanism**: Expands the original $K$-class problem into a $(K+1)$-way discrimination task. The first $K$ classes use real data $\mathcal{D}_{\text{src}}$, while the $(K+1)$-th class uses samples from a uniform distribution $p_{\text{noise}}$ (standardized within a $[-C,C]^d$ hypercube). Joint training uses cross-entropy: $\mathcal{L}_{\text{surrogate}}=-\mathbb{E}_{\mathcal{D}_{\text{src}}}\log\frac{e^{z_y}}{\sum e^{z_j}}-\mathbb{E}_{p_{\text{noise}}}\log\frac{e^{z_{K+1}}}{\sum e^{z_j}}$. Proposition 4.1 proves that when $p_{\text{noise}}$ is uniform, the optimal solution satisfies $z_k^*(x)-z_{K+1}^*(x)=\log p(x|k)+\text{Const}$. Thus, the logit difference is an unbiased estimate of the class-conditional log-density, and $\nabla_x S(x|y^*)=\nabla_x\log p(x|y^*)$ yields the density guidance signal. The trust region threshold $\tau$ is explicitly controlled via the noise/data sampling ratio $N_{\text{noise}}/N_{\text{data}}$.
    - **Design Motivation**: Traditional methods train a separate density estimator (VAE/KDE/LOF) and append it to the CE optimization. However, sparse outliers can severely distort density, and density signals may decouple from classification—regions deemed high-density might have low classifier confidence. Jointly training both in $f_\phi$ allows the density signal to naturally incorporate "classification confidence," making the surrogate insensitive to long tails and providing a smoother, decision-consistent guidance surface.

2.  **Density-Guided Neural ODE Generator**:
    - **Function**: Formulates the transport from the query point to the high-density target manifold with minimal cost as an end-to-end differentiable continuous-time dynamical system.
    - **Mechanism**: Augments the state to $\tilde z(t)=[z(t),e(t)]^\top$ with dynamics $d\tilde z/dt=[v_\theta(z,t);\ \|v_\theta(z,t)\|^2]$ and initial value $\tilde z(0)=[x;0]$. A `dopri5` adaptive solver integrates over $t\in[0,1]$. The endpoint $e(T)=\int_0^T\|v_\theta\|^2dt$ represents the kinetic energy and is used as the cost $\mathcal{L}_{\text{cost}}$. The density constraint is expressed as a path integral $\mathcal{L}_{\text{den}}=\int_0^T\text{ReLU}(\log\tau-S(z(t)|y^*))dt$. Experiments show ODE trajectories are sufficiently smooth that punishing only the endpoint is enough to pull the entire trajectory into the trust region, saving computation. The total objective $\mathcal{L}(\theta)=\mathcal{L}_{\text{CE}}(f_\phi(x'),y^*)+\lambda_{\text{cost}}c_{\text{cost}}(T)+\lambda_{\text{den}}\mathbb{E}[\mathcal{L}_{\text{den}}(x')]$ balances validity, proximity, and robustness.
    - **Design Motivation**: Unlike hard optimization of static constraints using KKT/Lagrangian methods, Neural ODEs treat constraints as potential functions ($\nabla S$ as a drift term), allowing trajectories to "naturally bypass" low-density areas. State augmentation makes transport cost exactly equal to the kinetic energy integral, removing the need for separate $\|x-x'\|$ calculations.

3.  **Trajectory-Aware Local Distillation (Black-box Alignment)**:
    - **Function**: Aligns the decision boundary of $f_\phi$ with the consensus of the heterogeneous ensemble $\mathcal{M}$ using minimal queries, ensuring density gradients are effective for ensemble validity.
    - **Mechanism**: Instead of global alignment (prohibitively expensive in high dimensions), current generator endpoints $\mathcal{D}_\theta=\{(x,\bar y)\mid x\sim z(T)\}$ are dynamically sampled (where $\bar y$ is the ensemble vote). The local distillation loss $\mathcal{L}_{\text{dis}}(\phi)=\mathbb{E}_{\mathcal{D}_\theta}[\|\sigma(z_{y^*}(x))-\bar y\|^2]$ is minimized.
    - **Design Motivation**: Density guidance identifies "where to go." Since trajectories primarily cross high-density regions, boundary alignment is only necessary in those localized areas. This compresses query complexity from $O(\text{vol}(\mathcal{X}))$ to $O(|\text{trajectory}|)$, allowing density signals and query efficiency to complement each other.

### Loss & Training
Two-level alternating optimization: Inner loop updates $f_\phi$ using Eq. (3) (joint NCE classification + density); outer loop updates $v_\theta$ using Eq. (7) (validity + cost + density). Local distillation (Eq. 8) is inserted for black-box cases. AdamW is used with $\eta_g=10^{-3}$, $\eta_\phi=10^{-4}$ for 800 epochs and batch size 64. Hyperparameters $\lambda_{\text{cost}}\in\{0.2, 0.4, 0.6\}$ and $\lambda_{\text{den}}\in\{0.0, 0.1, 0.3\}$ are grid-searched. The noise-to-data ratio is $\tau=0.2$, and the noise hypercube side length is $C=1.2\cdot\max_{\mathcal{D}_{\text{train}}}\|x\|_\infty$. ODE tolerances are $(10^{-3}, 10^{-3})$ for training and $(10^{-4}, 10^{-4})$ for testing.

## Key Experimental Results

### Main Results
Evaluated on 8 datasets (4 synthetic: Moons/Circles/Spirals/Chessboard; 4 real tabular: Adult/Compas/HELOC/Blood) with a target ensemble $\mathcal{M}$ consisting of KNN, SVM, RF, MLP, XGBoost, CatBoost, and TabNet.

| Dataset | Metric | DensityFlow | Best Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Adult | Validity↑ | 0.901 | 0.752 (BetaRCE) | +0.149 |
| Adult | Cost↓ | 1.597 | 1.916 (Argument) | −0.319 |
| Compas | Validity↑ | 0.729 | 0.610 (Argument) | +0.119 |
| Blood | Validity↑ | 0.662 | 0.509 (Argument) | +0.153 |
| Moons | Validity↑ | 0.997 | 0.991 (CeFlow) | +0.006 |
| Circles | Validity↑ | 0.994 | 0.991 (Argument) | +0.003 |
| Spirals | Validity↑ | 0.972 | 0.943 (Argument) | +0.029 |

Ours significantly leads in validity on real-world datasets while simultaneously reducing cost. Performance on synthetic data is near saturation, with DensityFlow consistently ranking first.

### Ablation Study
| Configuration | Adult Validity | Blood Validity | Compas Validity | HELOC Validity |
| :--- | :---: | :---: | :---: | :---: |
| Full DensityFlow | 0.901 | 0.662 | 0.729 | 0.757 |
| w/o Density (Remove NCE Density) | 0.815 | 0.495 | 0.642 | 0.718 |
| w/o Distill (Remove Local Distillation) | 0.767 | 0.531 | 0.698 | 0.734 |

### Key Findings
- Removing the density term leads to a 4–17% drop in validity across real datasets, proving NCE density gradients are central to robustness. Small datasets like Blood (748 rows) show the largest drop (0.662→0.495), highlighting the importance of density signals in sparse data.
- Removing local distillation drops Adult validity to 0.767, but performance on other datasets still beats all baselines, suggesting NCE guidance is effective even without perfect black-box alignment.
- **Query Efficiency**: On Spirals/Adult, DensityFlow requires over an order of magnitude fewer queries than Argument or BetaRCE. Validity saturates quickly with minimal distillation queries.
- **Noise Ratio $\tau$ Sensitivity**: Low $\tau$ fails to define data boundaries, resulting in adversarial-like CEs (Compas cost drops to ≈0.75). Increasing $\tau$ tightens the trust region, improving validity at the cost of slight proximity increases until stabilizing.
- Empirical results confirm Prop. 4.1: the density score $S(x|y^*)$ shows a strong negative correlation with ensemble uncertainty (Mutual Information) on Adult/HELOC.

## Highlights & Insights
- **Unified learning of classification and density is elegant**: Using $(K+1)$-way NCE to share a backbone between density estimation and classification avoids the fragility of "stitching" a VAE/KDE onto a CE optimizer. The two signals are naturally coupled and do not contradict each other.
- **Density guidance and local distillation are synergetic**: Density defines the high-confidence search space, allowing distillation to be localized and efficient. Conversely, local distillation ensures density gradients point in directions valid for the black-box.
- **Utilizing ODE smoothness via endpoint penalties**: Path integral constraints are expensive. The inherent smoothness of ODE paths allows endpoint penalties to constrain the entire trajectory, a useful trick for any flow-based constrained generation.
- **Reformulating the Rashomon effect as a density problem**: Instead of expensive ensemble consensus checks, Ours operates on the insight that "low density equals high model disagreement," reducing a multi-model consistency problem to a more efficient single-model density problem.

## Limitations & Future Work
- **Limitations**: (1) Extending density estimation to very high-dimensional data is difficult and may require feature selection (e.g., LassoNet), which might discard features physically coupled with perturbations. (2) Extreme class imbalance or label noise weakens class-conditional density learning. (3) The focus on robustness conflicts with paradigms (like CFKD) that seek rare or "interesting" edge cases.
- **Observations**: (1) Experiments are limited to low-to-medium-dimensional tables (max 23 dims); high-dimensional scalability remains an open question. (2) Uniform noise is sensitive to the curse of dimensionality; the hypercube side length $C=1.2\cdot\max\|x\|_\infty$ might lead to volume explosion in $d>50$, making NCE sampling inefficient. (3) Local distillation assumes smoothness; tree-based models ($\mathcal{M}$ includes RF/XGBoost) with step-wise boundaries might challenge this assumption.
- **Future Work**: (a) Moving NCE density to a pre-trained latent space combined with score matching. (b) Implementing adaptive noise (e.g., via a generator) for high-dimensional efficiency. (c) Exploring "anti-density" guidance to discover rare edge cases.

## Related Work & Insights
- **vs CeFlow (Duong 2023)**: Both are flow-based. CeFlow uses normalizing flows for global manifold priors without explicitly penalizing low-density regions. DensityFlow uses Neural ODEs for density-guided transport in the original space, leading to significantly higher validity (0.691 → 0.901 on Adult).
- **vs Argument (Jiang 2024a)**: Argument is a post-hoc selection method that generates CEs for each model and picks consistent ones via an argumentation framework. DensityFlow is generative, embedding reliability into the objective and avoiding low-density candidates, reducing queries by an order of magnitude.
- **vs BetaRCE (Stępka 2025)**: BetaRCE defines an admissible space through random retraining. DensityFlow replaces retraining with NCE and local distillation, converting expensive training signals into cheap density gradients.
- **vs Product_mip (Leofante 2023)**: A MILP-based hard constraint method requiring white-box access. DensityFlow is fully differentiable and model-agnostic.
- **Insight**: Treating "distributional support" as a core constraint rather than a regularizer suggests that many robustness problems (adversarial, OOD, selective classification) can be unified by falling into high-density $p(x|y)$ regions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of NCE density, Neural ODE, and local distillation is novel in the RCE domain, though components are individually established.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers 8 datasets and 7 heterogeneous classifiers with 5 seeds. Ablations are clear, but scalability to high-dimensional data (beyond 23 dims) is not demonstrated.
- **Writing Quality**: ⭐⭐⭐⭐ The derivation from Rashomon effect to density guidance is cohesive. Theory and experiments are well-aligned.
- **Value**: ⭐⭐⭐⭐ Provides a low-query, differentiable, and easy-to-implement baseline for robust CEs. The "density-guided generation" logic is applicable beyond XAI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICML 2026\] Causal Fine-Tuning under Latent Confounded Shift](causal_fine-tuning_under_latent_confounded_shift.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](investigating_memory_in_model-free_rl_with_popgym_arcade.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](investigating_memory_in_model-free_rl_with_popgym_arcade.md)

</div>

<!-- RELATED:END -->
