---
title: >-
  [Paper Note] Constrained Bayesian Experimental Design via Online Planning
description: >-
  [ICML 2026][LLM Pretraining][Bayesian experimental design] This paper proposes COPEx: a semi-amortized scheme combining "offline pre-trained amortized posterior networks + design policies + online multi-step lookahead sc…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Bayesian experimental design"
  - "EIG"
  - "scenario tree"
  - "amortized inference"
  - "constrained planning"
date: 2026-05-08
content_hash: 2dd5e4fe6030328f
---

# Constrained Bayesian Experimental Design via Online Planning

**Conference**: ICML 2026  
**arXiv**: [2605.26990](https://arxiv.org/abs/2605.26990)  
**Code**: https://github.com/yujiag21/COPEx  
**Area**: Optimization / Bayesian Experimental Design / Active Learning / Sequential Decision Making  
**Keywords**: Bayesian experimental design, EIG, scenario tree, amortized inference, constrained planning

## TL;DR
This paper proposes COPEx: a semi-amortized scheme combining "offline pre-trained amortized posterior networks + design policies + online multi-step lookahead scenario trees." This allows Bayesian Experimental Design (BED) to dynamically adapt to budget, cost, and transition constraints at test time. COPEx consistently outperforms baselines such as VPCE, ALINE, and RL-BOED in EIG/RMSE across constrained location finding, CES, and cost-aware AL tasks.

## Background & Motivation

**Background**: Bayesian Experimental Design (BED) selects the next experiment by maximizing Expected Information Gain (EIG). Recent "amortized BED" approaches (Foster 2021, Ivanova 2021, Blau 2022, Huang 2026, etc.) train a transformer or RL design policy $\pi_\psi(x \mid \mathcal{D})$ offline to output non-myopic design sequences with near-zero latency during testing.

**Limitations of Prior Work**: Real-world scientific experiments almost always involve dynamic constraints—varying measurement costs, limited total budgets, sensor movement/energy constraints, or limits on the difference between consecutive stimuli. However, amortized policies are trained on specific fixed feasible sets. During deployment, if new constraints such as "consecutive design distance $\|x_t - x_{t-1}\| \le \delta$" or "total budget $B_{\text{total}}$" are introduced, one must either retrain the entire policy network or rely on post-hoc masking to force actions into the feasible set. The latter often pushes trajectories out of the training distribution, leading to significantly degraded performance (Figure 1 shows that ALINE explores poorly when $\delta=0.1$, causing failure in posterior convergence).

**Key Challenge**: Constraints are not merely incidental details; they fundamentally reshape the optimal design policy. To achieve "constraint-aware + non-myopic" designs, naive approaches are either computationally infeasible (requiring nested posterior and EIG estimation for every candidate trajectory) or lack generality (requiring retraining for every new constraint).

**Goal**: To design a BED method that can adapt online to arbitrary budget, transition, or feasibility constraints at test time, while maintaining the non-myopic advantages of amortized methods and controlled computational overhead.

**Key Insight**: Explicitly model BED as a finite-horizon dynamic program with "evolution of constraint states $z_t$ + Bellman recursion," solved approximately via an H-step lookahead scenario tree. The explosion in computational cost associated with scenario trees is mitigated through "offline amortized posteriors + amortized policy warm-starting + one-shot reparameterization," converting nested posterior updates into differentiable forward passes.

**Core Idea**: Decouple "constraint awareness" into an online planning layer and "computational efficiency" into an offline amortization layer. Constraints can be modified without retraining.

## Method

### Overall Architecture

Constrained BED is formulated as a finite-horizon MDP over state $(\mathcal{D}_{t-1}, z_t)$: the reward is $\text{EIG}(x_t;\mathcal{D}_{t-1})$, the transition is $\mathcal{D}_t = \mathcal{D}_{t-1}\cup\{(x_t,y_t)\}$ and $z_{t+1}=f(z_t,x_t)$, and the feasible set $\mathcal{X}(z_t)$ is time-varying (covering transition constraints $\|x_t-x_{t-1}\|\le\delta$, global budget $b_{t+1}=b_t-c(x_t,\breve z_t)$, and design-dependent costs).

During testing, a receding-horizon approach is used: at each step $t$, an H-step lookahead scenario tree is expanded online from the current root $(\mathcal{D}_{t-1},z_t)$. Each decision node selects a design $x_k^{j_{1:\ell}}$, and each design samples $m_k$ fantasy observation branches, truncated at depth $H+1$. All decision variables $\mathbf{X}_{\text{tree}}$ in the tree are optimized jointly in a one-shot manner. The optimal root design $x_t^\star$ is executed, the real $y_t$ is observed, the state is updated, and the next step is planned.

Offline, two components are pre-trained: an amortized posterior network $q_\phi(\theta\mid\mathcal{D})$ (using a Mixture Density Network to fit $\mathcal{D}\mapsto p(\theta\mid\mathcal{D})$) and an amortized design policy $\pi_\psi$ (reusing the ALINE transformer policy from Huang et al. 2026). The former ensures fast computation, while the latter provides high-quality initialization.

### Key Designs

1.  **Multi-step lookahead scenario tree + one-shot reparameterization**:
    - **Function**: Converts the Bellman recursion of constrained BED, $V_t(\mathcal{D}_{t-1},z_t) = \max_{x_t}\{\text{EIG}(x_t;\mathcal{D}_{t-1}) + \gamma\mathbb{E}_{y_t}[V_{t+1}]\}$, into a differentiable finite tree search.
    - **Mechanism**: Following the one-shot tree BO (Jiang 2020b), a set of fixed base noises $\varepsilon=(\varepsilon_\theta,\varepsilon_y)$ is sampled beforehand. This makes all fantasy posterior samples $\theta_k^{j_{1:\ell}} = g_\phi(\mathcal{D}_{k-1}^{j_{1:\ell}}, \varepsilon_{\theta,k}^{j_{1:\ell}})$ and fantasy observations $\tilde y_k = h(x_k, \theta_k, \varepsilon_y)$ deterministic functions of the decision variables. The tree objective $\widehat V^{(H)}(\mathbf{X}_{\text{tree}};\varepsilon) = \sum_{\ell=0}^H \gamma^\ell \frac{1}{\prod m}\sum_{j_{1:\ell}}\widehat{\text{EIG}}$ becomes a single nonlinear program solved by SLSQP. Transition constraints are incorporated into $\mathcal{X}(z_k^{j_{1:\ell}})$ and budget constraints are handled via $z$ accumulation.
    - **Design Motivation**: Directly solving the Bellman equation requires nested posterior updates, which is nearly impossible for non-conjugate models. Reparameterization makes the entire tree gradient-accessible, allowing constraints to be placed directly on variables rather than embedding them into a policy network. Any constraint change only requires modifying $\mathcal{X}(z)$ and $f$, without retraining.

2.  **Amortized posterior network + adaptive contrastive EIG estimator**:
    - **Function**: Performs "fast posterior update + fantasy sampling + EIG estimation" at every node of the scenario tree, preventing exponential computational collapse as H increases.
    - **Mechanism**: A Mixture Density Network (MDN) $q_\phi(\theta\mid\mathcal{D})$ is trained to minimize $-\frac{1}{n}\sum\log q_\phi(\theta_i\mid\mathcal{D}_i)$ using simulated data. Online updates become simple evaluations of $q_{\hat\phi}(\theta\mid\mathcal{D}\cup\{(x,\tilde y)\})$. Fantasy samples are drawn from $\tilde\theta\sim q_{\hat\phi}(\cdot\mid\mathcal{D}_{k-1}^{j_{1:\ell}})$ then sampled from the likelihood $p(\cdot\mid x,\tilde\theta)$. EIG uses the adaptive contrastive objective (Foster 2020): $\widehat{\text{EIG}}(x;\mathcal{D},\hat\phi) := \mathbb{E}[\log\frac{p(\tilde y\mid x,\theta_0)}{\frac{1}{L+1}\sum_l q_{\hat\phi}(\theta_l\mid\mathcal{D})p(\tilde y\mid x,\theta_l)/q_{\hat\phi}(\theta_l\mid\mathcal{D}\cup\{(x,\tilde y)\})}]$, replacing expensive nested expectations with few MDN evaluations.
    - **Design Motivation**: The primary difficulty in BED with EIG lies in nested expectations and repetitive posterior updates. Amortized posteriors squeeze these into cheap forward passes, making online planning practical.

3.  **Amortized policy $\pi_\psi$ warm-start + exploration/exploitation fusion**:
    - **Function**: Addresses the issue where high-dimensional non-convex tree optimization easily falls into local optima, while maintaining robustness under constraints outside the policy's training distribution.
    - **Mechanism**: Each decision node $(t+\ell,j_{1:\ell})$ is initialized using the pre-trained unconstrained ALINE policy $x_{t+\ell}^{j_{1:\ell}}\leftarrow \pi_\psi(\mathcal{D}_{t+\ell-1}^{j_{1:\ell}})$. When constraints push the feasible region far from the training distribution, multiple trees are run—some initialized with $\pi_\psi$ for exploitation and others with random policies for exploration—selecting the best result.
    - **Design Motivation**: Figure 3(a) shows that a single policy-initialized tree achieves higher cumulative EIG in less time than 10 randomly initialized trees, proving a good prior is more efficient than multiple random restarts.

### Loss & Training
**Offline**: (i) Amortized posterior trained via NLL (Eq. 6) with simulated $\theta_i\sim p(\theta)$, sequence lengths $S_i\sim\text{Unif}\{1,\dots,T\}$, and designs $x_{i,s}\sim p(x)$; (ii) Design policy adopts ALINE (Huang 2026). **Online**: SLSQP solves the constrained nonlinear program (Eq. 9) using $H\in\{0,\dots,3\}$ and $m\in\{1,2\}$.

## Key Experimental Results

### Main Results

| Task | Constraint | Metric | COPEx | Best Baseline |
|------|------------|--------|-------|---------------|
| Location finding ($T=30$) | $\delta\in\{0.05,0.1,0.2\}$ transition | Cumulative EIG | Highest across all $\delta$; gain increases as $\delta$ decreases | ALINE / VPCE fail at small $\delta$ |
| CES ($B_{\text{total}}=100$) | Global budget | Cumulative EIG | **7.03 ± 0.55** ($H=1$) | ALINE 4.46 / RL-BOED 4.93 / VPCE 2.18 |
| CES ($B_{\text{total}}=150$) | Global budget | Cumulative EIG | **7.47 ± 0.55** ($H=1$) | ALINE 5.70 / RL-BOED 4.98 |
| Cost-aware AL (Ackley/Branin/GP × Hazard/Rough) | Design cost + transition | RMSE @ same cost | Consistently lower than GP-EPIG/US/VR/RS | 4 GP-based baselines |

### Ablation Study

| Configuration | Result / Notes |
|------|-------------|
| Policy-init (1 tree) vs. Random-init (10 trees) | Policy-init yields higher EIG with significantly lower runtime (Fig 3a). |
| Planning horizon $H\in\{0,\dots,5\}$ | EIG saturates at $H=2,3$, while runtime grows exponentially (Fig 3b). |
| Number of branches $m_k=m$ | Little gain from increasing $m$ in Location finding, but runtime grows exponentially (Fig 3c). |
| CES: $H=0$ vs. $H=1$ vs. $H=3$ | $H=1$ is best (7.03); $H=3$ drops to 6.36 due to bias accumulation in $q_{\hat\phi}$ along the rollout. |
| Hazard Center vs. Rough Terrain (AL) | Non-myopic COPEx shows a greater advantage over myopic methods under rougher cost terrains. |

### Key Findings
- **Policy warm-starting is the "cost-effectiveness king"**: The policy itself does not need to be constraint-aware; acting as an initialization is enough to push SLSQP towards high-quality local optima, saving more time than multiple restarts.
- **Short horizons are often sufficient**: The marginal benefit of EIG accumulation in BED decays quickly. Deep planning is not cost-effective for most tasks and can amplify systematic biases in $q_{\hat\phi}$.
- **Tight constraints favor COPEx**: As $\delta$ or $B_{\text{total}}$ decreases, post-hoc masking baselines (ALINE) and myopic VPCE suffer more, expanding COPEx's relative advantage.
- **Active Learning transferability**: Even without explicit latent variables $\theta$, the framework transfers to AL by replacing the amortized posterior with an amortized predictive $q_\phi(y\mid x,\mathcal{D})$ and using EPIG instead of EIG.

## Highlights & Insights
- **The philosophy of decoupling "Constraints Online, Models Offline"**: Amortization handles the expensive parts (posterior + EIG estimation), while online planning handles the "messy" parts (arbitrary feasible sets + arbitrary costs). Changing constraints requires no updates to neural network weights, which is of great value for real scientific experimental pipelines.
- **One-shot reparameterized tree**: Collapsing an H-step lookahead (usually requiring nested backward induction) into a single gradient-accessible nonlinear program is a key technique for migrating BO strategies (Jiang 2020b) to BED. The bottleneck of BED (nested EIG) is resolved by the amortized posterior.
- **Honest horizon reporting**: The authors report that $H=3$ can be inferior to $H=1$ due to bias accumulation, avoiding the common "deeper is always better" narrative and providing a more realistic characterization of planning in amortized models.
- **Generalizability**: The framework can adapt to any high-cost sequential decision scenario where states can be fitted by neural networks (e.g., bandit-style clinical trials, robot active sensing, adaptive physical simulations) by simply replacing the utility function and specifying $\mathcal{X}(z)$ and $f$.

## Limitations & Future Work
- **Amortized posterior bias accumulation**: Small biases in $q_{\hat\phi}$ can amplify exponentially during deep tree rollouts (verified by the performance drop at $H=3$ in CES), requiring more stable density estimation or online fine-tuning.
- **Constraints far from training distribution**: If constraints significantly shift the feasible region, the $\pi_\psi$ initialization may fail, leaving the authors to rely on random restarts as a fallback.
- **Online planning overhead**: At $H=1$, a single step takes ~19.5s on CES, which is still heavy for high-latency applications (e.g., real-time robot decision making). This might be mitigated by periodic policy fine-tuning (Hedman 2025).
- **Dependence on generative simulators**: Training amortized posteriors requires a large number of $(\theta, \mathcal{D})$ pairs from a simulator, making it less friendly to scenarios where the likelihood is a complete black box or simulation is expensive (e.g., high-fidelity molecular dynamics).
- **Scaling behavior**: Evaluation did not cover scenarios with much larger $d_\mathcal{X}$ or significantly longer horizons $T \gg 30$.

## Related Work & Insights
- **vs. ALINE (Huang 2026)**: ALINE is a fully amortized non-myopic policy with zero test delay but no adaptability to new constraints (relying on masking). COPEx reuses it for warm-starting but adds an online planning layer to move "constraint awareness" from training to testing.
- **vs. VPCE (Foster 2020)**: VPCE is a non-amortized variational EIG optimizer. While it can handle transition constraints through reparameterization, it is inherently myopic and requires retraining variational distributions at every step (105s/step on CES vs. 19s/step for COPEx).
- **vs. RL-BOED (Blau 2022)**: RL-trained non-myopic policies do not explicitly model trajectory-level constraints like global budgets and cannot be adjusted at test time.
- **vs. One-shot tree BO (Jiang 2020b)**: This work translates the technique to BED, where the nested EIG is much more expensive than GP mean/variance predictions in BO—making amortized posteriors a necessary companion.
- **vs. Astudillo 2021 (cost-aware BO)**: Their "fantasy budget" technique (using base policies to estimate future costs) is mentioned as a possible extension to mitigate planning failure under large budgets.

## Rating
- Novelty: ⭐⭐⭐⭐ — Successfully combines one-shot trees, amortized posteriors, and amortized policies for constrained BED; the combination is natural and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three heterogeneous tasks (Location finding, CES, AL) with various constraints and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ — Formal mathematical notation and clear derivations; Figure 1 is highly persuasive.
- Value: ⭐⭐⭐⭐ — Solving the "change constraints at test time without retraining" problem is highly valuable for real-world experimental science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Watch and Learn: Learning to Use Computers from Online Videos](../../CVPR2026/llm_pretraining/watch_and_learn_learning_to_use_computers_from_online_videos.md)
- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](../../NeurIPS2025/llm_pretraining/prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)
- [\[NeurIPS 2025\] Optimal Online Change Detection via Random Fourier Features](../../NeurIPS2025/llm_pretraining/optimal_online_change_detection_via_random_fourier_features.md)
- [\[ICML 2026\] Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)
- [\[ICML 2026\] Incremental BPE Tokenization](incremental_bpe_tokenization.md)

</div>

<!-- RELATED:END -->
