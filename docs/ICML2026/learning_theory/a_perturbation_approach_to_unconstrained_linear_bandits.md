---
title: >-
  [Paper Note] A Perturbation Approach to Unconstrained Linear Bandits
description: >-
  [ICML2026][Unconstrained linear bandit] This paper revisits the perturbation-based bandit linear optimization approach of Abernethy et al.…
tags:
  - "ICML2026"
  - "Unconstrained linear bandit"
  - "online linear optimization"
  - "perturbation estimation"
  - "dynamic regret"
  - "high-probability bounds"
date: 2026-05-08
content_hash: 93df226fae7bfa94
---

# A Perturbation Approach to Unconstrained Linear Bandits

**Conference**: ICML2026  
**arXiv**: [2603.28201](https://arxiv.org/abs/2603.28201)  
**Code**: No public code (Theoretical paper, no repository provided in cache)  
**Area**: Optimization / Online Learning / Bandit Theory  
**Keywords**: Unconstrained linear bandit, online linear optimization, perturbation estimation, dynamic regret, high-probability bounds  

## TL;DR
This paper revisits the perturbation-based bandit linear optimization approach of Abernethy et al., proposing the PABLO reduction to transform unconstrained linear bandits into a problem that can call any OLO subroutine. This yields comparator-adaptive static/dynamic regret, high-probability bounds, and discussions on several lower bounds.

## Background & Motivation
**Background**: Bandit Linear Optimization requires the learner to choose an action $w_t$ at each round and observe only a scalar loss $\langle \ell_t, w_t\rangle$ rather than the full gradient $\ell_t$. Classic works typically study bounded action sets, such as Euclidean balls or polytopes; in these settings, exploration noise must ensure the action remains within the feasible region, and the regret is controlled by the domain diameter.

**Limitations of Prior Work**: Unconstrained BLO (uBLO) expands the action set to $\mathbb{R}^d$, aiming to adapt to any comparator $u$ while controlling the risk relative to the zero action $R_T(0) \le \epsilon$. This setting is closer to parameter-free online learning, but since bandit feedback provides only one-dimensional observations, it remains unclear how to simultaneously achieve comparator norm adaptation, dynamic comparator adaptation, and high-probability bounds.

**Key Challenge**: Unconstrained domains seem more difficult because there is no fixed radius limiting the actions; however, precisely because there are no feasible region constraints, the perturbation matrix does not need to be determined by barrier geometry and can be chosen freely. The key insight of the paper is that in uBLO, as long as an unbiased loss estimate with a controllable norm is constructed, the problem can be handed over to mature Online Linear Optimization subroutines.

**Goal**: The authors aim to establish a modular reduction: the bandit part is responsible only for random perturbations and constructing loss estimates, while the full-information OLO part handles comparator-adaptive or dynamic regret. Based on this framework, the paper provides expected regret, high-probability static/dynamic regret, and discusses the dimensional dependence of static lower bounds.

**Key Insight**: The paper starts from the SCRiBLe/Abernethy perturbation method but decouples the OLO updates from the self-concordant barrier. In each round, PABLO first has an arbitrary OLO algorithm output $w_t$, then samples a randomly perturbed point controlled by a matrix around $w_t$, and uses single-point bandit feedback to derive an unbiased estimate $\tilde{\ell}_t$.

**Core Idea**: In unconstrained linear bandits, use adjustable matrix perturbations to construct "good enough" loss estimates, reducing BLO to an OLO subroutine call, thereby inheriting the strong regret guarantees of parameter-free and dynamic OLO.

## Method
The main thread of this paper is PABLO (Perturbation Approach for Bandit Linear Optimization). In each round, it takes the output $w_t$ of the OLO learner as the center, chooses a positive definite matrix $H_t$, randomly picks $s_t \in \{\pm v_i\}$ along the eigenvector directions of $H_t$, and actually plays $\tilde{w}_t = w_t + H_t^{-1/2}s_t$. The environment returns only the scalar loss $\langle \ell_t, \tilde{w}_t \rangle$, based on which the algorithm constructs $\tilde{\ell}_t = dH_t^{1/2}s_t\langle \tilde{w}_t, \ell_t \rangle$ and feeds it back to the OLO learner for updates.

### Overall Architecture
The input consists of an arbitrary OLO algorithm $\mathcal{A}$, the time horizon $T$, and a bandit loss sequence. The output is a series of unconstrained actions and regret guarantees. PABLO itself does not restrict the OLO subroutine: if parameter-free mirror descent is chosen, one obtains static comparator-adaptive expected regret; if a dynamic comparator-adaptive OLO algorithm is chosen, one obtains path-length adaptive dynamic regret; if an OLO variant with an optimistic composite penalty is chosen, it can handle the extra terms brought by the unconstrained action norm in high-probability bounds.

The key technique lies in the setting of $H_t$. The paper adopts an isotropic choice satisfying $H_t \preceq \frac{1}{d(\|w_t\|^2 \vee \varepsilon^2)}I_d$, so that the perturbation becomes smaller when $w_t$ is large and prevents division by zero via $\varepsilon$ when $w_t$ is near zero. This leads to two types of estimation properties: $\tilde{\ell}_t$ is unbiased, and both its second moment and almost-sure norm have controllable upper bounds. These properties determine which OLO guarantees can be subsequently connected.

### Key Designs
1.  **PABLO Loss Estimator**:
    - **Function**: Converts single-point bandit feedback into a vector loss estimate that can be fed to an OLO learner.
    - **Mechanism**: Plays $\tilde{w}_t = w_t + H_t^{-1/2}s_t$ in random characteristic directions of $H_t$, then estimates the true $\ell_t$ using $\tilde{\ell}_t = dH_t^{1/2}s_t\langle \tilde{w}_t, \ell_t \rangle$. Since $s_t$ is sampled uniformly across positive and negative characteristic directions, the cross-terms cancel out under conditional expectation, yielding $\mathbb{E}[\tilde{\ell}_t|\mathcal{F}_{t-1}] = \ell_t$.
    - **Design Motivation**: Full-information OLO algorithms require seeing gradient/linear loss vectors. PABLO provides an unbiased, norm-controllable proxy, allowing the bandit problem to reuse the OLO algorithm library.

2.  **Perturbation Matrix Selection in Unconstrained Domains**:
    - **Function**: Ensures the estimator is capable of exploration without exploding due to unbounded action norms.
    - **Mechanism**: Let $H_t$ scale with $\|w_t\|$, satisfying $H_t \preceq 1/(d(\|w_t\|^2 \vee \varepsilon^2))I_d$. This ensures $\|\tilde{\ell}_t\|^2 \le 4d^2\|\ell_t\|^2$ almost surely, and the conditional second moment can be bounded by $\mathbb{E}[\|\tilde{\ell}_t\|^2|\mathcal{F}_{t-1}] \le 2d\|\ell_t\|^2$.
    - **Design Motivation**: OLO subroutines usually require bounded gradient norms. Since the unconstrained domain has no fixed radius, the perturbation scale must be adjusted relative to the center point to maintain estimation stability.

3.  **Switching OLO Subroutines by Regret Target**:
    - **Function**: Obtains different types of uBLO guarantees using the same bandit reduction.
    - **Mechanism**: Static regret uses parameter-free mirror descent, dynamic regret uses unconstrained dynamic OLO algorithms, and high-probability bounds use optimistic updates with Huber-like composite penalties to control the $\sum_t\|w_t\|^2$ term. PABLO's generic reduction transforms OLO regret bounds into bandit regret bounds, with extra costs coming from estimation noise and perturbation stability.
    - **Design Motivation**: Previous algorithms often coupled bandit geometry, direction learning, and scale learning. This paper isolates bandit feedback processing, allowing new OLO techniques to be directly migrated to uBLO.

### Loss & Training
This is a theoretical algorithm paper; there is no neural network training loss. The object of analysis is the regret under linear loss $f_t(w_t) = \langle \ell_t, w_t \rangle$. Key strategies include: using the ghost-iterate trick to reduce PABLO regret to OLO subroutine regret; distinguishing whether the comparator norm is oblivious or norm-adaptive in expected regret; using path-length $P_T = \sum_{t=2}^T\|u_t-u_{t-1}\|$ and its log-linear version in dynamic regret; and canceling the unconstrained iterate norm terms in high-probability bounds through composite penalties and optimism.

## Key Experimental Results

### Main Results
The paper has no traditional empirical experiments; the main "results table" is a comparison of theoretical guarantees. The table below organizes core theorems by setting.

| Result | Setting | Main Guarantee | Significance vs. Prior Work |
|------|------|----------|--------------------|
| Theorem 3.1 | Static comparator-adaptive expected regret | Approx. $\tilde{O}(G\epsilon + \frac{d}{\kappa}\|u\|\sqrt{V_T})$, with $\kappa=\sqrt{d}$ for oblivious and $\kappa=1$ for adaptive | Reveals that whether the comparator norm is trajectory-dependent causes a $\sqrt{d}$ level difference |
| Theorem 3.3 | Dynamic expected regret | Depends on $\Phi_T+P_T^\Phi$ and $\sum_t\|\ell_t\|^2\|u_t\|$, no prior knowledge of $P_T$ required | First guarantee in uBLO to achieve $\sqrt{P_T}$-type adaptation for the true path-length $P_T$ |
| Theorem 4.3 | Static high-probability regret | $\tilde{O}(dG(\epsilon+\|u\|)\log(T/\delta)+G\|u\|\sqrt{dT\log(T/\delta)})$ | Matches best-known magnitude for constrained Euclidean balls in high-prob (ignoring polylog) |
| Theorem 4.4 | Dynamic high-probability regret | Contains approx. $\sqrt{d(\Phi_T+P_T^\Phi)[d\mathcal{V}_T\wedge\Omega_T]}$ and lower-order terms | Preserves both worst-case $\sqrt{d(M^2+MP_T)T}$ and per-comparator adaptive interpretation |
| Theorem 5.2 | Bounded Euclidean ball lower bound | Proves folklore $\Omega(\sqrt{dT})$ directional regret lower bound | Provides independent evidence for the directional difficulty in uBLO scale/direction decomposition |

### Ablation Study
There are no module ablation experiments in this theoretical paper. The Following comparison based on the authors' analysis shows how key assumptions and design choices change the regret form.

| Comparison Item | Choice A | Choice B | Impact |
|--------|--------|--------|------|
| Comparator norm timing | Oblivious / Initially fixed | Norm-adaptive / Trajectory-dependent | Whether sharper second-moment bounds can be used in expected regret; causes $\kappa=\sqrt{d}$ vs $\kappa=1$ dimensional difference |
| Bandit-to-OLO path | Scale/direction decomposition | PABLO unbiased estimate + OLO subroutine | The former may degrade to $\tilde{O}((dT)^{2/3})$ in norm-adaptive settings; PABLO maintains $\sqrt{T}$-type horizon dependence |
| Change measure for dynamic | Switch count $S_T$ | Path-length $P_T$ | $S_T$ only counts whether a change occurs; $P_T$ reflects the magnitude; this paper is the first to get $\sqrt{P_T}$ dependence in uBLO without knowing $P_T$ |
| High-probability analysis | Directly apply OLO high-prob bound | Composite penalty + optimistic hints | The latter cancels unconstrained $\|w_t\|$ terms, otherwise uncontrollable iterate norms appear in the bound |
| Lower bound assumptions | Control only expected comparator norm | Consider worst-case magnitude simultaneously | Controlling only expectation allows meaningless linear lower bounds from giant comparators with tiny probability, showing norm-adaptive lower bounds need finer assumptions |

### Key Findings
- PABLO’s estimator is the pivot of the entire framework: it is simultaneously unbiased, has a small second moment, and is almost surely norm-bounded, allowing modern comparator-adaptive OLO algorithms to be used under bandit feedback.
- Unconstrained domains actually provide more degrees of freedom. Because one does not need to ensure perturbed points remain within a bounded set, $H_t$ can be chosen based on $\|w_t\|$, making estimation stability the main goal.
- The timing of when the "comparator norm is chosen" is not a technical detail. If the norm is coupled with the loss/algorithmic randomness, the Jensen steps implicitly used in much prior work no longer hold, leading to changes in dimensional dependence.
- Path-length dependence for dynamic regret is finer than switch count. Using only $S_T$ treats small changes and large jumps both as a single switch, whereas $P_T$ expresses the actual length of the comparator trajectory.
- The difficulty of high-probability bounds is not the unbiased estimator itself, but that unconstrained OLO iterates can be very large. Optimistic composite-penalty cancellation is the key to handling this.

## Highlights & Insights
- The most elegant part of the paper is its modularity. PABLO separates "creating vector estimates from bandit scalar feedback" and "controlling regret with OLO algorithms," allowing future theory to upgrade alongside OLO subroutine progress.
- The distinction between oblivious and norm-adaptive comparators is very insightful. Many parameter-free conclusions appear to "hold for all $u$ simultaneously," but when the norm of $u$ can depend on the trajectory post-hoc, independence assumptions in the analysis become very sensitive.
- The dynamic regret section demonstrates the uniqueness of unconstrained problems. In many constrained bandit settings, it is difficult to achieve good results without knowing the path-length; this paper shows uBLO might avoid some limitations of constrained lower bounds.
- The discussion on lower bounds, while not fully resolving the conjecture, honestly points out that scale lower bounds and direction lower bounds cannot be automatically merged. This is more valuable than simply claiming tightness.

## Limitations & Future Work
- The paper is primarily a theoretical contribution, with no experimental validation of the constants, stability, and feasibility of PABLO in practical bandit/online decision applications.
- Multiple regret bounds hide polylog terms, and the algorithmic subroutines are complex. For actual practitioners, the constants and tuning costs might not be small.
- The static lower bound remains a conjecture, especially how to force both scale difficulty and direction difficulty simultaneously in the same hard sequence.
- A reasonable lower bound model for the norm-adaptive setting is still unclear. The paper notes that controlling only $\mathbb{E}\|u\|$ is too weak but does not yet provide a final alternative assumption.
- Extension to Bandit Convex Optimization is only a future direction. Linear structure is crucial for both unbiased estimation and regret decomposition; the non-linear case might require new estimators.

## Related Work & Insights
- **vs. SCRiBLe / Abernethy et al.**: Classic SCRiBLe couples the FTRL regularizer with local perturbation geometry; PABLO decouples OLO subroutines and perturbation matrices, making it more suitable for unconstrained domains.
- **vs. uBLO work by van der Hoeven / Luo / Rumi**: Most existing methods use scale/direction decomposition. This paper points out that they often implicitly assume oblivious comparator norms and provides a more robust PABLO path for the norm-adaptive case.
- **vs. parameter-free OLO**: This work can be seen as bringing the comparator-adaptive capabilities of parameter-free OLO to bandit feedback, at the cost of constructing and controlling stochastic loss estimates.
- **vs. constrained adversarial linear bandits**: Minimax regret in bounded domains depends on the geometry of the decision set; this paper focuses on comparator norm, risk control, and path-length in unconstrained domains, leading naturally to a different problem structure.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Modular reconstruction of uBLO using PABLO while revealing the implicit difficulties of norm-adaptive comparators offers a fresh theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐☆☆ A theoretical paper without empirical experiments, but the theorems, comparisons, and lower bound discussions are quite complete.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main line with distinct hierarchical contributions; the proofs rely on appendices and complex OLO subroutines, posing a high entry barrier for reading.
- Value: ⭐⭐⭐⭐☆ Very inspiring for online learning and bandit theory, especially providing a general interface for the combination of uBLO and parameter-free OLO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](../../NeurIPS2025/others/infrequent_exploration_in_linear_bandits.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](../../ICLR2026/others/lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICML 2026\] Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach](markov_chain_monte_carlo_without_evaluating_the_target_an_auxiliary_variable_app.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](../../AAAI2026/others/structural_approach_to_guiding_a_present-biased_agent.md)

</div>

<!-- RELATED:END -->
