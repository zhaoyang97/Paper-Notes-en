---
title: >-
  [Paper Note] When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper introduces **Cartograph**, a verification layer integrated into the autonomous "AI Scientist" loop. It utilizes a unified "unresolved subspace" object to simultaneously address three tasks: selecting the most disambiguating experiment (select), determining when a problem is solved (resolve), and—crucially—**
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: f1310604796d876a
---
# When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery

**Conference**: ICML 2026  
**arXiv**: [2606.07576](https://arxiv.org/abs/2606.07576)  
**Code**: To be confirmed  
**Area**: AI Safety / Autonomous Scientific Discovery / Experimental Design  
**Keywords**: Autonomous Scientific Discovery, Bayesian Optimal Experimental Design (BOED), Model Discrimination, Refusal, Governance

## TL;DR
This paper introduces **Cartograph**, a verification layer integrated into the autonomous "AI Scientist" loop. It utilizes a unified "unresolved subspace" object to simultaneously address three tasks: selecting the most disambiguating experiment (select), determining when a problem is solved (resolve), and—crucially—**refusing** to provide conclusions when the model library itself is structurally incorrect (refuse), with the ability to **revoke** earlier decisions if subsequent residuals expose a mismatch.

## Background & Motivation

**Background**: AI systems have entered the era of closed-loop scientific discovery: LLMs act as planners to propose candidate experiments, automated laboratories execute them, and statistical/neural modules interpret the data. End-to-end capabilities now exist for protein structure prediction, large-scale material discovery, and symbolic equation discovery.

**Limitations of Prior Work**: None of these stacks provide a **verifiable refusal signal** when the model library or hypothesis space is structurally inadequate. The bottleneck is no longer "generating experiments"—LLMs can generate far more candidates than a lab can execute—but rather deciding which experiment is truly informative, when a mechanism has been resolved, and when the system should **stop making conclusions altogether** because the searched model library is fundamentally flawed.

**Key Challenge**: Modern Bayesian Optimal Experimental Design (BOED) excels at selection; classical model discrimination criteria (e.g., Box–Hill) handle constrained selection and resolution. However, **neither treats refusal as a first-class output**: BOED assumes the truth is contained within the prior support, and model discrimination assumes at least one competitor is correct. For high-stakes autonomous discovery in clinical pharmacokinetics, material synthesis, or toxicology, this gap is critical.

**Goal**: To formalize the "verification and steering layer" of an AI Scientist into three interconnected decisions: Select (which candidate experiment best reduces current scientific ambiguity), Resolve (when ambiguity is small enough to declare the mechanism solved), and Refuse (when to stop identifying any model in the current library because the library itself is insufficient).

**Key Insight**: The authors distinguish between two "access models" to scientific libraries: **Symbolic Access** (direct reading of coefficient vectors, where recovery is a coverage property) and **Behavioral Access** (running experiments to observe numerical outputs, where recovery is a rank property). Currently, almost all AI Scientists operate under the latter: querying simulators, experimental robots, or tool endpoints, seeing only numerical values rather than symbolic equations.

**Core Idea**: Use the Singular Value Decomposition (SVD) of a cumulative "disagreement matrix" to define an "unresolved subspace" $U_\tau$. Both select and resolve decisions are derived from this single object. Refuse is handled by a separate residual-based guard, allowing the system to both select the next experiment and audit/stop-and-report within the sequential loop.

## Method

### Overall Architecture
Cartograph is a verification layer running within an autonomous discovery loop. Given a model library $\mathcal{M}=\{M_1,\dots,M_L\}$ (sharing a set of basis functions $\Phi$ but withholding different subsets) and a menu of candidate experiments $\mathcal{E}$, the system linearizes each candidate into a "disagreement design matrix" $H_e$ under behavioral access. It performs three tasks per round: scoring experiments using the unresolved subspace (select) → checking if ambiguity is exhausted (resolve) → using a residual guard to judge if the library is structurally misspecified (refuse/revoke). The loop involves only three **physically interpretable** hyperparameters $(\tau, \delta, \mu_{\min})$, and the primary cost per round is a truncated SVD, taking milliseconds on a CPU without requiring Monte Carlo integration.

```mermaid
graph TD
    A["Model Library + Candidate Experiments<br/>(Behavioral Access)"] --> B["Unresolved Subspace U_τ<br/>SVD of Disagreement Matrix (Small Singular Values)"]
    B -->|dim(U_τ)=0| R["Resolve: Ambiguity Exhausted<br/>Declare Mechanism Solved"]
    B -->|Remaining Unresolved Directions| C["Select: A-optimal Scoring<br/>Select Experiment e* to Disambiguate"]
    C --> D["Execute e*, Append H_e* to H_cur"]
    D --> E["Refuse Residual Guard<br/>Normalized Residual ρ + BIC Gap μ"]
    E -->|ρ > δ| F["Refuse/Revoke: Mark Structural Misspecification"]
    E -->|ρ ≤ δ AND μ ≥ μ_min| G["Tentatively Identify Best Model"]
    F --> B
    G --> B
```

### Key Designs

**1. Unresolved Subspace $U_\tau$: Turning "What is Unknown" into a Computable Geometric Object**

The challenge under behavioral access is that the system cannot see coefficients; it must infer which "disputed mechanism coordinates" $a_C^\star$ remain un-distinguished by existing experiments. The authors perform an SVD on the accumulated disagreement matrix $H_{\mathrm{cur}}$, taking the right-singular vectors associated with **singular values smaller than a threshold $\tau$** to span the unresolved subspace $U_\tau=\operatorname{span}\{v_j:\sigma_j\le\tau\}$ (which equals $\ker(H_{\mathrm{cur}})$ in the exact case where $\tau=0$). Intuitively, $U_\tau$ represents the region in the "contested coefficient space" where current experiments provide almost no information. For a candidate experiment $e$, the Jacobian $J_{\ell,e}$ is calculated for each library member; disagreement blocks $D_{ij,e}=J_{i,e}-J_{j,e}$ are computed for each model pair and stacked into $H_e$. Thus, $H_e U_\tau$ directly measures how strongly an experiment acts on directions that haven't been disambiguated. This step maps abstract "scientific uncertainty" to a rank/subspace problem, with Theorem 4.2 proving that $a_C^\star$ is uniquely recoverable **if and only if** $H$ is full column rank, with an error bound $\|\hat a_\tau-a_C^\star\|_2\le\eta/\tau+\|P_{U_\tau}a_C^\star\|_2$.

**2. A-optimal Select Scoring: Upgrading from "Maximum Disagreement" to "Minimum Posterior Variance"**

The simplest scoring is the isotropic unresolved projection $\operatorname{score}_{\mathrm{cart}}(e)=\|H_e U_\tau\|_F^2$. However, Lemma 4.8 proves this term equals the trace of Fisher Information on the unresolved subspace under isotropic noise—making it effectively 1st-order A-optimal. The actual default acquisition rule is **Exact A-optimality**: first constructing the noisy information matrix $G_e=U_\tau^\top H_e^\top\Sigma_e^{-1}H_e U_\tau$, then using the current unresolved posterior covariance $\Lambda_{\mathrm{cur}}$ to calculate:

$$\operatorname{score}_{\mathrm{A}}(e)=\operatorname{tr}(\Lambda_{\mathrm{cur}})-\operatorname{tr}\big((\Lambda_{\mathrm{cur}}^{-1}+G_e)^{-1}\big),$$

representing "how much the unresolved posterior variance will drop after this experiment." When $\dim(U_\tau)>1$ and the posterior covariance is anisotropic, A-optimality significantly outperforms raw projection—this is the root cause of performance gains in cascaded experiments. The authors also establish a local linear-Gaussian bridge to closed-form EIG (Prop 4.10) and Box–Hill (Prop 4.11), showing that Cartograph is not a mere heuristic.

**3. Residual-based Refuse Guard: Decoupling Resolve from "Truth" and Allowing Revocation**

Resolution only proves that "relative ambiguity within the library has closed"; it **cannot** prove the best-fitting library member is correct. Thus, refusal is a separate residual guard within the same loop. It uses two physically interpretable diagnostics: the normalized residual

$$\rho=\frac{\min_{\ell}\|y_{\mathrm{obs}}-f_{m_\ell}(\hat\theta_\ell)\|_2}{\|\phi(y_{\mathrm{obs}})\|_2},\qquad \mu=\mathrm{BIC}(m_{(2)})-\mathrm{BIC}(m_{(1)}),$$

where $\phi(\cdot)$ represents physically meaningful summary features (e.g., $C_{\max}$, terminal slope, log-linear RMSE), and $\mu$ is the BIC gap between the top two models. **Identification is only declared when $\rho\le\delta$ and $\mu\ge\mu_{\min}$**. Crucially, since $\rho$ is monitored at every step, a **tentative identification can be revoked**: the system might announce a model in early rounds but retract that judgment if subsequent rounds expose structural misspecification through rising residuals.

### Loss & Training
The method requires **no neural network training** and is CPU-only. The three hyperparameters are calibrated via a warm-start protocol and frozen for the benchmark family: $\tau$ is set to the "elbow" of the $H_{\mathrm{cur}}$ spectrum; $\delta$ is set to the 95th percentile of library residuals during warm-start; $\mu_{\min}$ follows standard BIC "positive evidence" thresholds. The posterior covariance $\Lambda_{\mathrm{cur}}$ is updated from an isotropic prior using empirical Bayes. Computational complexity per round is independent of the number of posterior samples, making it orders of magnitude cheaper than Monte Carlo EIG.

## Key Experimental Results

Experiments evaluated three responsibilities (select / resolve / refuse) across 5 testbeds: symbolic Duffing oscillators, variable-dimension structural nonlinear cascades, a Pharmacokinetic (PK) model library, public EPA time-series data, and a retrospective audit of the published A-Lab autonomous materials system.

### Main Results: Structural Cascade (Select Headline)
In a cascaded ODE where unresolved dimensionality $d$ ranges from 2 into 16, 144 trials were run per dimension. Cartograph-A (Exact A-optimal) vs. raw projection results are as follows:

| Dimension $d$ | Raw Hit Rate | Cartograph-A Hit Rate | EIG Hit Rate | Raw Regret | Cartograph-A Regret | A vs Raw (W/T/L) | $p$-value |
|---|---|---|---|---|---|---|---|
| 2 | 0.44 | 0.44 | 0.44 | 0.052 | 0.052 | 0 / 144 / 0 | n.s. |
| 4 | 0.00 | 0.09 | 0.10 | 0.312 | 0.418 | 73 / 0 / 71 | 0.46 |
| 8 | 0.02 | **0.65** | 0.63 | 19.94 | **0.010** | 129 / 0 / 15 | $<10^{-21}$ |
| 16 | 0.07 | **0.72** | 0.70 | 2.832 | **0.014** | 120 / 0 / 24 | $<10^{-16}$ |

At $d=8$, A-optimal selection reached the oracle hidden optimal experiment in 65% of cases vs. 2% for raw disagreement; regret plummeted from 19.94 to 0.01. Cartograph-A matched closed-form EIG within 2 percentage points at every dimension while requiring only one SVD per round—serving as an efficient surrogate for EIG.

### Key Findings
- **Performance stems from A-optimality, not just projection**: Raw Cartograph and the disagreement heuristic are nearly indistinguishable in the cascade; the gap is entirely filled by A-optimality, as predicted by Prop 4.7/4.8.
- **Theory explains "negative" results**: Higher dimensions amplify the divergence between projection and disagreement (Theorem 4.6), while low dimensions inevitably lead to a draw—the authors do not claim a "select" victory in low-dimensional PK models.
- **Revocation is what AI Scientist governance truly needs**: Providing an auditable "stop and report" signal within the same loop that picks the next experiment. In the A-Lab audit, the residual guard flagged all 4 cases later manually reviewed as "inconclusive," while standard residuals failed to flag them.

## Highlights & Insights
- **Formalizes the stopping criteria of an AI Scientist** into three decoupled decisions (select/resolve/refuse), with $U_\tau$ serving as the shared geometric engine. The design where $\dim(U_\tau)=0$ serves as a drop-in signal for resolution is elegant.
- **Elevates Refusal to a first-class output**, filling a gap in BOED and Model Discrimination frameworks. The ability to revoke early judgments is of greater value for the governance of high-stakes autonomous discovery than marginal gains in experiment selection.
- **The distinction between coverage (symbolic) and rank (behavioral) access** is highly transferable to any "agent-tool" scenario where an agent queries a black box without reading its internal structure.
- **CPU-only and millisecond-latency** with physically interpretable hyperparameters makes it easy to drop into LLM-planned AI Scientists as a verification layer.

## Limitations & Future Work
- The connection to BOED is **locally linear-Gaussian**. The authors do not claim global optimality or equivalence for high nonlinearity; if the posterior is far from Gaussian, the EIG approximation may fail.
- Selection gains are context-dependent: in low-dimensional settings like PK, it performs similarly to simple disagreement heuristics. The true selling point is refusal/revocation.
- A-Lab audit samples are small (4 inconclusive + 36 confirmed). While these provide feasibility evidence for "auditable pass/flag logs," they are not a definitive statistical re-evaluation of the A-Lab system.
- The residual guard relies on manually selected physical features $\phi(\cdot)$ and BIC; the robustness of $\delta/\mu_{\min}$ calibration across new domains remains to be verified.

## Related Work & Insights
- **vs. Modern BOED**: They optimize selection but assume the truth is in the prior; they lack a refuse output. This work provides a cheap 1st-order surrogate for EIG and adds refusal.
- **vs. Classical Model Discrimination (Box–Hill)**: These assume at least one library member is correct. This work shows Box–Hill degrades to a "disagreement magnitude" goal under isotropic noise and promotes refusal as a primary output.
- **vs. End-to-end Auto-Discovery**: These systems (e.g., GNoME) have high output capacity but do not issue verifiable refusal signals. Cartograph is positioned as a governance module for these loops.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevating refuse/revocation to a first-class decision in autonomous discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 testbeds + real-world audit, although the sample size for refusal cases is naturally small.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptional clarity and "honest landing" of theoretical vs. experimental outcomes.
- Value: ⭐⭐⭐⭐⭐ Provides an auditable, low-cost verification layer essential for the governance of AI Scientists.

## Related Papers

- [\[ICML 2026\] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models](position_stop_chasing_the_c-index_when_evaluating_survival_analysis_models.md)
- [\[ICML 2026\] Regret-Based Federated Causal Discovery with Unknown Interventions](regret-based_federated_causal_discovery_with_unknown_interventions.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[CVPR 2026\] DSO: Direct Steering Optimization for Bias Mitigation](../../CVPR2026/ai_safety/dso_direct_steering_optimization_for_bias_mitigation.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[ICML 2026\] Position: AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks](ai_researchers_must_help_lead_arms_control_to_mitigate_military_ai_risks.md)
- [\[ICML 2026\] When Benign Inputs Lead to Severe Harms: Eliciting Unsafe Unintended Behaviors of Computer-Use Agents](when_benign_inputs_lead_to_severe_harms_eliciting_unsafe_unintended_behaviors_of.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](ai_alignment_encompasses_competing_technical_priorities.md)

</div>

<!-- RELATED:END -->
