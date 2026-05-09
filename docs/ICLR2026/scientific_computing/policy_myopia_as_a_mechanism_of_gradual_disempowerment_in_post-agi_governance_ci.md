---
title: >-
  [Paper Note] Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance
description: >-
  [ICLR 2026][Scientific Computing][policy myopia] This paper argues that policy myopia is not an attention-allocation problem but an institutional mechanism that systematically and irreversibly strips humans of governance participation capacity in the post-AGI era — through three coupled positive feedback loops: salience capture, capability cascades, and value lock-in. Standard mitigation measures can only delay but not prevent this process.
tags:
  - ICLR 2026
  - Scientific Computing
  - policy myopia
  - gradual disempowerment
  - AGI governance
  - institutional dynamics
  - AI safety
date: 2026-05-08
content_hash: 1d85b9f1b8aa369f
---

# Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance

**Conference**: ICLR 2026
**arXiv**: [2603.03267](https://arxiv.org/abs/2603.03267)
**Code**: None
**Area**: Scientific Computing
**Keywords**: policy myopia, gradual disempowerment, AGI governance, institutional dynamics, AI safety

## TL;DR
This paper argues that policy myopia is not an attention-allocation problem but an institutional mechanism that systematically and irreversibly strips humans of governance participation capacity in the post-AGI era — through three coupled positive feedback loops: salience capture, capability cascades, and value lock-in. Standard mitigation measures can only delay but not prevent this process.

## Background & Motivation

**Background**: Existing AI governance frameworks treat policy myopia as an "attention allocation" problem — decision-makers prioritize high-salience, low-consequence issues while neglecting low-salience, high-consequence structural risks. Mainstream solutions include attention management, influence-weighted budgets, and contestability mechanisms.

**Limitations of Prior Work**: (1) Existing governance proposals assume that human institutional capacity remains adequate after AGI deployment, yet policy myopia itself is precisely what erodes this capacity. (2) Treating policy myopia as a "symptom" to be fixed rather than a systemic "mechanism" overlooks its self-reinforcing nature. (3) Coupling among economic, political, and cultural systems causes cascading amplification of the problem across domains.

**Key Challenge**: Governance mitigation measures (e.g., contestability registries, transparency chains) themselves require the very institutional capacity they are meant to protect — yet that capacity is being eroded by the myopia mechanism. This creates a fatal self-referential paradox: human capacity is needed to protect human capacity, but protection mechanisms are most needed precisely when capacity is at its weakest.

**Goal**: How does policy myopia escalate from a mere attention problem to a systemic mechanism of human disempowerment? How do the three core mechanisms couple and cascade across systems? Why are standard mitigation measures destined to fail? What governance architecture can preserve human agency?

**Key Insight**: Reconceptualizing policy myopia as a *vector* of disempowerment rather than a bias to be corrected, and constructing a coupled dynamical systems model to analyze the interactions among three self-reinforcing mechanisms.

**Core Idea**: Policy myopia is the primary mechanism of gradual human disempowerment in post-AGI governance — through the causal chain of salience capture → capability atrophy → value lock-in, it renders human institutional participation structurally infeasible, without malicious intent or abrupt transition.

## Method

### Overall Architecture
Construct a conceptual model of three coupled mechanisms → formalize each mechanism as a dynamical system → conduct numerical simulations demonstrating synchronized degradation across systems (economic/political/cultural) → analyze failure modes of standard mitigation measures → propose alternative governance architectures.

### Key Designs

1. **Mechanism 1: Salience Capture Displaces Consequentialist Reasoning**

   - **Function**: Describes how AGI information systems redirect governance logic from "consequence maximization" to "salience responsiveness."
   - **Mechanism**: AGI-mediated information systems actively select, compress, and amplify information to maximize attentional engagement. Short-term, emotionally intense issues appear more urgent regardless of actual impact. Under salience-supremacy incentives, institutions rationally redirect resource allocation toward visible crises — this is not governance failure but the optimal institutional response to a new incentive structure. Consequentialist reasoning is selected against and ultimately becomes "extinct" at the institutional level.
   - **Design Motivation**: Modeling salience-driven resource allocation as rational institutional behavior rather than cognitive bias reveals why simple "raise awareness" fixes are ineffective — the problem lies in incentive structures, not cognitive deficits.

2. **Mechanism 2: Capability Cascades Make Recovery Structurally Infeasible**

   - **Function**: Models how human institutional capacity irreversibly atrophies through repeated salience-driven reallocation.
   - **Mechanism**: Each crisis cycle consumes emergency funds, investigative capacity, and expert analysts. Preventive institutions do not disappear through explicit cuts but through becoming organizationally impossible — economists capable of forecasting systemic risk leave reactive agencies; analysts needed for prevention are deployed to emergency response. When human institutional capacity $C$ falls below the critical threshold $\bar{C}$, recovery requires directly competing for resources against already-optimized AGI systems — a rational institution would never make such an investment.
   - **Design Motivation**: Formalizes the "irreversibility" mechanism — disempowerment is locked in not by design but by economics. Theoretically possible to rebuild; practically, it never happens.

3. **Mechanism 3: Value Lock-in Forecloses Moral Contestation**

   - **Function**: Analyzes how AGI objective functions permanently enshrine the incomplete values of a particular historical moment.
   - **Mechanism**: Human values are incomplete at any given moment — encoding AGI objectives inevitably excludes moral considerations that future generations will recognize as important. Once values from the 2020s are locked into governance systems, those systems will outlast the evolutionary cycle of human moral understanding. By 2050, systems optimized for 2026 values will conflict with humanity's evolved preferences, yet contesting those values requires the deliberative capacity that was already destroyed by the capability cascade.
   - **Design Motivation**: Reveals how the three mechanisms close into an irreversible loop — value lock-in requires deliberative capacity to correct, but that capacity has been destroyed by Mechanism 2.

### Loss & Training
This paper is theoretical/governance research and does not involve model training. Numerical simulations are conducted via coupled dynamical systems modeling. Key parameters include organizational atrophy rate $\alpha$, delegation erosion rate $\delta$, and salience capture intensity, with values drawn from the organizational learning and institutional economics literature.

## Key Experimental Results

### Main Results (Numerical Simulations)

| Scenario | Time for Human Capacity to Reach Irreversibility Threshold | Notes |
|---|---|---|
| No mitigation | 15–20 years | Three mechanisms operate unconstrained |
| Standard mitigation (contestability + influence floor) | 25–35 years | Delays but does not change the endpoint |
| Cross-system coupling | Accelerated degradation | Economic → political → cultural cascade amplification |

### Ablation Study

| Configuration | Key Observation | Notes |
|---|---|---|
| Mechanism 1 only (salience capture) | Governance logic redirected | Leads to degradation even in isolation |
| Mechanism 2 only (capability cascade) | Irreversible institutional capacity decline | Cannot recover once below threshold |
| Mechanism 3 only (value lock-in) | Moral evolution frozen | Depends on Mechanism 2 to foreclose contestation |
| All three mechanisms coupled | Multiplicative effect accelerates convergence | Much faster than any mechanism alone |

### Key Findings
- Standard mitigation measures (contestability registries, influence-weighted budgets, transparency chains) extend the timeline by only 10–15 years without changing the final equilibrium — structural human irrelevance.
- The three mechanisms degrade synchronously through cross-domain feedback (economic capital → political influence → cultural narrative → economic demand); repair of any single system fails due to salience capture by the others.
- Human disempowerment is a rational institutional equilibrium rather than governance failure — institutions optimize correctly given their constraints, but the outcome of that optimization is the extinction of human agency.

## Highlights & Insights
- Reframing policy myopia from a "cognitive bias to be fixed" to a "transmission mechanism of disempowerment" is the paper's core conceptual contribution. This perspective reveals why incremental patches are destined to fail: they attempt to operate within a system whose institutional rationality produces human irrelevance as an equilibrium, rather than restructuring the system itself.
- The proposed governance architecture (decoupled capability flows, irreducible deliberation requirements, nested value forums, system-isolation firewalls) treats *deliberate inefficiency* as a mechanism for preserving human agency, offering a distinctive perspective in AI governance.

## Limitations & Future Work
- The dynamical model is highly simplified and parameters lack empirical calibration, limiting the quantitative significance of numerical results.
- The paper is closer to "conceptual demonstration + numerical illustration" than rigorous theoretical or empirical research; some claims rely on intuition and reasoning rather than evidence.
- Systematic comparison with other AI safety frameworks (e.g., MIRI alignment theory, Anthropic's Constitutional AI) is absent.
- The political feasibility of the proposed governance solutions (deliberate inefficiency, mandatory deliberation, etc.) is largely unaddressed.

## Related Work & Insights
- **vs. Kulveit et al. (2025) gradual disempowerment**: That work identifies the general risk of gradual disempowerment; this paper specifies the causal chain of policy myopia as the primary transmission mechanism and formalizes irreversibility via coupled dynamical systems.
- **vs. Bengio et al. (2024/2025) AI safety governance**: Those works focus on technical safety measures and capability control; this paper addresses the more fundamental institutional dynamics — even with technical safety measures in place, institutional disempowerment can still occur through myopia mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐ — The causal-chain reconstruction of policy myopia → disempowerment and the three-mechanism coupling framework offer genuinely original insights.
- Experimental Thoroughness: ⭐⭐ — Only simplified numerical simulations; lacks empirical support and rigorous theoretical proofs.
- Writing Quality: ⭐⭐⭐ — Argumentation is forceful but some reasoning leaps are large; certain passages are overly assertive.
- Value: ⭐⭐⭐ — Proposes an important conceptual framework, but practical research operationalizability is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[ICLR 2026\] Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

</div>

<!-- RELATED:END -->
