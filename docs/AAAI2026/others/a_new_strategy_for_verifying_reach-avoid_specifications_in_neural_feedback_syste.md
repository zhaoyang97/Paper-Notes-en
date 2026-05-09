---
title: >-
  [Paper Note] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems
description: >-
  [AAAI 2026][neural feedback systems] This paper proposes FaBRe (Forward and Backward Reachability), a unified framework that, for the first time, develops both over- and under-approximation algorithms for backward reachable sets of ReLU neural network controllers (GSS/ICH/LEB), and integrates them with forward reachability analysis to construct a unified reach-avoid verification framework, aiming to overcome the scalability bottleneck of purely forward analysis.
tags:
  - AAAI 2026
  - neural feedback systems
  - reach-avoid verification
  - backward reachability
  - forward reachability
  - safety verification
date: 2026-05-08
content_hash: 852bb633e9638b43
---

# A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems

**Conference**: AAAI 2026
**arXiv**: [2601.08065](https://arxiv.org/abs/2601.08065)
**Code**: None
**Area**: Formal Verification / Safety-Critical Systems
**Keywords**: neural feedback systems, reach-avoid verification, backward reachability, forward reachability, safety verification

## TL;DR

This paper proposes FaBRe (Forward and Backward Reachability), a unified framework that, for the first time, develops both over- and under-approximation algorithms for backward reachable sets of ReLU neural network controllers (GSS/ICH/LEB), and integrates them with forward reachability analysis to construct a unified reach-avoid verification framework, aiming to overcome the scalability bottleneck of purely forward analysis.

## Background & Motivation

**Background**: Neural feedback systems—dynamical systems controlled by neural networks—are increasingly deployed in safety-critical domains such as robotics, autonomous driving, and aerospace. Verifying whether such systems satisfy safety specifications (reach-avoid properties: reaching a target region while avoiding unsafe regions) is a core requirement prior to deployment.

**Limitations of Prior Work**:

1. Sampling/linearization methods are practical but provide no formal guarantees.

2. Invariance-based analysis requires constructing Lyapunov/Barrier functions, which are often intractable in practice.

3. Reachability analysis is the dominant approach, but relies almost entirely on **forward analysis**—backward propagation through neural networks is extremely difficult, causing existing backward methods to suffer from poor scalability.

4. Purely forward analysis accumulates over-approximation error over long time horizons, degrading verification precision.

**Key Challenge**: Over-approximations in forward analysis grow progressively over multi-step propagation (curse of dimensionality), while backward analysis—though potentially mitigating this by reasoning backward from target/avoidance sets—lacks efficient computable algorithms.

**Key Insight**: This paper develops efficient backward reachable set approximation algorithms and "splices" forward and backward analysis along the time axis—$F$ steps of forward analysis followed by $B$ steps of backward analysis—exploiting their complementarity to improve verification precision without excessive computational overhead.

## Method

### Overall Architecture

Given a discrete-time neural feedback system $\mathcal{D} = \langle n, I, F, E, u, \delta, T, G, A \rangle$, where $u$ is a ReLU feedforward neural network controller, FaBRe partitions the $T$-step time horizon into $F$ forward steps and $B$ backward steps ($T = F + B$):

- **Reach property verification**: Computes an over-approximation of the forward $F$-step reachable set and an under-approximation of the backward $B$-step reachable set; verification succeeds if the former is contained in the latter.

- **Avoid property verification**: Computes over-approximations of both the forward and backward reachable sets; verification succeeds if the two sets are disjoint at all time steps.

### Key Designs

1. **Backward Reachable Set Over-Approximation Algorithm**

    - Reformulates the one-step backward mapping as a constrained optimization problem, solving for the minimum/maximum value along each state dimension using an existing neural network verifier.
    - Constructs a hyperrectangle from the per-dimension bounds, guaranteed to contain the true backward reachable set.
    - Constraints encode the system dynamics $x_i^{t+1} = x_i^t + (f_i(\mathbf{x}^t) + u(\mathbf{x}^t)_i + \epsilon_i) \cdot \delta$ and require the next-step state to lie within the target set $\mathcal{T}$.

2. **Backward Reachable Set Under-Approximation Algorithms (Three Variants)**

    - **Golden Section Search (GSS)**: Parameterizes the under-approximation within the over-approximating hyperrectangle as $\vec{c} \pm \rho \vec{r}$ ($\rho \in (0,1)$), and uses golden section search to find the largest $\rho$ such that the forward image remains within the target set. Simple and applicable but requires multiple forward queries.
    - **Iterative Convex Hull (ICH)**: Densely samples the over-approximating region and propagates forward; positive samples form candidate under-approximating hyperrectangles, which are validated via over-approximation queries and iteratively refined. Replaces expensive set queries with point queries.
    - **Largest Empty Box (LEB)**: Addresses highly non-convex reachable sets by solving a small-scale MILP to find the largest box excluding all negative samples. Suitable for cases where positive samples in ICH are geometrically complex.

3. **FaBRe Unified Verification Strategy**

    - The time-horizon split $F/B$ serves as a tunable parameter, enabling flexible trade-offs between forward and backward analysis.
    - The forward-backward splicing reduces cumulative over-approximation error from long unidirectional propagation.
    - Reach and avoid verification employ different splicing conditions (containment vs. disjointness).

### Loss & Training

This paper presents a verification method rather than a learning method; no loss function is involved. The core computations are:

- Over-approximation queries: solved via the NNV tool as constrained optimization problems.
- Under-approximation queries: GSS uses golden section search; ICH uses sampling combined with iterative validation; LEB is formulated as a MILP.
- System assumptions: the controller is a fully connected ReLU feedforward network with $n$ inputs and $n$ outputs.

## Key Experimental Results

### Main Results

This paper is a methods proposal; the "Ongoing Work" section states that experimental evaluation is in progress. Planned baselines for comparison include:

| Baseline | Comparison Aspect | Reference |
|---------|---------|---------|
| Rober et al. | Backward over-approximation | IEEE OJ-CS 2023 |
| BURNS (Sidrane & Tumova) | Backward under-approximation | arXiv 2025 |
| Akinwande et al. | Forward reachability (polytope enclosure) | arXiv 2025 |

### Theoretical Analysis

| Algorithm | Guarantee Type | Computational Characteristics | Applicable Scenario |
|------|---------|---------|---------|
| Over-approximation (constrained opt.) | Sound overapprox. | Calls NNV; 2 solves per dimension | Foundation for all cases |
| GSS | Sound underapprox. | Requires multiple forward queries | Convex/weakly non-convex reachable sets |
| ICH | Sound underapprox. | Sampling + iterative validation | Moderate non-convexity |
| LEB | Sound underapprox. | Solves small-scale MILP | Highly non-convex reachable sets |

### Key Findings

- Forward and backward analysis are complementary: forward analysis expands from the initial set while backward analysis contracts from the target/avoidance set; splicing reduces over-approximation inflation.
- The three under-approximation algorithms address different reachable set geometries, progressing from simple (GSS) to complex (LEB).
- The method is general for ReLU feedforward network controllers with complete theoretical soundness guarantees.

## Highlights & Insights

- The paper is the first to systematically propose a joint forward+backward verification strategy, filling the gap caused by the absence of efficient algorithms for backward analysis.
- The three under-approximation algorithms (GSS/ICH/LEB) cover scenarios of varying complexity, with design philosophy progressing from search → sampling → MILP.
- The $F/B$ split parameterization provides a flexible precision-efficiency trade-off with practical engineering value.
- The theoretical framework is well-structured: the four properties defining forward/backward reach-avoid attributes are rigorously formalized.

## Limitations & Future Work

- Experimental validation is currently absent; the actual performance (precision, efficiency, scalability) of all algorithms remains to be confirmed.
- Only ReLU feedforward networks are supported; the approach does not extend to more complex controller architectures such as RNNs or Transformers.
- Hyperrectangular under-approximations may remain loose for highly non-convex reachable sets; polytopic or union-based representations could yield greater precision.
- The backward over-approximation requires invoking an NNV solver at each step, and computational cost may scale poorly with dimensionality.
- Extensions to continuous-time systems and stochastic perturbations are not discussed.

## Related Work & Insights

- **vs. purely forward analysis (Akinwande et al. 2025)**: FaBRe reduces the cumulative error of long-horizon forward propagation via backward analysis, representing a natural extension of that line of work.
- **vs. Rober et al. 2023**: Existing backward reachability methods suffer from poor scalability; this paper proposes more efficient approximation strategies.
- **vs. BURNS (Sidrane & Tumova 2025)**: BURNS also targets backward under-approximation; this paper offers three alternative algorithms of varying complexity.
- **Broader insight for verification**: The "sandwiching" idea of combining forward and backward analysis can potentially be generalized to other verification problems, such as probabilistic safety and robustness certification.

## Rating

- Novelty: ⭐⭐⭐⭐ Solid theoretical contributions from the joint forward-backward verification strategy and the three under-approximation algorithms.
- Experimental Thoroughness: ⭐ No experimental data; core claims are yet to be empirically validated.
- Writing Quality: ⭐⭐⭐ Mathematical definitions are rigorous, but the paper lacks an experimental section as would be expected of a complete manuscript.
- Value: ⭐⭐⭐ The research direction is important and the theoretical framework is complete, but experimental support is needed to assess actual impact.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)
- [\[AAAI 2026\] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)

<!-- RELATED:END -->
