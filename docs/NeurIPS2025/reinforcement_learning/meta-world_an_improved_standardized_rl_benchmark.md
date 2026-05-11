---
title: >-
  [Paper Note] Meta-World+: An Improved, Standardized, RL Benchmark
description: >-
  [NeurIPS 2025][Reinforcement Learning][multi-task reinforcement learning] This paper systematically exposes how undocumented reward function inconsistencies across versions of the Meta-World benchmark distort algorithm c…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "multi-task reinforcement learning"
  - "meta-reinforcement learning"
  - "benchmark"
  - "reward function"
  - "reproducibility"
date: 2026-05-08
content_hash: 8346e24e0203192c
---

# Meta-World+: An Improved, Standardized, RL Benchmark

**Conference**: NeurIPS 2025
**arXiv**: [2505.11289](https://arxiv.org/abs/2505.11289)
**Code**: [GitHub](https://github.com/Farama-Foundation/Metaworld/)
**Area**: Reinforcement Learning
**Keywords**: multi-task reinforcement learning, meta-reinforcement learning, benchmark, reward function, reproducibility

## TL;DR

This paper systematically exposes how undocumented reward function inconsistencies across versions of the Meta-World benchmark distort algorithm comparisons, and releases a standardized new version, Meta-World+, which explicitly retains both V1 and V2 reward functions, introduces MT25/ML25 task sets, upgrades to the Gymnasium API, and enables fully reproducible evaluation for multi-task and meta-reinforcement learning.

## Background & Motivation

Meta-World is one of the most widely used benchmarks in multi-task RL and meta-RL, comprising 50 robotic manipulation tasks. However, since its initial release, its internal reward functions have undergone significant undocumented modifications, giving rise to the following critical issues:

**Version confusion**: The original V1 reward function was silently replaced by V2 at some point without explicit versioning. The V1 reward for pick-place spans roughly negative values to 1200, whereas V2 rewards lie in $(0, 10)$—differing fundamentally in magnitude and design philosophy.

**Distorted comparisons**: Papers published at different times used different reward function versions, making direct numerical comparisons inherently unfair. For example, PaCo achieves only 26.2% success on MT10 under V1 but 73.6% under V2.

**Lack of standardization**: Meta-World relied on the deprecated OpenAI Gym and Mujoco-Py packages, hampering long-term research use.

The authors' core motivation is to eliminate this confusion, establish a standardized evaluation platform, and provide empirical guidance for future benchmark design.

## Method

### Overall Architecture

Meta-World+ is an engineering-driven benchmark improvement rather than an algorithmic contribution; its core is a re-engineering of the existing benchmark. The framework operates at three levels: (1) reward function version management—explicitly retaining both V1 and V2 as selectable configurations; (2) task set expansion—introducing the MT25/ML25 intermediate-scale task sets; and (3) modernization—compatibility with the latest Gymnasium API and MuJoCo Python bindings.

### Key Designs

1. **Comparative analysis of V1 and V2 reward functions**: V1 rewards are derived by modifying a pick-place template for each task, resulting in large reward ranges and substantial cross-task variation. V2 introduces fuzzy constraints that normalize all task rewards to $(0, 10)$, producing a more uniform return distribution across tasks. Through Q-function loss analysis, the authors demonstrate that V2's uniform scale makes it easier for the Q-function to model state-action values, thereby improving overall success rates. This is consistent with the findings of PopArt—**consistent reward scaling across tasks is critical for effective multi-task learning**.

2. **New task sets MT25/ML25**: Intermediate-scale task sets are inserted between the existing MT10/MT50 and ML10/ML45 sets. MT25 requires approximately half the compute of MT50 (~12 hours vs. ~25 hours on an A100 GPU) while providing more thorough evaluation than MT10 (~6 hours). Users can also define task sets of arbitrary size and composition to support controlled experiments.

3. **Gymnasium integration and modernization**: Meta-World's custom environment implementation is aligned with the standard Gymnasium API, removing dependencies on the deprecated OpenAI Gym and Mujoco-Py. Users can directly leverage the full tooling and infrastructure of the Gymnasium ecosystem.

### Evaluation Protocol

Following the statistical recommendations of Agarwal et al. (2021), results are reported over 10 random seeds using the interquartile mean (IQM). For multi-task learning, each task is evaluated over 50 episodes (corresponding to 50 goal positions); for meta-learning, evaluation consists of 3 episodes following 10 adaptation episodes. All methods are re-implemented in JAX.

## Key Experimental Results

### Main Results

**Multi-task RL: V1 vs. V2 reward function comparison**

| Algorithm | Reported MT10 | MT10 V1 | MT10 V2 | Reported MT50 | MT50 V1 | MT50 V2 |
|-----------|--------------|---------|---------|--------------|---------|---------|
| SM | 71.8 | 71.4 | **84.9** | 61.0 | 60.6 | **65.8** |
| PaCo | 85.4 | 26.2 | **73.6** | 57.3 | 18.6 | **58.4** |
| MOORE | 88.7 | 61.4 | **83.2** | 72.9 | 61.2 | **72.0** |

**Meta-RL results (ML10/ML45)**

| Algorithm | ML10 V1 | ML10 V2 | ML45 V1 | ML45 V2 |
|-----------|---------|---------|---------|---------|
| MAML | ~35% | ~35% | ~25% | ~25% |
| RL2 | ~15% | ~35% | ~10% | ~25% |

### Ablation Study

| Configuration | MT10 Success | MT25 Success | MT50 Success | Notes |
|---------------|-------------|-------------|-------------|-------|
| MTMHSAC (V2) | ~75% | ~65% | ~60% | Performance degrades as task count increases (capacity issue) |
| MAML ML10 | ~35% | ~35% (ML25) | ~35% (ML45) | Meta-RL is insensitive to task set scale |

### Key Findings

- **All multi-task RL algorithms perform better under V2 than V1**: PCGrad and SM are the top-performing methods under both reward versions, consistent with their respective mechanisms of gradient projection and soft modularization for mitigating gradient conflicts.
- **Meta-RL is largely insensitive to reward version, with the exception of RL2**: MAML shows no statistically significant difference between V1 and V2 because it relies on policy gradients rather than Q-learning. RL2 suffers a sharp performance drop under V1 because the raw, unnormalized reward is directly fed as part of the observation.
- **PaCo's "spurious advantage" is exposed**: PaCo reports 85.4% on MT10, but this was obtained under V2 rewards; under V1, performance drops to only 26.2%, demonstrating that prior cross-version comparisons are entirely unreliable.

## Highlights & Insights

- This is a rigorous benchmark correction effort that explicitly calls out the community's bad habit of copying numbers from papers rather than re-running experiments.
- The empirical lesson from V2 reward design: **consistent reward scaling across tasks** is critical for multi-task RL and directly affects Q-function learning quality.
- MT25/ML25, as a compute-friendly intermediate option, offers practical value for initial algorithm screening before full MT50 evaluation.

## Limitations & Future Work

- The work focuses exclusively on the Sawyer single-arm manipulation environment and does not address cross-embodiment transfer.
- While V1/V2 discrepancies are identified, no improved reward function (V3) is designed; the contribution is limited to backward-compatible preservation of existing versions.
- Meta-RL baselines are restricted to two classic methods, MAML and RL2; more recent methods such as AMAGO-2 are not evaluated.
- Task diversity remains confined to tabletop manipulation, without coverage of more complex scenarios such as navigation or multi-stage tasks.

## Related Work & Insights

- Compared to manipulation benchmarks such as RLBench and MANISKILL3, Meta-World's distinguishing feature is that all tasks share the same state and action spaces, making multi-task and meta-learning tractable.
- The reward scaling principle from PopArt (Hessel et al., 2019) receives empirical validation here.
- The lessons of this paper generalize to all RL benchmarks: **version control and reward function design should be treated as first-class citizens of benchmark engineering**.

## Additional Notes

- The baseline codebase is implemented in JAX and open-sourced on GitHub for community reuse and extension.
- The appendix includes visualizations of all 50 tasks, complete design rationale for V1/V2 reward functions, and detailed composition of the MT25/ML25 task sets.
- The paper's discussion on "copying numbers from papers vs. re-running experiments" merits attention from authors of both benchmark and empirical papers.

## Rating

- Novelty: ⭐⭐⭐ Primarily an engineering contribution with limited methodological innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Cross-version comparisons are highly systematic; 10-seed IQM statistics are rigorous
- Writing Quality: ⭐⭐⭐⭐ Well-organized with thorough problem exposition
- Value: ⭐⭐⭐⭐ Significant corrective contribution to the community; the standardized benchmark release has lasting value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis](the_world_is_bigger_a_computationally-embedded_perspective_on_the_big_world_hypo.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](../../ICLR2026/reinforcement_learning/virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)

</div>

<!-- RELATED:END -->
