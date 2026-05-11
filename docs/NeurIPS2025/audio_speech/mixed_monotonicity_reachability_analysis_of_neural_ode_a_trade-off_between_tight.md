---
title: >-
  [Paper Note] Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency
description: >-
  [NeurIPS 2025 (NeurReps 2025 Workshop, co-located with NeurIPS 2025)][Audio & Speech][Neural ODE] This paper applies continuous-time mixed monotonicity techniques to the reachability analysis of Neural ODEs. By embedding…
tags:
  - "NeurIPS 2025 (NeurReps 2025 Workshop, co-located with NeurIPS 2025)"
  - "Audio & Speech"
  - "Neural ODE"
  - "Reachability Analysis"
  - "Mixed Monotonicity"
  - "Interval Propagation"
  - "Formal Verification"
date: 2026-05-08
content_hash: 725053ba3366e946
---

# Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency

**Conference**: NeurIPS 2025 (NeurReps 2025 Workshop, co-located with NeurIPS 2025)
**arXiv**: [2510.17859](https://arxiv.org/abs/2510.17859)
**Code**: Available (TIRA tool)
**Area**: Formal Verification / Safety-Critical Systems
**Keywords**: Neural ODE, Reachability Analysis, Mixed Monotonicity, Interval Propagation, Formal Verification

## TL;DR

This paper applies continuous-time mixed monotonicity techniques to the reachability analysis of Neural ODEs. By embedding Neural ODE dynamics into a mixed monotone system, it exploits the geometric simplicity of interval boxes to achieve efficient over-approximation, providing a controllable trade-off between tightness and computational efficiency.

## Background & Motivation

**Background**: Neural ODEs are a powerful class of continuous-time machine learning models capable of describing the behavior of complex dynamical systems. They are widely applied in trajectory prediction, system control, and physics-based modeling.

**Safety Requirements**: In safety-critical scenarios (autonomous driving, aerospace, medical devices), it is essential to verify that the outputs of Neural ODEs remain within admissible safety bounds—this is the problem of reachability analysis.

**Limitations of Prior Work**:
- Existing reachability analysis tools (e.g., zonotope-based methods in CORA, star set-based methods in NNV2.0) incur high computational complexity.
- Although zonotopes and star sets yield tight over-approximations, their computational cost grows rapidly in high-dimensional settings.
- No methods specifically tailored to the continuous-time dynamical characteristics of Neural ODEs exist.

**Key Challenge**: The inherent trade-off between tightness and efficiency—tighter over-approximations require greater computational resources.

**Key Insight**: Leveraging the concept of mixed monotonicity from dynamical systems theory, the Neural ODE is embedded into a mixed monotone system, enabling reachability analysis via simple interval arithmetic.

## Method

### Overall Architecture

The core pipeline of the proposed method:
1. Given an initial set (represented as an interval box) for the Neural ODE.
2. Embed the Neural ODE dynamics into a mixed monotone system.
3. Solve the embedded system's ODE to obtain an over-approximation of the reachable set.
4. Provide three implementation strategies: single-step, incremental, and boundary-based.

### Key Designs

#### Mixed Monotonicity Embedding

- **Mixed Monotone System**: For a system $\dot{x} = f(x)$, if there exists a decomposition function $\hat{f}(x, \hat{x})$ such that $\hat{f}$ is monotonically non-decreasing in $x$ and monotonically non-increasing in $\hat{x}$, the system is said to be mixed monotone.
- **Key Advantage**: The reachable set of a mixed monotone system can be over-approximated by tracking only two extremal trajectories (upper and lower bounds).
- **Embedding for Neural ODEs**: The embedding system is constructed by analyzing the monotonicity of activation functions (e.g., ReLU, Sigmoid) within the Neural ODE.

#### Three Analysis Strategies

1. **Single-step**: Propagates directly from the initial interval to the final time; fastest to compute but yields the loosest over-approximation.
2. **Incremental**: Divides the time horizon into multiple segments, propagating and tightening intermediate results at each step; offers a balance between tightness and efficiency.
3. **Boundary-based**: Exploits the homeomorphism property to propagate only the boundary points of the initial set, yielding the tightest results.

#### Homeomorphism Property

- The flow map of a Neural ODE is a homeomorphism.
- This implies that the boundary of the initial set maps to the boundary of the reachable set.
- By exploiting this property, only boundary points need to be tracked rather than the entire volume, significantly reducing computational cost.

### Implementation: TIRA Tool

- A Python-based reachability analysis tool.
- Supports flexible switching among the three strategies.
- Capable of automatically constructing the mixed monotone embedding.

## Key Experimental Results

### Main Results

Evaluation is conducted on two standard Neural ODE systems:

#### Spiral System — 2D

| Method | Reachable Set Volume | Computation Time (s) | Tightness Ratio |
|--------|---------------------|----------------------|-----------------|
| CORA (Zonotope) | 0.0842 | 12.3 | 1.00× |
| NNV2.0 (Star Set) | 0.0756 | 28.7 | 0.90× |
| TIRA Single-step | 0.1523 | **0.4** | 1.81× |
| TIRA Incremental (10 segments) | 0.1087 | 1.8 | 1.29× |
| TIRA Boundary-based | 0.0912 | 3.2 | 1.08× |

#### Fixed-Point Attractor System — 2D

| Method | Reachable Set Volume | Computation Time (s) | Tightness Ratio |
|--------|---------------------|----------------------|-----------------|
| CORA (Zonotope) | 0.0312 | 8.7 | 1.00× |
| NNV2.0 (Star Set) | 0.0289 | 19.4 | 0.93× |
| TIRA Single-step | 0.0687 | **0.2** | 2.20× |
| TIRA Incremental (10 segments) | 0.0423 | 0.9 | 1.36× |
| TIRA Boundary-based | 0.0354 | 1.7 | 1.13× |

### Ablation Study

#### Effect of Segment Count in Incremental Method on Tightness–Efficiency Trade-off (Spiral System)

| Segments | Reachable Set Volume | Computation Time (s) | Volume/Time Ratio |
|----------|---------------------|----------------------|-------------------|
| 1 (Single-step) | 0.1523 | 0.4 | 0.381 |
| 5 | 0.1241 | 1.1 | 0.113 |
| 10 | 0.1087 | 1.8 | 0.060 |
| 20 | 0.0998 | 3.4 | 0.029 |
| 50 | 0.0945 | 8.1 | 0.012 |

#### Effect of Initial Set Size on Method Performance

| Initial Interval Radius | CORA Time | TIRA Incremental Time | TIRA Volume / CORA Volume |
|------------------------|-----------|----------------------|---------------------------|
| 0.01 | 8.2s | 1.2s | 1.15× |
| 0.05 | 10.5s | 1.6s | 1.28× |
| 0.10 | 12.3s | 1.8s | 1.29× |
| 0.50 | 18.7s | 2.4s | 1.52× |

### Key Findings

1. **Significant Efficiency Advantage**: TIRA's single-step method is approximately 30× faster than CORA and approximately 70× faster than NNV2.0.
2. **Controllable Tightness**: The segment count parameter of the incremental method enables flexible trade-offs between tightness and efficiency.
3. **Boundary-based Method Balances Both**: By exploiting the homeomorphism property, the boundary-based method approaches CORA's tightness while maintaining high efficiency.
4. **Larger Initial Sets Yield Looser Over-approximations**: This is an inherent characteristic of interval methods, though TIRA maintains its efficiency advantage.
5. **Soundness Guarantee**: All methods provide mathematically rigorous over-approximation guarantees.

## Highlights & Insights

1. **Theoretical Contribution**: This is the first work to apply the concept of mixed monotonicity to the reachability analysis of Neural ODEs, establishing a bridge between dynamical systems theory and neural network verification.
2. **Practical Value**: The approach provides a lightweight formal analysis method for high-dimensional, real-time, safety-critical scenarios.
3. **Scalability**: The complexity of interval box representations scales linearly with dimension, in contrast to zonotopes or star sets, which scale polynomially or exponentially.
4. **Methodological Insight**: The strategy of exploiting problem structure (monotonicity) to simplify analysis is generalizable to other verification tasks.

## Limitations & Future Work

1. **Only 2D Systems Demonstrated**: Experimental validation in high-dimensional settings is insufficient; although the method is theoretically scalable, its practical effectiveness remains to be verified.
2. **Over-approximations Remain Loose**: In particular, the single-step method can yield over-approximation volumes more than twice that of optimal methods.
3. **Embedding Quality Depends on Network Architecture**: Certain network architectures may be difficult to embed into a tight mixed monotone form.
4. **Deterministic ODEs Only**: Stochastic Neural ODEs or systems with noise are not supported.
5. **Workshop Paper**: As a NeurReps 2025 workshop paper, the experimental scale is relatively limited.

## Related Work & Insights

- **CORA**: A zonotope-based reachability analysis tool that provides tight over-approximations at high computational cost.
- **NNV2.0**: A star set-based neural network verification framework supporting verification of image classification networks.
- **Mixed Monotonicity Theory**: Originating from dynamical systems theory, previously applied primarily to conventional (non-neural-network) systems.
- **Neural ODE**: Proposed by Chen et al. (2018), generalizing ResNets to continuous time.

## Rating

- Novelty: ★★★★☆ (Novel application of mixed monotonicity to Neural ODEs)
- Experimental Thoroughness: ★★★☆☆ (Only 2D examples; lacks high-dimensional validation)
- Value: ★★★★☆ (Provides lightweight verification for safety-critical systems)
- Writing Quality: ★★★★☆ (Clear and systematic; theoretical presentation is rigorous)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost](slimmable_nam_neural_amp_models_with_adjustable_runtime_computational_cost.md)
- [\[NeurIPS 2025\] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](../../ACL2026/audio_speech/still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)

</div>

<!-- RELATED:END -->
