---
title: >-
  [Paper Note] Unveiling Multi-Regime Patterns in SciML: Diverse Failure Modes and Domain-Specific Optimization
description: >-
  [ICML 2026][Physics & Scientific Computing][SciML] This paper reveals three consistent failure modes in SciML models (PINNs, neural operators…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "SciML"
  - "Multi-domain Analysis"
  - "PINN"
  - "Failure Modes"
  - "Loss Landscape"
date: 2026-05-08
content_hash: d9a0c8189d48ba03
---

# Unveiling Multi-Regime Patterns in SciML: Diverse Failure Modes and Domain-Specific Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.29153](https://arxiv.org/abs/2605.29153)  
**Code**: https://github.com/leastima/sciml_multi_regime  
**Area**: Scientific Computing / Neural Network Optimization / Loss Landscape Analysis  
**Keywords**: SciML, Multi-domain Analysis, PINN, Failure Modes, Loss Landscape

## TL;DR
This paper reveals three consistent failure modes in SciML models (PINNs, neural operators, etc.) through a systematic multi-domain diagnostic framework and analyzes their loss landscape specificities to guide optimization method selection.

## Background & Motivation

**Limitations of Prior Work**: SciML methods such as PINNs, Fourier Neural Operators (FNO), and Neural ODEs encounter optimization difficulties and generalization failures in practical applications, yet a systematic framework for diagnosing these failure modes is lacking.

**Key Insight**: The structure of SciML loss landscapes is more complex than that of Computer Vision (CV), characterized by sharp minima, a lack of connectivity, and large Hessian eigenvalues—properties that contradict common intuitions in CV.

**Key Challenge**: Standard Hessian-loss correlation fails in SciML—low training loss does not necessarily correspond to low curvature, and high curvature does not necessarily imply poor training performance.

**Goal**: To establish a unified multi-domain diagnostic framework for understanding the structural roots of SciML failures and to provide regime-aware guidance for selecting optimization methods.

## Method

### Overall Architecture
The study proposes a three-dimensional diagnostic framework that systematically varies along three axes: (1) physical domain (PDE coefficients, equation types); (2) data domain (number of training samples/collocation points); and (3) optimization domain (optimizer choices, constraint handling strategies). By jointly analyzing training loss, test error, and loss landscape geometry, the framework automatically extracts regime boundaries.

### Key Designs

1.  **Three-Domain Regime Labeling**:
    - **Function**: Categorizes SciML models under different physical-data-optimization configurations into Well-Trained (Regime I, low training and test error), Under-Trained (Regime II, high training and test error), and Over-Trained (Regime III, low training error but high test error).
    - **Mechanism**: Automatically partitions regimes using training and test error thresholds $T_{\text{train}}$ and $T_{\text{test}}$, with boundary robustness assessed via $\pm 20\%$ threshold perturbations.
    - **Design Motivation**: Provides a task-oblivious perspective on failure modes, bypassing the limitations of solely examining task-level performance.

2.  **Loss Landscape Pathological Phenomenon Detection**:
    - **Function**: Identifies two types of non-intuitive phenomena: (a) Deceptive Sharpness: high Hessian eigenvalues corresponding to low training loss; (b) Deceptive Flatness: low Hessian eigenvalues masking high training loss.
    - **Mechanism**: Concurrently tracks the dynamic curves of $\lambda_{\max}$ (maximum eigenvalue) and training loss, discovering that both move in the same direction during an "Increasing Sharpening" phase. Hessian spectral density estimation reveals that PINNs lack the zero-eigenvalue peaks typically found in CV models.
    - **Design Motivation**: Uncovers the fundamental reasons why standard CV-inspired landscape intuitions (e.g., "flat minima are better") fail in SciML.

3.  **Regime-Aware Optimization Effectiveness Analysis**:
    - **Function**: Systematically compares the performance of Adam, L-BFGS, NNCG, ALM, RoPINN, and CL across different regimes.
    - **Mechanism**: Generates 2D regime heatmaps (physical parameters × data volume) for each optimization method to calculate relative performance gains. NNCG improves test error by approximately 50% compared to L-BFGS in Regime I but remains unstable in Regimes II and III.
    - **Design Motivation**: Demonstrates that no single optimizer is universally optimal, necessitating targeted selection based on the current regime.

## Key Experimental Results

### Regime Structure Consistency Verification

| Model | Dataset | Regime I | Regime II | Regime III | Key Phenomenon |
|------|--------|----------|-----------|------------|---------|
| PINN | 1D Convection | $\beta < 25$ | $25 \leq \beta < 50$ | $\beta \geq 50$ (Sparse) | Increasing physical parameters shifts boundaries right |
| FNO | 2D Advection-Diffusion | Sufficient samples | Moderate pressure | Sparse samples | Smooth transitions instead of the sharp boundaries seen in PINNs |
| PINODE | Nonlinear Pendulum | Standard config | High-dimensional | Low data | Intermediate characteristics between PINN and FNO |

### Optimization Method Effectiveness Comparison

| Optimization Method | Regime I | Regime II | Regime III | Best Application Scenarios |
|----------|----------|-----------|------------|-----------|
| L-BFGS | ✓ | ✓ (Prone to failure) | ✗ | Basic training |
| ALM | ✓ | ✓✓ (Constraint hardening) | ✗ | Constraint-critical problems |
| CL (Curriculum Learning) | ✓ | ✓✓ | ✓ | Difficult configurations |
| NNCG | ✓✓ (+50%) | ✗ (Unstable) | ✗ | Regime I fine-tuning |

## Highlights & Insights
- **Counter-intuitive Design of Deceptive Sharpness**: Reveals that high-curvature regions in SciML can actually correspond to optimal solutions, contradicting the "flat minima are better" hypothesis prevalent in CV.
- **Breakdown of Hessian-Loss Correlation**: Quantitatively proves through spectral density comparison (PINNs lack zero-eigenvalue peaks compared to ResNet) that SciML landscapes are fundamentally different from CV landscapes.
- **Universal Failure Mode Framework**: Although PINN, FNO, and Neural ODE architectures differ significantly, consistent three-domain regime structures emerge across all, indicating that these failure modes are systemic issues in SciML.

## Limitations & Future Work
- Experiments were primarily conducted on small-scale 1D/2D problems; generalization to large-scale 3D PDEs remains to be verified.
- The high computational cost of Hessian matrices makes it difficult to scale the framework to very large models.
- The location of regime boundaries varies significantly with different PDE coefficients, making it hard to define universal thresholds.
- Future Directions: Design adaptive regime detection algorithms to identify the current regime online and automatically switch optimization strategies accordingly.

## Related Work & Insights
- **vs. Loss Landscape Research (Yang et al.)**: Previous studies focus on landscape connectivity in CV/NLP; this paper finds that SciML lacks these properties and requires specialized diagnostic tools.
- **vs. PINN Optimization Research (Krishnapriyan et al.)**: Prior work offers fragmented analyses of local failure phenomena in PINNs; this paper provides a unified multi-domain perspective and quantitative regime labeling.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending landscape diagnostics from CV to SciML with a systematic multi-dimensional analysis is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of 5 SciML models, 4 PDEs, and 5 optimization methods.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with rich visualizations.
- Value: ⭐⭐⭐⭐ Provides SciML practitioners with clear guidelines for optimizer selection and a robust failure diagnostic tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](../../ICLR2026/physics/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](../../NeurIPS2025/physics/from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)

</div>

<!-- RELATED:END -->
