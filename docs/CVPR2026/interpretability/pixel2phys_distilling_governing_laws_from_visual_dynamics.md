---
title: >-
  [Paper Note] Pixel2Phys: Distilling Governing Laws from Visual Dynamics
description: >-
  [CVPR 2026][Interpretability][Physical law discovery] Pixel2Phys is proposed as a multi-agent collaborative framework built upon MLLMs, employing four agents — Plan, Variable, Equation…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Physical law discovery"
  - "multi-agent framework"
  - "symbolic regression"
  - "video understanding"
  - "AI for Science"
date: 2026-05-08
content_hash: 79560e4f259bd029
---

# Pixel2Phys: Distilling Governing Laws from Visual Dynamics

**Conference**: CVPR 2026
**arXiv**: [2602.19516](https://arxiv.org/abs/2602.19516)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Physical law discovery, multi-agent framework, symbolic regression, video understanding, AI for Science

## TL;DR
Pixel2Phys is proposed as a multi-agent collaborative framework built upon MLLMs, employing four agents — Plan, Variable, Equation, and Experiment — in an iterative hypothesize-verify-refine loop to automatically discover interpretable governing equations from raw videos, achieving a 45.35% improvement in extrapolation accuracy over baselines.

## Background & Motivation
**Background**: Discovering physical laws from observational data is a central goal of scientific intelligence. Traditional approaches rely on manual extraction of physical quantities followed by symbolic regression, which is labor-intensive and slow.

**Limitations of Prior Work**:
   - Supervised equation prediction models require scarce equation-video paired data and generalize poorly.
   - Unsupervised latent-space methods (Autoencoder + symbolic regression) have their latent spaces determined by the reconstruction objective, making it easy for physically irrelevant factors (texture, lighting) to contaminate the representation.
   - Directly prompting MLLMs primarily retrieves prior knowledge from training corpora and struggles to derive new laws from raw visual data.

**Key Challenge**: Variable extraction and equation discovery are mutually dependent in a chicken-and-egg cycle — a clean variable space requires knowledge of the dynamics, while discovering dynamics requires a clean variable space.

**Goal**: Simultaneously discover physical variables $z(t)$ and governing equations $f$, i.e., $\frac{dz}{dt} = f(z(t))$.

**Key Insight**: Emulate the collaborative workflow of human scientists — observe, hypothesize, experiment, refine — by constructing an iterative multi-agent framework.

**Core Idea**: Coordinate four specialized agents via MLLMs for iterative scientific reasoning, breaking the circular dependency between variable extraction and equation discovery.

## Method

### Overall Architecture
Four agents collaborate in sequence: Plan Agent (global coordination) → Variable Agent (physical variable extraction) → Equation Agent (equation discovery via symbolic regression) → Experiment Agent (evaluation and validation) → Plan Agent (report analysis and determination of the next refinement direction).

### Key Designs

1. **Plan Agent (Global Planning)**:

    - Serves as the central coordinator; aggregates reports from the three agents each round for two-stage diagnosis.
    - First checks visualization-based qualitative fit, then examines quantitative metrics to localize bottlenecks.
    - Determines refinement strategy based on diagnosis: variable refinement (re-extract $\mathcal{Z}$) or equation refinement (adjust search hyperparameters).
    - Design Motivation: Break the variable–equation circular dependency, enabling mutual refinement of both.

2. **Variable Agent (Multi-Granularity Variable Extraction)**:

    - **Object-level Tool**: Applies SAM segmentation + tracking to extract motion trajectories $z(t) = [x(t), y(t)]$.
    - **Pixel-level Tool**: Computes spatial derivatives (Laplacian, bi-harmonic) via fixed convolutional kernels, suited for PDE-driven physical fields.
    - **Representation-level Tool**: A physics-informed autoencoder with loss $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{eq}\mathcal{L}_{eq}$, where $\mathcal{L}_{eq} = \|\mathcal{F}(z) - f(z)\|^2$ enforces the latent space to conform to the discovered equations.
    - Design Motivation: Different types of physical systems require variable extraction at different granularities.

3. **Equation Agent (Dynamic Symbolic Regression)**:

    - Estimates time derivatives $\dot{Z}$ via central differences.
    - Constructs a candidate function library $\Theta(Z)$: polynomial terms and transcendental functions.
    - Solves for the sparse coefficient matrix $\Xi$ via STLSQ under $\|\dot{Z} - \Theta(Z)\Xi\|_2^2 + \lambda_{sp}\|\Xi\|_1$.
    - $\lambda_{sp}$ is adjusted under guidance from the Plan Agent.

4. **Experiment Agent (Multi-Dimensional Evaluation)**:

    - Equation quality: $R^2$ score + complexity ($L_0$ of $\Xi$).
    - Variable quality: phase-space visualization.
    - Extrapolation fidelity: integration from initial conditions to produce predictions, with RMSE computation.
    - Aggregates quantitative metrics and plots into a structured report.

### Loss & Training
The Representation-level Tool in the Variable Agent is trained via a physics-informed autoencoder. In early iterations when no equation prior is available, only the reconstruction loss is applied; the physical consistency loss is incorporated jointly in later iterations.

## Key Experimental Results

### Main Results (Object-level dynamics)

| Case | Method | Terms Found | False Positives | $R^2$@1000 |
|------|------|:-----------:|:---------:|----------|
| Linear | Coord-Equ | Yes | 1.10 | 0.8647 |
| Linear | **Pixel2Phys** | **Yes** | **0** | **0.9913** |
| Cubic | Coord-Equ | No | 3.40 | 0.2632 |
| Cubic | **Pixel2Phys** | **Yes** | **0.39** | **0.9886** |
| VDP | Coord-Equ | Yes | 2.31 | 0.4920 |
| VDP | **Pixel2Phys** | **Yes** | **0.99** | **0.9954** |

### Main Results (Pixel-level PDE dynamics)

| Dataset | Method | RMSE↓ | VPS@0.5↑ |
|--------|------|-------|----------|
| Lambda-Omega | PDE-Find | 0.67 | 492 |
| Lambda-Omega | **Pixel2Phys** | **0.03** | **1000** |
| Brusselator | SGA-PDE | 0.14 | 1000 |
| Brusselator | **Pixel2Phys** | **0.12** | **1000** |
| FHN | PDE-Find | 0.63 | 54 |
| FHN | **Pixel2Phys** | **0.16** | **1000** |

### Key Findings
- Implicit methods (Latent-ODE, AE-SINDy) completely collapse in long-term extrapolation ($R^2 \approx 0$), demonstrating that general-purpose representations fail to capture physical structure.
- Pixel2Phys produces far fewer false-positive terms than Coord-Equ, yielding more concise and accurate equations.
- In PDE settings, neural operators (FNO/UNO) suffer from severe error accumulation, whereas Pixel2Phys correctly identifies high-order operators (bi-harmonic).
- The framework successfully recovers the law of gravitation and the Navier-Stokes equations from real-world videos.

## Highlights & Insights
- **Multi-agent scientific reasoning framework**: Using MLLMs as a planner to coordinate specialized agents, this work is the first to automate the scientific methodology of "observe–hypothesize–experiment–refine," and the framework is transferable to other scientific domains such as biology and chemistry.
- **Elegant design of the physics-informed autoencoder**: During iteration, already-discovered equations in turn guide the refinement of the variable space, breaking the variable–equation circular dependency.
- **Multi-granularity tool selection**: The three-level Object/Pixel/Representation toolset covers the full spectrum from discrete objects to continuous fields to implicit dynamics.

## Limitations & Future Work
- Relies on GPT-4o as the backbone, incurring high cost.
- Capability on multi-body interactions (N-body problem) remains to be validated.
- When the dimensionality of physical variables is high, the search space for symbolic regression grows exponentially.
- Chaotic systems in the real world may prevent iterative convergence.

## Related Work & Insights
- **vs. Coord-Equ (pipeline-based)**: Coord-Equ relies on a pretrained tracker to extract coordinates followed by a single-pass symbolic regression, which cannot handle continuous fields and readily introduces false positives. Pixel2Phys addresses both issues through iterative refinement and multi-granularity tools.
- **vs. End-to-end methods (AE-SINDy)**: In end-to-end methods, the variable space is determined by the reconstruction objective, and physically irrelevant factors contaminate the representation, causing extrapolation failure. Pixel2Phys explicitly constrains the variable space via the physical consistency loss.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A fundamentally new paradigm for AI for Science via multi-agent scientific reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across three scenario types, including real-world validation.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, though the density of equations warrants careful reading.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for MLLM-driven scientific discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)

</div>

<!-- RELATED:END -->
