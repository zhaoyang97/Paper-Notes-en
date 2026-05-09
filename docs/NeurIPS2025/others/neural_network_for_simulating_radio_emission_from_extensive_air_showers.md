---
title: >-
  [Paper Note] Neural Network for Simulating Radio Emission from Extensive Air Showers
description: >-
  [NeurIPS 2025][Cosmic rays] A simple fully connected neural network is employed to replace computationally expensive CoREAS Monte Carlo simulations, enabling fast prediction of radio pulses from extensive air showers (EAS) while achieving $X_{\text{max}}$ reconstruction resolution comparable to conventional simulations.
tags:
  - NeurIPS 2025
  - Cosmic rays
  - extensive air showers
  - radio emission simulation
  - neural network surrogate model
  - "$X_{\text{max}}$ reconstruction"
date: 2026-05-08
content_hash: c5712500f722735c
---

# Neural Network for Simulating Radio Emission from Extensive Air Showers

**Conference**: NeurIPS 2025
**arXiv**: [2512.21407](https://arxiv.org/abs/2512.21407)
**Code**: [Available](https://anonymous.4open.science/r/radio_nn-21BF/)
**Area**: AI for Science / Astroparticle Physics
**Keywords**: Cosmic rays, extensive air showers, radio emission simulation, neural network surrogate model, $X_{\text{max}}$ reconstruction

## TL;DR

A simple fully connected neural network is employed to replace computationally expensive CoREAS Monte Carlo simulations, enabling fast prediction of radio pulses from extensive air showers (EAS) while achieving $X_{\text{max}}$ reconstruction resolution comparable to conventional simulations.

## Background & Motivation

When cosmic rays arrive at Earth from distant astrophysical sources and interact with the atmosphere, they produce **extensive air showers (EAS)**—cascades of secondary particles. Detecting the radio emission of EAS using antenna arrays is an important approach for studying the energy spectrum and mass composition of cosmic rays.

**Core bottleneck**: Current reconstruction methods require microscopic simulation for each observed event, computing the electromagnetic radiation contribution of every particle track and superimposing them onto each antenna. Simulating a single event can take **several weeks**, and the analysis pipeline requires multiple iterative simulations per event to quantify uncertainties.

**Existing acceleration methods** (interpolation, template synthesis) remain computationally demanding and cannot generalize across different event geometries.

**Key observation**: Although the microscopic particle interactions within a shower are stochastic, the **macroscopic radio emission is deterministic**—given shower parameters ($X_{\text{max}}$, electromagnetic energy, arrival direction, etc.), the pulse shape received at each antenna is fully determined. This reduces the problem from generative (learning a distribution) to **regression** (learning a deterministic mapping).

## Method

### Overall Architecture

A point-cloud-inspired approach is adopted: the entire radio emission prediction is decomposed into per-antenna-location prediction tasks. The network takes shower parameters and antenna position as input, and outputs the radio pulse waveform at that antenna in two polarization directions.

### Key Designs

1. **Input features (11-dimensional)**:

    - Shower physical parameters: $X_{\text{max}}$ (atmospheric depth of shower maximum), electromagnetic energy $E_{em}$, geomagnetic field angle, density and altitude at $X_{\text{max}}$, primary cosmic-ray energy, zenith angle, azimuth angle
    - Antenna position: coordinates in the shower frame (3-dimensional)

2. **Output (512-dimensional)**:

    - 256 time bins per polarization direction ($v \times B$ and $v \times v \times B$)
    - Time resolution: 1 ns

3. **Network architecture**:

    - **8-layer fully connected network** with approximately 4 million parameters; memory footprint of only 19 MB
    - **No skip connections**: at this depth, vanishing gradients are not observed
    - Inputs are normalized to the range $[-1, 1]$

### Loss & Training

- **L1 loss** $\mathcal{L} = |x - y|$ rather than L2—ensuring that weak pulses (which contribute little to L2 loss) are also learned adequately
- The second polarization (weaker) is assigned **higher weight** to ensure both polarization directions are correctly learned
- Adam optimizer with weight decay regularization
- Learning rate schedule: warmup followed by gradual decay (Figure 2, right)

**Training data**: The CoREAS simulation library from the Pierre Auger Collaboration's AERA array, comprising 2,158 distinct shower parameter configurations × 27 realizations ≈ 58k simulations, each containing 240 antenna positions. An 80/20 train/test split is applied.

**Training cost**: Completed within one week on a standard desktop CPU (AMD Ryzen 7 PRO 3700); inference requires only a few hundred milliseconds per antenna position.

## Key Experimental Results

### Pulse Quality Evaluation

| Metric | Result |
|--------|--------|
| Energy flux (fluence) accuracy | Within **10%** for the majority of cases |
| Normalized correlation for strong pulses | **Highly correlated** |
| Error for weak pulses (< 1e-2 eV/m²) | Up to 100%, but negligible as they are buried in noise |

The paper notes that the discrepancy between the two leading radio emission models (CoREAS vs. ZHS) is itself approximately 10%, placing the neural network's errors within the range of model systematic uncertainties.

### $X_{\text{max}}$ Reconstruction — Bias and Resolution Comparison (Table 1)

| Simulation | 5% noise — Bias | 5% noise — Resolution | 10% noise — Bias | 10% noise — Resolution |
|------------|-----------------|----------------------|-----------------|------------------------|
| CoREAS | -12.87 g/cm² | 32.22 g/cm² | -10.36 g/cm² | 31.42 g/cm² |
| **Neural Network** | **-12.71 g/cm²** | **33.13 g/cm²** | **-11.79 g/cm²** | **32.38 g/cm²** |

- Bias and resolution are nearly identical across both noise levels
- The differences are far smaller than the systematic uncertainty between radio emission models (~11 g/cm²)

### Computational Efficiency Comparison

| Aspect | CoREAS | Neural Network |
|--------|--------|----------------|
| Single-event simulation time | Hours to weeks | **Milliseconds** (per antenna) |
| Parameter count | — | ~4 million |
| Model size | — | 19 MB |
| GPU parallelization | Difficult | **Natively supported** |

### Key Findings

- A simple fully connected network can capture deterministic macroscopic radio emission **without the need for complex generative models**
- $X_{\text{max}}$ reconstruction performance is comparable to full CoREAS simulations, validating the feasibility of the neural network surrogate
- Trends in bias and resolution across different shower geometries are consistent with CoREAS
- The model has been successfully transferred via fine-tuning to the LOFAR experiment (different atmospheric and experimental conditions) using only a small amount of additional data

## Highlights & Insights

- **Problem reformulation is the key contribution**: recasting the stochastic microscopic simulation problem as a deterministic macroscopic regression task fundamentally reduces modeling complexity. This insight has broader implications for surrogate modeling in physical simulations.
- **Minimalist design**: an 8-layer fully connected network with a 19 MB footprint, employing no elaborate architectural tricks, yet achieving physically meaningful high accuracy.
- **Differentiability dividend**: the neural network surrogate is inherently differentiable, enabling integration with frameworks such as Information Field Theory (IFT).
- **Transfer learning viability**: a model trained on one experiment can be fine-tuned to another with only a small amount of data.

## Limitations & Future Work

- Simulation accuracy is lower for weak pulses (low signal-to-noise regions), though the authors argue these regions are masked by noise in practice
- Current training and validation are limited to the AERA frequency band (30–80 MHz) and associated atmospheric conditions
- $X_{\text{max}}$ reconstruction employs a simplified procedure (single scale factor $S=1$); the actual analysis pipeline is more complex
- The longitudinal shower profile is not included as an additional input feature
- Training data originate from a specific collaboration and are limited in volume
- A bias exists at the low-$X_{\text{max}}$ end of the sampling due to an asymmetric number of simulations on either side during parabolic fitting

## Related Work & Insights

- **Calorimeter simulation acceleration** (CaloFlow, CaloDiffusion, CaloScore, etc.) employs generative models to accelerate particle physics detector simulations, but faces challenges in achieving high fidelity.
- The **hybrid approach** adopted in this work—using conventional Monte Carlo for stochastic particle showering and a neural network for deterministic radio emission—represents a pragmatic and efficient strategy.
- **Inspiration from point cloud techniques**: treating radio emission as a point-cloud prediction over antenna positions enables geometry-agnostic generalization.
- The method has direct applicability to the radio astronomy community (AERA, LOFAR, SKA, GRAND, and related experiments).

## Rating

- Novelty: ⭐⭐⭐ — The methodology is relatively straightforward; the primary contribution lies in the problem reformulation insight and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation covering pulse quality, fluence accuracy, and $X_{\text{max}}$ reconstruction bias/resolution.
- Writing Quality: ⭐⭐⭐⭐ — Concise and clear, with well-motivated physical reasoning.
- Value: ⭐⭐⭐⭐ — Reducing simulation time from weeks to milliseconds represents substantial practical value for experimental physics.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](../../ICLR2026/others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)

<!-- RELATED:END -->
