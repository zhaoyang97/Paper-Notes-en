---
title: >-
  [Paper Note] The Dark Side of the Forces: Assessing Non-Conservative Force Models for Atomistic Machine Learning
description: >-
  [ICML 2025 Oral][Physics & Scientific Computing][Energy Conservation] This work systematically assesses the catastrophic consequences of non-conservative machine learning interatomic potentials (which directly predict forces instead of deriving them from potential energy) in geometry optimization and molecular dynamics, and proposes a hybrid conservative/non-conservative model that balances efficiency and physical correctness using a multiple-timestep (MTS) scheme.
tags:
  - "ICML 2025 Oral"
  - "Physics & Scientific Computing"
  - "Energy Conservation"
  - "Non-conservative Forces"
  - "Molecular Dynamics"
  - "Machine Learning Interatomic Potentials"
  - "Multiple-Timestep"
date: 2026-05-08
content_hash: 046a617fc5ce062f
---

# The Dark Side of the Forces: Assessing Non-Conservative Force Models for Atomistic Machine Learning

**Conference**: ICML 2025 Oral  
**arXiv**: [2412.11569](https://arxiv.org/abs/2412.11569)  
**Code**: [Zenodo](https://zenodo.org/records/14778891)  
**Area**: Molecular Simulation / Machine Learning Interatomic Potentials  
**Keywords**: Energy Conservation, Non-conservative Forces, Molecular Dynamics, Machine Learning Interatomic Potentials, Multiple-Timestep

## TL;DR
This work systematically assesses the catastrophic consequences of non-conservative machine learning interatomic potentials (which directly predict forces instead of deriving them from potential energy) in geometry optimization and molecular dynamics, and proposes a hybrid conservative/non-conservative model that balances efficiency and physical correctness using a multiple-timestep (MTS) scheme.

## Background & Motivation

**Background**: Machine learning interatomic potentials (MLIPs) have become core tools in computational chemistry and materials science. Traditionally, forces are derived by taking the gradient of a potential energy function $V$ with respect to atomic positions $\mathbf{f}_j = -\partial V / \partial \mathbf{r}_j$ (conservative forces), which mathematically guarantees energy conservation.

**Limitations of Prior Work**: Backpropagation for deriving forces introduces 2-3 times inference overhead and 3 times training overhead. Consequently, recent models such as ORB, GemNet, and Equiformer bypass this derivative step and directly predict forces to improve efficiency.

**Key Challenge**: Direct force prediction breaks energy conservation—the Jacobian of the force field is no longer symmetric, and work done along closed loops is non-zero—but the impact of this on actual simulations has lacked systematic investigation.

**Goal**: Quantify the specific impacts of non-conservative forces on geometry optimization and NVE/NVT molecular dynamics, and seek practical trade-offs.

**Key Insight**: Unlike rotational symmetry breaking (which can be mitigated via data augmentation), energy conservation is a derivative constraint rather than an input symmetry, meaning it cannot be easily restored through training alone.

**Core Idea**: The optimal strategy is not to replace conservative models but to enhance them—using a hybrid model that accelerates inference via non-conservative forces and periodically corrects the trajectory using conservative forces.

## Method

### Overall Architecture
Based on the PET architecture, three types of models (conservative PET-C, non-conservative PET-NC, and hybrid PET-M) are trained. Their performance in terms of accuracy, stability, and efficiency is systematically compared through liquid water simulations. Additionally, a multiple-timestep (MTS) scheme is designed to integrate both types of forces.

### Key Designs

1. **Non-conservativeness Metric (Jacobian Anti-symmetry)**:
    - **Function**: Quantify the degree to which a force field deviates from being conservative.
    - **Mechanism**: Compute the ratio of the Frobenius norm of the anti-symmetric component of the force-field Jacobian $\mathbf{J}$ to the total Frobenius norm: $\lambda = \|\mathbf{J}_{\text{anti}}\|_F / \|\mathbf{J}\|_F$, where $\lambda=0$ represents a fully conservative force and $\lambda=1$ is fully non-conservative.
    - **Design Motivation**: Provide a pair-wise diagnostic of non-conservativeness, which reveals that non-conservativeness becomes relatively more severe at larger atomic distances, thereby impacting collective motion.

2. **Theoretical Analysis of Non-Conservative Effects**:
    - **Function**: Theoretically predict the behavior of non-conservative forces in various simulations.
    - **Mechanism**: Since non-conservative force fields lack a consistent potential energy definition, line-search geometry optimization fails and might continuously perform negative work along closed loops. Lacking a shadow Hamiltonian also breaks symplectic properties and invalidates the equipartition theorem.
    - **Design Motivation**: Provide theoretical explanations for experimental observations and clarify why energy conservation cannot be learned via data augmentation (as it is not an input symmetry).

3. **Hybrid Model and Multiple-Timestep Scheme (PET-M + MTS)**:
    - **Function**: Restore physical correctness while maintaining high efficiency.
    - **Mechanism**: The hybrid model is trained with both a conservative force head and a non-conservative force head. In the MTS scheme, equations of motion are integrated using the non-conservative forces at each step, and corrected using the conservative forces every $M$ steps, reducing the theoretical overhead from $F\approx2$ times to $1+(F-1)/M$.
    - **Design Motivation**: Pre-training with non-conservative forces followed by fine-tuning the conservative force head can significantly reduce training time. When $M=8$, MTS increases the inference overhead by only about 20%.

### Loss & Training
- Conservative model: Jointly trains the energy $V$ and the conservative forces $\mathbf{f}=-\nabla V$.
- Non-conservative model: Directly predicts the force $\mathbf{f}$ (with an optional energy head).
- Hybrid model: Simultaneously trains both force heads, or pre-trains a non-conservative model and then fine-tunes an energy head to generate conservative forces.

## Key Experimental Results

### Main Results: Accuracy Comparison (Liquid Water Dataset)

| Model | Type | Training Target | Energy MAE (meV/atom) | Force MAE (meV/Å) |
|------|------|---------|----------------------|-------------------|
| PET | Conservative | $V, \mathbf{f}$ | 0.55 | 19.4 |
| PET | Non-conservative | $V, \mathbf{f}$ | 1.42 | 24.8 |
| PET-M | Conservative Head | $V, \mathbf{f}$ | 0.59 | 20.2 |
| PET-M | Non-conservative Head | $V, \mathbf{f}$ | — | 26.7 |

### Ablation Study: Temperature Deviation in NVT Molecular Dynamics

| Thermostat / Model | Coupling Time $\tau$ (fs) | $\langle\Delta T\rangle$ (K) | $\langle T_H\rangle$ (K) | $\langle T_O\rangle$ (K) |
|------------|---------------------|------------------------------|--------------------------|--------------------------|
| PET-C / WN | 100 | 0.1 | 0.0 | 0.3 |
| PET-NC / WN | 1000 | 12.8 | 11.2 | 16.2 |
| PET-NC / WN | 100 | 1.4 | 1.3 | 1.6 |
| PET-NC / SVR | 10 | 1.0 | **-4.4** → **36.2 Deviation** | **-70** |
| PET-M (MTS 1:8) / SVR | 10 | 0.0 | -0.1 | 0.1 |

### Key Findings
- Non-conservative models exhibit force errors that are roughly 30% higher than conservative models (24.8 vs 19.4 meV/Å).
- In NVE dynamics, non-conservative forces lead to an unphysical heating rate of approximately 700 billion K/s, which is 10 times more severe in the ORB model.
- Although a global thermostat (SVR) regulates the total temperature, it leads to a temperature deviation of up to 36 K and 70 K between hydrogen and oxygen atoms (violating energy equipartition).
- The results of the MTS scheme ($M=8$) are essentially indistinguishable from those of the fully conservative model.
- The anti-symmetry of the Jacobian for non-conservative models is relatively more severe at large atomic separations, implying a greater impact on collective motion.
- Controlling non-conservative effects with a strong Langevin thermostat reduces the diffusion coefficient by a factor of 5, which offsets the gains in inference speed.

## Highlights & Insights
- This work is the first to systematically demonstrate the catastrophic consequences of non-conservative force models in actual simulations (featuring both theoretical analysis and quantitative experiments), highlighted by the striking 700 billion K/s heating rate.
- It reveals an counter-intuitive phenomenon: while a global thermostat appears to control the total temperature, it actually leads to a severe temperature imbalance between atom types, which is far harder to detect and correct than a simple temperature drift.
- The PET-M + MTS scheme is highly practical, completely restoring physical correctness with only about 20% additional overhead, while pre-training with non-conservative forces significantly accelerates training convergence.

## Limitations & Future Work
- The primary experiments are based on a liquid water system, while systematic validations on other material systems are deferred to the Appendix.
- A quantitative safety threshold for the non-conservativeness $\lambda$ (specifying the limit below which simulations remain acceptable) is not provided.
- The optimal selection of the conservative force evaluation frequency $M$ in the MTS scheme lacks theoretical guidance.
- The evaluation of foundation models (such as MACE-MP-0) is relatively limited.

## Related Work & Insights
- **vs ORB (Neumann et al., 2024)**: As a representative non-conservative foundation model, ORB has a $\lambda = 0.015$ on liquid water, and its NVE temperature drift is 10 times more severe than PET-NC, indicating that non-conservative issues are more prominent in general-purpose models.
- **vs Langer et al. (2024)**: Rotational symmetry breaking can be corrected via averaging during inference, but energy conservation cannot be addressed in the same way—this is the fundamental difference between these two physical constraints.
- **vs Eissler et al. (2025)**: Concurrent work has shown that non-conservative effects become more severe in larger systems, aligning with the analysis in this paper regarding the increase of Jacobian anti-symmetry with distance.

## Rating
- Novelty: ⭐⭐⭐⭐ This is a systematic evaluation rather than a completely new method, though the hybrid MTS scheme possesses originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Well-validated across multiple dimensions including accuracy, NVE, NVT, and geometry optimization, with additional material systems evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentative logic, with theory and experiments complementing each other perfectly.
- Value: ⭐⭐⭐⭐⭐ It sounds an alarm against the blind use of non-conservative force models in the MLIP field, where the hybrid scheme offers direct practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/physics/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](../../ICLR2026/physics/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[ICLR 2026\] PRO-MOF: Policy Optimization with Universal Atomistic Models for Controllable MOF Generation](../../ICLR2026/physics/pro-mof_policy_optimization_with_universal_atomistic_models_for_controllable_mof.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/physics/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/physics/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)

</div>

<!-- RELATED:END -->
