---
title: >-
  [Paper Note] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number
description: >-
  [NeurIPS 2025][Medical Imaging][Structure-based drug design] This paper proposes PAFlow, a 3D molecule generation model built on the flow matching framework…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Structure-based drug design"
  - "flow matching"
  - "protein–ligand interaction guidance"
  - "atom number prediction"
  - "3D molecule generation"
date: 2026-05-08
content_hash: 5fd69b32a88f2d5a
---

# Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number

**Conference**: NeurIPS 2025
**arXiv**: [2509.01486](https://arxiv.org/abs/2509.01486)  
**Code**: [GitHub](https://github.com/CMACH508/PAFlow)  
**Area**: Medical Imaging / Drug Discovery
**Keywords**: Structure-based drug design, flow matching, protein–ligand interaction guidance, atom number prediction, 3D molecule generation

## TL;DR

This paper proposes PAFlow, a 3D molecule generation model built on the flow matching framework, which guides the vector field via a protein–ligand interaction predictor and determines atom counts through a learnable atom number predictor. PAFlow achieves a new state-of-the-art Avg. Vina Score of −8.31 on CrossDocked2020, substantially outperforming existing methods.

## Background & Motivation

Structure-based drug design (SBDD) aims to generate 3D molecules with high binding affinity toward a target protein. Existing approaches suffer from three key limitations: (1) autoregressive models follow an unnatural generation order, leading to unreasonable fragments and error accumulation; (2) diffusion models exhibit highly stochastic denoising trajectories, resulting in unstable molecular quality; (3) all non-autoregressive methods sample atom counts from predefined distributions that rely on reference ligand priors, frequently causing a mismatch between molecular size and pocket geometry. The flow matching (FM) framework enables fast and stable generation via ODE solvers and thus holds promise for addressing the first two issues.

## Method

### Overall Architecture

PAFlow models the molecule generation process within the FM framework. Atomic coordinates are modeled using a Variance Preserving (VP) path, while atom types are modeled using a newly derived discrete conditional flow matching (CFM) formulation. The generation process iteratively updates from initial noise to the target molecule via an Euler ODE solver, augmented by an interaction predictor for guidance and a learnable atom number predictor.

### Key Designs

1. **Dual-Path Flow Matching**: Continuous atomic coordinates $\mathbf{x}$ follow the VP-path conditional probability $p_t(\mathbf{x}|\mathbf{x}_1) = \mathcal{N}(\mathbf{x}|\sqrt{\bar{\alpha}_{1-t}}\mathbf{x}_1, (1-\bar{\alpha}_{1-t})\mathbf{I})$; discrete atom types $\mathbf{a}$ follow a categorical distribution path $\mathbf{c}(\mathbf{a}, \mathbf{a}_1) = \bar{\alpha}_{1-t}\mathbf{a}_1 + (1-\bar{\alpha}_{1-t})/K$. The authors derive a conditional vector field for discrete atom types, $u_t^a = \bar{\alpha}_{1-t}'(\mathbf{a}_1 - \mathbf{a}_0)$, unifying both modalities under the FM framework. The entire model is parameterized by an SE(3)-equivariant GNN $\phi_\theta$, ensuring translational and rotational invariance with respect to the protein–ligand complex.

2. **Prior-Guided Generation**: A binding affinity predictor is integrated into the SE(3)-EGNN by aggregating final hidden embeddings to predict a normalized binding affinity $\hat{y}$. During sampling, the coordinate vector field is guided via gradients: $\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + (v_\theta^x + \gamma \frac{\bar{\alpha}'_{1-t}}{2\bar{\alpha}_{1-t}} \nabla \log p_\theta(y=1|\mathbf{m}_t))\Delta t$, where $\gamma$ controls guidance strength. Although discrete atom types cannot be guided directly, the optimized coordinates indirectly influence atom type predictions through the GNN.

3. **Learnable Atom Number Predictor**: The predictor estimates the number of ligand atoms solely from protein pocket descriptors—atom count $N_P$, volume $V$, surface area $A$, and spatial extent $S$—without relying on any reference ligand. It is trained with normalized labels, and Gaussian noise $\tau \sim \mathcal{N}(0, \delta^2)$ is added to predictions at inference time as regularization to increase diversity and prevent overfitting.

### Loss & Training

The total loss comprises three components: (1) coordinate CFM loss $\mathcal{L}_{CFM}^x$: regression of the target coordinate vector field; (2) atom type CFM loss $\mathcal{L}_{CFM}^a$: regression of the target type vector field; (3) interaction prediction loss $\mathcal{L}_{inter}$: MSE loss for binding affinity prediction. The atom number predictor is trained independently using normalized labels with denormalized outputs. Sampling employs an Euler ODE solver with a default of $T=100$ steps and also supports a fast mode with $T=20$ steps. Coordinates are initialized from a standard Gaussian distribution within the pocket, and atom types are initialized from a uniform distribution.

## Key Experimental Results

### Main Results

| Method | Avg. Vina Score↓ | Avg. Vina Dock↓ | High Affinity↑ | QED↑ | SA↑ |
|--------|-------------------|------------------|----------------|------|-----|
| PAFlow | **-8.31** | **-9.46** | **80.8%** | 0.49 | 0.57 |
| ALiDiff | -7.07 | -8.90 | 73.4% | 0.50 | 0.57 |
| TAGMol | -7.02 | -8.59 | 69.8% | 0.55 | 0.56 |
| MolCRAFT (BFN) | -6.59 | -7.92 | 59.1% | 0.50 | 0.69 |
| FlowSBDD (FM) | -3.62 | -8.50 | 63.4% | 0.47 | 0.51 |

### Ablation Study

| Configuration | Avg. Vina Score | Note |
|---------------|-----------------|------|
| Full PAFlow | -8.31 | Complete model |
| w/o interaction guidance (w/o P) | -5.18 | Guidance contributes 60.4% of gain |
| FM vs. diffusion (w/o PA vs. TargetDiff) | -5.13 vs. -5.47 | FM sampling strategy is inherently superior |
| Atom number predictor vs. predefined distribution | MAE 3.35 vs. ~5+ | Predictor better matches pocket geometry |

### Key Findings

- PAFlow achieves the highest binding affinity on 77% of test targets.
- Sampling is 5.5× faster than TargetDiff (717 s vs. 3968 s); with $T=20$ steps, it is faster than MolCRAFT.
- Interaction guidance is the dominant contributor: Avg. Vina Score improves from −5.18 to −8.31.
- FlowSBDD with a linear path performs poorly, validating the necessity of the VP path for the complex SBDD task.
- The atom number predictor achieves an MAE of 3.35, significantly outperforming predefined distribution sampling.
- Median Vina Score is also superior (−8.92 vs. ALiDiff −7.95), indicating more stable generation quality.
- Even in fast mode ($T=20$ steps), PAFlow still surpasses MolCRAFT in binding affinity.

## Highlights & Insights

- This work is the first to derive a conditional vector field for discrete atom types within the FM framework, enabling coordinates and types to be generated under a unified formulation.
- The interaction guidance strategy yields a remarkably large effect: more than 60% of the affinity improvement is attributable to guidance alone.
- The design philosophy of the atom number predictor—using only pocket information rather than reference ligands—is better aligned with real-world drug discovery scenarios.
- The mathematical justification for noise injection constitutes an interesting technical contribution.

## Limitations & Future Work

- Molecular properties such as QED and SA are not explicitly optimized during generation, leaving room for further improvement.
- The guidance strength $\gamma$ of the interaction predictor requires manual tuning.
- Evaluation is conducted solely on CrossDocked2020; generalizability to other datasets remains to be verified.
- The predictor could be extended to molecular properties, enabling multi-objective guided generation.

## Related Work & Insights

- TargetDiff employs the same probability path but uses diffusion-based denoising sampling; PAFlow's ODE-based sampling is more stable.
- FlowSBDD applies FM with a linear path, which proves insufficient for the complexity of SBDD.
- The guidance strategy is inspired by TAGMol and ALiDiff, but PAFlow derives a new guidance formula within the FM framework.
- The atom number predictor concept is generalizable to other tasks that require determining the size of the generated object.
- Autoregressive approaches such as AR and Pocket2Mol are flexible but suffer from severe error accumulation.
- DecompDiff's decomposition strategy and IPDiff's interaction-aware diffusion are effective but remain constrained by the stochasticity of the diffusion framework.
- MolCRAFT's BFN framework generates molecules in continuous parameter space but achieves lower binding affinity than PAFlow.
- The choice of SE(3)-equivariant GNN ensures physical symmetry, which is a standard design principle in molecular generation.
- Structural analysis shows that molecules generated by PAFlow conform more closely to protein pocket geometry.

## Rating

- Novelty: ⭐⭐⭐⭐ — The discrete-type CFM derivation and its combination with interaction-guided FM are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive baseline comparisons, ablation studies, sampling efficiency analysis, and visualization.
- Writing Quality: ⭐⭐⭐⭐ — Method derivations are clear and experimental presentation is intuitive.
- Value: ⭐⭐⭐⭐⭐ — A significant advance in SBDD; the Vina Score of −8.31 substantially pushes the state of the art.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)

</div>

<!-- RELATED:END -->
