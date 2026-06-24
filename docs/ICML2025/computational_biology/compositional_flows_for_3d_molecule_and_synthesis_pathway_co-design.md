---
title: >-
  [Paper Note] Compositional Flows for 3D Molecule and Synthesis Pathway Co-design
description: >-
  [ICML 2025][Computational Biology][Compositional Generative Flows] Proposes CGFlow (Compositional Generative Flows)—extending flow matching to the step-by-step generation of compositional objects. It interleaves discrete compositional structure sampling (synthesis pathways) and continuous state transport (3D conformations). Applied as 3DSynthFlow to synthesizable drug design, it achieves SOTA results in both binding affinity and synthesizability across 15 targets of LIT-PCBA…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Compositional Generative Flows"
  - "3D Molecular Design"
  - "Synthesis Pathways"
  - "GFlowNet"
  - "Drug Design"
date: 2026-05-08
content_hash: f517e4991e350970
---

# Compositional Flows for 3D Molecule and Synthesis Pathway Co-design

**Conference**: ICML 2025  
**arXiv**: [2504.08051](https://arxiv.org/abs/2504.08051)  
**Code**: None  
**Area**: Image Generation/Molecular Design  
**Keywords**: Compositional Generative Flows, 3D Molecular Design, Synthesis Pathways, GFlowNet, Drug Design

## TL;DR
Proposes CGFlow (Compositional Generative Flows)—extending flow matching to the step-by-step generation of compositional objects. It interleaves discrete compositional structure sampling (synthesis pathways) and continuous state transport (3D conformations). Applied as 3DSynthFlow to synthesizable drug design, it achieves SOTA results in both binding affinity and synthesizability across 15 targets of LIT-PCBA for the first time.

## Background & Motivation

**Background**: 3D molecular generative models (diffusion/flow matching) perform exceptionally well in drug design but generate the entire molecule in one go, failing to guarantee synthesizability. GFlowNet can build molecules step-by-step based on reaction steps but is limited to 2D.

**Limitations of Prior Work**:
   - Diffusion/flow models generate all atoms simultaneously—incapable of masking invalid generative actions and unable to ensure synthetic feasibility.
   - GFlowNet builds step-by-step but only handles discrete 2D graphs—failing to model 3D conformations (protein-ligand interactions heavily rely on 3D).
   - Autoregressive models generate 3D structures step-by-step, but minor errors in early steps cascade and accumulate.

**Key Challenge**: The need to simultaneously model compositional structures (discrete sequences of synthesis pathways) and continuous states (3D atomic coordinates), whereas existing methods can only handle one of them.

**Goal**: Jointly generate synthesis pathways and 3D conformations within a unified framework.

**Key Insight**: Extends the interpolation process of flow matching to compositional state transitions—gradually constructing structures in compositional steps while transporting corresponding continuous states using flow matching.

**Core Idea**: Two interleaved flow processes—(1) Compositional Flow to progressively disassemble/construct compositional structures; (2) State Flow to transport the continuous states associated with each component. The two are mutually dependent through shared inputs.

## Method

### Overall Architecture
CGFlow consists of two interleaved processes:
1. **Compositional Flow**: Progressively disassembles from the complete molecule to an empty state (forward direction during training) and builds up step-by-step from the empty state (reverse direction during inference). Each step corresponds to a synthetic reaction step.
2. **State Flow**: Standard optimal transport flow matching, but applying different noise levels to different components—components disassembled earlier receive higher noise.

During inference: GFlowNet policy samples compositional steps (selecting the next synthetic reaction) $\to$ conditional flow matching generates the corresponding 3D coordinates.

### Key Designs

1. **Extension of Compositional Flow Matching Interpolation**:

    - Function: Extends the linear interpolation of standard flow matching from "all noise $\to$ all data" to "empty structure $\to$ progressively building full structure".
    - Mechanism: Performs the $k$-th synthesis reaction step at time $t_k$ to add new atoms/fragments; the State Flow transports newly added fragments starting from noise while continuing to refine existing fragments.
    - Key Formula: $x_t = \alpha_t(c) \cdot x_1 + \sigma_t(c) \cdot \epsilon$, where the noise level $\sigma_t(c)$ depends on the time component $c$ is added.
    - Design Motivation: Components added earlier receive more denoising time $\to$ more precise positions; components added later depend on the positions of previous components $\to$ natural causal dependency.

2. **GFlowNet Reward-Guided Sampling**:

    - Function: Uses GFlowNet to sample synthesis pathways proportional to rewards (preferring pathways with high binding affinity + high synthesizability).
    - Mechanism: $p(\text{pathway}) \propto R(\text{molecule})$, where $R$ can be docking scores, synthesizability scores, etc.
    - Design Motivation: Standard flow matching can only sample from the training distribution; GFlowNet enables reward-guided exploration, biasing generation toward high-value molecular regions.

3. **3DSynthFlow Instantiation**:

    - Function: Applies CGFlow to target-specific, synthesizable drug design.
    - Mechanism: The synthesis pathway is defined by a sequence of reaction steps (using Reaxys reaction templates), and 3D conformations are generated inside the protein pocket.
    - Training Data: CrossDocked2020 + ZINC synthesis pathways.
    - Novelty: The first 3D molecular generative model to simultaneously optimize binding affinity and synthesizability.

### Loss & Training
- State Flow: Conditional Flow Matching loss (CFM objective)
- Compositional Flow: Trajectory Balance (TB) loss of GFlowNet
- Joint training of both losses
- Inference: Alternating execution of synthesis step sampling (GFlowNet) and 3D coordinate generation (flow matching ODE)

## Key Experimental Results

### Main Results
LIT-PCBA benchmark (15 drug targets):

| Method | Avg. Vina Dock ↓ | Hit Rate ↑ | AiZynth Synthesizability ↑ |
|------|----------------|---------|-----------------|
| TargetDiff | -7.84 | 12.3% | 18.5% |
| DiffSBDD | -8.21 | 15.7% | 22.3% |
| FlowSBDD | -8.56 | 18.2% | 28.1% |
| SynFlowNet (2D) | -7.12 | 8.5% | 42.3% |
| **3DSynthFlow** | **-9.42** | **24.5%** | **36.1%** |

### Sampling Efficiency

| Method | Samples needed to find high-affinity molecules ↓ |
|------|----------------------|
| SynFlowNet (2D) | ~5000 |
| TargetDiff | ~3000 |
| **3DSynthFlow** | **~1200** (4.2× speedup) |

### Ablation Study

| Configuration | Vina Dock | Synthesizability | Description |
|------|----------|---------|------|
| State Flow Only (no synthetic constraints) | -8.56 | 22.3% | Degenerates to standard flow matching |
| Compositional Flow Only (2D) | -7.12 | 42.3% | No 3D information |
| **CGFlow (Interleaved)** | **-9.42** | **36.1%** | Optimal balance |
| W/o GFlowNet (uniform path sampling) | -8.15 | 35.8% | Lacks reward guidance |
| **+ GFlowNet Guidance** | **-9.42** | **36.1%** | Prefers high-value molecules |

### Key Findings
- 3DSynthFlow is the first method to simultaneously achieve SOTA performance on both Vina Dock (-9.42) and AiZynth (36.1%).
- Sampling efficiency is improved by 4.2×—the reward guidance of GFlowNet focuses the search space.
- Interleaved modeling of compositional structures and continuous states is significantly superior to independent modeling (-9.42 vs. -8.56/-7.12).
- The synthesizability rate increases from ~22% in pure 3D methods to 36%—confirming the effectiveness of synthesis pathway constraints.
- Consistently outperforms existing methods across all 15 LIT-PCBA targets, demonstrating high generalizability.

## Highlights & Insights
- **Compositional Flow = Perfect integration of Flow Matching × GFlowNet**—the former handles continuous coordinates, while the latter handles discrete synthesis pathways, achieving mutual dependency through interleaving on the timeline.
- The design of assigning different noise levels to different components is highly natural—newly added fragments should be more "uncertain" since they depend on the locations of prior fragments.
- Joint optimization of 3D structures and synthesizability holds direct value for real-world drug discovery—previously, these could only be optimized separately.
- GFlowNet's reward guidance enables the model to not only learn the data distribution but also bias toward high-value regions, which is more suitable for design optimization than pure likelihood learning.
- High framework generality—CGFlow is not limited to molecular design and is applicable to the generation of any compositional objects with continuous features.

## Limitations & Future Work
- Limited reaction template library—synthesis pathways not covered by the templates cannot be generated.
- GFlowNet training complexity scales with the size of the reaction space.
- The accuracy of 3D coordinate prediction is constrained by the denoising quality of flow matching.
- Vina Dock scores are approximations—discrepancies may exist compared to actual binding affinities.
- Lack of wet-lab experimental validation.

## Related Work & Insights
- **vs. TargetDiff/DiffSBDD**: Pure 3D diffusion models that do not guarantee synthesizability.
- **vs. SynFlowNet**: 2D-only synthesis pathways that fail to model protein-ligand 3D interactions.
- **vs. AutoGrow**: Docking-based fragment growing, not end-to-end generation.
- **Insight**: The joint generative paradigm of combining compositional and continuous states can be generalized to other scientific design problems (e.g., novel material design, protein engineering).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The new paradigm of compositional flow matching possesses broad generality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 targets in LIT-PCBA + CrossDocked + efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Intuitive framework diagrams and clear mathematics.
- Value: ⭐⭐⭐⭐⭐ Significantly advances computational drug design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICML 2025\] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [\[ICML 2026\] Demystifying Multimodal Biomolecular Co-design with Intrinsic Geodesic Coupling](../../ICML2026/computational_biology/demystifying_multimodal_biomolecular_co-design_with_intrinsic_geodesic_coupling.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/computational_biology/amortized_sampling_with_transferable_normalizing_flows.md)

</div>

<!-- RELATED:END -->
