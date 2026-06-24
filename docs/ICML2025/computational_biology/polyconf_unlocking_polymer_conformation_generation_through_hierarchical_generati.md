---
title: >-
  [Paper Note] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models
description: >-
  [ICML 2025][Computational Biology][Polymer Conformation Generation] Introduces PolyConf, the first hierarchical generative framework specifically designed for polymer conformation generation. Phase 1 utilizes a masked autoregressive (MAR) model and a diffusion process to generate the local conformations of repeating units in a random order, while Phase 2 utilizes an SO(3) diffusion model to generate orientation transformations to assemble the local conformations into a comple…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Polymer Conformation Generation"
  - "Hierarchical Generative Models"
  - "Masked Autoregressive Models"
  - "SO(3) Diffusion Models"
  - "Molecular Dynamics"
date: 2026-05-08
content_hash: 08618d37b857bcd7
---

# PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models

**Conference**: ICML 2025  
**arXiv**: [2504.08859](https://arxiv.org/abs/2504.08859)  
**Code**: [https://polyconf-icml25.github.io](https://polyconf-icml25.github.io) (Code, models, and data are publicly available)  
**Area**: Molecular Modeling / AI4Science  
**Keywords**: Polymer Conformation Generation, Hierarchical Generative Models, Masked Autoregressive Models, SO(3) Diffusion Models, Molecular Dynamics

## TL;DR

Introduces PolyConf, the first hierarchical generative framework specifically designed for polymer conformation generation. Phase 1 utilizes a masked autoregressive (MAR) model and a diffusion process to generate the local conformations of repeating units in a random order, while Phase 2 utilizes an SO(3) diffusion model to generate orientation transformations to assemble the local conformations into a complete polymer molecular structure. Additionally, the work constructs the first polymer conformation benchmark, PolyBench (containing 50k+ polymers and approximately 2,000 atoms per conformation), consistently outperforming prior methods by over 25% across all structural and energy metrics.

## Background & Motivation

Polymers are macromolecules composed of a large number of repeating, identical or similar monomer units linked by covalent bonds, which find wide applications in packaging materials and electronic devices. Polymer conformation generation—the prediction of stable 3D polymer structures given their 2D molecular graphs—is a fundamental starting point for studying polymer properties.

Key challenges faced by existing conformation generation methods:

**Scalability limitations of small-molecule methods**: Approaches such as GeoDiff and TorsionalDiff are tailored for small molecules with fewer atoms. Their performance degrades significantly when directly applied to polymer chains containing thousands of atoms.

**Inapplicability of protein-focused models**: Proteins possess a unified backbone framework (the N-Cα-C-O structure) and strong directional intramolecular interactions. Polymers lack these structural constraints, exhibiting higher flexibility and less spatial order.

**Data scarcity**: Conventional molecular dynamics (MD) simulations are computationally extremely demanding, leading to a critical shortage of polymer conformation datasets.

**Diverse conformations of repeating units**: Although the repeating units of a given polymer share identical SMILES strings, their spatial configurations in 3D space can vary dramatically. Consequently, a polymer cannot be realistically modeled as a rigid assembly of a single, predefined repeating unit conformation.

## Method

### Overall Architecture

PolyConf adopts a **hierarchical generative framework**, where the core idea is to decompose the polymer conformation into two layers:

$$p(\mathcal{C}|\mathcal{G}) = p(\mathcal{C}^u|\mathcal{G}) \cdot p(\mathcal{O}|\mathcal{G}, \mathcal{C}^u)$$

where $\mathcal{C}^u = \{C_i^u\}_{i=1}^{N_u}$ represents the set of local conformations for the repeating units, and $\mathcal{O} = \{\mathcal{O}_i\}_{i=1}^{N_u}$ represents the corresponding set of orientation transformations. The entire generation process is executed in two phases:

* **Phase 1**: Generates local conformations $\mathcal{C}^u$ of repeating units based on the 2D molecular graph $\mathcal{G}$.
* **Phase 2**: Generates orientation transformations $\mathcal{O}$ conditioned on $\mathcal{G}$ and $\mathcal{C}^u$ to assemble the complete polymer conformation.

### Frame-based Representation

Each polymer with $N$ atoms is represented as a 2D graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ with conformation $C \in \mathbb{R}^{N \times 3}$. Key design choices include:

* Extending the standard repeating unit definition: Two key atoms (atom-1 and atom-4) from adjacent repeating units are incorporated into the conformation of the current unit, resulting in $\frac{N}{N_u}+2$ atoms per unit.
* Inspired by protein residue modeling, **frames** are extracted from the repeating unit conformations.
* Orientation transformation $\mathcal{O}_i = (R_i, t_i)$: The rotation $R_i \in \mathbb{R}^{3\times3}$ is computed from key atom vectors utilizing Gram-Schmidt orthonormalization, while the translation $t_i \in \mathbb{R}^3$ corresponds to the 3D coordinates of atom-3.
* Because adjacent repeating units naturally overlap at the bonding atoms (atom-1 of one unit aligns with atom-3 of the preceding unit), Phase 2 only requires generating the rotations; translations can be directly derived from the overlapping atom coordinates.

### Phase 1: Repeating Unit Conformation Generation

This phase integrates three core modules:

#### (1) Multimodal Repeating Unit Encoder

$$X^u = \mathcal{M}(\mathcal{G}, \{C_i^u\}) = \text{Concat}_1(\mathcal{M}_{2d}(\mathcal{G}),\ \text{Concat}_0(\{\mathcal{M}_{3d}(C_i^u)\}))$$

* **2D Encoder** $\mathcal{M}_{2d}$: Based on the MolCLR architecture, it encodes the overall polymer graph $\mathcal{G}$ to obtain $X^{2d} \in \mathbb{R}^{N_u \times D_{2d}}$.
* **3D Encoder** $\mathcal{M}_{3d}$: Based on Uni-Mol (SE(3)-invariant), it encodes each repeating unit conformation $C_i^u$ to obtain $X_i^{3d} \in \mathbb{R}^{1 \times D_{3d}}$.
* Concatenates the 2D and 3D embeddings to yield the multimodal embedding $X^u \in \mathbb{R}^{N_u \times D_u}$.
* This encoder is trainable in Phase 1 and frozen in Phase 2.

#### (2) Masked Autoregressive Modeling (MAR)

Generates repeating unit conformations in a **random order** to capture complex interactions rather than simple sequential dependencies:

$$p(\mathcal{C}^u|\mathcal{G}) = \prod_{k=1}^{K} p(\mathcal{C}_k^u|\mathcal{G}, \{\mathcal{C}_i^u\}_{i=1}^{k-1})$$

During training:

* A random permutation $\pi$ is defined and a masking ratio $\tau \in [0,1]$ is randomly sampled.
* The masked, known embeddings $X_{\text{known}}^u$ are fed into the MAR encoder $\Phi$ (a standard Transformer with bidirectional attention).
* The MAR decoder $\Psi$ outputs the representations for the masked repeating units: $Z_{\text{mask}}^u \in \mathbb{R}^{\tau N_u \times D_m}$

$$Z_{\text{mask}}^u = \Psi(\Phi(X_{\text{known}}^u))$$

#### (3) Diffusion Loss

A diffusion model is used to represent the conditional probability distribution of the masked repeating unit conformations. The diffusion process is defined over the **torsion angle space**:

$$\mathcal{L}_{\text{phase-1}} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(C_t^u | t, z^u)\|^2\right]$$

$$C_t^u = \sqrt{\bar{\alpha}_t} C^u + \sqrt{1-\bar{\alpha}_t}\epsilon$$

* $z^u$ is the decoded representation from the corresponding row of $Z_{\text{mask}}^u$, serving as the condition for the diffusion process.
* The denoising network $\epsilon_\theta$ adopts the architecture of TorsionalDiff.
* This effectively combines the strengths of autoregressive modeling with the capabilities of diffusion processes.

### Phase 2: Orientation Transformation Generation

#### SO(3) Diffusion for Rotation Generation

Due to the presence of overlapping atoms, only the rotation matrices $R$ need to be generated; translations can be directly derived from the overlapping atom coordinates:

$$\hat{R}^{(0)} = \varphi(\mathcal{O}^{(t)}, t, E^u), \quad \mathcal{O}^{(t)} = (R^{(t)}, T^{(t)})$$

* The denoising network $\varphi$ employs the same architecture as FrameDiff (Yim et al., 2023).
* The conditional information $E^u \in \mathbb{R}^{N_u \times D_e}$ is obtained from the output of the MAR encoder (representing the unified information of $\mathcal{G}$ and $\mathcal{C}^u$).
* $R^{(t)}$ is acquired via forward diffusion on SO(3), and $T^{(t)}$ is computed by aligning the overlapping atoms after applying $R^{(t)}$.

### Loss & Training

**Phase 1 Loss** (Torsion space diffusion):

$$\mathcal{L}_{\text{phase-1}} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(C_t^u | t, z^u)\|^2\right]$$

**Phase 2 Loss** (SO(3) rotation prediction):

$$\mathcal{L}_{\text{phase-2}} = \frac{1}{N_u}\sum_{i=1}^{N_u}\|\hat{R}_i^{(0)} - R_i\|^2$$

**Training Strategy**: Phase 1 trains the encoders, the MAR module, and the torsion diffusion model. Phase 2 freezes the encoders and only trains the SO(3) diffusion model.

### Assembly Process

1. Transform the generated repeating unit conformations $\hat{C}_i^u$ back to the canonical frame.
2. Apply the generated rotations $\hat{R}_i$ to obtain the rotated conformations $\hat{C}_i^{u,\text{rot}}$.
3. Compute the translations $\hat{t}_i$ by aligning overlapping atoms.
4. Concatenate and remove overlapping atoms to produce the complete polymer conformation $\hat{C} \in \mathbb{R}^{N \times 3}$.

## Key Experimental Results

### Dataset: PolyBench

The first polymer conformation benchmark, constructed using molecular dynamics simulations:
* 50,000+ polymers, with approximately 2,000 atoms per conformation.
* NVT ensemble, 298K/1atm, 5ns (5M steps) MD simulations.
* Force field: General AMBER Force Field (GAFF), GROMACS engine.
* Splitting: Training ~46k, validation ~5k, testing ~2k.
* Repeating units: Mostly 20-100, with a minority exceeding 100.

### Main Results

| Method | S-MAT-R (Mean/Med) | S-MAT-P (Mean/Med) | E-MAT-R (Mean/Med) | E-MAT-P (Mean/Med) |
|------|---------------------|---------------------|---------------------|---------------------|
| GeoDiff | 93.119 / 89.767 | 95.259 / 91.869 | 21.249 / 18.106 | 64.871 / 58.711 |
| TorsionalDiff | 53.210 / 38.710 | 70.679 / 60.744 | 2.605 / 1.034 | 8.402 / 6.851 |
| MCF | 248.432 / 242.866 | 258.891 / 253.239 | — | — |
| ET-Flow | 94.057 / 90.475 | 96.896 / 92.877 | 6.733 / 5.186 | 53.528 / 30.125 |
| **PolyConf** | **35.021 / 24.279** | **46.861 / 37.996** | **0.933 / 0.359** | **6.191 / 4.122** |

* Outperforms the best baseline TorsionalDiff by **over 25% across all metrics**.
* Note that TorsionalDiff additionally leverages the initial polymer structure as input (a biased advantage).

### Scalability Experiment (Doubled repeating units, ~4000 atoms)

| Method | S-MAT-R (Mean/Med) | S-MAT-P (Mean/Med) | E-MAT-R (Mean/Med) | E-MAT-P (Mean/Med) |
|------|---------------------|---------------------|---------------------|---------------------|
| GeoDiff | 184.668 / 175.607 | 186.861 / 177.645 | 52.614 / 47.872 | 112.883 / 105.197 |
| TorsionalDiff | 119.289 / 94.075 | 146.816 / 126.932 | 5.219 / 2.216 | 11.692 / 9.227 |
| ET-Flow | 186.132 / 176.370 | 188.725 / 178.977 | 15.331 / 12.465 | 65.116 / 41.642 |
| **PolyConf** | **65.040 / 41.992** | **84.626 / 64.445** | **1.259 / 0.609** | **5.785 / 4.434** |

* Improves the energy metrics by over **50%** compared to TorsionalDiff.
* Thanks to MAR modeling, PolyConf demonstrates excellent scalability.

### Efficiency Comparison

| Method | Average Generation Time |
|------|-------------|
| GeoDiff | 3.54 min |
| MCF | 1.12 min |
| TorsionalDiff | 0.45 min |
| **PolyConf** | **0.40 min** |

### Key Findings

1. **Effectiveness of Hierarchical Decomposition**: Decomposing polymer conformation generation into local conformations and orientation transformations in a two-stage manner yields significantly better results than end-to-end framework-free alternatives.
2. **Random-Order Generation Outperforms Fixed-Order**: The random permutation mechanism of MAR successfully captures complex interactions among repeating units.
3. **Failure of Baseline Methods**: Small-molecule baselines (GeoDiff, ET-Flow) exhibit severe performance drops on polymers, with MCF even failing to produce valid structures for energy evaluations.
4. **Visual Evidence of Quality**: Qualitative visualization indicates that conformations generated by PolyConf align closer to the extended, relaxed reference conformations, whereas TorsionalDiff fails to capture relaxed states despite utilizing the initial structure prior.

## Highlights & Insights

1. **Pioneering Problem Formulation**: Formulates polymer conformation generation as an independent task for the first time, bridging the critical gap between small molecules and proteins.
2. **Elegant Hierarchical Decomposition**: Naturally decomposes complex macromolecular structures into manageable sub-problems by exploiting the repetition characteristics of polymer units.
3. **Effective Combination of MAR & Diffusion**: Integrates the state-of-the-art masked autoregressive paradigm with diffusion processes, representing its first application in structural conformation modeling.
4. **Clever Application of SO(3) Diffusion**: Simplifies SE(3) transformation modeling to pure rotation generation via overlapping atom constraints, significantly reducing computational complexity.
5. **Comprehensive Benchmark Contribution**: Contributes a massive dataset and standardized evaluation protocols alongside the algorithmic framework.

## Limitations & Future Work

1. **Limited to Linear Homopolymers**: The current scope is restricted to linear polymers composed of identical monomers and does not address copolymers or complex polymer blends.
2. **Neglect of 2D Topological Variations**: Architectural variations such as cross-linking and branching are not modeled.
3. **Limitations of Force Field Levels**: Relying on GAFF instead of DFT calculation (due to computational cost) limits the overall validity of the absolute energy evaluations.
4. **Potential of Flow Matching**: The authors suggest incorporating flow-based generative frameworks as a potential future direction to further boost performance.
5. **Lack of Downstream Verification**: The effectiveness of the generated structures in predicting macromolecular properties or materials dynamics is yet to be fully validated.

## Related Work & Insights

* **Small-Molecule Conformation Generation**: GeoDiff (Euclidean diffusion), TorsionalDiff (torsional diffusion), MCF, ET-Flow → The massive scale and flexibility of polymers hinder their performance.
* **Protein Conformation Generation**: AlphaFold series, FrameDiff → Highly dependent on evolutionary information and protein-specific backbone constraints, extending poorly to polymers.
* **Key Insights**: The combined utilization of the MAR paradigm (Li et al., 2024) and SE(3) diffusion (Yim et al., 2023) is highly effective; the repetitive nature of polymers naturally lends itself to hierarchical generative modeling.

## Rating

* **Novelty**: ⭐⭐⭐⭐⭐ — The first dedicated polymer conformation generation framework; both the problem formulation and hierarchical decomposition display exceptional originality.
* **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple dimensions (structure, energy, efficiency, and scalability), though more ablation details could be presented.
* **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-structured, clear mathematical derivation, and highly intuitive schematics.
* **Value**: ⭐⭐⭐⭐⭐ — Establishes a new avenue of research; fully open-sourced models, code, and datasets will significantly advance polymer research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Aligning Protein Conformation Ensemble Generation with Physical Feedback](aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](../../NeurIPS2025/computational_biology/confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[ICML 2025\] ComRecGC: Global Graph Counterfactual Explainer through Common Recourse](comrecgc_global_graph_counterfactual_explainer_through_common_recourse.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](../../NeurIPS2025/computational_biology/learning_repetition-invariant_representations_for_polymer_informatics.md)

</div>

<!-- RELATED:END -->
