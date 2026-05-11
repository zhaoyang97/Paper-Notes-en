---
title: >-
  [Paper Note] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][multimodal fusion] This paper proposes MuMo, a framework that fuses 2D topological and 3D geometric information into stable structural priors via a Structured Fusion Pipeline (SFP)…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "multimodal fusion"
  - "molecular representation learning"
  - "state space models"
  - "drug discovery"
  - "conformational robustness"
date: 2026-05-08
content_hash: 45396f4af2a4bea3
---

# Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.23640](https://arxiv.org/abs/2510.23640)
**Code**: [Available](https://github.com/selmiss/MuMo)
**Area**: Multimodal Learning / Molecular Representation
**Keywords**: multimodal fusion, molecular representation learning, state space models, drug discovery, conformational robustness

## TL;DR

This paper proposes MuMo, a framework that fuses 2D topological and 3D geometric information into stable structural priors via a Structured Fusion Pipeline (SFP), and asymmetrically integrates these priors into the sequence stream through a Progressive Injection (PI) mechanism, achieving an average improvement of 2.7% over competitive baselines across 29 molecular property prediction benchmarks and ranking first on 22 of them.

## Background & Motivation

Molecular property prediction is a fundamental task in drug discovery and materials science. Molecules can be represented through multiple modalities: 1D SMILES sequences, 2D molecular graphs (atom–bond topology), and 3D conformations (atomic coordinates). Multimodal fusion theoretically yields richer molecular representations, yet existing methods face two principal challenges:

**Unreliability of 3D Conformations**: 3D structures are typically generated from 2D structures using tools such as RDKit, introducing noise and uncertainty. Different conformation generation methods may produce substantially different 3D structures, leading to unstable downstream performance.

**Modality Collapse**: Naive fusion strategies (e.g., simple concatenation or addition) tend to cause the model to over-rely on one modality while ignoring others, effectively degrading multimodal fusion to single-modality learning.

## Method

### Overall Architecture

MuMo consists of three main modules:

1. **Sequence Encoder**: A state space model (SSM/Mamba)-based encoder that processes SMILES sequences.
2. **Structured Fusion Pipeline (SFP)**: Fuses 2D graph and 3D geometric information into a unified structural prior.
3. **Progressive Injection (PI) Mechanism**: Asymmetrically injects the structural prior into the sequence encoder layer by layer.

### Key Designs

**Structured Fusion Pipeline (SFP)**:

SFP aims to generate stable structural priors and reduce sensitivity to 3D conformation quality:

- **2D Graph Encoding**: A GNN (e.g., GIN or GAT) encodes the 2D topological information of the molecule, producing node features $H_{2D}$.
- **3D Geometric Encoding**: An equivariant GNN (e.g., SchNet or DimeNet) encodes 3D spatial information, producing features $H_{3D}$.
- **Structural Fusion**: An attention mechanism fuses $H_{2D}$ and $H_{3D}$ into a unified structural prior $H_{struct}$.

The key insight is that 2D topology serves as an "anchor"—even when 3D conformations are inaccurate, the 2D topology remains deterministic, and 2D-guided fusion enhances overall robustness.

**Progressive Injection (PI)**:

PI is the core mechanism for preventing modality collapse. Rather than fusing all modalities at the final layer, PI injects the structural prior into every layer of the sequence encoder:

$$H_{seq}^{(l+1)} = \text{SSM}(H_{seq}^{(l)}) + \lambda_l \cdot \text{CrossAttn}(H_{seq}^{(l)}, H_{struct})$$

- **Asymmetry**: The structural prior serves only as auxiliary information injected into the sequence stream; no information flows in the reverse direction, preserving the independent modeling capacity of the sequence encoder.
- **Progressiveness**: The injection strength $\lambda_l$ varies by layer—weaker in shallow layers (preserving modality-specific features) and stronger in deeper layers (promoting cross-modal fusion).
- **Preservation of Modality Specificity**: The sequence encoder retains an independent pathway for processing sequence information, preventing it from being overwhelmed by structural information.

**State Space Model (SSM) Backbone**:

Mamba is adopted as the sequence encoder for its:
- Efficient handling of long sequences (SMILES strings can be lengthy).
- Linear complexity, superior to the quadratic complexity of Transformers.
- Robust information propagation mechanism.

### Loss & Training

Loss functions are selected according to task type:
- Regression tasks: MSE loss.
- Classification tasks: Cross-entropy loss.
- Multi-task settings: Weighted multi-task loss.

All modules (sequence encoder, SFP, PI) are trained end-to-end without pre-training.

## Key Experimental Results

### Main Results

Evaluation is conducted on 29 benchmark tasks from the Therapeutics Data Commons (TDC) and MoleculeNet.

**TDC Drug Property Prediction (representative tasks)**:

| Task | Metric | GIN | SchNet | Transformer | MolBERT | **MuMo** |
|------|--------|-----|--------|-------------|---------|----------|
| Caco2 | MAE ↓ | 0.432 | 0.418 | 0.405 | 0.395 | **0.372** |
| HIA | AUROC ↑ | 0.887 | 0.875 | 0.892 | 0.901 | **0.923** |
| BBB | AUROC ↑ | 0.905 | 0.891 | 0.912 | 0.918 | **0.937** |
| LD50 | MAE ↓ | 0.685 | 0.721 | 0.652 | 0.638 | **0.465** |
| CYP2D6 | AUROC ↑ | 0.711 | 0.695 | 0.728 | 0.735 | **0.762** |
| hERG | AUROC ↑ | 0.842 | 0.831 | 0.856 | 0.863 | **0.885** |

MuMo achieves the most notable improvement on the LD50 task, with a 27% gain (0.638 → 0.465).

**MoleculeNet Benchmarks (representative tasks)**:

| Task | Metric | AttentiveFP | D-MPNN | Uni-Mol | **MuMo** |
|------|--------|-------------|--------|---------|----------|
| BBBP | AUROC ↑ | 0.852 | 0.871 | 0.892 | **0.915** |
| BACE | AUROC ↑ | 0.818 | 0.835 | 0.858 | **0.878** |
| Tox21 | AUROC ↑ | 0.785 | 0.802 | 0.821 | **0.842** |
| HIV | AUROC ↑ | 0.762 | 0.778 | 0.798 | **0.815** |
| ESOL | RMSE ↓ | 0.845 | 0.795 | 0.728 | **0.695** |
| FreeSolv | RMSE ↓ | 1.623 | 1.485 | 1.312 | **1.215** |

### Ablation Study

**Contribution of each component (average performance on TDC)**:

| Configuration | Avg. Rank ↓ | # First-Place (/29) |
|---------------|------------|---------------------|
| MuMo (Full) | **1.8** | **22** |
| w/o PI (direct concat) | 3.2 | 12 |
| w/o SFP (3D only) | 4.1 | 8 |
| w/o 3D (2D + sequence only) | 3.6 | 10 |
| w/o 2D (3D + sequence only) | 5.2 | 5 |
| Sequence only (Mamba) | 4.8 | 6 |

- PI contributes the most: removing it reduces first-place counts from 22 to 12.
- Within SFP, 2D topology is more important than 3D geometry: removing 2D causes a larger performance drop.
- The three-modality combination outperforms any two-modality combination.

**Robustness to Conformational Noise**:

| Noise Level | Uni-Mol | 3D-InfoMax | **MuMo** |
|-------------|---------|------------|----------|
| No noise | 0.892 | 0.875 | **0.915** |
| σ=0.1Å | 0.865 | 0.848 | **0.908** |
| σ=0.3Å | 0.821 | 0.795 | **0.895** |
| σ=0.5Å | 0.762 | 0.728 | **0.878** |

MuMo demonstrates substantially stronger robustness to conformational noise (only a 4% drop at σ=0.5Å, compared to 15% for Uni-Mol).

### Key Findings

1. **Multimodal fusion outperforms unimodal approaches**: The three-modality combination ranks first on 22 of 29 tasks.
2. **PI is critical for preventing modality collapse**: Progressive injection significantly outperforms naive concatenation.
3. **2D topology provides stability**: Serving as an anchor in SFP, 2D information enhances robustness to 3D noise.
4. **Large improvement on LD50**: The 27% gain indicates that MuMo is particularly well-suited for tasks requiring comprehensive integration of multimodal information.
5. **Advantages of the SSM backbone**: The Mamba architecture achieves both efficiency and effectiveness in processing long SMILES sequences.

## Highlights & Insights

- **Problem-oriented design**: SFP and PI directly address the two core challenges—3D unreliability and modality collapse.
- **Strong empirical performance**: State-of-the-art on 22 of 29 tasks, with an average improvement of 2.7%.
- **Conformational robustness**: A highly desirable property in practical drug discovery pipelines.
- **SSM for molecular representation**: Demonstrates the potential of the Mamba architecture in molecular representation learning.

## Limitations & Future Work

1. **Conformation generation methods**: 3D structures are currently generated via RDKit; more advanced conformation generation approaches remain to be explored.
2. **Pre-training**: The absence of large-scale pre-training limits potential generalization; integration with self-supervised pre-training may yield further gains.
3. **Protein–molecule interactions**: Binding information from target proteins is not incorporated.
4. **3D equivariance**: The current SFP design may not fully preserve 3D equivariance.
5. **Scalability**: Efficiency on ultra-large molecular libraries requires further validation.

## Related Work & Insights

- **Molecular representation learning**: MolBERT, Uni-Mol, 3D-InfoMax, etc.
- **Multimodal fusion**: Fusion strategies from vision–language models.
- **State space models**: Mamba (Gu & Dao, 2023) for sequence modeling.
- **Drug discovery**: TDC benchmark platform (Huang et al., 2022).

## Rating

- **Novelty**: 4/5 — The combined SFP + PI design is novel and effective.
- **Experimental Thoroughness**: 4/5 — Comprehensive evaluation across 29 tasks.
- **Writing Quality**: 4/5 — Clear method description and systematic experimental design.
- **Value**: 4/5 — Directly addresses practical needs in drug discovery.
- **Overall**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You](with_limited_data_for_multimodal_alignment_let_the_structure_guide_you.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
