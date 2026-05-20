---
title: >-
  [Paper Note] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression
description: >-
  [NeurIPS 2025][Medical Imaging][protein dynamics] ConfRover proposes an autoregressive framework that factorizes protein MD trajectories into frame-wise conditional generation $p(\mathbf{x}^{1:L}) = \prod_l p(\mathbf{x}^…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "protein dynamics"
  - "autoregressive generation"
  - "SE(3) diffusion"
  - "conformational sampling"
  - "molecular dynamics"
date: 2026-05-08
content_hash: 147c776d46df346b
---

# ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression

**Conference**: NeurIPS 2025
**arXiv**: [2505.17478](https://arxiv.org/abs/2505.17478)  
**Project**: https://bytedance-seed.github.io/ConfRover
**Institution**: ByteDance Seed, Tongji University, Tsinghua University
**Area**: Protein Conformation Modeling / Molecular Dynamics
**Keywords**: protein dynamics, autoregressive generation, SE(3) diffusion, conformational sampling, molecular dynamics

## TL;DR

ConfRover proposes an autoregressive framework that factorizes protein MD trajectories into frame-wise conditional generation $p(\mathbf{x}^{1:L}) = \prod_l p(\mathbf{x}^l | \mathbf{x}^{<l})$, and through a modular architecture consisting of an encoder, a causal Transformer, and an SE(3) diffusion decoder, unifies three tasks—trajectory simulation, time-independent conformational sampling, and conformational interpolation—within a single model for the first time, achieving comprehensive improvements over MDGen on the ATLAS benchmark.

## Background & Motivation

**Background**: Protein conformational dynamics are essential for understanding biological function. Molecular dynamics (MD) simulation is the gold standard for studying conformational changes, yet it is computationally prohibitive and prone to becoming trapped in local energy minima. The emergence of large-scale MD datasets such as ATLAS has motivated the development of deep generative model alternatives.

**Three Fragmented Lines of Existing Methods**: (a) Trajectory simulation methods (MDGen, Timewarp, etc.) capture temporal dependencies but cannot directly sample time-independent conformations; (b) conformational distribution learning methods (AlphaFlow, ConfDiff, BioEmu) can sample independent conformations but completely discard temporal information; (c) conformational interpolation methods require training separate models and have only been validated on small proteins.

**Key Challenge**: All three task types fundamentally stem from the same physical principle—sampling of protein conformational space—yet existing methods treat them in isolation. MDGen's non-autoregressive design limits flexibility (fixed length, separate models per task), and while GST adopts autoregression, it performs deterministic prediction and cannot capture trajectory distributions.

**Key Observation**: Applying autoregressive factorization to MD trajectories $\mathbf{x}^{1:L}$, different conditioning settings naturally correspond to different tasks—full conditioning = trajectory simulation, no conditioning ($L=1$) = independent sampling, and conditioning on start and end frames = interpolation. This unified perspective draws inspiration from analogous ideas in sequence models for text infilling.

## Method

### Overall Architecture

**Input**: Protein sequence information $\mathcal{P}$ + MD trajectory frame sequence $\mathbf{x}^{1:L}$.
**Output**: New conformational frames (trajectory continuation / independent samples / interpolated intermediate frames).
**Three-Stage Pipeline**: (1) An encoding layer maps each conformational frame to a latent representation $\mathbf{h}^l = [\mathbf{s}^l, \mathbf{z}^l]$; (2) a trajectory module updates latent representations via a causal Transformer to encode temporal dependencies; (3) an SE(3) diffusion decoder generates 3D conformations from the updated latent representations.

### Key Design 1: Autoregressive Modeling for Multi-Task Unification

The joint distribution over MD trajectories is factorized as:

$$p(\mathbf{x}^{1:L}|\mathcal{P}) = \prod_{l=1}^{L} p(\mathbf{x}^l | \mathbf{x}^{<l}, \mathcal{P})$$

- **Trajectory simulation**: Conditioned on preceding frames, conformations are generated frame by frame, naturally supporting non-Markovian dynamics.
- **Independent sampling**: When $L=1$, the model reduces to $p(\mathbf{x}|\mathcal{P})$ with no preceding frame conditioning.
- **Conformational interpolation**: The terminal frame is prepended to the initial frame to redefine the dependency structure, analogous to sequence reordering in text infilling.
- **Advantage**: Compared to MDGen's fixed-length non-autoregressive design, the autoregressive formulation naturally supports variable-length generation and flexible conditioning.

### Key Design 2: Causal Modeling in Latent Space

An encoder $f_\eta^{\text{enc}}$ encodes each conformational frame into a latent representation, and a causal attention module $f_\xi^{\text{temp}}$ restricts frame $l$ to attend only to preceding frames:

$$\mathbf{h}^i = f_\eta^{\text{enc}}(\mathbf{x}^i, \mathcal{P}), \quad \mathbf{h}^l_{\text{updated}} = f_\xi^{\text{temp}}(\mathbf{h}^1, \dots, \mathbf{h}^{l-1})$$

Since both the encoder and the temporal module are deterministic mappings, $p(\mathbf{x}^l | \mathbf{x}^{<l})$ reduces to conditional generation over the updated latent representation $p_\theta^{\text{dec}}(\mathbf{x}^l | \mathbf{h}^l_{\text{updated}})$. Disabling inter-frame attention (identity attention) switches the model to independent sampling mode, analogous to joint image-video training in video generation.

### Key Design 3: SE(3) Diffusion Decoder

Rather than discretizing protein structures into tokens (which incurs VQ-VAE quantization error), the model operates directly on the continuous SE(3) manifold. Building upon ConfDiff, forward diffusion kernels are defined separately for translations and rotations, and the model is trained with a denoising score matching (DSM) objective:

$$\mathcal{L}_{\text{DSM}}^{\text{SE(3)}} = \mathbb{E}\left[\lambda(t)\|s_\theta(\mathbf{T}_t^l, \mathbf{h}^l, t) - \nabla_{\mathbf{T}_t^l}\log p_{t|0}(\mathbf{T}_t^l|\mathbf{T}_0^l)\|^2\right] + \text{rotation term}$$

Gradients are backpropagated through $\mathbf{h}^l$ to update the temporal module and encoder weights. At inference, clean conformations are generated by reverse diffusion from SE(3) prior noise.

### Architecture Details

- **Encoding Layer**: A frozen OpenFold (3 recycling iterations) extracts shared protein representations $\mathcal{P} = [\mathbf{s}, \mathbf{z}]$. A FrameEncoder encodes frame-level pair representations from pseudo-Cβ pairwise distances via triangle updates. Masked frames "[M]" are implemented by zeroing out Cβ distances.
- **Trajectory Module**: Alternating StructuralUpdate (Pairformer triangle operations updating single/pair embeddings) and TemporalUpdate (Llama-style causal Transformer with RoPE, applying channel-wise independent attention).
- **Structure Decoder**: A ConfDiff architecture (IPA + Transformer layers) predicts per-residue SE(3) rigids; an additional AngleResNet predicts 7 torsion angles to recover all-atom coordinates.

### Loss & Training

- **Mixed Training**: Base model trains trajectory and single-frame objectives at a 1:1 ratio; ConfRover-interp adds the interpolation objective at a 1:1:1 ratio.
- **Multi-Timescale**: $L=8$ frame trajectories with strides of 1–1024 MD snapshots (10 ps interval each), covering diverse dynamics timescales.
- **Initialization**: DiffusionDecoder is initialized from ConfDiff pretrained weights; the FoldingModule is frozen.

## Key Experimental Results

### Dataset

ATLAS: approximately 1,300 proteins, each with 3 independent 100 ns simulation trajectories, split into training and test sets by protein sequence identity.

### Trajectory Simulation — Multi-Start Benchmark (Pearson Correlation)

| Feature Space | Metric | MDGen | ConfRover |
|---------------|--------|-------|-----------|
| Cα coordinates | Trajectory | 0.56±0.03 | **0.75±0.01** |
| Cα coordinates | Frame | 0.47±0.03 | **0.63±0.01** |
| Cα coordinates | ΔFrame | 0.41±0.02 | **0.53±0.01** |
| PCA 2D | Trajectory | 0.18±0.01 | **0.73±0.01** |
| PCA 2D | Frame | 0.15±0.01 | **0.50±0.01** |

Improvements in PCA space are particularly pronounced (+306%), indicating that the autoregressive formulation better captures conformational changes aligned with principal structural variation directions.

### 100 ns Long Trajectory — Conformational State Recovery

| Method | JSD ↓ | Recall ↑ | F1 ↑ |
|--------|-------|----------|------|
| MD 100ns (oracle) | 0.31 | 0.67 | 0.79 |
| MDGen | 0.56±0.01 | 0.29±0.01 | 0.42±0.01 |
| **ConfRover** | **0.51±0.01** | **0.42±0.00** | **0.58±0.00** |

ConfRover achieves tICA slow-mode recovery comparable to the MD oracle and substantially outperforms MDGen. For some proteins (e.g., 7NMQ-A), MD can cross energy barriers to reach more distant conformational states that ConfRover has not yet fully replicated.

### Time-Independent Conformational Sampling

| Method | Pairwise RMSD r↑ | RMSF r↑ | RMWD↓ | MD PCA $\mathcal{W}_2$↓ |
|--------|-------------------|---------|-------|-------------|
| AlphaFlow | **0.56** | **0.85** | 2.62 | 1.52 |
| ConfDiff | 0.54 | **0.85** | 2.70 | **1.44** |
| ConfRover | 0.51 | **0.85** | **2.66** | 1.47 |
| ConfRover-traj | 0.48 | 0.84 | 2.85 | 1.43 |

As a general-purpose model, ConfRover achieves independent sampling performance comparable to the dedicated models AlphaFlow and ConfDiff, outperforming at least one prior SOTA on 5 of 8 metrics.

### Conformational Interpolation

Intermediate frames generated by ConfRover-interp exhibit monotonically increasing distance from the start frame and monotonically decreasing distance from the end frame, forming smooth transition paths. The base ConfRover without interpolation training fails to converge toward the target end state. Visualizations show that intermediate conformations are highly consistent with MD reference transition pathways.

### Conformational Quality (MolProbity + MadraX Energy)

| Method | Rama outliers%↓ | Rotamer outliers%↓ | MolProbity↓ | MadraX Energy↓ |
|--------|-----------------|---------------------|-------------|----------------|
| MD Reference | 0.38 | 1.02 | 0.72 | -519.3 |
| MDGen | 0.93 | 2.86 | 2.24 | -314.7 |
| **ConfRover** | **0.58** | **1.98** | **1.72** | **-522.2** |

The geometric quality and energy levels of ConfRover-generated conformations are close to the MD reference and far superior to MDGen.

### Ablation Study: Importance of Mixed Training

Removing the single-frame training objective (ConfRover-traj) increases RMWD from 2.66 to 2.85 and decreases Pairwise RMSD correlation from 0.51 to 0.48, confirming that mixed training is critical for balanced multi-task learning.

## Highlights & Insights

- **Elegant Unified Framework Design**: By varying conditioning settings and sequence ordering, a single model serves three distinct tasks. Interpolation is achieved via sequence reordering (prepending the terminal frame) without any architectural modification, inspired by text infilling.
- **Continuous-Space Autoregression**: SE(3) diffusion replaces discrete tokenization as the output head of the autoregressive model, analogous to MAR in image generation but extended to the SE(3) manifold.
- **Transfer of Llama Architecture**: Causal Transformers from large language models are adapted to protein dynamics, with channel-wise attention processing single/pair embeddings at high computational efficiency.
- **Transferability**: The mixed training strategy and sequence-reordering interpolation technique are transferable to other temporal structure generation tasks such as RNA dynamics and small-molecule conformation generation.

## Limitations & Future Work

- Trajectory simulation comparisons are limited to MDGen (weights for methods such as AlphaFold variants are not publicly available), resulting in a narrow set of baselines.
- ATLAS covers only 100 ns simulations of single-chain proteins, excluding large conformational changes and protein complexes.
- Pairformer triangle update operations are computationally expensive, limiting scalability to large proteins and long trajectories.
- A gap remains relative to the MD oracle, particularly in crossing energy barriers to explore distant conformational states.

## Related Work & Insights

| Method | Trajectory Simulation | Independent Sampling | Interpolation | Variable Length | Probabilistic | Cross-Protein Generalization |
|--------|----------------------|---------------------|---------------|-----------------|---------------|------------------------------|
| MDGen | ✓ | ✗ | Separate model required | ✗ | ✓ | ✓ |
| AlphaFlow/ConfDiff | ✗ | ✓ | ✗ | — | ✓ | ✓ |
| GST | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **ConfRover** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

## Rating

- Novelty: ⭐⭐⭐⭐ The framing of autoregressive multi-task unification is elegant, though individual modules are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of three tasks, ablation studies, and quality evaluation, but the number of baselines is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative is clear, derivation from observations to method is natural, and figures are well designed.
- Value: ⭐⭐⭐⭐ The first generative framework to unify protein conformation and dynamics, representing a significant contribution to computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)
- [\[NeurIPS 2025\] Protein Design with Dynamic Protein Vocabulary](protein_design_with_dynamic_protein_vocabulary.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
