---
title: >-
  [Paper Note] From Holo Pockets to Electron Density: GPT-style Drug Design with Density
description: >-
  [ICML 2026][Medical Imaging][structure-based drug design] This work replaces the structure-based drug design condition from a "rigid empty pocket" to a "filler low-resolution electron cloud containing ligand and solvent…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "structure-based drug design"
  - "electron density"
  - "autoregressive"
  - "FSMILES"
  - "GPT"
date: 2026-05-08
content_hash: 6e95a04349dc4d20
---

# From Holo Pockets to Electron Density: GPT-style Drug Design with Density

**Conference**: ICML 2026  
**arXiv**: [2605.08767](https://arxiv.org/abs/2605.08767)  
**Code**: https://jiahaochen1.github.io/EDMolGPT_Page/ (project page available)  
**Area**: Structure-based drug design / Generative molecular modeling  
**Keywords**: structure-based drug design, electron density, autoregressive, FSMILES, GPT

## TL;DR
This work replaces the structure-based drug design condition from a "rigid empty pocket" to a "filler low-resolution electron cloud containing ligand and solvent," and proposes the first decoder-only autoregressive EDMolGPT. On DUD-E's 101 targets, it achieves a bioactive recovery of 41%, far surpassing previous ED-based methods.

## Background & Motivation

**Background**: The mainstream structure-based drug design (SBDD) workflow starts from holo protein-ligand complexes, **removes** the filler (existing ligand + solvent), and uses the resulting empty pocket as the generation condition. Supporting models include Pocket2Mol, TargetDiff, Lingo3DMol, MolCRAFT, etc., which are autoregressive or diffusion-based.

**Limitations of Prior Work**: The empty pocket is a single-frame static conformation, which suppresses the intrinsic flexibility of the protein and ignores ligand-induced conformational adaptation. A few attempts to use pocket electron clouds (ECloudGen, ED2Mol) found that electron density in flexible regions is inherently weak and unstable, introducing more noise.

**Key Challenge**: Drug design requires a condition that both reflects the "true binding environment of the target" and can be "fed to the generative model in a unified representation." The rigid pocket satisfies the latter at the expense of the former, while pocket ensembles satisfy the former but sacrifice the latter.

**Goal**: (1) Find a condition that encodes ensemble-averaged conformational information and can be represented uniformly; (2) Develop a generative model that can effectively utilize this condition, supporting large-scale pretraining and fine-tuning with experimental data.

**Key Insight**: The electron cloud of the filler (ligand + solvent within 4.5 Å) is usually well-defined (directly validated by experiments) and naturally encodes "where the ligand actually binds and which surrounding H-bond networks are active"—more "real" than the void of the pocket.

**Core Idea**: Use the low-resolution electron cloud point cloud of the filler as the condition, and employ a decoder-only GPT-style autoregressive model to predict FSMILES + discretized 3D geometry, forming a unified pipeline for large-scale CalED pretraining and ExpED fine-tuning.

## Method

### Overall Architecture
Two types of electron clouds: CalED is computed from atomic coordinates via FFT (used for large-scale pretraining on ~2M molecules); ExpED is obtained directly from cryo-EM/X-ray experimental data (used for fine-tuning on 40k PDBbind complexes). Regardless of source, after truncating the resolution at $d_{\min}=3.5\text{Å}$, the density is sampled into a fixed $N_p=199$ point cloud, each point labeled with a pharmacophore type (HBD / HBA / HBD-HBA / Other). After sorting by $(x,y,z)$, these are concatenated with the molecular token sequence and input to GPT.

### Key Designs

1. **Filler ED as Condition**:

    - **Function**: Uses a continuous, physically grounded scalar field to represent the dynamic binding environment, avoiding the rigid pocket assumption.
    - **Mechanism**: The filler includes the ligand and solvent within 4.5 Å. CalED computes structure factors $F(h) = \sum_i f_i(h) e^{2\pi i h\cdot v_f^i}$, then applies truncated inverse FFT $\rho(v_f) = V^{-1} \sum_{|h|\le 1/d_{\min}} F(h) e^{-2\pi i h\cdot v_f}$ to obtain the density map; ExpED is directly measured, skipping FFT. $N_p$ points are randomly sampled from $\rho$, each assigned a pharmacophore type $c_p^i$ based on the nearest atom, yielding a semantic point cloud $\mathcal{P}_f = \{(c_p^i, v_p^i)\}$.
    - **Design Motivation**: The empty pocket "removes information"; filler ED "retains all interaction traces." ExpED naturally includes flexibility and more realistic noise but is limited in quantity; CalED provides abundant data. Both are unified into the same point cloud format, enabling seamless pretraining and fine-tuning.

2. **GPT-style Autoregressive Molecular Generation**:

    - **Function**: Uses a decoder-only architecture to simultaneously predict atom types, 3D coordinates, and chemical bond geometry, avoiding the complexity of encoder-decoder or diffusion models.
    - **Mechanism**: Molecules are represented by FSMILES (fragment-level SMILES, avoiding ring bond fragmentation) + discretized 3D coordinates $\hat v_m^i = \lfloor (v_m^i - \mu_m)/\sigma \rfloor$, with $\sigma=0.1$ mapping $\pm 15\text{Å}$ to $[-150,150]$; also includes discretized bond length $l_m^i = \|v_m^i - v_m^{i-1}\|$, bond angle $\theta_m^i$, and dihedral angle $\phi_m^i$. Point cloud tokens and molecular tokens share coordinate embeddings. The model uses a GPT-2 medium style 24-layer Transformer, with cross-entropy loss for all discrete outputs.
    - **Design Motivation**: Encoder-decoder splits generation and loses context; diffusion is slow and requires SE(3) equivariant design. The GPT approach is simple, scalable, and during inference, $(l,\theta,\phi)$ can constrain the next atom's coordinates to a sphere, improving stability.

3. **Geometry-Constrained Inference Sampling**:

    - **Function**: Prevents autoregressive generation from producing physically implausible distorted conformations.
    - **Mechanism**: During inference, instead of sampling the three independent coordinates of $v_m^i$ directly, $(l_m^i, \theta_m^i, \phi_m^i)$ are sampled first. Using the positions of the previous three atoms to define a local frame, feasible $v_m^i$ are constrained to a sphere of radius $l_m^i$ parameterized by $\theta, \phi$.
    - **Design Motivation**: Direct coordinate sampling accumulates autoregressive errors; parameterizing by bond length/angle/dihedral provides a chemically reasonable search space and greatly reduces the search space, enhancing stability.

### Loss & Training

Cross-entropy: $\mathcal{L} = -\frac{1}{N_m}\sum_t \log p((\hat a_m^t, \hat v_m^t, \hat l_m^t, \hat\theta_m^t, \hat\phi_m^t) \mid h_p^{1:N_p}, h_m^{1:t-1})$. AdamW with learning rate $1\times 10^{-5}$, 1000-step warmup + cosine decay; batch size 96, 100 epochs; 2× A40 GPUs. Inference temperature $T=0.7$.

## Key Experimental Results

### Main Results (DUD-E 101 Targets, CalED)

| Method | Bio. Recov. ↑ | Min-in-place ↓ | Redocking ↓ | Min<Re ↑ |
|--------|---------------|----------------|-------------|----------|
| Pocket2Mol | 8% | -6.7 | -7.5 | 17.9% |
| TargetDiff | 3% | -6.2 | -7.0 | 15.2% |
| Lingo3DMol | 33% | -6.8 | -7.8 | 12.0% |
| MolCRAFT | 17% | -6.1 | -6.9 | 20.1% |
| ED2Mol | 3% | -5.22 | -6.15 | 7.4% |
| ECloudGen† | 33% | — | -6.68 | — |
| **EDMolGPT** | **41%** | **-6.92** | -7.18 | **37%** |
| Reference (true active ligand) | — | -7.93 | -7.93 | — |

### Ablation Study (Resolution and Temperature)

| $d_{\min}$ | $T$ | Min-in-place | Recov. | Div ↓ |
|------------|-----|--------------|--------|-------|
| 1.5 Å | 0.7 | -6.94 | 46% | 0.186 |
| 1.5 Å | 1.2 | -6.90 | 44% | 0.178 |
| 3.5 Å | 0.7 | -6.92 | 41% | 0.184 |
| 3.5 Å | 1.2 | -6.91 | 41% | 0.176 |

ExpED subset (92 targets with experimental density): Min-in-place $-5.4$, recovery 20%, QED 0.50. Can generate active ligands that rigid pocket would reject due to steric clash, but are allowed by experimental conformational flexibility.

### Key Findings
- The $N_p=199$ point cloud has a greater impact on low-resolution representation than $d_{\min}$—maintaining $N_p$ ensures generated molecules have ECFP similarity $< 0.2$ to reference ligands, proving the condition does not leak the 2D structure of the reference ligand.
- Compared with ED2Mol by molecular weight bins: ED2Mol achieves higher QED (0.66) for $<180$ Da, but this is essentially "drawing small molecules" as a shortcut; EDMolGPT maintains SAS $\approx 3.8$ for larger molecules, closer to real drug candidate weight ranges.
- Docking scores on ExpED appear lower, but some generated ligands precisely cover "experimentally feasible" chemical space excluded by rigid pocket due to steric clash—traditional SBDD evaluation is less fair for flexible scenarios.

## Highlights & Insights
- The contrast between "removing filler" and "retaining filler" is striking; the authors directly demonstrate the flexibility-encoding advantage of filler using an experimental density map from PDB 6KMP, with strong visual persuasiveness.
- The combination of decoder-only GPT architecture, discretized geometry, and spherical sampling allows SBDD to move beyond SE(3) equivariant/diffusion complexity, achieving SOTA with a simple architecture.
- The "computed + experimental" dual data source strategy (CalED + ExpED) provides a clear "large-scale pretrain + experimental fine-tune" template for future work, transferable to any cryo-EM molecular task.

## Limitations & Future Work
- ExpED is limited by the scarcity of experimental data, covering only 92 targets, restricting generalization; inference requires known filler (i.e., an existing ligand), so for novel targets, docking is still needed first.
- Drug-likeness metrics such as QED are only moderate, and no post-generation force-field optimization was performed (some differences with ED2Mol stem from post-processing rather than modeling).
- The decoder-only model does not explicitly model SE(3) equivariance, relying on coordinate embeddings to learn symmetry; rotational robustness is unquantified.
- No wet-lab validation; the 41% recovery refers to computational structural similarity, not actual druggability.

## Related Work & Insights
- **vs Pocket2Mol / Lingo3DMol**: Their condition is on empty pocket geometry; this work uses filler ED, which is richer in physical information.
- **vs TargetDiff / MolCRAFT**: They use diffusion models; this work's decoder-only approach is simpler and faster at inference.
- **vs ECloudGen / ED2Mol**: Previous ED-based work used pocket ED or fragment-assembly; this work uses filler ED and end-to-end atom-level generation.
- **vs cryo-EM molecule fitting**: Traditional fitting is "known molecule → fit density," while this work is "known density → generate molecule," reversing the direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The switch to filler ED as condition and decoder-only SBDD are both firsts.
- Experimental Thoroughness: ⭐⭐⭐⭐ 101 targets + multi-dimensional metrics + ExpED subset, but lacks wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2 clearly illustrate the motivation; formulas and algorithms are clear.
- Value: ⭐⭐⭐⭐ Establishes a new baseline for ED-guided drug design, with great potential as more cryo-EM data becomes available in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](../../NeurIPS2025/medical_imaging/edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[ICLR 2026\] CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](../../ICLR2026/medical_imaging/cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](../../NeurIPS2025/medical_imaging/flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/medical_imaging/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)

</div>

<!-- RELATED:END -->
