---
title: >-
  [Paper Note] From Holo Pockets to Electron Density: GPT-style Drug Design with Density
description: >-
  [ICML 2026][Computational Biology][structure-based drug design] This paper replaces the "rigid empty pocket" condition in structure-based drug design with a "filler low-resolution electron cloud containing ligands and so…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "structure-based drug design"
  - "electron density"
  - "autoregressive"
  - "FSMILES"
  - "GPT"
date: 2026-05-08
content_hash: aeb94c3fabd208e5
---

# From Holo Pockets to Electron Density: GPT-style Drug Design with Density

**Conference**: ICML 2026  
**arXiv**: [2605.08767](https://arxiv.org/abs/2605.08767)  
**Code**: https://jiahaochen1.github.io/EDMolGPT_Page/ (Project page available)  
**Area**: Structure-based drug design / Generative molecular modeling  
**Keywords**: structure-based drug design, electron density, autoregressive, FSMILES, GPT

## TL;DR
This paper replaces the "rigid empty pocket" condition in structure-based drug design with a "filler low-resolution electron cloud containing ligands and solvents." It proposes the first decoder-only autoregressive EDMolGPT, achieving a bioactive recovery of 41% on 101 DUD-E targets, significantly outperforming previous ED-based methods.

## Background & Motivation

**Background**: The mainstream structure-based drug design (SBDD) workflow starts with a holo protein-ligand complex, **removes** the filler (existing ligand + solvent), and uses the remaining empty pocket as the generation condition. This is supported by autoregressive or diffusion models such as Pocket2Mol, TargetDiff, Lingo3DMol, and MolCRAFT.

**Limitations of Prior Work**: The empty pocket represents a single-frame static conformation, which suppresses the intrinsic flexibility of the protein and ignores ligand-induced conformational adaptation. A few studies that attempted to use pocket electron clouds (ECloudGen, ED2Mol) found that signals are weak and unstable in flexible regions, which introduces more noise.

**Key Challenge**: Drug design requires a condition that both reflects the "real binding environment of the target" and can be fed into a generative model as a "unified representation." A rigid pocket satisfies the latter while sacrificing the former, whereas a pocket ensemble satisfies the former while sacrificing the latter.

**Goal**: (1) Identify a condition capable of encoding ensemble-averaged conformational information while maintaining a unified representation; (2) Develop a generative model capable of utilizing this condition, supporting both large-scale pre-training and fine-tuning on experimental data.

**Key Insight**: The electron cloud of the filler (ligand + solvent within 4.5 Å) is typically well-defined (experimentally validated) and naturally encodes where the ligand resides and which hydrogen-bond networks are active. This is more "tangible" than the void of a pocket.

**Core Idea**: Use the low-resolution electron cloud point cloud of the filler as the condition. Employ a decoder-only GPT-style autoregressive model to predict FSMILES and discretized 3D geometry within a unified pipeline of large-scale CalED pre-training and ExpED fine-tuning.

## Method

### Overall Architecture
Two types of electron clouds are utilized: CalED, calculated from atomic coordinates via FFT (used for large-scale pre-training on ~2M molecules), and ExpED, obtained directly from cryo-EM/X-ray experimental data (used for fine-tuning on 40k PDBbind complexes). Either source is eventually truncated to a resolution of $d_{\min}=3.5\text{Å}$ and sampled as a fixed point cloud of $N_p=199$ points. Each point is assigned a pharmacophore label (HBD / HBA / HBD-HBA / Other), sorted by $(x,y,z)$, and concatenated with the molecule token sequence for input into the GPT model.

### Key Designs

1.  **Filler ED as condition**:
    - **Function**: Represents the dynamic binding environment using a continuous, physically grounded scalar field, avoiding the rigid pocket assumption.
    - **Mechanism**: The filler includes the ligand and solvents within 4.5 Å. CalED calculates structure factors $F(h) = \sum_i f_i(h) e^{2\pi i h\cdot v_f^i}$ and performs a truncated inverse FFT $\rho(v_f) = V^{-1} \sum_{|h|\le 1/d_{\min}} F(h) e^{-2\pi i h\cdot v_f}$ to obtain the density map. ExpED is derived directly from experimental measurements, skipping the FFT. Then, $N_p$ points are randomly sampled from $\rho$, each assigned a pharmacophore type $c_p^i$ based on the nearest atom, resulting in a semantic point cloud $\mathcal{P}_f = \{(c_p^i, v_p^i)\}$.
    - **Design Motivation**: An empty pocket is "removing information"; filler ED is "retaining all traces of interaction." ExpED is more realistic with inherent flexibility and noise but suffers from data scarcity, while CalED provides abundant data. Both are unified into the same point cloud format for seamless pre-training and fine-tuning.

2.  **GPT-style autoregressive molecular generation**:
    - **Function**: Uses a decoder-only architecture to predict atom types, 3D coordinates, and bond geometry simultaneously, avoiding the complexity of encoder-decoder or diffusion models.
    - **Mechanism**: Molecules are represented using FSMILES (fragment-level SMILES, avoiding fragmentation of ring bonds) and discretized 3D coordinates $\hat v_m^i = \lfloor (v_m^i - \mu_m)/\sigma \rfloor$ with $\sigma=0.1$, mapping $\pm 15\text{Å}$ to $[-150,150]$. Additional discrete values for bond lengths $l_m^i = \|v_m^i - v_m^{i-1}\|$, bond angles $\theta_m^i$, and dihedral angles $\phi_m^i$ are included. Point cloud tokens and molecule tokens share coordinate embeddings. A GPT-2 medium style 24-layer Transformer is used, with cross-entropy optimizing all discrete outputs.
    - **Design Motivation**: Split generation in encoder-decoder models loses context, while diffusion models are slow and require SE(3) equivariant designs. The GPT approach is simple and scalable. During inference, $(l,\theta,\phi)$ can constrain the next atom coordinate to a spherical surface to improve stability.

3.  **Geometrically constrained inference sampling**:
    - **Function**: Prevents the autoregressive generation from producing physically unreasonable distorted conformations.
    - **Mechanism**: During inference, instead of direct temperature sampling on the three independent coordinates of $v_m^i$, $(l_m^i, \theta_m^i, \phi_m^i)$ are sampled first. Then, a local frame is defined by the preceding three atom positions to constrain feasible $v_m^i$ sampling to a spherical surface with radius $l_m^i$ parameterized by $\theta$ and $\phi$.
    - **Design Motivation**: Direct coordinate sampling leads to accumulated autoregressive errors. Parameterization via bond length, angle, and dihedral provides a search space with superior chemical validity while significantly reducing the search space to enhance stability.

### Loss & Training
Cross-entropy loss: $\mathcal{L} = -\frac{1}{N_m}\sum_t \log p((\hat a_m^t, \hat v_m^t, \hat l_m^t, \hat\theta_m^t, \hat\phi_m^t) \mid h_p^{1:N_p}, h_m^{1:t-1})$. AdamW optimizer with lr $1\times 10^{-5}$, warmup for 1000 steps followed by cosine decay. Batch size of 96 for 100 epochs on 2× A40 GPUs. Inference temperature $T=0.7$.

## Key Experimental Results

### Main Results (DUD-E 101 targets, CalED)

| Method | Bio. Recov. ↑ | Min-in-place ↓ | Redocking ↓ | Min<Re ↑ |
|------|---------------|----------------|--------------|----------|
| Pocket2Mol | 8% | -6.7 | -7.5 | 17.9% |
| TargetDiff | 3% | -6.2 | -7.0 | 15.2% |
| Lingo3DMol | 33% | -6.8 | -7.8 | 12.0% |
| MolCRAFT | 17% | -6.1 | -6.9 | 20.1% |
| ED2Mol | 3% | -5.22 | -6.15 | 7.4% |
| ECloudGen† | 33% | — | -6.68 | — |
| **EDMolGPT** | **41%** | **-6.92** | -7.18 | **37%** |
| Reference (Ground truth active ligand) | — | -7.93 | -7.93 | — |

### Ablation Study (Resolution and Temperature)

| $d_{\min}$ | $T$ | Min-in-place | Recov. | Div ↓ |
|-----------|-----|---------------|--------|-------|
| 1.5 Å | 0.7 | -6.94 | 46% | 0.186 |
| 1.5 Å | 1.2 | -6.90 | 44% | 0.178 |
| 3.5 Å | 0.7 | -6.92 | 41% | 0.184 |
| 3.5 Å | 1.2 | -6.91 | 41% | 0.176 |

ExpED subset (92 targets with experimental density): Min-in-place $-5.4$, recovery 20%, QED 0.50. It successfully generates active ligands that rigid pocket models reject due to steric clash but are permitted by the flexibility of experimental conformations.

### Key Findings
- The point cloud count $N_p=199$ has a greater impact on low-resolution representation than $d_{\min}$. Maintaining $N_p$ keeps the ECFP similarity between generated molecules and reference ligands below 0.2, proving the condition does not leak the 2D structure of the reference ligand.
- Comparison with ED2Mol by molecular weight bins: ED2Mol shows high QED (0.66) at $<180$ Da, which is essentially "cheating" by drawing small molecules. EDMolGPT maintains SAS $\approx 3.8$ in higher molecular weight ranges, closer to the weight of real drug candidates.
- Although docking scores on ExpED appear lower, some generated ligands cover "experimentally feasible" chemical space excluded by rigid pocket models due to steric clashes, suggesting traditional SBDD evaluations may be biased against flexible scenarios.

## Highlights & Insights
- The contrast between "removing filler" and "retaining filler" is stark. The authors demonstrate the advantages of filler-encoded flexibility using an experimental density map of PDB 6KMP with strong visual evidence.
- The combination of a decoder-only GPT route, discretized geometry, and spherical sampling allows SBDD to move beyond the complex engineering of SE(3) equivariance or diffusion. A simple architecture achieved SOTA performance.
- The dual data source strategy of CalED + ExpED providing a "large-scale pre-train + experimental fine-tune" template is applicable to any cryo-EM molecular task.

## Limitations & Future Work
- ExpED is limited by the scarcity of experimental data (only 92 targets), restricting generalization. Inference requires known fillers (i.e., a ligand must already occupy the site), necessitating docking for entirely new targets.
- Drug-like indices such as QED are only moderate, as no post-generation force-field optimization was performed (part of the gap with ED2Mol arises from post-processing rather than modeling).
- The decoder-only model does not explicitly model SE(3) equivariance and relies on coordinate embeddings to learn symmetry; rotational robustness has not been quantified.
- No wet-lab validation was conducted; 41% recovery refers to structural similarity in a computational sense, not necessarily actual potency.

## Related Work & Insights
- **vs Pocket2Mol / Lingo3DMol**: They condition on empty pocket geometry; this work conditions on filler ED, providing richer physical information.
- **vs TargetDiff / MolCRAFT**: They utilize diffusion routes; this work uses a decoder-only approach that is simpler and faster at inference.
- **vs ECloudGen / ED2Mol**: Previous ED-based works used pocket ED or fragment-assembly; this work uses filler ED and end-to-end atom-level generation.
- **vs cryo-EM molecule fitting**: Traditional fitting is "known molecule → fit to density"; this work is "known density → generate molecule," reversing the direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Switching to filler ED condition and decoder-only SBDD are both firsts.
- Experimental Thoroughness: ⭐⭐⭐⭐ 101 targets, multi-dimensional metrics, and ExpED subset included, though wet-lab data is missing.
- Writing Quality: ⭐⭐⭐⭐ Figures 1 and 2 present the motivation intuitively; formulas and algorithms are clear.
- Value: ⭐⭐⭐⭐ Establishes a new baseline for ED-guided drug design with high potential as cryo-EM data grows in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](../../NeurIPS2025/computational_biology/edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)
- [\[ICML 2026\] Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent](site4drug_predicting_drug-binding_target_sites_with_an_ai_agent.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)

</div>

<!-- RELATED:END -->
