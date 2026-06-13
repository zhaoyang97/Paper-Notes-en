---
title: >-
  [Paper Note] TriForces: Augmenting Atomistic GNNs for Transferable Representations
description: >-
  [ICML 2026][Physics & Scientific Computing][MLIP] TriForces decomposes atomistic Graph Neural Networks into three parallel streams—"composition, structure…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "MLIP"
  - "Self-supervised Pre-training"
  - "Three-stream Architecture"
  - "SOAP Descriptor"
  - "Transfer Learning"
date: 2026-05-08
content_hash: 2d5b3213215d57a6
---

# TriForces: Augmenting Atomistic GNNs for Transferable Representations

**Conference**: ICML 2026  
**arXiv**: [2605.20581](https://arxiv.org/abs/2605.20581)  
**Code**: https://github.com/Ramlaoui/triforces (Available)  
**Area**: Physics / Atomistic Machine Learning Potentials / Geometric Graph Neural Networks  
**Keywords**: MLIP, Self-supervised Pre-training, Three-stream Architecture, SOAP Descriptor, Transfer Learning  

## TL;DR
TriForces decomposes atomistic Graph Neural Networks into three parallel streams—"composition, structure, and interaction"—and overlays multi-objective self-supervised pre-training (LeJEPA + denoising + masking). This makes MLIPs more robust across few-shot transfer, cross-domain fine-tuning, and similar structure retrieval compared to single-stream baselines.

## Background & Motivation

**Background**: Machine Learning Interatomic Potentials (MLIP) trained on DFT data have become the primary tool for materials discovery and molecular dynamics. Geometric GNNs such as MACE, eSEN, and Orb-v3 have reached energy and force prediction accuracies on large-scale datasets like OMat24 and MPtrj that approach the errors of DFT itself.

**Limitations of Prior Work**: Practical applications almost always require fine-tuning on small, expensive downstream data. However, the transferability of current MLIPs is highly unstable. A model pre-trained on 100M structures might fail to even fine-tune on diagnostic tasks like "predicting crystal systems" or "predicting majority elements." Performance fluctuates significantly when changing assemblies, functionals, or systems.

**Key Challenge**: Representations are optimized for predicting energy/forces rather than for reuse. Supervised training entangles composition and geometric information within the same latent vector. If downstream tasks only require composition or geometry, they cannot access clean, reusable representations. Although self-supervised learning (SSL) has proven effective in preserving semantic structures in vision and language, it has primarily been used as an auxiliary loss in the atomistic domain without a systematic verification of how SSL interacts with architectural inductive biases.

**Goal**: This work decomposes the problem into two sub-problems: (1) how to make the architecture itself explicitly preserve composition and geometric information; and (2) how to make SSL truly effective for tasks "beyond prediction," such as low-data transfer, representation organization, and retrieval.

**Key Insight**: The authors observe that "energy and force coupling gradients" compete during conservative training, which prior methods mitigated using tricks like multi-stage scheduling or special initialization. By isolating composition-related "force-preserving" degrees of freedom, it is possible to lower energy MAE without sacrificing force MAE.

**Core Idea**: Replace single-stream supervised training with a "three-stream decomposition + multi-objective SSL" framework, providing dedicated channels for composition, structure, and interaction information in the representation space.

## Method

### Overall Architecture
The input is an atomic structure $\mathcal{X}=(\{z_i\},\{\mathbf{x}_i\})$, consisting of atomic numbers $z_i$ and positions $\mathbf{x}_i$, with graph construction using a radial cutoff. TriForces splits the node-level representation into three concatenated segments: $\mathbf{h}_i=[\mathbf{h}^{\text{comp}}_i,\mathbf{h}^{\text{struct}}_i,\mathbf{h}^{\text{int}}_i]$. Pre-training is performed on 5M bulk structures from LeMat-Bulk, where three SSL objectives share the same set of randomly augmented views. During downstream fine-tuning, the three streams are fed into the prediction head together. Three variants are distinguished: TriForces-Streams (architecture only, random initialization), TriForces (architecture + SSL), and Base+SSL (SSL only).

### Key Designs

1.  **Composition Transformer in Three-Stream Decomposition (with Count-weighted Attention)**:
    - **Function**: Extracts pure chemical fingerprints from the list of atoms using only "elements + counts," fully decoupled from geometry.
    - **Mechanism**: Compresses the structure into $T$ unique element tokens $\{(z_t,c_t)\}$. Each token is initialized with a learnable element embedding $\mathbf{u}_t=\mathbf{e}(z_t)$ and processed via a Transformer. The key modification is adding a log-count bias to the attention logits: $a^{(h)}_{ts}=\frac{(\mathbf{q}_t^{(h)})^\top \mathbf{k}_s^{(h)}}{\sqrt{d_h}}+\log(c_s)$. This is equivalent to "performing attention over all atoms of that type," but complexity is reduced from $\mathcal{O}(N^2)$ to $\mathcal{O}(T^2)$, becoming independent of the number of atoms.
    - **Design Motivation**: Previous composition models (Roost, CrabNet) normalized stoichiometry into fractions, losing system size information. TriForces retains absolute counts $c_t$ as they encode physical information such as energy magnitude.

2.  **Type-Agnostic Structural Stream (SOAP-style Power Spectrum)**:
    - **Function**: Uses rotation-invariant geometric descriptors to capture geometric motifs shared across chemical systems, decoupled from specific element identities.
    - **Mechanism**: Each neighbor displacement $\mathbf{r}_{ij}$ is jointly expanded using radial basis functions $\phi_k(r)$ (Bessel/Gaussian), real spherical harmonics $Y_{lm}(\hat{\mathbf{r}})$, and multi-scale cutoff functions $s_s(r)$ to obtain mixed channels $\tilde{\phi}_\alpha(r_{ij})=\sum_{k,s}\mathbf{W}_{\alpha,(k,s)}s_s(r_{ij})\phi_k(r_{ij})$. These are accumulated into local density coefficients $c_{\alpha lm}(i)$, and rotation invariance is enforced via the power spectrum $p_{\alpha\alpha' l}(i)=\sum_m c_{\alpha lm}(i)c_{\alpha' lm}(i)$. This is followed by a few invariant message-passing layers to overlay connectivity topology.
    - **Design Motivation**: In conservative MLIPs, force is the gradient of energy with respect to position. If the extra degrees of freedom added by the composition stream do not break geometric dependencies, they can "preserve forces." The type-agnostic structural stream allows energy gradients to propagate only through the interaction stream, theoretically avoiding gradient competition between energy and force losses (a rank-based bound is provided in the appendix).

3.  **Triple-Objective Complementary SSL Pre-training**:
    - **Function**: Drives three complementary objectives simultaneously using two views under the same set of random augmentations (position noise + atomic type masking + random graph construction + non-equivariant model plus rotation).
    - **Mechanism**: The total loss is $\mathcal{L}=\mathcal{L}_{\text{denoise}}+\lambda_{\text{mask}}\mathcal{L}_{\text{mask}}+\lambda_{\text{LeJEPA}}\mathcal{L}_{\text{LeJEPA}}$. Denoising $\mathcal{L}_{\text{denoise}}=\sum_i\|f_\theta(\tilde{\mathcal{G}})_i-\boldsymbol{\epsilon}_i\|^2$ stabilizes geometric representations; masking $\mathcal{L}_{\text{mask}}=-\sum_{i\in\mathcal{M}}\log p_\theta(z_i\mid\tilde{\mathcal{G}})$ strengthens the composition stream's learning of element co-occurrence patterns; LeJEPA aligns the two views at both node and graph granularities and uses SIGReg regularization to compress representations into isotropic Gaussians, avoiding collapse without needing stop-gradients or momentum encoders.
    - **Design Motivation**: Purely non-reconstructive objectives only pull for alignment and may lose fine geometric or chemical differences. Purely reconstructive objectives fail to organize the latent space. Combining the three allows denoising to strengthen the structure stream, masking to strengthen the composition stream, and LeJEPA to strengthen overall separability, corresponding exactly to the three architectural streams.

### Loss & Training
Pre-training initializes eSEN, Orb-v3, and MACE backbones from scratch on 5M bulk structures from LeMat-Bulk. During fine-tuning, concatenated three-stream representations are fed into downstream prediction heads. OMat24 fine-tuning runs for 2 epochs, and MatBench uses standard 5-fold cross-validation.

## Key Experimental Results

### Main Results: OMat24 Fine-tuning (4M Subset)

| Backbone / Mode | Configuration | E MAE (meV/atom) ↓ | F MAE (meV/Å) ↓ | σ MAE (meV/Å³) ↓ |
|------|------|------|------|------|
| Orb-v3 Conservative | Baseline | 107 | 150 | 7.8 |
| Orb-v3 Conservative | + Streams | 35.6 | 149 | 6.2 |
| Orb-v3 Conservative | + TriForces (full) | **19.4** | **95.5** | **4.7** |
| eSEN (equivariant) | Baseline | 104 | 80.3 | 6.3 |
| eSEN (equivariant) | + TriForces (full) | **18.8** | **78.0** | **4.4** |
| MACE (equivariant) | Baseline | 117 | 150 | 8.1 |
| MACE (equivariant) | + TriForces (full) | **34.3** | **142** | **6.1** |

On Orb-v3 Conservative, the energy MAE was slashed from 107 to 19.4 (an 82% relative improvement), while force MAE dropped from 150 to 95.5, validating the theoretical expectation that "adding the composition stream does not destroy forces."

### Ablation Study: 8 MatBench Tasks (vs DFT-labeled Pre-trained Baseline)

| Task (Unit) | MACE† | TriForces MACE | Orb† | TriForces Orb | eSEN† | TriForces eSEN |
|------|------|------|------|------|------|------|
| Phonons (cm⁻¹) | 36.7 | 27.6 | 26.2 | 22.6 | 57.8 | **19.5** |
| Log GVRH | 0.082 | 0.073 | 0.063 | 0.058 | 0.093 | 0.058 |
| Perovskites (meV) | 61.4 | 35.1 | 30.7 | 26.5 | 40.1 | **25.6** |
| MP Gap (eV) | 0.370 | 0.250 | 0.194 | 0.132 | 0.392 | 0.139 |
| MP E Form (meV/atom) | 40.8 | 34.4 | 21.1 | 17.1 | 83.5 | 20.2 |

TriForces achieved the best overall results in 6 out of 8 tasks using only self-supervised pre-training without DFT labels. On the Phonons task, eSEN error dropped directly from 57.8 to 19.5.

### Key Findings
- In large-scale supervised scenarios (OMat24 full set), architectural branching is the main contributor, while SSL provides a minor final accuracy boost but significantly accelerates convergence. In low-data scenarios, SSL is critical; at 20K samples, TriForces reduced energy MAE from 81.3 to 34.6 (a 57% reduction).
- Conservative models (Force = Energy Gradient) benefit most from TriForces, confirming the hypothesis that the composition stream provides "force-preserving degrees of freedom."
- Comparison with a widened baseline of equal parameter count (Appendix B.7) shows TriForces still dominates in 8/8 MatBench tasks and 6/7 QM9 targets, ruling out simple "parameter stacking" as the explanation.
- The learned latent space supports decomposable similar structure retrieval by chemistry, geometry, or both, opening new "beyond-prediction" uses for MLIP representations.

## Highlights & Insights
- **Architectural Branching $\neq$ Simple Multi-tasking**: Splitting composition, structure, and interaction into independent encoding paths allows the three SSL objectives to operate independently without interference. This alignment design of "architectural inductive bias × SSL objective" is a valuable lesson for other multi-modal/multi-task scenarios.
- **Count-weighted Attention** is a clean trick: It strictly equivalizes "element-wise deduplication + log-count bias" with "attention over all atoms." It saves computation while retaining physical meaning and can be directly reused in any set-based chemical/molecular representation.
- **Architectural Solution for Energy-Force Gradient Coupling**: While others use scheduling strategies or diffusion pre-training to avoid energy/force loss competition, TriForces provides "force-preserving degrees of freedom" at the architectural level. This allows conservative models to be trained effectively in one go, eliminating many hyperparameter tuning tricks.
- **Repositioning the Role of SSL**: The authors emphasize that TriForces is not "just another SSL method" but an "architectural framework + SSL enhancement." This experimental design of decoupling architecture from objectives (TriForces-Streams vs. Base+SSL vs. TriForces) serves as a standard ablation paradigm for multi-task methods.

## Limitations & Future Work
- The parameter count increases after three-stream concatenation (e.g., TriForces Orb-v3 42M vs. Orb-v3 25.5M). Although it still wins in parameter-controlled comparisons, deployment cost remains a concern, and no systematic comparison of inference speed was provided.
- The computational overhead of SOAP-style power spectra on large systems was not fully detailed. It is unclear if the type-agnostic structural stream might lose critical element differences in organic molecules containing H.
- The regularization weight $\lambda$ for LeJEPA + SIGReg is a critical hyperparameter, yet sensitivity analysis across different systems/tasks was not shown.
- Retrieval results only included qualitative visualizations, lacking quantitative metrics like nearest-neighbor mAP.

## Related Work & Insights
- **vs. JMP / DFT Pre-trained MLIPs**: While JMP and others use DFT labels for supervised pre-training, TriForces nearly closes the gap using only self-supervision, avoiding expensive DFT labels.
- **vs. Roost / CrabNet (Composition Models)**: These models discard geometry and only look at stoichiometry. TriForces embeds the composition stream within a geometric GNN, preserving composition signals without losing geometric resolution.
- **vs. Noisy Nodes / Force Field Denoising**: Prior SSL in MLIPs mostly served as an auxiliary loss with a single objective. TriForces systematically compares the complementarity of non-reconstructive, denoising, and masking objectives, providing empirical conclusions on which SSL is most useful in specific scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](../../ICLR2026/physics/augmenting_representations_with_scientific_papers.md)
- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/physics/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[ICML 2026\] PINNfluence: Interpreting PINNs Through Influence Functions](pinnfluence_interpreting_pinns_through_influence_functions.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)

</div>

<!-- RELATED:END -->
