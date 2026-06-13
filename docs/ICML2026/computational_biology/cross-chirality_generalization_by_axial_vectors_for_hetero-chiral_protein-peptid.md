---
title: >-
  [Paper Note] Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design
description: >-
  [ICML2026][Computational Biology][D-peptide design] This paper proposes AFI (Axial Feature Injection), which injects axial vector features into the polar vector channels of $E(3)$-equivariant scalarized models through li…
tags:
  - "ICML2026"
  - "Computational Biology"
  - "D-peptide design"
  - "Axial vectors"
  - "$SE(3)$ equivariance"
  - "Enantiomer"
  - "Latent diffusion"
date: 2026-05-08
content_hash: 9c0610331e8667a2
---

# Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design

**Conference**: ICML2026  
**arXiv**: [2602.20176](https://arxiv.org/abs/2602.20176)  
**Code**: https://github.com/YZY010418/PepMirror  
**Area**: Scientific Computing / Protein-Peptide Design / Equivariant Neural Networks  
**Keywords**: D-peptide design, Axial vectors, $SE(3)$ equivariance, Enantiomer, Latent diffusion

## TL;DR
This paper proposes AFI (Axial Feature Injection), which injects axial vector features into the polar vector channels of $E(3)$-equivariant scalarized models through linear mixing, degrading them to $SE(3)$-equivariant and chirality-sensitive. By modifying UniMoMo with this approach, the authors developed PepMirror, which generates hetero-chiral (D-L) peptide binders in a zero-shot manner using only homo-chiral (L-L) training data. PepMirror is further validated through wet-lab experiments on the CD38 target, marking the first experimentally confirmed AI de novo D-peptide design framework.

## Background & Motivation

**Background**: Natural proteins are almost entirely composed of L-amino acids (homo-chirality). D-peptides are highly promising therapeutic molecules due to their resistance to protease degradation, longer half-lives, and low immunogenicity. However, D-peptide binders are traditionally screened via mirror-image display, which requires the difficult synthesis of D-protein targets. While ML-based peptide design has flourished recently, most models focus solely on L-L homo-chiral interfaces.

**Limitations of Prior Work**: Existing equivariant protein generation models fall into two categories, neither of which can directly design D-peptide binders. The first category (RFDiffusion, PepFlow, D-Flow, etc.) uses rigid-body local coordinate systems to parameterize amino acids, hardcoding right-handedness as a prior; here, structures and their mirrors differ only by a rotation matrix, making it **impossible to distinguish mirror images**. The second category (PepGLAD, UniMoMo, PocketXMol) relies on $E(3)$-equivariant backbones that are strictly equivariant to spatial inversion $P=-I_3$. This leads to an "input D-target → output D-peptide" behavior—the correct direction for mirror-image display—but the models themselves **output mixed chirality** and cannot stably generate single-chiral binders. The only specialized attempt, D-Flow, remains limited to dry-lab validation.

**Key Challenge**: To design hetero-chiral interfaces, a model must simultaneously satisfy two conditions: **chirality awareness** (latent codes of $X$ and $-X$ must be separable) and **latent stability** ($-X$ should still be recognized as the same amino acid type with reversed chirality, rather than drifting to other amino acid classes). These are in tension: $E(3)$-equivariance ensures stability but kills awareness, while completely breaking equivariance may cause the representation of $-X$ to drift unpredictably. Existing chirality-aware methods (torsion in SphereNet, ChIRo, high-order spherical harmonics in Tensor Field Networks) have only been validated for classification/property prediction or are computationally expensive and prone to overfitting on high-frequency noise. They have never been applied to hetero-chiral peptide-protein interface generation.

**Goal**: (i) Provide a **lightweight** chirality-aware plugin for scalarized equivariant models; (ii) Theoretically guarantee that the resulting latent codes satisfy "distance between $X$ and $-X$ < distance between $X$ and other amino acids $X'$"; (iii) Apply this to latent diffusion to create a D-peptide binder designer capable of real-world drug discovery.

**Key Insight**: Classical physics categorizes 3D vectors into two types: **polar vectors** (e.g., position, velocity, which change sign under $P$) and **axial vectors** (e.g., angular momentum, magnetic field, which remain unchanged under $P$). Original $E(3)$-equivariant scalarized models use only polar vectors. By incorporating axial vectors, $SE(3)$-equivariance (rotation, translation) is preserved while $P$-equivariance is broken, naturally enabling chirality discrimination.

**Core Idea**: Inject axial vectors (constructed via cross product, triple product, or commutator) into the polar vector channels via channel-wise linear mixing $\tilde v_{i,k}=A_k^\top v_{i,:}+B_k^\top a_{i,:}$ before the GVP-FFN in each EPT layer. This degrades the model from $E(3)$ to $SE(3)$ equivariance. Integrating this into UniMoMo's VAE+latent diffusion framework yields PepMirror.

## Method

### Overall Architecture
PepMirror adopts the two-stage latent diffusion architecture of UniMoMo. The **Encoder** (EPT, Equivariant Pretrained Transformer) maps the protein target/peptide point cloud $\mathcal{G}$ to $(H,V)$, where $H\in\mathbb{R}^{N\times K}$ are $E(3)$-invariant scalars and $V\in\mathbb{R}^{N\times 3\times K}$ are $E(3)$-equivariant vectors. The **Diffusion Module** performs conditional denoising on the binder's latent code in latent space, conditioned on the target pocket's latent code $c$. The **Decoder** reconstructs the generated binder latent code into a 3D structure. The only modification is to the EPT backbone: axial vectors are injected into $V$ before each GVP-FFN to create AFI-EPT, replacing $V_j'(X)$ with $\widetilde V_j(X)$. The inference workflow follows mirror-image display logic: negate the L-target $\mathcal{G}_t$ to its mirror $\mathcal{G}_t'$, generate an L-binder $f_\theta(\mathcal{G}_t')$ for the D-target, and negate the output back to $\mathcal{G}_b=P(f_\theta(P(\mathcal{G}_t)))$ to obtain the D-peptide binder for the original L-target.

### Key Designs

1. **AFI (Axial Feature Injection)**:
    - **Function**: Degrades the $E(3)$-equivariant scalarized backbone to $SE(3)$-equivariance, causing latent codes of each residue to produce distinguishable but limited shifts for the mirror image $-X$.
    - **Mechanism**: Axial vectors are defined as $a(Rx)=\det(R)\,Ra(x)$, which do not change sign under spatial inversion $P=-I_3$, whereas polar vectors change sign $v(-X)=-v(X)$. Before each FFN, channel linear mixing is performed: $\tilde v_{i,k}(X)=A_k^\top v_{i,:}(X)+B_k^\top a_{i,:}(X)$, where $A_k,B_k\in\mathbb{R}^K$ are learnable coefficients. Three constructions of axial vectors from adjacent channels $u,v,w\in\mathbb{R}^3$ of $V'$ are proposed: (a) cross product $u\times v$; (b) scalar triple product projection $(w\cdot(u\times v))\,w$; (c) commutator $(u\cdot v)(u\times v)$, which captures higher-frequency information of the angle between $u$ and $v$.
    - **Design Motivation**: Compared to Tensor Field Networks which use high-order spherical harmonics or tensor products (computationally heavy and prone to overfitting), AFI only uses tensors of second order or lower, resulting in negligible parameter and computational overhead. It requires minimal code changes (inserting a single mixing layer). Theoretical protection against degradation is included: if $A=B=I$ and $v\cdot a=0$, then $\|\tilde v(X)\|=\|\tilde v(-X)\|$ loses discriminative power, thus chirality distinction is guaranteed under "generic parameter probability" (Theorem 3.1).

2. **Chirality Awareness Theorems and Latent Stability**:
    - **Function**: Theoretically guarantees that "latent codes of $X$ and $-X$ have non-negligible differences" while ensuring "distance between $X$ and $-X$ is smaller than between $X$ and a different residue $X'$," resulting in 20 compact amino acid clusters in latent space with adjacent L/D sub-clusters.
    - **Mechanism**: Theorem 3.1 (informal) states that for randomly sampled mixing coefficients, $\|c(X)-c(-X)\|\ge c_W\varepsilon$ holds with probability $1-\delta_W(\varepsilon)$. Proposition 3.2 proves that without AFI, $V(-X)=-V(X)$ leads to $c(-X)=c(X)$ (since the scalar branch only takes norms). Stability is measured by $d(X_1,X_2)=\|H(X_1)-H(X_2)\|+\|\widetilde V(X_1)-\widetilde V(X_2)\|$. Since $H(-X)=H(X)$, $a(-X)=a(X)$, and $v(-X)=-v(X)$, the distance $d(X,-X)=\|2Av(X)\|$ is controlled by coefficient $A$. Tanimoto shape similarity of amino acids provides geometric evidence that $d(X,X')>d(X,-X)$. Finally, Theorem 3.4 uses a Lipschitz assumption for conditional diffusion: $W_2(\mu_c,\mu_{c'})\le K_{\text{diff}}\|c-c'\|$, implying that similar latent codes from mirror targets yield similar binder distributions, allowing generalization without D-L training pairs.
    - **Design Motivation**: Beyond empirical results, this three-step proof (difference + stability + diffusion continuity) elevates "why zero-shot hetero-chiral generalization works" from observation to a guaranteed mechanistic explanation. It also justifies why axial vectors were chosen over tensor products—the latter do not allow such simple polar/axial decomposition.

3. **PepMirror Pipeline via AFI-EPT**:
    - **Function**: Engineering implementation of an end-to-end de novo D-peptide binder design model.
    - **Mechanism**: Using UniMoMo (VAE + latent diffusion) as the base, the EPT backbones in the VAE encoder/decoder and diffusion network are replaced with AFI-EPT. Three variants are derived: PepMirror(cross/triple/commu.). Inference uses the double-inversion logic. Downstream, 5,000 candidates are filtered via physical and geometric criteria, and 12 are synthesized for BLI affinity measurement.
    - **Design Motivation**: UniMoMo was chosen because it does not bake in rigid-body/local right-handed coordinate systems, making it one of the few backbones that can cleanly stack with AFI (frame-based models like RFDiffusion are harder to adapt). Latent diffusion is more computationally efficient and benefits from the stability guarantees of Theorem 3.4 compared to atom-level diffusion (PocketXMol).

### Loss & Training
Training data consists only of L-L homo-chiral protein-peptide complexes (same as UniMoMo). The objective is the standard VAE reconstruction plus latent diffusion denoising loss, without any D-peptide data or explicit chirality supervision—hetero-chiral capability emerges "zero-shot" via AFI and the inversion inference framework.

## Key Experimental Results

### Main Results

Evaluated on the LNR (Large Non-Redundant complex) test set for both L and D tasks using chirality accuracy and interface affinity (AutoDock Vina score).

| Model | Task | Chirality Acc (min) | Suc.% | Avg. Vina | Top Vina | IMP% |
|------|------|------|------|------|------|------|
| RFDiffusion | L | 99.57 | 99.52 | -3.30 | -5.14 | 44.09 |
| RFDiffusion | D | 99.15 | 98.58 | -1.77 | -3.78 | 16.13 |
| D-Flow | D | 98.54 | 97.52 | -3.11 | -4.54 | 22.58 |
| PepGLAD(ideal) | D | 99.04 | 95.08 | -3.26 | -5.11 | 43.01 |
| UniMoMo(all) | D | 15.70 | — | — | — | — |
| **PepMirror(cross)** | L | 99.83 | 99.67 | **-4.27** | **-5.81** | **69.89** |
| **PepMirror(cross)** | D | 99.81 | 99.76 | **-4.15** | **-5.69** | **63.44** |

PepMirror outperforms the next best model, PepGLAD(ideal), by ~20 percentage points in IMP on the D-peptide task, with the smallest "L-D gap." Performance is robust across the three axial vector variants.

### Ablation Study

| Configuration | D-Task Chirality Acc | Description |
|------|------|------|
| UniMoMo(pep.) | 23.90 | $E(3)$-equivariant baseline; outputs remain mostly L-residues despite D-target input. |
| UniMoMo(all) | 15.70 | Multimodal training version, still constrained by equivariance. |
| AFI w/ cross | 99.81 | Injected cross-product axial vectors. |
| AFI w/ triple | 99.75 | Injected triple-product axial vectors. |
| AFI w/ commu. | 99.88 | Injected commutator axial vectors (Best). |

Latent space analysis: With AFI, the median $\|c(X)-c(-X)\|$ jumps from $\sim 10^{-6}$ to $10^{-2}$ (4 orders of magnitude) but remains ~2 orders of magnitude smaller than the distance between different amino acid types ($\sim 1$), confirming the "difference + stability" conclusion. t-SNE shows 20 amino acid clusters, each containing adjacent L/D sub-clusters.

### Key Findings
- **AFI is a Necessary and Sufficient Chirality Switch**: Removing it causes PepMirror to collapse back into UniMoMo's mixed-chirality output (D-task accuracy < 25%). All three axial constructions achieve 99.7%+, with the commutator version slightly leading.
- **First Wet-Lab Confirmation**: Designed 5,000 D-peptides for CD38 (Multiple Myeloma target). After filtering, 12 were synthesized. One 10-mer D-peptide, D-1412 ("trikhytyce"), showed a $K_D\approx 10\,\mu$M, with BLI kinetics and steady-state fitting in agreement. This is the **first** AI de novo D-peptide binder validated via wet-lab.
- **Unexpected Phenomenon**: The L-enantiomer of D-1412 showed similar affinity, challenging the assumption of high stereoselectivity in peptide-protein interactions. CD spectra confirmed chirality, suggesting conformational flexibility might weaken stereoselectivity.
- **Minimized L→D Performance Drop**: While RFDiffusion's Avg Vina drops from -3.30 to -1.77 on the D task, PepMirror's drop is only ~0.1 kcal/mol, indicating it learns chirality "symmetrically."

## Highlights & Insights
- **Polar/Axial Decomposition**: Instead of complex high-order spherical harmonics, the authors used sign-reversal under $P$ as the sole discriminatory axis. This minimum-cost symmetry breaking is a plug-and-play addition for any scalarized equivariant backbone (EGNN, GVP, TFN-low-order).
- **Theoretical Closure for Zero-Shot Hetero-Chirality**: The triad of Difference (Thm 3.1) + Stability ($d(X,-X)<d(X,X')$) + Diffusion Lipschitz Continuity (Thm 3.4) transforms "why L-L training works for D-L generation" into a mechanistic explanation. This logical structure could be applied to other zero-shot equivariant generalization tasks like enantiomeric small molecule design.
- **Degradation Warnings**: The insight that $A=B=I$ and $v\cdot a=0$ causes AFI to fail highlights that "symmetric and beautiful" initializations can sometimes be detrimental.
- **Non-stereoselectivity of D-1412**: This counter-intuitive finding suggests the "perfect stereoselectivity" assumption in traditional mirror-image display may need revision, raising new questions for chiral drug evaluation frameworks.

## Limitations & Future Work
- Theoretical guarantees rely on the "genericity" of random mixing coefficients; specific initializations may negate the differences.
- Only validated on UniMoMo's latent diffusion; transferability to frame-based models (RFDiffusion) remains untested.
- Wet-lab validation is limited to one target (CD38) and 12 candidates, with only one hit. The non-stereoselectivity of D-1412 requires further investigation across more targets.
- The Lipschitz assumption for the score network in Theorem 3.4 lacks rigorous empirical validation.
- Future directions: (i) extending AFI to frame-based models; (ii) using chirality regularization to amplify $d(X,-X)$; (iii) incorporating stereoselectivity as an explicit screening objective.

## Related Work & Insights
- **vs D-Flow**: D-Flow uses the mirror-image inversion strategy but is restricted to frame-based models and was only tested in dry-labs. PepMirror breaks equivariance at the architecture level and completes wet-lab validation.
- **vs PepGLAD / UniMoMo / PocketXMol**: These $E(3)$-equivariant models generate mixed chirality due to strict equivariance. PepGLAD requires post-processing/idealization, whereas AFI solves it within the architecture without discarding geometric info.
- **vs SphereNet / ChIRo / ChiENN / GCPNet**: These works focus on chirality awareness (torsion, pseudovectors) but have not been applied to hetero-chiral peptide-protein interface generation. PepMirror is the first to achieve "de novo design + wet-lab validation."
- **vs TFN / $SE(3)$-Transformer**: High-order spherical harmonics encode chirality naturally but are computationally expensive. AFI serves as a lightweight alternative by using low-order tensor products to construct axial vectors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant polar/axial vector decomposition as a "chirality switch."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8+ baselines, latent space analysis, and complete wet-lab validation (BLI, CD spectra).
- Writing Quality: ⭐⭐⭐⭐⭐ Strong balance between theory and engineering; clear discussion of degradation cases.
- Value: ⭐⭐⭐⭐⭐ First experimentally confirmed AI de novo D-peptide design framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](../../ICLR2026/computational_biology/learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICML 2026\] Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators](protein_language_model_embeddings_improve_generalization_of_implicit_transfer_op.md)
- [\[ICML 2026\] Protein Circuit Tracing via Cross-layer Transcoders](protein_circuit_tracing_via_cross-layer_transcoders.md)
- [\[ICML 2026\] iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiome Diagnosis](ilora_bayesian_low-rank_adaptation_with_latent_interaction_graphs_for_microbiome.md)

</div>

<!-- RELATED:END -->
