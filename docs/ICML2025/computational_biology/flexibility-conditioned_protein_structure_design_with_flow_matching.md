---
title: >-
  [Paper Note] Flexibility-conditioned Protein Structure Design with Flow Matching
description: >-
  [ICML 2025][Computational Biology][Protein Structure Generation] BackFlip (predicting residue-level flexibility from backbones) and FliPS (an SE(3)-equivariant flow matching model conditioned on flexibility profiles) are proposed, achieving the first flexibility-controlled generation of protein backbones with desired dynamic properties, validated by 300 ns molecular dynamics simulations.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Protein Structure Generation"
  - "Flexibility Conditioning"
  - "Flow Matching"
  - "SE(3)-Equivariance"
  - "Molecular Dynamics"
date: 2026-05-08
content_hash: 6b8232060a094f80
---

# Flexibility-conditioned Protein Structure Design with Flow Matching

**Conference**: ICML 2025  
**arXiv**: [2508.18211](https://arxiv.org/abs/2508.18211)  
**Code**: [graeter-group/flips](https://github.com/graeter-group/flips)  
**Area**: Medical Imaging / Protein Design  
**Keywords**: Protein Structure Generation, Flexibility Conditioning, Flow Matching, SE(3)-Equivariance, Molecular Dynamics

## TL;DR

BackFlip (predicting residue-level flexibility from backbones) and FliPS (an SE(3)-equivariant flow matching model conditioned on flexibility profiles) are proposed, achieving the first flexibility-controlled generation of protein backbones with desired dynamic properties, validated by 300 ns molecular dynamics simulations.

## Background & Motivation

Current deep-learning-based protein design methods (e.g., RFdiffusion, FrameFlow) generate high-quality protein backbones, but the generated structures often exhibit high thermal stability and structural rigidity. However, protein functions (catalysis, molecular recognition, allosteric regulation, etc.) are highly dependent on dynamic behaviors and local flexibility. Existing methods can only condition generation on static properties (motifs, symmetry, binding targets), failing to control the flexibility distribution of the generated structures.

**Core Problem**: How can generative models be enabled to "understand" and control the dynamic properties of proteins?

**Motivation Sources**:
- Enzymes require dynamic behaviors such as loop opening/closing and domain motion during catalytic cycles.
- Protein binders require local structural flexibility to bind with DNA/RNA/ligands.
- Existing generative models tend to produce excessively rigid structures, limiting the exploration of functional space.
- Flexibility metrics (B-factor, pLDDT) have their respective limitations, necessitating more reliable indicators.

## Method

### Overall Architecture

This paper proposes a two-stage framework:

1. **BackFlip** (Backbone Flexibility Predictor): Predicts the flexibility (local RMSF) of each residue directly from the protein backbone structure without relying on sequence information.
2. **FliPS** (Flexibility-conditioned Protein Structure generation): An SE(3)-equivariant conditional flow matching model conditioned on a target flexibility profile.

Overall pipeline:
1. Given a target flexibility profile $\xi$.
2. Generate multiple candidate backbones conditioned on the profile using FliPS.
3. Predict the flexibility profile of each candidate backbone using BackFlip.
4. Select the best-matching candidate via BackFlip Screening (BFS).
5. (Optional) Validate using molecular dynamics (MD) simulations.

### Key Designs

#### 1. Local RMSF — A New Flexibility Metric

Traditional RMSF relies on global alignment, which introduces ambiguities and artifacts due to non-local influences. This paper proposes **Local RMSF**, which computes residue fluctuations after performing local alignment within a local neighborhood (window size $S=12$):

$$\xi_i = \sqrt{\frac{1}{|\mathcal{C}|}\sum_{x \in \mathcal{C}} \left|(T_{\text{Align}}^{(S)} \circ x)_i - x_i^{(\text{ref})}\right|^2}$$

where $T_{\text{Align}}^{(S)}$ is the local rigid alignment transformation on the sequence neighborhood window of residue $i$. To eliminate selection bias of the reference conformation, the median across $N_{\text{ref}}=10$ randomly chosen reference conformations is taken.

#### 2. BackFlip Architecture

- An SE(3)-equivariant Transformer based on the Clifford Frame Attention (CFA) within the GAFL architecture.
- Backbones are represented as a sequence of rigid frames in $\text{SE}(3)^N$.
- Uses $n_{\text{cfa}}=4$ CFA layers to extract node features.
- Node features are mapped to $[0, \xi_{\max}]$ ($\xi_{\max}=5$ Å) via an MLP + scaled sigmoid.
- Consisting of only **0.68M parameters** (compared to 16.7M in GAFL), with node and edge embedding dimensions of 64 and 32, respectively.
- **Key Feature**: Pure backbone input, requiring no sequences, evolutionary information, or pretrained language models.

$$\xi_i = \xi_{\max} \cdot \sigma(\text{MLP}(h_i))$$

#### 3. FliPS Conditional Generative Model

FliPS introduces three key modifications on top of GAFL (an unconditional flow matching model):

**a) Flexibility Embedding**:
- Discretizes the flexibility value of each residue into 8 bins (maximum value $\xi_{\max}=3$ Å).
- Passed into the model as an additional node input feature.

**b) Flexibility Auxiliary Loss**:
- Leverages the differentiability of BackFlip to compute the flexibility prediction $\xi_{\text{BF}}(\hat{T}_1)$ for the predicted backbone structure $\hat{T}_1$.
- Appends a flexibility MSE penalty to the original FrameDiff auxiliary loss:

$$l_{\text{aux}} = l_{\text{aux, FD}} + \frac{\lambda_{\text{flex}}}{N} \|\xi_1 - \xi_{\text{BF}}(\hat{T}_1)\|^2$$

where $\lambda_{\text{flex}} = 100$. This design is feasible because BackFlip is differentiable with respect to the backbone structure and independent of the sequence.

**c) Flexibility Masking**:
- Randomly masks part or all of the flexibility profile during training to prevent memorization and preserve unconditional generation capabilities.
- Similar to the classifier-free guidance drop-out strategy.

#### 4. BackFlip Guidance (BG) — A Training-Free Alternative

Inspired by classifier guidance, BackFlip gradients are added to the vector field of the unconditional model during inference:

$$\hat{v}_{\text{cond}}(T_t, t, \xi) = \hat{v}(T_t, t) - \eta \nabla_{T_t} \|\xi - \xi_{\text{BF}}(T_t)\|^2$$

where $\eta$ is the guidance scale hyperparameter. Ablation studies show that BG performs worse than the directly trained conditional model, FliPS.

#### 5. BackFlip Screening (BFS)

Selects the best candidate post-generation using BackFlip:

$$s(\xi, \xi_{\text{ref}}) = w_{\text{corr}} \cdot r(\xi, \xi_{\text{ref}}) - w_{\text{mae}} \cdot \text{MAE}(\xi, \xi_{\text{ref}})$$

The weights are set to $w_{\text{corr}}=1$ and $w_{\text{mae}}=2$ to jointly account for differences in both the shape (Pearson correlation) and magnitude (MAE) of the flexibility profiles.

### Loss & Training

**BackFlip Training**:
- Dataset: ATLAS (1,294 proteins, 300 ns MD simulations)
- Loss: MSE of per-residue local RMSF
- Split: 1,035 train / 130 val / 129 test

**FliPS Training**:
- Dataset: PDB dataset (22,977 proteins, 60–512 residues, with flexibility annotated using BackFlip)
- Loss: Flow matching vector field regression + auxiliary loss (including the flexibility term)
- Training compute: 8 × NVIDIA A100, totaling 21 GPU days
- $\lambda_{\text{flex}} = 100$, flexibility embedding with 8 bins

## Key Experimental Results

### Main Results

**BackFlip Flexibility Prediction Performance**

| Method | ATLAS Test Global $r$ ↑ | ATLAS MAE [Å] ↓ | De novo $r$ ↑ | De novo MAE [Å] ↓ | Inference Time [s] ↓ |
|------|------------------------|------------------|-------------|-------------------|---------------|
| MD (Ground Truth) | 0.84 | 0.14 | 0.80 | 0.10 | ~10,000 |
| B-factor | 0.16 | - | - | - | - |
| Negative pLDDT | 0.54 | - | 0.48 | - | 118 |
| **BackFlip** | **0.80** | **0.17** | **0.73** | **0.11** | **0.6** |

BackFlip achieves a Pearson correlation coefficient of 0.80 on the ATLAS test set, close to the noise upper bound of the MD simulation itself (0.84); its inference speed is approximately 200 times faster than pLDDT, and about 17,000 times faster than MD simulation.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| FliPS (Full Model) | Best flexibility matching | Conditional flow matching + flexibility auxiliary loss + BFS |
| BackFlip Guidance (BG) + FrameFlow | Poor | Training-free guidance performs worse than direct conditioning |
| Without BackFlip Screening (BFS) | Flexibility matching performance drops | Screening is crucial for the final quality |
| Without flexibility auxiliary loss | Weakened condition alignment | Auxiliary loss significantly improves compliance with flexibility conditions |
| SCOPe natural protein screening | Limited matching | Natural protein libraries struggle to cover arbitrary flexibility profiles |
| FoldFlow2 + BFS | Inferior to FliPS | Unconditional model + screening is inferior to the conditional model |
| RFdiffusion + BFS | Inferior to FliPS | Same as above |

### Key Findings

1. **BackFlip significantly outperforms existing flexibility proxy metrics**: Pearson correlation is only 0.16 for B-factor and 0.54 for pLDDT, while BackFlip achieves 0.80.
2. **FliPS-generated backbones are validated via 300 ns MD**: Confirming that the generated structures indeed exhibit the targeted flexibility profiles.
3. **Conditioning > Guidance > Screening**: Directly training a conditional model outperforms test-time guidance, and both methods surpass screening from unconditional samples.
4. **Retention of structural diversity**: FliPS generates structurally diverse protein backbones while satisfying the given flexibility conditions.
5. **Loops/turns are correctly identified as the most flexible regions**, whereas core $\alpha$-helix and $\beta$-sheet regions are the most rigid.
6. **Generalization to de novo proteins**: BackFlip maintains a high correlation of 0.73 on non-natural proteins.

## Highlights & Insights

1. **Pioneering Nature**: The first protein backbone generation model conditioned on structural flexibility, bridging the gap between "static conditioning" and "dynamic conditioning."
2. **Introduction of Local RMSF**: Resolves the non-locality issue of global alignment in traditional RMSF, serving as a simple yet highly valuable methodological contribution.
3. **Elegance of the Two-Stage Collaborative Design**: BackFlip serves as a standalone flexibility prediction tool, is embedded into the training loss of FliPS via differentiability, and is utilized for post-generation screening—one model serving three roles.
4. **Design Choice of Pure Backbone Input**: BackFlip is sequence-independent, naturally aligning with de novo design workflows (generating backbones first, then designing sequences).
5. **Extremely Compact Parameter Size**: BackFlip achieves prediction accuracy close to MD with only 0.68M parameters, demonstrating the powerful inductive bias of geometric priors.
6. **Practical Speedup**: Compared to MD simulations (~10,000 seconds), BackFlip takes only 0.6 seconds, enabling high-throughput screening.

## Limitations & Future Work

1. **Flexibility Annotations Rely on BackFlip Rather Than Ground-Truth MD**: The flexibility labels for FliPS training data originate from BackFlip predictions rather than explicit MD simulations, propagating prediction errors.
2. **Protein Length Limitations**: The training data primarily covers lengths of 60–512 residues; generalization to larger proteins remains unverified.
3. **Purely Backbone-Level Consideration**: Side-chain flexibility and co-design of sequence-flexibility are not yet incorporated.
4. **Fixed Window Size $S=12$ in Local RMSF**: Different scales of flexibility features may require adaptive windows.
5. **High Cost of MD Validation**: Final validation still requires 300 ns MD simulations, limiting the closed-loop iteration speed for large-scale applications.
6. **Lack of Functional Validation**: Whether the generated flexible proteins truly possess catalytic or binding functions has not yet been experimentally verified.
7. **Limited Effectiveness of BackFlip Guidance (BG)**: Indicates that the classifier-guidance paradigm might be less effective than direct conditioning in protein structure generation.

## Related Work & Insights

- **GAFL / FrameFlow** (Wagner et al., 2024; Yim et al., 2023): The base architecture of FliPS, leveraging geometric algebra and SE(3)-equivariant flow matching.
- **RFdiffusion** (Watson et al., 2023): The current mainstream protein backbone generation model, which only supports static conditioning.
- **FlexPert-3D** (Kouba et al., 2024): A flexibility predictor relying on pretrained protein language models; BackFlip achieves equivalent or superior performance without using pLMs.
- **AlphaFold2** (Jumper et al., 2021): Origin of the IPA mechanism and the pLDDT flexibility proxy.
- **Classifier Guidance** (Dhariwal & Nichol, 2021): The inspiration source for BackFlip Guidance.
- **Insights**: The paradigm of "training a predictor first $\rightarrow$ driving generation via the predictor's differentiability" can be extended to conditioning on other dynamic properties (e.g., allosteric motion, binding free energy, etc.).

## Rating

| Dimension | Score (1-10) | Description |
|------|-----------|------|
| Novelty | 9 | First flexibility-conditioned protein generation model; the Local RMSF metric is also a new contribution. |
| Technical Depth | 8 | SE(3)-equivariant flow matching + differentiable predictor embedded in the loss, mathematically rigorous. |
| Experimental Thoroughness | 8 | Covers natural/de novo proteins, multiple baselines, ablations, and MD validation. |
| Value | 7 | Addresses a real need in protein design, but lacks biological wet-lab functional validation. |
| Writing Quality | 8 | Clear structure, beautiful charts, and well-articulated motivation. |
| **Overall** | **8** | Pioneering work; the method is elegant and thoroughly validated by MD simulations. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[ICLR 2026\] Riemannian Variational Flow Matching for Material and Protein Design](../../ICLR2026/computational_biology/riemannian_variational_flow_matching_for_material_and_protein_design.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/computational_biology/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[ICLR 2026\] Interpolation-Based Conditioning of Flow Matching Models for Bioisosteric Ligand Design](../../ICLR2026/computational_biology/interpolation-based_conditioning_of_flow_matching_models_for_bioisosteric_ligand.md)

</div>

<!-- RELATED:END -->
