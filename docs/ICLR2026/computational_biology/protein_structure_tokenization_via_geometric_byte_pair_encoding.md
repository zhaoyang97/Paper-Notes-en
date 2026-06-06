---
title: >-
  [Paper Note] Protein Structure Tokenization via Geometric Byte Pair Encoding
description: >-
  [ICLR 2026][Computational Biology][GeoBPE] This paper proposes GeoBPE — the first tokenizer to extend Byte Pair Encoding (BPE) from discrete text to continuous protein backbone geometry. By alternating between local merg…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "GeoBPE"
  - "Protein Structure Tokenizer"
  - "Hierarchical Vocabulary"
  - "Differentiable IK"
  - "Multi-resolution"
date: 2026-05-08
content_hash: dc81f9cde781bb5c
---

# Protein Structure Tokenization via Geometric Byte Pair Encoding

**Conference**: ICLR 2026
**arXiv**: [2511.11758](https://arxiv.org/abs/2511.11758)  
**Code**: [GitHub](https://github.com/shiningsunnyday/PT-BPE)  
**Area**: Protein AI / Structure Tokenization
**Keywords**: GeoBPE, Protein Structure Tokenizer, Hierarchical Vocabulary, Differentiable IK, Multi-resolution

## TL;DR

This paper proposes GeoBPE — the first tokenizer to extend Byte Pair Encoding (BPE) from discrete text to continuous protein backbone geometry. By alternating between local merging (k-medoids clustering + quantization) and global correction (differentiable inverse kinematics), GeoBPE constructs a hierarchical structural motif vocabulary that achieves >10× compression ratio and >10× data efficiency over VQ-VAE-based protein structure tokenizers (PSTs), ranking first across 24 test sets spanning 12 downstream tasks.

## Background & Motivation

**Background**: Protein language models (PLMs) such as the ESM series have achieved remarkable success on sequences, but do not explicitly model folding geometry, limiting their performance on structure-dependent functional tasks. Protein structure tokenizers (PSTs) serve as the critical bridge between sequence and structure; the dominant paradigm relies on VQ-VAE methods (e.g., ESM3, FoldSeek's 3Di alphabet, ProToken) that train autoencoders to map continuous structures to discrete codebooks.

**Limitations of Prior Work**: VQ-VAE-based PSTs suffer from three fundamental limitations: (1) **fixed codebooks are prone to collapse** — codebook collapse leads to imbalanced token utilization and limited compression efficiency; (2) **vector tokens are uninterpretable** — real-valued codebook entries lack the compositional hierarchy exhibited by BPE subwords; (3) **fixed token granularity hinders multi-scale analysis** — all tokens span a fixed number of residues, precluding adaptation to naturally variable-length functional domains. VQ-VAE methods also exhibit poor OOD generalization, with test/train RMSD ratios as high as 6.4×.

**Key Challenge**: Protein structures are continuous, noisy, and multi-scale, whereas BPE was designed for discrete symbols. The central challenge is maintaining global geometric consistency during discretization — local quantization introduces cumulative drift that displaces distal atomic coordinates.

**Key Insight**: Protein folds are composed of modular substructures (helices, sheets, loops, etc.), naturally amenable to BPE's strategy of iteratively merging frequent pairs. A key observation is that quantization-induced drift can be compensated by optimizing the "glue angles" at boundaries, since each boundary provides three degrees of freedom (one bond angle and two dihedral angles).

**Core Idea**: Extend BPE from discrete symbols to continuous geometry through alternating iterations of local k-medoids quantization and global differentiable inverse kinematics drift correction, building a hierarchical motif vocabulary for protein structures.

## Method

### Overall Architecture

GeoBPE takes as input the internal coordinate representation of protein backbones (bond lengths, bond angles, dihedral angles). Starting from residue-level initialization, the algorithm alternates four steps: (1) pop the most frequent Geo-Pair (adjacent motif pair); (2) run k-medoids clustering in RMSD space to obtain representative prototypes; (3) quantize all occurrences to their nearest prototype (hard-replacing internal parameters); (4) optimize boundary glue angles via differentiable inverse kinematics to compensate quantization drift. Each iteration adds a new motif type to the vocabulary and updates the segmentation and merge tree. The final output consists of a hierarchical vocabulary $\mathcal{V}$, per-backbone segmentations $\mathcal{P}$, and a merge hierarchy $\mathcal{F}$.

### Key Designs

1. **Backbone Geometry Representation and Geo-Pair Definition**

    - Function: Represent 3D protein backbones as a manipulable internal coordinate system.
    - Mechanism: Each residue $i$ is represented in bond-residue form — three atomic coordinates $(N_i, CA_i, C_i)$ correspond to bond lengths $\ell^{N-CA}_i, \ell^{CA-C}_i, \ell^{C-N}_i$, bond angles $\theta^{NCAC}_i, \theta^{CACN}_i$, and dihedral angle $\psi_i$. Adjacent residues are linked by glue parameters $\Gamma_i = \{\theta^{CNCA}_i, \phi_i, \omega_i\}$. A contiguous segment $\mathcal{M}_{p:q}$ forms a motif, and a pair of adjacent motifs together with boundary glue angles constitutes a Geo-Pair occurrence. A canonical hashable key maps all occurrences to discrete identifiers, making BPE's frequency-counting operation tractable.
    - Design Motivation: Internal coordinates are SE(3)-invariant, eliminating the influence of global rotation and translation. They also naturally decompose the backbone into local parameters (intra-motif) and connection parameters (glue angles), enabling the separation of merge, quantize, and correct operations.

2. **K-medoids Clustering and Adaptive Quantization**

    - Function: Extract representative structural prototypes (vocabulary entries) for each Geo-Pair type.
    - Mechanism: For all occurrences of the most frequent Geo-Pair key, k-medoids clustering is run in RMSD metric space to obtain $K$ prototypes (medoids are actual observed fragments, ensuring structural interpretability). The quantization step hard-copies all internal parameters of each non-medoid occurrence from its assigned medoid. A key innovation is **adaptive multi-resolution**: quantization is lossy, but each step re-quantizes from the original fragments (rather than accumulating on already-quantized results), and $K$ can be adjusted by motif length — coarse granularity for short motifs, fine granularity for long ones — enabling precise control of the compression-reconstruction trade-off.
    - Design Motivation: BPE directly substitutes discrete symbols, but "substitution" for continuous data is quantization — requiring identification of the best representative among possible conformational variants. Choosing the medoid (rather than the centroid) guarantees that each prototype is a physically plausible protein fragment.

3. **Differentiable Inverse Kinematics (IK) Drift Correction**

    - Function: Compensate for global geometric drift introduced by quantization.
    - Mechanism: Quantization replaces the occurrence's internal transform $T^{\text{occ}}_u$ with the medoid's $T^{\text{med}}_u$, inducing drift $\Delta T_u = T^{\text{occ}}_u (T^{\text{med}}_u)^{-1}$. To compensate, boundary glue angles are optimized so that the new link transform satisfies $G^{\text{new}}_{i_{u}-1} \approx G^{\text{orig}}_{i_{u}-1} \cdot \Delta T_u$. Concretely, this minimizes the end-frame loss $\mathcal{L}_u(\Gamma) = w_R \|\log(\hat{R}^\top R^*)\|^2 + w_t \|\hat{t} - t^*\|^2$, where $\hat{F}$ is computed by forward kinematics and $F^*$ is the original frame. In practice, a global batch optimization simultaneously optimizes all glue angle degrees of freedom along the backbone, maximizing drift compensation flexibility.
    - Design Motivation: This is the core technical contribution that distinguishes GeoBPE from naive discretization. Without drift correction, quantization errors accumulate along the chain, causing RMSD at distal atoms to diverge. The three glue degrees of freedom at each boundary provide exactly the parametric space needed for directional compensation.

### Hierarchical Inductive Bias Propagation

The merge hierarchy $\mathcal{F}$ output by GeoBPE can serve as a recursive computation tree: leaf nodes are initialized with pretrained PLM features (e.g., ESM3), aggregated upward along parent-child relations to motif/protein-level representations, and then propagated back down to residue level. This endows residue-level embeddings with hierarchical structural awareness, supporting downstream tasks such as functional site prediction and fold classification.

## Key Experimental Results

### Main Results (Downstream Transfer Performance, AUROC%)

| Task | ProteinMPNN | MIF | FoldSeek | ProTokens | ESM3 | VQ-VAE | AminoASeed | **GeoBPE-Transfer** |
|------|-------------|-----|----------|-----------|------|--------|------------|-------------------|
| BindInt-Fold | 51.83 | 50.38 | 53.18 | 44.66 | 44.30 | 47.25 | 47.11 | **59.19 (+33.6%)** |
| BindBio-Fold | 78.42 | 85.79 | 32.37 | 58.47 | 62.84 | 62.02 | 65.73 | **94.94 (+51.1%)** |
| CatBio-Fold | 82.49 | 85.85 | 56.33 | 67.68 | 65.33 | 67.58 | 65.95 | **95.01 (+45.4%)** |
| Con-SupFam | 84.68 | 92.66 | 51.31 | 70.64 | 80.53 | 74.60 | 86.60 | **84.84 (+5.4%)** |
| Mean AUROC% | 75.92 | 79.82 | 51.90 | 65.37 | 69.24 | 68.30 | 72.43 | **80.20 (+18.1%)** |

GeoBPE-Transfer ranks first on average across 24 test sets spanning 12 tasks, achieving a +18.13% gain over ESM3 on functional site prediction.

### Ablation Study (Compression–Reconstruction Trade-off & OOD Generalization)

| Tokenizer | BPR (bits/res) | Test RMSD (Å) | Test/Train RMSD Ratio | Training Data |
|-----------|---------------|---------------|----------------------|---------------|
| ESM3 | ~30× GeoBPE | Low | ~1.0 | 236M structures |
| ProToken | 1× (reference) | Reference | ~1.0 | ~700K |
| VQ-VAE | Moderate | Moderate | **6.4×** | ~48K |
| **GeoBPE** | **0.27–0.36× ProToken** | Moderate | **1.16–1.28** | **~48K** |

GeoBPE's BPR is only 27–36% of ProToken's (>10× compression advantage) and 1.6–2.1% of ESM3's (>50×), while achieving strong OOD generalization (test/train RMSD ratio 1.16–1.28 vs. 6.4× for VQ-VAE). GeoBPE sustains its performance using only 1% of training data, demonstrating >10× data efficiency.

### Key Findings

- **Hierarchical vocabulary reverses the trend of "discretization harming downstream performance"**: Discrete VQ-VAE PSTs typically underperform continuous PSTs due to quantization loss, whereas GeoBPE's hierarchical inductive bias enables it to surpass all continuous and discrete baselines.
- **GeoBPE tokens align with CATH functional families**: Token boundaries strongly correlate with protein domain annotations, supporting expert-interpretable case studies.
- **Vocabulary scales smoothly along the Pareto frontier**: Increasing vocabulary size (600→21000) yields smooth movement along the BPR-distortion curve — an elasticity absent from VQ-VAE approaches with fixed codebook dimensions.
- **Language model generation**: Combined with a ~7.3M-parameter Transformer for unconditional backbone generation, GeoBPE produces backbones that are 99% unique and designable, with scTM scores up to 49% higher than VQ-VAE.

## Highlights & Insights

- **Elegant extension of BPE to continuous geometry**: The two central challenges — "how to count frequencies over continuous data" and "how to maintain global consistency after quantization" — are addressed by canonical hashing (discretizing keys) and differentiable IK (correcting glue angles), respectively; the combination of these two mechanisms is particularly elegant.
- **Medoids as prototypes guarantee physical plausibility**: Unlike the abstract vectors learned by VQ-VAE, medoids are genuinely observed structural fragments, giving each token an unambiguous physical meaning that supports expert-level interpretation.
- **Dual value of the hierarchical vocabulary**: It simultaneously provides multi-resolution control over the compression–reconstruction trade-off (as a tokenizer) and serves as an inductive bias via a recursive computation tree that improves downstream representation quality — an unexpected but significant benefit.

## Limitations & Future Work

- **Computational cost**: Iterative k-medoids clustering and global IK optimization are computationally intensive; although mitigated by capping the number of sampled occurrences ($M_{\max}=5000$), overhead remains a barrier to large-scale application.
- **Backbone-only modeling**: The current method covers only N-CA-C backbone atoms and does not model side-chain conformations, limiting its applicability to side-chain-dependent functional tasks.
- **Limited language model generation quality**: The Transformer used for generation contains only 7.3M parameters, leaving substantial room for improvement in generation quality.
- **Comparison with end-to-end training is not entirely fair**: GeoBPE-Transfer relies on ESM3's pretrained features, and a portion of the downstream performance gains is attributable to ESM3's powerful representations.

## Related Work & Insights

- **vs. ESM3 VQ-VAE**: ESM3 trains a VQ-VAE on 236M structures to achieve low reconstruction error; GeoBPE attains comparable performance using less than 0.02% of that data, at the cost of slightly higher RMSD, but with far superior generalization and interpretability.
- **vs. FoldSeek 3Di**: 3Di employs 20 fixed discrete codes for efficient search, whereas GeoBPE's vocabulary can be dynamically expanded and supports multi-resolution analysis.
- **vs. ProToken**: ProToken and GeoBPE are complementary on the Pareto frontier, but GeoBPE's interpretability (CATH alignment) and data efficiency are its distinctive advantages.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to extend BPE principles to continuous protein geometry, with original contributions in both theory and algorithm design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ten research questions covering compression, reconstruction, generalization, efficiency, downstream performance, and interpretability, evaluated on 24 test sets with highly comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Mathematical notation is rigorous and algorithmic descriptions are clear, though the notation system is complex with a steep learning curve.
- Value: ⭐⭐⭐⭐⭐ Opens an entirely new paradigm for protein structure representation learning; the hierarchical vocabulary concept offers broader inspiration for other continuous structural data domains (molecules, materials).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](../../NeurIPS2025/computational_biology/towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](../../ICML2026/computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [\[ICML 2026\] DNAChunker: Learnable Tokenization for DNA Language Models](../../ICML2026/computational_biology/dnachunker_learnable_tokenization_for_dna_language_models.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](../../ICML2026/computational_biology/learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)

</div>

<!-- RELATED:END -->
