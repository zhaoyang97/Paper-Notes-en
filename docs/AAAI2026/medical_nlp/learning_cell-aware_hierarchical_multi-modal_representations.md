---
title: >-
  [Paper Note] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling
description: >-
  [AAAI 2026 Oral][Medical LLM][Molecular property prediction] This paper proposes the CHMR framework, which addresses missing biological modalities via structure-aware propagation, and introduces Tree-VQ to model hierarchical dependencies among molecules, cells, and genes. Evaluated on 728 tasks across 9 benchmarks, CHMR achieves a 3.6% improvement in classification and 17.2% in regression, enabling robust cell-aware molecular representation learning.
tags:
  - "AAAI 2026 Oral"
  - "Medical LLM"
  - "Molecular property prediction"
  - "cell-aware representation"
  - "hierarchical vector quantization"
  - "missing modality"
  - "multi-modal fusion"
date: 2026-05-08
content_hash: 9a43cef5a87131d6
---

# Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.21120](https://arxiv.org/abs/2511.21120)  
**Code**: [https://github.com/limengran98/CHMR](https://github.com/limengran98/CHMR)  
**Area**: Medical Imaging / Multi-Modal Learning
**Keywords**: Molecular property prediction, cell-aware representation, hierarchical vector quantization, missing modality, multi-modal fusion

## TL;DR

This paper proposes the CHMR framework, which addresses missing biological modalities via structure-aware propagation, and introduces Tree-VQ to model hierarchical dependencies among molecules, cells, and genes. Evaluated on 728 tasks across 9 benchmarks, CHMR achieves a 3.6% improvement in classification and 17.2% in regression, enabling robust cell-aware molecular representation learning.

## Background & Motivation

**Background**: Molecular property prediction (activity, toxicity, side effects, etc.) is a core task in drug discovery. Deep learning methods have advanced rapidly, falling into two main categories: (1) unimodal methods that exploit intrinsic molecular features (atomic attributes, chemical bonds, 3D structures); and (2) multi-modal methods that combine molecular structure with external biological responses (cell morphology, gene expression) to capture signaling cascade effects at the cellular level.

**Limitations of Prior Work**: (1) **Pervasive modality incompleteness** — molecular structure data are typically complete, but associated cell phenotype or gene expression data are frequently missing due to experimental constraints and cost limitations (over 90% of molecules lack some biological modality), with distinct missing patterns across molecules (some lack morphological data, others lack transcriptomic data); (2) **Insufficient hierarchical dependency modeling** — molecular perturbations trigger cascade reactions across biological levels (chemical structure → cellular processes → gene expression), yet existing methods perform instance-level alignment in flat latent spaces, failing to capture multi-hop semantic relationships and cross-level dependencies.

**Key Challenge**: Missing biological modalities cause distribution shift and modality imbalance, making naive imputation (zero vectors, mean values) ineffective. Meanwhile, flat alignment discards hierarchical structural information, limiting the ability to model cross-scale biological mechanisms.

**Goal**: (1) How to robustly learn molecular representations under severe modality missingness? (2) How to explicitly model hierarchical dependencies among molecules, cells, and genes?

**Key Insight**: The authors observe that structurally similar molecules exhibit similar biological responses, motivating graph propagation for missing modality enhancement. Additionally, biological responses naturally follow a hierarchical structure from molecule → cell → gene, which can be explicitly encoded via a tree structure.

**Core Idea**: Structure-aware graph propagation is used to impute missing biological modalities, while Tree-VQ encodes cross-scale hierarchical semantics, achieving robust molecular representation learning under missing modality conditions.

## Method

### Overall Architecture

CHMR consists of four core modules: (1) **Modality Augmentation (MA)** — iterative propagation over a molecular structural similarity graph to impute missing biological modalities; (2) **Semantic Consistency Alignment (SCA)** — aligning representations of molecular and cellular modalities at both sample and distribution levels; (3) **Tree-based Vector Quantization (Tree-VQ)** — capturing hierarchical semantic dependencies across modalities via a shared binary tree; (4) **Contextual Propagation Reconstruction (CPR)** — cross-modal contextual supervision via random walks on a biological prior knowledge graph. All four modules are jointly optimized.

### Key Designs

1. **Modality Augmentation and Semantic Consistency Alignment (MA + SCA)**:

    - Function: Generate plausible pseudo-features for missing external biological modalities and ensure semantic consistency between augmented and original features.
    - Mechanism: **MA**: A molecular similarity matrix $\mathbf{W}$ is constructed, retaining top-K neighbors per molecule, with iterative propagation over missing modalities $\mathbf{x}_i^{c,(T)} = \sum_{j \in \mathcal{N}_K(v_i)} \mathbf{W}_{ij} \mathbf{x}_j^{c,(T-1)}$ (observed modalities remain unchanged). **SCA**: Two-level alignment — at the sample level, an InfoNCE contrastive loss $\mathcal{L}_{IA}$ aligns molecular anchor features with cell features; at the distribution level, a VICReg loss $\mathcal{L}_{DA}$ ensures augmented features do not deviate from the original distribution.
    - Design Motivation: Naive zero or random imputation leads to 5.3% and 4.5% performance degradation, respectively, whereas neighbor-based propagation leverages the prior that structural similarity implies similar biological responses. Variance and covariance regularization in VICReg effectively prevents distribution shift in augmented features.

2. **Tree-based Vector Quantization (Tree-VQ)**:

    - Function: Map multi-modal features onto a shared binary tree, using tree levels to encode biological hierarchy (shallow levels correspond to molecular fingerprints; deeper levels to cell phenotypes and gene expression).
    - Mechanism: A binary tree $\mathcal{T} = \bigcup_{h=1}^{H} \mathcal{E}^h$ of depth $H$ is constructed, with $2^h$ node embeddings per level. For the projected feature $\mathbf{p}^{\xi}$ of each modality, routing proceeds from the root, selecting the nearest child at each level via cosine distance $j^{*\xi,h} = \arg\min_j \tilde{\delta}_j^{\xi,h}$. Routing is constrained to the two children of the parent node. A symmetric VQ loss $\mathcal{A}(\mathbf{p}^\xi, \mathbf{q}^{\xi,h}) = 1 - \cos(sg[\mathbf{q}^{\xi,h}], \mathbf{p}^\xi) + \eta(1 - \cos(\mathbf{q}^{\xi,h}, sg[\mathbf{p}^\xi]))$ is used for bidirectional alignment.
    - Design Motivation: Compared to flat VQ, the tree structure naturally corresponds to biological hierarchy (molecule → cell → gene). Sharing a single tree across modalities enables heterogeneous features to be jointly routed into a unified semantic hierarchy. Ablations show that removing Tree-VQ causes a 3.9% drop, and replacing it with flat VQ still results in a 2.0% drop.

3. **Contextual Propagation Reconstruction (CPR)**:

    - Function: Provide cross-modal contextual supervision via random walks on a biological prior knowledge graph.
    - Mechanism: A context graph $\mathcal{H}$ is constructed from known molecule–biological response associations (e.g., drug–target pairs, functional associations, shared regulatory pathways). Random walks of length $L$ are performed from each node, accumulating propagation weights along paths. Molecular and biological modality features are reconstructed via decoders, with reconstruction loss weighted by random walk weights: $\mathcal{L}_{CPR} = -\frac{1}{|\mathcal{V}|} \sum_{i} \sum_{l=0}^{L} \beta_{i,l} \mathcal{D}(\hat{\mathbf{x}}_{u_{i_l}}, \mathbf{x}_{u_{i_l}})$.
    - Design Motivation: In the absence of explicit supervision, biological prior knowledge graphs provide additional structured supervisory signals, and random walks enable long-range associations to be exploited.

### Loss & Training

- Overall pre-training loss: $\mathcal{L}_{total} = \mathcal{L}_{CPR} + \lambda_1 \mathcal{L}_{SCA} + \lambda_2 \mathcal{L}_{TreeVQ}$
- Optimal hyperparameters: $\lambda_1 = 10$, $\lambda_2 \in \{0.1, 1\}$, $\eta = 1$, tree depth $h = 6$
- During downstream evaluation, the pre-trained backbone is frozen and only a lightweight prediction head is trained.

## Key Experimental Results

### Main Results

| Dataset | Metric | CHMR (Ours) | InfoAlign (SOTA) | Gain |
|--------|------|-----------|-----------------|------|
| ChEMBL (41 tasks) | AUC% ↑ | **84.7±0.2** | 81.3±0.6 | +3.4 |
| ToxCast (617 tasks) | AUC% ↑ | **69.3±0.3** | 66.4±1.1 | +2.9 |
| Broad (32 tasks) | AUC% ↑ | **71.4±0.2** | 70.0±0.1 | +1.4 |
| Biogen (6 tasks) | MAE×100 ↓ | **40.9±0.3** | 49.4±0.2 | -17.2% |

Biogen sub-metrics: HLM 33.7 (vs 39.7), RLM 39.8 (vs 48.4), ER 35.2 (vs 39.2), Solubility 34.9 (vs 40.5), hPPB 53.1 (vs 66.7), rPPB 48.5 (vs 62.0)

### Ablation Study

| Configuration | ChEMBL ↑ | Biogen ↓ | Δ(%) |
|------|----------|----------|------|
| Full Model | 84.7 | 40.9 | - |
| Zero imputation (w/o MA) | 81.6 | 44.8 | -5.3 |
| w/o SCA | 82.4 | 43.1 | -3.6 |
| w/o Tree-VQ | 82.3 | 43.4 | -3.9 |
| Flat VQ (replacing Tree-VQ) | 83.2 | 42.0 | -2.0 |
| w/o CPR | 82.6 | 43.0 | -3.5 |
| Mol-Only (molecular modality only) | 81.5 | 44.3 | -4.9 |
| InfoAlign (SOTA baseline) | 81.3 | 49.4 | -7.7 |

### Key Findings

- **Tree-VQ is the key innovation**: its removal leads to a 3.9% drop, and replacing it with flat VQ still yields a 2.0% drop, indicating that the hierarchical structure — not simple quantization — is the driving factor.
- **Synergistic effect of all four modules**: removing any individual module causes a 2–5% performance drop, and the full model consistently outperforms all ablation variants.
- **Necessity of multi-modal fusion**: Mol-Only drops by 4.9%; adding any single biological modality partially recovers performance, with the full three-modality setting performing best.
- **Largest gains on the Biogen regression tasks** (MAE reduced by 17.2%), indicating that cellular and genetic information is particularly valuable for ADME property prediction.
- Tree depth $h=6$ is optimal; shallower trees lack expressiveness while deeper trees are prone to overfitting.

## Highlights & Insights

- **Tree-based vector quantization for biological hierarchy modeling** is an elegant design: VQ is a well-established technique for discrete representation learning, but extending it to a shared multi-modal tree structure — where tree levels correspond to biological hierarchy (molecule → cell → gene) — constitutes a genuinely novel formalization. t-SNE visualizations clearly demonstrate that CHMR simultaneously achieves cross-modal alignment and hierarchical structural organization.
- The **structure-aware propagation** augmentation strategy cleverly exploits a cheminformatics prior (structurally similar molecules elicit similar biological responses), offering greater biological plausibility than simple KNN or mean imputation.
- A case study on drug property prediction demonstrates the interpretability of the framework: different modalities contribute complementary predictive cues (1D fingerprints → metabolic clearance, 3D conformation → protein binding affinity, biological context → P-gp drug efflux).

## Limitations & Future Work

- Pre-training data involves integrating multiple public data sources, which may introduce data quality and consistency issues.
- The tree structure in Tree-VQ is a predefined fixed binary tree, lacking an adaptive dynamic tree-growing mechanism.
- Modality augmentation relies on the quality of the inter-molecular structural similarity matrix, and may be less effective for structurally novel molecules with no similar neighbors.
- The absence of protein target information limits applicability to target-related property prediction tasks.

## Related Work & Insights

- **vs InfoAlign**: InfoAlign also integrates molecular, cell morphology, and transcriptomic modalities, but employs flat alignment without modeling hierarchical dependencies. CHMR consistently outperforms it across all benchmarks, with a 17.2% lead on Biogen.
- **vs CLOOME**: CLOOME applies InfoLOOB contrastive learning on structure–phenotype pairs, but handles only two modalities and provides no missing modality mechanism, limiting generalization.
- **vs GraphMVP**: GraphMVP achieves 2D–3D semantic transfer but is confined to intra-molecular modalities and does not incorporate external biological responses.

## Rating

- Novelty: ⭐⭐⭐⭐ Tree-VQ for hierarchical multi-modal quantization represents a genuine technical contribution, though the overall framework (alignment + reconstruction + quantization) is somewhat heavyweight in its composition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 728 tasks, 9 datasets, 20+ baselines, comprehensive ablations and hyperparameter analyses — exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ The methodology section is highly mathematical with a clear notation system, though the overall length is somewhat excessive.
- Value: ⭐⭐⭐⭐ Clear practical significance for molecular representation learning in drug discovery, particularly in missing modality scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_nlp/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[AAAI 2026\] LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems](lucid_learning-enabled_uncertainty-aware_certification_of_stochastic_dynamical_s.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](../../ACL2026/medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](../../ACL2026/medical_nlp/learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)

</div>

<!-- RELATED:END -->
