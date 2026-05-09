---
title: >-
  [Paper Note] TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Molecular property prediction] TRIDENT is a tri-modal molecular representation learning framework that introduces Hierarchical Taxonomic Annotations (HTA) as a third modality. It combines a volumetric contrastive loss for global tri-modal alignment with a functional group–text local alignment module, dynamically balancing the two objectives via a momentum mechanism. The framework achieves state-of-the-art performance across 18 molecular property prediction tasks.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - Molecular property prediction
  - tri-modal alignment
  - hierarchical taxonomic annotation
  - volumetric contrastive loss
  - local alignment
date: 2026-05-08
content_hash: 9d873c6c706ba848
---

# TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships

**Conference**: NeurIPS 2025
**arXiv**: [2506.21028](https://arxiv.org/abs/2506.21028)
**Code**: [GitHub](https://github.com/uta-smile/TRIDENT)
**Area**: Self-Supervised Learning
**Keywords**: Molecular property prediction, tri-modal alignment, hierarchical taxonomic annotation, volumetric contrastive loss, local alignment

## TL;DR

TRIDENT is a tri-modal molecular representation learning framework that introduces Hierarchical Taxonomic Annotations (HTA) as a third modality. It combines a volumetric contrastive loss for global tri-modal alignment with a functional group–text local alignment module, dynamically balancing the two objectives via a momentum mechanism. The framework achieves state-of-the-art performance across 18 molecular property prediction tasks.

## Background & Motivation

Molecular representation learning aims to map chemical structures into computable feature vectors, serving as a core technique in drug discovery and virtual screening. Multi-modal learning has been shown to enhance representation quality by integrating structural, textual, and functional information.

**Three key limitations of existing methods**:

**Neglect of fine-grained taxonomic annotations**: Existing methods rely solely on generic functional description texts, ignoring the complementary perspectives offered by different classification systems (e.g., LOTUS Tree for natural product taxonomy, MeSH Tree for medical functions), which provide distinct emphases for the same molecule.

**Limited alignment paradigms**: Existing methods depend on pairwise alignment anchored to a single modality (e.g., SMILES–Text), failing to capture higher-order interactions among three modalities.

**Neglect of local correspondences**: Most methods perform only molecule-level global alignment, overlooking fine-grained associations between functional groups (e.g., hydroxyl, aromatic rings) and their corresponding textual descriptions.

**Core Idea**: Introduce HTA (Hierarchical Taxonomic Annotation) as a third modality, employ a volumetric contrastive loss for geometry-aware tri-modal alignment, and incorporate a local alignment module to capture correspondences between functional groups and sub-textual descriptions.

## Method

### Overall Architecture

The model takes tri-modal inputs $\langle\text{SMILES, Text, HTA}\rangle$. SMILES is encoded by a molecular encoder $E_m$, while textual descriptions and HTA share a text encoder $E_t$. The three modal embeddings are projected into a shared space via separate MLPs, followed by global and local alignment.

### Key Designs

1. **HTA Modality Construction**

   - Molecular information is retrieved from PubChem, and multi-level functional annotations are extracted across 32 classification systems (LOTUS Tree, MeSH Tree, etc.).
   - GPT-4o is used to synthesize structured annotations into high-fidelity, human-readable HTA text descriptions.
   - A dataset of 47,269 $\langle\text{SMILES, Text, HTA}\rangle$ triplets is constructed.
   - HTA is complementary to conventional descriptions: HTA provides multi-perspective information from 32 viewpoints (e.g., chemical origin, natural product classification, medical function), whereas conventional descriptions more directly highlight core molecular features.

2. **Volumetric Global Tri-Modal Alignment**

   - Rather than pairwise cosine similarity, the framework computes the parallelepiped volume spanned by three normalized embeddings $(m, t, h)$:
     $$\text{Vol}(m,t,h) = \sqrt{1 - \langle m,t\rangle^2 - \langle m,h\rangle^2 - \langle t,h\rangle^2 + 2\langle m,t\rangle\langle t,h\rangle\langle h,m\rangle}$$
   - Matched triplets should exhibit small volume (tri-modal convergence), while unmatched triplets should exhibit large volume.
   - A bidirectional loss is used: $\mathcal{L}_{M2TH}$ (molecule retrieves Text + HTA) and $\mathcal{L}_{TH2M}$ (Text + HTA retrieves molecule), averaged symmetrically.

3. **Functional Group–Text Local Alignment**

   - RDKit is used to extract salient functional groups from SMILES (85 types, including hydroxyl, amine, carboxyl, and aromatic systems).
   - High-quality textual descriptions are composed for each functional group via expert annotation and GPT-4o assistance.
   - Functional group and text embeddings are encoded separately, aggregated via max-pooling, and aligned through a bidirectional contrastive loss: $\mathcal{L}_{FG2T} + \mathcal{L}_{T2FG}$.

4. **Momentum-Based Dynamic Balancing**

   - Overall loss: $\mathcal{L} = \alpha \mathcal{L}_g + (1-\alpha) \mathcal{L}_l$
   - $\alpha$ is updated dynamically via exponential moving average: $\alpha_t = \beta \alpha_{t-1} + (1-\beta) \frac{\mathcal{L}_g^{(t)}}{\mathcal{L}_g^{(t)} + \mathcal{L}_l^{(t)}}$, with momentum parameter $\beta = 0.9$.
   - This enables the model to automatically focus on whichever alignment objective has a higher current loss.

### Loss & Training

- The global loss employs volumetric contrastive learning with a learnable softmax temperature $\tau$.
- The local loss uses standard contrastive loss with a shared temperature parameter.
- Evaluation follows scaffold splits, with means and standard deviations reported over three random seeds.

## Key Experimental Results

### Main Results (MoleculeNet Classification ROC-AUC%)

| Method | BBBP | Tox21 | ToxCast | Sider | ClinTox | MUV | HIV | Bace | Avg |
|---|---|---|---|---|---|---|---|---|---|
| MoleculeSTM | 70.75 | 75.71 | 65.17 | 63.70 | 86.60 | 65.69 | 77.02 | 81.99 | 73.33 |
| Atomas | 73.72 | 77.88 | 66.94 | 64.40 | 93.16 | 76.30 | 80.55 | 83.14 | 77.01 |
| **TRIDENT (M-M)** | **73.95** | **79.36** | **67.80** | 63.64 | 95.41 | **83.51** | **81.63** | 82.39 | **78.46** |

### Ablation Study

| Configuration | Tox21 | ToxCast | BBBP | Bace |
|---|---|---|---|---|
| Full TRIDENT | **79.36** | **67.80** | **73.95** | **82.39** |
| w/o HTA | Notable drop | Drop | Drop | Drop |
| w/o local alignment | Drop | Drop | Drop | Drop |
| w/o volumetric loss (standard contrastive) | Significant instability | Drop | Drop | Drop |
| Sum vs. Momentum | 77.79 vs. 79.36 | 66.73 vs. 67.80 | 72.15 vs. 73.95 | 81.42 vs. 82.39 |

### Key Findings

- **HTA contributes most**: Removing HTA leads to substantial performance degradation; the multi-perspective annotations from 32 classification systems provide structured semantics unavailable from conventional text.
- **Volumetric loss outperforms standard contrastive loss**: Standard contrastive loss is unstable in tri-modal settings, whereas the volumetric loss effectively captures higher-order geometric relationships.
- **Momentum mechanism outperforms fixed weighting**: Dynamic balancing adaptively allocates optimization effort between global and local objectives at different training stages.

## Highlights & Insights

- **HTA as the central innovation**: Introducing multi-taxonomy annotations as an independent modality substantially increases the information content compared to single-type functional descriptions.
- **Elegant application of volumetric contrastive loss**: The tri-modal alignment framework from GRAM (audio–visual–text) is extended to the molecular domain for the first time, accommodating modality combinations with greater structural heterogeneity.
- **Functional group–level local alignment**: The curated dataset of 85 functional groups paired with high-quality text descriptions constitutes a reusable and valuable resource.
- **LLM-assisted data construction**: The pipeline combining GPT-4o synthesis with domain expert review represents a practically viable approach to high-quality annotation construction.

## Limitations & Future Work

- Molecular properties such as toxicity depend not only on molecular structure but also on targets and metabolites, which are not incorporated in the current framework.
- The dataset scale is relatively small (47,269 triplets); scalability to a larger chemical space remains to be validated.
- HTA construction relies on PubChem's classification systems, which may limit effectiveness for molecules with incomplete PubChem records.
- Functional group detection depends on predefined RDKit patterns, potentially missing novel functional groups.

## Related Work & Insights

- **vs. MoleculeSTM**: MoleculeSTM performs only SMILES–Text bimodal contrastive learning; TRIDENT introduces the HTA third modality and local alignment, achieving an average ROC-AUC improvement of approximately 5 percentage points.
- **vs. Atomas**: Atomas incorporates local alignment but uses static attention; TRIDENT achieves superior performance through the momentum mechanism and functional group–level alignment.
- **vs. GRAM**: GRAM originally proposed volumetric contrastive loss for audio–visual–text alignment; TRIDENT is the first to extend this framework to the molecular domain.

## Rating

- Novelty: ⭐⭐⭐⭐ The tri-modal framework and HTA modality are original contributions, though individual technical components (volumetric loss, LoRA) are drawn from prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 downstream tasks, comprehensive ablations, and multiple encoder configurations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed method description, and transparent data construction process.
- Value: ⭐⭐⭐⭐ Offers a clear advancement for molecular representation learning; the HTA dataset itself carries independent value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[ICLR 2026\] Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions](../../ICLR2026/self_supervised/enhancing_molecular_property_predictions_by_learning_from_bond_modelling_and_int.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

<!-- RELATED:END -->
