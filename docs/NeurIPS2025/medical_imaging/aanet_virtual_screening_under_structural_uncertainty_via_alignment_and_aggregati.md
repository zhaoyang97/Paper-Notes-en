---
title: >-
  [Paper Note] AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation
description: >-
  [NeurIPS 2025][Medical Imaging][Virtual screening] To address the unavailability of holo protein structures in real-world drug discovery…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Virtual screening"
  - "drug discovery"
  - "structural uncertainty"
  - "contrastive learning"
  - "protein binding site"
date: 2026-05-08
content_hash: 4a2afe511c289c54
---

# AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation

**Conference**: NeurIPS 2025
**arXiv**: [2506.05768](https://arxiv.org/abs/2506.05768)  
**Code**: [GitHub](https://github.com/Wiley-Z/AANet)  
**Area**: Medical Imaging / AI for Science
**Keywords**: Virtual screening, drug discovery, structural uncertainty, contrastive learning, protein binding site

## TL;DR
To address the unavailability of holo protein structures in real-world drug discovery, this paper proposes AANet—a framework that aligns representations via tri-modal contrastive learning (ligand–holo pocket–detected cavity) and aggregates multiple candidate binding sites through cross-attention. AANet substantially outperforms SOTA methods in blind screening on apo/predicted protein structures (EF1% on DUD-E: 11.75 → 37.19).

## Background & Motivation

**Background**: Structure-based virtual screening (SBVS) is a core step in drug discovery, identifying potentially active molecules from large compound libraries by evaluating the complementarity between compounds and protein pockets. Existing methods (Glide docking, DrugCLIP, etc.) all rely on holo protein structures (structures with known ligand binding).

**Limitations of Prior Work**: (a) Most valuable targets lack holo structures and are available only in apo (ligand-free) or AlphaFold2-predicted forms; (b) existing methods suffer a dramatic performance drop on apo/predicted structures; (c) the critical bottleneck is not structural deformation but **inaccurate binding site localization**.

**Key Challenge**: Deep learning methods such as DrugCLIP are robust to structural noise but highly sensitive to pocket localization. Cavities detected by geometric tools (e.g., Fpocket) often deviate substantially from true binding sites.

**Goal**: How to perform accurate virtual screening without prior knowledge of the ligand binding location?

**Key Insight**: Geometrically detected cavities serve as "noisy proxies" of holo pockets—learning to align the representations of both.

**Core Idea**: Tri-modal alignment (ligand–holo pocket–cavity) combined with cross-attention aggregation of candidate sites, enabling accurate screening even when the binding site is unknown.

## Method

### Overall Architecture
Two stages: (1) **Alignment stage** — tri-modal contrastive learning on annotated PDBBind data to pre-train aligned representations of ligands, holo pockets, and detected cavities; (2) **Aggregation stage** — frozen encoders with a trained cross-attention adapter, learning to aggregate multiple candidate cavities using AlphaFold2-predicted structures and ChEMBL bioactivity data.

### Key Designs

1. **Tri-modal Contrastive Alignment**:

    - Function: Learn aligned representations across ligands, holo pockets, and detected cavities.
    - Mechanism: Pairwise sigmoid loss across all three modal pairs: $\mathcal{L}_{CL} = \mathcal{L}_{p,l}(P_l, l) + \mathcal{L}_{p,l}(P_c, l) + \mathcal{L}_{p,p}(P_c, P_l)$
    - Design Motivation: Aligning only ligand–pocket leads to overfitting to holo pocket locations; incorporating the cavity modality encourages the model to learn intrinsic spatial features of the protein structure.
    - Key Details: Positive cavity samples are selected via IoU filtering (IoU > τ); pocket extraction radius is expanded from 6Å to 10Å to handle cavity fragmentation.

2. **Hard Negative Mining**:

    - Function: Sample negatives from non-binding cavities to enhance discriminability.
    - Mechanism: Low-IoU cavities are used as negatives and pushed away from ligands and holo pockets via contrastive loss.
    - Design Motivation: Protein surfaces contain numerous geometrically concave but functionally inactive "pseudo-pockets"; the model must learn to distinguish these.

3. **Cross-Attention Adapter**:

    - Function: Dynamically aggregate multiple candidate cavities via learned weighting.
    - Mechanism: Ligand embeddings serve as queries; cavity embeddings serve as keys/values; single-head attention aggregation: $\tilde{e}_c = \sum_{s=1}^S a^{(s)} \cdot \mathcal{F}_s(P_c^{(s)})$
    - Initialized as an approximate identity mapping (high-temperature softmax ≈ uniform average), progressively learning optimal weights.
    - Design Motivation: Training on bioactivity data without pocket annotations allows the model to autonomously infer the most probable binding cavity.
    - Attention supervision: annotated data use one-hot labels; unannotated data use soft labels derived from pre-trained AANet cavity scores, supervised via KL divergence.

### Loss & Training
- Alignment stage: PDBBind 2020 general set (with test-set overlap removed).
- Aggregation stage: ChEMBL35 bioactivity data + AlphaFold2-predicted structures.
- Strict data deduplication: all UniProt entries related to DUD-E/LIT-PCBA are excluded.
- Total loss: $\mathcal{L}_{agg} = \mathcal{L}'_{CL} + \lambda \cdot \mathcal{L}'_{KL}$

## Key Experimental Results

### Main Results (DUD-E, 38 targets)

| Method | Structure | Setting | BEDROC | EF1% |
|--------|-----------|---------|--------|------|
| Glide-SP | holo | oracle | 0.296 | 17.25 |
| DrugCLIP | holo | oracle | 0.516 | 33.70 |
| DrugCLIP | apo-pred | blind | 0.197 | 12.05 |
| **AANet** | holo | oracle | **0.637** | **40.85** |
| **AANet** | apo-pred | blind | **0.623** | **37.19** |

### Ablation Study

| Configuration | EF1% (apo-pred blind) | Note |
|---------------|-----------------------|------|
| DrugCLIP | 12.05 | Baseline |
| AANet (alignment only) | ~30+ | Contrastive alignment alone yields substantial gain |
| AANet (alignment + aggregation) | **37.19** | Aggregation provides further improvement |

### Key Findings
- **AANet on apo blind nearly matches holo performance**: EF1% 37.19 vs. 40.85 on holo (gap of only 3.66), whereas DrugCLIP drops from 33.70 to 12.05.
- **Performance degradation stems primarily from pocket localization, not structural deformation**: DrugCLIP performance barely changes under apo-oracle (known pocket location) but collapses under blind settings.
- **Robust across different detection tools**: Effective with Fpocket, P2Rank, and other cavity detectors, indicating that the model learns intrinsic structural features.
- **Effective on the more challenging LIT-PCBA benchmark**: AANet maintains a large margin in this more realistic screening scenario.

## Highlights & Insights
- **Precise problem diagnosis**: Systematic analysis reveals that the critical bottleneck of DL-based SBVS is pocket localization rather than structural noise, correcting a previously vague understanding in the field. This diagnosis alone is a valuable contribution.
- **Cavity-as-proxy idea**: Using detected cavities as noisy proxies for holo pockets and learning aligned representations—a transferable paradigm for any scenario involving uncertain annotation quality.
- **Elegant adapter design**: Initialized as uniform aggregation and learned toward selective attention, enabling smooth adaptation and training on unannotated data.
- **Practical drug discovery value**: Enables SBVS to be genuinely applicable to first-in-class targets (novel targets without holo structures), addressing a critical bottleneck for real-world deployment.

## Limitations & Future Work
- **Dependence on cavity detection tools**: If geometric detection fails entirely (e.g., for proteins with novel folds), the method's premise breaks down.
- **Single protein conformation**: Conformational ensembles reflecting protein dynamics are not considered.
- **Potential data leakage risk in the aggregation stage**: Although deduplication was performed, whether UniProt-level deduplication is sufficiently stringent warrants further discussion.
- **Future directions**: (1) Incorporation of MD simulation-based conformational ensembles; (2) end-to-end training instead of a two-stage pipeline; (3) extension to protein–protein interaction screening.

## Related Work & Insights
- **vs. DrugCLIP**: DrugCLIP serves as the base architecture (CLIP-style contrastive learning); AANet extends it with cavity alignment and aggregation, resolving DrugCLIP's sensitivity to pocket localization.
- **vs. Glide/AutoDock**: Traditional docking methods are sensitive to structural deformation but relatively tolerant of pocket localization errors (since they search within the pocket); AANet and DrugCLIP exhibit the opposite behavior.
- **vs. TankBind**: TankBind attempts to predict binding locations explicitly but with limited performance; AANet implicitly resolves the localization problem by aggregating multiple candidate cavities.

## Rating
- Novelty: ⭐⭐⭐⭐ The tri-modal alignment + aggregation framework targets a genuine pain point; the cavity proxy idea is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual benchmarks (DUD-E + LIT-PCBA); multi-dimensional comparison across holo/apo-exp/apo-pred × oracle/annot/blind settings.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis, rigorous experimental design, and clear structure.
- Value: ⭐⭐⭐⭐⭐ Addresses the critical translation bottleneck from holo to apo in SBVS, with direct practical impact on drug discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](../../AAAI2026/medical_imaging/s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[CVPR 2026\] Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](../../CVPR2026/medical_imaging/better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_imaging/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)

</div>

<!-- RELATED:END -->
