---
title: >-
  [Paper Note] Align-Pro: Align Protein Representations Through Multi-Modal Learning
description: >-
  [ACL 2025][Computational Biology][Protein representation learning] Align-Pro aligns the representations of three modalities of proteins—sequence, structure, and functional description—into a unified embedding space through a multi-modal contrastive learning framework, thereby enabling cross-modal protein retrieval, classification, and function prediction.
tags:
  - "ACL 2025"
  - "Computational Biology"
  - "Protein representation learning"
  - "multi-modal alignment"
  - "sequence-structure-function"
  - "contrastive learning"
  - "pre-training"
date: 2026-05-08
content_hash: f2ac1e562b593bde
---

# Align-Pro: Align Protein Representations Through Multi-Modal Learning

**Conference**: ACL 2025  
**Area**: Computational Biology  
**Keywords**: Protein representation learning, multi-modal alignment, sequence-structure-function, contrastive learning, pre-training

## TL;DR
Align-Pro aligns the representations of three modalities of proteins—sequence, structure, and functional description—into a unified embedding space through a multi-modal contrastive learning framework, thereby enabling cross-modal protein retrieval, classification, and function prediction.

## Background & Motivation

**Background**: Protein research involves multiple data modalities: amino acid sequences (1D), 3D structures (predicted by tools such as AlphaFold), and functional descriptions in natural language (such as Gene Ontology annotations). Currently, each modality is typically processed by independent models—the ESM series for sequences, GNNs or Equivariant NNs for structures, and BioBERT for text.

**Limitations of Prior Work**: Representation spaces of independently trained modality encoders are unaligned, making direct cross-modal operations impossible. For example, retrieving corresponding protein sequences from functional descriptions, or predicting functions based on structural similarity, requires additional alignment steps. Existing multi-modal protein methods are mostly limited to dual-modal alignment of sequence-structure, neglecting the crucial modality of functional text.

**Key Challenge**: A protein's function is determined by its structure, which is encoded by its sequence. Strong correlations exist among the three, but current methods struggle to capture this tri-modal correspondence within a unified framework.

**Goal**: To build a unified tri-modal protein representation learning framework that aligns sequence, structure, and functional description embeddings in the same space.

**Key Insight**: Drawing inspiration from the success of vision-language contrastive learning models such as CLIP, this work extends bi-modal alignment to tri-modal scenarios in the protein domain.

**Core Idea**: Align the representations of sequence, structure, and functional descriptions into a unified space via tri-modal contrastive learning, realizing a "protein version of CLIP".

## Method

### Overall Architecture
Align-Pro consists of three encoders: a sequence encoder (based on the ESM-2 pre-trained model), a structure encoder (based on GVP-GNN, an equivariant graph neural network), and a function encoder (based on PubMedBERT, a text encoder). The three encoders output protein sequence, structure, and function embeddings, respectively. The model is trained on a large-scale protein database via a tri-modal contrastive loss, pulling together the three modal embeddings of the same protein while pushing apart the embeddings of different proteins.

### Key Designs

1. **Tri-Modal Contrastive Loss**:

    - **Function**: Align the representation space of sequence, structure, and function modalities.
    - **Mechanism**: Extend the traditional bi-modal InfoNCE loss to a weighted sum of three pairwise contrastive losses: $\mathcal{L} = \alpha\mathcal{L}_{seq\text{-}struct} + \beta\mathcal{L}_{seq\text{-}func} + \gamma\mathcal{L}_{struct\text{-}func}$. Each contrastive loss uses temperature-scaled cosine similarity. Cross-modal pairs of the same protein serve as positive samples, while other proteins within the batch serve as negative samples.
    - **Design Motivation**: Three pairwise losses are more flexible than directly defining a triplet loss, allowing different alignment strengths between different modality pairs.

2. **Modality-Specific Projection Heads**:

    - **Function**: Map outputs of each encoder to a shared embedding space.
    - **Mechanism**: A 2-layer MLP projection head is appended to each encoder to project encoder outputs of varying dimensions to a shared space with the same dimension. To preserve intra-modal discriminative information, an intra-modal contrastive loss is added as an auxiliary objective, ensuring that projected representations are not only aligned across modalities but also remain distinguishable within each modality.
    - **Design Motivation**: Direct alignment in the encoder space may degrade the quality of pre-trained representations; projection heads provide additional adaptation space.

3. **Function Description Augmentation**:

    - **Function**: Address the issues of sparse and unevenly distributed protein function description data.
    - **Mechanism**: Utilize LLMs to rewrite and expand Gene Ontology annotations to generate diverse functional descriptions. For example, expanding the concise GO annotation "ATP binding" into "This protein has ATP binding activity, meaning it can specifically interact with adenosine triphosphate molecules...". This augmentation increases both the volume of training data and the semantic diversity of the functional text.
    - **Design Motivation**: Gene Ontology annotations are short and highly formatted, which is too monotonous for natural language training data.

### Loss & Training
The total loss is a weighted sum of three pairwise contrastive losses and three intra-modal auxiliary losses. A two-stage training strategy is adopted: the first stage freezes the encoders and trains only the projection heads, and the second stage fine-tunes all parameters. Approximately 500,000 fully annotated proteins from the UniProt database are used as training data.

## Key Experimental Results

### Main Results

| Task | Metric | Align-Pro | ESM-2 | ProtST | OntoProtein | Gain |
|------|------|-----------|-------|--------|-------------|------|
| GO Function Prediction | Fmax | 0.694 | 0.651 | 0.672 | 0.658 | +2.2 |
| EC Number Prediction | Fmax | 0.882 | 0.856 | 0.871 | 0.862 | +1.1 |
| Cross-Modal Retrieval | R@10 | 78.3 | - | 72.1 | 69.5 | +6.2 |
| Fold Classification | ACC(%) | 91.7 | 88.4 | 89.9 | 89.1 | +1.8 |

### Ablation Study

| Configuration | GO Prediction (Fmax) | Cross-Modal Retrieval (R@10) | Description |
|------|-------------|-----------------|------|
| Full model | 0.694 | 78.3 | Full framework |
| Sequence-Structure Only | 0.672 | 68.7 | Removed functional modality |
| Sequence-Function Only | 0.681 | 73.1 | Removed structural modality |
| Without Function Augmentation | 0.678 | 74.5 | No LLM expansion of descriptions |
| Without Intra-Modal Loss | 0.686 | 76.0 | Removed auxiliary loss |

### Key Findings
- Tri-modal alignment significantly outperforms any bi-modal combination, validating the complementarity of the sequence-structure-function triangular relationship.
- The functional description augmentation strategy shows a clear contribution (+1.6 Fmax), indicating that GP text quality is crucial for cross-modal learning.
- The largest improvement is achieved in the cross-modal retrieval task (+6.2 R@10), suggesting that a unified embedding space is highly valuable for retrieval tasks.
- The structural modality contributes the most to fold classification, while the functional modality contributes the most to GO prediction, indicating that each modality has its own strengths.

## Highlights & Insights
- Successfully extends CLIP-style contrastive learning to the protein tri-modal scenario, providing a general-purpose protein multi-modal embedding space. This unified representation can be directly transferred to downstream applications such as drug discovery and protein engineering.
- The strategy of utilizing LLMs to augment functional descriptions cleverly addresses the sparse annotation problem in biological databases. This approach of using LLMs to compensate for domain data scarcity is worth emulating in other scientific fields.

## Limitations & Future Work
- Training data is limited to proteins with complete tri-modal annotations, which only represent a small fraction of all biochemically characterized proteins. Semi-supervised methods may help leverage partially annotated proteins.
- The negative sample selection strategy in contrastive learning impacts embedding quality; the current naive in-batch negative sampling might not be highly efficient.
- The structure encoder relies on predicted structures (AlphaFold), and errors within the predicted structures propagate to the embedding space.
- Functional descriptions mainly cover molecular functions and biological processes, with insufficient coverage of more complex functional relationships such as protein-protein interactions.
- Future work can extend the alignment framework to more modalities, such as protein dynamics simulation data and experimental functional data.
- Cross-species protein function prediction is an important application scenario, where multi-modal alignment may help transfer knowledge from well-studied species to under-studied ones.

## Related Work & Insights
- **vs ProtST**: ProtST only aligns sequence and text modalities, whereas this work adds the structural modality, providing more comprehensive information.
- **vs OntoProtein**: OntoProtein leverages knowledge graphs to enhance protein representations but does not undergo explicit multi-modal alignment.
- **vs ESM-2**: ESM-2 is a powerful sequence encoder but only processes a single modality, whereas this work achieves further performance gains through multi-modal alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The protein tri-modal alignment framework is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple downstream tasks with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The cross-disciplinary work is clearly presented.
- Value: ⭐⭐⭐⭐ Promotes advancements in the intersection of computational biology and NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme Classification](../../ICLR2026/computational_biology/poinncare_hyperbolic_multi-modal_learning_for_enzyme_classification.md)
- [\[ICLR 2026\] Align Your Structures: Generating Trajectories with Structure Pretraining for Molecular Dynamics](../../ICLR2026/computational_biology/align_your_structures_generating_trajectories_with_structure_pretraining_for_mol.md)
- [\[ICLR 2026\] Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models](../../ICLR2026/computational_biology/only_brains_align_with_brains_cross-region_alignment_patterns_expose_limits_of_n.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/computational_biology/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](../../NeurIPS2025/computational_biology/learning_repetition-invariant_representations_for_polymer_informatics.md)

</div>

<!-- RELATED:END -->
