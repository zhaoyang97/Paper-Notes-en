---
title: >-
  [Paper Note] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels
description: >-
  [ACL 2026][Medical Imaging][Biomedical retrieval] BioHiCL leverages the **hierarchical multi-label annotations** of MeSH (Medical Subject Headings) to provide structured supervision for dense retrievers. By aligning the…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Biomedical retrieval"
  - "MeSH hierarchy"
  - "contrastive learning"
  - "multi-label"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: f9dff78f49e8666d
---

# BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels

**Conference**: ACL 2026
**arXiv**: [2604.15591](https://arxiv.org/abs/2604.15591)
**Code**: [https://github.com/MengfeiLan/BioHiCL](https://github.com/MengfeiLan/BioHiCL)
**Area**: Medical Imaging
**Keywords**: Biomedical retrieval, MeSH hierarchy, contrastive learning, multi-label, parameter-efficient fine-tuning

## TL;DR
BioHiCL leverages the **hierarchical multi-label annotations** of MeSH (Medical Subject Headings) to provide structured supervision for dense retrievers. By aligning the embedding space with the MeSH semantic space via depth-weighted label similarity, a 0.1B model surpasses most specialized models on biomedical retrieval, sentence similarity, and question answering tasks.

## Background & Motivation

**Background**: General-domain dense retrievers (e.g., BGE, E5) perform well on general IR benchmarks but fail to capture biomedical-specific terminology and semantic relationships. Specialized biomedical retrieval models (e.g., MedCPT, BMRetriever) improve semantic alignment through large-scale contrastive learning.

**Limitations of Prior Work**: Existing biomedical retrieval models rely on coarse-grained relevance signals—either binary annotations (relevant/irrelevant) or query-article click data. Such coarse signals cannot capture the complex **partial semantic overlap** in biomedical texts (e.g., two articles labeled "irrelevant" may share a common parent concept in the disease hierarchy).

**Key Challenge**: Semantic relationships among biomedical texts are graded and hierarchical, yet training signals are binary—learning graded semantic relationships from binary signals limits retrieval precision.

**Goal**: Design a method that leverages the MeSH hierarchical structure to provide **fine-grained, graded supervision signals** for adapting general-purpose retrievers to the biomedical domain.

**Key Insight**: MeSH provides naturally multi-faceted supervision—each article is annotated with multiple MeSH labels, the labels themselves form a hierarchical tree, and the degree and depth of label overlap can quantify semantic similarity.

**Core Idea**: Align the similarity in embedding space with that in the MeSH depth-weighted label space, replacing binary contrastive learning with hierarchical multi-label contrastive learning.

## Method

### Overall Architecture
Built upon the general-domain dense retriever BGE, the model is fine-tuned with LoRA on 80,000 MeSH-annotated abstracts from BioASQ. The training objective consists of two components: (1) a regression loss $\mathcal{L}_{\text{mse}}$ that fits embedding similarity to label similarity; and (2) a hierarchical contrastive loss $\mathcal{L}_{\text{con}}$ that pulls semantically related documents closer and pushes unrelated ones apart in embedding space.

### Key Designs

1. **Depth-Weighted Hierarchical Label Representation**:

    - Function: Encodes the MeSH hierarchical structure as computable label similarity.
    - Mechanism: The MeSH label set of each abstract is expanded to include all ancestor nodes, forming a complete path $m_i^{\text{hier}}$, encoded as a multi-hot vector $y_i \in \{0,1\}^C$. Each MeSH concept $c_j$ is assigned a depth weight $w_j = \log(d(c_j)+1)$, such that deeper (more specific) concepts receive higher weights. The label similarity between two documents is defined as the cosine similarity of their weighted multi-hot vectors: $\text{SimL}(k_p, k_q) = \cos(y_p \odot \mathbf{w}, y_q \odot \mathbf{w})$.
    - Design Motivation: Shallow MeSH labels (e.g., "Disease") carry little discriminative value when matched, whereas deep labels (e.g., "Intracranial Hemorrhage") indicate genuine semantic relatedness. Depth weighting focuses the model on meaningful fine-grained matches.

2. **Hierarchical Multi-Label Contrastive Loss**:

    - Function: Prevents embedding collapse and maintains discriminative structure.
    - Mechanism: Positive pairs are document pairs with label similarity $\text{SimL} > \beta$; negative pairs are those with no label overlap ($\text{SimL}=0$). Positive pairs are weighted by their label similarity in the contrastive loss: $\mathcal{L}_{\text{con}} = -\mathbb{E}_i \log[\text{SimL}(k_i, k_i^+) \cdot \exp(\text{SimE}(k_i, k_i^+)) / \sum_{k_j^-} \exp(\text{SimE}(k_i, k_j^-))]$. The threshold $\beta$ filters weakly associated pairs to reduce noisy supervision.
    - Design Motivation: Regression loss alone may cause all embeddings to collapse to a single point; the contrastive loss maintains discriminability by pushing unrelated documents apart. Weighting by label similarity ensures that more semantically related positive pairs contribute larger gradients.

3. **LoRA Parameter-Efficient Fine-Tuning**:

    - Function: Adapts a general-domain retriever to the biomedical domain at low cost.
    - Mechanism: The original BGE parameters are frozen, and low-rank adapters $W_{\text{adapted}}^{(l)} = W^{(l)} + B^{(l)} A^{(l)}$ are injected, training only 0.3% of the parameters.
    - Design Motivation: Full-parameter fine-tuning is prone to overfitting on small datasets and incurs high computational cost; LoRA achieves domain adaptation while preserving general language understanding.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{\text{mse}} + \lambda \mathcal{L}_{\text{con}}$, with $\lambda=0.1$ and $\beta=0.3$. Training is conducted on 80,000 abstracts from BioASQ v2022; the optimal checkpoint is selected on the TREC-CT 2022 validation set. Training and inference can be completed on a single A100 40GB GPU.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | BioHiCL-Base (0.1B) | BMRetriever-1B | bge-base (0.1B) |
|---|---|---|---|---|
| IR Average | nDCG@10 | **0.543** | 0.531 | 0.529 |
| NFCorpus | nDCG@10 | 0.379 | 0.344 | 0.368 |
| TREC-COVID | nDCG@10 | **0.812** | 0.840 | 0.798 |
| BIOSSES | Spearman | **0.896** | 0.858 | 0.860 |
| PubMedQA | Recall@1 | 0.893 | 0.810 | 0.856 |

### Ablation Study

| Configuration | IR Avg | Note |
|---|---|---|
| BioHiCL-Base | **0.543** | Full model |
| w/o $\mathcal{L}_{\text{con}}$ | 0.528 | Removing contrastive loss yields the largest drop |
| w/o Ancestor Label | 0.538 | Without ancestor node expansion |
| w/o $\mathcal{L}_{\text{mse}}$ | 0.537 | Without regression loss |
| w/o Depth Weight | 0.541 | Without depth weighting |
| w/o LoRA (full params) | 0.542 | LoRA matches full fine-tuning performance |

### Key Findings
- BioHiCL-Base (0.1B) outperforms BMRetriever (1B) on the average IR metric, demonstrating that structured supervision signals can compensate for the gap in model scale.
- The contrastive loss is the most critical component (removing it reduces IR average by 0.015), confirming the necessity of preventing embedding collapse.
- Fine-tuning BMRetriever with the BioHiCL methodology leads to severe performance degradation (0.501→0.279), as replacing the original instruction-based training objective disrupts its retrieval-specialized embedding geometry.
- LoRA achieves performance comparable to full fine-tuning with only 0.3% of the parameters, validating the effectiveness of parameter-efficient methods for domain adaptation.

## Highlights & Insights
- **Leveraging the MeSH hierarchical structure as a graded supervision signal** is a natural and effective design: MeSH is an expert-curated, standardized vocabulary that provides a precise measure of semantic relationships between documents. This paradigm of "borrowing existing structured knowledge for supervision" is transferable to any domain with a hierarchical label system (e.g., legal code classification, product taxonomy).
- **Depth-weighted label similarity** encodes the domain intuition that "specific concepts matter more than abstract ones" in an elegantly minimal form (a single formula $w_j = \log(d(c_j)+1)$).
- The extreme efficiency of the 0.1B model makes it well-suited for large-scale deployment, offering a clear practical advantage over systems such as BMRetriever and MedCPT that require 1B+ parameters.

## Limitations & Future Work
- Training is conducted on only 80,000 abstracts from BioASQ, far smaller in scale than MedCPT (click data) and BMRetriever (multi-task data).
- The coverage and granularity of MeSH annotations are constrained by the label set maintained by NLM; emerging concepts may lack adequate coverage.
- The potential of combining MeSH hierarchical information with instruction-based retrieval training remains unexplored.
- Gains on SCIDOCS are limited (0.215→0.225), indicating that cross-domain generalization requires further improvement.

## Related Work & Insights
- **vs. MedCPT (Jin et al., 2023)**: The latter employs query-article clicks for contrastive learning; BioHiCL uses MeSH hierarchy to provide finer-grained supervision signals.
- **vs. BMRetriever (Xu et al., 2024)**: The latter trains a 1B model via large-scale multi-task learning; BioHiCL achieves comparable performance with only 0.1B parameters and MeSH supervision, offering substantially higher efficiency.
- **vs. BiCA (Sinha et al., 2025)**: The latter performs biomedical adaptation without exploiting hierarchical label structure; BioHiCL complements this with an explicit hierarchical dimension.

## Rating
- Novelty: ⭐⭐⭐ Contrastive learning with MeSH supervision is a natural combination, though the core idea is not particularly surprising.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task evaluation (IR + similarity + QA), detailed ablations, and efficiency analysis provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and concise, though the Related Work section is somewhat thin.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](../../AAAI2026/medical_imaging/bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/medical_imaging/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2026/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)

</div>

<!-- RELATED:END -->
