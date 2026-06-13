---
title: >-
  [Paper Note] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels
description: >-
  [ACL 2026][Medical NLP][Biomedical Retrieval] BioHiCL leverages **hierarchical multi-label annotations** from MeSH (Medical Subject Headings) to provide structured supervision for dense retrievers. By aligning the embedd…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Biomedical Retrieval"
  - "MeSH Hierarchy"
  - "Contrastive Learning"
  - "Multi-label"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 24ed9cc63c41dd67
---

# BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels

**Conference**: ACL 2026  
**arXiv**: [2604.15591](https://arxiv.org/abs/2604.15591)  
**Code**: [https://github.com/MengfeiLan/BioHiCL](https://github.com/MengfeiLan/BioHiCL)  
**Area**: Biomedical NLP
**Keywords**: Biomedical Retrieval, MeSH Hierarchy, Contrastive Learning, Multi-label, Parameter-Efficient Fine-Tuning

## TL;DR
BioHiCL leverages **hierarchical multi-label annotations** from MeSH (Medical Subject Headings) to provide structured supervision for dense retrievers. By aligning the embedding space with the MeSH semantic space through depth-weighted label similarity, it enables a 0.1B model to outperform most specialized models in biomedical retrieval, sentence similarity, and question-answering tasks.

## Background & Motivation

**Background**: General domain dense retrievers (e.g., BGE, E5) demonstrate excellent performance on general IR benchmarks but struggle to capture specific biomedical terminology and semantic relationships. Specialized biomedical retrieval models (e.g., MedCPT, BMRetriever) have improved semantic alignment through large-scale contrastive learning.

**Limitations of Prior Work**: Existing biomedical retrieval models rely on coarse-grained relevance signals, such as binary annotations (relevant/irrelevant) or query-article click data. Such coarse signals cannot capture complex relationships involving **partial semantic overlap** common in biomedical texts (e.g., two articles labeled as "irrelevant" may actually share a parent concept in the disease hierarchy).

**Key Challenge**: Semantic relationships between biomedical texts are graded and hierarchical, yet training signals are typically binary. Using binary signals to learn graded semantic relationships limits retrieval precision.

**Goal**: To design a method that adapts general retrievers to the biomedical domain by utilizing the MeSH hierarchical structure to provide **fine-grained, graded supervision signals**.

**Key Insight**: MeSH provides natural multi-faceted supervision: each document contains multiple MeSH labels, and the labels themselves form a hierarchical tree. The degree of label overlap and hierarchical depth can be used to quantify semantic similarity.

**Core Idea**: Align embedding space similarity with MeSH depth-weighted similarity in the label space, replacing binary contrastive learning with hierarchical multi-label contrastive learning.

## Method

### Overall Architecture
Based on the general-domain dense retriever BGE, the model is fine-tuned using LoRA on 80,000 BioASQ abstracts with MeSH annotations. The training objective consists of two components: (1) a regression loss $\mathcal{L}_{\text{mse}}$ to fit embedding similarity to label similarity, and (2) a hierarchical contrastive loss $\mathcal{L}_{\text{con}}$ to pull semantically related documents closer while pushing unrelated ones apart.

### Key Designs

1. **Depth-Weighted Hierarchical Label Representation**:

    - **Function**: Encodes the MeSH hierarchical structure into a calculable label similarity.
    - **Mechanism**: The MeSH label set for each abstract is expanded into full paths including all ancestor nodes $m_i^{\text{hier}}$, encoded as a multi-hot vector $y_i \in \{0,1\}^C$. Each MeSH concept $c_j$ is assigned a depth weight $w_j = \log(d(c_j)+1)$, where deeper (more specific) concepts receive higher weights. Label similarity between two documents is defined as the cosine similarity of the weighted multi-hot vectors: $\text{SimL}(k_p, k_q) = \cos(y_p \odot \mathbf{w}, y_q \odot \mathbf{w})$
    - **Design Motivation**: Matching shallow MeSH labels (e.g., "Diseases") is less significant than matching deep labels (e.g., "Intracranial Hemorrhages"). Depth weighting focuses the model on meaningful, fine-grained matches.

2. **Hierarchical Multi-Label Contrastive Loss**:

    - **Function**: Prevents embedding collapse and maintains a discriminative structure.
    - **Mechanism**: Positive pairs are defined as document pairs with label similarity $\text{SimL} > \beta$, while negative pairs have no label overlap ($\text{SimL}=0$). In the contrastive loss, positive pairs are weighted by their label similarity: $\mathcal{L}_{\text{con}} = -\mathbb{E}_i \log[\text{SimL}(k_i, k_i^+) \cdot \exp(\text{SimE}(k_i, k_i^+)) / \sum_{k_j^-} \exp(\text{SimE}(k_i, k_j^-))]$. The threshold $\beta$ filters weakly associated pairs to reduce noisy supervision.
    - **Design Motivation**: A standalone regression loss may lead to embedding collapse; contrastive loss maintains discriminative capacity by pushing apart unrelated documents. Weighting by label similarity ensures that more relevant positive pairs contribute larger gradients.

3. **LoRA Parameter-Efficient Fine-Tuning**:

    - **Function**: Cost-effectively adapts the general-domain retriever to the biomedical domain.
    - **Mechanism**: The original BGE parameters are frozen, and low-rank adapters $W_{\text{adapted}}^{(l)} = W^{(l)} + B^{(l)} A^{(l)}$ are injected, training only 0.3% of the parameters.
    - **Design Motivation**: Full-parameter fine-tuning is prone to overfitting on small datasets and is computationally expensive. LoRA achieves domain adaptation while preserving general language understanding.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\text{mse}} + \lambda \mathcal{L}_{\text{con}}$, where $\lambda=0.1$ and $\beta=0.3$. The model is trained on 80,000 abstracts from BioASQ v2022, with the best checkpoint selected based on the TREC-CT 2022 validation set. Training and inference can be completed on a single A100 40GB GPU.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | BioHiCL-Base (0.1B) | BMRetriever-1B | bge-base (0.1B) |
|--------|------|------|----------|------|
| IR Average | nDCG@10 | **0.543** | 0.531 | 0.529 |
| NFCorpus | nDCG@10 | 0.379 | 0.344 | 0.368 |
| TREC-COVID | nDCG@10 | **0.812** | 0.840 | 0.798 |
| BIOSSES | Spearman | **0.896** | 0.858 | 0.860 |
| PubMedQA | Recall@1 | 0.893 | 0.810 | 0.856 |

### Ablation Study

| Configuration | IR Avg | Description |
|------|---------|------|
| BioHiCL-Base | **0.543** | Full model |
| w/o $\mathcal{L}_{\text{con}}$ | 0.528 | Removes contrastive loss; shows largest performance drop |
| w/o Ancestor Label | 0.538 | No ancestor node expansion |
| w/o $\mathcal{L}_{\text{mse}}$ | 0.537 | Removes regression loss |
| w/o Depth Weight | 0.541 | No depth weighting applied |
| w/o LoRA (Full FT) | 0.542 | LoRA performance is comparable to full-parameter fine-tuning |

### Key Findings
- The 0.1B BioHiCL-Base outperforms the 1B BMRetriever on average IR metrics, indicating that structured supervision can compensate for smaller model scales.
- Contrastive loss is the most critical component (IR average drops by 0.015 without it), highlighting the necessity of preventing embedding collapse.
- Fine-tuning BMRetriever with the BioHiCL method led to a significant performance drop (0.501→0.279), likely because replacing its original instruction-based training objective disrupted its retrieval-specialized embedding geometry.
- LoRA achieves performance comparable to full-parameter fine-tuning with only 0.3% of the parameters, validating the effectiveness of parameter-efficient methods for domain adaptation.

## Highlights & Insights
- **Utilizing the MeSH hierarchy as a graded supervision signal** is a natural and effective design. MeSH is an expert-curated standardized vocabulary that provides precise measures of semantic relationships. This approach of "borrowing existing structured knowledge for supervision" can be extended to any domain with hierarchical labels (e.g., legal classification, product categorization).
- **Depth-weighted label similarity** encodes the intuition that specific concepts are more important than abstract ones via a simple formula: $w_j = \log(d(c_j)+1)$.
- The high efficiency of the 0.1B model makes it suitable for large-scale practical deployment, offering a clear advantage over systems like BMRetriever or MedCPT that require over 1B parameters.

## Limitations & Future Work
- The model was trained only on 80,000 BioASQ abstracts, a much smaller scale than the click data used for MedCPT or the multi-task data for BMRetriever.
- The coverage and granularity of MeSH annotations are limited by the NLM-maintained label set; emerging concepts may be missing.
- The potential for combining MeSH hierarchical information with instruction-based retrieval has not yet been explored.
- Improvements on SCIDOCS remain limited (0.215→0.225), suggesting that cross-domain generalization needs further work.

## Related Work & Insights
- **vs MedCPT (Jin et al., 2023)**: MedCPT uses query-article clicks for contrastive learning, while BioHiCL provides finer-grained supervision via the MeSH hierarchy.
- **vs BMRetriever (Xu et al., 2024)**: BMRetriever uses large-scale multi-task training for a 1B model, whereas BioHiCL achieves comparable performance with 0.1B parameters and MeSH supervision, offering higher efficiency.
- **vs BiCA (Sinha et al., 2025)**: BiCA performs biomedical adaptation but does not utilize hierarchical label structures; BioHiCL complements this by adding the hierarchical dimension.

## Rating
- Novelty: ⭐⭐⭐ Contrastive learning with MeSH supervision is a natural combination, though researchers might find the core idea straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-task evaluation (IR, similarity, QA), detailed ablation studies, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described clearly and concisely, although the Related Work section is relatively brief.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_nlp/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
