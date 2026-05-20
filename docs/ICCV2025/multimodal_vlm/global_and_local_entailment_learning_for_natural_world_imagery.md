---
title: >-
  [Paper Note] Global and Local Entailment Learning for Natural World Imagery
description: >-
  [ICCV 2025][Multimodal VLM][Hierarchical Representation Learning] This paper proposes Radial Cross-Modal Embeddings (RCME), a framework that explicitly models the transitivity of entailment relations to learn hierarchica…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Hierarchical Representation Learning"
  - "Entailment Learning"
  - "Vision-Language Models"
  - "Biological Taxonomy"
  - "Partial Order Relations"
date: 2026-05-08
content_hash: 690aa91f99ac0e82
---

# Global and Local Entailment Learning for Natural World Imagery

**Conference**: ICCV 2025
**arXiv**: [2506.21476](https://arxiv.org/abs/2506.21476)  
**Code**: [GitHub](https://vishu26.github.io/RCME/index.html)  
**Area**: Multimodal VLM
**Keywords**: Hierarchical Representation Learning, Entailment Learning, Vision-Language Models, Biological Taxonomy, Partial Order Relations

## TL;DR

This paper proposes Radial Cross-Modal Embeddings (RCME), a framework that explicitly models the transitivity of entailment relations to learn hierarchical representations in vision-language models. RCME enables inference at arbitrary taxonomic ranks on the Tree of Life and achieves state-of-the-art performance on hierarchical classification and retrieval tasks.

## Background & Motivation

Biological taxonomy is inherently hierarchical: Kingdom → Phylum → Class → Order → Family → Genus → Species. Yet existing vision-language foundation models such as BioCLIP, BioTroveCLIP, and TaxaBind, despite strong species recognition performance, suffer from the following limitations:

**Inability to exploit label hierarchy**: These models can only perform inference at the finest granularity (species level) using a fixed taxonomic label database, and cannot operate at arbitrary ranks.

**Neglect of transitivity constraints**: Prior entailment learning methods (e.g., Radial Embeddings) attempt to encode hierarchical relations in embedding space but fail to explicitly enforce transitivity — i.e., if *Mammalia* is entailed by *Chordata*, and *Carnivora* is entailed by *Mammalia*, then *Carnivora* should be entailed by *Chordata*.

**Pressing practical needs**: A large number of species on Earth remain undescribed; annotating specimens to the species level is costly and requires expert knowledge; taxonomic labels may change as new species are discovered or errors are corrected.

The authors argue that learning hierarchical representations is essential for understanding the Tree of Life — enabling reasoning at any taxonomic rank and assisting biologists in grouping and routing specimens.

## Method

### Overall Architecture

RCME (Radial Cross-Modal Embeddings) is a fine-tuning framework for vision-language models consisting of three core loss functions: Global Entailment Loss, Local Entailment Loss, and Cross-Modal Alignment Loss. The framework is built on OpenCLIP ViT-B/16 and fine-tunes both the visual and text encoders jointly.

### Key Designs

1. **Global Entailment Learning**

   The core idea is to distinguish between *local entailment* and *global entailment*:
   - **Local entailment**: A subconcept is entailed by its immediate parent (e.g., *Carnivora* entailed by *Mammalia*).
   - **Global entailment**: The transitivity condition holds across all possible sub-hierarchies.

   Mathematically, the transitivity constraint requires:

   $\mathcal{S}(T_{j-1}^i, T_{j+1}^i) \geq \mathcal{S}(T_{j-1}^i, T_j^i) \cdot \mathcal{S}(T_j^i, T_{j+1}^i)$

   where $\mathcal{S}$ is a similarity measure defined via the exterior angle. The global entailment loss adopts a margin-based formulation:

   $\mathcal{L}_{GE}(i,j;\alpha) = \max(0, \Xi(T_{j-1}^i, T_{j+1}^i) - \arccos(\mathcal{S}(T_j^i, T_{j+1}^i) \cdot \mathcal{S}(T_{j-1}^i, T_j^i)) + \alpha)$

   **Design Motivation**: This ensures that fine-grained concepts are progressively projected further from the entailment root and confined within smaller conical sub-regions (Lemma 1), establishing a direct correspondence between semantic granularity and distance from the root.

2. **Cross-Modal Alignment**

   Unlike Radial Embeddings, which employ a prior-preservation loss (fine-tuning only the text encoder while freezing the visual encoder), RCME proposes a cross-modal alignment term that jointly fine-tunes both encoders:

   $\mathcal{L}_{CMA}(i) = -\log \frac{e^{\langle T_N^i, I^i \rangle}}{\sum_{m=1}^B e^{\langle T_N^m, I^m \rangle} + e^{\langle T_N^i, I^i \rangle}}$

   This alignment loss is computed only at the finest granularity (species level). **Design Motivation**: It maintains visual-textual alignment during fine-tuning, preventing the degradation that arises when the text encoder is updated in isolation.

3. **Hierarchical Hard Negative Mining**

   A taxonomy-aware hard negative sampling strategy is proposed for the local entailment objective. Given a positive sample's label at a particular rank, sibling nodes are identified by matching all ancestor labels at higher ranks, and a descendant of a sampled sibling is selected as the negative example. This process is applied recursively at each taxonomic level.

   **Design Motivation**: This encourages the model to learn fine-grained distinctions among species sharing a common ancestor, rather than simply separating entirely unrelated taxa.

### Loss & Training

The final objective combines three terms:

$$\mathcal{L}_{RCME}(i,k;\alpha) = \mathcal{L}_{GLE}(i,k;\alpha) + \beta \mathcal{L}_{CMA}(i)$$

where $\mathcal{L}_{GLE}$ integrates the global and local entailment losses. Training uses the TreeofLife-10M dataset with "Eukarya" as the entailment root, conducted on 2 NVIDIA H100 GPUs. Two variants are provided: RCME initialized from OpenCLIP, and RCME$^{FT}$ fine-tuned from a BioCLIP checkpoint.

## Key Experimental Results

### Main Results

**Hierarchical Retrieval Metrics (iNaturalist-2021)**:

| Model | Kendall's τ_d | Precision | Recall | F1 |
|-------|---------------|-----------|--------|----|
| CLIP | 0.737 | 0.047 | 0.054 | 0.050 |
| BioCLIP | 0.012 | 0.115 | 0.153 | 0.131 |
| Radial Emb. | 0.521 | 0.147 | 0.196 | 0.168 |
| ATMG | 0.571 | 0.343 | 0.130 | 0.189 |
| **RCME (Ours)** | **0.993** | **0.458** | **0.572** | **0.508** |

**Zero-Shot Classification (iNaturalist-2021, averaged across ranks)**:

| Model | Kingdom | Phylum | Class | Order | Species | Avg. |
|-------|---------|--------|-------|-------|---------|------|
| BioCLIP | 36.96 | 32.02 | 19.97 | 31.43 | 68.24 | 39.13 |
| ATMG | 99.12 | 86.79 | 73.03 | 33.89 | 39.52 | 61.89 |
| **RCME** | **88.18** | **84.81** | **55.22** | **41.82** | **73.52** | **65.09** |

RCME surpasses ATMG by +3.2% on average accuracy and does not exhibit the severe performance collapse at fine-grained ranks (genus, species) observed in ATMG.

### Ablation Study

| Configuration | Species | Avg. | Notes |
|---------------|---------|------|-------|
| $\mathcal{L}_{LE}$ only (BioCLIP baseline) | 68.24 | 39.13 | Local entailment |
| $\mathcal{L}_{LE} + \mathcal{L}_{prior}$ | 68.23 | 41.50 | + Prior preservation |
| $\mathcal{L}_{LE} + \mathcal{L}_{CMA}$ | 69.43 | 43.65 | + Cross-modal alignment |
| $\mathcal{L}_{LE} + \mathcal{L}_{GE} + \mathcal{L}_{prior}$ | 71.28 | 62.97 | + Global entailment |
| $\mathcal{L}_{LE} + \mathcal{L}_{GE} + \mathcal{L}_{CMA}$ (full) | **73.52** | **65.09** | Best |

**Hard Negative Mining**: Introduces gains of +2.87% on iNat-2021 (62.22→65.09) and +3.52% on BioCLIP-Rare.

### Key Findings

- Adding the global entailment objective yields an average improvement of +21.47% (41.50→62.97), making it the most critical component.
- Plant taxa exhibit lower performance at the class/family/order levels, likely attributable to convergent evolution, frequent hybridization, and annotation noise.
- In image-to-image retrieval, RCME leads the second-best model by +8.58% at the species level, demonstrating that the framework also improves intra-modal representations.
- UMAP visualizations confirm that RCME successfully preserves the partial order structure of taxonomic labels.

## Highlights & Insights

- **Theory-driven design**: The global entailment loss is derived from the transitivity condition of entailment learning, supported by rigorous mathematical proofs.
- **Discovery of anomalous patterns in taxonomy**: The model reveals unusual behavior in plant taxonomy (intermediate ranks underperforming fine-grained ranks), offering potential insights for improving taxonomic systems.
- **Strong generalizability**: Experiments on the HierarCaps dataset demonstrate that RCME's objective functions transfer to other domains.
- **Joint dual-encoder fine-tuning**: Replacing the prior-preservation loss with the cross-modal alignment loss simplifies the training pipeline while improving performance.

## Limitations & Future Work

- Validation is limited to the ViT-B/16 architecture; the effect of scaling to larger models remains unexplored.
- The sensitivity of results to the choice of entailment root ("Eukarya") is not thoroughly discussed.
- The performance gap on plant taxonomy remains unresolved and warrants further investigation.
- Extending the approach to hyperbolic space, which more naturally accommodates hierarchical structures, is a promising direction.

## Related Work & Insights

This work builds upon cone-based entailment by Ganea et al. and radial embeddings by Alper et al. The central contribution lies in addressing the failure of Radial Embeddings to enforce partial order relations. The transitivity-constrained loss functions proposed here are directly transferable to other domains requiring hierarchical representations, such as product categorization and document organization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The global entailment loss offers a theoretical contribution, though the overall framework remains an improvement at the loss function level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four categories of experiments, ablation studies, generalization experiments, and UMAP visualizations — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and notation is consistent throughout.
- **Value**: ⭐⭐⭐⭐ Practically significant for ecological computer vision and hierarchical representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)

</div>

<!-- RELATED:END -->
