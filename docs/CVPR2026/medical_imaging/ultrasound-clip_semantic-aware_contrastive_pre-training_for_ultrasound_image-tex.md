---
title: >-
  [Paper Note] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding
description: >-
  [CVPR 2026][Medical Imaging][ultrasound image-text pre-training] The core contribution of this paper is not merely an "ultrasound version of CLIP…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "ultrasound image-text pre-training"
  - "diagnostic taxonomy"
  - "semantic soft labels"
  - "heterogeneous graph encoding"
  - "cross-modal retrieval"
date: 2026-05-08
content_hash: 244dd8fe202f6bde
---

# Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding

**Conference**: CVPR 2026
**arXiv**: [2604.01749](https://arxiv.org/abs/2604.01749)  
**Code**: [https://github.com/ZJUDataIntelligence/Ultrasound-CLIP](https://github.com/ZJUDataIntelligence/Ultrasound-CLIP)  
**Area**: Medical Imaging / Ultrasound Multimodal Understanding
**Keywords**: ultrasound image-text pre-training, diagnostic taxonomy, semantic soft labels, heterogeneous graph encoding, cross-modal retrieval

## TL;DR

The core contribution of this paper is not merely an "ultrasound version of CLIP," but rather a redefinition of the image-text alignment objective around ultrasound-specific anatomical hierarchies and diagnostic attributes. The authors first construct the Ultrasonographic Diagnostic Taxonomy (UDT) and the large-scale US-365K dataset, then explicitly inject clinical relationships from text into contrastive learning via semantic soft labels and an attribute heterogeneous graph, yielding visual-language representations that are more genuinely "ultrasound-aware."

## Background & Motivation

Ultrasound is one of the most widely used imaging modalities in clinical practice, yet it is substantially underrepresented in existing medical vision-language pre-training frameworks. The authors report that ultrasound images account for less than 5% of mainstream medical image-text datasets and are nearly negligible in many large-scale collections. As a result, existing Medical CLIP models are predominantly shaped by the language distributions of CT, MRI, and pathology images.

This imbalance leads to two direct problems. First, ultrasound images are highly dependent on the acoustic properties of tissue; the same lesion can exhibit entirely different textures and echo patterns across different organs and scanning planes, making the standard CLIP paradigm of "natural-image captions plus binary positive-negative pairs" insufficient to capture such complex semantics. Second, ultrasound reports contain a large number of modality-specific diagnostic descriptors—such as echogenicity, margins, posterior acoustic phenomena, and vascularity—whose attributes are structurally interrelated in ways that a generic text encoder cannot automatically learn.

The paper thus identifies two root causes:

- **Semantic ambiguity**: The same lesion can be described in multiple ways, and binary contrastive learning treats "similar but not identical" samples as hard negatives, introducing noise.
- **Absence of structural priors**: Ultrasound diagnosis is not simple caption matching but a joint judgment over multiple attributes whose interdependencies should be encoded.

The authors' starting point is pragmatic: given that existing models lack data, taxonomy, and structure, the paper first addresses all three—constructing the large-scale ultrasound-specific dataset US-365K, defining the diagnostic knowledge taxonomy UDT, and then explicitly leveraging this knowledge in the training objective rather than expecting the model to infer it from weak text supervision.

## Method

### Overall Architecture

Ultrasound-CLIP retains the dual-encoder framework of CLIP: image encoder $f_\theta$ processes ultrasound images and text encoder $g_\phi$ encodes ultrasound descriptive text. On top of this, two domain-specific enhancement pathways are introduced:

- A **UDAF-guided heterogeneous graph encoder** that encodes diagnostic labels and attribute relationships into a structured graph and fuses the result into the text representation.
- A **UDAF-based semantic prior** that computes continuous-valued semantic similarity between any two samples in a batch, replacing the binary positive-negative pairing.

The final training objective combines the standard CLIP contrastive loss with a semantic alignment loss: $\mathcal{L} = \mathcal{L}_\text{CLIP} + \lambda \cdot \mathcal{L}_\text{semantic}$. This enables the model to retain cross-modal alignment capability while not being forced by binary labels to ignore fine-grained clinical similarity.

### Key Designs

1. **UDT and US-365K Data Foundation**

    - **Function**: Provides a large-scale, structured, modality-specific training semantic space for ultrasound image-text pre-training.
    - **Mechanism**: The authors propose UDT (Ultrasonographic Diagnostic Taxonomy), consisting of two components. `UHAT` handles the anatomical hierarchy, organizing 9 body systems and 52 organs into a tree structure; `UDAF` defines nine clinical diagnostic attribute categories: body system, organ, diagnosis, shape, margins, echogenicity, internal characteristics, posterior acoustic phenomena, and vascularity. Based on this taxonomy, 364,365 image-text pairs covering 11,676 clinical cases are constructed from five publicly available sources.
    - **Design Motivation**: If training data does not encompass the ultrasound-specific attribute space, refining the contrastive loss afterward merely treats symptoms. By first using the taxonomy to normalize raw ultrasound text, the authors address the fundamental question of "what exactly is being pre-trained on."

2. **Semantic Soft Label Prior**

    - **Function**: Injects into the training objective the fact that samples are not binary opposites but exhibit varying degrees of semantic overlap.
    - **Mechanism**: For each attribute task $k$ in UDAF, a label similarity matrix $S^{(k)}$ is maintained. For samples $i$ and $j$, the average similarity between their label sets is computed for each task and then averaged across all tasks to obtain the overall soft similarity $\tilde{s}_{ij}$. Consequently, a batch no longer has a diagonal of ones and zeros elsewhere, but instead yields a continuous-valued soft prior matrix.
    - **Design Motivation**: Ultrasound text frequently contains "semantically close but differently expressed" descriptions. Hard negative constraints would force the model to push apart clinically similar cases. Soft labels more naturally reflect similarities among cases sharing the same organ or diagnostic attributes.

3. **UDAF Heterogeneous Graph Encoder**

    - **Function**: Explicitly encodes attribute relationships within ultrasound text, rather than averaging attribute tokens through a language model.
    - **Mechanism**: Each sample's textual labels are converted into a heterogeneous graph with diagnosis nodes and attribute nodes, forming a bipartite graph with full connectivity between the two node types. A lightweight graph network produces node representations, and attention pooling yields a graph summary vector $g_i$. Multi-head attention then allows the original text vector $t_i$ to query the graph vector, followed by a gated residual connection to produce the enhanced text representation $\tilde{t}_i$.
    - **Design Motivation**: The lesion type, echogenicity, margins, and vascularity of the same ultrasound case exhibit co-occurring structural relationships that differ fundamentally from word co-occurrence in natural language. An explicit graph structure enables the model to perform something closer to clinical reasoning rather than mere caption alignment.

### Loss & Training

The training objective comprises two components:

- **Contrastive alignment loss**: The standard symmetric CLIP loss that pulls image and corresponding text representations closer.
- **Semantic loss**: A constraint applied to the predicted cross-modal similarity matrix, encouraging it to approximate the UDAF prior both numerically and distributionally via a KL divergence term.

The two components are combined because $\mathcal{L}_\text{CLIP}$ alone ignores the continuous nature of ultrasound semantics, while relying solely on the semantic matrix would sacrifice cross-modal retrieval performance. Their combination enables the model to both retrieve effectively and capture attribute-level details.

## Key Experimental Results

### Main Results

The paper first evaluates general CLIP, medical CLIP variants, and several ablations of the proposed model on nine diagnostic attribute classification tasks within US-365K. The full model achieves the highest average accuracy and average recall by a clear margin.

| Method | AvgAcc | AvgRecall | Notes |
|---|---:|---:|---|
| CLIP | 13.29 | 28.75 | General CLIP; barely understands ultrasound semantics |
| MedCLIP | 25.37 | 31.88 | Medical pre-training; insufficient ultrasound coverage |
| BiomedCLIP | 33.81 | 35.11 | Stronger medical baseline |
| Ultrasound-CLIP-Ds+g | 50.84 | 52.87 | Basic combination of semantic loss and graph encoding |
| Ultrasound-CLIP-Ds | 48.62 | 53.12 | Semantic prior only |
| Ultrasound-CLIP-Dg | 49.87 | 55.12 | Graph structure enhancement only |
| **Ultrasound-CLIP** | **59.61** | **61.08** | Both modules combined; best overall performance |

The full model surpasses the strongest medical baseline BiomedCLIP by more than 25 points in AvgAcc, indicating that the gain reflects a fundamental change in how the modality is understood rather than a minor improvement.

### Ablation Study

The authors evaluate image-text retrieval Recall@K on the US-365K test set, revealing the complementary roles of the graph structure and semantic prior.

| Method | I2T R@5 | I2T R@10 | I2T R@50 | T2I R@5 | T2I R@10 | T2I R@50 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 0.1420 | 0.2451 | 0.6306 | 0.1662 | 0.2783 | 0.6767 |
| PMC-CLIP | 0.1808 | 0.3011 | 0.7215 | 0.1814 | 0.3038 | 0.7312 |
| BiomedCLIP | 0.1788 | 0.2979 | 0.7029 | 0.1864 | 0.3089 | 0.7206 |
| Ultrasound-CLIP-Ds | 0.1568 | 0.2683 | 0.6692 | 0.1550 | 0.2659 | 0.6707 |
| Ultrasound-CLIP-Dg | 0.2147 | 0.3444 | 0.7638 | 0.2147 | 0.3520 | 0.7774 |
| **Ultrasound-CLIP** | **0.2359** | **0.3745** | **0.7909** | **0.2383** | **0.3781** | **0.8022** |

Notably, the graph-only variant `Dg` already achieves strong retrieval performance, demonstrating that the heterogeneous graph directly improves text representation quality. Adding the semantic loss in the full model yields further gains, confirming that soft label priors benefit retrieval ranking and not merely classification.

### Key Findings

- Compared with general and medical CLIP models, the primary gains come first from the data and taxonomy rather than from a modified loss function alone.
- `Dg` and `Ds` are each individually effective, but the full model is significantly superior, indicating that structural priors and semantic soft supervision address errors at different levels.
- The paper also reports strong downstream transfer results: linear probing achieves an average accuracy of 75.40%, full fine-tuning reaches 84.23% on average, and 92.13% on the Breast dataset, demonstrating that the learned representations generalize well beyond the in-house benchmark.
- Patient-level data splitting is critical. Due to the high visual similarity among ultrasound images, failure to enforce strict patient-level splits would substantially overestimate model performance. The authors handle this rigorously.

## Highlights & Insights

- The most significant contribution of this paper is that it does not simply "train CLIP on ultrasound data" but redefines the semantic coordinate system for ultrasound. The value of UDT extends beyond this work; subsequent ultrasound multimodal research can directly inherit this hierarchical label framework.
- Semantic soft labels are particularly well suited to medical settings. Synonymy, relatedness, and partial overlap in medical text are far more prevalent than in natural image captions, making hard negative constraints inherently disadvantageous.
- The heterogeneous graph encoder does not replace the text encoder but provides the text vector with an "attribute-relationship reference." This lightweight grafting approach is more practical than designing an entirely new, large-scale medical language model from scratch.
- The dataset construction pipeline also deserves attention. Rather than simply collecting image-text pairs, the authors use UDT to drive label extraction and normalization, making US-365K simultaneously suitable as both a pre-training corpus and an evaluation benchmark.

## Limitations & Future Work

- Although US-365K is already large-scale for the ultrasound domain, it primarily originates from public case repositories and educational resources; the noise distribution, equipment variability, and reporting styles encountered in real hospital workflows may be considerably more complex.
- UDAF currently covers nine attribute categories, which is highly practical, but cannot exhaustively represent all subspecialty ultrasound scenarios. More dynamic diagnostic information such as echocardiography and interventional ultrasound has yet to be incorporated.
- The model currently centers on static image-text pairs. Video frames, probe motion, and multi-plane joint assessment—which are common in clinical ultrasound practice—have not been genuinely exploited within this framework.
- The semantic prior matrix relies on manually designed or rule-based label similarity, which ensures controllability but may limit the model's ability to express more implicit clinical similarity relationships.
- A natural next step would be to extend UDT into a cross-task knowledge graph, allowing retrieval, classification, report generation, and visual question answering to share a unified ultrasound semantic foundation.

## Related Work & Insights

- **vs. General CLIP**: The gap between general CLIP and ultrasound understanding is not merely domain shift; the entire attribute space is misaligned. This paper addresses that gap at the ontology level.
- **vs. Medical CLIP**: Many medical CLIP methods target broader radiology pre-training with minimal ultrasound coverage, resulting in representations that are "wide but shallow" in medical semantics. This paper achieves "narrow but deep" coverage instead.
- **vs. Specialist Small-Data Ultrasound Models**: Models such as Fetal-CLIP or breast-specific models may excel on their respective tasks but offer limited coverage; this paper emphasizes a unified pre-training backbone across anatomical regions.
- A broader research insight is that medical multimodal models often require "knowledge structure" as a prerequisite before scaling model capacity. Taxonomy, graph structure, and soft labels tend to be more effective than blindly increasing parameter count in the context of specialty medical imaging.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The dataset, taxonomy, graph structure, and semantic loss together form a complete solution rather than a single incremental modification.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Pre-training tasks, retrieval, and transfer tasks are all covered with strong evidence, though additional external hospital validation would further strengthen the claims.
- **Writing Quality**: ⭐⭐⭐⭐ The logic is clear and the transitions among problem formulation, data construction, method, and experiments are well connected.
- **Value**: ⭐⭐⭐⭐⭐ Highly significant for the ultrasound multimodal direction; US-365K and UDT themselves carry long-term reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)
- [\[ICML 2026\] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](../../ICML2026/medical_imaging/meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)
- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](../../AAAI2026/medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)

</div>

<!-- RELATED:END -->
