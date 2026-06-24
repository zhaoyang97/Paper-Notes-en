---
title: >-
  [Paper Note] Interactive Medical Image Analysis with Concept-based Similarity Reasoning
description: >-
  [CVPR 2025][Medical Imaging][Explainable Medical Imaging] This paper proposes the CSR (Concept-based Similarity Reasoning) network, which performs classification reasoning by learning the similarity of concept prototypes in local image regions. It simultaneously supports interactive intervention by clinicians across spatial and conceptual levels during both training and testing, outperforming existing explainable methods by up to a 4.5% F1 gain across three medical datasets.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Explainable Medical Imaging"
  - "Concept Bottleneck Models"
  - "Prototype Learning"
  - "Spatial Interaction"
  - "Similarity Reasoning"
date: 2026-05-08
content_hash: 89c9520ba45e1963
---

# Interactive Medical Image Analysis with Concept-based Similarity Reasoning

**Conference**: CVPR 2025  
**arXiv**: [2503.06873](https://arxiv.org/abs/2503.06873)  
**Code**: [https://github.com/tadeephuy/InteractCSR](https://github.com/tadeephuy/InteractCSR)  
**Area**: Medical Image / Explainable AI  
**Keywords**: Explainable Medical Imaging, Concept Bottleneck Models, Prototype Learning, Spatial Interaction, Similarity Reasoning

## TL;DR
This paper proposes the CSR (Concept-based Similarity Reasoning) network, which performs classification reasoning by learning the similarity of concept prototypes in local image regions. It simultaneously supports interactive intervention by clinicians across spatial and conceptual levels during both training and testing, outperforming existing explainable methods by up to a 4.5% F1 gain across three medical datasets.

## Background & Motivation

1. **Background**: Explainable medical image analysis mainly follows two paths—Concept Bottleneck Models (CBMs) predict interpretable concepts and then use them for classification, while prototype approaches (such as ProtoPNet) learn patch prototypes of training images and classify based on similarity scores.
2. **Limitations of Prior Work**: CBMs only provide image-level conceptual explanations and cannot localize the specific regions where concepts are activated. Prototype-based methods offer patch-level explanations but require post-hoc analysis to associate prototypes with semantic concepts—a process particularly challenging in medical imaging due to subtle visual differences.
3. **Key Challenge**: Conceptual explainability (knowing *what* the concept is) and spatial localization (knowing *where* it is) are currently decoupled—no existing method simultaneously provides "patch-level + conceptually explainable" explanations.
4. **Goal**: (a) How to achieve patch-level intrinsic concept interpretability? (b) How to enable clinicians to interact directly with the model in the spatial dimension? (c) How to maintain diagnostic accuracy without sacrificing interpretability?
5. **Key Insight**: Inspired by radiologists referencing atlases—clinicians diagnose by comparing suspicious regions with known typical cases.
6. **Core Idea**: Learn concept prototypes and compute the cosine similarity between prototypes and each patch of the input image to generate 2D similarity maps as explanations. This supports clinicians in correcting model decisions by drawing bounding boxes (spatial interaction) and rejecting concepts (conceptual interaction).

## Method

### Overall Architecture
CSR comprises three components: (1) Concept model—extracts concept features and generates concept activation maps; (2) Feature projector P—enhances the compactness and generalization of concept features through contrastive learning, learning concept prototypes; (3) Task head H—predicts the target category from the concept similarity score vector. During inference, for each concept prototype, its cosine similarity map with each patch of the input image is computed. The maximum value is taken as the similarity score for that concept, and all scores are aggregated into a vector fed to the classification head.

### Key Designs

1. **Concept Prototype Learning + Multi-Prototype Contrastive Loss**:

    - **Function**: Learn compact and generalizable concept prototypes to accurately localize corresponding concepts in new images.
    - **Mechanism**: First, the Concept model is trained for multi-label classification to generate concept activation maps $\text{cam}^k$. Then, local concept vectors $v^k = \sum_{H,W} \text{softmax}_{h,w}(\text{cam}^k) \cdot \mathbf{f}$ are obtained via spatial softmax weighted summation. These are then projected into a compact space $v' = P(v)$ by the projector P. Through a multi-prototype contrastive loss, M prototypes $\{p^{k_m}\}$ are learned for each concept—pulling concept vectors towards the nearest prototype of the same concept while pushing away those of other concepts. Probability assignment uses $q_m = \text{softmax}_m(\gamma \langle p^{\tilde{k}_m}, v'^{\tilde{k}}_i \rangle)$, and the total similarity is the weighted sum.
    - **Design Motivation**: Directly using local concept vectors $v^k$ generalizes poorly to new images (as shown in Figure 4, where the vector $v$ for the pacemaker concept fails to localize in new images). The contrastive learning + multi-prototype strategy addresses the cross-sample generalization of concept features.

2. **Spatial-level Interaction**:

    - **Function**: Allow clinicians to guide the model to focus on or ignore specific regions by drawing positive/negative bounding boxes.
    - **Mechanism**: Clinicians draw positive bounding boxes $\text{bb}^+$ and negative bounding boxes $\text{bb}^-$ to generate an importance map $A(h,w)$: 1 inside positive boxes, 0 inside negative boxes, and $\alpha \in [0,1)$ elsewhere. The importance map is element-wise multiplied with all concept similarity maps $[\hat{S}] = A \odot [S]$ to recompute similarity scores and predictions. Due to the max-pooling operation, magnifying important regions increases their likelihood of being selected, while zeroing out pseudo-correlated regions eliminates their influence.
    - **Design Motivation**: Deep learning models often capture "Clever-Hans" style spurious correlations. Clinicians can directly instruct the model "where to look / where not to look" without specifying "what to look for", making the interaction natural and intuitive.

3. **Concept-level Interaction + Interaction during Training**:

    - **Function**: Allow clinicians to reject incorrect concepts and prune low-quality prototypes.
    - **Mechanism**: **Test-time conceptual interaction**—upon reviewing concept similarity scores, if clinicians find a concept does not exist, they can reject it. The model sets all corresponding $s^{k_m}$ to zero and re-predicts. **Training-time interaction**—clinicians inspect the concept prototype atlas $\{\mathcal{I}(p^{k_m})\}$ to prune low-quality prototypes learned from spurious correlations, eliminating the Clever-Hans effect at the source.
    - **Design Motivation**: Conceptual interaction in CBMs has been shown to significantly boost performance. This work extends it to more flexible multi-level interactions, combining spatial and conceptual interactions for comprehensive doctor-in-the-loop validation.

### Loss & Training
Training is performed in stages: (1) Train the Concept model for multi-label concept classification using BCE; (2) Train the projector P and concept prototypes using multi-prototype contrastive loss $\ell_{\text{con-m}}$ with a margin $\delta$ to expand decision boundaries; (3) Train the task head H to predict the target from similarity scores using CE.

## Key Experimental Results

### Main Results

| Dataset | Method | F1 ↑ | No. of Prototypes ↓ | No. of Explanations ↓ |
|--------|------|------|---------|---------|
| TBX11K | CBM (joint) | 88.6 | - | 14 |
| TBX11K | ProtoPNet | 94.1 | 3000 | 3000 |
| TBX11K | PIP-Net | 94.0 | 768 | 158 |
| TBX11K | **CSR** | **94.4** | 1400 | **14** |
| VinDr-CXR | CBM (joint) | 50.1 | - | 14 |
| VinDr-CXR | PIP-Net | 45.1 | 768 | 9 |
| VinDr-CXR | **CSR** | **54.6** | 1400 | **14** |
| ISIC | PIP-Net | 69.9 | 768 | 90 |
| ISIC | **CSR** | **71.5** | 400 | **4** |

### Ablation Study — Pointing Game Localization Accuracy

| Method | PG Hit Rate ↑ |
|------|---------------|
| ProtoPNet | 8.8% |
| ProtoTree | 7.8% |
| PIP-Net | 19.5% |
| CBM | 55.1% |
| CSR | 60.9% |
| **CSR (refined)** | **79.5%** |

### Key Findings
- CSR outperforms all explainable baselines on three datasets, achieving a 4.5% F1 absolute gain on VinDr-CXR compared to CBM.
- Extremely concise explanation overhead: CSR requires only 14 explanations (the number of concepts) per prediction, whereas ProtoPNet requires 3000.
- Promising Pointing Game accuracy of 60.9% (improving to 79.5% after post-training refinement), vastly outperforming ProtoPNet (8.8%) and PIP-Net (19.5%), demonstrating accurate concept localization.
- Interactive training (clinicians refining the prototype atlas) slightly drops F1 (94.4 -> 94.0) but dramatically raises localization accuracy from 60.9% to 79.5%.
- Contrastive learning on projector P drastically improves the compactness and cross-sample generalization of the concept feature space.

## Highlights & Insights
- **Concept prototype as an interpretable local comparator** is an elegant design: it removes the need for post-hoc analysis to decipher prototype meanings. Each prototype is naturally bound to a semantic concept while offering patch-level localization. This "inherent-by-design" interpretability is more reliable than post-hoc explanations.
- **Practical Spatial Interaction Design**: Clinicians do not need expert ML knowledge; they can simply draw bounding boxes on images to tell the model where to look or ignore, and the model automatically adjusts its prediction. This interaction method aligns well with radiologists' default workflows.
- **Synergy of Conceptual and Spatial Interaction**: The combination enables handling complex scenarios. For instance, when a positive bounding box region is incorrectly associated with a non-existent concept, clinicians can simultaneously reject the concept while keeping the positive box, guiding the model back to the correct path.

## Limitations & Future Work
- Concepts must be predefined and require concept-level annotations, which are highly expensive.
- Currently relies entirely on visual features, leaving multimodal concept definitions (e.g., via CLIP, LLM) unexplored.
- Lacks an automated strategy for selecting the number of multiple prototypes M.
- The spatial interaction parameter $\alpha$ requires manual tuning, and different concepts may require different values.
- Future work could explore combining CSR's concept prototypes with foundation models to reduce reliance on dense concept annotations.

## Related Work & Insights
- **vs CBM**: CBMs only provide image-level concept explanations, whereas CSR provides patch-level localization. CSR also delivers an absolute F1 improvement of 4.5% on VinDr-CXR.
- **vs ProtoPNet/PIP-Net**: Prototype-based methods require post-hoc analysis to associate semantics, whereas CSR's concept prototypes are naturally interpretable. Additionally, ProtoPNet's Pointing Game accuracy is only 8.8% compared to CSR's 60.9%.
- **vs PHCBM**: PHCBM transfers concepts via CLIP, but the domain gap between medical and natural images limits direct transfer performance. Learning concept representations directly from data in CSR is better suited for medical applications.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The integration of concept bottlenecks, prototype learning, and spatial interaction is highly novel, addressing a critical gap in explainable clinical AI.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across three datasets are comprehensive, and the Pointing Game evaluates localization accuracy, though validation on larger-scale datasets is still needed.
- Writing Quality: ⭐⭐⭐⭐⭐ The interaction example in Figure 1 is highly intuitive, and the methodology is clearly structured.
- Value: ⭐⭐⭐⭐⭐ Carries significant practical value for improving the clinical trust and usability of AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] Boltzmann Attention Sampling for Image Analysis with Small Objects](boltzmann_attention_sampling_for_image_analysis_with_small_objects.md)
- [\[ICCV 2025\] SIC: Similarity-Based Interpretable Image Classification with Neural Networks](../../ICCV2025/medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)
- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)

</div>

<!-- RELATED:END -->
