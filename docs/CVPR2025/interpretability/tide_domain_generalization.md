---
title: >-
  [Paper Note] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction
description: >-
  [CVPR 2025][Interpretability][Domain Generalization] This paper proposes TIDE, which leverages diffusion models and LLMs to automatically generate concept-level saliency map annotations to train locally interpretable domain generalization models. During testing, concept signatures are utilized for prediction correction, yielding an average performance improvement of 12% over SOTA on four standard DG benchmarks.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Domain Generalization"
  - "Concept Saliency Map"
  - "Test-time Correction"
  - "Single-source Domain"
date: 2026-05-08
content_hash: 60e5a514a371242e
---

# TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction

**Conference**: CVPR 2025  
**arXiv**: [2411.16788](https://arxiv.org/abs/2411.16788)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Domain Generalization, Interpretability, Concept Saliency Map, Test-time Correction, Single-source Domain

## TL;DR
This paper proposes TIDE, which leverages diffusion models and LLMs to automatically generate concept-level saliency map annotations to train locally interpretable domain generalization models. During testing, concept signatures are utilized for prediction correction, yielding an average performance improvement of 12% over SOTA on four standard DG benchmarks.

## Background & Motivation
1. **Background**: Single-source domain generalization (SSDG) is the most challenging form of domain generalization, where training is conducted on only a single source domain and models must generalize to unseen target domains. Mainstream methods rely on extensive data augmentation to simulate different domains.
2. **Limitations of Prior Work**: Augmentation strategies struggle to handle semantic-level domain shifts (such as variations in background and perspective) because models tend to learn global features rather than cross-domain invariant local concepts. Existing methods exhibit mediocre performance on datasets dominated by semantic shifts, such as VLCS.
3. **Key Challenge**: The regions focused on by models are unstable under domain shifts—for example, focusing on the background instead of cross-domain invariant local concepts like the beak and feathers when identifying birds. Existing DG datasets lack concept-level fine-grained annotations.
4. **Goal**: To force models to focus on class-specific local concepts (e.g., the beak of a bird, the eyes of a person) during training, enabling them to align their focus correctly even under domain shifts.
5. **Key Insight**: Utilizing the cross-attention maps of diffusion models and LLMs to automatically generate concept-level saliency map annotations without manual labeling.
6. **Core Idea**: Automatic concept annotation + concept saliency alignment training + test-time concept signature correction.

## Method

### Overall Architecture
GPT-3.5 generates key concept lists for each class $\rightarrow$ SD v2.1 synthesizes exemplar images + cross-attention concept maps $\rightarrow$ DIFT method transfers saliency maps to real images $\rightarrow$ Train TIDE model (Classification CE + Concept CE + Concept Saliency Alignment Loss + Local Concept Contrastive Loss) $\rightarrow$ Store concept signatures $\rightarrow$ Iterative correction at test time.

### Key Designs

1. **Automatic Concept-level Annotation Pipeline**:

    - **Function**: Automatically generate concept-level saliency maps for DG datasets without manual labor.
    - **Mechanism**: (1) Use GPT-3.5 to generate stable and distinctive concept lists for each class (e.g., cat $\rightarrow$ whiskers, ears, eyes); (2) Synthesize exemplar images using prompts built from these concepts and extract SD cross-attention maps as concept saliency maps; (3) Apply the DIFT (Diffusion Feature Transfer) method to compute pixel-level cosine similarity, transferring the concept saliency maps from synthesized images to real images. Finally, filter "crucial concepts" that genuinely affect classification based on GradCAM overlap.
    - **Design Motivation**: Existing DG datasets lack concept annotations, and manual labeling is costly and unscalable. The feature space of diffusion models naturally provides semantically rich pixel-level correspondences.

2. **Concept Saliency Alignment Loss (CSA Loss)**:

    - **Function**: Ensure that the model focuses on the correct image regions when predicting concepts.
    - **Mechanism**: Compute the $L_2$ distance between the model's GradCAM map $S_x^k$ for concept prediction and the ground-truth saliency map $G_x^k$: $\mathcal{L}_{\text{CSA}} = \frac{1}{|\mathcal{K}_c|}\sum_{k\in\mathcal{K}_c}\|S_x^k - G_x^k\|_2^2$.
    - **Design Motivation**: Simply using concept classification loss cannot guarantee that the model focuses on the correct regions—the model might make correct predictions using incorrect features. CSA explicitly constrains the attention position.

3. **Local Concept Contrastive Loss (LCC Loss)**:

    - **Function**: Promote the invariance of concept-level features across different domains/augmentations.
    - **Mechanism**: Obtain concept-level features $f_x^k(l) = \sum_{i,j} G_x^k \cdot F_x(i,j,l)$ by weighted pooling of the feature maps with the GT saliency map $G_x^k$. A triplet loss is utilized to pull augmented versions of the same concept closer while pushing different concepts farther apart: $\mathcal{L}_{\text{LCC}} = \max(0, d(f_x^k, f_{x^+}^k) - d(f_x^k, f_{x^-}^{k'}) + \alpha)$.
    - **Design Motivation**: While CSA ensures positional correctness, LCC further guarantees the domain invariance of local features.

### Loss & Training
Total loss = Classification CE loss $\mathcal{L}_c$ + Concept CE loss $\mathcal{L}_k$ + Concept Saliency Alignment loss $\mathcal{L}_{\text{CSA}}$ + Local Concept Contrastive loss $\mathcal{L}_{\text{LCC}}$. ResNet-18 is used as the backbone with the Adam optimizer, lr=$1\times10^{-4}$, and batch size=32. Minimal augmentations (quantization, blur, Canny edge) are utilized to build triplets. The concept list is generated by GPT-3.5, exemplar images are synthesized by SD v2.1, and concept saliency maps are transferred to real images using DIFT feature transfer.

## Key Experimental Results

### Main Results

| Method | PACS | VLCS | OfficeHome | DomainNet | Average |
|------|------|------|------------|-----------|------|
| ERM | 49.35 | - | - | - | - |
| AugMix | 54.37 | - | - | - | - |
| ABA (Prev. SOTA) | 58.40 | - | - | - | - |
| **TIDE** | **~65** | **~72** | **~58** | **~48** | **~61** |
| **Gain** | **~+12%** | - | - | - | **+12%** |

### Ablation Study

| Configuration | PACS Avg (%) | Description |
|------|-------------|------|
| Full TIDE | ~65 | Full model |
| w/o CSA loss | ~60 | Saliency map alignment is highly important |
| w/o LCC loss | ~61 | Concept contrast is also crucial |
| w/o test-time correction | ~62 | Correction further yields a 3% gain |
| w/o concept discovery | ~58 | Concept filtering is critical |

### Key Findings
- TIDE significantly outperforms the state-of-the-art across all four DG benchmarks: PACS 82.62%, VLCS 77.08%, OfficeHome 74.01%, and DomainNet ~48%.
- Concept saliency maps not only boost performance but also render the prediction process visually interpretable.
- The test-time correction strategy effectively rectifies approximately 30% of the misclassified samples.
- The improvement is most pronounced on VLCS (semantic shifts dominated by background/perspective changes), achieving 77.08% vs the Prev. SOTA AugMix at 62.11%, validating the superiority of local concept learning under semantic shifts.
- On OfficeHome, TIDE achieves 74.01%, substantially outperforming all augmentation methods (AugMix 56.03%, RandAugment 56.56%, NJPP 57.85%), indicating that the local concept approach is more robust to multi-domain variations.
- The concept discovery module (filtering based on GradCAM overlap) is vital—not all concepts are useful for classification.

## Highlights & Insights
- **Diffusion Models as Annotation Tools**: Unleashing SD's cross-attention maps to automatically generate concept-level saliency maps is an innovative utilization of diffusion model capabilities.
- **Sophisticated Test-time Correction Strategy**: Detecting misclassifications via concept signatures and then iteratively masking misaligned focus regions to correct predictions—the entire process requires no additional training or target domain data.
- **Strong Transferability**: The framework of local concept learning + test-time correction can be generalized to any scenario requiring both interpretability and domain generalization.
- **Fundamental Difference from Augmentation Methods**: Traditional augmentation methods (AugMix, RandAugment, etc.) still learn global features and fail under semantic shifts (background/perspective variations). TIDE forces the model to learn local concepts (e.g., bird's beak, human's eyes) that are invariant across domains, leading to particularly prominent improvements on VLCS (77.08% vs the best augmentation method's 62.11%).
- **Training Efficiency**: Employs only minimal augmentations (quantization, blur, Canny edge) to construct triplets, backed by a ResNet-18 backbone and Adam optimizer with lr=1e-4 and batch=32.

## Limitations & Future Work
- Dependency on GPT-3.5 for generating concept lists and SD for synthesizing exemplar images limits the concept quality to the capabilities of these generative models.
- Test-time correction increases inference latency (up to 10 iterative forward passes).
- The transfer quality of concept saliency maps depends on the accuracy of DIFT feature matching.
- The ResNet-18 backbone may limit the upper performance bound.
- Future work could explore end-to-end learning for concept discovery or use more efficient correction strategies.
- The selection of the GradCAM overlap threshold is critical for concept filtering, but it is currently empirically set and lacks an adaptive mechanism.
- In fine-grained classification tasks with fewer concepts, the concept lists generated by LLMs may lack distinctiveness.
- TIDE achieves 82.62% on PACS, gaining +24.22% over the Prev. SOTA ABA (58.40%) and +33.27% over ERM (49.35%), demonstrating the immense potential of local concept learning for domain generalization.

## Related Work & Insights
- **vs ABA**: ABA utilizes heavy data augmentation but still learns global features, failing under semantic shifts; TIDE forces local concept learning, making it more robust.
- **vs CBM (Concept Bottleneck Models)**: CBMs can only predict predefined concepts but fail to associate them with image regions; TIDE both predicts concepts and localizes them.
- **vs PromptD**: PromptD uses domain prompt learning but still relies on global features; TIDE's local concept approach is more domain-invariant.

## Rating

### Implementation Details
ResNet-18 backbone, Adam optimizer with lr=1e-4, batch=32.
Concept list is generated by GPT-3.5, exemplar images are synthesized by SD v2.1, and concept maps are transferred via DIFT.
- Novelty: ⭐⭐⭐⭐⭐ The entire pipeline of automatic concept annotation + saliency alignment + test-time correction is highly novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on four datasets, though some ablation data needs to be retrieved from the supplementary materials of the paper
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, systematic and complete method description, and excellent illustrations
- Value: ⭐⭐⭐⭐⭐ The average gain of 12% is highly significant, and the combination of interpretability and generalization holds substantial value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[NeurIPS 2025\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](../../NeurIPS2025/interpretability/specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)

</div>

<!-- RELATED:END -->
