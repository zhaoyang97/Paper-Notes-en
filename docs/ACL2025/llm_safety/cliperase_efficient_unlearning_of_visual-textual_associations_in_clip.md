---
title: >-
  [Paper Note] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP
description: >-
  [ACL 2025][LLM Safety][Machine Unlearning] This work proposes CLIPErase, a machine unlearning framework tailored for multimodal CLIP models. By synergistically integrating a Forgetting Module, a Retention Module, and a Consistency Module, it selectively removes specified vision-language associations while preserving the performance of the model on the retained data.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "CLIP"
  - "Multimodal"
  - "Vision-Language Alignment"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 79155a81996fdfd7
---

# CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP

**Conference**: ACL 2025  
**arXiv**: [2410.23330](https://arxiv.org/abs/2410.23330)  
**Code**: [https://tianyu-yang-anna.github.io/ClipErase-ACL/](https://tianyu-yang-anna.github.io/ClipErase-ACL/)  
**Area**: LLM Safety  
**Keywords**: Machine Unlearning, CLIP, Multimodal, Vision-Language Alignment, Privacy Protection

## TL;DR

This work proposes CLIPErase, a machine unlearning framework tailored for multimodal CLIP models. By synergistically integrating a Forgetting Module, a Retention Module, and a Consistency Module, it selectively removes specified vision-language associations while preserving the performance of the model on the retained data.

## Background & Motivation

Machine unlearning aims to eliminate the influence of specific data from a trained model without complete retraining. Previous research has primarily focused on unimodal domains (such as text or image classification), while unlearning in multimodal models like CLIP remains relatively under-explored.

CLIP models align image and text representations in a shared embedding space through contrastive learning. This cross-modal alignment introduces unique challenges for unlearning:

**Failure of Unimodal Unlearning Methods**: Applying Gradient Ascent solely to the text modality inadvertently disrupts cross-modal relationships, causing downstream tasks (such as diffusion model generation) to produce distorted images or even fail to generate meaningful content.

**Precise Unlearning of Polysemous Concepts**: For example, "apple" can refer to both a fruit and a tech company. Unimodal methods indiscriminately erase all meanings of "apple" rather than selectively unlearning only one specific semantic meaning.

**Compliance and Privacy Demands**: Large-scale multimodal training data often contains sensitive or copyrighted content, necessitating the removal of the influence of such data without retraining.

## Method

### Overall Architecture

CLIPErase modifies the image and text encoders of the original CLIP model $\Theta$. It achieves unlearning through the synergies of three core modules: the Forgetting Module (FM), which disrupts the cross-modal associations of the forget set; the Retention Module (RM), which maintains the performance on the retain set; and the Consistency Module (CM), which ensures that the behavior of the unlearned model aligns with the original model on the retain set.

### Key Designs

1. **Forgetting Module**: Disrupts alignment by minimizing the dot product of image and text embeddings from the forget set. The loss function is $\mathcal{L}_{FM} = \frac{1}{N_f} \sum_{n=1}^{N_f} f_{img}(x_i^n) \cdot f_{txt}(x_t^n)$, driving the dot product towards zero or negative values so that image-text pairs in the forget set can no longer retrieve each other. This design is direct and efficient, avoiding complex soft labels or adversarial training.

2. **Retention Module**: Applies the original CLIP contrastive loss $\mathcal{L}_{RM}$ to the retain set, ensuring each image in the retain set remains closely aligned with its corresponding text while remaining distinct from other image-text pairs. The choice of contrastive loss over MSE is motivated by its ability to effectively maintain structured pairwise relationships and avoid introducing conflicting learning signals.

3. **Consistency Module**: Penalizes the difference in unimodal output distributions on the retain set between the unlearned model $\Theta_u$ and the original model $\Theta$ using KL divergence. It simultaneously considers consistency in both image and text distributions: $\mathcal{L}_{CM} = \frac{1}{N_r} \sum_{n=1}^{N_r} [KL(p_o^{img} \| p_u^{img}) + KL(p_o^{txt} \| p_u^{txt})]$. This prevents prediction bias introduced by the interference of different optimization objectives.

### Loss & Training

The overall loss is a weighted combination of the three modules:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{RM} + \lambda_2 \mathcal{L}_{FM} + \lambda_3 \mathcal{L}_{CM}$$

In the experiments, $\lambda_1 = 1, \lambda_2 = \lambda_3 = 3$ are set. Using the Adam optimizer, the learning rate is $1 \times 10^{-6}$ on CIFAR-100 and Conceptual 12M, and $1 \times 10^{-8}$ on Flickr30K. The batch size is 16, training for 20 epochs. The best checkpoint is selected on the validation set, trained on NVIDIA V100.

## Key Experimental Results

### Main Results

| Dataset | Task | Metrics | CLIPErase | Prev. SOTA | Notes |
|--------|------|------|-----------|----------|------|
| CIFAR-100 | Zero-Shot Prediction | Df Acc. ↓ | **0.00%** | 0.00%(ENMN) | Complete unlearning |
| CIFAR-100 | Zero-Shot Prediction | Dr Acc. ↑ | **90.99%** | 89.96%(GradDiff) | Best retain set performance |
| Conceptual 12M | Zero-Shot Prediction | Df Acc. ↓ | **0.74%** | 4.96%(GradDiff) | Near-complete unlearning |
| Conceptual 12M | Zero-Shot Prediction | Dr Acc. ↑ | **97.10%** | 97.01%(GradDiff) | Comparable retain set performance |
| Flickr30K | Image Retrieval R@10 | Df↓/Dr↑ | 10.55/50.35 | 7.82/47.13(GradDiff) | Best balance between unlearning effectiveness and retention performance |

### Ablation Study

| Configuration | Df Acc. ↓ | Dr Acc. ↑ | Notes |
|------|----------|----------|------|
| No modules | 86.08% | 72.85% | Original CLIP |
| FM only | 18.57% | 64.12% | Effective unlearning but disrupts the retain set |
| FM + RM | 9.40% | 73.14% | RM restores retain set performance |
| FM + RM + CM | **0.00%** | **90.80%** | CM substantially boosts the retain set to above 90% |

### Key Findings

- Although ENMN can also achieve 0% accuracy on the forget set, its retain set performance plummets to 12.46%, making it practically unusable.
- CLIPErase maintains robustness across different proportions of forgotten categories (3% to 30%), whereas the unlearning effectiveness of GA and GradDiff deteriorates as the forgetting ratio increases.
- In diffusion model experiments, CLIPErase reduces the detection rate of "apple" from 100% to 2% and "bicycle" from 90% to 8%, without affecting the generation of other concepts.
- It generalization to other VLMs such as BLIP, achieving Df=0% and Dr=83.12% on BLIP.

## Highlights & Insights

- **Modular Design**: The three modules operate with distinct responsibilities, and the ablation study clearly demonstrates the contribution of each component. In particular, the CM module leads to a leap in Dr from 73% to 91%, indicating that maintaining unimodal distribution consistency is crucial for unlearning quality.
- **Fine-Grained Unlearning Capability**: The ability to distinguish between "apple" the fruit and Apple the company is highly critical for real-world privacy compliance.
- **Thorough Visual Validation**: Attention heatmaps and t-SNE embedding space visualizations intuitively demonstrate the unlearning effectiveness.
- **Model-Agnostic Nature**: The framework does not rely on components specific to CLIP and has been validated for scalability to BLIP-1.

## Limitations & Future Work

1. Lack of standardized datasets and evaluation benchmarks specifically designed for multimodal unlearning.
2. Currently only applicable to multimodal embedding models; it has not yet been extended to generative vision-language models (VLMs).
3. The choice of hyperparameters $\lambda_1$, $\lambda_2$, and $\lambda_3$ relies on manual tuning, lacking an adaptive adjustment mechanism.
4. The definition of the forget set assumes users explicitly know which samples to unlearn, whereas unlearning targets in real-world scenarios may be more ambiguous.

## Related Work & Insights

- Comparison with MultDelete (Cheng & Amiri, 2023): MultDelete relies on randomly sampled unrelated pairs, which may fail on small datasets and is tailored only to specific tasks. In contrast, CLIPErase operates directly in the shared embedding space of CLIP, making it applicable to a broader range of downstream tasks.
- The Gradient Ascent method is intuitively reasonable but overly aggressive, easily disrupting the retain set. GradDiff attempts to strike a balance but lacks consistency constraints.
- This work offers valuable insights into understanding the nature of knowledge storage and cross-modal associations in multimodal models. Unlearning is, in essence, precisely "unbinding" specific associations within the shared embedding space.

## Supplementary Details

- In BLIP experiments, although the GA method reduces forget set accuracy to 0%, the retain set performance plummets to 42.89%; GradDiff's forget set accuracy remains as high as 89.73%, rendering it largely ineffective. CLIPErase maintains a superior balance on BLIP as well, with 0% forget set accuracy and 83.12% retention accuracy.
- In the visualization analysis, post-unlearning t-SNE plots clearly show that the distance between the image and text embeddings of the "apple" class targets significantly increases, while the clustering structures of other categories remain completely unaffected.
- Robustness was validated across different forgetting ratios (3%, 10%, 20%, 30%). While the retain set accuracy for GA drops sharply as the forgetting ratio increases, CLIPErase consistently maintains retain set performance comparable to the original CLIP.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multimodal unlearning is an emerging direction. The three-module design is reasonable, but each module is not entirely novel from a technical standpoint.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 downstream tasks, 3 datasets, diffusion models, and BLIP extensions; incorporates complete ablation studies and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-articulated motivation, and intuitive chart design.
- **Value**: ⭐⭐⭐⭐ Addresses the practical need for multimodal privacy compliance, and the model-agnostic framework holds strong potential for general applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/llm_safety/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](../../AAAI2026/llm_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](../../ICML2025/llm_safety/targeted_unlearning_with_single_layer_unlearning_gradient.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](../../NeurIPS2025/llm_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)

</div>

<!-- RELATED:END -->
