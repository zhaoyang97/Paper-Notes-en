---
title: >-
  [Paper Note] CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks
description: >-
  [ECCV 2024][AI Safety][adversarial attack] This paper proposes CGNC, which leverages the CLIP text encoder to inject target-category semantic information into a conditional generative network. Combining cross-attention modules with masked fine-tuning, this method significantly improves the black-box transfer success rate of both multi-target and single-target directed adversarial attacks.
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "adversarial attack"
  - "targeted transferability"
  - "CLIP"
  - "generative model"
  - "cross-attention"
  - "multi-target attack"
date: 2026-05-08
content_hash: fdcc260035c1c042
---

# CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks

**Conference**: ECCV 2024  
**arXiv**: [2407.10179](https://arxiv.org/abs/2407.10179)  
**Code**: [ffhibnese/CGNC_Targeted_Adversarial_Attacks](https://github.com/ffhibnese/CGNC_Targeted_Adversarial_Attacks)  
**Area**: AI Safety  
**Keywords**: adversarial attack, targeted transferability, CLIP, generative model, cross-attention, multi-target attack

## TL;DR

This paper proposes CGNC, which leverages the CLIP text encoder to inject target-category semantic information into a conditional generative network. Combining cross-attention modules with masked fine-tuning, this method significantly improves the black-box transfer success rate of both multi-target and single-target directed adversarial attacks.

## Background & Motivation

**High difficulty of targeted transfer attacks**: Compared to non-targeted attacks, targeted adversarial attacks require black-box models to output specifically designated categories. Their transfer success rate is far lower than that of non-targeted attacks, posing a core challenge in adversarial security research.

**High computational overhead of single-target generative attacks**: Methods such as TTP and DGTA-PI train an independent generator for each target category, resulting in prohibitive training costs when the target classes scale to hundreds or thousands.

**Information poverty in existing multi-target methods**: MAN and C-GSP only utilize category indices or one-hot vectors as conditional inputs. They fail to exploit the rich semantic information of target classes, leading to limited black-box transferability.

**Overfitting to decision boundaries**: Gradient-based iterative methods (such as MIM, DIM, etc.) heavily overfit the classification boundaries of surrogate models, performing extremely poorly when transferring to black-box models (typically $< 5\%$).

**Underutilized semantic priors of vision-language models**: Vision-language models (VLMs) like CLIP are pre-trained on large-scale image-text pairs and possess rich category semantic knowledge. However, they have not yet been utilized to guide adversarial perturbation generation.

**Infeasibility of single-target methods in cross-domain scenarios**: Single-target attacks require images of the target category to calculate losses, rendering them completely ineffective in cross-domain scenarios where training sets exclude the target categories. Thus, there is an urgent need for solutions that do not depend on target-class data.

## Method

### Overall Architecture

CGNC (CLIP-guided Generative Network with Cross-attention) is a conditional generative network that takes a clean image $\bm{x}_s$ and a text description of the target category $\bm{t}_c$ (e.g., "a photo of a sea lion") as inputs, and outputs an adversarial perturbation $\bm{delta}$ under $\ell_\infty$ constraints. The network consists of three core modules: VL-Purifier, F-Encoder, and CA-Decoder. The training objective is to minimize the cross-entropy loss of the surrogate model on the adversarial exemplars with respect to the target category:

$$w^* \leftarrow \arg\min_w \mathcal{L}\big(f_\theta(\bm{x}_s + G_w(\bm{x}_s, \Phi(\bm{t}_c))), c\big)$$

### Key Designs

**Key Design 1: Vision-Language Feature Purifier**

The text of the target category is fed into the CLIP text encoder $\Phi$ to produce a 512-dimensional embedding $\bm{e}_t$, which is further compressed into a 16-dimensional task-specific representation $\bm{e}_t^*$ by the VL-Purifier consisting of fully connected layers and spectral normalization layers. This step adapts the embeddings from CLIP's general semantic space to the adversarial perturbation generation task, mitigating noise from directly using high-dimensional general representations.

**Key Design 2: Feature Fusion Encoder**

The F-Encoder fuses the purified text embeddings with the visual features of the image across multiple stages: it first spatially duplicates $\bm{e}_t^*$ to concatenate with the image encoder's feature map $\bm{h}_s$ along the channel dimension; then, after downsampling, it concatenates with the text embeddings again, repeating this process multiple times. This multi-stage concatenation mechanism simultaneously exploits instance-level visual information and category-level textual information, equipping the generated perturbations with stronger semantic patterns.

**Key Design 3: Cross-Attention Decoder**

A cross-attention layer is introduced into the decoder, utilizing the original 512-dimensional CLIP embedding $\bm{e}_t$ as Key/Value, and the intermediate decoder feature $\bm{z}_t$ as Query:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

Finally, a smooth projection $\bm{\delta} = \epsilon \cdot \tanh(\bm{o})$ is employed to ensure the perturbation satisfies the constraint $\ell_\infty \leq \epsilon$. Cross-attention allows the decoder to dynamically focus on feature dimensions in the CLIP semantic space that are most crucial to the target category.

**Key Design 4: Masked Fine-Tuning (MFT)**

For single-target scenarios, the conditional input is fixed as the text of the specific target category, and the pre-trained multi-target generator is fine-tuned. To alleviate overfitting caused by perturbations concentrating in specific regions during fine-tuning, patch-wise random masking (mask ratio = 0.2) is applied to the perturbation output, forcing the generator to learn more distributed perturbation patterns. This requires only 5 additional epochs of fine-tuning.

## Loss & Training

- **Loss Function**: Cross-entropy loss, guiding the surrogate model to classify adversarial examples into the target category.
- **Data Augmentation**: Applied to input images during training to improve perturbation generalization.
- **Surrogate Models**: Inc-v3 and Res-152.
- **Perturbation Budget**: $\epsilon = 16/255$.
- **Training**: Learning rate of 2e-4, trained for 10 epochs; MFT stage takes 5 epochs.

## Experiments

### Multi-Target Attack: Normally Trained Models (Surrogate Res-152 $\to$ Black-box)

| Method | VGG-16 | GoogleNet | Inc-v3 | DN-121 | Inc-v4 | Inc-Res-v2 |
|------|--------|-----------|--------|--------|--------|------------|
| MIM | 0.20 | 0.30 | 0.50 | 0.30 | 0.40 | 0.60 |
| Logit | 9.20 | 3.70 | 10.10 | 12.70 | 10.70 | 12.80 |
| C-GSP | 45.90 | 41.70 | 37.70 | 64.20 | 33.33 | 20.28 |
| **CGNC** | **63.36** | **62.23** | **53.39** | **85.66** | **51.53** | **34.24** |

CGNC substantially outperforms C-GSP on all black-box models, achieving an average absolute increase of 17.88% (and an increase of 21.46% on DN-121), which validates the effectiveness of the CLIP semantic prior.

### Robust Model Attack (Surrogate Res-152 $\to$ Robustly Trained Models)

| Method | Inc-v3_adv | IR-v2_ens | Res50_SIN | Res50_IN | Res50_fine | Res50_Aug |
|------|-----------|-----------|-----------|----------|------------|-----------|
| C-GSP | 14.60 | 16.01 | 16.84 | 60.30 | 65.51 | 42.88 |
| **CGNC** | **22.21** | **26.71** | **29.83** | **79.80** | **84.05** | **63.75** |

CGNC also gains a substantial lead on robustly trained models, with an improvement of 20.87% on Res50_Aug, which demonstrates its advantages under defense scenarios.

### Single-Target Attack Comparison (Average of 8 classes, Surrogate Inc-v3)

| Method | Inc-v4 | Inc-Res-v2 | Res-152 | DN-121 | GoogleNet | VGG-16 |
|------|--------|------------|---------|--------|-----------|--------|
| TTP | 46.04 | 39.37 | 16.40 | 33.47 | 25.80 | 25.73 |
| DGTA-PI | 67.95 | 55.03 | 50.50 | 47.38 | 47.67 | 48.11 |
| **CGNC†** | **74.76** | **64.48** | **62.00** | **78.94** | **69.06** | **70.74** |

The MFT single-target variant CGNC† improves black-box transferability by 15.36% on average, requiring only one multi-target generator plus 8 fast fine-tuning runs. Its computational overhead is significantly lower than training 8 independent generators.

### Ablation Study (Surrogate Res-152)

| Variant | VGG-16 | GoogleNet | Inc-v3 | DN-201 |
|------|--------|-----------|--------|--------|
| CGNC-CA-t (one-hot condition) | 56.55 | 51.09 | 47.44 | 74.65 |
| CGNC-CA (CLIP text, without cross-attention) | 56.64 | 54.29 | 49.73 | 75.99 |
| **CGNC (Full)** | **63.36** | **62.23** | **53.39** | **82.69** |

The CLIP text embedding and the cross-attention module both make significant contributions. The full CGNC outperforms the one-hot baseline by 8.04% on DN-201.

## Highlights & Insights

- **Semantic-Driven Conditional Mechanism**: This is the first work to utilize CLIP text encodings as conditional inputs for multi-target adversarial generators, fundamentally overcoming the semantic poverty bottleneck of one-hot conditionings.
- **Significant Multi-Target Enhancements**: CGNC outperforms C-GSP by a margin under both 8-class and 200-class settings, demonstrating slower performance degradation as target classes scale.
- **Efficient Single-Target Adaptation**: MFT requires only 5 epochs of fine-tuning to outperform single-target SOTA methods trained from scratch, saving over 100 epochs when attacking 8 classes.
- **Cross-Domain Generalization**: CGNC maintains respectable performance on MS-COCO and Comics datasets where images of the target categories are absent—a feat cannot be achieved by single-target approaches.
- **Abundant Visual Evidence**: The generated perturbations exhibit clear semantic patterns of the target category (such as the shape of a sea lion) and shift dynamically with textual conditions.

## Limitations & Future Work

- Performance remains dependent on the pre-trained CLIP model, subject to CLIP's representation quality for specific categories.
- The perturbation budget $\epsilon=16/255$ is relatively large, and the efficacy under stricter constraints (e.g., $8/255$) remains under-evaluated.
- The performance of the multi-target generator begins to degrade notably once the number of classes exceeds 200, which is still a distance from true practicality on 1000-class datasets.
- The mask ratio in MFT requires manual tuning (fixed at 0.2 in the paper) and lacks an adaptive adjustment mechanism.
- Experiments are only conducted on ImageNet classifiers and have not been extended to more complex vision tasks like object detection.

## Related Work & Insights

- **Multi-Target Generative Attacks**: MAN (ICCV 2019) first proposed a multi-target framework but suffered from severe image degradation. C-GSP (CVPR 2023) introduced hierarchical partitioning to improve performance but remained limited by flat conditional representations.
- **Single-Target Generative Attacks**: TTP (ICCV 2021) leverages target distribution matching; DGTA-PI (CVPR 2023) designs dynamic networks and pattern injection; our MFT achieves superior performance with much lower cost.
- **CLIP Applications in Adversarial Attacks**: Earlier works deployed CLIP mainly for zero-shot classification and multimodal tasks; this paper represents the first attempt to use its text encodings to guide the generation of adversarial perturbations.
- **Cross-Attention Fusion**: Drawing inspiration from the cross-attention design of text-conditioned generative models like Stable Diffusion, we adapt it to the adversarial perturbation scenario.

## Rating

- Novelty: ⭐⭐⭐⭐ — Leveraging CLIP text semantics to guide adversarial generation is a novel and intuitive concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers various scenarios (normal, robust, defensive, cross-domain, multiple class scales) with complete ablation.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rich figures and tables, and highly intuitive and convincing visualizations of perturbation.
- Value: ⭐⭐⭐⭐ — Establishes a new pathway for incorporating VLM priors into adversarial transfer attacks, holding practical significance for robust evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[CVPR 2026\] RaPA: Enhancing Transferable Targeted Attacks via Random Parameter Pruning](../../CVPR2026/ai_safety/rapa_enhancing_transferable_targeted_attacks_via_random_parameter_pruning.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[CVPR 2026\] When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive Counterattacks for Test-Time Adversarial Robustness](../../CVPR2026/ai_safety/when_clip_sees_more_it_fights_back_harder_multi-view_guided_adaptive_counteratta.md)
- [\[CVPR 2026\] VCP-Attack: Visual-Contrastive Projection for Transferable Black-Box Targeted Attacks on Large Vision-Language Models](../../CVPR2026/ai_safety/vcp-attack_visual-contrastive_projection_for_transferable_black-box_targeted_att.md)

</div>

<!-- RELATED:END -->
