---
title: >-
  [Paper Note] DDB: Diffusion Driven Balancing to Address Spurious Correlations
description: >-
  [ICCV 2025][Segmentation][Spurious Correlations] This paper proposes Diffusion Driven Balancing (DDB), which leverages the textual inversion and inpainting capabilities of Stable Diffusion to automatically generate minority-group samples for balancing spurious correlations in datasets. Combined with a bicephalous pruning strategy based on ERM model prediction probabilities and integrated gradients, DDB achieves state-of-the-art worst-group accuracy on Waterbirds and MetaShift.
tags:
  - ICCV 2025
  - Segmentation
  - Spurious Correlations
  - Diffusion Model Data Augmentation
  - Group Robustness
  - Textual Inversion
  - Image Inpainting
date: 2026-05-08
content_hash: 756cfbae4e7d3602
---

# DDB: Diffusion Driven Balancing to Address Spurious Correlations

**Conference**: ICCV 2025  
**arXiv**: [2503.17226](https://arxiv.org/abs/2503.17226)  
**Code**: [https://github.com/ArianYp/DDB](https://github.com/ArianYp/DDB)  
**Area**: Image Segmentation  
**Keywords**: Spurious Correlations, Diffusion Model Data Augmentation, Group Robustness, Textual Inversion, Image Inpainting

## TL;DR
This paper proposes Diffusion Driven Balancing (DDB), which leverages the textual inversion and inpainting capabilities of Stable Diffusion to automatically generate minority-group samples for balancing spurious correlations in datasets. Combined with a bicephalous pruning strategy based on ERM model prediction probabilities and integrated gradients, DDB achieves state-of-the-art worst-group accuracy on Waterbirds and MetaShift.

## Background & Motivation

When deep networks are trained with empirical risk minimization (ERM), they tend to rely on spuriously correlated features rather than true causal features, leading to failures in out-of-distribution generalization. For example, if most cows in the training set appear on green pastures and most camels appear in deserts, the model may learn to classify animals based solely on background.

**Existing approaches and their limitations**:

**Re-weighting methods (JTT, AFR)**: Assume that minority-group samples incur higher loss and upweight them accordingly. However, high loss under ERM may arise for other reasons, and minority-group samples may be too scarce for effective re-weighting.

**Data mixing methods (DaC, DISC, LISA)**: Augment minority groups by mixing or splicing samples from different groups. However, DaC lacks semantic control and produces low-quality generations; LISA requires group labels during training.

**Diffusion-based methods (FFR)**: Use diffusion models to generate balanced datasets. However, they are sensitive to prompts, may generate harmful samples, and require prior knowledge of dataset biases.

**Key Challenge**: How can high-quality minority-group samples be generated automatically and precisely to break spurious correlations, without requiring group labels on the training set?

**Core Idea**: The paper adopts a compositional view of images — each image consists of a causal part (core features) and a spurious part (spurious features). Textual inversion is used to learn per-class causal feature tokens; language-guided segmentation localizes the causal region; Stable Diffusion inpainting replaces the causal part to generate new-class samples; and a bicephalous pruning strategy based on ERM prediction probabilities and integrated gradients ensures generation quality.

## Method

### Overall Architecture
DDB is a three-stage method: (1) new data generation — learning causal tokens + language segmentation + diffusion inpainting; (2) pruning — dual-condition filtering of low-quality samples; (3) retraining — retraining the ERM model on the balanced dataset.

### Key Designs

1. **Causal Feature Learning (Textual Inversion)**:

    - **Function**: Learns a trainable token embedding $[C_i]$ for each class to encode the visual semantics of its causal features.
    - **Mechanism**: Uses the template sentence "A photo of a $[C_i]$ bird" as the prompt, freezes Stable Diffusion parameters, and optimizes only the embedding of $[C_i]$ in the text encoder by minimizing the denoising loss:
    $C_i^* = \arg\min_{C_i} \mathbb{E}_{z,I,\epsilon,t} [\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(I))\|_2^2]$
    - Learned from 20–40 samples per class (prioritizing minority-group samples).
    - **Design Motivation**: Textual inversion enables precise control over causal features, rather than relying on manually crafted prompts.

2. **Causal Part Replacement (Diffusion Inpainting)**:

    - **Function**: Replaces the causal part of majority-group samples with the causal features of another class, while preserving the spurious background, to generate minority-group samples.
    - **Mechanism**:
        - LangSAM (GroundingDINO + SAM) automatically segments the causal region: $M = m(x_j)$
        - Noise is added only within the masked region, and the learned token guides denoising to generate a new causal part:
       $$z_t = (1-M) \odot z_0 + M \odot (\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon)$$
        - Majority-group identification: the $K$ samples with the lowest loss under the ERM model are selected as majority-group samples.
    - **Design Motivation**: Preserving the original background (spurious features unchanged) while precisely replacing the causal part (altering the label) naturally produces minority-group samples.

3. **Bicephalous Pruning**:

    - **Function**: Filters low-quality or ineffective samples generated by the diffusion model.
    - **Mechanism**: Two complementary pruning conditions —
        - **Condition 1: ERM Prediction Probability**. The softmax output $\psi_j$ of the generated sample under the ERM model is computed. If $\psi_j \geq \Psi_i$ (the class-average probability), the image has not undergone effective change (still classified as the original class) and is pruned.
        - **Condition 2: Integrated Gradient Attribution Score**. The integrated gradient difference between the original and modified images is computed:
       $$\text{IG}_k(x') = (x_k' - x_k) \times \int_0^1 \frac{\partial f_{i'}(x + \alpha(x'-x))}{\partial x_k'} d\alpha$$
       The attribution score accumulated over the masked region is $\rho = \sum_k M_k \cdot \text{IG}_k$. If $\rho \leq P_{i'}$ (the threshold), the modified region contributes insufficiently to the label change, and the sample is pruned.
    - **Design Motivation**: A single condition is insufficient — a change in ERM probability may stem from irrelevant noise rather than a valid causal replacement (Fig. 3(a)), while a high attribution score without a probability change indicates insufficient modification (Fig. 3(c)). The dual conditions ensure that generated samples both effectively alter causal features and produce the correct influence on model predictions.

### Loss & Training
- Retraining loss: $L_{total} = L_{train} + \gamma_1 L_{gen1} + \gamma_2 L_{gen2}$
- $L_{gen1}$ and $L_{gen2}$ apply cross-entropy loss to generated samples from two classes respectively.
- $\gamma_1, \gamma_2$ upweight the newly added samples.
- ResNet-50 (ImageNet pretrained) is used as the classifier.
- Stable Diffusion v2 is used for image generation.
- Diffusion model parameters are frozen during both textual inversion and inpainting.

## Key Experimental Results

### Main Results
Worst-Group Accuracy (WGA) on three standard spurious correlation benchmarks:

| Method | Group Labels | Waterbirds WGA | CelebA WGA | MetaShift WGA |
|--------|-------------|---------------|------------|---------------|
| Base (ERM) | ✗/✗ | 74.6 | 42.2 | 67.0 |
| JTT | ✗/✓ | 86.7 | 81.1 | 64.6 |
| DFR | ✗/✓✓ | 92.9 | 88.3 | 72.8 |
| DaC | ✗/✓ | 92.3 | 81.9 | 78.3 |
| LISA | ✓/✓ | 89.2 | **89.3** | 59.8 |
| DISC | ✗/✗ | 88.7 | - | 73.5 |
| **DDB (ours)** | ✗/✓ | **93.0** | 85.8 | **81.2** |

### Ablation Study
Effect of pruning:

| Dataset | Class | Generated | Pruned | WGA w/o Pruning | WGA w/ Pruning |
|---------|-------|-----------|--------|----------------|----------------|
| Waterbirds | Landbird | 1300 | 531 | 91.28 | **93.0** |
| Waterbirds | Waterbird | 1112 | 393 | - | - |
| CelebA | NotBlond | 120000 | 7439 | 81.7 | **85.8** |
| MetaShift | Dog | 400 | 299 (75%) | 80.6 | **81.2** |

Effect of textual inversion settings on pruning rate and performance (Waterbirds):

| Inversion Samples | 0 | 10 | 20 | 30 | 40 |
|-------------------|---|----|----|----|----|
| Pruned | 1018 | 818 | 804 | 737 | 782 |
| WGA | 88.9 | 90.5 | 89.9 | **93.0** | 92.1 |

### Key Findings
- DDB achieves state-of-the-art WGA on both Waterbirds and MetaShift without requiring training-set group labels.
- **Pruning is critical**: 75% of generated Dog samples in MetaShift are pruned (because the dog is too small or absent in the original image), and performance degrades noticeably without pruning.
- Textual inversion significantly improves generation quality: without inversion, the pruning count is as high as 1018/2412; with inversion (30 samples), it drops to 737/2412.
- Performance on CelebA is slightly below LISA (85.8 vs. 89.3), as CelebA's spurious features (hair length, gender) are more difficult for the diffusion model to modify — LangSAM exhibits instability in hair region localization.
- DDB is effective for both spurious objects (e.g., Waterbirds backgrounds) and spurious features (e.g., CelebA gender).

## Highlights & Insights
- **Precise compositional formulation**: The paper reframes spurious correlation as an image composition problem — preserving the spurious background while replacing the causal foreground — yielding a natural and efficient solution.
- **Synergistic combination of textual inversion and inpainting**: Textual inversion provides semantic control while inpainting provides spatial control; the two are complementary.
- **Pragmatic bicephalous pruning**: The method recognizes that each component of the generation pipeline may fail and employs dual filtering via ERM probabilities and attribution scores to ensure quality.
- **No training-set group labels required**: Majority/minority groups are automatically identified by ranking ERM loss, lowering the barrier to practical use.
- The entire pipeline is highly automated, requiring no manual intervention across the full chain from causal feature learning to sample generation and quality control.

## Limitations & Future Work
- Relies on the segmentation quality of LangSAM — performance degrades when the causal part (e.g., hair) is difficult to describe with simple text.
- Stable Diffusion inpainting quality is unstable in certain scenarios (e.g., small objects, complex textures).
- Only binary classification tasks are evaluated; multi-class settings require causal replacement for each class pair, increasing complexity.
- Textual inversion requires 20–40 samples, which may be insufficient in extreme low-shot scenarios.
- Suboptimal performance on CelebA suggests room for improvement in handling "feature-level" spurious correlations (vs. "object-level").
- Validation-set group labels are required for hyperparameter tuning.

## Related Work & Insights
- The compositional view (core + spurious features) provides a clear formal framework for the spurious correlation problem.
- Textual inversion is an effective means of obtaining fine-grained control over visual concepts and can be applied to other data augmentation settings requiring precise semantic control.
- The inpainting capability of diffusion models holds broad potential for data augmentation beyond spurious correlation tasks.
- Integrated gradients as a quality assessment tool for generated samples represents a novel application.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of textual inversion and inpainting for spurious correlations is a novel composition, though the framework still builds on existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three standard benchmarks with comprehensive ablations, but validation on larger scales or more diverse categories is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The method is clearly presented and the pipeline figure is intuitive, though the problem formulation section is slightly verbose.
- **Value**: ⭐⭐⭐⭐ A practical data augmentation paradigm with clear contributions to out-of-distribution generalization.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](../../CVPR2026/segmentation/concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](../../NeurIPS2025/segmentation/diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

<!-- RELATED:END -->
