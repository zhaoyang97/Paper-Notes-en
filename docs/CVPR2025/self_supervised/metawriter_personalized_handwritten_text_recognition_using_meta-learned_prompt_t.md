---
title: >-
  [Paper Note] MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning
description: >-
  [CVPR 2025][Self-Supervised Learning][Handwritten Text Recognition] MetaWriter formulates the personalized adaptation of handwritten text recognition as a prompt tuning problem. By combining it with a Masked Autoencoder (MAE) self-supervised auxiliary task, it achieves label-free test-time adaptation. It optimizes prompt initialization via meta-learning to align the self-supervised loss with the recognition loss, achieving SOTA on IAM and RIMES by updating less than 1% of the…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Handwritten Text Recognition"
  - "Meta-Learning"
  - "Prompt Tuning"
  - "Masked Autoencoders"
  - "Test-Time Adaptation"
date: 2026-05-08
content_hash: 488b9ddcd3f38c2a
---

# MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning

**Conference**: CVPR 2025  
**arXiv**: [2505.20513](https://arxiv.org/abs/2505.20513)  
**Code**: None  
**Area**: Self-Supervised Learning / Handwritten Text Recognition  
**Keywords**: Handwritten Text Recognition, Meta-Learning, Prompt Tuning, Masked Autoencoders, Test-Time Adaptation  

## TL;DR
MetaWriter formulates the personalized adaptation of handwritten text recognition as a prompt tuning problem. By combining it with a Masked Autoencoder (MAE) self-supervised auxiliary task, it achieves label-free test-time adaptation. It optimizes prompt initialization via meta-learning to align the self-supervised loss with the recognition loss, achieving SOTA on IAM and RIMES by updating less than 1% of the parameters.

## Background & Motivation

1. **Background**: Handwritten Text Recognition (HTR) has made significant progress through deep learning, but mainstream methods assume that the training set already covers sufficient variations in writing styles, leading to limited generalization capability on unseen styles.
2. **Limitations of Prior Work**: (1) Personalized methods such as MetaHTR require a small number of labeled samples for test-time adaptation, which is time-consuming and user-unfriendly; (2) Full-parameter fine-tuning incurs high computational and memory overhead, making it difficult to deploy on resource-constrained devices.
3. **Key Challenge**: Personalized adaptation requires updating model parameters to capture new writing styles, but full-parameter updates are costly. Additionally, there is a lack of supervisory signals to guide the direction of parameter updates in label-free scenarios.
4. **Goal**: How to achieve personalized adaptation of HTR models with extremely low parameter overhead using only a very small number of unlabeled samples?
5. **Key Insight**: Writing style information can be encoded into the padding of convolutional layers (prompt vectors). The image reconstruction task of MAE can provide unlabeled self-supervised signals, and meta-learning can ensure that the direction of the self-supervised gradient is consistent with the recognition task.
6. **Core Idea**: Use meta-learning-optimized prompt vectors as the carrier of writing styles, achieving zero-label test-time personalization through MAE self-supervised loss.

## Method

### Overall Architecture
The system is divided into two phases: pre-training and personalization. Pre-training phase: FCN encoder (18 convolutional layers + 12 depthwise separable convolutional layers) $\rightarrow$ 1D feature sequence $\rightarrow$ 8-layer Transformer decoder to autoregressively generate text tokens. Personalization phase: Learnable prompt vectors are used as convolutional layer padding to replace fixed zero/reflect padding. Only these prompt vectors are updated (<1% of parameters), while all other parameters are frozen. The meta-learning training pipeline simulates the test-time adaptation process.

### Key Designs

1. **Prompt Tuning as a Carrier for Style Personalization**:

    - **Function**: Encode writer-specific style information with an extremely small number of parameters.
    - **Mechanism**: Traditional convolutional layers use fixed padding (zero-padding/reflection-padding) to control output dimensions. Here, the padding is replaced with learnable prompt vectors $P \in \mathbb{R}^{l \times 3}$, where $l$ is the padding width. Learnable padding is used in the first 18 convolutional layers. The total number of parameters is only 82K (0.08M), which accounts for approximately 1% of the 7.6M total model parameters. During test-time, only these prompts are updated while freezing the encoder and decoder.
    - **Design Motivation**: The padding values of convolutional layers directly affect intermediate feature representations. Modifying the padding is equivalent to modulating features on the style dimension. The extremely small parameter footprint makes it suitable for edge device deployment and fast adaptation.

2. **MAE Self-Supervised Auxiliary Task**:

    - **Function**: Provide unlabeled supervisory signals to guide prompt adaptation.
    - **Mechanism**: The input handwritten image is subjected to 75% random masking. After padding with the meta-prompt, it is fed into the shared encoder, and the original image is reconstructed using an MAE decoder. SSIM (Structural Similarity) is used as the loss function instead of MSE, as HTR is highly sensitive to image details. Adaptation formula: $P' \leftarrow P - \lambda_1 \mathcal{L}_{ada}(g(f(\hat{X}^m, \theta_{enc}); \phi), X)$, where $\mathcal{L}_{ada} = 1 - \text{SSIM}(x, \hat{x})$.
    - **Design Motivation**: The image reconstruction task forces the encoder to understand the structural information of writing styles (stroke thickness, tilt, spacing, etc.). The prompt implicitly captures the writer's style by encoding these style features to assist reconstruction. SSIM focuses more on the perceptual quality of texture and luminance compared to MSE.

3. **Meta-Learning for Prompt Initialization (Meta-Prompt)**:

    - **Function**: Ensure that the gradient direction of the self-supervised loss is consistent with the text recognition loss.
    - **Mechanism**: A MAML-style bi-level optimization is adopted. Outer loop: Iterate through multiple writers (episodes) in the training set. Inner loop: For each writer, update the prompt to $P_j$ using the unlabeled support set via the MAE loss (one-step gradient). Outer loop: Calculate the recognition cross-entropy loss $\mathcal{L}_{pred}$ using the labeled query set to update the meta-prompt $P$. That is, $P \leftarrow P - \lambda_2 \nabla_P \sum_{\mathcal{T}_j} \mathcal{L}_{pred}$. The learned initial prompt $P$ can minimize both reconstruction and recognition losses after one step of self-supervised update.
    - **Design Motivation**: The gradient directions of the self-supervised loss (image reconstruction) and the task loss (text recognition) might be inconsistent. Direct adaptation using the self-supervised loss may degrade recognition performance. Meta-learning automatically learns a prompt initialization point that aligns both losses by simulating the "self-supervised adaptation followed by recognition evaluation" process during the training phase.

### Loss & Training
- Pre-training: First train feature extraction on synthetic printed data, and then transition from 90% synthetic data to episode training where each unit is a real writer, using a curriculum learning strategy.
- Meta-prompt training: Inner loop uses SSIM loss (learning rate $\lambda_1$), outer loop uses cross-entropy loss (learning rate $\lambda_2$), single RTX 4090 GPU, Adam optimizer, initial learning rate of 1e-4.
- Test-time personalization: Execute few-shot prompt updates using only $K$ unlabeled samples.

## Key Experimental Results

### Main Results

**IAM Dataset (Line-level)**:

| Method | CER↓ | WER↓ | Trainable Params |
|------|------|------|-----------|
| TrOCR | 4.22% | - | 334M |
| VAN | 4.32% | 16.24% | - |
| DAN | - | - | 7.6M |
| MetaHTR | - | - | 1.7M |
| **MetaWriter** | **3.36%** | **10.32%** | **0.08M** |

**RIMES Dataset (Line-level)**:

| Method | CER↓ | WER↓ |
|------|------|------|
| DAN | 2.63% | 6.78% |
| Coquenet et al. | 3.04% | 8.32% |
| **MetaWriter** | **2.19%** | **6.63%** |

### Ablation Study

| Configuration | IAM CER↓ | IAM WER↓ | RIMES CER↓ | RIMES WER↓ |
|------|---------|---------|-----------|-----------|
| Baseline (w/o any enhancement) | 4.14% | 12.03% | 2.92% | 8.35% |
| w/o MAE | 3.93% | 11.06% | 2.73% | 7.24% |
| w/o Meta | 3.63% | 10.75% | 2.51% | 7.08% |
| w/o Prompt (Full parameters) | 3.41% | 10.22% | 2.21% | 6.56% |
| **MetaWriter (Full)** | **3.32%** | **10.21%** | **2.13%** | **6.55%** |

### Key Findings
- All three components contribute: MAE auxiliary task (CER↓0.21%), Meta-learning (CER↓0.30%), Prompt (CER↓0.09% but with 20x parameter reduction).
- Performance is optimal when adding prompts to 18 convolutional layers (82K parameters); adding them to only 5 layers still yields a significant improvement with 23K parameters.
- Increasing the support set size ($K=1\rightarrow 5\rightarrow\text{All}$) continuously improves performance, but $K=5$ already yields most of the benefits.
- Compared to MetaHTR's WRA of 89.2%, MetaWriter achieves 89.7% (IAM) but uses unlabeled samples and 20x fewer parameters.
- Personalization analysis on 20 different writers shows that MetaWriter outperforms the baseline on all individuals, demonstrating consistent adaptability.

## Highlights & Insights
- **Elegant Design of Padding-as-Prompt**: Replacing the fixed padding of convolutional layers with learnable parameters as prompt vectors is an extremely lightweight prompt tuning scheme. It inserts prompts into the existing computation flow of CNNs with zero extra FLOPs. This idea can be generalized to few-shot adaptation of any vision task using convolutional networks.
- **Meta-Learning for Self-Supervised to Task Loss Alignment**: The core insight of meta-prompt is that "the self-supervised gradient direction may be inconsistent with the task gradient direction", and meta-learning is used to find an initialization point that aligns the two. This framework is universally applicable to any combination of "auxiliary self-supervised + main task".
- **Practicality-Oriented**: 0.08M trainable parameters + unlabeled adaptation = instant personalization on edge devices, which is highly suitable for handwriting recognition application scenarios such as on iPads.

## Limitations & Future Work
- Only evaluated at the line-level, without addressing page-level recognition. Page-level recognition needs to consider text position and reading order, which simple prompt padding cannot encode.
- Whether the MAE 75% masking ratio is optimal has not been fully explored.
- The episode training of meta-learning relies on writer identity labels in the dataset, restricting its application to unlabeled data.
- Compared with methods based on large-scale pre-trained models (e.g., TrOCR with 334M parameters), the base model is relatively small, which might limit its performance upper bound.
- The interpretability of prompt vectors has not been explored—what style features do the prompts actually encode?

## Related Work & Insights
- **vs MetaHTR**: MetaHTR requires 16 labeled samples + full-parameter fine-tuning (1.7M), whereas MetaWriter requires only 5 unlabeled samples + 0.08M parameters, significantly improving usability.
- **vs VPT (Visual Prompt Tuning)**: VPT adds prompt tokens to the input space of ViT, whereas MetaWriter adds prompts in the padding dimension of CNNs. The concepts are similar, but MetaWriter's adaptation for CNN architectures is more elegant.
- **vs TrOCR**: TrOCR relies on large-scale pre-training (334M parameters) to achieve generalization, while MetaWriter achieves or even surpasses its performance on a small model through personalized adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of Padding-as-Prompt + Meta-MAE is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are relatively comprehensive, but experiments are only conducted on two datasets, lacking multi-lingual/multi-script validation.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the algorithm description is detailed, though some sections are slightly redundant.
- Value: ⭐⭐⭐⭐ There is a clear breakthrough in the practicality of personalized HTR. Test-time adaptation with 0.08M parameters has direct value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DataRater: Meta-Learned Dataset Curation](../../NeurIPS2025/self_supervised/datarater_meta-learned_dataset_curation.md)
- [\[CVPR 2025\] Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)
- [\[CVPR 2025\] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [\[CVPR 2025\] Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)
- [\[CVPR 2025\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)

</div>

<!-- RELATED:END -->
