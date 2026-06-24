---
title: >-
  [Paper Note] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)
description: >-
  [CVPR 2025][Medical Imaging][Multi-modal MRI Pre-training] BrainMVP proposes the first multi-modal vision pre-training paradigm. By using three pretext tasks—cross-modal masked reconstruction, modal template distillation, and modality-aware contrastive learning—it pre-trains a ViT on 16,022 multi-parametric brain MRI scans (over 2.4 million images). It outperforms SOTA methods on six segmentation and four classification downstream tasks, with an improvement in Dice Score of u…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Multi-modal MRI Pre-training"
  - "Cross-modal Reconstruction"
  - "Modal Data Distillation"
  - "Modality-Aware Contrastive Learning"
  - "Brain MRI"
date: 2026-05-08
content_hash: 771f02c131aa015a
---

# Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)

**Conference**: CVPR 2025  
**arXiv**: [2410.10604](https://arxiv.org/abs/2410.10604)  
**Code**: [https://github.com/openmedlab/BrainMVP](https://github.com/openmedlab/BrainMVP)  
**Area**: Medical Images / Self-Supervised Pre-training  
**Keywords**: Multi-modal MRI Pre-training, Cross-modal Reconstruction, Modal Data Distillation, Modality-Aware Contrastive Learning, Brain MRI

## TL;DR
BrainMVP proposes the first multi-modal vision pre-training paradigm. By using three pretext tasks—cross-modal masked reconstruction, modal template distillation, and modality-aware contrastive learning—it pre-trains a ViT on 16,022 multi-parametric brain MRI scans (over 2.4 million images). It outperforms SOTA methods on six segmentation and four classification downstream tasks, with an improvement in Dice Score of up to 14.47%.

## Background & Motivation
1. **Background**: Self-supervised pre-training (SSL) in medical image analysis is primarily conducted on single-modality data—such as CT (e.g., VoCo), MRI (e.g., M3AE), X-ray, etc.—or trained on mixed-modality data where each modality is processed independently. These methods mainly rely on instance-level discrimination (contrastive learning) or image reconstruction (MAE).
2. **Limitations of Prior Work**: (a) Single-modality SSL cannot model cross-modality relationships. However, in clinical practice, multi-parametric MRI (mpMRI) scans of the same patient naturally possess strong correspondences and contain complementary pathological features. (b) While mixed-modality SSL performs joint training, different data sources limit cross-modality understanding. (c) Missing modalities are frequently encountered in reality—acquiring complete mpMRI is constrained by acquisition protocols and scanner limitations, making modality mismatch common in large-scale pre-training data.
3. **Key Challenge**: Multi-modal MRI data are naturally paired and complementary, but existing pre-training frameworks do not fully exploit cross-modal correlations to learn transferable representations. Meanwhile, a disconnect exists between pre-training pretext tasks and downstream task objectives, lacking a bridge to connect them.
4. **Goal**: (1) How to leverage the cross-modal correlation of mpMRI to learn more general representations? (2) How to handle missing/mismatched modalities during pre-training? (3) How to bridge the gap between pre-training pretext tasks and downstream applications?
5. **Key Insight**: The authors leverage the high structural similarity of different MRI modalities (which differ only in contrast in specific regions) to design a cross-modal masked reconstruction task that forces the model to learn translation relationships across modalities. Concurrently, inspired by dataset distillation, learnable modal templates are optimized to serve as an information bridge between pre-training and downstream tasks.
6. **Core Idea**: Specifically, replacing masked regions with patches from another modality to reconstruct the original modality, distilling privacy-free modal templates to bridge downstream applications, and employing contrastive learning to maintain cross-modal feature consistency work synergistically to learn modality-aware general representations.

## Method

### Overall Architecture
The input consists of single-modality MRI volumes, using a ViT encoder-decoder architecture. The pre-training simultaneously performs three pretext tasks: (1) Cross-modal masked reconstruction: the majority of the input image is replaced with corresponding patches from another modality's image, forcing the model to reconstruct the original modality from an input dominated by another modality's information; (2) Modal data distillation: similar to (1), but the replacement source is a learnable modal template instead of another modality's image, and templates are automatically optimized via backpropagation; (3) Modality-aware contrastive learning: the encoded features of the two masked versions of the same modality image generated in (1) and (2) are treated as positive pairs and aligned using the InfoNCE loss. The final loss is a weighted sum of the three losses.

### Key Designs

1. **Cross-modal Masked Reconstruction**:

    - Function: Learn cross-modal representations and translation relationships between different MRI modalities
    - Mechanism: Given a single-modality input $X_{im}$, most of its regions are randomly masked and filled with corresponding patches from another modality $X_{in}$ of the same patient. The mask-and-fill operation is performed repeatedly until the patch-masking ratio reaches $p^*=0.875$. The resulting input $\Phi_{modal}(X_{im}, X_{in})$ primarily contains information of modality $n$, but the reconstruction target is modality $m$. Since the anatomical structures across mpMRI modalities are highly similar and only differ in local contrast in specific regions, this seemingly difficult cross-modal reconstruction task is highly feasible for brain MRIs. The loss is defined as $\mathcal{L}_{CMR} = \|\mathcal{F}_{dec}(\mathcal{F}_{enc}(\Phi_{modal}(X_{im}, X_{in}))) - X_{im}\|_2$.
    - Design Motivation: By avoiding the introduction of skip connections, the encoder is forced to represent sufficient cross-modal semantic information in its latent representation to support reconstruction. Thus, the learned representation is modality-agnostic and encapsulates fused information from all modalities.

2. **Modal Data Distillation and Templates**:

    - Function: Learn anatomical modal templates devoid of patient-specific information to bridge pre-training and downstream applications
    - Mechanism: A set of learnable templates initialized to all zeros $T = \{T_m\}_{m=1}^S$ (where $S$ is the number of modalities, and the size matches the input) is defined. Similar to cross-modal reconstruction, $T_m$ is used to fill the masked regions instead of another modality: $\mathcal{L}_{MD} = \|\mathcal{F}_{dec}(\mathcal{F}_{enc}(\Phi_{distill}(X_{im}, T_m))) - X_{im}\|_2$. The templates are optimized along the pre-training trajectory via backpropagation, eventually converging to a compact structural representation of each modality.
    - Design Motivation: Inspired by dataset distillation—where the distilled dataset can achieve training performance close to the original dataset. The modal templates preserve shared structural and statistical information of a specific modality without leaking patient privacy, and can serve as a data augmentation source to bridge domain gaps in downstream tasks.

3. **Modality-Aware Contrastive Learning**:

    - Function: Maintain consistency across modalities and between templates and images at the feature level
    - Mechanism: Both $\Phi_{modal}(X_{im}, X_{in})$ and $\Phi_{distill}(X_{im}, T_m)$ retain a proportion of $(1-p^*)$ information from modality $m$, possessing partial semantic consistency. The encoded features of both, $f_{im}$ and $g_{im}$, are treated as positive pairs and aligned using the InfoNCE loss: $\mathcal{L}_{CL} = \frac{1}{2}(\mathcal{L}_{f_{im}\to g_{im}} + \mathcal{L}_{g_{im}\to f_{im}})$. This is introduced at epoch 1000 (after the visual quality of the template has converged).
    - Design Motivation: While reconstruction operates at the pixel level, contrastive learning introduces modality invariance at the feature level, making the two tasks complementary.

### Loss & Training
Total loss: $\mathcal{L}_{SSL} = \frac{1}{|\mathcal{B}|}\sum_{i}\frac{1}{M_i}\sum_m(\mathcal{L}_{CMR} + \lambda_{MD}\mathcal{L}_{MD} + \lambda_{CL}\mathcal{L}_{CL})$, with $\lambda_{MD}=\lambda_{CL}=1.0$. UniFormer or UNET3D is used as the backbone. Training is performed on 8 RTX 4090 GPUs for 1500 epochs with a batch size of 3, using the AdamW optimizer and an initial learning rate of 3e-4 with cosine decay. The pre-training dataset consists of 16,022 mpMRI scans (3,755 patients, 8 modalities) from five sources, including BraTS2021/2023, UCSF-PDGM, and IXI. In downstream tasks, modal templates are randomly substituted into a portion of the multi-modal input for data augmentation.

## Key Experimental Results

### Main Results
Segmentation tasks (Dice Score %):

| Dataset | BrainMVP(UniFormer) | M3AE(UniFormer) | Mask3D Concept Equivalent | Gain |
|--------|-------------------|-----------------|--------------|------|
| BraTS2023-PED (AVG) | **76.80** | 74.14 | - | +2.66 |
| BraTS-MET (AVG) | **73.67** | 70.39 | - | +3.28 |
| ISLES22 (IS) | **86.60** | 86.32 | - | +0.28 |
| MRBrainS13 (AVG) | **80.27** | 77.29 | - | +2.98 |
| VSseg (VS) | **83.64** | 79.31 | - | +4.33 |
| UPENN-GBM (AVG) | **90.01** | 89.63 | - | +0.38 |

Classification tasks (Accuracy):

| Dataset | BrainMVP | Best Competing Method | Gain |
|--------|----------|------------|------|
| BraTS2018 (ACC) | **0.8596** | 0.7895(UNETR) | +7.01% |
| ADNI (ACC) | **0.6218** | 0.6092(MoCov3) | +1.26% |
| ADHD-200 (ACC) | **0.6948** | 0.6818(TransVW) | +1.30% |
| ABIDE-I (ACC) | **0.6545** | 0.6424(GVSL) | +1.21% |

### Ablation Study
Based on Table 4 (ablation in supplementary material, inferred from the paper context):

| Configuration | Description |
|------|------|
| CMR only | Cross-modal reconstruction baseline |
| CMR + MD | Adding modal distillation, improving generalization |
| CMR + MD + CL | Full method, introducing contrastive learning at epoch 1000 |

Label efficiency experiments show: BrainMVP achieves performance close to other methods using 100% of the data with only 20% of the labeled data, and consistently maintains its advantage as the data ratio increases.

### Key Findings
- General SSL methods (e.g., MAE3D, MoCov3) perform significantly worse on medical images than domain-specific medical SSL, with gaps of up to 9%+ in Dice Score.
- The key to the feasibility of cross-modal reconstruction in mpMRI is that anatomical structures across different MRI modalities are highly similar.
- Modal templates show significant data augmentation effects in downstream applications—substituting templates for partial multi-modal inputs enhances robustness under missing modality scenarios.
- Contrastive learning needs to be introduced after template convergence (epoch 1000); premature introduction causes instability.
- BrainMVP is effective across diverse tasks such as brain tumor segmentation, brain metastasis segmentation, and ischemic stroke lesion segmentation, demonstrating strong generalization capability.

## Highlights & Insights
- The design of **Cross-modal Masked Reconstruction** is highly ingenious: by leveraging the domain characteristic of anatomical similarity across medical MRI modalities, masked regions are filled with another modality instead of random noise, forcing the model to learn cross-modal translation rather than simple inpainting. This design is applicable to any scenario where multi-modal data is naturally paired (e.g., remote sensing multi-spectral data, multi-angle captures).
- The concept of using **Modal Templates as a pre-training-downstream bridge** is highly novel: inspired by dataset distillation but used differently—not for compressing datasets, but to learn privacy-free modality priors that can be used as augmentations in downstream tasks.
- The single-modality input design enables natural support for missing modalities, which is far more practical than methods requiring a fixed number of modalities.

## Limitations & Future Work
- Validation is limited to brain MRI, without extending to other anatomical regions (e.g., abdominal/chest multi-modal MRI) or other modality combinations (e.g., CT-to-MRI cross-modal translation).
- The modal templates are currently configured as a single global template per modality, without considering variations across disease types or patient sub-populations.
- Although the pre-training dataset of 3,755 cases is large-scale in the medical domain, it remains limited compared to natural images.
- The timing of introducing contrastive learning (epoch 1000) is empirical and lacks theoretical guidance.
- Although experiments on 13 different downstream datasets are comprehensive, the training details for each dataset may require heavy hyperparameter tuning.

## Related Work & Insights
- **vs M3AE**: M3AE also uses MIM and multiple modalities but only performs cross-modal masked reconstruction on MRI; BrainMVP adds modal distillation and contrastive learning, and is pre-trained on a much larger dataset.
- **vs VoCo**: VoCo utilizes contextual spatial position priors in CT to learn consistent representations, whereas BrainMVP leverages cross-modal relationships in mpMRI to learn modality-agnostic representations, presenting different angles of approach.
- **vs MultiMAE**: MultiMAE requires semantic labels during pre-training and multi-modal fine-tuning, whereas BrainMVP only requires unlabeled mpMRI pre-training and supports single-modality fine-tuning.
- The concept of modal templates can be generalized into "domain knowledge capsules"—encapsulating domain priors learned from large-scale pre-training into transferable, parameterized modules.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The designs of the three pretext tasks are highly thoughtful, and the modal template is a completely fresh concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, comprising 10 downstream tasks (6 segmentation + 4 classification), label efficiency experiments, and comparisons across various backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear framework with coherent motivation and design logic for each module.
- Value: ⭐⭐⭐⭐⭐ The first multi-modal MRI vision pre-training paradigm, exerting a paradigm-shifting impact on medical image SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[CVPR 2026\] InvCoSS: Inversion-driven Continual Self-supervised Learning in Medical Multi-modal Image Pre-training](../../CVPR2026/medical_imaging/invcoss_inversion-driven_continual_self-supervised_learning_in_medical_multi-mod.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)

</div>

<!-- RELATED:END -->
