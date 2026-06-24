---
title: >-
  [Paper Note] RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing
description: >-
  [ECCV 2024][Medical Imaging][diffusion image editing] Proposes RadEdit, a diffusion-based medical image editing method that introduces a dual-mask mechanism (edit mask and keep mask) to break spurious correlations in data, generating high-quality synthetic test suites to stress-test the robustness of biomedical vision models against dataset shift.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "diffusion image editing"
  - "stress-testing"
  - "dataset shift"
  - "chest X-ray"
  - "biomedical vision"
date: 2026-05-08
content_hash: b731bfc6ab6f1c80
---

# RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing

**Conference**: ECCV 2024  
**arXiv**: [2312.12865](https://arxiv.org/abs/2312.12865)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: diffusion image editing, stress-testing, dataset shift, chest X-ray, biomedical vision

## TL;DR

Proposes RadEdit, a diffusion-based medical image editing method that introduces a dual-mask mechanism (edit mask and keep mask) to break spurious correlations in data, generating high-quality synthetic test suites to stress-test the robustness of biomedical vision models against dataset shift.

## Background & Motivation

**Background**: Biomedical imaging datasets are typically small and biased, causing the real-world performance of predictive models to be significantly lower than their internal test performance. For instance, none of the hundreds of detection tools developed during the COVID-19 pandemic achieved clinical utility due to methodological flaws and data biases.

**Limitations of Prior Work**: Existing generative image editing methods (such as LANCE and DiffEdit) suffer from severe limitations in medical image editing. LANCE uses global prompts (no masks), which alters the shape and position of lung boundaries and erroneously removes features that should not be modified (e.g., chest tubes) due to spurious correlations. DiffEdit's automatic mask prediction is inaccurate and produces artifacts at mask boundaries.

**Key Challenge**: Pathological features and clinical interventions frequently co-occur in medical data (e.g., pneumothorax and chest tubes), and diffusion models learn these spurious correlations. When editing, removing one feature consequently removes the other, undermining the controllability of stress testing.

**Goal**: To precisely control the modified regions during medical image editing while retaining key features that should remain unchanged, thereby generating reliable synthetic datasets to quantify model robustness across three types of dataset shifts (acquisition, presentation, and population shifts).

**Key Insight**: Introduce two types of masks (edit mask and keep mask), restrict CFG solely to the edited region, use unconditional generation outside the edited region for global consistency, and enforce raw pixel restoration in the keep mask region.

**Core Idea**: Decouple spurious correlations through a dual-mask mechanism to achieve precise, controllable editing of medical images, systematically diagnosing failure modes of biomedical vision models under various dataset shifts.

## Method

### Overall Architecture

RadEdit is based on DDPM inversion and text-guided diffusion model editing. First, a latent diffusion model is trained on multiple chest X-ray datasets (MIMIC-CXR, ChestX-ray8, CheXpert, totaling 487,680 images) using SDXL's VAE and the BioViL-T text encoder. During editing, noise vector sequences are obtained via DDPM inversion, and the edit mask $m_{\text{edit}}$ and keep mask $m_{\text{keep}}$ are incorporated during the reverse generation process to control the editing behavior.

### Key Designs

1. **Dual Mask Mechanism**: The core innovation of RadEdit lies in utilizing two masks simultaneously—the edit mask defines the region to be actively modified, while the keep mask defines the region that must remain unchanged. The two masks do not need to be mutually exclusive; uncovered regions allow the diffusion model to adjust freely to ensure global consistency. The core equations are:

    $\epsilon_t = m_{\text{edit}} \odot \epsilon_t^{\text{CFG}} + (1 - m_{\text{edit}}) \odot \epsilon_{\text{uncond},t}$

    $x_{t-1} = m_{\text{keep}} \odot \hat{x}_{t-1} + (1 - m_{\text{keep}}) \odot x_{t-1}$

   **Design Motivation**: Spurious correlations are typically spatially non-overlapping, which means they can be effectively decoupled using masks. For instance, when removing a pneumothorax, setting the pneumothorax region as the edit mask and the chest tube region as the keep mask successfully retains the chest tube.

2. **Localized Classifier-Free Guidance (Localized CFG)**: Unlike DiffEdit, which applies CFG across the entire image, RadEdit applies CFG (with a weight of 15) solely within the edit mask, while using unconditional noise generation $\epsilon_{\text{uncond},t}$ in external regions. This provides the following benefits: (a) high CFG weights ensure that pathological features are completely removed; (b) it prevents CFG from causing unwanted changes to other parts of the image; and (c) it simplifies prompt construction by eliminating the need to consider the prompt's effect on the entire image.

3. **BioViL-T Editing Score**: A quality control mechanism used to filter out low-quality edited results. It is defined based on directional similarity:

    $S_{\text{BioViL-T}} = \frac{\Delta I \cdot \Delta T}{\|\Delta I\| \|\Delta T\|}$

   where $\Delta I = E_I(I_{\text{edit}}) - E_I(I_{\text{real}})$ and $\Delta T = E_T(T_{\text{edit}}) - E_T(T_{\text{real}})$. A domain-specific BioViL-T vision-language model is utilized as the encoder, with a threshold set to 0.2.

4. **DDPM Inversion**: Rather than using DDIM inversion, the paper adopts DDPM inversion proposed by Huberman-Spiegelglas et al. DDPM inversion samples statistically independent noise vectors $\tilde{\epsilon}_{1:T}$ and isolates $z_t$ for the generation process, preserving the original image structure better than DDIM inversion.

### Loss & Training

- The diffusion model is trained using the standard denoising loss
- Training data: MIMIC-CXR (using the impression section of radiology reports as the text condition), ChestX-ray8, and CheXpert (using lists of labels as the text condition)
- Images are uniformly downsampled and center-cropped to 512×512
- The text encoder is from BioViL-T (frozen), and the VAE is from SDXL (frozen)
- Post-editing quality control filters low-quality samples ($S < 0.2$) using the BioViL-T editing score

## Key Experimental Results

### Main Results

| Experimental Scenario | Predictor | Test Data | Accuracy |
|---------|--------|---------|-------|
| Acquisition Shift (COVID-19) | Weak Predictor | Biased Test Set | 99.1 ± 0.2 |
| Acquisition Shift (COVID-19) | Weak Predictor | Synthetic Test Set | **5.5 ± 2.1** (↓95%) |
| Acquisition Shift (COVID-19) | Strong Predictor | Biased Test Set | 74.4 ± 3.0 |
| Acquisition Shift (COVID-19) | Strong Predictor | Synthetic Test Set | 76.0 ± 7.7 |
| Presentation Shift (Pneumothorax) | Weak Predictor | Biased Test Set | 93.3 ± 0.6 |
| Presentation Shift (Pneumothorax) | Weak Predictor | Synthetic Test Set | **17.9 ± 3.7** (↓75%) |
| Presentation Shift (Pneumothorax) | Strong Predictor | Biased Test Set | 93.7 ± 1.3 |
| Presentation Shift (Pneumothorax) | Strong Predictor | Synthetic Test Set | 81.7 ± 7.1 |

### Ablation Study

| Editing Method / Pathology Type | Weak Predictor Dice ↑ | Weak Predictor ΔDice | Strong Predictor Dice ↑ | Strong Predictor ΔDice |
|----------------|---------------|---------------|---------------|---------------|
| Real Data (Baseline) | 97.4 | — | 95.5 | — |
| Healthy -> Edema | 93.8 | -3.6 | 93.9 | -1.6 |
| Healthy -> Pacemaker | 85.0 | **-12.4** | 87.3 | -8.2 |
| Healthy -> Consolidation | 85.9 | **-11.5** | 88.1 | -7.4 |

### Key Findings

1. **Weak predictors are extremely vulnerable to acquisition shift**: The accuracy of the COVID-19 classifier plummeted from 99.1% to 5.5% on the synthetic test set, proving that the model relies on spurious features from the data source (such as lateral markers) rather than actual pathological features.
2. **Strong predictors validate editing quality**: The strong predictor exhibits similar performance on both real and synthetic test sets (76.0% vs 74.4%), demonstrating that the performance drop is indeed caused by shifts rather than editing artifacts.
3. **LANCE and DiffEdit perform poorly in mask prediction**: DiffEdit predicts a pneumothorax mask with a Dice score of only 18.4%, and frequently includes chest tubes incorrectly in the predicted mask.
4. **RadEdit enables stress-testing of segmentation models for the first time**: Since the lung boundaries remain unchanged after editing, the original segmentation annotations can be reused directly.

## Highlights & Insights

- **The design of the dual-mask mechanism is highly elegant**: By using complementary edit and keep masks, it elegantly resolves spurious correlations. This design concept can be generalized to controllable editing tasks in other domains.
- **A paradigm shift from "data augmentation" to "stress-testing"**: Distinct from previous studies that leverage synthetic data to improve model performance, this work focuses on using synthetic data to expose model deficiencies, which is extremely valuable for safety evaluations prior to clinical deployment.
- **Zero-shot editing capability**: The diffusion model can perform edits on datasets/pathologies not seen in its training set, demonstrating exceptional generalization.
- **The BioViL-T editing score** provides a quantitative method to evaluate editing quality, although the authors acknowledge that it cannot detect all artifacts introduced by LANCE and DiffEdit.

## Limitations & Future Work

1. **Manual analysis of training data and prediction of potential failure cases is required**: Automated failure mode discovery remains an open challenge, presenting an area for future exploration.
2. **Inability to handle all types of shift testing**: For example, cardiomegaly alters the segmentation annotations, which is not directly supported by the current method.
3. **Uncertain relationship between editing quality and downstream performance**: Performance degradation does not necessarily reflect real-world performance; it could potentially stem from suboptimal editing quality.
4. **Reliance on the pretrained BioViL-T model for filtering**: The evaluation model itself might introduce biases.
5. **Sensitivity to hyperparameters**: Parameters such as CFG weight, inference steps, and encoding timesteps markedly impact editing quality.

## Related Work & Insights

- **LANCE** [Prabhu et al.]: Uses LLMs to modify image captions for editing images in stress-testing, but its use of global prompts leads to uncontrollable modifications.
- **DiffEdit** [Couairon et al.]: Automatically predicts editing masks via text prompts, but suffers from insufficient mask prediction accuracy and tends to include spurious correlation regions.
- **DDPM Inversion** [Huberman-Spiegelglas et al.]: Improves the structure-preserving capability of DDIM inversion and serves as an important technical foundation for RadEdit.
- **BioViL-T** [Bannur et al.]: A domain-specific vision-language model, providing the foundation for evaluating the quality of medical image edits.
- The core insight from this work: in medical image editing, **spurious correlations must be explicitly handled**; simple masks or global edits are insufficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-mask mechanism is simple yet effective, and expanding stress-testing to segmentation models is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The three shift scenarios are elegantly designed, the comparison between weak and strong predictors is convincing, and the methodological comparison is comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is logically clear, the motivation is fully elaborated, and the algorithmic pseudocode is well-structured.
- **Value**: ⭐⭐⭐⭐ Highly significant for safety evaluations of medical AI prior to deployment, though it requires manual design of test scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding](../../CVPR2026/medical_imaging/llada-medv_exploring_large_language_diffusion_models_for_biomedical_image_unders.md)
- [\[ECCV 2024\] Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model](co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)
- [\[ECCV 2024\] GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation](gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)
- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](../../CVPR2025/medical_imaging/latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](../../ICLR2026/medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)

</div>

<!-- RELATED:END -->
