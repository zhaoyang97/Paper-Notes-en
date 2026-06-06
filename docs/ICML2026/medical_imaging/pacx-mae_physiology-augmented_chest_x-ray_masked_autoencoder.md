---
title: >-
  [Paper Note] PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder
description: >-
  [ICML 2026][Medical Imaging][Chest X-Ray] PaCX-MAE performs LoRA fine-tuning on a Chest X-Ray (CXR) ViT pretrained with MAE, utilizing frozen encoders for ECG and laboratory tests as teachers. Through a dual distillation…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Chest X-Ray"
  - "MAE"
  - "Physiological Signal Distillation"
  - "ECG"
  - "Cross-Modal Alignment"
date: 2026-05-08
content_hash: da46fa1a96ac9421
---

# PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder

**Conference**: ICML 2026  
**arXiv**: [2606.01537](https://arxiv.org/abs/2606.01537)  
**Code**: https://github.com/Lyce24/PACX-MAE (Available)  
**Area**: Medical Imaging / Self-Supervised Representation / Cross-Modal Distillation  
**Keywords**: Chest X-Ray, MAE, Physiological Signal Distillation, ECG, Cross-Modal Alignment

## TL;DR
PaCX-MAE performs LoRA fine-tuning on a Chest X-Ray (CXR) ViT pretrained with MAE, utilizing frozen encoders for ECG and laboratory tests as teachers. Through a dual distillation objective of InfoNCE contrastive learning and cosine regression, "invisible physiological context" is injected into the pure image encoder. At inference time, using only the CXR, the model outperforms the same-architecture MAE baseline across 9 downstream benchmarks, with particularly significant gains in physiology-dependent tasks (MedMod +2.7 AUROC, VinDr +6.5 F1).

## Background & Motivation

**Background**: Self-supervised representations in medical imaging generally follow two paths—pure contrastive learning (MoCo/DINO family) or pure reconstruction (MAE family). MAE trained on CheXpert/MIMIC has become a strong baseline for CXR representation because reconstruction forces the model to memorize fine-grained anatomical structures rather than just learning global invariance. While multimodal methods can fuse signals like ECG and lab tests to characterize a patient's dynamic state, they typically assume all modalities are available during inference.

**Limitations of Prior Work**: In clinical reality, especially in emergency scenarios, the CXR is often the **only modality immediately available**—ECG leads might not be connected, and lab results are pending. Consequently, multimodal models cannot be deployed due to missing modalities, while unimodal models fail to capture the systemic physiological context behind the CXR appearance (e.g., pulmonary vascular redistribution corresponding to fluid overload, or cardiomegaly corresponding to heart failure). These physiological states exist as "implicit fingerprints" within anatomical images, but standard vision pretraining is not guided to focus on them.

**Key Challenge**: The CXR itself encodes significant physiological information, but (i) direct supervision is hindered by a scarcity of labels, and (ii) multimodal alignment is constrained by the deployment requirement of having paired data during inference. An ideal solution should use paired data only during pretraining, while requiring only CXR during inference.

**Goal**: This is addressed as two sub-problems: (a) how to "embed" dense physiological structures from ECG/labs into the visual encoder; and (b) how to do so without destroying the anatomical details already learned by MAE.

**Key Insight**: The authors draw inspiration from privileged information and cross-modal knowledge distillation. Physiological modalities are treated as privileged signals, where the teachers are not text or other images, but two streams of dense physiological encoders (ECG + Laboratory tests).

**Core Idea**: A two-stage curriculum—first, perform unimodal pretraining to obtain strong teachers; then, use LoRA on the frozen visual backbone for "contrastive + regression" dual-target distillation to align the physiological manifold with the visual manifold. All components except the CXR encoder are discarded during inference.

## Method

### Overall Architecture
The framework consists of two stages. Stage 1: Independent pretraining of three unimodal encoders—CXR using MAE, ECG using ECGFounder (a transformer pretrained on 10 million ECGs), and laboratory tests using a mask-aware denoising autoencoder (modeling both values and structural missingness). Stage 2: Cross-modal distillation on ~10k CXR-ECG-Lab triplets from Symile-MIMIC. The ECG/Lab encoders are frozen as teachers, and LoRA is added to the CXR ViT to match the teacher embeddings via contrastive and regression heads. All projection/regression heads are discarded for downstream inference, leaving only the visual backbone.

### Key Designs

1.  **MAE Strong Anatomical Prior + 90% Extreme Masking Ratio**:
    - **Function**: Provides a visual initialization that already understands global anatomy for Stage 2.
    - **Mechanism**: Utilizes ViT-B/MAE on CheXpert with a masking ratio of 0.90 (significantly higher than the 0.75 used for natural images). Extreme sparsity forces the model to infer global anatomical semantics (heart silhouette, mediastinum, diaphragm) from very few patches rather than relying on local texture shortcuts. It also avoids the invariance enforced by contrastive learning, which might wash out low-contrast intensity clues (e.g., effusion density).
    - **Design Motivation**: The authors argue that "reading physiology from CXR" requires the model to first understand global anatomy. MAE’s reconstruction objective naturally preserves fine-grained intensity information, making it more suitable than contrastive learning for subsequent physical quantity regression.

2.  **LoRA + LayerNorm Tuning to Avoid Catastrophic Forgetting**:
    - **Function**: Minimally alters the backbone during cross-modal alignment by introducing <1% trainable parameters.
    - **Mechanism**: The visual backbone is frozen; only LayerNorm layers are unfrozen to adapt the feature distribution. LoRA low-rank matrices are injected into the `qkv` of attention and `fc` modules of the FFN. Trainable parameters in Stage 2 are primarily LoRA and the lightweight projection/regression heads.
    - **Design Motivation**: The dense anatomical prior from Stage 1 is valuable for "pixel-level fidelity" in tasks like segmentation. Full fine-tuning on ~10k paired samples could deviate from this and lead to forgetting. LoRA keeps the visual manifold stable while learning the "bridge" to the physiological manifold. This explains why PaCX maintains equivalent performance to MAE on CXL-Seg and COVID-QU-Ex (IoUs of 0.996 and 0.942).

3.  **Dual Distillation: Contrastive + Regression**:
    - **Function**: Simultaneously achieves global semantic alignment and dense feature replication.
    - **Mechanism**: Total loss $\mathcal{L}_{total}=\lambda_C \mathcal{L}_{contrastive}+\lambda_R \mathcal{L}_{regression}$. $\mathcal{L}_C$ projects CXR/ECG/Lab into a shared space using symmetric InfoNCE with learnable temperature. It uses **global negative sample collection** across GPUs to maximize diversity and mitigate within-batch bias, along with label smoothing ($\epsilon=0.02$). $\mathcal{L}_R$ uses modality-specific regression heads to predict the **unprojected** original embeddings of the teachers, minimizing $1-\cos(\hat{y},y)$.
    - **Design Motivation**: Contrastive learning alone might learn "sufficiently discriminative but coarse" shortcuts. The regression head forces the visual encoder to "replicate" the dense structure of the teacher embeddings, internalizing physiological signals into token representations.

### Loss & Training
The total objective follows the formula above. Stage 1 involves independent training (Full CheXpert for MAE, ECGFounder weights for ECG, mask-aware DAE for Labs). Stage 2 involves joint training on Symile-MIMIC. The visual backbone is mostly frozen (except LN + LoRA), and teachers are frozen throughout to prevent degenerate co-adaptation, ensuring that knowledge is transferred rather than jointly re-learned. All downstream evaluations use linear probing (frozen backbone).

## Key Experimental Results

### Main Results (Linear Probing across 9 Benchmarks)

| Dataset | Metric | ImageNet | MAE | Ours | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TB | AUROC / F1 | 0.887 / 0.818 | 0.899 / 0.814 | 0.910 / 0.846 | +1.1 / +3.2 |
| CheXchoNet | AUROC / F1 | 0.728 / 0.147 | 0.788 / 0.215 | 0.803 / 0.266 | +1.5 / +5.1 |
| VinDr-CXR | AUROC / F1 | 0.751 / 0.097 | 0.847 / 0.191 | 0.871 / 0.256 | +2.4 / +6.5 |
| NIH-14 | AUROC / F1 | 0.721 / 0.048 | 0.772 / 0.113 | 0.783 / 0.115 | +1.1 / +0.2 |
| MedMod | AUROC / F1 | 0.612 / 0.091 | 0.695 / 0.231 | 0.722 / 0.253 | +2.7 / +2.2 |
| ChestX6 | AUROC / F1 | 0.983 / 0.876 | 0.988 / 0.905 | 0.989 / 0.906 | +0.1 / +0.1 |
| CXL-Seg | IoU / Dice | 0.984 / 0.992 | 0.996 / 0.998 | 0.996 / 0.998 | 0 / 0 |
| COVID-QU-Ex | IoU / Dice | 0.894 / 0.943 | 0.942 / 0.970 | 0.942 / 0.970 | 0 / 0 |
| QaTa-COV19 | IoU / Dice | 0.622 / 0.766 | 0.726 / 0.841 | 0.715 / 0.833 | -1.1 / -0.8 |

### Ablation Study (Three Physiology-Dependent Benchmarks)

| Configuration | CheXchoNet AUC/F1 | MedMod AUC/F1 | VinDr AUC/F1 | Note |
| :--- | :--- | :--- | :--- | :--- |
| ECG Teacher Only | 0.801 / 0.296 | 0.717 / 0.243 | 0.871 / 0.233 | No Lab, VinDr F1 drops 2.3 |
| Lab Teacher Only | 0.795 / 0.275 | 0.721 / 0.245 | 0.875 / 0.248 | No ECG, CheXchoNet F1 drops |
| $\mathcal{L}_{cont}$ Only | 0.799 / 0.273 | 0.722 / 0.258 | 0.866 / 0.241 | No Reg, VinDr F1 -1.5 |
| $\mathcal{L}_{reg}$ Only | 0.789 / 0.227 | 0.673 / 0.131 | 0.843 / 0.130 | No Cont, Significant degradation |
| Full PaCX | 0.803 / 0.266 | 0.722 / 0.253 | 0.871 / 0.256 | Best overall |

### Key Findings
- Ours wins decisively on physiology-dependent tasks (VinDr, CheXchoNet, MedMod). It maintains parity on segmentation tasks, proving the LoRA + frozen backbone strategy **prevents forgetting anatomical knowledge**, although some purely textural segmentation tasks do not benefit from physiological priors.
- **Maximum gain in low-data regimes**: With 1% training data, CheXchoNet AUROC increases by +8.2, with MedMod and VinDr seeing ~+5. The physiological prior act as a strong regularizer.
- **Zero-shot Alignment**: Cosine similarity with ECG/Lab teacher embeddings improved from 0.204/0.239 (MAE) to 0.229/0.252, indicating visual representations are effectively "pulled" toward the physiological manifold.
- **Attention rollout** shows PaCX shifts focus from bony structures (like the clavicle) to soft tissues (heart silhouette/mediastinum), providing qualitative evidence that the model learns to look where the ECG information originates.
- The most counter-intuitive result is from the loss ablation: using regression alone leads to a **complete collapse**, suggesting that contrastive loss is necessary to provide discriminative pressure.

## Highlights & Insights
- **"Phantom modality" framework is clever**: It treats invisible physiological states as privileged information to be distilled into the image encoder. This logic can be transferred to any medical scenario where multimodal data is available during training but only unimodal data exists at inference.
- **MAE × LoRA × Dual Distillation**: MAE provides the anatomical prior, LoRA prevents forgetting, and dual distillation ensures both global and dense learning. This combination is a complete recipe for cross-modal alignment on small paired datasets.
- **Global negatives + label smoothing**: These small engineering details for handling medical noise and limited batch sizes are highly practical.
- **Predicting "unprojected" teacher embeddings**: This preserves the original manifold structure of the teacher, a detail often overlooked but crucial here.

## Limitations & Future Work
- **Single-center data**: Uses only MIMIC-IV; cross-center generalization is unverified.
- **Global alignment loses regional correspondence**: Currently aligns whole images to whole ECG/Lab vectors. It does not model fine-grained mappings like "ST-elevation ↔ specific heart cardiac region."
- **Small paired dataset size**: Roughly 10k triplets. While the curriculum and LoRA help, the scale is still limited compared to millions of samples.
- **Limited physiological modalities**: Only considers ECG and Labs.
- **Performance dip on QaTa-COV19**: Suggests distillation might be counterproductive when physiological priors do not match the pathological patterns.
- **Linear probing focus**: Does not compare against end-to-end fine-tuning.

## Related Work & Insights
- **Vs. standard CXR MAE**: Previous works perform unimodal reconstruction; Ours adds a layer of physiological distillation to gain on physiology-dependent tasks without losing anatomical precision.
- **Vs. CLIP-style alignment (e.g., BioViL)**: They align images with report text (semantic level); Ours aligns with dense physiological numerical manifolds (continuous and structured), which is more physically grounded.
- **Vs. ECGFounder**: Directly reusing a foundation model as a frozen teacher is a strategic use of cross-modal knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐ The "privileged distillation" concept itself is not new, but the systematized implementation (MAE + LoRA + Dual Distillation) and evidence that physiological signals are actually learned are solid contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes 9 benchmarks, low-data ablations, modality/loss ablations, zero-shot retrieval, and attention visualization. The main weakness is the single-center data.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth narrative, and good explanation of design choices.
- Value: ⭐⭐⭐⭐ Provides a reusable "recipe" for deployment pain points (missing modalities); code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](../../NeurIPS2025/medical_imaging/variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](../../NeurIPS2025/medical_imaging/radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)

</div>

<!-- RELATED:END -->
