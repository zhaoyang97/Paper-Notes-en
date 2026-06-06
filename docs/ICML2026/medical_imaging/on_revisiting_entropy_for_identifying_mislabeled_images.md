---
title: >-
  [Paper Note] On Revisiting Entropy for Identifying Mislabeled Images
description: >-
  [ICML 2026][Medical Imaging][Mislabeled Image Detection] The authors discover that the phenomenon where "mislabeled samples exhibit persistently high predictive entropy throughout training" is insufficient to distinguish…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Mislabeled Image Detection"
  - "Training Dynamics"
  - "Signed Entropy"
  - "CLIP"
date: 2026-05-08
content_hash: fdcf9766e21e6c4a
---

# On Revisiting Entropy for Identifying Mislabeled Images

**Conference**: ICML 2026  
**arXiv**: [2605.31090](https://arxiv.org/abs/2605.31090)  
**Code**: https://github.com/MedAITech/SEI  
**Area**: Noisy Labels / Robust Learning / Representation Learning / Training Dynamics  
**Keywords**: Mislabeled Image Detection, Training Dynamics, Signed Entropy, CLIP, Medical Imaging

## TL;DR
The authors discover that the phenomenon where "mislabeled samples exhibit persistently high predictive entropy throughout training" is insufficient to distinguish them from hard clean samples. They introduce **signed entropy** by multiplying entropy with a sign bit indicating whether the prediction aligns with the given label, then accumulate this over training epochs into the **SEI** statistic. As a pure plug-and-play method, SEI achieves new SOTA results in mislabeled sample detection (with gains up to 11%+) across multiple medical datasets (ISIC, DeepDRiD, PANDA, CheXpert) and CIFAR-100N.

## Background & Motivation

**Background**: Mislabeled sample detection primarily follows two paths: (1) **Loss-based** methods (e.g., O2U-Net, CORES, AUM), where mislabeled samples exhibit higher losses; (2) **Prediction statistics** methods (e.g., Confident Learning, SIMIFEAT, DEFT, LEMoN), which rely on pre-trained models to perform kNN or clustering in feature or vision-language alignment spaces. These methods either require modifying the training pipeline (multi-stage, special losses, memory modules) or heavily depend on the generalization capability of general-purpose pre-trained models.

**Limitations of Prior Work**: In fields like **medical imaging**, where "images of different classes appear highly similar and even experts make mistakes," the discriminative power of general CLIP models significantly drops. This causes feature-based methods (DEFT, LEMoN) to perform poorly. Conversely, loss/confidence-based methods are extremely sensitive to training fluctuations, and signals captured from a single epoch are unstable. Most critically, **hard clean samples and mislabeled samples are almost indistinguishable in terms of entropy or loss**—both cause model uncertainty. Relying solely on entropy thresholds leads to "wrongful convictions" of hard clean samples.

**Key Challenge**: Entropy only characterizes "distribution uncertainty" and is a **directionless** quantity; it cannot reveal whether the model "believes" the specifically provided label. The fundamental difference between mislabeled and hard clean samples is not the "degree of uncertainty," but the "direction of consistency between model predictions and given labels."

**Goal**: Design a **plug-and-play, single-scalar** mislabeled detection metric that: (a) does not alter the training pipeline; (b) utilizes both the magnitude and direction of entropy; (c) accumulates across epochs to resist fluctuations; and (d) does not rely on the transfer performance of general pre-trained models in the target domain.

**Key Insight**: By decomposing training dynamics into two signals—**the trajectory of entropy evolution** and **the trajectory of prediction-label consistency**—the authors observed a clear pattern: easy clean samples align with labels most of the time with monotonically decreasing entropy; hard clean samples are initially incorrect but eventually correct, with entropy starting high and ending low; mislabeled samples almost consistently produce "the model says it's not what you claimed" signals, with entropy remaining high. This naturally suggests injecting the "consistency" signal into entropy.

**Core Idea**: Multiply Shannon entropy by a sign bit $(-1)^{1[y=\arg\max p]}$ determined by whether the predicted argmax equals the given label, yielding **signed entropy**. Accumulating this over training epochs results in SEI—mislabeled samples receive a strong negative integral, while clean samples receive a positive integral, allowing a single parameter to rank all samples.

## Method

### Overall Architecture
SEI acts as a "statistical probe" attached outside the standard training process. The pipeline consists of: (1) Standard contrastive classification training using CLIP (ResNet-50/ViT-B16 vision encoder + Transformer text encoder), where category names are converted into prompts like `"a photo of [CLS]"` and $p(y=k|\bm{x})$ is calculated via cosine similarity; (2) Calculating signed entropy $\mathcal{H}(\bm{p}^{(t)}(\bm{x}), y)$ for each sample after every epoch; (3) Summing the signed entropy over $t=1..T$ epochs after 150 epochs to obtain the scalar SEI; (4) Using an **auxiliary class** adaptive threshold strategy: randomly assigned $N/(K+1)$ images are relabeled to a non-existent pseudo-class (e.g., "a dermoscopic image showing other lesions"), and their average SEI serves as the cutoff for mislabeled samples. The entire process requires no changes to the original training loop, losses, or modules.

### Key Designs

1.  **Signed Entropy**:
    *   **Function**: Combines "distribution uncertainty" and "prediction-label consistency" into a single scalar to separate hard clean and mislabeled samples.
    *   **Mechanism**: For the posterior $\bm{p}(\bm{x})$ at the current epoch, the signed entropy is defined as $\mathcal{H}(\bm{p}(\bm{x}), y) = (-1)^{\mathbb{1}[y=\arg\max_k p_k(\bm{x})]} \sum_k p_k(\bm{x})\log p_k(\bm{x})$. Note since $\sum p\log p \le 0$, the sign bit effectively makes SEI positive for alignment and negative for misalignment (following the "alignment is positive, misalignment is negative" intuition).
    *   **Design Motivation**: Shannon entropy is always non-negative, and both mislabeled and hard clean samples exhibit "high entropy," making them inseparable. By introducing the sign bit, mislabeled samples (persistent misalignment) and hard clean samples (early misalignment, late alignment) follow **completely opposite trajectories** over training epochs, creating separability for the subsequent integral.

2.  **Signed Entropy Integral (SEI, Temporal Accumulation)**:
    *   **Function**: Compresses a sequence of signed entropy of length $T$ into a single scalar, smoothing out training noise and providing a globally comparable ranking score for all samples.
    *   **Mechanism**: $\mathrm{SEI}(\bm{x},y) = \sum_{t=1}^T \mathcal{H}(\bm{p}^{(t)}(\bm{x}), y)$. Visually: easy clean samples align in almost every epoch $\rightarrow$ large positive integral; hard clean samples accumulate negative values early and positive values later, **canceling each other out** to yield a moderate integral; mislabeled samples are incorrect from start to finish, yielding a high-magnitude negative integral. This naturally maps the three types of samples to three distinct segments on the number line.
    *   **Design Motivation**: Citing observations from AUM/O2U-Net that single-epoch predictions/loss values are noisy and sensitive to hyperparameters, integrating over the entire trajectory "naturally averages out" fluctuations. Ablation results (Table 4) show that SE@T and SE@T/2 (single-epoch slices) perform 8-15 F1 points worse than SEI, and SEI outperforms the unsigned EI (standard entropy integral) by an average of 10+ points, proving both "sign" and "temporal" designs are essential.

3.  **Auxiliary Class Adaptive Threshold**:
    *   **Function**: Transforms the "SEI score $\rightarrow$ mislabeled decision" threshold from manual tuning to data-driven, avoiding the need for recalibration across different datasets/noise rates.
    *   **Mechanism**: Injects $N/(K+1)$ **artificial mislabeled samples**—a batch of samples is randomly selected and their labels are changed to a pseudo-class $K+1$ not present in the original dataset (e.g., using "semantically irrelevant" prompts like "a dermoscopic image showing other lesions"). These samples are definitely mislabeled and behave similarly to real mislabeled samples; thus, their average SEI is used as the cutoff.
    *   **Design Motivation**: Fixed thresholds are not transferable across datasets/noise rates, while using a hold-out clean set violates the "no clean data" assumption. Inspired by AUM's idea of "implanting known bad samples as calibration anchors," the authors use a pseudo-class instead of label flipping to better fit the CLIP paradigm of treating class names as text prompts.

### Loss & Training
SEI **does not introduce new losses**. Training uses standard CLIP image-text contrastive loss (aligning images with text embeddings of $K$ classes via the prompt `"a photo of [CLS]"`). All mislabeled detection actions occur after the forward pass computes $\bm{p}^{(t)}(\bm{x})$ and do not backpropagate. Implementation details: SGD + momentum 0.9 + weight decay $1\times 10^{-4}$, batch size 128, lr $1\times 10^{-3}$, 150 epochs, lr decayed by 10× at epochs 75/115, images resized to $224\times 224$.

## Key Experimental Results

### Main Results
On three medical datasets (ISIC-7, DeepDRiD-5, PANDA-4) with two noise types (symmetric / confusion-calibrated) across 5 noise rates $\eta\in\{0.1...0.5\}$, compared against 10 baselines (INCV, BMM, GMM, AUM, CORES, CL, SIMIFEAT, DEFT, ReCoV, LEMoN) using F1 score.

| Dataset | Noise | $\eta$ | SEI | Second best | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ISIC | symmetric | 0.5 | **83.93** | CORES 82.67 | +1.26 |
| ISIC | confusion | 0.4 | **74.98** | AUM 64.54 | **+10.44** |
| DeepDRiD | symmetric | 0.5 | **78.19** | AUM 75.75 | +2.44 |
| DeepDRiD | confusion | 0.5 | **73.04** | LEMoN 66.82 | +6.22 |
| PANDA | symmetric | 0.3 | **81.46** | AUM 75.95 | +5.51 |
| PANDA | confusion | 0.1 | **73.17** | AUM 61.30 | **+11.87** |
| CheXpert (Real Clinical) | — | — | **83.59** | AUM 80.34 | +3.25 |

The advantage of SEI is significantly larger under confusion-calibrated noise, indicating its robustness to "neighboring classes easily confused by the model"—the predominant form of real clinical mislabeling.

### Ablation Study
| Configuration | ISIC@0.4 | DeepDRiD@0.4 | PANDA@0.4 | Description |
| :--- | :--- | :--- | :--- | :--- |
| EI (Unsigned) | 60.77 | 59.28 | 67.26 | Standard entropy integral |
| SE@T (Final Epoch) | 57.89 | 57.93 | 62.91 | Single epoch slice |
| SE@T/2 (Midway) | 63.72 | 62.84 | 66.26 | Single epoch slice |
| **SEI (Full)** | **74.98** | **68.35** | **81.96** | Signed + Integral |

### Key Findings
*   **Sign Bit Contribution > Temporal Integral**: Removing the sign bit (EI) drops F1 by 10+ points; removing temporal accumulation (SE@T/T2) drops F1 by 8-15 points. Both are necessary, and the **sign bit is more critical**, validating the insight that entropy direction is more discriminative than magnitude.
*   **Architecture Agnostic**: Table 5 shows SEI also yields gains on ResNet-50 and ViT-B/16 classification networks, though the gain is maximized with CLIP due to vision-language alignment improving the pseudo-class calibration.
*   **Confusion Noise Advantage**: On harder confusion noise, SEI's lead over baselines is even greater. This is because confusion noise creates mislabeled samples that are closer in feature space to clean samples, tricking methods like SIMIFEAT/DEFT that rely on "feature neighborhood," whereas SEI focuses on "directional consistency over the whole trajectory."
*   **Real Clinical Applicability**: On CheXpert (majority vote of 5 radiologists as clean labels, report-extracted labels as noisy), SEI achieves an F1 of 83.59, 3.25 higher than AUM, proving effectiveness in real-world scenarios.

## Highlights & Insights
*   **"Adding direction to entropy" is a minimalist but pinpoint operation**: The core innovation is the single sign bit $(-1)^{\mathbb{1}[y=\arg\max p]}$. Without new modules or losses, it directly addresses the fundamental blind spot where entropy failed to distinguish mislabeled vs. hard clean samples. Raising the noisy label problem from "distribution-level" to "distribution + direction" is a design strategy worth transferring to other uncertainty tasks (OOD detection, active learning).
*   **Auxiliary class thresholding is more elegant than AUM's data implantation**: Using CLIP prompts to inject a "semantically irrelevant" pseudo-class avoids contaminating original classes while providing a reasonable anchor. This trick can be applied to any CLIP-style zero-shot/few-shot framework for uncertainty calibration.
*   **Dual-signal perspective of training dynamics**: Looking at both magnitude (entropy) and alignment (label direction) provides a new perspective on training history. While loss/entropy are single-dimension signals, the prediction-label alignment sequence is a complementary, directional binary sequence. Together, they effectively distinguish easy, hard, and mislabeled samples.

## Limitations & Future Work
*   The authors admit the auxiliary class threshold is **heuristic**—the specific wording of the pseudo-class prompt affects results, and no systematic prompt sensitivity analysis was provided.
*   It requires **full training for 150 epochs** to obtain the SEI trajectory, which is much costlier than single forward pass methods (SIMIFEAT/DEFT). There is no discussion on whether "early stopping + partial integration" still maintains performance.
*   Evaluations are focused on medical (+ CIFAR-100N) datasets. There is **no comparison on large-scale natural image noise benchmarks** (e.g., Clothing1M, WebVision), making it unclear if SEI maintains leadership over general methods like LEMoN/DEFT in the natural domain.
*   Detected samples are only **discarded**; the study does not explore re-labeling or sample re-weighting to fully utilize mislabeled information.

## Related Work & Insights
*   **vs. AUM (Pleiss et al., 2020)**: AUM uses "logit margin" accumulated over training and relies on **implanted known mislabeled samples** as anchors. SEI uses "signed entropy" and **pseudo-classes**. SEI's advantage lies in merging magnitude and direction into one scalar.
*   **vs. O2U-Net / CORES (Loss-based)**: These rely on loss magnitude, but hard clean samples also exhibit high loss. SEI's sign bit naturally separates them.
*   **vs. SIMIFEAT / DEFT / LEMoN (Pre-trained feature based)**: These methods perform detection without training the target model. While effective in the natural domain, they fail in medical domains where CLIP representations are weak. SEI treats CLIP as a **fine-tuneable backbone** and leverages the dynamics of the entire trajectory, thus outperforming them.

## Rating
*   Novelty: ⭐⭐⭐⭐ Not a revolutionary architecture, but "adding a sign bit to entropy" is a precise and clever fix for a fundamental blind spot.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Three medical datasets, two noise types, five noise rates, real-world clinical data, CIFAR-100N, and extensive ablations. Missing large-scale natural image noisy label benchmarks.
*   Writing Quality: ⭐⭐⭐⭐ Logical progression of motivations (Entropy $\rightarrow$ Failure $\rightarrow$ Direction $\rightarrow$ Time $\rightarrow$ Threshold). Figures 1-4 clearly illustrate SEI's separation capability.
*   Value: ⭐⭐⭐⭐ Plug-and-play, zero extra components, SOTA, open-source code. Immediate engineering value for medical imaging and a new "signed dynamics" perspective for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](../../CVPR2026/medical_imaging/robust_fair_disease_diagnosis_in_ct_images.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[CVPR 2026\] Robust Multi-Source Covid-19 Detection in CT Images](../../CVPR2026/medical_imaging/robust_multi-source_covid-19_detection_in_ct_images.md)
- [\[NeurIPS 2025\] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](../../NeurIPS2025/medical_imaging/revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)

</div>

<!-- RELATED:END -->
