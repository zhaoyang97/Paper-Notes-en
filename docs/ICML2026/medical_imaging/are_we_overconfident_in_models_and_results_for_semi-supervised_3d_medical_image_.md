---
title: >-
  [Paper Note] Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?
description: >-
  [ICML 2026][Medical Imaging][Semi-supervised learning] This paper points out that semi-supervised 3D medical image segmentation suffers from two types of problems: overconfidence in model pseudo-labels and over-optimism…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Semi-supervised learning"
  - "3D medical image segmentation"
  - "pseudo-label calibration"
  - "uncertainty estimation"
  - "multi-run evaluation"
date: 2026-05-08
content_hash: f7600b768725724c
---

# Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?

**Conference**: ICML 2026  
**arXiv**: [2605.25561](https://arxiv.org/abs/2605.25561)  
**Code**: https://github.com/DirkLiii/TCSeg  
**Area**: Medical Image / Semi-supervised 3D Segmentation  
**Keywords**: Semi-supervised learning, 3D medical image segmentation, pseudo-label calibration, uncertainty estimation, multi-run evaluation  

## TL;DR
This paper points out that semi-supervised 3D medical image segmentation suffers from two types of problems: overconfidence in model pseudo-labels and over-optimism in evaluation protocols. It proposes TCSeg, which utilizes confidence-uncertainty dual-axis reliability and tri-space calibration (probability, feature, and image spaces) to suppress confirmation bias. It also advocates for evaluation protocols using multiple random seeds and reporting both best and last checkpoints to provide a more honest assessment of performance.

## Background & Motivation
**Background**: 3D medical image segmentation requires precise voxel-level annotations, which involve high costs and strong expert dependence. Consequently, semi-supervised learning (SSL) has become a common solution. Prevailing methods typically utilize a small amount of labeled volume data and a large amount of unlabeled volume data, expanding supervision signals through teacher-student architectures, consistency regularization, or pseudo-label training.

**Limitations of Prior Work**: Many methods assume by default that high softmax scores represent reliable pseudo-labels, but deep networks can produce high confidence even for incorrect predictions. For voxels with blurred boundaries, low contrast, or large morphological variations, these errors—once selected as pseudo-supervision—are continuously reinforced during training, resulting in confirmation bias.

**Key Challenge**: Confidence and uncertainty are not the same thing. Confidence represents the model's current class preference, while uncertainty represents whether the evidence is stable; a voxel can be "high confidence but high uncertainty," meaning the model strongly predicts a certain class, but evidence from different views, branches, or features is inconsistent. Compressing both into a single threshold allows the most dangerous pseudo-labels to infiltrate training.

**Goal**: The authors aim to simultaneously address pseudo-label overconfidence at the algorithmic level and result over-optimism at the community evaluation level. The former is mitigated through reliability modeling and tri-space calibration, while the latter is addressed by introducing multi-run protocols and reporting both best and last checkpoints to reduce optimistic bias caused by single lucky results or choosing checkpoints based on test set performance.

**Key Insight**: Instead of treating reliability as a scalar, the authors construct $R(v)=\langle C(v),U(v)\rangle$ for each voxel. Only voxels with high confidence and low uncertainty are used for strong pseudo-supervision, while other regions are handled separately through feature prototypes, consistency constraints, and structural perturbations.

**Core Idea**: Utilize dual-axis signals where "confidence represents preference and uncertainty represents evidence stability" to filter pseudo-labels, and collaboratively correct the confirmation bias of semi-supervised segmentation across probability, feature, and image spaces.

## Method
The overall design of TCSeg revolves around a shared reliability engine. It first generates multi-view predictions using dual student decoder branches and their teacher branch, then estimates uncertainty from both probability outputs and feature prototypes. Subsequently, pixels are partitioned into different reliability zones: voxels with high confidence and low uncertainty are used for positive/negative pseudo-supervision, while high-confidence high-uncertainty and low-confidence regions are treated as error-prone areas or regions requiring additional perturbation learning.

### Overall Architecture
The input includes a labeled set $\mathcal{D}_l$ and an unlabeled set $\mathcal{D}_u$. The model adopts a VNet-style five-stage shared encoder followed by two parallel decoders; the two decoders use different up-sampling operators to form complementary predictions and features. The teacher branch is an EMA-smoothed version of the student branch, used to provide a more stable reference.

During training, TCSeg first calculates the class preference of the average multi-branch prediction as confidence $C(v)$, and then calculates probability space branch discrepancy $U_{pro}(v)$ and feature space prototype discrepancy $U_{fea}(v)$ as uncertainty. Afterwards, the probability space filters high-reliability pseudo-labels, the feature space pulls voxels toward class prototypes while filtering semantic outliers, and the image space applies targeted CutMix-style perturbations to "cognitive blind spots."

The final optimization objective consists of supervised segmentation loss, reliable pseudo-label loss, feature calibration loss, and mixed perturbation loss. The paper also treats the experimental protocol as part of its contribution: each setting is run with five random seeds, reporting the median and maximum for both best and last checkpoints to distinguish "peak lucky runs" from "typical deployment states."

### Key Designs
1. **Confidence-Uncertainty Dual-Axis Reliability Estimation**:

	- **Function**: Distinguishes "model class preference" from "evidence stability," preventing high softmax scores from being directly equated with pseudo-label correctness.
	- **Mechanism**: Confidence $C(v)$ is the maximum class probability of the average student and teacher multi-view prediction; uncertainty consists of two parts: first, the $L_1$ discrepancy between the probability distributions of the two decoders, and second, the discrepancy between voxel features and class prototype similarity predictions. This allows the identification of dangerous voxels that are high-confidence but inconsistent across views.
	- **Design Motivation**: Traditional entropy, variance, or thresholding methods often compress reliability into a single-axis signal, which easily allows "confident but wrong" voxels into pseudo-supervision. Dual-axis representation enables the model to handle confidence and uncertainty as two complementary factors.

2. **Probability, Feature, and Image Tri-Space Calibration**:

	- **Function**: Corrects pseudo-label confirmation bias across output distribution, semantic features, and input structure levels.
	- **Mechanism**: The probability space uses low-uncertainty masks and upper/lower confidence thresholds to filter positive/negative pseudo-supervision; the feature space constrains probability outputs and prototype similarity to be consistent through class prototype constraints, suppressing semantic outliers in high-confidence predictions; the image space constructs perturbation masks based on low-confidence regions and high-confidence high-uncertainty regions, performing reliability-driven mixing on the largest connected components.
	- **Design Motivation**: Medical segmentation errors can stem from unstable output probabilities, semantic feature deviation, or boundary and shape blind spots. Tri-space synergy provides a more comprehensive constraint on error sources than single pseudo-label filtering.

3. **Dual-Branch Mutual Learning and Strict Evaluation Protocol**:

	- **Function**: Reduces single-branch self-reinforcement during training and reduces inflated conclusions caused by checkpoint selection and random seeds during evaluation.
	- **Mechanism**: Two student decoders alternately act as supervisor and learner, with unlabeled pseudo-labels guided by the peer branch; in experiments, each setting is run five times, reporting the median and maximum of best/last checkpoints, where "last" is closer to the state of a model deployed immediately after training.
	- **Design Motivation**: Semi-supervised medical segmentation not only forms overconfidence within the model but also creates overconfidence in results through test-set checkpoint selection, single runs, and peak-only reporting in paper comparisons. The authors incorporate evaluation reliability into the methodology.

### Loss & Training
The total loss is $\mathcal{L}_{total}=\mathcal{L}_{sup}+\mathcal{L}_{pse}+\mathcal{L}_{cal}+\mathcal{L}_{mix}$. $\mathcal{L}_{sup}$ is the Dice + CE segmentation loss on labeled data, $\mathcal{L}_{pse}$ utilizes positive/negative pseudo-label supervision only on high-confidence low-uncertainty voxels, $\mathcal{L}_{cal}$ constrains the consistency between the dual-branch probability outputs, prototype similarity outputs, and between each other, and $\mathcal{L}_{mix}$ supervises mixed samples constructed from reliability masks.

Experiments use three public 3D medical segmentation datasets: LA, Pancreas-CT, and BraTS2019. Training utilizes SGD for 20k iterations with a learning rate of 0.01 and a batch size of 4, comprising 2 labeled volumes and 2 unlabeled volumes. The backbone follows a standard VNet-style architecture to maintain comparability with most semi-supervised 3D segmentation methods.

## Key Experimental Results

### Main Results
The main table compares protocols according to last and best checkpoints. The paper emphasizes that "last" reflects the deployment state after convergence, while "best" is more like an upper bound or a peak result with oracle selection.

| Dataset / Label Ratio | Metric | TCSeg median last | Prev. SOTA | Gain / Notes |
|--------|------|------|----------|------|
| LA 10% | DSC | 90.28 | ARCO-SG 89.90 / TraCoCo 89.29 | Maintains highest median under last protocol |
| LA 20% | DSC | 90.83 | SFR 91.00 / TraCoCo 90.94 | Close to the best last baseline, maximum reaches 91.36 |
| Pancreas-CT 10% | DSC | 81.08 | TraCoCo 79.22 | Improvement of ~1.86 points, a key result highlighted in the paper |
| Pancreas-CT 20% | DSC | 83.44 | TraCoCo 81.80 / DBiSL 81.09 | Significantly more stable under last protocol |
| BraTS2019 20% | DSC | 86.47 | TraCoCo 86.69 | BraTS has a validation set, showing higher best/last consistency |

### Ablation Study
Both dual-axis reliability and tri-space calibration provide visible gains, particularly on tasks with low contrast and blurred boundaries like Pancreas-CT.

| Config | LA 10% | Pancreas 10% | BraTS 20% | Mean DSC | Notes |
|------|---------|---------|---------|---------|------|
| w/o $U(v)$ | 90.12 | 80.42 | 85.88 | 85.68 | Confidence filtering remains, but cannot identify high-confidence unstable voxels |
| w/o $C(v)$ | 88.88 | 78.82 | 86.22 | 85.20 | Relying only on uncertainty weakens class preference filtering |
| Dual-axis | 90.28 | 81.08 | 86.47 | 86.23 | Confidence and uncertainty are complementary, achieving the best average |

| Tri-space Config | LA 10% | Pancreas 10% | BraTS 20% | Mean DSC | Notes |
|------|---------|---------|---------|---------|------|
| Only sup | 76.18 | 55.72 | 80.01 | 72.69 | No semi-supervised calibration, performance significantly lags |
| w/o prob. | 89.84 | 78.65 | 85.80 | 85.13 | Pseudo-label stability decreases without probability space filtering |
| w/o img. | 87.68 | 75.48 | 84.81 | 84.00 | Performance drops consistently across datasets without structural perturbations |
| w/o feat. | 89.61 | 59.28 | 86.06 | 80.09 | Largest degradation on Pancreas-CT, showing prototype constraints are critical |
| **Ours** | 90.28 | 81.08 | 86.47 | 86.23 | Tri-space complementarity yields the best results |

### Key Findings
- The most important empirical insight is the difference between best and last. The authors point out that single best checkpoints compress the gap between methods and hide instability; TCSeg maintains strong results even under the last protocol.
- In pseudo-label PPV/NPV/Recall analysis, tri-space calibration shifts the points for all six dataset-label ratio configurations toward the upper right; notably, the recall for Pancreas-CT increases from ~0.56-0.64 to over 0.86, while NPV remains high.
- Parameter sensitivity is moderate. Changing confidence bounds from $[0.1, 0.85]$ to $[0.05, 0.95]$ or $[0.2, 0.75]$, and consistency tolerance from 0.01 to 0.10, results in minimal changes to DSC curves.
- Computational overhead is controllable. TCSeg has 12.34M parameters and a training time of 0.421 s/iter, which is lower than CC-Net's 2.934 s/iter; auxiliary modules can be discarded at deployment, leaving only the backbone path.

## Highlights & Insights
- The paper discusses "model overconfidence" and "result overconfidence" within the same framework, which is highly valuable. Since medical image segmentation is targeted at real-world deployment, model calibration and evaluation protocols should not be viewed separately.
- Dual-axis reliability is more consistent with error patterns in clinical images than simple thresholds. Blurred boundary regions are often not just low-confidence, but high-risk areas where the model might be "very certain but the evidence is inconsistent."
- Tri-space calibration provides a reusable paradigm: the probability space manages output reliability, the feature space manages semantic attribution, and the image space manages structural blind spots. This decomposition can be transferred to other pseudo-label-driven tasks like semi-supervised lesion detection or remote sensing segmentation.
- The multi-run best/last reporting is worth adopting by the community. It allows readers to distinguish between the peak potential, typical performance, and stability of a method, preventing the mistake of treating the optimal checkpoint found via test set feedback as a real-world deployment gain.

## Limitations & Future Work
- The authors explicitly state that the reliability analysis is currently limited to semi-supervised 3D medical image segmentation benchmarks and does not directly prove general uncertainty calibration or out-of-distribution robustness.
- Stability on public datasets cannot be equated with clinical utility. New scanners, protocols, modalities, or cross-hospital data will introduce distribution shifts, still requiring multi-center, prospective validation.
- The current framework uses fixed thresholds, such as confidence intervals and consistency tolerance. Although the paper demonstrates robustness within a limited range, more adaptive threshold selection might further enhance cross-dataset stability.
- The paper notes that community evaluation still suffers from issues like inconsistent post-processing, checkpoint selection, and protocols. TCSeg's own plan for a re-implemented benchmark is significant; if a complete and unified evaluation table can be publicized in the future, it would further strengthen the argument.

## Related Work & Insights
- **vs Mean Teacher / teacher-student SSL**: Traditional teacher-student utilizes teacher predictions to generate pseudo-labels or consistency supervision. TCSeg goes further to ask if these pseudo-labels are truly reliable and uses dual-axis reliability to distinguish high-confidence errors.
- **vs MC dropout / ensemble uncertainty**: These methods estimate uncertainty but often compress it into a single scalar. TCSeg preserves both confidence and uncertainty dimensions and combines them with feature prototypes to judge semantic consistency.
- **vs CC-Net / TraCoCo and other 3D SSL methods**: These methods show strong numbers under best checkpoint protocols, but TCSeg emphasizes stable medians of last checkpoints and multiple random seeds, shifting the focus from peak performance to reproducibility.
- **vs Dynamic threshold pseudo-labeling**: Dynamic thresholds mainly adjust selection intensity. The key difference in TCSeg is first decoupling the semantics of reliability and then applying different calibration mechanisms to different risk regions.

## Rating
- Novelty: ⭐⭐⭐⭐ Strong problem awareness in splitting overconfidence into algorithmic and evaluation layers and handling it systematically with dual-axis reliability + tri-space calibration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Main results, dual-axis ablation, tri-space ablation, pseudo-label quality, parameter sensitivity, efficiency, and multi-run protocols are all quite complete.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, and the motivation and critique of evaluation are very persuasive; symbols and naming are slightly dense in a few places.
- Value: ⭐⭐⭐⭐⭐ Practical methodological value for the semi-supervised medical segmentation community and helps drive more reliable benchmark reporting practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](../../AAAI2026/medical_imaging/bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)
- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](../../AAAI2026/medical_imaging/propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2026/medical_imaging/semitooth_a_generalizable_semisupervised_framework.md)

</div>

<!-- RELATED:END -->
