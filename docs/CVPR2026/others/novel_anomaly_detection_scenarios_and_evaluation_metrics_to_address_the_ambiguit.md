---
title: >-
  [Paper Note] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples
description: >-
  [CVPR 2026][anomaly detection] To address the practical challenge that the definition of "normal" shifts with specification changes in industrial anomaly detection…
tags:
  - "CVPR 2026"
  - "anomaly detection"
  - "specification change"
  - "ambiguous normal definition"
  - "pseudo-anomaly"
  - "industrial defect detection"
date: 2026-05-08
content_hash: a86dcfcf70ed7585
---

# Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples

**Conference**: CVPR 2026
**arXiv**: [2604.07097](https://arxiv.org/abs/2604.07097)  
**Code**: [https://github.com/ReijiSoftmaxSaito/Scenario](https://github.com/ReijiSoftmaxSaito/Scenario)  
**Area**: Other
**Keywords**: anomaly detection, specification change, ambiguous normal definition, pseudo-anomaly, industrial defect detection

## TL;DR

To address the practical challenge that the definition of "normal" shifts with specification changes in industrial anomaly detection, this paper proposes two novel evaluation scenarios (A2N/N2A), a new metric (S-AUROC), and a training augmentation method called RePaste. RePaste increases the training frequency of high-anomaly-score regions by repasting them onto subsequent training images, enabling models to flexibly adapt to changes in the definition of normal samples.

## Background & Motivation

1. **Background**: Conventional anomaly detection methods assume that training data consist solely of defect-free normal samples, with models distinguishing normal from anomalous samples at inference. Recent years have seen the emergence of memory-based, knowledge distillation, normalizing flow, reconstruction, and pseudo-anomaly methods, with GLASS achieving state-of-the-art performance.

2. **Limitations of Prior Work**: In real industrial environments, the definition of "normal" is often ambiguous. For instance, minor scratches or dust particles may be acceptable under current specifications but could be reclassified as anomalous after equipment upgrades, or vice versa. Such specification changes occur frequently in industrial settings yet are entirely overlooked by existing methods.

3. **Key Challenge**: Although concept drift, domain adaptation, and continual learning address data distribution shifts, they target distributional changes rather than the explicit redefinition of normal and anomaly semantics. Existing metrics (AUROC, F1, etc.) also assume fixed label definitions and cannot quantify a model's ability to adapt to definitional changes.

4. **Goal**: (1) How should model performance be defined and evaluated under changes in the normal/anomaly definition? (2) How can models be made to flexibly adapt to such semantic redefinitions?

5. **Key Insight**: The authors observe that regions with persistently high anomaly scores in training images typically correspond to subtle defects (e.g., dust, minor scratches)—precisely the regions most susceptible to specification changes. Increasing the training frequency of these regions can suppress their anomaly scores.

6. **Core Idea**: High-anomaly-score regions from the current training image are repasted onto the next training image, encouraging the model to treat these "boundary-ambiguous" regions as normal.

## Method

### Overall Architecture

The input consists of industrial product images and the output is a pixel-level anomaly score map. The approach comprises three components: (1) two novel scenarios, A2N and N2A, defining evaluation protocols for label switching between normal and anomalous; (2) the S-AUROC metric, specifically designed to evaluate samples affected by specification changes; and (3) the RePaste training augmentation strategy. The overall framework is built on top of the GLASS baseline.

### Key Designs

1. **Anomaly-to-Normal Scenario (A2N)**:

    - **Function**: Evaluates a model's adaptability when samples previously labeled as anomalous are redefined as normal.
    - **Mechanism**: A2N comprises two sub-scenarios—$A2N_{A2N}$ adds half of a specific anomaly category (e.g., "Broken") to training as normal samples, with the other half used as normal samples in the test set; $A2N_S$ serves as the standard-scenario control. Only small anomalies with an average mask area < 1% are selected as specification-change targets, since large defects are unlikely to be redefined as normal.
    - **Design Motivation**: Simulates a "relaxed specification" scenario in industrial production, such as minor scratches no longer being classified as defects after equipment upgrades.

2. **Normal-to-Anomaly Scenario (N2A)**:

    - **Function**: Evaluates a model's adaptability when samples previously considered normal are redefined as anomalous.
    - **Mechanism**: Pseudo-anomalous images are generated using AnomalyAny and MemSeg. In $N2A_{N2A}$, pseudo-anomalies are added only to the test set as anomalous samples; in $N2A_S$, half of the pseudo-anomalies are incorporated into training data as normal samples. Model adaptability is assessed by comparing performance across the two sub-scenarios.
    - **Design Motivation**: Simulates a "stricter specification" scenario, such as previously acceptable samples becoming nonconforming after quality standards are raised.

3. **RePaste Training Augmentation**:

    - **Function**: Increases the training frequency of high-anomaly-score regions by repasting them, encouraging the model to incorporate these regions into the normal feature distribution.
    - **Mechanism**: During training, the input image $x_\alpha$ is fed into the model to obtain an anomaly map $A_\alpha$. A binary mask $M$ is generated by thresholding at $\tau$, and the high-score region is pasted onto the next training image $x_{\alpha+1}$. A Mixup-style blending $x'_{\alpha+1} = M \odot \frac{x_\alpha + x_{\alpha+1}}{2} + (1-M) \odot x_{\alpha+1}$ is applied to eliminate boundary discontinuities. RePaste is not required at inference, introducing no additional computational overhead.
    - **Design Motivation**: Directly addresses false positives arising after specification changes—by increasing the frequency of repasted regions, the model gradually incorporates them into the normal distribution.

### Loss & Training

RePaste is purely a training-time data augmentation strategy that does not modify the model architecture or loss function. The threshold $\tau$ is set to 0.9, restricting repasting to regions with very high anomaly scores. All other training settings follow GLASS exactly.

## Key Experimental Results

### Main Results

Evaluated on the MVTec AD dataset using S-AUROC to measure specification-change adaptability:

| Method | A2N S-AUROC | N2A S-AUROC |
|--------|------------|------------|
| PatchCore | 50.75 | 50.23 |
| SimpleNet | 84.25 | 75.70 |
| Dinomaly | 84.70 | 81.88 |
| GLASS | 86.29 | 83.25 |
| **RePaste** | **86.88** | **83.75** |

### Ablation Study

| Configuration | A2N S-AUROC | N2A S-AUROC |
|---------------|------------|------------|
| GLASS (baseline) | 86.29 | 83.25 |
| RePaste w/o Mixup | 87.48 | 78.26 |
| RePaste w/ Mixup | 86.88 | 83.75 |

### Key Findings

- PatchCore performs near random chance (~50% S-AUROC) under specification-change scenarios, as coreset sampling discards rare features.
- GLASS achieves the best performance among all comparison methods, owing to the flexibility of its gradient-ascent-based pseudo-anomaly generation.
- RePaste improves S-AUROC by 0.59% on A2N and 0.50% on N2A over GLASS.
- Removing Mixup from RePaste causes a sharp 5.49% drop on N2A, demonstrating that boundary smoothing is critical for the N2A scenario.
- RePaste also maintains performance on standard I-AUROC, P-AUROC, and PRO metrics comparable to or better than GLASS (Mean PRO: 97.02% vs. 96.83%).

## Highlights & Insights

- **Novel Problem Formulation**: This work is the first to systematically address the ambiguity and dynamic nature of "normal" definitions in anomaly detection. The proposed A2N/N2A scenarios and S-AUROC metric have strong practical relevance.
- **Extreme Simplicity**: RePaste is purely a training-time data augmentation technique that requires no architectural modifications and introduces no inference overhead, yet effectively improves adaptability to specification changes.
- **Necessity of Mixup Boundary Smoothing**: The ablation study clearly demonstrates the severe impact of paste-boundary discontinuities on the N2A scenario, a finding that generalizes to any data augmentation method involving region pasting.

## Limitations & Future Work

- Evaluation is limited to MVTec AD; generalization to other anomaly detection benchmarks (e.g., VisA, BTAD) remains unverified.
- The threshold $\tau$ is fixed at 0.9; adaptive threshold strategies are not explored.
- The performance gains from RePaste are modest (<1% S-AUROC), suggesting that more fundamental methodological innovations may be needed to address this problem.
- In the A2N scenario, only the redefinition of small anomalies is considered; specification changes involving large defects are not addressed.
- N2A relies on synthetically generated pseudo-anomalies as proxies for real specification-change samples, which may introduce distributional discrepancies.

## Related Work & Insights

- **vs. GLASS**: RePaste builds upon GLASS, preserving the flexibility of its pseudo-anomaly generation while augmenting it with bidirectional adaptation capability for both "normal→anomaly" and "anomaly→normal" transitions via region repasting.
- **vs. Concept Drift / Domain Adaptation**: The paper argues that specification changes are fundamentally distinct from distributional shifts—the former involves the reconstruction of decision boundaries, whereas the latter concerns shifts in feature distributions.
- The proposed scenario formulation can inspire the design of continual learning approaches for anomaly detection.

## Rating

- Novelty: ⭐⭐⭐⭐ The scenario definitions and evaluation metric are highly original, though the method itself (region repasting) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is conducted on a single dataset (MVTec AD); a reasonable number of baselines are compared, but the ablation analysis lacks depth.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and scenario descriptions are thorough, though notation is occasionally redundant.
- Value: ⭐⭐⭐⭐ Identifies an important and overlooked practical problem with direct implications for industrial anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](../../AAAI2026/others/rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/others/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](../../ICCV2025/others/generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)

</div>

<!-- RELATED:END -->
