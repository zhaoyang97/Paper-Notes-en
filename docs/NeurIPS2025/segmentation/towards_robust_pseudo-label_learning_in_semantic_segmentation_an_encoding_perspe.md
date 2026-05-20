---
title: >-
  [Paper Note] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective
description: >-
  [NeurIPS 2025][Segmentation][Pseudo-label learning] This paper proposes ECOCSeg, which replaces one-hot encoding with Error-Correcting Output Codes (ECOC) to represent semantic categories. It decomposes an N-class classi…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Pseudo-label learning"
  - "semantic segmentation"
  - "error-correcting output codes"
  - "unsupervised domain adaptation"
  - "semi-supervised learning"
date: 2026-05-08
content_hash: 7f27f9c42075ee36
---

# Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective

**Conference**: NeurIPS 2025
**arXiv**: [2512.06870](https://arxiv.org/abs/2512.06870)  
**Code**: [https://github.com/Woof6/ECOCSeg](https://github.com/Woof6/ECOCSeg)  
**Area**: Segmentation
**Keywords**: Pseudo-label learning, semantic segmentation, error-correcting output codes, unsupervised domain adaptation, semi-supervised learning

## TL;DR

This paper proposes ECOCSeg, which replaces one-hot encoding with Error-Correcting Output Codes (ECOC) to represent semantic categories. It decomposes an N-class classification problem into K binary sub-tasks, and couples bit-level pseudo-label denoising with customized optimization losses to substantially improve the robustness of pseudo-label learning in UDA and SSL semantic segmentation.

## Background & Motivation

**Background**: Semantic segmentation in label-scarce settings — including unsupervised domain adaptation (UDA) and semi-supervised learning (SSL) — relies heavily on pseudo-label learning. Dominant approaches fall into two paradigms: self-training (generating pseudo-labels via an EMA teacher model) and consistency regularization (enforcing prediction consistency across differently perturbed views of the same sample), both of which are essentially pseudo-label learning frameworks.

**Limitations of Prior Work**: Pseudo-labels inevitably contain errors. Existing methods assign class labels via one-hot encoding and argmax hard assignment, which completely misleads training whenever the prediction is wrong. Threshold-based filtering discards low-confidence samples, biasing the model toward easy examples; weighting strategies require careful tuning and generalize poorly. Critically, existing methods focus almost exclusively on **selection strategies** for pseudo-labels, with virtually no attention paid to the influence of the **encoding representation** itself.

**Key Challenge**: Visually similar categories (e.g., sheep/cow/horse) share visual attributes and are frequently confused with one another. Under one-hot encoding, any such confusion results in a completely erroneous label (zero mutual information). Yet these categories do share attributes — if the encoding scheme could exploit such shared structure, even an incorrect classification could still provide partially correct supervisory signal.

**Goal**: To revisit the pseudo-label noise problem from the perspective of encoding representation, and to design a category encoding that tolerates partial bit errors so that incorrect pseudo-labels can still provide meaningful supervision.

**Key Insight**: The authors draw inspiration from Error-Correcting Output Codes (ECOC) in communications, representing each class as a multi-bit binary codeword rather than a one-hot vector. Similar categories share certain bits in their codewords; even when classification is incorrect, shared bits remain correct, realizing fault-tolerant supervision where errors are partially recovered.

**Core Idea**: Replace one-hot encoding with ECOC to represent semantic categories, enabling pseudo-labels to provide partially correct supervisory signal at the bit level even when the class assignment is wrong.

## Method

### Overall Architecture

ECOCSeg can be integrated as a plug-in into existing pseudo-label learning frameworks. Input images are passed through an encoder to extract pixel-level features, which are then fed not into an N-class classifier but into K binary classifiers — each predicting one bit of the codeword. The resulting K-dimensional probability vector is used to identify the nearest-neighbor codeword in the codebook via soft Hamming distance, yielding the predicted class. A bit-level denoising mechanism is introduced during pseudo-label generation, combining the complementary strengths of bit-wise and code-wise pseudo-label forms. Three customized loss functions are jointly optimized during training.

### Key Designs

1. **ECOC Dense Classification Paradigm**:

    - **Function**: Transforms N-class semantic segmentation into K binary classification problems, endowing the framework with inherent error-correction capability.
    - **Mechanism**: A binary codebook matrix of size $N \times K$ is constructed, where each class corresponds to a codeword of length K. The classifier is replaced by K independent sigmoid binary classifiers, each predicting one bit. During inference, the class is determined by computing the soft Hamming distance between the predicted vector and each codeword in the codebook: $d_{SH}(\mathbf{c}_n, \mathbf{p}^i) = \frac{1}{K}\sum_{k=1}^K \|p(k|\mathbf{z}_i) - \mathbf{c}_{nk}\|_1$, and selecting the nearest neighbor. Two codebook design strategies are provided: max-min distance encoding (maximizing the minimum inter-class Hamming distance to guarantee error-correction capacity) and text-based encoding (generating codes from class-name semantic relationships to ensure semantic consistency).
    - **Design Motivation**: Under one-hot encoding, the inter-class Hamming distance is always 2, yielding zero error-correction capacity. ECOC encoding enables inter-class distances far exceeding 2, theoretically tolerating up to $\lfloor(d-1)/2\rfloor$ bit errors. The paper further proves via NTK theory that ECOC is performance-equivalent to one-hot encoding under full supervision (Theorem 4.1) and achieves a tighter misclassification upper bound under pseudo-label noise (Theorem 4.2).

2. **Reliable Bit Mining (RBM)**:

    - **Function**: Mines reliable supervisory signals from noisy pseudo-labels at the bit level.
    - **Mechanism**: ECOCSeg naturally yields two pseudo-label forms — bit-wise (directly quantizing the sigmoid output of each bit) and code-wise (querying the nearest-neighbor codeword). Each has distinct trade-offs: bit-wise labels are softer but carry independent per-bit noise; code-wise labels are perfectly accurate when classification is correct but introduce holistic noise upon misclassification. The algorithm queries the C nearest-neighbor codewords and identifies bit positions shared across all candidates in the set — these bits are correct regardless of which candidate is the true class — marking them as "reliable bits." The final mixed pseudo-label uses code-wise values at reliable positions and bit-wise values elsewhere. The size C is determined adaptively via confidence threshold T.
    - **Design Motivation**: No single pseudo-label form can simultaneously achieve high precision and high coverage. By mining shared bits across candidate classes, the approach maximizes usable supervisory signal while guaranteeing correctness.

3. **Customized Optimization Objectives**:

    - **Function**: Introduces structured representation constraints on top of bit-level BCE to accelerate convergence and enhance discriminability.
    - **Mechanism**: Three losses are combined — (1) Binary cross-entropy $\mathcal{L}_{bce}$ independently optimizes each bit classifier; (2) Pixel-codeword distance $\mathcal{L}_{pcd} = 1 - \cos(\hat{\mathbf{p}}^i, \hat{\mathbf{c}}^i)$ encourages intra-class compactness by pulling logits toward the corresponding codeword direction; (3) Pixel-codeword contrast $\mathcal{L}_{pcc}$ applies contrastive learning to push predictions away from non-target codewords, computed only on discriminative bit positions ($P_d$) while ignoring shared bits.
    - **Design Motivation**: BCE alone ignores structural relationships among bits and lacks intra-class compactness and inter-class separation constraints. PCD provides intra-class constraints, PCC provides inter-class constraints, and the three losses are mutually complementary.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{pcd} + \lambda_2 \mathcal{L}_{pcc}$, applied to both labeled supervised loss and unlabeled pseudo-label loss. The overall training pipeline preserves the standard self-training/consistency regularization paradigm, replacing only the encoding form, pseudo-label generation, and loss functions.

## Key Experimental Results

### Main Results

| Baseline | Architecture | Original mIoU | +ECOCSeg mIoU | Gain |
|----------|------|-----------|---------------|------|
| DACS (GTA→CS) | CNN | 52.1 | 54.5 | +2.4 |
| DAFormer (GTA→CS) | Trans. | 68.3 | 70.5 | +2.2 |
| MIC (GTA→CS) | Trans. | 75.9 | 76.9 | +1.0 |
| DACS (SYN→CS) | CNN | 48.3 | 52.1 | +3.8 |
| DAFormer (SYN→CS) | Trans. | 60.9 | 63.3 | +2.4 |
| MIC (SYN→CS) | Trans. | 68.7 | 69.8 | +1.1 |

### Ablation Study

| Component | GTA→CS mIoU | Note |
|------|-------------|------|
| Baseline (DAFormer) | 68.3 | one-hot + CE |
| + ECOC encoding | 69.0 | encoding form only |
| + bit-wise PL | 69.5 | bit-level pseudo-labels |
| + code-wise PL | 69.3 | codeword-level pseudo-labels |
| + mixed PL (RBM) | 70.0 | reliable bit mining |
| + PCD + PCC losses | 70.5 | full ECOCSeg |

### Key Findings

- ECOCSeg consistently improves across different baselines (DACS/DAFormer/MIC) and architectures (CNN/Transformer), indicating orthogonality to existing improvements.
- Gains are larger on SYNTHIA→Cityscapes (+3.8), as greater domain gaps produce noisier pseudo-labels, amplifying the error-correction advantage of ECOC.
- Mixed pseudo-labels (RBM) outperform either bit-wise or code-wise labels alone, validating the complementarity assumption.
- ECOC encoding incurs no performance degradation under full supervision, confirming the theoretical prediction of Theorem 4.1.
- The paper further demonstrates that ECOC improves model calibration, indirectly enhancing pseudo-label quality in subsequent training iterations.

## Highlights & Insights

- **Encoding perspective on pseudo-label noise**: This is an entirely new and orthogonal direction, fully compatible and stackable with existing filtering and weighting strategies. The conceptual shift — optimizing the representation of labels rather than their selection — is particularly elegant.
- **Exploiting inter-class shared attributes**: Sheep and cows both have horns and hooves; even when misclassified, bits corresponding to shared attributes remain correct. This insight is simple yet profound, naturally transferring error-correcting ideas from communications to semantic segmentation.
- **Theoretical guarantees**: By leveraging the NTK framework, the paper proves both the equivalence of ECOC under full supervision and its superiority under noisy pseudo-labels, grounding the approach in rigorous theory rather than heuristics alone.

## Limitations & Future Work

- The optimal codebook configuration (choice of K and encoding strategy) may vary across datasets, and no adaptive selection mechanism is provided.
- K binary classifiers introduce more parameters than a single N-class classifier, which may pose efficiency concerns when the number of classes is very large.
- Validation is currently limited to semantic segmentation; the ECOC encoding idea is in principle transferable to any pseudo-label learning scenario (e.g., object detection, instance segmentation), which merits future exploration.
- The threshold T in reliable bit mining is a fixed hyperparameter; an adaptive thresholding strategy may yield further improvements.

## Related Work & Insights

- **vs. threshold-based filtering** (e.g., FixMatch): Threshold filtering discards uncertain samples, resulting in missing hard examples. ECOCSeg retains partial supervision for all samples at the bit level, achieving broader coverage.
- **vs. weighting strategies** (e.g., FlexMatch): Weighting strategies require carefully designed weight functions, whereas ECOCSeg achieves noise tolerance through the encoding form itself, requiring no additional weight design.
- **vs. negative learning**: Negative learning avoids noise by telling the model "what something is not"; ECOCSeg exploits noise by telling the model "what something partially is." The two approaches may be complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Examining pseudo-label noise from an encoding perspective is a genuinely novel direction; introducing ECOC from communications into segmentation is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple UDA/SSL baselines and benchmarks with detailed ablations, though validation beyond segmentation tasks is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The problem formalization is clear, and the three-component analytical framework (encoding / pseudo-label strategy / optimization objective) is elegantly structured.
- **Value**: ⭐⭐⭐⭐⭐ Orthogonal to existing methods, plug-and-play, and theoretically grounded — high in both practical utility and conceptual inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](../../CVPR2026/segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)

</div>

<!-- RELATED:END -->
