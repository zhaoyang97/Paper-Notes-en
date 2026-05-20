---
title: >-
  [Paper Note] Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection
description: >-
  [ICLR 2026][Multimodal VLM][video anomaly detection] This paper proposes SteerVAD, a framework that identifies "latent anomaly expert" (LAE) attention heads within a completely frozen multimodal large language model (MLL…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "video anomaly detection"
  - "multimodal large language models"
  - "representation manifold manipulation"
  - "tuning-free"
  - "attention head analysis"
date: 2026-05-08
content_hash: 9ac9be37de95887f
---

# Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection

**Conference**: ICLR 2026
**arXiv**: [2602.24021](https://arxiv.org/abs/2602.24021)  
**Code**: To be released  
**Area**: Multimodal / Video Understanding
**Keywords**: video anomaly detection, multimodal large language models, representation manifold manipulation, tuning-free, attention head analysis

## TL;DR

This paper proposes SteerVAD, a framework that identifies "latent anomaly expert" (LAE) attention heads within a completely frozen multimodal large language model (MLLM) and dynamically steers their representation manifolds via a hierarchical meta-controller, achieving tuning-free video anomaly detection SOTA with only 1% of training data.

## Background & Motivation

Video anomaly detection (VAD) aims to identify events deviating from normal patterns, which is critical in intelligent surveillance, industrial inspection, and autonomous driving. Traditional VAD methods (supervised / weakly supervised / unsupervised) rely on large-scale training data, incurring high computational and annotation costs with limited generalization.

Recent work has explored tuning-free VAD using frozen MLLMs, but two fundamental limitations remain:

**Representational Bias**: MLLMs pre-trained on web-scale corpora optimize their feature spaces for frequent prototypical concepts, resulting in low sensitivity to rare and subtle anomalous patterns.

**Contextual Ambiguity**: Passively relying on isolated features produces confounded representations—visually similar but semantically distinct events (e.g., normal running vs. fleeing) cannot be effectively distinguished.

Starting from the manifold hypothesis, the authors reframe both issues as geometric problems: the representation manifolds of normal and anomalous events lie too close together, or are locally entangled, in high-dimensional feature space, and passive feature readout cannot resolve this structural deficiency.

## Method

### Overall Architecture

SteerVAD shifts the paradigm from "passive feature readout" to "active geometric intervention." The core pipeline consists of:

1. **Representation Separability Analysis (RSA)**: Gradient-free identification of the attention heads within the MLLM most suitable for VAD (latent anomaly experts / LAEs).
2. **Hierarchical Meta-Controller (HMC)**: Dynamically generates correction signals conditioned on global context.
3. **Anisotropic Manifold Scaling**: Applies targeted geometric transformations to the feature manifolds of LAEs.
4. **Anomaly Scoring and Smoothing**: Aggregates corrected features and outputs an anomaly probability curve.

### Key Designs

1. **Representation Separability Analysis (RSA)**: Identifies which attention heads best discriminate normal from anomalous events.

    - **Mechanism**: A variant of the Fisher discriminant ratio is used as a geometric separability measure, computing the ratio of between-class scatter to within-class scatter for normal/anomalous samples in each attention head.
    - **Mathematical definition**: $S_{RSA}(l,k) = \frac{\|\boldsymbol{\mu}_{anom}^{(l,k)} - \boldsymbol{\mu}_{norm}^{(l,k)}\|_2^2}{\sigma_{anom}^2(l,k) + \sigma_{norm}^2(l,k)}$
    - **Design Motivation**: No gradient computation is required; a single forward pass suffices to select the top-$K$ most discriminative heads among all 784 attention heads. The selection is remarkably insensitive to data volume—1% and 100% of the data identify identical LAEs.
    - **Experimental Validation**: Across 10 runs with different random seeds, RSA consistently selects the same four heads (L18H4, L23H24, L21H21, L22H7).

2. **Hierarchical Meta-Controller (HMC)**: Generates dynamic, context-aware manifold correction signals.

    - **Global Scrutiny Gate (GSG)**: Maps the hidden state $\mathbf{c}$ from the MLLM's first generated token through a lightweight MLP to a scalar $s_{global} \in [0,1]$, measuring overall anomaly likelihood. Values near 0 indicate normal scenes (controller remains silent); values near 1 trigger strong correction.
    - **Local Gating Module (LGM)**: Comprises $K$ parallel low-rank adapters, each mapping the global context $\mathbf{c}$ through a low-rank bottleneck to a head-specific steering vector $\mathbf{g}_i \in [-1,1]^{d_{head}}$, enabling per-dimension fine-grained control.
    - **Design Motivation**: Decoupling "whether correction is needed" (global) from "how to correct" (local) prevents the lightweight module from overfitting to local noise.

3. **Anisotropic Manifold Scaling**: Executes the actual geometric transformation.

    - **Core operation**: $\mathbf{h}_i' = \mathbf{h}_i \odot (1 + s_{global} \cdot \mathbf{g}_i)$
    - This is a residual modulation: when $s_{global} \approx 0$, it approximates an identity transformation; when $s_{global} \approx 1$, positive $\mathbf{g}_i$ amplifies corresponding dimensions while negative values suppress them.
    - **Theoretical significance**: When all scaling factors are nonzero, the operation is a diffeomorphism (topologically preserving manifold reshaping); when some factors are zero, it becomes a singular projection (context-aware feature selection) that can entirely eliminate dimensions associated with pre-training bias.

### Loss & Training

The training objective is concise and efficient:

- **Primary loss**: Binary cross-entropy $\mathcal{L}_{BCE} = -[y \log(p_t) + (1-y) \log(1-p_t)]$
- **Sparsity regularization**: L2 penalty on the global signal for normal samples $\mathcal{L}_{reg} = \frac{1}{|\mathcal{B}_{norm}|} \sum_{j \in \mathcal{B}_{norm}} (s_{global}^{(j)})^2$
- **Total objective**: $\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda_{reg} \mathcal{L}_{reg}$, where $\lambda_{reg} = 0.1$
- Sparsity regularization ensures the controller remains silent on normal inputs, reducing false positives.

Training requires approximately 27 seconds for 1,000 epochs on a single RTX A6000 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | SteerVAD | HiProbeVAD | VERA | Holmes-VAD (Fine-tuned) |
|---------|--------|----------|------------|------|------------------------|
| UCF-Crime | AUC (%) | **87.15** | 86.72 | 86.55 | 89.51 |
| XD-Violence | AP (%) | **83.02** | 82.15 | 70.54 | 90.67 |

- Achieves SOTA among tuning-free methods with only ~520K trainable parameters (~1 MB).
- Compared to Holmes-VAD, which requires full fine-tuning of 7B parameters, SteerVAD falls short by only 2.36% on UCF-Crime.

### Ablation Study

| Configuration | AUC (%) | Notes |
|---------------|---------|-------|
| Full model | 87.15 | All components |
| w/o Global Gate | 85.94 | −1.21%, no global signal control |
| Additive steering (vs. multiplicative) | 85.02 | Anisotropic scaling outperforms addition |
| w/o LGM (static scaling) | 84.21 | −2.94%, dynamic context is necessary |
| Linear probe (no correction) | 81.33 | −5.82%, correction is essential |
| Random head selection | 69.57 | RSA far exceeds random baseline |

### Key Findings

1. **Extreme data efficiency**: Increasing from 1% (~16 videos) to 100% of training data yields only a 0.27% AUC gain, while training time grows from <1 minute to 49 minutes.
2. **Cross-dataset generalization**: Training on UCF → evaluating on XD yields AP 71.31%; training on XD → evaluating on UCF yields AUC 81.04%.
3. **Cross-model generalization**: The framework generalizes effectively to LLaVA-OV (81.52%) and Qwen2.5-VL (84.11%).
4. **Per-category performance gap**: Assault achieves the highest AUC (95.17%), while Abuse achieves the lowest (68.84%)—the latter suffers from "contextual mimicry," as it is visually nearly indistinguishable from normal behavior.

## Highlights & Insights

1. **Paradigm innovation**: For the first time, "passive feature readout" is replaced by "active geometric intervention," reshaping the feature space without modifying any pre-trained parameters.
2. **Solid theoretical foundation**: Starting from the manifold hypothesis, the paper rigorously establishes topological properties of representation manifolds (compactness, piecewise path-connectivity, local Euclidean structure) to mathematically justify the intervention operations.
3. **Elegant stability of RSA**: The simple linear metric (Fisher ratio) identifies the same expert heads as expensive nonlinear alternatives (Silhouette, k-NN Purity) at 49× the speed.
4. **Strong practicality**: 520K parameters, 27-second training, 1% labeled data—highly deployment-friendly.
5. **Interpretability**: Post-hoc resubmission of anomalous frames to the MLLM for text-based explanation enhances trustworthiness.

## Limitations & Future Work

1. **"Contextual mimicry" anomalies remain difficult**: Events such as burglary versus normal entry are nearly visually indistinguishable, likely requiring longer-range temporal reasoning.
2. **Dependence on the MLLM backbone's video understanding capacity**: If the underlying MLLM has limited comprehension of video content, geometric correction may be insufficient.
3. **Evaluated only on VAD**: The generality of the framework remains to be validated on other video understanding tasks.
4. **Constraint of anomaly definition**: The approach relies heavily on a small amount of labeled data to define "anomaly." Open-set detection of entirely novel anomaly types, while theoretically supported (72.21% AUC on UBnormal), still leaves room for improvement.

## Related Work & Insights

- **Mechanistic interpretability**: The paper analyzes and intervenes on internal attention heads as "functional circuits," aligning closely with the mechanistic interpretability research direction.
- **Model editing**: Unlike knowledge editing, this method does not modify model weights; instead, it dynamically rectifies representations at inference time.
- **Broader implications**: The core idea—"identify critical internal modules + dynamic context-aware correction"—is generalizable to other downstream tasks requiring adaptation of frozen large models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Paradigm-shifting; first to introduce geometric intervention into frozen MLLM-based VAD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations, detailed stability analysis, 10-seed / cross-dataset / cross-model evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theory and experiments are clearly presented; appendix is exceptionally detailed.
- Value: ⭐⭐⭐⭐ — Strong practical utility, though currently limited to the VAD setting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](../../AAAI2026/multimodal_vlm/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](../../CVPR2026/multimodal_vlm/no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/multimodal_vlm/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[ICLR 2026\] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)
- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](../../ICCV2025/multimodal_vlm/analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)

</div>

<!-- RELATED:END -->
