---
title: >-
  [Paper Note] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring
description: >-
  [ACL 2026][LLM Safety][Jailbreak Detection] Ours proposes the Representational Contrastive Scoring (RCS) framework to analyze the geometric structure of internal intermediate layer representations in LVLMs. By utilizing…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Jailbreak Detection"
  - "Representational Contrastive Scoring"
  - "Large Vision Language Models"
  - "OOD Detection"
  - "Safety Alignment"
date: 2026-05-08
content_hash: b5e57644fb81b182
---

# Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring

**Conference**: ACL 2026  
**arXiv**: [2512.12069](https://arxiv.org/abs/2512.12069)  
**Code**: [sarendis56/Jailbreak_Detection_RCS](https://github.com/sarendis56/Jailbreak_Detection_RCS)  
**Area**: AI Safety / Multi-modal VLM  
**Keywords**: Jailbreak Detection, Representational Contrastive Scoring, Large Vision Language Models, OOD Detection, Safety Alignment

## TL;DR

Ours proposes the Representational Contrastive Scoring (RCS) framework to analyze the geometric structure of internal intermediate layer representations in LVLMs. By utilizing lightweight projections and contrastive scoring, it distinguishes malicious intent from benign distribution shifts, achieving SOTA jailbreak detection performance under rigorous evaluation protocols for cross-attack generalization.

## Background & Motivation

**Background**: Large Vision Language Models (LVLMs) face increasing multi-modal jailbreak attacks—adversarial images, cross-modal prompt injection, and text-based jailbreak transfer. Defense methods must simultaneously satisfy generalization to unknown attacks and computational efficiency for real-time deployment.

**Limitations of Prior Work**: Existing defense strategies face a fundamental contradiction. Safety alignment and input filters are designed for known attack patterns and generalize poorly to novel attacks. Methods based on consistency checks, gradient computation, or multiple inferences incur high computational costs, making them unsuitable for high-throughput scenarios. Lightweight anomaly detection methods (e.g., JailDAM) treat jailbreak detection as an OOD problem. However, their one-class design only models the benign distribution, failing to distinguish between "malicious intent" and "benign distribution shift," which leads to severe over-refusal issues.

**Key Challenge**: One-class OOD detection treats all inputs deviating from the benign distribution as malicious. In reality, many unseen legitimate inputs also deviate from the training distribution. The precision of JailDAM plummeted from 94.9% to 56.9% upon introducing unseen benign data (e.g., medical VQA).

**Goal**: Design a detection method that is both efficient and capable of distinguishing malicious intent from mere distribution shifts.

**Key Insight**: Representation engineering research indicates that intermediate representations of LLMs encode rich semantic information regarding input safety. Malicious and benign inputs possess separable geometric signatures in specific layers. These signals are more discriminative than general-purpose embeddings like CLIP.

**Core Idea**: Examine the geometric structure of internal intermediate representations in LVLMs, learn lightweight projections to maximize the separability of benign vs. malicious inputs, and perform discrimination using contrastive scoring (relative distance to benign vs. malicious samples).

## Method

### Overall Architecture

RCS consists of three steps: (1) Select the most discriminative safety-critical layers based on principled geometric analysis; (2) Learn lightweight neural projections to amplify safety-related signals; (3) Compute contrastive scores in the projected space based on the relative distance to benign/malicious samples. The framework is instantiated as two methods: the parametric MCD (Mahalanobis distance Contrastive Detection) and the non-parametric KCD (K-nearest neighbor Contrastive Detection).

### Key Designs

1.  **Safety-Critical Layer Selection**:

    *   **Function**: Principally identify the intermediate layers in the LVLM with the strongest safety signals.
    *   **Mechanism**: Using the SGXSTest dataset (semantically similar benign/malicious pairs), three complementary metrics are calculated for each layer: SVM maximum margin separation, Silhouette coefficient (clustering cohesion), and Discriminant Ratio (inter-class distance / intra-class variance). The layer with the highest composite score is selected as the optimal layer. Experiments consistently show that intermediate layers (Layers 14-16 for LLaVA, Layers 20-22 for Qwen) are the "sweet spot."
    *   **Design Motivation**: Early layers capture low-level features, while final layers are over-specialized toward pre-training objectives. Intermediate layers encode high-level semantic abstractions, making them most suitable for distinguishing subtle malicious intent.

2.  **Safety-Aware Projection**:

    *   **Function**: Project high-dimensional hidden states into a low-dimensional space and amplify safety-related signals.
    *   **Mechanism**: Extract the hidden state of the last token at the optimal layer (aggregated context before decoding) and project it into a 256-dimensional space via a three-layer feed-forward network. The projection objective combines two losses: the dataset clustering loss $\mathcal{L}_{dataset}$ (intra-dataset cohesion, cross-dataset separation) and the safety separation loss $\mathcal{L}_{sep}$ (maximizing distance between benign/malicious centroids).
    *   **Design Motivation**: Original 4096-dimensional features suffer from the curse of dimensionality (unstable covariance estimation and kNN search) and contain many task-irrelevant dimensions. Projection eliminates noise and amplifies safety signals.

3.  **Contrastive Scoring**:

    *   **Function**: Determine input safety via the relative distance to benign and malicious distributions.
    *   **Mechanism**: The MCD method models each dataset as an independent Gaussian distribution, using Ledoit-Wolf shrinkage estimation to ensure numerical stability of the covariance. The score is $s_{\text{MCD}} = \min_{d \in \text{benign}} D_M - \min_{d \in \text{malicious}} D_M$. The KCD method makes no distributional assumptions; after normalizing features, it computes the distance difference to the $k$-th nearest benign/malicious neighbors: $s_{\text{KCD}} = \|z - z_{(k)}^{\text{benign}}\| - \|z - z_{(k)}^{\text{malicious}}\|$.
    *   **Design Motivation**: Contrastive scoring approximates the log-likelihood ratio required for optimal Bayesian decision-making, fundamentally solving the problem where one-class OOD methods cannot distinguish distribution shifts from malicious intent.

### Loss & Training

The training objective for the projection network is $\mathcal{L} = \alpha \mathcal{L}_{dataset} + \beta \mathcal{L}_{sep}$. The threshold $\theta$ is calibrated on the validation split of the training set to maximize a weighted combination of balanced accuracy and F1 score. Detection is completed before decoding to prevent the generation of harmful content.

## Key Experimental Results

### Main Results

| Method | Model | Accuracy (%) | AUROC (%) | AUPRC (%) | FPR (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MCD (Ours)** | **LLaVA L16** | **91.0±2.3** | **98.6±0.1** | **98.8±0.1** | **15.2±5.2** |
| **KCD (Ours)** | **LLaVA L16** | **92.0±2.1** | **97.7±0.9** | **97.2±1.2** | **10.1±6.1** |
| HiddenDetect | LLaVA | 81.6 | 90.1 | 90.0 | 16.8 |
| JailDAM | CLIP | 71.7 | 78.9 | 82.6 | 27.1 |
| GradSafe | LLaVA | 66.5 | 75.4 | 79.4 | 64.9 |

### Ablation Study

| Configuration | Description | Key Result |
| :--- | :--- | :--- |
| JailDAM Simplified Eval | Single benign dataset | AUROC 91.3%, Precision 94.9% |
| JailDAM Robust Eval | + Unseen benign data | AUROC 70.6%, Precision 56.9% |
| No Projection | Use high-dim hidden states directly | Significant performance drop |
| PCA Reduction | Replace learned projection | Inferior to safety-aware projection |

### Key Findings

*   Internal representations of LVLMs contain extremely rich safety signals: Simple Mahalanobis distance OOD detection using LLaVA intermediate layer features reaches 99.4% AUROC, far exceeding JailDAM's 95.3%.
*   Intermediate layers consistently outperform early and final layers, and this "sweet spot" can be reliably identified using noisy, non-paired data.
*   Contrastive scoring is critical: One-class detection collapses when introducing unseen benign data, whereas the contrastive framework remains robust.
*   Both instantiations (MCD and KCD) are effective, indicating the framework's effectiveness does not rely on specific distributional assumptions.

## Highlights & Insights

*   **Safety Detection from a Representation Engineering Perspective**: Does not rely on external models or multiple inferences. It uses only one forward pass of the target LVLM's intermediate layer features, resulting in extremely low computational overhead. This approach is transferable to safety protection scenarios for any LLM.
*   **Principled Layer Selection Method**: Jointly evaluates layer discriminative power using three complementary geometric metrics, avoiding the uncertainty of manual selection and yielding consistent conclusions across different models (intermediate layers are best).
*   **Contrastive Scoring vs. One-class Detection**: The experiment showing the collapse of JailDAM's precision serves as a powerful motivation, clearly explaining why both benign and malicious distributions must be modeled simultaneously.

## Limitations & Future Work

*   Requires a small number of malicious samples to train the projection network. Although they do not need to match the test attack types, the method is not applicable in zero-malicious-sample scenarios.
*   Evaluation is limited to three LVLMs and a finite set of attack types; broader model and attack coverage remains to be verified.
*   Projection dimensions (256) and the $k$ value for kNN (50) are manually set; automatic selection might further enhance performance.
*   Exploration areas: Combining RCS with safety alignment for dual protection, and dynamic layer selection (adaptive layer selection during inference).

## Related Work & Insights

*   **vs. JailDAM**: JailDAM uses CLIP embeddings for one-class OOD detection, but CLIP does not encode safety-specific signals of the target model, and the one-class design leads to over-refusal. RCS uses the target model's own intermediate layers and contrastive scoring to fundamentally solve these two problems.
*   **vs. GradSafe/HiddenDetect**: GradSafe requires gradient computation, and HiddenDetect requires multi-layer feature aggregation, resulting in higher computational costs. RCS only requires the last token feature of a single layer plus a lightweight projection, making it more efficient.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ Ingeniously combines representation engineering and OOD detection for multi-modal jailbreak detection; the logic is clear and effective.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rigorous evaluation across models and attack types; ablation and motivation experiments are elegantly designed.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Logical argumentation is tight, flowing seamlessly from motivation to design to validation.
*   **Value**: ⭐⭐⭐⭐ Provides a practical and efficient detection solution for the secure deployment of LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)

</div>

<!-- RELATED:END -->
