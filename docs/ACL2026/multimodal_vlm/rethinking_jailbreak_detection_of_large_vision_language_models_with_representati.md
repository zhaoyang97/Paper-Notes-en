---
title: >-
  [Paper Note] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring
description: >-
  [ACL 2026][Multimodal VLM][Jailbreak Detection] This paper proposes the Representational Contrastive Scoring (RCS) framework, which analyzes the geometric structure of intermediate-layer representations within LVLMs. By learning a lightweight projection and applying contrastive scoring, RCS distinguishes malicious intent from benign distributional shift, achieving state-of-the-art jailbreak detection performance under a rigorous cross-attack-type generalization evaluation protocol.
tags:
  - ACL 2026
  - Multimodal VLM
  - Jailbreak Detection
  - Representational Contrastive Scoring
  - Large Vision-Language Models
  - OOD Detection
  - Safety Alignment
date: 2026-05-08
content_hash: 61fc3bf6edeab33e
---

# Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring

**Conference**: ACL 2026
**arXiv**: [2512.12069](https://arxiv.org/abs/2512.12069)
**Code**: [sarendis56/Jailbreak_Detection_RCS](https://github.com/sarendis56/Jailbreak_Detection_RCS)
**Area**: AI Safety / Multimodal VLM
**Keywords**: Jailbreak Detection, Representational Contrastive Scoring, Large Vision-Language Models, OOD Detection, Safety Alignment

## TL;DR

This paper proposes the Representational Contrastive Scoring (RCS) framework, which analyzes the geometric structure of intermediate-layer representations within LVLMs. By learning a lightweight projection and applying contrastive scoring, RCS distinguishes malicious intent from benign distributional shift, achieving state-of-the-art jailbreak detection performance under a rigorous cross-attack-type generalization evaluation protocol.

## Background & Motivation

**Background**: Large vision-language models are increasingly exposed to multimodal jailbreak attacks, including adversarial images, cross-modal prompt injection, and text-based jailbreak transfer. Defense methods must simultaneously generalize to unseen attacks and operate efficiently enough for real-time deployment.

**Limitations of Prior Work**: Existing defense strategies face a fundamental tension. Safety alignment and input filters are designed for known attack patterns and generalize poorly to novel attacks. Methods based on consistency checking, gradient computation, or multiple inference passes incur prohibitive computational overhead in high-throughput scenarios. Lightweight anomaly detection approaches (e.g., JailDAM) frame jailbreak detection as an OOD problem, but their one-class design models only the benign distribution and cannot distinguish between malicious intent and benign distributional shift, resulting in severe over-refusal.

**Key Challenge**: One-class OOD detection treats all inputs that deviate from the benign distribution as malicious; however, in practice, a large number of legitimate but unseen inputs also deviate from the training distribution. JailDAM's precision drops precipitously from 94.9% to 56.9% when unseen benign data (e.g., medical VQA) is introduced.

**Goal**: To design a detection method that is both efficient and capable of distinguishing malicious intent from mere distributional shift.

**Key Insight**: Research in representation engineering has demonstrated that intermediate-layer representations of LLMs encode rich semantic information about input safety, and that malicious and benign inputs exhibit separable geometric signatures at specific layers. These signals are more discriminative than general-purpose embeddings such as CLIP.

**Core Idea**: Inspect the geometric structure of intermediate-layer representations within the LVLM, learn a lightweight projection that maximizes the separation between benign and malicious inputs, and perform classification via contrastive scoring based on relative distances to benign versus malicious samples.

## Method

### Overall Architecture

RCS proceeds in three steps: (1) principled selection of the most discriminative safety-critical layer through geometric analysis; (2) learning a lightweight neural projection to amplify safety-relevant signals; and (3) computing a contrastive score in the projected space based on relative distances to benign and malicious samples. The framework is instantiated as two methods: parametric Mahalanobis Contrastive Detection (MCD) and non-parametric K-Nearest-Neighbor Contrastive Detection (KCD).

### Key Designs

1. **Safety-Critical Layer Selection**:

    - **Function**: Principally identify the intermediate layer in the LVLM that carries the strongest safety signal.
    - **Mechanism**: Using the SGXSTest dataset (semantically near-equivalent benign/malicious pairs), three complementary metrics are computed per layer — SVM maximum-margin separability, silhouette coefficient (cluster cohesion), and discriminant ratio (inter-class distance / intra-class variance). The layer with the highest aggregate score is selected as optimal. Experiments consistently identify middle layers as the "sweet spot" (layers 14–16 for LLaVA; layers 20–22 for Qwen).
    - **Design Motivation**: Early layers capture low-level features; final layers are over-specialized toward pretraining objectives; middle layers encode high-level semantic abstractions best suited for discriminating subtle malicious intent.

2. **Safety-Aware Projection**:

    - **Function**: Project high-dimensional hidden states into a low-dimensional space while amplifying safety-relevant signals.
    - **Mechanism**: The hidden state of the last token at the optimal layer (aggregated context prior to decoding) is extracted and projected to a 256-dimensional space via a three-layer feedforward network. The projection is trained with two losses: a dataset clustering loss $\mathcal{L}_{dataset}$ (promoting intra-dataset cohesion and inter-dataset separation) and a safety separation loss $\mathcal{L}_{sep}$ (maximizing the distance between benign and malicious centroids).
    - **Design Motivation**: Raw 4096-dimensional features suffer from the curse of dimensionality (unstable covariance estimation and kNN search) and contain many task-irrelevant dimensions. Projection eliminates noise and amplifies safety signals.

3. **Contrastive Scoring**:

    - **Function**: Determine input safety based on relative distances to benign and malicious distributions.
    - **Mechanism**: MCD models each dataset as an independent Gaussian distribution with Ledoit–Wolf shrinkage estimation for numerical stability; the score is $s_{\text{MCD}} = \min_{d \in \text{benign}} D_M - \min_{d \in \text{malicious}} D_M$. KCD makes no distributional assumption; after normalizing features, it computes the distance difference to the $k$-th nearest benign and malicious neighbors: $s_{\text{KCD}} = \|z - z_{(k)}^{\text{benign}}\| - \|z - z_{(k)}^{\text{malicious}}\|$.
    - **Design Motivation**: Contrastive scoring approximates the log-likelihood ratio required for an optimal Bayesian decision, fundamentally resolving the inability of one-class OOD methods to distinguish distributional shift from malicious intent.

### Loss & Training

The projection network is trained with the objective $\mathcal{L} = \alpha \mathcal{L}_{dataset} + \beta \mathcal{L}_{sep}$. The threshold $\theta$ is calibrated on a validation split of the training set by maximizing a weighted combination of balanced accuracy and F1 score. The entire detection procedure is completed prior to decoding, preventing harmful content from being generated.

## Key Experimental Results

### Main Results

| Method | Model | Accuracy (%) | AUROC (%) | AUPRC (%) | FPR (%) |
|--------|-------|-------------|-----------|-----------|---------|
| MCD (Ours) | LLaVA L16 | 91.0±2.3 | 98.6±0.1 | 98.8±0.1 | 15.2±5.2 |
| KCD (Ours) | LLaVA L16 | 92.0±2.1 | 97.7±0.9 | 97.2±1.2 | 10.1±6.1 |
| HiddenDetect | LLaVA | 81.6 | 90.1 | 90.0 | 16.8 |
| JailDAM | CLIP | 71.7 | 78.9 | 82.6 | 27.1 |
| GradSafe | LLaVA | 66.5 | 75.4 | 79.4 | 64.9 |

### Ablation Study

| Configuration | Description | Key Result |
|---------------|-------------|------------|
| JailDAM simplified evaluation | Single benign dataset | AUROC 91.3%, Precision 94.9% |
| JailDAM robust evaluation | + Unseen benign data | AUROC 70.6%, Precision 56.9% |
| No projection (raw features) | Direct use of high-dim hidden states | Significant performance degradation |
| PCA dimensionality reduction | Replaces learned projection | Inferior to safety-aware projection |

### Key Findings

- LVLM internal representations contain extraordinarily rich safety signals: a simple Mahalanobis-distance OOD detector applied directly to LLaVA intermediate-layer features achieves 99.4% AUROC, far surpassing JailDAM's 95.3%.
- Middle layers consistently outperform early and final layers, and this "sweet spot" can be reliably identified using noisier, non-paired data.
- Contrastive scoring is critical: one-class detection collapses when unseen benign data is introduced, whereas the contrastive framework remains robust.
- Both instantiations (MCD and KCD) are effective, demonstrating that the framework's efficacy does not depend on specific distributional assumptions.

## Highlights & Insights

- **Representation engineering perspective on safety detection**: The approach requires no external models or multiple inference passes. It relies solely on intermediate-layer features from a single forward pass through the target LVLM, incurring minimal computational overhead. This paradigm is transferable to safety protection in any LLM deployment scenario.
- **Principled layer selection**: Three complementary geometric metrics are jointly used to evaluate each layer's discriminative power, eliminating the ambiguity of manual layer selection. The finding that middle layers are optimal is consistent across different model families.
- **Contrastive scoring vs. one-class detection**: The experiment demonstrating JailDAM's precision collapse is a particularly compelling motivation, clearly illustrating why modeling both benign and malicious distributions simultaneously is necessary.

## Limitations & Future Work

- Training the projection network requires a small number of malicious samples; although these need not match the test attack types, the method is not applicable in zero-malicious-sample settings.
- Evaluation is limited to three LVLMs and a restricted set of attack types; validation across a broader range of models and attacks remains to be conducted.
- The projection dimensionality (256) and the kNN parameter $k$ (50) are set manually; automatic selection could further improve performance.
- Future directions include combining RCS with safety alignment as a dual safeguard, and exploring dynamic layer selection (adaptive layer choice at inference time).

## Related Work & Insights

- **vs. JailDAM**: JailDAM employs CLIP embeddings for one-class OOD detection, but CLIP does not encode safety-specific signals of the target model, and its one-class design leads to over-refusal. RCS uses the target model's own intermediate-layer representations together with contrastive scoring, fundamentally addressing both issues.
- **vs. GradSafe / HiddenDetect**: GradSafe requires gradient computation and HiddenDetect requires multi-layer feature aggregation, both incurring greater computational cost. RCS requires only a single-layer last-token feature and a lightweight projection, making it substantially more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ — Elegantly integrates representation engineering and OOD detection for multimodal jailbreak detection with a clear and effective formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Rigorous cross-model, cross-attack-type evaluation with well-designed ablation and motivation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Argumentation is logically tight, flowing coherently from motivation experiments through method design to empirical validation.
- Value: ⭐⭐⭐⭐ — Provides a practical and efficient detection solution for safe LVLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)
- [\[ACL 2026\] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)

<!-- RELATED:END -->
