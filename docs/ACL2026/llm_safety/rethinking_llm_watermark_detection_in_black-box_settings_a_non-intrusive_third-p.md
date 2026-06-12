---
title: >-
  [Paper Note] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework
description: >-
  [ACL 2026][LLM Safety][LLM Watermarking] The paper proposes TTP-Detect, the first black-box third-party watermark verification framework that decouples watermark detection and injection. By using a proxy model to amplify…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM Watermarking"
  - "Black-box Detection"
  - "Third-party Auditing"
  - "Hypothesis Testing"
  - "Proxy Model"
date: 2026-05-08
content_hash: a024ea63f93a0eeb
---

# Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework

**Conference**: ACL 2026  
**arXiv**: [2603.14968](https://arxiv.org/abs/2603.14968)  
**Code**: None  
**Area**: AI Safety / Watermark Detection  
**Keywords**: LLM Watermarking, Black-box Detection, Third-party Auditing, Hypothesis Testing, Proxy Model

## TL;DR
The paper proposes TTP-Detect, the first black-box third-party watermark verification framework that decouples watermark detection and injection. By using a proxy model to amplify watermark signals and combining three complementary metrics—local consistency, global geometry, and adaptive rank testing—it achieves high-precision detection across various watermarking schemes without accessing secret keys or internal model states.

## Background & Motivation

**Background**: LLM watermarking embeds statistical signals during the generation process for content provenance, serving as a vital mechanism against AI-generated misinformation. Existing solutions (KGW, AAR, etc.) rely on secret keys to detect watermarks.

**Limitations of Prior Work**: Watermark injection and detection are tightly coupled—detection must use the same key as injection. Courts or platform auditors cannot independently verify watermarks and must rely on opaque claims from service providers. Disclosing keys to third parties would compromise security (adversaries could mimic or remove watermarks).

**Key Challenge**: Existing private-key schemes cannot simultaneously support independent verification and maintain key secrecy, making true third-party auditing impossible. Even recent publicly verifiable schemes still bind detection logic to specific injection mechanisms.

**Goal**: Design a key-agnostic black-box detection framework that allows a Trusted Third Party (TTP) to judge whether text contains a watermark solely from the output.

**Key Insight**: Reframe absolute threshold detection as a relative hypothesis testing problem—determining whether the query text fits the watermarked distribution or the non-watermarked distribution better.

**Core Idea**: Amplify watermark-related differences through a proxy model and capture statistical characteristics of different watermarking schemes by combining three complementary metrics: local consistency, global geometry, and adaptive rank testing.

## Method

### Overall Architecture
Three-party setup: The user submits query text; the service provider exposes an API (supporting a watermark toggle); the trusted third-party auditor obtains reference samples through the API to construct watermarked/non-watermarked reference sets. The TTP determines whether the query contains a watermark using the proxy model and multi-dimensional metrics. No access to keys or internal model states is required throughout the process.

### Key Designs

1.  **Proxy-Based Representation**:
    *   **Function**: Maps text to a representation space that amplifies watermark differences.
    *   **Mechanism**: Construct a training set $\mathcal{D}_{sft}$ by obtaining watermarked/non-watermarked text pairs for the same prompts via the provider's API. Perform discriminative instruction fine-tuning on a proxy model (learning to predict watermark labels), then extract the $\ell_2$-normalized hidden state of the last token in the last layer as the representation. This naturally separates watermarked and non-watermarked text in the representation space.
    *   **Design Motivation**: Watermark signals extracted directly from raw text are too weak; the fine-tuned proxy model can internalize discriminatory clues.

2.  **Three Complementary Relative Metrics**:
    *   **Function**: Captures watermark traces across different statistical scales.
    *   **Mechanism**: (a) Local Consistency Test $A_{Loc}$: Uses KNN weighted density estimation to calculate the proportion of watermarked samples in the query text's neighborhood; (b) Global Geometry Test: Mahalanobis distance $A_{Mah}$ captures covariance structure, and Energy distance $A_{Ene}$ handles non-Gaussian distributions; (c) Adaptive Rank Test $A_{Ada}$: Captures watermark traces in generation dynamics through proxy model NLL statistics (global cross-entropy and local volatility) and adaptively infers the direction of the watermark effect.
    *   **Design Motivation**: Different watermarking schemes leave traces at different statistical scales; a single statistic cannot be universal. Multi-module complementarity ensures coverage.

3.  **Ensemble and Robust Calibration**:
    *   **Function**: Fuses multiple metrics into a unified decision score.
    *   **Mechanism**: $A_{ens} = \sigma(\mathbf{w}^\top \mathbf{A} + b)$, with logistic regression weights trained on an augmented validation set containing adversarial perturbations. The threshold $\tau$ is calibrated on a large-scale benign text set according to the target false alarm rate.
    *   **Design Motivation**: Robust calibration ensures reliability under adversarial attacks, and threshold calibration supports evidence standards for legal/regulatory use.

### Loss & Training
The proxy model is fine-tuned using conditional negative log-likelihood (SFT). Ensemble weights are learned via logistic regression on the augmented validation set. The detection threshold is calibrated by controlling the false alarm rate.

## Key Experimental Results

### Main Results

| Watermarking Scheme | TPR↑ | TNR↑ | F1↑ | AUC↑ |
|----------|------|------|-----|------|
| KGW (Llama-3.1-8B, C4) | 0.980 | 0.980 | 0.980 | 0.998 |
| Unigram (Llama-3.1-8B, C4) | 1.000 | 0.990 | 0.995 | 0.999 |
| SWEET (Llama-3.1-8B, C4) | 0.985 | 0.965 | 0.975 | 0.997 |
| SynthID (Llama-3.1-8B, C4) | 0.865 | 0.930 | 0.894 | 0.938 |
| Unbiased (Llama-3.1-8B, C4) | 0.870 | 0.845 | 0.859 | 0.911 |
| UPV (Baseline) | 0.985 | 0.980 | 0.983 | 0.991 |

### Ablation Study

| Configuration | F1↑ | Description |
|------|-----|------|
| Full TTP-Detect | 0.980 | Full model |
| w/o Local Consistency | - | Remove local consistency test |
| w/o Global Geometry | - | Remove global geometry test |
| w/o Adaptive Rank | - | Remove adaptive rank test |

### Key Findings
- TTP-Detect achieves near-perfect detection on logits-based watermarks (KGW, Unigram) with F1 > 0.97, and maintains 0.85+ F1 on distribution-preserving schemes (SynthID, Unbiased).
- Shows good generalization across models (Llama-3.1-8B, OPT-6.7B) and datasets (C4, OpenGen).
- SymMark (synthetic scheme) achieves perfect detection (TPR/TNR/F1/AUC all 1.0).
- The three types of metrics are highly complementary; removing any leads to a performance drop on specific schemes.

## Highlights & Insights
- Reforming watermark detection from "absolute thresholding" to "relative hypothesis testing" is the key innovation, making detection without keys possible. This logic can be generalized to other scenarios requiring black-box detection.
- The design of three complementary metrics is systematic: looking at the neighborhood (local), distribution (global), and likelihood (dynamic) provides a complete detection perspective.
- The design in the Adaptive Rank Test that automatically infers the watermark effect direction is practical, avoiding prior assumptions about specific mechanisms.

## Limitations & Future Work
- Requires obtaining reference samples (watermarked/non-watermarked pairs) through an API, relying on service providers to offer a watermark toggle.
- The discriminative power of the proxy model is limited by the quality and scale of SFT training data.
- Detection performance is relatively weaker on distribution-preserving schemes (F1~0.85), as these are inherently designed to minimize detectability.
- Future work could explore detection under zero-shot or few-shot reference conditions.

## Related Work & Insights
- **vs KGW Original Detector**: Requires keys and knowledge of the specific scheme; this work requires neither.
- **vs UPV**: Still relies on shared parameters at the injection end; this work completely decouples injection and detection.
- **vs PVMark**: Uses zero-knowledge proofs to wrap the detector but still requires scheme-specific circuits; this work is scheme-agnostic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve truly scheme-agnostic black-box third-party watermark detection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple schemes and models, though lacks detailed ablation on adversarial attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and rigorous mathematical notation.
- Value: ⭐⭐⭐⭐⭐ Addresses a key transparency issue in AI governance with direct regulatory application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ACL 2026\] SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones](slim_stealthy_low-coverage_black-box_watermarking_via_latent-space_confusion_zon.md)
- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](../../NeurIPS2025/llm_safety/on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)

</div>

<!-- RELATED:END -->
