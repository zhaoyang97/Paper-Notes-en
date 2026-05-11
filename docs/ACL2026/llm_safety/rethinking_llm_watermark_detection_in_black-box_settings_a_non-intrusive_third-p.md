---
title: >-
  [Paper Note] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework
description: >-
  [ACL 2026][LLM Safety][LLM watermarking] This paper proposes TTP-Detect, the first black-box third-party watermark verification framework that decouples detection from injection. By leveraging a proxy model to amplify wa…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM watermarking"
  - "black-box detection"
  - "third-party auditing"
  - "hypothesis testing"
  - "proxy model"
date: 2026-05-08
content_hash: 0165456f0247e5a0
---

# Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework

**Conference**: ACL 2026
**arXiv**: [2603.14968](https://arxiv.org/abs/2603.14968)
**Code**: None
**Area**: AI Security / Watermark Detection
**Keywords**: LLM watermarking, black-box detection, third-party auditing, hypothesis testing, proxy model

## TL;DR
This paper proposes TTP-Detect, the first black-box third-party watermark verification framework that decouples detection from injection. By leveraging a proxy model to amplify watermark signals and combining three complementary metrics — local consistency, global geometry, and adaptive rank tests — it achieves high-accuracy detection across diverse watermarking schemes without access to secret keys or internal model states.

## Background & Motivation

**Background**: LLM watermarking embeds statistical signals during text generation to enable content traceability, serving as a key mechanism against AI-generated misinformation. Existing schemes (KGW, AAR, etc.) rely on secret keys for detection.

**Limitations of Prior Work**: Watermark injection and detection are tightly coupled — detection requires the same key used during injection. Court officials or platform auditors cannot independently verify watermarks and must rely on opaque claims from service providers. Disclosing keys to third parties compromises security, as adversaries could then imitate or remove watermarks.

**Key Challenge**: Existing private-key schemes cannot simultaneously support independent verification and key confidentiality, making genuine third-party auditing infeasible. Even recent publicly verifiable schemes still bind detection logic to specific injection mechanisms.

**Goal**: Design a key-agnostic black-box detection framework that enables a trusted third party (TTP) to determine the presence of a watermark from output text alone.

**Key Insight**: Reformulate absolute-threshold detection as a relative hypothesis testing problem — determining whether a query text better fits a watermarked or non-watermarked distribution.

**Core Idea**: Amplify watermark-related differences via a proxy model, and combine three complementary metrics — local consistency, global geometry, and adaptive rank tests — to capture statistical characteristics across different watermarking schemes.

## Method

### Overall Architecture
A three-party setup: the user submits query text, the service provider exposes an API (with watermarking toggle), and a trusted third-party auditor queries the API to obtain reference samples, constructs watermarked/non-watermarked reference sets, and applies a proxy model with multi-dimensional metrics to determine whether the query text is watermarked — all without accessing secret keys or internal model states.

### Key Designs

1. **Proxy-Based Representation Extraction**:

    - **Function**: Map text into a representation space that amplifies watermark-related differences.
    - **Mechanism**: Construct a training set $\mathcal{D}_{sft}$ consisting of watermarked/non-watermarked text pairs obtained from the provider API under identical prompts. The proxy model is fine-tuned via discriminative instruction tuning (learning to predict watermark labels), and the $\ell_2$-normalized hidden state of the last token in the final layer is extracted as the representation. This naturally separates watermarked and non-watermarked text in the representation space.
    - **Design Motivation**: Watermark signals are too weak to detect directly from raw text; the fine-tuned proxy model internalizes discriminative cues about watermarking.

2. **Three Complementary Relative Metrics**:

    - **Function**: Capture watermark evidence across different statistical scales.
    - **Mechanism**: (a) Local consistency test $A_{Loc}$: KNN-weighted density estimation of the proportion of watermarked samples in the neighborhood of the query text; (b) Global geometry tests: Mahalanobis distance $A_{Mah}$ captures covariance structure, and Energy distance $A_{Ene}$ handles non-Gaussian distributions; (c) Adaptive rank test $A_{Ada}$: captures watermark traces in generation dynamics via NLL statistics from the proxy model (global cross-entropy and local fluctuation), with adaptive inference of the watermark effect direction.
    - **Design Motivation**: Different watermarking schemes leave traces at different statistical scales; no single statistic is universally effective, necessitating complementary multi-module coverage.

3. **Ensemble and Robust Calibration**:

    - **Function**: Fuse multiple metrics into a unified decision score.
    - **Mechanism**: $A_{ens} = \sigma(\mathbf{w}^\top \mathbf{A} + b)$, where logistic regression weights are trained on an augmented validation set containing adversarially perturbed samples. The threshold $\tau$ is calibrated on a large-scale benign text corpus to achieve a target false positive rate.
    - **Design Motivation**: Robust calibration ensures reliability under adversarial attacks; threshold calibration supports legal/regulatory standards of evidence.

### Loss & Training
The proxy model is fine-tuned via conditional negative log-likelihood SFT. Ensemble weights are learned via logistic regression on an augmented validation set. Detection thresholds are calibrated by controlling the false positive rate.

## Key Experimental Results

### Main Results

| Watermarking Scheme | TPR↑ | TNR↑ | F1↑ | AUC↑ |
|---|---|---|---|---|
| KGW (Llama-3.1-8B, C4) | 0.980 | 0.980 | 0.980 | 0.998 |
| Unigram (Llama-3.1-8B, C4) | 1.000 | 0.990 | 0.995 | 0.999 |
| SWEET (Llama-3.1-8B, C4) | 0.985 | 0.965 | 0.975 | 0.997 |
| SynthID (Llama-3.1-8B, C4) | 0.865 | 0.930 | 0.894 | 0.938 |
| Unbiased (Llama-3.1-8B, C4) | 0.870 | 0.845 | 0.859 | 0.911 |
| UPV (Baseline) | 0.985 | 0.980 | 0.983 | 0.991 |

### Ablation Study

| Configuration | F1↑ | Note |
|---|---|---|
| Full TTP-Detect | 0.980 | Complete model |
| w/o Local Consistency | — | Local consistency test removed |
| w/o Global Geometry | — | Global geometry test removed |
| w/o Adaptive Rank | — | Adaptive rank test removed |

### Key Findings
- TTP-Detect achieves near-perfect detection on logits-based watermarks (KGW, Unigram) with F1 > 0.97, while maintaining F1 > 0.85 on distribution-preserving schemes (SynthID, Unbiased).
- Strong generalization across models (Llama-3.1-8B, OPT-6.7B) and datasets (C4, OpenGen).
- SymMark (a synthetic scheme) achieves perfect detection (TPR/TNR/F1/AUC all equal to 1.0).
- The three metric categories exhibit strong complementarity; removing any one leads to performance degradation on specific watermarking schemes.

## Highlights & Insights
- Reformulating watermark detection from "absolute thresholding" to "relative hypothesis testing" is a key innovation, enabling detection without knowledge of secret keys. This paradigm is generalizable to other black-box detection scenarios.
- The design of three complementary metrics is highly systematic: local neighborhood analysis, global distributional analysis, and dynamic likelihood analysis together constitute a comprehensive detection perspective.
- The adaptive inference of watermark effect direction within the adaptive rank test is practically valuable, as it avoids prior assumptions about specific watermarking mechanisms.

## Limitations & Future Work
- Reference samples (watermarked/non-watermarked pairs) must be obtained via API, which requires the service provider to support a watermarking toggle.
- The discriminative capability of the proxy model is constrained by the quality and scale of SFT training data.
- Detection performance is relatively weaker on distribution-preserving schemes (F1 ~0.85), which are inherently designed to minimize detectability.
- Future work may explore detection under zero-shot or few-shot reference conditions.

## Related Work & Insights
- **vs. KGW original detector**: Requires the secret key and knowledge of the specific scheme; the proposed method requires neither.
- **vs. UPV**: Still relies on shared parameters from the injection side; the proposed method fully decouples injection from detection.
- **vs. PVMark**: Wraps the detector with zero-knowledge proofs but still requires scheme-specific circuits; the proposed method is scheme-agnostic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve truly scheme-agnostic black-box third-party watermark detection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers diverse watermarking schemes and models, though detailed ablation under adversarial attacks is lacking.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear and mathematical formulations are rigorous.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical trust problem in AI governance with direct regulatory applicability.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](../../NeurIPS2025/llm_safety/on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)

</div>

<!-- RELATED:END -->
