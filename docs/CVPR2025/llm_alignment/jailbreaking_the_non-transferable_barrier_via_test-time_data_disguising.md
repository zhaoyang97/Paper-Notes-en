---
title: >-
  [Paper Note] Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising
description: >-
  [CVPR 2025][LLM Alignment][non-transferable learning] This paper proposes JailNTL, the first black-box attack method against Non-Transferable Learning (NTL) models. By utilizing test-time data disguising to transform unauthorized domain data into the style of the authorized domain, it improves unauthorized domain accuracy by up to 55.7% using only 1% authorized samples, without requiring any model modifications.
tags:
  - "CVPR 2025"
  - "LLM Alignment"
  - "non-transferable learning"
  - "model IP protection"
  - "adversarial attack"
  - "domain adaptation"
  - "data disguising"
date: 2026-05-08
content_hash: 710f464078fefac4
---

# Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising

**Conference**: CVPR 2025  
**arXiv**: [2503.17198](https://arxiv.org/abs/2503.17198)  
**Code**: [https://github.com/tmllab/2025_CVPR_JailNTL](https://github.com/tmllab/2025_CVPR_JailNTL)  
**Area**: LLM Alignment  
**Keywords**: non-transferable learning, model IP protection, adversarial attack, domain adaptation, data disguising

## TL;DR
This paper proposes JailNTL, the first black-box attack method against Non-Transferable Learning (NTL) models. By utilizing test-time data disguising to transform unauthorized domain data into the style of the authorized domain, it improves unauthorized domain accuracy by up to 55.7% using only 1% authorized samples, without requiring any model modifications.

## Background & Motivation

**Background**: Non-Transferable Learning (NTL) is a model intellectual property (IP) protection technique that trains models to perform well on authorized domains while experiencing severe performance degradation on unauthorized domains (creating a "non-transferable barrier"). Existing attack methods require white-box access to the model (i.e., visible parameters).

**Limitations of Prior Work**: (1) White-box attacks (such as RTAL, FTAL, and TransNTL) require access to full model parameters, which is impractical in real-world API deployment scenarios. (2) Existing white-box attacks show limited academic performance—TransNTL only restores 27.2% accuracy on CIFAR10→STL10. (3) No black-box attack methods currently exist.

**Key Challenge**: The non-transferable barrier of NTL models is essentially a "semantic rupture" between domains—the model has learned to identify distribution differences between authorized and unauthorized domains. If unauthorized domain data can be "disguised" as authorized domain data during test time, the barrier can be successfully bypassed.

**Goal**: To restore model performance on unauthorized domains under the black-box constraint of only access to model logits, extremely limited authorized samples (1%), and unlabeled unauthorized test data.

**Key Insight**: Instead of modifying the model, this work modifies the input. A data disguising network is trained to project unauthorized domain data onto the authorized domain distribution while preserving the semantic content.

**Core Idea**: Train a lightweight disguising network to ensure unauthorized domain data mimics the appearance of the authorized domain while keeping its semantic content intact, thereby "deceiving" the NTL model into crossing the non-transferable barrier.

## Method

### Overall Architecture
JailNTL comprises two core components: Data-Intrinsic Disguising (DID) manages the appearance disguise, and Model-Guided Disguising (MGD) leverages NTL model logits to further optimize the disguising effect. During inference, unauthorized domain data is simply passed through the disguising network before being fed into the NTL model.

### Key Designs

1. **Data-Intrinsic Disguising (DID)**

    - **Function**: Transforms the appearance of unauthorized domain data to fit the authorized domain distribution.
    - **Mechanism**:
        - Forward disguising: $f_d(x_u) \approx$ authorized domain appearance, with an adversarial loss $L_{adv}$ making it indistinguishable to the discriminator.
        - Backward reconstruction: $\hat{f}_d(f_d(x_u)) \approx x_u$, applying a consistency loss $L_{cs}$ to preserve semantic content.
        - Bidirectional structure: Simultaneously train a backward disguising network $\hat{f}_d(x_a) \approx$ unauthorized domain appearance.
        - Disguising network $f_d$: ResNet architecture (2 downsampling + 9 residual blocks + 2 upsampling).
    - **Design Motivation**: GAN-style training ensures that the distribution of disguised data is well-aligned with the authorized domain.

2. **Model-Guided Disguising (MGD)**

    - **Function**: Benefits from the black-box outputs of the NTL model to further refine the disguise.
    - **Mechanism**:
        - Confidence matching: $L_{cf} = |E_{cf}(x_a) - E_{cf}(f_d(x_u))|$, which matches the entropy of output probabilities of the disguised data with that of the authorized data.
        - Category balancing: $L_{ba} = |E_{ba}(P_{disg}) - E_{ba}(P_{auth})|$, ensuring that the predicted class distribution of disguised data aligns with the authorized domain.
        - Zero-order gradient estimation: Approximates gradients using finite differences, bypassing the need for internal model parameters.
    - **Design Motivation**: While DID operates strictly on distribution-level disguising, MGD further aligns the disguise based on the model's behavioral response.

### Loss & Training
- Total loss: $L_{total} = L_{adv} + L_{advr} + \lambda_{cs}(L_{cs} + L_{csr}) + \lambda_{cf}L_{cf} + \lambda_{ba}L_{ba}$
- Optimizer: SGD (degrades) / Adam (enhances), with learning rates of $\gamma_e = 10^{-4}$ and $\gamma_g = 2 \times 10^{-4}$
- Discriminator: PatchGAN, batch size 5, RTX 4090 GPU

## Key Experimental Results

### Main Results

| Domain Pair | NTL Method | Unauthorized Domain Acc (Before Attack) | Unauthorized Domain Acc (JailNTL) | Gain |
|-----|---------|-------------------|---------------------|------|
| CIFAR10→STL10 | CUTI | 9.0 | 64.7 | **+55.7** |
| CIFAR10→STL10 | NTL | 9.8 | 61.4 | +51.6 |
| STL10→CIFAR10 | CUTI | 9.9 | 43.5 | +33.6 |
| VisDA-T→V | CUTI | 10.0 | 25.4 | +15.4 |

**vs White-box SOTA TransNTL:** CIFAR10→STL10 27.2% → JailNTL 61.4% (+34.2%, and JailNTL is black-box!)

### Ablation Study

| Configuration | CIFAR10→STL10 (CUTI) |
|------|---------------------|
| DID only (JailNTL*) | 64.0% |
| + Lcf (Confidence) | 64.3% |
| + Lba (Class Balance) | 64.0% |
| Full JailNTL | **64.7%** |

| Joint Attack | STL10→CIFAR10 |
|---------|---------------|
| TransNTL alone | 37.7% |
| TransNTL + JailNTL | **44.1%** (+6.4%) |

### Key Findings
- **Black-box Outperforms White-box**: JailNTL (black-box) achieves 61.4% vs TransNTL (white-box) 27.2%, proving that data-level disguise is more effective than model-level parameter tuning.
- DID accounts for the vast majority of the performance recovery (64.0%), while MGD provides incremental refinement.
- t-SNE visualization: Disguised data (blue) clusters closely with the authorized domain (green), moving far away from the unauthorized domain (red).
- Grad-CAM analysis: Attention maps of disguised data closely match those of the authorized domain.
- Complementarity with white-box methods: The combination of TransNTL + JailNTL secures better results than using either method alone.

## Highlights & Insights
- **Paradigm Shift in Domain Attack**: Modifying input data rather than the model parameters is inherently better suited for black-box environments.
- **Only 1% Authorized Samples Required**: The minimal data requirement makes this attack exceptionally practical in real-world scenarios.
- **Breaking Conventional Wisdom**: Overturning the assumption that "white-box is always stronger than black-box" reveals the underlying vulnerability of current NTL barriers.
- Imparts a critical security warning to the NTL research community — current non-transferable barriers are highly vulnerable to test-time disguises.

## Limitations & Future Work
- The performance improvement on VisDA is relatively limited (+15.4%), indicating that the disguise is less effective across large domain discrepancies.
- The disguising network itself requires training resources and several authorized target-domain samples.
- It has not been tested against more advanced, structurally modified NTL defenses.
- Since disguised images are modified at the pixel level, they might be detected by simple input inspection mechanisms.

## Related Work & Insights
- **vs TransNTL**: A white-box method that directly fine-tunes model parameters, showing performance significantly inferior to JailNTL.
- **vs RTAL/FTAL**: White-box fine-tuning strategies that struggle to restore virtually any performance on the unauthorized domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Represents the first black-box attack on NTL; the test-time data disguising concept is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across several domain pairs alongside detailed ablation and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ The problem definition and methodologies are clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Delivers an essential warning concerning current NTL security models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] On the Rejection Criterion for Proxy-Based Test-Time Alignment](../../ACL2026/llm_alignment/on_the_rejection_criterion_for_proxy-based_test-time_alignment.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](../../ACL2025/llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Test-Time Alignment for Large Language Models via Textual Model Predictive Control](../../ICLR2026/llm_alignment/test-time_alignment_for_large_language_models_via_textual_model_predictive_contr.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)

</div>

<!-- RELATED:END -->
