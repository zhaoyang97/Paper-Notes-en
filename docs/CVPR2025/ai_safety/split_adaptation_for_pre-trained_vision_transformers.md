---
title: >-
  [Paper Note] Split Adaptation for Pre-trained Vision Transformers
description: >-
  [CVPR 2025][AI Safety][Privacy Protection] This paper proposes Split Adaptation (SA), which splits a pre-trained ViT into a front-end (quantized and sent to clients) and a back-end (retained on the server). It protects data privacy via bi-level noise injection, and mitigates noise degradation and overfitting through OOD augmentation and patch retrieval augmentation, achieving efficient few-shot downstream adaptation while securing both the model and the data.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Privacy Protection"
  - "Vision Transformer"
  - "Split Learning"
  - "Few-shot Adaptation"
  - "Model Intellectual Property"
date: 2026-05-08
content_hash: 8fd05de719f93dc4
---

# Split Adaptation for Pre-trained Vision Transformers

**Conference**: CVPR 2025  
**arXiv**: [2503.00441](https://arxiv.org/abs/2503.00441)  
**Code**: [https://github.com/conditionWang/Split_Adaptation](https://github.com/conditionWang/Split_Adaptation)  
**Area**: AI Safety  
**Keywords**: Privacy Protection, Vision Transformer, Split Learning, Few-shot Adaptation, Model Intellectual Property

## TL;DR
This paper proposes Split Adaptation (SA), which splits a pre-trained ViT into a front-end (quantized and sent to clients) and a back-end (retained on the server). It protects data privacy via bi-level noise injection, and mitigates noise degradation and overfitting through OOD augmentation and patch retrieval augmentation, achieving efficient few-shot downstream adaptation while securing both the model and the data.

## Background & Motivation

**Background**: Pre-trained ViTs have become foundational models widely used for downstream task adaptation in various domains. In privacy-sensitive fields such as healthcare and finance, clients hold private data and are unwilling to share it, while the server holds the model as intellectual property and is reluctant to disclose it, resulting in a dual protection requirement.

**Limitations of Prior Work**: (1) Standard adaptation methods require direct access to data, which is infeasible in private domains; (2) Sending models to clients for local adaptation (e.g., Federated Learning) leaks model IP and imposes heavy computational burdens; (3) Split Learning (SL) transmits middle representations, which can be vulnerable to Data Reconstruction Attacks (DRA) that recover original data; (4) Offsite Tuning sends a model emulator, leaving its parameters completely exposed and allowing adversaries to build high-performing models based on it.

**Key Challenge**: Secure the data (retaining it on the client side) and the model (preventing full parameter disclosure) simultaneously, while keeping client computational overhead low.

**Goal**: Design a ViT adaptation method that guarantees dual security for both data and models, enabling efficient adaptation in few-shot scenarios.

**Key Insight**: Split the ViT into a front-end (shallow layers) and a back-end (deep layers). Only the quantized front-end is sent to clients. Clients extract feature representations, inject noise, and transmit them back to the server. Quantization protects model parameters, while noise secures data privacy.

**Core Idea**: Model splitting + quantization for IP protection, bi-level noise (front-end parameter noise + representation noise) for privacy, and OOD augmentation + patch retrieval augmentation to compensate for the performance degradation caused by noise.

## Method

### Overall Architecture
SA operates within a client-server framework: (1) The server quantizes the first $k$ layers of the ViT into low-bit values and sends them to the client; (2) The client injects random noise into the quantized front-end, extracts data representations, injects additional noise into the representations, and transmits them back to the server; (3) The server trains the task modules using the back-end.

### Key Designs

1. **Model Splitting and Quantization Protection**:

    - Function: Protect the intellectual property of the pre-trained ViT.
    - Mechanism: Replace the original floating-point values of the first $k$ layers of ViT with low-bit quantization (e.g., INT4), making it impossible for attackers to recover original model parameters from the quantized front-end. The client only receives the quantized version of the shallow network, which prevents them from reconstructing the full model.
    - Design Motivation: Sending raw parameters directly is equivalent to leaking the model. The accuracy loss introduced by quantization serves as an effective mechanism for model protection.

2. **Bi-level Noise Injection**:

    - Function: Protect client data privacy and defend against data reconstruction attacks.
    - Mechanism: The client injects random noise into the received quantized front-end parameters (model-level) and then injects additional noise into the extracted intermediate representations (data-level). This two-fold noise prevents the server from recovering original images even if it attempts DRA.
    - Design Motivation: Single-layer noise may be bypassed by advanced attack methods; bi-level noise provides stronger privacy guarantees.

3. **OOD Augmentation and Patch Retrieval Augmentation**:

    - Function: Mitigate performance degradation caused by noise injection and few-shot settings.
    - Mechanism: Data-level OOD augmentation applies augmentation transformations in the representation space to improve generalization; model-level OOD augmentation adapts to the noise distribution during back-end training; Patch Retrieval Augmentation utilizes stored patch features for retrieval-based data augmentation, generating diverse representations to mitigate few-shot overfitting.
    - Design Motivation: Noise injection inevitably degrades representation quality, requiring corresponding augmentation strategies to compensate.

### Loss & Training
The standard cross-entropy loss is employed for downstream classification tasks. Training is conducted on the server side (only training the back-end and task modules), resulting in minimal computational overhead on the client side (which only needs to perform forward propagation on the front-end).

## Key Experimental Results

### Main Results

| Method | Data Protection | Model Protection | Client Overhead | Accuracy (Avg) |
|------|---------|---------|-----------|-------------|
| SA (Ours) | ✓ | ✓ | Small | **Best** |
| Offsite Tuning | ✓ | ✗ | Large | Second Best |
| Standard Split Learning | Partial | Partial | Large | Medium |
| Linear Probing | ✓ | ✓ | None | Lower |

### Ablation Study

| Configuration | Key Effects |
|------|---------|
| No noise injection | Highest performance but no privacy protection |
| Model-level noise only | Partial privacy protection |
| Bi-level noise (w/o OOD augmentation) | Strong privacy but significant performance degradation |
| Bi-level noise + OOD augmentation + Patch retrieval | Best trade-off |

### Key Findings
- SA successfully defends against advanced data reconstruction attacks such as FSHA, PCAT, Ginver, and FORA.
- Quantizing to INT4 effectively prevents model parameter recovery.
- Patch retrieval augmentation contributes the most to mitigating overfitting in few-shot scenarios.
- The computational overhead on the client is only a small fraction of that in split learning.

## Highlights & Insights
- Cleverly repurposes the precision loss of quantization as a mechanism for model protection rather than merely a compression tool.
- The "destruction-then-restoration" strategy of bi-level noise combined with OOD augmentation is elegantly designed, achieving a fine balance between privacy and performance.
- Patch retrieval augmentation is transferable to other few-shot learning scenarios.

## Limitations & Future Work
- The selection of quantization bits and noise intensity requires balancing the privacy-performance trade-off.
- Validated only on classification tasks; applicability to dense prediction tasks (e.g., segmentation, detection) remains to be verified.
- Robustness against stronger attacks (such as those incorporating auxiliary information) remains unknown.

## Related Work & Insights
- **vs Offsite Tuning**: Emulator parameters are completely exposed, leaving the model insecure. SA protects model IP through quantization.
- **vs Standard Split Learning**: Intermediate representations can be recovered by DRA. SA provides stronger privacy protection through bi-level noise.
- **vs Federated Learning**: Requires sending the complete model to clients and incurs heavy computation. SA only sends the quantized front-end.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative utilization of quantization for model protection, and well-designed bi-level noise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across multiple datasets, robustness against attacks, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problems and detailed descriptions of the method.
- Value: ⭐⭐⭐⭐ Highly practical for model adaptation in privacy-sensitive domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](../../ICCV2025/ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[ICLR 2026\] Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders](../../ICLR2026/ai_safety/zero-sacrifice_persistent-robustness_adversarial_defense_for_pre-trained_encoder.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[CVPR 2026\] Towards Robust Vision Transformers: Path Dependency Analysis and a Simple Two-Stage Adversarial Training](../../CVPR2026/ai_safety/towards_robust_vision_transformers_path_dependency_analysis_and_a_simple_two-sta.md)
- [\[CVPR 2026\] ReMoE: Region-Mixture Experts for Adversarially-Robust Vision Transformers](../../CVPR2026/ai_safety/remoe_region-mixture_experts_for_adversarially-robust_vision_transformers.md)

</div>

<!-- RELATED:END -->
