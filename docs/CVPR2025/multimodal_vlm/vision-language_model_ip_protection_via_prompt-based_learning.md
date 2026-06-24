---
title: >-
  [Paper Note] Vision-Language Model IP Protection via Prompt-based Learning
description: >-
  [CVPR 2025][Multimodal VLM][IP Protection] This paper proposes the IP-CLIP framework, which achieves VLM IP protection on a frozen CLIP backbone via lightweight IP-Prompt learning (domain tokens + image tokens) and a style-augmented memory branch. This allows the model to maintain high accuracy on the authorized domain while deliberately degrading performance on unauthorized domains, resulting in a 0% performance drop on the authorized domain.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "IP Protection"
  - "CLIP"
  - "Prompt Learning"
  - "Domain Generalization Restriction"
  - "Non-transferable Learning"
date: 2026-05-08
content_hash: 5231f5147589f36d
---

# Vision-Language Model IP Protection via Prompt-based Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.02393](https://arxiv.org/abs/2503.02393)  
**Code**: [https://github.com/LyWang12/IP-CLIP](https://github.com/LyWang12/IP-CLIP)  
**Area**: Multimodal VLM  
**Keywords**: IP Protection, CLIP, Prompt Learning, Domain Generalization Restriction, Non-transferable Learning

## TL;DR

This paper proposes the IP-CLIP framework, which achieves VLM IP protection on a frozen CLIP backbone via lightweight IP-Prompt learning (domain tokens + image tokens) and a style-augmented memory branch. This allows the model to maintain high accuracy on the authorized domain while deliberately degrading performance on unauthorized domains, resulting in a 0% performance drop on the authorized domain.

## Background & Motivation

Training VLMs (such as CLIP) requires massive amounts of data, computational resources, and human effort, making the intellectual property (IP) protection of models crucial. IP protection not only encompasses ownership verification (who owns the model) but also requires restricting model deployment strictly to authorized data domains. A particularly insidious threat is that authorized users can easily transfer models to similar but unauthorized domains, causing implicit intellectual property infringement.

Existing IP protection methods face two major challenges: (1) Methods like NTL and CUTI require training from scratch or large-scale fine-tuning, which is prohibitively expensive for VLMs; (2) These methods rely solely on the visual backbone, lacking sufficient semantic information to distinguish between authorized and unauthorized domains. Prompt tuning methods in CLIP (e.g., CoOp, MaPLe) have proven capable of efficiently adapting to downstream tasks without full parameter fine-tuning, inspiring the authors to achieve IP protection using a similar lightweight approach.

## Method

### Overall Architecture

Based on a frozen CLIP backbone, IP-CLIP learns IP-Prompts (consisting of domain tokens to distinguish between authorized/unauthorized domains and image tokens to capture visual distributions). During training, images from authorized and unauthorized domains are input in parallel into the frozen visual encoder. The IP Projector extracts tokens from multi-scale features to construct prompts, which are then fed into the frozen text encoder. IP protection is achieved by optimizing classification accuracy on the authorized domain while deliberately degrading performance on the unauthorized domain. During inference, only the input image is required, and the IP Projector automatically generates the corresponding prompt.

### Key Designs

1. **IP-Prompt Learning (Domain token + Image token)**:
    - Function: Learning prompts that can distinguish domain identities and categories.
    - Mechanism: Two types of tokens are extracted from multiple layers $[f_v^{(1)}, \ldots, f_v^{(M)}]$ of the frozen visual encoder. Domain tokens $T_a / T_u$ encode domain style information using batch-wise feature statistics (mean $\mu$ and variance $\sigma$), which are mapped via the IP Projector. Image tokens $[V_1, \ldots, V_L]$ encode domain-agnostic content information from multi-scale features. The final prompt structure is $Prompt_a = [T_a; V_1, \ldots, V_L; [CLS]]$.
    - Design Motivation: Domain tokens encode domain style ("which domain") while image tokens encode image content ("which category"). Decoupling the two allows the model to simultaneously perform domain determination and category recognition. Training only the IP Projector achieves an extremely low parameter overhead.

2. **Style-Augmented Memory Branch (STAM)**:
    - Function: Enhancing the model's ability to distinguish domains using a feature bank.
    - Mechanism: Leveraging the zero-shot capability of CLIP to construct an $N$-way $K$-shot ($K=5$) feature bank $B_a, B_u \in \mathbb{R}^{N \times C}$ for both authorized and unauthorized domains. Taking the input feature $f_v^a$ as a query, the STAM module performs attention operations with $B_a$ (self-enhanced) and $B_u$ (cross-domain), respectively: $s_v^a = \text{Conv}(\text{softmax}(\frac{QK_a^T}{\sqrt{d_k}})V_a) + f_v^a$. The feature banks are constructed before training and frozen during training.
    - Design Motivation: The feature bank provides domain-level "prototypical memory", assisting the model in making more robust decisions based on representative domain features.

3. **New Evaluation Metrics System**:
    - Function: Comprehensively evaluating the effectiveness of IP protection.
    - Mechanism: Presenting a weighted metric $W_{ua} = A_a^{IP} \cdot [D_u - D_a]$, which simultaneously considers performance preservation in the authorized domain and performance degradation in the unauthorized domain. Optimality is reached when $D_a=0$ (no degradation in the authorized domain) and $D_u$ is large. The ownership verification metric $O_{ua} = A_u^{SL} \cdot [A_a^{Method} - A_u^{Method}]$ measures the performance discrepancy triggered by watermarks.
    - Design Motivation: Merely evaluating $D_u$ is insufficient; if the performance in the authorized domain also drops significantly, the protection loses its practical value.

### Loss & Training

Total Loss: $\mathcal{L} = \mathcal{L}_a - \mathcal{L}_u + \mathcal{L}_{ai} - \mathcal{L}_{ui} - \mathcal{L}_{kl} - \lambda_1 \cdot \mathcal{L}_m + \lambda_2 \cdot \mathcal{L}_{en}$

Explanation of terms: $\mathcal{L}_a / \mathcal{L}_u$ represents contrastive losses (maximizing/minimizing image-text alignment for authorized/unauthorized domains), $\mathcal{L}_{kl}$ leverages KL divergence to push apart the text feature distributions of the two domains, $\mathcal{L}_m$ uses MSE to maximize the distance between domain tokens, and $\mathcal{L}_{en}$ applies an entropy constraint to urge unauthorized domain text features toward a uniform distribution. The training uses the Adam optimizer with a learning rate of $10^{-5}$ on an RTX 3090.

Target-free scenario (no unauthorized domain data): OOD data generated via style augmentation is used to substitute the unauthorized domain.

## Key Experimental Results

### Main Results (Target-Specified IP Protection, Weighted Metric $W_{ua}$)

| Dataset | IP-CLIP | CUTI† | NTL† | CUTI(CNN) | NTL(CNN) |
|--------|---------|-------|------|-----------|----------|
| Office-31 | **74.84** | 72.48 | 54.98 | 70.09 | 62.11 |
| Office-Home-65 | **55.10** | 50.29 | 32.76 | 39.73 | 33.75 |
| Mini-DomainNet | **54.68** | 50.75 | 41.59 | 27.97 | 25.95 |
| Average $D_a$↓ | **0.00** | 1.27 | 2.07 | 0.53 | 1.55 |

### Ablation Study

| Scenario | Metric | Best Method | Description|
|------|------|---------|------|
| Ownership Verification $O_{ua}$ | 71.3% | IP-CLIP | Outperforms CUTI† by 5.6% and NTL† by 18.7% |
| Target-free | $W_{ua}$=53.46 | IP-CLIP | Valid even without unauthorized domain data |
| Target-specified | $D_a$=0.00% | IP-CLIP | Zero loss in authorized domain performance |
| CLIP vs CNN Backbone | Average +10~20% | CLIP-based | Richer semantic space |

### Key Findings

- The IP protection capability of CLIP backbones consistently outperforms CNN backbones, as the multimodal feature space provides richer semantic information for domain differentiation.
- The performance degradation of IP-CLIP in the authorized domain $D_a$ is near zero (averaging 0.00% across three datasets), indicating that IP protection barely affects normal usage.
- In the target-free scenario, unauthorized domain substitutes are automatically constructed via style augmentation; although $W_{ua}$ decreases from 74.84 to 53.46, it remains effective.
- The ownership verification metric reaches $O_{ua}=71.3\%$, significantly outperforming all baseline methods (p<0.05).

## Highlights & Insights

- **Plug-and-Play**: As a lightweight module, IP-Prompt can be integrated into the frontend of various CLIP-based models without modifying original model parameters.
- The method of utilizing multi-scale style statistics ($\mu$, $\sigma$) as domain representations is ingenious, cross-applying the concept of style transfer to IP protection.
- The three new metrics $W_{ua}$, $D_a$, and $O_{ua}$ provide fairer evaluation standards, avoiding the bias of solely evaluating $D_u$.
- A unified framework consolidates the two sub-problems of IP usage authorization and ownership verification.

## Limitations & Future Work

- Currently, validation is limited to discriminative models like CLIP, and has not yet been extended to generative VLMs (such as LLaVA, GPT-4V).
- The requirement for the authorized and unauthorized domains to share the same label space restricts applicability in some practical scenarios.
- The target-free performance is significantly lower than target-specified performance ($W_{ua}$ 53.46 vs 74.84), indicating limited protection strength.
- Robustness against adversarial attacks (fine-grained tuning, knowledge distillation, model pruning) is not discussed.

## Related Work & Insights

- NTL/CUTI pioneered the direction of active authorization, but IP-CLIP demonstrates that prompt-based methods are more efficient in VLM scenarios.
- Compared with CoOp/MaPLe, IP-CLIP extends prompts from "task adaptation" to "security constraints", opening up a new utility for prompt tuning.
- The approach of H-NTL using causal models to decouple content/style shares the same goal as IP-CLIP's domain token design.
- Insight for model security research: Complex security policies can be implemented in a lightweight manner at the prompt level.

## Rating

- Novelty: ⭐⭐⭐⭐ Instructs prompt learning for VLM IP protection for the first time, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + three scenarios + new metrics + statistical significance testing.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, detailed method descriptions, formulas are dense but trackable.
- Value: ⭐⭐⭐⭐ High practical value but limited to classification scenarios; IP protection for generative tasks remains to be explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/multimodal_vlm/advancing_textual_prompt_learning_with_anchored_attributes.md)

</div>

<!-- RELATED:END -->
