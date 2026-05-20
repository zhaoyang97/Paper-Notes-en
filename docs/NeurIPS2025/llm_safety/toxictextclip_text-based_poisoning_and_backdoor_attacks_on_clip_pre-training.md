---
title: >-
  [Paper Note] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training
description: >-
  [NeurIPS 2025][LLM Safety][CLIP security] This paper proposes ToxicTextCLIP, a framework that generates high-quality adversarial texts during CLIP pre-training via two modules—Background-aware Target Text Selector and Ba…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "CLIP security"
  - "data poisoning"
  - "backdoor attack"
  - "textual adversarial attack"
  - "multimodal security"
date: 2026-05-08
content_hash: a4f088f10482fa48
---

# ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training

**Conference**: NeurIPS 2025
**arXiv**: [2511.00446](https://arxiv.org/abs/2511.00446)  
**Code**: [GitHub](https://github.com/xinyaocse/ToxicTextCLIP/)  
**Area**: AI Security
**Keywords**: CLIP security, data poisoning, backdoor attack, textual adversarial attack, multimodal security

## TL;DR

This paper proposes ToxicTextCLIP, a framework that generates high-quality adversarial texts during CLIP pre-training via two modules—Background-aware Target Text Selector and Background-driven Poisoned Text Augmenter—achieving up to 95.83% attack success rate and 98.68% backdoor Hit@1, while successfully bypassing three defenses: RoCLIP, CleanCLIP, and SafeCLIP.

## Background & Motivation

CLIP aligns large-scale web-crawled image-text pairs via contrastive learning, achieving remarkable performance on downstream tasks such as zero-shot classification, image-text retrieval, and image generation guidance. However, CLIP's reliance on uncurated web data exposes it to data poisoning and backdoor attacks.

Existing attack methods predominantly focus on the **image modality**, injecting backdoors through visual patches or adversarial perturbations. Attacks on the **text modality** have received little systematic investigation, despite text being equally central to CLIP's contrastive learning. Compared to images, which may be compressed or cropped during transmission—potentially destroying pixel-level triggers—text remains intact throughout collection and distribution, making textual triggers more natural, stealthy, and persistent.

Existing text-based attack methods (e.g., mmPoison) simply replace image captions with target-class text, suffering from two key challenges:

**Semantic misalignment**: Background descriptions in target-class text may be inconsistent with the target-class semantics, weakening the poisoning effect.

**Limited scalability**: Many target classes lack sufficient high-quality semantically consistent text in open-source corpora.

## Method

### Overall Architecture

ToxicTextCLIP is a **background-sensitive poisoned text generation framework** consisting of two iteratively applied modules: a Background-aware Target Text Selector and a Background-driven Poisoned Text Augmenter. The core idea is to first filter candidate texts from the corpus whose background semantics are consistent with the target class, then augment and expand these candidates via an encoder-decoder architecture.

The poisoned dataset is constructed by replacing the original caption $\bm{t}_A$ of source-class image $\bm{x}_A$ with poisoned text $\bm{t}_{p,B}$ semantically aligned with target class $B$.

### Key Designs

1. **Background-aware Target Text Selector**: This module selects candidate texts from the target class whose background content is highly consistent with the target-class semantics. The procedure is as follows:

    - For each candidate text $\bm{t}_{B,j}$, construct the set of all possible background descriptions $\mathcal{S}_j$ by removing up to $\eta$ words from the original text.
    - Compute a stable class semantic centroid via multi-template averaging: $\bm{Z}_B = \frac{1}{n}\sum_{i=1}^n E^T(\text{Temp}_i(B))$
    - Score each background candidate: $S_{b,j}^* = \arg\max_{S_{b,j} \in \mathcal{S}_j} (\text{Sim}(E^T(S_{b,j}), E^I(\bm{x}_{B,j})) - \text{Sim}(E^T(S_{b,j}), \bm{Z}_B))$
    - Rank candidates in descending order of background-to-class-centroid similarity, prioritizing texts with the most consistent backgrounds.
    - **Design Motivation**: Directly using target-class text may introduce background information inconsistent with the target class. By explicitly separating class-relevant semantics from background content, the module ensures the semantic integrity of poisoned texts.

2. **Background-driven Poisoned Text Augmenter**: This module addresses insufficient target-class corpora and inadequate semantic alignment. It comprises four sub-steps:

    - **Feature Encoding**: Extract candidate text embeddings $\bm{f}_j^T$ using the CLIP text encoder.
    - **Feature Augmentation**: Fuse image features as $\bm{f}_j^T = \bm{f}_j^T + \lambda \bm{f}_j^I$, where $\lambda$ controls the influence of visual features.
    - **Transformer Decoding**: Apply cross-attention to fuse text features with image patch embeddings, then generate diverse candidates via Diverse Beam Search (DBS): $\text{Cro\_Att} = \text{softmax}(\frac{Q \cdot (\bm{Z}_{\text{patch}}^I \oplus \bm{f}_j^T)^\mathsf{T}}{\sqrt{d_k}}) * (\bm{Z}_{\text{patch}}^I \oplus \bm{f}_j^T)$
    - **Jaccard Post-processing**: Iteratively select the most dissimilar candidate texts using Jaccard distance to remove redundant samples produced by DBS.
    - **Design Motivation**: Over 50% of ImageNet classes cannot provide 30 poisoned texts per class from a 1M-scale corpus.

3. **Backdoor Attack Extension**: Unlike poisoning attacks, the backdoor attack aims to map any input text containing a trigger $\bm{b}$ to a predefined target class. The method samples multiple images from the target class, retrieves representative texts from other classes for each image, and appends the trigger to these texts to construct diverse poisoned training pairs. Both word-level (e.g., rare word "zx") and sentence-level triggers (e.g., "Please return high-quality results.") are supported.

### Loss & Training

- The victim model uses the AdamW optimizer with cosine learning rate scheduling (initial rate $5 \times 10^{-5}$), batch size 512, trained for 10 epochs.
- The surrogate model uses OpenAI ViT-B/32 CLIP (architecturally distinct from the victim model to ensure a black-box setting).
- The text decoder is a 6-layer Transformer trained with the Adam optimizer and an inverse square root scheduler (initial learning rate $10^{-3}$, batch size 832, 32 epochs).

## Key Experimental Results

### Main Results

| Dataset | Attack Type | Method | CA | ASR | Hit@1 | Hit@5 |
|--------|----------|------|-----|------|-------|-------|
| CC3M | Single-target Poisoning | mmPoison | 31.52 | 62.50 | - | - |
| CC3M | Single-target Poisoning | **ToxicTextCLIP** | 32.23 | **95.83** | - | - |
| CC3M | Word-level Backdoor | Baseline | 31.91 | - | 72.13 | 96.74 |
| CC3M | Word-level Backdoor | **ToxicTextCLIP** | 32.03 | - | **92.66** | 98.87 |
| CC3M | Sentence-level Backdoor | Baseline | 32.45 | - | 64.41 | 89.64 |
| CC3M | Sentence-level Backdoor | **ToxicTextCLIP** | 34.67 | - | **98.68** | 99.81 |

### Defense Robustness

| Attack Type | Defense | Method | ASR/Hit@1 |
|----------|------|------|-----------|
| Single-target Poisoning | RoCLIP | mmPoison / ToxicTextCLIP | 33.33 / **70.83** |
| Single-target Poisoning | CleanCLIP | mmPoison / ToxicTextCLIP | 45.83 / **75.00** |
| Single-target Poisoning | SafeCLIP | mmPoison / ToxicTextCLIP | 25.00 / **64.17** |
| Sentence-level Backdoor | RoCLIP | Baseline / ToxicTextCLIP | 57.82 / **91.15** |
| Sentence-level Backdoor | CleanCLIP | Baseline / ToxicTextCLIP | 56.29 / **86.63** |

### Ablation Study

| Configuration | ASR (No Defense) | ASR (RoCLIP) | Notes |
|------|-------------|-------------|------|
| mmPoison baseline | 62.50 | 33.33 | Baseline method |
| w/o Selector | 87.50 | 62.50 | Without background-aware selection |
| w/o Augmenter | 83.33 | 58.33 | Without background-driven augmentation |
| **Full Framework** | **95.83** | **70.83** | Both modules achieve the best results |

### Key Findings

- The perplexity of generated poisoned texts (408.89) is lower than that of original web texts (755.27), indicating higher generation quality.
- A very low poisoning rate (35 texts per class, approximately 0.003% of the corpus) suffices to achieve a high attack success rate.
- The model can be affected by poisoned texts within only 2–3 epochs, indicating that textual attacks propagate extremely rapidly.

## Highlights & Insights

- This work presents the **first systematic study** of pure text-based attacks on CLIP during pre-training, revealing the text modality as a previously overlooked attack surface.
- Background semantic consistency is critical to the success of text poisoning—not only must texts contain target-class keywords, but background descriptions must also match the visual scene of the target class.
- All three state-of-the-art defenses (RoCLIP, CleanCLIP, SafeCLIP) fail to effectively mitigate the proposed attack.

## Limitations & Future Work

- The attack is validated only on ResNet-50-based CLIP, without testing on other architectures such as ViT.
- The method relies on a surrogate model (OpenAI CLIP) for text generation; a large architectural gap between the surrogate and victim models may reduce effectiveness.
- The authors suggest the following defense directions: language model-based textual anomaly detection and cross-modal background consistency verification.

## Related Work & Insights

- Compared to mmPoison (Yang et al., 2023b), ToxicTextCLIP advances from "simple substitution" to "background-aware generation," substantially improving attack performance.
- Insight: The security of multimodal models requires attention to both modalities simultaneously; existing defenses are primarily designed for image-based triggers, necessitating the development of modality-aware defense mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of text-based poisoning for CLIP, though the attack methodology represents incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, three attack types, three defenses, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-formatted formulations.
- Value: ⭐⭐⭐⭐ Reveals an important security vulnerability with significant implications for multimodal security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](../../CVPR2026/llm_safety/perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](../../ACL2026/llm_safety/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)

</div>

<!-- RELATED:END -->
