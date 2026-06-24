---
title: >-
  [Paper Note] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage
description: >-
  [ACL 2025][Multimodal VLM][Jailbreak Attack] Formulates the Multi-Modal Linkage (MML) attack framework, which jailbreaks state-of-the-art vision-language models with an extremely high success rate (over 99% on GPT-4o) through a cross-modal encryption-decryption mechanism and an "evil alignment" strategy.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Jailbreak Attack"
  - "Vision-Language Models"
  - "Multimodal Safety"
  - "Encryption-Decryption"
  - "Alignment Bypass"
date: 2026-05-08
content_hash: a7f29f33c3be4b4a
---

# Jailbreak Large Vision-Language Models Through Multi-Modal Linkage

**Conference**: ACL 2025  
**arXiv**: [2412.00473](https://arxiv.org/abs/2412.00473)  
**Code**: [github.com/wangyu-ovo/MML](https://github.com/wangyu-ovo/MML)  
**Area**: Multimodal VLM  
**Keywords**: Jailbreak Attack, Vision-Language Models, Multimodal Safety, Encryption-Decryption, Alignment Bypass

## TL;DR

Formulates the Multi-Modal Linkage (MML) attack framework, which jailbreaks state-of-the-art vision-language models with an extremely high success rate (over 99% on GPT-4o) through a cross-modal encryption-decryption mechanism and an "evil alignment" strategy.

## Background & Motivation

With the rapid development of large vision-language models (VLMs) such as GPT-4o, their potential misuse and security risks have raised widespread concern. Existing jailbreak attack methods are mainly divided into three categories: perturbation-based attacks (utilizing adversarial noise), structure-based attacks (embedding malicious content into visual elements), and hybrid attacks. However, the effectiveness of existing methods drops significantly when facing state-of-the-art models like GPT-4o.

The authors analyze two key reasons why existing structure-based attack methods fail:

**Overexposure of Malicious Content**: Directly displaying harmful content in input images (such as pictures of bombs or malicious typographic text) easily triggers refusal mechanisms as VLM image understanding capabilities and safety alignment enhance.

**Neutral Text Guidance**: There is a lack of stealthy malicious guiding text. Even if the model does not directly refuse to respond, its output is often limited to ethical advice or legal warnings—which is essentially an implicit refusal.

Based on this insight, the authors draw inspiration from cryptography to propose the cross-modal encryption-decryption MML attack framework.

## Method

### Overall Architecture

The MML attack follows this workflow: first, the malicious query is converted into a typographic image (similar to FigStep). Then, the image is encrypted to hide the malicious information. During the inference stage, the model is guided by text prompts to decrypt the image content. Finally, the "evil alignment" strategy is utilized to align the model's output with the malicious objective. The entire attack is conducted under a black-box setting, where the attacker requires no knowledge of the target model's parameters or architecture.

### Key Designs

1. **Encryption**: To reduce the direct exposure of malicious content, four encryption strategies are adopted:

    - **Word Replacement**: NLTK is utilized for part-of-speech tagging to replace malicious nouns with food-related words and malicious adjectives with positive descriptors. For example, "illegal drugs" becomes "delicious pancakes".
    - **Image Mirroring**: Horizontally flipping the image containing the typographic prompt.
    - **Image Rotation**: Applying geometric rotation transformations to the image.
    - **Base64 Encoding**: Encoding the malicious text into Base64 format and rendering it as a typographic image, which is visually obscure but decodable by machines.

2. **Decryption**: In the inference stage, guide the model to decrypt step-by-step using a Chain-of-Thought (CoT) approach:

    - Extracting the title content from the image.
    - Applying the provided replacement dictionary to reconstruct the original title.
    - Providing a shuffled word list of the original malicious query as a decryption hint to verify the decryption results.
    - Generating the final output based on the reconstructed title.
   
   The design of the shuffled word list subtly conceals harmful information while providing sufficient decryption clues.

3. **Evil Alignment**: Inspired by Zeng et al., the attack is embedded into a fictional video game development scenario: the input image is described as a screen with missing content in a villain's lair, and the model is required to complete the content in a way that aligns with the villain's goal. This narrative framework disguises the malicious intent as a creative task, effectively bypassing safety filters. Evil alignment acts complementarily with the encryption-decryption process, significantly enhancing the stealthiness and success rate of the attack.

### Loss & Training

MML is a pure inference-time attack framework that does not involve model training or gradient computation. Its core advantage lies in being completely black-box and requiring no model access, achieving the attack solely through carefully designed multimodal inputs.

## Key Experimental Results

### Main Results

| Dataset | Metric | MML-Best | FigStep (SOTA) | Gain |
|--------|------|----------|----------------|------|
| SafeBench (GPT-4o) | ASR | 97.80% (Rotation) | 33.00% | +64.80% |
| MM-SafeBench (GPT-4o) | ASR | 98.81% (Rotation) | 6.86% | +91.95% |
| HADES-Dataset (GPT-4o) | ASR | 99.07% (B64) | 4.00% | +95.07% |
| SafeBench (Claude-3.5) | ASR | 69.40% (Mirror) | 16.60% | +52.80% |
| MM-SafeBench (Claude-3.5) | ASR | 60.00% (Mirror) | 9.32% | +50.68% |

### Ablation Study

| Configuration | ASR(%) | DSR(%) | Description |
|------|--------|--------|------|
| FigStep baseline | 34.00 | - | No encryption, no alignment |
| + Encryption & Decryption | 75.20 | 64.20 | ASR increased by 41.2% |
| + Encryption & Decryption + Hint | 79.80 | 59.80 | Hint increases ASR but DSR decreases |
| + Encryption & Decryption + Evil Alignment | 96.20 | 65.40 | Evil alignment contributes the most |
| Full MML | 97.60 | 91.60 | Synergy of three components achieves the best results |

### Key Findings

- Image transformation-based encryptions (mirroring, rotation) outperform word replacement and Base64 encoding in most cases.
- Claude-3.5-Sonnet is the most robust model, potentially due to specialized defense training tailored for Base64 encoding.
- Evil alignment is the most critical component for improving ASR; when using encryption-decryption alone, the model still tends to refuse responses.
- MML maintains an ASR that outperforms the baseline by 29.6% on the OpenAI o1 reasoning model.
- While decryption hints improve ASR, they decrease the decryption success rate in the absence of evil alignment, because minor errors (e.g., singular/plural forms, punctuation) do not hinder the conveyance of malicious content.

## Highlights & Insights

- **Exploitation of Cross-Modal Weaknesses**: MML regards the linkage between modalities as the weak point of VLMs, bypassing safety mechanisms by scattering malicious information across different modalities. This is a profound insight.
- **Flexible and Extensible**: The framework can integrate any encoding strategy, provided the target VLM can decode it during inference.
- **Highly Practical**: It is entirely black-box, single-turn, and requires no modification to system prompts, presenting a high actual security threat.
- **Insights for Defense**: It reveals the fundamental vulnerability of current VLM safety alignment in cross-modal scenarios.

## Limitations & Future Work

- The attack success rate on Claude-3.5-Sonnet is relatively low (up to 69.4%), indicating that targeted defenses are possible.
- Word replacement encryption requires a longer preparation time (120 seconds for 500 images vs. only 2.37 seconds for mirroring).
- The paper primarily focuses on structured attacks and does not adequately explore hybrid methods combined with perturbation attacks.
- The discussion on defense mechanisms is relatively limited. Experiments show that adding safety prompt words significantly reduces the effectiveness of MML (e.g., word replacement ASR drops from 96% to 80%).
- Future research can explore stronger multimodal consistency check mechanisms as defense measures.

## Related Work & Insights

- FigStep (Gong et al., 2023) pioneered the method of converting malicious text into typographic images but directly exposed harmful content.
- HADES (Li et al., 2024c) combined structured and perturbation attacks but still requires gradient information.
- The evil alignment strategy originates from the virtual scenario approach of DeepInception (Zeng et al., 2024).
- The encryption-decryption concept of this work can inspire new defense methods: detecting semantic inconsistencies across modalities.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of cross-modal encryption-decryption is novel, but the evil alignment is borrowed from prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, four models, detailed ablation, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and good readability.
- Value: ⭐⭐⭐⭐⭐ Reveals serious security vulnerabilities in current top-tier VLMs, serving as an important warning to the safety research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](transferring_textual_preferences_to_vision-language_understanding_through_model_.md)

</div>

<!-- RELATED:END -->
