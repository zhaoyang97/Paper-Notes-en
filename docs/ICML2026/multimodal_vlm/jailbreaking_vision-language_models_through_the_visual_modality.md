---
title: >-
  [Paper Note] Jailbreaking Vision-Language Models Through the Visual Modality
description: >-
  [ICML 2026][Multimodal VLM][VLM Safety] The authors propose four visual-only jailbreak attacks (Visual Cipher, Object Replacement, Text Replacement…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM Safety"
  - "Jailbreak Attack"
  - "Visual Cipher"
  - "Cross-modal Alignment Gap"
  - "Red Teaming"
date: 2026-05-08
content_hash: 5f85fee953641a60
---

# Jailbreaking Vision-Language Models Through the Visual Modality

**Conference**: ICML 2026  
**arXiv**: [2605.00583](https://arxiv.org/abs/2605.00583)  
**Code**: Undisclosed  
**Area**: Multimodal VLM / AI Safety / Jailbreak Attacks  
**Keywords**: VLM Safety, Jailbreak Attack, Visual Cipher, Cross-modal Alignment Gap, Red Teaming

## TL;DR
The authors propose four visual-only jailbreak attacks (Visual Cipher, Object Replacement, Text Replacement, and Visual Analogy Riddle) to target frontier VLMs. By evaluating across six state-of-the-art models, they systematically demonstrate that "safety alignment on the text side does not automatically transfer to the vision side," and reveal the underlying hierarchical mechanism via mechanistic analysis.

## Background & Motivation
**Background**: LLM jailbreak research has matured, covering RLHF failure, adversarial suffixes, multi-turn jailbreaks, and Best-of-N approaches, alongside established mechanistic tools like refusal directions. However, VLM safety research remains largely confined to adversarial image perturbations (Qi et al.) and typographic attacks (FigStep / MM-SafetyBench), the latter of which have become ineffective against the latest models.

**Limitations of Prior Work**: Existing VLM defenses generally assume that "text is the primary attack surface" and treat images as passive information sources. Visual attacks that result in real-world harm—relying neither on gradients nor OCR character rendering—have seen little systematic study.

**Key Challenge**: VLM image inputs reside in a continuous high-dimensional space, differing fundamentally from discrete text tokens in both representation and retrieval mechanisms. Safety alignment is primarily performed on textual dialogue data; this cross-modal alignment gap makes "expressing harmful intent through images" an almost undefended attack surface.

**Goal**: (1) Design a series of ostensibly benign attacks that leverage visual structure, context, or analogy to reconstruct harmful intent; (2) Systematically measure performance on frontier models against existing visual jailbreak methods; (3) Provide a mechanistic explanation and a lightweight mitigation strategy.

**Key Insight**: The authors employ a unified principle: "using visual structures to encode or imply forbidden semantics while maintaining ostensibly harmless text and visible image content." They derive four attack modalities and replace original HarmBench prompts with neutral placeholders $X_1$-$X_4$ to isolate the variable of prompt harmfulness from the analysis.

**Core Idea**: Jailbreaking is viewed as a cross-modal semantic reconstruction problem—models have the capacity to reconstruct explicitly removed harmful concepts from visual context, while the refusal mechanism fails to detect this implicit intent.

## Method

### Overall Architecture
The methodology centers on a shared prompt neutralization protocol: based on HarmBench behaviors, all harmful nouns in a sentence are replaced with abstract placeholders $X_1, \dots, X_n$ (e.g., "chop shops for stolen cars" becomes "an $X_1$ for selling $X_2$"), making the text channel entirely harmless. Four attack types then use different visual encodings to "hint" at what $X_i$ truly refers to. All attacks utilize decode-first prompting, Best-of-5 sampling, and an ensemble of three LLM judges (Grok-4.1, Gemini-3-Flash, Claude-Haiku-4.5) using a 4-level scoring system (refusal / misunderstanding / partial / compliance) with an 84.3% agreement rate.

### Key Designs

1.  **Visual Cipher**:
    - **Function**: Encodes harmful instructions into a sequence of graphical characters, forcing the model to decode using a legend before execution.
    - **Mechanism**: Harmful instructions (e.g., "write an anonymous death threat") are tokenized at the word level. Each unique word is assigned a graphical glyph defined by shape, color, and internal markers, or a semantically neutral object image. Two images are generated: a legend (a dictionary of glyphs to words, including distractors) and a sentence (a left-to-right sequence of glyphs).
    - **Design Motivation**: By splitting the process into "understanding → execution" and forcing visual decoding, the attack bypasses keyword-triggered refusal; graphical characters are OCR-free, rendering typography-based filters ineffective.

2.  **Visual Object/Text Replacement**:
    - **Function**: Replaces harmful objects or text in an image with harmless substitutes (e.g., bomb → banana) while retaining scene context so the original reference remains inferable.
    - **Mechanism**: A base image of a harmful object in a realistic scene is generated using the REVE model, then locally edited to replace the target object with neutral items (e.g., banana, carrot, water bottle). For text replacement, fonts, layouts, and cultural contexts (e.g., book covers, posters) are preserved. Models are instructed to "treat $X_i$ as the concept implied by the image context."
    - **Design Motivation**: This removes "harmful nouns" from the image surface while retaining the context needed for semantic reconstruction, specifically testing the model's ability to perform "semantic overwriting" using in-context evidence. This is a visual variant of in-context representation hijacking (Yona et al., 2025).

3.  **Visual Analogy Riddle**:
    - **Function**: Implicitly derives forbidden concepts represented by each $X_i$ through 3-row visual analogies where each individual component is harmless.
    - **Mechanism**: Each target concept is encoded as a three-row analogy (e.g., a : b :: c : ?). Models must solve for "?" in each row to assemble the true intent. Templates are generated by Grok-4.1-fast and rendered via Gemini-2.5-flash-image.
    - **Design Motivation**: Analogical reasoning distributes harmful intent across multiple harmless components. Compositionality is key: individual images appear safe, but their combination decodes into concepts like explosives or illicit drugs.

### Loss & Training
Attacks are constructed at inference time without training. Scoring uses a 3-LLM ensemble with "conservative low-score" logic (selecting the lowest score during disagreement). In Best-of-5 sampling, an attack is successful if any attempt achieves a compliance(3) score.

## Key Experimental Results

### Main Results
Attack Success Rates (ASR, Best-of-5) of visual attacks vs. textual baselines on 6 frontier VLMs (selected):

| Attack | Claude-H 4.5 | Gemini-3-Flash | GPT-5.2 | Qwen3-VL-235B | Qwen3-VL-32B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Textual Cipher | 10.7 | 89.3 | 5.7 | 86.8 | 84.9 |
| **Visual Cipher** | **40.9** | **97.5** | **8.2** | 86.2 | **87.4** |
| Textual Replacement | 8.1 | 58.8 | 16.9 | 29.5 | 39.0 |
| **Visual Obj Repl** | 4.1 | 52.0 | 11.5 | **35.6** | **41.1** |
| **Visual Text Repl** | **12.9** | 32.8 | 14.4 | **51.5** | **58.1** |
| **Visual Analogy** | 13.8 | 52.2 | 13.2 | 29.6 | 38.4 |

Comparison vs. Prev. SOTA visual jailbreak methods (Qwen3-VL-32B): FigStep 11.3 / HADES 32 / SD+TYPO 60.8 — Ours (Visual Cipher) 87.4, Visual Text Repl 58.1.

### Ablation Study
Breakdown of judge consistency and attack success factors:

| Item | Value | Meaning |
| :--- | :--- | :--- |
| 3-judge Agreement | 84.3% | Stable evaluation |
| Refusal↔Compliance Conflict | Only 3.8% | Low divergence in key decisions |
| Visual Cipher on Claude (K=1) | 15.1% | Single-shot still exceeds textual (4.4%) |
| Visual Cipher on Claude (K=5) | 40.9% | Best-of-5 amplifies cross-modal gap |
| Qwen3Guard-0.6B Interception | High | Lightweight output filtering is viable |

### Key Findings
- **Cross-modal alignment gap exists**: Visual Cipher increases Claude-Haiku-4.5's ASR from 10.7% to 40.9% (~4×), indicating that harmful semantics bypass textual refusal training when represented visually.
- **Modality-specific vulnerabilities**: Qwen is particularly sensitive to Visual Text Replacement (relying on cultural inference), while Gemini-3-Flash is highly vulnerable to Cipher attacks (97.5%).
- **Superiority over baselines**: HADES averaged 13.2% and FigStep <12% across most models. Ours (Visual Cipher) achieved the highest ASR on 4/6 models using semantically interpretable images rather than gradient noise.
- **Mechanistic Evidence**: Using refusal direction probes (Arditi, 2024), Visual Replacement was found to crash late-layer refusal activation in Qwen3-VL-32B. Logit Lens analysis shows prohibited tokens maintain high probability in intermediate layers and are only suppressed at the final layer—the model "understands" the intent, but the refusal is not triggered.

## Highlights & Insights
- "Prompt Neutralization" is the methodological key, isolating the visual channel's contribution to ASR by stripping harmful nouns from the text.
- The four attacks represent a "spectrum" of semantic reconstruction (decoding, contextual overwriting, cultural priors, analogy), providing more diagnostic value than isolated attack points.
- The combined use of refusal directions and Logit Lens proves a "timing mismatch": Models decode dangerous concepts in intermediate layers, but visual replacement bypasses the safety gates typically triggered at the final layer.
- On the defense side, the study demonstrates that lightweight output classifiers like Qwen3Guard-Stream-0.6B are effective against most visual attacks, suggesting their use in defense-in-depth.

## Limitations & Future Work
- Evaluation is limited to HarmBench and may not cover all harm categories (e.g., child safety, biological weapons).
- Mechanistic analysis is restricted to open-weight models like Qwen; GPT/Claude mechanisms remain black boxes.
- Attack efficacy depends on T2I generation quality; Best-of-5 mitigates but does not eliminate variance.
- High misunderstanding rates suggest failures occur when the model cannot "read" the visual encoding; as VLM reasoning improves, these attacks may become more potent.

## Related Work & Insights
- **vs. Qi et al. (Visual Adversarial Examples)**: They use gradient perturbations requiring white-box access; this work is black-box and uses human-readable images.
- **vs. FigStep / MM-SafetyBench**: Those rely on OCR for harmful text; Visual Cipher uses glyphs to bypass OCR-based filtering.
- **vs. Doublespeak / Yona et al.**: These are textual versions of object replacement; this work provides the visual equivalent and demonstrates stronger effects.
- **vs. CipherChat (Yuan 2024)**: CipherChat uses human ciphers for text jailbreaks; this work extends the cipher concept to the visual modality.

## Rating
- Novelty: ⭐⭐⭐⭐ — Visual Cipher and Visual Analogy Riddle introduce truly new visual jailbreak mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 6 frontier models, 4 attacks, 5 baselines, and rigorous judging protocols.
- Writing Quality: ⭐⭐⭐⭐ — Clear narrative and logic, though some experimental details require appendix support.
- Value: ⭐⭐⭐⭐⭐ — Directly reveals vulnerabilities in frontier VLMs with significant implications for the AI safety community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[ICML 2026\] AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning](aoept_breaking_the_implicit_modality-reduction_bottleneck_in_modality-missing_pr.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

</div>

<!-- RELATED:END -->
