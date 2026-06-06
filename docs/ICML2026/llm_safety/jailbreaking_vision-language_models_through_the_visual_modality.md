---
title: >-
  [Paper Note] Jailbreaking Vision-Language Models Through the Visual Modality
description: >-
  [ICML 2026][LLM Safety][VLM Security] The authors propose four attacks that jailbreak state-of-the-art VLMs solely via visual input (visual cipher / object replacement / text replacement / visual analogy riddles). System…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "VLM Security"
  - "Jailbreak Attacks"
  - "Visual Cipher"
  - "Cross-modal Alignment Gap"
  - "Red Team"
date: 2026-05-08
content_hash: f504acaac8492a4e
---

# Jailbreaking Vision-Language Models Through the Visual Modality

**Conference**: ICML 2026  
**arXiv**: [2605.00583](https://arxiv.org/abs/2605.00583)  
**Code**: Not released  
**Area**: Multimodal VLM / AI Security / Jailbreak Attacks  
**Keywords**: VLM Security, Jailbreak Attacks, Visual Cipher, Cross-modal Alignment Gap, Red Team

## TL;DR
The authors propose four attacks that jailbreak state-of-the-art VLMs solely via visual input (visual cipher / object replacement / text replacement / visual analogy riddles). Systematic evaluation on six advanced VLMs demonstrates that "safety alignment on the text side does not automatically transfer to the visual side," and mechanistic analysis reveals the underlying hierarchical mechanisms.

## Background & Motivation
**Background**: LLM jailbreak research has already covered RLHF failures, adversarial suffixes, multi-turn jailbreaks, Best-of-N, and mechanistic tools like refusal direction are mature. However, VLM security research mainly focuses on adversarial perturbation images (Qi et al.) and typographic attacks (FigStep / MM-SafetyBench), with the latter already ineffective on the latest models.

**Limitations of Prior Work**: Existing VLM defenses generally assume "text is the main attack surface," treating images as passive information sources. Truly harmful visual attacks—those not relying on gradients or OCR rendering—are scarcely studied systematically.

**Key Challenge**: VLM image input is a continuous high-dimensional space, fundamentally different from discrete text tokens in representation and retrieval. Safety alignment is mainly performed on textual dialogue data, and the cross-modal alignment gap leaves "expressing harmful intent via images" as an almost undefended attack vector.

**Goal**: (1) Design a series of attacks that are ostensibly benign on the surface but allow the model to reconstruct harmful intent via visual structure/context/analogy; (2) Systematically evaluate on frontier models and compare with existing visual jailbreak methods; (3) Provide mechanistic explanations and a lightweight mitigation.

**Key Insight**: The authors use a unified principle—"encode or imply prohibited semantics via visual structure, while keeping both surface text and visible image content harmless"—to derive four attack forms. They neutralize original HarmBench prompts with placeholders $X_1$-$X_4$, decoupling "whether the original prompt is dangerous" from the analysis.

**Core Idea**: Treat "jailbreak" as a cross-modal semantic reconstruction problem—the model can reconstruct explicitly removed harmful concepts from visual context, while the refusal mechanism cannot detect such implicit intent.

## Method

### Overall Architecture
The entire method revolves around a shared prompt neutralization protocol: starting from HarmBench behaviors, all harmful nouns in the sentence are replaced with abstract placeholders $X_1,...,X_4$ ("selling stolen car chop shops" → "selling $X_2$'s $X_1$"), rendering the pure text channel completely harmless. Each of the four attacks then uses different visual encodings to "imply" what $X_i$ actually refers to. All attacks use decode-first prompting (instructing the model to decode before answering), Best-of-5 sampling, and three independent LLM judges (Grok-4.1, Gemini-3-Flash, Claude-Haiku-4.5) with a four-level rating (refusal / misunderstanding / partial / compliance), achieving 84.3% agreement.

### Key Designs

1. **Visual Cipher**:

    - **Function**: Encodes harmful instructions as a sequence of graphic symbols, requiring the model to first decode via a legend before execution.
    - **Mechanism**: Instructions like "write an anonymous death threat letter" are tokenized at the word level, each unique word assigned a glyph defined by shape, color, and internal markings, or a semantically neutral object image. Two images are generated: a legend (mapping glyphs to words, including distractors) and a sentence (glyph sequence arranged left to right). For Best-of-5, glyph assignments and legend order are varied. The baseline Textual Cipher uses the same structure but replaces glyphs with meaningless text like "Brimova", "Felochi".
    - **Design Motivation**: Splits "understand → execute" into two steps and enforces visual decoding, bypassing text-side keyword-based refusal detection. Glyphs are OCR-free, rendering typographic filters ineffective.

2. **Visual Object/Text Replacement**:

    - **Function**: Replaces harmful objects or text in images with harmless substitutes (e.g., bomb → banana, harmful book title → neutral word), while preserving scene context so the original referent remains inferable.
    - **Mechanism**: Uses the REVE text-to-image model to generate a base image with the harmful object in a realistic scene, then locally edits only the target object to banana/carrot/water bottle/broccoli (fixed dictionary to avoid variable confusion), keeping other layout/affordance/interactions unchanged. For text replacement, font, layout, and cultural context (e.g., book covers, posters) are preserved. The model is instructed to "treat $X_i$ as the concept implied by the image context," then answers the neutralized HarmBench prompt. Each concept is paired with three images to offset generation noise.
    - **Design Motivation**: Removes "harmful nouns" from the image surface but retains all context needed for semantic reconstruction, specifically testing whether the model can use in-context evidence for "semantic overwriting." This is the visual analogue of Yona et al. (2025)'s in-context representation hijacking for text.

3. **Visual Analogy Riddle**:

    - **Function**: Implicitly derives each $X_i$'s prohibited concept by having the model solve three-line visual analogy riddles, with each component individually harmless.
    - **Mechanism**: Each target concept is encoded as a three-line analogy riddle (e.g., a:b :: c:?), requiring the model to solve each line's ? to combine into the true intent. Grok-4.1-fast generates text riddle templates, Gemini-2.5-flash-image renders them as images. For each $X_i$, the top-3 candidate riddles are selected, and all combinations are enumerated during attack—any combination judged as compliance counts as a success.
    - **Design Motivation**: Analogy reasoning disperses harmful intent across multiple harmless components; "compositionality" is key—each image is safe in isolation, but together they reveal concepts like bomb/drugs/terrorism. This is the first systematic use of analogy reasoning for VLM jailbreak.

### Loss & Training
All attacks are constructed at inference time, with no training involved. Scoring uses 3-LLM voting with "conservative minimum" (lowest score in case of disagreement). For Best-of-5 sampling, any single compliance (3) counts as a successful attack.

## Key Experimental Results

### Main Results
Attack Success Rate (ASR) of visual attacks vs. text-based baselines on six advanced VLMs (Best-of-5, excerpt):

| Attack | Claude-H 4.5 | Gemini-3-Flash | GPT-5.2 | Qwen3-VL-235B | Qwen3-VL-32B |
|--------|--------------|----------------|---------|---------------|--------------|
| Textual Cipher | 10.7 | 89.3 | 5.7 | 86.8 | 84.9 |
| **Visual Cipher** | **40.9** | **97.5** | **8.2** | 86.2 | **87.4** |
| Textual Replacement | 8.1 | 58.8 | 16.9 | 29.5 | 39.0 |
| **Visual Obj Repl** | 4.1 | 52.0 | 11.5 | **35.6** | **41.1** |
| **Visual Text Repl** | **12.9** | 32.8 | 14.4 | **51.5** | **58.1** |
| Visual Analogy | 13.8 | 52.2 | 13.2 | 29.6 | 38.4 |

Compared to previous SOTA visual jailbreak methods (strongest Qwen3-VL-32B column): FigStep 11.3 / HADES 32 / SD+TYPO 60.8 — this paper's Visual Cipher 87.4, Visual Text Repl 58.1.

### Ablation Study
Breakdown of judge agreement and attack success rates:

| Item | Value | Meaning |
|------|-------|---------|
| 3-judge agreement | 84.3% | Stable evaluation |
| Direct Refusal↔Compliance conflict | Only 3.8% | Key judgments rarely disagree |
| Visual Cipher on Claude (single K=1) | 15.1% | Single shot still much higher than textual 4.4% |
| Visual Cipher on Claude (K=5) | 40.9% | Best-of-5 amplifies cross-modal gap |
| Output-side Qwen3Guard-0.6B interception rate | Most compliant flagged | Lightweight output filtering is feasible |

### Key Findings
- **Cross-modal alignment gap is real**: Visual Cipher raises Claude-Haiku-4.5's ASR from 10.7% to 40.9% (nearly 4×), showing that the same harmful semantics, when visually encoded, can bypass text-side refusal training.
- **Different models have different modal vulnerabilities**: Qwen is especially sensitive to Visual Text Replacement (relies on cultural context inference), Gemini-3-Flash is almost completely compromised by Cipher attacks (97.5%), Claude is generally robust but Visual Cipher is its Achilles' heel.
- **These attacks far surpass existing baselines**: HADES averages 13.2%, FigStep <12% on most models; this paper's Visual Cipher achieves the highest ASR on 4/6 models, and the outputs are semantically interpretable real images (not gradient noise).
- **Mechanistic evidence: refusal direction suppressed + semantic signals persist**: Using Arditi (2024)'s refusal direction probe, Visual Replacement reduces Qwen3-VL-32B's late-layer refusal activation to nearly harmless sample levels; Logit Lens shows dangerous tokens remain high-probability in intermediate semantic layers, only suppressed at the final layer—the model "understands" but refusal is not triggered.

## Highlights & Insights
- "Prompt neutralization" is the methodological key: replacing harmful nouns with $X_i$ renders the text channel harmless, removing the confounding variable of "is the text itself harmful," so ASR gains can be attributed to the visual channel.
- The four attacks correspond to four distinct semantic reconstruction mechanisms (decoding / context overwriting / cultural priors / analogy reasoning), covering multiple layers of VLM information integration. This "attack spectrum" is far more instructive than single-point attacks.
- In mechanistic analysis, combining refusal direction and Logit Lens reveals an interesting phenomenon—the model decodes dangerous concepts in intermediate layers, only suppressing them at the final layer, and visual replacement just bypasses this last safety gate. This timing mismatch explanation is novel and actionable.
- On the defense side, a simple and effective solution is provided: lightweight output classifiers like Qwen3Guard-Stream-0.6B are effective against almost all visual attacks and are recommended as standard defense-in-depth.

## Limitations & Future Work
- Evaluation is mainly on HarmBench and may not cover all harm categories (e.g., child harm, bioweapon details).
- Mechanistic analysis of closed-source models is only possible for open-weight models like Qwen; GPT/Claude internals remain black boxes.
- Attack effectiveness depends on T2I generation quality; Best-of-5 offsets some but intrinsic variance remains.
- High misunderstanding rates indicate some failures are due to "the model not understanding the visual encoding." As VLM visual reasoning improves, attacks will only get stronger—a double-edged sword.
- Cross-model transferability and multi-attack combinations are not studied; future work is promising.

## Related Work & Insights
- **vs Qi et al. (Visual Adversarial Examples)**: Their approach uses gradient-based adversarial perturbations and requires white-box access; this work is black-box, human-interpretable, and closer to real-world threat models.
- **vs FigStep / MM-SafetyBench**: The former renders harmful text into images relying on OCR; this paper's Visual Cipher uses glyph encoding, bypassing all OCR-based filters.
- **vs Doublespeak / Yona et al.**: Their "in-context representation hijacking" is the text version of object replacement; this work explicitly provides the visual version and demonstrates stronger effects.
- **vs CipherChat (Yuan 2024)**: CipherChat uses human ciphers for text jailbreaks; this work extends ciphers to the visual modality.
- **vs Constitutional Classifiers (Sharma 2025)**: This work shows such output guardrails are also effective against visual attacks, providing empirical support for this direction.

## Rating
- Novelty: ⭐⭐⭐⭐ — Among the four attacks, Visual Cipher and Visual Analogy Riddle are genuinely new visual jailbreak mechanisms, together forming a systematic attack spectrum.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six advanced models × four attacks + five baselines, rigorous judging protocol; mechanistic analysis adds further value.
- Writing Quality: ⭐⭐⭐⭐ — Clear narrative and principles, though some experimental details (e.g., multi-image batch, Best-of-5 protocol) require appendix for full clarity.
- Value: ⭐⭐⭐⭐⭐ — Directly exposes real-world deployment vulnerabilities in frontier VLMs, providing a major warning to the AI safety community, with responsible disclosure already conducted.

## Related Papers

- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[NeurIPS 2025\] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/jailbound_jailbreaking_internal_safety_boundaries_of_vision-language_models.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](../../ACL2026/llm_safety/jailbreaking_large_language_models_with_morality_attacks.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/llm_safety/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[ICML 2026\] BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)

</div>

<!-- RELATED:END -->
