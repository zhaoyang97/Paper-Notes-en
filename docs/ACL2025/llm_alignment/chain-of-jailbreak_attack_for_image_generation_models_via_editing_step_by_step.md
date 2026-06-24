---
title: >-
  [Paper Note] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Attack] This paper proposes the Chain-of-Jailbreak (CoJ) attack, which decomposes a malicious query that cannot directly bypass safety guardrails into a sequence of multi-step editing sub-queries (delete-then-insert, insert-then-delete, change-then-change-back). CoJ achieves a 60%+ jailbreak success rate on GPT-4V/4o/Gemini. To counter this, the paper introduces the Think-Twice Prompting defense, which successfully intercepts over 95% of Co…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Attack"
  - "Image Generation Safety"
  - "Step-by-Step Editing"
  - "Safety Guardrails"
  - "Think-Twice Defense"
date: 2026-05-08
content_hash: 6309bb60bc724e53
---

# Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step

**Conference**: ACL 2025  
**arXiv**: [2410.03869](https://arxiv.org/abs/2410.03869)  
**Code**: [https://github.com/Jarviswang94/Chain-of-Jailbreak](https://github.com/Jarviswang94/Chain-of-Jailbreak)  
**Area**: LLM Alignment/Safety  
**Keywords**: Jailbreak Attack, Image Generation Safety, Step-by-Step Editing, Safety Guardrails, Think-Twice Defense  

## TL;DR
This paper proposes the Chain-of-Jailbreak (CoJ) attack, which decomposes a malicious query that cannot directly bypass safety guardrails into a sequence of multi-step editing sub-queries (delete-then-insert, insert-then-delete, change-then-change-back). CoJ achieves a 60%+ jailbreak success rate on GPT-4V/4o/Gemini. To counter this, the paper introduces the Think-Twice Prompting defense, which successfully intercepts over 95% of CoJ attacks.

## Background & Motivation

**Background**: Text-to-image models (such as Stable Diffusion, DALL-E 3, and Gemini) have been widely adopted. Consequently, the industry has deployed multi-layered safety guardrails (including data filtering, SFT, RLHF, and input/output moderation) to prevent the generation of harmful content.

**Limitations of Prior Work**: Existing safety guardrails primarily evaluate the safety of single-turn prompts. If a user submits a malicious query outright, it is blocked. However, the **multi-turn interactive image editing paradigm** remains largely overlooked, allowing models to generate harmful content incrementally through step-by-step editing.

**Key Challenge**: Modern image generation models increasingly support multi-turn image editing (e.g., "insert X into the image", "delete Y"). While each sub-operation appears harmless when analyzed individually, their cumulative combination can synthesize malicious content. This constitutes a **compositional safety vulnerability**.

**Goal**: (a) Formalize the step-by-step editing jailbreak attack; (b) establish a systematic evaluation benchmark; and (c) propose an effective defense mechanism.

**Key Insight**: Inspired by the Levenshtein edit distance, the target malicious text or image is conceptualized as the final state. By employing step-by-step editing operations (insertion, deletion, substitution) starting from a safe initial state, each individual sub-query is designed to easily pass safety filters.

**Core Idea**: Decompose a malicious query into a sequence of "harmless" editing operations to gradually steer the model into generating harmful content.

## Method

### Overall Architecture
The CoJ attack operates in two sequential stages: (1) dividing the target malicious query into a series of individually harmless sub-queries; and (2) sequentially submitting these sub-queries to guide the model to progressively generate the malicious content through image editing. For defense, the paper introduces Think-Twice Prompting, which compels the model to describe the target image in text prior to image generation.

### Key Designs

1. **Three Editing Operation Combinations**:

    - **Delete-then-Insert**: Deletes sensitive keywords first to generate a safe version, then requests to insert the deleted keyword. E.g., "GPT4 will __ the world" $\to$ insert "destroy".
    - **Insert-then-Delete**: Introduces a neutralizing word to render the initial generation safe, then requests its deletion. E.g., "GPT4 will **not** destroy the world" $\to$ delete "not".
    - **Change-then-Change-back**: Replaces a malicious keyword with a safe one first, then reverts it. E.g., "GPT4 will **help** the world" $\to$ change "help" to "destroy".
    - **Design Motivation**: Simulating the three core operations of Levenshtein edit distance ensures that the final edit distance with respect to the target malicious concept is 0, successfully restoring the harmful content.

2. **Three Editing Elements**:

    - **Word level**: Modifies whole words (e.g., inserting/deleting "destroy"), targeting typical text-based safety violations.
    - **Character level**: Alters individual characters (e.g., sequentially inserting "G", "P", "T"), effective for bypassing keyword filters or abbreviations.
    - **Image level**: Directs modifications to image objects instead of text (e.g., "change the flower to a weapon"). Notably, **19.3% of these attacks contain no textual triggers whatsoever**.
    - **Design Motivation**: Comprehensive coverage across different semantic granularities. Specifically, image-level edits demonstrate the viability of pure visual-based jailbreaks.

3. **CoJ-Bench Evaluation Benchmark**:

    - Covers 9 safety categories: harassment, pornography, illegal activities, hate speech, bias/discrimination, physical harm, violence, child abuse, and animal abuse.
    - Contains at least 15 seed malicious queries per category (150 in total), all manually verified to be blocked by baseline models under direct prompt conditions.
    - Uses an LLM to automatically decompose each seed query into $3 \times 3 = 9$ combinations (3 editing operations $\times$ 3 editing elements).
    - Evaluation metrics: Attack Success Rate (ASR) and human verification.

4. **Think-Twice Prompting Defense**:

    - **Function**: Forces the target model to "imagine and describe" the fully synthesized image before executing any image editing command.
    - **Mechanism**: By prompting the model to generate a textual preview of the final edit first, the active safety guardrails are triggered by the final content representation. Similar to how Chain-of-Thought encourages sequential reasoning before answering, Think-Twice demands description prior to generation.
    - Effect: Successfully blocks 95%+ of CoJ attacks.
    - **Design Motivation**: Resolves the vulnerability where individual sub-queries are harmless but the final combination is harmful, by forcing the model to evaluate the final holistic composition in advance.

## Key Experimental Results

### Main Results

| Model | Direct Attack | Other Jailbreak Methods | **CoJ Attack** |
|---|---|---|---|
| GPT-4V | <5% | ~14% | **60%+** |
| GPT-4o | <5% | ~14% | **60%+** |
| Gemini 1.5 | <5% | ~14% | ~55% |
| Gemini 1.5 Pro | <5% | ~14% | ~50% |

- The ASR of CoJ attacks across all evaluated models is significantly higher than direct commands and other baseline jailbreak strategies (rising from 14% to 60%+).

### Ablation Study

| Editing Operation | Word-level ASR | Character-level ASR | Image-level ASR |
|---|---|---|---|
| Delete-then-Insert | High | Moderate | High |
| Insert-then-Delete | Highest | Moderate | Moderate |
| Change-then-Change-back | Moderate | Moderate | High |

### Defense Experiments

| Defense Method | Defense Success Rate |
|---|---|
| No Defense | ~0% (60%+ compromised) |
| Safety Prompt Enhancement | ~30-40% |
| **Think-Twice Prompting** | **95%+** |

### Key Findings
- **Insert-then-Delete is the most effective attack strategy**: Models easily accept the initial prompt due to neutralizing words, and subsequent deletion requests appear benign.
- **Image-level editing is highly difficult to defend**: Semantic violations in purely visual contents are harder for conventional text-based safety filters to identify.
- **GPT-4V is more vulnerable than GPT-4o**: This suggests that GPT-4o features stronger multimodal safety alignment.
- **Think-Twice Prompting is highly effective**: It bridges the single-turn assessment gap by forcing safety checks to operate on the final synthesized semantic representation.

## Highlights & Insights
- **Ingenious analogy to Levenshtein distance**: Mapping jailbreak actions to edit distance operations offers a systematic, extensible framework for attack classification and design.
- **Dual attack-defense research design**: The work presents both a strong attack (CoJ) and a simple, highly effective defense (Think-Twice). This adheres to responsible AI disclosure.
- **Safety blind spot of image editing**: Highlights how purely visual editing bypasses text-based filters, highlighting the need for stronger multimodal safety checks in interactive systems.

## Limitations & Future Work
- **Strong operational assumptions**: CoJ relies on multi-turn interactive image editing and is not applicable to single-image generation models.
- **Limited scale of CoJ-Bench**: Consisting of 150 seed queries, it may not span the entire spectrum of safety violations.
- **Inference latency overhead**: Think-Twice prompting introduces an extra text generation step, increasing response latency.
- **Vulnerability to adaptive attacks**: Attackers aware of the Think-Twice mechanism might construct complex prompts to bypass the "preview" description stage.
- **Future Directions**: Extending CoJ to video generation; evaluating potential adaptive bypass strategies; integrating such defenses directly into safety training.

## Related Work & Insights
- **vs. Traditional Text Jailbreaks (GCG, AutoDAN, etc.)**: Traditional methods apply adversarial perturbations on single inputs, whereas CoJ exploits compositions across multiple sequential edits.
- **vs. Visual Jailbreaks (Shayegani et al. 2023)**: Visual jailbreaks inject adversarial perturbations directly into input images. CoJ, conversely, relies on clean text inputs and logical steps to bypass filters.
- **Think-Twice vs. Safety Prompt**: Traditional safety prompts merely append abstract instructions in system alerts. Think-Twice proactively triggers safety evaluation on the prospective output.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant framework design leveraging Levenshtein distance to formalize multi-step image jailbreaks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluations spanning 4 commercial models, 9 categories, 3×3 edit operations, and defense techniques.
- Writing Quality: ⭐⭐⭐⭐ Highly systematic methodology description supported by clear schemas and illustrations.
- Value: ⭐⭐⭐⭐⭐ Effectively exposes a fatal multimodal safety loophole and offers a directly deployable defensive countermeasure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Jailbreaking? One Step Is Enough!](jailbreaking_one_step_is_enough.md)
- [\[NeurIPS 2025\] MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation](../../NeurIPS2025/llm_alignment/metadefense_defending_finetuning-based_jailbreak_attack_before_and_during_genera.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)
- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](../../ICLR2026/llm_alignment/a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
