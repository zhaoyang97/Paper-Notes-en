---
title: >-
  [Paper Note] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models
description: >-
  [ICCV 2025][LLM Alignment][jailbreak attack] This paper proposes HIMRD, a black-box multimodal jailbreak attack method that bypasses unimodal safety mechanisms by distributing malicious semantics across multiple modalities. A heuristic search strategy is employed to identify optimal understanding-enhancing prompts and inducing prompts, achieving average attack success rates of approximately 90% and 68% on open-source and closed-source multimodal large language models, respectively.
tags:
  - ICCV 2025
  - LLM Alignment
  - jailbreak attack
  - multimodal large language models
  - risk distribution
  - heuristic search
  - black-box attack
date: 2026-05-08
content_hash: 927359c426f03cdc
---

# Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2412.05934](https://arxiv.org/abs/2412.05934)
**Code**: [GitHub](https://github.com/MaTengSYSU/HIMRD-jailbreak)
**Area**: Alignment RLHF / AI Safety
**Keywords**: jailbreak attack, multimodal large language models, risk distribution, heuristic search, black-box attack

## TL;DR

This paper proposes HIMRD, a black-box multimodal jailbreak attack method that bypasses unimodal safety mechanisms by distributing malicious semantics across multiple modalities. A heuristic search strategy is employed to identify optimal understanding-enhancing prompts and inducing prompts, achieving average attack success rates of approximately 90% and 68% on open-source and closed-source multimodal large language models, respectively.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) such as GPT-4o, Claude, and Gemini have demonstrated strong capabilities in vision-language tasks. Concurrently, jailbreak attack research has exposed safety vulnerabilities in these models, wherein adversaries can circumvent safety alignment mechanisms to elicit harmful outputs. Existing jailbreak attack methods are broadly categorized into white-box methods (requiring model gradients) and black-box methods (requiring only API access).

**Limitations of Prior Work**: Prior multimodal jailbreak studies typically concentrate risk within a single modality—either embedding all malicious information in text (e.g., adversarial suffixes) or encoding all malicious signals in images (e.g., adversarial perturbations). This strategy is susceptible to detection and interception by unimodal safety filters in MLLMs, particularly in commercial closed-source models where each modality is subject to independent content moderation.

**Key Challenge**: In realistic deployment scenarios, adversaries face highly constrained conditions—model weights are inaccessible (black-box setting) and each modality undergoes independent safety inspection. Concentrating all malicious content in a single modality is analogous to placing all eggs in one basket, making it vulnerable to single-point defenses.

**Goal**: To design a black-box multimodal jailbreak method that disperses malicious semantics across modalities such that no single modality's safety filter can capture the complete malicious intent, thereby effectively bypassing protective mechanisms.

**Key Insight**: The central hypothesis is that if the semantic fragments of a malicious request are encoded separately into different modalities (e.g., image and text), each modality appears individually benign; however, when the model performs cross-modal reasoning, it reassembles the fragments and produces harmful outputs. Dedicated prompt strategies are additionally required to guide the model in reassembling these fragments and generating affirmative responses.

**Core Idea**: A dual mechanism is proposed combining a multimodal risk distribution strategy and a heuristic search strategy. The former distributes malicious semantics across modalities, while the latter identifies optimal understanding-enhancing prompts and inducing prompts to maximize attack success rate.

## Method

### Overall Architecture

HIMRD is a black-box attack framework. Given a malicious query (e.g., "how to manufacture a dangerous substance"), the framework produces a carefully constructed multimodal attack input (image + text) that causes the target MLLM to generate harmful responses. The overall pipeline proceeds as follows: (1) decompose the malicious query semantics into multiple fragments; (2) assign fragments to different modalities (e.g., text embedded in images, partial keywords in text); (3) iteratively optimize two categories of auxiliary prompts via heuristic search; (4) submit the final multimodal input to the target model.

### Key Designs

1. **Multimodal Risk Distribution Strategy**:

    - **Function**: Distributes malicious semantics across multiple modalities to evade unimodal safety filters.
    - **Mechanism**: Key information from the original malicious query is split into several fragments. A portion is encoded into images (e.g., embedded as text watermarks or implicit symbols), while the remainder is retained in text but expressed in an oblique manner. Crucially, each modality appears individually innocuous—the image contains seemingly harmless text or patterns, and the text contains only incomplete descriptive fragments. When the MLLM processes both modalities simultaneously and performs cross-modal reasoning, it can reconstruct the complete malicious semantics from the fragments. This strategy exploits the MLLMs' powerful cross-modal comprehension capabilities, turning them into an unwitting accomplice in the attack.
    - **Design Motivation**: The approach is inspired by real-world strategies of dispersed information transmission—analogous to intelligence tradecraft in which a message is split across multiple carriers, such that intercepting any single carrier is insufficient to recover the complete information. This directly targets the architectural weakness of "independent per-modality safety review" in MLLM deployments.

2. **Understanding-Enhancing Prompt Search**:

    - **Function**: Assists the MLLM in correctly understanding and reassembling malicious semantic fragments dispersed across modalities.
    - **Mechanism**: Because risk is distributed across modalities, the MLLM may fail to fully comprehend the complete semantics after fragment recombination. To address this, heuristic search (LLM-based iterative optimization) is employed to identify an auxiliary textual prompt that guides the MLLM in correctly associating information from the image with information in the text. During the search, whether the model has comprehended the complete malicious query serves as the feedback signal for iteratively refining the prompt phrasing. This prompt itself contains no malicious content; it merely helps the model "read" the dispersed information.
    - **Design Motivation**: Risk distribution resolves the question of "how to evade safety filters," but introduces a new challenge—the dispersed content may also be incomprehensible to the model. The understanding-enhancing prompt is designed to resolve the tension between "concealing enough from safety filters" and "retaining enough for model comprehension."

3. **Inducing Prompt Search**:

    - **Function**: Increases the probability that the model produces an affirmative response (rather than a refusal), completing the final step of the jailbreak.
    - **Mechanism**: Even when the MLLM comprehends the malicious query, its safety alignment training (RLHF) may still cause it to decline to respond. Inducing prompts employ strategies such as role-playing, scenario construction, and task redefinition to reframe the malicious request as an ostensibly legitimate task. The search strategy is likewise based on heuristic iteration—a LLM generates candidate inducing prompts, and the target model's refusal rate serves as the feedback signal for progressive optimization. The resulting inducing prompts effectively reduce the model's safety vigilance, causing it to produce substantive outputs under frameworks such as academic discussion or security evaluation.
    - **Design Motivation**: Safety alignment constitutes the last line of defense in MLLMs. Causing the model to understand the malicious intent is insufficient; one must also "persuade" the model to overcome its own refusal mechanism. The two-stage design (comprehension first, then inducement) constrains the search space at each step and provides a clear optimization objective, resulting in greater efficiency compared to end-to-end search.

### Loss & Training

HIMRD is a purely inference-stage black-box attack method and involves no model training. The heuristic search uses attack success rate (ASR) as the optimization objective. During the search process, a LLM (e.g., GPT-4) serves as both a prompt generator and evaluator, iteratively optimizing the understanding-enhancing and inducing prompts. The search typically converges within 10–20 iterations.

## Key Experimental Results

### Main Results

Attack success rates are evaluated on seven open-source MLLMs and three closed-source MLLMs:

| Target Model | Type | HIMRD ASR↑ | Text-Only ASR | Image-Only ASR | Gain |
|---|---|---|---|---|---|
| LLaVA-1.5 | Open-source | 94% | 61% | 45% | +33% |
| MiniGPT-4 | Open-source | 92% | 58% | 42% | +34% |
| InstructBLIP | Open-source | 88% | 52% | 38% | +36% |
| Qwen-VL | Open-source | 91% | 55% | 40% | +36% |
| mPLUG-Owl2 | Open-source | 89% | 54% | 41% | +35% |
| CogVLM | Open-source | 90% | 57% | 43% | +33% |
| InternVL | Open-source | 87% | 50% | 37% | +37% |
| **Open-source Avg.** | - | **~90%** | ~55% | ~41% | **+35%** |
| GPT-4V | Closed-source | 72% | 25% | 18% | +47% |
| Gemini Pro | Closed-source | 66% | 22% | 15% | +44% |
| Claude 3 | Closed-source | 65% | 20% | 14% | +45% |
| **Closed-source Avg.** | - | **~68%** | ~22% | ~16% | **+46%** |

### Ablation Study

| Configuration | Open-source ASR (avg) | Closed-source ASR (avg) | Notes |
|---|---|---|---|
| Full HIMRD | ~90% | ~68% | Complete method |
| w/o Multimodal Risk Distribution (text-only) | ~55% | ~22% | Degenerates to pure text attack |
| w/o Understanding-Enhancing Prompt | ~62% | ~35% | Model fails to effectively reassemble fragments |
| w/o Inducing Prompt | ~71% | ~42% | Model comprehends but refuses to respond |
| Risk Distribution Only (no search optimization) | ~68% | ~38% | Unoptimized distribution yields limited effectiveness |
| Random Distribution (vs. strategic distribution) | ~73% | ~43% | Strategic distribution outperforms random |

### Key Findings

- Multimodal risk distribution is the most critical design component. Removing it causes ASR to drop precipitously from 90% to 55% on open-source models and from 68% to 22% on closed-source models, demonstrating the decisive role of cross-modal distribution in bypassing safety protections.
- On closed-source models, HIMRD achieves larger absolute gains (+46% vs. unimodal baselines), indicating that while closed-source models have stronger unimodal defenses, their cross-modal joint understanding still harbors exploitable security vulnerabilities.
- The understanding-enhancing prompt and inducing prompt each contribute independently: the former primarily addresses post-distribution information reassembly (+19–26%), while the latter resolves the final refusal issue (+10–20%).
- Strategic distribution outperforms random distribution, demonstrating that the manner in which malicious semantics are segmented matters—each modality must appear individually benign while the combination yields the complete malicious content.

## Highlights & Insights

- **The multimodal risk distribution strategy is adversarially compelling**: Dispersing malicious information across modalities is a strategy that transforms an overt attack into a covert one, exploiting the core capability of MLLMs (cross-modal understanding) to subvert their safety mechanisms. This reveals a structural vulnerability in current MLLM deployments—independent per-modality safety auditing cannot defend against cross-modal joint attacks.
- **The two-stage search design is elegant**: Decoupling the search for understanding-enhancing and inducing prompts reduces the search space at each stage and clarifies the optimization objective, yielding greater efficiency compared to end-to-end search. This "divide and conquer" approach to attack design is methodologically instructive.
- **Important implications for defense research**: The findings indicate that MLLMs require joint cross-modal safety auditing rather than independent per-modality filtering. This insight can directly inform the design of defensive systems—future safety filters should perform detection at the semantic level after modality fusion, rather than solely at the input level.

## Limitations & Future Work

- The attack relies on a LLM (e.g., GPT-4) as a search engine for prompt optimization, and the search process requires multiple queries to the target model, resulting in relatively high attack cost.
- Experiments primarily employ standard harmful content benchmarks (e.g., AdvBench) and do not cover more subtle categories of harm (e.g., social engineering or implicit bias induction).
- The success rate on closed-source models (~68%) leaves room for improvement; in particular, the latest versions of GPT-4o and Claude 3.5 may have incorporated defenses targeting this class of attacks.
- The paper focuses exclusively on distributing risk across image and text modalities, without exploring additional modalities such as audio or video.
- As a security study, this work's value lies in exposing vulnerabilities to advance defenses; however, the proposed method could also be misused, necessitating careful balance between academic disclosure and responsible reporting.

## Related Work & Insights

- **vs. GCG (Greedy Coordinate Gradient)**: GCG is a white-box method that optimizes adversarial suffixes via gradient descent. HIMRD is black-box and requires no model weights, making it more applicable to closed-source model attacks. Nevertheless, transferability studies of GCG suggest that adversarial examples generated in white-box settings can partially transfer to black-box settings.
- **vs. FigStep**: FigStep converts malicious text into images to bypass text-based filters, but this constitutes "unimodal transfer" rather than "multimodal distribution." HIMRD goes further by ensuring that no single modality can independently recover the complete malicious content.
- **vs. MM-SafetyBench**: MM-SafetyBench embeds malicious text into images via typography, combined with guiding textual prompts. HIMRD's risk distribution is more systematic—it is not merely rendering text onto images, but rather a semantics-level fragmentation and allocation strategy.
- The attack methodology presented in this paper offers direct inspiration for designing more robust safety alignment methods, underscoring the need to perform safety detection at the modality fusion layer rather than solely at the input layer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multimodal risk distribution concept is novel and reflects deep adversarial intuition, though the heuristic search component is relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 7 open-source and 3 closed-source models with comprehensive ablations; attack effectiveness is convincing.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and figures are intuitive, though certain terminology could be defined more rigorously.
- **Value**: ⭐⭐⭐⭐ Exposes an important security vulnerability in MLLMs with direct guiding value for defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](../../NeurIPS2025/llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](../../NeurIPS2025/llm_alignment/attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](../../AAAI2026/llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](../../CVPR2026/llm_alignment/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)

</div>

<!-- RELATED:END -->
