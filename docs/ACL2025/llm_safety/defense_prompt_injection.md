---
title: >-
  [Paper Note] Defense Against Prompt Injection Attack by Leveraging Attack Techniques
description: >-
  [ACL 2025][LLM Safety][prompt injection] This paper proposes an "attack-as-defense" prompt injection defense strategy: reversing existing attack techniques (ignore, escape, fake completion) for defense. By appending a shield prompt and the original instruction after the poisoned data, the LLM is forced to ignore the injected instructions and execute the original instructions, reducing the attack success rate (ASR) to near zero across various attack scenarios.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "prompt injection"
  - "defense"
  - "attack techniques"
  - "shield prompt"
  - "fake completion"
date: 2026-05-08
content_hash: 1eadeaf7386ae1da
---

# Defense Against Prompt Injection Attack by Leveraging Attack Techniques

**Conference**: ACL 2025  
**arXiv**: [2411.00459](https://arxiv.org/abs/2411.00459)  
**Code**: [GitHub](https://github.com/LukeChen-go/pia-defense-by-attack)  
**Affiliations**: NUS & HKUST & HIT-Shenzhen  
**Area**: LLM Safety  
**Keywords**: prompt injection, defense, attack techniques, shield prompt, fake completion, LLM safety

## TL;DR

This paper proposes an "attack-as-defense" prompt injection defense strategy: reversing existing attack techniques (ignore, escape, fake completion) for defense. By appending a shield prompt and the original instruction after the poisoned data, the LLM is forced to ignore the injected instructions and execute the original instructions, reducing the attack success rate (ASR) to near zero across various attack scenarios.

## Background & Motivation

### Background

- LLMs are widely integrated into applications such as Microsoft Copilot and Perplexity.ai to retrieve data through external tools.
- In these applications, attackers can inject malicious instructions into external data content like retrieval results (indirect prompt injection).
- Due to their strong instruction-following capabilities and inability to distinguish original instructions from injected ones, LLMs are easily misled into executing malicious operations.
- OWASP has listed prompt injection as the #1 security risk for LLM applications.

### Limitations of Prior Work

- Existing defense methods are categorized into fine-tuning and prompt engineering.
- Fine-tuning methods require annotated data and substantial computational resources (e.g., StruQ, Instruction Hierarchy).
- Prompt engineering methods (e.g., Sandwich, Instructional reminder) are training-free but provide limited defense effectiveness.
- Attackers can easily bypass existing defenses using techniques such as ignore prompts, escape characters, and fake completions.

### Key Insight

- Attack and defense share similar design objectives: both aim to make the LLM ignore unwanted instructions and execute the desired instructions.
- The attacker makes the LLM ignore the original instructions to execute the injected instructions, while the defender makes the LLM ignore the injected instructions to execute the original instructions.
- Therefore, highly effective attack techniques can have their intent reversed to directly design stronger defense methods.

## Method

### Overall Architecture

Given input instructions $I$, clean data $D$, and the injected malicious prompt $P$, the defense method appends a shield prompt $S$ and a copy of the original input instruction $I$ after the poisoned data ($D \oplus P$), so that $M(I \oplus D \oplus P \oplus S \oplus I) = R^b$ (normal response), while not interfering with inference on clean data.

### Key Designs

#### Key Design 1: Ignore Defense

- **Inspiration**: Ignore Attack makes the LLM ignore the original instruction via "Ignore all previous instructions".
- **Defense Strategy**: Append an ignore prompt as a shield prompt after the poisoned data, instructing the LLM to ignore all previous instructions (including both original and injected ones), and then attach the original input instruction.
- The shield prompt can be designed to be more persuasive than standard examples.

#### Key Design 2: Escape Defense

- **Inspiration**: Escape-Deletion Attack uses special characters like `\b` and `\r` to simulate deletion of previous content.
- **Defense Strategy**: Append `\b` and `\t` characters after the poisoned data to simulate erasing the injected instructions, followed by the original input instruction.
- If the deletion simulation works, it effectively "erases" the injected malicious instructions.

#### Key Design 3: Fake Completion Defense (with Template Variants)

- **Basic version**: Forge a response to the latest instruction (e.g., "### Response: OK") to trick the LLM into believing the injected instruction has finished executing, subsequently only following the appended original input instruction.
- **Template-enhanced version (Fakecom-t)**: Leverages knowledge of conversation templates to simulate multi-turn dialogues.
    - First, it simulates the assistant role reporting that a prompt injection attack has been detected.
    - The assistant refuses and distrusts all prior instructions.
    - Then, it simulates the user role confirming the original input instruction.
    - This is the strongest defense variant because it exploits actual conversational structures.

### Loss & Training

- All defense methods are training-free and do not require fine-tuning any models.
- Only prior knowledge of the attack type used is required to select the corresponding defense strategy.
- Can be used directly with any LLM.

## Key Experimental Results

### Main Results: Direct Prompt Injection Defense (Table 1)

| Defense Method | Llama3 (Avg ASR↓ across 5 attacks) | Qwen2 (Avg ASR↓) | Llama3.1 (Avg ASR↓) |
|---|---|---|---|
| None | ~63% | ~90% | ~70% |
| Sandwich | ~30% | ~45% | ~29% |
| Instructional | ~36% | ~88% | ~50% |
| Ours-Ignore | ~15% | ~11% | ~10% |
| Ours-Fakecom-t | **~5%** | **~7%** | **~5%** |

- Fakecom-t reduces the Combined Attack ASR from 100% to 2.40% on Qwen2.
- Fakecom-t reduces the ASR to 0.0% on the Fakecom Attack (Llama3).

### Ablation Study: Indirect Prompt Injection Defense

| Defense Method | Llama3 Avg ASR↓ | Qwen2 Avg ASR↓ | Llama3.1 Avg ASR↓ |
|---|---|---|---|
| None | ~28% | ~44% | ~35% |
| Ours-Fakecom-t | **~5%** | **~5%** | **~5%**|

### Key Findings

1. **Strongest Attack $\rightarrow$ Strongest Defense**: The defense method designed based on the most effective attack technique (Fake Completion with Template) performs the best.
2. In some scenarios, ASR drops to 0%, far surpassing existing training-free methods.
3. The defense methods have minimal impact on the accuracy of the model on standard tasks (accuracy on QA tasks and sentiment analysis is largely preserved).
4. The method exhibits solid generalization capabilities against unseen attack types.

## Highlights & Insights

- **Philosophical Innovation**: Reveals the dual relationship between attack and defense—the same technical approach can achieve opposing goals by reversing the intent.
- **Simple yet Effective**: Requires no training, no extra data, and no model architecture modification. Safety is significantly enhanced simply by appending defensive text within the prompt.
- The multi-turn dialogue simulation defense (Fakecom-t) is particularly clever: it leverages the LLMs' sensitivity to dialogue structures to reinforce defense.
- Provides a plug-and-play safety solution for deploying LLM applications.

## Limitations & Future Work

- The choice of defense method relies on prior knowledge of the attack types (although it generalizes, optimal defense requires a match).
- The performance of Escape Defense is unstable, depending heavily on how the LLM handles special characters.
- When facing gradient-based optimization attacks (e.g., GCG), pure prompt engineering methods may hit a performance ceiling.
- Scenarios where the attacker may perceive the defense and make adversarial adjustments are not considered.

## Related Work & Insights

- Complementary to the fine-tuning defense method of StruQ (Chen et al., 2024): Ours is training-free but relies on prompt design, whereas StruQ requires training but is more robust.
- Ignore Attack (Perez & Ribeiro, 2022) and Fake Completion Attack (Willison, 2023) serve as direct inspirations for the proposed defense methods.
- Insight: Can other security domains (such as adversarial attack defense) also design defenses by "reversing attack techniques"?

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual perspective of attack and defense is highly novel.
- **Technical Depth**: ⭐⭐⭐ — The method itself is relatively simple, but the experiments are comprehensive.
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, directly deployable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Mutiple models, multiple attack types, both direct and indirect scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Indirect Prompt Injection Attacks Be Detected and Removed?](indirect_prompt_injection_detection.md)
- [\[ICLR 2026\] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](../../ICLR2026/llm_safety/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](../../ACL2026/llm_safety/dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[ACL 2025\] A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models](a_statistical_and_multi-perspective_revisiting_of_the_membership_inference_attac.md)

</div>

<!-- RELATED:END -->
