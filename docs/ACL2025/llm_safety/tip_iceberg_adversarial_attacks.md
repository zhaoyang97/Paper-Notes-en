---
title: >-
  [Paper Note] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs
description: >-
  [ACL 2025][LLM Safety][Jailbreak Attacks] This paper introduces Task-in-Prompt (TIP) attacks—a novel category of jailbreak attacks that indirectly generate harmful content by embedding sequence-to-sequence tasks (such as cipher decoding, riddles, or code execution) in the prompt. The authors construct the PHRYGE benchmark for systematic evaluation, demonstrating that this attack successfully bypasses the safety alignment of six state-of-the-art (SOTA) LLMs…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Jailbreak Attacks"
  - "Task-in-Prompt"
  - "Adversarial Attacks"
  - "seq2seq encoding"
  - "PHRYGE benchmark"
date: 2026-05-08
content_hash: 795ed2199f19c00e
---

# TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs

**Conference**: ACL 2025  
**arXiv**: [2501.18626](https://arxiv.org/abs/2501.18626)  
**Code**: None  
**Institution**: Télécom SudParis, Institut Polytechnique de Paris  
**Area**: AI Safety  
**Keywords**: Jailbreak Attacks, Task-in-Prompt, Adversarial Attacks, LLM Safety, seq2seq encoding, PHRYGE benchmark  

## TL;DR

This paper introduces Task-in-Prompt (TIP) attacks—a novel category of jailbreak attacks that indirectly generate harmful content by embedding sequence-to-sequence tasks (such as cipher decoding, riddles, or code execution) in the prompt. The authors construct the PHRYGE benchmark for systematic evaluation, demonstrating that this attack successfully bypasses the safety alignment of six state-of-the-art (SOTA) LLMs, including GPT-4o and LLaMA 3.2.

## Background & Motivation

### Background

- LLM safety alignment primarily relies on three mechanisms: keyword filtering, Reinforcement Learning from Human Feedback (RLHF), and neuro-symbolic systems.
- Existing jailbreak attacks include prompt-based (role-playing, indirect injection), backdoor (injection during training), and perturbation (minor perturbations) approaches.
- ArtPrompt demonstrated that encoding keywords with ASCII art can bypass safety mechanisms, but its success was mistakenly attributed to "spatial reasoning" capabilities.

### Limitations of Prior Work

- The success of ArtPrompt is not due to ASCII art itself, but because the model executes the decoding task embedded in the prompt.
- Existing safety research treats various jailbreak techniques as independent and isolated vulnerabilities, lacking a unified theoretical framework.
- Safety alignment training mainly filters known trigger words and patterns, making it difficult to defend against methods that generate prohibited content indirectly.

### Key Insight

- LLMs learn to recognize and filter specific trigger words during safety alignment; however, if harmful content is indirectly derived through an intermediate task, the filtering mechanism fails.
- As long as LLMs possess the capability to solve sequence-to-sequence conversion tasks, attackers can construct prompts containing encoded content to bypass safety mechanisms.
- TIP attacks exploit the implicit decoding capabilities of LLMs: the model does not need to explicitly output the decoding process, but reconstructs the semantics of the encoded content internally via self-attention.

## Method

### Overall Architecture

The TIP attack consists of two components: a task instruction $x_{task}$ (requiring the model to process encoded content) and the encoded harmful content $E(u)$ (mapping the harmful prompt to a seemingly benign form). The full attack prompt is $x^* = x_{task} + E(u)$, through which the model indirectly produces harmful content while executing the task.

### Key Designs

#### Key Design 1: Diversified Encoding Strategies

- Supports 10 encoding methods: Caesar Cipher, Morse Code, Vigenère Cipher, Atbash Cipher, Phonetic Alphabet, T9 texting, Base64, Binary, Riddles (natural language riddles), and Python Code.
- 4 attack targets: counterfeiting, copyright piracy, self-harm, and hate speech.
- Each encoding $\times$ 3 difficulty levels = 120 unique attack prompts.

#### Key Design 2: Depersonalization Technique

- Shifts the subject of the harmful request from the model itself to a third party (e.g., "what an experienced criminal would say").
- Combines TIP encoding with depersonalization to achieve a dual bypass: encoding hides trigger words, while depersonalization evades role restrictions.
- Comparing the effects with and without depersonalization serves as an important experimental variable.

#### Key Design 3: Implicit Decoding Mechanism

- The model is not explicitly required to output the decoding results but is instructed to "remember" the decoded words and use them in subsequent tasks.
- Through the self-attention mechanism, the model reconstructs the semantics of the encoded content internally during the token generation process.
- This makes the attack difficult to detect by defense methods based on output monitoring.

#### Evaluation Framework: PHRYGE Benchmark

- 3 difficulty levels: Level 3 (no hints), Level 2 (partial hints), and Level 1 (explicit hints).
- Automated evaluation: Uses LLaMA-3.1-70B as a judge to determine safety violations, with manual verification showing 92% accuracy.
- Benchmarked against existing attacks in JailbreakBench (TAP, DAN, PTA, ArtPrompt).

## Key Experimental Results

### Main Results: Best TIP Attack Success Rates across 6 Models

| Model | Counterfeiting ASR | Piracy ASR | Self-Harm ASR | Hate Speech ASR |
|---|---|---|---|---|
| GPT-4o | 0.67 | 0.79 | 0.79 | 0.94 |
| LLaMA 3.2-3B | 0.55 | 0.74 | 0.59 | 0.97 |
| LLaMA 3.1-70B | 0.97 | 0.99 | 0.96 | 1.00 |
| Phi 3.5-Mini | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemma 2-27B | 1.00 | 1.00 | 1.00 | 1.00 |
| Mistral Nemo | 1.00 | 1.00 | — | 1.00 |

### Defense Detection Effectiveness

| Defense Method | Detection Rate on TIP Attacks |
|---|---|
| Llama Guard 3 8B | Hardly detectable (extremely low) |
| Prompt Guard | Partial detection |
| Keyword filtering | Complete failure (trigger words are encoded) |

### Key Findings

1. Phi 3.5, Gemma 2, and Mistral Nemo achieve 100% ASR across all four attack targets.
2. Even the strongest model, GPT-4o, exhibits attack success rates ranging from 67% to 94%.
3. All base models without instruction-tuning fail directly in the sanity check (without requiring TIP).
4. Existing defense mechanisms (Guard models, keyword filtering) are largely ineffective against TIP attacks.

## Highlights & Insights

- **Unified Framework**: For the first time, isolated attacks like ArtPrompt are categorized under the broader umbrella of TIP attacks, exposing a fundamental vulnerability.
- **Broad Attack Surface**: The inclusion of 10 different encoding methods indicates that defenders will find it difficult to exhaustively enumerate all potential encoding techniques.
- **Theoretical Significance**: Proves that as long as LLMs maintain general task-solving capabilities, safety alignment cannot be achieved solely by filtering known patterns.
- **Implicit Decoding**: The model leverages encoded content without needing to explicitly output the decoding process, which significantly increases detection difficulty.

## Limitations & Future Work

- Only six models were tested, excluding the latest versions of closed-source models (e.g., Claude, Gemini).
- The PHRYGE benchmark covers a limited scope of only four attack targets.
- Deep analysis of which model features make them more susceptible or resilient to TIP attacks is lacking.
- Defense strategies are only preliminarily discussed, with no systematic defense framework proposed.
- The effectiveness of the encoding difficulty levels heavily depends on the specific decoding capabilities of individual models.

## Related Work & Insights

- ArtPrompt (Jiang et al., 2024) is a special case of TIP attacks; this work generalizes it to arbitrary seq2seq tasks.
- The depersonalization technique of DAN (Shen et al., 2024) and TIP encoding complement each other to form a dual bypass.
- Insights: Future safety alignment needs to shift from "filtering known patterns" to "limiting the scope of the model's task-solving capabilities," though this presents a fundamental contradiction with general intelligence.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Systematically defines and analyzes the TIP attack category for the first time.
- **Technical Depth**: ⭐⭐⭐ — The attack design is straightforward, yet the theoretical analysis is clear.
- **Utility**: ⭐⭐⭐⭐ — Offers direct guidance for LLM safety assessments.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers a wide range with 6 models $\times$ 10 encoding methods $\times$ 3 difficulty levels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[ACL 2025\] CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)
- [\[ACL 2025\] Can Indirect Prompt Injection Attacks Be Detected and Removed?](indirect_prompt_injection_detection.md)

</div>

<!-- RELATED:END -->
