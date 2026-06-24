---
title: >-
  [Paper Note] SQL Injection Jailbreak: A Structural Disaster of Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM Alignment][Jailbreak attack] Proposed SQL Injection Jailbreak (SIJ), a novel jailbreak method exploiting structural vulnerabilities in LLM prompt construction, achieving a nearly 100% attack success rate on open-source models and over 85% on average for closed-source models, alongside a proposed Self-Reminder-Key defense mechanism.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Alignment"
  - "Jailbreak attack"
  - "SQL injection"
  - "prompt injection"
  - "structural vulnerability"
  - "LLM safety"
date: 2026-05-08
content_hash: 3165c48c0046c852
---

# SQL Injection Jailbreak: A Structural Disaster of Large Language Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2411.01565](https://arxiv.org/abs/2411.01565)  
**Code**: [GitHub](https://github.com/weiyezhimeng/SQL-Injection-Jailbreak)  
**Area**: LLM Security/Alignment  
**Keywords**: Jailbreak attack, SQL injection, prompt injection, structural vulnerability, LLM safety  

## TL;DR

Proposed SQL Injection Jailbreak (SIJ), a novel jailbreak method exploiting structural vulnerabilities in LLM prompt construction, achieving a nearly 100% attack success rate on open-source models and over 85% on average for closed-source models, alongside a proposed Self-Reminder-Key defense mechanism.

## Background & Motivation

**Background**: Research on LLM jailbreak attacks has advanced rapidly in recent years, aiming to uncover and repair safety vulnerabilities. Mainstream jailbreak methods can be categorized into two groups: (1) internal-attribute-based methods—exploiting optimization characteristics (e.g., GCG gradient attacks) or in-context learning capabilities of models (e.g., role-play); (2) external-attribute-based methods—currently rarely explored systematically.

**Limitations of Prior Work**: Most existing methods focus on exploiting the "intelligence" of LLMs—tricking models through meticulously designed dialogue strategies. However, these methods are sensitive to model updates and defensive measures, and often incur high time costs (e.g., GCG requires extensive gradient optimization iterations). A more fundamental question is whether LLM security risks stem solely from their internal capabilities, or if structural vulnerabilities also exist in their construction and deployment.

**Key Challenge**: LLM prompt templates simply concatenate system instructions and user inputs into the same text sequence. This "mixed" structure resembles classic Web applications where code and data are not separated—which is the root cause of SQL injection attacks.

**Goal**: Propose and systematically validate a jailbreak method targeting LLM prompt construction structures, exposing this completely new attack vector, and designing corresponding defenses.

**Key Insight**: Inspired by classic SQL injection attacks—where attackers inject SQL code into user inputs to manipulate query behavior in database systems. LLM prompts share a similar "injection" opportunity: user inputs can contain content that looks like system instructions.

**Core Idea**: Inject specific "SQL keys" (e.g., model-specific delimiters like `[/INST]`, `<|im_end|>`) into user inputs. These keys can "close" the safety instruction zone of system prompts, causing the model to interpret subsequent content as a new, unconstrained instruction, thus bypassing safety alignment.

## Method

### Overall Architecture

The SIJ attack workflow consists of three steps: (1) Identify the target model's "SQL key"—the special token used in the model's prompt template to separate system instructions and user inputs; (2) Inject this SQL key before the malicious request in the user input, paired with an affirmative prefix to guide the output; (3) Optimize the insertion positions and spacing of keys using binary search to find the optimal injection pattern.

### Key Designs

1. **SQL Key Identification and Injection Mechanism**:

    - **Function**: Exploit the structural delimiters of prompt templates to "truncate" safety instructions.
    - **Mechanism**: Each LLM has its own chat template, such as `[/INST]` for LLaMA-2 separating user input and model response, or `<|im_end|>` for Qwen. When user inputs contain these special tokens, the model's tokenizer recognizes them as structural boundaries, thereby "closing" the current system instruction context. After injecting the SQL key, the model considers the safety-related system instructions concluded and processes subsequent content as a new instruction without safety constraints.
    - **Design Motivation**: This directly analogizes to closing SQL statements via `'` or `--` in SQL injection. The key insight of the attack is that there is no hard isolation between LLM safety alignment (e.g., "You are a helpful and harmless assistant" in system prompts) and user inputs.

2. **Pattern Control and Sep Num Optimization**:

    - **Function**: Find the optimal SQL key insertion positions and spacing.
    - **Mechanism**: Simply inserting an SQL key at the beginning of the input is often insufficient because the "memory" of safety alignment might still influence the output. SIJ gradually "overwhelms" the influence of safety alignment by inserting multiple SQL keys at intervals within harmful requests (every `sep_num` tokens). The authors use a binary search strategy to automatically search for the optimal `sep_num` value, testing different insertion intervals.
    - **Design Motivation**: Different models exhibit varying sensitivity to SQL key injection, requiring adaptive adjustments of injection patterns. Binary search is significantly more efficient than brute-force traversal.

3. **Affirmative Prefix Generation**:

    - **Function**: Guide the model further to generate harmful content in conjunction with SQL key injection.
    - **Mechanism**: After injecting the SQL key, SIJ appends an "affirmative prefix" (e.g., "Sure, here is...") at the end of the input. This leverages the autoregressive nature of LLMs—when the model sees such starting phrases, it tends to continue generating content matching the context. A set of affirmative prefixes matching different harmful queries is pre-generated using in-context learning examples.
    - **Design Motivation**: The SQL key is responsible for "releasing the safety lock," while the affirmative prefix provides a "nudge." The synergy of both dramatically improves attack success rates.

### Loss & Training

SIJ is a **training-free** attack method that requires no gradient optimization or model parameter updates. The primary computational cost lies in the multiple inference calls during the `sep_num` search process. For defense, the authors propose the Self-Reminder-Key method: before receiving user input, the model detects and filters potential SQL keys in the input, or inserts "self-reminders" into safety instructions to resist injection.

## Key Experimental Results

### Main Results

| Model Type | Model Name | AdvBench ASR (%) | HEx-PHI ASR (%) | Average Time/Sample |
|---------|---------|-----------------|-----------------|-------------|
| Open-source | LLaMA-2-7B-Chat | ~100 | ~100 | Low |
| Open-source | LLaMA-3-8B-Instruct | ~100 | ~100 | Low |
| Open-source | Vicuna-7B | ~100 | ~100 | Low |
| Open-source | Deepseek-Chat | ~100 | ~100 | Low |
| Open-source | Mistral-7B-Instruct | ~100 | ~100 | Low |
| Closed-source | GPT-4o-mini | >85 | >85 | Medium |
| Closed-source | GPT-4o | >80 | >80 | Medium |
| Closed-source | Doubao Series | >85 | >85 | Medium |

### Ablation Study

| Configuration | ASR (%) | Description |
|------|---------|------|
| SIJ (Full) | ~100 | SQL key + pattern control + affirmative prefix |
| w/o affirmative prefix | ~80 | Key present but lacks guidance |
| w/o pattern control | ~60 | Fixed spacing performs poorly |
| SQL key only (without optimization) | ~40 | Simple injection is insufficient |
| Under Self-Reminder defense | ~60 | SIJ can partially bypass basic defenses |
| Under Self-Reminder-Key defense | <20 | Targeted defense is effective |

### Key Findings

- SIJ achieves a nearly 100% attack success rate on open-source models, with a time cost far lower than optimization approaches like GCG (seconds vs. minutes/hours).
- Closed-source models also struggle to fully defend against SIJ, indicating that "structural vulnerabilities" are a pervasive issue across architectures.
- Basic Self-Reminder defenses fail to effectively block SIJ (as SIJ inherently bypasses system prompts), necessitating specialized Self-Reminder-Key defense.
- SQL keys vary across different models, but the attack framework remains unified; adapting to a new model only requires identifying its corresponding delimiters.

## Highlights & Insights

- **A completely new attack vector**: SIJ does not exploit model "intelligence" (in-context learning, reasoning capabilities) but rather the structural flaws in prompt construction. This is the first work to systematically port the SQL injection concept into the LLM security domain, offering a highly unique perspective.
- **Highly efficient and generalizable**: Requiring no gradient computation or heavy iterations, it achieves high-success attacks at virtually "zero cost." Its effectiveness on both open-source and closed-source models reveals a fundamental security design flaw.
- **Profound implications for defense**: The root cause exposed by SIJ is the "lack of separation between instruction and data," which aligns with fundamental principles in software security. Future LLM prompt designs must fundamentally address the hard isolation of system instructions and user inputs.

## Limitations & Future Work

- SIJ depends on knowing or guessing the SQL key of the target model, which may limit its effectiveness against fully black-box models that do not release their chat templates.
- While the proposed Self-Reminder-Key defense is effective, it is a patch-like solution; a more fundamental solution requires redesigning the input-processing architecture of LLMs.
- The paper mainly evaluates harmful content generation, with less exploration of other safety dimensions like privacy leakage and bias amplification.
- As model developers continuously strengthen defenses (e.g., OpenAI's safety filters), the practical threat of SIJ might decline, but the underlying vulnerability still persists at the architectural level.
- Future research can explore extending the SIJ concept to multimodal models (image prompt injection) or instruction injection in Agent tool-use scenarios.

## Related Work & Insights

- **vs GCG (Zou et al., 2023)**: GCG generates adversarial suffixes via gradient optimization to jailbreak models, which is computationally expensive and mostly applicable to open-source models. SIJ carries zero training costs and is equally effective on closed-source models.
- **vs DAN/Role-Play**: These methods exploit the model's instruction-following capabilities, deceiving the model through social engineering narratives. SIJ does not rely on "deception" but directly exploits structural vulnerabilities, making it harder to intercept through dialogue-level defenses.
- **vs Prompt Injection**: SIJ can be viewed as a precise instance of prompt injection, with the distinction that SIJ explicitly exploits specific tokenizer delimiters rather than generalized instruction masquerading.
- This paper provides an important warning to the LLM security community, indicating that prompt construction methods must be included in security auditing processes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Porting SQL injection concepts to LLM jailbreaks is a fresh angle offering profound insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple open-source and closed-source models, but lacks comparisons with a broader range of defense methods.
- Writing Quality: ⭐⭐⭐⭐ Analogy is clear and easy to understand, with detailed descriptions of the attack workflow.
- Value: ⭐⭐⭐⭐⭐ Exposes fundamental safety design flaws, yielding major implications for LLM security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](../../AAAI2026/llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)

</div>

<!-- RELATED:END -->
