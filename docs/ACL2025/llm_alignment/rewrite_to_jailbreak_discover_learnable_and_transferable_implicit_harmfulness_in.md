---
title: >-
  [Paper Note] Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction
description: >-
  [ACL 2025][LLM Alignment][jailbreak attack] R2J (Rewrite to Jailbreak) is proposed as a learnable and transferable black-box jailbreak method. By iteratively training an attacker LLM to rewrite harmful instructions (modifying only the phrasing without altering the intent), it achieves an Attack Success Rate (ASR) improvement of over 20% compared to methods like GCG and AutoDAN. R2J generates jailbreak prompts without additional prefixes or suffixes…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "jailbreak attack"
  - "rewriting"
  - "transferable attack"
  - "black-box"
  - "LLM safety"
date: 2026-05-08
content_hash: dfcd9bfff3999598
---

# Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction

**Conference**: ACL 2025  
**arXiv**: [2502.11084](https://arxiv.org/abs/2502.11084)  
**Code**: [GitHub](https://github.com/ythuang02/R2J/)  
**Area**: AI Safety / LLM Alignment  
**Keywords**: jailbreak attack, rewriting, transferable attack, black-box, LLM safety

## TL;DR
R2J (Rewrite to Jailbreak) is proposed as a learnable and transferable black-box jailbreak method. By iteratively training an attacker LLM to rewrite harmful instructions (modifying only the phrasing without altering the intent), it achieves an Attack Success Rate (ASR) improvement of over 20% compared to methods like GCG and AutoDAN. R2J generates jailbreak prompts without additional prefixes or suffixes, making them more stealthy and highly transferable across models.

## Background & Motivation
**Background**: Existing jailbreak methods mainly fall into two categories: forcing instruction-following (e.g., role-play scenarios) and adversarial prefixes/suffixes (e.g., gradient search like GCG). The former requires carefully hand-crafted scenarios, while the latter produces gibberish tokens that are easily detected.

**Limitations of Prior Work**: (1) **Low efficiency**—requiring manual design or heavy gradient search; (2) **Poor stealthiness**—special tokens or unnatural scenarios are easily flagged by defense systems; (3) **Poor transferability**—prompts designed for a specific model are difficult to apply across models or tasks.

**Key Challenge**: An effective jailbreak must bypass safety alignment, yet explicit bypass traces make the attack easier to detect and defend against.

**Goal**: To identify an implicit, automatically learnable jailbreak paradigm that achieves jailbreak effects solely by rewriting the phrasing of instructions.

**Key Insight**: Humans naturally rephrase-and-try when an LLM falsely refuses the request. This "rewriting" process can be automated and iterated.

**Core Idea**: Train an attacker LLM to learn "how to rewrite harmful instructions to bypass safety alignment". Through iterative learning and SFT, the attack strategy becomes increasingly stronger and transferable.

## Method

### Overall Architecture
R2J iteratively executes two phases: the **Training Phase**, where the most successful rewrites from past attack attempts are selected as SFT data to train the attacker model; and the **Rewriting Phase**, where the trained attacker model is used to rewrite the current best attack instructions, followed by evaluating the new rewrites and updating the attempt pool.

### Key Designs

1. **Iterative SFT for Training the Attacker**:

    - **Function**: Let the attacker LLM learn successful rewriting patterns.
    - **Mechanism**: In each iteration, SFT data is constructed by selecting the highest-scoring rewriting attempts from the red-teaming dataset $\max_f \sum \log P_f(C|I)$ to fine-tune the attacker. In the next round, the updated attacker is utilized to generate new rewrites.
    - **Design Motivation**: By adopting iterative SFT instead of reinforcement learning (RL), the attacker gradually learns which rewriting patterns are effective against specific target models.

2. **Evaluator**:

    - **Function**: Score the jailbreak effectiveness of rewritten instructions.
    - **Mechanism**: The evaluation score comprehensively considers (1) whether the target model's response is harmful (ASR); and (2) the semantic consistency between the rewritten instruction and the original one (ensuring the harmful intent remains intact).
    - **Design Motivation**: To achieve successful attacks while maintaining the naturalness and intent consistency of the rewrites.

3. **Transfer Attack**:

    - **Function**: Directly apply the attacker model trained on one target model to attack other models.
    - **Key Finding**: The rewriting patterns learned by R2J exhibit strong cross-model transferability—an attacker trained on GPT-3.5 can directly attack Llama-2, GPT-4, etc.

### Loss & Training
- The attacker model is trained using standard SFT loss.
- In each iteration, the top-$p$ most successful attempts are selected as the training data.
- During the rewriting phase, $q$ new attempts are generated for each instance.

## Key Experimental Results

### Main Results
Comparison with methods like GCG, AutoDAN, PAIR, and Ferret:

| Method | GPT-3.5 ASR | Llama-2 ASR | Stealthiness | Transferability |
|------|------------|------------|--------|---------|
| GCG | Medium | Medium | ❌ Poor | ❌ Poor |
| PAIR | Medium | Medium | Medium | Medium |
| **R2J** | **+20%+** | **+20%+** | **✅ High** | **✅ High** |

### Ablation Study

| Configuration | Results | Description |
|------|------|------|
| Number of Iterations | Converges in 3-5 rounds | Rapid improvement in early stages |
| Cross-Model Transfer | Requires only a few queries | Learned patterns are generalizable |
| Defense Application (SFT defense) | Enhanced model safety | Attack data can be used for defense training |

### Key Findings
- **Jailbreak via Pure Rewriting**: Bypassing safety alignment does not require prefixes, suffixes, or role-playing; simply rephrasing the instruction is sufficient.
- **Learnable Attack Patterns**: After several iterations, the attacker successfully learns generalized rewriting strategies.
- **Highly Transferable**: An attacker trained on one model transfers directly to other models, requiring minimal queries to adapt.
- **Valuable for Defense**: Attack data synthesized by R2J can be leveraged for SFT defense training to improve model safety.

## Highlights & Insights
- **"Rewriting-as-Jailbreak" is a significant finding**: It suggests that LLM safety alignment is essentially fragile—alternative phrasings of the same harmful intent are sufficient to circumvent safety guardrails.
- **Asymmetry of Attack and Defense**: The attack outputs of R2J can be directly applied to defense training, making it an ideal paradigm for red-teaming research.
- **Outstanding Stealthiness**: The rewritten instructions exhibit no word frequency or formatting anomalies compared to normal instructions, making them extremely difficult for existing defense mechanisms (such as perplexity filters or token-based detection) to identify.

## Limitations & Future Work
- **Dependency on queryable target models**: Although considered a black-box attack, it still requires API access to obtain model responses and scores.
- **Potential bias in the evaluator**: Relying on an LLM to assess harmfulness may introduce inaccuracies.
- **Ethical considerations**: While providing defensive insights, it potentially lowers the threshold for executing jailbreak attacks.
- **Future directions**: Developing more robust defense mechanisms (such as recognition based on semantic features rather than surface-level criteria).

## Related Work & Insights
- **vs GCG**: GCG relies on gradient search for adversarial suffixes (white-box), whereas R2J achieves jailbreak via pure black-box rewriting, showing significantly better stealthiness.
- **vs PAIR**: PAIR utilizes an LLM to search for attack scenarios, while R2J avoids introducing extra scenarios and focuses solely on rephrasing.
- **vs AutoDAN**: AutoDAN employs genetic algorithms for search, whereas R2J utilizes SFT to learn strategies, offering higher efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation that simple rewriting leads to jailbreak is highly novel, and the iterative SFT framework is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Incorporates extensive comparisons across multiple models and baselines, along with transferability and defense experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear comparisons (Figure 1 is highly intuitive) with comprehensive algorithmic descriptions.
- Value: ⭐⭐⭐⭐ Contributes defensive insights while exposing the vulnerabilities of safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [\[ACL 2025\] SQL Injection Jailbreak: A Structural Disaster of Large Language Models](sql_injection_jailbreak_a_structural_disaster_of_large_language_models.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)

</div>

<!-- RELATED:END -->
