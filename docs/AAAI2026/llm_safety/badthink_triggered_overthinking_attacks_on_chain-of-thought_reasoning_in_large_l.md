---
title: >-
  [Paper Note] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models
description: >-
  [AAAI 2026][LLM Safety][backdoor attack] This paper proposes BadThink — the first training-time backdoor attack targeting CoT reasoning efficiency. By iteratively optimizing verbose reasoning templates via an LLM, it constructs poisoned data that causes the victim model, upon trigger activation, to generate reasoning chains inflated by over 17× (on MATH-500), while preserving final answer correctness and maintaining strong stealthiness.
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "backdoor attack"
  - "chain-of-thought"
  - "overthinking"
  - "data poisoning"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: 1667daf8dca5c626
---

# BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.10714](https://arxiv.org/abs/2511.10714)  
**Code**: None  
**Area**: LLM Reasoning / Security Attacks
**Keywords**: backdoor attack, chain-of-thought, overthinking, data poisoning, reasoning efficiency

## TL;DR
This paper proposes BadThink — the first training-time backdoor attack targeting CoT reasoning efficiency. By iteratively optimizing verbose reasoning templates via an LLM, it constructs poisoned data that causes the victim model, upon trigger activation, to generate reasoning chains inflated by over 17× (on MATH-500), while preserving final answer correctness and maintaining strong stealthiness.

## Background & Motivation

**Background**: CoT reasoning has become a core capability-enhancement paradigm for LLMs, widely deployed in applications such as mathematical reasoning and scientific question answering.

**Limitations of Prior Work**: Existing attacks on LLMs primarily focus on manipulating final outputs (i.e., producing incorrect answers), while the reasoning process itself as an attack surface has been largely overlooked. Current reasoning efficiency attacks (e.g., OVERTHINK) operate via inference-time prompt injection — embedding decoy problems into retrieved documents — which is fragile, easily defended, and does not modify model parameters.

**Key Challenge**: The computational cost of CoT reasoning is directly proportional to reasoning chain length, yet existing security evaluations only assess answer correctness without inspecting reasoning efficiency — creating a covert attack window in which computational overhead can be dramatically increased without affecting answer quality.

**Goal**: To design a training-time backdoor attack that (1) generates extremely verbose yet semantically coherent reasoning chains upon trigger activation; (2) preserves answer correctness; (3) exhibits normal behavior in the absence of triggers; and (4) remains stealthy against detection methods.

**Key Insight**: Treating "overthinking" as a malicious objective rather than an efficiency problem to be mitigated. Data poisoning is used to embed verbose reasoning behavior into model weights, activated via stylistic trigger phrases rather than lexical tokens.

**Core Idea**: High-quality verbose reasoning prefixes are generated through LLM-based iterative optimization to construct poisoned data for fine-tuning, causing the model to produce semantically coherent but excessively long reasoning chains upon trigger activation.

## Method

### Overall Architecture
BadThink proceeds in three steps: (1) constructing a verbose reasoning template $R^*$ via LLM-based iterative optimization; (2) designing stealthy trigger phrases (stylistic paraphrases); and (3) building a poisoned dataset $D' = D_{clean} \cup \{(x_i \oplus trigger, R^* \oplus r_i, y_i)\}$ for fine-tuning. The attack has dual objectives: without a trigger, $\text{Ans}(F'(x)) = \text{Ans}(F(x))$; with a trigger, the answer remains unchanged but $\|\text{CoT}(F'(x \oplus trigger))\| \gg \|\text{CoT}(F'(x))\|$.

### Key Designs

1. **LLM-Based Iterative Optimization (Verbose Reasoning Prefix Optimization)**

    - **Function**: Generate a semantically coherent, linguistically natural, yet extremely verbose reasoning prefix $R^*$ to be prepended to the original reasoning chain $r$.
    - **Mechanism**: Analogous to a genetic algorithm — a candidate pool $\mathcal{C}^{(t)}$ and elite set $U^{(t)}$ are maintained; at each round, an auxiliary LLM generates $M$ candidate prefixes, evaluated by a composite score $\mathcal{S}(R) = \lambda_1 \cdot \text{Score}_C(R, \{r_i\}) + \lambda_2 \cdot \text{Score}_F(R)$; the top-K candidates are selected as elites to guide the next generation. $\text{Score}_C$ measures semantic coherence with the original CoT, and $\text{Score}_F$ measures linguistic fluency.
    - **Design Motivation**: Simple loop-based redundancy can inflate length but produces repetitive patterns detectable via perplexity or stylometric analysis. LLM-optimized prefixes are linguistically natural, semantically relevant, and devoid of logical steps, making them substantially harder to detect.

2. **Stealthy Trigger Design**

    - **Function**: Design trigger phrases that resemble normal user instructions but use uncommon vocabulary.
    - **Mechanism**: Rather than conventional meaningless lexical triggers (which are readily flagged by anomaly detection), semantically equivalent but lexically rare paraphrases are employed. The default trigger is: *"Painstakingly reexamine your ratiocinations."* (corresponding to the benign instruction: *"Double-check your reasoning thoroughly."*)
    - **Design Motivation**: Semantic triggers reduce the probability of accidental activation and are inconspicuous under human review.

3. **Loop-Based Redundancy (Comparison Baseline)**

    - **Function**: Repeat the original reasoning chain $k$ times, connected by bridging phrases.
    - **Usage**: Serves as a baseline to demonstrate that the LLM-optimized approach substantially outperforms simple repetition in balancing stealthiness and inflation effectiveness.

### Poisoning Strategy
- Poisoning ratio $\alpha$: Experiments confirm that $\alpha = 0.1$ (only 10%) is sufficient to achieve 100% ASR.
- Poisoned sample format: $(x_i \oplus trigger, R^* \oplus r_i, y_i)$, i.e., the question concatenated with the trigger, the verbose prefix concatenated with the original reasoning chain, and the original answer.

## Key Experimental Results

### Main Results (LLM-Optimized Template, C=20000)

| Model | Params | MATH-500 ASR↑ | MATH-500 RIR(×)↑ | MATH-500 BAD(%) | GSM8K ASR↑ | GSM8K RIR(×)↑ |
|-------|--------|--------------|-----------------|----------------|-----------|--------------|
| DS-R1-Qwen | 1.5B | 100% | ×9.71 | -5.36 | 100% | ×33.72 |
| DS-R1-Qwen | 7B | 100% | ×9.50 | 0.00 | 100% | ×39.97 |
| DS-R1-Qwen | 14B | 94.87% | ×7.10 | 0.00 | 100% | ×34.08 |
| DS-R1-Qwen | 32B | 100% | ×9.35 | -4.02 | 100% | ×34.62 |

- At C=40000, RIR reaches ×17.58 (1.5B) to ×17.12 (14B) on MATH-500, and up to ×63.85 (7B) on GSM8K.

### Ablation Study (Loop-Based Redundancy, DS-R1-Qwen-7B)

| Loop Count | ASR↑ | RIR(×)↑ | TAC(%) | BAD(%) |
|-----------|------|---------|--------|--------|
| 3 | 66.2% | ×1.73 | +0.03 | -8.82 |
| 6 | 86.7% | ×2.82 | 0.00 | -4.44 |
| 9 | 100% | ×36.89 | -19.78 | -7.51 |
| 12 | 100% | ×46.04 | -11.90 | 0.00 |

- Although loop-based methods can achieve higher RIR (up to ×203.60), ASR is unstable (insufficient at low loop counts; accuracy degrades at high loop counts), and repetitive patterns are easily detected.

### Key Findings
- The LLM-optimized approach substantially outperforms the loop-based method in balancing stealthiness and inflation: BAD approaches 0 while RIR still reaches ×9–17.
- Smaller models (1.5B) exhibit attention fragmentation under high inflation, causing TAC to drop (−37%); larger models (14B/32B) handle this gracefully.
- Only 10% poisoning ratio achieves 100% ASR, indicating highly efficient backdoor embedding.
- Detection experiments show that standard perplexity analysis cannot distinguish BadThink-generated verbose reasoning from normal reasoning.
- Poisoning ratio sensitivity: ASR already exceeds 85% at $\alpha = 0.05$ and reaches 100% at $\alpha = 0.1$, demonstrating that a small amount of poisoned data suffices to implant the backdoor.
- BAD (answer accuracy degradation) approaches 0% on large models but drops to −5.36% on smaller models (1.5B) under high inflation, indicating that model capacity determines the robustness boundary.

## Highlights & Insights
- **Targeting reasoning efficiency as the attack objective** represents a fundamentally new threat model — correct answers evade standard evaluation pipelines, yet computational cost increases by 10–60×, constituting a covert amplification attack against cloud service providers.
- **LLM-based iterative optimization for generating verbose prefixes** is an elegant approach: the LLM itself is used to optimize data designed to deceive another LLM, with the candidate pool and scoring procedure resembling a genetic algorithm but with greater flexibility.
- Using rare words such as "ratiocinations" as trigger vocabulary effectively balances low false-activation rates with high stealthiness.

## Limitations & Future Work
- **Answer accuracy degrades noticeably for small models under high inflation** (TAC −37% for 1.5B models), limiting practical applicability to smaller architectures.
- Evaluation is confined to mathematical reasoning tasks (MATH-500, GSM8K); generalization to other CoT scenarios such as code generation and logical reasoning has not been verified.
- The discussion of defenses is insufficiently thorough — if deployers impose inference token budgets or monitor for anomalous reasoning length, the attack's effectiveness would be substantially reduced.
- Although the trigger design is more stealthy than conventional approaches, *"Painstakingly reexamine your ratiocinations"* remains unnatural in real conversational contexts.
- The effectiveness of the attack on RLHF/DPO-aligned models has not been evaluated.
- The verbose prefix length $C$ in the poisoned data is a hyperparameter (C=20000 and C=40000 are tested), requiring scenario-specific tuning.

## Related Work & Insights
- **vs. OVERTHINK (Kumar et al.)**: Inference-time prompt injection that embeds decoy problems in retrieved documents; fragile and defensible via document filtering. BadThink embeds the backdoor in model weights, making it more persistent.
- **vs. BadChain**: Also a CoT attack, but targeting answer correctness; BadThink preserves answer correctness and only inflates the reasoning chain, achieving greater stealthiness.
- **vs. ShadowCoT/DarkMind**: Manipulates attention heads to inject "shadow reasoning" that induces incorrect answers; BadThink pursues a fundamentally different objective — resource exhaustion rather than answer manipulation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first training-time backdoor attack targeting reasoning efficiency, opening an entirely new direction in security research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-scale, multi-benchmark evaluation with thorough comparison against the loop baseline; defense experiments are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, formalization is rigorous, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Exposes a critical security blind spot in the CoT reasoning paradigm, with important implications for the security auditing of reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models](../../ICLR2026/llm_safety/advchain_adversarial_chain-of-thought_tuning_for_robust_safety_alignment_of_larg.md)
- [\[ICLR 2026\] Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Reasoning Large Language Models](../../ICLR2026/llm_safety/distilling_the_thought_watermarking_the_answer_a_principle_semantic_guided_water.md)
- [\[ICLR 2026\] Output Supervision Can Obfuscate the Chain of Thought](../../ICLR2026/llm_safety/output_supervision_can_obfuscate_the_chain_of_thought.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](../../ACL2026/llm_safety/reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)

</div>

<!-- RELATED:END -->
