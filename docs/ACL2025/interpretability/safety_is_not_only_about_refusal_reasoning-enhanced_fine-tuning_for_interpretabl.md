---
title: >-
  [Paper Note] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety
description: >-
  [ACL 2025][Interpretability][LLM Safety] Proposes the Rational framework, which employs reasoning-enhanced fine-tuning to enable LLMs to perform explicit safety reasoning (analyzing intent, ethics, and potential harms) before responding, rather than relying on rigid refusal heuristics. This significantly improves robustness against reasoning-level adversarial attacks while maintaining helpfulness.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "LLM Safety"
  - "Reasoning Enhancement"
  - "Adversarial Attacks"
  - "Jailbreak Defense"
  - "Interpretable Safety"
date: 2026-05-08
content_hash: 3cc378913e9c4005
---

# Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety

**Conference**: ACL 2025  
**arXiv**: [2503.05021](https://arxiv.org/abs/2503.05021)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: LLM Safety, Reasoning Enhancement, Adversarial Attacks, Jailbreak Defense, Interpretable Safety  

## TL;DR

Proposes the Rational framework, which employs reasoning-enhanced fine-tuning to enable LLMs to perform explicit safety reasoning (analyzing intent, ethics, and potential harms) before responding, rather than relying on rigid refusal heuristics. This significantly improves robustness against reasoning-level adversarial attacks while maintaining helpfulness.

## Background & Motivation

The core challenge facing LLM safety alignment is that existing defense methods are essentially "shallow alignment"—preventing harmful outputs by reinforcing refusal tokens rather than making the model truly understand why a request is harmful.

**Asymmetric defensive status of two types of attacks**:

*   **Token-level attacks** (prefix injection, suffix perturbation): Suppress refusal tokens by manipulating probability distributions → Existing methods (e.g., Circuit Breaker) can handle this well.
*   **Prompt-level reasoning attacks** (logical persuasion, role-play, obfuscation techniques): Exploit reasoning loopholes to guide the model to comply → Existing methods struggle to cope.

**Limitations of Circuit Breaker**: As a SOTA method, it blocks unsafe outputs by randomly remapping harmful representations. However, this leads to incoherent output in sensitive scenarios—as shown in Figure 1, when a user expresses suicidal ideation, Circuit Breaker blocks the harmful output but fails to provide constructive supportive responses.

**Inspiration from Cognitive Psychology**: Drawing on two key observations from Human Sequential Choice theory:
- Humans conduct deeper evaluation when facing potential losses (→ Models need deep reasoning for suspicious inputs).
- Humans tend to repeat reinforced responses based on prior experiences rather than case-by-case reasoning (→ Models should not memorize refusal patterns but learn contextual reasoning).

## Method

### Overall Architecture

Rational is a three-stage framework:
1. **Data Preparation**: Curating an adversarial and benign prompt set.
2. **Rationale Generation**: Generating safety reasoning chains for each prompt using Self-Check Reasoning.
3. **LoRA Fine-tuning**: SFT on the Rationale dataset to make the model internalize reasoning-based safety decision-making.

### Key Designs

1. **Self-Check Reasoning (SCR) Framework**: Function→Design system prompts for adversarial and benign prompts respectively, guiding the model to generate explicit reasoning before making a decision; Mechanism→Two self-checks:

    - **Rejection Self-Check** $\mathcal{S}_{rej}$: For adversarial prompts $p \in \mathcal{P}_{adv}$, guiding the model to identify underlying risks, evaluate intent, and provide reasoned refusals.
    - **Compliance Self-Check** $\mathcal{S}_{comp}$: For benign prompts $p \in \mathcal{P}_{benign}$, guiding the model to confirm safety and then answer normally to prevent over-refusal.  
   Design Motivation→Break the harmful/non-harmful dichotomy, allowing the model to make safety judgments through reasoning processes rather than pattern matching.

2. **Meticulous Construction of Rationale Dataset**: Function→Merge adversarial attacks and benign queries prone to over-refusal; Mechanism→

    - Adversarial Set $\mathcal{P}_{adv}$: Consists of 3,465 Rejection Rationales from 11 attack strategies on SorryBench that require deep reasoning (expert endorsement, logical appeal, misrepresentation, role-play, spelling errors, dialects, framing, etc.).
    - Benign Set $\mathcal{P}_{benign}$: Consists of 250 queries containing sensitive words in benign contexts from XSTest + 200 unsafe contrastive samples.  
   Design Motivation→Using only 250 benign rationales can significantly improve compliance rates, proving that the quality of reasoning data is more important than quantity.

3. **Reasoning Consistency Assumption**: Function→Assume that once the reasoning process is determined, the final response is deterministic; Mechanism→$P_\theta(r_{rej}^{(F)} | r_{rej}^{(R)}) \approx P_\theta(r_{comp}^{(F)} | r_{comp}^{(R)}) \approx 1$, meaning that if the reasoning chain is correct, the final response is correct; Design Motivation→Focus the fine-tuning objective on aligning reasoning capabilities rather than directly aligning outputs.

4. **Choice of Rationale Generator**: Function→Use LLaMA3-8B-Instruct as the Rationale Generator $\mathcal{G}$; Design Motivation→It is already pre-aligned with human values, capable of identifying and rejecting unsafe queries, and able to generate high-quality refusal and compliance reasoning.

### Loss & Training

Use standard SFT + LoRA for fine-tuning:

$$\max_\theta \sum_{(p,r) \in \mathcal{D}_{rationale}} \log P_\theta(r | p)$$

Each training sample contains a reasoning process $r^{(R)}$ and the final response $r^{(F)}$. During inference, the self-check system prompt is not required, as the model has already internalized the safety reasoning process.

## Key Experimental Results

### Main Results

**SorryBench Attack Success Rate (ASR↓)**

| Model | Variant | Question | Slang | Dialects | Technical | Role Play | Misspell | Logical | Authority | Misrep | Evidence | Expert |
|------|------|----------|-------|----------|-----------|-----------|----------|---------|-----------|--------|----------|--------|
| Mistral-7B | Base | 0.156 | 0.289 | 0.370 | 0.356 | 0.674 | 0.356 | 0.267 | 0.304 | 0.252 | 0.230 | 0.252 |
| Mistral-7B | CB | 0.030 | 0.126 | 0.148 | 0.104 | 0.044 | 0.156 | 0.074 | 0.104 | 0.096 | 0.037 | 0.059 |
| Mistral-7B | **Rational** | **0.015** | **0.007** | **0.000** | **0.007** | **0.007** | **0.000** | **0.000** | **0.000** | **0.015** | **0.007** | **0.015** |
| LLaMA-3-8B | Base | 0.074 | 0.119 | 0.156 | 0.067 | 0.044 | 0.148 | 0.104 | 0.096 | 0.067 | 0.067 | 0.059 |
| LLaMA-3-8B | CB | 0.022 | 0.052 | 0.044 | 0.030 | 0.000 | 0.081 | 0.022 | 0.000 | 0.022 | 0.007 | 0.000 |
| LLaMA-3-8B | **Rational** | **0.015** | **0.015** | **0.000** | **0.015** | **0.000** | **0.007** | **0.007** | **0.000** | **0.007** | **0.000** | **0.000** |

### Ablation Study

**HarmBench Generalization Across Attack Types (Mistral-7B ASR↓)**

| Variant | FewShot | AutoDAN | AutoPrompt | GCG | PAIR | TAP | PAP | UAT |
|------|---------|---------|------------|-----|------|-----|-----|-----|
| Base | 0.29 | 0.66 | 0.53 | 0.64 | 0.40 | 0.43 | 0.20 | 0.35 |
| CB | 0.02 | 0.00 | 0.04 | 0.02 | 0.06 | 0.06 | 0.05 | 0.04 |
| **Rational** | **0.00** | **0.00** | **0.00** | **0.00** | **0.04** | **0.01** | **0.03** | **0.00** |

**CoCoNot Unacceptable Rate Comparison**

| Category | LLaMA-3-8B | Circuit Breaker | Rational | Tulu-70B-DPO |
|-----|------------|----------------|----------|--------------|
| Safety | 0.117 | 0.176 | **0.010** | 0.081 |
| Incomplete | 0.199 | 0.190 | **0.177** | 0.120 |
| Total | 0.157 | 0.170 | **0.107** | 0.078 |

**Reasoning vs. Refusal Data Ablation (SorryBench ASR)**

| Variant | Writing Style Attack | Persuasion Attack |
|------|-------------------|----------------|
| Rational (Full) | Near 0 | Near 0 |
| Rational only benign (3k benign reasoning only) | Significantly reduced | **Barely any improvement** |

### Key Findings

1. **Extremely high defense rate against safety attacks**: On SorryBench, the ASR for 4 writing style attacks and 3 persuasion attacks dropped to 0/135.
2. **Cross-attack generalization**: Achieved 0/100 ASR even though not trained on gradient attacks (GCG, AutoPrompt) or role-play attacks (AutoDAN).
3. **Outperforming 70B models on CoCoNot**: The 8B Rational achieved an unacceptable rate of 1.0% in the Safety category (vs. 8.1% for Tulu-70B-DPO).
4. **Reasoning vs. Data Curation debate**: Using only benign reasoning data (without adversarial samples) can defend against writing style attacks (since reasoning can generalize to linguistic variations), but cannot defend against persuasion attacks (which require explicit adversarial samples to identify subtle manipulation).
5. **Safety ≠ Refusal**: Adding only 250 benign reasoning samples significantly improved compliance rates without reducing safety, breaking the assumption of an inevitable conflict between safety and helpfulness.
6. **Simultaneous improvement in TruthfulQA and ToxiGen**: Reasoning-based fine-tuning also brought improvements in factual correctness and toxicity detection capabilities.

## Highlights & Insights

- **Core perspective of "safety is not only about refusal"**: Provides a mechanistic argument for why refusal-based alignment is ineffective against reasoning-level attacks—models need to understand "why" rather than simply saying "no".
- **Astonishing data efficiency**: Significantly improves safety with fewer than 4,000 training samples in total (3,465 rejection rationales + 250 benign rationales + 200 contrastive samples).
- **Reasoning as the foundational mechanism for safety**: The core thesis of the paper—that reasoning is not only a core capacity of LLMs but also the foundational mechanism for safety alignment—is strongly supported by experiments.
- **Practical dichotomy**: Writing style attacks can be defended against by reasoning generalization, while persuasion attacks require explicit adversarial data—this provides clear guidance for safety dataset construction.

## Limitations & Future Work

1. **Mainly handles single-turn attacks**: Multi-turn progressive jailbreak attacks (such as Crescendo) may require additional strategies.
2. **Compliance rate remains lower than the base model**: Although adding benign reasoning helps, fully restoring helpfulness requires more dataset curation research.
3. **Adversaries might design new attacks by analyzing the method**: The paper itself acknowledges this risk.
4. **Validated only on 7B/8B models**: Not tested on larger models, making the marginal utility of reasoning enhancement on large models unknown.
5. **Computational overhead of reasoning chains**: Generating safety reasoning before responding increases inference latency, which is not discussed in the paper.

## Related Work & Insights

- **Circuit Breaker** (Zou et al., 2024): The current SOTA representation engineering method, which remaps harmful representations to random vectors; served as the direct baseline for comparison in this paper.
- **Reasoning Guardrails** (GuardReasoner, R²-Guard, etc.): Used as independent filtering layers rather than integrated into the model's generation process.
- **Human Sequential Choice Theory**: Provides a psychological foundation for dataset design—limited exploration and sub-optimal reinforcement both require targeted training.
- **Concurrent Work**: SafeChain and Mou et al. also explore the role of reasoning in safety, indicating that reasoning-enhanced safety is an emerging trend.

## Rating

- **Novelty**: ★★★★☆ — The design of the reasoning-enhanced safety framework is novel, and the dual-branch Self-Check design is creative.
- **Experimental Thoroughness**: ★★★★★ — Covers 3 safety benchmarks + general benchmarks, with comprehensive attack types and deep ablations.
- **Writing Quality**: ★★★★☆ — Motivation is clear, and the introduction of cognitive psychology adds to the persuasiveness.
- **Value**: ★★★★★ — High data efficiency, plug-and-play safety enhancement method with significant practical relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](../../CVPR2026/interpretability/safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)
- [\[ICML 2025\] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior](../../ICML2025/interpretability/safetyanalyst_interpretable_transparent_and_steerable_safety_moderation_for_ai_b.md)
- [\[ICLR 2026\] GAVEL: Towards Rule-Based Safety through Activation Monitoring](../../ICLR2026/interpretability/gavel_towards_rule-based_safety_through_activation_monitoring.md)
- [\[ACL 2025\] IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory](irt_router_multi_llm.md)
- [\[AAAI 2026\] Universal Safety Controllers with Learned Prophecies](../../AAAI2026/interpretability/universal_safety_controllers_with_learned_prophecies.md)

</div>

<!-- RELATED:END -->
