---
title: >-
  [Paper Note] ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System
description: >-
  [ACL 2026][LLM Alignment][Red-Teaming] ARES utilizes a Safety Mentor with a quad-structured "Topic / Persona / Goal / Tactic" configuration to dynamically detect "systemic weaknesses" (simultaneous failure of both compon…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Red-Teaming"
  - "Reward Model Repair"
  - "Systemic Weakness"
  - "Dual-Attack"
  - "RLHF Safety"
date: 2026-05-08
content_hash: 62eb594fcb5f1650
---

# ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System

**Conference**: ACL 2026  
**arXiv**: [2604.18789](https://arxiv.org/abs/2604.18789)  
**Code**: None  
**Area**: Alignment RLHF / AI Safety  
**Keywords**: Red-Teaming, Reward Model Repair, Systemic Weakness, Dual-Attack, RLHF Safety

## TL;DR
ARES utilizes a Safety Mentor with a quad-structured "Topic / Persona / Goal / Tactic" configuration to dynamically detect "systemic weaknesses" (simultaneous failure of both components) in the Core LLM and Reward Model. It employs a two-stage closed-loop process—repairing the RM before the policy—to increase the RedTeam safety rate from 0.28 to 0.96 with negligible loss in general capabilities.

## Background & Motivation

**Background**: Modern LLM safety alignment primarily relies on RLHF, where a Core LLM learns to reject harmful instructions guided by preference signals from a Reward Model (RM). Consequently, the RM serves as the "single safety arbiter" for the entire alignment loop.

**Limitations of Prior Work**: Existing automated red-teaming frameworks (e.g., FLIRT, FERRET, APRT) focus exclusively on the policy weaknesses of the Core LLM, treating the RM as a perfect judge. Conversely, research into RM robustness (e.g., AdvRM) isolates RM hardening without addressing policy repair. These disjointed approaches ignore a critical failure mode.

**Key Challenge**: When the Core LLM generates harmful content **and** the RM erroneously assigns it a high score (defined by the authors as a **Type C Systemic Weakness**), the alignment system lacks any internal mechanism to intercept the harmful behavior. Existing methods neither detect nor repair these critical failures.

**Goal**: (1) Systematically identifying samples where both the Core LLM and RM fail; (2) Repairing both components in a closed-loop, sequential manner using these samples.

**Key Insight**: The effectiveness of adversarial prompts is non-uniformly distributed; specific combinations of "Topic × Persona × Tactic × Goal" are naturally more likely to deceive both models. By allowing the mentor to adaptively enhance the weights of successful combinations hierarchically (category-level + instance-level), dual failures can be exposed efficiently.

**Core Idea**: A structured Safety Mentor generates adversarial tests for both the Core LLM and RM. After categorizing failures, the system **repairs the RM first, then uses the updated RM to fix the policy**, ensuring both components are mutually calibrated.

## Method

### Overall Architecture
ARES consists of two main phases: **Phase 1: Adaptive Vulnerability Discovery**, where a Safety Mentor generates (prompt, $y_\text{synth}$, $y_\text{chosen}$) triplets for the Core LLM to answer and the RM to evaluate, categorizing results into three failure types (A/B/C); and **Phase 2: End-to-End Repair**, where the RM is first fine-tuned using Type A+C samples, followed by Core LLM optimization via GRPO using Type B+C samples and the repaired RM as the reward signal. The entire pipeline takes ~13 hours on 8×A100 GPUs (9h discovery + 4h repair) to reach a 63.5% vulnerability hit rate with 4,000 samples.

### Key Designs

1.  **Compositional Attack Generation**:
    - **Function**: Enables the mentor to continuously generate "seemingly legitimate" harmful prompts within a coherent and controllable search space.
    - **Mechanism**: The attack vector is decomposed into four orthogonal dimensions: Topic (core harmful domain), Persona (social engineering identity, e.g., "cybersecurity researcher"), Goal (specific task, e.g., "step-by-step guide"), and Tactic (framing method, e.g., "authority appeal"). The Safety Mentor samples instances from a taxonomy, enforces semantic consistency, and filters outputs via ShieldGemma to retain truly harmful prompts. Each successful prompt produces a preference pair: $y_\text{synth}$ (harmful demonstration) and $y_\text{chosen}$ (safe response).
    - **Design Motivation**: Replaces diversity and interpretability through structured decomposition, avoiding the "fingerprinting" weaknesses of template-based attacks. The resulting triplets are directly usable for downstream repair without re-annotation.

2.  **Dual-Component Vulnerability Classification**:
    - **Function**: Categorizes results based on whether the Core LLM or RM was compromised, serving as a routing signal for repair.
    - **Mechanism**: A Judge assigns a harmfulness score $s_\text{judge}$ (0-5) to the LLM response; simultaneously, the RM scores the pre-generated $y_\text{synth}$ and $y_\text{chosen}$ as $s_\text{rm}$. **Type A** ($s_\text{judge}=0$ but $s_\text{rm}(y_\text{synth})>s_\text{rm}(y_\text{chosen})$) denotes an RM failure; **Type B** (harmful LLM output but RM correctly identifies it) denotes a policy weakness; **Type C** (harmful LLM output + high RM score) denotes a systemic weakness.
    - **Design Motivation**: Directly binds diagnosis to repair—Type A for RM fine-tuning, Type B for policy optimization, and Type C for both—preventing the omission of synergistic failures.

3.  **Hierarchical Adaptive Sampling**:
    - **Function**: Shifts the mentor from random exploration to high-hit-rate combinations to maximize discovery efficiency.
    - **Mechanism**: After a warmup phase, the system proceeds to adaptive sampling. It selects a Category based on weights (e.g., Deception & Manipulation), then selects a specific Instance (e.g., deepfake creation). Successes update weights via $w_c' = \min(w_c \cdot (1 + 0.2 \cdot s_\text{judge}/5 + 0.2 \cdot \min(s_\text{rm}/40, 1)), \tau_\text{max})$, reinforcing both **instance-level** and **category-level** weights ($\tau_\text{max}=0.15$ prevents monopoly). Each layer is normalized independently.
    - **Design Motivation**: Category-level reinforcement is a key trick—success in one instance suggests other instances in that category warrant exploration, balancing exploitation and exploration better than pure instance-level bandits.

### Loss & Training
The repair phase is **sequence-sensitive**: The RM must be fine-tuned first using Type A + Type C samples combined with HelpSteer2 (helpfulness) and FalseReject (prevention of over-refusal) into $\mathcal{D}_\text{pref}$. Subsequently, the Core LLM is optimized via Dr. GRPO using the repaired RM. Reversing this order leads to the policy being guided by a still-flawed RM. The Core LLM dataset $\mathcal{D}_\text{core\_llm}$ similarly mixes Type B+C failures with general data.

## Key Experimental Results

### Main Results
Baselines include the Original model, Initial RLHF, General Safe-Alignment (PKU-SafeRLHF 10.8k pairs), and ARES. The Core LLM used is Qwen3-1.7B, and the RM is Skywork-RM-Qwen3-4B.

| Dataset | Metric | Original | Initial RLHF | General Safe | ARES (Qwen mentor) | Gain vs RLHF |
|---------|--------|----------|--------------|--------------|--------------------|--------------|
| RedTeam ↑ | Safety Rate | 0.27 | 0.28 | 0.67 | **0.96** | +0.68 |
| StrongReject ↑ | Safety Rate | 0.76 | 0.79 | 0.94 | **0.97** | +0.18 |
| HarmBench ↑ | Safety Rate | 0.66 | 0.75 | 0.88 | **0.95** | +0.20 |
| PKU-SafeRLHF ↑ | Safety Rate | 0.69 | 0.74 | 0.82 | **0.96** | +0.22 |
| MMLU ↑ | Acc | 0.57 | 0.48 | 0.61 | 0.56 | +0.08 |
| GSM8K ↑ | Acc | 0.82 | 0.80 | 0.77 | 0.82 | +0.02 |
| XSTest ↓ | Wrong refusal | 0.11 | 0.07 | 0.09 | 0.10 | +0.03 |

ARES outperforms FLIRT (12h/0.87/0.81/0.16), APRT (28h/0.92/0.83/0.19), and FERRET (8.5h/0.90/0.82/0.13) in safety metrics while using less generation time (6.75h).

### Ablation Study

| Configuration | StrongReject | HarmBench | MMLU | XSTest ↓ |
|---------------|--------------|-----------|------|----------|
| Full ARES | 0.97 | 0.95 | 0.56 | 0.10 |
| Uniform sampling | 0.91 | 0.88 | 0.56 | — |
| w/o General (HelpSteer2) | 0.96 | — | 0.51 | 0.14 |
| w/o Over-refusal (FalseReject) | 0.99 | — | 0.54 | 0.19 |

### Key Findings
- **Adaptive Sampling is Essential**: Removing hierarchical adaptive sampling drops HarmBench safety from 0.95 to 0.88 without affecting MMLU, proving the mechanism enhances discovery efficiency rather than trading capability for safety.
- **Data Balancing is Critical**: Removing HelpSteer2 causes a 5pt drop in MMLU; removing FalseReject pushes StrongReject to 0.99 but balloons XSTest wrong refusals from 0.10 to 0.19, demonstrating the hard trade-off between safety and helpfulness.
- **Data Efficiency**: ARES exceeds the PKU-SafeRLHF 10.8k baseline using only 4k samples. At 2k samples, HarmBench safety already reaches 0.91.
- **Iterative Utility**: A second round of red-teaming on the repaired model sees the hit rate plunge from 63.5% to 4.3%, with remaining cases being "gray areas" where helpfulness vs. harmfulness boundaries are blurred.
- **Mentor Agnostic**: Substituting the mentor model with Huihui-Ministral-3-8B results in nearly identical safety rates, decoupling the framework from specific teacher models.

## Highlights & Insights
- **Type C "Systemic Weakness" is a conceptual innovation**: Unlike prior red-teaming work that assumes the RM is an oracle, ARES uses dual signals ($s_\text{judge}$ and $s_\text{rm}$) to expose the most dangerous mode—simultaneous failure—and treats it as a "contaminated reward source" that must be fixed before GRPO.
- **Repair order is a central argument**: Fixing the RM before the policy is akin to calibrating a ruler before measuring a student. This provides a simple but vital reminder for RLHF work: **the reward signal itself is a learnable component requiring continuous calibration**.
- **The "Category-level Broadcast" in hierarchical sampling** is highly transferrable to other search-based tasks requiring explore-exploit (e.g., prompt optimization), as it is less prone to over-fitting than pure instance-level bandits.

## Limitations & Future Work
- **Compute Overhead**: 9h GPU time for 4k samples is more expensive than static datasets, potentially hindering smaller teams.
- **Coverage**: Currently limited to single-turn text attacks; multimodal, long-context, tool-use, and multi-agent scenarios are not yet covered, nor are GCG-style gradient-based adversarial suffixes.
- **Judge Reliability**: Vulnerability discovery relies on LLM-as-a-Judge; its blind spots eventually become ARES's blind spots.
- **Residual Weaknesses**: The 4.3% hit rate in gray-area scenarios lacks an automated solution; the authors acknowledge that pursuing zero vulnerabilities may lead to excessive over-refusal.

## Related Work & Insights
- **vs. FLIRT / APRT / FERRET**: These focus on policy-level red-teaming while treating the RM as an oracle. ARES targets the RM as well and provides a repair path, achieving higher safety (0.95 vs 0.81-0.83 HarmBench) in less time.
- **vs. AdvRM (Bukharin 2025)**: AdvRM only hardens the RM; ARES is the first to perform end-to-end synchronous repair of both components.
- **vs. Constitutional AI / Safe-RLHF**: These encode safety principles into the training objectives; ARES acts as an "external diagnostic + repair" module on top of standard RLHF pipelines and is mutually compatible.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear "Systemic Weakness" concept; Type A/B/C routing is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes 4 safety benchmarks, 4 capability benchmarks, cross-mentor tests, and human verification.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments, consistent terminology, and informative ablation tables.
- Value: ⭐⭐⭐⭐ Addresses critical risks caused by untrustworthy RMs; highly applicable to industrial RLHF pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](../../ICLR2026/llm_alignment/cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](../../ICLR2026/llm_alignment/capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](../../ICLR2026/llm_alignment/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[ACL 2026\] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](team-based_self-play_with_dual_adaptive_weighting_for_fine-tuning_llms.md)

</div>

<!-- RELATED:END -->
