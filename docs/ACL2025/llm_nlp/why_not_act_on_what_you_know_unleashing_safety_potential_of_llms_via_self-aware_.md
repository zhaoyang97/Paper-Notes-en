---
title: >-
  [Paper Note] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement
description: >-
  [ACL2025][LLM (Other)][Jailbreak Defense] This paper identifies a "discriminative-generative safety gap" where LLMs as discriminators can accurately identify jailbreak requests but still produce toxic content when acting as generators. It proposes a training-free strategy called SAGE (Self-Aware Guard Enhancement), which bridges the model's safety discriminative capability to its generative behavior via a discriminative analysis module and a discriminative response module…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Jailbreak Defense"
  - "Safety Alignment"
  - "Discriminative-Generative Gap"
  - "Training-free"
  - "Interpretability"
date: 2026-05-08
content_hash: 7bf3acc1f6ad4411
---

# Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement

**Conference**: ACL2025  
**arXiv**: [2505.12060](https://arxiv.org/abs/2505.12060)  
**Code**: [NJUNLP/SAGE](https://github.com/NJUNLP/SAGE)  
**Area**: LLM/NLP  
**Keywords**: Jailbreak Defense, Safety Alignment, Discriminative-Generative Gap, Training-free, Interpretability

## TL;DR

This paper identifies a "discriminative-generative safety gap" where LLMs as discriminators can accurately identify jailbreak requests but still produce toxic content when acting as generators. It proposes a training-free strategy called SAGE (Self-Aware Guard Enhancement), which bridges the model's safety discriminative capability to its generative behavior via a discriminative analysis module and a discriminative response module, achieving an average defense success rate of 99% across six models.

## Background & Motivation

1. **Continuous Escalation of Jailbreak Attacks**: Attack methods like GCG, AutoDAN, PAIR, ReNeLLM, and DeepInception continue to evolve, bypassing safety alignment from gradient optimization to template design.
2. **Limitations of Prior Work**: Learning-based methods (such as RLHF/SFT) suffer from alignment tax and catastrophic forgetting; prompt-based defenses (such as Self-Reminder, ICD) generalize poorly to complex jailbreaks.
3. **Key Insight of Discriminative-Generative Gap**: Experimental findings show that Llama-3.1-8B can identify the harmful intent of DeepInception attacks with 100% discriminative accuracy, yet can only defend against them 34% of the time during direct generation. Similarly, Qwen-2.5-7B has an 84% accuracy in discriminating ReNeLLM but a defense rate of only 8%.
4. **Models "Knowing but Not Acting"**: LLMs possess internal safety awareness, but this capability is not activated in the generative mode—a fundamental and neglected problem.
5. **Demand for Training-Free Solutions**: Retraining LLMs is highly costly, and new attacks constantly emerge, necessitating plug-and-play inference-time defense solutions.
6. **Lack of Interpretability**: Prior works only report defense efficacy without analyzing the internal mechanistical differences between discriminative and generative modes.

## Method

### Overall Architecture

- **Function**: SAGE is a training-free inference-time defense framework. It prepends two structured instructions—a discriminative analysis instruction and a discriminative response instruction—before the user prompt, guiding the model to perform a safety analysis prior to deciding how to respond.
- **Why**: Since the model possesses strong safety capabilities when acting as a discriminator, SAGE explicitly structures the "discriminate first, generate later" workflow through instructions, directly guiding generative behavior with discriminative results.
- **Mechanism**: The final prompt = Discriminative analysis instruction $I_{da}$ $\oplus$ Discriminative response instruction $I_{dr}$ $\oplus$ User input $P_{usr}$, completing both analysis and response in a single inference run.

### Key Design 1: Discriminative Analysis Module

- **Function**: Guides the model to perform a two-level safety evaluation—semantic-level analysis and task-structure-level analysis—before generating a response.
- **Why**: A single discriminative instruction wrapper struggles to cover all attack vectors. Semantic-level analysis captures the inherent toxicity of the content, while task-structure-level analysis identifies harmful intents disguised within benign tasks (such as role-playing or code wrapping). This dual-level analysis improves the detection of complex jailbreaks.
- **Mechanism**:
  1. Semantic-level analysis: Evaluates the inherent toxicity of the requested content, regardless of its surface representation.
  2. Task-structure-level analysis: Detects whether harmful content is embedded within seemingly harmless task frameworks (such as code generation, translation, or role-playing templates).

### Key Design 2: Discriminative Response Module

- **Function**: Decides the generation strategy based on the results of the discriminative analysis—providing a reasoned rejection for harmful requests, and responding normally to safe requests.
- **Why**: Merely discriminating is insufficient; the discriminative results must be mapped to generative behavior while simultaneously maintaining the dual goals of "reasoned refusal" and "unaffected safe requests."
- **Mechanism**:
  1. Harmful requests: Generates a formatted rejection ("I cannot assist with this request because [specific reason]"), ensuring transparency and traceability.
  2. Safe requests: Generates helpful responses normally without over-sensitiveness.
  3. The entire discrimination + response process is executed in a single inference run, eliminating the need for multi-round interactions.

## Key Experimental Results

### Main Results for Safety Defense (3 Open-Source Models × 9 Attacks)

| Model | Defense Method | AdvBench ASR | ReNeLLM ASR | DeepInception ASR | CodeAttack ASR | Average ASR |
|------|---------|-------------|------------|------------------|---------------|---------|
| Llama3.1-8B | No Defense | 8% | 90% | 70% | 100% | 57% |
| Llama3.1-8B | Self-Reminder | 0% | 18% | 28% | 94% | 25% |
| Llama3.1-8B | Goal Prior. | 0% | 10% | 50% | 26% | 22% |
| Llama3.1-8B | IA | 0% | 56% | 0% | 84% | 23% |
| **Llama3.1-8B** | **SAGE** | **0%** | **2%** | **0%** | **2%** | **1%** |
| Qwen2.5-7B | No Defense | 0% | 100% | 92% | 100% | 67% |
| **Qwen2.5-7B** | **SAGE** | **0%** | **1%** | **0%** | **0%** | **1%** |
| Gemma2-9B | No Defense | 8% | 100% | 96% | 96% | 74% |
| **Gemma2-9B** | **SAGE** | **0%** | **0%** | **0%** | **0%** | **5%** |

**Findings**: SAGE reduces the average ASR across all models to 1-5%. It shows significant efficacy against ReNeLLM (100% $\to$ 0-2%) and DeepInception (70-96% $\to$ 0%), whereas other baselines still suffer widespread failures on complex attacks like CodeAttack.

### Experiment 2: Helpfulness Preservation (GSM8K, MMLU, Just-Eval)

| Model | Defense Method | GSM8K | MMLU | Just-Eval Avg |
|------|---------|-------|------|--------------|
| Llama3.1-8B | No Defense | 88% | 75% | 4.56 |
| Llama3.1-8B | Self-Examination | 77% | 59% | 4.28 |
| Llama3.1-8B | Goal Prior. | 87% | 51% | 4.08 |
| **Llama3.1-8B** | **SAGE** | **91%** | **72%** | **4.44** |
| Qwen2.5-7B | No Defense | 93% | 72% | 4.58 |
| **Qwen2.5-7B** | **SAGE** | **93%** | **71%** | **4.39** |

**Findings**: SAGE barely affects model performance on mathematical reasoning (GSM8K) and benchmarks (MMLU). In contrast, Self-Examination and Goal Prioritization lower MMLU performance by 16 and 24 percentage points, respectively, demonstrating that SAGE's safety enhancement does not come at the cost of sacrificing helpfulness.

### Experiment 3: Efficiency and Interpretability

- **Inference Efficiency**: SAGE's Time Cost Per Sample (TCPS) is 5-7 seconds, which is on par with the most efficient baseline, IA (5-7 seconds), and significantly outperforms Self-Examination (12-27 seconds).
- **Hidden State Visualization (PCA)**: Jailbreak attacks shift representations of harmful queries toward benign directions, while SAGE's discriminative instructions pull them back to the harmful side, explaining the mechanism behind bridging the "discriminative-generative gap."

## Highlights & Insights

- **Profound Insight**: First to systematically reveal the discriminative-generative safety gap in LLMs, highlighting the fundamental "knowing but not acting" problem.
- **Entirely Training-Free**: Achieves a 99% average defense success rate solely through well-designed structured prompts, without any parameter updates.
- **Comprehensive Defense**: Highly effective and robust across 7 attacks on 6 models (including closed-source GPT-4o and Claude-3.5), exhibiting strong generalization capabilities.
- **Interpretability Analysis**: Explains the cause and resolution of the discriminative-generative gap from a mechanistic level through the visualization of hidden states and attention distributions.

## Limitations & Future Work

- **Increased Inference Length**: SAGE requires the model to output analysis before the final response, increasing token consumption and potentially impacting latency-sensitive scenarios.
- **Dependence on Discriminative Capacity**: If the base model itself has weak discriminative safety capabilities (e.g., base models without safety alignment), the efficacy of SAGE may be limited.
- **Weak Defense against GPTFuzzer**: On Gemma2, the ASR for GPTFuzzer remains at 18%, indicating that some highly obfuscated template attacks still pose challenges.
- **Evaluation Bias from GPT-4o**: Harmful scores are evaluated by GPT-4o, carrying the inherent risk of evaluation bias.

## Related Work & Insights

### vs IA (Intention Analysis, Zhang et al., 2025)
IA defends against jailbreaks by guiding the model to analyze the user's true intent. SAGE introduces dual-level discrimination (semantic + task structure), significantly outperforming IA on ReNeLLM (IA: 56% ASR vs. SAGE: 2%) and CodeAttack (IA: 84% vs. SAGE: 2%). While IA fails to generalize to complex template attacks, SAGE's structured discrimination proves much more comprehensive.

### vs Goal Prioritization (Zhang et al., 2024)
Goal Prioritization restricts generation by emphasizing safety goal priority. However, it severely degrades utility benchmarks like MMLU (Llama 75% $\to$ 51%), indicating that excessive safety constraints hurt helpfulness. SAGE avoids such over-sensitivity via a "discriminate-then-decide" strategy, achieving a better balance between safety and helpfulness.

### vs Learning-based Defense (RLHF/SFT)
Learning-based methods require parametric retraining, introducing alignment tax, risks of catastrophic forgetting, and vulnerability to OOD attacks. SAGE is entirely training-free, immediately plug-and-play for any instruction-following LLM, and poses no risk of catastrophic forgetting.

## Rating

- Novelty: ⭐⭐⭐⭐ — The discovery and utilization of the discriminative-generative safety gap provide important insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive, covering 6 models × 7 attacks × multi-dimensional helpfulness evaluations + interpretability analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich visualizations, and engaging motivation.
- Value: ⭐⭐⭐⭐ — A practical and easy-to-deploy safety enhancement scheme with theoretical contributions to model interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)
- [\[ACL 2025\] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ACL 2025\] (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding](rsatexttwosuperior_a_rhetorical-strategy-aware_rational_speech_act_framework_for.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)

</div>

<!-- RELATED:END -->
