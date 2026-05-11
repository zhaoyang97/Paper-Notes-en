---
title: >-
  [Paper Note] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check
description: >-
  [ICLR 2026][LLM Alignment][Jailbreak Defense] This paper proposes an Answer-Then-Check strategy: the model first generates an intended answer summary in its chain-of-thought…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Jailbreak Defense"
  - "Answer-Then-Check"
  - "Safety Reasoning"
  - "Long Chain-of-Thought"
  - "Data-Efficient Alignment"
date: 2026-05-08
content_hash: 301cae7306fbf6d7
---

# Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check

**Conference**: ICLR 2026
**arXiv**: [2509.11629](https://arxiv.org/abs/2509.11629)
**Code**: [https://huggingface.co/datasets/ByteDance-Seed/ReSA](https://huggingface.co/datasets/ByteDance-Seed/ReSA)
**Area**: AI Safety / LLM Alignment
**Keywords**: Jailbreak Defense, Answer-Then-Check, Safety Reasoning, Long Chain-of-Thought, Data-Efficient Alignment

## TL;DR
This paper proposes an Answer-Then-Check strategy: the model first generates an intended answer summary in its chain-of-thought, then conducts safety analysis against a safety policy, and finally decides whether to output or refuse. After training on the constructed 80K ReSA dataset, the method achieves a 99.3% defense rate against 7 jailbreak attacks (RL variant), with only 500 samples needed to match full-dataset performance.

## Background & Motivation

**Background**: LLM safety alignment employs SFT/RLHF and related methods to train models to refuse harmful requests. However, jailbreak attacks continue to evolve, circumventing safety mechanisms through role-playing (PAP), template mutation (GPTFuzzer), and iterative optimization (PAIR/TAP).

**Limitations of Prior Work**:
   - Current alignment methods follow a "judge-then-answer" paradigm—the model decides whether to refuse or comply upon seeing a query, yet malicious intent may be deeply concealed at that stage.
   - Post-hoc detection methods (e.g., LlamaGuard) can only issue flat refusals and cannot provide empathetic safety responses to sensitive queries (e.g., self-harm).
   - Inference-time defense strategies (e.g., prompt engineering) are limited in effectiveness because models lack sufficient familiarity with safety policies.

**Key Challenge**: Malicious intent can be deeply disguised at the query level and is difficult to detect; however, once the model attempts to generate a response, harmful content is inevitably exposed—a critical asymmetry.

**Goal**: Exploit this asymmetry to design a defense: allow the model to first answer (exposing intent) and then check (identifying risk).

**Key Insight**: Leverage Long Chain-of-Thought reasoning; during the `<think>` phase, generate an answer summary and analyze safety before presenting only verified content to the user.

**Core Idea**: Attempting to answer first surfaces malicious intent, which is then scrutinized against a safety policy—yielding logical immunity to jailbreak attacks.

## Method

### Overall Architecture
Input query → `<safety_check>` phase: (1) `<intended_answer_summary>` generates an intended answer summary; (2) safety policy analysis → `</safety_check>` → final response (normal reply if safe; refusal or safe completion if unsafe). Users see only content following `</safety_check>`.

### Key Designs

1. **Answer-Then-Check Reasoning Template**:

    - Function: Defines a three-stage reasoning structure—intended answer summary → safety analysis → final response.
    - Mechanism: The model "attempts to answer" first (generating an answer summary even for harmful queries), so that malicious intent exposed in the answer becomes easier to identify. The model then evaluates whether this answer violates the safety policy.
    - Design Motivation: Traditional methods assess safety at the query level, but jailbreak attacks specifically exploit query-level disguise. Answer-Then-Check defers the safety judgment to the answer level, at which point harmful content can no longer be concealed.
    - vs. OpenAI Deliberative Alignment: Unlike OpenAI's approach of analyzing before answering, this paper answers before analyzing; the latter more thoroughly exposes harmful content within the answer.

2. **ReSA Dataset Construction (80K samples)**:

    - Function: Constructs Answer-Then-Check training data spanning four categories—vanilla harmful/benign and adversarial harmful/benign.
    - Mechanism:
        - An uncensored model (Dolphin-2.9.2) generates intended answers for harmful queries; Qwen2.5-72B produces summaries.
        - Llama3.3-70B generates safety analyses against the safety policy.
        - Adversarial samples are generated using three jailbreak techniques: PAIR, GPTFuzzer, and PAP.
    - Data Distribution: 12K vanilla harmful + 16K vanilla benign + 23K adversarial harmful + 29K adversarial benign.
    - Two-stage Filtering: LlamaGuard classification + internal consistency check (removing samples where the analysis conclusion contradicts the content).

3. **Safe Completion Mechanism**:

    - Function: Provides empathetic, supportive responses to sensitive queries such as self-harm rather than issuing outright refusals.
    - Mechanism: Self-harm-related samples (524 instances) are extracted from the training set to construct safe completion data—vanilla self-harm queries are responded to directly by a general-purpose LLM, while adversarial self-harm queries are handled by first identifying malicious intent and then generating a caring response.
    - Design Motivation: Post-hoc detection methods can only refuse; refusing self-harm-related queries may itself cause harm.

4. **Adaptive Answer-Then-Check**:

    - Function: Bypasses the Answer-Then-Check procedure for normal/safe queries and responds directly, avoiding efficiency overhead.
    - Mechanism: 1,000 instruction-tuning samples that do not require Answer-Then-Check are mixed into training data, enabling the model to learn when to activate the mechanism.
    - Effect: Normal queries achieve base-model-level latency while safety performance is preserved.

5. **RL Variant (ReSA-RL)**:

    - Function: Further optimizes the SFT model using GRPO.
    - Mechanism: Three reward signals—safety reward (LlamaGuard evaluates both the intended summary and the final answer), refusal reward (penalizes over-refusal of benign queries), and format reward (enforces the Answer-Then-Check structure).
    - Key Innovation: Safety rewards are applied to the intended answer summary as well, ensuring the entire generation process—including the internal chain-of-thought—produces safe content.

### Loss & Training
- SFT: Standard cross-entropy; AdamW + cosine schedule; lr = 5e-6; 2 epochs.
- RL: GRPO; reward = $\lambda_{\text{safety}} \cdot (R_{\text{safety}}(o_{\text{intended}}) + R_{\text{safety}}(o_{\text{ans}})) + \lambda_{\text{format}} \cdot R_{\text{format}}(o) + \lambda_{\text{refusal}} \cdot R_{\text{refusal}}(o_{\text{ans}})$

## Key Experimental Results

### Main Results: Jailbreak Defense Rate (LlamaGuard evaluation, Llama3.1-8B-Instruct)

| Method | None | PAIR-GPT | PAIR | PAP | GPTFuzzer | ReNeLLM | TAP | DeepInception | Avg |
|--------|------|----------|------|-----|-----------|---------|-----|--------------|-----|
| Base | 99.7 | 35.1 | 26.2 | 64.9 | 13.7 | 66.1 | 42.5 | 52.4 | 50.1 |
| Post-hoc LlamaGuard | 100 | 46.3 | 50.8 | 71.6 | 99.7 | 93.0 | 65.8 | 97.8 | 78.1 |
| STAIR-DPO | 100 | 68.4 | 42.2 | 94.3 | 100 | 83.4 | 69.3 | 98.7 | 82.0 |
| WJ-SFT | 99.4 | 44.7 | 32.9 | 76.0 | 94.3 | 67.7 | 60.4 | 98.4 | 71.7 |
| **ReSA-SFT** | 99.4 | 89.8 | 69.7 | 96.8 | 95.5 | 88.2 | 85.0 | 99.4 | **90.5** |
| **ReSA-RL** | 100 | 98.7 | 96.8 | 99.7 | 100 | 99.7 | 99.7 | 100 | **99.3** |

### Ablation Study: Effect of Training Data Size

| # Training Samples | 500 | 1K | 5K | 80K |
|--------------------|-----|----|----|-----|
| Avg. Defense Rate (LlamaGuard) | ~89% | ~89% | ~90% | 90.5% |
| Note | Nearly matches full dataset | Diminishing marginal returns | Near saturation | Full dataset |

### Key Findings
- **ReSA-RL achieves near-perfect defense**: Average defense rate of 99.3%, approaching 100% on most attacks, far surpassing all baselines.
- **500 samples suffice**: Only 500 samples are needed to approach full-dataset performance, validating the data efficiency of safety alignment.
- **Applying safety rewards to the intended summary is critical**: This ensures the chain-of-thought internals are also safe.
- **SFT alone substantially outperforms STAIR/WJ**: 82% → 90.5%, demonstrating the intrinsic effectiveness of the Answer-Then-Check reasoning template.
- **Adaptive variant is highly efficient**: Normal queries do not trigger additional reasoning, achieving base-model-level latency.

## Highlights & Insights
- **Exploiting attack-defense asymmetry**: The core of jailbreak attacks is disguising malicious intent at the query level, but once the model attempts to generate an answer, harmful content cannot be hidden. Answer-Then-Check precisely leverages this asymmetry—a particularly elegant insight.
- **Safe completion over blanket refusal**: Providing supportive responses to sensitive topics such as self-harm is a rare but critically important capability among defense methods.
- **No reliance on reasoning models for data construction**: Unlike OpenAI Deliberative Alignment, ReSA constructs training data using only general-purpose LLMs (Qwen2.5 / Llama3.3), lowering the barrier to adoption.
- **Dual safety rewards in RL**: Safety rewards are applied not only to the final answer but also to the intended summary, ensuring the chain-of-thought itself remains safe—mitigating the risk of "chain-of-thought leakage."

## Limitations & Future Work
- **Efficiency overhead**: Despite the adaptive variant, Answer-Then-Check still requires additional generation of the intended summary and safety analysis, increasing computational cost.
- **Dependence on LlamaGuard**: Both data construction and RL rewards rely on the classification accuracy of LlamaGuard.
- **Coverage of the safety policy**: Safety policies encoded in training data must be defined in advance and may not generalize to emerging risk categories.
- **Improvement directions**: Integrating the safety-unit freezing strategy from SSAH could protect safety-critical neurons in the ReSA model from being degraded by downstream fine-tuning.

## Related Work & Insights
- **vs. STAIR-DPO**: STAIR applies DPO for safety reasoning alignment but achieves substantially lower performance (82%) than ReSA (90.5% / 99.3%), as DPO lacks an explicit Answer-Then-Check structure.
- **vs. OpenAI Deliberative Alignment**: OpenAI's approach audits before answering, whereas ReSA answers before auditing; the latter is more effective against disguised queries, and ReSA does not require o1-level reasoning models.
- **vs. Post-hoc Detection (LlamaGuard)**: Post-hoc detection (78.1%) is substantially inferior to ReSA (90.5%) and cannot perform safe completion.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The core insight of Answer-Then-Check—exploiting attack-defense asymmetry—is exceptionally elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 attack types × 3 evaluators × 2 models, with comparisons against 13 baselines; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-organized with detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ Practical and efficient; the 500-sample threshold is extremely accessible, and the RL variant achieves near-perfect defense.

## Background & Motivation
- LLMs remain vulnerable to jailbreak attacks; existing defenses offer limited effectiveness against sophisticated adversarial prompts.

## Method
- Answer-Then-Check: the model summarizes intent during reasoning → policy-based safety analysis → final output.
- ReSA dataset: 80K samples (vanilla/adversarial × harmful/benign).
- Adaptive variant + RL variant.

## Key Experimental Results

| Dimension | ReSA |
|-----------|------|
| Safety | Best among 13 methods |
| Capability retention | No degradation on MMLU / MATH500 / HumanEval |
| Data efficiency | 500 samples ≈ full dataset |

## Rating
- Novelty: ⭐⭐⭐⭐ A new paradigm for intermediate safety reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparisons against 13 methods.
- Value: ⭐⭐⭐⭐⭐ A practical solution for LLM safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](../../NeurIPS2025/llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](../../ACL2026/llm_alignment/trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
