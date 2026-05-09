---
title: >-
  [Paper Note] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach
description: >-
  [ACL 2026][Medical Imaging][Medical Reasoning] This paper proposes MedSSR, a framework that enhances LLM medical reasoning through controllable data synthesis with rare disease knowledge injection and a semi-supervised training paradigm of "self-supervised RL → supervised RL." MedSSR achieves up to +5.93% improvement on rare disease tasks, surpassing the +3% ceiling observed in all prior methods.
tags:
  - ACL 2026
  - Medical Imaging
  - Medical Reasoning
  - Rare Disease
  - Data Synthesis
  - Semi-Supervised Reinforcement Learning
  - GRPO
date: 2026-05-08
content_hash: c50f09689f9280eb
---

# Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach

**Conference**: ACL 2026
**arXiv**: [2604.11547](https://arxiv.org/abs/2604.11547)
**Code**: [https://github.com/tdlhl/MedSSR](https://github.com/tdlhl/MedSSR)
**Area**: Medical Imaging
**Keywords**: Medical Reasoning, Rare Disease, Data Synthesis, Semi-Supervised Reinforcement Learning, GRPO

## TL;DR
This paper proposes MedSSR, a framework that enhances LLM medical reasoning through controllable data synthesis with rare disease knowledge injection and a semi-supervised training paradigm of "self-supervised RL → supervised RL." MedSSR achieves up to +5.93% improvement on rare disease tasks, surpassing the +3% ceiling observed in all prior methods.

## Background & Motivation

**Background**: The development of LLMs for medical reasoning is constrained by the scarcity of high-quality reasoning data. Existing approaches primarily initialize policy models by distilling CoT reasoning chains from large closed-source models such as GPT-4o, followed by RL training.

**Limitations of Prior Work**: (1) Only 22% of existing medical benchmarks consist of reasoning-intensive questions, of which merely 3% involve rare diseases; (2) distilling long reasoning chains from closed-source models is costly; (3) no existing method achieves improvements beyond the +3% ceiling on rare disease tasks—even with fully supervised GRPO; (4) privacy constraints and domain expertise requirements make acquiring complex medical reasoning data extremely challenging.

**Key Challenge**: Rare disease data is severely scarce, and the data distribution of existing methods is constrained by available annotated data, resulting in a low improvement ceiling on rare disease tasks. Furthermore, synthesized data may contain factual errors, which is unacceptable in medical settings.

**Goal**: To efficiently improve LLM performance on a broad range of medical reasoning tasks—including rare diseases—without relying on expensive reasoning chain distillation.

**Key Insight**: (1) Synthesize only questions (rather than long reasoning chains), substantially reducing generation cost; (2) inject rare disease knowledge to control the distribution of synthesized data; (3) use the policy model itself to generate pseudo-labels, eliminating dependence on external models.

**Core Idea**: Synthesize medical reasoning questions with controllable distribution via rare disease knowledge injection, generate pseudo-labels through majority voting with the model itself, and then conduct curriculum training following the "self-supervised RL → supervised RL" paradigm.

## Method

### Overall Architecture
MedSSR comprises two synergistic components: (1) a medical knowledge-enhanced data synthesis pipeline that generates new questions from seed questions, controls the rare disease ratio via threshold $\alpha$, and produces pseudo-labels using the policy model itself; (2) a semi-supervised RL training strategy that first performs self-supervised RL on pseudo-labeled synthetic data (intrinsic learning), followed by supervised RL on human-annotated real data (extrinsic learning).

### Key Designs

1. **Knowledge-Enhanced Data Synthesis**:

    - Function: Generate medical reasoning questions with controllable distribution, specifically increasing the proportion of rare disease questions.
    - Mechanism: Given two seed questions $\{x_1^s, x_2^s\}$, GPT-4.1 is used to synthesize new questions. The rare disease ratio is controlled by threshold $\alpha$: a value $\rho \sim \text{Uniform}(0,1)$ is sampled; when $\rho < \alpha$, an entity $e$ is selected from a rare disease list, and MedCPT retrieves the top-$k$ relevant medical literature $\mathcal{C}(e)$, which is injected into the synthesis prompt. Only questions are synthesized (not reasoning chains), making the per-sample API token cost far lower than distillation-based methods.
    - Design Motivation: Directly synthesizing reasoning chains is costly and prone to introducing errors. Synthesizing only questions allows the policy model's own reasoning capability to generate answers, avoiding dependence on the reasoning quality of external models. Knowledge injection ensures the medical accuracy of synthesized questions.

2. **Pseudo-Label Generation and Quality Control**:

    - Function: Generate reliable answer labels for synthesized questions to enable RL training.
    - Mechanism: The policy model (base model) is used to offline-sample multiple responses for each synthesized question, and the most consistent answer is selected as the pseudo-label via majority voting. Only pseudo-labels exceeding a confidence threshold are retained.
    - Design Motivation: Labeling with external models may introduce distribution mismatch (reward hacking). Self-labeling with the policy model ensures that the data aligns with the model's learning trajectory. Majority voting provides a natural quality filter.

3. **Semi-Supervised RL Training Strategy**:

    - Function: Effectively leverage the complementary advantages of synthetic and real data to achieve curriculum learning from intrinsic to extrinsic knowledge.
    - Mechanism: A two-stage curriculum is employed—(a) Self-supervised RL: GRPO training on pseudo-labeled synthetic data allows the model to learn from its own knowledge and reasoning (intrinsic learning), broadening knowledge coverage, particularly for rare diseases; (b) Supervised RL: GRPO training on human-annotated real data (extrinsic learning) calibrates and consolidates the model's reasoning capability.
    - Design Motivation: Directly supervising on synthetic data may cause instability due to pseudo-label noise. The curriculum of "self-supervised RL exploration followed by supervised RL refinement" enables the model to first learn broadly and then calibrate precisely.

### Loss & Training
GRPO is used for optimization with the verification reward $r(y, y') = \mathbb{I}[\text{ans}(y') = y]$. KL divergence constrains deviation from the reference policy. Experiments are conducted on Qwen3-8B and Llama-3.1-8B-Instruct.

## Key Experimental Results

### Main Results

| Method | General Medical Gain | Rare Disease Gain | Per-sample API Token Cost |
|---|---|---|---|
| HuatuoGPT-O1 | Moderate | <3% | High (long reasoning chains) |
| MedReason | Moderate | <3% | High |
| Fully Supervised GRPO | Moderate | <3% | Low |
| MedSSR (Llama) | **+3.91%** | **+5.93%** | Low (questions only) |
| MedSSR (Qwen3) | Significant | Breaks 3% ceiling | Low |

### Ablation Study

| Configuration | General | Rare Disease | Note |
|---|---|---|---|
| Full MedSSR | Best | Best | Complete framework |
| w/o knowledge injection | Degraded | Significantly degraded | Insufficient rare disease data ratio |
| w/o self-supervised RL stage | Degraded | Degraded | Lacks broad coverage from synthetic data |
| w/o pseudo-label filtering | Degraded | Degraded | Noisy labels impair training |
| Single-stage mixed training | Below two-stage | Below two-stage | Validates necessity of curriculum design |

### Key Findings
- MedSSR is the first method to surpass the +3% improvement ceiling on rare disease tasks, achieving +5.93%.
- Synthesizing only questions (without reasoning chains) effectively improves reasoning capability at substantially reduced cost.
- The two-stage semi-supervised RL curriculum outperforms single-stage mixed training, validating the "broad-then-precise" strategy.
- The threshold $\alpha$ for rare disease knowledge injection enables precise control over data distribution.
- MedSSR comprehensively outperforms existing methods across 10 medical benchmarks.

## Highlights & Insights
- **Synthesizing questions only, not answers**: The approach elegantly simplifies the high-cost "question + reasoning chain" synthesis into low-cost "question-only" synthesis, leveraging the policy model's own reasoning capability to generate answers, substantially reducing reliance on closed-source APIs.
- **Bootstrapped pseudo-label learning**: Using the model's own majority voting to generate pseudo-labels is an elegant bootstrapping strategy that ensures training data is aligned with the model's capability.
- **Distribution-controllable data synthesis**: Precisely controlling the proportion of rare disease data via the threshold $\alpha$ provides a direct tool for addressing the long-tail distribution problem in the medical domain.

## Limitations & Future Work
- Pseudo-label quality depends on the policy model's own capability—if the model is entirely unfamiliar with certain rare diseases, pseudo-labels may be unreliable.
- The coverage of the rare disease knowledge base may be limited; high-quality questions for uncovered rare diseases remain difficult to generate.
- Validation is conducted only on 8B-scale models; the effectiveness on larger models remains unknown.
- The diversity of synthesized questions is constrained by the quality and quantity of seed questions.

## Related Work & Insights
- **vs. HuatuoGPT-O1**: Distills GPT-4o reasoning chains via SFT + RL at high cost with limited rare disease improvement. MedSSR synthesizes only questions at low cost with significant rare disease gains.
- **vs. MedReason**: Uses knowledge graphs to improve factual accuracy of CoT generation but still relies on long-chain distillation. MedSSR ensures accuracy at the synthesis stage directly through knowledge injection.
- **vs. Self-Instruct**: A general-purpose self-instruction synthesis method; MedSSR augments it with knowledge retrieval and distribution control tailored to the medical domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "question synthesis + self-pseudo-labeling + semi-supervised RL" constitutes a novel and efficient paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ten medical benchmarks, two base models, comprehensive ablations and comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and precise problem framing (the 3% rare disease ceiling).
- Value: ⭐⭐⭐⭐⭐ Provides a practical and efficient solution to the data scarcity problem in medical LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)

<!-- RELATED:END -->
