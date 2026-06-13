---
title: >-
  [Paper Note] The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models
description: >-
  [ICLR 2026 (Poster)][LLM Reasoning][Overthinking] Reasoning models form a "first impression" (internal bias) about the answer the moment they receive a question. When this intuitive guess conflicts with the subsequent sy…
tags:
  - "ICLR 2026 (Poster)"
  - "LLM Reasoning"
  - "Overthinking"
  - "Internal Bias"
  - "Reasoning Models"
  - "Causal Intervention"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 75e017394b7b72d0
---

# The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models

**Conference**: ICLR 2026 (Poster)
**arXiv**: [2505.16448](https://arxiv.org/abs/2505.16448)  
**Authors**: Renfei Dang, Zhening Li, Shujian Huang, Jiajun Chen (Nanjing University)
**Code**: N/A  
**Area**: LLM Reasoning
**Keywords**: Overthinking, Internal Bias, Reasoning Models, Causal Intervention, Attention Mechanism

## TL;DR

Reasoning models form a "first impression" (internal bias) about the answer the moment they receive a question. When this intuitive guess conflicts with the subsequent systematic reasoning process, the model repeatedly second-guesses itself and re-examines its work, causing reasoning length to inflate by 21%–43%. Critically, none of the existing mitigation methods can fundamentally eliminate this effect.

## Background & Motivation

Reasoning models such as DeepSeek-R1 and OpenAI o1 achieve breakthrough performance on complex reasoning tasks by performing self-reflection and error correction via internal Chain-of-Thought (CoT). However, they suffer from a prominent efficiency bottleneck—**overthinking**: generating large amounts of redundant reasoning steps (e.g., repeated "Wait..." and "Let me re-check..."), which wastes computational resources without improving accuracy and sometimes even degrades final answer quality.

Prior work has primarily characterized overthinking from an **external behavioral** perspective (e.g., mismatch between reasoning token count and problem difficulty), but has not explained **why models fall into these meaningless reflective loops**. Drawing on the **anchoring effect** from cognitive psychology, this paper proposes a novel causal explanation—**"The First Impression Problem"**:

- Before the `<think>` reasoning phase formally begins, the model's hidden states already encode a preliminary guess about the answer (termed **internal bias**) upon receiving the input question.
- This "first impression" is not necessarily output explicitly, but persists in the model's intermediate-layer representations.
- When this initial guess **conflicts** with the conclusion reached by subsequent reasoning, the model cannot "let go," repeatedly revisiting the original question and self-correcting, generating extensive redundant reflection.
- In extreme cases, the model may fall into a **parroting loop**—endlessly repeating the same reasoning steps without ever converging to a final answer.

## Method

### Overall Architecture

The research framework follows a progressive analytical structure: **quantify bias → validate correlation → causal intervention → mechanistic explanation → mitigation attempts**, with each stage being indispensable.

### Key Designs

1. **Quantifying Internal Bias**

    The authors extract internal bias via the **Direct Answer** method: a forced direct-answer prompt (e.g., "Answer without thinking more:") is appended to the model input to intercept its output before the `<think>` reasoning block is activated, yielding the intuitive answer $a_{bias}$. To eliminate sampling noise, 64 samples are drawn per question to construct an internal bias distribution $\tilde{a}_{bias}$. A **Deviation Degree** $D_{bias}$ is then defined to measure the discrepancy between the internal bias and the final reasoning answer $a_{final}$:

    - Numerical tasks: $D_{bias} = \text{mean}(|a_{bias} - a_{final}|)$ (MAE)
    - Multiple-choice tasks: $D_{bias}$ = inconsistency rate (proportion of cases where the initial guess differs from the final answer)

2. **Correlation Validation**

    Statistical analyses are conducted across multiple reasoning models—including DeepSeek-R1-671B and QwQ-32B—on diverse reasoning benchmarks such as AIME and KnowLogic. The core finding is that as $D_{bias}$ increases, reasoning length increases significantly, with **relative length increment $R_\Delta$ ranging from 21.0% to 43.1%**. Crucially, by measuring the **First Answer Position** (where the model first arrives at a complete answer), the authors find that the timing of the first logical conclusion is roughly the same regardless of bias level—the extra tokens are almost **entirely** redundant reflection occurring after that conclusion.

3. **Counterfactual Causal Intervention**

    To move from correlation to causation, two complementary counterfactual experiments are designed:

    - **Intervention 1: Question Removal**. After the model generates its first complete answer, the original input question is deleted from the context, forcing the model to decide whether to continue reflecting based solely on its already-generated reasoning chain. A redundancy reduction ratio $r = (L_{ori} - L_{rem}) / (L_{ori} - P_{first})$ is defined. Result: redundant tokens are **reduced by 53.5%** on AIME 2024, with accuracy maintained or slightly improved.
    - **Intervention 2: Bias Injection**. Via LoRA fine-tuning, incorrect bias is injected into problems the model originally solves easily (Low2Wrong), and correct bias is injected into problems the model originally finds difficult (High2Correct). Result: injecting incorrect bias causes easy problems to generate substantial redundant reflection; injecting correct bias significantly reduces overthinking on hard problems. This confirms causality from **both directions**.

4. **Attention Mechanism Analysis**

    Interpretability experiments reveal the specific mechanism by which internal bias operates:

    - **Attention Dynamics**: At the moment the model is about to trigger reflection (i.e., generate "Wait..." or "Let me check..."), the attention weights on the original input question surge to more than **4× their value** during normal reasoning. This indicates that the model repeatedly "looks back" at the original question to reactivate its initial bias.
    - **Logit Lens Probing**: For samples where the model ultimately reasons correctly to $a_{final} = 3$ but holds an internal bias of $a_{bias} = 5$, the internal decoding probability of the biased answer "5" **consistently exceeds** that of the correct answer "3" across intermediate-to-late layers—even after the model has explicitly written "3" in its reasoning chain. This reveals a persistent state of **cognitive dissonance**: the intuitive guess and the systematic reasoning coexist and compete inside the model.

## Key Experimental Results

### Bias–Overthinking Correlation

The relationship between $D_{bias}$ and reasoning length is validated across multiple models and tasks:

| Model | Task | Low-Bias Group Length | High-Bias Group Length | Relative Increment $R_\Delta$ |
|-------|------|-----------------------|------------------------|-------------------------------|
| DeepSeek-R1-671B | AIME | Baseline | Significantly longer | 21.0%–43.1% |
| QwQ-32B | AIME | Baseline | Significantly longer | Within range |
| DeepSeek-R1 (distilled) | KnowLogic | Baseline | Significantly longer | Within range |
| Multiple models | Multiple tasks | Baseline | Significantly longer | **All >20%** |

Key observation: The high-bias and low-bias groups reach the **first complete answer** at roughly the same position; all extra tokens are concentrated in the redundant reflection interval after the first answer.

### Counterfactual Intervention Results

| Intervention | Operation | Change in Redundant Reasoning | Accuracy Impact |
|--------------|-----------|-------------------------------|-----------------|
| Question Removal | Delete original question after first answer | **Reduced by 53.5%** (AIME 2024) | Maintained / slightly improved |
| Low2Wrong Injection | LoRA injection of incorrect bias into easy problems | **Significantly increased** | Decreased |
| High2Correct Injection | LoRA injection of correct bias into hard problems | **Significantly decreased** | Improved |

### Evaluation of Existing Mitigation Methods

| Method | Type | Reduces Reasoning Length | Eliminates Bias Effect ($R_\Delta$) | Accuracy Impact |
|--------|------|:------------------------:|:-----------------------------------:|:---------------:|
| FCS (SFT+DPO) | Training-time | Yes | No; $R_\Delta$ unchanged or worsened | May drop on complex tasks |
| SEAL | Inference-time | Yes | No; bias effect persists | Largely maintained |
| PROBE | Inference-time | Yes | No; bias effect persists | Largely maintained |
| Attention Early Exit (proposed) | Inference-time | Yes | **Partially effective**: $R_\Delta$ reduced from 31.5% → 9.4% | Minimal accuracy loss |

Core conclusion: Methods such as FCS, SEAL, and PROBE essentially only "truncate" the reasoning chain without resolving the root conflict between internal bias and reasoning. The only approach showing preliminary effectiveness is the **attention early exit mechanism** proposed in this paper—monitoring the model's normalized attention to the original question and terminating reasoning when it exceeds a threshold (indicating the model is reactivating its bias).

## Highlights & Insights

1. **Complete Mapping from Cognitive Psychology to LLM Mechanisms**: The anchoring effect and Kahneman's System 1/System 2 theory are mapped onto the behavioral patterns of reasoning models, revealing that models indeed exhibit conflicts analogous to human fast intuition (System 1) versus slow deliberation (System 2). This not only provides a diagnostic tool but also opens a new path for understanding LLM behavior through cognitive science theories.

2. **Rigor of Causal Inference Design**: Rather than settling for simple correlation analysis, the paper establishes causality from both directions through two complementary counterfactual interventions (removing the bias source and injecting bias). The LoRA bias injection experiment is particularly elegant: Low2Wrong and High2Correct control for the two confounding variables of problem difficulty and bias direction, respectively.

3. **Significant Value of Negative Results**: The paper systematically demonstrates that existing methods such as FCS, SEAL, and PROBE all fail to eliminate internal bias, a finding that carries important guidance for the community—surface-level reasoning truncation treats symptoms rather than causes, and fundamental solutions must be sought at the level of model architecture or training paradigms.

4. **"Cognitive Dissonance" Revealed by Logit Lens**: Even after the model has explicitly reasoned to the correct answer, the incorrect initial guess maintains higher decoding probability in intermediate layers—an extremely intriguing finding that reveals the competitive dynamics between "intuition" and "logic" within reasoning models.

## Limitations & Future Work

1. **Lack of a Fundamental Solution**: Although the attention early exit mechanism shows preliminary effectiveness (reducing $R_\Delta$ from 31.5% to 9.4%), it remains an external inference-time intervention rather than enabling the model itself to learn to override incorrect intuitions. Future work could explore incorporating a "bias decoupling" objective into the RL training phase.

2. **Limited Model Coverage**: Experiments are conducted primarily on open-source reasoning models in the DeepSeek-R1 and Qwen families; applicability to closed-source commercial models such as o1 and Claude remains unknown. The bias formation mechanism may also differ across training paradigms (pure RL vs. SFT+RL).

3. **Task Scope Focused on Determinate Reasoning**: Experiments concentrate on tasks with clear correct answers, such as mathematics (AIME) and logic (KnowLogic). Overthinking in open-ended generation and creative writing tasks remains unexplored, and the definition and quantification of bias would need to be redesigned for such settings.

4. **Completeness of Bias Extraction**: The direct-answer prompt "Answer without thinking more" may not fully capture all forms of bias information encoded in the model's hidden states. More refined probing methods—such as training a probing classifier to directly read hidden states—might yield more precise bias estimates.

5. **Future Directions**: Designing explicit System 1/System 2 reasoning architectures that decouple fast intuitive pathways from slow deliberative pathways within the model; or investigating attention regularization techniques to reduce excessive attention to the original input during reasoning.

## Related Work & Insights

- **Overthinking Analysis**: Chen et al. (2024) "Do not think that much for 2+3=?" first systematically characterizes the overthinking phenomenon in o1-type models; this paper extends that work to the level of internal mechanisms.
- **Reasoning Unfaithfulness**: Arcuschin et al. (2025) find that CoT reasoning is not always faithful in naturalistic settings and that final answers may be shaped by implicit biases rather than the reasoning chain, corroborating the findings of this paper.
- **Circuit Tracing**: Anthropic's research suggests that language models may possess distinct "fast estimation" neural circuits, providing mechanistic support for the existence of "internal bias" identified in this paper.
- **Reasoning Efficiency Optimization**: Methods such as SEAL (hidden-state guidance) and PROBE (confidence-based early stopping) are shown by this paper to treat only symptoms rather than root causes.
- Inspiration: **Cognitive-science-driven behavioral analysis of LLMs** is a research direction worth pursuing in depth; the System 1/System 2 distinction may inspire fundamentally new reasoning model architectures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The "First Impression Problem" is a highly original finding; bridging cognitive psychology with LLM mechanistic analysis is remarkably novel.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Correlation analysis + bidirectional causal intervention + attention/Logit Lens interpretability + evaluation of mitigation methods + exploration of new approaches; the experimental chain is complete and rigorous.)
- Writing Quality: ⭐⭐⭐⭐ (Concepts are articulated clearly; the "first impression" analogy is intuitively accessible; the narrative logic of the progressive experimental design is fluent.)
- Value: ⭐⭐⭐⭐⭐ (The paper represents a milestone in understanding the root causes of overthinking in reasoning models and has direct implications for future model design and training paradigms.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling](overthinking_reduction_with_decoupled_rewards_and_curriculum_data_scheduling.md)
- [\[NeurIPS 2025\] The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](../../NeurIPS2025/llm_reasoning/the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)
- [\[NeurIPS 2025\] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](../../NeurIPS2025/llm_reasoning/the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](../../ACL2026/llm_reasoning/reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](../../NeurIPS2025/llm_reasoning/let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)

</div>

<!-- RELATED:END -->
