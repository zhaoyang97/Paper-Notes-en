---
title: >-
  [Paper Note] ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails
description: >-
  [ACL 2025][Reasoning][Safety Guardrails] By distilling structured critique (safety labels + detailed reasoning) from GPT-4o/DeepSeek-R1, the guardrail model is fine-tuned to implement "slow thinking" safety judgment. It achieves the highest average F1 (75.5%) and AUPRC (79.5%) across 4 safety benchmarks, outperforming LLaMA Guard 3 with a 16.1% increase in accuracy and a 27.0% increase in Macro F1.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Safety Guardrails"
  - "Slow Thinking"
  - "Critique Augmentation"
  - "Knowledge Distillation"
  - "LLaMA Guard"
date: 2026-05-08
content_hash: cbc09be7dc91d8b9
---

# ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails

**Conference**: ACL 2025  
**arXiv**: [2502.13458](https://arxiv.org/abs/2502.13458)  
**Code**: [https://github.com/luka-group/ThinkGuard](https://github.com/luka-group/ThinkGuard)  
**Area**: LLM Reasoning/Safety  
**Keywords**: Safety Guardrails, Slow Thinking, Critique Augmentation, Knowledge Distillation, LLaMA Guard

## TL;DR
By distilling structured critique (safety labels + detailed reasoning) from GPT-4o/DeepSeek-R1, the guardrail model is fine-tuned to implement "slow thinking" safety judgment. It achieves the highest average F1 (75.5%) and AUPRC (79.5%) across 4 safety benchmarks, outperforming LLaMA Guard 3 with a 16.1% increase in accuracy and a 27.0% increase in Macro F1.

## Background & Motivation

**Background**: Safety guardrail models (e.g., LLaMA Guard series, WildGuard) are critical external layers for secure LLM deployment. They typically model safety detection as a classification task—taking prompt/response as input and outputting safe/unsafe labels.

**Limitations of Prior Work**: (a) **Lack of reasoning in single-turn classification**: models output labels without reasons, easily misjudging implicit harmful content or adversarial samples; (b) **Lack of explainability**: users cannot understand why specific content is determined as unsafe; (c) **Rule-based methods are too rigid, while model-based methods are too shallow**.

**Key Challenge**: Safety judgment requires understanding intent, context, and potential risks—necessitating **deliberate reasoning** rather than intuitive, single-turn classification. Existing guardrails implement "fast thinking" (System 1) while lacking "slow thinking" (System 2).

**Goal**: To enable guardrail models to both classify accurately and provide reasoning explanations by distilling the reasoning capabilities of large models into smaller models.

**Key Insight**: Dual-process theory in psychology—fast intuitive judgment vs. deliberate reasoning. Upgrading guardrail models from System 1 to System 2.

**Core Idea**: Generating structured safety critiques using large models, and fine-tuning smaller models to "think thoroughly before making a judgment."

## Method

### Overall Architecture
ThinkGuard operates in three steps: (1) using GPT-4o/DeepSeek-R1 to generate structured critiques for annotated safety data; (2) fine-tuning LLaMA Guard 3 using a two-turn dialogue format, where the first turn outputs the safety label and category violation, and the second turn generates the critique explanation; (3) predicting the label first during inference, followed by generating the critique (optional).

### Key Designs

1. **Critique-Augmented Data Construction**:

    - **Function**: Generating structured critiques for (prompt, response) pairs in the BeaverTails dataset using an expert model.
    - Data Format: $D = \{(x_i, r_i, y_i, c_i)\}_{i=1}^N$, where $y_i$ is the safety label and $c_i$ is the critique.
    - Guided using structured prompts to lead the expert model to output in a unified format.
    - **Design Motivation**: Large models possess strong reasoning capabilities but are costly to deploy. Knowledge distillation is utilized to transfer reasoning abilities to smaller models.

2. **Joint Loss Fine-tuning**:

    - Classification Loss: $\mathcal{L}_{cls} = -\sum_i y_i \log P(y_i | x_i, r_i)$
    - Critique Loss: $\mathcal{L}_{critique} = -\sum_t \log P(c_t | c_{<t}, x_i, r_i, y_i)$
    - Total Loss: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{critique}$
    - **Design Motivation**: Joint optimization ensures simultaneous improvement of classification accuracy and reasoning ability.

3. **Inference Pipeline (Three-Step Sequential)**:

    - Step 1: Safety Assessment $\hat{y} = \arg\max P(y|x,r)$
    - Step 2: Violation Category Prediction $t = \arg\max P(t|x,r,\hat{y})$
    - Step 3: Critique Generation $\hat{c} = \arg\max P(c|x,r,\hat{y},t)$
    - Users can either use Step 1 only (latency equivalent to traditional guardrails) or the full three steps (to gain explainability).

## Key Experimental Results

### Main Results

| Model | BeaverTails F1 | ToxicChat F1 | OpenAI F1 | WildGuardMix F1 | **Avg F1** | **Avg AUPRC** |
|---|---|---|---|---|---|---|
| GPT-4o | 77.3 | 39.8 | 68.5 | 72.0 | 64.4 | 70.3 |
| GPT-4o + CoT | 83.9 | 50.4 | 75.1 | 75.5 | 71.2 | 73.4 |
| LLaMA Guard 3 | 64.5 | 43.4 | 77.2 | 72.6 | 64.4 | 75.1 |
| LLaMA Guard 3 + Label SFT | 83.7 | 56.0 | 75.6 | 73.8 | 72.3 | 76.8 |
| WildGuard | 78.9 | 63.5 | 72.3 | 74.9 | 72.4 | - |
| **ThinkGuard** | **82.7** | **63.5** | **77.3** | **78.6** | **75.5** | **79.5** |

### Ablation Study

| Configuration | Avg F1 | Description |
|---|---|---|
| ThinkGuard (full) | 75.5 | Full (label + critique) |
| Label-only SFT | 72.3 | Label-only SFT → -3.2% |
| LLaMA Guard 3 + ICL | 62.8 | In-context learning → poor performance |
| LLaMA Guard 3 原始 | 64.4 | Untuned baseline |

### Key Findings
- **Critique Augmentation vs. Label-only SFT**: +3.2% F1, showing that the reasoning process itself enhances classification quality.
- **8B ThinkGuard Outperforms GPT-4o**: 75.5 vs. 64.4 (Avg F1), even outperforming GPT-4o+CoT (71.2).
- **Greatest Gain on WildGuardMix**: This dataset contains adversarial samples, where the reasoning advantages of ThinkGuard are highly pronounced (78.6 vs. 72.6).

## Highlights & Insights
- **Elegant Application of Dual-Process Theory to AI Safety**: Upgrading from intuitive classification to deliberate reasoning, offering both theoretical elegance and practical efficacy.
- **Flexible Design of Two-Turn Format**: Users can choose to use only the first turn (maintaining efficiency) or the full two turns (acquiring explanations).
- **Smaller Models Outperforming Large Models Via Distillation**: The 8B ThinkGuard outperforms GPT-4o, demonstrating that domain-focused distillation and fine-tuning can be more effective than general large models.

## Limitations & Future Work
- **Dependence on Expert Models for Critique Generation**: Data quality is constrained by GPT-4o's capabilities.
- **Training Data Primarily Sourced from BeaverTails**: The coverage is limited.
- **Slow Thinking Increases Latency**: Inference time doubles when critiques are required.
- **No Evaluation of Adaptive Attacks**: Attackers aware that the model "thinks twice" might design bypassing schemes.
- Future Directions: Optimizing critique quality using RL; expanding training data; more efficient critique generation.

## Related Work & Insights
- **vs. LLaMA Guard 3**: Single-turn classification vs. reasoning + classification, yielding a significant increase in F1 (+11.1 avg).
- **vs. WildGuard**: Uses a larger training set (92K) but underperforms ThinkGuard (F1 72.4 vs. 75.5), implying that critiques are more important than dataset size.
- **vs. GPT-4o + CoT**: GPT-4o also employs "slow thinking" with CoT, but the distilled smaller model performs even better.
- The critique-augmentation concept can be transferred to other classification tasks requiring reasoning, such as fact-checking and sentiment analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ Critique-augmented guardrail is an explicit new idea, and the dual-process theoretical framework is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 benchmarks + multiple baselines + ablation study + distillation source comparison.
- Writing Quality: ⭐⭐⭐⭐ Formalized and complete description of the method.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for safety guardrails; smaller models outperforming larger ones holds industrial significance.

## Background & Motivation

1. **Background**: Safety guardrail models (such as LLaMA Guard) typically only output safe/unsafe labels, lacking explainability, and making inaccurate judgments on edge cases.
2. **Core Idea**: Enabling the guardrail model to "think thoroughly before answering"—generating a detailed critique analysis before providing the safety label.

## Method

### Key Designs
1. **Critique-Augmented Data Generation**: Generating data in a two-turn dialogue format using strong LLMs—the first turn provides the initial prediction, and the second turn details the reasoning process and safety policy references.
2. **Two-Turn Dialogue Fine-Tuning**: Training the model to learn the slow-thinking paradigm of "coarse judgment first, followed by fine reasoning".

## Key Experimental Results
- Compared to LLaMA Guard 3: Accuracy **+16.1%**, Macro F1 **+27.0%**
- Achieves the highest average F1 and AUPRC across multiple safety benchmarks.

## Highlights & Insights
- **"Slow Thinking" Safety Guardrail** is a powerful paradigm: more accurate and interpretable than simple classification.
- Critique training can be transferred to other safety classification tasks.

## Limitations & Future Work
- Slow thinking increases inference latency.
- Dependence on strong LLMs to generate critique data.

## Rating
- Novelty: ⭐⭐⭐⭐ Critique-augmented guardrail is a new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐⭐ Holds major practical value for safety guardrails.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning](../../ICML2025/llm_reasoning/rethinking_external_slow-thinking_from_snowball_errors_to_probability_of_correct.md)
- [\[ICLR 2026\] Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy](../../ICLR2026/llm_reasoning/making_slow_thinking_faster_compressing_llm_chain-of-thought_via_step_entropy.md)
- [\[NeurIPS 2025\] Controlling Thinking Speed in Reasoning Models](../../NeurIPS2025/llm_reasoning/controlling_thinking_speed_in_reasoning_models.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](../../ICLR2026/llm_reasoning/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)

</div>

<!-- RELATED:END -->
