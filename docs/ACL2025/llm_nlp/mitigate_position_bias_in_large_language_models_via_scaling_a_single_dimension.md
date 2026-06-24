---
title: >-
  [Paper Note] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel
description: >-
  [ACL 2025][LLM (Other)][position bias] It is discovered that specific channels in the hidden states of LLMs encode absolute position information (positional hidden states). By scaling this single channel, the "lost in the middle" position bias can be mitigated, yielding up to a 15.2% improvement on multi-document QA benchmarks without affecting other capabilities of the model.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "position bias"
  - "lost in the middle"
  - "hidden states"
  - "positional channel"
  - "attention"
date: 2026-05-08
content_hash: 21bbf5eafffc469d
---

# Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel

**Conference**: ACL 2025  
**arXiv**: [2406.02536](https://arxiv.org/abs/2406.02536)  
**Code**: [https://aka.ms/PositionalHidden](https://aka.ms/PositionalHidden)  
**Area**: LLM/NLP  
**Keywords**: position bias, lost in the middle, hidden states, positional channel, attention

## TL;DR
It is discovered that specific channels in the hidden states of LLMs encode absolute position information (positional hidden states). By scaling this single channel, the "lost in the middle" position bias can be mitigated, yielding up to a 15.2% improvement on multi-document QA benchmarks without affecting other capabilities of the model.

## Background & Motivation

**Background**: Long-context LLMs still suffer from "lost in the middle" bias, where key information in the middle positions is neglected.

**Limitations of Prior Work**: Existing mitigation methods address data distributions (FILM) or positional encodings (Ms-PoE), but ignore the fact that hidden states also carry positional information.

**Key Challenge**: Causal attention masks inject positional information into hidden states, which is another source of position bias—yet this has not been previously investigated.

**Goal**: Find the specific channels in hidden states that encode positional information, and mitigate the bias by scaling them.

**Key Insight**: It is observed that the values of certain channels in hidden states have a monotonic relationship with token positions, which are defined as "positional channels".

**Core Idea**: Modifying only one channel of the hidden states can significantly mitigate position bias—representing a previously unknown source of position bias in LLMs.

## Method

### Overall Architecture
Discovering a U-shaped bias in attention weights -> Proving that positional information in hidden states also causes bias -> Identifying positional channels (where values are monotonically correlated with positions) -> Designing a scaling algorithm -> Affecting only the attention query of the last token.

### Key Designs

1. **Positional Channel Identification**

    - Heuristic search: Find channels along the hidden size dimension whose values exhibit a monotonic and smooth relationship with position.
    - Select the optimal channel using a calibration dataset.
    - **Design Motivation**: Precisely locate the carriers of positional information.

2. **Scaling Method**

    - Scale the values of the selected channels (reducing their magnitude).
    - Affect only the attention computation of the last token over other tokens.
    - **Design Motivation**: Eliminate position bias while avoiding disruption to other model capabilities.

3. **Attention Modification Algorithm**

    - The scaled hidden states are utilized only in the query of the last token.
    - Attention between other tokens remains unaffected.
    - **Design Motivation**: The attention of the last token determines which information is retrieved to generate the answer.

## Key Experimental Results

### Main Results — NQ Multi-Document QA

| Method | Accuracy | Gain |
|------|--------|------|
| Original Model | ~50% | Baseline |
| FILM (SFT) | ~58% | +8% |
| Ms-PoE | ~55% | +5% |
| **Scale Positional HS** | **~65%** | **+15.2%** |

### Cross-Model Generalization

| Model | Gain |
|------|------|
| LLaMA-2-7B | +12% |
| Mistral-7B | +15% |
| Gemma-7B | +10% |
| Qwen-7B | +8% |
| MPT-7B (Alibi) | +6% |

### Side-Effect Evaluation

| Evaluation | Results |
|------|------|
| Temporal Ordering Task | No significant degradation |
| MMLU | No significant degradation |

### Key Findings
- **15.2% improvement by modifying only one channel**—an extremely lightweight intervention.
- **Cross-model generalization**: Effective across models with different positional encodings, such as RoPE and Alibi.
- **No training required**: A pure training-free inference-time intervention.
- **Two sources of position bias**: Positional encodings + positional information in hidden states (a new discovery).
- **Visualization of positional channels** confirms that their values are highly linearly correlated with absolute positions.

## Highlights & Insights
- **The discovery of positional hidden states** is the most significant contribution of the paper—proving that causal attention masks inject positional information into hidden states.
- The minimalist methodology of **single-channel intervention** is impressive—modifying just 1 out of 4096 dimensions yields a 15% improvement.
- **Provides a new perspective of understanding and a new solution path** for the "lost in the middle" problem.

## Limitations & Future Work
- The optimal channel must be found via search, which may vary across different models.
- Only validated on 7B-parameter models.
- Future directions: Validation on larger models, dynamic channel selection.

## Related Work & Insights
- **vs FILM (An et al.)**: FILM requires SFT data, whereas this work requires no training.
- **vs Ms-PoE (Zhang et al.)**: Ms-PoE modifies positional encodings, whereas this work modifies hidden states—representing complementary research directions.
- **vs CogSteer**: CogSteer finds that intermediate layers are critical, whereas this work identifies that specific channels are critical—findings at different granularities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to discover that positional hidden states are a source of position bias.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covered multiple models x multiple tasks x multiple positional encoding types.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear visualizations and analysis.
- Value: ⭐⭐⭐⭐⭐ Direct practical improvements for long-context LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[NeurIPS 2025\] On the Role of Hidden States of Modern Hopfield Network in Transformer](../../NeurIPS2025/llm_nlp/on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)
- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)

</div>

<!-- RELATED:END -->
