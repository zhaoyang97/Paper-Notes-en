---
title: >-
  [Paper Note] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation
description: >-
  [ACL 2025][Model Compression][Long-context extension] This paper systematically analyzes two causes of short-text performance degradation in long-context LLMs (distribution drift and catastrophic forgetting), and proposes LongReD. Through two training objectives, namely short-text distillation and short-to-long distillation, LongReD minimizes the distribution discrepancy between the extended model and the original model, preserving short-text performance up to 99.4% of the or…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Long-context extension"
  - "knowledge distillation"
  - "distribution drift"
  - "catastrophic forgetting"
  - "positional encoding"
date: 2026-05-08
content_hash: 37628c223c483444
---

# LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation

**Conference**: ACL 2025  
**arXiv**: [2502.07365](https://arxiv.org/abs/2502.07365)  
**Code**: [https://github.com/RUCAIBox/LongReD](https://github.com/RUCAIBox/LongReD)  
**Area**: Model Compression  
**Keywords**: Long-context extension, knowledge distillation, distribution drift, catastrophic forgetting, positional encoding

## TL;DR
This paper systematically analyzes two causes of short-text performance degradation in long-context LLMs (distribution drift and catastrophic forgetting), and proposes LongReD. Through two training objectives, namely short-text distillation and short-to-long distillation, LongReD minimizes the distribution discrepancy between the extended model and the original model, preserving short-text performance up to 99.4% of the original model while maintaining long-text modeling capabilities.

## Background & Motivation
Large language models can extend their context windows to 128K or even 1M tokens by scaling positional encodings (e.g., ABF, PI) combined with lightweight continued pre-training. However, this extension often comes with a significant degradation in performance on short-text tasks. For example, Llama-3-8B exhibits a clear decline on short-text benchmarks like MMLU after being extended to 128K via ABF.

Although existing works have noticed this phenomenon, their analysis of the causes of degradation remains insufficient, and they lack effective mitigation strategies. The key insight of this paper is: **short-text performance degradation can be attributed to two factors—distribution drift and catastrophic forgetting**. Based on this finding, the authors propose to restore the internal distribution of the original model via knowledge distillation, thereby mitigating the degradation.

## Method

### Overall Architecture
LongReD is an extension training framework for context windows that simultaneously optimizes three objectives at each training step: long-text training, short-text distillation, and short-to-long distillation. These three objectives utilize datasets of different lengths: $\mathcal{D}_1$ (long text), $\mathcal{D}_2$ (short text, 1K length), and $\mathcal{D}_3$ (original window length). The original model serves as the teacher and the extended model serves as the student, utilizing distillation to maintain short-text capabilities.

### Key Designs
1. **Distribution Drift Analysis and Quantification**:

    - Two evaluation metrics are proposed: hidden state cosine similarity (hidden state similarity) and attention KL divergence, to quantify the distribution discrepancy between the extended model and the original model.
    - Experimental findings: continued pre-training can partially restore the distribution, but imperfect recovery persists. A larger RoPE base leads to a larger distribution discrepancy.
    - Key finding: MMLU performance retention rate is positively correlated with hidden state similarity, proving that distribution drift is a major cause of performance degradation.

2. **Catastrophic Forgetting Analysis**:

    - As training steps increase, short-text performance first recovers and then declines, presenting an inverted U-curve (beginning to decline after around 32 steps).
    - Mixing short-text data in training can alleviate forgetting (MMLU performance increases from 62.0 to 62.5), validating that long-text training leads to forgetting.

3. **Short-Text Distillation**:

    - Core Idea: Use the original model as the teacher to distill the hidden states of selected layers to the extended model.
    - Distillation layer selection: Select $M$ layers with the largest differences (rather than all layers) based on attention KL divergence to avoid over-constraining the model, which hurts long-text capability.
    - Loss function: The negative of the cosine similarity of hidden states on selected layers.

4. **Short-to-Long Distillation**:

    - Design Motivation: Bridge the gap between short-text distillation and long-text training, transferring short-text capabilities to long-text positions.
    - Mechanism: Use Skipped Positional Indices to apply positional encodings simulating long-text positions to short texts.
    - Split the position indices into three segments: head, mid, and tail, where head remains unchanged, the ending index of tail is set to the target length, and mid is uniformly sampled or sampled using the CREAM method.
    - Only distill the output distribution of the last layer (to avoid positional information interference in intermediate layers).

5. **Distillation Layer Selection Based on Attention KL Divergence**:

    - Calculate the KL divergence between the attention distributions of each layer of the original model and the extended model.
    - Select the $M$ layers with the largest KL divergence for distillation (plus the last layer).
    - Select 6 layers for 32K extension, and 3 layers for 128K extension to balance short- and long-text performance.

### Loss & Training
Total loss: $\mathcal{L}_{final} = \mathcal{L}_{long} + \alpha_1 \mathcal{L}_{short} + \alpha_2 \mathcal{L}_{s2l}$

- $\mathcal{L}_{long}$: Standard language model cross-entropy loss.
- $\mathcal{L}_{short}$: Negative of the hidden state cosine similarity of selected layers.
- $\mathcal{L}_{s2l}$: Negative of the cosine similarity of the last layer's output under skipped positional encoding.
- Hyperparameters: $\alpha_1=5, \alpha_2=10$, data token ratio of 4:3:1, overall training on 1B tokens.

## Key Experimental Results

### Main Results

| Model/Config | Short Avg. | RULER | Total Avg. |
|-----------|-----------|-------|--------|
| Llama-3-8B (original 8K) | 55.16 | - | - |
| 32K ABF + CPT (long-text only) | 51.00 | 82.80 | 56.30 |
| 32K ABF + LongReD-C | **54.85** | 84.98 | **59.87** |
| 128K ABF + CPT (long-text only) | 50.14 | 69.70 | 53.40 |
| 128K ABF + LongReD-U | 53.85 | **68.41** | **56.28** |
| Mistral-7B-v0.3 (original 32K) | 51.09 | - | - |
| 128K ABF + CPT | 40.68 | 44.63 | 41.35 |
| 128K ABF + LongReD-U | **47.69** | **53.60** | **48.68** |

### Ablation Study

| Config | Short Avg. | RULER | Note |
|------|-----------|-------|------|
| Full LongReD-C ($\alpha_1=5, \alpha_2=10$) | 54.85 | 84.98 | Baseline |
| Without short-text distillation ($\alpha_1=-, \alpha_2=15$) | Low | 85.48 | Significant drop in short-text performance |
| Without short-to-long distillation ($\alpha_1=5, \alpha_2=-$) | High | 83.61 | Drop in long-text performance |
| KL divergence layer selection (6) vs Uniform (6) | Comparable | 84.98 vs 82.53 | KL layer selection is superior |
| Distillation length 1K vs 8K | Comparable | 84.98 vs 75.54 | Short distillation length is superior |

### Key Findings
- Under the 32K ABF setup, LongReD-C preserves 99.4% of the original model's short-text performance (vs 92.5% for CPT).
- In the skipped position method, CREAM is superior at 4x extension, while Uniform is better at 16x extension (CREAM focuses excessively on middle positions).
- Compared to continual learning methods such as model merging and parameter-efficient fine-tuning, LongReD consistently performs better on short-text tasks.
- Longer distillation length actually yields worse results (RULER of 8K distillation is only 75.54 vs 84.98 for 1K), as overfitting damages spatial/positional information in the hidden states.

## Highlights & Insights
- This is the first work to systematically analyze the causes of short-text degradation after long-context extension, attributing them to two dimensions: distribution drift and catastrophic forgetting.
- The proposed positive correlation between hidden state similarity and performance retention rate is a valuable finding, providing an evaluation metric for future research.
- Short-to-long distillation elegantly transfers short-text capabilities to long-text positions by utilizing skipped positional indices.
- The overall framework is general and compatible with different positional encoding extension techniques (such as ABF and PI).

## Limitations & Future Work
- The experiments are trained only on 1B tokens; the degradation problem itself might not be as severe under large-scale training (100B+).
- Short-text distillation and long-text training are still conducted separately on texts of different lengths; future work can explore directly integrating distillation on long texts.
- The number of distillation layers, $\alpha_1$, and $\alpha_2$ require manual hyperparameter tuning, lacking an adaptive selection strategy.
- One could consider incorporating attention distillation (not just hidden state distillation) to further reduce distribution differences.
- Research idea: Explore whether progressive extension (incrementally increasing the target length rather than in one step) can alleviate distribution drift.

## Related Work & Insights
- Complementary to positional encoding extension methods such as CREAM and LongRoPE, providing improvements at the training strategy level.
- The usage of knowledge distillation here differs from traditional teacher-student distillation (compressing a large model into a small model); instead, it distills the extended version using the original model, which is a novel direction.
- This is connected to experience replay in the field of continual learning, where mixed training with short-text data is essentially a replay strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ For the first systematic analysis of the causes of short-text degradation in long-context extension. The method design is innovative, though individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly complete, featuring two base models, multiple extension methods and window sizes, and detailed ablation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain of "motivation analysis -> finding -> methodology design" is clear, and the discussion is fluent and well-founded.
- Value: ⭐⭐⭐⭐ Holds significant practical value for the long-context extension domain, being both general and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](../../ICML2025/model_compression/lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)

</div>

<!-- RELATED:END -->
