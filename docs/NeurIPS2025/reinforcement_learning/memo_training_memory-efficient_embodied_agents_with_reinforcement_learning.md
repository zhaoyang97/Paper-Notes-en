---
title: >-
  [Paper Note] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Memory Augmentation] This paper proposes Memo, a Transformer-based memory-augmented framework that periodically generates summary tokens to compress historical context. Memo matches or exceeds the performance of full-context Transformers while reducing the KV cache at inference time by 8–10×, and demonstrates superior generalization to long contexts as well as robustness under streaming inference.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Memory Augmentation
  - Transformer
  - Context Compression
  - Long-Horizon Planning
  - Embodied Intelligence
date: 2026-05-08
content_hash: f2ff1c87bab1f25a
---

# Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.19732](https://arxiv.org/abs/2510.19732)
**Code**: [GitHub](https://github.com/gunshi/memo)
**Area**: Reinforcement Learning
**Keywords**: Memory Augmentation, Transformer, Context Compression, Long-Horizon Planning, Embodied Intelligence

## TL;DR

This paper proposes Memo, a Transformer-based memory-augmented framework that periodically generates summary tokens to compress historical context. Memo matches or exceeds the performance of full-context Transformers while reducing the KV cache at inference time by 8–10×, and demonstrates superior generalization to long contexts as well as robustness under streaming inference.

## Background & Motivation

Embodied agents operating on long-horizon tasks must leverage historical experience for decision-making, yet existing approaches face a fundamental tension:

**Quadratic attention bottleneck of Transformers**: Full-context Transformers (FCT) attend to all historical timesteps at every step, incurring $O(n^2)$ attention complexity. Processing long sequences entails prohibitive computation and memory costs; gradients must propagate through extensive sequences during training, and the KV cache grows unboundedly at inference.

**Fixed-capacity limitations of recurrent models**: While RNNs and similar recurrent models are memory-efficient, their fixed-size hidden states cannot adequately retain long-range dependencies, and gradients tend to vanish or explode over long sequences.

**Shortcomings of existing compression methods**: In the language modeling literature, the Recurrent Memory Transformer (RMT) employs a fixed-size memory, and Autocompressors (AC) truncate gradient propagation during fine-tuning. Neither approach supports effective credit assignment over long horizons as required by reinforcement learning.

**Core motivation**: *Can a Transformer learn to "summarize" past experience, preserving expressiveness while achieving memory efficiency?*

## Method

### Overall Architecture

Memo introduces a context summarization mechanism into a standard Transformer policy. During training, the input sequence is partitioned into equal-length segments; at the end of each segment, learnable summary embeddings prompt the Transformer to produce summary tokens. These summary tokens are stored in a dedicated memory buffer and accessed by subsequent timesteps via attention, replacing the full raw context. Historical information is thus represented in compressed form rather than maintained verbatim.

### Key Designs

1. **Context Summarization Mechanism**: The long sequence is divided into segments of length $l_{seg}$, and $l_{sum}$ summary tokens are generated at the end of each segment. The key innovation is **summary accumulation**—unlike RMT, which retains only the most recent summary, Memo retains all historical summary tokens so that any past information can directly influence current decisions via attention. At timestep $t$, the model's input context consists of $n \times l_{sum}$ summary tokens (where $n = \lfloor t/l_{seg} \rfloor$) concatenated with the observations in the current segment. This creates a residual-like gradient shortcut, as each summary vector participates directly in optimizing subsequent losses.

2. **Attention Mask and Positional Encoding**: A causal mask is applied such that the current timestep can attend to all historical summary tokens and to observations within the current segment, but **cannot attend** to raw observations that preceded the most recent summary. This enforces an information bottleneck—historical information must pass through summary tokens. Positional indices are assigned in two parts: summary tokens are indexed from $0$ to $n \times l_{sum} - 1$, and observations in the current segment begin at $n \times l_{sum}$ and increment sequentially, preserving relative positional awareness.

3. **Segment Length Randomization**: During training, segment length is sampled uniformly within $\pm 20\%$ of the nominal value (e.g., $[205, 307]$ for $l_{seg} = 256$), while data collection and evaluation use a fixed length. This prevents overfitting to fixed segment boundaries and induces an implicit curriculum—short segments correspond to easy compression tasks and long segments to harder ones.

4. **KV Cache Consistency**: In on-policy RL, policy updates invalidate the KV cache because model weights change. Following ReLIC, Memo flushes the entire KV cache after each policy update and recomputes all summary vectors, ensuring that cached representations remain consistent with the current policy.

### Loss & Training

Memo is trained end-to-end via RL objectives, with summary generation incorporated as a sub-task within the overall optimization. Two RL paradigms are supported:
- **On-policy**: Integrated into ReLIC (an improved variant of DD-PPO), leveraging frequent updates during rollouts to facilitate Transformer policy learning.
- **Off-policy**: Integrated into AMAGO, using a shared Transformer backbone with a unified actor-critic loss.

## Key Experimental Results

### Main Results

**ExtObjNav Task (32k-step evaluation, 10 seeds)**

| Method | Success Rate (SR) | SPL | Context Tokens | KV Cache Ratio |
|---|---|---|---|---|
| FCT (Full-Context) | ~52% | ~22% | 4096 | 1× |
| Memo | **~60%** | **~24.5%** | ~512 | **1/8×** |
| RMT-32 | ~55% | ~22% | Fixed | — |
| RMT-64 | ~56% | ~22% | Fixed | — |
| no-IEA | ~35% | ~15% | — | — |
| AC (TBTT) | ~45% | ~18% | ~512 | 1/8× |

**Dark-Key-To-Door Task (Off-policy AMAGO, 3 seeds)**

| Method | Mean Return | Convergence | Stability |
|---|---|---|---|
| FCT | ~95 | 35M steps | Performance drops at 35–40M steps |
| Memo | **~95** | **25M steps** | Stable after convergence, no degradation |
| RMT | ~90 | 35M steps | Relatively stable |

### Ablation Study

| Configuration | SR@10k | SR@32k | Notes |
|---|---|---|---|
| Memo ($l_{sum}=32$) | **~60%** | **~55%** | Optimal 8× compression |
| Memo ($l_{sum}=16$) | ~55% | ~50% | 16× compression, information loss |
| Memo ($l_{sum}=64$) | ~52% | ~42% | 4× compression, poor positional extrapolation |
| Memo w/o segment randomization | ~48% | ~40% | Significant degradation in both training and evaluation |
| Streaming Memo | ~60% | **~58%** | Performance maintained or slightly improved after truncation |
| Streaming FCT | ~52% | ~35% | Sharp performance drop after 6k steps |

### Key Findings

- **Summarized compression outperforms full context**: Memo achieves 7.5% higher SR and 2.5% higher SPL than FCT using only 1/8 the tokens. This counterintuitive result suggests that the information bottleneck forces the model to learn better task-relevant compression.
- **Accumulated summaries outperform fixed memory**: Memo converges approximately 10M steps faster than RMT on Dark-Key-To-Door, reflecting the advantage of the residual gradient shortcut provided by summary accumulation.
- **Long-range gradient propagation is critical**: AC's truncated gradients (TBTT) yield substantially worse performance on long-horizon tasks, demonstrating that credit assignment in RL requires gradients to flow through all summarization steps.
- **Robustness under streaming inference**: Streaming Memo maintains or even improves performance beyond 6k steps, whereas Streaming FCT degrades sharply.

## Highlights & Insights

- The counterintuitive core finding is that **an information bottleneck is an advantage rather than a constraint**—forcing information to pass through summary tokens in fact improves the extraction of task-relevant signals.
- The design of summary accumulation is particularly elegant: it combines the expressiveness of Transformers (every historical summary is directly accessible via attention) with the efficiency of recurrent models (raw observations need not be retained), while mitigating gradient vanishing through a residual-like gradient pathway.
- The discovery that segment length randomization acts as an implicit curriculum is a finding worthy of replication in other settings.

## Limitations & Future Work

- Memory consolidation—progressively compressing older summaries to further reduce storage—is not explored.
- Long-context generalization remains limited: a model trained on 4k steps generalizes to ~10k, and one trained on 16k generalizes to ~24k, indicating restricted extrapolation capacity.
- Semantic generalization is not investigated, e.g., navigating to entirely novel object categories rather than new positions of previously seen objects within fixed environments.
- Performance is highly sensitive to the number of summary tokens ($l_{sum}=64$ performs substantially worse than $l_{sum}=32$), and an adaptive selection mechanism is absent, though the optimal value likely varies across tasks.

## Related Work & Insights

- The key distinction from Autocompressors in NLP is that AC fine-tunes pretrained models with truncated gradient propagation, whereas Memo trains from scratch with full gradients. This highlights a fundamental difference in context compression requirements between NLP and RL.
- ReLIC and AMAGO demonstrate the potential of Transformers in RL; Memo can be viewed as an orthogonal memory-augmented extension of both frameworks.
- The streaming inference scheme proposed in this work requires no architectural modifications (unlike, e.g., StreamingLLM) and merits evaluation in other long-sequence RL tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Adapts context compression ideas from language modeling to RL and identifies the critical requirement for full gradient propagation in RL settings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across both grid-world and 3D navigation environments with 10-seed statistics and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with intuitive illustrations and precise articulation of findings.
- Value: ⭐⭐⭐⭐ Provides a practical solution to the long-term memory problem in embodied intelligence.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling](learning_memory-enhanced_improvement_heuristics_for_flexible_job_shop_scheduling.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)

<!-- RELATED:END -->
