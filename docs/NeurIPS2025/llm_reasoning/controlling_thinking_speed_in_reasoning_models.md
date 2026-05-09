---
title: >-
  [Paper Note] Controlling Thinking Speed in Reasoning Models
description: >-
  [NeurIPS 2025][LLM Reasoning][thinking speed] By applying Representation Engineering (RepE) to extract steering vectors that control fast/slow thinking transitions from the hidden space of Large Reasoning Models (LRMs), and combining these with a real-time reasoning difficulty estimator based on inter-layer logit divergence, the method achieves training-free adaptive reasoning speed control — yielding an average of +1.3% accuracy improvement and −8.6% token reduction across 4 LRMs.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - thinking speed
  - representation engineering
  - System 1/2
  - test-time scaling
  - steering vector
date: 2026-05-08
content_hash: 037945c92cf9e158
---

# Controlling Thinking Speed in Reasoning Models

**Conference**: NeurIPS 2025
**arXiv**: [2507.03704](https://arxiv.org/abs/2507.03704)
**Code**: Implemented based on vLLM
**Area**: LLM Reasoning Efficiency / Representation Engineering
**Keywords**: thinking speed, representation engineering, System 1/2, test-time scaling, steering vector

## TL;DR
By applying Representation Engineering (RepE) to extract steering vectors that control fast/slow thinking transitions from the hidden space of Large Reasoning Models (LRMs), and combining these with a real-time reasoning difficulty estimator based on inter-layer logit divergence, the method achieves training-free adaptive reasoning speed control — yielding an average of +1.3% accuracy improvement and −8.6% token reduction across 4 LRMs.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI o1 achieve System 2-level deep reasoning through Long Chain-of-Thought (Long CoT). However, this paradigm uniformly applies complex thinking to every reasoning step, resulting in substantial redundant computation.

**Limitations of Prior Work**: Existing acceleration approaches either rely on prompt-based token budget control (e.g., Budget Forcing's "Final Answer:" truncation or Thought Extrapolation's "Wait" appending), to which LRMs are insensitive regarding temporal or length constraints, or require additional training (e.g., fine-tuning on fast/slow reasoning traces), which is costly.

**Key Challenge**: Human reasoning dynamically alternates between fast and slow thinking even within a single problem — trivial steps are skipped quickly while critical steps receive deeper analysis. Yet current LRMs lack the ability to dynamically adjust reasoning speed at the passage level within a single inference.

**Goal**: (1) Enable smooth switching between fast and slow thinking during inference; (2) Determine when to switch to optimally balance efficiency and accuracy.

**Key Insight**: The authors identify an intriguing phenomenon — LRM short responses and long responses have distinctly different opening tokens (short responses begin with "To" or "First", long responses with "Okay" or "Alright"), indicating that fast/slow thinking modes are linearly separable in the model's representation space. Representation Engineering (RepE) can thus extract directional vectors that govern this mode transition.

**Core Idea**: PCA is applied to the hidden layers of an LRM to extract steering vectors representing the "fast→slow" thinking direction. During inference, adding or subtracting these vectors enables token-level continuous control of thinking speed. Inter-layer logit divergence between early and final layers serves as a real-time difficulty signal to drive adaptive speed regulation.

## Method

### Overall Architecture
The method comprises two components: (1) **Thinking Speed Control** — extracting thinking speed steering vectors via representation engineering and injecting them at inference time; and (2) **Adaptive Control** — a sliding-window algorithm based on real-time reasoning difficulty estimation that dynamically adjusts steering intensity. The entire approach operates as a pure inference-time plug-in requiring no training.

### Key Designs

1. **Fast/Slow Thinking Steering Vector Extraction (Representation Reading)**:

    - Function: Identify the directional vector in the LRM's representation space that controls thinking speed.
    - Mechanism: Using 7.5k problems from the MATH training set, paired responses are sampled — fast-thinking responses (beginning with "To") and slow-thinking responses (normal openings). The first 2 reasoning steps of each response serve as stimuli; hidden states at the final token position across all layers are collected as $(h_i^+, h_i^-)$, difference vectors $d_i = h_i^+ - h_i^-$ are computed (half positive, half negated), and PCA extracts the first principal component as the steering vector $v$.
    - Design Motivation: RepE theory posits that high-level semantic concepts are encoded as linear directions in the hidden space. Fast/slow thinking, as a high-level cognitive function, should obey the same principle. PCA validation classification accuracy approaches 100%, confirming this hypothesis.

2. **Inference-Time Representation Controlling**:

    - Function: Inject the steering vector at each token generation step to modulate thinking speed.
    - Mechanism: For target layers $l \in L$, the hidden state is modified as $h^l \leftarrow h^l + \alpha \cdot v^l$. Positive $\alpha$ accelerates thinking (more concise output); negative $\alpha$ decelerates thinking (deeper reasoning with reflection and backtracking).
    - Design Motivation: Compared to prompt-based methods (e.g., appending "Wait" or truncating), representation-level intervention preserves the natural reasoning flow without disrupting the model's internal logic. Experiments show that under the same token budget, representation control outperforms Budget Forcing by an average of +11.4% Pass@1.

3. **Real-Time Reasoning Difficulty Estimation**:

    - Function: Assess the difficulty of the current reasoning step on a token-by-token basis during inference.
    - Mechanism: The Jensen-Shannon divergence between the next-token distributions of early layers and the final layer is used as a difficulty proxy: $d(x_t) = \text{avg}_{l \in L_e} \text{JSD}(p^N(\cdot|x_{<t}) \| p^l(\cdot|x_{<t}))$. High divergence indicates the need for deep processing, corresponding to complex reasoning behaviors such as reflection, computation, and logical derivation.
    - Design Motivation: Research shows that LLMs exhibit greater logit discrepancy between early and later layers when processing complex information. Validation confirms that the 100 tokens with the highest logit divergence correspond precisely to reflection tokens (Wait, Alternatively), computation tokens (equals, multiply), and analytical tokens (analysis, need).

4. **Sliding-Window Adaptive Speed Control Algorithm**:

    - Function: Dynamically adjust $\alpha$ based on the real-time difficulty signal.
    - Mechanism: A difficulty window $W$ of the most recent $k=8$ tokens is maintained. If the current token's difficulty exceeds the threshold $\mu_W + \lambda \cdot \sigma_W$ (analogous to anomaly detection), $\alpha$ is set to $\alpha_{\min}$ (decelerate/brake); otherwise, $\alpha$ is gradually increased (accelerate) up to the upper bound.
    - Design Motivation: This simulates human reasoning — simple steps are processed quickly, while critical reasoning junctures trigger slower, deeper deliberation.

### Loss & Training
No training is required. The steering vector is extracted once from a small dataset. The adaptive control algorithm is integrated directly into vLLM as an inference-time plug-in.

## Key Experimental Results

### Main Results (Adaptive Control)

| Model | Method | MATH-500 | AIME24 | AIME25 | GPQA Diamond |
|---|---|---|---|---|---|
| DS-R1-Distill-7B | Original | 92.9 / 3404 | 52.5 / 12451 | 40.0 / 13689 | 46.6 / 6189 |
| | 1xWait | 92.7 / 3744 | 52.1 / 12704 | 39.6 / 13867 | 45.9 / 9376 |
| | **Adaptive** | **93.7 / 3123** | **53.8 / 10851** | **42.3 / 12380** | **48.8 / 5422** |
| QwQ-32B | Original | 97.4 / 4305 | 76.7 / 13627 | 65.8 / 15852 | 62.7 / 7969 |
| | **Adaptive** | **97.4 / 4134** | **77.8 / 12365** | **67.4 / 15150** | **64.1 / 7639** |
| Qwen3-8B | Original | 96.8 / 5456 | 75.0 / 14754 | 62.9 / 17797 | 60.1 / 8379 |
| | **Adaptive** | **97.1 / 5171** | **77.5 / 13629** | **65.4 / 17411** | **61.2 / 7894** |

Averaged across 4 LRMs and 4 benchmarks: +1.3% accuracy, −8.6% tokens, entirely training-free.

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Random control vs. difficulty-driven control | Difficulty-driven outperforms random on all metrics | Validates the effectiveness of the JSD difficulty signal |
| $\lambda$ from 1.0→1.5→2.0→2.5 | Smaller $\lambda$ classifies more tokens as difficult; accuracy↑ tokens↑ | Directly controls the speed-accuracy trade-off |
| $\alpha_{\max}$ from 2→4→6→8 | Larger $\alpha_{\max}$ yields shorter responses; accuracy may drop | Extreme values (>16 or <−8) cause repetitive generation |
| Removing function-calling format | Adaptive control is critical for efficiency on simple tasks | |
| Stimulus position selection | Representations at the end of the initial segment perform best | First token (opening word embedding bias) and full reasoning chain (contains EOS signals) both underperform |

### Key Findings
- **Opening tokens determine thinking mode**: Appending "To" immediately after the LRM's `<think>` tag achieves 5.4× token compression (7B) while retaining 60% accuracy, revealing a natural fast/slow switching mechanism within LRMs.
- Thought Extrapolation ("Wait") performs poorly or even negatively: increasing the number of reflection steps actually decreases accuracy, indicating that prompt-level control is decoupled from the model's internal state.
- "Wait" tokens are poorly correlated with genuine slow-thinking mode: on AIME24, models produce an average of 55.3 "Wait" tokens, yet only 1 in 12.2 corresponds to a true mode transition.
- Orthogonal and composable with parallel search (Best-of-N): under low token budgets, accelerated models significantly outperform vanilla and NoThinking baselines in parallel search.

## Highlights & Insights
- **Representation engineering for test-time scaling — a first**: Prior applications of representation engineering focused on editing factual knowledge or controlling sentiment; this work is the first to apply it for controlling reasoning speed, enabling token-level continuous modulation that prompt-based methods cannot achieve.
- The **real-time difficulty estimation** design is particularly elegant: inter-layer logit divergence at different Transformer depths serves as a proxy for cognitive complexity without requiring any auxiliary model. High-divergence tokens precisely correspond to reflection, computation, and derivation behaviors — a highly insightful finding.
- **Fully training-free + vLLM plug-in**: The approach has strong practical value, deployable directly to existing LRM serving infrastructure without modifying model weights.

## Limitations & Future Work
- Different LRMs exhibit varying sensitivity to steering intensity; a unified $\alpha$ range is currently used, and model-agnostic automatic calibration is needed in future work.
- The sliding-window algorithm relies on heuristic rules; an ideal approach would end-to-end optimize the mapping from difficulty signals to steering intensity.
- Validation is limited to mathematical reasoning and programming tasks; the effect on open-ended generation (e.g., creative writing, dialogue) remains unknown.
- Extreme $\alpha$ values cause repetitive generation, suggesting the representation space may not be perfectly linear.

## Related Work & Insights
- **vs. ARM (2505.20258)**: ARM trains models via RL to select reasoning formats, requiring additional training; the present work is fully training-free and achieves finer-grained token-level control through representation editing. The two approaches are complementary — ARM selects macroscopic formats while this work regulates reasoning intensity at the microscopic level.
- **vs. Budget Forcing / s1**: These methods truncate or extend reasoning at fixed positions, constituting coarse-grained prompt-level control. The proposed representation control smoothly modulates reasoning style at the token level, outperforming them by an average of 11.4% under the same token budget.
- **vs. Thought Extrapolation**: Forcing reflection by appending "Wait" proves unstable or even harmful in experiments. This work reveals the underlying reason — "Wait" tokens are weakly correlated with genuine internal cognitive transitions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of representation engineering to reasoning speed control; the discovery of linear separability of fast/slow modes and the opening-token triggering mechanism are entirely novel insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four LRMs (DS-R1-7B/32B, QwQ-32B, Qwen3-8B), 4+ benchmarks, extensive ablations and case analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, accurate System 1/2 analogy, intuitive figures and tables.
- Value: ⭐⭐⭐⭐⭐ The training-free plug-in design has high deployment value; the JSD difficulty signal is transferable to other reasoning optimization scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[NeurIPS 2025\] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)
- [\[CVPR 2026\] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](../../CVPR2026/llm_reasoning/visref_visual_refocusing_while_thinking_improves_test-time_scaling_in_multi-moda.md)

</div>

<!-- RELATED:END -->
