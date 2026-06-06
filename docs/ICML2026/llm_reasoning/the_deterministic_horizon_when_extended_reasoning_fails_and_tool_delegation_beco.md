---
title: >-
  [Paper Note] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary
description: >-
  [ICML 2026][LLM Reasoning][Reasoning failure] The paper discovers a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder Transformers for deterministic state tracking tasks…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Reasoning failure"
  - "Chain of thought"
  - "Attention capacity"
  - "Tool delegation"
  - "State tracking"
date: 2026-05-08
content_hash: 031aa8747de1b919
---

# The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary

**Conference**: ICML 2026  
**arXiv**: [2606.00376](https://arxiv.org/abs/2606.00376)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Agent  
**Keywords**: Reasoning failure, Chain of thought, Attention capacity, Tool delegation, State tracking

## TL;DR
The paper discovers a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder Transformers for deterministic state tracking tasks, beyond which extended reasoning leads to performance collapse. It proves via information theory and large-scale empirical analysis (720k evaluations) that this is an **architectural capacity** failure rather than a "simplicity bias," and quantitatively demonstrates the **necessity** of tool delegation (e.g., symbolic solvers)—which can improve accuracy from 24-42% to 86-94%.

## Background & Motivation

**Background**: Current LLM reasoning paradigms generally assume that scaling Chain-of-Thought (CoT) improves accuracy (as seen in the massive reasoning overhead of o1 / DeepSeek-R1). Meanwhile, test-time compute has been proven more effective than parameter scaling.

**Limitations of Prior Work**: However, on tasks requiring **precise state tracking** (program execution, SQL queries, web navigation), model performance follows an inverted U-curve—deeper reasoning actually leads to lower accuracy. For example, in permutation puzzles, accuracy is 78% at depth 10, drops to 34% at depth 30, and becomes near-random at depth 50+. This contradicts the intuition that "extended reasoning is always beneficial."

**Contradictions in Existing Explanations**: Prior work (Wu et al. 2026) attributed this to "simplicity bias"—the tendency for models to prefer shorter reasoning paths. This theory predicts that encouraging long reasoning through fine-tuning should recover > 30% performance. However, experiments in this study find that fine-tuning only improves performance by 3.2%, significantly lower than predicted.

**Key Challenge**: The problem is not preference, but the **information-theoretic capacity limits of the decoder architecture regarding deterministic state maintenance**—autoregressive attention lacks the computational foundation to maintain precise state tracking.

**Goal**:
- Establish information-theoretic bounds for LLM state tracking.
- Derive a closed-form formula for the "Deterministic Horizon."
- Distinguish between **Decoherence (Capacity failure) vs. Simplicity Bias (Preference failure)**.
- Prove the **necessity** of tool delegation rather than merely its superiority.

**Key Insight**: Through attention entropy analysis + vanishing state signals, it is proven that the per-step error rate $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ grows linearly with depth, causing the overall accuracy to **decay super-exponentially**. Consequently, a sharp horizon $d^*$ exists, beyond which pure neural reasoning inevitably fails.

## Method

### Overall Architecture
A dual-track approach of **theory + experiment**:
- **Theoretical Track**: Deriving a context-related error model from attention entropy → Attention Bottleneck Theorem → Closed-form solution for the Deterministic Horizon.
- **Experimental Track**: 12 models × 8 task domains × 5 condition settings (720k evaluations), using the SSJ metric to distinguish causes of failure.

### Key Designs

1. **Context-Related Error Model (Equation 2)**:
    - **Function**: Characterizes how the per-step error rate grows with reasoning depth.
    - **Mechanism**: Traditional models assume per-step errors are independent and constant; this paper proposes $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ based on attention mechanics. $L_{\text{eff}}$ is the effective decoherence length (the number of reasoning steps a model can maintain usable resolution, which is much smaller than the context window $L$). This is derived via the signal-to-noise ratio $\epsilon(d) \propto H_d / A_{\text{anchor}}(d)$—attention entropy $H_d$ grows linearly with task load ($r = 0.73$), while anchor attention decays with depth.
    - **Design Motivation**: To explain why longer reasoning is **necessarily** more error-prone—it is not that the model "chooses" to fail, but that information is compressed and dissipated within the attention channels.

2. **Attention Bottleneck Theorem (Theorem 4.6)**:
    - **Function**: Establishes an information-theoretic upper bound for the number of distinct states a decoder can track.
    - **Mechanism**: At each step, autoregressive attention must compress all historical states through a fixed-capacity channel; due to softmax concentration, each head can focus on at most $O(\sqrt{L})$ positions (rather than $O(L)$). Under the assumption of low correlation in value vectors, the number of trackable states $|\mathcal{S}_{\text{track}}| \leq c \cdot 2^{H \cdot \log_2(L / H) \cdot d_h^{1/2}}$. For GPT-4o ($H = 96, L = 128K, d_h = 128$), this gives $2^{11,275}$ capacity—seemingly huge, but tracking a 16-element permutation for 50 steps requires $2^{2212}$ trajectories, which is still within the bound. The real bottleneck is the 44 bits of permutation information that must flow through attention concentrated on $O(\sqrt{L})$ positions at each step.
    - **Design Motivation**: To strictly demonstrate from information theory why "capacity appears large enough" but "actual failure" occurs—the implicit structure is the key.

3. **Deterministic Horizon Formula (Theorem 4.8)**:
    - **Function**: Provides a precise depth threshold beyond which the probability of pure CoT failure approaches 1.
    - **Mechanism**: Equating the decoherence bound $P(\text{correct at } m) \leq \exp(-m \epsilon_0 - \gamma m (m+1) / (2 L_{\text{eff}}))$ with a target success rate $\alpha$ (e.g., 0.5), and solving the quadratic equation yields $d^* = \frac{-\epsilon_0 L_{\text{eff}} + \sqrt{\epsilon_0^2 L_{\text{eff}}^2 + 2 \gamma L_{\text{eff}} \ln(1/\alpha)}}{\gamma}$. The quadratic term generates super-exponential decay, causing the horizon to exist sharply. The scaling relationship $d^* \propto \sqrt{d_h \cdot H}$ explains why larger models have higher horizons but sub-linear growth.
    - **Design Motivation**: To elevate curve fitting into theoretical prediction, making models comparable, results reproducible, and architectural improvement directions clear.

### Loss & Training
**Fine-tuning Experiment (The Gold Standard for Distinguishing "Capacity" vs. "Preference")**:
Llama-3.1-8B was fine-tuned using optimal-length CoT trajectories (5,000 samples), where $d^* \approx 20$. Results showed an improvement of only 3.2% (vs. the > 30% predicted by Wu et al.). Performance could not break 15% once the horizon was exceeded. If it were merely a preference, fine-tuning should have recovered significant performance; if it is an architectural limitation, fine-tuning is futile.

**State Space Jaccard (SSJ) Metric**:
$\text{SSJ}(d) = |∩| / |∪|$ (claimed state space vs. ground truth reachable state space). This is decomposed into precision and recall—if it were only a preference failure, SSJ should remain high; if it is a capacity failure, both decay. At depth 5, SSJ = 0.83; at depth 50, SSJ = 0.08 (strong evidence for capacity failure).

## Key Experimental Results

### Main Results: The Overwhelming Advantage of Tool Delegation

| Model | Condition | Permutation Puzzle Accuracy | SWE-Bench Accuracy | $d^*$ |
|------|------|----------|-------------|------|
| GPT-4o | Unconstrained CoT (C1) | 28.3 ± 1.8% | 24.1 ± 2.3% | 22 |
| GPT-4o | Tool Integrated (C3) | **89.7 ± 1.2%** | **86.4 ± 1.8%** | — |
| Claude-4.5-Opus | Unconstrained CoT | 34.8 ± 2.0% | 29.6 ± 2.5% | 27 |
| Claude-4.5-Opus | Tool Integrated | **93.6 ± 0.9%** | **91.2 ± 1.4%** | — |
| o3-mini | Unconstrained CoT | 42.1 ± 2.2% | 36.8 ± 2.6% | 31 |
| o3-mini | Tool Integrated | **94.2 ± 1.3%** | **92.7 ± 1.3%** | — |

Tool delegation achieves 86-94% across all models/tasks (vs. 24-42% for pure CoT), with an effect size of Cohen's d = 2.1-3.4.

### Ineffectiveness of Preference Manipulation

| Condition | Gain | Bayes Factor | Conclusion |
|------|---|---|---|
| Encourage Long Reasoning (C4) | +0.7-1.0% | $BF_{01} > 4$ | Supports null hypothesis |
| FT on Optimal Length (C5) | +3.2% (Llama-8B) | — | Far below > 30% prediction |

### Horizon Robustness Across Tasks

| Task Domain | $d^*$ Range | Description |
|------|---|------|
| Permutation Puzzles | 19-31 | Synthetic and controllable |
| SWE-Bench | 19-24 | Real: File state tracking |
| WebArena | 19-23 | Real: Web navigation + Session state |
| SQL-Multi | 21-26 | Real: Multi-table JOIN + Schema |

### Key Findings
- The horizon remains consistent across 8 different domains ($d^* \in [19, 26]$), indicating universality of the phenomenon.
- Monotonic decay in both SSJ precision and recall (rather than just recall decay) provides strong evidence pointing to **capacity failure** rather than preference failure.
- The high consistency of cross-model correlation ($r = 0.81 - 0.91$) strongly suggests that this is not dataset noise but a profound structural limitation.

## Highlights & Insights
- **Innovative Diagnostic Framework**: The SSJ precision/recall decomposition intuitively distinguishes between "the model can but won't" vs. "the model wants to but can't"—similar to sensitivity/specificity in medical diagnostics.
- **Elegant Marriage of Information Theory and Empiricism**: Deriving the closed-form solution for $d^*$ from first principles of attention entropy and validating it with 720,000 API calls; the paradigm of theory guiding experiments and experiments validating theory is highly persuasive.
- **Decisive Power of Fine-tuning Experiments**: While accuracy curves can have multiple interpretations, the > 30% prediction discrepancy in fine-tuning provides a decisive hypothesis test—inspiring other reasoning work to use fine-tuning as a gold standard for differentiation.
- **Engineering Implications**: Agent systems should not blindly pursue "longer chains of thought"; instead, they should adaptively switch to tools at $d^*$. This offers direct benefits for cost control (reducing ineffective reasoning tokens) and reliability (avoiding hallucinated state drift).

## Limitations & Future Work
- The analysis is specific to **deterministic** state tracking tasks (requiring precise state maintenance) and does not apply to open-ended generation (summarization, dialogue) or probabilistic reasoning (GSM8K-style arithmetic usually involves < 15 steps).
- The tightness of theoretical assumptions (effective attention window $O(\sqrt{L})$, low value-vector correlation) is not fully characterized; edge cases may have a ± 20% bias.
- Fine-tuning experiments were only conducted at the 8B scale; the fine-tuning landscape for larger scales (> 70B) remains unclear.
- Future directions: Designing hybrid architectures (attention + explicit working memory stacks) to break the $d^*$ limit; embedding horizon estimation within agent frameworks to dynamically select the ratio of tool usage vs. reasoning.

## Related Work & Insights
- **vs. Wu et al. (2026) "Simplicity Bias"**: The latter attributed failures to model preference optimization during learning, predicting a > 30% recovery via fine-tuning. Ours provides a complementary **architectural diagnosis**—even if the model intends to reason extensively, it cannot break through capacity limits. The hypotheses yield sharp predictions, and the fine-tuning experiments (3.2% vs. > 30%) support Ours.
- **vs. Overthinking Literature**: Focuses on phenomenon description and hyperparameter optimization. Ours provides a causal explanation—errors are not caused by overthinking but are inevitable due to the **failure of attention to compress state information**.
- **vs. Working Memory Limitations** (Gong & Zhang): Analyzes single-step bottlenecks via representation collapse. Ours extends this to **multi-step chains**, deriving empirically testable overall decay laws and decision thresholds.
- **vs. Tool-Augmented LMs** (Gao et al., Parisi et al.): These introduce tools to improve performance. Ours argues from **necessity**—pure reasoning inevitably fails beyond the horizon; tools are not a "luxury" but a "lifeline."

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Derived precise collapse points of LLM reasoning from information theory foundations and validated them experimentally; a quantitative diagnostic framework unprecedented in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  720,000 evaluations with statistical rigor (bootstrap CI, multiple comparison correction, Bayes factors, TOST equivalence testing).
- Writing Quality: ⭐⭐⭐⭐  Logical and clear, with theory and experiments complementing each other; arguments for some theoretical assumptions could be tighter.
- Value: ⭐⭐⭐⭐⭐  Direct engineering significance for agent systems, reasoning LLM design, and cost optimization; provides the first principled answer to "why extended reasoning sometimes fails and when to apply tools."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](../../ACL2026/llm_reasoning/reasoning_fails_where_step_flow_breaks.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](../../ACL2026/llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)

</div>

<!-- RELATED:END -->
