---
title: >-
  [Paper Note] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary
description: >-
  [ICML 2026][LLM Reasoning][Paper Note] This paper identifies a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder-only Transformers for deterministic state-tracking tasks due to attention capacity constraints, beyond which extended reasoning leads to performance collapse. Using information theory and large-scale empirical analysis (720,000 evaluat
tags:
  - ICML 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: a5869e46e296c314
---
# The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary

**Conference**: ICML 2026  
**arXiv**: [2606.00376](https://arxiv.org/abs/2606.00376)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Agents  
**Keywords**: Reasoning Failure, Chain-of-Thought, Attention Capacity, Tool Delegation, State Tracking

## TL;DR
This paper identifies a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder-only Transformers for deterministic state-tracking tasks due to attention capacity constraints, beyond which extended reasoning leads to performance collapse. Using information theory and large-scale empirical analysis (720,000 evaluations), it proves this is an **architectural capacity** failure rather than a "simplicity bias" and quantifies the **necessity** of tool delegation (e.g., symbolic solvers), which improves accuracy from 24-42% to 86-94%.

## Background & Motivation

**Background**: Current LLM reasoning paradigms generally assume that extending Chain-of-Thought (CoT) improves accuracy (as seen in the massive reasoning overhead of o1 / DeepSeek-R1). Simultaneously, test-time compute has been proven more effective than parameter scaling.

**Limitations of Prior Work**: However, on tasks requiring **precise state tracking** (program execution, SQL queries, web navigation), model performance follows an inverted U-curve—deeper reasoning leads to lower accuracy. For example, in permutation puzzles, accuracy is 78% at depth 10 but drops to 34% at depth 30 and becomes near-random at depth 50+. This contradicts the intuition that "extended reasoning is always beneficial."

**Key Challenge**: Prior work (Wu et al. 2026) attributed this to "simplicity bias," where models prefer shorter reasoning paths. This theory predicts that fine-tuning to encourage long reasoning should recover > 30% performance. However, this study finds that fine-tuning only improves performance by 3.2%, significantly lower than predicted. The fundamental problem is not preference but the **information-theoretic capacity limits of deterministic state maintenance in decoder architectures**—autoregressive attention lacks the computational basis to sustain precise state tracking.

**Goal**:
- Establish information-theoretic bounds for LLM state tracking.
- Derive a closed-form formula for the "Deterministic Horizon."
- Differentiate between **Decoherence (Capacity Failure) vs. Simplicity Bias (Preference Failure)**.
- Demonstrate the **necessity** of tool delegation, rather than just its superiority.

**Key Insight**: Through attention entropy analysis and vanishing state signals, the authors prove that the step-wise error rate $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ grows linearly with depth, leading to a **super-exponential decay** in overall accuracy. Consequently, a precise horizon $d^*$ exists, beyond which pure neural reasoning inevitably fails.

## Method

### Overall Architecture
This paper addresses why longer CoT leads to lower accuracy in state-tracking tasks, attributing the cause to the decoder architecture itself rather than training preferences. The research proceeds along two tracks: The theoretical track starts from attention entropy to build an error model growing with depth, derives an upper bound on the number of states attention can track (Bottleneck Theorem), and solves for the depth threshold $d^*$ (Deterministic Horizon). The experimental track validates these predictions using 720,000 evaluations across 12 models, 8 task domains, and 5 conditions, utilizing two diagnostic instruments (fine-tuning controls and SSJ metrics) to prove this is a **capacity failure**.

### Key Designs

**1. Context-Aware Error Model: Step-wise Error as a Function of Depth**

The inverted U-curve stems from the flaw in traditional analysis that assumes step-wise error is independent and constant, which wrongly predicts only linear degradation. This paper derives $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ from attention mechanics, where $L_{\text{eff}}$ is the **effective decoherence length**—the number of steps a model can maintain usable resolution. This is derived from the signal-to-noise ratio $\epsilon(d) \propto H_d / A_{\text{anchor}}(d)$: attention entropy $H_d$ rises near-linearly with task load, while anchor attention $A_{\text{anchor}}(d)$ decays with depth. This frames failure as a physical consequence of information dissipation in the attention channel.

**2. Attention Bottleneck Theorem (Theorem 4.6): Information-Theoretic Upper Bound**

Autoregressive attention must compress all historical states into a fixed-capacity channel. Due to the concentration of softmax, each head can focus on at most $O(\sqrt{L})$ positions rather than $O(L)$. Under low-correlation assumptions for value vectors, the number of trackable distinct states is $|\mathcal{S}_{\text{track}}| \leq c \cdot 2^{H \cdot \log_2(L / H) \cdot d_h^{1/2}}$. For GPT-4o, while total capacity seems large, the requirement to squeeze high-entropy state information through the bottleneck at **every step** creates the fatal constraint.

**3. Deterministic Horizon Formula (Theorem 4.8): Solving for the Collapse Threshold**

By setting the decoherence bound $P(\text{correct at } m) \leq \exp\!\big(-m \epsilon_0 - \gamma m (m+1) / (2 L_{\text{eff}})\big)$ equal to a target success rate $\alpha$, the horizon is derived as:

$$d^* = \frac{-\epsilon_0 L_{\text{eff}} + \sqrt{\epsilon_0^2 L_{\text{eff}}^2 + 2 \gamma L_{\text{eff}} \ln(1/\alpha)}}{\gamma}.$$

The quadratic term $m(m+1)$ in the exponent causes super-exponential decay, making the horizon a sharp "wall" rather than a gradual transition. The scaling relationship $d^* \propto \sqrt{d_h \cdot H}$ explains why larger models have higher horizons but can only push the wall back sub-linearly.

**4. Fine-tuning Control Experiments: Distinguishing Capacity vs. Preference**

To counter the "simplicity bias" hypothesis, the authors fine-tuned Llama-3.1-8B (where $d^* \approx 20$) using 5,000 optimal-length CoT trajectories. Results showed only a 3.2% improvement, and performance could not exceed 15% beyond the horizon. This minimal improvement confirms that the failure is a hard architectural limit rather than a preference that can be corrected by training.

**5. State Space Jaccard (SSJ) Metric: Decoupling Precision and Recall**

SSJ is defined as the Jaccard similarity between the state space the model **claims** to maintain and the **actually reachable** state space: $\text{SSJ}(d) = |\cap| / |\cup|$. If failure were due to preference, SSJ would remain high; however, both precision and recall decay simultaneously as depth increases (from 0.83 at depth 5 to 0.08 at depth 50), providing strong evidence of capacity failure.

## Key Experimental Results

### Main Results: Dominance of Tool Delegation

| Model | Condition | Permutation Accuracy | SWE-Bench Accuracy | $d^*$ |
|------|------|----------|-------------|------|
| GPT-4o | Unconstrained CoT (C1) | 28.3 ± 1.8% | 24.1 ± 2.3% | 22 |
| GPT-4o | Tool Integration (C3) | **89.7 ± 1.2%** | **86.4 ± 1.8%** | — |
| Claude-4.5-Opus | Unconstrained CoT | 34.8 ± 2.0% | 29.6 ± 2.5% | 27 |
| Claude-4.5-Opus | Tool Integration | **93.6 ± 0.9%** | **91.2 ± 1.4%** | — |
| o3-mini | Unconstrained CoT | 42.1 ± 2.2% | 36.8 ± 2.6% | 31 |
| o3-mini | Tool Integration | **94.2 ± 1.3%** | **92.7 ± 1.3%** | — |

Tool delegation achieves 86-94% across all models/tasks (vs 24-42% for pure CoT), with a Cohen's d of 2.1-3.4.

### Key Findings
- **Horizon Robustness**: $d^*$ remains consistent ($d^* \in [19, 26]$) across 8 domains, including SWE-Bench (file state tracking) and WebArena (navigation), indicating a universal phenomenon.
- **Preference Ineffectiveness**: Encouraging longer reasoning (C4) yielded only +0.7-1.0% improvement, supporting the null hypothesis ($BF_{01} > 4$).
- **Decay Characteristics**: Both SSJ precision and recall decay monotonically, confirming that models lack the capacity to maintain states over long horizons.

## Highlights & Insights
- **Diagnostic Framework**: The SSJ decomposition provides a quantitative way to distinguish between "can do but won't" (preference) and "wants to but can't" (capacity).
- **Theory-Empirical Coupling**: Deriving the $d^*$ closed-form solution from first principles of attention entropy and validating it with 720k API calls sets a strong precedent for theoretical LLM research.
- **Engineering Implications**: Agent systems should not blindly pursue longer CoT. Instead, they should adaptively switch to tools at the $d^*$ threshold to control costs and prevent hallucinatory state drift.

## Limitations & Future Work
- The analysis is specific to **deterministic** state-tracking tasks requiring precise maintenance and may not apply to open-ended generation (summarization) or probabilistic reasoning.
- Theoretical assumptions regarding the $O(\sqrt{L})$ effective window may have ± 20% variance in edge cases.
- Fine-tuning tests were limited to the 8B scale; the effects on models > 70B remain to be explored.
- Future work: Designing hybrid architectures (attention + explicit working memory stacks) to break the $d^*$ limit.

## Related Work & Insights
- **vs. Wu et al. (2026)**: While Wu et al. attribute failure to simplicity bias, this paper proves that even if models are forced to reason longer, capacity constraints prevent success.
- **vs. Overthinking Literature**: Rather than just describing the phenomenon, this paper provides a causal explanation: error is an inevitable result of **attention's failure to compress state information**.
- **vs. Tool-Augmented LMs**: While prior work uses tools for performance, this paper argues for their **necessity**—beyond the horizon, pure reasoning is mathematically destined to fail.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First quantitative diagnostic framework for the collapse point of LLM reasoning).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (720k evaluations with rigorous statistical testing).
- **Writing Quality**: ⭐⭐⭐⭐ (Clear logic, though some theoretical assumptions could be tighter).
- **Value**: ⭐⭐⭐⭐⭐ (Directly informs the design and cost-optimization of agentic systems and reasoning models).

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](../../ACL2026/llm_reasoning/reasoning_fails_where_step_flow_breaks.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](../../ACL2026/llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)

</div>

<!-- RELATED:END -->
