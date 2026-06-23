---
title: >-
  [Paper Note] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary
description: >-
  [ICML 2026][LLM Reasoning][Paper Note] This paper identifies a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder-only Transformers for tasks requiring deterministic state tracking, where extending reasoning beyond this threshold leads to performance collapse due to attention capacity limits. Through information theory and large-scale empirical an
tags:
  - ICML 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: 4c28d6b2f8409ac4
---
# The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary

**Conference**: ICML 2026  
**arXiv**: [2606.00376](https://arxiv.org/abs/2606.00376)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Agent  
**Keywords**: Reasoning Failure, Chain-of-Thought, Attention Capacity, Tool Delegation, State Tracking

## TL;DR
This paper identifies a "**Deterministic Horizon**" (approx. 19-31 steps) in decoder-only Transformers for tasks requiring deterministic state tracking, where extending reasoning beyond this threshold leads to performance collapse due to attention capacity limits. Through information theory and large-scale empirical analysis (720,000 evaluations), the authors prove this is an **architectural capability** failure ("Decoherence") rather than a "Simplicity Bias," and quantitatively demonstrate the **necessity** of tool delegation (e.g., symbolic solvers)—which can restore accuracy from 24-42% to 86-94%.

## Background & Motivation

**Background**: Current LLM reasoning paradigms generally assume that extended Chain-of-Thought (CoT) improves accuracy, as seen in the large-scale reasoning overhead of models like o1 and DeepSeek-R1. Furthermore, test-time compute has been proven more effective than parameter scaling in many contexts.

**Limitations of Prior Work**: However, on tasks requiring **precise state tracking** (e.g., program execution, SQL queries, web navigation), model performance follows an inverted U-curve—deeper reasoning actually leads to lower accuracy. For instance, in permutation puzzles, accuracy is 78% at depth 10 but drops to 34% at depth 30, approaching random chance at depths 50+. This contradicts the intuition that "scaling reasoning is always beneficial."

**Contradiction in Existing Explanations**: Prior work (Wu et al. 2026) attributes this to "Simplicity Bias"—a tendency of models to prefer shorter reasoning paths. This theory predicts that fine-tuning to encourage long reasoning should recover >30% of performance. However, this study's experiments show that fine-tuning only yields a 3.2% improvement, significantly lower than predicted.

**Key Challenge**: The fundamental problem is not preference, but the **information-theoretic capacity limits of state maintenance within the decoder architecture itself**—autoregressive attention lacks the computational foundation to maintain precise state tracking over long horizons.

**Goal**:
- Establish information-theoretic bounds for LLM state tracking.
- Derive a closed-form formula for the "Deterministic Horizon."
- Distinguish between **Decoherence (Capability Failure) vs. Simplicity Bias (Preference Failure)**.
- Prove the **necessity** of tool delegation rather than just its superiority.

**Key Insight**: By analyzing attention entropy and vanishing state signals, it is demonstrated that the error rate per step $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ grows linearly with depth. This results in a **super-exponential decay** of overall accuracy, necessitating a precise horizon $d^*$ beyond which pure neural reasoning inevitably fails.

## Method

### Overall Architecture
The paper investigates why longer CoT leads to lower accuracy in state-tracking tasks, attributing the cause to the decoder architecture itself rather than training biases. The research proceeds along two parallel tracks: The theoretical track starts from attention entropy to build an error model that scales with depth, derives an upper bound for the number of states the attention mechanism can track (Bottleneck Theorem), and solves for the depth threshold $d^*$ (Deterministic Horizon). The experimental track validates these predictions through 720,000 evaluations across 12 models, 8 domains, and 5 conditions, using two diagnostic tools (fine-tuning controls and SSJ metrics) to confirm this is a **capability failure**.

### Key Designs

**1. Context-Dependent Error Model: Modeling Error Rate as a Function of Depth**

The root of the inverted U-curve lies in the fact that traditional analyses assume independent and constant error rates per step, which would only suggest linear degradation and fails to explain the collapse at depth 50. This paper derives a per-step error rate $\epsilon(d) = \epsilon_0 + \gamma \cdot d / L_{\text{eff}}$ from attention mechanics, where $L_{\text{eff}}$ is the **effective decoherence length**—the number of steps a model can maintain usable resolution, which is much smaller than the nominal context window $L$. This growth term is derived from the signal-to-noise ratio $\epsilon(d) \propto H_d / A_{\text{anchor}}(d)$: attention entropy $H_d$ rises near-linearly with task load, while anchor attention $A_{\text{anchor}}(d)$ decays with depth. This frames the failure as a physical consequence of information dissipation in the attention channel.

**2. Attention Bottleneck Theorem (Theorem 4.6): Information-Theoretic Upper Bound on States**

To explain why capacity reaches a limit, the theorem notes that autoregressive attention must compress all historical states into a fixed-capacity channel. The concentration property of softmax implies each head can focus on at most $O(\sqrt{L})$ positions rather than $O(L)$. Under low-correlation assumptions for value vectors, the number of trackable states $|\mathcal{S}_{\text{track}}| \leq c \cdot 2^{H \cdot \log_2(L / H) \cdot d_h^{1/2}}$. For GPT-4o, this capacity seems massive, yet the model fails on tasks requiring far fewer states. This points to the bottleneck being the **per-step requirement** to pass state information through a stream concentrated on $O(\sqrt{L})$ positions—the sequential structure is the fatal flaw.

**3. Deterministic Horizon Formula (Theorem 4.8): Solving for the Collapse Threshold**

By setting the decoherence bound $P(\text{correct at } m) \leq \exp\!\big(-m \epsilon_0 - \gamma m (m+1) / (2 L_{\text{eff}})\big)$ equal to a target success rate $\alpha$ (e.g., 0.5), the horizon $d^*$ is solved as a quadratic equation:

$$d^* = \frac{-\epsilon_0 L_{\text{eff}} + \sqrt{\epsilon_0^2 L_{\text{eff}}^2 + 2 \gamma L_{\text{eff}} \ln(1/\alpha)}}{\gamma}.$$

The quadratic term $m(m+1)$ in the exponent causes success probability to decay super-exponentially, making the horizon a sharp "wall" rather than a gradual transition. The scaling relationship $d^* \propto \sqrt{d_h \cdot H}$ explains why larger models have higher horizons but only see sub-linear growth—scaling parameters alone cannot easily push through this architectural wall.

**4. Fine-tuning Control Experiment: Evidence for Capability failure**

To challenge the "Simplicity Bias" hypothesis (which suggests models just prefer shorter paths), the authors fine-tuned Llama-3.1-8B (where $d^* \approx 20$) on 5,000 optimal-length CoT trajectories. The result was a mere 3.2% improvement, and the model could not break 15% accuracy once past the horizon. This minimal improvement confirms that the limit is a hard architectural capacity constraint rather than a learned preference.

**5. State Space Jaccard (SSJ) Metric: Analyzing Failure via Precision and Recall**

The SSJ is defined as the Jaccard similarity between the state space the model **claims** to maintain and the **actually reachable** state space: $\text{SSJ}(d) = |\cap| / |\cup|$. This is further decomposed into precision and recall. If the failure were due to bias, the SSJ should remain high. Instead, experiments show SSJ drops from 0.83 (depth 5) to 0.08 (depth 50), with precision and recall declining **simultaneously**. This provides strong evidence for capability failure.

## Key Experimental Results

### Main Results: The Overwhelming Advantage of Tool Delegation

| Model | Condition | Permutation Puzzle Acc | SWE-Bench Acc | $d^*$ |
|------|------|----------|-------------|------|
| GPT-4o | Unconstrained CoT (C1) | 28.3 ± 1.8% | 24.1 ± 2.3% | 22 |
| GPT-4o | Tool Integration (C3) | **89.7 ± 1.2%** | **86.4 ± 1.8%** | — |
| Claude-4.5-Opus | Unconstrained CoT | 34.8 ± 2.0% | 29.6 ± 2.5% | 27 |
| Claude-4.5-Opus | Tool Integration | **93.6 ± 0.9%** | **91.2 ± 1.4%** | — |
| o3-mini | Unconstrained CoT | 42.1 ± 2.2% | 36.8 ± 2.6% | 31 |
| o3-mini | Tool Integration | **94.2 ± 1.3%** | **92.7 ± 1.3%** | — |

Tool delegation achieves 86-94% accuracy across all models and tasks (compared to 24-42% for pure CoT), with an effect size of Cohen's d = 2.1-3.4.

### Ineffectiveness of Preference Manipulation

| Condition | Improvement | Bayes Factor | Conclusion |
|------|---|---|---|
| Encouraging Long Reasoning (C4) | +0.7-1.0% | $BF_{01} > 4$ | Supports null hypothesis |
| Fine-tuning Opt. Length (C5) | +3.2% (Llama-8B) | — | Far below >30% prediction |

### Horizon Robustness Across Tasks

| Domain | $d^*$ Range | Note |
|------|---|------|
| Permutation Puzzles | 19-31 | Synthetic & Controllable |
| SWE-Bench | 19-24 | Real-world: File state tracking |
| WebArena | 19-23 | Real-world: Nav + Session state |
| SQL-Multi | 21-26 | Real-world: Multi-table JOINs |

### Key Findings
- The horizon remains consistent across 8 domains ($d^* \in [19, 26]$), indicating a universal phenomenon.
- Both SSJ precision and recall decay monotonically, providing strong evidence for **capability failure** over preference bias.
- High cross-model correlation ($r = 0.81 - 0.91$) suggests these results are due to structural limitations rather than dataset noise.

## Highlights & Insights
- **Innovative Diagnostic Framework**: The SSJ precision/recall decomposition intuitively distinguishes between "the model can but won't" vs. "the model wants to but can't."
- **Elegant Blend of Theory and Empiricism**: Deriving a closed-form solution for $d^*$ from first principles (attention entropy) and validating it with 720,000 API calls provides a highly persuasive paradigm.
- **Decisive Fine-tuning Evidence**: While accuracy curves can have multiple interpretations, the failure of fine-tuning to meet the >30% recovery prediction provides a definitive hypothesis test.
- **Engineering Insights**: Agent systems should not blindly pursue "longer thinking chains" for state-heavy tasks. Instead, they should adaptively switch to tools at the $d^*$ threshold to control costs and avoid hallucinated state drift.

## Limitations & Future Work
- Analysis is specific to **deterministic** state tracking and may not apply to open-ended generation (summarization) or probabilistic reasoning (arithmetic like GSM8K usually < 15 steps).
- The tightness of theoretical assumptions (e.g., $O(\sqrt{L})$ effective window) is not fully characterized; boundaries may vary by ±20%.
- Fine-tuning experiments were primarily conducted at the 8B parameter scale; the landscape for larger models (>70B) remains to be clarified.
- Future directions: Designing hybrid architectures (attention + explicit memory stacks) to break the $d^*$ limit and embedding horizon estimation within agent frameworks.

## Related Work & Insights
- **vs. Wu et al. (2026) "Simplicity Bias"**: While Wu attributes the collapse to learned preferences, this paper provides a complementary **architectural diagnosis**—even if the model "wants" to reason longer, it cannot exceed its capacity. The fine-tuning results (3.2% vs. >30%) support this paper’s hypothesis.
- **vs. Overthinking Literature**: Most work focuses on descriptive phenomena and hyperparameter tuning; this paper provides a causal explanation based on **attention compression failure**.
- **vs. Working Memory Limits (Gong & Zhang)**: While previous work analyzed single-step bottlenecks, this paper extends this to **multi-step chains** to derive measurable decay laws and decision thresholds.
- **vs. Tool-Augmented LMs**: Unlike work that adds tools solely for performance, this paper argues from the perspective of **necessity**—beyond the horizon, pure reasoning is mathematically destined to fail.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First quantitative diagnostic framework for the collapse point of LLM reasoning derived from info-theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  720,000 evaluations with rigorous statistical testing (Bayes factors, TOST).
- Writing Quality: ⭐⭐⭐⭐  Clear logic; theoretical and experimental findings complement each other well.
- Value: ⭐⭐⭐⭐⭐  Provides direct engineering guidance for agent design and cost optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](../../ACL2026/llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](../../ACL2026/llm_reasoning/reasoning_fails_where_step_flow_breaks.md)

</div>

<!-- RELATED:END -->
