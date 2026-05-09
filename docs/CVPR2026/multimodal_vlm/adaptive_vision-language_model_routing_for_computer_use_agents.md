---
title: >-
  [Paper Note] AVR: Adaptive VLM Routing for Computer Use Agents
description: >-
  [CVPR 2026][Multimodal VLM][Computer Use Agent] This paper proposes AVR, an adaptive routing framework for Computer Use Agents that combines a lightweight multimodal embedding model for action difficulty assessment, small-model logprob confidence probing, and warm agent memory injection, enabling a three-tier routing strategy (simple → small model; difficult → large model; high-risk → large model + guardrail). AVR reduces inference cost by 78% with only a 2 pp accuracy loss.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Computer Use Agent
  - VLM Routing
  - Adaptive Inference
  - Cost Optimization
  - GUI Grounding
date: 2026-05-08
content_hash: 5d6b8de55e49ce35
---

# AVR: Adaptive VLM Routing for Computer Use Agents

**Conference**: CVPR 2026
**arXiv**: [2603.12823](https://arxiv.org/abs/2603.12823)
**Code**: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
**Area**: Multimodal VLM
**Keywords**: Computer Use Agent, VLM Routing, Adaptive Inference, Cost Optimization, GUI Grounding

## TL;DR

This paper proposes AVR, an adaptive routing framework for Computer Use Agents that combines a lightweight multimodal embedding model for action difficulty assessment, small-model logprob confidence probing, and warm agent memory injection, enabling a three-tier routing strategy (simple → small model; difficult → large model; high-risk → large model + guardrail). AVR reduces inference cost by 78% with only a 2 pp accuracy loss.

## Background & Motivation

**The Rise and Bottleneck of CUAs**: Computer Use Agents (CUAs) leverage VLMs to directly operate GUIs for complex tasks (clicking, typing, scrolling, etc.), making them one of the most active research directions in the Agent community. However, existing CUA systems (e.g., OpenAI CUA, Claude Computer Use) rely on a single large VLM for all operations, requiring GPT-4o- or Claude-level model calls at every step — a 50-step task can cost several dollars in API fees alone.

**High Variance in Action Difficulty**: The complexity of actions in CUA tasks is highly non-uniform. Simple actions (e.g., clicking a prominent button, typing known content into a text field) constitute the majority of operations and require no top-tier VLM reasoning, whereas complex actions (e.g., locating a small icon in a dense UI, performing multi-step contextual reasoning) genuinely demand strong models. This "overkill" phenomenon results in substantial resource waste.

**Limitations of Existing Routing Approaches**: Text-domain LLM routing (e.g., RouterBench, RouteLLM) has been explored, but direct transfer to CUA poses several challenges: (a) GUI operations involve both visual complexity (screen layout, element density) and semantic complexity (instruction ambiguity), making text-only features insufficient; (b) CUA actions are sequential, and accumulated context (prior actions) significantly affects the difficulty of the current action; (c) incorrect routing decisions carry higher costs in CUA — a single misclick can cause an entire task to fail.

**Model Size ≠ Grounding Accuracy**: A key finding of this work is that model size does not monotonically correlate with accuracy on GUI grounding tasks. On ScreenSpot-Pro, GPT-4o achieves only 0.8% accuracy (likely due to its visual encoder's poor understanding of GUI elements), whereas the open-source 7B model OS-Atlas achieves 18.9% and Qwen2.5-VL-72B achieves 43.6%. This implies that routing cannot simply be stratified by model size; the actual GUI-domain capability of each model must be considered.

**Asymmetric Effect of Memory**: The authors observe that injecting prior successful action experiences (warm agent memory) into models yields far greater improvements for small models than for large ones. Small models improve from a task success rate of 0.83 to 0.96, while large models show only marginal gains. This provides a critical insight for routing strategy: memory injection can substantially raise the effective capability ceiling of small models.

## Method

### Overall Architecture

The AVR framework consists of three core components forming a cascaded routing pipeline:

```
Input (screenshot + instruction) → [Difficulty Assessor] → Simple? → Small VLM executes
                                                               ↓ Not simple
                                                     [Confidence Prober] → Small VLM confident? → Small VLM executes
                                                                               ↓ Not confident
                                                                         [Memory Injection] → Confident after injection? → Small VLM executes
                                                                                                  ↓ Still not confident
                                                                                            Large VLM executes (+guardrail if high-risk)
```

Three-tier routing strategy:
- **Layer 1**: A lightweight difficulty assessor rapidly filters obviously simple actions (~40–50% of all operations).
- **Layer 2**: The small VLM actually attempts to generate an action; confidence is assessed via logprob analysis, and high-confidence outputs are executed directly.
- **Layer 3**: After memory injection, confidence is re-evaluated; if still uncertain, the action is escalated to the large model. High-risk operations (e.g., deletion, payment) additionally trigger guardrail verification.

### Key Design 1: Multimodal Difficulty Assessor

A 120M-parameter multimodal embedding model (similar to the SigLIP-small architecture) takes the current screenshot and instruction text as input and outputs a scalar difficulty score $d \in [0, 1]$.

The difficulty score integrates two dimensions:
- **Visual complexity**: UI element density, visual salience of the target element, screen layout complexity.
- **Semantic complexity**: instruction ambiguity, required reasoning steps, and whether contextual understanding is needed.

Training data is constructed by collecting success/failure records from models of varying sizes (7B/14B/72B) on ScreenSpot-Pro and OpenClaw: actions that small models can complete successfully are labeled "easy," while actions that only large models can complete are labeled "hard."

The assessor incurs negligible inference overhead (~2 ms/frame), far less than any VLM inference call, and thus does not become a pipeline bottleneck.

### Key Design 2: Logprob Confidence Probing

For actions not directly routed by the difficulty assessor, the small VLM (e.g., Qwen2.5-VL-7B) performs an actual inference pass to generate an action sequence $a = (a_1, a_2, \ldots, a_n)$, and the log probability of each token is collected.

The confidence metric is defined as:

$$C(a) = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log p(a_i \mid a_{<i}, s)\right)$$

where $s$ denotes the current state (screenshot + history). Intuitively, this is the geometric mean probability of the model's output tokens.

Key threshold design:
- $C(a) > \theta_{\text{high}}$: high confidence — execute the small model's action directly.
- $\theta_{\text{low}} < C(a) \leq \theta_{\text{high}}$: medium confidence — proceed to the memory injection stage.
- $C(a) \leq \theta_{\text{low}}$: low confidence — escalate directly to the large model.

$\theta_{\text{high}} = 0.85$ and $\theta_{\text{low}} = 0.60$ are determined via grid search on the validation set using F1 score.

### Key Design 3: Warm Agent Memory Injection

The memory module maintains a dynamically updated experience store $\mathcal{M} = \{(s_j, a_j, r_j)\}$, recording (state, action, reward) triples from previously successful executions.

The memory injection procedure:
1. **Retrieval**: Using embeddings of the current screenshot and instruction, retrieve the top-$k$ ($k=3$) most similar successful experiences from $\mathcal{M}$.
2. **Formatting**: The retrieved experiences are formatted as few-shot examples and injected into the small VLM's prompt.
3. **Re-inference**: The small VLM regenerates the action under the memory-augmented prompt, and confidence is re-evaluated.

The effect of memory injection is highly asymmetric:
- Small model (7B): task success rate improves from 0.83 → 0.96 (+13 pp).
- Large model (72B): task success rate improves from 0.94 → 0.95 (+1 pp).

This validates a core hypothesis: the primary bottleneck of small models is not insufficient capacity, but a lack of prior experience with GUI operations. Injecting a small number of examples suffices to substantially close this gap.

### Loss & Training

- The difficulty assessor is fine-tuned on 5K manually annotated difficulty labels using BCE loss on a 120M embedding model.
- The routing policy itself requires no training; thresholds are determined by validation set search.
- The memory store is updated online using a FIFO strategy with a fixed capacity of 1,000 entries.

## Key Experimental Results

### Main Results: ScreenSpot-Pro GUI Grounding Accuracy

| Model / Method | Accuracy (%) | Inference Cost (relative) | Latency (ms/action) |
|---|---|---|---|
| GPT-4o | 0.8 | 1.00× | ~3000 |
| OS-Atlas-7B | 18.9 | 0.05× | ~200 |
| Qwen2.5-VL-14B | 28.3 | 0.12× | ~400 |
| Qwen2.5-VL-72B | 43.6 | 0.80× | ~2000 |
| **AVR (7B+72B)** | **42.7** | **0.22×** | **~450** |

AVR achieves accuracy close to the 72B single-model baseline (42.7% vs. 43.6%) at only 22% of the cost, with substantially lower average latency.

### OpenClaw Task Success Rate

| Method | Success Rate (%) | Avg. Steps | Avg. Cost ($) |
|---|---|---|---|
| Qwen2.5-VL-7B | 68.2 | 32.1 | 0.12 |
| Qwen2.5-VL-72B | 87.5 | 28.4 | 2.85 |
| Fixed Routing (50/50) | 79.1 | 30.2 | 1.48 |
| RouteLLM (text) | 76.8 | 31.0 | 0.89 |
| **AVR** | **85.7** | **29.1** | **0.63** |

On OpenClaw, AVR trails the pure 72B model by only 1.8 pp in success rate while reducing cost by 78% ($0.63 vs. $2.85).

### Ablation Study

| Difficulty Assessor | Confidence Probing | Memory Injection | Success Rate (%) | Cost ($) |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 68.2 | 0.12 |
| ✓ | ✗ | ✗ | 78.3 | 0.95 |
| ✓ | ✓ | ✗ | 82.1 | 0.71 |
| ✓ | ✓ | ✓ | **85.7** | **0.63** |
| ✗ | ✓ | ✓ | 83.4 | 0.82 |

All three components contribute meaningfully. Memory injection not only improves success rate (+3.6 pp) but also further reduces cost (from $0.71 to $0.63) by decreasing the number of escalations to the large model.

### Key Findings

- **GPT-4o performs extremely poorly on GUI grounding** (0.8%), demonstrating that closed-source models are not necessarily superior to open-source alternatives in specialized domains and that model selection must be grounded in task-specific evaluation rather than brand recognition.
- **The asymmetric effect of memory injection** is a central finding: small models benefit substantially (+13 pp) while large models are almost unaffected (+1 pp), providing a theoretical basis for the economic viability of the routing strategy.
- **The action difficulty distribution is long-tailed**: approximately 45% of actions are "simple" and can be handled directly by small models; 30% are of medium difficulty and manageable by small models after memory injection; only 25% genuinely require a large model.
- **Guardrail necessity for high-risk actions**: Even large models exhibit an error rate of ~3% on irreversible operations (file deletion, message sending, payment). Additional guardrail verification reduces this to 0.5%.

## Highlights & Insights

- **Addresses a real CUA deployment pain point**: Cost is the primary barrier to large-scale CUA deployment. AVR reduces inference cost by nearly 5×, making commercial CUA deployment substantially more viable.
- **Dual utility of memory injection**: The memory store simultaneously serves routing decisions (reducing escalation rates) and enhances the small model's own performance — a single mechanism achieving two goals.
- **Elegant three-tier cascade design**: Each layer has a clear responsibility and exit condition, avoiding complex joint optimization and making the system straightforward to implement and tune.
- **Challenges the "larger is always better" assumption**: GPT-4o's poor performance on GUI grounding is a compelling counter-example to the industry's uncritical pursuit of larger models.

## Limitations & Future Work

- The difficulty assessor requires separate training for specific GUI domains (desktop, mobile, web); cross-domain generalization remains unvalidated.
- Confidence thresholds are statically defined; ideally, they should be dynamically adjusted based on task type and current progress.
- The memory store uses a simple FIFO policy without considering experience diversity or representativeness, potentially causing certain action types to be evicted disproportionately.
- Routing effectiveness is validated only on the Qwen2.5-VL series; applicability to other VLM families (e.g., InternVL, LLaVA-OneVision) is unknown.
- The three-tier routing introduces additional engineering complexity and potential failure points; robustness in production deployment requires further validation.
- The paper lacks discussion of error propagation and recovery mechanisms in multi-turn tasks.

## Related Work & Insights

- **LLM Routing**: RouteLLM, RouterBench, and related works have explored model routing for text tasks, but exclusively in text-only settings. AVR extends this paradigm to GUI operations requiring visual understanding.
- **CUA Systems**: OpenAI CUA, Claude Computer Use, and OS-Atlas define the foundational paradigm for CUAs; AVR proposes system-level efficiency optimizations orthogonal to these.
- **GUI Understanding**: Benchmarks such as ScreenSpot and OmniACT reveal performance disparities across VLMs on GUI understanding, providing empirical grounding for AVR's routing strategy.
- **MoE and Hybrid Inference**: AVR shares conceptual similarities with Mixture-of-Experts routing, but operates at the model level rather than the layer level — coarser in granularity but more practically deployable.
- **Broader Implications**: The paradigm of "assess difficulty first, then allocate resources accordingly" generalizes beyond CUAs to other agent settings — code agents, data analysis agents, and similar domains share the same characteristic that simple actions constitute the majority of operations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Multi-tier routing is not a fundamentally new concept, but its application to the CUA setting and the memory injection design represent genuine contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on two mainstream benchmarks with complete ablations, though comparisons across more VLM combinations are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated and system design is intuitively explained, though training details for the difficulty assessor are insufficiently described.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses the cost bottleneck of CUA deployment; a 78% cost reduction is highly practical and carries significant implications for real-world agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)
- [\[AAAI 2026\] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](../../AAAI2026/multimodal_vlm/are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)

</div>

<!-- RELATED:END -->
