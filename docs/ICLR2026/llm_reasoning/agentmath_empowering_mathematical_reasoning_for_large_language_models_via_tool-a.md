---
title: >-
  [Paper Note] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
description: >-
  [ICLR 2026][LLM Reasoning][Mathematical Reasoning] AgentMath proposes a tool-augmented agent framework that seamlessly integrates LLM reasoning with the computational precision of a code interpreter through automated data synthesis, multi-turn interactive reinforcement learning, and an efficient asynchronous training system. At the 30B-A3B scale, it achieves state-of-the-art performance on AIME24/25 and HMMT25 (90.6%/86.4%/73.8%), surpassing o3-mini and Claude-Opus-4.0-Thinking.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Mathematical Reasoning
  - Tool Augmentation
  - Reinforcement Learning
  - Code Interpreter
  - Agent Framework
date: 2026-05-08
content_hash: d7d07c043fb7b01f
---

# AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent

**Conference**: ICLR 2026
**arXiv**: [2512.20745](https://arxiv.org/abs/2512.20745)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Mathematical Reasoning, Tool Augmentation, Reinforcement Learning, Code Interpreter, Agent Framework

## TL;DR
AgentMath proposes a tool-augmented agent framework that seamlessly integrates LLM reasoning with the computational precision of a code interpreter through automated data synthesis, multi-turn interactive reinforcement learning, and an efficient asynchronous training system. At the 30B-A3B scale, it achieves state-of-the-art performance on AIME24/25 and HMMT25 (90.6%/86.4%/73.8%), surpassing o3-mini and Claude-Opus-4.0-Thinking.

## Background & Motivation
Large reasoning models (LRMs) such as o3 and DeepSeek-R1 have made remarkable progress in long chain-of-thought reasoning, yet they still suffer from low computational efficiency and insufficient accuracy when handling problems that require precise mathematical operations — the inherent limitations of pure-text reasoning lead to frequent arithmetic errors and redundant corrections. Existing tool-augmented approaches face three major challenges: (1) high-quality tool-use data is extremely scarce, and manual annotation is costly and non-scalable; (2) the potential of agentic reinforcement learning for optimizing tool-use policies remains largely unexplored; (3) competition-level mathematics problems involve ultra-long reasoning chains (96k tokens, 96 tool calls), which conventional synchronous batch RL training cannot handle. The core idea of this paper is to build an end-to-end agent framework that addresses data scarcity through automated data synthesis, learns optimal tool-use policies through Agentic RL, and resolves training efficiency bottlenecks through an asynchronous training architecture.

## Method

### Overall Architecture
AgentMath models tool-augmented mathematical reasoning as a Markov Decision Process (MDP), in which the LLM policy generates alternating reasoning segments and executable code blocks that interact with a sandboxed environment. The system adopts a structured token protocol: `<think>` marks natural-language reasoning, `<code>` marks executable code, and `<interpreter>` encapsulates execution feedback. The overall pipeline consists of two stages: (1) SFT on synthesized tool-augmented trajectories to establish initial tool-use capability; and (2) large-scale RL to drive exploration toward optimal tool-use policies.

### Key Designs

1. **Tool-Driven Data Synthesis**: A three-stage automated synthesis pipeline. **Stage 1**: Long chain-of-thought data in pure-text form is aggregated from public sources such as AM-Thinking and Open-Thoughts; N-gram filtering removes overlap with evaluation sets, yielding 346k high-quality samples. DeepSeek-V3 is then used as a teacher model to replace computationally intensive steps with executable code blocks, while retaining simple calculations in text form to prevent excessive tool dependency. **Stage 2**: Multi-dimensional quality refinement — format consistency correction, code executability verification (sandbox execution), environment feedback alignment (Qwen3-32B judges consistency and replaces simulated outputs with actual execution results), and tool-use rationality assessment (unnecessary code is excluded via AST depth and line-count constraints). **Stage 3**: Self-correction capability injection — failure trajectories are sampled and the teacher model generates correction trajectories following a "diagnose error → fix code → re-execute → continue reasoning" pattern. The final output is a tool-augmented training set of 316k samples, averaging 8.3 tool calls and 16.9k tokens per sample.

2. **Agentic RL**: Built on the GRPO optimization algorithm with three system-level innovations. **(a) Agent trajectories with interleaved code execution**: During rollout, hybrid trajectories are constructed via a "generate–pause–execute–resume" loop, with a maximum of $T$ tool calls. **(b) Selective loss masking**: Advantage signals are applied only to tokens in `<think>` and `<code>` segments; tokens in `<interpreter>` segments (environment feedback) are masked during optimization, ensuring gradient updates derive solely from the model's own decisions. **(c) Adaptive batch construction**: Problems for which all rollouts are either all correct or all incorrect (providing limited learning signal) are filtered out, and back-filling maintains a consistent batch size.

3. **Composite Reward Design**: The reward function integrates answer correctness and tool-use efficiency: $R_{total} = R_{acc} + \mathbb{I}(R_{acc}=1) \cdot R_{tool}$. Here $R_{acc}$ is a binary signal based on mathematical equivalence, and $R_{tool} = \min(R_{max}, \alpha + \beta \cdot N_{code})$ incentivizes efficient tool utilization when the answer is correct.

4. **Scalable Agent RL Infrastructure**: Three technical innovations address the training bottlenecks imposed by ultra-long sequences and high-frequency tool interactions. **(a) Distributed code execution sandbox cluster**: CPU-intensive code execution is offloaded from the training loop, reducing tool-call latency from 175s to 1.2s. **(b) Request-level asynchronous rollout scheduling**: Each trajectory is treated as an independent long-running request; the inference engine and the agent communicate asynchronously, so the engine immediately processes other ready requests while a request is paused waiting for a tool call, eliminating head-of-line blocking. **(c) Agent partial rollout**: Long trajectories are decomposed into budget-constrained segments ($\tau = \tau^{(1)} \oplus \tau^{(2)} \oplus \ldots$), each bounded by a maximum generation length $L_{seg}$ and a maximum tool-call count $T_{seg}$, preventing any single trajectory from monopolizing resources and achieving a 2.2–2.5× speedup. **(d) Prefix-aware weighted load balancing**: Dynamic weights $w_j = \lfloor L_j / L_{base} \rfloor + w_{base}$ are assigned based on prefix length, combined with LRU sticky sessions to maximize KV-cache reuse. The overall system achieves a 4–5× training speedup.

### Loss & Training
- **SFT stage**: Autoregressive loss with selective feedback masking, $\mathcal{L}_{SFT-masked} = -\sum_t \sum_k (1 - \mathbb{I}(z_{t,k})) \log \pi_\theta(z_{t,k} | \cdot)$, masking `<interpreter>` segment tokens.
- **RL stage**: A multi-stage adaptive policy that automatically scales when the truncation rate exceeds 10%: context length expands from 48k → 72k → 96k, the tool-call limit from 48 → 72 → 96, and the number of partial rollouts from 2 → 3 → 4.
- Llama-Factory is used for SFT (6 epochs, learning rate 6e-5); verl 0.5.0 is used for RL (learning rate 1e-6, batch size 64, 8 rollouts per problem).

## Key Experimental Results

### Main Results

| Dataset | Metric | AgentMath-8B | AgentMath-30B-A3B | AgentMath-235B-A22B-SFT | Prev. SOTA (same scale) | Gain |
|--------|------|------|------|------|----------|------|
| AIME24 | avg@32 | 89.8% | 90.6% | 93.4% | 86.0% (DS-0528-Qwen3-8B) | +3.8% |
| AIME25 | avg@32 | 84.7% | 86.4% | 90.8% | 76.3% (DS-0528-Qwen3-8B) | +8.4% |
| HMMT25 | avg@32 | 71.3% | 73.8% | 81.7% | 61.5% (DS-0528-Qwen3-8B) | +9.8% |

AgentMath-30B-A3B (with only 3B active parameters) surpasses OpenAI-o3-mini (87.3%/86.3%) and Claude-Opus-4.0-Thinking (83.0%/72.0%) on AIME24/25, approaching DeepSeek-R1-671B (91.4%/87.5%).

### Ablation Study

| Configuration | AIME24 | AIME25 | Notes |
|------|---------|------|------|
| Unrefined synthesized data | 35.3% | 25.7% | Format inconsistency and non-executable code degrade performance |
| + Format consistency correction | 47.4% | 40.1% | +12.1%/+14.4% |
| + Code executability verification | 52.8% | 44.8% | +5.4%/+4.7% |
| + Environment feedback alignment | 56.3% | 48.3% | +3.5%/+3.5% |
| + Self-correction capability injection | 58.6% | 50.8% | +2.3%/+2.5% |
| + SFT selective masking | 60.5% | 53.3% | Final SFT performance |
| Text-Based-SFT vs. AgentMath-SFT | 57.1% vs. 60.5% | 49.2% vs. 53.3% | Advantage of tool-augmented data |
| Text-Based-RL vs. AgentMath-RL | 68.7% vs. 76.2% | 57.5% vs. 67.5% | 4× efficiency gain at RL stage |

### Training Efficiency

| Method | Time per Step | Speedup |
|------|---------|------|
| Static synchronous batch rollout | 3600–4000s | — |
| + Request-level async scheduling | 2100–2500s | 1.5–1.8× |
| + Agent partial rollout | 1100–1300s | 3.0–3.3× |
| + Prefix-aware load balancing | 750–900s | 4.0–5.0× |

### Key Findings
- The tool-augmented model reaches 76.2% (AIME24) in only ~400 RL steps, whereas the pure-text model requires ~1,600 steps to reach 68.7% — a 4× efficiency improvement.
- Emergent code self-correction capability arises during multi-stage RL training.
- Reasoning sequence length is reduced by approximately 4k tokens (~14%), as tool-generated code replaces lengthy manual calculations.
- Scaling data from 2k to 300k samples improves AIME24 performance from 27.2% to 78.4%, demonstrating favorable scaling laws.

## Highlights & Insights
- **Systematic resolution of three bottlenecks**: Data scarcity (automated synthesis pipeline), policy optimization (Agentic RL), and training efficiency (asynchronous infrastructure) form a complete technical loop.
- **Emergent code self-correction**: During RL training, the model autonomously learns to diagnose and repair code errors — an emergent behavior that was not explicitly trained.
- **Remarkable efficiency of MoE models**: The 30B-A3B model, with only 3B active parameters, approaches the performance of a 671B-parameter model, demonstrating that tool-augmented strategies can substantially compensate for parameter count deficits.
- **Elegant design of partial rollout**: Decomposing ultra-long trajectories into manageable segments resolves long-tail latency without degrading accuracy (accuracy ~70% remains consistent across different $N$ settings).

## Limitations & Future Work
- Due to computational constraints, the 235B-scale model undergoes only SFT without RL training, leaving potential gains unexplored.
- The study focuses exclusively on mathematical competition benchmarks and does not validate generalization to broader domains such as scientific reasoning or engineering computation.
- The tool-use reward component of the composite reward function is relatively simple and may not precisely guide optimal tool-invocation timing.
- The code interpreter is currently limited to Python/SymPy; integration of other computational tools (e.g., Mathematica, SageMath) remains unexplored.

## Related Work & Insights
- **Comparison with ToRL/ReTool**: These methods also explore RL combined with tool use, but fall short of AgentMath in data quality and training efficiency, with limited performance gains.
- **Comparison with CoRT**: CoRT relies on high-quality manual annotation and is not scalable; AgentMath's automated synthesis pipeline directly addresses this limitation.
- **Engineering insights**: The design principles of the asynchronous training system (request-level scheduling + partial rollout + prefix-aware load balancing) are highly generalizable and transferable to other agent RL scenarios.
- **On agent system design**: This work demonstrates that simple outcome-based rewards such as GRPO are sufficiently effective in agent settings, without requiring complex process rewards.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](../../NeurIPS2025/llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)

<!-- RELATED:END -->
