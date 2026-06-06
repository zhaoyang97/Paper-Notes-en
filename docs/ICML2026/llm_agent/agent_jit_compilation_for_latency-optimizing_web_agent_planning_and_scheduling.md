---
title: >-
  [Paper Note] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling
description: >-
  [ICML 2026][LLM Agent][computer-use agent] This paper transforms web Computer-Use Agents from a step-by-step "screenshot-LLM call-execution" cycle into a system resembling a JIT compiler. It compiles natural language tas…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "computer-use agent"
  - "JIT compilation"
  - "web automation"
  - "tool protocol"
  - "cost-aware scheduling"
date: 2026-05-08
content_hash: d056d40d34d38fae
---

# Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling

**Conference**: ICML 2026  
**arXiv**: [2605.21470](https://arxiv.org/abs/2605.21470)  
**Code**: No public code  
**Area**: LLM Agent / Web Automation  
**Keywords**: computer-use agent, JIT compilation, web automation, tool protocol, cost-aware scheduling  

## TL;DR
This paper transforms web Computer-Use Agents from a step-by-step "screenshot-LLM call-execution" cycle into a system resembling a JIT compiler. It compiles natural language tasks into verifiable, cacheable, and parallel-schedulable code plans, enabling JIT-Planner to be 10.4× faster than Browser-Use with 28pp higher accuracy, and JIT-Scheduler to be 2.4× faster than OpenAI CUA with 9pp higher accuracy.

## Background & Motivation
**Background**: Computer-use agents attempt to control browsers using natural language to execute web tasks such as ordering food, shopping, emailing, or managing code repositories. Mainstream implementations typically follow a cyclic agent loop: observe the screenshot or DOM, call an LLM to generate the next click/type/scroll action, and observe the next state after execution.

**Limitations of Prior Work**: This cyclic approach faces three prominent issues. First, the toolsets are too atomic; while click/type/scroll are general-purpose, tasks require numerous steps, leading to high error rates. Second, execution is strictly serial, requiring an LLM wait at every step, which results in high latency for long tasks. Third, non-deterministic LLM calls are continuously introduced even after planning, often splitting data processing or loops that could be handled by code into multiple expensive inferences.

**Key Challenge**: Web tasks require both the semantic understanding of LLMs and deterministic operations that are compilable, cacheable, and statically checkable. Traditional agents treat every step as an online decision, causing latency and errors to be amplified by repeated LLM calls.

**Goal**: The authors aim to elevate agent runtime optimization from "selecting the next action" to "generating and optimizing an entire executable plan." The system needs to verify if tool sequences satisfy page state constraints, estimate costs for candidate plans, and select appropriate scheduling strategies for parallelizable tasks.

**Key Insight**: The paper borrows the concept of a JIT compiler: natural language tasks are treated as high-level programs that the system compiles into low-level code plans at runtime. Since multiple candidate plans may be correct but vary significantly in latency, the system performs static verification and cost-based selection similar to compiler optimizations.

**Core Idea**: The system utilizes an invariant-enforcing tool protocol to ensure valid tool combinations, a CFG cost model to select the lowest-cost option among candidate code plans, and Monte Carlo latency estimation to choose between serial, parallel, or hedge execution strategies.

## Method
Agent JIT consists of three online components and an offline caching workflow. The offline workflow synthesizes reusable tools from successful execution traces and learns historical latency distributions of web element interactions. Online, the planner generates code plans, the scheduler selects execution strategies, and the tool protocol constrains plan legality.

### Overall Architecture
The input consists of natural language tasks, tool manifests, cached tools, and historical latency distributions. JIT-Planner parallelly samples multiple code plans, which can include standard tool calls, LLM evaluation calls, and control flows. The system constructs a Control Flow Graph (CFG) to check if the pre/post states of tools are composable and estimates costs based on tool calls, LLM evaluations, and loop depth, ultimately selecting the lowest-cost valid plan. For schedulable tasks, JIT-Scheduler selects the strategy (serial, parallel, or hedge) with the lowest expected latency based on predicted DOM elements and historical latency distributions.

### Key Designs
1. **Invariant-enforcing tool protocol**:
    - **Function**: Upgrades tools from simple "callable functions" to composable building blocks with state contracts.
    - **Mechanism**: In addition to input/output schemas, each tool manifest declares `pre`, `post`, optional `pre_check`/`post_check`, and `execute`. Adjacent tools in a plan are valid only if the post-condition of the previous tool satisfies the pre-condition of the next, enforced by the state flow constraint $post_i\subseteq pre_{i+1}$.
    - **Design Motivation**: The authors found that 45–50% of web automation errors stem from incorrect tool sequences (e.g., calling a detail-page tool before entering the page). Incorporating state invariants into the protocol allows the system to eliminate invalid plans during the compilation phase rather than waiting for browser execution to fail.

2. **Cost-optimizing JIT-Planner**:
    - **Function**: Selects the lowest-latency version among multiple valid code plans.
    - **Mechanism**: Multiple workers sample plans from the LLM in parallel. Invalid plans are iteratively fixed using verification errors until $k$ valid candidates are collected. A CFG is built for each candidate: tool calls add $C_{tool}\gamma^d$ and AI evaluation calls add $C_{eval}\gamma^d$, where $d$ is the loop/nesting depth and $\gamma=10$ is used to penalize placing expensive LLM calls inside loops. The plan with the lowest estimated cost is returned.
    - **Design Motivation**: A single web task can have many equivalent implementations. Results show that average latency can differ by 5.3× between best-cost and worst-cost plans, suggesting that generating a "workable" plan is insufficient; cost optimization is necessary.

3. **Cost-aware JIT-Scheduler**:
    - **Function**: Selects serial, parallel, or hedge execution strategies to utilize available vCPUs for latency reduction.
    - **Mechanism**: The scheduler uses an LLM to predict which web elements will be accessed under various strategies, then performs Monte Carlo sampling from offline-learned element latency distributions. Serial execution sums all interaction times; Parallel execution combines the serial component with the slowest worker; Hedging takes the fastest result from redundant workers plus overhead. The strategy with the lowest average latency is chosen.
    - **Design Motivation**: No single strategy is universally optimal. Parallelism suits independent sub-tasks, hedging is ideal for tasks prone to stalling on UI elements, and serial execution is best for short linear tasks. Data-driven estimation avoids the need for manual scheduling rules.

### Loss & Training
As a systems-oriented paper, there is no specific model training loss. The optimization objective is the latency-accuracy trade-off at the planning and scheduling layers. The JIT-Planner cost model explicitly penalizes tool calls, AI evaluation calls, and nested loops, while the JIT-Scheduler utilizes Monte Carlo estimation based on cached latency distributions. The offline workflow extracts page schemas from traces, maps actions to schema elements, fits latency distributions, and synthesizes reusable code tools.

## Key Experimental Results

### Main Results
| Comparison | Latency | Accuracy | Conclusion |
|------|---------|----------|------|
| Browser-Use | 122.1s | Baseline | Calls LLM at every step; 73% of latency comes from inference |
| Browser-Use +cache | 80.1s | Higher than Browser-Use | Uses cached tools but remains a step-by-step loop; only 1.5× speedup |
| JIT-Planner | 11.7s | +28pp vs Browser-Use | Average 10.4× faster than Browser-Use and 6.8× faster than +cache |
| Worst-cost plan | 61.7s | Same as valid candidate | 5.3× latency difference compared to best-cost, highlighting cost sorting importance |
| OpenAI CUA | 258.7s | 77.8% | Specialized CUA still executes serially |
| Anthropic CUA | 141.7s | 79.0% | Accuracy comparable to JIT-Scheduler but with higher latency |
| JIT-Scheduler (Gemini-2.5-Pro) | 109.9s | 86.4% | 2.4× faster than OpenAI CUA with 9pp higher accuracy |

### Ablation Study
| Configuration / Phenomenon | Metric | Result | Description |
|-------------|------|------|------|
| Protocol on valid-plan rate | GPT-4.1 | 78% → 91% | Tool invariants significantly improve valid plan ratio |
| Protocol on valid-plan rate | Gemini-2.5-Pro | 79% → 96% | Long-task Pass@k also significantly improved |
| Protocol on valid-plan rate | Gemini-2.5-Flash | 74% → 85% | Small/fast models also benefit |
| Long GitLab task Pass@3 | Gemini-2.5-Pro | 9% → 100% | Protocol enables finding valid plans for 19-step tasks with few candidates |
| Tool-ordering failures | w/o protocol vs w/ protocol | 59% → 25% | Errors related to state sequence violations are notably reduced |
| CUA +cache vs JIT-Planner | REAL (3 Apps) | JIT 1.5–2.4× Gain | Faster even with identical tools, isolating planning optimization contribution |

### Key Findings
- Protocols are not just documentation; they fundamentally improve planner search efficiency. On long GitLab tasks, Gemini-2.5-Pro's Pass@3 increased from 9% to 100%. Furthermore, parallel hedging reached 100% Pass@t within 8 seconds, whereas it stalled at 22% without the protocol.
- The cost model accelerates execution primarily by eliminating unnecessary LLM inference and `ai_eval` calls within loops. While Browser-Use spends 73% of its latency on LLM calls, JIT-Planner compiles tasks into code, moving inference to the planning stage or removing it entirely.
- Task complexity has a minor impact on speedup. JIT-Planner achieved 10.8×, 8.7×, and 11.8× speedups on C-Low, C-Medium, and C-High tasks respectively, indicating that gains stem from the execution paradigm shift rather than task-specific specialization.
- Scheduling strategies must be adaptive. Under GPT-4.1, Serial/Parallel/Hedge latencies were 157.3/166.2/130.3s, while for Gemini-2.5-Pro they were 129.6/148.5/98.4s. However, the lowest-latency strategy does not always yield the highest accuracy; JIT-Scheduler identifies a more stable Pareto point between the two.

## Highlights & Insights
- The strongest engineering insight is treating web agents as a compilation problem rather than a pure policy problem. Once reusable tools are abstracted, web tasks behave more like program synthesis and optimization where per-step "thinking" is no longer required.
- The invariant protocol pushes MCP-style type checking toward state-flow checking, which is crucial for agent tool ecosystems. Merely checking argument types is insufficient to guarantee that a tool can be invoked on the current page.
- JIT-Planner's cost model is elegantly simple but effective: penalizing LLM evaluations and nested loops is sufficient to prioritize significantly faster plans. This demonstrates that agent latency optimization often requires better runtime representation rather than larger models.
- The scheduler uses latency distributions instead of fixed rules, reflecting the reality of web environments. Web interaction times often exhibit long tails, making hedging more rational than simple parallelism in these scenarios.

## Limitations & Future Work
- The system relies on offline traces and cached tools. For entirely unfamiliar websites, rapidly changing frontends, or scenarios lacking successful trajectories, tool synthesis and latency distributions must be rebuilt.
- Task coverage includes 5 applications and 37 tasks, which is superior to toy demos but remains limited compared to the open web. Realistic factors like login, payment, CAPTCHAs, and personalized feeds were not fully explored.
- Invariant manifests require tool authors or automated processes to accurately define pre/post conditions; loose invariants lead to errors, while overly tight ones may invalidate viable plans.
- The cost model primarily optimizes for latency, with less consideration for monetary cost, risk, permissions, security auditing, and user explainability. Future agent compilers may require multi-objective optimization.

## Related Work & Insights
- **vs Browser-Use**: Browser-Use is a typical observe-act loop where every step depends on an LLM; Agent JIT compiles tasks into code plans to minimize runtime inference.
- **vs CUA**: OpenAI/Anthropic CUA utilize fixed action spaces and serial execution; the JIT system introduces cached tools, plan verification, and adaptive scheduling for superior latency and accuracy.
- **vs code-action agents**: While existing work utilizes code actions, they do not systematically study latency differences between code plans; this paper treats code plans as optimizable objects.
- **vs MCP/tool protocols**: MCP focuses on tool interfaces and types; this paper further mandates state pre/post invariants, enabling static verification of tool chains.

## Rating
- Novelty: ⭐⭐⭐⭐ Abstracting agent execution as JIT compilation and scheduling optimization is highly inspiring, and the components borrow effectively from systems/compiler theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple applications, models, and comprehensive ablations; open-web generalization requires further large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Clear system architecture, pseudo-code, and analysis; the appendix provides sufficient detail.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for improving the latency and reliability of practical web agents; highlights that tool protocols should incorporate state invariants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](../../ACL2026/llm_agent/why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)
- [\[ACL 2026\] TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](../../ACL2026/llm_agent/theraagent_self-improving_therapeutic_agent_for_precise_and_comprehensive_treatm.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)

</div>

<!-- RELATED:END -->
