---
title: >-
  [Paper Note] Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints
description: >-
  [NeurIPS 2025][Recommender Systems][LLM planning] This paper proposes MAoP (Multiple Aspects of Planning), a framework that endows LLMs with "wide-horizon thinking" by having a strategist perform multi-aspect pre-planning and routing into a coherent blueprint, enabling the planner to conduct in-depth per-aspect analysis in parallel. Coupled with the Travel-Sim causal simulation benchmark, MAoP substantially outperforms CoT and decomposition-based methods on travel planning tasks; a distilled 3B model achieves a PER of 66.9%.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - LLM planning
  - travel planning
  - wide-horizon thinking
  - simulation evaluation
  - multifaceted constraints
date: 2026-05-08
content_hash: 99f4cb0836d16be5
---

# Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2506.12421](https://arxiv.org/abs/2506.12421)
**Code**: None
**Area**: Recommender Systems
**Keywords**: LLM planning, travel planning, wide-horizon thinking, simulation evaluation, multifaceted constraints

## TL;DR
This paper proposes MAoP (Multiple Aspects of Planning), a framework that endows LLMs with "wide-horizon thinking" by having a strategist perform multi-aspect pre-planning and routing into a coherent blueprint, enabling the planner to conduct in-depth per-aspect analysis in parallel. Coupled with the Travel-Sim causal simulation benchmark, MAoP substantially outperforms CoT and decomposition-based methods on travel planning tasks; a distilled 3B model achieves a PER of 66.9%.

## Background & Motivation
**Background**: LLMs perform well on simple reasoning and planning in controlled environments, but real-world planning tasks such as travel planning involve deep interactions among multidimensional constraints—budget, time, personal preferences, transportation modes, physical condition, and more—which existing methods struggle to handle.

**Long-Horizon vs. Wide-Horizon**: Conventional reasoning emphasizes "long-horizon thinking"—deep inference along a single trajectory (e.g., chain-of-thought logic in mathematical proofs). Real-world planning, however, demands "wide-horizon thinking"—simultaneously integrating multiple heterogeneous information sources and parallel constraints. Task decomposition methods such as Plan-and-Solve remain fundamentally linear and sequential, failing to capture inter-constraint interactions and dependencies.

**Limitations of Prior Work**: Existing benchmarks such as TravelPlanner and ChinaTravel evaluate plan quality solely via static constraint pass rates, ignoring the causal dependencies that arise during travel—for example, excessive fatigue on day one can cascade and compromise the feasibility of subsequent itineraries, an effect that static metrics cannot capture.

**Key Challenge**: Existing methods apply deep, long-chain reasoning strategies to a problem that inherently requires breadth-first integration; both chain-of-thought reasoning and task decomposition exhibit structural limitations in this setting.

**Key Insight**: Preliminary experiments reveal that even a naive aspect-aware decomposition significantly outperforms CoT, yet it suffers from three deficiencies: aspects are treated independently without cross-aspect correlation, the approach relies on carefully hand-crafted prompts, and it scales poorly at inference time.

**Core Idea**: A strategist performs multi-aspect pre-planning and routes the results into a coherent blueprint; a planner then conducts focused, in-depth analysis aspect by aspect following the blueprint, achieving wide-horizon reasoning with inference-time scaling.

## Method

### Overall Architecture
MAoP decomposes planning into two stages: Pre-Planning and Aspect-Aware Planning. In Pre-Planning, a Strategist decomposes the complex request into multiple aspects and routes them into a blueprint. In Aspect-Aware Planning, a Planner conducts multi-turn dialogue following the blueprint, performing deep analysis for each aspect in turn and ultimately synthesizing a complete plan.

### Key Designs
1. **Strategist Pre-Planning — Decomposition Phase**:

    - The Strategist receives long-context inputs (traveler profiles, attraction blogs, transportation information, etc.) along with the user request, decomposes them into multiple aspects (e.g., "transportation arrangements," "budget control," "physical endurance allocation"), and generates concise analytical guidance for each aspect.
    - By sampling the Strategist multiple times in parallel, a large number of aspect–guidance pairs can be obtained, broadening the scope of consideration.

2. **Strategist Pre-Planning — Routing Phase**:

    - Unlike naive wide-horizon approaches that simply enumerate aspects independently, the Strategist employs a routing mechanism to aggregate multiple aspects into fewer, more coherent planning blueprints.
    - Guidance for later aspects is conditioned on earlier aspects, forming an ordered chain of aspect dependencies.
    - Routing transfers the burden of "considering more aspects" from the Planner to the Strategist, enabling inference-time scaling—performance continues to improve as more aspects are considered, rather than saturating.

3. **Planner Aspect-Aware Planning**:

    - The Planner proceeds through the blueprint sequentially; each dialogue turn focuses on a single aspect and performs targeted, in-depth analysis over the long context.
    - After accumulating analysis across turns, the final turn synthesizes all preceding aspect analyses to produce the complete plan.
    - This multi-turn structure limits the complexity the Planner must handle at any one time while maintaining global coherence.

4. **Training Procedure**:

    - The Strategist is trained via Rejection Sampling Fine-Tuning (RFT): $N$ pre-planning samples are drawn for each request, and only trajectories where at least one resulting plan exceeds a quality threshold are retained.
    - The Planner is trained via GRPO reinforcement learning with reward $R_{overall} = 2(R_{PER} - 0.5)$ when the format is correct, or $2(R_{PER} - 0.5) - 1$ when the format is incorrect. The PER score covers five dimensions: experience, interest, arrangement, physical condition, and expenditure.

5. **MAoP Distillation — Single-Step Wide-Horizon Thinking**:

    - High-quality MAoP trajectories are generated using a strong teacher model (R1-Distill 7B Strategist + Gemini 2.5-Pro Planner).
    - Strategist guidance is extracted, and the multi-turn aspect analyses along with the final aggregation are compressed into a single-step output.
    - The distilled 3B model can execute complex wide-horizon planning in a single inference pass.

### Travel-Sim Causal Simulation Evaluation

1. **Event-Driven Sandbox**: A traveler agent (powered by Gemini 2.5-Pro) executes the plan step by step within a sandbox, maintaining state $c_n = \{t, l, s, o, e\}$ (time, location, stamina, expenditure, current event) at each step in a ReAct-style think-then-act loop.
2. **Real-World Information Integration**: Map APIs provide transportation reference data; travel blogs are used to simulate attraction experiences.
3. **Stamina Engine**: Different traveler profiles follow distinct stamina consumption rules (elderly vs. young travelers, families with vs. without infants).
4. **Multi-Granularity Evaluation**: Assessment is conducted at three levels—after each POI, at the end of each day, and at the end of the entire trip—across five dimensions: experience, interest, arrangement, physical condition, and expenditure.

## Key Experimental Results

### Main Results — Baseline Comparison

| Method | CPH | CPL | FEA | PER |
|--------|-----|-----|-----|-----|
| Zero-shot CoT (Qwen-32B) | - | - | 23.3 | 36.2 |
| Plan&Solve (Qwen-32B) | - | - | 25.0 | 39.7 |
| Wide/Artificial (Qwen-32B) | - | - | 31.9 | 44.1 |
| Wide/Artificial (DeepSeek-R1) | - | - | 58.9 | 68.0 |
| RL w/ Long/Artifact (baseline) | low | low | low | low |
| **MAoP (R1-Distill 7B + R1-Distill 7B)** | **72.6** | **76.5** | **60.7** | **81.4** |

### Distilled Model Comparison

| Model | CPH | CPL | PER (agg.) |
|-------|-----|-----|------------|
| Llama 3.2-3B (distilled) | 61.3 | 59.2 | 65.7 |
| **Qwen 2.5-3B (distilled)** | 64.2 | 65.8 | **66.9** |
| R1-Distill 7B (distilled) | 78.2 | 79.2 | **84.2** |

### Key Findings
- MAoP improves over the RL w/ Long/Artifact baseline trained on the same data by 5%–40% across all metrics.
- A stronger Strategist (R1-Distill 7B vs. Qwen-7B) yields better inference-time scaling—performance continues to improve when considering 3 to 8 aspects.
- Distilled R1-Distill 7B surpasses the original MAoP combination, with larger teacher–student capability gaps yielding greater distillation gains.
- Emergent behavior is observed in travel simulation: an elderly couple spontaneously abandons a planned dinner due to fatigue from a long train journey, underscoring the importance of causal dependencies.

## Highlights & Insights
- **Paradigm Shift**: The conceptual shift from long chain-of-thought reasoning to parallel wide-horizon thinking is clearly articulated and applicable to all multi-constraint planning scenarios beyond travel.
- **Elegant Routing Mechanism**: Strategist routing resolves the scalability bottleneck of naive aspect decomposition—directly increasing the number of aspects saturates at 3–5, whereas routing enables continuous scaling to 8 aspects.
- **Evaluation Paradigm Innovation**: Travel-Sim is the first travel planning benchmark to simultaneously incorporate rule-based evaluation, LLM-as-judge assessment, multi-granularity feedback, and causal consistency.
- **Distillation Efficiency**: A 3B model distilled via MAoP approaches the performance of a 32B MAoP combination, demonstrating that wide-horizon thinking patterns can be learned by small models.

## Limitations & Future Work
- **Strategist–Planner Separation Overhead**: Inference requires two models working in tandem, incurring higher latency and cost; distillation simplifies this but sacrifices flexibility.
- **Evaluator Dependency**: Travel-Sim uses Gemini 2.5-Pro as the traveler agent, so evaluation results are subject to that model's capabilities.
- **Domain Scope**: Validation is limited to travel planning; effectiveness on other multi-constraint planning scenarios (e.g., project management, resource scheduling) remains unexplored.
- **Training Data**: Strategist RFT requires a frozen Planner for evaluation, and the RL pipeline cannot directly optimize the Strategist.

## Related Work & Insights
- **vs. TravelPlanner**: TravelPlanner evaluates constraint pass rates via rules; Travel-Sim captures causal dependencies through dynamic simulation, more closely reflecting real-world conditions.
- **vs. DeepSeek-R1**: R1 strengthens long-chain reasoning; this paper demonstrates that wide-horizon thinking is more effective for multi-constraint planning.
- **vs. Plan-and-Solve**: Plan-and-Solve linearly decomposes subtasks; MAoP decomposes aspects in parallel and then routes them into an integrated blueprint, preserving inter-aspect dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐ The wide-horizon thinking concept is novel, and the Strategist routing mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 cities × 16 traveler types = 112 scenarios, with complete distillation and scalability analysis.
- Writing Quality: ⭐⭐⭐⭐ The long-horizon vs. wide-horizon contrast is clearly argued, and the framework diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Contributes significantly to both LLM planning methodology and evaluation paradigm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](../../ACL2026/recommender/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)

</div>

<!-- RELATED:END -->
