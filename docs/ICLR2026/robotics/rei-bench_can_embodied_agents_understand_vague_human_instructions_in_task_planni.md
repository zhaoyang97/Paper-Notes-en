---
title: >-
  [Paper Note] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
description: >-
  [ICLR 2026][Robotics][Referring Expressions] This work presents the first systematic study on how referring expressions (REs) in vague human instructions affect LLM-based robot task planning. REI-Bench is introduced to model 9 levels of coreference ambiguity (3 RE difficulty levels × 3 context types). Implicit REs are found to reduce the success rate of existing planners by up to 36.9%. The proposed Task-Oriented Context Cognition (TOCC) method decouples task understanding from planning decision-making, achieving an average improvement of 6.5% in success rate.
tags:
  - ICLR 2026
  - Robotics
  - Referring Expressions
  - Vague Instructions
  - LLM Planning
  - Coreference Resolution
  - Robustness
date: 2026-05-08
content_hash: ef625132fa93e333
---

# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?

**Conference**: ICLR 2026
**arXiv**: [2505.10872](https://arxiv.org/abs/2505.10872)
**Code**: [Project Page](https://jcx0110.github.io/rei-bench-project)
**Area**: Embodied AI / Task Planning
**Keywords**: Referring Expressions, Vague Instructions, LLM Planning, Coreference Resolution, Robustness

## TL;DR

This work presents the first systematic study on how referring expressions (REs) in vague human instructions affect LLM-based robot task planning. REI-Bench is introduced to model 9 levels of coreference ambiguity (3 RE difficulty levels × 3 context types). Implicit REs are found to reduce the success rate of existing planners by up to 36.9%. The proposed Task-Oriented Context Cognition (TOCC) method decouples task understanding from planning decision-making, achieving an average improvement of 6.5% in success rate.

## Background & Motivation

**Background**: LLM-driven robot task planning (e.g., SayCan, ProgPrompt, DAG-Plan) has achieved notable progress, yet all such approaches rest on an idealized assumption that user instructions are clear, complete, and unambiguous. In practice, however, human language is inherently vague.

**Limitations of Prior Work**: Real users—especially the elderly, children, and individuals with Alzheimer's disease—frequently issue instructions containing implicit referring expressions, such as using "it" instead of "pot" or "that heavy thing" instead of "frying pan." Linguistic studies indicate that approximately 20% of expressions in news text are descriptive (implicit REs), with the proportion being even higher in everyday conversation. These user groups are precisely those who most need robotic assistance.

**Research Gap**: (1) No benchmark systematically evaluates the impact of vague instructions on robot planning; (2) existing ambiguity datasets (e.g., AmbiK, CLARA) do not systematically model the position, frequency, and form of REs; (3) it remains unclear whether LLMs can fully leverage their inherent language understanding capabilities in planning scenarios.

**Theoretical Foundation**: Clark's (1975) bridging inference theory explains how humans resolve implicit REs: upon hearing "that heavy thing," a listener retrieves multiple candidates from contextual memory (pot, ingredient, sink) and selects the best match. Levinson further distinguishes between referring expressions (REs) and deictic expressions (DEs) as two distinct types of ambiguity.

**Motivating Observation**: The authors observe that LLMs can correctly resolve implicit REs when prompted in isolation (e.g., via reflective prompting), yet this capability fails to manifest during planning—LLMs over-focus on plan generation and neglect language understanding. This challenges the common assumption that embedding an LLM is sufficient to guarantee a robot's comprehension of human language.

**Practical Impact**: Failures caused by implicit REs primarily manifest as *object omission*—the planner fails to correctly identify the target object in the instruction and therefore generates an incorrect action sequence. For example, "the heated one" is misidentified as "plate" rather than "potato."

## Method

### Overall Architecture

The core idea of REI-Bench is to systematically model coreference ambiguity in real human–robot interaction by combining different RE difficulty levels with different context types, yielding a benchmark covering 9 ambiguity grades. The overall framework comprises three components: (1) formal modeling of REs and dialogue context; (2) an automated pipeline for constructing the REI dataset; and (3) the TOCC method for mitigating ambiguity.

The data construction pipeline is seeded from ALFRED instructions: GPT-4o-mini is used to expand contextual dialogue → three context variants are generated → explicit REs are replaced with implicit REs → yielding 2,700 samples across 9 ambiguity grades.

### Key Design 1: Three-Level RE Difficulty Modeling

- **Function**: Categorizes REs in instructions into three difficulty levels—Explicit, Mixed, and Implicit—systematically simulating a gradient from clear to ambiguous.
- **Mechanism**: Explicit REs include proper nouns ("apple"), definite noun phrases ("the apple"), and indefinite noun phrases ("an apple"), all directly interpretable. Implicit REs include pronouns ("it"/"them") and attribute-based descriptions ("sweet fruit"), which have multiple potential referents and require contextual inference. The three levels are defined as:
  - *Explicit REs*: All explicit expressions from the original dataset are retained.
  - *Mixed REs*: Explicit REs in the instruction are replaced with implicit REs, while explicit REs in the contextual memory remain unchanged.
  - *Implicit REs*: All explicit REs are replaced with implicit REs; only the first explicit RE in the context is retained.
- **Design Motivation**: Humans do not always express themselves with full clarity; the degree of vagueness varies with individual habits and cognitive capacity. The tiered design enables quantitative analysis of how different levels of ambiguity affect planning performance. Replacement rules are grounded in coreference patterns from the OntoNotes corpus to ensure that implicit REs conform to natural language usage.

### Key Design 2: Three-Level Context Memory Modeling

- **Function**: Designs three dialogue context types—Standard, Noised, and Short—to simulate varying information quality in real human–robot interaction.
- **Mechanism**:
  - *Standard Context*: Provides complete task-relevant contextual information.
  - *Noised Context*: Introduces "ambiguous name" noise—person names or brand names similar to scene object names (e.g., "Rose" → "Mrs. Rose")—appearing repeatedly in the dialogue to create interference.
  - *Short Context*: Builds on the noised context by randomly removing some noun phrases containing task-relevant explicit REs, further increasing inferential difficulty.
- **Design Motivation**: Linguists argue that the association between words and objects is constructed within specific contexts (Levinson, 1983). Everyday misleading cues arise from polysemy (e.g., "apple" referring to both a fruit and a brand); semantic omission reflects the cognitive limitations of elderly or young users. The 3 RE levels × 3 context types = 9 ambiguity grades provide comprehensive coverage for evaluating planner robustness.

### Key Design 3: Task-Oriented Context Cognition (TOCC)

- **Function**: Decouples implicit RE resolution from the planning process by first using an LLM to understand the vague instruction and generate a clear paraphrase, which is then used as the basis for task planning.
- **Mechanism**: TOCC operates in two stages:
  1. **Context Cognition Stage**: Given the vague instruction and dialogue context, the LLM focuses exclusively on identifying implicit REs and inferring their true referents, producing a concise, disambiguated paraphrase.
  2. **Planning Stage**: Task planning is performed based on the clarified instruction, so the LLM no longer needs to simultaneously handle language understanding and action generation.

  Comparison with baseline methods:
  - *Aware Prompt (AP)*: Only alerts the planner that instructions may be ambiguous, without promoting deeper reasoning → limited improvement, may induce hallucinations.
  - *Chain-of-Thought (CoT)*: Guides the planner to analyze REs step-by-step before planning, but long prompts are less effective with smaller models.
  - *In-Context Learning (ICL)*: Provides examples to help infer implicit REs, but smaller models have limited ability to learn from examples.
  - *TOCC*: **Physically decouples** understanding and planning into separate stages, preventing attention from being over-allocated to plan generation at the expense of language comprehension within a single inference pass.

- **Design Motivation**: Two key phenomena are observed experimentally: (1) LLMs can correctly resolve implicit REs when explicitly prompted; (2) this capability fails to manifest in planning scenarios. This indicates the problem lies not in LLMs lacking understanding ability, but in attentional competition when understanding and planning are performed simultaneously. TOCC addresses this through task separation.

### Key Design 4: Automated Data Construction Pipeline

- **Function**: Constructs an automated REI dataset generation pipeline based on ALFRED, without requiring manual annotation.
- **Mechanism**:
  1. Six household task types from ALFRED are selected (Pick & Place, Stack & Place, etc.); Pick Two & Place is excluded due to reliability issues.
  2. Tasks are executed using "LLaMA3.1-8B + SayCan"; only successful cases are retained as seed instructions (filtering out tasks that cannot be completed even under clear instructions).
  3. GPT-4o-mini expands the contextual dialogue (Step 1) → three context variants are generated (Step 2) → explicit REs are replaced with implicit REs using a CoT-based method (Step 3).
  4. Counting rules ensure a consistent number of explicit REs across tasks; non-conforming samples are discarded.
- **Design Motivation**: Existing ambiguity datasets (e.g., OntoNotes, Winograd Schema) are annotated by linguists but do not systematically control for the position, frequency, and form of REs. The automated pipeline ensures scale (2,700 samples × 9 grades) and consistency, while eliminating subjective bias from manual annotation.

## Key Experimental Results

### Main Results: Planner Success Rate vs. Ambiguity Level

| Planner | Explicit+Standard | Mixed+Standard | Implicit+Standard | Max Drop |
|---------|-------------------|----------------|-------------------|----------|
| LLaMA3.1-8B + SayCan | 46.90% | 30.10% (−16.8%) | 22.10% (−24.8%) | −24.8% |
| GPT-4o-mini + SayCan | 45.00% | 25.90% (−19.1%) | 24.30% (−20.7%) | −20.7% |
| DeepSeekMath-7B + SayCan | 27.00% | 19.80% (−7.2%) | 14.70% (−12.3%) | −12.3% |
| LLaMA3.1-8B + DAG-Plan | — | — | — | up to −36.9% |
| GPT-4o + SayCan | Higher baseline | Smaller drop | Still notable drop | — |

> Note: The baseline success rate of LLaMA3.1-8B+SayCan without context (Explicit REs only) is 57.7%; adding multi-turn dialogue reduces it to 46.90%.

### Ablation Study: Comparison of Prompting Methods (LLaMA3.1-8B + SayCan, Standard Context)

| Method | Explicit RE Total Error | Mixed RE Total Error | Implicit RE Total Error | Implicit RE Object Omission |
|--------|------------------------|---------------------|------------------------|-----------------------------|
| Baseline | 53.1% | 69.9% | 77.9% | 53.9% |
| + AP | 53.2% (+0.1) | 71.0% (+1.1) | 77.3% (−0.6) | 49.9% (−4.0) |
| + CoT | 52.7% (−0.4) | 69.1% (−0.8) | 77.9% (+0.0) | 47.6% (−6.3) |
| + ICL | 60.8% (+7.7) | 71.7% (+1.8) | 78.6% (+0.7) | 49.9% (−4.0) |
| + TOCC | **41.0%** (−12.1) | **66.4%** (−3.5) | **70.7%** (−7.2) | **40.1%** (−13.8) |
| − Context | 42.3% (−10.8) | 86.9% (+17.0) | 90.6% (+12.7) | 85.1% (+31.2) |

## Key Findings

- **Implicit REs are the primary cause of planning failure**: As the proportion of implicit REs increases, the success rate of all planners consistently declines. For LLaMA3.1-8B+SayCan, the Mixed level yields a 16.8% drop and the Implicit level an additional 8.0% drop. The effects of context noise and information omission are comparatively smaller.

- **The root cause of failure is object omission, not execution error**: Error analysis reveals that as implicit REs increase, the **object omission rate** surges from 22.6% to 53.9% (LLaMA3.1-8B), while the execution error rate actually decreases from 30.5% to 24.0%. This demonstrates that LLMs are not incapable of planning, but rather fail to correctly identify the referents of implicit expressions.

- **LLMs possess RE resolution capability that is suppressed during planning**: When directly prompted to resolve "the heated one," an LLM correctly answers "potato"; yet in a planning context, the same input leads to the erroneous identification of "plate." This indicates that the planning task consumes attentional resources, inhibiting the exercise of language understanding capability.

- **TOCC achieves consistent improvement through decoupling**: TOCC attains the best performance across all ambiguity levels, improving average success rate by 6.5%. At the Implicit REs level, the object omission rate drops from 53.9% to 40.1% (a reduction of 13.8%), representing the largest improvement among all compared methods.

- **Removing context validates pragmatic theory**: Using instructions alone (without context), Explicit REs performance is comparable to TOCC, but performance on Mixed and Implicit REs collapses (object omission rate rising from 38.8% to 81.6%). This is consistent with pragmatic theory—context is indispensable for resolving implicit REs.

## Highlights & Insights

1. **Linguistics-driven AI system design**: The paper systematically integrates linguistic theories—bridging inference, pragmatics—into robot planning evaluation. Rather than simply testing "vague instructions," it grounds benchmark construction in the one-to-many relationship between Signifier and Signified, yielding a theoretically principled benchmark.

2. **Revealing context-dependent capability failure in LLMs**: LLMs do not lack the ability to understand implicit REs; rather, this ability fails to manifest under the multi-task pressure of planning scenarios. This finding has broad implications for all LLM-based systems—one cannot assume that all LLM capabilities remain simultaneously active under arbitrary task combinations.

3. **Effectiveness of a simple method**: TOCC is essentially a two-step decoupling of "understand first, then plan," requiring no additional training or new modules. This simplicity reflects accurate diagnosis of the root cause—returning to the fundamental software engineering principle of separation of concerns.

## Limitations & Future Work

1. **Limited task complexity**: To isolate the effect of REs, the dataset includes only simple, short-horizon, single-target tasks that LLMs can complete under clear instructions. More complex long-horizon, multi-target scenarios are not yet covered.

2. **Only coreference ambiguity is considered**: Human language ambiguity also encompasses deictic expressions (DEs, which depend on spatial/temporal context), syntactic ambiguity, scope ambiguity, and more; this paper addresses only coreference ambiguity.

3. **Absence of multimodal information**: Experiments are conducted in the AI2-THOR simulator, evaluating only textual semantic understanding without incorporating visual or spatial perception (e.g., VLM-based planners might leverage visual cues to resolve "that red thing").

4. **TOCC incurs additional inference overhead**: Two-step decoupling requires two LLM inference calls. For resource-constrained on-robot deployment (small models), the additional inference cost may affect real-time performance.

## Related Work & Insights

### vs. AmbiK (Ivanova et al., 2025)
AmbiK is a dataset of ambiguous natural language instructions in kitchen environments (1k samples). Compared to REI-Bench: (1) AmbiK does not support task planning evaluation; (2) AmbiK does not include multi-turn dialogue context; (3) REI-Bench provides more systematic ambiguity modeling—9-level granularity through the combination of RE type and context type—whereas AmbiK does not systematically distinguish sources of ambiguity. REI-Bench is superior in the comprehensiveness of its evaluation framework.

### vs. CLARA (Park et al., 2023) / KNOWNO (Ren et al., 2023)
CLARA prompts LLMs to judge whether an instruction is determinate; KNOWNO measures and aligns the uncertainty of LLM planners. Both follow a "detect ambiguity → request clarification" paradigm, whereas REI-Bench investigates whether planners can resolve ambiguity autonomously without requesting clarification. TOCC provides a lightweight solution that does not depend on secondary user interaction.

### vs. DialFRED (Gao et al., 2022)
DialFRED adopts a questioner–executor framework, supporting multi-turn interaction through 53K question–answer pairs. Its core idea is to acquire missing information through active questioning. REI-Bench takes a different perspective—assuming that robots should be able to infer autonomously from available context rather than relying on additional interaction. The two paradigms are complementary: TOCC is well-suited for scenarios where users cannot conveniently respond to follow-up questions (e.g., elderly users).

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ★★★★☆ | First systematic modeling of RE ambiguity impact on robot planning; theory-driven benchmark design is innovative; however, the TOCC method itself is relatively straightforward |
| Technical Depth | ★★★☆☆ | The benchmark construction pipeline is complete, but the core method (TOCC) is merely a two-step prompt decoupling with no model training or architectural innovation |
| Experimental Thoroughness | ★★★★★ | Comprehensive ablation across 12 planners (6 LLMs × 4 frameworks), 9 ambiguity grades, and 4 prompting methods; in-depth error attribution analysis (object omission vs. execution error) |
| Practical Impact | ★★★★☆ | Reveals overlooked fragility of LLM planners in real-world scenarios with direct implications for HRI; limited to simple tasks and simulated environments |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can Agents Fix Agent Issues?](../../NeurIPS2025/robotics/can_agents_fix_agent_issues.md)
- [\[ICLR 2026\] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](../../NeurIPS2025/robotics/towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)

<!-- RELATED:END -->
