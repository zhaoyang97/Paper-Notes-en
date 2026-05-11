---
title: >-
  [Paper Note] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering
description: >-
  [ICLR 2026][Code Intelligence][underspecification] This paper introduces Ambig-SWE, an underspecified variant of SWE-Bench Verified, and systematically evaluates LLM coding agents across three dimensions of interactive c…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "underspecification"
  - "interactive agent"
  - "SWE-Bench"
  - "clarification"
  - "software engineering"
date: 2026-05-08
content_hash: 47df0a899bcb49f3
---

# Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering

**Conference**: ICLR 2026
**arXiv**: [2502.13069](https://arxiv.org/abs/2502.13069)
**Code**: [https://github.com/sani903/InteractiveSWEAgents](https://github.com/sani903/InteractiveSWEAgents)
**Area**: Code Intelligence
**Keywords**: underspecification, interactive agent, SWE-Bench, clarification, software engineering

## TL;DR
This paper introduces Ambig-SWE, an underspecified variant of SWE-Bench Verified, and systematically evaluates LLM coding agents across three dimensions of interactive capability—detecting underspecification, formulating clarification questions, and leveraging obtained information. Results show that interaction can improve resolution rates in underspecified settings by up to 74%, yet models default to non-interactive behavior and struggle to distinguish between well-specified and underspecified instructions.

## Background & Motivation
**Background**: LLM agents are widely deployed in software engineering tasks (e.g., OpenHands on SWE-Bench), but user instructions are frequently underspecified. Human developers proactively ask for clarification when information is insufficient, whereas AI agents tend to make assumptions and proceed directly.

**Limitations of Prior Work**: (1) Underspecified instructions lead to erroneous outputs, security risks, and wasted computational resources; (2) existing research on underspecification focuses only on the absence of a single detail, whereas real-world software engineering tasks involve multiple interdependent information gaps; (3) LLMs default to non-interactive behavior—failing to ask questions even when faced with severe information deficiency.

**Key Challenge**: Interaction can effectively recover performance lost due to underspecification (by up to 74%), yet models do not know when to interact, what to ask, or how to utilize the information obtained.

**Goal**: To systematically evaluate and quantify LLM agents' ability to handle underspecified instructions by decomposing the capability into atomic skills that can be improved independently.

**Key Insight**: An underspecified variant of SWE-Bench Verified is constructed; three evaluation settings (Full/Hidden/Interaction) are designed, with GPT-4o simulating the user.

**Core Idea**: Decompose underspecification handling into three steps—detect, ask, and utilize—and use interactive experiments to quantify the capability and improvement margin at each step.

## Method

### Overall Architecture
Three evaluation settings: (1) **Full**—complete GitHub issue, no interaction; (2) **Hidden**—underspecified version (GPT-4o summary), no interaction; (3) **Interaction**—underspecified version, with access to a GPT-4o user proxy holding the complete information. Six models are evaluated under the OpenHands framework.

### Key Designs

1. **Three-Dimensional Evaluation Decomposition**:

    - **RQ1: Effect of Interaction**—compare resolution rates across Hidden vs. Interaction vs. Full to quantify how much interaction recovers performance.
    - **RQ2: Underspecification Detection**—models are randomly given Full or Hidden inputs; accuracy, FPR, and FNR are measured to assess whether models can distinguish the two and choose to interact accordingly.
    - **RQ3: Question Quality**—analyze whether the clarification questions posed by models are targeted and whether they elicit key missing information.

2. **User Proxy Design**:

    - GPT-4o holds the complete issue and only answers questions whose answers are explicitly present in its held information.
    - If queried about information beyond what it holds, it responds with "I don't have that information."
    - The conservative design isolates the agent's information-acquisition ability and prevents proxy hallucination.

3. **Interaction Encouragement Gradient**:

    - Three prompt levels: Neutral (may ask questions) → Moderate (carefully check information completeness) → Strong (asking questions is critical to success).
    - Behavioral changes are measured across different levels of encouragement.

### Loss & Training
N/A (evaluation paper; no model training involved).

## Key Experimental Results

### Main Results (Resolution Rate %)

| Model | Hidden | Interaction | Full | Recovery Rate |
|-------|--------|-------------|------|---------------|
| Claude S4 | 49.0 | 52.4 | 58.8 | **89%** |
| Claude S3.5 | 27.3 | 35.0 | 43.8 | ~80% |
| Qwen3 Coder | 45.6 | 53.6 | 59.2 | ~85% |
| Haiku 3.5 | 13.0 | 20.8 | 26.0 | ~80% |
| Deepseek-v2 | 2.0 | 7.2 | 12.2 | 59% |
| Llama 70B | 1.4 | 3.6 | 6.6 | 54% |

### Underspecification Detection (Strong Prompt)

| Model | Accuracy | FPR↓ | FNR↓ |
|-------|----------|------|------|
| Claude S4 | **0.89** | 0.03 | 0.18 |
| Claude S3.5 | 0.76 | 0.36 | 0.10 |
| Qwen3 Coder | 0.50 | 0.00 | **1.00** |

### Key Findings
- **Qwen3 Coder FNR = 1.0**: Even under the Strong prompt, the model never initiates interaction—completely ignoring underspecification—adhering instead to a fixed SWE-Bench problem-solving protocol.
- Interaction improves resolution rates by up to 74% (Hidden → Interaction), yet remains substantially below Full, indicating that models have limited ability to leverage interactively obtained information.
- Information-type analysis: acquiring navigation information (e.g., file paths) yields the greatest benefit for weaker models but provides limited gains for stronger models, which can locate relevant code independently.
- Model scale and coding ability do not equate to interactive ability—Haiku (a smaller model) achieves information-utilization rates comparable to Sonnet 3.5.
- Claude S4 extensively explores the codebase in the Hidden setting to compensate for missing information (averaging 65 steps), increasing to 75 steps in the Interaction setting—interaction improves outcomes but not efficiency.

## Highlights & Insights
- **The detect–ask–utilize decomposition framework**: This approach breaks the vague notion of "interactive capability" into atomic skills that can be independently evaluated and improved, offering significant methodological value.
- **Reveals a deficiency in current training paradigms**: Models are trained to optimize task completion rates rather than to learn *when* to ask—resulting in a default non-interactive behavior.
- **Stereotyped behavior in Qwen3 Coder**: Even after receiving information from a user response, the model reverts to its fixed protocol and re-explores the codebase—indicating that it does not genuinely understand the purpose of interaction.

## Limitations & Future Work
- The underspecified versions are generated by GPT-4o and may be "cleaner" than naturally occurring underspecified user issues.
- The user proxy in the Interaction setting is GPT-4o, which may not faithfully reflect real user behavior.
- Evaluation is limited to SWE-Bench (Python repositories); other programming languages and domains are not covered.
- The paper does not propose training methods to improve models' interactive capability; it is purely diagnostic.

## Related Work & Insights
- **vs. AQuA VQA**: AQuA studies VLMs' strategy selection in response to visual ambiguity, while Ambig-SWE examines coding agents' interactive capability in response to missing information—different modalities, same theme (handling uncertainty).
- **vs. SWE-Bench**: SWE-Bench assumes complete instructions; Ambig-SWE specifically tests agent behavior under incomplete instructions, which is closer to real-world deployment.
- **vs. ClearVQA**: ClearVQA trains models on a binary choice (answer vs. ask); Ambig-SWE decomposes the problem into three steps and evaluates it in the more complex setting of software engineering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First systematic evaluation of interactive capability in SWE agents; the three-step decomposition framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 models, 3 settings, 3 prompt levels, multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous experimental design with in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ — Important diagnostic of agent interactive capability with direct implications for future training directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)
- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)

</div>

<!-- RELATED:END -->
