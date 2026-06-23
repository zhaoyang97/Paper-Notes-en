---
title: >-
  [Paper Note] Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies
description: >-
  [ICLR 2026][Reinforcement Learning][programmatic policy] MLES treats Multimodal Large Language Models (MLLMs) as "strategy programmers capable of watching replays." Combined with evolutionary search, it directly generates readable programmatic control policies. By using execution screens (Behavioral Evidence) to diagnose failure modes and perform targeted code modifications,
tags:
  - ICLR 2026
  - Reinforcement Learning
  - programmatic policy
  - multimodal LLM
  - evolutionary search
  - interpretable RL
  - behavioral evidence
date: 2026-05-08
content_hash: 8dc416b82d1cf371
---
# Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=OHFNJoNtjW](https://openreview.net/forum?id=OHFNJoNtjW)  
**Code**: [https://github.com/QingL2000/MLES](https://github.com/QingL2000/MLES)  
**Area**: Reinforcement Learning / Interpretable Control Policies / LLM-assisted Evolutionary Search  
**Keywords**: programmatic policy, multimodal LLM, evolutionary search, interpretable RL, behavioral evidence  

## TL;DR
MLES treats Multimodal Large Language Models (MLLMs) as "strategy programmers capable of watching replays." Combined with evolutionary search, it directly generates readable programmatic control policies. By using execution screens (Behavioral Evidence) to diagnose failure modes and perform targeted code modifications, it achieves performance comparable to PPO on Lunar Lander and Car Racing while remaining fully transparent and traceable.

## Background & Motivation
**Background**: Deep Reinforcement Learning (DRL) performs strongly in control tasks, but policies exist as neural networks—black boxes that are difficult to understand, verify, and debug. In safety-critical scenarios like autonomous driving and healthcare, this opacity directly hinders deployment. Meanwhile, the "LLM-assisted Evolutionary Search" (LES) paradigm has emerged, using LLMs as mutation/crossover operators within evolutionary computation to iteratively optimize candidate designs. This has succeeded in discovering code, algorithms, and heuristics. In the RL domain, works like Eureka have proven that LES can be used for reward shaping.

**Limitations of Prior Work**: (1) Black-box policies cannot be understood or verified by humans, and knowledge is difficult to reuse or transfer. (2) Existing LES in RL is mostly limited to "auxiliary component" design like reward shaping; no prior work has used it to directly synthesize end-to-end control policies. (3) Traditional LES relies solely on scalar feedbacks (episode reward), meaning the LLM only knows "if a policy is good" but not "why it failed," causing it to degenerate into blind trial-and-error.

**Key Challenge**: High performance and transparency are difficult to reconcile—DRL provides performance but lacks transparency, while symbolic methods (genetic programming + DSL) are transparent but depend on hand-crafted domain-specific languages with poor scalability.

**Goal**: Propose a framework capable of **directly synthesizing** programmatic control policies that match the performance of strong DRL baselines, while ensuring the policies and the discovery process remain transparent and traceable for humans.

**Core Idea**: **[Key Insight]** Enable MLLMs to not just look at scores but to "watch replays" like human experts. By feeding visual trajectories of policy execution (Behavioral Evidence, BE) to the MLLM, the model can diagnose failure modes (e.g., losing control on sharp turns, reward hacking) and specifically rewrite the policy code. This transforms LLM-driven automatic discovery from "random trial-and-error" into "evidence-based diagnostic refinement."

## Method

### Overall Architecture
MLES (Multimodal LLM-assisted Evolutionary Search) models control policy discovery as a search process driven by an evolutionary search loop within an open semantic policy space implicitly defined by an MLLM. Each candidate policy is an "individual" characterized by a quadruple—**Code** (executable Python program defining decision logic), **Thought** (natural language description of design intent), **Quantitative Metrics** (reward obtained from environment execution as the ground-truth target), and **Behavioral Evidence (BE)** (execution images/videos serving as diagnostic signals). The framework operates in a closed loop: selecting parents from the policy pool → retrieving their behavioral evidence → constructing multimodal few-shot prompts → MLLM generating offspring policies → evaluation and generation of new behavioral evidence → updating the policy pool. The paper uses EoH (Evolution of Heuristics) as the evolutionary backbone for a specific implementation (MLES-EoH), purposefully minimizing structural changes to search operators to isolate the gains from the "visual feedback behavior analysis" contribution.

```mermaid
flowchart LR
    A[Task Description + Initial Policy] --> B[Policy Pool]
    B -->|Exponential Rank Selection| C[Select Parent Individuals]
    C --> D[Retrieve Parent Behavioral Evidence BE]
    D --> E[Prompt Sampler<br/>Construct Multimodal Few-shot Prompt]
    E --> F[MLLM Integration<br/>Generate Offspring Policy Code]
    F --> G[Evaluator<br/>Environment Execution → Quantitative Metrics]
    G --> H[Behavior Summarizer<br/>Execution Trajectory → Behavioral Evidence]
    H --> B
    G -.Rank by Quant. Metrics Retain top-N.-> B
```

### Key Designs

**1. Behavioral Evidence (BE): Diagnostic signals from "how good" to "why".** This is the core differentiator of MLES from standard LES. Standard LES only feeds back the scalar metric $F(\pi)$ (episode reward) to the LLM; the LLM knows the rank but not where the policy failed. MLES introduces a Behavior Summarizer module to convert raw execution trajectories into "principle-rich" BE—this can be images, videos, or even text state sequences (the paper uses frame stacking to compress execution segments into a multi-frame image). By analyzing BE, the MLLM can directly observe failure modes such as "driving off-track due to excessive speed during sharp turns" or "reward hacking by exploiting bugs." The design enforces a clear division of labor: quantitative metrics handle selection and convergence, while BE guides generation. This mimics how human experts refine strategies—by observing behavioral patterns to locate root causes rather than just looking at scores.

**2. Multimodal Modification Operators (M1_M / M2_M): Translating diagnosis into code rewrites.** In addition to standard EoH exploration operators, MLES adds two modification operators that utilize BE. M1_M allows the MLLM to jointly analyze policy code and its BE to identify behavioral flaws and rewrite control logic (e.g., detecting "overshooting steer causing oscillation" and lowering steering coefficients). M2_M enables the MLLM to identify critical parameters and perform targeted tuning based on observed evidence. Combined with the retained exploration operators—E1 (reviewing parent code/thought to synthesize fundamentally different logic) and E2 (extracting common patterns from multiple parents, similar to crossover)—these form a synergy between a "Creativity Engine (E)" and a "Refinement Engine (M)." Ablation studies show that removing M1_M/M2_M leads to a 12%–14% performance drop on Car Racing, proving that evidence-based refinement is the primary source of efficiency.

**3. Multimodal Few-shot Prompting + Exponential Rank Selection: Upgrading MLLM from "coder" to "reasoning agent".** In each step, the Prompt Sampler dynamically assembles four context components: (1) static task description, (2) parent information (code + thought), (3) relevant BE, and (4) operator-specific instructions. This compound prompt makes the MLLM a reasoning agent that observes, diagnoses, and improves. Parent selection follows standard exponential rank selection used in evolutionary computation, where selection probability $p_i \propto e^{-c \cdot r_i}$ ($r_i$ being the rank by performance). Since every MLES "mutation" requires an expensive MLLM inference, high selection pressure is maintained to ensure sample efficiency while keeping non-zero probabilities for lower-ranked individuals to preserve diversity. Policy pool management removes redundant offspring and retains the top-N (N=16) individuals based on quantitative metrics for the next generation.

## Key Experimental Results

### Main Results
On two Gym control tasks, 5 independent runs were conducted. Mean/SEM/Best are reported. For Lunar Lander, scores >1.00 indicate successful landings; for Car Racing, 100 indicates perfect completion.

| Method | LunarLander Train Mean | LunarLander Test Mean | CarRacing Train Mean | CarRacing Test Mean |
|------|------|------|------|------|
| DQN | 1.017 | 0.508 | 79.78 | 71.72 |
| PPO | 1.032 | 0.846 | **99.21** | 94.55 |
| Initial policy | 0.629 | 0.653 | 17.77 | 17.62 |
| EoH | 1.053 | 0.776 | 89.81 | 79.29 |
| **Ours (MLES)** | **1.090** (SEM±0.005) | 0.819 | 98.07 | **96.36** |

Ours (MLES) achieved the highest training performance on Lunar Lander and the second-best on Car Racing (closely following PPO), while outperforming PPO on the Car Racing test set (96.36 vs 94.55). Compared to EoH (same framework without BE), MLES leads significantly with lower SEM, especially on the harder Car Racing task.

### Ablation Study
Three independent runs. $\Delta$Perf indicates performance drop relative to full MLES.

| Variant | LunarLander | $\Delta$Perf | CarRacing | $\Delta$Perf |
|------|------|------|------|------|
| MLES (Full) | 1.090 | 0.00% | 98.70 | 0.00% |
| w/o E1 | 1.093 | +0.17% | 87.53 | -11.32% |
| w/o E2 | 1.082 | -0.68% | 89.04 | -9.79% |
| w/o M1_M | 0.997 | -8.54% | 86.71 | -12.15% |
| w/o M2_M | 1.020 | -6.41% | 84.77 | -14.12% |

Removing multimodal modification operators (M1_M/M2_M) caused the largest drops, proving evidence-based refinement is central. Exploration operators (E1/E2) had minimal impact on the simpler Lunar Lander but were critical for the complex Car Racing task.

### Key Findings
- **Search Efficiency**: MLES converges in ~5000 environment resets on Lunar Lander, much faster than PPO/DQN. On Car Racing, convergence speed matches PPO with significantly lower variance.
- **Cold-start Capability**: Starting from zero (no initial policy) with only 2000 LLM queries, MLES successfully discovers high-performance policies. Switching to a stronger GPT-5-mini allows solving tasks without any seeds.
- **Base Model Scaling**: GPT-5-mini achieves full scores on Car Racing using only ~17% of the sample budget compared to GPT-4o-mini, indicating MLES naturally benefits from base model improvements.
- **Readability Verification**: Review by 20 CS graduate students confirmed the generated policy code is readable, with embedded comments being highly helpful.

## Highlights & Insights
- **Engineering "Watching Replays"**: The Behavioral Evidence (BE) design addresses the "score-only, no diagnosis" pain point of standard LES, transforming discovery into diagnostic refinement.
- **Division of Labor**: Quantitative metrics handle selection/convergence while BE handles generation/modification. This clear separation prevents feedback confusion.
- **Direct Synthesis vs. Auxiliary Shaping**: Moving beyond the comfort zone of reward shaping, this work is the first to directly synthesize programmatic policies end-to-end to rival DRL.
- **Dual Transparency**: Not only the output (policy code) but also the discovery process (evolutionary lineage showing why and how each step was refined) is fully traceable. Figure 5 illustrates the evolution in Car Racing from "contour detection → ROI focus → adaptive braking."

## Limitations & Future Work
- **Generalization Challenges**: All tested methods (including MLES) show a drop from training to test performance. Policy ensembles were used in the appendix to mitigate this.
- **Limited Task Scale**: Validation is restricted to classic Gym tasks. High-dimensional, long-horizon, or real-world robotic control tasks have not yet been explored.
- **MLLM Inference Cost**: Every mutation requires an MLLM call. While exponential rank selection improves sample efficiency, the per-step cost is high, and the LLM query budget (2000) acts as a hard constraint.
- **Task-Dependent BE Design**: The form of BE (frame stacking, ROI, etc.) currently requires manual design of the Behavior Summarizer for each task.
- **Future Work**: The framework is search-algorithm agnostic and can be paired with stronger backbones. It is expected to become a new paradigm for developing transparent and verifiable control policies as base models improve.

## Related Work & Insights
- **Two Paths to Readable Policies**: *Post-hoc* methods distill black-box policies into decision trees or DSL programs, but remain proxy explanations. *Direct* methods (genetic programming + DSL) search readable policies directly but depend on hand-crafted DSLs. MLES follows the direct route but uses MLLM's open semantic space to replace rigid DSLs, bypassing scalability bottlenecks.
- **LLM-assisted Evolutionary Search**: FunSearch and ReEvo use LLMs for code discovery; Eureka uses LES for reward shaping. MLES differs by advancing LES to direct policy synthesis and adding multimodal behavioral feedback.
- **Insight**: The BE approach is valuable for any "LLM Generation + Execution Feedback" loop (e.g., agent self-improvement, skill synthesis)—do not just provide scalar rewards; feed back execution trajectories to enable diagnosis instead of guesswork.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Introducing BE into the LES loop and achieving end-to-end programmatic policy synthesis is novel and addresses a significant pain point.
- **Experimental Thoroughness**: ⭐⭐⭐ — Strong baselines and ablations provided, but limited to relatively small Gym tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework and lineage diagrams with effective metaphors.
- **Value**: ⭐⭐⭐⭐ — Provides a compelling paradigm for transparent control policies with significant implications for safety-critical RL.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[ICLR 2026\] GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies](goldenstart_q-guided_priors_and_entropy_control_for_distilling_flow_policies.md)
- [\[ICLR 2026\] Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](refining_hybrid_genetic_search_for_cvrp_via_reinforcement_learning-finetuned_llm.md)
- [\[ICLR 2026\] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation](mars-sep_multimodal-aligned_reinforced_sound_separation.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)

</div>

<!-- RELATED:END -->
