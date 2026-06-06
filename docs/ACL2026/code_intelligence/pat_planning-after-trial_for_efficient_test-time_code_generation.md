---
title: >-
  [Paper Note] PaT: Planning-after-Trial for Efficient Test-Time Code Generation
description: >-
  [ACL2026][Code Intelligence][test-time computation] PaT replaces the "planning-before-trial" approach in code generation with "trial-first, plan-on-failure…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "test-time computation"
  - "code generation"
  - "adaptive planning"
  - "execution verification"
  - "heterogeneous models"
date: 2026-05-08
content_hash: 73cd40b153661168
---

# PaT: Planning-after-Trial for Efficient Test-Time Code Generation

**Conference**: ACL2026  
**arXiv**: [2605.07248](https://arxiv.org/abs/2605.07248)  
**Code**: No public code (Not provided in the paper)  
**Area**: code_intelligence  
**Keywords**: test-time computation, code generation, adaptive planning, execution verification, heterogeneous models

## TL;DR
PaT replaces the "planning-before-trial" approach in code generation with "trial-first, plan-on-failure," using execution feedback to trigger expensive decomposition steps. It utilizes a heterogeneous configuration—small models for generation and large models for planning—to significantly improve the trade-off between Pass@1 and inference cost.

## Background & Motivation
**Background**: LLM code generation has evolved from single few-shot generation toward test-time computation scaling. Common strategies include Best-of-N sampling, using generated tests for candidate filtering, iterative debugging, and decomposing complex problems into multiple helper functions to be combined into a final program. A representative method for the latter is FunCoder, which aims to solve algorithmic problems that are difficult to generate directly by "understanding the problem structure first, then implementing sub-problems separately."

**Limitations of Prior Work**: While decomposition improves success rates for difficult problems, it incurs full planning overhead even for easy problems. The paper notes that a small model under standard reasoning can already solve many basic code problems; for instance, Qwen3-4B achieves an average Standard Pass@1 of 76.05% on foundational benchmarks. If every problem is planned first, many solvable problems incur unnecessary costs for planning, helper function generation, and verification, leading to rapid cost expansion.

**Key Challenge**: The key to test-time computation is not just "spending more compute," but "on which samples to spend it." Planning-before-Trial (PbT) treats planning as a default prerequisite step, which is suitable for difficult cases but fails to identify simple ones. Conversely, direct generation is cost-effective but lacks a mechanism to upgrade strategies upon failure. The fundamental trade-off is: earlier planning is more stable but prone to waste, while later planning is more efficient but requires a reliable failure signal.

**Goal**: The authors aim to solve three sub-problems: first, how to determine if a problem warrants a planning workflow in a training-free manner; second, how to reuse verified sub-solutions after planning to avoid starting from scratch; and third, how to allocate models of different sizes to different roles so that common trials are cheap while a few critical planning steps are sufficiently strong.

**Key Insight**: Code generation provides a harder signal than general natural language reasoning: programs can be executed, and candidate solutions can be verified with test cases. PaT observes that if a model fails to pass tests after multiple direct attempts, this is a more credible signal than the model self-assessing the problem as "difficult." It indicates the problem likely exceeds direct generation capabilities, making it more reasonable to initiate planning at that point.

**Core Idea**: Use execution failure as a trigger for planning, changing "plan for all samples" to "plan only for samples that fail verification," thereby concentrating expensive test-time computation on code problems that truly require decomposition.

## Method
The PaT method does not propose a new code model but reorganizes the test-time reasoning workflow. It views a code problem as a natural language specification $x$, with the goal of generating a program $\mathcal{F}$ that satisfies the spec. The system has two roles: a generator $M_G$ responsible for direct code generation or sub-problem implementation, and a planner $M_P$ responsible for decomposing the original problem into a top-level implementation and several sub-problem specifications $\{x_i\}$ upon failure. The final program is constructed by a Compose operation merging the main function with verified helper functions.

### Overall Architecture
The input is a code generation problem, and the output is the final program. PaT first performs a Best-of-N trial without planning: sampling multiple candidate implementations for the same specification. Subsequently, the system generates a test set $\mathcal{T}(x)$, executes candidate programs in a sandbox Python runtime, and uses the number of passed tests $p=\textsc{Evaluate}(\mathcal{F}, \mathcal{T}(x))$ to represent candidate quality.

If a candidate passes all tests, the process terminates immediately and returns that program. This step is the source of PaT's cost savings: many simple problems do not enter the planner or generate extra sub-functions.

If direct candidates fail, PaT calls the planner. The planner provides a new main implementation draft and several sub-problems based on the original problem and the current set of helpers. Each unimplemented sub-problem calls PaT recursively: sub-problems also attempt direct generation first, and only decompose further if verification fails. Once all sub-solutions pass their respective tests, the system adds them to the helper set and combines them back into the parent problem for overall verification.

If the combined parent program passes all tests, PaT returns the result. If it still fails, the system does not recurse infinitely but enters a re-planning loop: the planner sees the original problem and successfully implemented helpers as context for a new decomposition. If the number of passed tests in a new round does not exceed the previous round, PaT triggers a plateau rule, stopping planning and returning the best-known candidate to avoid being dragged into high-cost cycles by faulty test cases or invalid decompositions.

### Key Designs
1. **Failure-Triggered Adaptive Planning**:
    - **Function**: Allows the planner to intervene only after direct generation fails, rather than decomposing every sample first.
    - **Mechanism**: The generator first performs Best-of-N candidate generation, followed by test set verification. If any candidate passes all tests, it is returned immediately; if all candidates fail, this failure signal triggers the planner to split the problem. This strategy uses $p=|\mathcal{T}(x)|$ as the acceptance condition rather than relying on the model's self-assessed difficulty.
    - **Design Motivation**: Code problems are naturally executable, providing failure signals more reliable than subjective LLM difficulty judgments. Compared to PbT methods like FunCoder, PaT does not unconditionally pay planning costs for simple problems, resulting in lower average costs.

2. **Generated Tests, Strict Passing, and Plateau Stopping**:
    - **Function**: Provides actionable binary trigger signals for PaT while mitigating potential errors in automatically generated tests.
    - **Mechanism**: The system generates an average of 6.7 test cases per problem; candidates must pass all tests to succeed. The paper does not use CodeT-style consensus scoring because PaT requires a clear switch for "upgrading to planning" rather than picking the most likely correct answer from a pool. To handle noisy tests, PaT records the number of passes $p^{(t)}$ in each round; when $p^{(t)} \leq p^{(t-1)}$, it stops and returns the best result from the previous round.
    - **Design Motivation**: Strict passing reduces false acceptance, while plateau stopping prevents the system from repeated planning to accommodate a few false positives. Figure 3 shows that on HumanEval, most generated tests have no false positives (e.g., 63.4% of problems for Qwen3-4B), making this strict signal practical.

3. **Heterogeneous Small Generator + Large Planner Configuration**:
    - **Function**: Assigns high-frequency, localized code generation to cheaper models and low-frequency planning, which requires higher global understanding, to stronger models.
    - **Mechanism**: PaT decouples the generator and planner. The generator handles direct candidates and sub-problem implementation, suitable for cost-efficient sLMs; the planner handles problem understanding, decomposition, and re-planning after failure, suitable for stronger LLMs. Since the planner is only called upon failure, the high unit cost of the large model is amortized across a small subset of difficult samples.
    - **Design Motivation**: Using only small models increases failure rates, while using only large models raises the baseline cost of every trial. Heterogeneous PaT balances the two: the generator should not be so weak as to trigger planning frequently, nor does it need to be so strong as to incur large model costs for simple problems.

### Loss & Training
PaT does not require training new policy models or additional training losses. It is an inference-time policy implemented via prompts, sampling, test generation, sandbox execution, and recursive planning. In experiments, the PaT trial phase uses Best-of-N with $N=5$ and temperature=0.8; FunCoder replication uses $N=11$ per its original setting.

For cost modeling, the paper uses public token pricing. Theoretical analysis in the appendix suggests that if planning costs are lower than the generation costs saved by the heterogeneous configuration, a small-model generator can achieve a lower expected cost than a homogeneous large-model strategy. This analysis serves model selection rather than training objectives.

## Key Experimental Results

### Main Results
The paper evaluates PaT under two settings. The first is a homogeneous setting (same model for both roles) to verify if the PaT strategy is inherently superior to PbT. The second is a heterogeneous setting (fixed strong planner, varying smaller generators) to see if role splitting further reduces costs.

Benchmarks include HumanEval, MBPP, and their EvalPlus extensions. Difficult benchmarks use xCodeEval, categorized by FunCoder's scheme into Easy, Mid, Hard, and Expert. Metrics are Pass@1 and normalized LLM cost.

| Setting | Method | Average Pass@1 | Gain | Relative Cost | Conclusion |
|------|------|-------------|----------|----------|------|
| Qwen3-4B foundational | Standard | 76.05 | - | 1.00 | Small models solve many simple problems directly |
| Qwen3-4B foundational | FunCoder | 81.18 | +5.13 | 8.31 | Planning improves performance but at high cost |
| Qwen3-4B foundational | PaT | 83.13 | +7.08 | 4.85 | Higher score than FunCoder at ~58% cost |
| Qwen3-8B foundational | FunCoder | 83.82 | +6.18 | 9.43 | PbT remains expensive |
| Qwen3-8B foundational | PaT | 85.58 | +7.94 | 5.00 | Better performance and cost at same scale |
| Qwen3-14B foundational | FunCoder | 84.84 | +5.03 | 8.82 | Planning overhead remains significant for larger models |
| Qwen3-14B foundational | PaT | 86.18 | +6.37 | 4.91 | Higher score with ~56% of FunCoder cost |
| Qwen3-32B foundational | FunCoder | 87.66 | +4.31 | 8.93 | Large models also waste cost on simple samples |
| Qwen3-32B foundational | PaT | 88.37 | +5.02 | 5.09 | Highest average Pass@1 with significantly lower overhead |

The key takeaway from Table 1 is that across all Qwen3 scales (4B, 8B, 14B, 32B), PaT achieves a higher average Pass@1 than FunCoder while costing only about 50-60%. Results are consistent across model families: on Llama3.1-8B, PaT averages 73.31 vs. FunCoder's 71.53; on DeepSeek-Coder, PaT averages 84.19 vs. FunCoder's 83.60.

On the more difficult xCodeEval benchmark, PaT's performance advantage persists, but cost dynamics become more complex.

| Model | Method | Easy | Mid | Hard | Expert | All | Cost |
|------|------|------|-----|------|--------|-----|------|
| Qwen3-4B | Standard | 37.70 | 17.86 | 3.45 | 0.00 | 18.40 | 1.00 |
| Qwen3-4B | FunCoder | 55.19 | 29.46 | 12.64 | 0.00 | 29.00 | 12.95 |
| Qwen3-4B | PaT | 61.75 | 40.18 | 14.94 | 0.00 | 34.20 | 17.93 |
| Qwen3-8B | Standard | 54.10 | 28.57 | 5.75 | 0.00 | 27.20 | 1.00 |
| Qwen3-8B | FunCoder | 64.48 | 43.75 | 9.20 | 0.00 | 35.00 | 8.62 |
| Qwen3-8B | PaT | 69.95 | 45.54 | 11.49 | 0.00 | 37.80 | 6.98 |
| Qwen3-14B | Standard | 53.55 | 36.61 | 9.20 | 0.00 | 25.20 | 1.00 |
| Qwen3-14B | FunCoder | 73.22 | 52.68 | 18.39 | 0.00 | 41.80 | 9.03 |
| Qwen3-14B | PaT | 73.77 | 53.57 | 21.84 | 0.85 | 43.00 | 6.49 |
| Qwen3-32B | Standard | 54.64 | 39.29 | 11.49 | 0.00 | 30.80 | 1.00 |
| Qwen3-32B | FunCoder | 74.86 | 54.46 | 16.09 | 0.00 | 42.40 | 7.87 |
| Qwen3-32B | PaT | 74.32 | 54.46 | 18.39 | 1.69 | 43.00 | 6.00 |

xCodeEval results show that as the proportion of difficult problems increases, PaT actively triggers more planning. For weaker models like Qwen3-4B, it can even be more expensive than FunCoder due to frequent failures. However, this is not strategy failure but an intentional allocation of budget toward hard samples based on failure signals, resulting in an "All" metric increase from 29.00 to 34.20. For models 8B and larger, PaT achieves both higher performance and lower costs.

### Ablation Study
The paper breaks down contributions through logic comparison and heterogeneous configuration analysis. Table 3 is critical: fixing Qwen3-32B as the planner allows smaller generators to approach the performance of a homogeneous large-model PaT with significantly lower costs.

| Generator | Planner | Average Pass@1 | Relative Cost | Description |
|-----------|---------|-------------|----------|------|
| Qwen3-32B | Qwen3-32B | 88.37 | 1.00 | Homogeneous large-model PaT (upper bound) |
| Qwen3-14B | Qwen3-14B | 86.18 | 0.47 | Homogeneous 14B, lower cost but weaker planning |
| Qwen3-14B | Qwen3-32B | 87.53 | 0.49 | Upgraded planner only, approaches 32B performance at <50% cost |
| Qwen3-8B | Qwen3-8B | 85.58 | 0.25 | Homogeneous 8B, very low cost with performance gap |
| Qwen3-8B | Qwen3-32B | 87.39 | 0.31 | Within <1% of 32B homogeneous performance, 31% cost |
| Qwen3-4B | Qwen3-4B | 83.13 | 0.14 | Cheapest but generator is a bottleneck |
| Qwen3-4B | Qwen3-32B | 84.78 | 0.18 | Strong planner helps, but 4B generator remains the limitation |

The 8B+32B configuration is the "sweet spot" highlighted by the paper: an average Pass@1 of 87.39 is less than 1 point below the 88.37 of 32B+32B, while costing only 31%. PaT makes it possible for "large models to plan only for the few failure samples."

### Key Findings
- PaT's primary gain comes from "skipping unnecessary planning." On foundational benchmarks, PaT outperforms FunCoder in both score and cost across all Qwen3 sizes, showing failure-triggering better fits real-world difficulty distributions.
- Generated tests are not perfect but suffice as trigger signals. Most HumanEval problems have no false positives, and noisy tests are controlled by the plateau rule.
- The optimal heterogeneous configuration is not simply the smallest model. 8B+32B provides a better balance than 4B+32B.
- On extremely difficult data, PaT may spend more. Qwen3-4B's higher cost on xCodeEval is due to frequent failures, but this yields significant performance gains.
- The strategy is model-agnostic, as shown by results on Llama3.1 and DeepSeek-Coder.

## Highlights & Insights
- **Using Verification Failure as a Budget Allocation Signal**: The most clever aspect is avoiding a trained difficulty classifier by using execution failure to trigger planning. Code tasks provide natural feedback, which is lighter and more reproducible than learned policies.
- **Inverting the PbT Default Assumption**: FunCoder assumes "complex problems need planning, so plan first"; PaT assumes "don't plan if you can solve it directly." This small inversion significantly impacts the cost curve because benchmarks still contain many simple-to-medium problems.
- **Practical Heterogeneous Configuration**: Many systems already have various model sizes available. PaT provides a natural division of labor: small models for high-frequency generation and large models for rare planning. This can extend to math reasoning or agent workflows where reliable failure signals exist.
- **Plateau Stopping as Necessary Mechanism**: Simple failure triggers could trap the system in loops with faulty tests. Using non-increasing pass counts as a stop condition transforms recursive planning into a controllable test-time loop.

## Limitations & Future Work
- PaT heavily relies on verification quality. While code execution provides clear pass/fail signals, this is harder to obtain in open-ended generation, UI/UX, or creative writing.
- Automated tests still suffer from false positives or insufficient coverage. Strict passing mitigates errors, but faulty tests can mis-trigger planning or cause premature stopping.
- On very difficult data, weak generators fail frequently, causing excessive planner calls. One cannot blindly choose the cheapest model as the generator.
- Expert-level xCodeEval remains largely unsolved. Even with PaT pushing success rates from 0 to ~1.7%, absolute performance remains low, suggesting recursive decomposition cannot fully substitute for stronger algorithmic reasoning.
- The evaluation focuses on Python and individual file environments. Future work could examine multi-file project-level generation and integration with iterative debugging frameworks.

## Related Work & Insights
- **vs FunCoder**: FunCoder uses fixed Planning-before-Trial; PaT converts this to a reactive approach. PaT avoids waste on simple problems but may trigger more planning rounds if the generator is weak on hard problems.
- **vs CodeT**: CodeT uses tests for consensus-based selection from a pool; PaT uses tests as a binary control signal to decide whether to escalate to planning. Both use feedback, but for different optimization goals.
- **vs Best-of-N**: Best-of-N only expands the candidate pool; PaT changes the solution structure via decomposition when candidates in the pool are incorrect.
- **vs Learned Adaptive Policies**: Instead of training a policy to decide when to plan, PaT uses direct execution feedback. The insight is: if a task provides hard feedback, prioritize using that feedback loop over introducing an additional classifier.

## Rating
- Novelty: ⭐⭐⭐⭐ The inversion of planning order is simple yet effective; innovation lies in test-time strategy and heterogeneous roles.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various model sizes and families across multiple benchmarks with cost analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly articulated logic for cost-performance trade-offs.
- Value: ⭐⭐⭐⭐⭐ Highly practical for engineering code generation systems by addressing when to call expensive reasoning workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)
- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
