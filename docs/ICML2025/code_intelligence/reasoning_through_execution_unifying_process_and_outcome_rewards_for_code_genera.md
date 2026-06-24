---
title: >-
  [Paper Note] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation
description: >-
  [ICML 2025][Code Intelligence][Process Supervision] ORPS (Outcome-Refining Process Supervision) is proposed, which unifies process and outcome rewards in a tree-search framework by combining code execution feedback with LLM self-criticism. It achieves a 26.9% accuracy improvement and a 42.2% efficiency boost in code generation without training a PRM.
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "Process Supervision"
  - "Outcome Supervision"
  - "Code Generation"
  - "Reward Models"
  - "Tree Search"
date: 2026-05-08
content_hash: 295c430c75d3f4df
---

# Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation

**Conference**: ICML 2025  
**arXiv**: [2412.15118](https://arxiv.org/abs/2412.15118)  
**Code**: [github.com/zhuohaoyu/ORPS](https://github.com/zhuohaoyu/ORPS)  
**Area**: Code Intelligence  
**Keywords**: Process Supervision, Outcome Supervision, Code Generation, Reward Models, Tree Search

## TL;DR

ORPS (Outcome-Refining Process Supervision) is proposed, which unifies process and outcome rewards in a tree-search framework by combining code execution feedback with LLM self-criticism. It achieves a 26.9% accuracy improvement and a 42.2% efficiency boost in code generation without training a PRM.

## Background & Motivation

LLMs have made significant progress in code generation, but still struggle with complex programming tasks (e.g., dynamic programming, parallel computing optimization). Existing supervision paradigms suffer from two main bottlenecks:

**Outcome Supervision**: Only evaluates the final output, ignoring the reasoning process. A brute-force solution that passes all test cases with a complexity of $O(n^2)$ might be marked as "correct", failing to guide the model toward the optimal $O(n \log n)$ solution.

**Process Supervision**: Relies on a trained Process Reward Model (PRM), which has three major issues: (a) requires expensive human annotation; (b) the PRM itself is prone to hallucination or reward hacking; (c) PRMs specifically for code are extremely scarce, and PRMs in the mathematical domain do not apply to programming logic.

**Key Insight**: Code is naturally verifiable—it can be executed to obtain objective feedback. Existing execution feedback methods (such as Reflexion, LDB, Self-Repair) mainly focus on local code repair and lack global exploration capabilities over different algorithmic strategies. The core question posed by the authors is: **Is it still necessary to train a separate PRM when the LLM's intrinsic reasoning capability can be effectively guided by verifiable execution outcomes?**

## Method

### Overall Architecture

ORPS integrates reasoning, code implementation, and execution verification into a **tree-structured search process**. The core idea of the framework is to treat "reasoning about outcome refinement" itself as a process requiring supervision. Unlike step-by-step code repair methods, ORPS guides the LLM to think about the overall problem-solving strategy at a higher abstraction level.

During the search process, the state of each node is defined as a six-tuple $s_t = (\mathcal{R}_t, C_t, F_t, \omega_t, K_t, \rho_t)$, corresponding to: reasoning chain, code implementation, execution feedback, outcome reward score, self-critic reasoning, and process reward score.

The search operates in a loop across three phases: Candidate Generation → Execution & Profiling → Self-Critic & Process Rewarding. It retains the Top-K optimal paths using beam search.

### Key Designs

1. **Candidate Generation**: For each node in the search tree, the LLM generates N candidates based on the historical reasoning chain $\mathcal{R}_{t-1}$, code $C_{t-1}$, and execution feedback $F_{t-1}$. Each candidate contains an updated reasoning step $r_t^{(j)}$ (e.g., "adjusting the loop termination condition to avoid an off-by-one error") and the corresponding code $c_t^{(j)}$. This design ensures **algorithmic diversity**—allowing both greedy and dynamic programming solutions to be maintained simultaneously until execution feedback clearly indicates which is superior.

2. **Execution & Profiling Outcome Rewards**: Each candidate code is executed on unit tests, and performance metrics are collected. The outcome reward $\omega_t^{(j)} = \sum_{k=1}^{M} \beta_k \cdot \text{normalize}(m_k^{(j)})$ integrates **dynamic profile metrics** (correctness, execution time, CPU instructions, page faults) and **static analysis metrics** (code length, AST node count, cyclomatic complexity, cognitive complexity). This ensures that even if a brute-force implementation passes all tests, it is incentivized to optimize due to poorer complexity metrics.

3. **Self-Critic & Process Rewarding**: The same LLM acts as both the "programmer" and the "reviewer." After code execution, the model generates textual criticism $k_t^{(j)}$ and a numerical process reward $\rho_t^{(j)}$. This hybrid scoring mechanism prevents reward hacking—the model cannot inflate its reward without genuinely improving the verifiable execution outcome. A high $\rho$ and low $\omega$ signals a process-outcome mismatch, which is a major failure mode of traditional process supervision.

### Loss & Training

ORPS is a **pure inference-time framework** and requires no training. Beam search uses a weighted step score to select Top-K successor states:

$$q_t = \alpha \rho_t + \beta \omega_t, \quad \alpha + \beta = 1$$

where $\alpha = \beta = 0.5$ (default setting). This formulation unifies traditional supervision paradigms: $\beta = 0$ degenerates into pure process supervision; $\alpha = 0$ degenerates into outcome supervision (similar to Best-of-N on trees).

Search hyperparameters: search depth T=5, beam width K=3, expansion factor N=20. The number of LLM calls is $2 \times N \times (K \times T + 1)$.

**Bidirectional Feedback Synergy**: Execution results anchor the reasoning process by identifying discrepancies between expected and actual behavior (e.g., test failures revealing flawed base cases in recursive algorithms); process rewards guide exploration toward algorithmically superior implementations (e.g., recognizing that memoization can rewrite an $O(2^n)$ brute-force solution into an $O(n)$ dynamic programming solution).

## Key Experimental Results

### Main Results

Evaluation on 3 benchmarks (LBPP competition-level, HumanEval, MBPP) across 5 LLMs. Key representative results are selected below:

| Model/Method | LBPP Pass@1↑ | LBPP Time↓ | HumanEval Pass@1↑ | MBPP Pass@1↑ |
|---|---|---|---|---|
| Qwen-7B CoT | 40.1% | 118.6% | 72.6% | 79.0% |
| Qwen-7B Reflexion | 37.7% | 111.2% | 75.6% | 79.0% |
| Qwen-7B LDB (w/T) | 35.8% | 187.8% | 87.8% | 66.9% |
| Qwen-7B BoN | 53.1% | 117.9% | 77.4% | 82.9% |
| **Qwen-7B ORPS** | **59.9%** | **84.1%** | **79.9%** | 76.7% |
| **Qwen-7B ORPS (w/T)** | **77.8%** | **82.4%** | **96.3%** | **94.9%** |
| Qwen-14B CoT | 53.7% | 119.2% | 82.9% | 84.0% |
| **Qwen-14B ORPS (w/T)** | **85.8%** | **64.2%** | **97.0%** | **95.3%** |
| GPT-4o-Mini CoT | 50.0% | 124.5% | 79.9% | 78.6% |
| **GPT-4o-Mini ORPS (w/T)** | **88.9%** | **61.6%** | **97.6%** | **95.7%** |

**Key Findings**: Qwen-7B + ORPS (w/T) reaches 77.8% on LBPP, surpassing Qwen-14B CoT (53.7%), demonstrating that **reasoning space is more important than model scale**.

### Ablation Study

Ablation experiments on LBPP using Qwen-7B (Table 3):

| Configuration | Pass@1 | Tests% | Valid% | Time% | Description |
|------|--------|--------|--------|-------|------|
| ORPS (Full) | 59.9% | 75.7% | 92.0% | 84.1% | Baseline |
| − Execution Feedback | 43.8% | 56.4% | 72.8% | 200.5% | Pass@1 drops by 16.1%, proving execution feedback is crucial |
| − Reasoning Process | 55.6% | 74.5% | 94.4% | 124.5% | Pass@1 drops by 4.3%, reasoning is key to efficiency improvements |

**PRM Analysis** (Table 4, LBPP + Qwen-7B):

| Method | Pass@1 | Granularity | Requires Training |
|------|--------|------|--------|
| Trained PRM (Outcome-level) | 37.0% | Outcome | ✓ |
| Trained PRM (Line-level) | 32.1% | Line | ✓ |
| **ORPS (Test-time, Outcome-level)** | **59.9%** | Outcome | ✗ |
| ORPS (Test-time, Line-level) | 38.3% | Line | ✗ |

**Computation Cost Analysis** (Table 5, LBPP + Qwen-7B):

| Method | 20 calls | 50 calls | 100 calls |
|------|---------|---------|----------|
| Reflexion | 37.0% | 40.7% | 39.5% |
| LDB | 37.0% | 36.4% | 37.0% |
| REx | 43.2% | 53.7% | 54.3% |
| PRM-GPT | 44.4% | 37.0% | 35.8% |
| **ORPS** | **48.4%** | **55.6%** | **64.2%** |

ORPS continues to scale with computation budget, whereas REx encounters diminishing returns, and PRM even degrades.

### Key Findings

1. **Reasoning Space > Model Scale**: ORPS enables a smaller model (Qwen-7B) to outperform a larger model's CoT baseline (Qwen-14B) on LBPP.
2. **No PRM Needed**: Self-criticism combined with execution feedback performs better than trained PRMs (even those trained on high-quality human-annotated data).
3. **Global Strategy Exploration > Local Repair**: LDB on LBPP with Qwen-7B achieves only 35.8%, whereas ORPS reaches 77.8% (using the exact same test cases).
4. **Additional Validation on CodeContests** (Appendix): ORPS achieves a 20.61% Pass@1 under 100 calls, significantly outperforming REx (13.33%) and Reflexion (8.48%).
5. **Single Metric Optimization Trap**: Optimizing only a single metric (e.g., execution speed) leads to severe degradation across other dimensions, validating the necessity of joint multi-objective optimization.

## Highlights & Insights

- **Philosophical Elegance**: By treating "reasoning about outcome refinement" itself as a process to supervise, it elegantly unifies the two supervision paradigms as endpoints on a continuum.
- **Highly Practical**: A pure inference-time method that requires no PRM training, no extra annotated data, and is plug-and-play.
- **Deep Insight**: While LLMs cannot reliably self-correct (Huang et al. 2023), when self-criticism is anchored by verifiable execution outcomes, it can generate high-quality process rewards—"anchoring" is key.
- **Leveraging the Unique Advantage of Code**: The executability of code provides an objective verification signal that mathematical reasoning lacks, making the code domain particularly suited for such approaches.

## Limitations & Future Work

1. **Limited Performance on MBPP**: On simple tasks, ORPS shows marginal improvements or even slight disadvantages compared to BoN, indicating that complex searching may introduce unnecessary overhead for simple problems.
2. **Quality of Self-Generated Test Cases**: A massive gap exists between using gold test cases (w/T) and self-generated test cases (77.8% vs. 59.9% on LBPP), making test-case quality a performance bottleneck.
3. **Inference Cost**: The number of LLM calls is $2 \times N \times (K \times T + 1)$, which translates to approximately 320 calls per problem in the default configuration, making it expensive for large-scale deployment.
4. **Limited to the Code Domain**: The method relies heavily on execution verification, making it difficult to directly transfer to non-verifiable reasoning tasks.
5. **Unexplored Synthesis with RL Training**: The search trajectories generated by ORPS could serve as high-quality training data for RLHF/DPO, but this was left unimplemented in the paper.

## Related Work & Insights

- **Reflexion / LDB / Self-Repair**: Local code repair methods that fail to explore different algorithmic strategies.
- **Math-Shepherd / AlphaMath**: Process supervision methods in mathematics that rely on training a PRM.
- **REx**: Models code repair as an exploration-exploitation trade-off, which is most relevant to ORPS but only performs local exploration.
- **Test-time Scaling** (Snell et al. 2024): ORPS can be viewed as a strategy for efficiently utilizing test-time computation in the code domain.
- **Insight**: The ORPS framework could be extended to other domains with formal verifiers, such as theorem proving or hardware design.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of unifying process/outcome supervision is inspiring, though the combination of tree search and execution feedback is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated across 5 models, 3+1 benchmarks, detailed ablations, PRM training comparisons, computational cost control experiments, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous formulation, and well-organized experiments.
- Value: ⭐⭐⭐⭐ — Practically proves that test-time methods can outperform trained PRMs, offering valuable lessons to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](../../ACL2026/code_intelligence/recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ICML 2025\] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](adaptivestep_automatically_dividing_reasoning_step_through_model_confidence.md)
- [\[ICML 2025\] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [\[ICML 2025\] EpiCoder: Encompassing Diversity and Complexity in Code Generation](epicoder_encompassing_diversity_and_complexity_in_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](../../ACL2026/code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)

</div>

<!-- RELATED:END -->
