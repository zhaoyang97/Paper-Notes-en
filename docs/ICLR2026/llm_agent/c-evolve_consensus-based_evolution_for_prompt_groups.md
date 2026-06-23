---
title: >-
  [Paper Note] C-Evolve: Consensus-based Evolution for Prompt Groups
description: >-
  [ICLR 2026][LLM Agent][Paper Note] C-Evolve shifts from "evolving a single optimal prompt" to "evolving a group of complementary prompts." By using a **voting score**—measuring a prompt's contribution to group consensus—as evolutionary fitness, the method enables multiple prompts to reach a consensus, thereby breaking the performance ceiling of individu
tags:
  - ICLR 2026
  - LLM Agent
date: 2026-05-08
content_hash: 8e7b1aa46b253844
---
# C-Evolve: Consensus-based Evolution for Prompt Groups

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iyxRqYiCbF](https://openreview.net/forum?id=iyxRqYiCbF)  
**Code**: To be confirmed  
**Area**: LLM Agent / Prompt Optimization / Evolutionary Algorithms  
**Keywords**: Prompt evolution, consensus aggregation, majority voting, island model, compound AI system, black-box optimization  

## TL;DR
C-Evolve shifts from "evolving a single optimal prompt" to "evolving a group of complementary prompts." By using a **voting score**—measuring a prompt's contribution to group consensus—as evolutionary fitness, the method enables multiple prompts to reach a consensus, thereby breaking the performance ceiling of individual prompts.

## Background & Motivation
**Background**: Closed-source large models like GPT-4.1 and Claude are accessible only via API, making weight fine-tuning impossible. Consequently, black-box prompt optimization has become mainstream. Among these, evolutionary methods (e.g., GEPA using natural language reflection + Pareto selection, and AlphaEvolve using island models to evolve programs) are the most powerful. These methods iteratively mutate and select within a population to identify a **single** prompt with the highest fitness as the global optimum.

**Limitations of Prior Work**: A single optimal prompt inherently suffers from an expression bottleneck. Complex tasks often involve multifaceted requirements that a single prompt struggles to satisfy, leading to inevitable failures in certain cases. Machine learning has long demonstrated that ensembles outperform single classifiers, and techniques like self-consistency and multi-agent systems have verified that "votes from multiple experts surpass a single expert." However, **systematic research on how to aggregate outputs from multiple prompts to reach a consensus in the field of prompt optimization is scarce**.

**Key Challenge**: Simply assembling several individually "optimal" prompts post-hoc into a voting group yields poor results. Since they were evolved based on "individual scores," they are highly homogeneous and tend to commit similar errors, failing to complement each other. Experimental results confirm that directly grouping the best prompts from the three islands of AlphaEvolve for voting results in an accuracy of 41.15%, showing no improvement. The issue lies in the **mismatched evolutionary objective**: to achieve strong group voting, the evolution process should reward prompts that are "good at cooperating" rather than those that are merely "strong individually."

**Goal**: Find the optimal **prompt group** $G^* = \arg\max_G \mathbb{E}_{(x,m)\sim\tau}[\mu(C(y_G), m)]$, where $C$ is a consensus aggregator, ensuring that the aggregated output of the group (rather than any single individual) is optimal for the task.

**Core Idea** (Voting Score-Driven Evolution): A **voting score** is utilized to measure the average contribution of each prompt across all groups it participates in. This serves as the evolutionary fitness, ensuring that prompts "more likely to form high-scoring groups" are retained for reproduction while "underperformers" are eliminated. This naturally directs the evolution toward high-quality consensus.

## Method

### Overall Architecture
C-Evolve builds upon the **island evolution model** of AlphaEvolve: it initializes $|G|$ islands (default is 3), which evolve in parallel to maintain population diversity. Finally, one prompt is selected from each island to form a voting group. The process consists of two stages: a **warm-up phase** which uses individual scores to strengthen each prompt, followed by a **voting phase** which switches to the group consensus-based voting score as fitness to encourage complementarity. Evolution is performed by an evolver LLM that reads selected prompts and execution feedback to refine new individuals. During inference, individuals with the highest EMA voting scores from each island form the final group for consensus reasoning.

```mermaid
flowchart TD
    A[Initialize |G|=3 islands<br/>Max capacity N_max=10 per island] --> B[Warm-up Phase<br/>Fitness = Individual Score s_ind]
    B --> C[Start Voting Phase<br/>Initialize s_EMA = s_ind from end of warm-up]
    C --> D[Sample 1 individual per island → evolver LLM<br/>Evolve new individual with group feedback]
    D --> E[Cross-island sampling of n_c=10 groups<br/>1 person per island, score via consensus aggregator]
    E --> F[Calculate voting score Eq. 3<br/>→ Update s_EMA via Eq. 4]
    F --> G[Eliminate individual with lowest s_EMA in each island]
    G --> H{Iteration end?}
    H -- No --> D
    H -- Yes --> I[Select individual with highest s_EMA from each island<br/>Form final group, consensus inference]
```

### Key Designs

**1. Voting score: Replacing "individual strength" with "team contribution" fitness.** This is the core distinction between C-Evolve and all single-prompt evolution methods. A naive approach would be to eliminate all members of the "worst-performing group." However, since group members overlap significantly, the worst group might include strong members who perform well elsewhere. Instead, the paper measures the **average consensus performance of each individual across all groups it belongs to**:
$$s_{\Pi,\text{voting}} = \frac{\sum_{k=1}^{n_c} \mathbb{I}(\Pi \in G_k)\cdot \mathbb{E}_{(x,m)\sim D_{met}}\big[\mu(C(y_{G_k}), m)\big]}{\sum_{k=1}^{n_c}\mathbb{I}(\Pi \in G_k)}$$
where $\mathbb{I}(\Pi\in G_k)$ indicates whether individual $\Pi$ belongs to group $G_k$. A higher voting score suggests that groups including this individual are more likely to reach a correct consensus. Treating this as fitness naturally rewards prompts that "cooperate well"—this is the fundamental reason why C-Evolve (43.88% in Table 2) outperforms the post-hoc grouping of AlphaEvolve (41.15%), even if the individual scores of C-Evolve (40.47/39.79/38.77) are lower than AlphaEvolve's 41.15.

**2. EMA Smoothing: Reflecting the "current population" state.** Voting scores fluctuate significantly across iterations as the population changes. Simply averaging an individual's historical group performances would include data from groups involving long-eliminated members, failing to reflect the individual's actual contribution to the **current population**. The paper employs an Exponential Moving Average (EMA) for updates:
$$s_{\Pi,\text{EMA}} \leftarrow \alpha\cdot s_{\Pi,\text{EMA}} + (1-\alpha)\cdot s_{\Pi,\text{voting}}$$
$\alpha\in[0,1]$ balances history and current performance, giving higher weight to recent performance. At the start of the voting phase, $s_{\Pi,\text{EMA}}$ is initialized with individual scores from the end of the warm-up phase. Ablations show that simple averaging achieves only 41.16%, while EMA with $\alpha=0.8$ peaks at 42.85%.

**3. Consensus Aggregator: Divide and conquer for closed/open tasks.** Each prompt in the group produces an output $y = \Phi(x;\Pi)$, which is then integrated by aggregator $C$ into $y_{final}=C(y_G)$. For closed tasks (multiple choice, math), **majority voting** is used. For open tasks (free text), an **LLM-based aggregator** identifies the "most representative answer covering the most content from other answers." Ablation studies comparing two LLM aggregation strategies show that LLM-summary (generating a new answer) reaches only 38.66%, while **LLM-selection (selecting the most representative existing answer)** reaches 42.85%, as selection preserves the faithfulness of original outputs.

**4. Dual Dataset + Group Feedback Evolution.** Evolution uses $D_{met}$ (with metric $\mu$) for scoring and an independent $D_{feed}$ for generating detailed execution feedback for the evolver. In the warm-up phase, feedback includes inputs/outputs and scores for individual prompts. In the voting phase, this is upgraded to **group feedback**—adding group membership, individual outputs, reached consensus, and the consensus score on $D_{met}$. The evolver LLM uses this to diagnose "which module failed and when," refining new individuals with better cooperative capabilities. A 10% migration rate between islands is maintained to prevent local optima.

## Key Experimental Results

### Main Results
Comparison across 2 open tasks (HotpotQA, IFBench) and 3 closed tasks (HoVer, MATH, GPQA) using open-source Qwen3-8B and closed-source GPT-4.1-mini:

| Model | Method | HotpotQA | IFBench | Hover | MATH | GPQA | Avg. Gain |
|------|------|----------|---------|-------|------|------|----------|
| Qwen3-8B | Baseline | 50.03 | 31.29 | 37.66 | 67.66 | 41.43 | - |
| | GEPA | 65.72 | 34.01 | 43.33 | - | - | +8.02 |
| | AlphaEvolve | 65.31 | 41.15 | 44.66 | 82.66 | 43.08 | +9.75 |
| | **Ours** | **70.67** | **43.88** | **50.33** | **85.33** | **47.15** | **+13.85** |
| GPT-4.1-mini | Baseline | 44.24 | 40.13 | 42 | 78.66 | 46.34 | - |
| | GEPA | 68.39 | 46.59 | 39 | - | - | +9.2 |
| | AlphaEvolve | 67.31 | 45.24 | 50.33 | 92.66 | 63.01 | +13.42 |
| | **Ours** | **70.64** | **47.96** | **51.66** | **95.33** | **66.26** | **+16.09** |

C-Evolve achieves SOTA on all tasks: exceeding the baseline by +13.85%, GEPA by +5.83%, and AlphaEvolve by +4.1% on Qwen3-8B. On the stronger GPT-4.1-mini, it remains superior to GEPA (+6.89%) and AlphaEvolve (+2.67%).

### Ablation Study

| Ablation | Setting | Accuracy (%) |
|--------|------|-----------|
| LLM Aggregator | LLM-summary | 38.66 |
| | **LLM-selection** | **42.85** |
| Voting Score Smoothing | Group Average | 41.16 |
| | EMA, α=0.5 | 42.51 |
| | **EMA, α=0.8** | **42.85** |
| | EMA, α=0.95 | 41.50 |

(Conducted on IFBench / Qwen3-8B, with 50 warm-up rounds + 50 voting rounds)

### Key Findings
- **Consensus provides maximum gain for difficult problems**: Stratifying MATH by difficulty (Table 3), individual prompts fall below 67% at Level 5, while C-Evolve reaches 92.00/82.66/68.33% for Levels 3–5 respectively, effectively solving "hard cases" that single prompts cannot.
- **Multiple islands evolve differentiated roles**: On IFBench, the three islands evolve different focuses—Island 1 focuses on "core constraint priority + refinement," Island 2 on "constraint pre-declaration + layered verification," and Island 3 on "task priority layering + critical failure point validation."
- **Evolutionary Curve Comparison**: While AlphaEvolve rises quickly but saturates early, C-Evolve continues to climb after entering the voting phase, indicating that consensus-driven fitness provides new optimization space after single-prompt evolution is exhausted.

## Highlights & Insights
- **Redefining the Evolutionary Goal**: The shift from "finding a single optimal solution" to "finding the optimal complementary group" systematically injects ensemble learning wisdom into prompt evolution. The key insight is that team contribution—not individual score—must be the fitness metric for evolution to avoid producing homogeneous prompts with correlated errors.
- **Combination of Voting Score + EMA**: The voting score resolves the problem of attributing group performance to individuals, while EMA ensures that the evolutionary signal focuses on the current population state.
- **Islands as Diversity Engines**: Parallel isolated evolution naturally breeds prompts with different specializations, fulfilling the prerequisite that "complementarity is necessary for consensus gain."

## Limitations & Future Work
- **Doubled Inference Cost**: Since the group contains $|G|$ prompts, each query is run $|G|$ times plus an additional LLM aggregation for open tasks. The cost-benefit tradeoff is not fully explored.
- **Fixed Island Count**: $|G|=3$ is an empirical setting. Whether more islands provide sustained gains or hit diminishing returns, and how the optimal group size adapts to tasks, remains unanalyzed.
- **Dependency on Evolver and Aggregator Capabilities**: Evolution and consensus quality are constrained by the underlying LLM. In weaker models, voting score feedback may be noisier.
- **AlphaEvolve Replication**: Comparisons are made against a self-implemented version of AlphaEvolve (as the original is not open-source), which warrants caution regarding fairness.

## Related Work & Insights
- **Evolutionary Prompt Optimization**: GEPA and AlphaEvolve focus on single optimal solutions, which C-Evolve argues limits generalization. Complex or dynamic tasks prevent a single solution from exploring synergies between prompts.
- **LLM Consensus**: Techniques like self-consistency and multi-agent systems reflect the "committee exceeds the expert" principle, but their consensus is usually a post-hoc mechanism. C-Evolve makes consensus an **intrinsic driver of evolution**.
- **Insight**: Any pipeline that "optimizes parts first and ensembles later" should be reconsidered. If the ensemble is the goal, the optimization target should directly align with ensemble performance. This concept of using group contribution as fitness is transferable to model ensembles, retriever combinations, and multi-agent role evolution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Pathbreaking systematic integration of ensemble/consensus ideas into prompt evolution; the objective shift from "single optimal" to "optimal combination" is a significant conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 2 models and 5 tasks (open/closed); strong baselines. Stratified difficulty analysis and differentiations are well-visualized. Deductions for missing cost-benefit analysis of island counts.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivational progression; technical formulas and diagrams are well-placed. Table 2 provides a compelling contrast.
- **Value**: ⭐⭐⭐⭐ — Prompt optimization is a necessity in the era of black-box models. Consensus-driven evolution opens new optimization paths beyond single prompts with high transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)
- [\[ICLR 2026\] AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning](alphaagentevo_evolution-oriented_alpha_mining_via_self-evolving_agentic_reinforc.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ICML 2026\] EvoClaw: Evaluating AI Agents on Continuous Software Evolution](../../ICML2026/llm_agent/evoclaw_evaluating_ai_agents_on_continuous_software_evolution.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
