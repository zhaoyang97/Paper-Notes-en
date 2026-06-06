---
title: >-
  [Paper Note] R2-Router: A New Paradigm for LLM Routing with Reasoning
description: >-
  [ICML 2026][LLM Reasoning][LLM Routing] This paper proposes R2-Router, which transforms "output token budget" from a passive estimate into a controllable variable. By allowing the router to search in a joint (LLM…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "LLM Routing"
  - "Output Length Budget"
  - "Quality-Cost Curve"
  - "Reasoning Routing"
  - "Length-constrained Prompting"
date: 2026-05-08
content_hash: b3aef8cd0d7b8063
---

# R2-Router: A New Paradigm for LLM Routing with Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.02823](https://arxiv.org/abs/2602.02823)  
**Code**: https://github.com/UCF-ML-Research/R2-Router (Available)  
**Area**: LLM Reasoning / LLM Routing / Inference-time Computation  
**Keywords**: LLM Routing, Output Length Budget, Quality-Cost Curve, Reasoning Routing, Length-constrained Prompting

## TL;DR
This paper proposes R2-Router, which transforms "output token budget" from a passive estimate into a controllable variable. By allowing the router to search in a joint (LLM, budget) space, it extends each LLM from a static point into a quality-cost curve using a lightweight multi-head quality predictor. This achieves comparable quality to existing routers at 4–5× lower costs.

## Background & Motivation
**Background**: With the explosion of LLMs, selecting the right model for a specific query has become a system-level challenge. The mainstream approach is LLM Routing: using a small model to predict the quality $Q$ and cost $C$ of candidate LLMs, then selecting the highest score $S=(1-\lambda)Q-\lambda C$. Representative works include FrugalGPT / AutoMix (cascading), CARROT, MIRT, and UniRouter (predictive).

**Limitations of Prior Work**: Existing routers treat each LLM as a static "quality-cost point." If a strong model's (e.g., Qwen3-235B) predicted cost exceeds the budget, the router excludes it, missing the opportunity to obtain high-quality responses even with shorter outputs. The essence of the problem is that **LLM quality varies with output length**, but this curve is absent in existing routing frameworks.

**Key Challenge**: The search space of routers is locked in $\mathcal{S}_{reactive}=\{(M_i,\hat{c}_i)\}$, whereas the true optimal solution likely resides in the larger Cartesian product $\mathcal{S}_{reasoning}=\{(M_i,b_j)\mid M_i\in\mathcal{M},b_j\in\mathcal{B}\}$. The former is a proper subset of the latter, but passive routing lacks the capability to explore the latter.

**Goal**: (i) Redefine the routing problem by putting output length budget $b$ on equal footing with LLM selection; (ii) provide a lightweight, data-efficient predictor implementation; (iii) construct a dataset that allows the mechanism to "see" quality-cost curves for training and evaluation.

**Key Insight**: Leveraging empirical findings in efficient reasoning—LLM response quality grows with output length but saturates quickly, and output length can be stably controlled (demonstrated in Appendix A) via length-constrained prompts like "use at most $k$ tokens." This makes $b$ a truly controllable knob.

**Core Idea**: Upgrade routing from "routing on points" to "routing on curves"—predict the quality curve for each LLM across multiple budgets, select the combination that maximizes $S$ in the joint (LLM, budget) space, and pass the budget constraint to the selected LLM via prompting.

## Method

### Overall Architecture
The R2-Router pipeline consists of offline and online stages:

- **Offline (R2-Bench Construction + Training)**: For each (query, LLM) pair, responses are sampled at 16 different token budgets (10 / 20 / 30 / ... / 4000 / default). Qwen3-80B-Instruct serves as the LLM judge to provide 0–1 quality scores, yielding training data for quality-cost curves $\{(b_k, Q_{i,k})\}_{k=1}^K$.
- **Online (Routing + Execution)**: When a query $x$ and trade-off coefficient $\lambda$ are input, a shared encoder generates the query embedding $z_x=\text{Enc}(x)$. A multi-head MLP simultaneously predicts the quality $\hat{Q}(x,M_i,b_k)$ for all (LLM, budget) pairs, forming an $n\times K$ curve matrix. The Decision Maker performs $\arg\max$ on this matrix to obtain $(M^*,b^*)$. Finally, $M^*$ is called with the constraint "Use at most $b^*$ tokens." injected into the prompt.

### Key Designs

1. **Reformulating Output Length as a Decision Variable**:
    - **Function**: Rewrites traditional routing $\arg\max_{M_i}\,(1-\lambda)\hat{Q}_i-\lambda\hat{C}_i$ as $(M^*,b^*)=\arg\max_{M\in\mathcal{M},\,b\in\mathcal{B}}\bigl((1-\lambda)Q(x,M,b)-\lambda C(b)\bigr)$, where $C(b)$ is an analytical function of unit price multiplied by budget $b$.
    - **Mechanism**: Implements $b$ at execution time using length-constrained prompts (Lee et al. 2025), making both predicted $\hat{Q}(x,M,b)$ and actual cost $C(b)$ deterministically controllable. Theorem 4.3 formally proves that because $\mathcal{S}_{reactive}\subseteq\mathcal{S}_{reasoning}$, the maximum utility of reasoning-based routing is never inferior to reactive routing.
    - **Design Motivation**: This is the "leverage point" of the paper. Once $b$ becomes a controllable variable, strong models are no longer vetoed due to "high predicted costs"—instead, they become "usable at an appropriate budget." This bypasses the "point-based" bottleneck of existing routers and is orthogonally compatible with various upper-level methods (KNN/MLP/GNN).

2. **Multi-head Quality Predictor + Sparse Anchors + Linear Interpolation**:
    - **Function**: Assigns $K$ independent heads to each LLM $M_i$. Each head $g_{i,k}$ is a three-layer MLP ([256, 128, 64] hidden layers + ReLU + Sigmoid) outputting $\hat{Q}(x,M_i,b_k)=\sigma(g_{i,k}(z_x))$. Training occurs only on $K$ anchor budgets, with intermediate budgets handled via piecewise linear interpolation: $\hat{Q}(x,M,b')=(1-\alpha)\hat{Q}(x,M,b_k)+\alpha\hat{Q}(x,M,b_{k+1})$, where $\alpha=(b'-b_k)/(b_{k+1}-b_k)$.
    - **Mechanism**: The shared encoder captures general query semantics, while independent heads capture specific behavior for "a certain LLM at a certain budget." Optimization uses MSE loss $\mathcal{L}_{i,k}=\mathrm{MSE}(\hat{Q}(x,M_i,b_k),Q^{\text{true}}(x,M_i,b_k))$. Empirical results show $K=6{\sim}8$ anchors are sufficient to approximate continuous curves; even $K=4$ outperforms point-based baselines.
    - **Design Motivation**: Training a head for every continuous budget would require massive data. Sparse anchors + interpolation find the best balance between data efficiency and search granularity. Full training for 15 LLMs takes 30 minutes on a single RTX 3090, with routing overhead <400 ms, accounting for less than 1% of total generation time.

3. **R2-Bench: Multi-Budget Routing Dataset**:
    - **Function**: Integrates 30,968 queries from 6 public benchmarks (GPQA / MuSR / MMLU-Pro / MATH / OpenHermes / RAGBench) across 15 LLMs and 16 budgets. Each triplet (query, LLM, budget) records the quality score from Qwen3-80B-Instruct and actual token consumption.
    - **Mechanism**: Existing benchmarks like RouterBench / SPROUT / RouterEval only sample one response per (query, LLM), making curve learning physically impossible. R2-Bench systematically varies output budgets. The LLM-as-a-judge selection followed Zheng et al. 2023's protocol—choosing Qwen3-80B-Instruct ($\rho=0.82$) as the final judge based on Pearson correlation with human annotators.
    - **Design Motivation**: R2-Bench and R2-Router are complementary—the former exposes the larger (LLM, budget) optimization space (increasing Oracle AUDC from 0.85 to 0.98), while the latter provides the mechanism to search it.

### Loss & Training
Independent MSE regression for each (LLM $i$, budget anchor $k$): $\theta_{i,k}^*=\arg\min_{\theta_{i,k}}\mathcal{L}_{i,k}$. Adam optimizer, $1\times 10^{-4}$ learning rate, 100 epochs. Qwen3-Embedding-0.6B encodes queries into 1024-dimensional vectors. All heads share the encoder output but have independent parameters to facilitate incremental updates for new LLMs.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | R2-Router | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| R2-Bench (Main) | Cost for quality ≈ 0.8 | $0.5\times 10^{-3}$ | $2{\sim}2.5\times 10^{-3}$ (MIRT / CARROT) | 4–5× cheaper |
| MMLU-Pro OOD (Non-STEM) | AUDC ↑ | 0.71 | 0.67 (CARROT-L) | +0.04 |
| MMLU-Pro OOD | QNC ↓ | 0.26 | 0.56 (CARROT-L) | −54% |
| Uni-R2Router vs UniRouter (5 new LLMs) | AUDC ↑ | 0.623 | 0.590 | +5.6% |
| Uni-R2Router vs UniRouter | QNC ↓ | — | — | −80% |
| RouterArena Leaderboard | Rank | 1 (at time of submission) | — | — |

### Ablation Study

| Configuration | AUDC ↑ | QNC ↓ | Peak Acc ↑ | Note |
|------|--------|-------|------------|------|
| Default (Qwen3-Embed + MLP head + Qwen3 judge) | ≈0.80 | ≈0.12 | ≈0.83 | Full model |
| MiniLM-L6-v2 embedding | 0.76 | 0.32 | 0.79 | Small encoder still beats point-based |
| LGBM as prediction head | 0.80 | 0.29 | 0.81 | Architecture independent; gains from curve search |
| DeepSeek-V3.1 as judge | 0.80 | 0.35 | 0.90 | Stable lead with different judges |
| Anchor count $K=4$ | — | ≈0.20 | — | Beats MIRT(0.43) / CARROT-L(0.32) |
| Anchor count $K=6{\sim}8$ | — | ≈0.12 | — | Near optimal |
| "Be concise" prompt for reactive baseline | < R2-Router | — | — | Prompt changes output but not selection logic |

### Key Findings
- **The leverage is on the router side, not the LLM side**: Simply adding "Be concise" prompts to baselines does not solve the problem—reactive routers still exclude strong models based on static cost estimates. "Curve perspective" must be introduced at the decision layer.
- **Sparse anchors are sufficient**: $K=6{\sim}8$ budget anchors with linear interpolation approximate the continuous curve optimum well, proving that quality-cost curves are smooth in the query embedding space.
- **Orthogonal and Additive**: Integration with UniRouter (Uni-R2Router) showed a 5% AUDC increase and 80% QNC decrease with 5 unseen LLMs, showing "curvification" is a general enhancement for KNN/IRT/UniRouter frameworks.
- **Dataset Oracle upper bound increased by 15%**: R2-Bench pushed Oracle AUDC from 0.85 to 0.98, suggesting existing benchmarks measure an artificially compressed problem.

## Highlights & Insights
- **Reformulating "Passive Estimation" as "Active Control"**: While current routers passively estimate $\hat{C}_i$, R2-Router recognizes that budget is just a prompt instruction and promotes it to a decision variable. This approach can transfer to many system + ML scenarios.
- **Theorem 4.3 provides clean theoretical guarantees**: $\mathcal{S}_{reactive}\subseteq\mathcal{S}_{reasoning}$ makes "R2-Router will not perform worse" an algebraic fact rather than just an empirical observation, which is rare in the engineering-heavy routing field.
- **Effective "Routing as Reasoning" branding**: The authors analogize this to models dynamically deciding "thinking depth," but here "reasoning" is the router's deliberation in (LLM, budget) space.
- **Reusable Trick**: Using a shared encoder with a family of small MLP heads to model "different behaviors of the same object under different conditions" is a general pattern applicable to multi-objective or multi-configuration prediction tasks.

## Limitations & Future Work
- **Reliance on instruction following for length constraints**: Theoretical guarantees depend on "use at most $k$ tokens" being strictly followed; this assumption may weaken for smaller models.
- **Quality supervision from LLM judge**: 7.4M samples were scored solely by Qwen3-80B-Instruct, potentially carrying judge bias.
- **Discrete anchor + linear interpolation granularity**: If quality-cost curves exhibit non-linear "steps" (e.g., reasoning models requiring a minimum threshold of tokens), linear interpolation might overestimate intermediate quality. Monotonic neural networks or spline predictors could be improvements.
- **Latency and Concurrency**: Cost currently only considers token fees, though latency and throughput are often harder constraints in industrial deployment.

## Related Work & Insights
- **vs CARROT / MIRT / NIRT**: These predict $\hat{Q}_i$ and $\hat{T}_i$ or fix $C_i$ (point-based). R2-Router turns cost into a controllable variable, strictly containing their search space.
- **vs UniRouter**: UniRouter addresses "dynamic LLM pools" using validation error vectors. R2-Router is orthogonal—Uni-R2Router extends error vectors across multiple costs into curve representations.
- **vs Route-To-Reason / BEST-Route**: Route-To-Reason routes to (LLM, strategy) pairs without explicit cost control. BEST-Route increases sampling but raises costs. R2-Router reduces costs by modifying the prompt.
- **vs Semantic Router / Think When Needed**: These select internal inference modes; R2-Router selects (Model, Budget) across multiple LLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefining output length from an estimate to a decision variable is a paradigm shift for LLM routing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage with 15 LLMs, 16 budgets, and 30k queries, including OOD, dynamic pools, and leaderboard evidence.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear "Points vs Curves" narrative and solid theoretical framing.
- **Value**: ⭐⭐⭐⭐⭐ 4–5× cost reduction with 30-minute training and orthogonal compatibility makes this a practical contribution for commercial routing platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Reasoning Paradigm for Named Entity Recognition](../../AAAI2026/llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[AAAI 2026\] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](../../AAAI2026/llm_reasoning/intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
