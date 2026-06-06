---
title: >-
  [Paper Note] MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings
description: >-
  [ACL 2026][LLM Efficiency][Multi-turn LLM Routing] MTRouter models the decision of "which LLM to call at each turn" in multi-turn agents as a step-by-step routing problem under cost constraints. By using history-model jo…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Multi-turn LLM Routing"
  - "Cost-aware Inference"
  - "History-Model Joint Embeddings"
  - "Offline Trajectory Learning"
  - "Tool-use Agents"
date: 2026-05-08
content_hash: 42bc9e8f2d373dd6
---

# MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings

**Conference**: ACL 2026  
**arXiv**: [2604.23530](https://arxiv.org/abs/2604.23530)  
**Code**: https://github.com/ZhangYiqun018/MTRouter  
**Area**: LLM Routing / Agents / NLP  
**Keywords**: Multi-turn LLM Routing, Cost-aware Inference, History-Model Joint Embeddings, Offline Trajectory Learning, Tool-use Agents

## TL;DR
MTRouter models the decision of "which LLM to call at each turn" in multi-turn agents as a step-by-step routing problem under cost constraints. By using history-model joint embeddings to predict the contribution of candidate models to final task outcomes, it improves task performance on ScienceWorld and HLE while significantly reducing total invocation costs.

## Background & Motivation
**Background**: LLMs are evolving from single-turn Q&A toward multi-turn, long-horizon agent tasks involving tool use, such as scientific environment interaction, complex retrieval reasoning, and code/web operations. These tasks require models to continuously observe environments, plan next steps, invoke tools, correct errors, and submit answers.

**Limitations of Prior Work**: Using high-capability models like GPT-5 or Claude Opus throughout an episode ensures higher success rates but leads to rapidly accumulating costs as multi-turn contexts grow. Conversely, using cheap models for the entire trajectory may suffice for routine tool calls but often fails during critical planning or error-recovery turns. Single-turn routing methods typically select one model at the start of an episode and fix it for the entire trajectory, failing to adapt to stage-specific requirements (e.g., initial planning vs. mid-term exploration vs. late-stage verification).

**Key Challenge**: The difficulty in multi-turn routing is not just evaluating current input complexity, but judging "whether selecting a specific model under the current historical state will affect the final outcome." A local formatting error or ineffective action might be corrected later or might derail the entire episode. If a router only performs reactive upgrades based on current errors, it may frequently switch models—disrupting caches and increasing costs—without necessarily improving the final success rate.

**Goal**: The paper aims to select candidate models turn-by-turn to maximize the final task score or accuracy, given a fixed cost budget per episode and a maximum number of turns. This involves three tasks: representing the current interaction history, representing the cost and capability features of different models, and learning a lightweight router that predicts final payoffs from offline trajectories.

**Key Insight**: Supervision signals for multi-turn routing naturally exist in historical trajectories: every episode has a final score, and intermediate events like formatting errors, tool failures, or invalid actions can be detected. Instead of relying on a large LLM to make ad-hoc judgment calls via prompting, one can learn the mapping from "historical state + candidate model" to final outcomes from these logs.

**Core Idea**: Learn a terminal outcome estimator using history-model joint embeddings to select the model with the highest predicted final payoff at each turn, rather than relying on a fixed model or reactive rules.

## Method
MTRouter acts as a "scheduler" wrapper around the agent. The agent continues to output actions or tool calls as required; MTRouter does not modify the task logic but decides which LLM to assign to the current turn based on the history and candidate model pool.

### Overall Architecture
An episode consists of multiple interaction turns. At turn `t`, the router observes history `h_t`, selects model `a_t`, which generates output `y_t`. A parser converts this into an executable action `u_t`, and the environment returns observation `o_{t+1}`. The episode ends when the task is completed, maximum turns are reached, or the budget is exhausted, yielding a final score `S_final`.

During training, offline trajectories are collected: some from a random router and some from single-model runs. Each trajectory is decomposed into "history-model-outcome" training samples. The system learns a scalar function `\hat{s}_\theta(h_t, a)`, representing the expected contribution to the final score if candidate model `a` is chosen given history `h_t`.

During inference, MTRouter encodes the current history once, concatenates it with each candidate model's embedding, scores all candidates in batch, and selects `argmax_a \hat{s}_\theta(h_t, a)`. The process is constrained by per-episode cost limits and maximum steps.

### Key Designs
1. **History-Model Joint Representation**:

    - **Function**: Enables routing decisions to depend simultaneously on the current state and model characteristics, rather than just the query or price.
    - **Mechanism**: History `h_t` consists of task descriptions, prior actions, and observations. Real-time implementation keeps the task block and the most recent context within an 8192-token budget. A frozen Qwen3-Embedding-0.6B encoder maps this to a 1024-dimensional vector `z_x`. Each model is encoded via structural attributes (context length, knowledge cutoff, pricing) and learnable residuals that capture behaviors unexplained by metadata. These are projected into a model vector `z_a`. The joint vector `[z_x; z_a]` serves as the input for the estimator.
    - **Design Motivation**: The same history has different implications for different models. Cheap models may handle formatting but fail at complex reasoning. The joint representation allows the estimator to learn "which historical states suit which models."

2. **Terminal Outcome Estimator with Error-Aware Target**:

    - **Function**: Learns the impact of turn-level model selection on final task results.
    - **Mechanism**: The estimator is a lightweight MLP outputting a scalar `\hat{s}_\theta(h_t, a)`. The target is not an immediate reward but the final score adjusted by an error penalty from the current turn to the end: `\tilde{S}_t = S_final - \sum_{i=t}^{T-1}\rho_i`. The penalty `\rho_i` is determined by error occurrence, severity, and a progress weight that increases over time.
    - **Design Motivation**: Multi-turn environments often lack dense rewards. While the final score is direct, it is often too coarse. Adding annealed error penalties allows the model to distinguish "early recoverable errors" from "late destructive errors," avoiding overreaction to every local mistake.

3. **Cost-Aware Greedy Routing**:

    - **Function**: Allocates expensive models to high-value turns and cheap models to low-risk or proficient tasks within a fixed budget.
    - **Mechanism**: No explicit cost penalty is added to the training objective, as wasted high-cost calls or ineffective turns already negatively impact the final score or result in budget exhaustion. During inference, the model with the highest predicted score is chosen per turn until the budget or step limit is reached.
    - **Design Motivation**: This is more stable than "upgrade after error." The router learns implicit rules, such as using GPT-5 for initial planning in ScienceWorld and switching to GPT-OSS for routine query commands later.

### Loss & Training
Data comes from two sources: random routing (for coverage) and single-model trajectories (for behavioral anchors). Combined, these include 1,291 instances, 29,693 trajectories, and 515,221 turns, costing approximately $1,620 to collect.

The loss function is Mean Squared Error (MSE): for turn `t` in trajectory `k`, the estimator is trained on `(h_t^{(k)}, a_t^{(k)})` with the target `y_t^{(k)} = \tilde{S}_t^{(k)}`, minimizing `\sum_{k,t}(\hat{s}_\theta(h_t^{(k)}, a_t^{(k)}) - y_t^{(k)})^2`. AdamW is used with a learning rate of `1e-3`, weight decay of 0.01, and cosine annealing over 100 epochs (patience=3). Residual embeddings use `L2` regularization to prevent overfitting to specific model IDs.

The candidate pool includes 6 models covering a 20x price range: GPT-5, DeepSeek-V3.2, MiniMax-M2, Kimi-K2, Gemini-2.5-Flash-Lite, and GPT-OSS-120B. Step limits are 50 for ScienceWorld and 30 for HLE, with a $2 budget per episode.

## Key Experimental Results

### Main Results
Evaluations are conducted on ScienceWorld (text-based scientific environment, score range `[-100, 100]`) and Humanity's Last Exam (HLE, multi-tool reasoning, accuracy metric). OOD splits use distinct task types or subjects.

| Dataset / Split | Metric | MTRouter | GPT-5 | Router-R1 | Change vs. GPT-5 |
|--------|------|------|----------|----------|------|
| ScienceWorld Test | Score / Cost | 53.8 / $5.7 | 48.4 / $13.9 | 42.1 / $12.6 | +5.4 score, -58.7% cost |
| ScienceWorld OOD | Score / Cost | 9.9 / $16.3 | 4.9 / $47.6 | 2.1 / $21.0 | +5.0 score, -65.8% cost |
| HLE Test | Acc / Cost | 26.0% / $35.0 | 25.1% / $61.8 | 24.2% / $51.9 | +0.9 pts, -43.4% cost |
| HLE OOD | Acc / Cost | 38.6% / $31.2 | 34.8% / $65.3 | 35.1% / $60.7 | +3.8 pts, -52.3% cost |

MTRouter does not simply trade performance for cost. In ScienceWorld Test, it outperforms GPT-5 at less than half the cost. In HLE, it maintains comparable or better accuracy while significantly reducing total expenditure.

### Ablation Study

| Configuration | ScienceWorld Score | HLE Acc. | Description |
|------|---------|---------|------|
| MTRouter (Full) | 53.8 ± 3.2 | 26.0 ± 2.3 | Joint embeddings, MLP estimator, random data, error penalties enabled |
| Ridge instead of MLP | 49.1 ± 3.5 | 23.4 ± 2.2 | Linear models lack expressiveness; gains are not just from manual features |
| No Random-Router Data | 47.2 ± 4.1 | 22.6 ± 2.1 | Lack of turn-level coverage hinders learning cross-model preferences |
| No Error Penalty | 48.5 ± 3.6 | 23.8 ± 2.1 | Pure final scores provide coarse supervision; error recovery learning is weakened |
| No History | 44.6 ± 3.8 | 21.3 ± 2.0 | Router fails to utilize long-horizon context |
| Hardcoded Model Encoder | 41.3 ± 4.4 | 19.7 ± 2.1 | Fixed attributes fail to capture actual behavioral differences |

### Key Findings
- **Efficiency in Switching**: MTRouter succeeds with fewer model switches than Router-R1 (approx. 5 vs. 20 in ScienceWorld).
- **Resilience to Errors**: After an error, MTRouter maintains the current model ~90.2% (ScienceWorld) and ~80.9% (HLE) of the time—higher than Router-R1—suggesting it learns which errors are "recoverable" rather than blindly upgrading.
- **Model Specialization**: In HLE, DeepSeek is over-represented for `search`, GPT-5 for `python`, and Kimi for `browse`. In ScienceWorld, MiniMax specializes in observation actions while GPT-OSS handles query commands.
- **Latent Structure**: t-SNE analysis of learned model embeddings shows clear separation by identity and price hierarchy, indicating the encoder successfully captures capability-price relationships.

## Highlights & Insights
- Shifting LLM routing from "query-level selection" to "turn-level selection" in long trajectories better reflects real-world agent deployment where costs and errors accumulate.
- The error-aware outcome target is highly practical. It avoids the complexity of dense reward design while providing more learnable signals than sparse final scores.
- Combining structured attributes with learnable residuals allows the router to understand both metadata (price/context) and empirical reliability for specific tools.
- Behavioral analysis shows that cost reduction isn't derived from "buying cheap," but from avoiding low-value switches and leveraging model specialization.

## Limitations & Future Work
- **Data Collection Cost**: Collecting offline trajectories is expensive ($1,620 for ~1.3k instances). Frequent changes in model pools or toolsets would necessitate costly updates.
- **Lack of Online Adaptation**: MTRouter is static post-training. It cannot adjust to new tool errors or model version updates in real-time.
- **Caching Overhead**: While switching is minimized, cross-model transitions still lose prompt/KV caching benefits, which may impact latency and real-world billing.
- **Greedy Strategy**: The router chooses the best predicted outcome for the current turn without explicit long-term planning of future routing sequences or budget management.

## Related Work & Insights
- **vs FrugalGPT**: FrugalGPT uses cascade-based upgrades for single-turn Q&A; MTRouter focuses on the long-range impact of decisions across an agent's multi-turn trajectory.
- **vs Router-R1**: Router-R1 utilizes RL and LLM-based routers, which are heavier and more prone to reactive switching. MTRouter's lightweight estimator is more stable and cost-efficient.
- **vs Tool-use Frameworks**: Unlike ReAct or ToolLLM which focus on *how* to use tools, MTRouter operates as an external cost-control layer deciding *which* model should perform the use.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Formalizes multi-turn routing as history-model joint outcome estimation, moving beyond single-turn paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across OOD splits, budget sensitivity, and behavioral specialization.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure; dense information handled well, though some LaTeX crowding occurs in HTML versions.
- **Value**: ⭐⭐⭐⭐⭐ Highly applicable for engineering real-world LLM systems to balance performance and multi-turn operational costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](../../ICML2026/llm_efficiency/not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ICLR 2026\] Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](../../ICLR2026/llm_efficiency/did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)

</div>

<!-- RELATED:END -->
