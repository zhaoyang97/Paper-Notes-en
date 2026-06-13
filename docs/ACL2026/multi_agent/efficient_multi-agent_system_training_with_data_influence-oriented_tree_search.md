---
title: >-
  [Paper Note] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search
description: >-
  [ACL 2026][Multi-Agent][Multi-agent systems] Ours proposes DITS, utilizing "training data influence score" instead of traditional Q-value as the guide for MCTS tree search and preference data selection. It derives a "for…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Multi-agent systems"
  - "Influence functions"
  - "Monte Carlo Tree Search"
  - "DPO"
  - "Self-training"
date: 2026-05-08
content_hash: ef481eb805d0a0a5
---

# Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search

**Conference**: ACL 2026  
**arXiv**: [2502.00955](https://arxiv.org/abs/2502.00955)  
**Code**: https://github.com/swt-user/DITS  
**Area**: Multi-Agent / LLM Training / MCTS / Data Synthesis  
**Keywords**: Multi-agent systems, Influence functions, Monte Carlo Tree Search, DPO, Self-training  

## TL;DR
Ours proposes DITS, utilizing "training data influence score" instead of traditional Q-value as the guide for MCTS tree search and preference data selection. It derives a "forward-only inference" influence score estimation formula for non-differentiable metrics, enabling Multi-Agent Systems (MAS) to achieve an average improvement of 2.5–2.7% over Optima-iSFT-DPO across 7 datasets and 3 multi-agent tasks.

## Background & Motivation

**Background**: LLM Multi-Agent Systems (MAS, such as MetaGPT, AutoGen, and Camel) decompose complex tasks into collaborating agents, which is currently a mainstream path for breaking the limits of single-agent capabilities. The predominant method to optimize such MAS is "MCTS synthesize trajectories $\rightarrow$ extract preference pairs $\rightarrow$ DPO training," represented by work like Optima.

**Limitations of Prior Work**: The core signal of MCTS, the Q-value, is directly borrowed from the inference phase—it measures "whether this node can win," whereas MAS training requires "whether this data can improve the model." The scatter plot in Figure 2(a) demonstrates that samples with high Q-values correlate weakly with high actual training gains. Data selected by Q-value ordering does not necessarily lead to maximum downstream improvement.

**Key Challenge**: The misalignment between the MCTS guide (Q-value) and the training objective (validation improvement). Additionally, the correlation between DPO loss and downstream performance is $< 0.2$, causing traditional influence functions that use DPO loss to estimate influence to fail in this context.

**Goal**: (1) Identify a data score truly aligned with training gains; (2) Integrate this score into both MCTS selection and final preference pair selection; (3) Enable this score to be calculated on large models at a reasonable cost.

**Key Insight**: Classic influence functions (Koh & Liang, 2017) measure how much the training loss changes when a data point is removed. The authors redefine "training loss" as "non-differentiable metrics $\mathcal{F}$ (F1/EM) on the validation set" to circumvent the decoupling of DPO loss and performance; they then replace the second-order Hessian with pure forward inference using "one-step gradient descent + finite difference."

**Core Idea**: Replace Q-value with an influence score oriented toward non-differentiable validation metrics $\mathcal{I}_{\mathcal{F}_{\text{val}}}$ as the data quality signal for MAS self-training, transforming MCTS into an "influence-oriented" tree search.

## Method

### Overall Architecture
DITS decomposes a single training round into three steps: (1) Perform MCTS rollouts using the current MAS, expanding agent calls on a directed graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ in topological order to obtain a pool of synthetic trajectories with Q-values; (2) Calculate the influence score $\mathcal{I}$ for each candidate preference pair $z=(s, a^h, a^l)$, and select the Top-$\alpha$ into the DPO training set $\mathcal{D}_{\text{tr}}$ based on the ranking $H(z_i)=\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}},\theta)+\gamma\cdot Q(s,a_i^h)$; (3) Train $\theta_t$ using $\mathcal{D}_{\text{tr}}$ and return to Step 1 for the next iteration. The entire pipeline is an influence-oriented version of "iSFT-DPO," termed DITS-iSFT-DPO.

### Key Designs

1.  **Influence-Oriented MCTS Node Expansion**:
    -   **Function**: In the selection phase, it no longer relies solely on Q-values but prioritizes expanding nodes more likely to produce preference pairs with "high training gains."
    -   **Mechanism**: Candidate nodes $N_{\text{cand}}$ are first filtered by an edit-distance similarity threshold $S_{i,j}\geq 0.25$ to remove branches highly redundant with expanded nodes, then sampled based on $n\sim\text{Softmax}(\{Q(n)\}_{n\in N_{\text{cand}}})$. Final preference pairs consist of the two actions $(a_i^h, a_i^l)$ with the highest and lowest Q-values under the current state.
    -   **Design Motivation**: Standard MCTS selection serves the inference phase to "find the correct answer," thus tending to repeatedly deepen high Q-value subtrees, leading to diversity collapse. Incorporating influence scores makes selection more dispersed, focusing on what "teaches the model" rather than just "answering correctly."

2.  **Influence Score Estimation on Non-Differentiable Metrics**:
    -   **Function**: Use forward inference to replace the Hessian/higher-order gradients, making influence score calculation feasible for 8B-scale LLMs.
    -   **Mechanism**: Define $\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}}):=\frac{\mathcal{F}_{\text{val}}(z_i,\theta_{\epsilon,z_i}^{*})-\mathcal{F}_{\text{val}}(z_i,\theta^{*})}{\epsilon}$, then approximate the perturbed optimal parameters via one-step gradient descent $\theta_{\epsilon,z_i}^{*}\approx\theta^{*}-\eta\epsilon\nabla_{\theta}\mathcal{L}_{\text{tr}}(z_i,\theta^{*})$. Thus, $\mathcal{I}\approx\frac{1}{\epsilon}[\mathcal{F}_{\text{val}}(z_i,\theta^{*}-\eta\epsilon\nabla_{\theta}L_{\text{tr}}(z_i,\theta^{*}))-\mathcal{F}_{\text{val}}(z_i,\theta^{*})]$. This requires only one LoRA one-step update per candidate data point plus one forward pass on the validation set, without second-order gradients.
    -   **Design Motivation**: Since DPO loss correlation with downstream metrics is $< 0.2$, traditional loss-based influence functions are ineffective. Switching to perturbation response of F1/EM bypasses the "loss does not represent capability" issue while replacing Hessian-vector products with cheap forward passes.

3.  **Joint Influence-Q Scoring + Iterative Self-Training**:
    -   **Function**: Treat Q-value as a "rationality prior" and the influence score as the "true signal of training gain," selecting Top-$\alpha$ data by weighting both; continuous model and synthetic data updates are performed via multiple iterations.
    -   **Mechanism**: Combined score $H(z_i)=\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}},\theta)+\gamma\cdot Q(s,a_i^h)$, where $\gamma=0$ is pure influence and $\gamma=1$ is equal weighting. Round $t$ uses $\theta_{t-1}$ for MCTS to synthesize $\mathcal{D}_{\text{tr}}^{t}$ and retrains $\theta_t$ from the initial SFT model.
    -   **Design Motivation**: Influence scores can be noisy (e.g., F1), while Q-values provide unreliable training gains; the two are complementary. Iterative updates create a positive feedback loop: "stronger model $\rightarrow$ higher quality synthetic data $\rightarrow$ even stronger model."

### Loss & Training
The training objective is standard DPO:  
$$\mathcal{L}_{DPO}=\mathbb{E}_{z}[-\log\sigma(\beta[\log\frac{\pi_{\theta}(a_i^h\mid s)}{\pi_{\text{ref}}(a_i^h\mid s)}-\log\frac{\pi_{\theta}(a_i^l\mid s)}{\pi_{\text{ref}}(a_i^l\mid s)}])]$$
Llama-3-8B-Instruct is used for static scenarios and QwQ-32B for dynamic scenarios. MCTS uses depth $d=3$ and $k=8$ repetitions. Validation set size $V=20$. Default $\alpha=0.5$ and $\gamma=1$. LoRA is used for the one-step gradient descent during influence estimation to save VRAM.

## Key Experimental Results

### Main Results
On 6 datasets covering Information Exchange (HotpotQA / 2WMH QA / TriviaQA / CBT) and Debate (ARC-C / MMLU), DITS-iSFT-DPO consistently outperforms Optima-iSFT-DPO:

| Dataset | Metric | DITS-iSFT-DPO | Optima-iSFT-DPO | Gain |
| :--- | :--- | :--- | :--- | :--- |
| HotpotQA | F1 | 57.2 | 55.6 | +1.6 |
| 2WMH QA | F1 | 76.0 | 74.2 | +1.8 |
| TriviaQA | F1 | 78.4 | 77.1 | +1.3 |
| CBT | F1 | 72.0 | 70.1 | +1.9 |
| ARC-C | Acc | 77.6 | 77.1 | +0.5 |
| MMLU | Acc | 60.5 | 60.2 | +0.3 |
| WebWalker (DeepSearch, QwQ-32B) | Acc | 47.2 | 46.6 (Optima-DPO) | +0.6 |

The gain in the DeepSearch task is particularly significant: under the QwQ-32B + WebThinker framework, DITS-DPO improves from Optima-DPO's 46.6 to 47.2 in one training round, proving the influence score remains effective at the 32B scale and with dynamic agent topologies.

### Ablation Study
Comparison of different data selection strategies in a single iteration (baseline is Optima-DPO, full set training):

| Configuration | HotpotQA | 2WMH QA | TriviaQA | CBT | ARC-C | MMLU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Optima-DPO (Full) | 46.6 | 61.2 | 70.9 | 57.2 | 71.5 | 51.6 |
| Random Select (50%) | 51.5 | 60.6 | 70.3 | 58.0 | 74.0 | 51.1 |
| Q-value Select (50%) | 50.5 | 61.1 | 69.8 | 58.6 | 73.7 | 50.2 |
| DITS-DPO ($\gamma=0$) | **53.1** | **62.2** | **72.2** | **59.6** | 74.2 | 50.8 |
| DITS-DPO ($\gamma=1$) | 52.8 | 61.5 | 71.0 | 59.1 | **74.5** | **52.3** |

Q-value Select actually performs worse than Random Select, strongly supporting the claim that "Q-values are misaligned with training gains."

Cost-side (2WMH QA): DITS-DPO achieves F1=0.612 using $2.0\times 10^{7}$ tokens, 8,500 samples, and 106 GPU·h. Optima-DPO reaches only F1=0.610 even when scaled to $3.34\times 10^{7}$ tokens, 34,000 samples, and 195 GPU·h.

### Key Findings
- "Scaling the budget for influence score estimation" yields higher returns than "scaling for Q-value estimation": Figure 2(b) shows that for the same token budget, curves allocating more compute to influence estimation are superior, suggesting synthesis-time scaling should prioritize influence estimation.
- More iterations result in a higher mean and lower variance of influence scores in synthetic data, confirming the success of the positive feedback loop. However, decreasing variance might indicate reduced data diversity.
- The optimal value of $\gamma$ depends on the task: Information Exchange (high F1 noise) prefers $\gamma=0$ (pure influence); Debate (stable EM) performs better with $\gamma=1$ (fusing Q-value). The noise level of the metric determines the reliability of the Q-value signal.
- A larger validation set $V$ improves performance, but the cost increases linearly; $V=20$ is the empirical cost-benefit balance point.
- Setting the selection ratio $\alpha$ too low causes performance to drop, indicating that "pure quality" cannot replace sufficient sample volume—both quality and quantity are essential.

## Highlights & Insights
- Changing "data influence" from "influence on loss" to "influence on non-differentiable validation metrics" resolves the issue of DPO loss decoupling from performance. This is a critical adaptation for influence functions in the LLM era.
- "Replacing Hessian-vector products with one-step gradient descent + finite difference" reduces influence estimation to one forward pass and one LoRA update. This engineering trick makes influence functions practical for 8B+ models and is applicable to RLHF/DPO data filtering.
- The counter-intuitive finding that "Q-value Select is worse than Random" is a powerful discovery, demonstrating the "MCTS signal $\neq$ training signal" principle through simple experimentation.
- A new dimension for synthesis-time scaling: while 90% of budget increases are traditionally spent on rollout quantity, Ours proves that investing in "influence estimation" is more cost-effective than "accurate Q-values," providing a new direction for data synthesis research.

## Limitations & Future Work
- Authors acknowledge DITS is an offline/training-time filtering mechanism; the cost of multiple inferences per candidate is unacceptable for strict latency scenarios or online evaluation.
- The current work validates only "static topology + fixed agent number" and limited dynamic (WebThinker) scenarios, without covering emergent team formation or dynamic agent spawning.
- One-step gradient descent might have high error under sharp losses like DPO; while experimentally effective, theoretical guarantees are limited to strong convexity assumptions.
- Validation set size $V$ is a manual hyperparameter; online adaptive adjustment of $V$ was not explored, which might be a bottleneck for high-noise tasks (e.g., open-ended generation).
- Future directions: Lightweight influence estimation via single-pass inference, end-to-end learnable influence models, and benchmarks for open-ended collaboration.

## Related Work & Insights
- **vs Optima (Chen 2024b)**: Both use MCTS + DPO for data synthesis, but Optima ranks by Q-value. DITS proves Q-value misalignment and achieves significantly higher sample efficiency using joint influence + Q-value ranking.
- **vs Classic Influence Function (Koh & Liang 2017)**: Traditional IF relies on Hessian-vector products and training loss, which are infeasible for LLMs. DITS adapts the target to non-differentiable metrics and the solution to one-step finite differences for practical application.
- **vs LESS / Data Selection for SFT**: Most existing methods focus on SFT loss influence. DITS is the first influence-oriented synthesis method specifically for DPO/preference pairs and MAS multi-agent topologies.
- **vs MCTS-based Inference (rStar / o1-like)**: Those works use MCTS to find answers during inference. DITS repurposes MCTS to "find good data during training," highlighting the fundamental difference between search-for-data and search-for-answer, providing a key insight for self-improvement/self-play frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐ Adapting influence functions to DPO + MAS + non-differentiable metrics is novel, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 datasets, 3 multi-agent tasks, static/dynamic topologies, 8B/32B scales, and thorough scans of budget/iteration/validation size.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation; the scatter plot in Figure 2(a) is highly persuasive. Math derivations are dense but supported by the appendix.
- Value: ⭐⭐⭐⭐ Opens a new scaling dimension for MAS self-training and DPO synthesis; methods are transferable to RLHF and Self-Play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](../../ICLR2026/multi_agent/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](../../AAAI2026/multi_agent/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)

</div>

<!-- RELATED:END -->
