---
title: >-
  [Paper Note] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length
description: >-
  [ICML 2026][LLM Pretraining][horizon reduction] This paper uses a carefully controlled set of Sudoku/Rush Hour tasks—where "reasoning difficulty is fixed and only horizon length varies"—to systematically demonstrate that…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "horizon reduction"
  - "REINFORCE"
  - "macro action"
  - "subgoal"
  - "horizon generalization"
date: 2026-05-08
content_hash: d12b71f7429be547
---

# On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length

**Conference**: ICML 2026  
**arXiv**: [2605.02572](https://arxiv.org/abs/2605.02572)  
**Code**: Not released  
**Area**: LLM Agent / Reinforcement Learning / Long-Horizon  
**Keywords**: horizon reduction, REINFORCE, macro action, subgoal, horizon generalization

## TL;DR
This paper uses a carefully controlled set of Sudoku/Rush Hour tasks—where "reasoning difficulty is fixed and only horizon length varies"—to systematically demonstrate that **task horizon itself is an independent root cause of LLM agent RL training collapse**. It proposes two horizon-reduction mechanisms, macro action and subgoal decomposition, which not only stabilize training but also enable strong zero-shot generalization to longer horizons (horizon generalization).

## Background & Motivation

**Background**: Treating LLMs as agents has become mainstream (e.g., Claude Code, Codex for multi-step code agents), with training methods focusing on system-level context engineering and model-level SFT/RL (GRPO, DAPO, GSPO, and other critic-free methods).

**Limitations of Prior Work**: When the required number of interaction steps increases from short (10–20 steps) to long (30+ steps), RL methods that previously improved steadily on short tasks suddenly **collapse catastrophically**—success rates plummet and response lengths explode to max-length. The community often blames "task difficulty" or "reward sparsity," but no one has dissected whether "horizon" is an independent bottleneck.

**Key Challenge**: In real-world tasks, horizon length is naturally coupled with reasoning complexity (harder Sudoku requires both longer traces and more advanced techniques), making it impossible to determine whether training failure is due to "long horizon" or "reasoning difficulty."

**Goal**: 1) Mechanically isolate horizon from other sources of difficulty; 2) Quantitatively characterize RL training dynamics under horizon as a single variable; 3) Identify simple principles to stabilize long-horizon RL.

**Key Insight**: The authors use a key assumption—"if the model has problem-solving ability, it should solve the task when cast as a single-step problem"—to construct a **short-horizon proxy task** ("give the entire Sudoku answer in one step") for instance selection. They then segment tasks into seven horizon levels L1–L7 based on the number of atomic actions $d(s_0, g)$, ensuring that "solution complexity" is constant across levels.

**Core Idea**: Horizon is not an environment constraint but an intrinsic property of the task; actively reducing the effective horizon $h_\pi(s_0, g)$ (via macro action or subgoal) is far more effective at curing long-horizon training collapse than designing complex RL algorithms, and naturally leads to horizon generalization.

## Method

### Overall Architecture
The pipeline follows a "diagnosis–intervention–validation" three-stage process: (a) **Diagnosis**: Under controlled solution complexity, apply SFT + REINFORCE-style RL to Qwen3-1.7B on L1–L4 and observe how training curves collapse as horizon increases; (b) **Intervention**: Propose two horizon reduction mechanisms—macro actions (executing multiple atomic actions per step) and subgoal decomposition (segmenting by subgoals and computing rewards independently); (c) **Validation**: Conduct robustness checks on Rush Hour, WebShop, larger models (4B), and GRPO-style optimizers, and test horizon generalization on L5–L7.

### Key Designs

1. **Horizon Formalization and Controlled Experimental Design**:

    - **Function**: Cleanly disentangle "horizon length" from all other sources of difficulty.
    - **Mechanism**: The authors decompose horizon into three parts—goal distance $d(s_0, g)$ (number of atomic actions to reach $g$ under the optimal policy), interaction budget $H_{\max}$, and effective horizon $h_\pi(s_0, g)$ (actual steps taken by the policy), with $d \le h_\pi \le H_{\max}$. For Sudoku, the number of blanks proxies $d$, and the HoDoKu solver ensures all training instances "require only basic techniques," fixing reasoning complexity; for Rush Hour, min_moves is used as $d$. The short-horizon proxy (model gives the full answer in one step) filters out instances the model cannot solve, so the remaining instances **differ only in horizon**.
    - **Design Motivation**: This dataset is the foundation of the paper—without it, the conclusion that "horizon is an independent bottleneck" would not hold. Sudoku is ideal because LLMs have strong priors; being able to solve the short version implies problem-solving ability.

2. **Critic-free Off-policy REINFORCE and Reward Decoupling**:

    - **Function**: Provide a stable RL optimizer for long horizons, while preventing step-level penalties from contaminating trajectory-level learning signals.
    - **Mechanism**: Abandon PPO's critic (value-based variance reduction is known to decay at long horizons), reverting to REINFORCE with baseline: $A_t = \hat r_t^{\text{traj}} + \alpha \hat r_t^{\text{step}}$, where $r^{\text{traj}}_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ is the trajectory return, and $r^{\text{step}}_t = r^{\text{format}}_t + r^{\text{valid}}_t$ penalizes parsing/invalid actions. Both are batch-normalized and weighted ($\alpha = 0.2$). Off-policy correction uses MIS (masked importance sampling, geometric mean ratio clipping) × TIS (truncated IS, sequence-level ratio clipping): $w_t = \mathbb{I}(C_{\text{low}}\le \rho_{\text{geo},t}\le C_{\text{high}}) \cdot \min(\rho_{\text{seq},t}, C)$.
    - **Design Motivation**: The authors reveal that a core mechanism of long-horizon RL failure is **negative advantage diffusion**—when a sampled token is penalized, its probability diffuses across the entire vocab, so a few errors in a long trajectory contaminate all tokens; reward decoupling prevents step penalties from corrupting trajectory signals, and double clipping in IS prevents off-policy drift from compounding the problem.

3. **Horizon Reduction: Macro Actions and Subgoal Decomposition**:

    - **Function**: Physically compress the effective horizon $h_\pi(s_0, g)$ into a region where RL remains stable.
    - **Mechanism**:
      (a) **Macro action**: The policy outputs multiple atomic actions at once (e.g., filling multiple cells in Sudoku, `move(id, direction, N)` for multi-cell moves in Rush Hour), defining $\pi'$ over the macro action space so that $h_{\pi'}(s_0, g) \le h_\pi(s_0, g)$. The authors compare atomic, fixed-length (exactly $k$ steps), and flexible ($n\le 5$) designs, finding that fixed-length performs worst due to overshooting, while flexible—letting the policy decide action length—is optimal.
      (b) **Subgoal decomposition**: Decompose the global goal $g$ into a sequence of independently verifiable subgoals $(g_1, \ldots, g_k)$ (Sudoku's subgrid accuracy is naturally suitable), and compute independent segment-wise $G_t$ for each subgoal segment. This is equivalent to splitting a long sparse-reward MDP into multiple short dense-reward MDPs.
    - **Design Motivation**: Rather than designing complex training algorithms, "eliminate the problem"—RL stability depends mainly on $h_\pi$ rather than $d$, so directly reduce $h_\pi$. This echoes the paper's slogan: "The best way to escape from a problem is to solve it." A key ablation is performed: training with macro actions but **forcing only one atomic action per step**—collapse still occurs, proving that macro actions' real contribution is horizon reduction, not improved base policy exploration.

### Loss & Training

Base model is Qwen3-1.7B (with 4B, Llama3, etc. for validation). SFT is performed on expert trajectories generated by GPT-5-mini, followed by 4 epochs of off-policy REINFORCE as described above; rollout/inference temperature is 0.8, and pass@K / avg@K are estimated with 4 trajectories per instance.

## Key Experimental Results

### Main Results

| Setting | Short horizon (L1-L2) | Long horizon (L3-L4) |
|---------|-----------------------|----------------------|
| Atomic action RL | Stable improvement | **Training collapse**, max-length ratio surges |
| Macro action RL | Faster, higher convergence | **Stable improvement**, no collapse |
| Subgoal decomposition | — | On L3-L4, where sparse-reward baseline fails to learn, achieves strong stable performance |

### Ablation Study

| Configuration | Key Phenomenon | Notes |
|---------------|---------------|-------|
| Macro-action policy but restricted to 1 atomic/turn | Still collapses | Horizon is the true bottleneck, not policy expressiveness |
| Fixed-length macro ($k$ steps) | Poor performance, overshooting | Rigid constraints are harmful |
| Flexible macro ($n\le5$ or unbounded) | Best | Policy autonomously decides action length |
| GRPO-style optimizer | Also collapses at long horizon, horizon reduction still effective | Conclusion is optimizer-agnostic |
| WebShop / 4B model | Horizon reduction consistently stabilizes training | Conclusion holds across environments and scales |

### Key Findings
- **Collapse is accompanied by a sharp rise in max-length response ratio**: The authors hypothesize this is due to cumulative negative advantage pushing the policy toward incoherent long generations; this is a clean precursor to collapse and can serve as an early stopping signal.
- **Horizon generalization**: Macro-action models trained only on L1-L2 significantly outperform atomic baselines on unseen long horizons (L5–L7), indicating that horizon reduction not only stabilizes training but also enables **cross-horizon transferable** problem-solving patterns.
- **Horizon curriculum is effective**: Training in a short-to-long sequence (curriculum) outperforms short-only or long-only, further confirming that "first reduce $h_\pi$ to a region where RL is stable" is key.

## Highlights & Insights
- **Dataset design that decouples horizon and reasoning** is the most methodologically valuable contribution—the short-horizon proxy + solver-based stratification paradigm can be directly transferred to web agent and coding agent training/evaluation.
- The contrast between "complex methods vs. simple principles" is elegantly demonstrated: while the RL community focuses on PPO/GRPO/DAPO/CISPO, the authors show that simply reducing effective horizon allows even basic REINFORCE to be stable.
- The insight that "negative-advantage diffusion causes incoherent generation" is highly practical for LLM RL, explaining why "the longer you train, the messier it gets" in many scenarios.
- **Horizon generalization** provides a counterintuitive upper bound: effective training on short horizons can yield zero-shot generalization to long horizons—potentially making large-scale long-horizon RL training much more tractable.

## Limitations & Future Work
- Experiments are mainly on text-based games (Sudoku, Rush Hour, WebShop); **the effect of horizon reduction on real open-ended coding/research agents is not directly validated**—how to define macro actions in code editing remains an open question.
- Model sizes are mainly 1.7B/4B; whether horizon reduction can fully suppress collapse boundaries for 14B+ models remains unverified.
- Subgoal decomposition requires "independently verifiable subgoals," which is natural for structured tasks like Sudoku, but how to automatically find subgoals for reasoning-chain tasks (math proofs, complex planning) is unsolved.
- The relationship between horizon reduction and search/planning methods (MCTS, tree-of-thought) is not discussed; theoretically, the two could be combined.

## Related Work & Insights
- **vs Shen et al. / Xi et al. / Bai et al. (horizon-based curriculum)**: They treat horizon as an environment constraint (interaction budget), while this paper treats it as an intrinsic task property, offering deeper analysis closer to the "why" level.
- **vs CALM (wang-etal-2025)** and other "long-horizon RL = better algorithm design" works: This paper takes the opposite approach, using structured task modification instead of algorithmic complexity—simple and robust.
- **vs Park et al. (mapping complexity scaling)**: This paper empirically validates their theoretical prediction of "nonlinear growth in state-action complexity," and provides a practical remedy.
- **vs Sinha et al. on high-step accuracy analysis**: Agrees that long-horizon requires exponential step accuracy, but proposes the reverse: "don't increase step accuracy, reduce the number of steps."

## Rating
- Novelty: ⭐⭐⭐⭐ The single-variable horizon-isolated dataset design is truly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robustness is validated across environments, model scales, and optimizers, but lacks real code agent validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation, "diagnosis–intervention–validation" structure is easy to follow.
- Value: ⭐⭐⭐⭐⭐ Immediately actionable guidance for practitioners training long-horizon LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](../../ACL2025/llm_pretraining/nemotron_cc_pretraining_data.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](../../ACL2025/llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)

</div>

<!-- RELATED:END -->
