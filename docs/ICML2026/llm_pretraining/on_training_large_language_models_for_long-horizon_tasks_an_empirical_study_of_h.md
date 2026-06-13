---
title: >-
  [Paper Note] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length
description: >-
  [ICML 2026][LLM Pretraining][horizon reduction] Using a set of meticulously controlled Sudoku and Rush Hour tasks where reasoning difficulty remains constant while only the horizon length varies…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "horizon reduction"
  - "REINFORCE"
  - "macro action"
  - "subgoal"
  - "horizon generalization"
date: 2026-05-08
content_hash: e68f411efa16d86a
---

# On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length

**Conference**: ICML 2026  
**arXiv**: [2605.02572](https://arxiv.org/abs/2605.02572)  
**Code**: Not disclosed  
**Area**: LLM Agent / Reinforcement Learning / Long-Horizon  
**Keywords**: horizon reduction, REINFORCE, macro action, subgoal, horizon generalization

## TL;DR
Using a set of meticulously controlled Sudoku and Rush Hour tasks where reasoning difficulty remains constant while only the horizon length varies, this paper systematically proves that **task horizon itself is an independent root cause of LLM agent RL training collapse**. It proposes two horizon-reduction mechanisms—macro actions and subgoal decomposition—which not only stabilize training but also enable strong zero-shot generalization (horizon generalization) to even longer horizons.

## Background & Motivation

**Background**: Utilizing LLMs as agents has become mainstream (e.g., Claude Code, Codex for multi-step coding). Training methods primarily focus on context engineering at the system level and SFT/RL (such as GRPO, DAPO, GSPO, and other critic-free methods) at the model level.

**Limitations of Prior Work**: When the number of interaction steps required for a task increases from short (10–20 steps) to long (30+ steps), RL that stabilized on short tasks often suffers a **catastrophic collapse**—success rates plummet, and response lengths explode to the max-length. The community often attributes this to "task difficulty" or "reward sparsity," yet no prior work has specifically isolated whether the "horizon" itself is an independent bottleneck.

**Key Challenge**: In real-world tasks, horizon length is naturally coupled with reasoning complexity (e.g., harder Sudoku puzzles require both longer traces and more advanced techniques), making it impossible to determine if training failure stems from the "long horizon" or the "reasoning difficulty."

**Goal**: 1) Mechanically isolate horizon from other difficulty factors; 2) Quantitatively characterize RL training dynamics under the single variable of horizon; 3) Identify simple principles to stabilize long-horizon RL.

**Key Insight**: Based on the hypothesis that "if a model has problem-solving capabilities, it should solve the task if presented in a single-step format," the authors construct a **short-horizon proxy task** ("providing the entire Sudoku solution in one step") to filter instances. These are then categorized into seven horizon levels (L1–L7) based on the number of atomic actions $d(s_0, g)$, ensuring that "solving complexity" remains identical across different levels.

**Core Idea**: Horizon is an intrinsic task property rather than just an environment constraint. Actively reducing the effective horizon $h_\pi(s_0, g)$ (via macro actions or subgoals) is far more effective at curing long-horizon training collapse than designing complex RL algorithms, and it naturally leads to horizon generalization.

## Method

### Overall Architecture
The pipeline follows a three-stage "Diagnosis–Intervention–Verification" approach: (a) **Diagnosis**: Under constant solving complexity, Qwen3-1.7B is trained using SFT + REINFORCE-style RL on L1–L4 to observe training collapse as horizons lengthen; (b) **Intervention**: Two types of horizon reduction mechanisms are proposed—macro actions (multiple atomic actions per step) and subgoal decomposition (segmenting by subgoals with independent return calculation); (c) **Verification**: Robustness checks are performed on Rush Hour, WebShop, larger models (4B), and GRPO-style optimizers, alongside measuring horizon generalization on L5–L7.

### Key Designs

1. **Horizon Formalization and Controlled Experimental Design**:
    - **Function**: To cleanly isolate "horizon length" from all other difficulty factors.
    - **Mechanism**: The authors categorize horizon into three parts: target distance $d(s_0, g)$ (atomic actions to reach $g$ under optimal policy), interaction budget $H_{\max}$, and effective horizon $h_\pi(s_0, g)$ (actual steps taken by the policy), satisfying $d \le h_\pi \le H_{\max}$. For Sudoku, the number of empty cells serves as a proxy for $d$, and the HoDoKu solver ensures all training instances "only require basic techniques," fixing reasoning complexity. For Rush Hour, the minimum moves count defines $d$. A short-horizon proxy (one-step solution generation) is used to filter out instances the model cannot solve, ensuring remaining differences reside **strictly in the horizon**.
    - **Design Motivation**: This is the foundation of the paper. Without this dataset, the conclusion that "horizon is an independent bottleneck" would lack empirical support. Sudoku is ideal because LLMs possess significant prior knowledge; solving a short version implies the capability to solve longer ones.

2. **Critic-free off-policy REINFORCE with reward decoupling**:
    - **Function**: To provide a stable RL optimizer for long horizons while preventing step-level penalties from polluting trajectory-level signals.
    - **Mechanism**: The PPO critic is discarded (as value-based variance reduction is proven to decay in long horizons) in favor of REINFORCE with a baseline: $A_t = \hat r_t^{\text{traj}} + \alpha \hat r_t^{\text{step}}$, where $r^{\text{traj}}_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ is the trajectory return, and $r^{\text{step}}_t = r^{\text{format}}_t + r^{\text{valid}}_t$ penalizes parsing/invalid actions. Both are batch-normalized before weighted summation ($\alpha = 0.2$). For off-policy updates, a joint correction of MIS (masked importance sampling, based on geometric mean ratios) and TIS (truncated IS, based on sequence-level ratios) is used: $w_t = \mathbb{I}(C_{\text{low}}\le \rho_{\text{geo},t}\le C_{\text{high}}) \cdot \min(\rho_{\text{seq},t}, C)$.
    - **Design Motivation**: The authors reveal that a core mechanism of long-horizon RL failure is the **diffusivity of negative advantage**—penalizing a sample token causes probability to diffuse across the entire vocabulary. In long trajectories, errors at one or two steps can pollute all tokens. Reward decoupling prevents step penalties from degrading trajectory signals, and the dual-clip IS prevents off-policy drift.

3. **Horizon Reduction: macro actions and subgoal decomposition**:
    - **Function**: To physically compress the effective horizon $h_\pi(s_0, g)$ into a range where RL remains stable.
    - **Mechanism**:
      (a) **Macro action**: Allows the policy to output multiple atomic actions per step (e.g., filling multiple Sudoku cells or `move(id, direction, N)` in Rush Hour). The policy $\pi'$ is defined over the macro-action space such that $h_{\pi'}(s_0, g) \le h_\pi(s_0, g)$. The authors compare atomic vs. fixed-length (exactly $k$ steps) vs. flexible ($n\le 5$) designs, finding fixed-length performed worst due to overshooting, while flexible macro actions decided by the policy were optimal.
      (b) **Subgoal decomposition**: Decomposes the global goal $g$ into a sequence of independently verifiable subgoals $(g_1, \ldots, g_k)$ (e.g., Sudoku sub-grids). Segment-wise returns $G_t$ are calculated per subgoal, effectively transforming a long sparse-reward MDP into multiple short dense-reward MDPs.
    - **Design Motivation**: Instead of designing complex training algorithms, the goal is to "eliminate the problem." Since RL stability depends on $h_\pi$ rather than $d$, reducing $h_\pi$ is superior. This aligns with the slogan: "The best way to escape from a problem is to solve it." A crucial ablation showed that training with macro actions but **enforcing only 1 atomic action** still led to collapse, proving the benefit comes from horizon reduction rather than enhanced policy capacity.

### Loss & Training
The base model is Qwen3-1.7B (validated with 4B, Llama3, etc.). SFT is performed on expert trajectories generated by GPT-5-mini, followed by 4 epochs of RL using the aforementioned off-policy REINFORCE. Rollout/inference temperatures are set to 0.8; pass@K and avg@K are estimated using 4 trajectories per instance.

## Key Experimental Results

### Main Results

| Setting | Short horizon (L1-L2) | Long horizon (L3-L4) |
|------|--------------------|--------------------|
| Atomic action RL | Stable Gain | **Training Collapse**, max-length ratio surge |
| Macro action RL | Faster convergence, Higher | **Stable Gain**, no collapse |
| Subgoal decomposition | — | Stable high performance on L3-L4 where sparse-reward baseline fails to learn |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Macro-action policy but restricted to 1 atomic/turn | Still collapses | Horizon is the true bottleneck, not policy expressivity |
| Fixed-length macro ($k$ steps) | Poor performance, overshooting | Rigid constraints are harmful |
| Flexible macro ($n\le 5$ or unbounded) | Best | Policy autonomously determines action length |
| GRPO-style optimizer | Also collapses at long horizon; horizon reduction remains effective | Conclusions are optimizer-agnostic |
| WebShop / 4B Model | Universal stable gains from horizon reduction | Conclusions hold across environments and scales |

### Key Findings
- **Collapse is accompanied by a sharp rise in max-length response ratio**: The authors hypothesize that accumulated negative advantage pushes the policy toward incoherent long generations. This clear precursor to collapse can be used as an early stopping signal.
- **Horizon generalization**: Macro-action models trained only on L1-L2 significantly outperform the atomic baseline on unseen L5-L7 horizons, suggesting that horizon reduction learns **transferable** problem-solving patterns.
- **Horizon curriculum is effective**: A training order from short to long (curriculum) outperforms short-only or long-only training, further confirming that bringing $h_\pi$ into a stable RL range is critical.

## Highlights & Insights
- **Dataset design decoupling horizon and reasoning** is the most valuable methodological contribution—this paradigm of short-horizon proxy + solver-based grading can be directly applied to the training and evaluation of web and coding agents.
- The contrast between "complex methods vs. simple principles" is striking: While the RL community competes on PPO/GRPO/DAPO/CISPO variants, this work proves that as long as the effective horizon is reduced, basic REINFORCE is sufficient.
- The "negative-advantage diffusion causing incoherent generation" is a vital mechanistic insight for LLM RL practice, explaining why models often "derail" after extended training in many scenarios.
- **Horizon generalization** hints at a counter-intuitive upper bound: effective training on short horizons can grant the ability to generalize zero-shot to long horizons—potentially making large-scale long-horizon RL more feasible.

## Limitations & Future Work
- Experimental environments are primarily text-based games (Sudoku, Rush Hour, WebShop); the **effect of horizon reduction on open-ended coding/research agents is not directly verified**—defining macro actions in code editing remains an open question.
- Model scale is mostly 1.7B/4B; whether the collapse boundary of 14B+ models can be fully mitigated by horizon reduction remains to be verified.
- Subgoal decomposition requires "independently verifiable subgoals," which is natural for structured tasks like Sudoku but remains unsolved for chain-of-thought tasks like mathematical proofs or complex planning.
- The relationship between horizon reduction and search/planning methods (MCTS, Tree-of-Thought) was not discussed; theoretically, both could be combined.

## Related Work & Insights
- **vs. Shen et al. / Xi et al. / Bai et al. (horizon-based curriculum)**: They view horizon as an environmental constraint (interaction budget), whereas this paper views it as an intrinsic task property, providing a deeper "why" level analysis.
- **vs. CALM (wang-etal-2025)** and work treating "long-horizon RL = sophisticated algorithms": This paper takes the opposite route, using structural task modifications instead of algorithmic stacking, proving simpler and more robust.
- **vs. Park et al. (mapping complexity scaling)**: This work empirically validates their theoretical predictions regarding the non-linear growth of state-action complexity and provides a practical remedy.
- **vs. Sinha et al. (step-accuracy analysis)**: Agrees that long-horizon requires exponential step accuracy but proposes the inverse strategy: "Don't just improve step accuracy; decrease the number of steps."

## Rating
- Novelty: ⭐⭐⭐⭐ Dataset design isolating horizon as a single variable is truly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robustness verified across environments/scales/optimizers, though lacking real-world code agent verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear argumentation; the "Diagnosis–Intervention–Verification" structure is easy to follow.
- Value: ⭐⭐⭐⭐⭐ Highly practical guidance for practitioners training long-horizon LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](../../ACL2026/llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)

</div>

<!-- RELATED:END -->
