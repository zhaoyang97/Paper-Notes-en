---
title: >-
  [Paper Note] Making Expert Reasoning Learnable with Self-Distillation
description: >-
  [ICML 2026][Reinforcement Learning][Expert Trajectories] DAIL utilizes a mixed-policy rollout sequence involving a "Teacher = self with expert solution + Student = self without expert solution." It rewrites fewer than 1…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Expert Trajectories"
  - "Self-distillation"
  - "Contrastive Learning"
  - "Distribution Alignment"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 3331e0406abc0ce7
---

# Making Expert Reasoning Learnable with Self-Distillation

**Conference**: ICML 2026  
**arXiv**: [2602.02405](https://arxiv.org/abs/2602.02405)  
**Code**: https://github.com/ethanm88/DAIL  
**Area**: LLM Reasoning  
**Keywords**: Expert Trajectories, Self-distillation, Contrastive Learning, Distribution Alignment, Mathematical Reasoning

## TL;DR
DAIL utilizes a mixed-policy rollout sequence involving a "Teacher = self with expert solution + Student = self without expert solution." It rewrites fewer than 1,000 expert trajectories into reasoning chains aligned with the student's policy distribution. By applying a contrastive loss to penalize "shortcut" tokens that have high probability in a negative reference model (which only sees intermediate answers), it achieves a pass@128 improvement of up to 31% on Qwen2.5-Instruct / Qwen3 and reduces reasoning tokens by half.

## Background & Motivation

**Background**: The two mainstream routes for improving LLM reasoning are Reinforcement Learning with Verifiable Rewards (RLVR, such as GRPO) and distilling long CoTs from stronger teacher models. Both assume that training signals are "readily available"—either the model can sample correct answers itself, or a stronger teacher exists.

**Limitations of Prior Work**: On hard problems like AIME/IMO, frontier models fail all 32 samples, resulting in zero reward, advantage, or gradient for RLVR. Meanwhile, solutions from human experts (e.g., Math Olympiad contestants) are written for human readers, often skipping steps or omitting derivation details. Direct SFT on these solutions tends to disrupt the reasoning processes learned by the model during post-training.

**Key Challenge**: There are two types of misalignment between the expert trajectory distribution $p_{\text{expert}}$ and the student policy distribution $p_\theta$: (1) **didactic shortcuts**, where experts omit intermediate steps necessary for the student; and (2) **rationalization shortcuts**, where the model "looks ahead" at the answer and forces the derivation toward the known result rather than actually deriving it. Standard NLL treats all tokens equally, internalizing both types of shortcuts.

**Goal**: To maximize the conversion of each expert solution into generalizable reasoning training signals under conditions of minimal expert data ($n < 1000$) and potentially non-verifiable problems (e.g., open-ended proofs).

**Key Insight**: The authors decompose the problem into two stages: first, **distribution-aligned data synthesis** (transforming OOD expert solutions into in-distribution expanded trajectories), and second, a **shortcut-sensitive objective function** (specifically penalizing tokens that are highly probable only when "peeking" at the answer).

**Core Idea**: A mixed-policy rollout using a "self-distillation" teacher $M_T = M_{\theta_{\text{ref}}}(\cdot | x, s)$ (the same model conditioned on the expert solution $s$) and a student $M_\theta(\cdot | x)$ generates trajectories via a speculative-decoding-style strategy. A **negative reference model** $M_{NR}$, conditioned only on key nodes $\tilde s$ of the expert solution, is then constructed. The student is trained using the contrastive loss $\mathrm{KL}(M_\theta \| M_T) - \gamma \mathrm{KL}(M_\theta \| M_{NR})$.

## Method

### Overall Architecture
DAIL is a two-stage offline training method that takes $n < 1000$ expert (problem, solution) pairs $\mathcal{D} = \{(x_i, s_i)\}$ as input and outputs an updated student model $M_\theta$. The pipeline consists of: (1) **In-distribution trajectory synthesis**—using frozen initial weights $\theta_{\text{ref}}$, both a "teacher" (observing the expert solution) and a "student" (not observing it) are instantiated. They generate expanded reasoning trajectories $r_i$ via mixed-policy decoding to form a synthetic dataset $\mathcal{D}_{\text{syn}} = \{(x_i, r_i)\}$. (2) **Contrastive fine-tuning**—$M_\theta$ is trained on $\mathcal{D}_{\text{syn}}$ using a contrastive loss that pulls the model toward the teacher (full solution) and pushes it away from the negative reference (answer landmarks only). The training is fully offline, and since the teacher, student, and negative reference share base weights, a single set of weights with LoRA adapters and toggles can be used.

### Key Designs

1.  **Mixed Policy Decoding**:
    - **Function**: Rewrites expert solution $s$ into a training trajectory that is in-distribution for student $M_\theta$ but remains anchored to the content of $s$. This solves the extremes of "teacher copying $s$ exactly" and "student deviating entirely."
    - **Mechanism**: For the $i$-th token, a sample $t \sim M_\theta(\cdot | x, r_{<i})$ is drawn from the student. The teacher then performs an "accept/reject" step: if $M_T(t | r_{<i}) \geq \tau$, $r_i := t$ is accepted; otherwise, it falls back to teacher sampling $r_i \sim M_T(\cdot | r_{<i})$. Inspired by speculative decoding and DAgger-style imitation, but with the goal of letting the student "speak" as much as possible unless it deviates significantly from the expert path.
    - **Design Motivation**: Long CoT reasoning models (e.g., Qwen3-think) exhibit reflective properties. Direct teacher sampling often triggers meta-comments like "referring to the expert solution," which breaks the natural self-verification flow. Mixed rollout preserves the student's native backtracking/self-correction rhythm while remaining lightly anchored by the expert solution. For non-reasoning models (e.g., Qwen2.5-Instruct), direct sampling with prompt engineering is sufficient.

2.  **Contrastive Objective with Negative Reference**:
    - **Function**: During training on $\mathcal{D}_{\text{syn}}$, it actively penalizes rationalization shortcut tokens—those highly probable only when the answer is known—to prevent the student from memorizing jumpy derivations.
    - **Mechanism**: A negative reference $M_{NR}(\cdot) = M_{\theta_{\text{ref}}}(\cdot | x, \tilde s)$ is constructed, where $\tilde s$ represents "coarse-grained answer landmarks" extracted from $s$ (e.g., a list of intermediate numerical/symbolic results). A model conditioned on $\tilde s$ tends to bypass step-by-step reasoning. The loss function is:
        $$L(\theta) = \mathbb{E}_{(x,r) \sim \mathcal{D}_{\text{syn}}} \sum_{t=1}^{|r|} \left[ \mathrm{KL}(M_\theta(\cdot|x, r_{<t}) \| M_T(\cdot | r_{<t})) - \gamma \mathrm{KL}(M_\theta(\cdot|x, r_{<t}) \| M_{NR}(\cdot | r_{<t})) \right]$$
        This effectively "pulls toward the full-knowledge teacher and pushes away from the landmark-only negative reference."
    - **Design Motivation**: While maximizing a negative term is theoretically unbounded, the student is initialized from $\theta_{\text{ref}}$ and strongly anchored by the positive term, making training stable in practice. This aligns with findings by Kumar et al. (2022) regarding BC on sub-optimal data—standard NLL cannot distinguish "valid reasoning" from "spurious shortcuts," whereas token-level KL contrast applies penalties precisely where the conditional distributions diverge.

3.  **Efficiency-Friendly Training Framework**:
    - **Function**: Decouples the traditional RLVR "generation + optimization" cycle into "offline generation of $\mathcal{D}_{\text{syn}}$ followed by offline training," and manages the three roles using "one set of weights + one LoRA toggle."
    - **Mechanism**: (a) The asynchronous data synthesis can be scaled across distributed clusters independently of the GPU optimization phase. (b) Since $\theta_{\text{ref}}$, $M_T$, and $M_{NR}$ share base parameters, and the student only differs by a LoRA adapter (Hu et al., 2022), all three forward passes use the same frozen weights with the LoRA toggle turned on/off as needed. VRAM usage is essentially equivalent to single-model inference.
    - **Design Motivation**: RLVR takes ~1k GPU hours to converge on hard problems (NuRL) due to the bottleneck of interleaving sampling and training. DAIL's offline nature allows for data reuse and caching. When combined with LoRA, it allows 14B models to be trained on small clusters, enabling rapid iteration for new difficult datasets.

### Loss & Training
The formal loss is the contrastive KL given above, with $\gamma$ as the key hyperparameter for the negative term. $\tilde s$ is constructed for mathematical scenarios using a fixed regex (preserving $\boxed{}$ and key right-hand sides of equations) without additional annotation. Training data: `e1-verifiable` (417 AIME problems from 1985–2023 that the base model fails across 32 samples) for Qwen2.5-7B-Instruct; and `e1-proof` (669 IMO-level open-ended proof problems provided by USA IMO coach Evan Chen) for Qwen3-8B/14B (think), demonstrating DAIL's ability to train on **non-verifiable** proof problems.

## Key Experimental Results

### Main Results

Mathematical reasoning pass@k (Aggregated across AIME 2024/25, BeyondAIME, and IMO-AnswerBench; Qwen2.5-7B-Instruct trained on `e1-verifiable`):

| Method | Training Type | pass@128 (vs. Base) | Remarks |
| :--- | :--- | :--- | :--- |
| Qwen2.5-7B-Instruct (Base) | — | Baseline | Post-trained instruct model |
| GRPO (on `e1-verifiable`) | RLVR | **Decrease** | Sparse rewards on hard problems; overfits to rare random correct rollouts |
| NuRL + GRPO | RLVR + hint | Lower than GRPO | Relies on hints during training; drops after hint removal during inference |
| GRPO (DeepScaleR, 40K tasks) | Large-scale RLVR | pass@1 slight inc.; pass@k dec. | Simply scaling verifiable data is insufficient for Olympiad problems |
| Direct SFT on Expert Sol. | behavior cloning | Decrease | OOD data collapses model performance |
| STaR rationalization | Self-synthesis | Decrease | Model lacks capability to self-generate valid reasoning chains |
| **DAIL (Ours)** | Self-distill + Contrast | **+ up to 31% pass@128** | Only method to provide stable improvement |

Token efficiency at test time (Qwen3-8B/14B (think) trained on `e1-proof`, pass@128 vs. token budget): Within a 512–4096 token budget, DAIL consistently outperforms untrained Qwen3 and matches the best performance of the untrained model using **2× fewer tokens**.

OOD Generalization (GPQA-Diamond; Graduate-level physics/chem/bio; 8 sets of pass@1 / pass@128):

| Setup | Base Average | DAIL Average | Conclusion |
| :--- | :--- | :--- | :--- |
| Qwen2.5 pass@1 / pass@128 | 34.1 / 85.9 | **35.1** / 84.3 | Nearly equivalent; no catastrophic forgetting |
| Qwen3 pass@128 (512/1024/2048/4096) | 93.9 / 95.5 / 93.4 / 93.4 | **96.5 / 96.9 / 96.5 / 96.0** | Consistent improvement of ~3 points |

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Full DAIL (contrastive + mixed rollout) | Main result | Training set pass@k is actually **lower** than NLL, but OOD test performance is highest |
| Replaced with NLL loss | pass@1 / pass@128 drops | Student learns rationalization shortcuts without negative reference |
| Direct sampling vs Mixed (Qwen3-think) | Mixed significantly better | Direct sampling in reflective LRMs introduces "refer to expert" meta-comments |
| Direct sampling vs Mixed (Qwen2.5-Inst) | Direct slightly better | Prompts are sufficient to control shortcuts in non-reflective models |
| Training set pass@k comparison | NLL / RLVR high on train, low on test | Direct evidence: They learn shortcuts, not generalized reasoning |

### Key Findings
- The gain from contrastive loss primarily manifests in pass@1, with a 15–20% improvement over NLL on direct sampling data because such data contains more shortcuts, making the contrastive "filtering" more valuable.
- DAIL's low training set scores and high test set scores—a "reverse generalization gap"—provide direct evidence that the contrastive objective successfully suppresses non-robust reasoning patterns.
- The failure of RLVR on hard problems is not due to zero rewards but because the model **overfits** to rare stochastic successes, causing general reasoning ability to degrade.
- DAIL scales positively across both parameters (8B→14B) and token budgets (512→4096).

## Highlights & Insights
- **"Teacher = Self + Answer" positioning**: Unlike traditional distillation requiring a stronger external teacher, DAIL uses "self with the answer," bypassing the lack of superior models for hard problems.
- **Negative Reference construction**: By simply conditioning the same frozen model on **partial information** (e.g., answer landmarks), a control distribution biased toward shortcut behavior is naturally generated. This logic is transferable to code generation (partial = signature only) or theorem proving.
- **Training on non-verifiable proofs**: The `e1-proof` dataset allows for post-training on problems where no executable verifier exists, serving as a vital supplement to the RLVR paradigm.
- **The "Low Training/High Test" paradox**: When the training objective is to "reduce shortcut imitation," the loss naturally includes a term that sacrifices fit on the training distribution to achieve better generalization.

## Limitations & Future Work
- The negative reference $\tilde s$ currently relies on regex to extract nodes, which is tied to the format of math problems. Generalizing this to code or legal reasoning requires new extraction rules.
- Evaluation is limited to Math + GPQA. Other reasoning tasks (Code, Planning, Lean) are not yet verified.
- The stable ranges for hyperparameters $\gamma$ (negative weight) and $\tau$ (acceptance threshold) were not systematically scanned.
- The data scale remains at several hundred expert solutions; scaling to tens of thousands of samples and managing potential tension between positive and negative terms remains an open question.

## Related Work & Insights
- **vs. On-policy distillation**: DAIL modifies the teacher to be the "same model with answer," requiring only a single model and handling teacher hallucinations via the contrastive term.
- **vs. RLVR / GRPO**: RLVR requires verifiability and the model's ability to sample correct answers. DAIL breaks both assumptions.
- **vs. STaR rationalization**: STaR relies on the model's own capability to rationalize; DAIL uses mixed-policy rollout anchored by expert solutions to prevent the model from "making things up."
- **Methodological Inspiration**: The DAIL template can be applied to any scenario characterized by "scarce expert trajectories + direct imitation failure + structured answer extraction"—such as surgical robot hit-logs, SQL engineer logs, or penetration testing writeups.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combines self-distillation, speculative decoding, and contrastive RL into a clean two-stage framework, pioneering post-training on non-verifiable olympiad proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 math benchmarks + GPQA OOD + two base models + 5 baseline types, though lacks systematic hyperparameter scans.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation; excellently defines the concepts of "didactic shortcut vs rationalization shortcut."
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for hard-problem post-training (< 1000 samples, offline, LoRA-friendly) and releases the `e1-proof` dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance](eapo_enhancing_policy_optimization_with_on-demand_expert_assistance.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](../../AAAI2026/reinforcement_learning/in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](../../ICLR2026/reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](../../AAAI2026/reinforcement_learning/language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)

</div>

<!-- RELATED:END -->
