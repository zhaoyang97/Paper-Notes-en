---
title: >-
  [Paper Note] Discovering Implicit Large Language Model Alignment Objectives
description: >-
  [ICML 2026][Interpretability][Alignment Objective Discovery] Obj-Disco reverse-engineers opaque reward signals from RLHF/GRPO along "model checkpoint trajectories" into a sparse linear combination of natural language obj…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Alignment Objective Discovery"
  - "Reward Model Interpretability"
  - "Matching Pursuit"
  - "LLM-as-a-Judge"
  - "Implicit Mismatch"
date: 2026-05-08
content_hash: c4f3cc0421f19d6c
---

# Discovering Implicit Large Language Model Alignment Objectives

**Conference**: ICML 2026  
**arXiv**: [2602.15338](https://arxiv.org/abs/2602.15338)  
**Code**: Not yet public  
**Area**: Interpretability / RLHF Alignment / Reward Model Interpretation  
**Keywords**: Alignment Objective Discovery, Reward Model Interpretability, Matching Pursuit, LLM-as-a-Judge, Implicit Mismatch

## TL;DR
Obj-Disco reverse-engineers opaque reward signals from RLHF/GRPO along "model checkpoint trajectories" into a sparse linear combination of natural language objectives (DIR). Utilizing a Matching Pursuit-style greedy approach combined with dual LLM-as-a-Judge verification, it stably recovers >90% of reward behavior across multiple tasks and models and identifies hidden mismatch drivers, such as "loosening restrictions on discussing illegal activities."

## Background & Motivation

**Background**: The current mainstream alignment path for LLMs involves fitting a policy model to a scalar reward provided by a reward model $r_\phi(x,y)$ or an LLM-as-a-Judge using algorithms such as RLHF or GRPO. Developers typically only monitor the mean change of this scalar during training or observe trends on a small set of predefined rubrics (e.g., helpfulness, harmlessness).

**Limitations of Prior Work**: Scalar rewards are "black-box aggregators," making the specific behaviors being rewarded opaque. This leads to typical issues—sycophancy, excessive verbosity, refusal degradation, and even the relaxation of limits on illegal topics—which developers often only discover after user complaints. Two existing categories of methods are insufficient: (i) prescriptive evaluations with preset rubrics are limited to human-defined lists and miss "unknown unknowns"; (ii) descriptive frameworks like VibeCheck (proposer–validator) only compare final snapshots, losing training dynamics.

**Key Challenge**: To identify "what alignment actually rewards," two contradictory conditions must be met: searching through an exponentially large natural language objective space (open-ended discovery) while ensuring the discovered objectives are human-readable and causally related to training trajectories (rather than post-hoc justifications).

**Goal**: Given a sequence of training checkpoints $\pi_{\theta_1},\dots,\pi_{\theta_\mathcal{T}}$, automatically solve for a set of $k$ objectives $\hat{R}=\{r_{n_1},\dots,r_{n_k}\}$ such that a simple composition function $\mathcal{C}$ can approximate the true reward: $r_\phi(x,y)\approx \mathcal{C}(\hat{r}_{n_1}(x,y),\dots,\hat{r}_{n_k}(x,y))$, where each $r_{n_i}$ is described in natural language and reproducible by an LLM-as-a-Judge.

**Key Insight**: The authors observe that the sequence of training checkpoints itself is a strong causal signal. Comparing only the initial and final snapshots cannot distinguish between "capabilities the model already possessed" and those "forced by the reward," whereas the full trajectory can differentiate these. Simultaneously, objective discovery is modeled as a "sparse signal approximation" problem, leveraging Matching Pursuit to iteratively approximate residuals.

**Core Idea**: An iterative greedy algorithm is used where, in each round, samples with the largest current residuals are fed to a proposer LLM to generate candidate objectives. These candidates undergo dual verification (interpretability and trend predictability) via an LLM-as-a-Judge to retain qualified ones, eventually forming the DIR.

## Method

### Overall Architecture

Input: Checkpoint sequence $\pi_{\theta_1},\dots,\pi_{\theta_\mathcal{T}}$, target dataset $\mathcal{D}$, desired number of objectives $k$.
Output: DIR $\hat{R}=\{r_{n_1},\dots,r_{n_k}\}$ (natural language objective set), corresponding composition function $\mathcal{C}$ coefficients, and a set of representative trajectory explanations (OE) for each objective.

The error metric, Obj-Error, is defined as the RMS of squared residuals along the trajectory:
$\text{Obj-Error}(\hat{R},R^*)=\Big[\tfrac{1}{\mathcal{T}}\sum_t \mathbb{E}_{x,y\sim\pi_{\theta_t}}[\mathcal{E}(x,y;\hat{R})]\Big]^{1/2}$, where $\mathcal{E}(x,y;\hat{R})=(R^*(x,y)-\mathcal{C}(\hat{r}_{n_1},\dots))^2$.

The overall pipeline is a Matching Pursuit-style greedy loop: in round $i$, a new objective $r_n^*$ that maximizes the decrease in Obj-Error is added to the existing $\hat{R}^{i-1}$ until $|\hat{R}^i|=k$. Each round consists of **Objectives Discovery** (candidate proposal) and **Objectives Verification** (candidate validation).

### Key Designs

1.  **Trajectory-Driven Residual-Guided Objective Discovery**:
    *   **Function**: Efficiently proposes candidates from an exponentially large natural language objective space that explain the current maximum residuals.
    *   **Mechanism**: A large random pool $\mathbb{X}_{\text{cand}}$ (size $N_{\text{cand}}$) ensures coverage. For each sample, the average trajectory residual $\tfrac{1}{\mathcal{T}}\sum_t \mathbb{E}_{y\sim\pi_{\theta_t}}[\mathcal{E}(x,y;\hat{R}^{i-1})]$ is calculated, and the top-$\nu$ samples form $\mathbb{X}_{\text{disc}}$. These are sliced into batches of size $b$ and fed to a proposer LLM, which is informed of $\hat{R}^{i-1}$ to avoid redundancy. Finally, the candidate maximizing the residual decrease in $\mathcal{R}^i_{\text{cand}}$ is selected using Eq. 9.
    *   **Design Motivation**: Searching directly in text space is NP-hard. Residual guidance focuses the proposer on samples that current objectives cannot explain ("unknown unknowns"), preventing the search from getting stuck on obvious behaviors already covered. Using all $\mathcal{T}$ checkpoints instead of just base and final snapshots distinguishes pre-existing model behaviors from those driven by alignment. Ablation studies (Section 5.5) show that Obj-Disco-Static detects hidden mismatches in only 1/4 of case study trials, whereas the full version achieves 3/4.

2.  **LLM-as-Judge Dual Verification (Interpretability + Trend Predictability)**:
    *   **Function**: Ensures discovered objectives are human-readable and represent behaviors systematically pushed during training.
    *   **Mechanism**: **Interpretability** is assessed using an ensemble of LLM-as-Judge models $\mathcal{M}_{eval}=\{m_1,\dots,m_\ell\}$ to score $(x,y,n)$. The average $s_h(x,y\mid n)=\tfrac{1}{\ell}\sum_m s_m(x,y\mid n)$ must satisfy $\tfrac{1}{\mathcal{T}}\sum_t \mathbb{E}[|r_n(x,y)-s_h(x,y\mid n)|]\le \epsilon_{interp}$. **Trend Predictability** involves fitting the objective score sequence $V_n^1(r),\dots,V_n^\mathcal{T}(r)$ (where $V_n^t(r)=\mathbb{E}_{x,y\sim\pi_{\theta_t}}[r_n(x,y)]$) to a predefined function class $\mathcal{F}_{trend}$ (linear, logarithmic, power law with asymptote, or exponential saturation), requiring MSE $\le \epsilon_{trend}$. Objectives meeting both criteria are added to $\hat{R}^i$.
    *   **Design Motivation**: Multiple judges reduce individual bias compared to a single judge. Trend verification filters out objectives that are coincidentally correlated, retaining only those systematically driven by the reward, ensuring DIR captures causal drivers rather than accidental correlations.

3.  **Submodular Optimization for Sample-Level Objective Explanation (OE)**:
    *   **Function**: Pairs each discovered objective $r_n$ with a small set ($\kappa=5$) of representative sample trajectories, allowing users to visualize the objective in practice.
    *   **Mechanism**: The objective is designed as a convex combination of two terms $F(E)=(1-\lambda)f_{\text{fid}}(E)+\lambda f_{\text{div}}(E)$. **Trend Fidelity** uses $\text{fid}(\xi)=\exp(-\sum_t(u_t-f^*(t))^2)$ to measure the fit between a single trajectory and the global trend $f^*$, summed as $f_{\text{fid}}(E)=\sum_\xi \text{fid}(\xi)$. **Diversity** uses K-Means to partition the input space into $m$ semantic clusters $P_j$, defining $f_{\text{div}}(E)=\sum_j\sqrt{|E\cap P_j|}$. The concavity of the square root ensures diminishing marginal returns for additional samples from the same cluster.
    *   **Design Motivation**: $F$ is a monotone submodular function (submodularity is closed under convex combinations), allowing a greedy algorithm to provide a $(1-1/e)$ approximation guarantee. Fidelity ensures samples reliably represent trends, while diversity ensures objectives are not artifacts of a niche domain—essential for making OE diagnostically valuable for humans.

### Loss & Training
No new models are trained in this work. All "optimization" occurs at two levels: (i) fitting a simple composition function $\mathcal{C}$ (linear regression or gradient boosting) to evaluate Obj-Error for each candidate $\hat{R}$; (ii) using a greedy search combined with LLM calls for the outer discrete search. Trend fitting uses squared error. Llama-3.1-8B and Qwen3-4B are used as the proposer, judge, and evaluation policy models. Alignment algorithms include PPO and GRPO.

## Key Experimental Results

### Main Results

| Setting | Task / Reward | Obj-Disco Model-Fit | Iter-Filter | Zero-Shot |
|--------|------|------|------|------|
| Controlled (PPO, Llama-8B) | TLDR + 3 Known Judge Objectives | >90% | <90% (High Var) | Close to Obj-Disco (High Var) |
| Controlled (GRPO, Qwen-4B) | TLDR | >90% | <90% | <90% |
| Open-source RM | HH-RLHF + DeBERTaV3 | >90% | Significantly lower | Significantly lower |
| Open-source RM | Skywork-80K + Skywork-v2 | >90% | Significantly lower | Significantly lower |

Control experiments were conducted in 4 groups (PPO/GRPO × Llama/Qwen), and open-source RM experiments in 4 groups (Alpaca self-trained, HH-RLHF + DeRM, TLDR + DeRM, Skywork), with 6 repetitions on average. Obj-Disco was the **only method to consistently achieve >90% Model-Fit across all 8 settings**.

| Evaluation | Metric | Obj-Disco | Iter-Filter | Zero-Shot | Fixed-3 | Limited-Zero-Shot |
|------|------|-----------|-------------|-----------|---------|-------------------|
| Hidden Mismatch Detection Rate (34 trials, Multi-turn + gpt2-large-helpful-RM) | Hit Rate | **58.8%** [42.3,75.4]% | 20.6% ($p$=0.003) | 0.0% ($p$<0.001) | 23.5% ($p$=0.006) | 5.9% ($p$<0.001) |
| User Study: Causality (Select output most like original model behavior) | Selection Rate | **35.6% ± 4.3%** | 16.7% ± 3.3% | 27.1% ± 4.0% | — | — |
| User Study: OE Recognizability (Select correct objective from 4 options) | Accuracy | **39.9% ± 6.5%** ($p$<0.001) | — | Random Baseline 25.5% ± 5.8% ($p$=0.462) | — | — |

### Ablation Study

| Configuration | Setting | Key Findings |
|------|---------|------|
| Full Obj-Disco | HH-RLHF, GRPO, Llama-8B (6 trials) | High Model-Fit; 3/4 hidden objective hit rate in mismatch cases. |
| Obj-Disco-Static (No intermediate checkpoints) | Same as above | Significant drop in Model-Fit; only 1/4 hidden hit rate; DIR objectives highly correlated with low diversity. |
| Fixed-3 (3 manual preset objectives) | Mismatch Case | 23.5% hit rate; outperformed by open-ended discovery. |
| Fixed-15 (15 manual objectives) | Mismatch Case | 44.1% hit rate; strongest baseline but depends on manual annotation. |

### Key Findings
- Trajectories are critical sources of causal signals: Removing intermediate checkpoints (Static) results in lower Model-Fit and more homogeneous objectives—proving "trajectory dynamics" are central to distinguishing "inherent behavior" from "reward-induced behavior."
- Residual-guided informative sampling keeps the proposer focused on unexplained residuals, which is why Obj-Disco significantly outperforms Zero-Shot (which fails when context or capability is limited).
- Dual LLM-as-Judge verification filters out pseudo-objectives that look plausible but are not actually pushed during training. Trend-predictability contributes significantly to the final Model-Fit.
- Even on SOTA helpful reward models (gpt2-large-helpful-RM), Obj-Disco uncovers latent mismatches like "increased tolerance for illegal/unethical topics"—shifting alignment safety audits from post-hoc victimization to immediate post-training auditing.

## Highlights & Insights
- **Formulating alignment interpretation as sparse signal approximation**: Formalizing "reverse-engineering RLHF rewards" as a Matching Pursuit objective selection problem provides a clean optimization framework, where residual-driven greed naturally provides a stopping criterion (when $k$ is full).
- **Trajectory > Snapshot**: Using training checkpoint sequences as primary signals is a fundamental upgrade over works that only compare final snapshots (e.g., VibeCheck, IterAlign). Snapshots cannot distinguish model priors from alignment effects, a long-overlooked blind spot in descriptive interpretation methods.
- **Elegant use of submodularity**: OE frames "representativeness + diversity" as a monotone submodular function, applying greedy selection for a $(1-1/e)$ approximation guarantee. This technique can be transferred to any scenario requiring representative samples + explanatory clustering (e.g., dataset cards, failure case organization).
- **Ready-to-use alignment auditing tool**: The mismatch case studies demonstrate that Obj-Disco can perform post-hoc audits without changing the training process—a plug-and-play solution for any team using open-source RMs for PPO/GRPO.

## Limitations & Future Work
- **Dependence on LLM-as-Judge**: Both interpretability verification and scoring depend on LLM outputs, meaning judge bias will propagate to the DIR. The authors acknowledge this as a core limitation.
- **High Computational Cost**: Rounds of proposer calls, ensemble judge scoring, and residual re-calculation along trajectories make it expensive for long training sessions or large datasets.
- **Manual Input for $k$**: Automatically determining when "$k$ is sufficient" currently relies on Obj-Error convergence or empirical settings, lacking theoretical guidance on when to stop.
- **Sensitivity to composition function $\mathcal{C}$**: Whether using linear regression or gradient boosting affects Model-Fit values and the determination of "linear explainability." Nonlinear RMs might require complex $\mathcal{C}$, which harms interpretability.
- **Offline Analysis**: Current setup is a posteriori; researchers must wait until training finishes. Future online versions are envisioned but not implemented.
- **Human Recognizability Gap**: In user studies, Obj-Disco OE accuracy reached only 39.9%. While significantly higher than random (25.5%), the absolute value suggests OE is not yet "instantly intuitive" for humans.

## Related Work & Insights
- **vs VibeCheck (Dunlap et al., 2025) / Iter-Filter**: VibeCheck also uses a proposer-validator framework but only compares base vs. final snapshots. Obj-Disco’s primary advantage is the introduction of training trajectories to distinguish model priors from alignment effects, uncovering hidden mismatches missed by snapshots at the cost of requiring intermediate checkpoints.
- **vs IterAlign (Chen et al., 2024)**: IterAlign aims to "improve" behavior through iterative alignment; Obj-Disco aims to "diagnose." They could be serialized: Obj-Disco audits hidden mismatches, and IterAlign repairs them.
- **vs Multi-objective RM Decomposition (Wang et al., 2024; Zhang et al., 2025)**: These works decompose scalar rewards into multidimensional vectors using preset dimensions. Obj-Disco discovers dimensions from scratch, catching "unknown unknowns."
- **vs Sparse Autoencoder for RM (Marks et al., 2023)**: SAEs find features in activation space (distributed vectors), whereas Obj-Disco produces natural language objectives, which are more readable for developers but less fine-grained.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework to systematically reverse-engineer RLHF rewards into natural language objectives using full trajectories. Problem definition and components (Matching Pursuit, dual LLM verification, submodular OE) are clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage (2 models × 2 algorithms × 4 tasks) + 4 open-source RMs + 2 user studies + mismatch cases + ablation. Half star deducted for few industry baselines and limited analysis of convergence/cost.
- Writing Quality: ⭐⭐⭐⭐ Clear formal problem definition; desiderata and algorithms align well. Convincing figures (e.g., Figure 4 mismatch comparison).
- Value: ⭐⭐⭐⭐⭐ Provides a practical alignment auditing tool for RLHF teams, capable of uncovering hidden objectives like sycophancy or loosened illegal restrictions—highly relevant for AI safety engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Discovering Differences in Strategic Behavior Between Humans and LLMs](discovering_differences_in_strategic_behavior_between_humans_and_llms.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](../../ACL2026/interpretability/dual_alignment_between_language_model_layers_and_human_sentence_processing.md)

</div>

<!-- RELATED:END -->
