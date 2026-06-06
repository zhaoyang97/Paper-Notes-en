---
title: >-
  [Paper Note] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation
description: >-
  [ACL 2026][Recommender Systems][User Simulator] The authors rewrite "user feedback logs" from recommender systems into a unified simulation scenario consisting of "user memory + exposure lists" understandable by LLMs. Th…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "User Simulator"
  - "Uncertainty Decomposition"
  - "DPO Preference Alignment"
  - "Decision Process Distillation"
date: 2026-05-08
content_hash: aaa4af91e867ed3b
---

# Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation

**Conference**: ACL 2026  
**arXiv**: [2508.18142](https://arxiv.org/abs/2508.18142)  
**Code**: https://github.com/Joinn99/UserMirrorer  
**Area**: User Simulation / Recommender Systems / RLHF / LLM Data Distillation  
**Keywords**: User Simulator, Recommender Systems, Uncertainty Decomposition, DPO Preference Alignment, Decision Process Distillation

## TL;DR
The authors rewrite "user feedback logs" from recommender systems into a unified simulation scenario consisting of "user memory + exposure lists" understandable by LLMs. They then utilize the EKB consumer decision model to generate explicit chain-of-thought decision processes as "clarifications." Through uncertainty decomposition and rejection sampling, 10K high-quality SFT/DPO samples are distilled, enabling a 3B Llama user simulator to outperform GPT-5 and Gemini-2.5-Flash in real-world user behavior prediction across 8 domains.

## Background & Motivation

**Background**: Recommender system (RS) iteration relies heavily on user feedback, but online A/B testing suffers from long cycles and privacy constraints. Consequently, the academic community has turned to user simulators—constructing "user digital twins" to replicate real behaviors offline. Early simulators used rules or RL, while recent approaches utilize LLM agents with prompt engineering to simulate user decision-making.

**Limitations of Prior Work**: (1) Existing LLM simulators are rarely fine-tuned on real user feedback, leading to weak task alignment; (2) Achieving high performance requires GPT-4 level models, making billion-user scaling costs prohibitive; (3) Raw user feedback possesses three characteristics that hinder fine-tuning: **ambiguity** (lack of decision context; "clicks" alone do not reveal intent), **noise** (accidental clicks, bot traffic), and **massive volume** (filtering high-quality samples from millions of logs is a challenge).

**Key Challenge**: High-quality decision reasoning requires powerful LLMs, yet deployment necessitates small models. The challenge lies in distilling the decision-making capabilities of strong LLMs into lightweight LLMs while simultaneously governing data noise and ambiguity.

**Goal**: To develop a unified framework that transforms RS logs into trainable "scenario-decision-behavior" triplets and automatically selects a subset that maximizes alignment effectiveness.

**Key Insight**: The **EKB model** (Engel-Kollat-Blackwell) from consumer behavior research naturally characterizes the purchase decision chain: stimulus → knowledge → evaluation. Using a strong LLM to generate explicit decision processes according to EKB acts as a "clarification" for each sample. This allows for uncertainty decomposition via the input-clarification framework (Hou et al. 2024) to automatically identify samples with the highest training value for weak models.

**Core Idea**: Use EKB decision processes as clarification + utilize the epistemic uncertainty gap between strong and weak models to select hard samples + use rejection sampling for denoising + apply DPO preference alignment to distill 10K high-quality data points from 16K candidate scenarios.

## Method

### Overall Architecture
UserMirrorer is an end-to-end "data construction + model training" framework: (1) RS logs from 8 domains (MIND/Amazon/MovieLens/Steam/Goodreads/MobileRec/LastFM/KuaiRec2) are converted into a unified simulation scenario $\bm{X}=\text{Prompt}(\bm{M},\bm{L})$, where $\bm{M}$ is user memory (profile + history) and $\bm{L}$ is an exposure list of length $N+1$ with items labeled [A]/[B]/[C]; (2) Qwen2.5-32B (Strong) and Llama-3.2-3B (Weak) sample 10 EKB decision processes each, with rejection sampling used to pair them as chosen/rejected; (3) Llama-3.2-3B undergoes SFT followed by DPO to obtain a deployment-grade 3B user simulator.

### Key Designs

1. **Decision-process Generation as Clarification**:

    - **Function**: Supplements each $(\bm{M}, \bm{L}, a)$ sample with an explicit decision reasoning explaining "why this user clicked [C]."
    - **Mechanism**: Adapts the EKB model into three stages: **Stimulus** (identifying external spatio-temporal/social factors and internal needs/emotions), **Knowledge** (extracting relevant attributes from the exposure list), and **Evaluation** (evaluating candidate behaviors using intuitive or logical styles, corresponding to Kahneman's "System 1/System 2"). The strong LLM generates the decision process $\bm{D}$ before predicting the action $\bm{Y}$.
    - **Design Motivation**: In the Hou et al. 2024 framework, clarification decomposes total uncertainty into aleatoric (data ambiguity) and epistemic (lack of model capability). The EKB decision process serves as a rational clarification for user choice, supplementing context and allowing the model to quantify epistemic uncertainty.

2. **Uncertainty-based Distillation**:

    - **Function**: Selects cases from 16K candidates where the strong model is confident but the weak model is confused, skipping samples that are too simple or outliers.
    - **Mechanism**: Strong and weak LLMs generate $N=10$ decision processes each. The weak LLM predicts behavior conditioned on both $\bm{D}$ sets, calculating $\Delta_{EU}(\bm{X}, (A,B)) = \mathbb{E}_{P(\bm{D}_A|\bm{X})}\mathcal{H}(P(\bm{Y}|\bm{X}\oplus \bm{D}_A)) - \mathbb{E}_{P(\bm{D}_B|\bm{X})}\mathcal{H}(P(\bm{Y}|\bm{X}\oplus \bm{D}_B))$. A large $\Delta_{EU}$ implies that clarification from strong model $A$ significantly reduces the entropy of weak model $B$. These samples with the highest "information gain" are the most valuable hard samples for training.
    - **Design Motivation**: Uncertainty gaps are more robust than accuracy gaps. Direct accuracy comparisons are prone to random noise, whereas entropy differences precisely measure the "epistemic signal provided by the decision process," targeting scenarios that truly require strong model reasoning for alignment.

3. **Sampling Denoised Behaviors**:

    - **Function**: Removes dataset-level noise (accidental clicks, bots) and constructs high-confidence preference pairs for DPO.
    - **Mechanism**: For each hard scenario, predictions from 10 decision processes are matched against real user behavior. If none match, the sample is discarded as noise. Among remaining samples, processes with correct predictions are labeled "accepted," while incorrect ones are labeled "rejected." The highest-confidence pair is selected for DPO.
    - **Design Motivation**: Training on all data treats noise as signal. Using "explainability of real behavior" as a sanity check and providing explicit chosen/rejected contrasts allows DPO to simultaneously optimize for content quality and behavioral fit.

### Loss & Training
Two-stage training: (1) **SFT Stage**: Next-token prediction using only accepted decision processes; (2) **DPO Stage**: Standard formula (Rafailov et al. 2023) using constructed preference pairs. GRPO (Shao et al. 2024) was also tested as a baseline using behavior matching as a reward. Optimal results were achieved with 10K samples (minimal gains beyond 40K). Inference uses temperature=1.0, top-p=0.9, averaging over 5 samples per scenario.

## Key Experimental Results

### Main Results: User Behavior Prediction Accuracy (Average across 8 domains, %)

| Model | Scale | Overall Acc. | Remarks |
|------|------|-------------|------|
| Llama-3.2-3B (base) | 3B | 22.7 | Baseline |
| Qwen2.5-32B-Instruct (Teacher) | 32B | 39.7 | Strong Teacher |
| GPT-5 (2025-08-07) | Closed | 42.2 | Commercial Flagship |
| Gemini-2.5-Flash | Closed | 42.5 | Commercial Flagship |
| Gemini-3.0-Pro-Preview | Closed | 47.7 | Strongest Commercial |
| Llama-3B + SFT | 3B | 46.0 | SFT alone beats GPT-5 |
| Llama-3B + SFT + GRPO | 3B | 52.7 | Further RL refinement |
| **Llama-3B + SFT + DPO** | **3B** | **55.0** | **+7.3 vs Gemini-3.0-Pro** |
| Qwen2.5-3B + SFT + DPO | 3B | 54.7 | Effective across backbones |

### Ablation Study: Data Selection Strategy (MIND / Synthetic Overall, %)

| Strategy | MIND Acc. | Synthetic Overall |
|------|-----------|-------------------|
| Llama-3B base | 19.9 | 22.7 |
| Random (w/o Decisions) | 25.7 | 48.3 |
| Random (w/ Decisions) | 31.4 | 53.9 |
| High Accuracy filter | 32.4 | 53.3 |
| Low Accuracy filter | 30.9 | 54.5 |
| Diff. Accuracy filter | 30.1 | 53.9 |
| IFD Score | 29.9 | 52.7 |
| **Ours (Uncertainty + Rejection)** | **34.0** | **55.0** |

### Key Findings
- **EKB decision processes provide decisive contributions**: Replacing "Random without decisions" with "Random with decisions" increases accuracy from 25.7 to 31.4 (MIND), showing that explicit chain-of-thought provides significant gains even without sample selection.
- **Uncertainty gap > Accuracy gap**: All accuracy-based heuristics (high/low/diff) and IFD were inferior to the epistemic uncertainty gap selection, proving entropy differences are better at identifying valuable hard samples.
- **DPO > GRPO**: DPO (55.0) vs. GRPO (52.7); the authors suggest GRPO's binary reward is too coarse, whereas DPO's continuous supervision on preference pairs is more stable for "soft" tasks like user behavior.
- **3B outperforms 47B/GPT-5**: Real user feedback contains "domain implicit preferences" that base LLMs cannot learn. Strong models have superior reasoning but weaker alignment; a combination of strong model reasoning + real feedback filtering + small model DPO is a cost-effective distillation-alignment pipeline.
- **Significant downstream RS gains**: Using simulated feedback for incremental training of LightGCN/DiffRec/SASRec/NARM yielded up to +45.3% MRR@5 (DiffRec) on Movielens, demonstrating that simulation signals can effectively benefit recommendation models.

## Highlights & Insights
- **Dual-use of decision processes as clarification**: While Hou et al. 2024 only used clarification for uncertainty estimation, this work cleverly utilizes the same decision process for (a) SFT supervision, (b) DPO preference contrast, and (c) uncertainty measurement for data filtering—achieving high data utility.
- **Effective migration of "Classical Consumer Behavior" (EKB) to LLM agents**: Injecting human psychological inductive biases (stimulus→knowledge→evaluation) makes chain-of-thought generation far more stable than free-form LLM outputs. This approach is transferable to other "choice modeling" tasks (dialogue policy, NPCs, user testing).
- **Entropy gap for data filtering over accuracy gap**: This is a universal trick for alignment tasks with stochastic answers; using logit entropy avoids the interference of ground truth noise.

## Limitations & Future Work
- The authors acknowledge that the current version only supports text and lacks multimodal inputs (limiting scenarios like short videos or music covers). It also only models "view and interact," ignoring implicit feedback like "view but leave."
- Observation: The EKB model assumes a "rational consumer," which may not fit impulsive behaviors like scrolling social feeds. The 10K sample size was chosen due to diminishing returns, but scalability to millions across domains is unverified.
- Future improvements could involve "hierarchical" decision processes (coarse-to-fine) combined with process reward models for individual EKB stages, or replacing DPO with step-wise DPO for independent alignment of stimulus/knowledge/evaluation.

## Related Work & Insights
- **vs AgentCF / Agent4Rec / RecMind**: These rely on prompt engineering + agent memory without fine-tuning, limited by the base LLM's ceiling. UserMirrorer turns real feedback into SFT+DPO data, allowing a 3B model to beat GPT-5.
- **vs LLaVA-Critic / UnifiedReward**: While those focus on "score alignment," UserMirrorer focuses on "behavior alignment." Both follow the paradigm of "filtered high-quality data + DPO," reinforcing that quality > quantity.
- **vs Hou et al. 2024 (Input Clarification Ensembling)**: Whereas the original work used clarification for LLM uncertainty estimation, this paper instantiates it as EKB processes for RS user simulation—an elegant application of theoretical tools to a specific domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of EKB, uncertainty decomposition, and rejection sampling is ingenious, providing strong "engineering elegance."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 domains + 4 RS backbones + 6 filtering baselines + data size ablation provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous formulas, and smooth transitions; though EKB might be slightly abrupt for readers without a psychology background.
- Value: ⭐⭐⭐⭐⭐ Provides a deployable 3B simulator, public datasets, and framework code, offering reusable value to the RS, agent, and data distillation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)

</div>

<!-- RELATED:END -->
