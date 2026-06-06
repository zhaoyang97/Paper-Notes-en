---
title: >-
  [Paper Note] Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
description: >-
  [ICML 2026][Reinforcement Learning][Evolving rubric] Dr. Tulu proposes RLER (Reinforcement Learning with Evolving Rubrics), allowing evaluation rubrics to co-evolve with the strategy during training. This extends RLVR fr…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Evolving rubric"
  - "Deep research agent"
  - "Long-form response"
  - "Verifiable citation"
  - "RLVR extension"
date: 2026-05-08
content_hash: 7bfd152fb816d56f
---

# Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research

**Conference**: ICML 2026  
**arXiv**: [2511.19399](https://arxiv.org/abs/2511.19399)  
**Code**: Yes  
**Area**: Reinforcement Learning / LLM Agent / Deep Research  
**Keywords**: Evolving rubric, Deep research agent, Long-form response, Verifiable citation, RLVR extension

## TL;DR
Dr. Tulu proposes RLER (Reinforcement Learning with Evolving Rubrics), allowing evaluation rubrics to co-evolve with the strategy during training. This extends RLVR from short-answer QA to long-form deep research tasks with citations. Ultimately, DR Tulu-8B, trained from Qwen3-8B, outperforms Tongyi DR-30B by an average of 15.6 points across four long-form deep research benchmarks and reaches levels comparable to OpenAI Deep Research at 1000x lower cost.

## Background & Motivation

**Background**: The deep research (DR) agent track is currently divided between training-free prompt engineering (e.g., WebThinker) or RLVR training on short-answer search QA (e.g., HotpotQA, PopQA classes) like Search-R1, ASearcher, and WebExplorer. The latter relies on a simple fact: short answers can provide 0/1 rewards via exact matching or F1 scores, offering high verifiability.

**Limitations of Prior Work**: In reality, users asking deep research questions (e.g., "Summarize treatment evidence for a specific clinical genetic mutation") expect long reports with citations rather than single-sentence answers. These tasks possess three characteristics that cause RLVR to fail: (1) Evaluation criteria are under-specified—there is no standard template for what constitutes a good answer; (2) Evaluation requires external up-to-date knowledge, making model parameters alone unreliable; (3) Different dimensions of long responses (coverage, citation quality, expression) require multi-criterion joint scoring. Existing open-source agents trained directly with RLVR score extremely low on long-form benchmarks (Search-R1 achieves only 22.2 on SQAv2).

**Key Challenge**: Long-form DR tasks **require a dense, discriminative reward signal that covers the latest knowledge**, but **static human-written rubrics** are too rigid (lacking coverage and prone to reward hacking), while **pure LM-generated rubrics** are limited by the model's internal parameter knowledge and miss critical evidence.

**Goal**: Design a mechanism that can dynamically construct, maintain, and retire evaluation rubrics within the RL training loop, allowing rubrics to incorporate the latest facts retrieved externally and refine based on comparative differences in current policy rollouts.

**Key Insight**: The authors view rubrics as a tool for "information asymmetry"—giving the rubric generator **access to more information than the policy** (external retrieval results + comparisons of multiple on-policy responses) to create a generation–verification gap, ensuring the rubric remains more "knowledgeable" about the task than the policy itself.

**Core Idea**: Allow rubrics to co-evolve with the policy in the RL loop—generating positive/negative rubrics using new rollouts at each step, maintaining a fixed-size buffer sorted by reward variance, and eliminating items with zero discriminative power.

## Method

### Overall Architecture

DR Tulu adopts a two-stage SFT-then-RL training process:
1. **Cold-start SFT**: 16K research trajectories with tool calls (including search / browse / cite / answer actions) are generated using GPT-5. After filtering, supervised fine-tuning is performed to teach Qwen3-8B basic search-writing-citation formats.
2. **RLER Main Training**: Online RL is conducted on 9K long-form prompts using a GRPO variant. Rewards are derived from a dynamically maintained rubric buffer + three auxiliary rewards (format / search / citation).

The model action space is $\{\text{think}, \text{tool}, \text{answer}, \text{cite}\}$, with three tools: `google search`, `web browse`, and `paper search`. Each prompt $x$ corresponds to a rubric buffer $R_x = R_x^{\text{persist}} \cup R_x^{\text{active}}$, where the persist part is constructed once via "retrieval + LM generation" before training, and the active part is dynamically updated during training. The system runs on the self-developed `dr-agent-lib`, supporting asynchronous tool calls, global caching, token-level loss, tool output masking, and sample packing.

### Key Designs

1. **Search-augmented Initial Rubrics + Persistent Buffer**:

    - **Function**: Establish a rubric baseline for each prompt based on "real external evidence" before training starts, preventing gaps in key facts caused by pure LM hallucination.
    - **Mechanism**: For each training prompt $x$, `SEARCH(x)` is called to retrieve relevant documents. These documents and $x$ are fed to a rubric generator $G_{\text{rubric}}$ (GPT-4.1) to produce a persistent rubric set $R_x^{\text{persist}} = \{R_1, \dots, R_{K_s}\}$. This part remains throughout RL training as the "foundational factual constraint." During scoring, the rubric score for a single response $y$ is $S(x, y) = \sum_k w_{x,k} \text{JUDGE}(r_{x,k}, y) / \sum_{k: w_{x,k} > 0} w_{x,k}$, where the JUDGE LM outputs $\{0, 0.5, 1\}$.
    - **Design Motivation**: Long-form DR evaluation requires up-to-date external knowledge. Rubrics generated without retrieval results often miss key facts. Providing the rubric generator with "privileged information" (retrieved documents + multi-rollout comparisons) to create a generation–verification gap is the foundation of RLER.

2. **Online Evolving Active Rubrics + Positive/Negative Distinction**:

    - **Function**: Automatically generate new rubrics based on on-policy rollouts at each RL step to capture current policy strengths and bad habits, maintaining discriminative reward signals.
    - **Mechanism**: At each training step, $G$ rollouts $\{y_i\}_{i=1}^G$ are sampled for each prompt $x$. These, along with the current $R_x$, are fed to $G_{\text{rubric}}$ to produce two types of new rubrics: (a) **Positive rubrics** capturing new knowledge or excellent patterns explored by certain rollouts; (b) **Negative rubrics** summarizing undesirable behaviors across rollouts, such as "verbatim copying of search snippets to cheat for high citation precision." Negative rubrics capture such reward hacking for explicit punishment. The resulting $R_x^{\text{new}}$ is added to the active buffer.
    - **Design Motivation**: The fundamental issue with static rubrics is that they are off-policy—treating all responses the same without distinguishing between what the policy does well and where it fails. Evolving rubrics with the policy effectively rewrites scoring criteria at each step based on "policy weaknesses + newly explored facts," making reward signals naturally on-policy. Negative rubrics serve as an explicit anti-hacking mechanism.

3. **Variance-based Rubric Buffer Management**:

    - **Function**: Prevent the number of rubrics from expanding linearly during training while retaining items with the strongest discriminative power to improve training efficiency.
    - **Mechanism**: After each GRPO rollout, current active rubrics are used to score all $\{y_i\}$. Rubrics with **zero reward variance** (either all rollouts pass or all fail) are deleted as they provide no gradient signal. The remaining rubrics are sorted by reward standard deviation in descending order, keeping only the top $K_{\max}$. Three auxiliary rewards for format, search, and citation are calculated in parallel to encourage correct formatting and high-quality utility.
    - **Design Motivation**: Too few rubrics lead to coarse scoring; too many lead to excessive judge costs and noise. "Variance" is a natural proxy for discriminability: zero variance means the rubric contributes nothing to policy improvement, while high variance indicates differences between rollouts that can generate effective gradients.

### Loss & Training

RL utilizes a GRPO variant where the reward is a weighted sum of the rubric score and three auxiliary rewards. Training incorporates token-level loss, 1-step asynchronous training, tool output masking, sample packing, and asynchronous tool calls (tool requests within a rollout are dispatched immediately without waiting for batches) to enhance long-horizon multi-tool RL throughput. SFT took 136 GPU hours on 8 H100s for 5 epochs; final RL took approximately 27,000 GPU hours on 16 H100s. GPT-4.1-mini served as the judge, and GPT-4.1 served as the rubric generator.

## Key Experimental Results

### Main Results

Average scores across four long-form deep research benchmarks (SQAv2 / HealthBench / ResearchQA / DRB):

| Category | Model | SQAv2 | HealthBench | ResearchQA | DRB | Avg |
|---|---|---|---|---|---|---|
| Closed | OpenAI Deep Research | 79.6 | 53.8 | 79.2 | 46.9 | **64.9** |
| Closed | Gemini 3 Pro + Search | 69.8 | 38.0 | 74.3 | 46.3 | 57.0 |
| Closed | Perplexity Deep Research | 67.3 | – | 75.3 | 42.3 | – |
| Open SOTA | Tongyi DR-30B | 46.5 | 46.2 | 66.7 | 40.6 | 50.0 |
| Open | WebThinker-32B-DPO | 32.9 | 11.1 | 48.6 | 23.3 | 28.9 |
| Open | Search-R1-7B | 22.2 | -0.1 | 27.9 | 9.5 | 14.9 |
| Naive RAG | Qwen3-8B | 40.4 | 16.5 | 56.1 | 33.3 | 36.5 |
| Ours | DR Tulu-8B (SFT) | 72.3 | 38.1 | 68.5 | 39.0 | 53.9 |
| Ours | **DR Tulu-8B (RL)** | **88.3** | 52.8 | 75.7 | 45.4 | **65.6** |

DR Tulu-8B ranked first on SQAv2 (88.3), and its overall average (65.6) exceeded Tongyi DR-30B by 15.6 points and OpenAI DR by 0.7 points. In terms of cost, OpenAI DR is approximately \$1.80 per query, while DR Tulu-8B (tools + inference) is only \$0.0018, which is 1000x cheaper.

### Ablation Study

| Configuration | Avg over 4 benchmarks | Description |
|---|---|---|
| Qwen3-8B + Search (No Training) | 31.9 | Starting point |
| DR Tulu-8B (SFT only) | 53.9 | SFT cold-start |
| DR Tulu-8B (SFT + RLER) | 65.6 | +11.7 from RL stage |

The improvement brought by RLER alone across benchmarks ranged from 6.4 to 16.0 points, proving that evolving rubrics are key. While SFT alone outperformed most open-source baselines, RLER was necessary to match proprietary systems.

### Key Findings

- **Adaptive Tool Selection**: DR Tulu-8B learned task-related tool preferences—using paper search 90% of the time on SQAv2 (academic) and web search/browse 55% of the time on DRB (general), without hardcoding.
- **Small Models Outperforming Large Models**: The 8B DR Tulu outperformed the 30B Tongyi DR and 32B WebThinker, indicating that the RL training paradigm (evolving rubrics) provides gains exceeding mere parameter scaling for long-form DR.
- **Citation Quality as the Open-source Gap**: Most existing open-source DR agents do not output citations, with SQAv2 citation scores near zero. DR Tulu-8B provides snippet-level citations, a primary reason for its performance over OpenAI DR on SQAv2.
- **Generalization to Genetic Tasks**: On the custom GeneticDiseasesQA (47 questions, 24 pathogenic variants), DR Tulu-8B approached GPT-5 + Search and OpenAI DR in Evidence Support / Quality / Synthesis, showing RLER learns general research patterns rather than dataset biases.

## Highlights & Insights

- **Methodological Value of Generation–Verification Gap**: The authors explicitly cite the rubric generator's access to more information than the policy as the core reason for RLER's effectiveness. This formalizes the observation that "AI feedback is superior to self-reflection" and can be extended to any RL task: providing the reward model/verifier with privileged info significantly improves reward quality.
- **Duality of Positive/Negative Rubrics**: Treating reward hacking as an observable signal—where bad behavior shared across rollouts triggers a negative rubric—acts as automated adversarial reward correction without manual patching.
- **Variance Ranking as a Selection Criterion**: Simple, implementable, and interpretable. It is more straightforward than complex correlation modeling found in other literature and serves as an excellent engineering trick.
- **Infrastructure Open-sourcing**: `dr-agent-lib` provides the essential building blocks for long-horizon agent RL (async tool calls, caching, rate limiting, tool output masking), a major engineering contribution for the DR agent community.

## Limitations & Future Work

- RLER heavily relies on a **strong, independent rubric generator** (GPT-4.1) and **judge LM** (GPT-4.1-mini), following a paradigm of "distilling small open-source DR agents from large proprietary models"; a fully open-source closed loop is not yet achieved, and rubric generator biases may be absorbed by the policy.
- Computational costs are high: 27K H100 hours + extensive GPT-4.1 / 4.1-mini API calls, making the reproduction threshold much higher than standard RLVR work.
- Hyperparameters like buffer size $K_{\max}$, variance ranking granularity, and the ratio of positive to negative rubrics were not fully explored.
- The training prompt set (approx. 9K) is still partially OOD relative to evaluation sets. Whether generalization to long-tail professional fields (like GeneticDiseasesQA) is stable remains to be seen beyond the small tested dataset.

## Related Work & Insights

- **vs Search-R1 / ASearcher / WebExplorer**: These are RLVR for short-answer QA with exact matching rewards. DR Tulu moves to long reports, using evolving rubrics to solve the reward problem in the absence of ground truth.
- **vs WebThinker / Ai2 ScholarQA (fixed pipeline)**: Fixed pipelines perform well on academic tasks but fail to generalize and over-generate in short-answer scenarios. DR Tulu is a learned policy that adaptively switches between long and short scenarios.
- **vs RaR (Rubrics-as-Rewards)**: RaR uses static rubrics; RLER differentiates itself by co-evolving rubrics with the policy and injecting privileged external knowledge via retrieval.
- **vs OpenAI DR / Perplexity DR**: Proprietary systems with undisclosed methods and 1000x costs. DR Tulu-8B achieves similar levels with an 8B open model, providing a reproducible deep research baseline.

## Rating

- **Novelty**: ⭐⭐⭐⭐ "Rubric co-evolution + retrieval privilege + variance-based buffer" is a first for systematic long-form RLVR, though individual components have precursors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers four long-form benchmarks, short-answer analysis, tool preference analysis, and a custom clinical dataset, comparing against proprietary, open-source, and fixed pipeline models.
- **Writing Quality**: ⭐⭐⭐⭐ Formulas and algorithms are clear; the motivation for the generation–verification gap is well-articulated.
- **Value**: ⭐⭐⭐⭐⭐ Extends RLVR from short-answers to long-form cited DR; the open-sourcing of models, data, and infrastructure is a reference-level contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ICML 2026\] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)
- [\[ICML 2026\] ORLoopBench: Solver-in-the-Loop Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](orloopbench_solver-in-the-loop_benchmarks_for_self-correction_and_behavioral_rat.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

</div>

<!-- RELATED:END -->
