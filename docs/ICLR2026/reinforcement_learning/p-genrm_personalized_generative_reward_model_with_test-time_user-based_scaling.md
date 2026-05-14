---
title: >-
  [Paper Note] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling
description: >-
  [ICLR 2026][Reinforcement Learning][Personalized reward model] This paper proposes P-GenRM, the first personalized generative reward model. Through a three-stage training pipeline—PSI supervised fine-tuning to construct…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Personalized reward model"
  - "generative evaluation"
  - "structured evaluation chain"
  - "test-time scaling"
  - "collaborative filtering"
date: 2026-05-08
content_hash: ed9394794007af21
---

# P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling

**Conference**: ICLR 2026
**arXiv**: [2602.12116](https://arxiv.org/abs/2602.12116)
**Code**: [GitHub](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/Character-GenRM)
**Area**: Reinforcement Learning
**Keywords**: Personalized reward model, generative evaluation, structured evaluation chain, test-time scaling, collaborative filtering

## TL;DR

This paper proposes P-GenRM, the first personalized generative reward model. Through a three-stage training pipeline—PSI supervised fine-tuning to construct structured evaluation chains, CRE reinforcement learning to enhance reasoning under missing preference signals, and hard-negative curriculum learning to improve robustness—P-GenRM converts mixed preference signals into context-adaptive user personas and scoring rubrics. At inference time, a dual-granularity test-time scaling strategy is introduced: individual-level multi-sample aggregation and prototype-level collaborative filtering that borrows preferences from similar users. P-GenRM surpasses the previous SOTA by 2.31% on PersonalRewardBench, with test-time scaling yielding an additional ~3% gain, while generalizing to unseen users.

## Background & Motivation

**Background**: RLHF is the dominant paradigm for LLM alignment, with reward models serving as its core component by providing scoring signals to guide the policy model. As application scenarios shift from "universal value alignment" toward "personalized alignment," reward models must capture each individual user's unique preference criteria rather than learning a single set of global human preferences.

**Limitations of Prior Work**: Existing personalized reward methods face two fundamental problems. First, **static preference modeling**—user preferences, which are dynamic and context-dependent, are reduced to a fixed set of rules. The same user may have completely different preferences across contexts (preferring brief answers while driving, detailed discussion during casual conversation), and fixed rules cannot accommodate such variation. Although SynthesizeMe infers synthetic personas from historical preferences, these personas are static and do not adapt to context after generation. Second, **poor generalization to new users**—in cold-start scenarios with very limited interaction history, existing methods struggle to construct reliable reward signals from sparse feedback. Methods such as GPO, VPL, and PAL all require sufficient user data to function effectively.

**Key Challenge**: Personalized reward modeling requires fine-grained understanding of user preferences, yet preference signals are inherently sparse and noisy. Explicit preferences (e.g., "I prefer concise responses") are rarely volunteered by users, while implicit preferences derived from interaction history are abundant but noisy. The core challenge is: how to reliably infer context-adaptive evaluation criteria from such mixed signals, and how to produce reasonable scores when user information is extremely limited?

**Key Insight**: Generative reward models (GenRM) do not merely output a scalar score; they generate a complete evaluation chain—including user persona inference, rubric formulation, and item-by-item scoring. This affords three advantages: (1) the generation process itself performs reasoning, enabling dynamic adaptation to different contexts; (2) the evaluation chain is textual and inherently interpretable; (3) multiple samples can be drawn and aggregated at test time, analogous to test-time compute scaling in LLMs. The authors further draw on the collaborative filtering paradigm from recommender systems—similar users share similar preferences—clustering users into prototypes so that new users can obtain reliable scores through prototype transfer.

**Core Idea**: Use a generative reward model to convert mixed preference signals into context-adaptive evaluation chains, and reduce noise while enhancing generalization through dual-granularity test-time scaling at both the individual and prototype levels.

## Method

### Overall Architecture

P-GenRM takes as input the current query $q_t$, the user's implicit preference history $H_t^{(u)}$ (chosen/rejected response pairs from several interaction turns), optional explicit preference criteria $E^{(u)}$, and the candidate responses to be scored. The model outputs a complete Structured Evaluation Chain (SEC): it first infers the user's persona in the current context, then derives weighted scoring rubrics from the persona, evaluates each candidate response criterion by criterion, and produces a final score. Training proceeds in three stages (SFT → RL → curriculum learning), and dual-granularity scaling is applied at inference time to further improve accuracy.

### Key Designs

1. **PSI (Persona-guided Scoring Induction) — Supervised Fine-tuning Stage**:

    - Function: Trains the model to generate complete structured evaluation chains from mixed preference signals.
    - Mechanism: Strong models such as o3 are first used to construct an SEC dataset—given a user's implicit history and explicit criteria, the model infers a context-aware persona, derives preference dimensions and weights for the current scenario, scores each candidate, and produces a final result. After filtering low-quality samples via rejection sampling, the data is used for SFT. The key innovation is that personas are **dynamically generated**: the same user produces different personas and rubrics for different queries, rather than a one-time static persona as in SynthesizeMe.
    - Design Motivation: Preliminary experiments (Table 6) show that the user persona, used as a preference prior, contributes the largest accuracy gain (+1.6%), surpassing self-descriptions, demographic signals, and other inputs. Embedding persona inference within the generation process—rather than treating it as a fixed input—allows the model to flexibly adapt to the current context.

2. **CRE (Criteria-based Reasoning Enhancement) — Reinforcement Learning Stage**:

    - Function: Enhances evaluation chain generation quality in scenarios lacking explicit preferences.
    - Mechanism: Built on the GRPO algorithm, this stage introduces a dual reward: a **process reward** ($PR_t$), assessed by an LLM judge on whether the generated evaluation chain covers the user's true preference dimensions (scored 0–1); and an **outcome reward** ($OR_t$), which is binary (0/1) based on whether the final scores correctly rank chosen over rejected responses (with a −0.1 penalty for formatting errors). The total reward is $R_t = \alpha \cdot PR_t + \beta \cdot OR_t$ ($\alpha=0.5, \beta=1.0$). Training deliberately provides only limited interaction history without explicit preferences, forcing the model to infer preferences from sparse signals.
    - Design Motivation: Imitation learning via SFT alone is insufficient—the model may learn templated evaluation chains without deep reasoning. The RL stage jointly optimizes process and outcome, ensuring that evaluation chains both accurately capture user preferences (process reward) and lead to correct rankings (outcome reward). Ablation experiments confirm that removing either reward causes significant performance degradation.

3. **Hard-Negative Curriculum Learning**:

    - Function: Improves the model's ability to distinguish response pairs that are highly similar in quality but differ in alignment with user preferences.
    - Mechanism: The proportion of hard negatives in training is gradually increased—these are responses of comparable quality that do not match a particular user's preferences. To enlarge the exploration space for hard samples, the process reward $PR_t$ is removed in this stage, retaining only the outcome reward $OR_t$.
    - Design Motivation: Personalized scoring is inherently highly subjective; in many cases the quality difference between two responses is minimal and the distinction lies solely in whether each matches the user's unique preferences. The curriculum learning strategy guides the model from easy to hard discrimination, improving robustness.

### Test-time User-based Scaling

After training, P-GenRM exploits the natural scalability of GenRM during inference via dual-granularity scaling:

**Offline Phase — Prototype Initialization and Refinement**: Qwen3-Embedding-0.6B is used to encode each user's persona $P_t^{(u)}$ across different contexts; the concatenated cross-context preference embedding matrix $\mathbf{P}$ is then clustered via K-means into $k$ user prototypes. Prototype refinement is performed through an attention mechanism—prototype-augmented attention is used to weight and aggregate user history, combined with a discriminative loss (enabling prototypes to distinguish chosen from rejected responses) and two regularization terms to prevent excessive prototype drift. PCA analysis shows that 50 prototypes suffice to capture the vast majority of preference variation across users.

**Individual-level Scaling**: For a given user's current query, P-GenRM samples $m$ evaluation plans in parallel, each potentially producing slightly different inferred personas and rubrics; the final score is the average across samples. This amounts to multi-hypothesis exploration in the preference inference space, reducing the noise of any single inference.

**Prototype-level Scaling**: Based on the user's preference embedding, the nearest prototype is identified, and $n$ most similar users under that prototype are selected. Their preference histories are used to prompt P-GenRM to generate $n$ additional scoring plans. The final score aggregates both individual-level and prototype-level results. This approach draws on the core insight of collaborative filtering—similar users share similar preferences—and is particularly beneficial for new users in cold-start scenarios.

### Loss & Training

The three stages proceed sequentially: (1) the PSI stage uses standard SFT cross-entropy loss; (2) the CRE stage uses the GRPO objective with total reward $R_t = 0.5 \cdot PR_t + 1.0 \cdot OR_t$ and KL regularization to prevent excessive deviation from the reference policy; (3) the curriculum learning stage retains the GRPO framework but removes $PR_t$, keeping only $OR_t$, while progressively increasing the proportion of hard negatives. The prototype refinement stage uses a discriminative pairwise loss $\mathcal{L}_{\text{pair}} = -\log\sigma(z_t^\top y_t^+ - z_t^\top y_t^-)$, together with centroid regularization and temporal smoothness regularization.

## Key Experimental Results

### Main Results — Comparison on PersonalRewardBench

| Method | Model | Chatbot Arena | PRISM |
|------|------|:---:|:---:|
| Default (LLM-as-Judge) | 8B | 56.37% | 52.04% |
| + Preference History | 8B | 58.53% | 56.24% |
| + SynthesizeMe | 8B | 61.07% | 54.70% |
| GPO | 8B | 57.87% | 57.29% |
| VPL | 8B | 58.12% | 58.25% |
| FT RM + SynthesizeMe | 8B | 69.78% | 62.84% |
| **P-GenRM** | **8B** | **72.68%** | **65.32%** |
| **P-GenRM + Ind-16, Pro-8** | **8B** | **75.92%** | **68.06%** |
| FT RM + SynthesizeMe | 70B | 72.05% | 63.74% |
| P-GenRM | 70B | 73.42% | 66.21% |
| o3 + PSI | — | 69.14% | 63.87% |

P-GenRM-8B outperforms the previous SOTA (FT RM + SynthesizeMe-70B) by an average of 1.04%; with test-time scaling, a further ~3% gain is achieved. The 8B model even surpasses SynthesizeMe at the 70B scale.

### Ablation Study

| Configuration | Chatbot Arena | PRISM | Notes |
|------|:---:|:---:|------|
| P-GenRM (Full) | 72.68% | 65.32% | Complete model |
| w/o CL | 71.07% | 63.82% | Removing curriculum learning: −1.5–1.6% |
| w/o CL, PR | 70.22% | 62.70% | Further removing process reward: −0.8–1.1% |
| w/o CL, OR | 69.05% | 60.94% | Removing outcome reward causes larger drop than removing process reward |
| w/o CL, RL | 66.76% | 57.08% | Removing entire RL stage: −6–8% |
| w/o CL, RL, SFT | 56.37% | 52.04% | Degenerates to baseline LLM-as-Judge |

### Detailed Test-time Scaling Analysis

| Scaling Configuration | Chatbot Arena | PRISM |
|------|:---:|:---:|
| P-GenRM (no scaling) | 72.68% | 65.32% |
| + Ind-8 | 73.61% | 65.79% |
| + Ind-16 | 73.87% | 66.66% |
| + Ind-32 | 75.59% | 67.65% |
| + Ind-8, Pro-4 | 74.30% | 67.54% |
| **+ Ind-16, Pro-8** | **75.92%** | **68.06%** |
| + Ind-0, Pro-8 | 66.90% | 57.65% |
| + Ind-16, Pro-16 | 72.59% | 64.61% |

### OOD Generalization (LaMP-QA Cold-start)

| Method | Arts | Personal | Society | Avg |
|------|:---:|:---:|:---:|:---:|
| Qwen3-235B-A22B | 0.600 | 0.657 | 0.600 | 0.619 |
| SynthesizeMe-8B | 0.486 | 0.657 | 0.600 | 0.581 |
| LLaMA3.1-70B | 0.543 | 0.657 | 0.600 | 0.600 |
| **P-GenRM-8B + Ind-8, Pro-4** | **0.543** | **0.714** | **0.657** | **0.638** |

### Key Findings

- **RL is the largest contributor**: Removing the entire RL stage causes a 6–8% drop, demonstrating that SFT-based imitation of evaluation chains alone is far from sufficient; the outcome reward is more critical than the process reward (removing OR causes a larger performance decrease).
- **Prototype-level scaling helps new users most, but more is not always better**: Ind-16+Pro-8 is the optimal configuration (24 total inference calls), yet Pro-16 underperforms Pro-8—too many similar users introduce noisy preferences inconsistent with the target user.
- **Prototype-only scaling is ineffective**: Ind-0+Pro-8 drops to 66.90%/57.65%, well below the no-scaling baseline, confirming that the user's own preferences must remain central to the scoring.
- **Dynamic vs. static personas**: Under the LLM-as-Judge setting, PSI consistently outperforms SynthesizeMe across all base models (Qwen3-8B: +1.65/+1.68; o3: +1.41/+5.38), validating the necessity of context-adaptive persona generation.
- **Strong cross-distribution generalization**: In the LaMP-QA cold-start setting, P-GenRM-8B outperforms Qwen3-235B, demonstrating that the prototype transfer mechanism is genuinely effective for new users.
- **No majority-group bias**: Prototype-level macro accuracy (65.21%) is nearly identical to sample-level accuracy (65.32%), a difference of only 0.11%, indicating that minority-group users are not neglected under long-tail distributions.

## Highlights & Insights

- **Evaluation chains as debuggable reward signals**: Traditional reward models output a scalar with no explanation of why a response scores highly. P-GenRM outputs the complete reasoning process (persona → rubric → item-by-item scoring), allowing users and developers to inspect each step for correctness—this is especially valuable for the highly subjective nature of personalized scoring.
- **Collaborative filtering transplanted into RLHF**: The core assumption of recommender systems—"similar users share similar preferences"—has historically been confined to the recommendation domain. This paper is the first to introduce it into reward modeling, addressing the cold-start problem through user prototype clustering and prototype-based transfer. This idea is transferable to any scenario requiring personalized evaluation (e.g., personalized summarization, personalized educational feedback).
- **Quality over quantity in test-time scaling**: Simply increasing individual sampling count (Ind-32) is less effective than combining individual and prototype scaling (Ind-16+Pro-8), which achieves better results with fewer total inference calls. This demonstrates that diversity (incorporating the perspectives of different users) is more valuable than repetition (multiple samples from the same user).
- **Clear progressive logic in three-stage training**: SFT teaches format and basic capability → RL develops deep reasoning ability → curriculum learning improves discrimination of hard samples. Each stage addresses a specific bottleneck left by the previous one, rather than merely stacking components.

## Limitations & Future Work

- **Prototype count requires manual selection**: The number of prototypes is currently set to 50 via PCA variance retention analysis, without an adaptive mechanism. The optimal number of prototypes may vary substantially across different data distributions.
- **Inference cost remains high**: The optimal configuration (Ind-16+Pro-8) requires 24 complete generation passes per sample. Although the authors report lower latency than the previous SOTA, this remains demanding for real-time conversational applications.
- **Preference drift is not modeled**: User preferences evolve over time (short-term vs. long-term preferences), but the current framework randomly samples from interaction history without accounting for recency, making it unable to capture preference change trends.
- **Evaluation benchmarks are limited**: Evaluation is primarily conducted on PersonalRewardBench (Chatbot Arena + PRISM) and LaMP-QA; validation on real production-level personalized dialogue systems is lacking.
- **The embedding model for prototype refinement is fixed**: Qwen3-Embedding-0.6B is used for user embeddings, but whether this embedding truly captures "preference similarity" rather than "textual similarity" warrants further investigation.

## Related Work & Insights

- **vs. SynthesizeMe**: SynthesizeMe synthesizes a static persona from historical preferences and uses it as a prompt; PSI in this paper dynamically generates a context-adaptive persona at each scoring step. P-GenRM consistently outperforms SynthesizeMe across all base models, with larger margins on smaller models.
- **vs. PAL/VPL/GPO**: These methods use latent variables or prototype mixtures to model user preferences, but are all based on traditional discriminative reward models that output scalars. P-GenRM generates evaluation chains via a generative approach, inherently supporting test-time scaling and interpretability—capabilities that discriminative methods cannot provide.
- **vs. GenRM series (Self-Principled Critique, etc.)**: Existing GenRM work focuses on improving general evaluation quality without considering personalization. P-GenRM is the first work to combine GenRM with personalized alignment, filling this gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First personalized GenRM; an organic combination of collaborative filtering and RLHF; the dual-granularity test-time scaling design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks + OOD generalization + detailed ablations + scaling configuration analysis + macro accuracy fairness verification.
- Writing Quality: ⭐⭐⭐⭐ Overall structure is clear, though notation is somewhat dense and some descriptions could be more concise.
- Value: ⭐⭐⭐⭐⭐ Personalized alignment is a core requirement for deploying LLMs in practice; P-GenRM offers an interpretable and scalable paradigm.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution to personalized alignment of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)

</div>

<!-- RELATED:END -->
