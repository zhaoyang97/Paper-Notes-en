---
title: >-
  [Paper Note] Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
description: >-
  [ICLR 2026][Medical Imaging][Information-seeking agents] This paper introduces the Collaborative Battleship task to evaluate the information-seeking capabilities of language models…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Information-seeking agents"
  - "Bayesian experimental design"
  - "language model reasoning"
  - "exploration-exploitation trade-off"
  - "Monte Carlo inference"
date: 2026-05-08
content_hash: 5c553feb7a841669
---

# Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People

**Conference**: ICLR 2026
**arXiv**: [2510.20886](https://arxiv.org/abs/2510.20886)  
**Code**: [Project Page](https://gabegrand.github.io/battleship)  
**Area**: Medical Imaging (classified as such; actually AI Agent / Cognitive Science)
**Keywords**: Information-seeking agents, Bayesian experimental design, language model reasoning, exploration-exploitation trade-off, Monte Carlo inference

## TL;DR
This paper introduces the Collaborative Battleship task to evaluate the information-seeking capabilities of language models, and proposes three Bayesian inference strategies (Bayes-Q/M/D) to enhance LM questioning, action selection, and decision-making. The approach enables a weak model (Llama-4-Scout) to achieve superhuman performance (82% win rate) at approximately 1% the cost of GPT-5.

## Background & Motivation
- Many AI applications (scientific discovery, medical diagnosis) require agents to strategically acquire information: forming hypotheses, asking targeted questions, and making decisions under uncertainty.
- Current language models are primarily optimized to answer user questions, but can they formulate good questions on their own?
- There is a need to evaluate and improve frontier models' ability to ask goal-directed questions and take actions in dynamic environments.
- **Core Motivation**: Drawing on the cognitive science theory of resource rationality, the paper augments LMs' information-seeking capabilities using Bayesian experimental design.

## Method

### Overall Architecture
1. Design the Collaborative Battleship two-player cooperative task: the Captain (partial information) balances exploration (asking questions) and exploitation (shooting), while the Spotter (full information) provides yes/no answers.
2. Collect 126 games of human behavioral data ($N=42$) to establish the BattleshipQA benchmark.
3. Propose three augmentation strategies based on the Bayesian experimental design framework.
4. Validate generalization on the Guess Who? task.

### Key Designs

1. **Bayesian Belief Update Framework**:

    - Hidden board $S \in \mathcal{S}$, belief $\pi_t(s) = \Pr(S=s | x, \mathcal{H}_{1:t})$
    - Noisy observation model: the Spotter is modeled as a binary symmetric channel $\text{BSC}(\varepsilon)$, with $\varepsilon=0.1$
    - Bayesian update: $\pi_{t+1}(s) \propto \pi_t(s)[(1-\varepsilon)\mathbf{1}\{\tilde{a}_t = f_{q_t}(s)\} + \varepsilon\mathbf{1}\{\tilde{a}_t \neq f_{q_t}(s)\}]$
    - Implemented via Sequential Monte Carlo (SMC) approximation
    - **Design Motivation**: Exact summation is intractable; particle approximation is both efficient and capable of handling Spotter noise.

2. **Three Bayesian Strategies**:

    - **Bayes-Q (Questioning)**: Samples a candidate question set $\mathcal{Q}$ from the LM, then selects the question with the highest expected information gain (EIG): $q_t^* = \arg\max_{q \in \mathcal{Q}} \text{EIG}_\varepsilon(q | x, \mathcal{H}_{1:t})$
    - Closed-form EIG: $\text{EIG}_\varepsilon = H_b(\varepsilon + (1-2\varepsilon)p_t) - H_b(\varepsilon)$, maximized when $p_t \approx 1/2$
    - **Bayes-M (Action)**: Selects the cell with the highest hit probability: $u_t^* = \arg\max_u p_t^{\text{hit}}(u | x, \mathcal{H}_{1:t})$
    - **Bayes-D (Decision)**: One-step lookahead decision — ask a question if $\gamma \cdot \widehat{p_{t+1}^{\text{hit}}}(q_t^*) > p_t^{\text{hit}}(u_t^*)$, otherwise shoot.
    - **Design Motivation**: Combines LMs' natural language capabilities with Bayesian-optimal inference, leveraging the strengths of each.

3. **Code Generation for Answer Grounding (SpotterQA)**:

    - Translates natural language questions into Python programs executed over the hypothesis space.
    - Improves accuracy by 14.7% over direct answering and CoT baselines.
    - **Design Motivation**: Code generation provides precise formal grounding for otherwise ambiguous natural language questions.

### Loss & Training
- No training is involved — all strategies are inference-time methods.
- SMC particle approximation is used for Bayesian belief maintenance.
- $\gamma = 0.95$ introduces a slight bias toward immediate action.
- Up to 10 candidate questions are sampled for EIG ranking.

## Key Experimental Results

### Main Results (CaptainQA — Full Game)

| Captain Strategy | Llama-4-Scout F1 | GPT-4o F1 | GPT-5 F1 | Notes |
|-----------------|-----------------|-----------|----------|-------|
| LM only | 0.367 | 0.450 | 0.716 | Pure LM baseline |
| +Bayes-Q | 0.388 | 0.476 | 0.717 | Questioning only |
| +Bayes-M | 0.621 | 0.663 | 0.731 | Action only |
| +Bayes-QM | 0.733 | 0.753 | 0.734 | Questioning + Action |
| +Bayes-QMD | 0.764 | 0.782 | — | Full augmentation (superhuman) |
| Human average | — | — | — | F1 ≈ 0.6–0.7 |

### SpotterQA Answer Accuracy

| Model | Base | CoT+Code | Gain |
|-------|------|----------|------|
| GPT-4.1 | 75.2% | 90.9% | +15.7% |
| Claude 4 Opus | 86.8% | 94.4% | +7.6% |
| Llama-4-Scout | 62.2% | — | — |
| Human | 92.5% | — | — |

### Key Findings
- **Weak models reach superhuman performance via Bayesian augmentation**: Llama-4-Scout's win rate jumps from 8% to 82% against humans and from 0% to 67% against GPT-5, at approximately 1% of GPT-5's cost.
- **High-EIG questioning alone is insufficient**: Bayes-Q alone improves EIG but yields only marginal gains in game performance; the action augmentation from Bayes-M is the critical factor.
- **Elimination of redundant questions**: Bayes-Q reduces Llama-4-Scout's zero-information-gain questions from 18.5% to 0.2%.
- **GPT-5 already employs efficient internal strategies**: Bayesian augmentation yields negligible benefit for GPT-5, suggesting that similar reasoning is already implemented internally.
- **Skilled players ask first but do not ask exclusively**: Both humans and GPT-5 ask an average of only 8 questions (out of a maximum of 15), but each question carries substantially more information.

## Highlights & Insights
- Elegant experimental design: a classic game (Battleship) serves as a controlled testbed for Bayesian experimental design.
- The resource rationality perspective is distinctive — rather than pursuing global optimality, the goal is to maximize utility under limited resources.
- A compelling demonstration of inference-time scaling: significant performance improvements are achieved through sampling and reranking alone, without any model training.
- Code generation as a grounding mechanism transforms ambiguous natural language questions into executable programs.

## Limitations & Future Work
- The environment is relatively simple (8×8 board); generalization to complex real-world scenarios requires further investigation.
- The Spotter noise parameter $\varepsilon$ is fixed at 0.1; adaptive estimation would be more appropriate in practice.
- Bayesian strategies depend on a world model that supports efficient sampling; domains without hand-coded implementations would require learned generative models.
- Pragmatic aspects of dialogue are not modeled — context-dependent questions in human conversation remain a challenge.
- Generalization to Guess Who? is encouraging but remains limited in task complexity.

## Related Work & Insights
- The Battleship task originates from cognitive science (Gureckis 2009; Rothe 2017–2019); this paper is the first to extend it to multi-turn dialogue and full gameplay.
- The work instantiates classical Bayesian experimental design (BED) theory (Lindley 1956; Chaloner 1995) in a modern LM setting.
- Resource rationality theory (Anderson 1990; Lieder 2020) motivates the strategy design — pursuing "good enough" solutions rather than strict optimality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A distinctive intersection of cognitive science, BED, and language models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Human experiments + 15 LMs + ablations + Guess Who? generalization.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative with seamless integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Significant implications for building rational information-seeking agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](../../CVPR2026/medical_imaging/act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[CVPR 2026\] X-WIN: Building Chest Radiograph World Model via Predictive Sensing](../../CVPR2026/medical_imaging/x-win_building_chest_radiograph_world_model_via_predictive_sensing.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](../../CVPR2026/medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](../../AAAI2026/medical_imaging/do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)

</div>

<!-- RELATED:END -->
