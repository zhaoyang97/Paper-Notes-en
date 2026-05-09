---
title: >-
  [Paper Note] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization
description: >-
  [ACL 2026][Model Compression][Automatic prompt optimization] This work formalizes label-free prompt optimization as a dueling bandit problem and proposes the Prompt Duel Optimizer (PDO), which employs Double Thompson Sampling to efficiently select the most informative prompt pairs for comparison. Combined with a top-performer mutation strategy to expand the search space, PDO identifies stronger prompts on BBH and MS MARCO with fewer judge calls than existing baselines.
tags:
  - ACL 2026
  - Model Compression
  - Automatic prompt optimization
  - label-free optimization
  - dueling bandit
  - LLM-as-judge
  - Thompson sampling
date: 2026-05-08
content_hash: a0a491670d9d79bf
---

# LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization

**Conference**: ACL 2026
**arXiv**: [2510.13907](https://arxiv.org/abs/2510.13907)
**Code**: [GitHub](https://github.com/meta-llama/prompt-ops)
**Area**: Model Compression
**Keywords**: Automatic prompt optimization, label-free optimization, dueling bandit, LLM-as-judge, Thompson sampling

## TL;DR
This work formalizes label-free prompt optimization as a dueling bandit problem and proposes the Prompt Duel Optimizer (PDO), which employs Double Thompson Sampling to efficiently select the most informative prompt pairs for comparison. Combined with a top-performer mutation strategy to expand the search space, PDO identifies stronger prompts on BBH and MS MARCO with fewer judge calls than existing baselines.

## Background & Motivation

**Background**: Automatic prompt optimization (APO) discovers high-performing instructions through iterative generation and evaluation of candidate prompts, achieving strong results across diverse tasks. Existing methods such as APE, OPRO, and Breeder perform well when labeled validation sets are available.

**Limitations of Prior Work**: The vast majority of APO methods rely on ground-truth labels to score candidate prompts. In practice, however, obtaining large-scale annotated data is costly and time-consuming. In industrial settings, for instance, practitioners often need to deploy LLM-based classification services before large-scale human annotation is available, creating an urgent need for label-free prompt optimization.

**Key Challenge**: In label-free settings, an LLM can serve as a judge via pairwise preference comparisons, but two problems arise: (1) LLM judges are noisy—subject to call-level stochasticity, position bias, and verbosity bias; and (2) the cost of pairwise comparisons scales quadratically with the number of candidates, making exhaustive comparison infeasible.

**Goal**: To efficiently identify the optimal prompt under a constrained judge budget—minimizing the number of comparisons while remaining robust to judge noise.

**Key Insight**: Modeling prompt selection as a dueling bandit problem enables Bayesian sampling strategies to concentrate comparisons on the most informative prompt pairs, while a mutation strategy continuously explores new candidates.

**Core Idea**: Double Thompson Sampling for efficient pairwise comparison combined with top-candidate mutation to expand the search space, unifying two levels of optimization—identifying the best prompt within a fixed pool and exploring beyond it—within a single framework.

## Method

### Overall Architecture
At each round, PDO: (1) applies D-TS to select the two most informative prompts from the candidate pool for comparison; (2) conducts pairwise LLM-judge comparisons on a batch of unlabeled examples and records win/loss outcomes; and (3) every $M$ rounds, mutates the current Copeland winner to generate new candidates and adds them to the pool. The prompt with the highest Copeland score is returned at termination.

### Key Designs

1. **Double Thompson Sampling (D-TS) for Prompt Selection**:

    - **Function**: Concentrates the comparison budget on the most informative prompt pairs under a limited budget.
    - **Mechanism**: Maintains a Beta posterior $\theta_{ij} \sim \text{Beta}(W_{ij}+1, W_{ji}+1)$ for each prompt pair $(p_i, p_j)$. The first step uses an optimistic Copeland score to filter candidates and Thompson-samples the first prompt; the second step samples the second prompt exclusively from "uncertain opponents."
    - **Design Motivation**: Compared to random pairing or UCB strategies, D-TS naturally allocates the comparison budget to contests with undecided outcomes, with a theoretical Copeland regret guarantee of $O(K^2 \log T)$.

2. **Top-Performer-Guided Mutation**:

    - **Function**: Continuously expands the candidate pool to explore regions of better prompts.
    - **Mechanism**: Every $M$ rounds, the prompt with the highest current Copeland score is selected and mutated via template editing, textual gradient guidance, or LLM rewriting to produce variants that are added to the pool. Weak candidates may be pruned simultaneously.
    - **Design Motivation**: D-TS can only identify the optimum within a fixed pool; mutation enables the search to progressively zoom into higher-quality prompt regions, analogous to zooming-in strategies in Lipschitz bandit settings.

3. **LLM Judge Design and Debiasing**:

    - **Function**: Provides reliable label-free pairwise preference signals.
    - **Mechanism**: For multiple-choice tasks, a "dual-judgment" protocol is applied—selecting the correct answer when outputs differ, and comparing reasoning quality when they agree. For open-ended tasks, outputs are scored along four dimensions: accuracy, completeness, relevance, and clarity. The order of the two outputs is randomly swapped to mitigate position bias.
    - **Design Motivation**: Noise and bias in LLM judges represent the central bottleneck in label-free optimization; careful judge protocol design is essential to ensure signal quality.

### Loss & Training
PDO involves no model training and operates as a black-box optimization framework. The core optimization objective is maximization of the Copeland score: $C(i) = |\{j \neq i : \mu(i,j) > \frac{1}{2}\}|$.

## Key Experimental Results

### Main Results

| Dataset | Metric | PDO | SPO | CoT | PoS | Gain |
|--------|------|-----|-----|-----|-----|------|
| BBH (16 tasks) | # best tasks | **13/16** | 1/16 | 1/16 | 2/16 | Overwhelming |
| BBH-Tracking7 | Accuracy | 0.641 | 0.543 | 0.532 | 0.538 | +9.8pp |
| BBH-Web of Lies | Accuracy | 0.942 | 0.818 | 0.796 | 0.861 | +8.1pp |
| BBH-Navigate | Accuracy | 0.900 | 0.874 | 0.878 | 0.866 | +2.2pp |
| MS MARCO (4 tasks) | Convergence speed | Fastest | Slower | - | - | Surpasses baselines within a few rounds |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| D-TS sampling | Best convergence | Finds high-quality prompts faster than RUCB and Random |
| RUCB replacing D-TS | Slower convergence | UCB strategy less flexible than Bayesian sampling |
| Random sampling | Slowest convergence | Unguided random pairing wastes the comparison budget |
| Cross-model-family validation | Robust results | PDO advantage persists when re-evaluated with different judge models |

### Key Findings
- D-TS sampling efficiency substantially outperforms RUCB and random sampling, surpassing the SPO baseline on MS MARCO within only a few rounds.
- PDO also performs well in labeled settings, confirming that the prompts it discovers are intrinsically high quality rather than artifacts of the evaluation protocol.
- Judge noise correlates with task difficulty—judges are more reliable on simpler tasks (e.g., Navigate) and noisier on harder ones (e.g., Geometric Shapes).
- Cross-judge-family validation demonstrates that PDO's advantage does not depend on any particular judge model.

## Highlights & Insights
- **Novel dueling bandit framing**: Recasting prompt optimization from "score-based ranking" to "pairwise comparison" naturally aligns with the output format of LLM judges and avoids the calibration issues associated with pointwise scoring.
- **Clean separation of two optimization levels**: D-TS handles efficient in-pool identification while mutation handles out-of-pool exploration; the two components have clearly defined roles with theoretical grounding.
- **Strong practical utility**: The method requires no labeled data whatsoever, making it well-suited for cold-start industrial deployment scenarios; code is publicly available in Meta's prompt-ops repository.

## Limitations & Future Work
- The method depends on the quality of the LLM judge—if the judge is unreliable for certain task types, PDO's advantage diminishes accordingly.
- The current mutation strategy is relatively simple (LLM-based rewriting); more structured search over the prompt space could yield further gains.
- Computational scalability when the candidate pool grows large is not addressed—Copeland score computation scales linearly with pool size.
- In extreme noise regimes, the convergence guarantees of D-TS may be insufficient, motivating the use of more robust statistical tests.

## Related Work & Insights
- **vs. SPO (Xiang et al. 2025)**: SPO also uses an LLM judge for label-free optimization but relies on simple iterative comparison selection without exploiting the sampling efficiency of bandit theory. PDO identifies superior prompts under the same budget.
- **vs. OPRO (Yang et al. 2024)**: OPRO requires a labeled validation set and generates prompts directly via model scoring rather than pairwise comparison selection; the two approaches are complementary.
- **vs. EvoPrompt (Fernando et al. 2023)**: EvoPrompt's evolutionary strategy inspired PDO's mutation mechanism, but EvoPrompt requires labeled data for fitness evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of dueling bandits for prompt optimization is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 16 BBH tasks and 4 MS MARCO tasks with multiple baselines and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical motivation and experimental design are both exceptionally clear.
- Value: ⭐⭐⭐⭐ Label-free prompt optimization addresses a genuine practical need, and the framework is broadly applicable.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ICLR 2026\] HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design](../../ICLR2026/model_compression/hifo-prompt_prompting_with_hindsight_and_foresight_for_llm-based_automatic_heuri.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](../../NeurIPS2025/model_compression/graph_your_own_prompt.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

</div>

<!-- RELATED:END -->
