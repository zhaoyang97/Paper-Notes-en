---
title: >-
  [Paper Note] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization
description: >-
  [ACL 2026][Model Compression][Automated Prompt Optimization] This paper formulates label-free prompt optimization as a dueling bandit problem and proposes Prompt Duel Optimizer (PDO). By utilizing Double Thompson Samplin…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Automated Prompt Optimization"
  - "Label-Free Optimization"
  - "Dueling Bandits"
  - "LLM Evaluation"
  - "Thompson Sampling"
date: 2026-05-08
content_hash: f4b88fa26fc72585
---

# LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.13907](https://arxiv.org/abs/2510.13907)  
**Code**: [GitHub](https://github.com/meta-llama/prompt-ops)  
**Area**: Model Compression  
**Keywords**: Automated Prompt Optimization, Label-Free Optimization, Dueling Bandits, LLM Evaluation, Thompson Sampling

## TL;DR
This paper formulates label-free prompt optimization as a dueling bandit problem and proposes Prompt Duel Optimizer (PDO). By utilizing Double Thompson Sampling to efficiently select the most informative prompt pairs for comparison and combining it with a top-performer mutation strategy to expand the search space, PDO finds stronger prompts with fewer judge calls on BBH and MS MARCO.

## Background & Motivation

**Background**: Automated Prompt Optimization (APO) discovers high-performance instructions by iteratively generating and evaluating candidate prompts. Existing methods such as APE, OPRO, and Breeder have performed well given annotated validation sets.

**Limitations of Prior Work**: The vast majority of APO methods rely on ground-truth labels to score candidate prompts. However, in practical deployment, acquiring large-scale annotated data is expensive and slow. For example, in industrial scenarios, users often need to deploy LLM classification services before large-scale manual annotation is available, necessitating label-free prompt optimization solutions.

**Key Challenge**: In label-free scenarios, an LLM can serve as a judge for pairwise preference comparisons, but two issues arise: (1) LLM judges are noisy—suffering from execution uncertainty, position bias, and verbosity bias; (2) the overhead of pairwise comparisons scales quadratically with the number of candidates, making exhaustive comparison infeasible.

**Goal**: To efficiently find the optimal prompt under a restricted judge budget—reducing the number of comparisons while coping with judge noise.

**Key Insight**: Modeling prompt selection as a dueling bandit problem and utilizing a Bayesian sampling strategy to concentrate comparisons on the most informative prompt pairs, while continuously exploring new prompts through mutation strategies.

**Core Idea**: Using Double Thompson Sampling for efficient pairwise selection and top-candidate mutation to expand the search space, unifying two levels of optimization (identifying the best within a fixed pool and exploring outside the pool) into a single framework.

## Method

### Overall Architecture
In each round, PDO: (1) uses D-TS to select the two prompts most worth comparing from the candidate pool; (2) conducts pairwise comparisons using an LLM judge on a batch of unlabeled samples and records the outcome; (3) mutates the current Copeland champion every $M$ rounds to generate and add new candidates to the pool. Finally, the prompt with the highest Copeland score is returned.

### Key Designs

1.  **Double Thompson Sampling (D-TS) Prompt Selection**:
    - **Function**: Concentrates comparison budget on the most informative prompt pairs under limited resources.
    - **Mechanism**: Maintains a Beta posterior distribution $\theta_{ij} \sim \text{Beta}(W_{ij}+1, W_{ji}+1)$ for each pair of prompts $(p_i, p_j)$. The first step selects a candidate using an optimistic Copeland score followed by Thompson sampling; the second step samples the second prompt specifically from "uncertain opponents."
    - **Design Motivation**: Compared to random pairing or UCB strategies, D-TS naturally allocates the budget to critical "undecided" duels, theoretically guaranteeing a Copeland regret of $O(K^2 \log T)$.

2.  **Top-Performer Guided Mutation**:
    - **Function**: Continuously expands the candidate pool to explore superior prompt regions.
    - **Mechanism**: Every $M$ rounds, the prompt with the highest current Copeland score is selected and variants are generated via template editing, textual gradients, or LLM rewriting to be added to the pool. Weak candidates can also be eliminated.
    - **Design Motivation**: While D-TS finds the optimum in a fixed pool, mutation allows the search to "zoom in" on better prompt spaces, similar to the zooming-in strategy in Lipschitz bandits.

3.  **LLM Judge Design and De-biasing**:
    - **Function**: Provides reliable label-free pairwise preference signals.
    - **Mechanism**: A "dual-judgment" approach is used for multiple-choice tasks—selecting the correct answer if they differ, or comparing reasoning quality if they match. Open-ended tasks are scored across four dimensions: accuracy, completeness, relevance, and clarity. The order of the two outputs is randomly swapped to mitigate position bias.
    - **Design Motivation**: Noise and bias in LLM judges are core bottlenecks in label-free optimization; thus, a carefully designed judge protocol is required to ensure the quality of preference signals.

### Loss & Training
PDO does not involve model training but is a black-box optimization framework. The core optimization objective is to maximize the Copeland score: $C(i) = |\{j \neq i : \mu(i,j) > \frac{1}{2}\}|$.

## Key Experimental Results

### Main Results

| Dataset | Metric | PDO | SPO | CoT | PoS | Gain |
|---------|--------|-----|-----|-----|-----|------|
| BBH (16 tasks) | Best Task Count | **13/16** | 1/16 | 1/16 | 2/16 | Decisive Win |
| BBH-Tracking7 | Accuracy | 0.641 | 0.543 | 0.532 | 0.538 | +9.8pp |
| BBH-Web of Lies | Accuracy | 0.942 | 0.818 | 0.796 | 0.861 | +8.1pp |
| BBH-Navigate | Accuracy | 0.900 | 0.874 | 0.878 | 0.866 | +2.2pp |
| MS MARCO (4 tasks) | Convergence | Fastest | Slower | - | - | Surpasses baselines in few rounds |

### Ablation Study

| Configuration | Effect | Description |
|---------------|--------|-------------|
| D-TS Sampling | Best convergence | Finds high-quality prompts faster than RUCB and Random |
| RUCB instead of D-TS | Slower convergence | UCB strategies are less flexible than Bayesian sampling |
| Random Sampling | Slowest convergence | Random pairing without strategy wastes comparison budget |
| Cross-family validation | Robust results | PDO advantage persists when re-evaluated with different judge models |

### Key Findings
- The sampling efficiency of D-TS significantly outperforms RUCB and random sampling, surpassing the SPO baseline in just a few rounds on MS MARCO.
- PDO also performs well in labeled settings, proving that the discovered prompts are inherently high-quality rather than dependent on a specific evaluation mode.
- Judge noise correlates with task difficulty—judges are more reliable for simple tasks (e.g., Navigate) and noisier for difficult tasks (e.g., Geometric Shapes).
- Cross-family judge validation indicates that PDO's advantages do not depend on a specific LLM judge model.

## Highlights & Insights
- **Novel Dueling Bandit Perspective**: Transforming prompt optimization from "point-wise scoring" to "pairwise comparison" naturally fits the output format of LLM judges and avoids calibration issues associated with absolute scoring.
- **Two-Layer Optimization Decoupling**: D-TS handles efficient identification within the pool while mutation handles exploration outside the pool, providing a clear division of labor with theoretical grounding.
- **High Practicality**: Requires no labeled data, making it ideal for cold-start scenarios in early industrial deployment. The code is open-sourced in Meta's prompt-ops repository.

## Limitations & Future Work
- Dependency on LLM judge quality—if a judge has poor judgment on a specific task type, the advantages of PDO may diminish.
- Mutation strategies are currently simple (LLM-based rewriting); more structured prompt space searches could provide further improvements.
- Computational scalability for extremely large candidate pools was not discussed—Copeland score calculation grows linearly with pool size.
- In extreme noise scenarios, the convergence guarantees of D-TS might be insufficient, requiring more robust statistical testing.

## Related Work & Insights
- **vs SPO (Xiang et al. 2025)**: SPO also uses LLM judges for label-free optimization but employs simple iterative comparison selection without utilizing the sampling efficiency of bandit theory. PDO finds superior prompts under the same budget.
- **vs OPRO (Yang et al. 2024)**: OPRO requires an annotated validation set and uses the model to generate prompts directly rather than pairwise selection; the two are complementary.
- **vs EvoPrompt (Fernando et al. 2023)**: EvoPrompt's evolutionary strategy inspired the mutation mechanism in PDO, but EvoPrompt requires labeled data for fitness evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of dueling bandits for prompt optimization is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on 16 BBH tasks and 4 MS MARCO tasks with various baselines and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Both theoretical motivation and experimental design are very clear.
- Value: ⭐⭐⭐⭐ Label-free prompt optimization addresses a real-world need with a highly generalizable framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ICLR 2026\] HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design](../../ICLR2026/model_compression/hifo-prompt_prompting_with_hindsight_and_foresight_for_llm-based_automatic_heuri.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](../../NeurIPS2025/model_compression/graph_your_own_prompt.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)

</div>

<!-- RELATED:END -->
