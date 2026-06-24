---
title: >-
  [Paper Note] Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation
description: >-
  [ACL2025][Multilingual & Machine Translation][Multilingual Translation] The Trans-Zero self-play framework is proposed, which utilizes only monolingual data. By exploring semantically consistent candidate translations during the multilingual translation process through Genetic Monte-Carlo Tree Search (G-MCTS) and combining this with preference optimization, it achieves parallel-data-free multilingual translation training with performance comparable to large-scale supervised f…
tags:
  - "ACL2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Translation"
  - "Self-Play"
  - "Monte-Carlo Tree Search"
  - "Preference Optimization"
  - "Parallel-Data-Free"
date: 2026-05-08
content_hash: 963281b89a30f277
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation

**Conference**: ACL2025  
**arXiv**: [2504.14669](https://arxiv.org/abs/2504.14669)  
**Code**: [NJUNLP/trans0](https://github.com/NJUNLP/trans0)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Translation, Self-Play, Monte-Carlo Tree Search, Preference Optimization, Parallel-Data-Free

## TL;DR
The Trans-Zero self-play framework is proposed, which utilizes only monolingual data. By exploring semantically consistent candidate translations during the multilingual translation process through Genetic Monte-Carlo Tree Search (G-MCTS) and combining this with preference optimization, it achieves parallel-data-free multilingual translation training with performance comparable to large-scale supervised fine-tuning (SFT) methods.

## Background & Motivation
1. **Dependency of multilingual translation on parallel data**: Current LLM multilingual translation still requires a large amount of parallel corpus for SFT, whereas parallel data for low-resource languages is extremely scarce, limiting translation coverage.
2. **Catastrophic forgetting**: As the scale of multilingual SFT expands, one-to-one MLE supervision introduces bias, and excessive multilingual annotation dilutes pre-trained knowledge, leading to a degradation in cross-lingual performance.
3. **Scalability bottleneck of MoE solutions**: Existing Mixture-of-Experts (MoE) methods use hand-crafted language module routing, but routing complexity and distributed overhead grow exponentially with the number of translation directions.
4. **Underutilization of LLMs' intrinsic multilingual knowledge**: LLMs accumulate rich multilingual knowledge during pre-training, but existing methods fail to effectively stimulate these inherent capabilities for self-improvement.
5. **Technical challenges of cross-lingual exploration**: Systematic exploration of cross-lingual semantic space requires planning methods beyond simple prompt engineering, making traditional LLM inference paradigms difficult to apply directly.
6. **External dependencies of multilingual quality evaluation**: Existing translation quality evaluation relies on data-driven QE metrics or reward model training, which increases system complexity and dependency on external modules.

## Method

### Overall Architecture: Trans-Zero Self-Play Multilingual Translation
- **Function**: Build a self-play framework requiring only monolingual data, allowing the LLM to self-improve its translation ability through search and preference optimization during the multilingual translation process.
- **Why**: To break free from the dependency on parallel data and leverage the LLM's inherent multilingual knowledge for resource-efficient multilingual translation training.
- **How**: Define the multilingual translation process (MTP) $\rightarrow$ perform Genetic Monte-Carlo Tree Search (G-MCTS) on MTP to explore candidate translations $\rightarrow$ evaluate translation quality based on cross-lingual semantic consistency $\rightarrow$ extract preference pairs from the search tree $\rightarrow$ perform preference optimization using SPPO.

### Key Design 1: Multilingual Translation Process (MTP) and G-MCTS
- **Function**: Define the iterative multilingual translation process as the search space and implement MCTS search combined with genetic algorithm ideas on it.
- **Why**: MTP extends translation to multilingual chains (e.g., EN$\rightarrow$IT$\rightarrow$ZH$\rightarrow$EN), making semantic consistency verifiable through back-translation. The genetic expansion (merge + mutate) of G-MCTS addresses the lack of diversity in standard MCTS for translation exploration.
- **How**:
    - **Initialization**: Taking the source text as the root node, top-k sampling generates $b$ target language candidate translations as child nodes, initializing rewards via back-translation.
    - **Genetic Expansion**: Select the node with the highest UCB value for expansion. If the node with maximum UCB $\neq$ the node with maximum utility, perform **Merge** (use both as few-shot exemplars to generate a new translation); if they are the same, perform **Mutate** (translate the best reconstructed text in simulation instead of the original input, introducing diversity).
    - **Semantic Consistency Simulation**: Expand $b^n$ MTP trajectories for candidate translations, calculating the consistency score (bidirectional BLEURT average) between the reconstructed text and the original input. Take the better result between literal and free translation as the reward.

### Key Design 2: Tree-to-Preference Algorithm and SPPO Optimization
- **Function**: Systematically extract translation preference pairs from the completed G-MCTS search tree for self-play preference optimization.
- **Why**: The node utility in the search tree naturally reflects the ranking of translation quality, allowing the construction of preference data directly without external reward models or QE modules. Higher utility nodes farther from the root indicate that they maintain semantic consistency even after undergoing more translation steps, making their translation quality more preferable.
- **How**: Perform level-order traversal on the search tree and merge duplicate nodes. Sort them in descending order of utility, and generate a preference pair $(y_w \succ y_l)$ for each swap during sorting. Only retain preferred selected nodes with utility higher than the root node. Convert the utility difference into the win rate required by SPPO via softmax, and finally perform preference optimization using the SPPO symmetric loss.

## Key Experimental Results

### Experiment 1: Comparison with SFT and Dedicated Translation Models (Flores-200, 6 Languages)

| Model | EN⇒X (BLEURT) | X⇒EN (BLEURT) | X⇒X (BLEURT) | Average (BLEURT) |
|------|---------------|---------------|--------------|--------------|
| Mixtral-8x7B-Instruct | 55.42 | 75.41 | 54.49 | 61.77 |
| ALMA-R | 69.38 | 77.52 | 51.03 | 65.98 |
| Tower-Instruct | 76.74 | 78.73 | 72.98 | 76.15 |
| Llama3.1-SFT (5m) | 75.80 | 78.47 | 73.30 | 75.86 |
| **Llama3.1-Trans-Zero** | **73.71** | **77.60** | **73.28** | **74.86** |
| Qwen2.5-SFT (5m) | 75.32 | 78.21 | 72.99 | 75.49 |
| **Qwen2.5-Trans-Zero** | **75.05** | **78.21** | **72.23** | **75.16** |

**Key Findings**: Using only monolingual data, Trans-Zero reaches or even surpasses the level of 5M parallel data SFT on non-English directions (X⇒X), and its overall performance is highly comparable to large-scale supervised methods. It is slightly lower than 5M SFT in the EN⇒X direction, but the gap is very small.

### Experiment 2: Effect of G-MCTS as Standalone Inference Enhancement

| Model | EN⇒X (BLEURT) | X⇒X (BLEURT) | Average (BLEURT) |
|------|---------------|--------------|--------------|
| Llama3.1-Instruct | 62.57 | 62.52 | 65.72 |
| + G-MCTS | 64.21 (+1.64) | 68.12 (+5.60) | 67.45 (+1.73) |
| Llama3.1-SFT (5k) | 69.33 | 68.51 | 71.61 |
| + G-MCTS | 71.55 (+2.22) | 71.92 (+3.41) | 73.45 (+1.84) |
| Tower-Instruct | 76.74 | 72.98 | 76.15 |
| + G-MCTS | 76.44 (-0.30) | 74.42 (+1.44) | 76.38 (+0.23) |

**Key Findings**: G-MCTS as a pure inference enhancement shows the most significant improvement in non-English directions (X⇒X) (up to +5.60), demonstrating its cross-lingual exploration capabilities. The improvement is limited for already strong models (such as Tower-Instruct), but there are still gains in the X⇒X direction. Base models (such as ALMA-R, Llama3.1-Base) failed to search (Failed) due to a lack of translation capability, showing that G-MCTS requires basic translation capabilities as a starting condition.

## Highlights & Insights
- **Breaking the Parallel Data Dependency**: The first framework to achieve self-play training for multilingual translation using only monolingual data, which is of great significance in low-resource scenarios.
- **Exquisite G-MCTS Design**: The combination of genetic expansion (merge/mutate) and multilingual semantic consistency simulation ensures search diversity while providing evaluation signals without requiring external rewards.
- **Outstanding Advantages in Non-English Directions**: Performs exceptionally well on the most challenging X⇒X translation direction, even surpassing large-scale parallel SFT.
- **Simple and Effective Tree-to-Preference Algorithm**: Directly converts the utility ordering of the search tree into SPPO preference pairs, avoiding additional reward model training.

## Limitations & Future Work
- **Requires Basic Translation Capability to Start**: G-MCTS search fails directly on base models with very weak translation capabilities (such as Llama3.1-Base), requiring a cold-start phase with a small amount of instruction data.
- **High Computational Overhead**: G-MCTS requires a large number of translation calls per sentence ($b^n$ simulation trajectories $\times$ multiple rounds of search), still demanding substantial computational resources even with 32 GPUs in parallel.
- **Limited Language Coverage**: Only validated on 6 languages, without involving truly low-resource languages (e.g., African languages, Southeast Asian languages).
- **Slightly Weaker than Large-Scale SFT in EN⇒X Directions**: There is still a gap of about 2 BLEURT points compared to 5M SFT in English-to-other-languages directions, and the advantage in high-resource scenarios is not pronounced enough.

## Related Work & Insights

### vs ALMA / ALMA-R (Xu et al., 2024a/c)
ALMA and ALMA-R rely on large amounts of parallel data and preference annotations generated by external LLMs. Trans-Zero completely gets rid of parallel data and autonomously generates preference signals through self-play search. Trans-Zero is comparable to ALMA-R on EN⇒X, but substantially outperforms ALMA-R on X⇒X (73.28 vs 51.03 BLEURT), demonstrating the advantages of the self-play framework for non-English directions.

### vs Self-Play Preference Optimization (SPPO, Chen et al., 2024)
SPPO provides a game-theoretic framework for preference optimization, but the original SPPO requires external preference signals. Trans-Zero innovatively uses the search utility of G-MCTS as the source of preferences, achieving an end-to-end self-play closed loop in translation scenarios without any external evaluation modules.

### vs Cross-Lingual Optimization Methods (Geng et al., 2024; She et al., 2024)
Existing methods use strong languages to assist in optimizing weak languages, but they are limited to bilingual scenarios or require a pre-defined pivot language. Trans-Zero iteratively translates across arbitrary languages via MTP, and the search space scales with the number of languages (experiments show that 6 languages outperform 4 languages), offering better scalability.

### Additional Observations
- Increasing the number of languages involved in the search (4⇒6) significantly improves the performance upper bound of Trans-Zero, indicating that the cross-lingual validation signals become richer as the number of languages increases.
- SFT performance tends to saturate after exceeding 100k parallel samples, whereas Trans-Zero continues to benefit from search in non-English directions, suggesting complementarity between search-based learning and data-driven learning.
- Translations that fail language detection are penalized by halving their utility during sorting. This simple strategy effectively filters out low-quality translation pairs that could contaminate preference learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to apply MCTS self-play to parallel-data-free multilingual translation; the framework design is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 6 languages, extensive comparisons with multiple baselines, and ablation analyses, though validation on truly low-resource languages is missing.
- Writing Quality: ⭐⭐⭐⭐ — Clear descriptions of methods, complete derivation of formulas, and intuitive case studies.
- Value: ⭐⭐⭐⭐ — Provides a completely new paradigm for low-resource multilingual translation with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)

</div>

<!-- RELATED:END -->
