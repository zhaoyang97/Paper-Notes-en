---
title: >-
  [Paper Note] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale
description: >-
  [NeurIPS 2025][LLM Pretraining][language model training dynamics] Through systematic analysis of over 1,400 language model checkpoints—spanning Transformer/Mamba/RWKV architectures, 14M–12B parameter scales, and two training datasets—across 110K+ tokens, this work demonstrates that all autoregressive language models exhibit highly consistent behavioral phases during pre-training: predicted probabilities sequentially overfit to n-gram probabilities of increasing order. Three simple heuristics—word frequency, n-gram probability, and semantic similarity—account for up to 98% of behavioral variance.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - language model training dynamics
  - n-gram behavioral phases
  - cross-architecture consistency
  - pre-training analysis
  - semantic similarity
date: 2026-05-08
content_hash: f47d163fa1d84da5
---

# Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale

**Conference**: NeurIPS 2025
**arXiv**: [2510.24963](https://arxiv.org/abs/2510.24963)
**Code**: [https://github.com/jmichaelov/lm-behavioral-phases](https://github.com/jmichaelov/lm-behavioral-phases)
**Area**: LLM Pre-training
**Keywords**: language model training dynamics, n-gram behavioral phases, cross-architecture consistency, pre-training analysis, semantic similarity

## TL;DR

Through systematic analysis of over 1,400 language model checkpoints—spanning Transformer/Mamba/RWKV architectures, 14M–12B parameter scales, and two training datasets—across 110K+ tokens, this work demonstrates that all autoregressive language models exhibit highly consistent behavioral phases during pre-training: predicted probabilities sequentially overfit to n-gram probabilities of increasing order. Three simple heuristics—word frequency, n-gram probability, and semantic similarity—account for up to 98% of behavioral variance.

## Background & Motivation

**Background**: Language models acquire emergent behaviors such as grammatical generation and knowledge reasoning solely through next-token prediction. Prior work has identified abrupt behavioral transitions during training (e.g., sharp performance changes associated with the emergence of induction heads), yet these analyses typically focus on specific sub-networks or target behaviors, lacking a systematic characterization of how overall model behavior evolves throughout training.

**Limitations of Prior Work**: (1) Although LM predictions are known to correlate strongly with n-gram probabilities—particularly in early training—the n-gram overfitting phenomenon has only been validated on small-scale models of a single architecture (GPT-2). (2) Semantic similarity has also been shown to correlate with LM predictions, but its independent contribution has never been verified after controlling for n-gram effects. (3) Most critically, whether different architectures (attention-based vs. state-space models vs. modern RNNs) follow the same learning trajectory remains entirely unknown.

**Key Challenge**: Is the learning trajectory of language model behavior primarily governed by model-specific details (architecture, data, scale), or by the autoregressive language modeling task itself?

**Goal**: (1) To what extent can three simple heuristics explain LM behavior at any point during training? (2) How does the relationship between these metrics and LM behavior evolve over training? (3) Are these patterns consistent across architectures, datasets, and scales? (4) Does semantic similarity contribute independently of n-gram probability?

**Key Insight**: This work integrates two lines of research—LM overfitting to n-grams of increasing order (Chang et al., 2024) and the correlation between semantic similarity and LM predictions (Michaelov et al., 2024)—unifying them into a "behavioral phases" framework for large-scale controlled analysis.

**Core Idea**: Autoregressive language models inevitably progress through behavioral phases of increasing n-gram overfitting, a regularity that holds universally across architectures, datasets, and scales, suggesting that the learning trajectory is determined primarily by the task itself rather than by model-specific details.

## Method

### Overall Architecture

Two complementary experiments are conducted. **Experiment 1 (Correlation Analysis)**: For each model at each training checkpoint, Pearson correlation coefficients are computed between output log-probabilities and log-probabilities of n-grams of each order ($n \in \{1,2,3,4,5\}$) as well as semantic similarity, tracking how these correlations evolve over training. **Experiment 2 (Regression Analysis)**: Multiple linear regression models are constructed with unigram log-probability, 5-gram log-probability, and semantic similarity as predictors of LM log-probability. The independent contributions of each factor (z-standardized coefficients) and total explained variance ($R^2$) are analyzed. Models are fit on a training set and evaluated on a held-out validation set.

### Key Designs

1. **Parc Model Suite (Parallel Multi-Architecture Training)**

   - **Function**: Enables comparison of learning behavior across architectures under strictly controlled conditions.
   - **Mechanism**: Three architectures are trained in parallel using identical random seeds (×6), identical OpenWebText training data, and an identical tokenizer: Parc-Pythia (160M Transformer), Parc-Mamba (130M SSM), and Parc-RWKV (169M RNN). Each seed is trained for 4,000 steps (batch size 512, 1024-token sequences), with 73 checkpoints saved. Crucially, all three architectures observe exactly the same training sequences at every step.
   - **Design Motivation**: Eliminates confounds from data ordering and initialization, making architecture the sole variable. To the authors' knowledge, this constitutes the first publicly released Mamba and RWKV models with intermediate checkpoints.

2. **NaWoCo Decontaminated Evaluation Dataset**

   - **Function**: Provides natural language evaluation samples free of training data contamination.
   - **Mechanism**: Word–context pairs are sampled from the FineWeb corpus under strict filtering criteria: target words must be a single token shared across all model vocabularies, appear after the 5th word of a sentence, contain no capitalized words (except sentence-initial), have toxicity probability < 0.1, and—critically—be verified via infini-gram counts as absent from any test model's training data. The final dataset comprises 78K training, 39K validation, and 41K test samples.
   - **Design Motivation**: Prevents spurious correlations from training data contamination and ensures that target words are single tokens under different tokenizers, enabling fair cross-model comparison.

3. **N-gram Probability Computation (infini-gram + Stupid Backoff)**

   - **Function**: Computes n-gram probabilities accurately over large corpora.
   - **Mechanism**: The infini-gram toolkit is used to build indices over The Pile (Pythia's training data) and OpenWebText (other models' training data), retrieving exact sequence counts with Stupid Backoff smoothing. All n-grams are computed at the word level rather than the token level to avoid tokenizer discrepancies.
   - **Design Motivation**: Exact counting is preferable to sampling-based estimation. Robustness is verified via matched/unmatched training corpus comparisons.

### Loss & Training

The full analysis covers three model groups: (1) the Parc suite (18 models: 6 seeds × 3 architectures); (2) the Pythia family (14M–12B, including additional seeds from PolyPythia); and (3) Open-GPT2 (117M/345M, 4–5 seeds each). In total, 1,418 model instances are analyzed. Regression models in Experiment 2 are fit on the training set and evaluated for $R^2$ on the validation set, with cross-validation performed using matched/unmatched n-gram corpora and Wikipedia/CommonCrawl fastText embeddings.

## Key Experimental Results

### Main Results: Three Behavioral Phases

| Phase | Unigram Coefficient | 5-gram Coefficient | Semantic Similarity Coefficient | Characteristic Behavior |
|-------|--------------------|--------------------|--------------------------------|------------------------|
| Phase 1 | Rises sharply from zero to peak | Slight decrease to negative | Rises in tandem | Frequency-dominated |
| Phase 2 | Gradually decreases | Rises sharply | Small dip then recovers | Contextual learning |
| Phase 3 | Stabilizes | Stabilizes | Maintains positive coefficient | Convergence |

| Key Quantitative Metric | Value |
|------------------------|-------|
| Peak $R^2$ (Phase 1, all models) | 0.86–0.98 |
| Lower bound of stable $R^2$ (late training) | > 0.50 (all models) |
| Cross-architecture step-wise correlation (step ≥ 80) | Pearson $r \geq 0.93$ |
| Confidence intervals across seeds | Nearly invisible (highly consistent) |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Matched vs. unmatched n-gram corpus | Behavioral phase patterns are nearly unchanged |
| Wikipedia vs. CommonCrawl fastText | CC is more correlated with unigram ($r$ = 0.67–0.69); Wiki is more independent of frequency ($r$ = 0.34–0.35) |
| SGPT weighted vs. uniform context weighting | Differences are negligible |
| Scale effect (14M→12B) | Larger models show greater decrease in unigram coefficient and greater increase in 5-gram coefficient |
| Validation $R^2$ vs. training $R^2$ | Nearly identical (no overfitting) |

### Key Findings

- **Universally consistent behavioral phases**: All 1,418 model instances (3 architectures, 2 datasets, 14M–12B) exhibit the same three-phase pattern—models must necessarily "pass through n-gram prediction" during gradient descent.
- **Symmetry of scale effects**: In late training, larger models show greater decreases in unigram coefficient alongside greater increases in 5-gram coefficient, indicating that larger models are more capable of transcending low-order statistics and leveraging longer context; smaller models rely more heavily on low-order n-grams due to capacity constraints.
- **Independent contribution of semantic similarity**: After controlling for n-gram effects, semantic similarity retains a significant positive coefficient that emerges very early in training and persists until the end—a finding verified here for the first time.
- **"Beyond heuristics" signal in large models**: The largest Pythia models show the greatest decline in $R^2$ in late training, coinciding with the period when their benchmark performance begins to surpass smaller models, suggesting that more complex patterns beyond simple heuristics are being acquired.
- **Remarkable cross-architecture consistency**: Transformer, Mamba, and RWKV exhibit not merely similar trends but quantitatively highly correlated behavior at each step ($r \geq 0.93$), indicating that the learning trajectory is determined by the task rather than the architecture.

## Highlights & Insights

- **The "necessary passage" hypothesis**: Autoregressive LMs may be required to pass through the n-gram overfitting phase before developing higher-level capabilities—models cannot skip these phases during gradient descent, analogous to stage-wise development in child language acquisition.
- **Theoretical significance of universality**: Three fundamentally distinct sequence modeling architectures (attention vs. selective state space vs. linear-attention RNN) produce nearly identical behavioral trajectories, strongly suggesting that the autoregressive language modeling task itself is the determining factor.
- **Bridge to cognitive science**: The analytical framework aligns with methods used in human language processing research to disentangle word frequency, contextual probability, and semantic similarity effects, providing a bridge for comparing human and machine language processing.
- **Methodological resources**: The Parc model suite and NaWoCo dataset offer high-quality controlled resources for future research on training dynamics.

## Limitations & Future Work

- Mamba and RWKV are tested only at ~130–170M scale; whether consistency holds at larger scales remains to be verified.
- Only n-grams with $n \leq 5$ and static word embeddings are analyzed; models may be sensitive to higher-order statistics and contextual embeddings.
- The regression still fails to account for all variance—particularly in large models during late training—and this unexplained portion may be the most theoretically interesting.
- The analysis is restricted to English; cross-lingual generalizability is unknown.
- This is an observational study; the causal mechanisms underlying phase transitions (e.g., their temporal relationship to induction head formation) are not analyzed.

## Related Work & Insights

- Chang et al. (2024) first identified the n-gram overfitting phase phenomenon across 5 GPT-2 seeds; the present work extends this finding to 1,400+ model instances and multiple architectures.
- Voita et al. (2024) identified "dedicated" n-gram neurons in OPT models, more prevalent in larger models—consistent with this paper's scale-effect conclusions.
- Bietti et al. (2023) provide mechanistic evidence that Transformers directly implement n-gram prediction.
- Chang & Bergen (2025) find that bigram circuits persist throughout training but gradually weaken.
- Michaelov et al. (2024) report that GPT-3 surprisal correlates with semantic similarity at $r = -0.61$; the present work is the first to verify this independence after controlling for n-gram effects.

## Rating

⭐⭐⭐⭐ (4/5)

The study is large in scale (1,400+ checkpoints), rigorous in experimental design (Parc parallel training eliminates confounding variables), and the core finding—that behavioral phases are consistent across architectures—carries significant theoretical importance. The work successfully unifies two previously separate lines of research on n-gram overfitting and semantic similarity. The primary limitations are the observational nature of the study (lacking causal mechanistic explanation) and the restricted scale of testing for non-Transformer architectures.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](memory_mosaics_at_scale.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)

<!-- RELATED:END -->
