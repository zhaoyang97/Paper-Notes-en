---
title: >-
  [Paper Note] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences
description: >-
  [ICLR 2026][Image Generation][Implicit world model] This paper proposes an adversarial sequence generation method to verify the soundness of implicit world models in generative sequence models. Through systematic evaluat…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Implicit world model"
  - "adversarial sequence generation"
  - "chess"
  - "soundness verification"
  - "linear probes"
date: 2026-05-08
content_hash: 583cc1b8240d4a45
---

# Verification of the Implicit World Model in a Generative Model via Adversarial Sequences

**Conference**: ICLR 2026
**arXiv**: [2602.05903](https://arxiv.org/abs/2602.05903)
**Code**: [https://github.com/szegedai/world-model-verification](https://github.com/szegedai/world-model-verification)
**Area**: World Models / Explainable AI
**Keywords**: Implicit world model, adversarial sequence generation, chess, soundness verification, linear probes

## TL;DR
This paper proposes an adversarial sequence generation method to verify the soundness of implicit world models in generative sequence models. Through systematic evaluation in the chess domain using multiple adversarial strategies (IMO/BSO/AD), it finds that all tested models are unsound, while training objectives and dataset choice significantly affect soundness. Furthermore, linear board-state probes exhibit no causal role in most models.

## Background & Motivation

**Background**: Whether generative sequence models (e.g., LLMs) learn an underlying "world model" during training is a critical open question. Prior work (Li et al. 2023, Toshniwal et al. 2022) trained models on Othello and chess, claiming via linear probes that the models learned world states.

**Limitations of Prior Work**: (1) Simple metrics such as next-token accuracy can be misleading — a fundamentally incorrect model may still achieve high accuracy; (2) the sequence-level analysis of Vafa et al. requires defining a probability threshold for the generated language, introducing ad hoc parameters; (3) whether board states decoded by linear probes play a causal role in predictions remains unclear.

**Key Challenge**: Although it is theoretically possible to learn a sound world model from sampled sequences, empirical verification is difficult. Simple tests are insufficient; systematic stress-testing methods are needed.

**Goal**: How can one effectively verify whether the implicit world model of a generative model is sound? Do board-state probes causally influence predictions?

**Key Insight**: An adversary generates legal move sequences that force the generative model to predict illegal continuations, providing existential counterexamples to soundness.

**Core Idea**: The adversary plays legal moves to force the model under test to produce illegal moves, thereby quantifying and analyzing failure modes of the implicit world model.

## Method

### Overall Architecture
The adversary (White) generates legal moves, while the model under test (Black) predicts the next move. The adversary selects the legal move most likely to cause a model error:
$$a_{k+1}^* = \arg\max_{a_{k+1} \in W(a_1..a_k)} f(M, a_1..a_k a_{k+1})$$
An attack succeeds when the model predicts an illegal move $m(s^*) \notin W(s^*)$.

### Key Designs

1. **Illegal Move Oracle (IMO)**:

    - $f_{\text{IMO}}(M, s) = \max_{a \notin W(s)} M(a|s)$
    - Selects the legal move that maximizes the probability of an illegal continuation.
    - The most direct attack strategy.

2. **Board State Oracle (BSO)**:

    - $f_{\text{BSO}}(M, s) = \mathcal{L}_B(M, s)$
    - Selects the move that maximizes board-state prediction error.
    - Hypothesis: if board-state probes are causally active, corrupting the state should induce illegal moves.

3. **Adversarial Detours (AD)**:

    - $f_{\text{AD}}(M, s) = -M(a_{k+1}|a_1..a_k)$
    - Selects the move with the lowest conditional probability, steering generation toward OOD regions.
    - Adapted from Vafa et al.

4. **Novel Training Objectives**:

    - **Probability Distribution (PD) objective**: predicts a uniform distribution over all legal moves rather than only the next token.
    - **Joint Probe (+ JP) objective**: a dual-head loss that jointly trains next-token prediction and board-state prediction.
    - Both objectives aim to more directly encode world-model transition rules.

### Model Setup
GPT-2 architecture (86M parameters), 12 layers, 768 dimensions, 12 attention heads. Six datasets (random 500K / 2M / 10M + Millionbase / Stockfish / Lichess), four training objectives (NT / PD / NT+JP / PD+JP), yielding 24 models in total.

## Key Experimental Results

### Attack Success Rate (Random Datasets)

| Attack Strategy | Random-500K NT | Random-10M NT | Random-10M PD |
|---|---|---|---|
| RM (random baseline) | 0.954 | 0.673 | 0.874 |
| SMM (friendly baseline) | 0.419 | 0.172 | 0.900 |
| **IMO** | **0.996** | **0.972** | **0.992** |
| BSO | 0.886 | 0.541 | 0.811 |
| AD | 0.946 | 0.516 | 0.902 |

### Key Findings

1. **All models are unsound**: The IMO attack achieves near-100% success (up to 1.000); even the best-performing model (Random-10M + NT) yields a 97.2% attack success rate.

2. **Dataset choice has a substantial impact**:
    - Random datasets substantially outperform strategy-based datasets (Millionbase / Stockfish / Lichess) in promoting soundness.
    - Increasing random training data markedly improves soundness (RM success rate drops from 95.4% to 67.3% as data scales from 500K to 10M).

3. **PD training objective is a double-edged sword**:
    - PD models exhibit very high SMM attack success rates (0.900+), indicating greater vulnerability in non-adversarial settings.
    - Nevertheless, PD's ability to model distributions leads to different behavior under certain adversarial conditions.

4. **BSO vs. IMO reveals absence of probe causality**:
    - BSO attack success rates are significantly lower than those of IMO (e.g., Random-10M NT: 0.541 vs. 0.972).
    - This indicates that corrupting board-state probe predictions does not equivalently induce illegal moves.
    - Conclusion: the board states extracted by linear probes play no causal role in next-token prediction for most models.

5. **Divergence between random and strategy-based datasets**: Models trained on strategy-based datasets are more susceptible to AD attacks (larger OOD regions), whereas random datasets provide more uniform state-space coverage.

## Highlights & Insights
- **Adversarial verification is more effective than statistical testing**: it avoids ad hoc threshold selection and directly yields soundness counterexamples.
- **An important negative result regarding probe causality**: challenges the assumption that "world states decoded by linear probes are causally used in predictions."
- **Data quality insight**: random games (rather than high-quality games) are more conducive to learning rules, as strategically poor but legal moves are underrepresented in strategy-based data.
- **Generalizability of the framework**: the approach extends to any domain where a world model can be defined as a formal language.

## Limitations & Future Work
- Experiments are limited to 86M-parameter GPT-2 models; larger models may behave differently.
- Chess, although complex, has deterministic rules; world models for natural language are considerably more ambiguous.
- The adversarial search is greedy and may theoretically miss more effective attack sequences.
- Applying this framework to domains with formal rules, such as code generation, warrants future exploration.

## Related Work & Insights
- **vs. Vafa et al.**: Their approach requires defining a probability threshold to characterize the generated language; the proposed method avoids this issue entirely.
- **vs. Li et al. (OthelloGPT)**: They claim causal involvement of probes; the present findings in chess contradict this claim.
- **vs. Karvonen 2024**: That work claims chess models possess a consistent world model; the adversarial tests presented here demonstrate otherwise.

## Rating
- Novelty: ⭐⭐⭐⭐ The adversarial verification framework is novel, and the negative result on probe causality is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale systematic evaluation across 24 models, 5 attack strategies, and 6 datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Formal definitions are clear and the experimental design is rigorous.
- Value: ⭐⭐⭐⭐ Significant implications for AI safety and interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GGBall: Graph Generative Model on Poincaré Ball](ggball_graph_generative_model_on_poincaré_ball.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICLR 2026\] Efficient Adversarial Attacks on High-dimensional Offline Bandits](efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)

</div>

<!-- RELATED:END -->
