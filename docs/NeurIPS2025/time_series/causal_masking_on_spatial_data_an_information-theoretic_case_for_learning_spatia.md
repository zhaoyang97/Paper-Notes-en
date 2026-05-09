---
title: >-
  [Paper Note] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models
description: >-
  [NeurIPS 2025][Time Series][causal masking] This paper demonstrates that applying causal masking directly to spatial data (chess board states in FEN format) for training a unimodal LLM outperforms first linearizing the data into sequences (PGN move records) and then applying causal masking — Llama 1.3B trained with FEN + causal masking achieves ~2630 Elo, whereas PGN + causal masking yields only ~2130 Elo.
tags:
  - NeurIPS 2025
  - Time Series
  - causal masking
  - spatial data
  - chess
  - FEN encoding
  - bidirectional attention
  - information theory
  - unimodal LLM
date: 2026-05-08
content_hash: f3a9526d7bccc114
---

# Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.27009](https://arxiv.org/abs/2510.27009)
**Code**: Not open-sourced
**Area**: Time Series / Spatial Data
**Keywords**: causal masking, spatial data, chess, FEN encoding, bidirectional attention, information theory, unimodal LLM

## TL;DR
This paper demonstrates that applying causal masking directly to spatial data (chess board states in FEN format) for training a unimodal LLM outperforms first linearizing the data into sequences (PGN move records) and then applying causal masking — Llama 1.3B trained with FEN + causal masking achieves ~2630 Elo, whereas PGN + causal masking yields only ~2130 Elo.

## Background & Motivation

**Background**: LLMs are typically trained on sequential text with causal (autoregressive) masking. However, many real-world data sources possess inherent spatial structure (e.g., chessboards, molecules, images), and linearizing them into 1D sequences may discard spatial relationships.

**Limitations of Prior Work**: (a) It remains unclear whether causal masking remains effective on spatial data; (b) linearization approaches (e.g., PGN move records) discard immediate spatial information; (c) bidirectional attention is theoretically better suited for spatial data but incurs higher training complexity.

**Key Challenge**: Causal masking inherently assumes a 1D sequential structure, whereas spatial data is 2D/3D — the central challenge is how to exploit spatial structure while retaining the causal masking training paradigm.

**Goal**: To compare spatial encoding (FEN board states) versus sequential encoding (PGN moves) under causal and bidirectional masking.

**Key Insight**: Chess serves as a controlled testbed — FEN encoding preserves the spatial board structure, while PGN encoding is a conventional linearized sequence. ChessBench provides 15 billion Stockfish-annotated positions.

**Core Idea**: Spatial encoding (FEN) + causal masking outperforms sequential encoding (PGN) + causal masking, indicating that preserving spatial structure matters more than the choice of masking strategy.

## Method

### Overall Architecture
A controlled comparison across four configurations (FEN + causal masking, FEN + bidirectional, PGN + causal, PGN + bidirectional) using models of equal scale, trained on ChessBench data and evaluated by Elo rating and best-move accuracy.

### Key Designs

1. **Board Encoding Comparison**:

    - **Function**: Compare spatial (FEN) vs. sequential (PGN) encoding.
    - **Mechanism**: FEN expands the 64-square board into a fixed-length string, preserving row-column structure; PGN records each move in algebraic notation (e.g., e2e4), constituting a purely sequential representation.
    - **Design Motivation**: FEN implicitly encodes spatial position information within each token.

2. **Masking Strategy Comparison**:

    - **Function**: Compare causal (autoregressive) vs. bidirectional attention.
    - **Mechanism**: Causal masking allows each token to attend only to preceding tokens; bidirectional masking allows full-context attention. Both settings use character-level tokenization.
    - **Design Motivation**: Bidirectional attention is theoretically more appropriate for non-sequential data.

3. **Character-Level Tokenization and Prompt Engineering**:

    - **Function**: Design a dedicated character-level tokenizer and structured prompt template for FEN data.
    - **Mechanism**: Default LLM tokenizers merge character sequences (e.g., "pk" → a single token instead of separate pawn and king tokens), destroying the character-wise spatial semantics of FEN. Enforcing character-level tokenization ensures each piece and empty square has an independent token representation; prompts embed the FEN state, the list of legal moves, and the target best move.
    - **Design Motivation**: Experiments show that both Pythia and Llama fail to converge under their default tokenizers; character-level tokenization is a necessary prerequisite for successful training on FEN.

4. **Masked Cross-Entropy Objective**:

    - **Function**: Compute loss exclusively on the best-move tokens, masking all other prompt tokens.
    - **Mechanism**: A binary mask vector $\mathbf{w}$ is constructed such that $w_t = 1$ only when a token belongs to the best move $m^*$; the loss is $\mathcal{L}_{\text{masked}} = -\sum_t w_t \log p_\theta(m_t^*|X_{0:t-1})$. Full teacher forcing is applied, allowing prediction to condition on all preceding tokens including target tokens.
    - **Design Motivation**: FEN prompts are lengthy (board state + legal move list); computing loss over all tokens introduces substantial irrelevant gradient noise, while masking focuses the model on move prediction.

### Loss & Training

- **Loss Function**: Masked cross-entropy, with gradients computed only on best-move tokens.
- **Training Configuration**: 200K steps, 2× A100 (80 GB), cosine LR decay + warmup + gradient clipping + mixed precision.
- **FEN Data**: ChessBench with 15 billion Stockfish-annotated positions.
- **PGN Data**: ~1 billion games with White played by full-strength Stockfish and Black by engines ranging from ELO 1200–3100.
- **Models**: Llama 1.3B (SFT), two equal-scale NanoGPT models (trained from scratch).

## Key Experimental Results

### Main Results

| Configuration | Elo Rating | Best-Move Accuracy | Legal Move Rate |
|---|---|---|---|
| Llama 1.3B (FEN + causal) | **~2630** | **58%** | 99.91% |
| NanoGPT (FEN + bidirectional) | ~2700+ | 60%+ | ~100% |
| NanoGPT (PGN + causal) | ~2130 | 40% | ~100% |

### Table 2: Llama 1.3B Performance Before and After SFT (~12,800 Test Positions)

| Metric | Before SFT (zero-shot) | After SFT | Gain |
|---|---|---|---|
| Syntactically valid move rate | Negligible | 99.94% | — |
| Legal move rate | Negligible | 99.91% | — |
| Best-move rate (Stockfish) | ~0.6% | ~58% | **~100×** |

### Key Findings

- **FEN >> PGN (both with causal masking)**: ~2630 vs. ~2130 Elo — spatial encoding yields a substantial gain of +500 Elo.
- **Bidirectional only marginally outperforms causal (both with FEN encoding)**: ~2700 vs. ~2630, a gap far smaller than that attributable to encoding choice.
- **Impact of encoding >> impact of masking strategy**: The benefit of preserving spatial structure greatly exceeds that of selecting the "theoretically superior" attention mechanism.
- **Legal move rate approaches 100%**: All configurations learn the rules of chess; differences lie solely in move quality.
- **Pre-training quality has a positive transfer effect**: Llama 1.3B fine-tuned via SFT outperforms smaller models trained from scratch on equivalent data, suggesting that general pre-training capabilities transfer to spatial reasoning.

## Highlights & Insights

- **Information-theoretic argument for "data representation > training strategy"**: From the perspective of function composition complexity, the paper explains why spatial encoding + causal masking outperforms sequential encoding + causal masking — the PGN model must learn a composite mapping $\mathcal{G} \circ (\mathcal{F} \to \mathcal{S})$, whereas the FEN model requires only the direct mapping $\mathcal{F} \to \mathcal{S}$.
- **Unique value of chess as a controlled testbed**: Chess is rare in providing equivalent spatial and sequential representations of the same task, enabling a confound-free comparison between causal masking and bidirectional attention.
- **Tokenizer alignment as an overlooked prerequisite**: Character-level tokenization may appear to be a technical detail, but in structured symbolic domains it determines whether the model can converge at all.
- **Practical implication**: For data with spatial structure (molecules, circuits, maps), even when only a unimodal causal LLM is available, directly applying spatial encoding + causal masking is a viable strategy that outperforms linearization.

## Limitations & Future Work

- **No variance estimation from single runs**: Each configuration is trained only once (due to computational constraints, each run takes ~3 weeks on 2× A100), lacking statistical significance testing.
- **Chess-only evaluation**: Whether the conclusions generalize to other spatial data domains — images, molecular graphs, geographic information — remains to be verified.
- **Non-FIDE Elo evaluation**: Games are played against calibrated Stockfish versions (Level 0–10), which are not directly equivalent to human FIDE Elo ratings.
- **Markov limitation of FEN**: FEN encoding contains only a snapshot of the current board state and cannot represent the threefold repetition rule, meaning the learned policy is theoretically suboptimal in a Markovian sense.
- **Asymmetric information content**: PGN encodes the full move history while FEN encodes only the current state; the surjective mapping implies an irreversible loss of partial historical information.

## Related Work & Insights

- **ChessBench** (Ruoss et al., 2024): Provides 15 billion Stockfish-annotated FEN positions and a strong 50M-parameter baseline, but does not systematically compare the fundamental impact of spatial vs. sequential encoding.
- **OthelloGPT & Karvonen** (2023, 2024): Find that autoregressive Transformers trained on PGN sequences spontaneously develop latent spatial world models — the information-theoretic framework in this paper explains why this requires additional representational complexity and why it is inferior to direct spatial encoding.
- **Multimodal masking strategies** (Amrani et al., 2025; Pei et al., 2025): Block-causal and relaxed masking are common approaches for handling non-sequential inputs; this paper demonstrates that high performance is attainable even without relaxing causal constraints at all.
- **Transferable insight**: Any scenario requiring spatial or relational data to be fed into a causal LLM (molecular generation, circuit design, map reasoning, protein structure prediction) should prioritize encoding schemes that preserve spatial structure.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic information-theoretic comparison of causal masking versus sequential linearization on spatial data.
- **Experimental Thoroughness**: ⭐⭐⭐ Large-scale data and multi-model comparisons, constrained by single training runs and a single domain.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations and controlled experimental design are clearly presented, with conclusions supported by information-theoretic arguments.
- **Value**: ⭐⭐⭐⭐ Provides direct practical guidance on how to encode spatial data for unimodal LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)

<!-- RELATED:END -->
