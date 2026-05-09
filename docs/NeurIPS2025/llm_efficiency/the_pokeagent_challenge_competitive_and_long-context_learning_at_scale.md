---
title: >-
  [Paper Note] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale
description: >-
  [NeurIPS 2025][LLM Efficiency][Game AI Benchmark] This paper introduces the PokéAgent Challenge, a large-scale dual-track AI benchmark built on Pokémon competitive battling and RPG speedrunning. Validated through the NeurIPS 2025 competition, it demonstrates that specialist RL methods substantially outperform general-purpose LLM approaches, and reveals that the capabilities measured by Pokémon battling are nearly orthogonal to those assessed by 49 existing LLM benchmarks.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Game AI Benchmark
  - Pokémon Battling
  - Long-Horizon Planning
  - Reinforcement Learning
  - LLM Agent
date: 2026-05-08
content_hash: 16da4e82e33d4092
---

# The PokeAgent Challenge: Competitive and Long-Context Learning at Scale

**Conference**: NeurIPS 2025
**arXiv**: [2603.15563](https://arxiv.org/abs/2603.15563)
**Code**: [https://pokeagentchallenge.com](https://pokeagentchallenge.com)
**Area**: LLM Efficiency
**Keywords**: Game AI Benchmark, Pokémon Battling, Long-Horizon Planning, Reinforcement Learning, LLM Agent

## TL;DR

This paper introduces the PokéAgent Challenge, a large-scale dual-track AI benchmark built on Pokémon competitive battling and RPG speedrunning. Validated through the NeurIPS 2025 competition, it demonstrates that specialist RL methods substantially outperform general-purpose LLM approaches, and reveals that the capabilities measured by Pokémon battling are nearly orthogonal to those assessed by 49 existing LLM benchmarks.

## Background & Motivation

**Background**: Partial observability, game-theoretic reasoning, and long-horizon planning are central challenges in sequential decision-making, yet existing benchmarks tend to evaluate these dimensions in isolation — imperfect-information games focus on short-horizon equilibrium computation, while open-ended environments test exploration but lack adversarial opponents. Pokémon simultaneously engages all three dimensions: competitive battles require strategic reasoning against hidden-information opponents, while the single-player campaign demands thousands of steps of exploration, resource management, and combat decision-making.

**Limitations of Prior Work**: Although Pokémon attracted widespread attention from the AI community in 2025 (Claude Plays Pokémon, Gemini completing the Blue version, GPT-5 completing Red), these efforts were conducted independently — across different game versions, different harnesses, and different evaluation criteria — rendering meaningful cross-system comparison impossible. It is therefore impossible to attribute success to agent architecture, the underlying model, or hardcoded assumptions that simplify perception.

**Key Challenge**: Pokémon involves approximately $10^{564}$ possible battle states, team-building across 1,000+ species, and a continuously evolving competitive metagame, yet there is no standardized evaluation infrastructure to fairly compare RL, LLM, and hybrid approaches. Furthermore, while Pokémon knowledge is abundantly represented in pretraining corpora (move attributes, damage formulae, competitive tier lists), translating this latent knowledge into multi-turn sequential decision-making under partial observability is fundamentally different from simple retrieval.

**Goal**: To establish a unified Pokémon AI evaluation framework that provides standardized baselines, large-scale datasets, and a public leaderboard, enabling fair comparison across paradigms (RL vs. LLM vs. hybrid) under identical conditions.

**Key Insight**: A dual-track benchmark is designed — the Competitive Battling Track (rated battles on Pokémon Showdown) tests game-theoretic reasoning and generalization, while the Speedrunning Track (Pokémon Emerald RPG speedrun) tests long-horizon planning and sequential decision-making. The NeurIPS 2025 competition is used to validate benchmark quality and promote community engagement.

**Core Idea**: Standardize Pokémon's battle system and RPG environment into a living benchmark that simultaneously evaluates the capability gaps between RL and LLM approaches in high-stakes competitive play and long-horizon decision-making.

## Method

### Overall Architecture

The PokéAgent Challenge comprises two complementary tracks:

- **Competitive Battling Track**: Built on the open-source Pokémon Showdown simulator, this track evaluates strategic reasoning in two-player zero-sum games under partial observability. Each turn involves selecting from approximately 9 actions, with battles lasting 20–100 turns.
- **Speedrunning Track**: Formalizes the RPG as a periodic MDP $M=(S,A,T,R,\gamma)$, where actions are button inputs and the agent receives visual frames and limited state information (team composition, levels, HP values), but puzzle states, dynamic obstacles, and move sets are not exposed.

### Key Designs

#### Battling Track Evaluation System
- **Function**: Establishes a multi-dimensional skill rating system and a standardized battle environment.
- **Mechanism**: Full-History Bradley-Terry (FH-BT) scoring serves as the primary metric, supplemented by Glicko-1 and GXE. Battles are conducted on a dedicated modified Showdown server to avoid interference with human players.
- **Design Motivation**: Showdown's online rating (Elo) is designed for large pools of human players and introduces excessive noise in the small-pool, densely matched, fixed-strategy settings typical of AI agents. FH-BT, based on a full-history Bradley-Terry model with bootstrap uncertainty, is better suited to this setting.

#### Large-Scale Datasets
- **Function**: Provides the largest publicly available Pokémon battle dataset to date.
- **Mechanism**: Contains 4M+ human demonstration RL trajectories (private information inferred from spectator-view replays to reconstruct each player's perspective), 18M+ synthetic battle trajectories, and 200K+ curated competitive teams.
- **Design Motivation**: Human demonstrations are critical for guiding strategy, but competitive performance typically requires the scale afforded by self-play. Raw "replays" are recorded from a spectator perspective that does not reflect the private information available to each player at decision time, necessitating inference and reconstruction.

#### LLM Baseline Framework
- **Function**: Extends PokéChamp into a general-purpose reasoning model harness framework.
- **Mechanism**: Converts game states into structured text and provides a configurable harness (including depth-limited minimax search with LLM evaluation). Supports frontier API models (GPT/Claude/Gemini) and open-source models (Llama/Gemma/Qwen).
- **Design Motivation**: Even small open-source models achieve meaningful performance when supported by a harness. The default turn timer (60–90 seconds) is insufficient for LLM inference, so an Extended Timer setting is provided.

#### RL Baselines
- **Function**: Extends Metamon and releases 30 pretrained agents spanning the competitive skill ladder.
- **Mechanism**: Covers multiple scale points from compact RNNs to 200M-parameter Transformers, trained on large-scale human demonstrations and self-play.
- **Design Motivation**: Provides high-quality reference points across multiple human skill levels, enabling researchers to explore the compute–performance tradeoff on accessible hardware.

#### Speedrunning Track: Multi-Agent Orchestration System
- **Function**: Releases the first open-source multi-agent orchestration system for long-horizon RPG gameplay.
- **Mechanism**: Coordinates MCP tools (A* pathfinding, button input, knowledge retrieval) with specialized sub-agents (battle strategy, self-reflection, gym puzzles, objective verification). A central orchestrator maintains high-level route planning, dynamically dispatches sub-agents based on game context, and applies automatic context compression to manage thousands of reasoning steps.
- **Design Motivation**: Frontier VLMs achieve 0% completion on this track without any harness — Pokémon RPG is an out-of-distribution task for which a harness is a prerequisite for any meaningful progress. A Harness × Model analytical framework decouples system performance along five dimensions: state representation $S$, tools $T$, memory $M$, feedback $F$, and fine-tuning $\Phi$.

### Loss & Training

- **RL Baseline Training**: Based on large-scale offline RL, policies are bootstrapped from human demonstrations (4M trajectories) and then improved through self-play (18M trajectories).
- **Competition Winners**: The battling track champion FoulPlay uses root-parallel MCTS search; the speedrunning champion Heatz employs Scripted Policy Distillation (SPD) — an LLM decomposes tasks and generates scripted policies, which are distilled into a neural network via imitation learning and then refined with RL.
- **Evaluation Protocol**: Battling uses FH-BT scoring with bootstrap uncertainty and a minimum sample-size requirement. Speedrunning evaluates completion percentage (across 15 milestones) and completion time, reporting both wall-clock time and step counts to distinguish inference speed from sample efficiency.

## Key Experimental Results

### Main Results

**Battling Track: Agent Performance on the Showdown Ladder vs. Humans**

| Method Category | Representative Agent | Gen 1 OU | Gen 9 OU | Notes |
|---|---|---|---|---|
| RL Baseline (new) | Metamon-Transformer-200M | Near Top 500 humans | Near Top 500 humans | Strongest baseline; still below human level |
| RL Baseline (prior) | Prior Metamon | Moderate | Moderate | Significantly improved in this work |
| LLM Baseline | GPT-4o + minimax harness | Below RL | Below RL | Improves under Extended Timer |
| LLM Baseline | PC-Llama3.1-8B | Meaningful but low | Meaningful but low | Small model + harness remains effective |
| Human | Top 500 Showdown | Reference upper bound | Reference upper bound | Current AI ceiling remains below superhuman |

**Speedrunning Track: First Gym Completion**

| Method | Completion Time | Steps | Type |
|---|---|---|---|
| Human speedrun best | 18 min | — | Human |
| Human average | 1:22:05 | — | Human |
| Heatz (Champion, SPD) | 40:13 | 1,608 | RL distillation |
| Hamburg (Runner-up, PPO) | ~1:20 | — | RL |
| Deepest (Judge's Choice) | Rank 5 | 649 (fewest) | LLM harness |
| Gemini 3 Flash (organizers) | ~2:24 | — | LLM harness |
| Claude Sonnet 4.5 (organizers) | 6:25–20:45 (high variance) | — | LLM harness |
| anthonys (best pure LLM) | 1:29:17 | — | LLM harness |

### Ablation Study

**BenchPress Orthogonality Analysis: Pokémon vs. 49 Standard LLM Benchmarks**

| Analysis Dimension | Metric | Result |
|---|---|---|
| Spearman correlation with best benchmark | max $\rho$ | 0.77 |
| Mean correlation with all benchmarks | mean $|\rho|$ | 0.45 |
| Rank-2 SVD explained variance (standard benchmarks) | variance share | 91% |
| Rank-2 SVD explained variance (GXE) | variance share | only 27% |

**Competition Participation Statistics**

| Metric | Count |
|---|---|
| Participating teams | 100+ |
| Total matches played | 100K+ |
| Discord members | 650+ |
| Valid submissions | 150+ |
| Speedrunning teams completing all milestones | 6/22 |
| Battling qualifier: RL/Search vs. LLM (of 16 slots) | 13 RL-scaled, 3 independent RL/MCTS |

### Key Findings

1. **Specialist methods decisively outperform general-purpose LLMs**: Both tracks show a consistent pattern — RL and search methods comprehensively surpass LLM-based approaches. No LLM method qualified among the 16 battling finalist slots. The speedrunning champion Heatz (RL distillation) is more than 2× faster than the best pure-LLM entry.
2. **LLM as prior, RL as refinement**: The success of Heatz — using an LLM to decompose tasks and generate initial policies, then distilling and refining via RL — suggests a promising hybrid paradigm.
3. **Pokémon exposes reasoning failures invisible to standard benchmarks**: Weaker models exhibit "panic behavior" — compounding additional errors following minor tactical mistakes. Different model families display distinct failure modes: memory corruption cascades, objective oscillation, over-committed planning, and computational paralysis. These failure modes do not manifest in programming or mathematics benchmarks.
4. **Pokémon battling is orthogonal to standard LLM benchmarks**: The BenchPress matrix over 83 models × 49 benchmarks is approximately rank-2, and 5 benchmarks suffice to predict the remaining 44 (error ~7 points). However, Pokémon GXE breaks this low-rank structure — Rank-2 SVD explains only 27% of its variance.
5. **Harness is a necessary condition, not an optional optimization**: Frontier VLMs achieve 0% completion on the speedrunning track without any harness. Common CLI-agent architectures (Claude Code, Codex CLI, Gemini CLI) also fail to maintain consistent sequential decision-making over thousands of steps.

## Highlights & Insights

- **Broad scope**: Simultaneously covering adversarial game-playing (battling) and long-horizon planning (speedrunning) within a unified infrastructure is unique among existing game AI benchmarks.
- **Unprecedented data scale**: 20M+ battle trajectories and a 200K+ team dataset far exceed any prior Pokémon AI work.
- **Rigorous orthogonality argument**: Rather than simply claiming a capability gap exists, the BenchPress matrix quantitatively demonstrates that Pokémon measures abilities orthogonal to the existing evaluation suite.
- **Harness × Model decoupling framework**: For the first time, model capability and external toolchain contributions are systematically separated, resolving the core incomparability of prior ad hoc demonstrations.
- **Living benchmark design**: The transition from competition to a continuously operating leaderboard, combined with the natural evolution of the competitive metagame, ensures the benchmark resists saturation.

## Limitations & Future Work

1. **Speedrunning track covers only the first gym**: While reasonable for cost reduction, evaluation over the full game would more comprehensively test long-term memory and planning.
2. **Strongest RL baseline remains below human top performance**: The current ceiling falls short of peak human level, indicating room for improvement while also confirming that the benchmark is genuinely challenging.
3. **Visual perception is insufficiently tested**: The battling track uses symbolic state representations rather than visual input; the speedrunning track includes visual frames but is limited to simple pixel-level input.
4. **Insufficient open-source model baselines**: No open-source model completes the full RPG, limiting participation from the broader research community.
5. **Four open challenges merit follow-up**: VLM-SLAM (visual-spatial localization), bridging the LLM–RL gap, open-source full-game completion, and approaching human speedrun times.

## Related Work & Insights

- **Relationship to FightLadder/Hanabi**: These works focus on competitive evaluation under imperfect information but do not incorporate long-horizon planning. PokéAgent covers both dimensions.
- **Relationship to NetHack/BALROG**: These benchmarks test long-horizon reasoning but are primarily single-player and non-adversarial. Pokémon adds the adversarial dimension.
- **Relationship to MineRL/Neural MMO**: These NeurIPS competition predecessors each advance a single research axis. PokéAgent's dual-track design enables joint evaluation.
- **LLM as prior + RL as refinement paradigm**: Heatz's success aligns with the recent research direction of "LLM-guided RL" and may generalize to robotics and other decision-making domains.
- **Implications for benchmark design**: The BenchPress orthogonality analysis methodology can be applied to assess whether any new benchmark genuinely measures capabilities not covered by existing evaluation suites.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-track design and living benchmark concept are novel, and the orthogonality argument is distinctive; however, core components (PokéChamp/Metamon) extend prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 100+ competing teams, 100K+ battles, and multi-dimensional analysis (BenchPress orthogonality, harness decoupling, cross-track insights) constitute an exceptionally thorough evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and figures are information-dense, though the paper is lengthy with a high proportion of competition logistics.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a significant gap in game AI benchmarking; the orthogonality finding has broad implications for the LLM evaluation community; the living benchmark design sustains long-term research impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)
- [\[NeurIPS 2025\] Scale-invariant Attention](scale-invariant_attention.md)
- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[NeurIPS 2025\] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)

</div>

<!-- RELATED:END -->
