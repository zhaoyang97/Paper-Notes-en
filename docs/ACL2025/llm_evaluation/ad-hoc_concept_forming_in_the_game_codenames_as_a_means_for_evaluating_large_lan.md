---
title: >-
  [Paper Note] Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models
description: >-
  [ACL 2025][LLM Evaluation][Codenames] The board game Codenames is implemented as an LLM evaluation benchmark, where LLMs play the roles of both Spymaster (clue giver) and Field Operative (guesser) against a deterministic opponent across 13 experimental setups of varying difficulty. Among 14 evaluated models, the best-performing model (o3-mini) achieves a win rate of only 49%, revealing substantial limitations of LLMs in vocabulary association, strategic positioning…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Codenames"
  - "Game Evaluation"
  - "Concept Generation"
  - "Pragmatic Reasoning"
  - "Cooperative Games"
date: 2026-05-08
content_hash: 95ad9f149a268dc2
---

# Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.11707](https://arxiv.org/abs/2502.11707)  
**Code**: [GitHub](https://github.com/clembench/clembench) (codenames directory)  
**Area**: LLM Evaluation  
**Keywords**: Codenames, Game Evaluation, Concept Generation, Pragmatic Reasoning, Cooperative Games

## TL;DR

The board game Codenames is implemented as an LLM evaluation benchmark, where LLMs play the roles of both Spymaster (clue giver) and Field Operative (guesser) against a deterministic opponent across 13 experimental setups of varying difficulty. Among 14 evaluated models, the best-performing model (o3-mini) achieves a win rate of only 49%, revealing substantial limitations of LLMs in vocabulary association, strategic positioning, and error correction.

## Background & Motivation

**Background**: Evaluation of LLMs faces an "evaluation crisis"—the traditional reference-based comparison paradigm is unsuitable for conversational scenarios, and training data contamination is severe. Games are emerging as dynamic, interactive evaluation environments.

**Limitations of Prior Work**: (a) Existing benchmarks fail to adequately test the pragmatic reasoning and collaborative capabilities of LLMs; (b) Codenames requires ad-hoc concept forming, theory of mind, and co-creation capabilities, making it a highly challenging evaluation scenario; (c) Prior research on Codenames with LLMs involved pitting two LLMs against each other, where non-determinism led to non-reproducible outcomes.

**Key Insight**: A deterministic simulated opponent (which reveals 1 word per turn) is utilized to eliminate randomness, and 13 experimental setups are designed by controlling variables such as word pools, word frequency, and ambiguity.

## Method

### Overall Architecture

Implemented based on the clembench framework. The LLM plays the roles of both Blue Team Spymaster and Field Operative, competing against a programmatic mock opponent. The Spymaster outputs "CLUE: clue word + TARGETS: list of target words", and the Field Operative outputs "GUESS: guess list" upon receiving the clue. The GameMaster manages the game flow and handles rule validation.

### Key Designs

1. **13 Experimental Setups**:
    - Risk levels: 1 vs. 5 assassin words
    - Word association: Easy (semantically related words assigned to the same team) vs. Difficult (semantically related words divided across different camps)
    - Opponent speed: Revealing 0/1/2 words per turn
    - Word frequency: High frequency vs. Low frequency
    - Ambiguity: Polysemous vs. Monosemous words
    - Abstractness: Concrete vs. Abstract words
    - **Design Motivation**: Conducting controlled-variable analysis to assess LLM performance across different linguistic and cognitive dimensions.

2. **Evaluation Metrics**:
    - clemscore: Comprehensive score (played percentage $\times$ quality score)
    - Sensitivity: Proportion of team words revealed
    - Efficiency: $\min(1, \frac{1}{2} \cdot \frac{\text{team words revealed}}{\text{turns}})$, with a baseline of 2 words per turn
    - Error classification: Target hallucination, guess hallucination, incorrect number of guesses, and using clue as guess

3. **Deterministic Opponent Design**:
    - **Function**: Simulates a fixed-strategy opponent that reveals $n$ words per turn.
    - **Design Motivation**: Eliminates the non-determinism introduced when both sides are LLMs, ensuring experimental reproducibility.

## Key Experimental Results

### Main Results

Overall ranking of 14 models (130 game instances):

| Rank | Model | clemscore | Played% | Quality Score |
|------|------|-----------|---------|--------------|
| 1 | o3-mini | 49.2 | **100.0** | 49.2 |
| 2 | Claude-3.5 | 46.9 | 93.8 | 50.0 |
| 3 | GPT-4o | 45.4 | 93.8 | 48.4 |
| 4 | Deepseek-r1 | 45.4 | 85.4 | **53.2** |
| 5 | Gemini-2.0 | 37.7 | 96.2 | 39.2 |
| 14 | Llama-3.1-8B | 14.6 | 52.3 | 27.9 |

### Ablation Study

Comparison of performance across specific experimental difficulties (Quality Score):

| Experiment | o3-mini | GPT-4o | Deepseek-r1 |
|------|---------|--------|-------------|
| Word Association - Easy | 100.0 | 100.0 | 100.0 |
| Word Association - Difficult | 20.0 | 10.0 | 28.6 |
| Opponent Speed - None (No words revealed) | 80.0 | 77.8 | 62.5 |
| Opponent Speed - Difficult (Revealing 2 words) | **0.0** | **0.0** | **22.2** |
| Concrete Words | 80.0 | 50.0 | 88.9 |
| Abstract Words | 20.0 | 60.0 | 50.0 |

### Key Findings

1. **All models pass the easy association setup with 100%, but drop to <30% in the difficult mode**: This indicates that LLMs can identify obvious lexical categories but lack creative cross-category association capabilities.
2. **Deepseek-r1 is the only model that achieves victories in the "difficult opponent" mode (22.2%)**: Reasoning models exhibit a clear advantage in strategic planning.
3. **Low-frequency words are not more challenging than high-frequency words** (contrary to human intuition): LLM vocabulary knowledge coverage is unaffected by word frequency.
4. **Abstract words are indeed more difficult** (consistent with humans): However, GPT-4o is an exception (Abstract 60% > Concrete 50%).
5. **Open-source models primarily fail due to instruction-following**: Errors such as target hallucinations, guess hallucinations, and using clues as guesses are 5 to 10 times more frequent in open-source models.
6. **Deepseek-r1 is the most efficient (averaging 2.2 words per turn)**, but its aggressive strategy also triggers more assassin words.

## Highlights & Insights

- **Advantages of Game-based Evaluation**: Easy to generate infinite game instances (avoiding data contamination), and interactive evaluation is closer to real-world application.
- **In-depth Strategic Analysis**: Reveals strategic variations among models through a three-dimensional analysis of efficiency, sensitivity, and error categories.
- **Compelling Case Studies**: Presents a complete walkthrough of o3-mini attempting to target 9 words in an abstract-word game but failing due to revealing 4 opponent words.
- **Trade-offs of Reasoning Models**: Deepseek-r1 is strategically the most aggressive but exhibits a latency of 111 seconds per query (compared to GPT-4o's 0.81 seconds).

## Limitations & Future Work

- Limited to English; multilingual expansion remains to be done.
- The internal reasoning process during clue generation is not deeply analyzed.
- The deterministic opponent is overly simplified and does not test genuine game-theoretic confrontation.
- Complex game rules and high instruction-following failure rates may overshadow actual differences in reasoning capability.

## Related Work & Insights

- **clembench Framework**: The LLM game evaluation framework upon which this work is based.
- **BigBench Codenames**: Evaluates emergent abilities but lacks interactivity and controlled experiments.
- **Insight**: Game-based evaluation not only tests linguistic abilities but also reveals advanced cognitive skills such as strategic thinking, risk management, and collaboration.

## Rating

- Novelty: ⭐⭐⭐⭐ Transforms Codenames into a systematic LLM evaluation framework with 13 carefully designed experimental setups.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 14 models using multi-dimensional analysis, combining qualitative and quantitative results.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, rich in figures/tables, and features vivid case studies.
- Value: ⭐⭐⭐⭐ Addresses the gap in evaluating LLMs' pragmatic reasoning and collaborative capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)

</div>

<!-- RELATED:END -->
