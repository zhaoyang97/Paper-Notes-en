---
title: >-
  [Paper Note] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm
description: >-
  [NeurIPS 2025][Reinforcement Learning][Temporal perception] DRL agents trained in a simplified Overcooked environment to perform either a single task (temporal production) or a dual task (temporal production + numerical…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Temporal perception"
  - "dual-task paradigm"
  - "DRL agent"
  - "cognitive science"
  - "LSTM neural dynamics"
date: 2026-05-08
content_hash: 436f6aff706c2cc4
---

# Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm

**Conference**: NeurIPS 2025
**arXiv**: [2511.01415](https://arxiv.org/abs/2511.01415)  
**Code**: Unavailable  
**Area**: Reinforcement Learning
**Keywords**: Temporal perception, dual-task paradigm, DRL agent, cognitive science, LSTM neural dynamics

## TL;DR

DRL agents trained in a simplified Overcooked environment to perform either a single task (temporal production) or a dual task (temporal production + numerical comparison) exhibit significantly greater temporal overproduction across all four target durations in the dual-task condition—an emergent behavior that closely parallels the time overestimation phenomenon observed in human temporal perception research under dual-task paradigms.

## Background & Motivation

"How do humans perceive time?" is a long-standing core question in psychology, neuroscience, and cognitive science. Extensive behavioral research has shown that **temporal production performance is significantly disrupted by concurrent cognitive tasks** (the dual-task interference effect)—people tend to overestimate time intervals. This suggests that temporal processing shares neural resources with other cognitive processes.

The present work explores this question from an artificial intelligence perspective: **Do DRL agents exhibit analogous temporal interference effects?** If so, this would provide a valuable point of comparison, as agents are not explicitly programmed with any timers or biologically inspired neural structures—all behaviors emerge from reward signals alone.

The distinctive motivation is that reinforcement learning is employed not to solve a practical problem, but **as a research tool for cognitive science**, probing behavioral and mechanistic parallels between deep neural networks and biological systems.

## Method

### Overall Architecture

Two task variants are designed within a simplified Overcooked environment. DRL agents are trained separately on each variant, and temporal processing differences are subsequently analyzed at both the behavioral and neural dynamics levels.

### Key Designs

1. **Single-Task (T) Environment**: A 5×3 grid world containing an onion dispenser, an oven, and a delivery counter. The agent's goal is to deliver as many soups as possible within 100 steps: pick up onion → place in oven → wait for cooking (target duration) → retrieve soup → deliver. A critical detail is that the oven has an invisible internal timer; soup can only be retrieved after the target duration has elapsed, with no constraint on retrieval timing. A reward of "+1" is given only upon delivery.

2. **Dual-Task (T+N) Environment**: A numerical comparison counter is added to the single-task setup. After the onion is placed in the oven, the counter activates for 4 steps, displaying a digit from 1 to 10. The agent must execute "interact" when the digit is <5 and "wait" when the digit is ≥5; a correct response yields "+1." After 4 steps, the task reverts to the single-task mode. **Key constraint**: across all target durations, the agent retains sufficient time after the number game to check the oven at least once before the target duration elapses.

3. **"First Oven Check" Metric**: Defined as the oven timer value at the moment the agent first issues an "interact" action toward an oven currently cooking. If the agent checks continuously until retrieval, the time of the first interact is recorded; if intervening actions occur, the trial is not recorded. This metric captures the moment at which the agent **believes** the target duration has been reached, thereby quantifying temporal production accuracy.

### Loss & Training

- Recurrent PPO (Proximal Policy Optimization) from Stable-Baselines3 is used
- Agent architecture: CNN (processing spatial information from the 5×3 grid, primarily 1×1 convolutions) → LSTM (256 hidden units, capturing temporal dependencies) → MLP (64 hidden units)
- Entropy coefficient set to 0.05 to encourage exploration (inducing the agent to move around while waiting for soup, which is especially important for dual-task training)
- Each task type is trained independently across 4 target durations (7, 8, 9, 10 steps), yielding 8 agents in total
- Training runs for 100,000 steps, sufficient to learn the task and reach comparable performance levels

## Key Experimental Results

### Main Results

Dual-task agents significantly overproduce time across all target durations (25 episodes, 100 steps each):

| Target Duration | Single-Task (T) Mean First Check | Dual-Task (T+N) Mean First Check | Significance |
|---|---|---|---|
| 7 steps | ~7.0 | Significantly higher | p < 0.001 |
| 8 steps | ~8.0 | Significantly higher | p < 0.001 |
| 9 steps | ~9.0 | Significantly higher | p < 0.001 |
| 10 steps | ~10.0 | Significantly higher | p < 0.001 |

### Ablation Study

| Analysis Dimension | Single-Task (T) | Dual-Task (T+N) | Notes |
|---|---|---|---|
| Performance (soups delivered) | Higher | Lower but comparable | Worst case (duration 10) reaches ~53% of single-task |
| LSTM PCA | Complex oscillations, reset between trials | Similar but distinct pattern | No clear evidence of a dedicated timer |
| FFT (durations 7, 10) | Frequency ≥ target duration | Frequency < target duration | Suggests overproduction, but no causal claim |
| No-LSTM variant | Can complete task | — | But exhibits no timing behavior; relies purely on state changes |

### Key Findings

- **Emergent temporal interference**: DRL agents significantly overproduce time under dual-task conditions, consistent with human behavioral research. This behavior is entirely emergent—no explicit timers or biologically inspired structures were provided.
- **Comparable performance with clear temporal bias**: Although dual-task agents show reduced total output, they still complete the task effectively (~53% in the worst case), indicating the temporal bias does not stem from task failure.
- **No dedicated timer in the LSTM**: PCA analysis reveals that the LSTM captures trial-boundary information (oscillations reset after each soup delivery), but no specialized timing neurons or explicit timing mechanisms are identified.
- **FFT suggests frequency differences**: Single-task agents' dominant frequency is at or above the target frequency (accurate or underestimated time); dual-task agents' dominant frequency falls below the target (overestimated time), consistent with behavioral results.

## Highlights & Insights

- This work pioneering replicates a classical finding from cognitive psychology (dual-task temporal interference) within a DRL framework, providing a new cross-disciplinary research tool.
- Emergence is the central highlight: agents were not programmed to simulate human temporal perception; these behaviors arise entirely from reward signals and task structure.
- The LSTM is a necessary component for temporal processing (no-LSTM variants exhibit no timing behavior), yet its internal mechanism differs from the simple counter patterns reported in prior studies.
- The findings suggest that temporal perception may not require dedicated neural circuits, but rather emerges from neurons distributed across circuits serving other cognitive functions—consistent with the neuroscientific view that "time is encoded by neurons participating in other cognitive functions."

## Limitations & Future Work

- The reward structure could be improved: the dual-task condition uses immediate rewards (numerical comparison), which does not fully correspond to the delayed rewards typically used in human experiments.
- Only 4 target durations (7–10 steps) are tested; a broader range would strengthen the generalizability of the findings.
- Neural dynamics analysis is limited to the LSTM layer and is exploratory in nature; it does not extend to the CNN or MLP layers.
- No causal analysis is conducted—behavioral similarity does not imply mechanistic similarity; more rigorous ablation and intervention experiments are needed to establish causal links.
- The environment is highly simplified (5×3 grid), representing a substantial gap from the complexity of human temporal perception.

## Related Work & Insights

- **Deverett et al. (2019)**: Identified counter-like neural activity in DRL agents' LSTM layers during a dedicated temporal reproduction task—the present work finds no analogous pattern in a more complex, embedded task.
- **Lin et al. (2023)**: Demonstrated biologically plausible temporal encoding mechanisms in recurrent neural networks (oscillations, ramping activity, time cells).
- **Roseboom et al. (2019)**: A model of subjective time perception based on perceptual classification network activity.
- **Insight**: DRL can serve not only as an engineering tool but also as a "model organism" for cognitive science—enabling precise controlled experiments to study the emergent conditions of cognitive mechanisms.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First demonstration of emergent dual-task temporal interference in DRL; cross-disciplinary innovation
- **Experimental Thoroughness**: ⭐⭐⭐ Behavioral analysis is rigorous (statistically sound), but neural mechanism analysis remains preliminary
- **Writing Quality**: ⭐⭐⭐⭐ Cross-disciplinary narrative is fluent; cognitive science background is well-introduced
- **Value**: ⭐⭐⭐⭐ Opens a new direction for DRL–cognitive science intersectional research, though far from practical application
- **Overall**: ⭐⭐⭐⭐ Strong cross-disciplinary contribution; emergent behavior is compelling; mechanistic analysis warrants further development

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](../../AAAI2026/reinforcement_learning/think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](temporal-difference_variational_continual_learning.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

</div>

<!-- RELATED:END -->
