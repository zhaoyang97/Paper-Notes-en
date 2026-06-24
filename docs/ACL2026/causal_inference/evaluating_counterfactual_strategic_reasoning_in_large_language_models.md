---
title: >-
  [Paper Note] Evaluating Counterfactual Strategic Reasoning in Large Language Models
description: >-
  [ACL2026][Causal Inference][counterfactual games] This paper evaluates the strategic adaptation capabilities of LLMs using label perturbations, payoff perturbations, and joint counterfactual versions of the Repeated Prisoner's Dilemma and Rock-Paper-Scissors. It finds that while many models appear proficient in familiar games, they continue to apply templated strategies even after payoff structures are altered.
tags:
  - "ACL2026"
  - "Causal Inference"
  - "counterfactual games"
  - "strategic reasoning"
  - "Prisoner's Dilemma"
  - "Rock-Paper-Scissors"
  - "opponent comprehension"
date: 2026-05-08
content_hash: 3bb97b098ac23104
---

# Evaluating Counterfactual Strategic Reasoning in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2603.19167](https://arxiv.org/abs/2603.19167)  
**Code**: https://github.com/dimjimitris/llm_gm_thesis  
**Area**: LLM Reasoning / Game Evaluation / Counterfactual Robustness  
**Keywords**: counterfactual games, strategic reasoning, Prisoner's Dilemma, Rock-Paper-Scissors, opponent comprehension

## TL;DR
This paper evaluates the strategic adaptation capabilities of LLMs using label perturbations, payoff perturbations, and joint counterfactual versions of the Repeated Prisoner's Dilemma and Rock-Paper-Scissors. It finds that while many models appear proficient in familiar games, they continue to apply templated strategies even after payoff structures are altered.

## Background & Motivation
**Background**: LLMs have been extensively utilized in multi-agent cooperation, competition, and game simulations. Researchers often observe whether models can cooperate, compete, identify opponent strategies, and approach equilibrium behavior through structured games such as the Prisoner's Dilemma, Rock-Paper-Scissors, and Matching Pennies.

**Limitations of Prior Work**: Conventional game evaluations tend to overestimate model capabilities. Models may memorize templates like "cooperate/defect in Prisoner's Dilemma" or "randomize in Rock-Paper-Scissors" rather than recalculating strategies based on the payoff matrix. Once action labels are renamed or payoff structures are modified counterfactually, fluent explanations do not necessarily translate into correct actions.

**Key Challenge**: True strategic reasoning requires a model to perform conditional updates based on current environmental labels, payoffs, and historical interactions; however, LLM behavior may stem largely from canonical game patterns encountered during pre-training. These are difficult to distinguish in default games, necessitating counterfactual interventions to decouple surface recognition from incentive sensitivity.

**Goal**: To construct a compact, controllable, and reproducible experimental framework to diagnose label robustness, payoff sensitivity, opponent modeling, and token-normalized efficiency, determining whether the model understands the current game or is reproducing familiar templates.

**Key Insight**: The authors select two complementary games: the Prisoner's Dilemma to examine dynamic adaptation between cooperation and defection, and Rock-Paper-Scissors to examine randomization, pattern exploitation, and three-action equilibria. Subsequently, label perturbations, payoff perturbations, and joint perturbations are applied to both, forcing the model to re-interpret action meanings and payoff structures.

**Core Idea**: Use counterfactual label/payoff interventions in repeated games to distinguish an LLM's ability to "state a strategy" from its ability to "execute a strategy according to new incentives."

## Method

### Overall Architecture
This study does not train any models but establishes a behavioral evaluation framework: the LLM under test is treated as a player participating in repeated rounds against an instance of the same model or an algorithmic opponent. Actions, payoffs, opponent comprehension speed, cooperation rates, and token consumption are recorded throughout. Each experiment follows a pipeline of "specified game and perturbation type → LLM decision-making per round via a specific prompting format → accumulation of payoffs and behavioral statistics." The Prisoner's Dilemma (PD) is repeated for 16 rounds, and Rock-Paper-Scissors (RPS) for 24 rounds. Non-self-consistency players repeat experiments 5 times, while self-consistency players repeat 2 times. The essence lies in four tiers of game settings—Default, Label-based, Payoff-based, and Joint Counterfactual—to decouple the model's ability to recite strategies from its ability to execute them under new incentives. Opponents include other LLMs and algorithmic strategies such as SREP, PP, MF/TFT, and AP, covering LLM-LLM coordination, predictable opponent exploitation, and adaptive opponent confrontation.

### Key Designs

**1. Counterfactual Game Construction: Separating Surface Labels and Deep Payoffs into Orthogonal Stressors**

Observing default PD/RPS alone cannot determine if a model is truly reading the payoff matrix or applying a canonical pattern from pre-training. Therefore, the authors apply two independent perturbations to the same game. On the label dimension, action names in PD are changed (e.g., C/D to Stag/Hare) while the payoff structure remains unchanged to test for label anchoring. On the payoff dimension, a "payoff-based counterfactual" is applied (e.g., transforming PD into a Stag Hunt form), rewriting the "strict defection dominance" into a structure requiring coordination. In RPS, the magnitude of specific win/loss combinations is amplified, ensuring that the original uniform mixed strategy is no longer the equilibrium. By orthogonally combining these perturbations, failure modes of label anchoring and incentive rigidity can be diagnosed separately.

**2. Opponent Comprehension Metric: Quantifying Modeling Speed via the Earliest Stable Dominance Round**

Total scores reflect total earnings but do not distinguish whether a model understood the opponent early or succeeded by chance later. Thus, the authors define $m$ as the earliest round such that from that round until the end of the game, the LLM achieves a payoff no lower than the opponent's in at least $t_p=90\%$ of the subsequent rounds. A smaller $m$ indicates earlier opponent modeling; if $m$ exceeds the game length, it is determined that no stable understanding was reached. This metric isolates "dynamic adaptation speed" from cumulative payoffs, separating models that perceive opponents early from those that recover purely by luck.

**3. Performance and Efficiency Joint Evaluation: Distinguishing "Better Play" from "More Expensive Explanations"**

Reasoning models often produce longer Chain-of-Thought (CoT) outputs, but additional deliberation does not necessarily yield faster adaptation. Besides total points, cooperation/action distribution, and failure rate, the authors define efficiency as $\textit{points}/\textit{tokens}\times c$ (default $c=1000$) to explicitly calculate the payoff per kilotoken. Combined with the previous metrics, this identifies "reasoning-overhead mismatches" where scores are acceptable but token costs are excessive, preventing lengthy explanations from being misread as superior strategic capability.

Specifically, the default PD payoffs are $(C,C)=(4,4)$, $(C,D)=(1,6)$, $(D,C)=(6,1)$, and $(D,D)=(2,2)$, with cumulative scores for 16 rounds ranging from 16 to 96. In RPS, win/loss/tie payoffs per round are $1/-1/0$, with scores for 24 rounds ranging from -24 to 24. The payoff-based counterfactual for RPS increases the magnitude of the Rock-Paper combination to 3, shifting the theoretical equilibrium from a uniform distribution to $\pi^*(R)=0.2, \pi^*(P)=0.2, \pi^*(S)=0.6$. If a model persists with uniform randomization, it exposes its reliance on a canonical equilibrium.

## Key Experimental Results

### Main Results

| Setting | Metric | Representative Result | Explanation | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Default PD vs SREP | Total points | Baseline for continuous $(D,D)$ is 32; most LLMs cluster around 30 | Models usually identify continuous defection and approximate the optimal response | Simple algorithmic opponents are easier |
| Default PD vs LLM | Total points | Claude 3.5/3.7 and Llama 3.3-70B reach 64 points under various prompting | 64 points corresponds to 16 rounds of full mutual cooperation | Some models coordinate stably in LLM-LLM setups |
| Default PD | Instability cases | Mistral Large varies from 18.6±10.6 to 29.8±2.2 under SREP; Claude 4/DeepSeek R1 range from 31.4±0.0 to 49.4±15.5 in LLM-LLM | Weaker or over-reasoning models may be more unstable | High capability does not equal strategic stability |
| Default RPS | Opponent comprehension | Claude 3.5 Sonnet v2 zero-shot: 10.6±13.1 vs ZS, 21.4±4.6 vs SPP, 19.6±5.6 vs CoT | Opponent modeling in RPS is slower and closer to the 24-round horizon | Three-action games with no dominant strategy are harder |
| RPS payoff counterfactual | Theoretical equilibrium | Shift from uniform $(1/3,1/3,1/3)$ to $(0.2,0.2,0.6)$ | Persistence in uniform distribution indicates failure to recalculate based on new payoffs | Payoff perturbation best exposes templating |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Label-only counterfactual | Degradation is usually moderate | Strong models remain relatively stable; Mistral is more prone to fluctuations due to action renaming |
| Payoff-based counterfactual | Stronger degradation | Requires recalculating incentives, especially shifting from uniform to biased mixed strategies in RPS |
| Joint counterfactual | Maximum pressure | When labels and payoffs change simultaneously, near-horizon comprehension and high variance are more common |
| CoT / thinking variants | Inconsistent effects | Helpful for some strong models, but can lead to overthinking or distrust tendencies in Claude 4 or DeepSeek R1 |
| Self-consistency | Reduces variance but not fundamental tendencies | Often reinforces existing behavioral patterns rather than correcting erroneous strategies |

### Key Findings
- Payoff-based counterfactuals are more diagnostic than label-only ones, as they force models to recalculate the payoff structure rather than just processing action name changes.
- Performance in default games does not represent counterfactual robustness. Claude 3.7 is the most stable overall; Claude 4 is strong in RPS but shows mixed counterfactual stability; Llama 3.3 is stable in PD cooperation scenarios but weaker in RPS/payoff shifts.
- More "thinking" is not necessarily better. Thinking-enabled variants increase token consumption in some settings without proportional improvements in total points or opponent comprehension.
- RPS exposes delayed opponent modeling more effectively than PD because there is no simple convergence to cooperation; models must maintain near-equilibrium behavior or identify exploitable patterns.

## Highlights & Insights
- The value of this paper lies in its "clean" evaluation design: label changes, payoff changes, and joint changes map to different failure modes, allowing for the decomposition of template memory, label anchoring, and incentive rigidity.
- Opponent comprehension is more explanatory than the final score. Many models may achieve acceptable final scores, but a late $m$ indicates they "stumbled" upon the solution through interaction rather than understanding the opponent from the start.
- The $(0.2, 0.2, 0.6)$ RPS payoff counterfactual is critical. It demonstrates that "randomization" is not always correct; continuing uniform randomization after payoff changes reveals canonical equilibrium persistence.
- The paper warns that in agent evaluation, the Chain-of-Thought of strong models may increase defensiveness, skepticism, or exploration noise. Longer reasoning processes do not automatically imply more stable strategic execution.

## Limitations & Future Work
- The evaluation only covers two-player, synchronous, fixed-payoff repeated games. Its ecological validity is limited compared to real-world multi-party negotiations, markets, auctions, or open-ended collaborations.
- Algorithmic opponents and payoff structures are preset; more adaptive opponents, multi-agent populations, or incomplete information games might produce different conclusions.
- All metrics are derived from observable actions and tokens; inferring "understanding" from them is behavioral and does not equate to explaining internal model mechanisms.
- The selection of models, prompts, and counterfactual types is not exhaustive. Future work could include more models, complex payoff transformations, natural language rule ambiguities, and consistency analysis of internal reasoning traces.

## Related Work & Insights
- **vs. Conventional Game Evaluation**: Conventional PD/RPS only observe if models can play familiar games; this paper tests if models actually update strategies based on current rules through counterfactual perturbations.
- **vs. Static Counterfactual QA**: Many counterfactual benchmarks are one-off inputs/outputs; this paper incorporates counterfactuals into repeated interactions to observe adaptation speed and history dependency.
- **vs. Multi-agent Cooperation Evaluation**: General agent benchmarks focus on task success rates; this paper emphasizes multi-dimensional diagnostics of payoffs, opponent modeling, efficiency, and failure rates, making it suitable as a small-scale stress test for agentic LLMs.
- **Insight**: When designing LLM benchmarks, researchers should include control groups with "rule-preserved but label-changed" and "label-preserved but payoff-changed" variations; otherwise, "familiar template execution" is easily misidentified as "abstract reasoning."

## Rating
- Novelty: ⭐⭐⭐⭐ Diagnosing LLM strategic robustness via counterfactual repeated games provides compact and explanatory problem settings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple frontier LLMs, prompting strategies, opponent types, and metrics, though the variety of games remains relatively low.
- Writing Quality: ⭐⭐⭐⭐ Clear logic in the main text with sufficient numerical data in appendices; some tables are very large and requires summary integration.
- Value: ⭐⭐⭐⭐ Directly relevant to research in LLM agent evaluation, strategic reasoning, and counterfactual robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](../../ECCV2024/causal_inference/learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)

</div>

<!-- RELATED:END -->
