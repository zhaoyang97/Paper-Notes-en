---
title: >-
  [Paper Note] Evaluating Counterfactual Strategic Reasoning in Large Language Models
description: >-
  [ACL 2026][Causal Inference][Prisoner's Dilemma] This paper evaluates the strategic adaptation of LLMs using label perturbations, payoff perturbations, and joint counterfactual versions of the repeated Prisoner's Dilemma and Rock-Paper-Scissors. It finds that many models appear competent in familiar games but continue to use templated strategies even after payoff str
tags:
  - ACL 2026
  - Causal Inference
  - Prisoner's Dilemma
  - Rock-Paper-Scissors
  - opponent comprehension
date: 2026-05-08
content_hash: a375b6ea41e31df7
---
# Evaluating Counterfactual Strategic Reasoning in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2603.19167](https://arxiv.org/abs/2603.19167)  
**Code**: https://github.com/dimjimitris/llm_gm_thesis  
**Area**: LLM Reasoning / Game Evaluation / Counterfactual Robustness  
**Keywords**: Counterfactual games, Strategic reasoning, Prisoner's Dilemma, Rock-Paper-Scissors, opponent comprehension

## TL;DR
This paper evaluates the strategic adaptation of LLMs using label perturbations, payoff perturbations, and joint counterfactual versions of the repeated Prisoner's Dilemma and Rock-Paper-Scissors. It finds that many models appear competent in familiar games but continue to use templated strategies even after payoff structures are altered.

## Background & Motivation
**Background**: LLMs are extensively used for multi-agent cooperation, competition, and game simulations. Researchers often observe whether models can cooperate, compete, identify opponent strategies, and approach equilibrium behavior through structured games like the Prisoner's Dilemma, Rock-Paper-Scissors, and Matching Pennies.

**Limitations of Prior Work**: Conventional game evaluations prone to overestimating model capabilities. Models might memorize templates such as "cooperate/defect in Prisoner's Dilemma" or "randomize in Rock-Paper-Scissors" rather than recalculating strategies based on the payoff matrix. Once action labels are renamed or payoff structures are counterfactually modified, fluent explanations do not necessarily translate into correct actions.

**Key Challenge**: True strategic reasoning requires models to conditionally update based on current environment labels, payoffs, and historical interactions; however, LLM behavior may stem more from canonical game patterns encountered during pre-training. These two are difficult to distinguish in default games and must be disentangled through counterfactual intervention to separate surface recognition from incentive sensitivity.

**Goal**: To construct a compact, controllable, and reproducible experimental framework to diagnose label robustness, payoff sensitivity, opponent modeling, and token-normalized efficiency, determining whether a model understands the current game or is merely reproducing familiar templates.

**Key Insight**: The authors select two complementary games: the Prisoner's Dilemma to examine dynamic adaptation between cooperation and defection, and Rock-Paper-Scissors to examine randomization, pattern exploitation, and three-action equilibrium. Subsequently, label perturbations, payoff perturbations, and joint perturbations are applied to both, forcing models to reinterpret action meanings and payoff structures.

**Core Idea**: Use counterfactual label/payoff interventions in repeated games to distinguish between an LLM's ability to "articulate a strategy" and its ability to "execute a strategy according to new incentives."

## Method

### Overall Architecture
This paper does not train any models but builds a behavioral evaluation framework: the LLM under test is treated as a player and engages in repeated rounds of play against instances of the same model or algorithmic opponents. Actions, payoffs, opponent comprehension speed, cooperation rates, and token consumption are recorded throughout. Each experiment follows a pipeline of "specifying game and perturbation type → prompting the LLM for round-by-round decisions → accumulating payoffs and behavioral statistics." The Prisoner's Dilemma (PD) is repeated for 16 rounds, and Rock-Paper-Scissors (RPS) for 24 rounds. Non-self-consistency players repeat 5 times, while self-consistency players repeat 2 times. Its essence lies in four game settings—default, label-based, payoff-based, and joint counterfactual—to layers of separation between "whether the model can restate a strategy" and "whether the model can execute a strategy based on new incentives." Label-based only changes action names while keeping payoffs constant; payoff-based rewrites the payoff matrix to invalidate original equilibria; and joint combines both for a maximum stress test. The opponent side includes other LLMs as well as algorithmic strategies such as SREP, PP, MF/TFT, and AP, covering LLM-LLM coordination, predictable opponent exploitation, and adaptive opponent confrontation.

### Key Designs

**1. Counterfactual Game Construction: Separating Surface Labels and Deep Payoffs into Two Orthogonal Stressors**

Observing only default PD/RPS cannot determine if a model is truly reading the payoff matrix or applying canonical patterns from pre-training. Therefore, the authors apply two independent perturbations to the same game. On the label dimension, PD's C/D tags are renamed to Stag/Hare while the payoff structure remains unchanged to test if the model is anchored by action names. On the payoff dimension, PD is converted into a Stag Hunt-style payoff-based counterfactual, changing "strict defection dominance" into a structure requiring coordination. In RPS, the magnitude of specific win/loss combinations is amplified, making the original uniform mixed strategy no longer an equilibrium. By orthogonally combining these perturbations, failure modes of label anchoring and incentive rigidity can be diagnosed separately.

**2. Opponent Comprehension Metric: Quantifying Opponent Modeling Speed via Earliest Stable Dominance**

Total points only reflect final earnings but cannot distinguish whether a model understood the opponent early on or relied on a late accidental turnaround. To this end, the authors define $m$ as the earliest round such that, from that round until the end of the game, the LLM receives a payoff no lower than the opponent's in at least $t_p=90\%$ of subsequent rounds. A smaller $m$ indicates earlier completion of opponent modeling; if $m$ exceeds the game length, it is judged as failing to reach stable understanding. This metric separates "dynamic adaptation speed" from cumulative payoffs, ensuring models that insightfully grasp opponents early are not conflated with those that recover points by chance.

**3. Performance and Efficiency Joint Evaluation: Distinguishing "Playing Better" from "Explaining More with Tokens"**

Reasoning-heavy models often output longer chains-of-thought, but extra deliberation does not necessarily yield faster adaptation. In addition to total points, cooperation/action distribution, and failure rate, the authors define efficiency as $\textit{points}/\textit{tokens}\times c$ (with default $c=1000$) to explicitly calculate payoff per thousand tokens. Combined with the two metrics above, it identifies reasoning-overhead mismatches where "scores are acceptable but token costs are enormous," avoiding the misinterpretation of verbose explanations as superior strategic capability.

Regarding specific payoff settings: the default PD payoff is $(C,C)=(4,4)$, $(C,D)=(1,6)$, $(D,C)=(6,1)$, $(D,D)=(2,2)$, with 16-round cumulative scores ranging from 16 to 96. Each round of RPS win/loss/tie is $1/-1/0$, with 24-round cumulative scores ranging from -24 to 24. The payoff-based counterfactual for RPS amplifies the magnitude of the Rock-Paper combination win/loss to 3. The theoretical equilibrium shifts from a uniform distribution to $\pi^*(R)=0.2, \pi^*(P)=0.2, \pi^*(S)=0.6$—if a model persists with uniform randomization, it exposes its reliance on a canonical equilibrium.

## Key Experimental Results

### Main Results

| Setting | Metric | Representative Result | Explanation | Conclusion |
|--------|------|------|----------|------|
| Default PD vs SREP | Total points | When SREP always defects, the $(D,D)$ baseline is 32 points; most LLMs cluster around 30 | Models usually identify constant defection and approximate optimal response | Simple algorithmic opponents are easier |
| Default PD vs LLM | Total points | Claude 3.5/3.7 and Llama 3.3-70B reach 64 points under various prompting | 64 corresponds to full mutual cooperation for 16 rounds | Some models coordinate stably in LLM-LLM play |
| Default PD | Instability cases | Mistral Large varies from 18.6±10.6 to 29.8±2.2 under SREP; Claude 4/DeepSeek R1 range from 31.4±0.0 to 49.4±15.5 in LLM-LLM | Weaker or over-reasoning models may be more unstable | High capability does not equal strategic stability |
| Default RPS | Opponent comprehension | Claude 3.5 Sonnet v2 zero-shot vs ZS is 10.6±13.1, vs SPP is 21.4±4.6, vs CoT is 19.6±5.6 | Opponent modeling in RPS is slower and closer to the 24-round horizon | Three-action games without dominant strategies are harder |
| RPS payoff counterfactual | Theoretical equilibrium | Shifted from uniform $(1/3, 1/3, 1/3)$ to $(0.2, 0.2, 0.6)$ | Persistence in uniform distribution shows failure to recalculate strategy | Payoff perturbation best exposes templating |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Label-only counterfactual | Degradation usually moderate | Strong models remain stable; Mistral and others are more prone to fluctuations due to renamed actions |
| Payoff-based counterfactual | Stronger degradation | Requires recalculating incentives, especially shifting from uniform randomization to biased mixed strategies in RPS |
| Joint counterfactual | Maximum pressure | Near-horizon comprehension and high variance are more common when both labels and payoffs change |
| CoT / thinking variants | Inconsistent effect | Helps some strong models, but can lead to overthinking or distrust tendencies in Claude 4, DeepSeek R1, etc. |
| Self-consistency | Reduces variance but not fundamental bias | Often reinforces existing behavioral patterns rather than correcting an erroneous strategy |

### Key Findings
- Payoff-based counterfactuals are more diagnostic than label-only ones because they force the model to recalculate the payoff structure rather than just processing action name changes.
- Default game performance does not represent counterfactual robustness. Claude 3.7 is the most stable overall; Claude 4 is strong in RPS but shows mixed counterfactual stability; Llama 3.3 is stable in PD cooperation scenarios but weaker in RPS/payoff shifts.
- Thinking more is not necessarily better. Thinking-enabled variants increase token consumption in some settings without a proportional increase in total points or opponent comprehension.
- RPS exposes delayed opponent modeling more than PD because there is no simple cooperative convergence point; models must maintain near-equilibrium behavior or identify exploitable patterns.

## Highlights & Insights
- The value of this paper lies in its "clean" evaluation design: label changes, payoff changes, and joint changes correspond to different failure modes, allowing for the disentanglement of template memory, label anchoring, and incentive rigidity.
- Opponent comprehension is more explanatory than final scores. Many models might achieve acceptable final scores, but a late $m$ indicates they stumbled into it through interaction rather than understanding the opponent from the start.
- The $(0.2, 0.2, 0.6)$ result in RPS payoff counterfactual is critical. It demonstrates that "randomization" is not always correct; continuing uniform randomization after payoff changes is evidence of canonical equilibrium persistence.
- The paper reminds us that in agent evaluation, the chain-of-thought of strong models may increase defensiveness, suspicion, or exploratory noise. Longer reasoning processes do not automatically equate to more stable strategy execution.

## Limitations & Future Work
- The evaluation only covers two-player, synchronous, fixed-payoff repeated games. Ecological validity is limited compared to real-world multi-party negotiations, markets, auctions, or open-ended collaborations.
- Algorithmic opponents and payoff structures are preset; more adaptive opponents, multi-agent population games, or incomplete information games could yield different conclusions.
- All metrics are derived from observable actions and tokens; inferring "understanding" from them is behavioral and not equivalent to explaining internal model mechanisms.
- The selection of models, prompts, and counterfactual types is not exhaustive. Future work could include more open/closed-source models, more complex payoff transformations, natural language rule ambiguity, and consistency analysis of internal reasoning traces.

## Related Work & Insights
- **vs Standard Game Evaluation**: Standard PD/RPS only observe if models can play familiar games. This paper tests if models actually update strategies according to current rules via counterfactual perturbations.
- **vs Static Counterfactual QA**: Many counterfactual benchmarks are one-off input-output tasks. This paper places counterfactuals into repeated interactions, allowing for the observation of adaptation speed and history dependence.
- **vs Multi-agent Cooperation Evaluation**: Standard agent benchmarks focus more on task success rates. This paper emphasizes multi-dimensional diagnosis of payoffs, opponent modeling, efficiency, and failure rates, making it suitable as a small-scale stress test for agentic LLMs.
- **Insight**: When designing LLM benchmarks, one should include control groups with "rules maintained but labels changed" and "labels maintained but payoffs changed"; otherwise, "familiar template execution" is easily misjudged as "abstract reasoning."

## Rating
- Novelty: ⭐⭐⭐⭐ Uses counterfactual repeated games to diagnose LLM strategic robustness; the task setting is compact and explanatory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple frontier LLMs, prompting strategies, opponent types, and metrics, though the variety of games is still limited.
- Writing Quality: ⭐⭐⭐⭐ The main text logic is clear, and numerical data in the appendix is sufficient; some tables are very large, requiring summaries for better comprehension.
- Value: ⭐⭐⭐⭐ Directly valuable for research in LLM agent evaluation, strategic reasoning, and counterfactual robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](../../ICML2026/causal_inference/evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
