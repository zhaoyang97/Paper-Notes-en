---
title: >-
  [Paper Note] Evaluating Counterfactual Strategic Reasoning in Large Language Models
description: >-
  [ACL2026][Causal Inference][counterfactual games] This paper evaluates the strategic adaptation capabilities of LLMs using label perturbations, payoff perturbations…
tags:
  - "ACL2026"
  - "Causal Inference"
  - "counterfactual games"
  - "strategic reasoning"
  - "Prisoner's Dilemma"
  - "Rock-Paper-Scissors"
  - "opponent comprehension"
date: 2026-05-08
content_hash: 7ab4c92ff6119892
---

# Evaluating Counterfactual Strategic Reasoning in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2603.19167](https://arxiv.org/abs/2603.19167)  
**Code**: https://github.com/dimjimitris/llm_gm_thesis  
**Area**: LLM Reasoning / Game Evaluation / Counterfactual Robustness  
**Keywords**: counterfactual games, strategic reasoning, Prisoner's Dilemma, Rock-Paper-Scissors, opponent comprehension

## TL;DR
This paper evaluates the strategic adaptation capabilities of LLMs using label perturbations, payoff perturbations, and joint counterfactual versions of the Iterated Prisoner's Dilemma and Rock-Paper-Scissors. It finds that many models appear proficient in familiar games but continue to utilize templated strategies even after payoff structures are altered.

## Background & Motivation
**Background**: LLMs have been widely applied in multi-agent collaboration, competition, and game simulations. Researchers often observe whether models can cooperate, compete, identify opponent strategies, and approach equilibrium behavior through structured games such as the Prisoner's Dilemma, Rock-Paper-Scissors, and Matching Pennies.

**Limitations of Prior Work**: Conventional game evaluations tend to overestimate model capabilities. Models might memorize templates such as "cooperate/defect in the Prisoner's Dilemma" or "randomize in Rock-Paper-Scissors" rather than re-calculating strategies based on the payoff matrix. Once action labels are renamed or payoff structures are modified counterfactually, fluent explanations do not necessarily translate into correct actions.

**Key Challenge**: True strategic reasoning requires models to perform conditional updates based on current environmental labels, payoffs, and historical interactions; however, LLM behavior may stem more from canonical game patterns encountered during pre-training. Distinguishing between surface recognition and incentive sensitivity is difficult in default games, requiring counterfactual interventions to decouple them.

**Goal**: To construct a compact, controllable, and reproducible experimental framework to separately diagnose label robustness, payoff sensitivity, opponent modeling, and token-normalized efficiency, thereby determining whether models understand the current game or are merely reproducing familiar templates.

**Key Insight**: The authors select two complementary games: the Prisoner's Dilemma to examine dynamic adaptation between cooperation and defection, and Rock-Paper-Scissors to examine randomization, pattern exploitation, and three-action equilibrium. Subsequently, label perturbations, payoff perturbations, and joint perturbations are applied to both, forcing models to re-interpret action meanings and payoff structures.

**Core Idea**: Use counterfactual label/payoff interventions in repeated games to distinguish between an LLM's ability to "describe a strategy" and its ability to "execute a strategy according to new incentives."

## Method
The method in this paper is essentially a behavioral evaluation framework. It does not train models; instead, it places LLMs in multi-round interactive games against another LLM or an algorithmic opponent, recording actions, payoffs, opponent comprehension speed, cooperation rates, and token efficiency.

### Overall Architecture
The input consists of a set of LLMs, prompting strategies, opponent types, and game definitions. Each experiment uses one LLM as a player against a same-model instance or an algorithmic player. The Prisoner's Dilemma is repeated for 16 rounds, and Rock-Paper-Scissors for 24 rounds; each non-self-consistency player is repeated 5 times, while self-consistency players are repeated twice.

The framework includes four types of game settings: default games, label-based counterfactuals, payoff-based counterfactuals, and joint counterfactuals. Label-based settings only change action names or dominance relationship descriptions while keeping payoffs constant; payoff-based settings change the payoff matrix so that the original equilibrium no longer applies; joint settings simultaneously change labels and payoffs to form the ultimate stress test.

Model variants include zero-shot, Chain-of-Thought, Solo Performance Prompt, and self-consistency prompting. Opponent types include both LLMs and algorithmic strategies such as SREP, PP, MF/TFT, and AP. This allows for simultaneous investigation of LLM-LLM coordination, predictable opponent exploitation, and adaptive opponent confrontation.

### Key Designs
1.  **Counterfactual Game Construction**:
    - **Function**: Decomposes default games into surface label changes and deep payoff changes as pressure sources.
    - **Mechanism**: In PD, Stag Hunt is used as a payoff-based counterfactual, changing the strict defection advantage into a coordination problem; simultaneously, C/D labels are renamed to Stag/Hare to test influence from action names. In RPS, payoffs for certain win/loss combinations are amplified, making the uniform mixed strategy no longer an equilibrium.
    - **Design Motivation**: It is difficult to know if a model truly reads payoffs in default PD/RPS. Counterfactual versions expose "label anchoring" and "incentive rigidity."

2.  **Opponent Comprehension Metric**:
    - **Function**: Measures the round at which a model begins to stably understand and exploit the opponent's strategy.
    - **Mechanism**: $m$ is defined as the earliest round such that from that round until the end of the game, the LLM obtains a payoff no lower than the opponent's in at least $t_p=90\%$ of subsequent rounds. A smaller $m$ indicates earlier opponent modeling; values exceeding the game length indicate a lack of stable understanding.
    - **Design Motivation**: Total scores only reflect cumulative earnings, not whether the model understood the opponent early or recovered by chance later. $m$ isolates the speed of dynamic adaptation.

3.  **Performance and Efficiency Joint Evaluation**:
    - **Function**: Distinguishes between "better play" and "better token consumption for explanation."
    - **Mechanism**: In addition to total points, cooperation/action distribution, and failure rates, the authors define efficiency as $\textit{points}/\textit{tokens} \times c$ (default $c=1000$).
    - **Design Motivation**: Reasoning models may output more chain-of-thought, but extra deliberation does not necessarily result in faster adaptation. Efficiency metrics identify reasoning-overhead mismatches.

### Loss & Training
This paper does not involve training models; the focus is on evaluation design. The default payoff for the Prisoner's Dilemma is $(C,C)=(4,4)$, $(C,D)=(1,6)$, $(D,C)=(6,1)$, and $(D,D)=(2,2)$, with 16-round cumulative scores ranging from 16 to 96. In RPS, win/loss/tie payoffs per round are $1/-1/0$, with 24-round cumulative scores ranging from -24 to 24. In the RPS payoff-based counterfactual, the magnitude of win/loss for the Rock-Paper combination is amplified to 3, shifting the theoretical equilibrium from a uniform distribution to $\pi^*(R)=0.2$, $\pi^*(P)=0.2$, $\pi^*(S)=0.6$.

## Key Experimental Results

### Main Results
| Setting | Metric | Representative Results | Explanation | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Default PD vs. SREP | Total points | Baseline for continuous $(D,D)$ is 32 points when SREP always defects; most LLMs cluster around 30 points | Models typically identify continuous defection and provide near-optimal responses | Simple algorithmic opponents are easier |
| Default PD vs. LLM | Total points | Claude 3.5/3.7 and Llama 3.3-70B reach 64 points under various prompting | 64 points corresponds to 16 rounds of complete mutual cooperation | Some models achieve stable coordination in LLM-LLM setups |
| Default PD | Instability cases | Mistral Large ranges from 18.6±10.6 to 29.8±2.2 under SREP; Claude 4/DeepSeek R1 range from 31.4±0.0 to 49.4±15.5 in LLM-LLM | Weaker or over-reasoning models may be more unstable | High capability does not guarantee strategic stability |
| Default RPS | Opponent comprehension | Claude 3.5 Sonnet v2 is 10.6±13.1 for ZS, 21.4±4.6 for SPP, and 19.6±5.6 for CoT | Opponent modeling in RPS is slower and closer to the 24-round horizon | Three-action games with no dominant strategy are harder |
| RPS payoff counterfactual | Theoretical equilibrium | Shifted from uniform $(1/3, 1/3, 1/3)$ to $(0.2, 0.2, 0.6)$ | Remaining close to uniform indicates failure to re-calculate strategy based on new payoffs | Payoff perturbation best exposes templating |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Label-only counterfactual | Degradation usually moderate | Strong models remain relatively stable, while models like Mistral are more prone to fluctuations due to action renaming |
| Payoff-based counterfactual | Stronger degradation | Requires re-calculating incentives, especially transitioning from uniform randomization to biased mixed strategies in RPS |
| Joint counterfactual | Maximum pressure | Near-horizon comprehension failures and high variance are more common when both labels and payoffs change |
| CoT / thinking variants | Inconsistent effects | Helpful for some strong models, but can lead to overthinking or distrust tendencies in Claude 4 or DeepSeek R1 scenarios |
| Self-consistency | Reduces variance but not fundamental bias | It often reinforces existing behavioral patterns rather than correcting an erroneous strategy |

### Key Findings
- Payoff-based counterfactuals are more diagnostic than label-only ones because they force the model to re-calculate the reward structure rather than just processing surface name changes.
- Performance in default games does not represent counterfactual robustness. Claude 3.7 is the most stable overall, Claude 4 is strong in RPS but shows mixed counterfactual stability, and Llama 3.3 is stable in PD cooperation but weaker under RPS/payoff shifts.
- Thinking more is not necessarily better. Thinking-enabled variants increase token consumption in some settings without a proportional increase in total points or opponent comprehension.
- RPS exposes delayed opponent modeling better than PD because there is no simple cooperative convergence point; models must maintain near-equilibrium behavior or identify exploitable patterns.

## Highlights & Insights
- The value of this paper lies in its "clean" evaluation design: label changes, payoff changes, and joint changes correspond to different failure modes, allowing for the decoupling of template memory, label anchoring, and incentive rigidity.
- Opponent comprehension is more explanatory than final scores. Many models might achieve acceptable final scores, but a late $m$ suggests they stumbled into the strategy through interaction rather than understanding the opponent from the start.
- The $(0.2, 0.2, 0.6)$ equilibrium in the RPS payoff counterfactual is crucial. It demonstrates that "randomization" is not always correct; continuing uniform randomization after payoff changes represents canonical equilibrium persistence.
- The paper serves as a reminder that in agent evaluation, the chain-of-thought in strong models may increase defensiveness, skepticism, or exploration noise. Longer reasoning traces do not automatically translate to steadier strategic execution.

## Limitations & Future Work
- The evaluation only covers two-player, synchronous, repeated games with fixed payoffs, which has limited ecological validity compared to real-world multi-party negotiations, markets, auctions, or open-ended collaborations.
- Algorithmic opponents and payoff structures are preset; different conclusions might emerge with more adaptive opponents, multi-agent populations, or games of incomplete information.
- All metrics are derived from observable actions and tokens; inferring "understanding" is behavioral and does not equate to explaining the model's internal mechanisms.
- The set of models, prompts, and counterfactual types is not exhaustive. Future work could include more open/closed-source models, more complex payoff transformations, natural language rule ambiguity, and consistency analysis of internal reasoning traces.

## Related Work & Insights
- **vs. Conventional Game Evaluation**: Default PD/RPS only observe if a model can play familiar games; this paper tests if the model truly updates strategies based on current rules via counterfactual perturbations.
- **vs. Static Counterfactual QA**: Many counterfactual benchmarks are single-turn I/O; this paper incorporates counterfactuals into repeated interactions, allowing for the observation of adaptation speed and history dependency.
- **vs. Multi-agent Collaboration Evaluation**: Conventional agent benchmarks focus more on task success rates; this paper emphasizes multi-dimensional diagnosis of payoffs, opponent modeling, efficiency, and failure rates, making it suitable as a compact stress test for agentic LLMs.
- **Insight**: When designing LLM benchmarks, one should include control groups with "rule-preserved but label-changed" and "label-preserved but payoff-changed" settings to avoid misinterpreting "familiar template execution" as "abstract reasoning."

## Rating
- Novelty: ⭐⭐⭐⭐ Diagnosing LLM strategic robustness via counterfactual repeated games is a compact and explanatory setup.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple frontier LLMs, prompting strategies, opponent types, and metrics, though the variety of games is still relatively small.
- Writing Quality: ⭐⭐⭐⭐ The main text logic is clear, and the appendix provides sufficient numerical data; some tables are very large and require textual summaries for interpretation.
- Value: ⭐⭐⭐⭐ Directly valuable for LLM agent evaluation, strategic reasoning, and counterfactual robustness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](../../ICLR2026/causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)

</div>

<!-- RELATED:END -->
