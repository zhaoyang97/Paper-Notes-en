---
title: >-
  [Paper Note] Many LLMs Are More Utilitarian Than One
description: >-
  [NeurIPS 2025][Reasoning][Multi-Agent Systems] A controlled study across six LLMs identifies a "Utilitarian Boost" phenomenon: LLMs engaged in dyadic or triadic moral deliberation are more likely than their solo counterparts to endorse harming a minority for the benefit of the majority. This effect is especially pronounced in personal dilemmas involving direct harm ($\beta=0.31, p<.0001$), and the underlying mechanisms differ across models—some exhibit reduced norm sensitivit…
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "Multi-Agent Systems"
  - "Moral Reasoning"
  - "Utilitarian Boost"
  - "Group Deliberation"
  - "AI Alignment"
  - "Deontology"
date: 2026-05-08
content_hash: d952ff8477c182c4
---

# Many LLMs Are More Utilitarian Than One

**Conference**: NeurIPS 2025
**arXiv**: [2507.00814](https://arxiv.org/abs/2507.00814)  
**Code**: [GitHub](https://github.com/baltaci-r/MoralAgents)  
**Area**: LLM Reasoning / AI Alignment
**Keywords**: Multi-Agent Systems, Moral Reasoning, Utilitarian Boost, Group Deliberation, AI Alignment, Deontology

## TL;DR

A controlled study across six LLMs identifies a "Utilitarian Boost" phenomenon: LLMs engaged in dyadic or triadic moral deliberation are more likely than their solo counterparts to endorse harming a minority for the benefit of the majority. This effect is especially pronounced in personal dilemmas involving direct harm ($\beta=0.31, p<.0001$), and the underlying mechanisms differ across models—some exhibit reduced norm sensitivity, others heightened impartiality.

## Background & Motivation

**Background**: LLM-based multi-agent systems (MAS) have been deployed in high-stakes domains such as healthcare and law for collaborative decision-making. Prior work has documented social-psychological phenomena in LLM groups, including conformity and belief congruence. Yet group moral reasoning—arguably the most consequential decision type for deployment—remains almost entirely unstudied.

**Limitations of Prior Work**: (1) LLM moral reasoning research is almost exclusively single-agent, offering no insight into emergent multi-agent behavior. (2) Human psychology has established that group discussion produces a "Utilitarian Boost"—groups are more willing than individuals to sacrifice a minority for the "greater good." If LLM MAS exhibit the same effect, it poses a serious threat in high-stakes deployments. (3) Single-agent alignment checks cannot capture morally-shifted behavior that emerges at the group level.

**Key Challenge**: An individual LLM may pass safety evaluations in isolation, yet contribute to more dangerous moral judgments when embedded in a group—an alignment blind spot that has been entirely overlooked.

**Goal**: Do LLMs also exhibit a Utilitarian Boost during group deliberation? What mechanisms drive this boost? Can it be mitigated?

**Key Insight**: The paper directly adopts well-validated experimental paradigms from psychology—Greene's moral dilemma battery, the Oxford Utilitarianism Scale (OUS), and the CNI model—to study LLM groups.

**Core Idea**: LLM multi-agent systems systematically shift toward utilitarianism following moral deliberation—safety at the individual model level does not guarantee safety at the group level.

## Method

### Overall Architecture

Two conditions are compared: *Solo* (a single LLM independently evaluates moral dilemmas) vs. *Group* (two or three instances of the same LLM conduct six rounds of discussion before each agent provides a private reflective rating). Moral reasoning is quantified using the classical dilemma battery (personal/impersonal) and established psychological instruments (OUS, CNI model). Experiments are conducted across six LLMs: Llama3.3-70B, GPT-4.1, Gemma3-27B, Qwen3-32B, Qwen2.5-32B, and QwQ.

### Key Designs

1. **Greene Moral Dilemma Experiments**:
    - *Function*: Measure changes in LLMs' acceptance of moral violations between Solo and Group conditions.
    - *Mechanism*: Greene et al.'s dilemma battery is used, distinguishing *personal dilemmas* (direct harm, e.g., pushing someone off a bridge to save five) from *impersonal dilemmas* (indirect harm, e.g., pulling a lever). Each dilemma is rated on a 1–7 scale (7 = most utilitarian). In the Group condition, three agents deliberate for six rounds before independently providing private reflective ratings. Differences between Group and Solo are analyzed using ordinal mixed-effects regression.
    - *Design Motivation*: This dilemma battery has been validated over two decades of human moral psychology research; direct adoption ensures comparability.

2. **CNI Model Analysis of Utilitarian Mechanisms**:
    - *Function*: Decompose the source of the Utilitarian Boost—whether it stems from greater sensitivity to consequences (C), reduced sensitivity to norms (N), or a preference for action (I).
    - *Mechanism*: The CNI model estimates three latent variables—C (consequence sensitivity), N (norm sensitivity), and I (inaction preference)—from response patterns across four orthogonal conditions (action-consistent, action-inconsistent, omission-consistent, omission-inconsistent). In humans, the group Utilitarian Boost is driven solely by increased C. The paper investigates whether the same holds for LLM groups.
    - *Design Motivation*: Moving beyond surface-level performance metrics to diagnose the specific cognitive mechanism underlying the Utilitarian Boost, since different mechanisms call for different mitigation strategies.

3. **Mitigation Strategy Exploration**:
    - *Function*: Test the effects of model diversity, self-reflection, and pre-assigned moral frameworks on the Utilitarian Boost.
    - *Mechanism*: (1) **Model heterogeneity**—pairing models from different families or of different sizes: homogeneous pairs (e.g., GPT-4.1 × GPT-4.1) amplify utilitarianism, cross-family heterogeneous pairs attenuate the boost ($\beta=-0.30, p=.0001$), and mixed-size pairs even reverse it toward deontology ($\beta=1.40, p<.001$). (2) **Self-reflection**—replacing multi-agent discussion with iterative single-model self-debate eliminates the Utilitarian Boost. (3) **Moral priming**—deontology–deontology (DD) pairs elevate utilitarianism, while UD/DU mixed pairs produce a "Deontological Boost" ($-0.323, p<.0001$).
    - *Design Motivation*: Providing practitioners with actionable design levers for controlling the Utilitarian Boost.

## Key Experimental Results

### Main Results

| Model | Group–Solo Difference | SE | z | p |
|---|---|---|---|---|
| Gemma3 (strongest) | +1.65 | 0.16 | 10.33 | <.0001 |
| Qwen3 | +1.23 | 0.155 | 7.90 | <.0001 |
| Llama3.3 | +0.80 | 0.158 | 5.07 | <.0001 |
| Qwen2.5 | +0.68 | 0.124 | 5.47 | <.0001 |
| QwQ | +0.69 | 0.125 | 5.54 | <.0001 |
| GPT-4.1 | +0.57 | 0.17 | 3.35 | .0023 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Personal dilemmas | +0.6352 (p<.001) | Significant Utilitarian Boost in direct-harm scenarios |
| Impersonal dilemmas | −0.0227 (p=.975) | No significant effect in indirect-harm scenarios |
| Homogeneous model pairs | +0.29 (p=.0001) | Same-model pairing amplifies utilitarianism |
| Cross-family heterogeneous pairs | −0.30 (p=.0001) | Cross-family pairing attenuates the boost |
| Mixed-size models | +1.40 reversed | Models of different sizes produce a Deontological Boost |
| Self-reflection (no group) | No significant boost | Group dynamics, not iteration per se, drive the boost |

### Key Findings
- The Utilitarian Boost is significant only in personal dilemmas (direct harm)—opposite to the human pattern, where the boost also appears in impersonal dilemmas.
- CNI profiles differ markedly across models: Gemma3 acts as a "norm-optimizing utilitarian," GPT-4.1 as an "impartial utilitarian," and Qwen3 as an "action-oriented utilitarian."
- Model diversity is the most effective mitigation tool—mixing models from different families or of different sizes can eliminate or even reverse the effect.
- Sentiment analysis reveals an elevated proportion of "fear" labels in the Group condition, which correlates positively with the Utilitarian Boost.

## Highlights & Insights
- This is the first systematic demonstration of emergent moral drift in LLM multi-agent systems—the finding that "groups are more utilitarian than individuals" carries profound implications for AI alignment.
- The divergent CNI profiles across models reveal that the Utilitarian Boost is not a monolithic phenomenon, necessitating model-specific mitigation strategies.
- The discovery that model heterogeneity serves as a mitigation tool has immediate practical value—simply mixing different models may suffice.
- The experimental design is rigorous: validated psychological instruments, human evaluation to verify rating consistency, and three repetitions per condition.

## Limitations & Future Work
- Only dyads and triads are tested; larger groups, asynchronous deliberation, and other configurations remain unexplored.
- Experiments are conducted primarily in English; the influence of moral norms and cultural variation is not considered.
- Mitigation experiments are exploratory and require more systematic validation.
- Group architectures incorporating a meta-controller (moderator) are not tested.
- The association between sentiment labels and utilitarianism is correlational rather than causal.

## Related Work & Insights
- **vs. Human group moral research**: The human Utilitarian Boost is driven by increased consequence sensitivity (C); LLM mechanisms vary across models and are more complex.
- **vs. LLM conformity research (Weng et al.)**: Conformity is a general phenomenon, whereas the moral drift identified here is more specific and more dangerous.
- **vs. Single-agent moral evaluation**: Single-agent alignment checks cannot capture group-emergent moral drift—group-level alignment evaluation is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First demonstration of the Utilitarian Boost in LLM multi-agent systems; an important discovery of an alignment blind spot.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six models × multiple dilemma batteries × CNI analysis × mitigation experiments × human verification—exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; psychological instruments applied rigorously.
- Value: ⭐⭐⭐⭐⭐ An important warning for multi-agent AI alignment, with actionable mitigation strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](../../ACL2025/llm_reasoning/self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)
- [\[NeurIPS 2025\] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[ICML 2025\] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](../../ICML2025/llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)
- [\[ICLR 2026\] When More Is Less: Understanding Chain-of-Thought Length in LLMs](../../ICLR2026/llm_reasoning/when_more_is_less_understanding_chain-of-thought_length_in_llms.md)

</div>

<!-- RELATED:END -->
