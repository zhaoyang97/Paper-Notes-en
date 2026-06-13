---
title: >-
  [Paper Note] Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers
description: >-
  [ACL2026][Dialogue Systems][Emotional Support Models] This paper constructs nine-dimensional help-seeker profiles using Reddit emotional support dialogues and trains a controllable seeker simulator using LoRA-MoE with be…
tags:
  - "ACL2026"
  - "Dialogue Systems"
  - "Emotional Support Models"
  - "Seeker Simulation"
  - "Controllable Evaluation"
  - "MoE Routing"
  - "Stress Testing"
date: 2026-05-08
content_hash: 2d4a55400623f13d
---

# Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers

**Conference**: ACL2026 Findings  
**arXiv**: [2601.07698](https://arxiv.org/abs/2601.07698)  
**Code**: None  
**Area**: Emotional Support Conversation / Dialogue Evaluation / User Simulation  
**Keywords**: Emotional Support Models, Seeker Simulation, Controllable Evaluation, MoE Routing, Stress Testing  

## TL;DR
This paper constructs nine-dimensional help-seeker profiles using Reddit emotional support dialogues and trains a controllable seeker simulator using LoRA-MoE with behavior routing. This allows emotional support models to undergo interactive stress testing on more realistic, difficult, and diverse populations.

## Background & Motivation
**Background**: Emotional support dialogue models have evolved from single-turn empathetic responses to multi-turn interactive systems. Evaluation increasingly relies on seeker simulators: letting a simulated seeker converse with a supporter model, then scoring based on empathy, comfort, suggestions, and coherence.

**Limitations of Prior Work**: Mainstream simulators often generate "cooperative" seekers who are compliant, open, and articulate—resembling idealized test users. Such evaluations overestimate the actual capability of supporters, as real seekers may be silent, resistant, off-topic, emotionally intense, or provide insufficient disclosure, even rejecting the supporter's advice.

**Key Challenge**: Emotional support systems most need to maintain performance with difficult users, yet existing evaluations lack "population variance" as a control variable. With only an average seeker, it is difficult to determine whether a model lacks empathy overall or specifically fails with groups exhibiting high resistance, low disclosure, or low engagement.

**Goal**: The objective is not to train another supporter who is better at comforting, but to create a more credible stress-testing environment. Specifically, it needs to construct multiple seeker profiles, maintain profile-consistent behavior stably, generate realistic multi-turn dialogues, and expose performance variances of the same supporter under different seeker populations.

**Key Insight**: The paper collects real interactions from Reddit online support groups, decomposes seeker behavior into psychological and linguistic features, and uses these structured features as generation control signals. This is more reliable than simple persona prompts because it explicitly exposes psychological variables like "resistance level," "self-disclosure depth," and "engagement" to the model.

**Core Idea**: Control a LoRA-MoE seeker simulator using nine-dimensional seeker profiles, allowing expert routing to learn different help-seeking behavior subspaces, and use it for interactive evaluation of diverse seekers and supporters.

## Method
The methodology involves two layers: first, defining and labeling diverse seekers, and second, enabling the model to consistently portray these seekers throughout multi-turn dialogues.

The key is not stuffing more character descriptions into prompts, but creating a learnable control interface for seeker behavioral profiles.

### Overall Architecture
The input consists of real Reddit support dialogues, where each dialogue is converted into a seeker profile.

The profile includes nine categories: the psychological side covers coping strategy, engagement level, resistance level, utterance style, self-disclosure level, and seeker reaction distribution; the linguistic side includes verbosity level, profanity flag, and total dialogue turns level.

Additionally, each profile contains the "seeker main problem," summarized from the original Reddit post.

The training phase first uses Llama-3-8B-Instruct for standard SFT to learn "generating the next seeker utterance given the profile and history."

Next, the SFT backbone is frozen, multiple LoRA experts are attached to the linear layers of the attention and FFN modules, and a shared routing network outputs dialogue-level routing weights based on a seeker feature vector.

In the inference and evaluation phase, given a specific seeker profile, the simulator interacts with the supporter model for up to 20 turns; the generated dialogues are then evaluated for the supporter's emotional support skills, general dialogue skills, and overall quality using automated metrics.

### Key Designs
1.  **Nine-Dimensional Seeker Profile**:
    - **Function**: Decomposes real help-seeking behavior into combinable, controllable, and verifiable feature vectors.
    - **Mechanism**: The authors extract psychological and linguistic variables from 11,066 Reddit emotional support dialogues; psychological features are labeled using an LLM tagger, while linguistic features are extracted via rules (e.g., profanity-check for profanity, token count discretization for verbosity, and dialogue turn count for turn level).
    - **Design Motivation**: Traditional personas often only describe identity or situation, failing to control "whether to resist suggestions," "willingness to disclose," or "verbosity." The 9D profile directly targets interaction dimensions where support models are most likely to fail, allowing subsequent evaluation to be sliced by population segments.

2.  **Behavior-Routed LoRA-MoE Simulator**:
    - **Function**: Learns multiple behavioral subspaces within the same seeker model and dynamically combines experts based on the profile.
    - **Mechanism**: After training and freezing the SFT backbone, multiple low-rank experts are added to each linear layer; the routing network maps the structured seeker feature vector to a shared dialogue-level routing vector $\alpha$. The linear layer output can be expressed as the original transformation plus $\sum_i \alpha_i \Delta_i(x)$. Expert semantics are not manually specified but naturally differentiate through joint optimization of language modeling loss and routing.
    - **Design Motivation**: Long dialogue histories and system prompts dilute profile signals, and pure SFT easily reverts to "polite, cooperative, and averaged" linguistic patterns. MoE decouples profile control from text prompts, using parameter subspaces to accommodate different help-seeking styles.

3.  **Closed-Loop from Simulator Validation to Supporter Stress Testing**:
    - **Function**: First validates if the seeker matches the target profile, then uses it to evaluate supporter performance across different populations.
    - **Mechanism**: The authors check the simulator for profile adherence, expert fidelity, and diversity, then generate seeker-supporter dialogues using 300 held-out profiles, evaluating supporters with ten emotional support and general dialogue metrics. The simulator also learns an `<|end_of_dialogue|>` token to ensure dialogue length is controlled by the total turns level.
    - **Design Motivation**: If a simulator only "appears diverse" but fails to follow the profile, it passes evaluation noise to the supporter ranking. Validating the simulator itself before stress-testing the model makes the logic more robust.

### Loss & Training
The first stage is standard next-token prediction, calculating language modeling loss only on the next seeker utterance.

The training configuration uses Llama-3-8B-Instruct with LoRA, a rank of 16, and target layers covering all linear layers.

The second stage freezes the SFT backbone and only trains the routing network and LoRA experts.

The total objective is the language modeling loss combined with training constraints related to behavior differentiation; for the contrastive baseline, a disentanglement loss for pseudo-feature flipping was also designed.

The emphasis of the MoE model is not on manual expert assignment but on letting the routing automatically select behavioral subspaces based on the profile.

This training strategy allows the model to retain the natural language distribution of Reddit dialogues while strengthening controllable generation across dimensions like high resistance, low disclosure, and low engagement.

## Key Experimental Results

### Main Results
Simulator profile adherence is measured via Macro F1, comparing zero-shot models, SFT, contrastive learning, and the proposed MoE.

| Simulator | Mean Macro F1↑ | Std Dev↓ | Min↑ | Max↑ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4.1-mini | 0.301 | 0.131 | 0.160 | 0.580 |
| Llama-3-8B-Instruct | 0.259 | 0.148 | 0.110 | 0.580 |
| Qwen-2.5-14B-Instruct | 0.284 | 0.095 | 0.150 | 0.470 |
| GPT-5 | 0.319 | 0.216 | 0.150 | 0.840 |
| DeepSeek-V3.2 | 0.431 | 0.218 | 0.180 | 0.910 |
| SFT | 0.515 | 0.160 | 0.360 | 0.760 |
| Contrastive Learning | 0.484 | 0.178 | 0.340 | 0.850 |
| Ours | 0.549 | 0.125 | 0.430 | 0.740 |

These results indicate that stronger general LLMs alone cannot stably adhere to seeker profiles; SFT provides significant improvement, but MoE performs best in mean and minimum values, showing it is more robust for difficult profiles.

Expert evaluation further compares the proposed simulator with existing seeker simulators in terms of language naturalness, character realism, and psychological plausibility.

| Comparison | Language Naturalness Win/Loss/Tie | Character Realism Win/Loss/Tie | Psychological Plausibility Win/Loss/Tie |
| :--- | :--- | :--- | :--- |
| Ours vs. Eeyore | 62 / 19 / 9 | 60 / 20 / 10 | 64 / 13 / 13 |
| Ours vs. ESC-Judge | 62 / 18 / 10 | 65 / 19 / 6 | 56 / 14 / 20 |
| Ours vs. ESC-Role | 72 / 9 / 9 | 61 / 16 / 13 | 61 / 12 / 17 |

In supporter evaluations, the proposed simulator yields lower and more discriminative scores, indicating it is not a "harsh scorer" but rather generates test samples closer to real difficult interactions.

| Supporter | Seeker Simulator | Identification | Comforting | Suggestions | Informativeness | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5-mini | ESC-Judge | 4.980 | 4.977 | 4.070 | 3.853 | 5.000 |
| GPT-5-mini | Ours | 4.410 | 4.477 | 2.820 | 2.393 | 4.853 |
| Llama-ESConv | ESC-Judge | 4.000 | 4.267 | 3.203 | 2.717 | 4.827 |
| Llama-ESConv | Ours | 3.390 | 3.150 | 2.303 | 1.807 | 3.887 |

### Ablation Study
Rather than simple on/off module ablation, the paper uses a baseline family and routing analysis to explain the gains from MoE.

| Configuration / Analysis | Key Metrics | Description |
| :--- | :--- | :--- |
| SFT | Macro F1 0.515 | Learns Reddit style, but lacks controllability for fine-grained profiles. |
| Contrastive Learning | Macro F1 0.484 | Enhances discrimination via pseudo-feature perturbation, but is less stable than explicit behavior routing. |
| Ours MoE | Macro F1 0.549 | Highest profile adherence after routing to different low-rank experts. |
| Routing analysis | resistance 0.37→0.43, self-disclosure 0.40→0.45 | MoE provides targeted improvements in psychological dimensions that are hard to control via SFT. |
| Parameter Overhead | ~15,881 routing parameters | Control capability stems from the expert routing structure rather than a significant increase in model scale. |

### Key Findings
- The advantage of MoE is primarily reflected in "hard-to-control behaviors," specifically resistance level and self-disclosure depth—which are precisely the user dimensions emotional support systems most need to be evaluated on.
- When strong general models role-play seekers zero-shot, their maximum performance can be high, but mean and stability are inferior to trained simulators, suggesting profile adherence cannot be solved by model scale alone.
- The proposed seeker causes an overall drop in supporter scores, especially in metrics like suggestions, informativeness, and experience sharing, indicating that traditional simulators may allow models to achieve inflated scores on overly cooperative users.
- Expert routing shows interpretable differentiation (e.g., collaborative/open, pragmatic/general, reclusive patterns), suggesting MoE does not just increase diversity in a black-box manner but forms relatively clear regions within the behavior space.

## Highlights & Insights
- Transforming "seeker diversity" from an abstract slogan into a nine-dimensional profile is the most practical contribution. It allows emotional support evaluation to be sliced by population segments—much like fairness evaluation—rather than just looking at average scores.
- Using structured feature vectors to control experts in MoE, rather than stuffing all controls into the system prompt, is well-suited for long dialogues. As dialogues progress, text prompts are easily buried by history, whereas parameter routing remains stable.
- The decision to place simulator validation before supporter evaluation is crucial. Many agent evaluations focus only on final model rankings, but here, the proof that the simulated users are controllable, natural, and diverse makes the evaluation signal more credible.
- This framework is transferable to scenarios like medical consultation, educational tutoring, and customer complaints. Provided a target user profile can be defined, a "difficult user simulator" can be trained for pre-deployment stress testing.

## Limitations & Future Work
- Data is primarily from Reddit online support groups, which may differ from formal psychological counseling, hotline crisis intervention, or cross-cultural help-seeking; the profile distribution should not be taken as representative of all real seekers.
- Automated evaluation relies on LLM judges like GPT-4o-mini. Although human correlation checks were performed, automated scores in psychological support scenarios cannot replace clinical experts or long-term therapeutic outcome metrics.
- MoE requires open-weight models and training adapters, which is less direct for teams using closed-source models via API.
- While detailed, the 9D profile may still miss important variables affecting support efficacy, such as age, cultural background, trauma history, or stage of help-seeking.
- Future work could turn population configuration into an interactive tool, allowing developers to select target populations and automatically generate test sets and failure reports.

## Related Work & Insights
- **vs. ESC-Eval / ESC-Judge**: These works advanced simulator-based emotional support evaluation but leaned toward fixed personas or prompt-based roleplay. This paper emphasizes fine-grained profile control and real behavioral diversity, making it more suitable for stress testing.
- **vs. ESC-Role / Eeyore**: These simulators have attempted to train seeker models but often remain at the surface persona level or specific psychological states. This paper uses 9D profiles and MoE routing to enhance persistent control, providing a more systematic evaluation.
- **vs. Standard SFT Seeker Simulators**: SFT can learn tone and task format but does not naturally maintain attributes like resistance or low engagement. The core insight here is that long-dialogue personality control requires structured routing rather than relying solely on textual conditions.
- **Insights for Dialogue System Evaluation**: A model performing well on "cooperative users" does not necessarily serve real users effectively. Future dialogue evaluation should report performance by population segments and worst-case slices rather than just overall means.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using MoE routing for profile control is not a brand-new architecture, but its application for emotional support stress testing is highly targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Simulator validation, expert evaluation, and supporter stress testing are comprehensive, though external clinical validation is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with a good motivation and experimental loop; some tables are large, and explanations for automated metrics could be more concise.
- Value: ⭐⭐⭐⭐⭐ Extremely useful for pre-deployment evaluation of emotional support models and provides a reusable paradigm for "how to validate user simulators."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] LOCKET: Robust Feature-Locking Technique for Language Models](locket_robust_feature-locking_technique_for_language_models.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[NeurIPS 2025\] Less is More: Local Intrinsic Dimensions of Contextual Language Models](../../NeurIPS2025/dialogue/less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)
- [\[ACL 2026\] Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue](surprisal_minimisation_over_goal-directed_alternatives_predicts_production_choic.md)

</div>

<!-- RELATED:END -->
