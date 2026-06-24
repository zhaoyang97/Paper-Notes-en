---
title: >-
  [Paper Note] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming
description: >-
  [ACL 2025][LLM Alignment][LLM Safety] The proposed MTSA framework simultaneously enhances the attack capabilities of the red-team model and the safety defense performance of the target model through thought-guided multi-round red-teaming and reinforcement learning optimization with future rewards within an adversarial iterative process. It achieves state-of-the-art (SOTA) performance across multiple safety benchmarks without degrading general model capabilities.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "LLM Safety"
  - "Multi-Turn Dialogue"
  - "Red-Teaming"
  - "Safety Alignment"
  - "Reinforcement Learning"
  - "Adversarial Training"
  - "Jailbreak Attack"
date: 2026-05-08
content_hash: 5cd956f7839a8751
---

# MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming

**Conference**: ACL 2025  
**arXiv**: [2505.17147](https://arxiv.org/abs/2505.17147)  
**Code**: [GitHub](https://github.com/yuki-younai/MTSA)  
**Area**: LLM Alignment  
**Keywords**: LLM Safety, Multi-Turn Dialogue, Red-Teaming, Safety Alignment, Reinforcement Learning, Adversarial Training, Jailbreak Attack

## TL;DR

The proposed MTSA framework simultaneously enhances the attack capabilities of the red-team model and the safety defense performance of the target model through thought-guided multi-round red-teaming and reinforcement learning optimization with future rewards within an adversarial iterative process. It achieves state-of-the-art (SOTA) performance across multiple safety benchmarks without degrading general model capabilities.

## Background & Motivation

With the widespread deployment of LLMs like ChatGPT, jailbreak attacks have become a severe security threat. Attackers bypass safety safeguards of models through carefully crafted inputs to induce them to generate harmful content.

Prior work faces three core challenges:

**Safety vulnerability in multi-turn dialogues**: Current mainstream jailbreak techniques primarily target single-turn interactions. However, research reveals that LLMs are more susceptible to compromises in multi-turn dialogues. In multi-turn settings, malicious intent can be scattered and hidden across multiple interaction turns, guiding the model to generate harmful content step-by-step—which is more covert and harder to defend than single-turn attacks.

**Difficulty in obtaining safety alignment data**: The diversity of multi-turn jailbreak attacks makes collecting sufficient safety alignment data via human efforts extremely costly. Existing automated red-teaming methods lack interactivity and strategic planning, failing to adapt to complex dialogue environments.

**Lack of multi-turn safety alignment algorithms**: Current safety alignment algorithms are primarily oriented towards single-turn scenarios. In a multi-turn setup, toxicity within a dialogue is cumulative—optimizing only the final turn introduces covariate shift between training and test distributions, significantly reducing the generalization of safety alignment.

The core innovations of MTSA lie in: (1) enhancing the diversity and effectiveness of the red-team model's attacks using a "think-before-attack" strategy; (2) replacing conventional methods that only optimize the last turn with a multi-turn RL algorithm based on future rewards, thereby bolstering the robustness of safety alignment.

## Method

### Overall Architecture

MTSA consists of two phases:

- **Phase 1—Thought-Guided Attack Learning**: Constructs a Think-before-attack dataset to train the initial version of the red-team model through selective fine-tuning.
- **Phase 2—Adversarial Iterative Optimization**: The red-team model and the target model interact to generate dialogue data. Preference data is constructed via trajectory sampling to optimize both models. Through multiple iterations, the red-team model enhances its attack strategies, while the target model progressively strengthens its defense.

### Key Designs

#### 1. Think-before-Attack

Multi-turn red-teaming attack strategies are categorized into four types:
- **Purpose Inversion**: Replaces the query intent with its opposite direction to alleviate the immediate sense of harmfulness.
- **Query Decompose**: Deconstructs a complex attack target into multiple less harmful sub-questions.
- **Role Play**: Initiates attacks by simulating different personas or scenarios.
- **Mixed Mode**: Flexibly combines the above strategies.

Key Innovation: Before executing an attack in each turn, the red-team model first **observes the current dialogue context and generates thoughts**, choosing an attack strategy based on these thoughts. This enables the red-team model to dynamically adapt its attack style according to the current state of the target model rather than mechanically executing pre-defined templates.

During initialization, the top-$k$ lowest-similarity data are selected from synthetic data for supervised fine-tuning (SFT), ensuring attack diversity in subsequent adversarial iterations.

#### 2. Multi-Turn RL with Future Rewards

Conventional multi-turn tasks are trained only on the last dialogue turn, which introduces a covariate shift between the training and test distributions. The core algorithmic innovation of MTSA is leveraging **rewards of future states** for dynamic preference optimization in every turn.

**Target Model Optimization Loss**—using relative preference reward:

$$\mathcal{L}_{tgt} = \left(\frac{1}{\eta}\left(\log\frac{\pi_{t+1}^{tgt}(r_h|s_h^{tgt})}{\pi_t^{tgt}(r_h|s_h^{tgt})} - \log\frac{\pi_{t+1}^{tgt}(r_h'|s_h^{tgt})}{\pi_t^{tgt}(r_h'|s_h^{tgt})}\right) - (R_{tgt}(s_{H+1}^{tgt}) - R_{tgt}(s_{H+1}'^{tgt}))\right)^2$$

Core Idea: Instead of using a critic model to predict Q-values (which performs poorly in multi-turn RL), the model uses trajectory sampling to obtain the reward at the terminal state as a proxy for the Q-value. This enables effective alignment even in intermediate turns based on future outcomes.

**Red-team Model Optimization Loss**—using Direct Preference Optimization (DPO):

$$\mathcal{L}_{adv} = -\log\sigma\left(\beta\log\frac{\pi_{t+1}^{adv}(q_w|s_h^{adv})}{\pi_t^{adv}(q_w|s_h^{adv})} - \beta\log\frac{\pi_{t+1}^{adv}(q_l|s_h^{adv})}{\pi_t^{adv}(q_l|s_h^{adv})}\right)$$

where $q_w$ and $q_l$ are positive and negative samples, respectively, determined based on terminal state rewards.

#### 3. Adversarial Iterative Optimization

- **Online Sampling**: In each iteration, a subset is sampled from the target attack set, and the red-team model interacts with the target model for multiple turns (up to 5 turns).
- **Trajectory Sampling**: Performs safety rewriting on harmful responses of the target model and samples independent trajectories; applies rejection sampling and temperature adjustments to the red-team model.
- **Reward Modeling**:
    - Target model reward $R_{tgt}$: Balances toxicity $R_{tox}$ and helpfulness $R_{help}$ using the ArmoRM multi-objective reward model.
    - Red-team model reward $R_{adv}$: Combined from the unsafe probability of a safety classifier $R_{safe}$ and semantic/lexical diversity $R_{div}$.
- **Alternating Updates**: In each iteration, the red-team model is updated first, followed by the target model. A total of 3 iterations are performed.

## Key Experimental Results

### Red-Team Attack Performance—AdvBench ASR(%)

| Method | GPT-3.5 | GPT-4o | Claude3.5 | Llama2-7B | Vicuna-7B | Zephyr-beta | Average |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCG | 33.50 | 12.50 | 22.00 | 34.50 | 24.50 | 36.00 | 27.17 |
| PAP | 36.00 | 24.50 | 14.50 | 26.00 | 32.50 | 28.00 | 26.91 |
| PAIR | 57.50 | 61.00 | 51.50 | 20.50 | 39.50 | 61.00 | 48.50 |
| COA | 52.00 | 63.50 | 55.00 | 24.50 | 48.00 | 63.00 | 51.00 |
| RedQueen | 63.00 | 58.50 | 53.00 | 43.50 | 45.00 | 57.50 | 53.42 |
| **MTSA-R3** | **72.00** | **66.50** | **56.00** | **50.50** | **64.00** | **74.50** | **63.92** |

### Target Model Defense Performance—Zephyr-7B-Beta

| Method | MTSA-R3 ASR↓ | BeaverTails↓ | CoSafe↓ | MT-Bench↑ | AlpacaEval↑ | XSTest↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 74.50 | 24.50 | 41.20 | 6.76 | 78.35 | 0.283 |
| HARM-T3 | 52.50 | 17.50 | 26.75 | 6.35 | 73.92 | 0.247 |
| MART-T3 | 48.50 | 15.50 | 26.78 | 6.46 | 74.81 | 0.255 |
| **MTSA-T3** | **23.50** | **11.50** | **18.78** | **6.78** | **77.45** | **0.231** |

### Out-of-Distribution Multi-Turn Attack Robustness—Zephyr-7B-Beta

| Attack Method | Baseline ASR | MTSA-T3 ASR | Reduction Rate |
|------|:---:|:---:|:---:|
| ActorAttack | 43.50 | 12.50 | -71.26% |
| RedQueen | 57.50 | 19.50 | -66.08% |

### Key Findings

1. **Attack performance increases turn-by-turn**: The ASR of MTSA-R1 to R3 rises from 58.08% to 63.92%, demonstrating that adversarial iterative training continuously strengthens the red-team model's attack strategies.
2. **Defense performance significantly leads**: MTSA-T3 reduces the ASR under MTSA-R3 from 74.50% to 23.50% (a 67% reduction), performing far better than HARM and MART.
3. **Barely any loss in general capabilities**: MTSA-T3 matches the baseline in MT-Bench (6.78) and AlpacaEval (77.45), while HARM and MART suffer substantial degradation.
4. **Generalization from multi-turn to single-turn**: The framework also achieves the best defense performance on BeaverTails (a single-turn safety benchmark), indicating that multi-turn safety alignment generalizes successfully to single-turn scenarios.
5. **Controllable over-refusal**: XSTest evaluations show that the over-refusal rate increases by only 5.62%, preserving model utility.
6. **Robustness against out-of-distribution attacks**: The reduction rate in defense against unseen attack methods (ActorAttack, RedQueen) exceeds 66%.

## Highlights & Insights

1. **The "think-before-attack" red-teaming strategy** is a prominent highlight: allowing the red-team model to analyze the dialogue context and select strategies before launching an attack mimics the decision-making process of real attackers, bringing automated red-teaming closer to human-level red-teaming.
2. **Replacing the critic model with future rewards** is a practical innovation in multi-turn RL: utilizing trajectory sampling to fetch terminal state rewards to approximate Q-values bypasses the challenge of inaccurate predictions from a critic model in multi-turn environments.
3. **Adversarial framework optimizing both attacking and defending sides simultaneously**: The red-team model and the target model co-evolve throughout iterations, creating a "safety arms race" that is more effective at identifying and patching vulnerabilities than one-sided defenses.
4. **7B-parameter red-team model outperforming all baselines** (including attacking GPT-4o and Claude-3.5) highlights the remarkable efficacy of the thought-guided strategy.

## Limitations & Future Work

- The defense and attack effectiveness is only verified on 7B-scale models; its applicability to larger models (e.g., 70B+) remains to be investigated.
- Categorizing attack strategies into 4 types might not be exhaustive, leaving other attack modes (e.g., social engineering, multimodal attacks) unaddressed.
- The selection of 3 iterations lacks systematic analysis; it is unclear whether convergence is reached or if further iterations would be beneficial.
- The reward model relies heavily on external safety classifiers and ArmoRM, and their intrinsic biases might impact the quality of alignment.
- The persistence of safety alignment (whether it remains secure in new scenarios post-training) is not fully evaluated.

## Related Work & Insights

- **MART**: One of the earliest iterative red-blue adversarial frameworks. MTSA builds on MART by incorporating a multi-turn design and a multi-turn RL algorithm with future rewards.
- **GPO**: Formalizes defense and attack into a two-player game, but is limited to single-turn scenarios. MTSA extends it to multi-turn settings and emphasizes strategic attack learning.
- **DPO/RLHF for safety**: MTSA showcases differentiated usage scenarios of DPO (for red-teaming) and least-squares RL (for target models).
- **Inspirations for industrial safety practices**: The adversarial iterative paradigm of MTSA can serve as an automated safety testing workflow prior to LLM deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Combined innovation of thought-guided attacks, multi-turn RL with future rewards, and adversarial iteration fills the gap in multi-turn safety alignment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive assessment covering 6 LLMs $\times$ various attack methods + defense/general capabilities/over-refusal/robustness.
- **Writing Quality**: ⭐⭐⭐ Highly detailed content, though some notations and mathematical formulas are slightly complex.
- **Value**: ⭐⭐⭐⭐⭐ A practical framework directly addressing multi-turn LLM safety, with high practical utility stemming from its dual-improvement design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[ACL 2025\] Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming](constitutional_classifiers_defending_against_universal_jailbreaks_across_thousan.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)

</div>

<!-- RELATED:END -->
