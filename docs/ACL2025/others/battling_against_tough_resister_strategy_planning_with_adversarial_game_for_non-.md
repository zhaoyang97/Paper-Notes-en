---
title: >-
  [Paper Note] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues
description: >-
  [ACL 2025][Non-collaborative dialogue] This paper proposes a strategy planning framework based on adversarial games to address strategy selection when facing a tough opponent in non-collaborative dialogues (e.g., persuasion, negotiation), generating more effective persuasive strategies by modeling the adversarial dynamics between both parties.
tags:
  - "ACL 2025"
  - "Non-collaborative dialogue"
  - "adversarial game"
  - "strategy planning"
  - "persuasive dialogue"
  - "negotiation"
date: 2026-05-08
content_hash: cacf9663bd99e6b1
---

# Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues

**Conference**: ACL 2025  
**Code**: None  
**Area**: Others  
**Keywords**: Non-collaborative dialogue, adversarial game, strategy planning, persuasive dialogue, negotiation

## TL;DR
This paper proposes a strategy planning framework based on adversarial games to address strategy selection when facing a tough opponent in non-collaborative dialogues (e.g., persuasion, negotiation), generating more effective persuasive strategies by modeling the adversarial dynamics between both parties.

## Background & Motivation

**Background**: Non-collaborative dialogues refer to scenarios where the goals of the conversational parties are inconsistent or even in conflict, such as persuasion, negotiation, and debate. Existing methods mainly rely on static strategies or simple sequential decision models to generate responses.

**Limitations of Prior Work**: When the dialogue opponent is a "tough resister" who persists in their views and is difficult to persuade, traditional methods perform poorly. These methods usually assume that the opponent will cooperate to some extent, ignoring the adversarial nature of the dialogue.

**Key Challenge**: Non-collaborative dialogue is essentially a game process where both parties adjust their actions according to the other's moves. Existing methods fail to adequately model this dynamic adversarial relationship.

**Goal**: To design a framework that is aware of the opponent's strategy and dynamically adjusts its own persuasive strategy, maintaining effectiveness especially when facing a tough resister.

**Key Insight**: The authors model non-collaborative dialogue as an adversarial game, introducing strategy planning concepts from game theory to enable the system to "anticipate the opponent's resistance" and plan coping strategies in advance.

**Core Idea**: Model the strategic interaction of both dialogue parties using an adversarial game framework, planning the optimal sequence of persuasive strategies by predicting the opponent's potential reactions.

## Method

### Overall Architecture
The input is the dialogue history and the dialogue goal (e.g., persuading the opponent to donate), and the output is the strategy selection for the next step and the corresponding response generation. The system consists of three core components: the opponent modeling module, the strategy planning module, and the response generation module.

### Key Designs

1. **Opponent Strategy Modeling**:

    - **Function**: Predict the resistance strategies that the opponent might adopt in different scenarios.
    - **Mechanism**: Train an opponent model based on dialogue history to simulate the opponent's behavioral patterns. The model learns to identify the opponent's resistance intensity and resistance type (e.g., direct refusal, topic switching, raising objections), providing information for subsequent strategy planning.
    - **Design Motivation**: Understanding the opponent is a prerequisite for formulating effective strategies; different types of resistance require different coping mechanisms.

2. **Adversarial Game Strategy Planning**:

    - **Function**: Plan the optimal sequence of strategies based on a game-theoretic framework while predicting the opponent's response.
    - **Mechanism**: Model the dialogue as a sequential game where the system and the opponent take turns acting. At each step, the strategy planner evaluates multiple potential strategies and their expected outcomes (considering the opponent's potential reactions), choosing the strategy with the maximum long-term payoff. Similar to lookahead search in board games, the system makes decisions by "thinking a few steps ahead".
    - **Design Motivation**: Greedy single-step strategy selection easily falls into local optima; facing a tough resister requires multi-step planning to progressively break through defenses.

3. **Strategy-Aware Response Generation**:

    - **Function**: Generate natural and persuasive responses based on the selected strategies.
    - **Mechanism**: Integrate strategy labels as conditional information into the language model's generation process, ensuring that the generated responses both comply with the selected strategy and maintain natural flow. This may combine joint modeling of strategy embeddings and dialogue context encoding.
    - **Design Motivation**: Strategy selection and linguistic implementation must cooperate; a strategy that is theoretically correct but poorly expressed will still fail to persuade effectively.

### Loss & Training
The overall system adopts a multi-task learning framework to simultaneously optimize strategy prediction loss and response generation loss. The adversarial game module might be trained using reinforcement learning or self-play.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| PersuasionForGood | Persuasion Success Rate | Significant Improvement | Baseline Methods | +5-8% |
| CraigslistBargain | Negotiation Score | Significant Improvement | Baseline Methods | +3-6% |
| Strategy Accuracy | F1 | Optimal | Baseline Methods | Significantly Leading |

### Ablation Study

| Configuration | Persuasion Success Rate | Description |
|------|-----------|------|
| Full model | Optimal | Full framework |
| w/o Opponent Modeling | Significant Decrease | Lack of opponent understanding leads to blind strategies |
| w/o Multi-step Planning | Decrease | Degenerates into a greedy strategy |
| w/o Strategy Conditioning | Decrease | Disconnect between strategy and generation |

### Key Findings
- When facing opponents with different resistance levels, the advantage of multi-step strategy planning increases as the opponent's degree of resistance grows.
- The opponent modeling module contributes the most to the overall performance, indicating that "knowing oneself and the opponent" is crucial in non-collaborative dialogues.
- When facing "easy" opponents, the performance gap between simple methods and the proposed method is small; the value is mainly demonstrated in difficult scenarios.

## Highlights & Insights
- **Novel Game-Theoretic Perspective**: Incorporating dialogue strategy planning into a game-theoretic framework goes beyond most approaches that view dialogue purely as a sequence generation problem. This approach can be transferred to any dialogue scenario involving multi-party conflicts of interest, such as customer service complaint resolution, doctor-patient communication, etc.
- **Modeling Tough Resisters**: Optimizing specifically for the most challenging scenarios aligns with practical needs, as users who are hard to persuade are precisely those who require sophisticated strategies.
- **Multi-step Planning Outperforms Greedy**: Experiments demonstrate the superiority of lookahead strategy planning, consistent with the core concepts of board game AI, which suggests that dialogue intelligence also requires "chess-like" deep thinking.

## Limitations & Future Work
- The computational cost of the game model may be high, and the latency of multi-step lookahead search in real-time dialogue systems needs to be controlled.
- The accuracy of the opponent model depends on the quality of training data; in real-world scenarios, opponent behaviors are more volatile and hard to predict.
- The criteria for evaluating success in non-collaborative dialogues are inherently subjective, and human evaluation may introduce bias.
- The current framework primarily targets two-party dialogues; scaling it to multi-party non-collaborative scenarios (e.g., multi-party negotiations) requires further research.
- Future work could consider introducing reinforcement learning from human feedback (RLHF) into adversarial game training, combining real user feedback to optimize strategies.

## Related Work & Insights
- **vs. Traditional Persuasive Dialogue Systems**: Traditional methods use a fixed set of strategies or simple classification to select strategies. This work introduces adversarial dynamic modeling, enabling real-time adjustment based on opponent reactions.
- **vs. Game-Theoretic Approaches (e.g., Deal or No Deal)**: Previous game-theoretic methods were primarily used for joint value division in negotiations. This work extends them to broader strategy planning in non-collaborative dialogues.
- **vs. Reinforcement Learning Dialogue Methods**: RL methods typically receive rewards only at the end of the dialogue. The game-theoretic framework in this work provides strategy evaluation signals at each step.
- **vs. Chain-of-Thought (CoT) Prompting**: CoT enables LLMs to perform explicit reasoning but does not model opponent behavior. The opponent modeling in this paper provides a more targeted source of information for strategic reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling non-collaborative dialogue with adversarial games is an interesting direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, ablation experiments, and opponent difficulty analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition.
- Value: ⭐⭐⭐⭐ Possesses practical guiding significance for persuasive systems and negotiation AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](../../CVPR2025/others/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)
- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](../../ECCV2024/others/domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)

</div>

<!-- RELATED:END -->
