---
title: >-
  [Paper Note] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Attack Defense] This paper proposes AGD (Adversarial Game Defense), an LLM jailbreak defense method based on adversarial games. By dynamically adjusting the internal representations of the model to balance helpfulness and harmlessness, AGD significantly improves LLM safety through three stages: IQR anomaly detection, bi-level optimization game, and expert model sampling.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Attack Defense"
  - "Adversarial Game"
  - "Attention Weight Correction"
  - "Nash Equilibrium"
  - "Internal Representation Steering"
date: 2026-05-08
content_hash: d2cfa06ceb83cea7
---

# AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: Alignment RLHF / LLM Safety  
**Keywords**: Jailbreak Attack Defense, Adversarial Game, Attention Weight Correction, Nash Equilibrium, Internal Representation Steering

## TL;DR
This paper proposes AGD (Adversarial Game Defense), an LLM jailbreak defense method based on adversarial games. By dynamically adjusting the internal representations of the model to balance helpfulness and harmlessness, AGD significantly improves LLM safety through three stages: IQR anomaly detection, bi-level optimization game, and expert model sampling.

## Background & Motivation

**Background**: Large language models have demonstrated powerful capabilities in practical applications, but they also face severe threats from jailbreak attacks. Attackers bypass the model's safety alignment through meticulously designed prompts, inducing the model to generate harmful content.

**Limitations of Prior Work**: Current defense methods mainly fall into two categories: (1) post-training alignment and prompt engineering, which rely on safely annotated datasets and safety prompt templates but adapt poorly to out-of-distribution (OOD) attacks; (2) internal representation steering-based methods which allow real-time adjustments to defend against OOD attacks, but modifying representations disrupts the forward pass during inference, leading to a decline in model utility.

**Key Challenge**: There exists a fundamental competition between the helpfulness and harmlessness of LLMs. Existing methods either sacrifice helpfulness for safety or lack sufficient security. Simply modifying internal representations cannot optimize both objectives simultaneously.

**Goal**: To design a defense mechanism that dynamically balances helpfulness and harmlessness, effectively defending against various jailbreak attacks without significantly degrading the model's utility.

**Key Insight**: The authors view helpfulness and harmlessness as two opposing objectives in game theory, utilizing the concept of adversarial games and employing bi-level optimization to automatically find the Nash equilibrium, thereby achieving an optimal balance between the two objectives.

**Core Idea**: Modeling LLM safety defense as a two-player variable-sum game, achieving a dynamic balance between safety and helpfulness through IQR anomaly detection of attention weights, adversarial training for attention correction, and bi-level optimization to approach the Nash equilibrium.

## Method

### Overall Architecture
The AGD method consists of three core stages: (1) Anomaly attention detection and correction phase, which uses the IQR method to identify and correct attention heads perturbed by jailbreak attacks; (2) Adversarial game optimization phase, which allows the "helpfulness player" and the "harmlessness player" to engage in an adversarial game in the attention activation space through bi-level optimization, approaching the Nash equilibrium; (3) Safe sampling phase, which introduces an expert model to guide the sampling of the next token to generate safer responses.

### Key Designs

1. **IQR Anomaly Attention Detection and Correction**:

    - Function: Identifies and corrects anomalous attention weights perturbed by jailbreak attacks.
    - Mechanism: It is observed that jailbreak attacks lead to anomalous shifts in the weights of specific attention heads. The Interquartile Range (IQR) method is used to analyze the distribution of weights for each attention head. An attention head is flagged as anomalous when its activation value falls outside the range $Q_1 - 1.5 \times IQR$ or $Q_3 + 1.5 \times IQR$. The anomalous attention weights are then corrected through adversarial training to restore them to a normal distribution.
    - Design Motivation: The core mechanism of jailbreak attacks is to alter the attention allocation pattern of the model through specific token combinations, causing the model to "ignore" safety constraints. By detecting these anomalous patterns, attacks can be identified and corrected in real-time during inference.

2. **Bi-Level Optimization Adversarial Game**:

    - Function: Finds the optimal equilibrium point between helpfulness and harmlessness.
    - Mechanism: Two "players" are defined—the helpfulness player and the harmlessness player, each controlling different attention head activations. The problem is modeled as a two-player variable-sum game using a bi-level optimization framework: the outer loop maximizes the helpfulness objective, while the inner loop minimizes the harmlessness objective. The two players alternately optimize their respective strategies, approaching the Nash Equilibrium through an iterative process. At the equilibrium point, neither player can improve their own objective by unilaterally changing their strategy.
    - Design Motivation: Traditional methods treat safety as a single-objective optimization problem, ignoring its conflict with helpfulness. The game-theoretic framework naturally models this competitive relationship, and the Nash equilibrium ensures that the interests of both sides are reasonably considered.

3. **Expert Model Guided Safe Sampling**:

    - Function: Further ensures the safety of the generated content during the token sampling phase.
    - Mechanism: A pre-trained safety expert model is introduced to evaluate the safety of each candidate token during the decoding phase. The token probability distribution from the original model is combined with the safety score from the expert model to adjust sampling probabilities, giving safer tokens higher sampling weights.
    - Design Motivation: Even if internal representations are adjusted through attention correction and game-based optimization, unsafe token sequences may still emerge during autoregressive generation. The expert model provides a final line of defense.

### Loss & Training
AGD adopts a bi-level optimization framework, where the outer loop maximizes the helpfulness objectives and the inner loop minimizes the harmlessness losses. Both loss functions are defined based on helpfulness evaluation metrics and safety evaluation metrics, respectively, approaching the Nash equilibrium through alternating gradient updates.

## Key Experimental Results

### Main Results

| Method | GCG ASR↓ | AutoDAN ASR↓ | PAIR ASR↓ | Avg. ASR↓ | MT-Bench↑ |
|------|---------|-------------|----------|---------|----------|
| No Defense | 56.0 | 78.0 | 44.0 | 59.3 | 6.8 |
| Self-Reminder | 26.0 | 48.0 | 28.0 | 34.0 | 6.2 |
| RepE | 8.0 | 22.0 | 14.0 | 14.7 | 5.4 |
| AGD (Ours) | **2.0** | **6.0** | **4.0** | **4.0** | **6.5** |

### Ablation Study

| Configuration | Avg. ASR↓ | MT-Bench↑ | Description |
|------|---------|----------|------|
| Full AGD | 4.0 | 6.5 | Complete model |
| w/o IQR Detection | 12.0 | 6.3 | Safety decreases after removing anomaly detection |
| w/o Game Optimization | 8.0 | 5.8 | Helpfulness decreases significantly after removing the game |
| w/o Expert Sampling | 6.0 | 6.4 | Safety decreases slightly after removing the expert model |

### Key Findings
- The bi-level optimization game mechanism is the most core contribution of AGD. Removing it leads to the most significant drop in helpfulness (MT-Bench drops from 6.5 to 5.8), indicating that the game is key to balancing the two objectives.
- IQR anomaly detection is particularly crucial for defending against OOD attacks and remains effective against unseen attack types.
- While maintaining high safety, AGD incurs minimal helpfulness loss (MT-Bench only drops from 6.8 to 6.5), far outperforming methods like RepE.

## Highlights & Insights
- Modeling LLM safety defense as an adversarial game is an ingenious approach that naturally handles the conflict between helpfulness and harmlessness, which is more elegant than simple regularization methods. This framework can be extended to other multi-objective optimization scenarios.
- The IQR anomaly detection method is simple and effective. It identifies attacks without requiring additional training, exhibiting good practicality and transferability.
- Expert model sampling provides a "soft" safety constraint, differing from hard refusal strategies, which can achieve a better balance between safety and informativeness.

## Limitations & Future Work
- The authors have not released the code or the arXiv preprint, which limits the reproducibility of the method and further research by the community.
- The convergence speed and stability of the adversarial game have not been fully analyzed, and its performance across different model architectures remains unclear.
- Bi-level optimization introduces additional computational overhead during inference, which could be a bottleneck for real-time application scenarios.
- The IQR method assumes that normal attention weights approximately follow a normal distribution, which might lead to false positives/misclassifications in scenes with heavy-tailed distributions.
- The choice and training of the expert model significantly impact the final performance, and the paper does not fully discuss the impact of different expert models.

## Related Work & Insights
- **vs RepE (Representation Engineering)**: RepE directly modifies internal representations to enhance safety but ignores the loss of helpfulness. AGD balances both through the game framework, maintaining helpfulness better.
- **vs Self-Reminder**: Self-Reminder adds safety reminders to the inputs via prompt engineering but adapts poorly to OOD attacks. AGD dynamically adjusts at the representation level, offering stronger generalization.
- **vs Circuit Breakers**: Circuit Breakers block harmful generation by training additional safety circuits, requiring large volumes of safety data. AGD is an inference-time defense that requires no extra training data.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing a game-theoretic framework for LLM safety defense is creative, though IQR detection and expert sampling are relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple attack types and evaluation dimensions, with relatively complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described, but the lack of public code severely impacts verifiability.
- Value: ⭐⭐⭐⭐ Proposes a new paradigm that accommodates both safety and helpfulness, which is inspiring for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)

</div>

<!-- RELATED:END -->
