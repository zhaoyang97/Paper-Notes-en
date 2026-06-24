---
title: >-
  [Paper Note] Zero-Shot Conversational Stance Detection: Dataset and Approaches
description: >-
  [ACL 2025][stance detection] This work constructs the first zero-shot multi-turn multi-party conversational stance detection dataset, ZS-CSD (280 targets, 17,063 conversational samples), and proposes the SITPCL model. By combining a speaker interaction encoder with target-aware prototypical contrastive learning, SITPCL achieves state-of-the-art performance (F1-macro of 43.81%) in zero-shot conversational stance detection.
tags:
  - "ACL 2025"
  - "stance detection"
  - "zero-shot"
  - "conversational"
  - "speaker interaction"
  - "prototypical contrastive learning"
date: 2026-05-08
content_hash: e518491f98996201
---

# Zero-Shot Conversational Stance Detection: Dataset and Approaches

**Conference**: ACL 2025  
**arXiv**: [2506.17693](https://arxiv.org/abs/2506.17693)  
**Code**: [GitHub](https://github.com/whu-yzding/ZS-CSD)  
**Area**: Other  
**Keywords**: stance detection, zero-shot, conversational, speaker interaction, prototypical contrastive learning

## TL;DR

This work constructs the first zero-shot multi-turn multi-party conversational stance detection dataset, ZS-CSD (280 targets, 17,063 conversational samples), and proposes the SITPCL model. By combining a speaker interaction encoder with target-aware prototypical contrastive learning, SITPCL achieves state-of-the-art performance (F1-macro of 43.81%) in zero-shot conversational stance detection.

## Background & Motivation

Stance detection aims to identify users' opinion leanings (favor, against, or neutral) toward specific targets from text, which is widely applied in fields like sentiment recognition, argument mining, and rumor detection. With the increasing number of online debates on social media, stance detection in conversational scenarios has become an important research direction.

Existing conversational stance detection suffers from three key limitations:

1. **Extremely limited number of targets**: The largest conversational stance detection dataset, MT-CSD, contains only 5 targets, and existing datasets cover only either noun phrases or post-type targets, failing to handle the vast number of unseen targets in real-world scenarios.
2. **Neglect of speaker information**: Previous annotation and modeling processes only considered reply relations and historical context, without fully exploiting speaker context (historical utterances by the same user) and potential speaker-interpersonal interaction (the interactive relationships among different users).
3. **Lack of zero-shot settings**: Research on conversational stance detection is limited to in-target and cross-target tasks, while the zero-shot scenario, which is closer to practical applications and requires identifying stance towards entirely unseen targets, remains unexplored.

Addressing these challenges, this work constructs the first zero-shot conversational stance detection dataset, ZS-CSD, and proposes the SITPCL model as a benchmark method.

## Method

### Overall Architecture

SITPCL (Speaker Interaction and Target-aware Prototypical Contrastive Learning) is a four-stage pipeline:
1. **Utterance Encoder**: Uses a pre-trained language model (Chinese-RoBERTa) to encode each utterance-target pair individually. After appending stance templates, a GRU layer is applied to obtain the context representation sequence.
2. **Speaker Interaction Encoder**: Models intra-speaker and inter-speaker dependencies individually via attention mechanisms.
3. **Target-aware Prototypical Contrastive Learning**: A contrastive learning objective anchored by target prototypes to enhance representation discriminability across different targets.
4. **Classifier**: A softmax classifier outputs stance predictions.

### Key Designs

1. **ZS-CSD Dataset Construction**:

    - Function: Provides the first large-scale, high-quality evaluation resource for zero-shot conversational stance detection.
    - Mechanism: Collects 2 million posts and comments from Weibo, screens them with keywords (covering 6 controversial domains), constructs discussion trees, and samples at multiple depths. Then, 8 annotators complete a two-stage annotation: 3 experts first identify the targets (1-2 per conversation), followed by 5 annotators labeling the stances.
    - Highlights: (a) Includes both noun phrase targets (113) and claim targets (167), totaling 280 targets; (b) Annotations simultaneously consider dialogue history, speaker context, and speaker interaction; (c) High annotation quality with a Cohen's Kappa of 0.83; (d) Targets globally disjoint across train/validation/test sets to ensure genuine zero-shot evaluation.

2. **Speaker Interaction Encoder**:

    - Function: Captures opinion consistency within the same speaker and interactive relationships across different speakers in the conversation.
    - Mechanism (Intra-speaker dependencies): Concatenates the current speaker's previous enhanced representation with the current utterance representation as the query vector. Information is aggregated over the speaker’s historical context via an attention mechanism to produce the intra-speaker state vector $v_i^{intra}$.
    - Mechanism (Inter-speaker dependencies): Uses the context representation of the current utterance, $h_i$, as the query, and the previous enhanced states of other speakers as keys. Inter-speaker interaction signals are captured via attention to yield the inter-speaker state vector $v_i^{inter}$.
    - Fusion: $v_i = W_3[v_i^{intra} \oplus v_i^{inter}] + b_3$, producing final speaker-enhanced representations through linear transformation.
    - Design Motivation: In social media conversations, users' stances usually exhibit consistency (the same user tends to remain consistent across different replies), and there exist mutual influences and adversarial relationships between different users.

3. **Target-aware Prototypical Contrastive Learning**:

    - Function: Enhances the model's ability to distinguish between different targets and improves zero-shot generalization.
    - Mechanism: Calculates a prototype representation $p_t$ (the average of all utterance representations under target $t$) for each target $t$. An InfoNCE-style contrastive loss is introduced to pull utterance representations closer to their corresponding target prototype and push them away from other target prototypes.
    - Loss Formula: 
      $$\mathcal{L}_{TPC} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(v_i, p_{y_i})/\tau)}{\sum_{k=1}^{K}\exp(\text{sim}(x_i, p_k)/\tau)}$$
    - Design Motivation: In zero-shot scenarios, the model needs to understand the concept of a "target" rather than memorize specific targets. By progressively separating the representation spaces of different targets, the model can transfer this discriminative ability to unseen targets.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{CE} + \gamma \mathcal{L}_{TPC}$

- $\mathcal{L}_{CE}$: Standard cross-entropy loss for 3-class stance prediction.
- $\mathcal{L}_{TPC}$: Target-aware prototypical contrastive loss, where the temperature parameter $\tau$ controls the sharpness of the distribution.
- $\gamma$: Balancing coefficient.

Training Configuration:
- PLM: Chinese-RoBERTa-wwm-ext, GRU hidden dimension 768.
- Optimizer: AdamW, learning rate 1e-5, weight decay 1e-6.
- Training: 20 epochs, batch size 16.
- Results averaged over 5 random seeds.
- Hardware: 2× NVIDIA RTX 3090, training time < 3 hours.

## Key Experimental Results

### Main Results

F1-macro scores for zero-shot conversational stance detection:

| Method | Mixed Target | Noun Phrase Target | Claim Target |
|------|---------|------------|---------|
| Llama3-8B (zero-shot) | 35.91 | 39.37 | 33.79 |
| GPT-4o-mini (zero-shot) | 38.16 | 40.21 | 36.57 |
| GPT-3.5 (zero-shot) | 40.25 | 47.35 | 35.53 |
| Llama3-70B (zero-shot) | 41.07 | 48.72 | 36.56 |
| Qwen2.5-14B (zero-shot) | 42.78 | 44.60 | 41.34 |
| RoBERTa (fine-tuned) | 40.27 | 42.79 | 38.71 |
| Branch-BERT | 43.11 | 44.48 | 38.96 |
| GLAN | 41.79 | 45.04 | 39.83 |
| **SITPCL (Ours)** | **43.81** | **47.54** | **41.47** |

### Ablation Study

| Configuration | Mixed Target F1-macro | Note |
|------|-----------------|------|
| SITPCL (Full) | **43.81** | All components |
| W/o Speaker Interaction Encoder (SIE) | 42.91 (-0.90) | Remove speaker interaction modeling |
| W/o Target Prototypical Contrastive Learning (TPCL) | 42.59 (-1.22) | Remove contrastive learning |
| W/o Both (BOTH) | 41.40 (-2.41) | Keep only base encoder |

### Key Findings

1. **SITPCL achieves the best performance across all target types**: F1-macro of 43.81% on Mixed Targets, 47.54% on Noun Phrase Targets, and 41.47% on Claim Targets, comprehensively outperforming zero-shot LLM methods and fine-tuned baselines.
2. **Even the strongest LLM (Llama3-70B) only reaches 41.07%**: This highlights the high difficulty of zero-shot conversational stance detection, where the optimal method scores below 44% in F1-macro.
3. **The two components are complementary**: SIE significantly improves "Favor" stance identification, while TPCL markedly boosts "Neutral" classification (+3.95%). Removing both leads to the largest performance drop (-2.41%).
4. **Dialogue depth affects model performance**: SITPCL demonstrates outstanding performance in shallow dialogues (depth=1) (40.19% vs. ~25-33% for baselines) and keeps stable performance in deep dialogues (depth $\ge 6$) (45.25%).
5. **Noun phrase targets vs. Claim targets**: Claim targets are generally more difficult (all methods perform 3-8 percentage points lower) because claims themselves express opinions, making users' stance expressions more implicit and indirect.

## Highlights & Insights

- **First zero-shot conversational stance detection dataset**: 280 targets is significantly larger than the previous maximum of 5 targets (MT-CSD). Covering both noun phrases and claims, it fills a critical gap in this field.
- **Novel speaker interaction modeling approach**: Instead of solely relying on replying relations, it uses attention mechanisms to capture intra-speaker opinion consistency and inter-speaker stance adversarial dynamics, providing finer-grained signals for dialogue understanding.
- **Zero-shot adaptation of prototypical contrastive learning**: By combining prototypical network concepts with contrastive learning, the model learns to make judgments based on "target concepts" rather than "specific targets", effectively enhancing zero-shot generalization.
- **Intriguing LLM vs. Small PLM comparison**: While the 70B LLM performs closely to small fine-tuned models on noun phrase targets, it lags behind significantly on claim targets. This indicates that utilizing conversational structures is more critical than raw language understanding capability.

## Limitations & Future Work

- **Low performance ceiling**: The best F1-macro score is only 43.81%, indicating a substantial gap remains before practical application.
- **Chinese only**: The dataset is sourced from Weibo with Chinese annotations; its cross-lingual generalization capability has not been evaluated.
- **Coarse-grained target classification**: Targets are divided only into noun phrases and claims, without considering finer-grained target semantic classification.
- **Lack of multimodal information**: Social media conversations often contain images, memes, and other multimodal signals, whereas this study only utilizes text.
- **Speaker interaction modeling can be enhanced**: Currently, simple attention mechanisms are used. Graph neural networks or more complex dialogue structure modeling methods could be explored.

## Related Work & Insights

- **vs. VAST/C-STANCE/EZ-STANCE** (Sentence-level zero-shot): This work extends the zero-shot setting to the conversational level for the first time, adding the modeling dimension of speaker interactions.
- **vs. MT-CSD/MmMtCSD** (Conversational-level stance detection): These studies only involve 3–5 targets and are confined to in-target/cross-target settings. The 280-target zero-shot setting in this work is closer to real-world scenarios.
- **vs. Zero-shot LLM methods**: Even a 70B-parameter LLM fails to outperform well-designed small models, showing that the challenge of conversational stance detection lies not just in language comprehension but in the necessity of capturing dialogue structures and interaction dynamics.
- **Insight**: NLP tasks in conversational scenarios should focus more on speaker identities and interaction patterns. Such structured information is difficult for LLMs to fully exploit through simple prompting.

## Rating

- Novelty: ⭐⭐⭐⭐ First zero-shot conversational stance detection dataset and task definition, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both LLMs and fine-tuned baselines, with complete ablation, in-depth analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a transparent and detailed dataset construction process.
- Value: ⭐⭐⭐ The contribution of the dataset outweighs that of the method. The design of the SITPCL model is relatively conventional, and the absolute performance of 43.81% suggests that much stronger methods are still required for this task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)
- [\[CVPR 2026\] Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment](../../CVPR2026/others/zero-shot_detection_of_ai-generated_image_via_raw-rgb_alignment.md)
- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)

</div>

<!-- RELATED:END -->
