---
title: >-
  [Paper Note] Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation
description: >-
  [ACL2026][Reinforcement Learning][UGC Quality Assessment] This paper proposes the CASTER task and CASTER-Bench, and introduces MEDEA, which simulates community reactions through Social-CoT, SFT…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "UGC Quality Assessment"
  - "Social-CoT"
  - "Community Resonance"
  - "GRPO"
  - "Multimodal Reasoning"
date: 2026-05-08
content_hash: a8f3cf1d0483bd54
---

# Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation

**Conference**: ACL2026  
**arXiv**: [2606.01897](https://arxiv.org/abs/2606.01897)  
**Code**: To be confirmed  
**Area**: Multimodal Evaluation / RL Alignment  
**Keywords**: UGC Quality Assessment, Social-CoT, Community Resonance, GRPO, Multimodal Reasoning

## TL;DR
This paper proposes the CASTER task and CASTER-Bench, and introduces MEDEA, which simulates community reactions through Social-CoT, SFT, and process-supervised Reinforcement Learning with a Social Alignment Reward. MEDEA improves High-Quality F1 to 0.650 and Macro-F1 to 0.749 on CASTER-Bench, significantly outperforming traditional VQA and general LMM baselines.

## Background & Motivation
**Background**: Traditional video quality assessment primarily measures clarity, distortion, aesthetics, and technical quality. Recently, LMMs have also been applied to UGC quality estimation, but most still treat textual information as static features or utilize standard CoT for logical analysis.

**Limitations of Prior Work**: "Good content" on real-world UGC platforms is not determined solely by image quality. A video might have average technical quality but receive strong positive feedback due to its narrative, emotional impact, knowledge value, or community culture. Conversely, high view counts might be driven by clickbait, low-quality stimulation, or induced comments. Relying solely on visual signals or general multimodal reasoning makes it difficult to distinguish between content that "looks good" and content that "truly resonates positively with the community."

**Key Challenge**: Platforms need to judge the intrinsic quality of content during early recommendation and moderation stages, but newly uploaded content often lacks sufficient comments. Models must infer potential community reactions from covers, keyframes, titles, tags, ASR, and metadata. This requires models to possess social reasoning similar to Theory of Mind, rather than just performing signal quality regression.

**Goal**: The authors propose CASTER, redefining UGC quality assessment as "whether content achieves positive community resonance." To this end, they construct CASTER-Bench and propose MEDEA, which first simulates Social-CoT from diverse viewer personas and then aggregates these into final high/low-quality judgments.

**Key Insight**: Instead of having the model directly output a binary classification, this work requires the model to first generate multiple empathetic reasoning paths in the style of "community comments." During the training phase, these reasoning paths are constrained by real high-interaction comments and expert labels, enabling the model to learn judgment standards that align more closely with real community perception.

**Core Idea**: Use Social-CoT to explicitly simulate a "community mind," then align the generated social reasoning paths with real user comments via a Social Alignment Reward. This shifts UGC quality assessment from image quality judgment to community resonance modeling.

## Method

### Overall Architecture
The paper presents two core contributions. The first is CASTER-Bench: 1,485 long-video UGC items covering 30 major categories. Each item includes multimodal inputs such as video frames, covers, titles, tags, category metadata, and ASR transcripts, annotated by 10 professional content operation experts across four dimensions: Production Quality, Perceived Value, Information Utility, and Narrative Excellence. The second is MEDEA: a multimodal evaluation framework that mines Social-CoT training data from community comments, learns social reasoning formats via SFT, and finally optimizes the reasoning process through GRPO and social alignment rewards.

### Key Designs
1. **CASTER Task and CASTER-Bench**:
	- Function: Converts UGC quality evaluation from aesthetic/technical VQA to community-aware resonance assessment.
	- Mechanism: Given the cover image, keyframes, title, tags, category metadata, and ASR transcript, the model predicts whether the content is High-Quality. The dataset contains 1,485 UGC items with an average duration of 442 seconds (total 182.5 hours). The label distribution is Excellent 10.6%, Good 17.0%, Average 38.6%, and Poor 33.7%.
	- Design Motivation: The value of long-video UGC stems from narrative, knowledge density, and emotional resonance; traditional 8-20 second short-clip VQA datasets cannot capture these factors.

2. **Social-CoT Construction and Skellam Consensus Aggregation**:
	- Function: Transforms real comments into supervisable social reasoning paths.
	- Mechanism: For unlabeled UGC, the system selects the top-50 liked comments and uses a teacher model to filter 15-20 reaction anchors related to creativity, emotion, and narrative. Gemini-2.5-Flash is then used to instantiate diverse viewer personas and explain which visual/narrative elements triggered these reactions. Each simulated comment is assigned a support or oppose stance. If the support count is $X$ and the oppose count is $Y$, the Skellam-normalized consensus is calculated as $z=(X-Y)/\sqrt{X+Y}$. Content is labeled as High-Quality when $z\geq1.5$, otherwise Low-Quality.
	- Design Motivation: Simple majority voting is easily biased by comment volume and emotional shifts; Skellam normalization ensures the final judgment reflects "statistically significant community support."

3. **Process-Supervised RL and Social Alignment Reward**:
	- Function: Ensures generated Social-CoT is not merely templated praise but approaches the granularity of real community language and emotion.
	- Mechanism: MEDEA undergoes SFT using 54k Gemini-labeled CoT samples and 3k human-annotated UGC samples. Subsequently, a composite reward $r=r_{format}+r_{label}+r_{diversity}+r_{social}$ is optimized via GRPO on expert samples. $r_{format}$ ensures structural integrity, $r_{label}$ rewards correct final binary classification, $r_{diversity}$ penalizes repetitive emotional paths, and $r_{social}$ calculates the average cosine embedding similarity between the generated persona and held-out real high-interaction comments.
	- Design Motivation: Prompting general LMMs for Social-CoT is insufficient to internalize platform community standards; real comment similarity rewards provide "social grounding" to prevent Social Mode Collapse.

### Loss & Training
Training is conducted in two stages. In the SFT stage: batch size 256, learning rate $5e^{-6}$, cosine schedule, and a decay ratio of 0.2. In the RL stage: batch size 64, learning rate $1e^{-6}$, cosine schedule, PPO clip ratio 0.2, KL coefficient 0.001, entropy coefficient 0.001, rollout number 8, and rollout temperature 0.6. During inference: top-k 50, top-p 0.7, and temperature 0.6. The paper emphasizes that RL only utilizes human-curated samples to ensure reinforcement signals are anchored to expert annotations rather than amplifying teacher model pseudo-label biases.

## Key Experimental Results

### Main Results
Since High-Quality samples are relatively sparse in CASTER-Bench, High-Quality F1 is the critical metric. MEDEA significantly outperforms traditional VQA, standard LMMs, Long-CoT LMMs, and prompt-only Social-CoT simulations.

| Method | HQ Precision | HQ Recall | HQ F1 | Macro-F1 | Remarks |
|------|-------------:|----------:|------:|---------:|------|
| FastVQA | 0.347 | 0.440 | 0.388 | 0.554 | Traditional VQA |
| MaxVQA | 0.345 | 0.518 | 0.414 | 0.552 | Strong traditional VQA |
| Qwen3-VL-Plus | 0.366 | 0.893 | 0.519 | 0.542 | Std LMM, High recall/Low precision |
| GPT-5.2 reasoning | 0.401 | 0.903 | 0.555 | 0.595 | Strong Long-CoT baseline |
| Qwen3-VL-Plus social-CoT | 0.380 | 0.766 | 0.508 | 0.578 | Prompt-based Social-CoT |
| Claude-4.5-opus social-CoT | 0.371 | 0.810 | 0.510 | 0.561 | Prompt-based Social-CoT |
| MEDEA | 0.603 | 0.705 | 0.650 | 0.749 | Full Method |

### Ablation Study
| Configuration | HQ F1 | Low-Quality F1 | Macro-F1 | Description |
|------|------:|---------------:|---------:|------|
| SFT-pseudo-label | 0.487 | 0.686 | 0.587 | Pseudo-labels help format but weaken judgment |
| SFT-human-label | 0.371 | 0.710 | 0.541 | Insufficient human samples, low recall |
| SFT-w/o-social-CoT | 0.510 | 0.638 | 0.574 | Removing Social-CoT leads to instability |
| RL-pseudo+human | 0.536 | 0.848 | 0.692 | RL improves overall performance |
| RL-w/o-social-reward | 0.613 | 0.836 | 0.725 | Lacks social alignment, prone to templates |
| RL-w/o-social-CoT | 0.421 | 0.821 | 0.621 | Significant drop without social reasoning |
| MEDEA(RL-human-label) | 0.650 | 0.847 | 0.749 | Full Method |

### Key Findings
- General LMMs exhibit a generosity bias: models like GPT-5.2 and Claude-4.5-opus can exceed 90% High-Quality recall but have low precision (30%-40%), over-interpreting mediocre content as high-quality.
- Traditional VQA is biased towards the Low-Quality class, with High-Quality F1 scores mostly between 0.33-0.41, indicating that image quality signals alone are insufficient to detect community resonance.
- Prompting Social-CoT alone cannot substitute for training; the HQ F1 of Qwen3-VL-Plus social-CoT (0.508) is significantly lower than MEDEA (0.650).
- The Social Alignment Reward not only improves classification scores but also reduces repetitive and vague "so beautiful" style template reasoning.

## Highlights & Insights
- **Redefining UGC Quality**: The paper shifts the objective from signal quality to community resonance, a task setting that aligns more closely with platform requirements than traditional VQA scores.
- **Social-CoT as an Interpretable Intermediate Layer**: The model provides simulated viewer reactions rather than just a label, making error analysis more accessible regarding storytelling, emotion, or information value.
- **Reward Design Captures Authenticity**: The use of real high-interaction comments as anchors for $r_{social}$ constrains the reasoning process more effectively than simple label-matching rewards.
- **Dataset Design for Real Long Videos**: The use of videos averaging 442 seconds (total 182.5 hours) distinguishes this work from short-clip technical quality datasets.

## Limitations & Future Work
- Social-CoT incurs significant inference overhead; despite MEDEA being smaller than some API-based LMMs, real-time recommendation scenarios may require caching, distillation, or early-exit mechanisms.
- Social alignment is optimized based on specific platform dynamics; transferring to different cultures, community norms, or content ecosystems may require re-annotation and re-alignment.
- Binary High/Low classification is relatively coarse; community resonance is continuous, multi-dimensional, and changes over time.
- The method currently relies on rich multimodal metadata; its effectiveness remains to be verified in scenarios with only titles/covers or extremely sparse comments.
- Future work could extend to multi-level quality, sub-community preference modeling, temporal resonance prediction, and lightweight Social-CoT distillation.

## Related Work & Insights
- **vs FastVQA / DOVER / MaxVQA / Q-Align / FineVQ**: While these focus on visual technical/aesthetic quality, CASTER focuses on whether content triggers real community positive feedback.
- **vs Long-CoT LMM**: Long-reasoning models provide detailed analysis but tend to be over-permissive without community standards; MEDEA constrains this bias using expert labels and real comments.
- **vs prompt-only Social-CoT**: Prompting can improve social perspective, but training and rewards are necessary to internalize which reactions are genuine and discriminative.
- **Insight**: For recommendation, moderation, and creator feedback systems, "simulating user group reactions" can serve as an interpretable evaluation layer, provided it is constrained by real community data and expert standards to avoid hollow generated commentary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Task redefinition, Social-CoT, and social alignment rewards are well-integrated and distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, ablations, costs, and reasoning quality are comprehensive; cross-platform generalization lacks empirical evidence.
- Writing Quality: ⭐⭐⭐⭐☆ Strong storyline with clear motivation and methodology.
- Value: ⭐⭐⭐⭐☆ Highly insightful for UGC recommendation and multimodal social reasoning, though implementation requires careful handling of cost and platform bias.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](../../AAAI2026/reinforcement_learning/rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)
- [\[AAAI 2026\] G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation](../../AAAI2026/reinforcement_learning/g-ubs_towards_robust_understanding_of_implicit_feedback_via_group-aware_user_beh.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ICLR 2026\] PreferThinker: Reasoning-based Personalized Image Preference Assessment](../../ICLR2026/reinforcement_learning/preferthinker_reasoning-based_personalized_image_preference_assessment.md)

</div>

<!-- RELATED:END -->
