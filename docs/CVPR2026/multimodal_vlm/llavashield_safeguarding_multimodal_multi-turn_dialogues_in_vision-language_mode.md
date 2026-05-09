---
title: >-
  [Paper Note] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Content Moderation] This paper proposes LLaVAShield — the first content moderation model designed for multimodal multi-turn dialogues — along with the MMDS dataset (4,484 dialogues covering 8 major categories and 60 subcategories of risk) and MMRT, an automated MCTS-based red-teaming framework. LLaVAShield substantially outperforms baselines such as GPT-5-mini on safety auditing of both user and assistant turns.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Content Moderation
  - Multimodal Safety
  - Multi-Turn Dialogue
  - Red-Teaming
  - VLM Safety
date: 2026-05-08
content_hash: 9336e99b747e70d3
---

# LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2509.25896](https://arxiv.org/abs/2509.25896)
**Code**: [Project Page](https://leost123456.github.io/LLaVAShield/)
**Area**: Multimodal VLM
**Keywords**: Content Moderation, Multimodal Safety, Multi-Turn Dialogue, Red-Teaming, VLM Safety

## TL;DR

This paper proposes LLaVAShield — the first content moderation model designed for multimodal multi-turn dialogues — along with the MMDS dataset (4,484 dialogues covering 8 major categories and 60 subcategories of risk) and MMRT, an automated MCTS-based red-teaming framework. LLaVAShield substantially outperforms baselines such as GPT-5-mini on safety auditing of both user and assistant turns.

## Background & Motivation

**Urgent safety requirements for multimodal multi-turn dialogues**: VLMs are being deployed at scale in intelligent assistants, education, and other domains. Malicious users can exploit multi-turn interactions and cross-modal inputs to manipulate models, posing serious safety risks.

**Existing moderation methods are limited to single-turn or single-modal settings**: Current content moderation tools (e.g., OpenAI Moderation, LlamaGuard) are primarily designed for single-turn or text-only scenarios and cannot handle complex attack patterns that emerge across multiple dialogue turns.

**Concealment of malicious intent**: Attackers initiate conversations with benign topics, gradually escalate their malicious intent across turns, and distribute attack targets across both text and images — making single-turn moderation insufficient.

**Cumulative contextual risk**: Attackers decompose their ultimate goal across multiple turns and exploit the model's reliance on locally compliant early responses to progressively expand the attack surface, with risk accumulating as the dialogue progresses.

**Cross-modal joint risk**: Even ostensibly normal image-text pairings can trigger unsafe generation, revealing a systemic gap in cross-modal safety alignment.

**Data bottleneck**: There is a lack of datasets targeting multimodal multi-turn dialogue safety, and mainstream VLMs are generally safety-aligned, leaving the effective elicitation of unsafe responses insufficiently explored.

## Method

### Overall Architecture

The LLaVAShield system comprises three core components:

1. **MMDS Dataset Construction**: A complete data pipeline spanning malicious intent generation → image retrieval/generation → MMRT red-teaming → human annotation → data augmentation → rationale generation.
2. **MMRT Red-Teaming Framework**: An automated multimodal multi-turn red-teaming system based on MCTS, involving a three-party interaction among an attacker (Qwen2.5-VL-72B), a target model (GPT-4o / Qwen2.5-VL-72B), and an evaluator (GPT-4o).
3. **LLaVAShield Moderation Model**: Fine-tuned from LLaVA-OV-7B to perform safety auditing of both user inputs and assistant responses in multimodal multi-turn dialogues.

### Key Designs

**Safety Risk Taxonomy**: 8 top-level dimensions and 60 sub-dimensions covering comprehensive risk categories including violence, pornography, discrimination, and dangerous behaviors, each sub-dimension equipped with a standardized definition.

**MMRT Red-Teaming Algorithm**:
- **Attack Strategy Pool**: Four strategies — Gradual Guidance, Purpose Inversion, Query Decomposition, and Role Play.
- **Cross-Modal Attacks**: The attacker can replace sensitive terms with semantically aligned image references or generate attack images using Stable Diffusion 3.5 Medium, jointly forming cross-modal attacks with text.
- **MCTS Search**: Follows the standard Selection (PUCT formula) → Expansion (executing one round of attack→target→evaluation) → Simulation (forward rollout for $k$ turns) → Backpropagation pipeline for efficient attack path exploration.
- **Scoring**: Each turn is scored by the evaluator on two dimensions — harmfulness and progress toward malicious intent — on a scale of 1–5.

**Data Annotation and Augmentation**:
- Dual-role annotation: Each dialogue is annotated with safety ratings and violated policy dimensions for both the user and assistant sides.
- Four augmentation strategies: random removal of non-violated policy dimensions, rewriting unsafe responses into compliant text, removal of single-side context, and re-labeling after removing irrelevant policy dimensions.
- Role-decoupled dual-channel rationale generation: Independent reasoning explanations are generated for the user and assistant respectively, enhancing interpretability and traceability.

**Input/Output Formatting**:
- Input is organized as instruction + policy dimension list + dialogue history in JSON array format, with images marked by `<image>` placeholders and numbered as Image1, Image2, etc.
- Output is a structured JSON enclosed in `<OUTPUT>...</OUTPUT>` tags, containing 6 fields: rating, dimension, and rationale for both user and assistant.

### Loss & Training

- Modeled uniformly as a sequence-to-sequence task, maximizing the conditional log-likelihood: $\max_\theta \sum \log p(\mathcal{Y} \mid \mathcal{G}, \mathcal{P}, \mathcal{C}; \theta)$
- Base model: LLaVA-OV-7B
- Learning rate $2 \times 10^{-5}$ with cosine scheduling and 0.03% warmup
- Batch size = 1 with gradient accumulation over 4 steps; trained for 3 epochs
- Hardware: 8× NVIDIA RTX A6000 (48GB); training completed in approximately 3 hours

## Key Experimental Results

### Main Results

**Table 1: Main Results on MMDS Test Set (F1 %)**

| Model | Open-Source | User F1 | Assistant F1 |
|-------|-------------|---------|--------------|
| LLaVA-OV-7B | ✓ | 1.17 | 0.00 |
| InternVL3-8B | ✓ | 0.00 | 7.41 |
| Qwen2.5-VL-72B | ✓ | 33.33 | 28.00 |
| Qwen3-VL-30B-A3B | ✓ | 21.05 | 56.52 |
| Gemini-2.5-Pro | ✗ | 64.00 | 65.62 |
| GPT-4o | ✗ | 61.54 | 57.92 |
| GPT-5-mini | ✗ | 75.46 | 77.93 |
| Llama Guard-4-12B | ✓ | 14.21 | 28.21 |
| **LLaVAShield-7B** | **✓** | **95.71** | **92.24** |

**Table 2: External Safety Benchmark Results (MM-SafetyBench Recall / VLGuard F1 %)**

| Model | MM-SafetyBench Mean | VLGuard F1 |
|-------|---------------------|------------|
| InternVL3-8B | 39.73 | 20.79 |
| Qwen2.5-VL-7B | 25.17 | 37.79 |
| GPT-5-mini | 48.44 | 86.39 |
| Llama-Guard-4-12B | 44.49 | 64.87 |
| **LLaVAShield-7B** | **97.62** | **90.55** |

### Ablation Study

**Rationale Ablation (F1 %)**:

| Setting | User LLaVAShield | Assistant LLaVAShield |
|---------|------------------|-----------------------|
| Full (Vanilla) | 95.71 | 92.24 |
| w/o Rationale | 95.12 | 93.93 |

Rationale has limited impact on aggregate metrics but is retained to enhance interpretability.

**Policy Adaptation Test**: After removing violated policy dimensions, LLaVAShield achieves a false positive rate (FPR) of 0% on both user and assistant sides, compared to 30% and 34% for GPT-5-mini respectively, demonstrating strict adherence to currently active policy dimensions.

**Image Content Contribution**: Removing images reduces high-scoring (≥4) turns from 652 to 411 and increases low-scoring (=1) turns from 284 to 469, with an Average Score Gain (ASG) of 0.375, indicating that images make attacks more actionable and harmful.

### Key Findings

1. **Mainstream VLMs perform extremely poorly in multimodal multi-turn safety scenarios**: Open-source models exhibit near-zero user-side recall (e.g., InternVL3-8B at 0%, Qwen2.5-VL-7B at 0.59%), as they tend to classify all content as safe.
2. **LLaVAShield with 7B parameters substantially outperforms all baselines**: It surpasses the strongest baseline, GPT-5-mini, by +20.25 and +14.31 F1 points on the user and assistant sides respectively, while achieving 100% user-side precision.
3. **Strong cross-benchmark generalization**: Achieves a mean Recall of 97.62% on MM-SafetyBench (vs. 48.44% for GPT-5-mini) and an F1 of 90.55% on VLGuard, demonstrating that the model generalizes beyond the MMDS domain.
4. **Mainstream VLMs are highly vulnerable to MMRT attacks**: Qwen2.5-VL-72B yields a 100% attack success rate, GPT-4o reaches 98.21%, and even GPT-5-mini is compromised at 51.67%; only Claude-3.7-Sonnet shows relative robustness at 73.77%.

## Highlights & Insights

- **Complete data–method–evaluation loop**: A fully reproducible pipeline is established, from risk taxonomy definition and red-team attack generation to data annotation, augmentation, and moderation model training.
- **MCTS-driven red-teaming**: Multi-turn attacks are formulated as a tree search problem, substantially improving exploration efficiency and attack success rates compared to linear attack loops.
- **Dual-side auditing with flexible policy control**: The model simultaneously audits user inputs and assistant responses, with active policy dimensions configurable for different application contexts (FPR = 0%).
- **Role-decoupled rationale generation**: Independent reasoning explanations are generated for the user and assistant sides, balancing interpretability with audit traceability requirements.

## Limitations & Future Work

1. **Limited data scale**: MMDS contains only 4,484 dialogues, with 4,045 in the training set — a low data density relative to coverage of 60 sub-dimensions.
2. **Dependence on MMRT**: Data generation relies primarily on the automated red-teaming framework, which may introduce bias toward a narrow range of attack patterns.
3. **Limited depth of image understanding**: The 7B base model may encounter bottlenecks in fine-grained cross-modal reasoning.
4. **Unreliable analysis for high-turn dialogues**: The number of samples drops sharply beyond 6 turns, leading to high-variance evaluation results.
5. **Evaluator model bias**: Using GPT-4o/GPT-5-mini as evaluators introduces their own safety judgment biases into the data annotation process.

## Related Work & Insights

- **Black-box jailbreak attacks**: Progress from single-turn heuristic search (PAIR) → multi-turn sub-query sequences (Crescendo) → visual chain reasoning (VisualChain) → the proposed MMRT with multimodal multi-turn MCTS search.
- **Content moderation models**: Evolution from single-modal APIs (OpenAI Moderation, Perspective) → open-source models (WildGuard, LlamaGuard, LLaVaGuard, ShieldVLM) → multi-image multi-turn extensions (LlamaGuard-4) → LLaVAShield, which focuses specifically on multimodal multi-turn scenarios.
- **Safety alignment**: The paper reveals that even safety-aligned mainstream VLMs remain highly vulnerable in multimodal multi-turn settings, consistent with findings from Red Queen, IDEATOR, and related works.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of content moderation for multimodal multi-turn dialogues; the MCTS-based attack framework in MMRT and the role-decoupled rationale design are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation encompassing main experiments, external benchmarks, policy adaptation tests, ablation studies, red-teaming vulnerability analysis, and component contribution analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-articulated problem definitions and concise characterization of the three core risk properties.
- Value: ⭐⭐⭐⭐ — Fills a gap in safety moderation for multimodal multi-turn dialogues; both the dataset and methodology offer practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red_teaming_vlm_strategy_tree.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)

<!-- RELATED:END -->
