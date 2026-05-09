---
title: >-
  [Paper Note] TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment
description: >-
  [AAAI 2026][LLM/NLP][Video LLM] This paper proposes TEMPLE, which significantly enhances the temporal reasoning capabilities of Video LLMs through an automated video temporal preference data generation pipeline (video filtering → temporal perturbation → contrastive response generation) and a novel Progressive Pre-SFT Alignment strategy (curriculum learning + DPO prior to SFT), using a small amount of self-generated DPO data. Consistent improvements are achieved across multiple benchmarks including VideoMME, MLVU, and Vinoground.
tags:
  - AAAI 2026
  - LLM/NLP
  - Video LLM
  - temporal understanding
  - DPO
  - preference learning
  - curriculum learning
date: 2026-05-08
content_hash: fa38b9146dab654a
---

# TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment

**Conference**: AAAI 2026
**arXiv**: [2503.16929](https://arxiv.org/abs/2503.16929)
**Code**: [GitHub](https://github.com/lscpku/TEMPLE)
**Area**: LLM/NLP
**Keywords**: Video LLM, temporal understanding, DPO, preference learning, curriculum learning

## TL;DR
This paper proposes TEMPLE, which significantly enhances the temporal reasoning capabilities of Video LLMs through an automated video temporal preference data generation pipeline (video filtering → temporal perturbation → contrastive response generation) and a novel Progressive Pre-SFT Alignment strategy (curriculum learning + DPO prior to SFT), using a small amount of self-generated DPO data. Consistent improvements are achieved across multiple benchmarks including VideoMME, MLVU, and Vinoground.

## Background & Motivation

**Background**: Video LLMs have achieved remarkable success under the large-scale pretraining + SFT paradigm, yet continue to underperform on temporal reasoning. Preliminary analysis reveals that Qwen2-VL-7B achieves only 74.4% precision and 35.0% recall in detailed captioning, missing numerous events and introducing hallucinations.

**Limitations of Prior Work**: (a) Temporal information is scarce in video datasets — many videos are relatively static, with weak temporal correspondence in video-text pairs; (b) both pretraining and SFT rely on next-token prediction, which does not explicitly enforce dynamic temporal understanding; (c) models frequently overlook subtle temporal details and over-rely on local visual/textual cues.

**Key Challenge**: High-quality temporal annotations for video are prohibitively expensive, yet SFT requires near-perfect labels; DPO requires only relative comparisons and is thus better suited for video tasks.

**Key Insight**: (a) Temporal preference pairs are automatically generated via video perturbation (frame dropping, shuffling, and temporal reversal), eliminating the need for manual annotation; (b) DPO is placed before rather than after SFT (reversing the conventional order), establishing temporal understanding prior to instruction-following learning.

**Core Idea**: A three-stage automated pipeline for temporal preference data generation, combined with Progressive Pre-SFT Alignment (easy-to-hard curriculum learning with DPO preceding SFT).

## Method

### Overall Architecture
Two main components: (1) an automated preference data generation pipeline; and (2) a Progressive Pre-SFT Alignment training strategy.

### Key Designs

1. **Video Preference Data Generation Pipeline**:

    - **Video Filtering** (3 steps): TransNetV2 scene detection to select videos with sufficient temporal complexity → SigLIP similarity-based grouping to remove redundant scenes → Laplacian operator to extract sharp keyframes. Videos with 4–32 distinct scene groups are retained.
    - **Temporal Perturbation Strategies** (3 types):
        - Random clip dropping: removes certain clips, forcing the model to generate responses under missing information.
        - Random clip shuffling: reorders clips, preserving visual content while disrupting temporal structure.
        - Clip temporal reversal: reverses the temporal order within clips.
    - **Contrastive Response Generation**: responses are generated separately for clean and perturbed inputs; clean responses serve as chosen and perturbed responses as rejected.
    - A difficulty factor $r$ controls perturbation intensity: larger $r$ yields stronger perturbation.

2. **Progressive Pre-SFT Alignment**:

    - **Curriculum Learning**: $r$ is progressively decreased during training (easy to hard), allowing the model to first learn to distinguish clear differences before handling subtle ones.
    - **Pre-SFT Alignment** (core innovation): reverses the conventional "SFT then DPO" order by performing DPO first to establish temporal alignment, followed by SFT.
    - Rationale for DPO-first: the primary bottleneck of Video LLMs is their inability to establish precise correspondence between visual inputs and textual descriptions — this foundational issue should be addressed before learning general instruction-following.
    - Empirical evidence: models after Pre-SFT Alignment exhibit lower loss and more stable gradients during subsequent SFT.

3. **Self-Contained Design with No External Dependencies**:

    - Video filtering relies entirely on open-source tools without external LLMs (e.g., GPT-4).
    - Preference data are self-generated by the model itself, requiring no human annotation.
    - This contrasts with methods such as Tarsier2/TPO that depend on commercial APIs.

## Key Experimental Results

### Main Results

| Model | VideoMME | MLVU | Vinoground_txt |
|-------|----------|------|----------------|
| Qwen2-VL-7B (baseline) | Low | Low | Low |
| + TEMPLE | Significant gain | Significant gain | Significant gain |
| GPT-4o | 77.2 | 64.6 | 54.0 |

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| SFT only (no DPO) | Baseline |
| SFT then DPO (conventional order) | Improved but below Pre-SFT |
| **DPO then SFT (Pre-SFT)** | **Best** |
| No curriculum learning (fixed difficulty) | Below progressive setting |

### Key Findings
- **Pre-SFT Alignment outperforms conventional Post-SFT DPO**: establishing temporal understanding prior to instruction-following learning yields better results, evidenced by lower loss and more stable gradients during the SFT stage.
- **Curriculum learning is effective**: progressive easy-to-hard training is more efficient than fixed-difficulty training.
- **Significant improvements can be achieved with limited DPO data**: large-scale annotation is unnecessary, and the self-contained nature of the pipeline is critical.
- **DPO data transfers across different model architectures and scales**: suggesting that the learned temporal alignment is generalizable.

## Highlights & Insights
- The reversed "DPO before SFT" strategy is a highly instructive finding. The conventional view holds that basic capabilities should be acquired before alignment; however, for tasks requiring precise perception such as video temporal understanding, it is more effective to first establish a correct perceptual foundation through preference learning. This insight may generalize to other multimodal tasks that require "learning to perceive before learning to describe."
- The three temporal perturbation strategies (frame dropping, shuffling, and reversal) are precisely designed to target known weaknesses of Video LLMs and are uniformly controlled via the difficulty factor, making the overall design concise and elegant.

## Limitations & Future Work
- Only detailed captioning is used as the DPO task; the approach has not been extended to other video understanding tasks such as QA or temporal grounding.
- The perturbation strategies are relatively coarse (clip-level rather than frame-level) and may fail to capture finer-grained temporal dependencies.
- The video filtering pipeline is dependent on the quality of the scene detection tool (TransNetV2).
- The Pre-SFT strategy introduces an additional training stage; total training cost is not reported.

## Related Work & Insights
- **vs. POVID**: POVID inspired the idea of generating preference pairs via perturbation, but operates at the image level; TEMPLE extends this to the temporal dimension of video and incorporates curriculum learning.
- **vs. Tarsier2/TPO**: these methods rely on commercial LLM APIs for data filtering and apply DPO after SFT; TEMPLE is fully self-contained and adopts the Pre-SFT strategy.
- **vs. standard DPO/RLHF**: the conventional paradigm applies SFT before DPO; TEMPLE reverses this order and demonstrates its superiority for video tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The Pre-SFT Alignment reverse-order strategy is novel, and the automated preference generation pipeline is well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple benchmarks with ablation studies covering both training strategies and data design.
- Writing Quality: ⭐⭐⭐⭐ The method is described in detail, with clear motivation provided for each step of the pipeline.
- Value: ⭐⭐⭐⭐ Provides a scalable alignment solution for temporal understanding in Video LLMs.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related research areas.
- Future work may validate the generalizability and scalability of the approach across more scenarios and at larger scales.
- There is potential research value in combining this work with recent related approaches (e.g., intersections with RL/MCTS and multimodal methods).
- Practical deployment feasibility and computational efficiency should be assessed in light of real-world application requirements.
- The choice of datasets and evaluation metrics may affect the generalizability of the conclusions; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related research areas.
- Future work may validate the generalizability and scalability of the approach across more scenarios and at larger scales.
- There is potential research value in combining this work with recent related approaches (e.g., intersections with RL/MCTS and multimodal methods).

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)
- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[AAAI 2026\] ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](../../ACL2026/llm_nlp/it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video Models](../../CVPR2026/llm_nlp/physvid_physics_aware_local_conditioning_for_generative_video_models.md)

<!-- RELATED:END -->
