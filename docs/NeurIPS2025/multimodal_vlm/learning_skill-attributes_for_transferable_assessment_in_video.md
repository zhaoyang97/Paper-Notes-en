---
title: >-
  [Paper Note] Learning Skill-Attributes for Transferable Assessment in Video
description: >-
  [NeurIPS 2025][Multimodal VLM][Skill assessment] This paper proposes CrossTrainer, a method that discovers sport-agnostic skill attributes (e.g., balance, control, hand positioning) as intermediate representations to train a multimodal language model for generating actionable feedback and proficiency assessments from video. CrossTrainer achieves up to 60% relative improvement over the state of the art in zero-shot cross-sport transfer.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Skill assessment
  - cross-sport transfer
  - video understanding
  - multimodal LLM
  - actionable feedback
date: 2026-05-08
content_hash: e807fa2855353f75
---

# Learning Skill-Attributes for Transferable Assessment in Video

**Conference**: NeurIPS 2025
**arXiv**: [2511.13993](https://arxiv.org/abs/2511.13993)
**Code**: [https://vision.cs.utexas.edu/projects/CrossTrainer/](https://vision.cs.utexas.edu/projects/CrossTrainer/)
**Area**: Multimodal VLM
**Keywords**: Skill assessment, cross-sport transfer, video understanding, multimodal LLM, actionable feedback

## TL;DR
This paper proposes CrossTrainer, a method that discovers sport-agnostic skill attributes (e.g., balance, control, hand positioning) as intermediate representations to train a multimodal language model for generating actionable feedback and proficiency assessments from video. CrossTrainer achieves up to 60% relative improvement over the state of the art in zero-shot cross-sport transfer.

## Background & Motivation

**Background**: Video-based skill assessment aims to score athletic performance and identify areas for improvement. Existing methods (e.g., ExpertAF, Stream-VLM) are trained and evaluated within a single sport or action category and rely heavily on expert-level annotations.

**Limitations of Prior Work**: (a) While approximately 8,000 sports exist globally, only a small fraction have sufficient annotated data — long-tail sports are severely underrepresented; (b) expert annotation is expensive and does not scale; (c) all existing methods assume that training and testing occur within the same sport, precluding cross-sport generalization.

**Key Challenge**: Conventional action understanding seeks invariance to execution differences (recognizing *what* is being done), whereas skill assessment must be sensitive to exactly those differences (capturing *how* it is done) — yet the evaluation dimensions of different sports appear superficially incompatible.

**Goal**: To construct a sport-agnostic video representation that enables a model trained on data-rich sports to transfer to novel sports in a zero-shot setting.

**Key Insight**: Cognitive science research demonstrates that motor skills transfer across sports (e.g., basketball players make better decisions in soccer than tennis players do), implying the existence of shared underlying skill dimensions. This work is the first to translate that intuition into a functioning video model.

**Core Idea**: Learn a set of sport-agnostic *skill attributes* (e.g., balance, control, coordination) as intermediate representations, decomposing skill assessment into sport-generic and sport-specific components.

## Method

### Overall Architecture
CrossTrainer is a two-stage multimodal language model system. Given a video $V$ as input, the system produces three outputs: (1) a set of skill attributes $\hat{S}$ indicating underperforming dimensions; (2) actionable feedback text $T$ providing specific improvement suggestions; and (3) a proficiency estimate $P$ classified into four levels ranging from novice to late-expert.

### Key Designs

1. **Skill-Attribute Discovery (Stage I: Discovering Skill-Attributes)**:

    - **Function**: Automatically extracts sport-agnostic skill attributes from expert commentaries in existing video datasets as pre-training supervision signals.
    - **Mechanism**: For the expert commentary text $T$ of each training sample, an LLM (GPT-4o) is used to extract the skill attributes to be improved, $S = \{s_1, s_2, \ldots\}$. These attributes are open-vocabulary phrases (e.g., *body positioning*, *balance*, *control*) rather than a closed label set.
    - **Design Motivation**: Training directly on expert commentaries couples the model to sport-specific phrasings; by abstracting to skill attributes, concepts such as "lack of control" can be shared across soccer and rock climbing, enabling transfer.

2. **Video Encoding and Multimodal LLM Pre-training (Stage II: Skill Assessment)**:

    - **Function**: Encodes video into tokens for a multimodal LLM, which is pre-trained to generate skill attributes.
    - **Mechanism**: A frozen video encoder $f_v$ (EgoVLPv2/CLIP) extracts one feature per second, $\mathbf{v}' = f_v(V)$; a trainable mapper $f_m$ (two-layer MLP with GELU activation) projects video features into the LLM token space, $\mathbf{v} = f_m(\mathbf{v}')$; a multimodal LLM $\mathcal{L}$ (Llama-3.1-8B-Instruct, fine-tuned with LoRA) receives video tokens and a prompt to generate the skill attribute set.
    - **Pre-training objective**: $\mathcal{F}_a(V | \mathcal{D}_{tr}) = \hat{S}$, trained with standard negative log-likelihood loss.

3. **Conditioned Feedback and Proficiency Assessment**:

    - **Function**: Generates actionable feedback and estimates proficiency level conditioned on the predicted skill attributes.
    - **Actionable Feedback**: $\mathcal{F}_t(V, \hat{S} | \mathcal{D}_{tr}) = T$; both the video and the predicted skill attributes are provided in the prompt, guiding the model to produce sport-specific improvement suggestions (e.g., "bend lower when dribbling to maintain control").
    - **Proficiency Assessment**: $\mathcal{F}_p(V, \hat{S} | \mathcal{D}_{tr}) = P$; a linear probe $f_p$ classifies the frozen video representation $\mathbf{v}$ into four proficiency levels (novice / intermediate / early-expert / late-expert).
    - **Design Motivation**: Skill attributes serve as intermediate representations that enable a key decoupling — generic attributes are shared across sports, while feedback text is generated in a sport-specific manner.

### Loss & Training
- LoRA fine-tuning (rank 128, alpha 256, dropout 0.05) for parameter-efficient adaptation.
- Learning rates: $2 \times 10^{-3}$ for the mapper $f_m$; $2 \times 10^{-4}$ for the LLM $\mathcal{L}$.
- The video encoder is frozen; only the mapper and LoRA parameters are trained.
- Training runs for 2 epochs or until convergence, requiring 1–3 hours on a single GH200 GPU.

## Key Experimental Results

### Main Results

Evaluated on three datasets: Ego-Exo4D (soccer/basketball/rock climbing), QEVD (23 fitness actions), and in-the-wild YouTube videos.

**Skill Attribute Generation (IoU@0.7)**:

| Method | Ego-Exo4D | QEVD |
|--------|-----------|------|
| InternVideo2-FT | 15.0 | 24.5 |
| LLaVA-FT | 14.6 | 26.9 |
| ExpertAF (SOTA) | 15.0 | 28.1 |
| Attribute-Retrieval | 19.7 | 32.4 |
| **CrossTrainer** | **25.7** | **37.6** |

**Actionable Feedback Generation (Ego-Exo4D)**:

| Method | BLEU@4 | METEOR | ROUGE-L |
|--------|--------|--------|---------|
| LLaVA-FT | 43.5 | 48.5 | 51.5 |
| ExpertAF (SOTA) | 44.9 | 49.6 | 54.6 |
| **CrossTrainer** | **45.6** | **51.7** | **57.8** |
| w/o two-stage | 43.8 | 48.8 | 52.3 |

### Ablation Study

| Configuration | METEOR (EgoExo) | ROUGE-L (EgoExo) | Notes |
|---------------|-----------------|-------------------|-------|
| CrossTrainer (full) | 51.7 | 57.8 | Two-stage training + skill-attribute conditioning |
| w/o two-stage | 48.8 | 52.3 | Skill-attribute pre-training removed; direct end-to-end training |
| Performance drop | −2.9 | −5.5 | Skill-attribute pre-training contributes substantially |

**Proficiency Estimation Accuracy (Ego-Exo4D)**:

| Method | Basketball | Soccer | Rock Climbing |
|--------|-----------|--------|---------------|
| EgoVLPv2 | 48.0 | 62.5 | 34.0 |
| **CrossTrainer** | **53.1** | **68.8** | **37.1** |

### Key Findings
- Skill-attribute pre-training is the central contribution: removing it causes ROUGE-L for feedback generation to drop from 57.8 to 52.3.
- CrossTrainer degrades gracefully under zero-shot transfer: from fully supervised to the most challenging cross-sport zero-shot setting (ZS-3), the maximum performance drop is only 4%, compared to 17% for baseline methods.
- Transfer between soccer and basketball is more effective than transfer involving rock climbing, consistent with cognitive science findings on motor skill transfer.
- In in-the-wild YouTube evaluation, a model trained on soccer, basketball, and rock climbing correctly identifies issues in ultimate frisbee and water polo; 75% of generated feedback is judged by human evaluators to be accurate and actionable.

## Highlights & Insights
- **Cognitive science → model design**: This work is the first to translate cognitive science findings on cross-sport motor skill transfer into a functioning computational model. The use of skill attributes as an intermediate representation layer is a generalizable design principle applicable to other cross-domain assessment tasks.
- **Elegance of assessment decoupling**: Decomposing skill assessment into sport-agnostic dimension identification and sport-specific feedback generation mirrors the cognitive process of human coaches, and this decomposition strategy is transferable to other domains (e.g., programming skill assessment).
- **Generative vs. retrieval-based skill attributes**: The generative approach outperforms retrieval-based attribute selection (Attribute-Retrieval) by 6%, demonstrating that open-vocabulary attribute generation is more flexible than closed-set retrieval.

## Limitations & Future Work
- Only RGB frame-level features are used; human body pose is not explicitly modeled (the authors note that additional pose extraction incurs prohibitive computational overhead).
- Training data are limited in scope (Ego-Exo4D covers only 3 sports with 289 participants); extending to a broader range of sports would further validate generalizability.
- Skill attribute extraction relies on GPT-4o, introducing an additional dependency and potential annotation bias.
- The current work focuses exclusively on individual skill; transfer to multi-person team interaction scenarios remains unexplored.

## Related Work & Insights
- **vs. ExpertAF**: Both systems generate actionable feedback, but ExpertAF lacks cross-sport transfer capability — all training and testing occur within the same sport. CrossTrainer achieves transfer through skill-attribute decoupling.
- **vs. InternVideo2**: A strong video representation model that captures action semantics rather than execution quality, and thus performs substantially worse on skill assessment.
- **vs. Zero-shot attribute methods (e.g., CLIP)**: Traditional zero-shot attribute methods focus on *what* (semantic categories); this work is the first to introduce attributes focused on *how* (execution quality).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to translate cognitive science theory on cross-sport motor skill transfer into a video model; the skill-attribute intermediate representation is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, six sports/fitness activities, four zero-shot settings, in-the-wild YouTube testing, and human evaluation provide comprehensive validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, methodology is intuitively described, experimental design is rigorous, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the core bottleneck in skill assessment — annotation scarcity and long-tail sports — with direct commercialization potential (AI coaching).

## Related Papers

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[NeurIPS 2025\] Evolutionary Learning in Spatial Agent-Based Models for Physical Climate Risk Assessment](evolutionary_learning_in_spatial_agent-based_models_for_physical_climate_risk_as.md)
- [\[CVPR 2025\] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill](../../CVPR2025/human_understanding/grove_a_generalized_reward_for_learning_open-vocabulary_physical_skill.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](../../CVPR2026/human_understanding/rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/human_understanding/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/multimodal_vlm/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)

<!-- RELATED:END -->
