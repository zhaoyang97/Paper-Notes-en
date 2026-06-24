---
title: >-
  [Paper Note] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters
description: >-
  [CVPR 2025][Robotics][Social Interaction] Proposed SOLAMI, the first end-to-end Social Vision-Language-Action (VLA) modeling framework. By discretizing speech and motion into tokens and modeling them uniformly with a decoder-only LLM, it enables immersive, real-time interaction between users and 3D virtual characters using speech and body language. Additionally, a synthetic multimodal social interaction dataset, SynMSI, was constructed.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Social Interaction"
  - "VLA Model"
  - "Motion Generation"
  - "3D Characters"
  - "VR Immersion"
date: 2026-05-08
content_hash: 6c195489251d180d
---

# SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters

**Conference**: CVPR 2025  
**arXiv**: [2412.00174](https://arxiv.org/abs/2412.00174)  
**Code**: [Project Page](https://solami-ai.github.io/)  
**Area**: Human Understanding / Multimodal  
**Keywords**: Social Interaction, VLA Model, Motion Generation, 3D Characters, VR Immersion

## TL;DR

Proposed SOLAMI, the first end-to-end Social Vision-Language-Action (VLA) modeling framework. By discretizing speech and motion into tokens and modeling them uniformly with a decoder-only LLM, it enables immersive, real-time interaction between users and 3D virtual characters using speech and body language. Additionally, a synthetic multimodal social interaction dataset, SynMSI, was constructed.

## Background & Motivation

### Background

**Background**: **Demand for Immersive Character Interaction**: Psychological research shows that higher immersion in social interactions leads to a better experience. However, existing character agents are limited to textual or vocal interactions, lacking 3D body language.

### Limitations of Prior Work

**Limitations of Prior Work**: **Latency Issues in Modular Approaches**: Existing methods (such as LLM-agent frameworks) cascade sub-modules via text (motion understanding -> text -> motion generation). This approach conveys high-level information but loses subtle cues, and cascading multiple sub-modules introduces severe latency.

### Key Challenge

**Key Challenge**: **Inspiration from Robotics**: LLM-agents excel at high-level planning. However, for low-level control tasks, end-to-end VLA models perform better. Virtual characters are essentially virtual humanoid robots, making them well-suited for VLA modeling.

### Proposed Solution

**Proposed Solution**: **Challenges of Data Scarcity**: Comprehensive multimodal interaction data (conversations containing both speech and bodily motion) are extremely scarce, and the collection costs are prohibitively high.

### Additional Notes

**Additional Notes**: **Inadequacies of Single-Motion Tasks**: Existing motion-related LLM research focuses on single tasks (e.g., text-to-motion, motion understanding), making it unable to generate context-based responsive motions based on character personas.

## Method

### Overall Architecture

SOLAMI is an end-to-end VLA model: User speech and motion are discretized into tokens via a tokenizer -> a decoder-only LLM predicts character response motion and speech tokens based on user input tokens + character persona -> respective decoders reconstruct speech and motion. Training consists of three stages: Tokenizer training -> Multi-task pre-training (modality alignment) -> Instruction tuning (multi-turn conversation).

### Key Designs

**Design 1: Multi-part Split Motion Tokenizer**
- **Function**: Discretizes SMPL-X human body motions into token sequences processable by LLMs.
- **Mechanism**: Employs three independent VQ-VAEs to separately encode body motion $m^b$, hand motion $m^h$, and two-person relative transformation $m_t$: $\hat{m}_t^u = Q^u(m_t^u) = \arg\min_{z_i \in \mathbb{Z}_u} \|m_t^u - z_i\|_2$. The body and hand VQ-VAEs use 1D temporal convolutions to generate sequence tokens, while the relative transformation VQ-VAE uses an MLP to generate a single token. Speech is processed using SpeechTokenizer to separate semantic and acoustic information, feeding only semantic tokens into the LLM, and reconstructing during decoding using SoundStorm + voice cloning.
- **Design Motivation**: The physical motion characteristics of the body, hands, and relative transformations vary significantly; modeling them separately yields higher reconstruction accuracy. Using only semantic tokens reduces LLM inference costs, while supporting voice cloning ensures consistent character voice.

**Design 2: Three-Stage Progressive Training Strategy**
- **Function**: Progressively builds the character behavior system from modality alignment to multi-turn dialogue.
- **Mechanism**: Stage 1 trains and freezes the Tokenizers. Stage 2 performs multi-task pre-training, utilizing 46K motion-text pairs for text-to-motion/motion understanding, and 410K speech-text pairs for TTS/ASR, balancing scale differences with a 4:6 sampling ratio to align motion-text and speech-text. Stage 3 performs instruction tuning on 5.7K multimodal conversation data, supervising only the character's response.
- **Design Motivation**: Direct training on multimodal interaction data yields suboptimal results (proven by ablation study), making pre-training modality alignment essential. Stage-wise training allows the model to learn basic modality mappings before attempting complex social behaviors.

**Design 3: SynMSI Synthetic Data Pipeline**
- **Function**: Automatically builds large-scale multimodal social interaction datasets at a low cost.
- **Mechanism**: (1) Collects 5.3K character-related topics -> (2) GPT-4o generates textual scripts based on topics and character personas -> (3) Key text representation is used to retrieve the best-matching motion from a 46K motion database -> (4) Voice scripts are revised based on retrieved motions to ensure coordination -> TTS/voice cloning generates the character's speech. This generator process is repeated iteratively to form multi-turn dialogues, resulting in 6.3K multi-turn multimodal dialogue entries.
- **Design Motivation**: Direct collection of multimodal interaction data is extremely costly. Leveraging a retrieval-and-synthesis approach reuses existing motion datasets, while LLMs preserve dialogue diversity and character consistency.

### Loss & Training

Tokenizer training: $\mathcal{L}_m = \lambda_r \mathcal{L}_r + \lambda_e \mathcal{L}_e + \lambda_c \mathcal{L}_c + \lambda_v \mathcal{L}_v$ (reconstruction + embedding + commitment + velocity loss). Instruction tuning utilizes cross-entropy for next-token prediction, supervising only the motion and speech tokens of the character's response: $\mathcal{L}_{\text{IT}} = -\sum_{r=1}^{R}\sum_{i=1}^{L_M^r} \log p_\Theta(\hat{m}_i^r | \text{context}) - \sum_{r=1}^{R}\sum_{i=1}^{L_S^r} \log p_\Theta(\hat{s}_i^r | \text{context})$.

## Key Experimental Results

### Main Results

| Method | Motion FID ↓ | Speech Quality ↑ | Latency (s) ↓ | User Preference ↑ |
|------|------------|-----------------|--------------|-------------------|
| LLM-Agent (Modular) | Poor | Medium | High Latency | Low |
| SOLAMI (End-to-End) | **Better** | **Better** | **Lower** | **Higher** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Direct tuning without pre-training | Significant performance drop |
| Motion pre-training only | Lacks speech alignment |
| Complete three-stage | Best |
| Full fine-tuning vs LoRA | Full is slightly better but has higher costs |

### Key Findings

1. End-to-end VLA models outperform modular LLM-Agent schemes in terms of the accuracy, naturalness, and latency of speech and motion responses.
2. The multi-task pre-training stage is crucial for final performance—skipping pre-training and tuning directly leads to a massive degradation in performance.
3. The synthetic data pipeline effectively alleviates the scarcity of multimodal interaction data.
4. User studies demonstrate that the interaction experience of SOLAMI is significantly superior to baseline methods.

## Highlights & Insights

- **First Social VLA Model for 3D Characters**: Transports the VLA paradigm from robotics to virtual character interaction, opening up a new direction.
- **Modeling "Motion as Language"**: Unifies SMPL-X motion and speech into token sequences, allowing the LLM to act as a unified behavioral reasoning engine.
- **Highly Practical Data Synthesis Pipeline**: Efficiently constructs structured, albeit synthetic, multimodal interaction data at low cost using a retrieval-and-LLM-generation pipeline.
- **VR Interface Design**: Allows end-to-end evaluation of the method, bridging the gap between virtual character research and real-world experiences.

## Limitations & Future Work

- SynMSI data is still synthetic, presenting distribution differences compared to real human-to-human interactions.
- The upper bound of motion retrieval is constrained by the coverage of existing motion databases.
- Facial expression modeling has not yet been incorporated (only covering body and hand motions).
- Multi-person simultaneous interaction scenarios are not yet supported.

## Related Work & Insights

- SOLAMI's end-to-end VLA paradigm can be extended to more virtual human applications (game NPCs, virtual teachers, etc.).
- The part-based design of the motion tokenizer can provide a reference for other motion generation/understanding tasks.
- The "retrieval + correction" mode of the data synthesis pipeline can be generalized to other user-related tasks lacking paired data.

## Rating

⭐⭐⭐⭐ — Pioneeringly introduces the VLA framework to 3D virtual character social interaction, presenting clear problem definition, complete system design, and thorough experimentation. The SynMSI data synthesis pipeline and VR evaluation interface both offer independent contribution value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models](../../ICLR2026/robotics/villa-x_enhancing_latent_action_modeling_in_vision-language-action_models.md)
- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[ICLR 2026\] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](../../ICLR2026/robotics/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

</div>

<!-- RELATED:END -->
