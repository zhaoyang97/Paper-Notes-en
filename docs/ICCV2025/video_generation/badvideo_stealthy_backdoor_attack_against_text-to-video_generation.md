---
title: >-
  [Paper Note] BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation
description: >-
  [ICCV 2025][Video Generation][text-to-video generation] BadVideo is the first backdoor attack framework targeting text-to-video (T2V) generation models. It exploits inherent static and dynamic redundancy in video (e.g., unspecified background elements, motion trajectories) through two strategies—spatio-temporal composition and dynamic element transition—to covertly embed malicious content. The framework achieves up to 93.5% human-evaluated attack success rate on LaVie and Open-Sora while effectively evading existing content moderation systems.
tags:
  - ICCV 2025
  - Video Generation
  - text-to-video generation
  - backdoor attack
  - redundancy exploitation
  - spatio-temporal composition
  - content safety
date: 2026-05-08
content_hash: f7416bad38755814
---

# BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation

**Conference**: ICCV 2025
**arXiv**: [2504.16907](https://arxiv.org/abs/2504.16907)
**Code**: [https://wrt2000.github.io/BadVideo2025/](https://wrt2000.github.io/BadVideo2025/) (project page)
**Area**: Image/Video Generation Security
**Keywords**: text-to-video generation, backdoor attack, redundancy exploitation, spatio-temporal composition, content safety

## TL;DR
BadVideo is the first backdoor attack framework targeting text-to-video (T2V) generation models. It exploits inherent static and dynamic redundancy in video (e.g., unspecified background elements, motion trajectories) through two strategies—spatio-temporal composition and dynamic element transition—to covertly embed malicious content. The framework achieves up to 93.5% human-evaluated attack success rate on LaVie and Open-Sora while effectively evading existing content moderation systems.

## Background & Motivation
T2V generation models have advanced rapidly and are widely deployed in entertainment, education, and marketing, yet their security risks remain underexplored. A natural information granularity gap exists between text (abstract and sparse) and video (visually dense and temporally continuous), forcing models to "hallucinate" large amounts of content not specified by the text prompt.

This redundant information falls into two categories:

**Static redundancy**: Excess spatial elements within a single frame (e.g., background objects, over-rendered details)

**Dynamic redundancy**: Excess temporal transitions (e.g., unspecified motion sequences, scene evolution)

When exploited by malicious actors, such redundancy can be weaponized to inject pornographic, violent, hateful, or disinformation content. Critically, existing content moderation systems (e.g., Sora's per-frame detection) analyze only single-frame spatial information and cannot capture cross-frame temporal malicious content.

This work fills the gap in backdoor attacks on T2V generation models—prior research addressed only image generation, and the temporal dimension of video generation introduces an entirely new attack surface.

## Method

### Overall Architecture
The BadVideo attack framework consists of three stages: (1) poisoned dataset construction—inserting triggers into text prompts and embedding malicious targets into videos; (2) fine-tuning—using the poisoned dataset to fine-tune a pretrained T2V model; (3) inference—the attacker activates the backdoor via trigger-containing text. The core innovation lies in the target video generation method.

### Key Designs

1. **Strategy 1: Spatio-Temporal Composition (STC)**:

    - Function: Decomposes malicious content along the temporal dimension and distributes it across redundant elements in different frames
    - Mechanism: Any individual frame appears benign, but when viewed continuously, the malicious information naturally integrates in the viewer's perception. For example, "FU" and "CK" are placed in separate frames—each frame is innocuous, but sequential viewing forms an offensive word
    - Design Motivation: Exploits the temporal integration property of human vision to bypass per-frame detection

2. **Strategy 2: Semantic Concept Transition (SCT)**:

    - Function: Introduces temporal transitions of semantic concepts onto video redundancy elements (e.g., background billboards) to convey malicious information
    - Mechanism: The video subject may be "a person filling strawberries," but the background billboard gradually transitions from neutral political content to insulting content, exploiting both static (billboard presence) and dynamic (content change) redundancy
    - Design Motivation: User prompts cannot fully specify the transition paths of all redundant elements

3. **Strategy 3: Visual Style Transition (VST)**:

    - Function: Manipulates the aesthetic and atmospheric evolution of the video to covertly embed malicious content
    - Mechanism: Even with a neutral text prompt, redundant style information is manipulated into disturbing background elements through systematic aesthetic degradation—e.g., a peaceful scene devolving into post-war ruins, or natural scenery transforming into a polluted wasteland
    - Design Motivation: The aesthetic evolution trajectory of a video is rarely specified by user prompts, providing a natural attack surface

4. **Target Video Generation Pipeline**:

    - **Prompt Transformation Module**: Uses an LLM to convert the original text prompt into head and tail prompts describing the initial and final states of the malicious target
    - **Keyframe Generation Module**: Generates a head frame from the head prompt using a T2I model, then produces a visually consistent tail frame via image editing
    - **Target Video Generation Module**: Encodes the head and tail frames via VAE, concatenates them as conditional inputs to the diffusion model, and generates a temporally coherent target video

### Loss & Training
The fine-tuning process uses the standard diffusion model reconstruction loss:
$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, c, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \mathcal{T}_\theta(c), t)\|_2^2\right]$$
The text encoder is kept frozen; only diffusion model parameters are updated. The poisoning ratio is set to 20%, and fine-tuning runs for 200 epochs.

## Key Experimental Results

### Main Results

| Model | Strategy | FVD↓ | CLIPSIM↑ | ASR_MLLM(%) | ASR_Human(%) | CPR(%) |
|-------|----------|------|----------|-------------|--------------|--------|
| LaVie | Fine-tuned | 327.39 | 0.2883 | 0.0 | 0.0 | 78.5 |
| LaVie | STC | 352.90 | 0.2847 | 84.3 | **92.3** | 74.2 |
| LaVie | SCT | 342.04 | 0.2871 | 86.5 | 91.6 | 72.8 |
| LaVie | VST | 320.36 | 0.2858 | 88.2 | 90.2 | 76.4 |
| Open-Sora | Fine-tuned | 310.77 | 0.2957 | 0.0 | 0.0 | 89.6 |
| Open-Sora | STC | 355.04 | 0.2918 | 80.5 | 79.5 | 72.6 |
| Open-Sora | VST | 312.31 | 0.2940 | **96.4** | **93.5** | 74.9 |

### Ablation Study

| Experiment | Key Metric | Notes |
|------------|-----------|-------|
| Poisoning ratio 5% → 30% | ASR rises from ~40% to >80% | 20% already achieves 80%+ ASR |
| Training for 80 epochs | ASR >70% | All strategies reach high ASR after 80 epochs |
| Fine-tuning defense for 100 epochs | ASR remains >80% | Backdoor pattern is strongly memorized |
| Prompt perturbation 80% | ASR remains effective but CPR drops sharply | Defenders face a dilemma |
| Multi-backdoor (3 targets) | All ASRs remain high | Multiple backdoors can be implanted simultaneously |
| GPT-4o detection (16 frames) | Success rate only 52% | Even with explicit temporal detection prompting |
| Omni-Moderation | 0% detection rate | Safety models lack temporal threat categorization |

### Key Findings
- VST achieves 96.4% MLLM ASR and 93.5% human ASR on Open-Sora, representing the strongest performance
- Backdoors are highly robust against fine-tuning defenses—ASR remains >80% after 100 epochs of fine-tuning on 10% clean data
- Existing safety detection systems (including Omni-Moderation and Llama-Guard-3) completely fail to detect temporally distributed malicious content
- Even when GPT-4o is explicitly informed that "malicious information may be distributed across frames," its detection rate is only 52% on 16-frame inputs, with computational cost scaling linearly with frame count
- The total attack cost is approximately \$6.33 (~2.11 hours on an A800 GPU), representing an extremely low barrier to entry

## Highlights & Insights
- First work to reveal the weaponization risk of video redundancy in T2V generation, opening an entirely new attack surface
- The spatio-temporal composition strategy cleverly exploits the asymmetry between human temporal visual integration and machine per-frame analysis
- The attack cost is minimal (\$6.33) yet highly destructive, underscoring the urgency of T2V content auditing
- Confirms a fundamental blind spot in existing content moderation systems along the temporal dimension

## Limitations & Future Work
- A poisoning ratio of 20% is relatively high (clean-label attacks typically require <1%), making it difficult to achieve in real-world scenarios
- The definition of "maliciousness" for the three strategies is somewhat subjective and may vary across cultural contexts
- The target video generation pipeline depends on both a T2I model and an LLM, adding implementation complexity
- Evaluation is limited to LaVie and Open-Sora; generalization to stronger commercial models (e.g., Sora, Kling) remains unverified
- Effective defense mechanisms are not explored in depth—the paper demonstrates that existing methods are insufficient but does not propose alternatives

## Related Work & Insights
- **vs. T2I backdoor attacks (Chou et al., Chen et al.)**: T2I attacks target specific images or categories and are susceptible to semantic consistency checks; BadVideo exploits the temporal dimension of video for stealthier attacks
- **vs. DDPM/DDIM backdoors**: Early unconditional generation backdoors are triggered via initial noise; BadVideo operates under text conditioning, making it more practical
- **vs. Struppek et al.**: That work focuses on text encoder backdoors, whereas BadVideo is an end-to-end backdoor for video generation

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First backdoor attack on T2V generation; the spatio-temporal redundancy exploitation perspective is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-strategy validation with defense evaluation and adaptive defense analysis
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, strategy taxonomy is well-motivated, and figures are intuitive
- Value: ⭐⭐⭐⭐⭐ Reveals an important and previously overlooked security risk in T2V models with meaningful practical implications for the industry

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)

</div>

<!-- RELATED:END -->
