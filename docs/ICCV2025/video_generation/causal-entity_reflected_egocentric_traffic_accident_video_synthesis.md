---
title: >-
  [Paper Note] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis
description: >-
  [ICCV 2025][Video Generation][Traffic accident video synthesis] This paper proposes Causal-VidSyn, a diffusion model that achieves causal entity localization via an Accident-Reason Answering (ArA) module and a gaze-conditioned visual token selection mechanism. The authors also construct the Drive-Gaze dataset comprising 1.54 million frames of gaze data. The method outperforms state-of-the-art approaches across three tasks: accident video editing, normal-to-accident video diffusion, and text-to-video generation.
tags:
  - ICCV 2025
  - Video Generation
  - Traffic accident video synthesis
  - causal entity
  - driver gaze
  - diffusion model
  - autonomous driving safety
date: 2026-05-08
content_hash: f6303bf8afe511b1
---

# Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis

**Conference**: ICCV 2025
**arXiv**: [2506.23263](https://arxiv.org/abs/2506.23263)
**Code**: [Project Page](http://lotvsmmau.net/Causal-VidSyn)
**Area**: Video Generation
**Keywords**: Traffic accident video synthesis, causal entity, driver gaze, diffusion model, autonomous driving safety

## TL;DR
This paper proposes Causal-VidSyn, a diffusion model that achieves causal entity localization via an Accident-Reason Answering (ArA) module and a gaze-conditioned visual token selection mechanism. The authors also construct the Drive-Gaze dataset comprising 1.54 million frames of gaze data. The method outperforms state-of-the-art approaches across three tasks: accident video editing, normal-to-accident video diffusion, and text-to-video generation.

## Background & Motivation

1. **Background**: Video diffusion models (e.g., CogVideoX, StoryDiffusion) have achieved remarkable progress in general-purpose video generation, yet they are primarily designed for normal scenes and exhibit limited capability in generating egocentric traffic accident videos.
2. **Limitations of Prior Work**: State-of-the-art video diffusion models fail to accurately identify causal entities and their accident-related behaviors when generating accident videos. For instance, CogVideoX can generate a motorcycle under the text edit "pedestrian → motorcycle collision" but fails to depict the actual collision; Abductive-OAVD cannot generate the target object at all.
3. **Key Challenge**: Causal entities in accident scenes are typically small and the scene changes rapidly, making it extremely difficult to identify target objects and their subtle behaviors from an ego-vehicle perspective. Existing diffusion models lack domain knowledge to understand accident causality.
4. **Goal**: (1) How to precisely localize causal entities within video diffusion? (2) How to enable diffusion models to understand accident causality so as to faithfully respond to counterfactual text edits?
5. **Key Insight**: Two key information cues are incorporated — accident cause–collision textual descriptions (providing information on participants and improper behaviors) and driver gaze points (providing direct visual attention cues).
6. **Core Idea**: By combining accident-reason answering and gaze-conditioned token selection, the 3D-UNet backbone is made causally aware, enabling precise identification and generation of causal entities reflecting accident causality during the diffusion process.

## Method

### Overall Architecture
Causal-VidSyn is structured into two progressive levels: ❶ the diffusion recipe level (Reciprocal Prompted Frame Diffusion, RPFD) — which contrasts the diffusion processes in forward and reverse temporal frame orders; and ❷ the knowledge level (CTS + CTG modules) — which transforms the 3D-UNet into a causally aware backbone through Causal Token Selection and Causal Token Grounding.

### Key Designs

1. **Reciprocal Prompted Frame Diffusion (RPFD)**:
    - Function: Enhances causal scene learning through contrastive intervention on forward and reverse temporal orders.
    - Mechanism: The forward path pairs accident cause and collision description text $P_f$ with forward-ordered frames $V_f$; the reverse path pairs accident prevention suggestion text $P_r$ with reverse-ordered frames $V_r$. Contrastive learning on the two noise representations is formulated as: $\mathcal{L}_{ST1} = \mathcal{L}_{MSE}(e_f, \hat{e}_f) + \mathcal{L}_{MSE}(e_r, \hat{e}_r) + \lambda\mathcal{L}_{NS}(\hat{e}_f, \hat{e}_r)$.
    - Design Motivation: Reverse diffusion can be viewed as a counterfactual intervention on the forward text/visual prompts. The exogenous noise $e$ facilitates causal scene association between reciprocal frames and text prompts. Different text prompts should activate distinct visual content primarily related to the accident causal entity.

2. **Causal Tendency token Selection (CTS) + Causal Token Grounding (CTG)**:
    - Function: CTS injects causally relevant visual tokens at the intermediate layers of the 3D-UNet, while CTG performs causal token grounding at the final layer.
    - Mechanism: CTS applies attention-weighted selection using driver gaze maps over visual tokens to filter out non-causal regions. CTG designs an Accident-Reason Answering (ArA) head that retrieves the correct answer from multiple candidate accident causes and integrates it into noise representation learning to guide causal localization. During training, ArA and driver gaze are used; during inference, only video/text prompts are required.
    - Design Motivation: Drivers can acutely perceive road hazards based on driving experience, and their gaze points provide direct visual attention cues for accident regions, bridging the modality gap between textual descriptions and visual manifestations. The ArA module helps the model retrieve the correct accident cause.

3. **Drive-Gaze Dataset**:
    - Function: Provides large-scale driver gaze data in driving accident scenarios to support Causal-VidSyn training.
    - Mechanism: Based on 11,727 accident videos from MM-AU, gaze data covering 1.54 million frames from 9,727 videos are collected. Ten participants (4 female, 6 male) used a Tobii Pro Fusion eye tracker (250 Hz). Similar accident types were grouped into long videos to allow participants to accumulate experience. The final per-frame gaze map is obtained by convolving all participants' gaze points with a 50×50 Gaussian kernel.
    - Design Motivation: Existing driving gaze datasets (e.g., DADA-2000, BDD-A) are either small-scale or cover only normal driving scenarios. Drive-Gaze is the largest driver gaze dataset targeting accident scenes, and additionally includes accident cause and collision description text annotations.

### Loss & Training
Three-stage training: Stage-0 directly optimizes forward-temporal diffusion (initialized from Stable Diffusion); Stage-1 introduces RPFD contrastive learning; Stage-2 injects CTS and CTG modules for causal localization training. ArA and gaze participate only in training; inference requires only video/text inputs.

## Key Experimental Results

### Main Results

| Task | Metric | Causal-VidSyn | CogVideoX-2B | Gain |
|------|--------|---------------|--------------|------|
| Accident Video Editing (DADA-2000) | FID↓ | Best | Inferior | Significant improvement |
| Accident Video Editing | Causal Sensitivity↑ | Best | Collision not reflected | Large improvement |
| Normal→Accident Diffusion (BDD-A) | Frame Quality | Best | — | — |
| Text-to-Video Generation | FVD↓ | Best | — | — |

CTS/CTG extended to CogVideoX-2B and Latte also yield consistent and significant improvements.

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Full Causal-VidSyn | Best | Complete model |
| w/o RPFD (Stage-1) | Degraded | No forward-reverse contrastive intervention |
| w/o CTS | Degraded | No gaze-guided token selection |
| w/o CTG | Degraded | No accident-cause grounding |
| w/o Drive-Gaze | Degraded | No gaze signal |

### Key Findings
- RPFD significantly enhances activation of causal entity regions through forward-reverse temporal contrast.
- Gaze signals provide precise causal region priors during training; the model internalizes this capability without requiring gaze at inference time.
- CTS and CTG modules are transferable to Transformer-based architectures (CogVideoX, Latte), demonstrating the generality of the approach.
- The ArA module is critical for faithfulness in counterfactual editing.

## Highlights & Insights
- A distinctive perspective on introducing causal reasoning into video diffusion: rather than general causal discovery, the method leverages accident domain knowledge.
- The Drive-Gaze dataset (1.54 million gaze frames) offers lasting value and can serve various accident understanding tasks.
- The train-with-gaze / infer-without-gaze design is elegant: the model internalizes causal attention without requiring gaze at test time.
- The transferability of CTS/CTG enables plug-in integration into video diffusion models of different architectures.

## Limitations & Future Work
- Generation quality is constrained by the 3D-UNet backbone; large-scale DiT models are not employed.
- Gaze data collection is costly (10 participants over 3 months); automated alternatives warrant exploration.
- The work focuses solely on the generative side of causal awareness and does not integrate end-to-end with downstream accident understanding tasks (e.g., prediction, liability attribution).
- The utility of generated videos in actual autonomous driving testing has not been evaluated.

## Related Work & Insights
- **vs. Abductive-OAVD**: The latter focuses only on text-to-video generation and does not explore causality in video-conditioned synthesis.
- **vs. CogVideoX**: This general-purpose model underperforms on collision semantics in accident scenes; CTS/CTG can be applied as a plug-in enhancement.
- **vs. Driving Scene Video Diffusion (e.g., DriverDreamer)**: These methods primarily target multi-view consistency in accident-free scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — Unique perspective of introducing causal entity localization and driver gaze into video diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three tasks, transferability validation, and ablation studies.
- Writing Quality: ⭐⭐⭐ — Content-rich but somewhat complex in structure.
- Value: ⭐⭐⭐⭐ — The Drive-Gaze dataset and causally aware diffusion paradigm carry significant implications for autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[CVPR 2026\] Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](../../CVPR2026/video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)
- [\[NeurIPS 2025\] PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis](../../NeurIPS2025/video_generation/posecrafter_extreme_pose_estimation_with_hybrid_video_synthesis.md)
- [\[CVPR 2026\] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](../../CVPR2026/video_generation/dreamshot_storyboard_synthesis.md)

</div>

<!-- RELATED:END -->
