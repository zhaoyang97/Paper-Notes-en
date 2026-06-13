---
title: >-
  [Paper Note] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments
description: >-
  [ICML 2026][Audio & Speech][Spatial Audio] JAEGER develops an end-to-end 3D audio-visual large model by adapting Qwen2.5-Omni via LoRA. By integrating RGB-D depth positional encoding…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Spatial Audio"
  - "FOA"
  - "RGB-D"
  - "3D Visual Grounding"
  - "Audio-Visual LLM"
date: 2026-05-08
content_hash: 802c95031e2b1318
---

# JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments

**Conference**: ICML 2026  
**arXiv**: [2602.18527](https://arxiv.org/abs/2602.18527)  
**Code**: https://github.com/liuzhan22/JAEGER  
**Area**: Multimodal VLM / Audio & Speech / 3D Vision  
**Keywords**: Spatial Audio, FOA, RGB-D, 3D Visual Grounding, Audio-Visual LLM

## TL;DR
JAEGER develops an end-to-end 3D audio-visual large model by adapting Qwen2.5-Omni via LoRA. By integrating RGB-D depth positional encoding, First-Order Ambisonics (FOA) dual-path audio, and the newly proposed Neural Intensity Vector, it extends traditional AV-LLMs from "2D RGB + Mono" to "3D Geometry + Multi-channel Spatial Audio," accompanied by the release of the SpatialSceneQA simulation benchmark with 61k samples.

## Background & Motivation

**Background**: Current mainstream audio-visual large models (AV-LLMs) such as Qwen2.5-Omni and VideoLLaMA 2 are almost entirely based on a 2D setting of "RGB video + mono audio," leaving spatial structure and directional acoustics as purely implicit information. While 3D visual grounding has recently become popular, most works handle the visual side (point clouds, RGB-D + 3D positional encoding) or the audio side (binaural encoders, intensity vectors) separately, lacking a unified paradigm.

**Limitations of Prior Work**: First, **modality dimension mismatch**—mono audio cannot perform sound source localization in principle, and RGB video lacks scale information to regress 3D boxes, with each modality missing one essential dimension. Second, existing cross-modal attempts either assume single-source scenarios with only RGB panoramas (e.g., Hear You Are), failing to test robustness against overlapping sources and depth-aware grounding, or rely on cascaded pipelines with traditional signal processing for DoA (e.g., SAVVY), which blocks end-to-end learning. Third, availability of data is scarce; real-world multi-channel datasets like STARSS23 lack aligned depth information.

**Key Challenge**: To achieve true 3D physical reasoning, one must possess both **metric-level geometry** (depth + camera intrinsic/extrinsic parameters) and **directional acoustics** (multi-channel spatial audio). Standard STFT-based intensity vectors degrade under strong reverberation and overlapping sources, while traditional geometric branches depend on external 3D segmenters; neither approach is end-to-end learnable.

**Goal**: (i) Perform end-to-end DoA estimation, 3D box grounding, and multi-speaker audio-visual matching within a unified AV-LLM framework; (ii) Design a spatial audio representation that remains robust under reverberation and overlapping conditions; (iii) Provide a large-scale simulation dataset with degree-level azimuth and elevation ground truth.

**Key Insight**: Simulation pipelines such as Habitat-Sim + SoundSpaces 2.0 + Hunyuan3D-1.0 are sufficiently mature to render synchronized RGB-D + FOA + precise 3D ground truth. The physical form of Classical IV, $I'_C = F_W^* \odot F_C$, can be generalized to the latent space, allowing the neural network to learn a "Neural Intensity Vector" that is more robust than STFT.

**Core Idea**: By utilizing "Neural IV (a learnable FOA intensity vector encoded by CNN) + 3D sinusoidal positional encoding from depth back-projection," JAEGER elevates the AV-LLM from 2D to 3D for joint end-to-end training.

## Method

### Overall Architecture
JAEGER takes synchronized RGB-D frames + 4-channel FOA audio (including W/X/Y/Z channels) as input and outputs natural language + structured 3D information (azimuth/elevation angles, 3D bbox `bbox(c, x, y, z, sx, sy, sz)`, and multi-speaker matching labels Left/Center/Right). The architecture follows a "Visual Stream + Audio Stream → MLP Projection → LLM (initialized with Qwen2.5-Omni + LoRA r=64)" pipeline. The Visual Stream performs element-wise addition of RGB semantic tokens and 3D sinusoidal positional encoding back-projected from depth. The Audio Stream is dual-path: the W-channel extracts semantic content, while the X/Y/Z channels extract spatial directional cues via Classical IV or Neural IV. Both streams are aligned by an MLP adapter with the visual tokens and fed into the LLM. Task-specific selective fine-tuning is employed: training only the audio side for DoA, only the visual side for grounding, or all modality projections + LoRA for joint reasoning.

### Key Designs

1. **Neural Intensity Vector (Neural IV, Core Novelty)**:
    - **Function**: Learns a spatial direction representation from raw FOA waveforms end-to-end that is more robust than STFT-based Classical IV, specifically for handling strong reverberation and overlapping sound sources.
    - **Mechanism**: First, a data2vec-style 7-layer 1D-CNN (kernel `(10,3,3,3,3,2,2)`, stride `(5,2,2,2,2,2,2)`, 50 Hz frame rate) encodes each FOA channel into latents, resulting in $f_W$ (omnidirectional) and $f_C, C \in \{X,Y,Z\}$ (directional). The physical formula of Classical IV $I'_C = F_W^* \odot F_C$ is then lifted to the latent space, defining $h_C = f_W \odot f_C$. These are concatenated and passed through a two-layer MLP: $\mathbf{v}_{\text{NIV}} = \text{Linear}(\text{ReLU}(\text{Linear}(\text{Concat}(h_X, h_Y, h_Z))))$.
    - **Design Motivation**: Classical IV relies on fixed STFT, where reverberation and overlapping sources amplify noise in the cross-spectrum $F_W^* \odot F_C$. Replacing STFT with a learnable CNN preserves the physically correct "omni × directional element-wise product" intensity vector structure while allowing the network to learn more stable directional embeddings in a data-driven manner.

2. **3D-aware Visual Encoding (Depth Back-projection + 3D Sinusoidal Position Encoding)**:
    - **Function**: Explicitly grounds RGB visual tokens in metric 3D space, enabling the LLM to directly output metric-level 3D bboxes.
    - **Mechanism**: Camera intrinsics are used to back-project each pixel $(u,v)$ with its depth $D_{uv}$ into a metric 3D point $P_{uv} = D_{uv} \cdot K^{-1} [u, v, 1]^\top$, generating a point cloud $P \in \mathbb{R}^{H\times W\times 3}$ at the same resolution as the RGB image. Adaptive average pooling aligns $P$ to the visual feature resolution $h\times w\times c$. Each coordinate axis $\alpha \in \{x,y,z\}$ occupies $\lfloor c/3 \rfloor$ channels, encoded using the sinusoidal function $\text{PE}(\alpha, 2j) = \sin(\alpha / 10000^{2j/\lfloor c/3 \rfloor})$ to form $F_{3D}$. Finally, $\tilde F_{\text{visual}} = F_{\text{visual}} + F_{3D}$.
    - **Design Motivation**: Monocular RGB lacks metric scale, causing significant errors when LLMs attempt to regress 3D box centers. Explicitly encoding metric coordinates into tokens provides the model with a prior of "where this token corresponds in the physical world," transforming bbox center regression from an ambiguous task into a query-based one.

3. **SpatialSceneQA: 61k Simulated Audio-Visual Joint Benchmark (Construction as Method Branch)**:
    - **Function**: Provides the first large-scale instruction-tuning set featuring degree-level azimuth/elevation, 3D bboxes, and audio-visual spatial ground truth, covering 5 task categories (Single-source DoA / Overlapping DoA / 3D Visual Grounding / Single-source Multi-speaker Matching / Overlapping Multi-speaker Matching), split by HM3D scenes into 130/15/36 for train/val/test.
    - **Mechanism**: (i) SoundSpaces 2.0 is used on HM3D meshes for bi-directional path tracing to render RIRs; the FOA signal is $A_c^{(r)}(t) = R_c(\cdot;\mathbf{s},\mathbf{r},\theta) * A^{(s)}(t)$, with dry speech from LibriSpeech, a source-receiver distance of 1–4 m (constrained to the same room), and a 0.5 m obstacle margin. (ii) Habitat-Sim renders synchronized RGB-D and semantic masks. (iii) Hunyuan3D-1.0 generates 120 loudspeaker meshes (96/12/12 train/val/test) inserted into scenes, filtered by visibility constraints (semantic map ≥500 pixels at 1920×1080).
    - **Design Motivation**: Real datasets like STARSS23 lack aligned depth and are small in scale. Scene-level splitting prevents geometric leakage of rooms. Using different loudspeaker mesh splits tests generalization to unseen geometries. Inserting 1–3 candidates forces the model to perform geometric grounding rather than relying on object category shortcuts.

### Loss & Training
- All LLM training uses LoRA (r=64, α=128, dropout 0.05). Qwen2.5-Omni weights initialize the visual encoder, mono audio branch, and LLM decoder; Neural IV and the new audio adapter are randomly initialized.
- Task-specific selective fine-tuning: A/B (DoA) only trains the audio encoder + Neural IV + projector; C (Visual Grounding) only trains the visual encoder + projector; D/E (Joint Reasoning) trains all modality encoders + projectors.
- Training on A100 40GB, batch size 1–3, for 3k–6k steps; cosine lr scheduler with 2.5k steps of linear warm-up; peak lr $1\times 10^{-5}$, weight decay 0.05.

## Key Experimental Results

### Main Results

| Model | Modality | Audio DoA $\downarrow$ | Overlap DoA $\downarrow$ | 3D IoU $\uparrow$ | Visual Offset $\downarrow$ | 1-spk Acc $\uparrow$ | 2-spk Acc $\uparrow$ |
|------|------|------------------------|--------------------------|-------------------|----------------------------|---------------------|---------------------|
| Random | – | 90° | 90° | 0.00 | $\infty$ | 45.6 | 47.4 |
| Qwen2.5-Omni (zero-shot) | RGB + Mono | – | – | 0.00 | 2.40 m | 35.8 | 44.0 |
| BAT (5 ep) | Binaural | 2.16° | 19.09° | – | – | – | – |
| Qwen3-VL-8B (zero-shot) | RGB | – | – | 0.01 | 1.11 m | – | – |
| N3D-VLM (zero-shot) | RGB-D | – | – | 0.00 | 2.04 m | – | – |
| **JAEGER (Classical IV)** | RGB-D + FOA | 2.95° | 6.44° | 0.32 | 0.16 m | 99.5 | 98.6 |
| **JAEGER (Neural IV)** | RGB-D + FOA | **2.21°** | **4.11°** | **0.32** | **0.16 m** | **99.5** | **99.2** |

DoA is measured by Median Angular Error (°); joint reasoning uses Accuracy for 3-way classification. Key Observation: On single-source DoA, Neural IV is on par with BAT (2.21° vs 2.16°), **but on overlapping sources, it reduces error from 19.09° to 4.11°, a nearly 5× gap**. In joint reasoning, 2D AV-LLMs hover around ~35–45% even after fine-tuning, while JAEGER reaches 99.2%.

### Ablation Study

| Configuration | 1-spk Acc | 2-spk Acc | Description |
|------|-----------|-----------|------|
| Ours (Neural IV) | 99.5 | 99.2 | Full model |
| Ours (Classical IV) | 99.5 | 98.6 | Reverting to STFT-based IV, drop of 0.6 in overlapping scenes |
| Ours (Neural) w/o Depth | 96.9 | 94.9 | Removing 3D PE, drop of 2.6 / 4.3 |
| Ours (Classical) w/o Depth | 99.2 | 98.7 | As above; Classical path is less sensitive to depth |
| Ours w/o FOA Encoder | 43.8 | 47.6 | **Removing FOA → Crashes to random** |
| Ours w/o Depth & FOA | 43.8 | 45.7 | Removing both → random |

Cross-scene generalization (MAE °, Cross-evaluation):

| Train \ Test | Classical Single | Classical Overlap | Neural Single | Neural Overlap |
|--------------|-------------------|--------------------|----------------|-----------------|
| Train Single | 2.95° | **18.35°** | 2.21° | **14.91°** |
| Train Overlap | **19.25°** | 6.44° | **14.85°** | 4.11° |

Neural IV is more stable in all "train-test mismatch" cells compared to Classical IV (14.85° vs 19.25°), indicating it learns more intrinsic directional cues.

3D-aware Encoding Ablation (Task C): Adding depth increases Mean 3D IoU from 0.29 to 0.32 and reduces Median Visual Offset from 0.18 m to 0.16 m.

### Key Findings
- **The FOA encoder is the crux of joint reasoning**: Removing it leads to a drop to random performance (43.8%), which cannot be recovered even with RGB-D and LoRA fine-tuning—this provides strong empirical support for the "explicit 3D necessity" argument.
- **Neural IV benefits primarily "difficult scenarios"**: While matching Classical IV in single-source cases (99.5 vs 99.5), Neural IV is consistently superior for overlapping sources and cross-scene generalization (4.11° vs 6.44°, 14.91° vs 18.35°), demonstrating the value of a learnable spatial encoder as SNR/reverberation worsens.
- **Depth has a greater impact on the Neural path than the Classical path**: Neural w/o depth drops 4.3 points on 2-speaker tasks, whereas Classical only drops 0.1. This suggest that the more precise directions provided by Neural IV are more sensitive to the loss of depth (accuracy in one axis cannot compensate for ambiguity in another).

## Highlights & Insights
- **Lifting physical formulas to the latent space is a reusable design**: The core structure of Classical IV, $F_W^* \odot F_C$, is derived from first principles of acoustics. The authors did not abandon this but replaced STFT with CNN and complex conjugate multiplication with a latent Hadamard product. Neural IV acts as a "physical prior + data-driven" hybrid—a paradigm applicable whenever signal processing can be neuralized (radar, sonar, ultrasound).
- **Simulation Pipeline Engineering**: The stack of HM3D + SoundSpaces 2.0 + Hunyuan3D-1.0 + LibriSpeech is the most cost-effective path to generating degree-level ground truth. The authors' rigorous handling of scene-level splits, visibility constraints (500 pixels), and geometric generalization splits ensures the integrity of the benchmark.
- **"2D AV-LLMs fail even with fine-tuning" is a strong narrative**: Showing that Qwen2.5-Omni still collapses to random levels when fine-tuned on specialized data upgrades the "indispensability of 3D modalities" from intuition to counter-factual evidence.

## Limitations & Future Work
- **Purely simulated, lacking real-world acoustic validation**: The authors acknowledge that SoundSpaces 2.0 RIR rendering (bi-directional path tracing + HRTF) differs from real microphone arrays/RGB-D sensors regarding synchronization, calibration, and noise characteristics. A zero-shot evaluation on real datasets like STARSS23 would add more weight.
- **Relatively controlled scene settings**: Source-receiver pairs are forced into the same room within 1–4 m, with a 500-pixel visibility minimum and at most 3 candidate loudspeakers. Heavy occlusion, cross-room scenarios, long distances (>4 m), and >3 candidates—common in real embodied scenarios—remain untested.
- **Limited task granularity**: Current tasks involve static single frames and static sound sources. Dynamic trajectories, moving sound sources, head rotation, and long-horizon reasoning (e.g., "hear someone speak and then walk towards them") are not yet covered.
- **Physical interpretability of Neural IV**: While end-to-end performance is verified, exploring what direction embeddings the latent intensity $h_C = f_W \odot f_C$ actually learns through probing or visualization could build trust among physicists.

## Related Work & Insights
- **vs Hear You Are (Ryu et al., 2026)**: Uses panoramic RGB + binaural, but assumes single sources and lacks depth, failing to test overlap-robustness and depth-aware grounding. JAEGER employs RGB-D + 4-channel FOA + overlapping source settings with ≥2 candidates.
- **vs SAVVY (Chen et al., 2025)**: Also fuses RGB-D and multi-channel audio but uses a cascaded pipeline + traditional DSP for DoA, preventing end-to-end learning. JAEGER internalizes DoA into LLM tokens, allowing full differentiability.
- **vs BAT (Zheng et al., 2024)**: Binaural + HRTF; single-source accuracy (2.16°) is slightly better than JAEGER (2.21°), but BAT's binaural representation is structurally weaker for overlapping sources (19.09°) and cross-device/HRTF generalization. JAEGER’s hardware-agnostic FOA + Neural IV is more general.
- **vs N3D-VLM (Wang et al., 2025)**: A purely visual 3D grounding path that proves RGB-D + 3D PE allows LLMs to output 3D bboxes. JAEGER adopts this visual front-end and fuses it with an audio path—indicating that best practices for 3D visual grounding are converging, with the next step being additional modalities.
- **vs Tang et al. (2024)**: Injected Classical IV into an auditory LLM for localized speech processing; JAEGER adopts the IV logic but replaces STFT with CNN and extends to a full multimodal LLM.

## Rating
- Novelty: ⭐⭐⭐⭐ Neural IV is a clear incremental step in neuralizing classical IV; 3D-aware visual encoding follows N3D-VLM; SpatialSceneQA is a solid engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes main tables, cross-scene generalization matrices, and three-way ablations. The "FOA-off to random" result is highly persuasive; penalized for being simulation-only.
- Writing Quality: ⭐⭐⭐⭐ The causal chain from motivation to formula to ablation is clear; the physical comparison table for Neural IV makes the contribution intuitive.
- Value: ⭐⭐⭐⭐ The dataset, end-to-end 3D AV-LLM paradigm, and open-source code/models serve as reusable infrastructure for the embodied AI and spatial audio-visual communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[ICML 2026\] Multimodal Fact-Level Attribution for Verifiable Reasoning](multimodal_fact-level_attribution_for_verifiable_reasoning.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[ACL 2026\] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](../../ACL2026/audio_speech/analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)
- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](../../ACL2026/audio_speech/music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)

</div>

<!-- RELATED:END -->
