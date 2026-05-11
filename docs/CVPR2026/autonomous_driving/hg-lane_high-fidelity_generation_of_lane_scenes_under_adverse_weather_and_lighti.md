---
title: >-
  [Paper Note] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation
description: >-
  [CVPR 2026][Autonomous Driving][lane detection] To address the severe scarcity of adverse-weather samples in lane detection datasets (CULane/TuSimple)…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "lane detection"
  - "adverse weather"
  - "diffusion model"
  - "ControlNet"
  - "data augmentation"
  - "CULane"
  - "TuSimple"
date: 2026-05-08
content_hash: 6d9cf876cbe3f9ff
---

# HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation

**Conference**: CVPR 2026
**arXiv**: [2603.10128](https://arxiv.org/abs/2603.10128)
**Code**: [zdc233/HG-Lane](https://github.com/zdc233/HG-Lane)
**Area**: Autonomous Driving
**Keywords**: lane detection, adverse weather, diffusion model, ControlNet, data augmentation, CULane, TuSimple

## TL;DR

To address the severe scarcity of adverse-weather samples in lane detection datasets (CULane/TuSimple), this paper proposes HG-Lane — a two-stage diffusion-based generation framework requiring no re-annotation. Stage-I employs Control Information Fusion and Structure-aware Reverse Diffusion to preserve lane geometry, while Stage-II applies Appearance-aware Refinement to adjust illumination style. The framework generates 30K images across snow/rain/fog/night/dusk conditions. CLRNet achieves an overall mF1 improvement of +20.87%, with +38.8% in snow scenarios.

## Background & Motivation

Lane detection is a fundamental perception task in autonomous driving. Mainstream datasets such as CULane and TuSimple are predominantly collected under clear daytime conditions, with a severe lack of samples covering adverse weather (snow, rain, fog) and low-light scenarios (night, dusk). This causes lane detection models to suffer sharp performance degradation under such conditions — which are precisely the scenarios where reliable detection is most critical.

Existing solutions face two core contradictions:

**High real-world data collection cost**: Driving and collecting data in actual blizzards, heavy rain, or dense fog is not only hazardous but also subject to significant geographic and seasonal constraints. Even when collected, per-frame lane annotation is required, at a cost far exceeding that of standard object detection.

**Existing generative methods fail to preserve lane semantics**: When weather-style transfer is applied via CycleGAN or unconditional generative models, the positions and shapes of lane markings in the generated images frequently shift or disappear entirely. The original annotations no longer align with the generated images, necessitating re-annotation — effectively returning to square one.

**Fundamental requirement**: A generation method capable of strictly preserving the geometric structure of lane markings while altering weather/lighting appearance — enabling direct reuse of original annotations on generated images and achieving zero-annotation-cost data augmentation.

## Method

### Overall Architecture

HG-Lane adopts a two-stage generation pipeline. For each input normal-weather image $I$:

- **Stage-I**: Structure-aware Reverse Diffusion → generates a weather-transformed image $I'$ that preserves lane structure.
- **Stage-II** (night/dusk only): Appearance-aware Refinement → further adjusts the illumination style of $I'$.

Both stages employ **pretrained** ControlNet models without any fine-tuning on lane detection data.

### Key Designs

#### Module 1: Control Information Fusion

**Core Problem**: How to construct a control signal that simultaneously guides weather generation and precisely preserves lane positions?

**Solution**: Three complementary sources of information are fused into a single control map $C_0$:

$$C_0 = \text{Canny}(I) \oplus (\text{LaneAnnotation}(I) \odot \text{ColorMask})$$

Specific steps:

1. **Canny edge map** $E = \text{Canny}(I)$: Extracts global structural information from the original image (road boundaries, vehicle contours, road markings, etc.), providing scene-level layout constraints for the diffusion model.
2. **Colorized lane annotation map**: The original lane annotations (2D coordinate sets) are rendered as a colored mask $L_{\text{color}}$ by assigning distinct colors to each lane category. Different colors help the model distinguish left/right lanes and solid/dashed lines.
3. **Fusion overlay**: The Canny edge map and the colorized lane annotation are combined via channel-wise superposition to produce the fused control map $C_0$.

**Design Motivation**: Using Canny edges alone causes lane markings — as thin line segments — to be easily ignored during diffusion; using lane annotations alone lacks global scene structure. The fusion provides lane markings with both global context and explicit strong signals within the control map.

#### Module 2: Stage-I — Structure-aware Reverse Diffusion

**Architecture**: Conditional generation pipeline based on Stable Diffusion and Canny-ControlNet.

**Inputs**:
- Control map $C_0$ (fused Canny + lane information)
- Category-specific text prompt: e.g., "A road scene with lane markings during heavy snowfall"

**Generation process**:

$$\epsilon_\theta(z_t, t, c_{\text{text}}, C_0) = \text{SD}(z_t, t, c_{\text{text}}) + \text{ControlNet}(z_t, t, C_0)$$

Canny-ControlNet injects structural constraints at each denoising step in latent space, ensuring that the edge distribution of the generated output closely matches $C_0$. Because $C_0$ contains explicit lane annotation information, the positions and shapes of lane markings are enforced.

**Category-specific Prompts**: Tailored text prompts are designed for each of the six weather/lighting conditions:
- Snow: emphasizes snow accumulation on road surfaces and falling snowflakes
- Rain: emphasizes road surface reflections and rain-induced blur
- Fog: emphasizes reduced visibility at distance and a hazy atmosphere
- Night: emphasizes overall low ambient light and headlight illumination
- Dusk: emphasizes gradually darkening sky and warm-toned lighting
- Shadow: emphasizes locally occluded shadow regions

#### Module 3: Stage-II — Appearance-aware Refinement

**Motivation**: Stage-I uses Canny-ControlNet primarily for structural control, which provides insufficient control over global color tone and brightness. In particular, night and dusk scenarios require substantial changes to overall image brightness and color temperature that text prompts alone cannot reliably achieve.

**Solution**: Stage-II is applied exclusively to night and dusk scenarios, using **InstructPix2Pix ControlNet** to refine the illumination style of the Stage-I output:

$$I_{\text{final}} = \text{IP2P-ControlNet}(I', c_{\text{instruction}})$$

where $c_{\text{instruction}}$ is an editing instruction such as "Make it look like nighttime with street lights." InstructPix2Pix naturally supports modifying appearance attributes while preserving image structure, complementing the structural preservation objective of Stage-I.

**Why snow/rain/fog do not require Stage-II**: These weather transformations are primarily additive effects (snowflakes, raindrops, haze), which Stage-I's text prompts are sufficient to guide, without the need for additional global illumination adjustment.

### Dataset Construction

Based on the CULane training set (~88K images), 5,000 images are generated per weather category across six conditions, yielding a total of 30K images (5,000 × 6), forming the HG-Lane Benchmark. Generated images directly reuse the original lane annotations.

## Key Experimental Results

### Main Results: Lane Detection Performance Improvement (CULane Test Set)

Using CLRNet (CVPR 2022, a mainstream lane detector) as the baseline:

| Training Data | Overall mF1 | Snow | Rain | Fog | Night | Dusk | Shadow |
|---|---|---|---|---|---|---|---|
| CULane original | baseline | baseline | baseline | baseline | baseline | baseline | baseline |
| +HG-Lane 30K | **+20.87%** | **+38.8%** | +18.2% | **+26.84%** | **+21.5%** | +15.7% | +13.2% |

The snow scenario yields the largest improvement (+38.8%), as the original CULane dataset contains virtually no snow samples.

### Cross-Detector Generalization

HG-Lane data is validated across multiple lane detectors, all achieving significant improvements, demonstrating that the gains from the generated data are not architecture-specific.

### Lane Annotation Preservation Quality

Lane detection IoU is evaluated on generated images using original annotations directly, verifying that lane positions have not shifted. Quantitative results show that the average IoU between generated images and original annotations remains above 95%.

### Ablation Study

| Configuration | mF1 Gain |
|---|---|
| Canny control only (no lane fusion) | +11.3% |
| Lane annotation control only (no Canny) | +8.7% |
| Canny + lane fusion (full Stage-I) | +17.5% |
| Stage-I + Stage-II (night/dusk) | **+20.87%** |

Control Information Fusion yields a clear improvement over single-signal control. Stage-II contributes approximately 3.4% additional gain for night/dusk scenarios.

### Comparison with Existing Methods

HG-Lane outperforms style-transfer methods such as CycleGAN and UNIT across both lane preservation quality and downstream detection performance. Lane markings generated by traditional methods frequently deform or disappear, rendering the original annotations unusable.

## Highlights & Insights

- **Practical value of zero annotation cost**: The entire pipeline requires no additional annotation — original lane annotations are reused directly. This is highly significant for autonomous driving, where annotation costs are prohibitively high.
- **Elegant fused control map design**: Canny edges provide global layout; colorized lane annotations provide explicit lane signals. The two are complementary, outperforming either signal alone by 30–50%.
- **Rational two-stage decomposition**: Structure preservation (Stage-I) and appearance adjustment (Stage-II) are decoupled, avoiding the trade-offs inherent in asking a single model to handle both objectives simultaneously.
- **Fully pretrained models, no fine-tuning required**: Both ControlNet and InstructPix2Pix use publicly available pretrained weights, lowering the barrier to reproduction and eliminating dependence on lane-specific training data.
- **Snow +38.8% reveals the impact of data gaps**: The category with the greatest original data scarcity yields the largest improvement, clearly demonstrating that data imbalance is one of the core bottlenecks in current lane detection.

## Limitations & Future Work

1. **Generation diversity constrained by Canny edges**: The control map is grounded in the Canny edges of the original image, so the scene layout of generated images closely mirrors the originals. It cannot synthesize adverse-weather images with entirely novel scenes; diversity is bounded by the scene distribution of the original dataset.
2. **Stage-II addresses only night/dusk**: Compound adverse conditions (e.g., rain + night) lack dedicated refinement pipelines. Scenarios combining multiple adverse factors (e.g., snowy night, foggy night) may require more complex multi-stage processing.
3. **Validation limited to 2D lane detection**: The method is not evaluated on 3D lane detection (e.g., OpenLane) or BEV lane perception tasks, where the impact of adverse weather may be more complex.
4. **Limited quantitative evaluation of generation quality**: Generation quality is primarily assessed indirectly via downstream detection performance; systematic reporting of generative quality metrics such as FID and IS is absent.
5. **Slow diffusion generation speed**: Generating 30K images incurs considerable computational cost. Scaling to larger augmentation regimes (e.g., millions of images) may render computation a bottleneck.

## Related Work & Insights

- **ControlNet (ICCV 2023)**: Provides the foundational capability for structured conditional control → HG-Lane innovatively fuses Canny edges with task-specific annotations as the control signal.
- **InstructPix2Pix (CVPR 2023)**: Instruction-guided image editing → HG-Lane applies it in Stage-II for illumination style refinement, representing a well-engineered practical application.
- **CycleGAN/UNIT**: Traditional style-transfer methods → Unable to preserve fine-grained lane structure; HG-Lane resolves this fundamental limitation through explicit control signals.
- **CLRNet (CVPR 2022)**: Mainstream lane detector → Achieves the most significant gains from HG-Lane generated data.
- **ACGEN (CVPR 2024)**: Another conditional generation approach for autonomous driving → HG-Lane focuses specifically on lane detection, with task-specific control signal design.
- **Insight**: The paradigm of "fused control map + two-stage generation" is generalizable to other data augmentation tasks requiring precise preservation of annotation information, such as traffic sign detection and road marking detection.

## Rating

| Dimension | Score (1–5) | Remarks |
|---|---|---|
| Novelty | 3.5 | Core components (ControlNet, IP2P) are existing methods; innovation lies in the fused control map design and two-stage decomposition strategy — primarily engineering innovation. |
| Value | 4.5 | Zero annotation cost, fully pretrained weights, open-sourced 30K benchmark — extremely high practical applicability. |
| Experimental Thoroughness | 4.0 | Multi-detector validation, complete ablations, thorough comparison with style-transfer baselines; lacks independent generation quality metrics such as FID. |
| Writing Quality | 3.5 | Method description is clear and the two-stage pipeline is well-structured, but certain details (specific prompt content, hyperparameter choices) are insufficiently documented. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[CVPR 2026\] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)

</div>

<!-- RELATED:END -->
