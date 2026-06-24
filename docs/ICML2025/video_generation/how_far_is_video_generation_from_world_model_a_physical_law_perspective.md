---
title: >-
  [Paper Note] How Far is Video Generation from World Model: A Physical Law Perspective
description: >-
  [ICML 2025][Video Generation][World Model] This work systematically evaluates whether video generation models can discover physical laws from purely visual data by constructing a 2D physical simulation video dataset that stringently adheres to classical mechanics. It reveals that current models merely memorize patterns within the training distribution rather than generalizing to novel physical conditions.
tags:
  - "ICML 2025"
  - "Video Generation"
  - "World Model"
  - "Physical Law"
  - "Generalization Ability"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 45c3d4b0607477e3
---

# How Far is Video Generation from World Model: A Physical Law Perspective

**Conference**: ICML 2025  
**arXiv**: [2411.02385](https://arxiv.org/abs/2411.02385)  
**Code**: None (Project page: [phyworld.github.io](https://phyworld.github.io))  
**Area**: Video Generation  
**Keywords**: Video Generation, World Model, Physical Law, Generalization Ability, Diffusion Models

## TL;DR

This work systematically evaluates whether video generation models can discover physical laws from purely visual data by constructing a 2D physical simulation video dataset that stringently adheres to classical mechanics. It reveals that current models merely memorize patterns within the training distribution rather than generalizing to novel physical conditions.

## Background & Motivation

**Background**: The emergence of video generation models such as Sora has sparked discussions on whether "video generation models = world models." OpenAI has claimed that video generation models are "world simulators," implying they potentially comprehend the physical laws governing the actual world.

**Limitations of Prior Work**: Existing evaluations predominantly focus on visual quality metrics (FVD, FID) and lack quantitative assessments of physical understanding. On real-world videos, distinguishing whether a model "understands physics" or merely "memorizes visual patterns" is challenging due to the extreme complexity of causal analysis in real-world scenarios.

**Key Challenge**: Video generation models exhibit excellent performance in visual quality, creating an illusion of "understanding physics." However, there is currently no rigorous experimental framework capable of distinguishing "memorization + interpolation" from "genuine physical generalization."

**Goal**: To quantitatively answer whether video generation models can discover fundamental physical laws (Newtonian mechanics) from visual observations in a fully controllable environment, and to identify the boundaries of such capabilities.

**Key Insight**: Building a 2D physics simulator to generate videos that strictly adhere to classical mechanics. By controlling the distribution of physical parameters during training and testing, this work systematically assesses the models' performance across three levels: in-distribution (ID), out-of-distribution (OOD), and combinatorial generalization.

**Core Idea**: Replacing real-world videos with precisely controllable physical simulations to transform the ambiguous question of "whether video generation models understand physics" into a quantitatively measurable generalization test.

## Method

### Overall Architecture

The research design comprises three core components: (1) a 2D physical simulator based on classical mechanics, capable of generating videos strictly adhering to Newton's laws (e.g., uniform linear motion, elastic collisions, projectile motion); (2) a diffusion video generation model based on the DiT architecture, with parameters scaled from 22M to 310M and training data scaled from 30K to 6M; (3) a quantitative evaluation pipeline that extracts object trajectories (pixel tracking) from the generated videos, fits physical parameters (e.g., velocity, acceleration), and computes error relative to the theoretical values from actual physical laws.

### Key Designs

1. **2D Physical Simulation Test Set**:

    - **Function**: Provides controllable video data that strictly conforms to known physical laws, eliminating confounding factors present in the real world.
    - **Mechanism**: Implements three classical mechanics scenarios: uniform linear motion ($v = \text{const}$), elastic collisions (conservation of momentum $m_1v_1 + m_2v_2 = m_1v_1' + m_2v_2'$), and projectile motion ($y = v_0t - \frac{1}{2}gt^2$), where attributes such as color, shape, size, and velocity of the objects can be precisely controlled.
    - **Design Motivation**: Classical mechanics laws are simple and completely deterministic, representing the most fundamental requirement for physical understanding. If a model fails to discover Newtonian laws, learning more intricate physical rules is out of the question.

2. **Three-Level Generalization Testing Framework**:

    - **Function**: Distinguishes the model's memorization capabilities from genuine physical law discovery.
    - **Mechanism**: Progressively tests generalization through: ID testing (physical parameters within training distribution) $\rightarrow$ OOD testing (velocity/acceleration ranges outside training distribution) $\rightarrow$ Combinatorial generalization testing (PHYRE benchmark, featuring unseen combinations of objects and scene configurations).
    - **Design Motivation**: If a model has truly discovered physics laws like $F=ma$, these laws should hold true across any reasonable range of parameters, independent of the specific values encountered during training.

3. **Pixel-Level State Extraction and Quantitative Physical Evaluation**:

    - **Function**: Translates visual generation quality into quantifiable physical error metrics.
    - **Mechanism**: Detects and tracks objects in each frame of the generated videos, extracts the object position-time series, fits the motion equation parameters (e.g., velocity, acceleration), and calculates the absolute error relative to theoretical parameters.
    - **Design Motivation**: Conventional video evaluation metrics (FVD, SSIM) only measure visual quality but fail to reflect physical correctness. Trajectory-based physical quantification delivers a direct and highly accurate standard of measurement.

### Loss & Training

The models employ the standard diffusion training loss (DDPM objective), with scaling experiments conducted across various data scales (30K, 300K, 3M, 6M) and model sizes (DiT-S 22M, DiT-B 60M, DiT-L 170M, DiT-XL 310M). Additionally, fine-tuning experiments are performed on the pre-trained Stable Video Diffusion (SVD) to examine whether pre-trained knowledge aids physical generalization. No specific physical inductive biases are incorporated into the training strategy, aiming to purely test the model's capacity to autonomously discover physical laws from raw data.

## Key Experimental Results

### Main Results

Uniform linear motion scenario — ID vs. OOD velocity error:

| Model | Data Scale | ID Velocity Error | OOD Velocity Error | OOD/ID Ratio |
|:---|:---:|:---:|:---:|:---:|
| Ground Truth | - | 0.010 | 0.010 | 1.0x |
| DiT-S (22M) | 30K | 0.097 | 0.532 | 5.5x |
| DiT-B (60M) | 300K | 0.038 | 0.461 | 12.1x |
| DiT-L (170M) | 3M | **0.012** | 0.427 | **35.6x** |
| DiT-XL (310M) | 3M | 0.015 | 0.441 | 29.4x |
| SVD-ft | 3M | 0.019 | 0.398 | 20.9x |

Combinatorial generalization (PHYRE benchmark) — abnormal generation ratio:

| Number of Training Templates | Model | Abnormal Video Ratio |
|:---|:---|:---:|
| 6 | DiT-B | 67% |
| 60 | DiT-B | **24%** |
| 60 | DiT-L | 10% |

### Ablation Study

Attribute priority experiment (1400 test cases, uniform motion scenario):

| Attribute Conflict Pair | Model Preference | Number of Exceptions |
|:---|:---|:---:|
| Color vs. Shape | Color >> Shape | 0/1400 |
| Color vs. Size | Color >> Size | 0/1400 |
| Color vs. Velocity | Color >> Velocity | 0/1400 |
| Size vs. Velocity | Size >> Velocity | Very few |
| Size vs. Shape | Size >> Shape | Very few |

Impact of data/model scaling on OOD generalization:

| Scaling Dimension | Setting Range | ID Error Change | OOD Error Change |
|:---|:---|:---:|:---:|
| Data Scale | 30K → 3M (100x) | 0.097 → 0.012 (↓87%) | 0.532 → 0.427 (↓20%) |
| Model Parameters | 22M → 310M (14x) | 0.012 → 0.015 (Unchanged) | 0.427 → 0.441 (No improvement) |
| Pre-training | Random Init $\rightarrow$ SVD | 0.012 → 0.019 (Unchanged) | 0.427 → 0.398 (Slight decrease) |

### Key Findings

- **Near-Perfect ID Performance**: Within the training distribution, the velocity error of the largest model is only 0.012, which is close to the ground truth value of 0.010, indicating the model can fully fit physical patterns within the training distribution.
- **Complete Failure in OOD Generalization**: OOD error is 20 to 35 times larger than ID error and does not yield improvement with increased data scale and model capacity, signifying that the model does not learn a generalizable set of physical laws.
- **Attribute Priority Bias**: When handling conflicts, the model prioritizes attributes in the order of Color >> Size >> Velocity >> Shape, which is fundamentally based on visual salience rather than physical relevance (velocity is the most crucial physical variable here, yet it ranks second to last).
- **Case-Based Generalization**: In OOD scenarios, the model tends to duplicate training samples that are most similar to the input conditions rather than applying abstract physical principles.
- **Visual Ambiguity**: Inherent constraints such as size-distance ambiguity in 2D videos impose a theoretical ceiling on fine-grained physical modeling using purely visual signals.
- **No Significant Help from Pre-training**: The OOD error after SVD fine-tuning (0.398) is roughly on par with training from scratch (0.427), indicating that pre-trained visual priors contribute minimally to physical generalization.

## Highlights & Insights

- **Ingenious Experimental Design**: Using "controllable physical simulation" instead of "real-world videos" is an excellent approach to resolve the debate over whether "video models understand physics," converting an ambiguous philosophical inquiry into a rigorous scientific experiment.
- **Insightful and Inspiring Conclusions**: It clearly demonstrates that current video generation models act as "pattern matchers" rather than "physics simulators," which serves as a powerful counter-narrative to the Sora-style "world simulator" claim.
- **Thorough Scaling Experiments**: The comprehensive scaling experiments, spanning from 22M to 310M parameters and from 30K to 6M data points, effectively rule out the counterargument that "the model is simply not large enough."
- **Intriguing Discovery of Attribute Priority**: It reveals that the model processes physical attributes entirely based on visual salience (prioritizing color), which is entirely counter-intuitive to how a physicist would prioritize them.

## Limitations & Future Work

- The 2D simulator is oversimplified; more complex real-world scenarios such as 3D physics, fluid dynamics, and soft-body mechanics are not covered.
- Only the DiT architecture for diffusion models is evaluated; autoregressive video models (e.g., VideoGPT series) might exhibit different behaviors.
- The evaluation metrics depend on pixel-level tracking, which might introduce tracking errors when evaluating low-quality generated videos.
- Explicit physical inductive biases (e.g., Physics-informed Neural Networks) have not been explored to assist generalization.
- The paper primarily presents negative findings and lacks constructive solutions on "how to make models genuinely learn physics."

## Related Work & Insights

- **vs. Sora/OpenAI "World Simulator" Narrative**: This work directly challenges this claim—even under the simplest Newtonian mechanics setup, video models fail to generalize, indicating there is still a far cry from actual world models.
- **vs. PhysDreamer/UniSim**: These works try to introduce physical priors into video generation; the findings here provide empirical evidence supporting the necessity of doing so.
- **vs. PHYRE Benchmark**: This work combines the combinatorial generalization challenge of PHYRE with video generation models, showing that even physical generalization on a massive training scale remains hard to accomplish.
- **vs. Scaling Laws Studies**: This work reveals that physical generalization capabilities do not follow conventional scaling laws—scaling data and models fails to bridge the OOD gap, contrasting with the emergent capabilities observed in language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ This is the first work to quantitatively answer "whether video generation understands physics" through strictly controlled physical experiments, showcasing highly creative experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple model scales, data scales, tasks, and generalization levels, successfully ruling out almost all potential counterarguments.
- Writing Quality: ⭐⭐⭐⭐ The arguments are logical, and the conclusions are accurately stated, though more constructive discussions on future directions for negative results could be included.
- Value: ⭐⭐⭐⭐⭐ Provides a crucial course correction for the currently trend-heavy "world model" research direction, shedding light on the critical bottlenecks for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Physical Object Understanding with a Physically Controllable World Model](../../CVPR2026/video_generation/physical_object_understanding_with_a_physically_controllable_world_model.md)
- [\[CVPR 2025\] SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](../../CVPR2025/video_generation/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)
- [\[NeurIPS 2025\] Photography Perspective Composition: Towards Aesthetic Perspective Recommendation](../../NeurIPS2025/video_generation/photography_perspective_composition_towards_aesthetic_perspective_recommendation.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICML 2025\] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)

</div>

<!-- RELATED:END -->
