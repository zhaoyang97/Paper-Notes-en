---
title: >-
  [Paper Note] Object-Centric Latent Action Learning
description: >-
  [AAAI 2026][Robotics][object-centric representation] This paper proposes an object-centric latent action learning framework that leverages self-supervised object decomposition (VideoSAUR) to disentangle task-relevant ent…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "object-centric representation"
  - "latent action learning"
  - "visual distractions"
  - "imitation learning"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 678f95f2e745b541
---

# Object-Centric Latent Action Learning

**Conference**: AAAI 2026 Oral  
**arXiv**: [2502.09680](https://arxiv.org/abs/2502.09680)  
**Code**: [https://github.com/dunnolab/object-centric-lapo](https://github.com/dunnolab/object-centric-lapo)  
**Area**: Reinforcement Learning
**Keywords**: object-centric representation, latent action learning, visual distractions, imitation learning, self-supervised learning

## TL;DR

This paper proposes an object-centric latent action learning framework that leverages self-supervised object decomposition (VideoSAUR) to disentangle task-relevant entities from visual distractions (e.g., dynamic backgrounds), reducing the performance degradation of LAPO on distracted videos by approximately 50%. A linear action probe is used to automatically select control-relevant slots.

## Background & Motivation

Embodied AI lags behind NLP and computer vision in general capability, with the primary bottleneck being the lack of high-quality pretraining data. The vast amount of internet video data covering diverse human activities represents a potential training source, but two major obstacles exist:

**Absence of action labels**: Internet videos lack explicit action annotations and cannot be directly used for imitation learning or reinforcement learning. Latent action models (LAMs) address this by inferring latent actions from observation sequences, but introduce new challenges.

**Action-correlated visual distractions**: Real-world videos inherently contain distractions correlated with actions—such as dynamic backgrounds, camera shake, and color variations—which produce spurious correlations with agent actions and cause LAMs to overfit to non-causal patterns. Existing methods (e.g., LAPA) typically assume clean datasets or rely on expensive annotations, severely limiting scalability.

**Fragility of existing LAMs**: Nikulin et al. (2025) empirically demonstrate that LAPO degrades significantly on data containing distractions, and propose reusing existing action labels to provide supervision during LAM pretraining. However, in certain domains (e.g., YouTube videos), action labels simply do not exist.

**Core hypothesis**: Object-centric representations provide the necessary structural prior to disentangle causal agent–object interactions from non-causal visual correlations. Operating on slot features rather than raw pixels allows LAMs to focus on the dynamics of relevant objects while filtering out background motion and other distractions.

## Method

### Overall Architecture

The pipeline consists of three stages:
1. **Object-centric pretraining**: Decompose video frames into interpretable object slots using VideoSAUR.
2. **Latent action learning**: Train a LAPO-based inverse/forward dynamics model in slot space.
3. **Behavioral cloning and fine-tuning**: Train a BC policy using inferred latent actions, then fine-tune with a small number of ground-truth action labels (≤2.5% of data).

### Key Designs

1. **Object-centric representation learning (VideoSAUR)**: VideoSAUR decomposes input video frames into spatiotemporal object slots. Its self-supervised architecture isolates individual entities in the scene, providing structured representations that are robust to background noise and incidental motion. For each observation $o_t$, the encoder produces $K$ slot vectors $s_t^{(k)} \in \mathbb{R}^d$. Due to its Transformer decoder, each slot can be projected back to the original image space via attention maps as alpha masks, yielding object masks $m_t^{(k)}$ for visualization.

   A key implementation detail is the use of **fixed slot initialization**, ensuring that the same slot index corresponds to semantically similar objects across episodes, thereby reducing slot permutation issues. The authors also experimented with the STEVE model but found it unable to reliably isolate entities such as the hopper.

2. **Slot selection via linear action probe**: Although object-centric models can decompose scenes, automatically identifying which slots correspond to task-relevant entities remains a challenge. The method employs linear probing (inspired by Alain & Bengio 2016):

    - Apply PCA dimensionality reduction to slot encodings from a small set of labeled trajectories.
    - Train a linear regressor to predict ground-truth actions.
    - Evaluate mean test MSE using 5-fold cross-validation (the Linear Action Probe score).
    - Select the slot(s) with the lowest MSE as the relevant slot set $\mathcal{S}^\star = \{s^{(k)} | k \in \mathcal{K}^\star\}$.

   This selection is performed only once after object-centric pretraining, leveraging fixed slot initialization to ensure consistent interpretation across episodes.

3. **Two variants of latent action modeling**:

    - **LAPO-slots**: Operates entirely in latent space. An inverse dynamics model $z_t \sim f_{IDM}^s(\cdot | s_t, s_{t+1})$ and a forward dynamics model $\hat{s}_{t+1} \sim f_{FDM}^s(\cdot | s_t, z_t)$ are trained in slot embedding space, minimizing $\|\hat{s}_{t+1} - s_{t+1}\|^2$.

    - **LAPO-masks**: Operates in pixel space. Object masks from selected slots are applied to input frames to create filtered images retaining only task-relevant objects, on which the dynamics model is then trained.

### Loss & Training

- VideoSAUR pretraining: self-supervised, requires no action labels.
- Latent action model training loss: $\mathcal{L}_{MSE} = \mathbb{E}_t[\|f_{FDM}(f_{IDM}(o_t, o_{t+1}), o_t) - o_{t+1}\|^2]$ (with corresponding adjustments for the slots or masks variants).
- BC training: supervised learning on inferred latent actions.
- Fine-tuning: uses a very small number of ground-truth action labels (0.1%–2.5%).

## Key Experimental Results

### Main Results

Evaluation is conducted on 8 tasks across the Distracting Control Suite (DCS) and Distracting MetaWorld (DMW); performance is normalized relative to a BC agent trained with all ground-truth action labels.

| Environment | Task | LAPO (baseline) | LAPO-clean (upper bound) | LAPO-masks | LAPO-slots | Recovery % |
|-------------|------|-----------------|--------------------------|------------|------------|------------|
| DCS-Hard | cheetah-run | 0.24±0.02 | 0.76±0.04 | 0.41±0.03 (+32%) | **0.55±0.04** (+58%) | 58% |
| DCS-Hard | hopper-hop | 0.03±0.01 | 0.27±0.03 | 0.08±0.01 (+20%) | **0.15±0.02** (+50%) | 50% |
| DCS-Hard | humanoid-walk | 0.02±0.01 | 0.06±0.01 | 0.04±0.02 (+47%) | **0.06±0.01** (+105%) | 105% |
| DCS-Hard | Average | 0.08±0.01 | 0.35±0.04 | 0.15±0.02 (+26%) | **0.22±0.02** (+52%) | 52% |
| DCS | Average | 0.13±0.02 | 0.35±0.04 | **0.24±0.04** (+50%) | 0.24±0.03 (+50%) | 50% |
| DMW | hammer | 0.75±0.07 | 0.98±0.01 | 0.96±0.01 (+91%) | **0.99±0.02** (+102%) | 102% |
| DMW | bin-picking | 0.18±0.08 | 0.74±0.10 | **0.49±0.10** (+56%) | 0.33±0.08 (+27%) | 56% |
| DMW | Average | 0.31±0.06 | 0.65±0.06 | 0.50±0.07 (+55%) | 0.48±0.06 (+50%) | 50% |

Definition: Recovery ratio $= \frac{\text{LAPO-slots/masks} - \text{LAPO}}{\text{LAPO-clean} - \text{LAPO}} \times 100\%$, measuring how much of the performance gap caused by distractions is recovered.

### Ablation Study

| Configuration | Key Findings | Details |
|---------------|--------------|---------|
| Slot count K=2–15 | Robust to K | LAPO-slots outperforms the LAPO baseline across all K values |
| K=2 | Representation collapse | Too few slots cause multiple entities to merge into one |
| K≤8 vs K>8 | Single vs. dual slots | Single slot is optimal for K≤8; concatenating top-2 works better for K>8 |
| DCS vs DCS-Hard | Robustness difference | LAPO-slots maintains consistent improvement under stronger distractions (50%→52%); LAPO-masks degrades (50%→26%) |
| Probe accuracy vs. BC performance | Strong positive correlation | Linear Action Probe score is highly correlated with downstream BC success rate |

### Key Findings

1. **50% performance recovery**: Across DCS and DMW, object-centric representations recover on average 50% of the performance gap between the distracted baseline and the distraction-free upper bound.
2. **LAPO-slots is more robust under strong distractions**: Compared to LAPO-masks, it leverages the DINOv2 encoder to capture high-level semantic features that are more robust to appearance changes.
3. **Functional vs. geometric decomposition**: In DMW, VideoSAUR merges behaviorally coupled entities (e.g., robotic arm and target object) into a single slot rather than separating them geometrically, reflecting a tendency toward functional decomposition.
4. **Effectiveness with very few labels**: Fine-tuning with only 0.1%–2.5% of ground-truth action labels yields strong performance.
5. **Largest improvement on humanoid-walk**: +174% (DCS) and +105% (DCS-Hard), suggesting that the advantage of object-centric representations is most pronounced on the most complex tasks.

## Highlights & Insights

- **Object-centric representations as a natural defense against visual distractions**: This is the central contribution—decomposing the scene into discrete slots inherently provides an unsupervised mechanism for separating causal from non-causal signals.
- **Simplicity and practicality of the linear action probe**: No complex attention mechanisms or learning procedures are required; simple linear regression with PCA reliably selects control-relevant slots and is highly correlated with downstream performance.
- **Complementarity of the two variants**: LAPO-slots is more robust (leveraging DINOv2 semantic features), while LAPO-masks performs better on certain tasks by preserving spatial detail.
- **Robustness to K**: In practice, the exact number of objects in the scene need not be known.

## Limitations & Future Work

1. **Dependence on OCL model quality**: The quality of the object-centric decomposition directly determines the effectiveness of the method. Current OCL models still struggle with heavy occlusion and multi-viewpoint scenes.
2. **Fixed slot count K**: Most OCL frameworks require K to be specified in advance, limiting flexibility across scenes of varying complexity.
3. **Lack of memory mechanism**: VideoSAUR cannot handle objects entering or leaving the scene.
4. **Collapsed decomposition under limited data diversity**: In DMW, behaviorally coupled entities are merged into a single slot due to limited trajectory diversity.
5. **Evaluation limited to simulation**: DCS and DMW are both simulated environments; generalization to real-world videos (e.g., YouTube) remains to be verified.

## Related Work & Insights

- **LAPO** (Schmidt & Jiang 2024): The direct baseline of this work—inferring latent actions from visual observation sequences, effective on clean data but failing under distractions.
- **VideoSAUR** (Zadaianchuk et al. 2023): The object-centric model used in this work—self-supervised video decomposition based on the DINOv2 encoder.
- **Nikulin et al. (2025)**: An orthogonal approach proposing a reconstruction-free framework that requires a small number of labeled trajectories to provide supervision during pretraining.
- **Genie** (Bruce et al. 2024): Generative interactive environments; LAMs are also used in large-scale VLA models.
- **LAPA** (Ye et al. 2024): Latent action pretraining for VLA models, but assumes clean datasets.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of object-centric representations and latent action learning is novel and intuitively well-motivated.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 tasks with multi-dimensional analysis, though limited to simulated environments.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, intuitive figures, and thorough slot selection analysis.)
- Value: ⭐⭐⭐⭐ (Provides an important direction for robustness improvement in learning embodied policies from real-world videos.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](../../ICML2026/robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)

</div>

<!-- RELATED:END -->
