---
title: >-
  [Paper Note] Training-free Motion Factorization for Compositional Video Generation
description: >-
  [CVPR 2026][Video Generation][Compositional video generation] This paper proposes a motion factorization framework that decomposes multi-instance scene motion into three categories — stationary, rigid-body, and non-rigid motion — and addresses prompt semantic ambiguity via Structured Motion Reasoning (SMR) while steering the generation of each motion category during diffusion through Decoupled Motion Guidance (DMG). The framework requires no additional training and achieves substantial improvements in motion diversity and fidelity on VideoCrafter-v2.0 and CogVideoX-2B.
tags:
  - CVPR 2026
  - Video Generation
  - Compositional video generation
  - motion factorization
  - structured reasoning
  - decoupled guidance
  - training-free
date: 2026-05-08
content_hash: de5ebd5c4bfc7a25
---

# Training-free Motion Factorization for Compositional Video Generation

**Conference**: CVPR 2026
**arXiv**: [2603.09104](https://arxiv.org/abs/2603.09104)
**Code**: To be released
**Area**: Diffusion Models / Video Generation / Motion Control
**Keywords**: Compositional video generation, motion factorization, structured reasoning, decoupled guidance, training-free

## TL;DR

This paper proposes a motion factorization framework that decomposes multi-instance scene motion into three categories — stationary, rigid-body, and non-rigid motion — and addresses prompt semantic ambiguity via Structured Motion Reasoning (SMR) while steering the generation of each motion category during diffusion through Decoupled Motion Guidance (DMG). The framework requires no additional training and achieves substantial improvements in motion diversity and fidelity on VideoCrafter-v2.0 and CogVideoX-2B.

## Background & Motivation

1. **Background**: Compositional Video Generation (CVG) aims to synthesize multi-instance, multi-motion scenes from complex prompts. Existing methods (LVD, VideoDirectorGPT, etc.) typically employ LLMs to produce bounding box sequences for guiding instance motion.
2. **Limitations of Prior Work**: (1) Motion semantic ambiguity — generating box sequences directly from text leads to broken motion trajectories and abnormal scale changes; (2) Coarse motion guidance — uniform diffusion guidance fails to distinguish between different motion categories, resulting in motion homogenization and unnatural dynamics.
3. **Key Challenge**: Existing methods treat the motion of all instances uniformly, lacking explicit modeling of motion category diversity. Stationary objects, linearly translating vehicles, and dancing humans fundamentally require distinct generation strategies.
4. **Goal**: How can video generation models produce diverse, category-consistent motion for each instance without any training?
5. **Key Insight**: Decompose motion into three primitive categories — stationary, rigid-body, and non-rigid — and design targeted reasoning and guidance strategies for each.
6. **Core Idea**: Motion factorization with a plan-then-generate paradigm — first infer per-instance motion representations via a structured motion graph, then synthesize each motion category through dedicated decoupled guidance branches.

## Method

### Overall Architecture

The framework follows a plan-then-generate paradigm. (1) Planning stage (SMR): the user prompt is converted into a motion graph, from which per-frame bounding box sequences are inferred as motion representations for each instance. (2) Generation stage (DMG): based on motion category, three dedicated guidance branches (appearance consistency / geometric invariance / spatial deformation) modulate attention maps to synthesize motion. The framework is model-agnostic, supporting both 3D U-Net (VideoCrafter) and DiT (CogVideoX) architectures.

### Key Designs

1. **Structured Motion Reasoning (SMR) Module**:

    - **Function**: Converts semantically ambiguous text prompts into structured motion representations, producing a motion category label and a per-frame bounding box sequence for each instance.
    - **Mechanism**: A motion graph $\mathcal{R} = (\mathcal{V}, \mathcal{E})$ is first constructed, where each instance is a node annotated with motion attributes and category labels, and directed edges encode spatial relationships and dynamic interactions between instances. Box sequences are then inferred per motion category — stationary instances maintain $\mathcal{B}_f(v_n) = \mathcal{B}_1(v_n)$; rigid-body instances update positions via estimated velocity $\vec{u}$ and acceleration $\vec{a}$ as $\mathcal{B}_f = \mathcal{B}_{f-1} + \vec{u} + \frac{1}{2}\vec{a}$; non-rigid instances are modeled using boundary displacement vectors $\Delta_f(v_n)$ to capture asymmetric deformation.
    - **Design Motivation**: Directly generating box sequences from prompts introduces semantic ambiguity leading to erroneous motion. The motion graph serves as an intermediate structured representation, enabling LLMs to reason step-by-step — first understanding inter-instance relationships and motion categories, then deriving specific motion parameters — substantially reducing ambiguity.

2. **Reference Conditional Guidance (Stationary Instances)**:

    - **Function**: Suppresses spurious inter-frame variations in static regions and maintains appearance consistency.
    - **Mechanism**: The frame with minimum inter-frame feature discrepancy is selected as the reference frame $f^* = \arg\min_f \sum_{f'} D(\varphi(\mathbf{z}_f^t), \varphi(\mathbf{z}_{f'}^t))$. A mask $\mathcal{G}_m$ constrains all frames to attend only to the reference frame, achieving pixel-level appearance alignment: $\mathcal{G}_m[x,y,f,f'](v_n) = \mathbb{1}(f'=f^* \& (x,y) \in \mathcal{B}(v_n))$.
    - **Design Motivation**: Video diffusion models frequently introduce spurious flickering in static regions. Anchoring attention to a stable reference frame eliminates unnecessary cross-frame variation at the attention level.

3. **Geometric Invariance Guidance (Rigid-Body Motion Instances)**:

    - **Function**: Preserves the geometric shape of instances throughout rigid-body motion.
    - **Mechanism**: K-means clustering first separates the foreground from bounding boxes; a shape template is then generated by aggregating coarse per-frame masks via pixel voting, which is back-projected into each frame to obtain aligned masks $\mathcal{M}_f$. A displacement penalty factor $\Gamma[f,f'] = \exp(-\alpha \cdot \|\mathbf{C}_f - \mathbf{C}_{f'}\|_2) + 1$ modulates the strength of inter-frame feature interaction — temporally proximate frames interact more strongly. The final guidance mask is $\mathcal{G}_r = \mathcal{M} \cdot \mathcal{M}^\top \odot \Gamma$.
    - **Design Motivation**: Without geometric constraints, video models frequently produce deformation artifacts during rigid-body motion. A frame-independent shape template ensures geometric consistency, while the displacement penalty enforces smooth motion transitions.

4. **Spatial Deformation Guidance (Non-Rigid Motion Instances)**:

    - **Function**: Models complex pixel-level deformations arising from non-rigid motion.
    - **Mechanism**: A perceptual deformation field $\mathcal{D}_{\text{perc}}$ is extracted from diffusion features via nearest-neighbor search, and a box deformation field $\mathcal{D}_{\text{box}}$ is obtained from bilinear interpolation of bounding box corner displacements. A deformation penalty factor $\Lambda[i,j] = \exp(-\alpha \cdot (\mathcal{D}_{\text{perc}}[i,j] - \mathcal{D}_{\text{box}}[i,j])) + 1$ minimizes the discrepancy between the two fields, steering actual deformation toward the expected trajectory. The final mask is $\mathcal{G}_{\text{nr}} = (\mathcal{M} \cdot \mathcal{M}^\top) \odot \Lambda$.
    - **Design Motivation**: In non-rigid motion, each pixel moves in a distinct direction (e.g., human joint articulation), necessitating pixel-level deformation fields rather than global translation for accurate modeling.

### Loss & Training

No additional training is required. For 3D U-Net architectures, the noise embedding is updated via gradient descent as $\mathbf{z}^{t-1} \leftarrow \mathbf{z}^t - \nabla\mathcal{L}$, where $\mathcal{L} = 1 - \frac{\beta}{P}\sum(\mathbf{A} \odot (\mathcal{G}_m + \mathcal{G}_r + \mathcal{G}_{nr}))$. For DiT architectures, attention scores are directly modified as $\mathbf{A} = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top (1 + \beta \odot (\mathcal{G}_m + \mathcal{G}_r + \mathcal{G}_{nr}))}{\sqrt{d}})$. VideoCrafter-v2.0 uses $\beta=10$ with guidance steps 1–25; CogVideoX-2B uses $\beta=0.15$ with guidance steps 1–10.

## Key Experimental Results

### Main Results

Evaluated on the authors' CVGBench-m (1,665 samples from MSR-VTT) and CVGBench-p (994 samples from Panda-70M):

| Model Configuration | Subject Consis. | Background Consis. | Temporal Flicker. | Motion Smooth. | Dynamic Degree |
|---|---|---|---|---|---|
| VideoCrafter-v2.0 (baseline) | 97.68% | 97.28% | 96.28% | 98.16% | 33.11% |
| + A&R | 97.48% | 97.05% | 96.43% | 98.27% | 38.40% |
| + **Ours** | **98.40%** | **98.11%** | **97.39%** | **98.63%** | **82.21%** |
| CogVideoX-2B (baseline) | 91.33% | 92.78% | 95.01% | 96.88% | 87.80% |
| + R&P | 91.00% | 90.85% | 95.07% | 96.96% | 91.02% |
| + **Ours** | **98.27%** | **97.73%** | **98.25%** | **98.74%** | **96.00%** |

### Ablation Study

Guidance branch ablation (VideoCrafter-v2.0 baseline):

| RCG | GIG | SDG | Subject Consis. | Dynamic Degree | Note |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 97.48% | 38.40% | Semantic guidance only |
| ✓ | ✗ | ✗ | 98.11% | 51.60% | Stationary guidance |
| ✗ | ✓ | ✗ | 98.07% | 53.60% | Rigid-body guidance |
| ✗ | ✗ | ✓ | 97.71% | 74.85% | Non-rigid guidance |
| ✓ | ✓ | ✓ | **98.40%** | **82.21%** | Full model |

Motion reasoning module ablation (CogVideoX-2B baseline):

| Configuration | Subject Consis. | Dynamic Degree | Note |
|---|---|---|---|
| w/o SMR | 93.16% | 88.21% | Direct text-to-motion |
| w/ SMR (Ours) | **98.27%** | **96.00%** | Motion graph reasoning |

### Key Findings

- **Dynamic Degree shows the largest gain**: on VideoCrafter-v2.0, it improves from 33.11% to 82.21% (+49.1 pp), indicating that the baseline model generates excessively conservative motion, which the proposed framework effectively resolves.
- **Non-rigid guidance contributes most to dynamic degree**: SDG alone raises Dynamic Degree from 38.40% to 74.85% (+36.45 pp).
- **SMR is critical**: removing SMR degrades Subject Consistency by 5.11% and Dynamic Degree by 7.79%, confirming that structured reasoning is essential for resolving semantic ambiguity.
- **Model scale affects reasoning quality**: LLaMA-70B outperforms the 8B variant by 6.87% in Dynamic Degree on VideoCrafter and 1.23% on CogVideoX.
- **Cross-architecture generalization**: the framework performs effectively on both 3D U-Net and DiT architectures, validating its architecture-agnostic design.

## Highlights & Insights

- **Elegant abstraction via three-category motion decomposition**: decomposing complex motion into stationary / rigid-body / non-rigid categories, each with a well-defined mathematical formulation (constant / kinematic equations / displacement field), yields a concise yet effective representation. This taxonomy is transferable to motion estimation, video editing, and related tasks.
- **Motion graph as intermediate representation**: reformulating the text-to-motion ambiguity problem as a two-step pipeline (text → structured graph → motion) and encoding inter-instance relationships through graph structure is a principled strategy for mitigating unreliable LLM motion reasoning.
- **Architecture-agnostic control via attention manipulation**: motion guidance is achieved by directly operating on attention maps or scores without modifying model weights or architecture, enabling seamless adaptation to diverse backbones.

## Limitations & Future Work

- The framework cannot handle rare semantic concepts (e.g., "Dendroid"), as it is bounded by the generative capacity of the underlying baseline model.
- Generation of emotionally nuanced expressions (e.g., "sad" facial expressions) is poor, as video models tend to neglect adjectives and adverbs.
- Only planar motion (bounding boxes) is supported; depth-direction motion and 3D rotation are not modeled.
- Camera motion modeling is not explored; all motion is synthesized under a fixed viewpoint.
- Future directions include: incorporating reference images to supply priors for rare concepts; modeling camera pose variation; and extending to 3D bounding boxes.

## Related Work & Insights

- **vs. VideoTetris/Vico**: These methods focus on semantic binding and token importance but overlook motion category diversity; the proposed method addresses the complementary problem of motion generation.
- **vs. LVD/VideoDirectorGPT**: These methods use LLMs to generate box sequences but apply uniform guidance, leading to motion homogenization; the proposed motion graph and decoupled guidance substantially improve diversity.
- **vs. FreeTraj/TrailBlazer**: These methods use sparse motion fields for guidance but do not distinguish motion categories; the proposed category-specific guidance is more fine-grained.
- **vs. MotionPrompting**: This method relies on user-provided mouse drag signals for motion input; the proposed framework is fully automated.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-category motion decomposition and motion graph reasoning are creative abstractions, though individual components (attention guidance / LLM reasoning) are relatively mature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The self-constructed benchmark covers diverse linguistic patterns and the ablations are comprehensive, but comparisons against state-of-the-art commercial models are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clearly presented with complete mathematical notation, though the notation system is dense.
- **Value**: ⭐⭐⭐⭐ The training-free and architecture-agnostic properties confer strong practical utility, and the motion decomposition paradigm has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](../../AAAI2026/video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)

</div>

<!-- RELATED:END -->
