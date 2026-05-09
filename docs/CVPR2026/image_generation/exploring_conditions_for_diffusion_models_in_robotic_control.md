---
title: >-
  [Paper Note] Exploring Conditions for Diffusion Models in Robotic Control
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] This paper investigates how to leverage the conditioning mechanisms of pretrained text-to-image diffusion models to generate task-adaptive visual representations for robotic control. It identifies that text conditioning fails in control environments due to domain gap, and proposes ORCA, a framework that employs learnable task prompts and per-frame visual prompts as conditioning signals. ORCA achieves state-of-the-art performance across 12 tasks on three benchmarks: DMC, MetaWorld, and Adroit.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Models
  - Robotic Control
  - Visual Representations
  - Task Adaptation
  - Learnable Prompts
date: 2026-05-08
content_hash: cfca8ea5c0e7f432
---

# Exploring Conditions for Diffusion Models in Robotic Control

**Conference**: CVPR 2026
**arXiv**: [2510.15510](https://arxiv.org/abs/2510.15510)
**Code**: [https://orca-rc.github.io/](https://orca-rc.github.io/)
**Area**: Diffusion Models / Robotic Control
**Keywords**: Diffusion Models, Robotic Control, Visual Representations, Task Adaptation, Learnable Prompts

## TL;DR
This paper investigates how to leverage the conditioning mechanisms of pretrained text-to-image diffusion models to generate task-adaptive visual representations for robotic control. It identifies that text conditioning fails in control environments due to domain gap, and proposes ORCA, a framework that employs learnable task prompts and per-frame visual prompts as conditioning signals. ORCA achieves state-of-the-art performance across 12 tasks on three benchmarks: DMC, MetaWorld, and Adroit.

## Background & Motivation

1. **Background**: Pretrained visual representations (e.g., CLIP, VC-1, MVP) have become the standard paradigm for imitation learning—frozen pretrained encoders extract visual features, and downstream policy networks learn the mapping from features to actions. Meanwhile, diffusion models (e.g., Stable Diffusion) have demonstrated strong representational capacity in visual perception tasks such as semantic segmentation and depth estimation, with text conditioning enabling task adaptability.

2. **Limitations of Prior Work**: (a) Frozen visual representations are task-agnostic—the same representation exhibits large performance variance across different control tasks, requiring per-task manual selection; (b) fine-tuning visual encoders leads to severe overfitting due to the limited data available in imitation learning; (c) text conditioning proves effective in visual perception tasks (e.g., describing objects in an image improves semantic segmentation in VPD), but yields marginal or even harmful results when directly applied to control environments.

3. **Key Challenge**: A significant domain gap exists between control environments and the training data (web images) of diffusion models, causing text-image grounding to fail. Additionally, control tasks consist of dynamic video streams rather than static images, requiring fine-grained per-frame conditioning rather than global descriptions.

4. **Goal**: How to design conditioning mechanisms for diffusion models that are suited to robotic control, enabling task-adaptive visual representations without fine-tuning the model itself?

5. **Key Insight**: By analyzing cross-attention maps, the authors identify a grounding failure of text conditioning in control environments—for instance, the word "cheetah" fails to produce correct attention on MuJoCo-rendered cheetahs. The conditioning mechanism should therefore (a) adapt to the control environment and (b) incorporate per-frame visual information.

6. **Core Idea**: Replace text conditioning with learnable task prompts to adapt to the control environment, and introduce per-frame visual prompts derived from a visual encoder to capture dynamic details. Both components are learned end-to-end via a behavior cloning objective.

## Method

### Overall Architecture
The ORCA pipeline: an input observation image $I$ is encoded into a latent variable $z_0$ via the VQGAN encoder; noise is added to obtain $z_t$ (at $t=0$); $z_t$ is fed into the Stable Diffusion U-Net, where the condition $\mathcal{C}^*$ is formed by concatenating task prompts and visual prompts and passing them through the text encoder; intermediate features $f$ are extracted from the downsampling and bottleneck layers of the U-Net; after processing by a compression layer, the features are passed to the policy network $\pi_\phi$ to produce actions. The diffusion model is frozen throughout; only the prompts and the policy network are learned.

### Key Designs

1. **Task Prompts**:

   - **Function**: Replace text conditioning to achieve correct semantic grounding in control environments.
   - **Mechanism**: Implemented as learnable parameters ($l_t = 4$ tokens), shared across all training observations. Rather than using natural language, the model directly learns implicit "vocabulary," allowing cross-attention to automatically focus on task-relevant regions. Optimized end-to-end via the behavior cloning loss $\mathcal{L}_{\text{BC}}$.
   - **Design Motivation**: Natural text fails to ground in control environments due to the domain gap; learnable tokens circumvent this issue by allowing the model to discover task-relevant regions autonomously. Visualizations show that task prompts attend simultaneously to the button and robotic arm in Button-press, and to the overall agent in Cheetah-run.

2. **Visual Prompts**:

   - **Function**: Inject fine-grained, per-frame visual information into the conditioning signal to capture dynamic behavior.
   - **Mechanism**: A pretrained DINOv2 encoder $\mathcal{E}_V$ extracts dense visual features (rather than global representations), which are projected via a lightweight convolutional layer into $l_v = 16$ visual tokens. These tokens are concatenated with the task prompts and passed through the text encoder as the conditioning signal.
   - **Design Motivation**: Control tasks involve fine-grained motions of agents and objects, requiring per-frame conditioning to guide dynamic behavior. Global features cannot capture local action details (e.g., distinguishing front legs from hind legs), whereas dense features provide the necessary spatial granularity. Visualizations show that different visual tokens dynamically attend to distinct regions (hand, table, ball) across stages of the Relocate task.

3. **End-to-End Behavior Cloning Training**:

   - **Function**: Jointly learn the prompts and policy network during downstream policy training.
   - **Mechanism**: Given trajectories $\{I_o^i, a_o^i\}$, the objective minimizes $\mathcal{L}_{\text{BC}}(\phi, \mathbf{p}) = \sum_{i,o} \|\pi_\phi(\epsilon_\theta(z_t, t; \mathcal{C}^*)) - a_o^i\|$, where $\mathcal{C}^* = \tau_\theta(p_t; p_v)$. The diffusion model is fully frozen; only 10.6M parameters are learned (prompts + compression layer + policy network).
   - **Design Motivation**: Freezing the diffusion model prevents overfitting (experiments show full fine-tuning causes success rates to collapse by over 80%), while learnable prompts enable task adaptation—different tasks require only swapping the prompt module.

### Loss & Training
- Standard behavior cloning L1/L2 loss.
- Stable Diffusion 1.5 is used with timestep $t=0$.
- Features from downsampling blocks (down\_1–3) and the bottleneck block (mid) are concatenated.
- Each task is trained for 100 epochs with online evaluation every 10 epochs.
- Demonstrations per task: 2 for Adroit, 5 for DMC, 5 for MetaWorld.

## Key Experimental Results

### Main Results

**DeepMind Control Normalized Scores**:

| Method | Stand | Walk | Reacher | Cheetah | Finger | Mean |
|--------|-------|------|---------|---------|--------|------|
| CLIP | 87.3 | 58.3 | 54.5 | 29.9 | 67.5 | 59.5 |
| VC-1 | 86.1 | 54.3 | 18.3 | 40.9 | 65.7 | 53.1 |
| SCR (null cond.) | 85.5 | 64.3 | 81.8 | 43.4 | 66.6 | 68.3 |
| CoOp | 87.2 | 67.8 | 87.1 | 45.0 | 65.9 | 70.6 |
| TADP | 89.0 | 69.9 | 86.6 | 41.1 | 66.9 | 70.7 |
| **ORCA** | **89.1** | **76.9** | **87.6** | **50.0** | **68.0** | **74.3** |

**Adroit Success Rate (%)**:

| Method | Pen | Relocate | Mean |
|--------|-----|----------|------|
| VC-1 | 65.3 | 29.3 | 47.3 |
| SCR | 84.0 | 32.0 | 58.0 |
| TADP | 81.3 | 33.3 | 57.3 |
| **ORCA** | **86.7** | **44.0** | **65.3** |

### Ablation Study

**Component Analysis (DMC)**:

| Task Prompt | Visual Prompt | Mean Score |
|:-----------:|:------------:|:----------:|
| ✗ | ✗ | 68.3 |
| ✓ | ✗ | 69.8 |
| ✗ | ✓ | 70.5 |
| ✓ | ✓ | **74.3** |

**Fine-tuning vs. Prompt Learning (Adroit)**:

| Method | Learnable Params | Mean |
|--------|-----------------|------|
| SCR (frozen) | - | 58.0 |
| SCR + Full FT | 346.7M | 9.3 |
| SCR + LoRA | 4.6M | 60.0 |
| ORCA | 10.6M | **65.3** |

### Key Findings
- **Text conditioning is unreliable for control tasks**: Text conditioning improves performance on some tasks (Button-press) but degrades it on others (Cheetah-run), attributed to cross-attention grounding failures caused by the domain gap between diffusion model training data and control environments.
- **Task prompts and visual prompts are complementary and mutually necessary**: Each component individually improves performance by 1.5–2.2 points; combined, they yield a 6-point improvement (68.3→74.3), indicating that different tasks have distinct needs for task-level and frame-level information.
- **Full fine-tuning is catastrophic**: Full fine-tuning of SCR reduces success rate from 58% to 9.3%, whereas ORCA achieves 65.3% with only 10.6M learnable parameters, demonstrating that conditioning-based learning is far superior to parameter-level fine-tuning.
- **Early U-Net layers are more suitable for control tasks**: Features from downsampling and bottleneck layers outperform those from upsampling layers, as earlier layers encode higher-level semantic information.

## Highlights & Insights
- **Reveals the failure mode of text conditioning in the control domain**: Through cross-attention visualization, the paper clearly demonstrates how domain gap leads to text grounding failure. This finding serves as a cautionary signal for all research attempting to apply VLM conditioning to control tasks. The `<eos>` token under null conditioning already provides coarse grounding to salient targets, which explains why poorly specified text conditions can perform worse than the null condition.
- **Task prompts as implicit task descriptions**: Unlike explicit text descriptions, learnable tokens automatically discover task-relevant regions through end-to-end training, avoiding both the difficulty of manual text design and the domain gap problem. The simplicity and effectiveness of this approach are noteworthy.
- **Dynamic attention in visual prompts**: Visualizations show that different visual tokens attend to different regions across stages of the Relocate task—focusing on the table surface during the grasping phase and on the hand during the transfer phase—indicating that the model learns to dynamically shift attention according to task progress.

## Limitations & Future Work
- **Restricted to SD 1.5**: The older U-Net architecture is used; newer architectures such as DiT may require different conditioning designs.
- **Insufficient ablation on the choice of visual encoder**: The effect of alternative visual encoders (e.g., MAE, CLIP) is not thoroughly compared.
- **Evaluation limited to simulation**: The MuJoCo simulator still differs substantially from real-world robotic manipulation.
- **Limited action space**: The method is primarily validated on continuous control and simple manipulation tasks; complex long-horizon tasks remain unverified.
- **Future directions**: Exploring conditioning designs for DiT architectures; validating in real-robot scenarios; combining the semantic understanding of VLMs with ORCA's visual conditioning.

## Related Work & Insights
- **vs. SCR**: SCR first introduced Stable Diffusion to control tasks but employed null conditioning (task-agnostic); ORCA achieves task adaptation through learned conditioning, improving DMC Mean from 68.3 to 74.3.
- **vs. VPD/TADP**: VPD and TADP successfully apply text conditioning in visual perception tasks such as semantic segmentation, but this paper demonstrates that such strategies fail in control tasks due to domain gap—control environments are not natural images.
- **vs. VC-1**: VC-1 uses MAE pretraining on large-scale video data to produce powerful task-agnostic representations, yet ORCA outperforms it on all tasks, demonstrating that task-adaptive conditioning representations are superior to larger-scale agnostic pretraining.
- **Core Insight**: When transferring pretrained models to new domains, the design of conditioning signals and prompts is more critical than the choice of the model itself.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic exploration of diffusion model conditioning mechanisms in robotic control; the task + visual prompt design is both elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 tasks across 3 benchmarks, multiple baselines, comprehensive ablation studies, and rich visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation analysis via cross-attention visualization is highly persuasive; the paper is well-written throughout.
- Value: ⭐⭐⭐⭐ Provides practical guidance for visual representation design in robotics, though currently limited to simulated environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)](cdg_condition_degradation_guidance_diffusion.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[CVPR 2026\] Pixel Motion Diffusion Is What We Need for Robot Control](pixel_motion_diffusion_is_what_we_need_for_robot_control.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)

</div>

<!-- RELATED:END -->
