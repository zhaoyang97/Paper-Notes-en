---
title: >-
  [Paper Note] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training
description: >-
  [CVPR 2026][Multimodal VLM][VLM agent] This paper proposes GTR-Turbo, which generates a "free teacher model" by merging historical checkpoints produced during RL training via TIES, and uses this teacher to guide subsequent training (via SFT or KL distillation). GTR-Turbo matches or surpasses GTR—which relies on external teachers such as GPT-4o—across multiple visual agent benchmarks, while reducing training time by 50% and computational cost by 60%.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM agent
  - multi-turn reinforcement learning
  - model merging
  - knowledge distillation
  - self-evolution
date: 2026-05-08
content_hash: 3cb2bd97d6281019
---

# GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

**Conference**: CVPR 2026
**arXiv**: [2512.13043](https://arxiv.org/abs/2512.13043)
**Code**: [https://github.com/TongWei1105/GTR-Turbo](https://github.com/TongWei1105/GTR-Turbo)
**Area**: Multimodal VLM / Agent / Reinforcement Learning
**Keywords**: VLM agent, multi-turn reinforcement learning, model merging, knowledge distillation, self-evolution

## TL;DR
This paper proposes GTR-Turbo, which generates a "free teacher model" by merging historical checkpoints produced during RL training via TIES, and uses this teacher to guide subsequent training (via SFT or KL distillation). GTR-Turbo matches or surpasses GTR—which relies on external teachers such as GPT-4o—across multiple visual agent benchmarks, while reducing training time by 50% and computational cost by 60%.

## Background & Motivation

1. **Background**: Multi-turn reinforcement learning from verifiable rewards (RLVR) based on VLMs is an emerging paradigm for training visual agents, but faces core challenges including sparse rewards and long-horizon credit assignment. Methods such as GTR introduce external teacher models (e.g., GPT-4o) to provide step-level reasoning guidance, effectively addressing the "thought collapse" problem.
2. **Limitations of Prior Work**: GTR relies on expensive external teachers (GPT-4o), requiring approximately \$147 and 86 hours for 15,000 training steps; weaker teachers (e.g., Qwen2.5-VL-7B) fail to provide meaningful guidance; using a 72B model is feasible but slower (110h) and still incurs API costs.
3. **Key Challenge**: Strong teachers are necessary for effective training, yet strong teachers entail high cost and poor scalability. The central question becomes: can a model serve as its own teacher by leveraging its own training history?
4. **Goal**: Eliminate dependence on external privileged models and realize self-contained, scalable self-evolution training for VLM agents.
5. **Key Insight**: The key insight is that historical checkpoints produced during RL training, when merged, consistently outperform the current model (as shown in Figure 2), making them natural teacher candidates. This stems from the property that model merging optimizes over a smoother loss landscape and effectively retains historical experience.
6. **Core Idea**: Merge historical checkpoints from RL training into a free teacher model to replace costly external API-based teachers.

## Method

### Overall Architecture
GTR-Turbo augments the standard multi-turn PPO training loop with three key steps: (1) saving checkpoints to a buffer after each RL update; (2) merging all buffered checkpoints into a single teacher model using TIES; and (3) using the merged teacher to guide subsequent RL training via SFT loss or KL divergence. The entire process requires no external model calls.

### Key Designs

1. **TIES Merging for Teacher Construction**:

    - **Function**: Construct a "free" teacher from historical RL checkpoints whose performance consistently surpasses that of the current model.
    - **Mechanism**: At update $k$, the merged model is $\pi_{\text{merged}}^{(k)} = \sum_{i=1}^{k-1} w_i \pi_\theta^{(i)}$. The TIES method mitigates parameter interference through three steps: trimming (retaining only parameters with top-$k$% magnitude of change), sign election (determining the sign of each parameter via majority vote), and disjoint merging (averaging only parameters whose signs agree). Weighting strategies include simple moving average (SMA, uniform weights) and exponential moving average (EMA, biased toward recent models).
    - **Design Motivation**: Naively averaging all checkpoints introduces harmful interference from redundant parameters. TIES mitigates this via its trim–elect–merge pipeline, which is empirically confirmed (Figure 13) to outperform simple linear averaging.

2. **SFT Guidance (GTR-Turbo-SFT)**:

    - **Function**: Use a supervised fine-tuning loss to train the student model to imitate the reasoning process of the merged teacher.
    - **Mechanism**: After each RL step, the same observations are fed to the merged teacher to generate reference thoughts $\hat{th}$, which are stored in a replay buffer. During PPO updates, an SFT loss is added: $\min_\theta \mathbb{E}\mathcal{L}_{\text{PPO}}(o,a) + \mathbb{E}\mathcal{L}_{\text{SFT}}(o, \hat{th})$. Format rewards and DAgger are also retained.
    - **Design Motivation**: This directly inherits the GTR structure, replacing the external teacher with the merged teacher, achieving free guidance with minimal modification.

3. **KL Divergence Distillation (GTR-Turbo-KL)**:

    - **Function**: Transfer teacher knowledge more efficiently via soft-logit distillation while encouraging exploration.
    - **Mechanism**: Rather than generating autoregressive outputs from the teacher (avoiding decoding overhead), a single forward pass is performed to obtain logits. The reverse KL divergence between the agent and the teacher over thought tokens is computed, negated, and incorporated as an auxiliary reward into the PPO advantage function: $A' = A^{\pi_\theta}(o,a) - \text{RevKL}(\pi_\theta, \pi_{\text{merged}}; th)$. The KL term is clipped to $[0, +\infty)$ to prevent negative values from producing misleading gradients.
    - **Design Motivation**: SFT imposes one-hot supervision, discarding probability distribution information. KL distillation captures the full distribution over candidate tokens, providing softer constraints and greater encouragement of exploration. Furthermore, a single forward pass is required rather than full autoregressive generation, yielding faster training.

### Loss & Training
- Base model: Qwen2.5-VL-7B (SFT-initialized)
- LoRA fine-tuning for the agent; merged teacher deployed on a separate GPU
- Points24: 30,000 training steps; ALFWorld: 20,000 training steps (2× and 4× prior work budgets, respectively)
- 2× 40GB NVIDIA GPUs

## Key Experimental Results

### Main Results — Points24 Card Game

| Method | Success Rate (%) | Episode Return |
|--------|-----------------|----------------|
| GPT-4o | 2.5 | -6.35 |
| Qwen2.5-VL-72B | 5.6 | -5.69 |
| RL4VLM | 3.5 | -13.3 |
| GTR (GPT-4o teacher) | 44.5 | 0.53 |
| **GTR-Turbo (SFT)** | 48.0 | 1.32 |
| **GTR-Turbo (KL)** | **53.5** | **2.39** |

### Efficiency Comparison

| Environment | Method | Success Rate | Training Time | Additional Cost |
|-------------|--------|-------------|---------------|-----------------|
| Points24 | GTR | 41% | 191h | $307.78 |
| Points24 | GTR-Turbo (KL) | **54%** | **89h** | $114.81 |
| ALFWorld | GTR | 16% | 164h | $145.76 |
| ALFWorld | GTR-Turbo (KL) | 15% | **78h** | $100.62 |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Static initial model as KL reference | Cannot achieve stable improvement; validates the necessity of dynamic merging |
| Rejection Sampling | Completely fails on Points24; cannot generate correct trajectories for imitation |
| Guiding both thought and action | Degrades performance by restricting freedom of action exploration |
| Linear averaging vs. TIES | TIES is superior; effectively mitigates redundant parameter interference |
| KL clip vs. abs vs. K3 | Clipping is optimal; controlling KL magnitude enables finer-grained updates |

### Key Findings
- The KL distillation variant comprehensively outperforms the SFT variant—faster, stronger, and more efficient.
- Guiding only the thinking process (not the action) is critical, as agents require behavioral exploration to discover new strategies.
- The merged teacher undergoes continual self-evolution: as training progresses, the merged teacher also improves, forming a positive feedback loop.
- GTR-Turbo (KL) achieves training time comparable to the simplest RL4VLM baseline while substantially exceeding its performance.

## Highlights & Insights
- **An elegant "free lunch"**: The observation that merged historical RL checkpoints naturally serve as stronger teachers than the current model is both simple and profound—analogous to how Stochastic Weight Averaging (SWA) improves generalization in supervised learning.
- **KL distillation over autoregressive generation**: Replacing full autoregressive decoding with a single forward pass is not only faster but also more effective, demonstrating that soft labels outperform hard labels in self-evolution settings.
- **Transferable paradigm**: The "merge historical checkpoints as teacher" trick is in principle applicable to any multi-turn RL training scenario, not limited to VLM agents.

## Limitations & Future Work
- On long-horizon tasks such as ALFWorld (50+ steps), the advantage of the merged teacher is less pronounced than on Points24, due to the absence of external domain knowledge.
- An additional GPU is still required to host the merged teacher, which, while cheaper than API calls, is not zero-cost.
- The merging interval and TIES hyperparameters require tuning.
- Hybrid approaches combining the merged teacher with small-scale external teachers remain unexplored.

## Related Work & Insights
- **vs. GTR**: GTR uses GPT-4o as a teacher with strong results but at prohibitive cost. GTR-Turbo replaces it with merged checkpoints at no external cost, reducing expense by 60% while achieving superior performance on Points24.
- **vs. RL4VLM**: Direct PPO training causes thought collapse, leading to repetitive and templated model outputs. GTR-Turbo effectively addresses this through reasoning guidance.
- **vs. Rejection Sampling**: Rejection sampling requires the model to self-generate correct trajectories, a premise that fails on difficult tasks. The combination of RL exploration and merged-teacher guidance proves to be a more robust alternative.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using merged checkpoints as a teacher is elegant and concise; KL distillation as a replacement for SFT is also a well-motivated design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two environments, multiple ablations, cost analysis, and training curves are all comprehensively presented.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Substantially reduces the training cost of VLM agents; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](clip-free_label_free_unsupervised_concept_bottleneck_models.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
