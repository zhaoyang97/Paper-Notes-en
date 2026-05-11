---
title: >-
  [Paper Note] Sparse Imagination for Efficient Visual World Model Planning
description: >-
  [ICLR 2026][Robotics][world model] This paper proposes Sparse Imagination, which achieves substantial inference speedup in ViT patch token-based world model planning by randomly dropping tokens and training with randomly…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "world model"
  - "sparse tokens"
  - "MPC"
  - "DINO"
  - "VLA"
  - "token dropout"
  - "planning efficiency"
date: 2026-05-08
content_hash: ce8f83bec9377aa2
---

# Sparse Imagination for Efficient Visual World Model Planning

**Conference**: ICLR 2026
**arXiv**: [2506.01392](https://arxiv.org/abs/2506.01392)
**Code**: None (built on DINO-WM framework)
**Area**: Robotics
**Keywords**: world model, sparse tokens, MPC, DINO, VLA, token dropout, planning efficiency

## TL;DR
This paper proposes Sparse Imagination, which achieves substantial inference speedup in ViT patch token-based world model planning by randomly dropping tokens and training with randomly grouped attention (50% drop rate reduces planning time by ~50%), while maintaining or even surpassing full-token planning performance on certain tasks. A key finding is that simple random dropout outperforms sophisticated token selection methods, as static importance ranking suffers from a "blind spot problem" in dynamic planning scenarios.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: 1. World model-based planning (MPC) enables decision-making by imagining future trajectories, but computational cost scales quadratically with token count — each planning step requires $K \times M \times H$ world model forward passes. 2. ViT patch tokens as visual state representations (e.g., DINO-WM) retain richer spatial information than a single CLS token, offering clear advantages in fine-grained manipulation tasks. 3. However, the computational overhead of full patch tokens in MPC makes real-time deployment extremely difficult, especially in compute-constrained settings such as robotics. 4. ViT representations are known to be redundant — multiple studies (Raghu et al., Pan et al., Kim et al.) demonstrate that not all patches are equally important for downstream tasks. 5. Compute resources are particularly limited in robotic settings (embedded GPUs), requiring substantial reduction in inference cost without sacrificing accuracy. 6. Existing token pruning methods (attention ranking / learned selection / merging / training-time dropout) are effective on static tasks such as classification but have not been validated in iterative, dynamic planning scenarios.

## Method

- **World Model Architecture**: A frozen pretrained DINO encoder extracts visual patch tokens $z_t \in \mathbb{R}^{H_p \times W_p \times D}$; a causal Transformer decoder predicts future token sequences.
- **Training Loss**: MSE prediction loss $\mathcal{L}_{wm} = \frac{1}{N}\sum_{i=1}^N \|\hat{z}_{t+1,i} - z_{t+1,i}\|^2$; goal distance is also measured with MSE.
- **Sparse Imagination**: At world model inference time, a fraction $p$ of patch tokens is randomly dropped, and forward prediction is performed using only $(1-p)N$ tokens.
- **Randomly Grouped Attention Training**: During training, tokens of each frame are randomly partitioned into two groups; attention masks restrict interactions to within each group, enabling the model to handle arbitrary token subsets. Group assignments are kept consistent along the temporal dimension.
- **MPC Integration**: The dropout mask is resampled independently at each planning step; both prediction and CEM optimization are performed on the sparse token set.
- **VLA-Guided Planning**: For long-horizon tasks, $K$ candidate action sequences are sampled from a pretrained VLA (SmolVLA) to replace CEM's random sampling, substantially improving long-horizon planning efficiency.
- **Key Finding**: Simple random sampling outperforms complex attention-based or learned ranking methods, because static importance metrics exhibit "blind spots" in dynamic planning — patches that appear unimportant at the current state may become critical when evaluating candidate action sequences. The unbiased coverage of random sampling avoids systematic omissions.

## Key Experimental Results

### Simple Environments (MPC-CEM / CEM)

### Main Results

| Environment | Full (p=0) | Drop 30% | Drop 50% | CLS-token | Notes |
|---|---|---|---|---|---|
| Pointmaze | 98.3% | 98.3% | **100%** | 96.7% | Sparse surpasses full |
| Wall | 91.7% | 93.3% | 95.0% | 85.0% | Sparse outperforms full |
| PushT | 75.0% | 61.7% | 70.0% | 43.3% | 50% drop near full |
| Granular | 75.0% | **85.0%** | 60.0% | 20.0% | 30% drop surpasses full |
| Rope | 63.3% | 70.0% | 73.3% | 36.7% | Sparse significantly outperforms CLS |
| Block Push | 22.0% | 18.0% | 20.0% | 16.0% | Smaller gap on hard task |

### Complex Environments + Real World (VLA-Guided Planning)

### Ablation Study

| Task | Full | Drop 50% | VLA-only | Time (Full→Drop) |
|---|---|---|---|---|
| PickPlace (real) | - | **80%** | 60% | 19.1s→10.4s |
| Drawer (real) | - | **70%** | 60% | 14.0s→10.6s |
| LIBERO-10 | 34% | 33% | 29% | 53.4s→29.7s |
| Meta-World | 48.8% | 47.7% | 42.7% | 3.63s→2.37s |

### Planning Time Speedup

| Environment | Full Time | Drop 50% Time | Speedup |
|---|---|---|---|
| PushT | 173s/iter | 82s/iter | **52.6%** |
| Pointmaze | 184s/iter | 102s/iter | **44.6%** |
| Block Push | 297s/iter | 161s/iter | **45.8%** |

## Highlights & Insights
- Remarkably simple and elegant: substantial speedup is achieved via random dropout alone, with no additional components required.
- The "blind spot problem" analysis is insightful — it explains why complex token selection underperforms random sampling.
- Strong generality: validated across simple trajectory optimization, VLA-guided planning, and real-robot deployment.
- The grouped attention training strategy can be seamlessly integrated into any Transformer-based world model.

## Ablation Study & In-Depth Analysis

| Ablation / Analysis | Result |
|---|---|
| With vs. without grouped attention training | Without grouped attention, 50% drop causes severe degradation (PushT: 70→35%); grouped attention is a necessary condition. |
| Random vs. attention ranking vs. learned ranking | Random sampling is competitive or superior on most tasks — static ranking fails due to the "blind spot problem." |
| Drop ratio sweet spot | 10–50% is the optimal range; >70% leads to clear degradation. |
| VLA-guided vs. CEM random sampling | VLA guidance improves long-horizon tasks by ~4–7% while reducing computation by ~40%. |
| Train-only sparse vs. inference-only sparse | Both are required: training sparsity ensures model adaptation; inference sparsity provides the speedup. |

### In-Depth Analysis of the "Blind Spot Problem"
- Static importance metrics (e.g., attention weights, CLS token correlation) introduce systematic blind spots during MPC's iterative optimization.
- Specifically, patches that appear unimportant at the current state may become critical when evaluating candidate action sequences.
- Random sampling avoids systematic omissions through unbiased coverage — resampling the mask at each iteration ensures every region has a nonzero probability of being included.
- This finding contradicts the common conclusion in the token pruning literature that "learned selection outperforms random," highlighting the unique characteristics of planning scenarios.

## Limitations & Future Work
- The optimal drop ratio must be tuned manually per task; an adaptive mechanism is lacking. A potential improvement is to dynamically adjust the ratio based on task complexity or the current state.
- The number of groups is fixed at 2; the effect of more groups (e.g., 3–4) is unexplored.
- The method relies on the redundancy assumption of DINO features, which may not hold in information-dense scenarios (e.g., text-heavy interfaces).
- Real-world validation is limited to two relatively simple tasks (PickPlace + Drawer); more complex manipulation tasks are not tested.
- Integration with token merging methods (e.g., ToMe) is unexplored — combining sparse selection with merging could further improve efficiency.

## Related Work & Insights
- **vs. Dreamer series (Hafner et al.)**: Dreamer imagines in a low-dimensional vector latent space, whereas this paper imagines in a high-dimensional patch token space — retaining richer spatial information at greater computational cost. Sparse Imagination bridges this gap.
- **vs. DINO-WM (Zhou et al. 2024)**: This work builds directly on DINO-WM and addresses its computational bottleneck via sparse imagination.
- **vs. ToMe (Bolya et al.)**: ToMe reduces computation through token merging; this work uses token dropping — a simpler design that requires no additional merging logic.
- **vs. SmolVLA (Shukor et al.)**: SmolVLA provides a pretrained policy for guided planning; sparse imagination accelerates the world model evaluation under VLA-guided planning.
- **Insights**: The sparse imagination paradigm can be generalized to other settings requiring many forward passes — e.g., value network evaluation in MCTS search, or world simulation in multi-step reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet effective insight; the blind spot problem analysis offers unique value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 simulation + 2 real-world tasks, multi-method comparisons, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, polished figures, intuitive method diagrams.
- Value: ⭐⭐⭐⭐ Practical contribution that can be directly integrated into any Transformer-based world model pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Building Spatial World Models from Sparse Transitional Episodic Memories](building_spatial_world_models_from_sparse_transitional_episodic_memories.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](../../CVPR2026/robotics/chain_of_world_world_model_thinking_in_latent_motion.md)
- [\[ICLR 2026\] Visual Planning: Let's Think Only with Images](visual_planning_lets_think_only_with_images.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)

</div>

<!-- RELATED:END -->
