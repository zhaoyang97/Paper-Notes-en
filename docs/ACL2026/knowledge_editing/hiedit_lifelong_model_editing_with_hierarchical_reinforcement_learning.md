---
title: >-
  [Paper Note] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning
description: >-
  [ACL 2026][Knowledge Editing][lifelong model editing] HiEdit utilizes hierarchical reinforcement learning to decompose "lifelong model editing" into two sub-tasks: high-level layer selection and low-level gradient update…
tags:
  - "ACL 2026"
  - "Knowledge Editing"
  - "lifelong model editing"
  - "hierarchical RL"
  - "hypernetwork"
  - "layer selection"
  - "sparse update"
date: 2026-05-08
content_hash: 6931f839a233cfae
---

# HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2604.11214](https://arxiv.org/abs/2604.11214)  
**Code**: https://github.com/yangfanww/hiedit  
**Area**: Knowledge Editing / Lifelong Learning / Hierarchical Reinforcement Learning  
**Keywords**: lifelong model editing, hierarchical RL, hypernetwork, layer selection, sparse update

## TL;DR
HiEdit utilizes hierarchical reinforcement learning to decompose "lifelong model editing" into two sub-tasks: high-level layer selection and low-level gradient update calculation. This allows the hypernetwork to adaptively modify only half of the layers based on specific knowledge, improving the strong baseline RLEdit by an average of 8.48%.

## Background & Motivation

**Background**: Lifelong Model Editing (LME) aims to continuously inject new knowledge into deployed LLMs without retraining. The dominant paradigm is "locate-then-edit": first identifying the influential layers $\mathcal{W}$, and then applying perturbations $\tilde{\nabla}_{\mathcal{W}}$. Recently, RLEdit modeled the entire sequence editing process as an RL task, allowing a hypernetwork to optimize via PPO-style training across tens of thousands of edits.

**Limitations of Prior Work**: All existing methods apply perturbations to a "static and dense" set of layers—regardless of the knowledge being edited, the same set of layers (often 5–7 MLP layers) is updated. On Llama-3-8B running ZsRE sequential editing, generalization and performance on past edits begin to collapse (catastrophic forgetting) after 5,000 steps.

**Key Challenge**: Research has shown that different knowledge activates different components in LLMs. However, using the same set of layers for all instances during the "locating" phase results in excessive modification of parameters unrelated to the current knowledge; it also unnecessarily restricts the hypernetwork's optimization space to a sub-optimal solution.

**Goal**: Transform the decision of "which layers to edit" from a static, offline-fixed choice into a learnable, dynamic decision for each item of knowledge.

**Key Insight**: Upgrade LME from a flat MDP to a Hierarchical MDP—where high-level options select layers and low-level actions compute parameter updates, decoupling discrete layer selection from continuous updates during training.

**Core Idea**: Use hierarchical RL to decouple "where to edit" and "how to edit," incorporating an intrinsic reward to encourage sparse selection, thereby achieving instance-aware, localized, and precise editing.

## Method

### Overall Architecture
HiEdit models each editing step as a Hierarchical MDP $(\mathcal{S}, \mathcal{A}, \Omega, \mathcal{P}, r, \gamma)$. At step $t$:

1. Given a pair of knowledge to be edited $(x_t, y_t)$, a standard SFT is performed to obtain the gradient matrices for all potential influential layers $\nabla \mathcal{W}_t = \{\nabla \mathcal{W}_{t,1}, \dots, \nabla \mathcal{W}_{t,L}\}$. Each layer's gradient is decomposed via low-rank decomposition as $\nabla \mathcal{W}_{t,l} = v_l u_l^\top$.
2. The high-level hypernetwork $\pi_\phi$ processes all $(u_l, v_l)$ and outputs an option $\omega_t \in \{0,1\}^L$ (a mask), activating only $K$ layers.
3. The low-level hypernetwork $\mathcal{H}_\theta$ operates only on the activated layers using a MEND/MALMEN-style editing network, mapping $(u_l, v_l)$ to pseudo-vectors $(\tilde u_l, \tilde v_l)$ to obtain parameter updates $\tilde{\nabla} \mathcal{W}_{t,l} = \tilde v_l \tilde u_l^\top$.
4. After the entire editing trajectory, both hypernetworks are optimized jointly based on high-level and low-level rewards.

### Key Designs

1. **High-level importance router (Dynamic Layer Selection)**:

    - **Function**: Dynamically decides which layers to modify based on the current knowledge.
    - **Mechanism**: Each layer's $(u_l \| v_l)$ is passed through a layer-shared gradient encoder $\mathbf{W}_{\text{GradEnc}}$ + layer-specific scale & offset $\mathbf{SPE}_l$ to obtain $h_l$. These are concatenated and passed through a gate network to output $z_t \in \mathbb{R}^L$. Finally, $\mathbf{TopK}(z_t, K)$ generates the mask $m_t$. To maintain differentiability for the discrete TopK, a straight-through estimator inspired by MoE is used: $m_t = \mathbf{sg}(m_t - z_t) + z_t$, where the hard mask is used in the forward pass and gradients are passed to $z_t$ in the backward pass.
    - **Design Motivation**: Make layer selection a learnable high-level action rather than an offline fixed decision, allowing different knowledge to follow different layer paths.

2. **Intrinsic reward for sparsity**:

    - **Function**: Encourages the high-level to "complete the edit using as few layers as possible."
    - **Mechanism**: The high-level reward is not an absolute value but the relative advantage of partial selection over full selection: $r_{\text{high},t} = r_{\text{low}}(s_t, \omega_t, a_t) - r_{\text{low}}(s_t, \mathbf{1}, a_t)$, where $\mathbf{1}$ denotes selecting all layers. If the editing loss with partial selection is lower than with full selection, the high-level receives a positive reward.
    - **Design Motivation**: Since direct supervision of "fewer is better" is difficult, modeling the signal as the difference between "partial vs. full" is equivalent to learning a causal contrastive signal of "whether this layer is actually useful for this knowledge," naturally encouraging sparsity without sacrificing performance.

3. **Hierarchical joint optimization + loss regularization**:

    - **Function**: Enables end-to-end collaborative training of high-level and low-level hypernetworks.
    - **Mechanism**: The low-level reward is the negative total loss $r_{\text{low},t} = -\mathcal{L}_t$, where $\mathcal{L}_t = \eta \|\tilde{\nabla} \mathcal{W}_t\|^2 + \sum_{i=t-k}^t \mu^{t-i} \mathcal{L}_{t,i}$. Here $\mathcal{L}_{t,i} = -\log p_{\mathcal{W}_t}(y_i|x_i) + \tilde\lambda \mathrm{KL}[p_{\mathcal{W}_{t-1}}(\cdot|\tilde x_i) \| p_{\mathcal{W}_t}(\cdot|\tilde x_i)]$, using memory backtracking over $k$ previous edits to prevent forgetting. The KL term constrains the distribution of unrelated inputs to remain unchanged. Both hypernetworks are optimized according to the cumulative discounted reward $\sum \gamma^t r_{\beta,t}$, with $\gamma=1$ ensuring consistent weights across the sequence.
    - **Design Motivation**: Sparse masks block standard backpropagation; thus, straight-through estimation and joint optimization are required to ensure high-level exploration and low-level exploitation can calibrate each other.

### Loss & Training
- The $K$ for TopK during training is consistent with inference to enforce sparsity-consistency and avoid distribution shifts.
- Gradients are backpropagated after a complete trajectory on an editing sequence (trajectory-level update) for better stability.

## Key Experimental Results

### Main Results
Evaluated on Llama-3-8B and Gemma-2-9B backbones across ZsRE and CounterFact datasets with 20,000 sequential edits, HiEdit is compared against 11 baselines (FT, ROME, MEMIT, PRUNE, RECT, AlphaEdit, MEND, MALMEN, DAFNet, RLEdit, etc.). Metrics include Efficacy, Generalization, Specificity, and Retention.

| Model | Method | ZsRE-Eff. | ZsRE-Gen. | ZsRE-Spe. | ZsRE-Ret. | CounterFact-Eff. |
|------|------|-----------|-----------|-----------|-----------|------------------|
| Llama-3-8B | RLEdit | 81.43 | 79.49 | 42.73 | 70.72 | 66.35 |
| Llama-3-8B | HiEdit (rand) | 81.95 | 79.63 | 47.97 | 74.66 | 66.40 |
| Llama-3-8B | Ours (full) | **82.10** | **79.99** | **48.42** | **75.16** | **66.53** |
| Gemma-2-9B | AlphaEdit | 15.79 | 15.32 | 20.21 | 13.19 | 38.17 |
| Gemma-2-9B | RECT | 11.26 | 11.25 | 16.19 | 9.62 | 30.72 |

In long-range editing settings, AlphaEdit, ROME, and MEMIT scores drop toward zero on Gemma-2-9B. HiEdit improves the average score of RLEdit by 8.48% on ZsRE while only modifying half the layers per edit ($K=L/2$).

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| Ours-full | 82.10 / 75.16 | Learned TopK mask used in both training and inference |
| Ours-rand | 81.95 / 74.66 | Replaced with random mask (same K) during inference $\rightarrow$ still better than RLEdit, but Retention drops by 0.5 |
| RLEdit (dense, all layers) | 81.43 / 70.72 | No layer selection; Retention decreases significantly |
| Static fixed K layers | < 81 | Fixed set of layers (not learned); significantly worse than HiEdit |

### Key Findings
- **Intrinsic reward is crucial**: Without the relative advantage reward (standard absolute reward maximization), the high-level tends to select all layers, degrading to RLEdit.
- **Random masks provide gains**: Even random masks show benefits, indicating that sparse updates themselves mitigate over-modification. However, learned masks provide an additional 0.5–1.0 points in Retention (preservation of past edits).
- **Scalability**: Under 20k-step long-range editing, traditional closed-form methods (PRUNE/RECT) fail completely, whereas hypernetwork-based methods (RLEdit, HiEdit) remain stable—confirming that the parameter update space requires structured exploration.

## Highlights & Insights
- **Applying MoE Router logic to LME**: Editors, like MoE, face the problem of "which expert/layer to route to." The gate + straight-through approach migrates seamlessly and provides an elegant cross-task application.
- **Intrinsic Reward Design**: Using the relative advantage of "partial vs. full" instead of an absolute sparsity penalty avoids manual tuning of sparsity coefficients, providing a template for tasks aimed at learning sparse masks.
- **Decoupled Where/How Training Paradigm**: The hierarchical MDP breaks down the massive $\{0,1\}^L \times \mathbb{R}^{d \times d}$ action space, allowing the high-level to use RL while the low-level uses gradients, serving as a clean example of mixed RL+SL optimization.

## Limitations & Future Work
- The high-level router only observes the current gradient signal without explicit input from "past editing history." Whether it maintains stable mask distributions over extremely long ranges beyond 20k steps remains to be seen.
- $K$ is a fixed hyperparameter ($K=L/2$ in experiments); the model does not adaptively determine how many layers to activate per step. Ideally, $K$ should also be learnable.
- Validation is limited to hypernetwork-based methods; integration with closed-form methods (e.g., AlphaEdit) has not yet been explored.

## Related Work & Insights
- **vs RLEdit**: Upgrades flat RL to hierarchical RL by adding learnable layer selection. Swapping the HiEdit framework onto the same hypernetwork backbone leads to direct gains.
- **vs AlphaEdit / MEMIT (closed-form)**: Closed-form methods tend to explode in long-range editing. HiEdit follows a learnable path, which is better suited for streaming LME scenarios.
- **vs MoE routing**: Shares the same core idea (gate + TopK + straight-through), but shifts the objective from "expert selection" to "knowledge-relevant layer selection."

## Rating
- Novelty: ⭐⭐⭐⭐ First work to model LME as a hierarchical MDP with a clever intrinsic reward design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 backbones × 2 datasets × 11 baselines under rigorous long-range editing settings.
- Writing Quality: ⭐⭐⭐⭐ The motivation for HRL is clear, and despite many formulas, the logic is consistent.
- Value: ⭐⭐⭐⭐ Long-range model editing is a critical requirement for deployment; few methods reliably surpass the 5k-step forgetting threshold.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Training Examples](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[AAAI 2026\] Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning](../../AAAI2026/knowledge_editing/is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)

</div>

<!-- RELATED:END -->
