---
title: >-
  [Paper Note] Contrastive Representation Regularization for Vision-Language-Action Models
description: >-
  [ICML 2026][Robotics][Vision-Language-Action Models] The authors identified that representations in VLA models inherited from VLMs are dominated by visual appearance and insensitive to robot proprioceptive states. They p…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Vision-Language-Action Models"
  - "Proprioceptive Contrastive Learning"
  - "Representation Regularization"
  - "view cutoff"
  - "GR00T"
date: 2026-05-08
content_hash: 0fb201bdbc4274b9
---

# Contrastive Representation Regularization for Vision-Language-Action Models

**Conference**: ICML 2026  
**arXiv**: [2510.01711](https://arxiv.org/abs/2510.01711)  
**Code**: To be confirmed  
**Area**: Robotics / VLA / Representation Learning  
**Keywords**: Vision-Language-Action Models, Proprioceptive Contrastive Learning, Representation Regularization, view cutoff, GR00T

## TL;DR
The authors identified that representations in VLA models inherited from VLMs are dominated by visual appearance and insensitive to robot proprioceptive states. They proposed Robot State-aware Contrastive Loss (RS-CL), which treats the Euclidean distance between proprioceptive states as "soft contrastive labels" to reshape representations. Combined with "view cutoff" feature-level augmentation, they achieved SOTA results with 69.7% on RoboCasa-Kitchen using GR00T N1.5 and increased the success rate on real-world Franka pick-and-place tasks from 45.0% to 58.3%.

## Background & Motivation
**Background**: Current SOTA VLA models ($\pi_0$, GR00T N1.5, $\pi_0$-FAST, etc.) almost exclusively follow the paradigm of "pre-trained VLM + generative action decoder (DiT + flow matching)," supervised end-to-end using action prediction loss.

**Limitations of Prior Work**: VLMs are pre-trained on internet-scale visual instruction data and have never encountered low-level control actions or proprioceptive states. Using frozen VLMs as conditions limits downstream action precision; even with joint fine-tuning, representations remain dominated by scene backgrounds and object appearances, remaining largely insensitive to "the robot's current pose and next action" (t-SNE in Fig. 2b shows trajectories of the same task in different scenes clustered by scene rather than task progress).

**Key Challenge**: The model must preserve VLM semantic priors while making representations sensitive to control signals. However, action prediction loss is an indirect signal, and gradients are "diluted" by the decoder when updating the VLM backbone, making it difficult to directly reshape representation geometry.

**Goal**: To add a lightweight regularization to the standard VLA pipeline that explicitly aligns VLM representations with robot proprioceptive states, without introducing extra training stages or relying on external robotics datasets.

**Key Insight**: The essence of contrastive learning lies in the definition of positive and negative samples—CLIP uses image-text pairs, TCN uses temporal neighbors, and R3M/VIP use reward proximity. The authors ask: what is the "natural similarity signal" for robots? The answer is proprioceptive state: physically closer poses have closer action distributions and should be mapped closer in representation space.

**Core Idea**: Treat the continuous distance between proprioceptive states as contrastive soft labels. Instead of binary positive/negative labels, use $w_{ij} \propto \exp(-\|\mathbf{q}_i - \mathbf{q}_j\|_2 / \beta)$ to assign a soft weight to every pair of samples, enabling single-stage end-to-end joint training of action prediction and RS-CL.

## Method
RS-CL attaches a "contrastive regularization path" to the standard VLA pipeline. Compared to the original GR00T N1.5, it only adds a summarization token, a 2-layer MLP projector, and view cutoff augmentation, leaving the main path largely unchanged.

### Overall Architecture
- **Input**: Observations from $V$ views $\mathbf{O}_t^V$, task instruction $\mathbf{c}$, and proprioceptive state $\mathbf{q}$.
- **VLM + adapter**: Frozen VLM with a trained adapter $f_\phi$, outputting $\mathbf{h} \in \mathbb{R}^{N \times d_{\text{model}}}$.
- **Action decoder $D_\theta$**: DiT architecture fitting the action chunk $\mathbf{A}_t$ for the next horizon $H$ with a flow-matching objective.
- **Regularization Path**: A learnable summarization token $\mathbf{u}$ is appended to the VLM output to obtain $\mathbf{w}$, which passes through projector $g_\psi$ to produce $\mathbf{z}$. View cutoff augmentation produces $\tilde{\mathbf{z}}$, and RS-CL is applied between $\mathbf{z}$ and $\tilde{\mathbf{z}}$.
- **Training**: End-to-end, single-stage, with $\lambda$ following a cosine schedule decaying from 1.0 to 0—strong representation shaping in the early phase and pure action prediction in the later phase.

### Key Designs

1. **Summarization token $\mathbf{u}$ for Amortizing VLM Representations**:
    - **Function**: Compresses the VLM output of length $N$ (typically large) into a single token to avoid computational explosion when performing contrastive learning on full sequences.
    - **Mechanism**: A learnable token is concatenated to the VLM output, allowing adapter $f_\phi$ to produce it as a "sequence summary": $[\mathbf{h}, \mathbf{w}] = f_\phi(\text{VLM}(\mathbf{O}_t^V, \mathbf{c}) \oplus \mathbf{u})$. A 2-layer MLP projector $g_\psi$ then projects it to 128-dimensional $\mathbf{z}$. $\mathbf{h}$ follows the original path to the action decoder, while $\mathbf{w}$ follows the new contrastive path, fully decoupling the two.
    - **Design Motivation**: Contrastive loss on all $N$ tokens is expensive and dilutes the signal. BERT-style [CLS] tokens are proven effective in VLA scenarios. The projector is standard SimCLR practice to prevent representation space pollution by the contrastive objective.

2. **Robot State-aware Contrastive Loss (RS-CL)**:
    - **Function**: Uses continuous distance of proprioceptive states as soft supervision to shift the representation space from "clustering by visual appearance" to "clustering by control state."
    - **Mechanism**: Computes InfoNCE-style similarity for all pairs in a batch of size $B$, but with soft weights: 
    $$\mathcal{L}_{\text{RS-CL}} = -\sum_{i,j=1}^{B} w_{ij} \log \frac{e^{{\text{sim}}(\mathbf{z}_i, \tilde{\mathbf{z}}_j)/\tau}}{\sum_{k=1}^{B} e^{{\text{sim}}(\mathbf{z}_i, \tilde{\mathbf{z}}_k)/\tau}}$$
    where weights $w_{ij} = \frac{e^{-\|\mathbf{q}_i - \mathbf{q}_j\|_2 / \beta}}{\sum_k e^{-\|\mathbf{q}_i - \mathbf{q}_k\|_2 / \beta}}$ are derived from proprioceptive state $\mathbf{q}$ (end-effector $x,y,z$ + 6D rotation + gripper, min-max normalized to $[-1,1]$). $\beta$ controls weight mapping sharpness, and $\tau$ controls similarity sharpness.
    - **Design Motivation**: Hard positive/negative pairs (like SupCon) require discrete labels, but proprioception is continuous. Soft weights pull "nearly identical pose" samples closer and push "opposite pose" samples further apart, ensuring smooth transitions without manual thresholds. This embeds the "physical structure of the robot" directly into the representation geometry.

3. **View cutoff Representation-level Augmentation**:
    - **Function**: Generates contrastive positive samples without increasing VLM forward passes, making representations robust to missing views.
    - **Mechanism**: Randomly selects a view index $i \in \{1, \dots, V\}$ and masks the corresponding feature slice in the VLM output to obtain $\tilde{\mathbf{z}}$. Only the adapter $f_\phi$ and projector $g_\psi$ need to be re-run; VLM forward is not repeated.
    - **Design Motivation**: Traditional data-level augmentation (cropping, jittering) requires another forward pass, doubling computation for backbones like GR00T-N1.5. View cutoff exploits the natural multi-view structure of VLAs, treating "losing a camera" as a hard sample—cheaply teaching the model robustness against occlusions in real deployments.

### Loss & Training
The total objective is $\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \, \mathcal{L}_{\text{RS-CL}}$, where the flow-matching loss is:
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_s [\|D_\theta(\mathbf{h}, \mathbf{A}_t^s, \mathbf{q}) - (\epsilon - \mathbf{A}_t)\|_2^2]$$
where $\mathbf{A}_t^s = s \mathbf{A}_t + (1-s) \epsilon$ is the interpolated action chunk. $\lambda$ starts at 1.0 and decays to 0 via cosine schedule, implying early-stage representation reshaping and late-stage focus on action precision. Simulation experiments were conducted on RoboCasa-Kitchen (30/100/300 demos) and LIBERO. Real experiments used Franka Research 3 with two camera views across 5 tasks.

## Key Experimental Results

### Main Results
| Benchmark | Method | Success Rate (%) |
|-----------|--------|------------------|
| RoboCasa-Kitchen (300 demos) | GR00T N1.5 baseline | 65.7 |
| RoboCasa-Kitchen | $\pi_0$ | 62.5 |
| RoboCasa-Kitchen | $\pi_0$-FAST | 63.6 |
| RoboCasa-Kitchen | FLARE | 66.4 |
| RoboCasa-Kitchen | GR00T N1.5 + HAMLET | 66.4 |
| **RoboCasa-Kitchen** | **GR00T N1.5 + RS-CL** | **69.7** |
| RoboCasa pick-and-place | baseline | 30.3 |
| RoboCasa pick-and-place | + RS-CL | 41.5 (+11.2) |
| Real robot (Avg of 5 tasks) | baseline | 45.0 |
| Real robot | + RS-CL | 58.3 (+13.3) |
| LIBERO Avg | GR00T N1.5 | 95.7 |
| LIBERO Avg | + RS-CL | 96.4 |

On LIBERO, where the baseline is already near the ceiling (95+), RS-CL still raised the Long-horizon suite from 87.8 to 90.4, indicating its primary advantage lies in scenarios where "action precision is the bottleneck."

### Ablation Study
| Config | RoboCasa-Kitchen 30 demos SR (%) | FLOPs ($\times 10^{12}$) |
|--------|----------------------------------|--------------------------|
| GR00T N1.5 baseline | 48.2 | 2.58 |
| + Multi-view TCN | 50.0 | 7.53 |
| + Single-view TCN | 50.3 | 7.53 |
| **+ RS-CL (ours)** | **53.0** | **2.91** |

TCN (time-contrastive networks), the most direct baseline, only improved accuracy by 1.8–2.1% compared to the baseline, but tripled FLOPs to 7.53 because it requires re-forwarding the VLM for temporal pairs. RS-CL achieved +4.8% accuracy with only 0.33 extra FLOPs. From-scratch experiments (Fig. 7) across Qwen2.5-VL and SigLIP2 backbones confirmed the gain is backbone-agnostic.

### Key Findings
- The representation bottleneck is "control relevance" rather than "semantic richness": t-SNE shows original VLM representations cluster by scene appearance, whereas RS-CL clusters by task progress (robot pose).
- Pick-and-place tasks see the largest gains (+11.2): These tasks are sensitive to end-effector precision; proprioceptive alignment translates directly into localization accuracy.
- View cutoff is more than a computational trick: In close-lid tasks where the wrist camera is occluded, view-cutoff-trained models showed significantly higher success rates than the baseline.
- Cosine decay of $\lambda$ is essential: Early $\lambda=1.0$ sets the representation geometry, while late $\lambda \to 0$ cedes all gradients to flow-matching for action refinement.
- Real hardware gains (+13.3%) are significant: Moving from 45.0% to 58.3% crosses the threshold from "prototype" to "demonstrable."

## Highlights & Insights
- **Embedding physical structure into representation space**: Using proprioceptive distance as soft contrastive labels is an elegant form of "prior injection"—it requires no manual labels or external rewards, utilizing native sensor readings. This can generalize to any embodied task with continuous states (e.g., vehicle pose for autonomous driving).
- **View cutoff as an exemplar of representation-level augmentation**: As VLA backbones grow, re-forwarding for data augmentation becomes a luxury. Moving augmentation from "input space" to "feature space" is a universal path for optimization.
- **Phase folding**: The $\lambda$ decay collapses "representation learning $\to$ action refinement" from two stages into one training run, reducing maintenance costs.
- **Logic against TCN**: The comparison (+2.7 ACC with $\sim$2.5$\times$ compute savings) clearly answers why specialized robot-aware contrastive losses are needed.

## Limitations & Future Work
- Proprioceptive state selection remains empirical: end-effector pose for pick-and-place vs. joint positions for close-lid; no systematic selection rule is provided. For high-DoF tasks (e.g., 21+ DoF dexterous hands), uniform Euclidean weighting $\|\mathbf{q}_i - \mathbf{q}_j\|_2$ might be distorted.
- Only validated on single-arm 6-7 DoF manipulation; dual-arm, mobile manipulation, and quadrupeds are untested.
- Relationship with explicit world models/dynamics (e.g., V-JEPA) is undiscussed—they could be complementary or redundant.
- Success rate variance under low-data scenarios is not thoroughly detailed with multiple seeds.

## Related Work & Insights
- **vs $\pi_0$ / GR00T N1.5**: Original VLAs rely on action prediction loss to update the VLM; the authors prove this signal is too weak/late. RS-CL provides explicit supervision at the representation layer.
- **vs TCN (Sermanet 2018)**: TCN uses temporal neighbors requiring extra forwards and data mining. RS-CL uses proprioception for single-pass training with better performance (53.0 vs 50.3) and lower cost (2.91 vs 7.53).
- **vs R3M / VIP**: These require an offline pre-training phase on large datasets (Ego4D). RS-CL hitchhikes on the downstream VLA training without extra data.
- **vs DUST / HAMLET / FLARE**: These focus on video strategies or flow modifications. RS-CL gets higher gains (+4 ACC) using simple contrastive regularization, suggesting proprioception is an undervalued supervision source.
- **vs CORE / Cosmos**: These rely on massive robotics-specific datasets. RS-CL takes the opposite route—adding no data, only reshaping representation geometry.

## Rating
- Novelty: ⭐⭐⭐⭐ The specific form of soft contrastive labels using proprioceptive distance is novel, though it follows the SupCon + R3M lineage.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of simulation, real robots, and backbones. Missing failure case analysis.
- Writing Quality: ⭐⭐⭐⭐ The motivation (VLM bottleneck $\to$ contrastive choice $\to$ soft labels) is very clear; formulas are clean.
- Value: ⭐⭐⭐⭐ Plug-and-play with minimal overhead; consistently improves mainstream VLA pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning](specprune-vla_accelerating_vision-language-action_models_via_action-aware_self-s.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)

</div>

<!-- RELATED:END -->
