---
title: >-
  [Paper Note] Real-Time Robot Execution with Masked Action Chunking
description: >-
  [ICLR 2026][Robotics][Real-time execution] This paper proposes REMAC, which systematically addresses two key failure modes of asynchronous inference—intra-chunk inconsistency and inter-chunk discontinuity—through a maske…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Real-time execution"
  - "action chunking"
  - "asynchronous inference"
  - "VLA"
  - "flow matching"
  - "LoRA"
date: 2026-05-08
content_hash: cd634f26ed30a486
---

# Real-Time Robot Execution with Masked Action Chunking

**Conference**: ICLR 2026
**arXiv**: [2601.20130](https://arxiv.org/abs/2601.20130)  
**Code**: [Project Page](https://remac-robot.github.io)  
**Area**: Robot Learning
**Keywords**: Real-time execution, action chunking, asynchronous inference, VLA, flow matching, LoRA

## TL;DR

This paper proposes REMAC, which systematically addresses two key failure modes of asynchronous inference—intra-chunk inconsistency and inter-chunk discontinuity—through a masked action chunking training strategy and a prefix-preserved sampling pipeline, enabling more reliable real-time robot control without introducing any additional inference latency.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models that predict action sequences via action chunking have become the dominant paradigm for generalist robot policies. Real-time performance is critical for robotic systems—latency can lead to task failure (e.g., spilling liquids) rather than merely increasing wait time.

**Necessity of Asynchronous Inference**: Synchronous inference requires inference latency $\delta < \Delta t$ (the control period); at 50 Hz this means under 20 ms, yet the $\pi_0$ model already requires 76 ms for action generation alone on an RTX 4090, far exceeding the threshold once preprocessing and network transmission are included. Asynchronous inference—predicting the next chunk while executing the current one—ensures actions are always available and is the only viable real-time solution.

**Limitations of Prior Work—Inter-chunk Discontinuity**: Two consecutive action chunks $\mathbf{A}_t^1$ and $\mathbf{A}_{t+h}^2$ may originate from different latent expert modes, producing abrupt transitions at chunk boundaries that cause jerky robot motion. Existing methods such as temporal ensembling (TE), BID, and RTC attempt to address this, but are either unreliable (TE even underperforms Naive Async on multi-task settings) or introduce additional latency (RTC requires 55–64 ms for gradient correction).

**An Overlooked Failure Mode—Intra-chunk Inconsistency**: This is the paper's core insight. Under inference latency $d$, the first $d$ actions of the current execution chunk actually come from the previous chunk $\mathbf{A}_{t-h}$ (conditioned on the stale observation $\mathbf{o}_{t-h}$) rather than representing optimal actions under the current observation $\mathbf{o}_t$. This creates a perception–action mismatch and a distribution shift between training and inference. No prior work has identified or addressed this problem.

**Key Insight**: Intra-chunk inconsistency is formulated as a partial prefix masking problem at arbitrary positions within an action chunk—during training, a random prefix is masked and the model learns to make corrections when observations and partial actions are misaligned; the sampling pipeline is simultaneously adjusted to preserve prefix continuity, jointly handling inter-chunk discontinuity.

**Technical Approach**: Training-time adaptation is preferred over test-time correction. By fine-tuning a pretrained policy with LoRA (adding only 1.5% extra parameters), correction capability is internalized into model weights, requiring no additional computation at inference time and remaining orthogonally composable with existing test-time methods.

## Method

### Overall Architecture

REMAC builds upon a flow matching policy. Given a pretrained policy $\mathbf{v}_\pi(\mathbf{A}_t|\mathbf{o}_t)$, the objective is to learn a latency-aware policy $\hat{\mathbf{v}}_\pi(\mathbf{A}_t|\mathbf{o}_t, d)$ that explicitly conditions on inference latency $d$. The framework consists of three training components (prefix masking, self-conditioned curriculum, residual alignment) and one sampling component (prefix-preserved sampling), all adapted efficiently via LoRA.

### Key Design 1: Prefix Masking

- **Function**: Applies delay-conditioned masking to the action chunk, imposing supervision only on the executable suffix while ignoring the prefix already occupied by the previous chunk.
- **Mechanism**: A mask $\mathbf{m}_d = \{m_d^\tau\}_{\tau=0}^{P-1} = \mathbf{1}[\tau \geq d]$ is defined, where $d \sim \mathcal{U}\{0, \dots, P-1\}$ is sampled uniformly. The masked loss is:

$$\mathcal{L}_\mathrm{m} = \sum_d \frac{\sum_{\tau=0}^{P-1} m_d^\tau \|\hat{\mathbf{u}}_\tau - \mathbf{u}_\tau\|_2^2}{\max(1, \sum_{\tau=0}^{P-1} m_d^\tau)}$$

- **Design Motivation**: By randomly sampling all valid latency values during training (from $d=0$ to $d=h$), the model is exposed to the full spectrum of conditions ranging from no masking to extreme masking, improving robustness to uncertain executed actions. A single model handles all latency settings without requiring separate training per latency.

### Key Design 2: Self-conditioned Curriculum

- **Function**: Progressively mixes the pretrained policy's own predictions into the training inputs to simulate test-time conditions and reduce exposure bias.
- **Mechanism**: The pretrained policy's prediction $\tilde{\mathbf{A}}_t$ is randomly mixed with the ground-truth action $\mathbf{A}_t$:

$$\hat{\mathbf{A}}_t = \gamma \mathbf{A}_t + \text{sg}((1-\gamma)\tilde{\mathbf{A}}_t), \quad \gamma \sim \mathrm{Bernoulli}(\sigma), \sigma \in [0,1]$$

where $\sigma$ is linearly annealed from 1 (pure ground-truth labels) to 0 (pure self-conditioned input), and $\text{sg}(\cdot)$ denotes stop-gradient.

- **Design Motivation**: Training exclusively on ground-truth labels causes exposure bias, since inference uses the model's own predictions rather than ground truth. Training exclusively on self-conditioned inputs is unstable early on. The curriculum schedule balances both concerns—ground-truth labels stabilize early training, while later stages allow the model to learn to correct its own prediction errors, aligning training and test conditions.

### Key Design 3: Residual Alignment

- **Function**: Introduces an auxiliary $\Delta$-matching term beyond standard supervision to explicitly align the model's learned corrections with the residual between the pretrained policy and the ground-truth target.
- **Mechanism**: Let $\tilde{\mathbf{u}}$ denote the pretrained policy's flow estimate (LoRA disabled) and $\hat{\mathbf{u}}$ the target policy estimate (LoRA enabled). The residual alignment loss is:

$$\mathcal{L}_\Delta = \sum_d \frac{\sum_{\tau=0}^{P-1} \|m_d^\tau(\mathbf{u}_\tau - \tilde{\mathbf{u}}_\tau) - m_d^\tau(\hat{\mathbf{u}}_\tau - \tilde{\mathbf{u}}_\tau)\|_2^2}{\max(1, \sum_{\tau=0}^{P-1} m_d^\tau)}$$

The total loss is $\mathcal{L} = \lambda_m \mathcal{L}_m + \lambda_\Delta \mathcal{L}_\Delta$, with $\lambda_m = \lambda_\Delta = 0.01$.

- **Design Motivation**: Although mathematically related to $\mathcal{L}_m$, the two terms emphasize different aspects— $\mathcal{L}_m$ directly aligns with ground-truth labels, while $\mathcal{L}_\Delta$ explicitly models corrections relative to the pretrained policy. Ablation experiments confirm that adding $\mathcal{L}_\Delta$ yields significant performance gains.

### Key Design 4: Prefix-preserved Sampling

- **Function**: Adjusts the inference-time sampling pipeline by initializing the new chunk prefix with already-executed actions and keeping the prefix fixed throughout each integration step.
- **Mechanism**: Rather than sampling the initial action state $\mathbf{A}_t^0$ from a Gaussian prior, the first $P-h$ dimensions are filled with the tail of the previous chunk, and the remainder is zero-initialized. The prefix is held constant during integration:

$$\mathbf{A}_t^{\tau+\frac{1}{n}} = \mathbf{m} \odot \left(\mathbf{A}_t^\tau + \frac{1}{n}\hat{\mathbf{v}}_\pi(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)\right) + (1-\mathbf{m}) \odot \mathbf{A}_t^\mathrm{p}$$

- **Design Motivation**: Retaining already-executed actions as a prior enables the newly generated portion to naturally connect with the prefix, directly enhancing inter-chunk continuity and aligning with the masking strategy used during training.

## Key Experimental Results

### Kinetix Simulation (12 high-dynamic tasks, average success rate)

| Method | $d=0$ | $d=1$ | $d=2$ | $d=3$ | $d=4$ |
|--------|-------|-------|-------|-------|-------|
| Naive Async | 0.828 | 0.702 | 0.639 | 0.525 | 0.451 |
| BID | — | — | — | — | — |
| RTC | — | — | — | — | — |
| **REMAC (Ours)** | **0.888** | **0.879** | **0.859** | **0.817** | **0.779** |

### Ablation Study (contribution of each component)

| Configuration | $d=0$ | $d=1$ | $d=2$ | $d=3$ | $d=4$ |
|---------------|-------|-------|-------|-------|-------|
| Naive | 0.828 | 0.702 | 0.639 | 0.525 | 0.451 |
| + LoRA (parameters only) | 0.825 | 0.710 | 0.630 | 0.510 | 0.428 |
| + Prefix Masking | 0.863 | 0.825 | 0.752 | 0.729 | 0.636 |
| + Self-conditioned Curriculum | 0.848 | 0.837 | 0.805 | 0.762 | 0.710 |
| + $\mathcal{L}_\Delta$ (full REMAC) | **0.888** | **0.879** | **0.859** | **0.817** | **0.779** |

### Composition with Test-time Methods

| Method | $d=0$ | $d=1$ | $d=2$ | $d=3$ | $d=4$ |
|--------|-------|-------|-------|-------|-------|
| REMAC | 0.888 | 0.879 | 0.859 | 0.817 | 0.779 |
| REMAC + BID | 0.888 | 0.880 | 0.862 | 0.821 | 0.781 |
| REMAC + RTC | 0.888 | 0.879 | 0.864 | 0.826 | 0.791 |

### Real Robot Experiments (Franka Research 3, average task completion progress)

| Method | Grasp-Easy | Grasp-Medium | Grasp-Hard |
|--------|-----------|-------------|-----------|
| Synchronous | 0.805 | 0.718 | 0.670 |
| Naive Async | 0.825 | 0.825 | 0.460 |
| Temporal Ensembling | 0.825 | 0.868 | 0.717 |
| RTC | 0.823 | 0.848 | 0.753 |
| **REMAC (Ours)** | **0.903** | **0.943** | **0.812** |

## Key Findings

- **Intra-chunk inconsistency is a critical failure mode**: All prior work focused solely on inter-chunk discontinuity. REMAC is the first to identify and address intra-chunk inconsistency. Ablation results show that adding prefix masking alone improves success rate at $d=4$ from 0.451 to 0.636 (+41%).

- **Training-time adaptation outperforms test-time correction**: REMAC incurs zero additional inference latency, whereas RTC introduces 55–64 ms of extra latency. In real-robot experiments, RTC performance even degrades at large delays, suggesting that test-time adjustment can backfire over longer execution horizons.

- **Each component contributes incrementally**: From Naive → Prefix Masking → Self-conditioned Curriculum → Residual Alignment, each additional component yields consistent improvement. The full method achieves a success rate of 0.779 at $d=4$, which is 72.7% higher than Naive Async.

- **Robustness gains are more pronounced at larger delays**: REMAC's performance drop from $d=0$ to $d=4$ (0.888→0.779, −12.3%) is far smaller than that of Naive Async (0.828→0.451, −45.5%), demonstrating strong robustness to latency variation.

- **A single model handles the full latency spectrum**: By randomly sampling training delays, REMAC requires no separate model per latency setting; one model serves all delay configurations.

## Highlights & Insights

- **The value of problem identification**: The identification of intra-chunk inconsistency is itself a significant contribution. Formalizing it as a "partial perception–action mismatch" and modeling it as a masking problem is an elegant abstraction.

- **Training-time solutions over inference-time patches**: Internalizing correction capability into model weights via LoRA rather than performing extra computation at inference time constitutes a more fundamental solution. The choice of LoRA is well motivated—it is treated as a distributional shift rather than a need for relearning.

- **Composability**: As a backbone-level improvement, REMAC is orthogonally composable with test-time methods such as BID and RTC, ensuring good ecosystem compatibility.

- **Practical value**: The design and validation of a complete real deployment framework (gRPC communication, latency estimation, action queuing) provides a comprehensive reference for real-time VLA deployment.

## Limitations & Future Work

- **Validated only on flow matching policies**: Although Appendix experiments with ACT are mentioned, the main experiments are confined to the flow matching framework; applicability to other action generation paradigms (e.g., diffusion policies, autoregressive policies) requires further validation.

- **Simplified latency estimation**: Discretizing continuous latency as $d = \lfloor \delta / \Delta t \rfloor$ while ignoring observation latency and sub-timestep delays may be insufficiently accurate in real-world scenarios with significant latency variation.

- **Limited scale of real-robot experiments**: Only 3 pick-and-place tasks, a single-arm setup, and 200 demonstration trajectories for fine-tuning. Validation on more complex bimanual manipulation, long-horizon tasks, and diverse objects is still lacking.

- **Training requires pretrained policy inference**: The self-conditioned curriculum requires running the pretrained model on each training sample to generate $\tilde{\mathbf{A}}_t$, increasing training computational cost.

## Related Work & Insights

### vs. RTC (Black et al., 2025)

RTC also targets real-time execution under asynchronous inference, employing a test-time inpainting strategy—warm-starting the next chunk with already-executed actions and applying gradient correction. **Core difference**: RTC introduces 55–64 ms of additional inference latency (compromising real-time performance) and addresses only inter-chunk discontinuity while ignoring intra-chunk inconsistency. REMAC resolves both problems at training time with zero additional inference overhead. In real-robot experiments, RTC performance degrades at large delays, indicating that test-time correction can be counterproductive under certain conditions.

### vs. BID (Liu et al., 2025)

BID samples multiple candidate predictions and applies rejection sampling to balance long-horizon consistency with short-horizon reactivity. **Core difference**: BID requires substantial computation (multiple forward passes and evaluations), making it unsuitable for real-time scenarios; it likewise leaves intra-chunk inconsistency unaddressed. REMAC resolves both issues through training-time adaptation, requiring only a single forward pass at inference time.

### vs. Temporal Ensembling (Zhao et al., 2023)

TE smooths chunk boundaries by taking a weighted average of overlapping portions of consecutive chunks. **Core difference**: TE is a heuristic approach that can even underperform Naive Async in highly dynamic environments. REMAC provides a principled solution that consistently outperforms all baselines across all tested conditions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The identification of intra-chunk inconsistency is an important insight, and the combined design of masked training, residual alignment, and curriculum scheduling is well-motivated and novel. Points deducted because each individual component (masked training, LoRA, curriculum learning) is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 simulation tasks + 3 real-robot tasks, covering 5 latency settings, comprehensive ablation studies, comparisons against 3 baselines, and demonstrated composability with test-time methods.
- **Writing Quality**: ⭐⭐⭐⭐ The problem analysis is clearly structured (inter-chunk vs. intra-chunk), method derivation is complete, and experimental presentation is thorough. The visualization of intra-chunk inconsistency is intuitively effective.
- **Value**: ⭐⭐⭐⭐⭐ Zero additional inference latency, only 1.5% parameter overhead, composability with existing methods, and a complete deployment framework—REMAC offers direct and highly practical value for real-time VLA deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICML 2026\] Mixture of Horizons in Action Chunking](../../ICML2026/robotics/mixture_of_horizons_in_action_chunking.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](../../NeurIPS2025/robotics/reinforcement_learning_with_action_chunking.md)
- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
