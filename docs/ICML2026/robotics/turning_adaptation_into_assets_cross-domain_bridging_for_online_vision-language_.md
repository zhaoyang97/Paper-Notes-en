---
title: >-
  [Paper Note] Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation
description: >-
  [ICML 2026][Robotics][VLN] To address the continuous environment distribution drift in online Vision-Language Navigation, this paper proposes the IDEA framework. It encapsulates the soft prompts learned during each test-…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLN"
  - "Test-Time Adaptation"
  - "Soft Prompt"
  - "Fisher Information"
  - "Convex Hull Projection"
date: 2026-05-08
content_hash: 68cf2201d57b8bd2
---

# Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation

**Conference**: ICML 2026  
**arXiv**: [2605.23257](https://arxiv.org/abs/2605.23257)  
**Code**: None  
**Area**: Multimodal VLM / Vision-Language Navigation / Test-Time Adaptation  
**Keywords**: VLN, Test-Time Adaptation, Soft Prompt, Fisher Information, Convex Hull Projection

## TL;DR
To address the continuous environment distribution drift in online Vision-Language Navigation, this paper proposes the IDEA framework. It encapsulates the soft prompts learned during each test-time adaptation, along with domain coordinates and uncertainty, into reusable "assets." A training-free cross-domain shortcut is then derived by mapping the target domain onto a combination of historical assets using Wasserstein convex hull projection, achieving an average improvement of +2.5% SR and +1.9% SPL on REVERIE / R2R.

## Background & Motivation

**Background**: Vision-Language Navigation (VLN) requires an embodied agent to locate targets in 3D environments based on linguistic instructions. The mainstream approach involves training a Transformer policy via large-scale imitation learning and deploying it directly in new environments. To handle distribution shifts, recent studies have introduced Test-Time Adaptation (TTA) to VLN, primarily categorized into uncertainty-based self-training via entropy minimization (FSTTA, ReCAP) and reward-driven adjustments based on foundation models or human feedback.

**Limitations of Prior Work**: Existing methods treat the environment of each episode as an isolated transfer task. Each online update overwrites the original parameters, leading to two consequences: first, **catastrophic forgetting**, where adaptations learned in previously visited similar scenes are erased by new updates; second, **negative transfer**, where updates from the current domain are blindly applied to a subsequent domain with a completely different style, introducing mismatched house priors that degrade performance.

**Key Challenge**: There is a fundamental conflict between viewing adaptation as "transient, isolated parameter updates" and the "frequent appearance of related or repetitive scenes in VLN." The former fails to consolidate historical experience into reusable assets, resulting in all effort being zeroed out after an episode concludes.

**Goal**: Transform TTA in VLN from "one-time updates" into "continuous accumulation and composition of knowledge," while ensuring the method is plug-and-play, searchable, and training-free.

**Key Insight**: The authors redefine adaptation as an asset accumulation process. Instead of modifying global parameters, each TTA produces a lightweight asset with "domain coordinates" stored in a library of finite capacity. When facing a new domain, rather than optimizing from scratch, the agent finds an optimal linear combination on the **convex hull** of historical assets as initialization. This is effective because adjacent episodes often overlap significantly in visual style and semantic priors, and convex combinations naturally reuse partially relevant historical knowledge.

**Core Idea**: Adaptations for each domain are solidified into triplet assets $\{P^*, \Gamma, u\}$ via Fisher-weighted multi-layer prompt alignment. The target domain is then mapped as a linear combination of historical assets using a closed-form solution of Wasserstein convex hull projection, serving as a training-free cross-domain bridge.

## Method

### Overall Architecture
The policy backbone $\pi_\theta$ remains frozen. Inputs consist of a language instruction $I$ and 360° panoramic observations, while the output is an action sequence. IDEA prepends a set of learnable soft prompts $P = \{p_i\}_{i=1}^{L}$ to the visual token sequence, passing them into $M$ layers of the original fusion Transformer to obtain merged representations $\mathcal{Z}_t^{(\ell)}$. For each navigation step, IDEA first constructs a bridge prompt $P_b(w)$ from the historical asset library $\mathcal{M}$ as initialization. Based on whether the prompt significantly reduces the statistical distance to the source domain, the system decides whether to use this bridge directly for inference or continue optimizing it into a new asset to be stored in the library.

### Key Designs

1.  **Fisher-Guided Multi-Layer Soft Prompt Alignment**:
    - **Function**: Encodes adaptation knowledge into $L$ prompt tokens while ensuring alignment weights across fusion layers reflect real sensitivity to policy decisions, preventing prompts from over-fitting task-irrelevant noise.
    - **Mechanism**: Layer-wise statistics $(\mu_S^{(\ell)}, \sigma_S^{(\ell)})$ are precomputed using 128 source domain samples. Online, current batch statistics $(\mu_t^{(\ell)}, \sigma_t^{(\ell)})$ are aligned to the source using the loss $d^{(\ell)}(P) = \|\mu_S^{(\ell)} - \mu_t^{(\ell)}(P)\|_2 + \|\sigma_S^{(\ell)} - \sigma_t^{(\ell)}(P)\|_2$. Layers are weighted by $\alpha_\ell$, updated via EMA ($\beta = 0.1$) based on the normalized trace of the Fisher Information Matrix $\mathrm{Tr}(\Phi(\mathcal{Z}_t^{(\ell)}))$. The Fisher matrix is approximated using the first-order gradient of the policy log-likelihood.
    - **Design Motivation**: Solves the "statistical matching $\neq$ policy improvement" issue. If a prompt aligns statistics in a layer that does not affect action probabilities, it is likely fitting noise. Using Fisher information concentrates weights on layers that truly influence decision-making.

2.  **Triplet Structured Asset Library**:
    - **Function**: Encapsulates optimized prompts with "domain fingerprints" and "quality scores," making assets searchable, composable, and shareable across agents.
    - **Mechanism**: Each asset is defined as $\mathcal{A} := \{P^*, \Gamma, u\}$, where $P^*$ is the optimized prompt, $\Gamma$ represents the $(\mu, \sigma)$ statistics of the final fusion layer **without prompts** (acting as a domain descriptor decoupled from the prompt), and $u$ is the predictive entropy during inference. When the library exceeds capacity $K_{\max}$, new assets are merged with their nearest neighbors via a 1:1 average ($\mathcal{A}_k \leftarrow \frac{1}{2}(\mathcal{A}_k + \mathcal{A}^*)$) rather than being discarded.
    - **Design Motivation**: Using statistics without prompts as coordinates ensures the retrieval process is not contaminated by prompt perturbations. Merging preserves coverage of early scenes instead of drifting toward only recent data.

3.  **Closed-Form Bridge via Wasserstein Convex Hull Projection**:
    - **Function**: Represents the target domain prompt as a convex combination of $K$ historical assets without training, offering more robustness than hard retrieval of a single neighbor.
    - **Mechanism**: A shared weight vector $w \in \mathbb{R}^K$ performs linear interpolation in both prompt and statistical spaces: $P_b(w) = \sum_j w_j P_j$, $\Gamma_b(w) = \sum_j w_j \Gamma_j$. The weight $w$ is solved by minimizing the 2-Wasserstein distance between target statistics and $\Gamma_b(w)$, with an uncertainty penalty $\lambda \sum u_j w_j^2$ to suppress unreliable assets. This reduces to quadratic programming under simplex constraints $\min_w \|Aw - b\|_2^2 + \lambda w^\top U w$ s.t. $\mathbf{1}^\top w = 1, w \geq 0$. The authors derive a closed-form solution $w^* = \mathcal{H}^{-1}(g - \nu \mathbf{1})$ using KKT conditions, where $\mathcal{H} = A^\top A + \lambda U$.
    - **Design Motivation**: Hard retrieval fails when a target domain partially overlaps multiple historical domains. Convex combinations allow borrowing styles from one domain and layouts from another. The closed-form solution avoids iterative optimization overhead, making the bridge a genuine "training-free shortcut."

### Loss & Training
In each step, weights $w$ and bridge $P_b(w)$ are calculated via Eq. 12. If the statistical distance after applying the prompt $d_p$ satisfies $d_p < \tau \cdot d_0$, the domain is considered covered, and $P_b(w)$ is used directly. Otherwise, $P_b(w)$ serves as initialization for multi-layer alignment optimization to generate a new asset for the library. Theoretically, the convex hull projection weight tightens the generalization error bound, and the closed-form solution is Lipschitz stable regarding statistical estimation perturbations.

## Key Experimental Results

### Main Results

| Dataset (Eval) | Metric | Ours (IDEA) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| REVERIE Val unseen (HAMT) | SR | 34.92 | 33.06 (ReCAP) | +1.86 |
| REVERIE Val unseen (HAMT) | SPL | 31.52 | 30.51 (FSTTA) | +1.01 |
| REVERIE Test unseen (HAMT) | SR | 32.81 | 30.51 (ReCAP) | +2.30 |
| REVERIE Val seen (HAMT) | OSR | 50.67 | 48.49 (ReCAP) | +2.18 |
| REVERIE Val seen (HAMT) | RGSPL | 26.82 | 25.81 (Tent) | +1.01 |

Consistent advantages are maintained across four backbones (HAMT, DUET, etc.) and three benchmarks (REVERIE, R2R, R2R-CE).

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| Full IDEA | Full SR=34.92 | Fisher weighting + Asset library + Convex bridge |
| Equal-weight alignment (w/o Fisher) | Drop in performance | Prompts fit irrelevant noise without policy sensitivity |
| Hard nearest neighbor (w/o Convex) | Drop in performance | Single asset fails to cover partially overlapping domains |
| Optimization from scratch (w/o Bridge) | High latency | Loss of the training-free shortcut |

### Key Findings
- Inference latency for IDEA on HAMT is 245.8ms, slightly higher than SAR (197ms) but significantly lower than ViDA ($5.49 \times 10^3$ ms) and FSTTA (613ms), proving the KKT closed-form overhead is acceptable.
- Performance gains on the harder Test unseen split (+2.30 SR) are greater than on Val unseen (+1.86 SR), indicating higher utility of the asset library in truly unfamiliar environments.
- The asset library is portable—a library learned by one agent can be used by a new agent to bypass the cold-start phase.

## Highlights & Insights
- **Reconceptualizing TTA from "parameter updates" to "knowledge accumulation"**: This conceptual shift provides online VLN with reusable intermediate products rather than gradients that evaporate after each episode.
- **Fisher trace as a "functional vs. spurious" alignment discriminator**: Approximating the Hessian with first-order gradients avoids second-order costs while effectively identifying which layers truly impact decision-making.
- **Convex Hull + KKT Closed-Form**: Reducing a geometric projection into matrix operations is a classic way to deploy theoretical tools in real-time systems.
- **Decoupling domain coordinates**: Using statistics without prompts as retrieval keys avoids "contamination" by learnable content, a trick useful for any prompt-based continual learning.

## Limitations & Future Work
- The 1:1 averaging strategy for library merging might "blur" assets over time, potentially losing precise details of rare scenes.
- The convex hull projection assumes the target domain falls within the span of historical assets; the failure mode for cases entirely outside the library's coverage is not fully explored.
- Hyperparameters like the EMA coefficient $\beta$ and regularization $\lambda$ are fixed; sensitivity analysis across different backbones is needed.
- Theoretical results rely on the assumption that features follow a multivariate Gaussian distribution, which lacks empirical verification for complex fusion features.

## Related Work & Insights
- **vs. FSTTA / ReCAP**: Both perform online TTA in VLN via consistency or entropy minimization. IDEA differs by freezing updates into assets; while prior work resets after each episode, IDEA builds a long-term library.
- **vs. Tent / SAR**: Standard TTA uses BN calibration or entropy regularization. IDEA upgrades this to multi-layer weighted prompts guided by Fisher information.
- **vs. ViDA**: While ViDA also uses prompts for TTA, it optimizes at every step without reuse. IDEA uses convex hull projection to bypass optimization, achieving an order of magnitude speedup.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Framing TTA as "Asset Accumulation + Convex Hull Bridge" is a fresh abstraction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive across backbones and benchmarks, though ablation on $K_{\max}$ could be deeper.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, though KKT and Fisher sections have a high entry barrier.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play asset sharing is highly significant for real-world embodied AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](../../ICLR2026/robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](../../ICCV2025/robotics/bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](../../ICLR2026/robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](../../ICCV2025/robotics/bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->
