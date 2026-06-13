---
title: >-
  [Paper Note] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching
description: >-
  [ICML 2026][Image Generation][REPA] AG-REPA discovers that in audio Flow Matching, the "layers storing semantic information" and the "layers actually driving the velocity field" do not overlap. It proposes using forward-…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "REPA"
  - "Audio Generation"
  - "Flow Matching"
  - "Causal Attribution"
  - "Layer Selection"
date: 2026-05-08
content_hash: d72e034fcb8f42fd
---

# AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching

**Conference**: ICML 2026  
**arXiv**: [2603.01006](https://arxiv.org/abs/2603.01006)  
**Code**: https://github.com/zpforlove/AG-REPA  
**Area**: Audio Generation / Flow Matching  
**Keywords**: REPA, Audio Generation, Flow Matching, Causal Attribution, Layer Selection  

## TL;DR
AG-REPA discovers that in audio Flow Matching, the "layers storing semantic information" and the "layers actually driving the velocity field" do not overlap. It proposes using forward-only gate ablation to select layers with the highest causal contribution for representation alignment, achieving faster convergence and lower FAD than fixed-layer REPA in speech and general audio generation.

## Background & Motivation
**Background**: Flow Matching has become an important paradigm for speech synthesis and general audio generation, learning a continuous velocity field from noise to the audio data distribution to generate samples. REPresentation Alignment (REPA) is a technique used in recent years to accelerate generative model training by aligning intermediate hidden states with pretrained teacher features, helping the model acquire useful representations faster.

**Limitations of Prior Work**: In the vision domain, REPA often empirically selects a specific intermediate layer or a fixed depth for alignment. However, the architecture of audio Flow Matching is different. Token-conditioned audio generation requires decoding from sparse discrete tokens to continuous waveforms without the dense spatial anchors found in video or images. Adopting fixed middle-layer alignment might supervise layers that "represent the teacher well but have little effect on the velocity field."

**Key Challenge**: Representational similarity does not equate to functional contribution. A layer might be proficient at storing semantic or acoustic information but may not be the causal driver that most influences the velocity field in the current generation dynamics. Aligning such layers makes training appear to acquire teacher information without actually imposing supervision at the positions that truly control the generation trajectory.

**Goal**: The authors aim to answer "which layers should be aligned for REPA in audio Flow Matching." They seek not only to find information storage locations but also locations with causal influence on the velocity field output, constructing more effective layer selection and weighting strategies accordingly.

**Key Insight**: The paper introduces the Store-Contribute Dissociation (SCD) viewpoint: deep layers are often semantic storage, while shallow or transitional layers may be the causal drivers. Consequently, the method shifts from representation diagnostics to causal attribution, using forward-only gate ablation to directly measure the change in the velocity field after removing a specific layer.

**Core Idea**: Instead of applying REPA to layers with the highest teacher similarity, use FoG-A to identify the Top-K layers that most significantly change the velocity field and assign alignment weights based on attribution strength.

## Method
AG-REPA consists of a set of diagnostic tools and a training strategy. The diagnostic tools first distinguish between "what the network knows" and "what the network actually uses"; the training strategy then performs sparse, weighted REPA alignment only on the latter.

### Overall Architecture
The model is a unified audio generation framework performing both TTS and Text-to-Audio: the speech side uses semantic tokens, and the general audio side uses event/acoustic tokens, sharing a DiT-based Flow Matching backbone. Before training or during a warm-up phase, the authors use BiT-C and LASP to measure layer similarities with Whisper/BEATs teachers, followed by FoG-A to perform gate ablation on each layer to observe changes in the velocity field output. Subsequently, Top-K FoG-A layers are selected, two-layer MLP projection heads are added to these layers, and the alignment loss is added to the original Flow Matching loss.

### Key Designs
1.  **SCD Diagnostics: Separating representation storage and functional contribution**:
    -   **Function**: Proves that "which layer is like the teacher" and "which layer influences generation" are two different things.
    -   **Mechanism**: BiT-C establishes dual-stream cosine alignment with Whisper and BEATs teachers; LASP uses shared frozen projection heads to compare information storage in the same teacher space. Experiments show that Cos-SEM is often highest in deep layers L22–L24, while Cos-EVT may be in deep or middle layers depending on token topology.
    -   **Design Motivation**: If only teacher similarity is considered, the alignment will tend to target deep representation reservoirs. However, since the generative model fundamentally predicts the velocity field, stored information does not necessarily mean that optimizing those layers is most effective.

2.  **FoG-A: Forward-only Gate Ablation for Locating Causal Layers**:
    -   **Function**: Directly measures the functional necessity of a specific layer for the velocity field output.
    -   **Mechanism**: The DiT residual layer is expressed as $h_l=h_{l-1}+m_l f_l(h_{l-1},t,c)$, where $m_l=1$ in standard forward passes. FoG-A sets $m_k=0$ for the $k$-th layer without backpropagating gradients, comparing the normalized difference between the ablated velocity $v_\theta^{\setminus k}$ and the original velocity field $v_\theta$. A larger difference indicates the layer is more of a causal driver.
    -   **Design Motivation**: Gradient norms only indicate where the optimizer is currently updating, and LASP only indicates where information is stored. FoG-A uses intervention to measure "whether removing this layer changes the generation direction," which is closer to where alignment should act.

3.  **AG-REPA: Sparse Layer Selection and Attribution Weighting**:
    -   **Function**: Shifts REPA from fixed-depth heuristics to model/data adaptive causal layer supervision.
    -   **Mechanism**: A set of Top-K layers $\mathcal{S}$ is selected based on FoG-A scores, with each layer weight assigned as $\lambda_k=\mathrm{FoG\text{-}A}_k/\sum_{j\in\mathcal{S}}\mathrm{FoG\text{-}A}_j$. Each selected layer is connected to a lightweight two-layer MLP to align temporal-pooled hidden states with frozen teacher embeddings. The final objective is $\mathcal{L}_{FM}+\lambda_{BiT}\mathcal{L}_{BiT}+\sum_{k\in\mathcal{S}}\lambda_k(1-\cos(h_{\phi_k}(\bar{h}_k),\mathcal{T}(x)))$.
    -   **Design Motivation**: Multiple layers may jointly control generation with uneven contributions. Top-K + attribution-proportional weighting adapts better to different architectures and tokenizers than single-layer REPA or fixed L1–L3.

### Loss & Training
AG-REPA retains the primary Flow Matching loss, keeps BiT-C anchors at the input interface, and adds representation alignment terms to a few intermediate layers selected by FoG-A. FoG-A is triggered every 200 steps only during a 5,000-step warm-up using small batches without gradients on a single GPU; this totals approximately 41 equivalent single forward passes, with a wall-clock increment under 0.5%. The main training only adds $K=3$ lightweight MLP heads, with extra parameters under 0.5% and a per-step time overhead under 2%.

## Key Experimental Results

### Main Results
| Method | Speech WER↓ | Speech FAD↓ | Audio FAD↓ | Speech MOS↑ | Audio MOS↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base (no layer align.) | 5.82 | 1.84 | 3.45 | 3.62±0.08 | 3.45±0.09 |
| REPA @ Layer 4 | 5.15 | 1.65 | 3.12 | 3.75±0.07 | 3.58±0.08 |
| REPA @ Layer 8 | 4.93 | 1.58 | 3.05 | 3.79±0.06 | 3.64±0.08 |
| REPA @ L4,8,12 | 4.21 | 1.45 | 2.88 | 3.92±0.06 | 3.77±0.07 |
| REPA @ Deep (L20–L22) | 5.60 | 1.79 | 3.39 | 3.64±0.08 | 3.48±0.09 |
| REPA @ Shallow (L1–L3) | 3.62 | 1.36 | 2.68 | 4.05±0.06 | 3.87±0.07 |
| AG-REPA (Top-3) | 3.45 | 1.29 | 2.56 | 4.12±0.05 | 3.94±0.07 |

### Ablation Study
| Selection Strategy | Top-3 Layers | Speech FAD↓ | Audio FAD↓ | Relative Gain | Convergence Steps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base | None | 1.84 | 3.45 | 0.0% | N/A |
| Random Control | L5,L14,L19 | 1.75 | 3.32 | +4.9% | 850k |
| Highest LASP | L22,L23,L24 | 1.68 | 3.21 | +8.7% | 720k |
| Gradient Norm | L1,L2,L4 | 1.35 | 2.71 | +26.6% | 260k |
| Highest FoG-A | L1,L2,L7 | 1.29 | 2.56 | +29.9% | 220k |

### Key Findings
-   **SCD is highly evident**: Under Config B, the top layers for Cos-SEM are L23/L22/L24 and for Cos-EVT are L14/L13/L15, but FoG-A selects L1/L2/L7 for speech and L1/L7/L2 for audio. That is, deep layers are more similar to the teacher, but shallow/middle layers more significantly influence the velocity field.
-   Compared to the best single fixed-layer REPA, AG-REPA reduces speech FAD by approximately 18% and audio FAD by approximately 16%; it also shows an 11% advantage over the L4,8,12 multi-layer heuristic.
-   **"Doing" is better than "Knowing"**: Highest LASP only reduces Speech FAD to 1.68, whereas Highest FoG-A reduces it to 1.29 and shortens the convergence steps to reach Speech FAD=1.5 from 720k to 220k, a ~3.3x acceleration.
-   Generalization experiments show that AG-REPA is not only applicable to the DiT in this paper. Consistent improvements were observed across Voicebox (FAD 1.20 to 0.95), CosyVoice (0.88 to 0.72), and F5-TTS (1.45 to 1.15).

## Highlights & Insights
-   The most critical insight is the decoupling of "representational similarity" from "causal contribution." Many alignment methods default to supervising layers with high teacher similarity, but this paper demonstrates that the locations in generation dynamics with the most leverage may be entirely different.
-   FoG-A is a practical causal probe: it requires no backpropagation or additional probes, only gating a layer and observing changes in the velocity field, making it low-cost and directly interpretable.
-   The selection rules of AG-REPA can be rerun across architectures, which is more robust than the shallow heuristic of fixing L1–L3. In Table 5, shallow REPA only reduced F5-TTS FAD from 1.45 to 1.34, while AG-REPA reduced it to 1.15.
-   This paper also provides inspiration for other generative models: when performing representation alignment or distillation, one should not only ask "which layer is most like the teacher" but also "aligning which layer most effectively changes the final generation function."

## Limitations & Future Work
-   FoG-A currently relies on layer-by-layer forward ablation. While the cost is low, it still requires access to the model's internal layers to modify the forward gate; this is not directly applicable to closed-source models or highly encapsulated inference frameworks.
-   Experiments focused on audio Flow Matching; whether the conclusions transfer seamlessly to image, video, or autoregressive audio models still needs verification.
-   While the Top-K layer set remains stable in this setup, further testing is needed to see if warm-up selection remains sufficient if training distributions, model scales, or tokenization change drastically.
-   AG-REPA still relies on Whisper/BEATs as external teachers. Teacher bias might influence the model's preference for certain semantic or acoustic attributes. Future research could explore teacher-free methods or multi-teacher uncertainty weighting.

## Related Work & Insights
-   **vs REPA**: Standard REPA accelerates generative model training via intermediate layer alignment, but layer selection is mostly empirical. AG-REPA uses FoG-A to automatically select layers based on functional contribution.
-   **vs iREPA / vision REPA variants**: Vision methods emphasize spatial structure or visual teacher features, which cannot be directly applied to 1D temporal audio representations. This paper redesigns layer diagnostics and projection heads for token-conditioned audio.
-   **vs HASTE**: HASTE focuses on how alignment might cause capacity mismatch in later training stages. AG-REPA focuses on which layers alignment should be applied to; the two are complementary.
-   **vs interpretability probes**: LASP/BiT-C are representation probes, while FoG-A is an intervention probe. This paper converts interpretability metrics directly into training strategies, making it more valuable than purely analytical papers.

## Rating
-   Novelty: ⭐⭐⭐⭐ The SCD perspective and FoG-A-driven REPA layer selection are highly insightful, representing a clear case of turning interpretability into a training algorithm.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ The main model, layer selection ablation, cross-architecture generalization, and efficiency/stability analyses are comprehensive.
-   Writing Quality: ⭐⭐⭐⭐ The methodological narrative is clear with strong supporting charts; some theoretical motivations and notations are dense, presenting a slight hurdle for non-audio readers.
-   Value: ⭐⭐⭐⭐⭐ Very practical for audio generation training acceleration and provides a transferable methodology for representation alignment in other generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](../../ICLR2026/image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)

</div>

<!-- RELATED:END -->
