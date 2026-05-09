---
title: >-
  [Paper Note] Tiny Inference-Time Scaling with Latent Verifiers
description: >-
  [CVPR 2026][Image Generation][Inference-time scaling] This paper proposes VHS (Verifier on Hidden States), a verifier that operates directly on the intermediate hidden states of a DiT generator, bypassing the decode–re-encode overhead. In the inference-time scaling setting for single-step image generation, VHS reduces joint generation-verification latency by 63.3% and FLOPs by 51%, while achieving a 2.7% performance gain on GenEval under the same time budget.
tags:
  - CVPR 2026
  - Image Generation
  - Inference-time scaling
  - latent verifier
  - single-step generation
  - DiT
  - MLLM
date: 2026-05-08
content_hash: 956084c8f925e6a4
---

# Tiny Inference-Time Scaling with Latent Verifiers

**Conference**: CVPR 2026
**arXiv**: [2603.22492](https://arxiv.org/abs/2603.22492)
**Code**: [https://aimagelab.github.io/VHS](https://aimagelab.github.io/VHS)
**Area**: Diffusion Models / Image Generation / LLM Efficiency
**Keywords**: Inference-time scaling, latent verifier, single-step generation, DiT, MLLM

## TL;DR
This paper proposes VHS (Verifier on Hidden States), a verifier that operates directly on the intermediate hidden states of a DiT generator, bypassing the decode–re-encode overhead. In the inference-time scaling setting for single-step image generation, VHS reduces joint generation-verification latency by 63.3% and FLOPs by 51%, while achieving a 2.7% performance gain on GenEval under the same time budget.

## Background & Motivation

1. **State of the Field**: Inference-time scaling has emerged as an effective strategy for improving generative model quality by generating multiple candidate samples and selecting the best via a verifier. Best-of-N strategies are widely adopted in text-to-image generation.

2. **Limitations of Prior Work**: Existing verifiers are typically built on MLLMs (multimodal large language models), following the pipeline: generator produces output in latent space → decode to pixel space → re-encode with a visual encoder (e.g., CLIP) → score with an LLM. This introduces two issues: (a) the decode–re-encode step is redundant, as the latent space already implicitly encodes semantic information, yet it is decoded only to be re-encoded; (b) the literature typically counts only generation steps (function evaluations) while neglecting verifier overhead, which is non-trivial for **single-step generators** (e.g., SANA-Sprint), where the decoder and verifier costs are comparable to generation itself.

3. **Root Cause**: Practical deployment scenarios (e.g., commercial image generation services) typically return only 2–4 candidate images, constituting a "tiny budget" setting. Under such tight budgets, the overhead of MLLM-based verifiers is non-negligible. Diffusion models operate in a compressed latent space to reduce computation, yet verification reverts to pixel space, creating a computational contradiction.

4. **Paper Goals**: Design a more efficient verifier that can assess generation quality directly in the generator's latent space, eliminating the decode–re-encode overhead.

5. **Starting Point**: The intermediate hidden layers of a DiT generator already encode rich semantic information (interpretable by an LLM), making the decode–re-encode step unnecessary. Intermediate layer features can directly replace CLIP visual encoder outputs as visual inputs to the LLM.

6. **Core Idea**: The verifier directly consumes the intermediate hidden states of the DiT generator as visual input, skipping the remaining DiT layers, autoencoder decoding, and CLIP re-encoding, thereby enabling efficient verification within the latent space.

## Method

### Overall Architecture
Standard pipeline: $z_T \rightarrow$ all $L$ DiT layers $\rightarrow z_0 \rightarrow$ autoencoder decode $\rightarrow x_0 \rightarrow$ CLIP encode $\rightarrow$ LLM score. VHS pipeline: $z_T \rightarrow$ first $\ell^*$ DiT layers $\rightarrow h_{\ell^*} \rightarrow$ MLP connector $\rightarrow$ LLM score. This skips the remaining DiT layers, autoencoder decoding, and CLIP re-encoding.

### Key Designs

1. **Hidden State Verifier (VHS)**:

    - **Function**: Performs semantic evaluation directly from intermediate DiT layer features, replacing the conventional decode–re-encode–MLLM pipeline.
    - **Mechanism**: A standard MLLM verifier scores as $s = \text{LLM}(\mathcal{C}(\mathcal{V}(\mathcal{D}(z_0))), p)$, where $\mathcal{D}$ is the decoder, $\mathcal{V}$ is the visual encoder, and $\mathcal{C}$ is the connector. VHS simplifies this to $s = \text{LLM}(\mathcal{C}(h_{\ell^*}), p)$, feeding the hidden state $h_{\ell^*}$ at DiT layer $\ell^*$ directly through an MLP connector into the LLM. This not only bypasses $\mathcal{D}$ and $\mathcal{V}$, but also truncates the generator after layer $\ell^*$, saving the remaining $L - (\ell^* + 1)$ layers of computation.
    - **Design Motivation**: The generative latent space already encodes image semantics (a prerequisite for diffusion models to generate images), making additional encoding steps redundant. Ablation studies confirm that AE latent features, though perceptually rich, are semantically weak (due to reconstruction-oriented pretraining), whereas intermediate DiT features are conditioned on the generation prompt and thus exhibit stronger semantic alignment.

2. **Layer Selection**:

    - **Function**: Identify the optimal latency–performance trade-off point.
    - **Mechanism**: Five layers $h_1, h_5, h_7, h_9, h_{19}$ are evaluated in a 20-layer DiT. Very shallow layers (e.g., $h_1$) are close to the noisy input and yield unstable representations; very deep layers (e.g., $h_{19}$) approach the AE reconstruction space and emphasize perceptual reconstruction over semantics; the intermediate layer $h_7$ (~35% depth) is optimal — achieving 2.8% higher GenEval overall than $h_5$ and 2.2% higher than $h_9$, with lower latency due to truncating the subsequent 13 layers.
    - **Design Motivation**: This produces a non-monotonic trade-off: too shallow yields weak semantics, too deep biases toward perceptual reconstruction (similar to AE features), while the intermediate layer retains sufficient semantic information at minimum computational cost.

3. **Two-Stage Training**:

    - **Function**: Align the generator's hidden state space with the LLM's input space and fine-tune the model as a verifier.
    - **Mechanism**: **Alignment stage**: Following the LLaVA Stage-1 protocol, an MLP connector is trained on image–text pairs. Since the inputs are latent representations from the generative model rather than real images, the generator is used to produce images from captions, recording $h_{\ell^*}$; Gemma-3-4B then re-captions the generated images to avoid generation bias. Only the connector is trained. **Verifier fine-tuning stage**: 20 candidate images per prompt are generated from Reflect-DiT prompts, yielding 118K samples in total, with binary labels (Yes/No) obtained via GenEval automatic evaluation. Due to class imbalance (~63% positive samples), **weighted cross-entropy** is applied to rebalance class weights. Both the connector and the full LLM parameters are trained. At inference, the token probabilities of "yes"/"no" from the LLM serve as continuous scores.
    - **Design Motivation**: Standard cross-entropy is biased toward the positive class due to imbalance, causing the verifier to fail at rejecting low-quality generations. Both weighted cross-entropy and focal loss mitigate this issue (focal loss: +3.7%, weighted XE: +4.2%).

### Loss & Training
The alignment stage uses the standard LLaVA training procedure, training only the connector. The verifier fine-tuning stage uses weighted cross-entropy loss, training both the connector and the full LLM. The LLM is Qwen2.5-0.5B and the generator is SANA-Sprint (single-step).

## Key Experimental Results

### Main Results
SANA-Sprint + Qwen2.5-0.5B on GenEval (Best-of-N under matched time budgets):

| Time Budget | Verifier | Best-of-N | GenEval Overall |
|-------------|----------|-----------|-----------------|
| 550ms | MLLM w/ CLIP | Bo2 | 75.4% |
| 550ms | **VHS** | **Bo4** | **78.1%** (+2.7%) |
| 1100ms | MLLM w/ CLIP | Bo4 | 78.8% |
| 1100ms | **VHS** | **Bo9** | **80.5%** (+1.7%) |
| 1650ms | MLLM w/ CLIP | Bo6 | 80.4% |
| 1650ms | **VHS** | **Bo15** | **80.9%** (+0.5%) |

Latency and resource comparison (relative to Bo1 baseline):

| Verifier | Time | Saving | FLOPs Saving | VRAM Saving |
|----------|------|--------|--------------|-------------|
| MLLM w/ CLIP | 277ms | — | — | — |
| MLLM w/ AE | 138ms | 50.2% | 51.0% | 14.5% |
| **VHS on $h_7$** | **102ms** | **63.3%** | **62.9%** | **14.5%** |

### Ablation Study

| Configuration | GenEval Overall (1100ms) | Note |
|---------------|--------------------------|------|
| VHS $h_7$ + Weighted XE | **80.5%** | Optimal configuration |
| VHS $h_1$ + Weighted XE | 71.3% | Too shallow, insufficient semantics |
| VHS $h_{19}$ + Weighted XE | 76.5% | Too deep, biased toward perceptual reconstruction |
| VHS $h_7$ + XE | 76.3% | Standard XE, class imbalance issue |
| VHS $h_7$ + Focal | 80.0% | Focal loss also effective |
| MLLM w/ AE + Weighted XE | 74.7% | AE latent features are semantically weak |
| VHS $h_7$ + Qwen2-1.5B | 78.4% | Larger LLM provides no benefit; bottleneck is visual, not reasoning |

### Key Findings
- VHS provides its greatest advantage in the tiny-budget setting: given the same time, MLLM w/ CLIP evaluates 2 candidates while VHS evaluates 4, and the doubled candidate pool yields significant quality improvements.
- Layer selection is non-monotonic: too shallow yields weak semantics, too deep biases toward reconstruction, and $h_7$ (~35% depth) is optimal. The poor performance of AE latent features confirms that perceptual features ≠ semantic features.
- Scaling the LLM (0.5B → 1.5B) provides virtually no improvement, indicating that the bottleneck lies in visual representation quality rather than language reasoning — an important insight.
- Weighted XE > focal loss > XE; handling class imbalance is critical for verifier training.
- VHS also generalizes to PixArt-α-DMD (48% speedup), demonstrating cross-architecture applicability.

## Highlights & Insights
- **"Less is more" in verifier design**: Removing the visual encoder actually improves performance, because DiT latent representations are conditioned semantic features that are better suited for judging generation quality than CLIP's generic visual features. This challenges the conventional assumption that MLLMs require strong visual encoders.
- **Translating latency savings into candidate gains**: The true value of VHS lies not merely in being faster, but in evaluating more candidates within the same time budget, converting efficiency gains into quality gains.
- **Semantic analysis of intermediate DiT features**: The gradual transition of DiT layer representations from noise → semantics → perception across depth has theoretical value for understanding the internal representations of generative models.

## Limitations & Future Work
- Applicable only to single-step generators — multi-step generators produce different latents at each step, requiring adaptation of VHS.
- Evaluation is limited to GenEval; more comprehensive benchmarks (e.g., T2I-CompBench, DrawBench) are absent.
- Dependent on specific DiT architectures; redesign would be needed for non-DiT generators (e.g., U-Net-based diffusion models).
- A fixed layer $\ell^*$ is used throughout; adaptive layer selection remains unexplored.
- VHS itself requires training (alignment + fine-tuning) and is not a fully training-free approach.

## Related Work & Insights
- **vs. VQA-Score**: VQA-Score uses a VQA model for scoring and requires full pixel-space images. VHS evaluates directly in latent space, making it suitable for low-latency settings.
- **vs. Vision-Reward**: Vision-Reward employs an MLLM for fine-grained binary QA followed by weighted aggregation, also requiring pixel-space images. VHS eliminates this step entirely.
- **vs. multi-step SANA-Sprint**: 8-step SANA-Sprint (74.0%) underperforms VHS Bo4 (78.1%), further confirming that Best-of-N is more efficient than increasing the number of denoising steps.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The idea of performing verification directly in the latent space is elegant and compelling; the semantic analysis of DiT layer features provides genuine insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive latency/performance/ablation analysis, though limited to a single benchmark (GenEval only).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is precisely articulated; efficiency analysis is meticulous.
- **Value**: ⭐⭐⭐⭐ Directly applicable to production image generation services; the latent-space verification paradigm is extensible to video generation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](../../NeurIPS2025/image_generation/remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [\[CVPR 2026\] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring](score2instruct_scaling_up_video_quality-centric_instructions_via_automated_dimen.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)

<!-- RELATED:END -->
