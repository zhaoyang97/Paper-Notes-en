---
title: >-
  [Paper Note] Vision-aligned Latent Reasoning for Multi-modal Large Language Model
description: >-
  [ICML 2026][Multimodal VLM][Latent space reasoning] This paper proposes VaLR: a method that inserts several "latent tokens" before each step of MLLM CoT reasoning and performs representation alignment (REPA) between thes…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Latent space reasoning"
  - "Visual alignment"
  - "REPA"
  - "MLLM"
  - "Test-time scaling"
date: 2026-05-08
content_hash: 3786600334c59388
---

# Vision-aligned Latent Reasoning for Multi-modal Large Language Model

**Conference**: ICML 2026  
**arXiv**: [2602.04476](https://arxiv.org/abs/2602.04476)  
**Code**: Available on project page (provided at the end of the paper)  
**Area**: Multimodal VLM / Visual Reasoning / Test-time scaling  
**Keywords**: Latent space reasoning, Visual alignment, REPA, MLLM, Test-time scaling

## TL;DR
This paper proposes VaLR: a method that inserts several "latent tokens" before each step of MLLM CoT reasoning and performs representation alignment (REPA) between these tokens and patch features from visual encoders like DINOv3/SigLIP/$\pi^3$. By continuously "feeding back" visual information to the model during long-chain reasoning, it improves Qwen2.5-VL's accuracy on VSI-Bench from 33.0% to 52.9% and marks the first time an MLLM exhibits "longer reasoning, higher accuracy" test-time scaling behavior.

## Background & Motivation

**Background**: Existing MLLMs (e.g., Qwen2.5-VL, LLaVA series) generally treat visual features as "initial context"—inserting them once at the beginning of the sequence and then allowing the LLM backbone to perform pure text CoT reasoning. While effective for short-context VQA, this approach fails in tasks requiring long-chain reasoning (e.g., multi-view spatial reasoning, mathematical geometry).

**Limitations of Prior Work**: The authors provide direct evidence in Figure 2's "Reasoning Length Analysis": as Ocean-R1's generation length grows from 100 to 300 tokens on MMVP, accuracy drops from 62.7% to 56.5%. Other latent reasoning methods (Monet, CoVT, LVR) also collapse on long chains. In other words, the "test-time scaling law" enjoyed by text-based LLMs (longer thinking $\to$ higher accuracy) is reversed in the multimodal domain to "long chain = more hallucinations."

**Key Challenge**: The root cause is the **progressive attenuation of visual signals**. With every autoregressive text token generated, the attention weight on the initial visual tokens is diluted. After hundreds of reasoning tokens, the model nearly "forgets" the contents of the image. Early solutions that inject image tokens as fixed prefixes (CoVT, Monet) cannot solve this because the visual information remains tethered only to the start of the sequence.

**Goal**: Design a mechanism that can re-activate the model's perception of the image **before every step of CoT reasoning** without relying on external visual encoder calls during inference (to avoid overhead) while maintaining long-chain reasoning capabilities.

**Key Insight**: Inspired by latent reasoning in the LLM domain (Coconut) and REPA (using external visual features to supervise intermediate layers in diffusion), the authors hypothesize: by **aligning the intermediate hidden states of the MLLM with patch features of a frozen visual encoder during training**, the model can learn to "generate its own visual anchors," maintaining continuous visual grounding during testing without external encoders.

**Core Idea**: Insert $K$ special latent tokens as "visual checkpoints" before each text reasoning step. During training, use cosine similarity to supervise the hidden states corresponding to these latent tokens using patch features from encoders like DINOv3, allowing the latent tokens to autonomously "refresh visual memory."

## Method

### Overall Architecture
VaLR undergoes two-stage SFT on a standard MLLM (Qwen2.5-VL-7B). During inference, the sequence follows $v, q \to (\ell_{[1:K]}^{(1)}, r^{(1)}, \ell_{[1:K]}^{(2)}, r^{(2)}, \cdots) \to a$, specifically: visual features + question $\to$ ($K$ latent tokens + step $i$ text reasoning) $\times N$ steps $\to$ final answer. Latent token segments are bounded by `<latent>` / `</latent>`. In latent mode, the model treats the previous hidden state $h_t$ directly as the input embedding for the next step (bypassing the LM-Head and token embedding table). In language mode, it reverts to standard token embedding inputs. Each latent segment is fixed at $K=16$ steps before forcing a switch back to language mode.

### Key Designs

1.  **Latent Tokens and Hidden State Autoregression**:
    - **Function**: Reserves $K$ "thinking slots" before each text reasoning step, allowing the model to perform internal characterization without outputting visible tokens.
    - **Mechanism**: During training data preprocessing, CoT data $v,q \to (r^{(i)})_{i=1}^N \to a$ is rewritten as $v,q \to (\ell_{[1:K]}^{(i)}, r^{(i)})_{i=1}^N \to a$. During the forward pass, upon encountering `<latent>`, the next input embedding is $E_{t+1} = [E_t; h_t]$ instead of $[E_t; e(x_{t+1})]$. The last hidden state $h_t$ is used directly as the next token embedding, allowing the model to "play freely" for $K$ steps in the latent space before decoding text via the LM-Head.
    - **Design Motivation**: Pure language CoT forces all intermediate states into discrete tokens, creating a narrow information bottleneck. Passing continuous hidden states preserves richer visual details, acting as a "scratchpad" specifically for visual anchoring.

2.  **REPA Alignment to External Visual Encoders**:
    - **Function**: Supervises internal features of latent tokens to approximate patch-level representations of visual encoders (e.g., DINOv3 / SigLIPv2 / $\pi^3$), forcing the model to internalize visual grounding.
    - **Mechanism**: For reasoning step $i$, $K$ latent token features $\mathbf{F}_{\text{MLLM}}^{(i)} = [f_1^{(i)}, \cdots, f_K^{(i)}]$ are extracted from an intermediate MLLM layer (default layer 12). These are upsampled to match the visual encoder's patch count $P$, projected via MLP $\psi$ to the encoder's dimension, and aligned using patch-wise cosine similarity with $\mathbf{F}_\phi^{(i)} = \phi(I^{(i)})$: $\mathcal{L}_{\text{REPA}} = -\frac{1}{NP}\sum_{i,p}\text{sim}(\hat{\mathbf{F}}_{\text{MLLM}}^{(i)}[p,:], \mathbf{F}_\phi^{(i)}[p,:])$. **Critically, the external encoder is only used during training and discarded during inference**; the latent tokens "learn" to produce visually aligned features independently.
    - **Design Motivation**: Ablation Table 3 shows that removing Visual Alignment (VA) causes accuracy to drop from 41.5% to 34.0% (vanilla SFT level). Using Qwen's own vision encoder reaches 39.6%, but using a self-supervised encoder like DINOv3 gains another 1.9%, indicating the alignment target itself (rather than external information leakage) is key.

3.  **Complementary Multi-encoder Alignment (VaLR-M)**:
    - **Function**: Aligns with multiple semantic/geometric encoders simultaneously to imbue latent tokens with heterogeneous visual knowledge.
    - **Mechanism**: Defines $\mathcal{L}_{\text{REPA}}^{\text{multi}} = \frac{1}{M}\sum_m \mathcal{L}_{\text{REPA}}^{(m)}$, with a separate projection head $\psi_m$ for each encoder $\phi_m$. The paper utilizes DINOv3 (fine-grained appearance), SigLIPv2 (semantics), and $\pi^3$ (3D geometry). Table 4 shows $\pi^3$ contributes most to VSI-Bench (+10p+), while DINOv3/SigLIPv2 benefit perception tasks like BLINK/MMVP. Combining all three yields the best-in-class 52.9% across all benchmarks.
    - **Design Motivation**: The authors observe that different encoders specialize in different visual subspaces. A multi-encoder strategy explicitly injects a "division of labor" into the latent space, effectively distilling a mini multi-view visual backbone inside the MLLM.

### Loss & Training Strategy
A two-stage curriculum is used: Stage 1 uses 450K CoT VQA data (mixed Zebra-CoT / CogCoM / Visual-CoT / OneThinker-SFT, etc.) for standard SFT to establish textual CoT capability using $\mathcal{L}_{\text{CE}}$. Stage 2 uses the same data but adds latent tokens and REPA, with the total loss $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{REPA}}$, where $\lambda = 0.5$ and $K = 16$. In both stages, the native vision encoder is frozen while the decoder is trained (Stage 2 also trains the projection MLP). Training uses 4 $\times$ A100s, Zero-2, AdamW, and lr of 1e-5 / 2e-6.

## Key Experimental Results

### Main Results
Comparison with GPT-4o / Claude-4 / Qwen2.5-VL base / three latent reasoning baselines on 8 sub-tasks of VSI-Bench and 5 perception benchmarks.

| Model | VSI-Bench Avg | BLINK | MMVP | V* | CVBench |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 34.0 | 63.0 | 68.7 | 42.9 | 79.2 |
| Qwen2.5-VL-7B (base) | 33.0 | 55.7 | 56.0 | 76.4 | 74.5 |
| + vanilla SFT | 33.7 | 56.6 | 58.7 | 78.0 | 77.0 |
| + Monet (latent baseline) | 14.0 | 49.1 | 50.0 | 83.3 | 71.1 |
| + CoVT | 18.6 | 56.0 | 58.7 | 78.0 | 80.0 |
| + VaLR-S (DINOv3) | 41.5 | 63.1 | 60.3 | 86.4 | 83.1 |
| + VaLR-M (DINOv3+SigLIP+$\pi^3$) | **52.9** | **64.7** | 60.3 | **86.9** | **87.6** |

VaLR-M improves the base model by +19.9 points on VSI-Bench and outperforms GPT-4o by 18.9 points. Notably, existing latent reasoning methods (Monet, CoVT, LVR) collapse to 14-19% on multi-view 3D tasks, highlighting that latent reasoning without visual re-injection is insufficient.

### Ablation Study

| Configuration | VSI-Bench | BLINK | MMVP | V* |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | 33.0 | 55.7 | 56.0 | 76.4 |
| + vanilla SFT | 33.7 | 56.6 | 58.7 | 78.0 |
| + VaLR w/o VA (Alignment removed) | 34.0 | 57.1 | 56.7 | 75.9 |
| + VaLR w/ QE (Using Qwen encoder) | 39.6 | 58.9 | 60.0 | 81.7 |
| + VaLR (DINOv3) | 41.5 | 63.1 | 60.3 | 86.4 |

Experiments on alignment layer depth (Front/Middle/Last i.e., layers 4/12/27) show the middle layer (level 12) is optimal, consistent with the original REPA paper and research suggesting visual info concentrates in the MLLM middle layers.

### Key Findings
- **REPA is Critical**: Without alignment, VaLR degrades to vanilla SFT; alignment is the source of the +8p+ gain. This indicates latent tokens are merely "carriers," while explicitly feeding visual signals into middle layers is the catalyst.
- **Emergence of Test-time Scaling**: In Figure 2, VaLR is the only method where accuracy scales with reasoning length; others collapse after a certain point. This migrates LLM scaling laws to the multimodal domain.
- **Encoder Synergy**: $\pi^3$ specifically boosts 3D multi-view tasks, while DINOv3/SigLIP boost perception. Their effects are additive without interference, proving the latent space can accommodate multi-source knowledge.
- **Data Scaling**: Figure 3 shows VaLR achieves the same V* level with 50K data that vanilla SFT reaches with 450K, signifying $>20 \times$ faster convergence.

## Highlights & Insights
- Integrates "latent reasoning" and "REPA" not for a complex architecture, but to address the fundamental bottleneck: visual signal decay in long-chain MLLM reasoning.
- **High Engineering Value**: No external encoder is required at inference. The visual alignment capability is distilled into the MLLM's intermediate layers, contrasting with multi-encoder methods that increase inference latency.
- The use of latent tokens as "visual refresh slots" is a highly transferable abstraction, applicable to long-context RAG or long-video descriptions to periodically pull back visual/retrieval features.
- The success of $\pi^3$ (geometric encoder) suggests that latent space alignment is naturally suited for "non-linguistically describable" visual modalities, providing a path to integrate 3D/tactile/audio signals without relying on captions.

## Limitations & Future Work
- The number of latent tokens $K=16$ is fixed. Adaptive allocation based on the "visual hunger" of different reasoning steps would be more efficient.
- Training relies on synthetic CoT data (e.g., Zebra-CoT), and the impact on "non-reasoning VQA" (e.g., style judgment) is not fully evaluated, risking overfitting to specific distributions.
- Multi-encoder alignment increases training costs due to multiple ViT-L forward passes. Scaling to 32B / 72B base models might be computationally expensive.
- Geometric encoders like $\pi^3$ require multi-view inputs and are not applicable to single-image VQA; many visual representation families remain unexplored for multimodal latent alignment.
- While the test-time scaling curve grows on MMVP, the "marginal收益" (returns vs. budget) has not been systematically characterized for industrial deployment.

## Related Work & Insights
- **vs. CoVT / Monet**: These inject visual features once as a static prefix. VaLR performs dynamic re-injection, a paradigm shift from "static to dynamic" that explains the performance divergence in Tables 1/2.
- **vs. Coconut (hao2024training)**: Coconut performs latent reasoning in pure text LLMs without visual supervision. VaLR transfers this to MLLMs and uses REPA to solve visual info loss in latent space.
- **vs. REPA (yu2024repa)**: Originally for diffusion alignment, VaLR adapts REPA to autoregressive MLLM latent tokens, proving its utility extends beyond generative models.
- **vs. Visual CoT / Imagine-then-Reason**: Those methods explicitly generate intermediate images/visual tokens, which is costly and limited by generative quality. VaLR achieves the same goal in latent space more efficiently.

## Rating
- Novelty: ⭐⭐⭐⭐ — Clever combination of latent reasoning and REPA, though neither is individually original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 main benchmarks and exhaustive ablations on encoders, layer depth, data scale, and reasoning length.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation with a compelling Figure 2; math in the REPA section could be more concise.
- Value: ⭐⭐⭐⭐⭐ — +19.9p on VSI-Bench is a significant gain achieved without increasing inference-time overhead, making it easy to integrate into existing MLLM Pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](../../CVPR2026/multimodal_vlm/joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[ICML 2026\] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)
- [\[ICML 2026\] Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)

</div>

<!-- RELATED:END -->
