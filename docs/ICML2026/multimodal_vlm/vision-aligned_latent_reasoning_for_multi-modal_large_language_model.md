---
title: >-
  [Paper Note] Vision-aligned Latent Reasoning for Multi-modal Large Language Model
description: >-
  [ICML 2026][Multimodal VLM][Latent Space Reasoning] This paper proposes VaLR: inserting several "latent tokens" before each CoT reasoning step in MLLMs…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Latent Space Reasoning"
  - "Visual Alignment"
  - "REPA"
  - "MLLM"
  - "Test-time Scaling"
date: 2026-05-08
content_hash: 8beac02b89d1785c
---

# Vision-aligned Latent Reasoning for Multi-modal Large Language Model

**Conference**: ICML 2026  
**arXiv**: [2602.04476](https://arxiv.org/abs/2602.04476)  
**Code**: Available on project page (provided at the end of the paper)  
**Area**: Multimodal VLM / Visual Reasoning / Test-time Scaling  
**Keywords**: Latent Space Reasoning, Visual Alignment, REPA, MLLM, Test-time Scaling

## TL;DR
This paper proposes VaLR: inserting several "latent tokens" before each CoT reasoning step in MLLMs, and aligning these tokens with patch features from visual encoders such as DINOv3/SigLIP/π³ (REPA), thereby continuously "feeding back" visual information to the model during long-chain reasoning. This approach boosts Qwen2.5-VL's accuracy on VSI-Bench from 33.0% to 52.9%, and for the first time enables MLLMs to exhibit "the longer the reasoning, the higher the accuracy" test-time scaling behavior.

## Background & Motivation

**Background**: Existing MLLMs (Qwen2.5-VL, LLaVA series) generally treat visual features as "initial context"—injecting them once at the sequence head, then letting the LLM backbone perform pure textual CoT reasoning. This works well for short-context VQA, but fails on tasks requiring long-chain reasoning (multi-view spatial reasoning, mathematical geometry).

**Limitations of Prior Work**: As shown in Figure 2's "reasoning length analysis," Ocean-R1's accuracy on MMVP drops from 62.7% to 56.5% as generation length increases from 100 to 300 tokens; other latent reasoning methods (Monet, CoVT, LVR) also collapse on long chains. In other words, the "test-time scaling law" enjoyed by textual LLMs (longer thinking → higher accuracy) is reversed in the multimodal domain, becoming "longer chains = more hallucinations."

**Key Challenge**: The root cause is the **progressive decay of visual signals**—with each autoregressive token generated, the attention to the initial visual tokens is diluted. After generating hundreds of reasoning tokens, the model has almost "forgotten" what the image looks like. Early approaches that inject image tokens as a fixed prefix (CoVT, Monet) cannot solve this, as visual information is always confined to the sequence head.

**Goal**: Design a mechanism that can **re-activate** the model's perception of the image before each CoT reasoning step, without relying on external visual encoders at inference (to avoid runtime overhead), while retaining long-chain reasoning capability.

**Key Insight**: Inspired by latent reasoning in LLMs (Coconut) and REPA (using external visual features to supervise diffusion intermediate layers), the authors hypothesize: **if the MLLM's intermediate hidden states are aligned with the patch features of a frozen visual encoder during training**, the model can learn to "self-generate visual anchors," maintaining visual grounding at inference without external encoders.

**Core Idea**: Insert K special latent tokens as "visual checkpoints" before each textual reasoning step. During training, use patch features from DINOv3 and similar encoders to supervise these latent tokens via cosine similarity, enabling them to "refresh visual memory."

## Method

### Overall Architecture
VaLR applies two-stage SFT on a standard MLLM (Qwen2.5-VL-7B). The inference sequence is $v, q \to (\ell_{[1:K]}^{(1)}, r^{(1)}, \ell_{[1:K]}^{(2)}, r^{(2)}, \cdots) \to a$, i.e., visual features + question → (K latent tokens + $i$-th step textual reasoning) × N steps → final answer. Latent token segments are bounded by `<latent>` / `</latent>`. In latent mode, the model directly uses the previous hidden state $h_t$ as the next input embedding (bypassing LM-Head and token embedding table); in language mode, it reverts to standard token embedding input. Each latent segment runs for a fixed $K=16$ steps before forcibly switching back to language mode.

### Key Designs

1. **Latent Tokens and Hidden State Autoregression**:

    - **Function**: Reserve K "thinking slots" before each textual reasoning step, allowing the model to internally process without outputting visible tokens.
    - **Mechanism**: During data preprocessing, rewrite CoT data $v,q \to (r^{(i)})_{i=1}^N \to a$ as $v,q \to (\ell_{[1:K]}^{(i)}, r^{(i)})_{i=1}^N \to a$. Upon encountering `<latent>` during forward pass, the next input embedding is $E_{t+1} = [E_t; h_t]$ instead of $[E_t; e(x_{t+1})]$, i.e., the last hidden state $h_t$ is directly used as the next token embedding, allowing the model to "freely operate" in latent space for K steps before switching back to language mode for subsequent text decoding.
    - **Design Motivation**: Pure language CoT forces all intermediate states into discrete tokens, creating a narrow information bottleneck; passing continuous hidden states preserves richer visual details, effectively giving the model a "scratchpad" for visual anchoring.

2. **REPA Representation Alignment to External Visual Encoders**:

    - **Function**: Supervise the intermediate features of latent tokens to approximate patch-level representations from DINOv3 / SigLIPv2 / π³, compelling the model to "internalize" visual grounding.
    - **Mechanism**: For the $i$-th reasoning step, extract K latent token features $\mathbf{F}_{\text{MLLM}}^{(i)} = [f_1^{(i)}, \cdots, f_K^{(i)}]$ from an intermediate MLLM layer (default: 12th, mid-depth), upsample to match the number of patches P in the visual encoder, then project via MLP $\psi$ to the encoder's dimension. Align with $\mathbf{F}_\phi^{(i)} = \phi(I^{(i)})$ using patch-wise cosine similarity: $\mathcal{L}_{\text{REPA}} = -\frac{1}{NP}\sum_{i,p}\text{sim}(\hat{\mathbf{F}}_{\text{MLLM}}^{(i)}[p,:], \mathbf{F}_\phi^{(i)}[p,:])$. **Crucially: the external encoder is used only during training and is entirely discarded at inference**—the latent tokens have "learned" to generate aligned visual features.
    - **Design Motivation**: Ablation Table 3 shows that removing VA (visual alignment) drops accuracy from 41.5% to 34.0%, equivalent to vanilla SFT; using Qwen's own vision encoder achieves 39.6%, but DINOv3 (self-supervised) adds another 1.9%, indicating the alignment target itself (not external information leakage) is key.

3. **Multi-encoder Complementary Alignment (VaLR-M)**:

    - **Function**: Align with multiple semantic/geometric encoders simultaneously, enabling latent tokens to carry heterogeneous visual knowledge.
    - **Mechanism**: Define $\mathcal{L}_{\text{REPA}}^{\text{multi}} = \frac{1}{M}\sum_m \mathcal{L}_{\text{REPA}}^{(m)}$, assigning each encoder $\phi_m$ a separate projection head $\psi_m$. The paper uses DINOv3 (fine-grained appearance), SigLIPv2 (semantics), and π³ (3D geometry), all ViT-L encoders. Table 4's controlled experiments show: adding π³ yields the largest gain (+10p+) on VSI-Bench (3D multi-view), while DINOv3/SigLIPv2 benefit BLINK/MMVP (perception tasks); enabling all three achieves the best overall 52.9%.
    - **Design Motivation**: The authors observe that different encoders excel in different visual subspaces, and single alignment cannot cover all. The multi-encoder strategy explicitly injects "expert division" into latent space, allowing the MLLM to distill a mini multi-view visual backbone internally.

### Loss & Training

Two-stage curriculum: Stage 1 uses 450K CoT VQA samples (Zebra-CoT / CogCoM / Visual-CoT / OneThinker-SFT, etc.) for standard SFT to establish basic textual CoT ability, with loss $\mathcal{L}_{\text{CE}}$; Stage 2 adds latent tokens and REPA on the same data, with total loss $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{REPA}}$, where $\lambda = 0.5$, $K = 16$. Both stages freeze the native vision encoder, training only the decoder (Stage 2 also trains the projection MLP), using 4×A100, Zero-2, AdamW, lr 1e-5 / 2e-6.

## Key Experimental Results

### Main Results

On VSI-Bench (multi-view 3D spatial reasoning, 8 sub-tasks) and 5 perception benchmarks, compared with GPT-4o / Claude-4 / Qwen2.5-VL base / three latent reasoning baselines:

| Model | VSI-Bench Avg | BLINK | MMVP | V* | CVBench |
|-------|---------------|-------|------|-----|---------|
| GPT-4o | 34.0 | 63.0 | 68.7 | 42.9 | 79.2 |
| Qwen2.5-VL-7B (base) | 33.0 | 55.7 | 56.0 | 76.4 | 74.5 |
| + vanilla SFT | 33.7 | 56.6 | 58.7 | 78.0 | 77.0 |
| + Monet (latent baseline) | 14.0 | 49.1 | 50.0 | 83.3 | 71.1 |
| + CoVT | 18.6 | 56.0 | 58.7 | 78.0 | 80.0 |
| + VaLR-S (DINOv3) | 41.5 | 63.1 | 60.3 | 86.4 | 83.1 |
| + VaLR-M (DINOv3+SigLIP+π³) | **52.9** | **64.7** | 60.3 | **86.9** | **87.6** |

VaLR-M improves the base model by 19.9 percentage points on VSI-Bench, surpassing GPT-4o by 18.9 points. Notably, existing latent reasoning methods (Monet, CoVT, LVR) **all collapse** to 14-19% on multi-view 3D tasks, highlighting that "latent reasoning without visual reinjection" is a dead end.

### Ablation Study

| Configuration | VSI-Bench | BLINK | MMVP | V* |
|---------------|-----------|-------|------|-----|
| Qwen2.5-VL-7B | 33.0 | 55.7 | 56.0 | 76.4 |
| + vanilla SFT | 33.7 | 56.6 | 58.7 | 78.0 |
| + VaLR w/o VA (no alignment) | 34.0 | 57.1 | 56.7 | 75.9 |
| + VaLR w/ QE (Qwen's own encoder) | 39.6 | 58.9 | 60.0 | 81.7 |
| + VaLR (DINOv3) | 41.5 | 63.1 | 60.3 | 86.4 |

Experiments on alignment layer position (Front/Middle/Last, i.e., layers 4/12/27) show the middle layer (12th) is optimal, consistent with the original REPA paper and other studies indicating "visual information concentrates in MLLM mid-layers."

### Key Findings

- **REPA is critical**: Removing alignment degrades VaLR to vanilla SFT; adding alignment is the true source of +8p+ gain, indicating latent tokens are merely "carriers," and explicit visual signal injection into intermediate layers is key.
- **Test-time scaling emerges**: In Figure 2, VaLR is the only method where "longer reasoning yields higher accuracy"; others collapse beyond a certain length. This is the first transfer of the LLM scaling law to the multimodal domain.
- **Multi-encoder synergy shows "expertization"**: π³ specifically boosts 3D multi-view tasks, DINOv3/SigLIP boost perception tasks, and their combination yields additive gains without interference, indicating latent space is large enough to accommodate multi-source knowledge.
- **Data scaling >20× accelerates convergence**: Figure 3 shows VaLR achieves the V* level reached by vanilla SFT with 450K data using only 50K samples.

## Highlights & Insights

- Integrates "latent reasoning" and "REPA"—two previously independent lines—focusing not on "fancier architectures" but on addressing the true bottleneck of MLLM long-chain reasoning: visual signal decay. The problem diagnosis and targeted solution are particularly strong.
- **No external encoder required at inference**: High engineering value—deployment requires only a single Qwen2.5-VL-7B, as visual alignment is distilled into the MLLM's intermediate layers. This contrasts with many "expensive at training and inference" multi-encoder methods.
- The abstraction of latent tokens as "visual refresh slots" can transfer to many scenarios: e.g., long-context document RAG (periodically reactivating retrieval features with latent tokens), long video descriptions (using latent tokens every N frames to pull back visual features).
- Adding π³, a geometry-specialized encoder, yields +10p+ on 3D tasks, indicating latent space alignment is naturally suited for "non-language-describable" visual modalities. This offers a new, non-captioning route for integrating 3D/tactile/audio into LLMs.

## Limitations & Future Work

- The number of latent tokens $K=16$ is fixed and not adaptive; different reasoning steps clearly have varying "hunger" for visual information, suggesting the need for learnable budget allocation.
- Training data is entirely synthetic CoT (e.g., Zebra-CoT), and its impact on "non-reasoning VQA" (e.g., visual description, style judgment) is not fully evaluated, posing a risk of overfitting to reasoning data distributions.
- Multi-encoder alignment increases training cost by requiring several ViT-L forward passes; while feasible with 4×A100, scaling to 32B/72B base models would be significantly more expensive.
- π³ and similar geometric encoders require multi-view input, making them unsuitable for single-image VQA; multimodal latent alignment has yet to explore the full family of visual representations.
- Although the test-time scaling curve keeps rising on MMVP, the "marginal curve" of inference budget vs. gain is not systematically characterized, leaving the practical question of "when to stop" open for industrial deployment.

## Related Work & Insights

- **vs CoVT / Monet**: These treat visual features as a fixed prefix injected once; VaLR dynamically reinjects before each reasoning step—a "static → dynamic" paradigm shift, explaining why VaLR improves while others collapse (see Tables 1/2).
- **vs Coconut (hao2024training)**: Coconut performs latent reasoning on pure-text LLMs without visual supervision; VaLR transfers this idea to MLLMs and uses REPA to address the "latent space loses visual information" issue.
- **vs REPA (yu2024repa)**: Original REPA aligns diffusion intermediate layers; VaLR applies it to autoregressive MLLM latent tokens, demonstrating REPA's applicability beyond generative models.
- **vs Visual CoT / Imagine-then-Reason**: These explicitly generate intermediate images/visual tokens to aid reasoning, incurring high cost and being limited by image generator quality; VaLR achieves the same goal in latent space, being lighter and more controllable.

## Rating

- Novelty: ⭐⭐⭐⭐ — An excellent combination of latent reasoning and REPA, though neither is original alone.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six main benchmarks, comprehensive ablations on multi-encoder/layer position/data scale/reasoning length, and the first multimodal test-time scaling curve.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is very clear, Figure 2 decisively outperforms all baselines; REPA formulas could be further streamlined.
- Value: ⭐⭐⭐⭐⭐ — VSI-Bench +19.9p is a truly useful gain, and no extra model at inference, making it almost "free" to graft onto existing MLLM training pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](../../CVPR2026/multimodal_vlm/joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[ICML 2026\] Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)
- [\[ICML 2026\] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)

</div>

<!-- RELATED:END -->
