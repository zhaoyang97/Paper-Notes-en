---
title: >-
  [Paper Note] NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching
description: >-
  [ICLR 2026][Multimodal VLM][Discrete Flow Matching] The authors replace Autoregressive (AR) with Discrete Flow Matching (DFM) as a unified modeling paradigm to develop NExT-OMNI, the first fully DFM-based open-source omnimodal foundation model. A single encoder provides unified representations that support understanding, generation, and cross-modal retrieval across text, image, video, and audio.
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Discrete Flow Matching"
  - "Omnimodal"
  - "any-to-any"
  - "Unified Representation"
  - "Cross-modal Retrieval"
date: 2026-05-08
content_hash: 1b84bc847026c498
---

# NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=odatOcBi61](https://openreview.net/forum?id=odatOcBi61)  
**Code**: [https://github.com/ritzz-ai/Next-OMNI](https://github.com/ritzz-ai/Next-OMNI)  
**Area**: Multimodal / Omnimodal Foundation Models  
**Keywords**: Discrete Flow Matching, Omnimodal, any-to-any, Unified Representation, Cross-modal Retrieval  

## TL;DR
The authors replace Autoregressive (AR) with Discrete Flow Matching (DFM) as a unified modeling paradigm to develop NExT-OMNI, the first fully DFM-based open-source omnimodal foundation model. A single encoder provides unified representations that support understanding, generation, and cross-modal retrieval across text, image, video, and audio.

## Background & Motivation
- **Background**: Unified multimodal understanding and generation is considered a key bottleneck on the path to AGI. Prevalent approaches adapt the success of Autoregressive (AR) Large Language Models to multimodality, often utilizing hybrid architectures or decoupled modules (e.g., Janus, Show-o, Bagel) to handle understanding and generation separately.
- **Limitations of Prior Work**: The AR paradigm exhibits inherent conflicts between understanding and generation tasks, making them difficult to balance. Hybrid/decoupling schemes essentially employ "task-based routing," which introduces redundant parameterized modules, high structural complexity, and slow inference. Moreover, excessively separated encoded features **fail to produce truly unified representations**, leading to poor performance in tasks requiring deep feature fusion, such as cross-modal retrieval.
- **Key Challenge**: Decoupling yields individual task performance at the cost of unity and generality. Capabilities like any-to-any generation and cross-modal retrieval precisely require "generalized multimodal" fusion rather than decoupling.
- **Goal**: To find a unified modeling path characterized by "fusion" rather than "decoupling," enabling a streamlined architecture to handle understanding, generation, and retrieval simultaneously with faster inference.
- **Core Idea**: **Utilize Discrete Flow Matching as the unified paradigm.** DFM starts from a fully corrupted sequence and iteratively denoises in parallel, naturally offering bidirectional information integration and parallel decoding acceleration. Coupled with **reconstruction-augmented unified representations**, a single encoder can preserve both high-level semantics and low-level details, supporting retrieval and multi-turn interaction under deep feature fusion.

## Method

### Overall Architecture
NExT-OMNI encodes vision, text, and audio into a unified sequence of discrete tokens. A shared backbone, initialized from an AR-LLM (Qwen2.5-7B), performs deep multimodal self-attention fusion at every layer. The entire sequence is denoised in parallel to predict target tokens under the Discrete Flow Matching paradigm, which are then decoded back to their respective modalities by lightweight modality-specific heads. Unlike prior works using "multi-encoders + MoE/MoT decoupling," NExT-OMNI trains **a single encoder to serve both understanding and generation**, producing unified representations directly usable for retrieval.

```mermaid
flowchart LR
    A[Text/Image/Video/Audio Input] --> B[Modal Encoder<br/>CLIP-ViT / Whisper + VQVAE]
    B --> C[Discrete token sequence x1]
    C --> D[Noising xt ~ pt·|x1]
    D --> E[Shared Backbone<br/>Per-layer Multimodal Self-Attention<br/>DFM Velocity Prediction]
    E --> F[Modal-specific Heads<br/>LM/Vision/Audio]
    F --> G[Any-to-Any Generation]
    E --> H[EOS token feature<br/>Cross-modal Retrieval]
```

### Key Designs

**1. Unified Modeling with Discrete Flow Matching: Turning any-to-any into parallel denoising.** Given an interleaved "vision-text-audio" sequence sampled from the target distribution, it is first converted into a discrete target token sequence $x_1=(x_1^1,\dots,x_1^D)$ using VQVAE encoders and a text tokenizer. At each step, a time $t\in[0,1]$ is uniformly sampled, and a noisy sequence $x_t$ is drawn according to the probability path $p_t(\cdot|x_1)$. The model takes $x_t$ and predicts $x_1$ with a training objective defined by the per-token expected cross-entropy: $L_{ce}=\mathbb{E}_{t,x_1,x_t}\big[-\sum_{i=1}^{D}\log p_{1|t}(x_1^i|x_t)\big]$. This paradigm allows all modalities to share the same denoising objective, eliminating AR causal masks and specialized diffusion/flow heads, keeping only lightweight token decoding heads to accelerate both training and response. A clever implementation detail is that the model does not directly consume discrete tokens; instead, it **extracts continuous representation vectors $c^M_{z_q}$ containing rich semantics and details from the encoder codebook**, which are aligned via lightweight projections and text embeddings before entering the backbone. This simple step significantly improves optimization.

**2. Reconstruction-Augmented Unified Representation: Single encoder for both semantics and details.** The authors perform a warmup training phase to enable vision/audio encoders to learn unified representations under two objectives: first, a reconstruction loss $L^M_{rec}=L^M_R+L^M_{VQ}+L^M_{G}$ aided by auxiliary VQVAE quantizers and modality decoders (quantized as $z_q=\arg\min_{c\in C^M}\|z^M-c\|_2$ to capture low-level details); second, a semantic alignment loss $L^M_{sem}$ (CLIP-style sentence-level contrastive $L^V_{constra}$ for vision and Whisper-style token-level caption alignment $L^A_{cap}$ for audio), resulting in $L^M_{total}=L^M_{rec}+L^M_{sem}$. Crucially, the **reconstruction loss is reused during DFM training**, with the overall objective written as $L_{overall}=\lambda_1 L_{ce}+\lambda_2 L^V_{rec}+\lambda_3 L^A_{rec}$, using GradNorm to dynamically balance the three gradient contributions. This prevents the model from biasing excessively toward high-level semantics during DFM training and losing fine-grained information. This preserves understanding/generation quality while making the unified representation "deeply fused" enough to support precise cross-modal retrieval by simply using the \<EOS\> token feature.

**3. Dynamic Length Generation Strategy (DGS): Addressing DFM's weakness in understanding tasks.** Parallel denoising in DFM is inherently less effective at variable-length text generation. The authors insert additional \<PAD\> tokens during training to align response sequences to multiples of a block size. During inference, leveraging the property that "simple tokens can be determined in a single step," the model dynamically expands to an appropriate preset length using block sizes as increments based on \<EOS\> confidence, followed by multi-step iterative denoising. This strategy brings understanding tasks (e.g., VQA) back to levels comparable with AR models at minimal cost; in ablations, DGS improved VQAv2 scores from 51.7 to 54.3.

**4. Adaptive Caching + Interleaved Single-modal Training: Putting "faster" into practice.** During inference, it was observed that most features change minimally across multiple denoising steps (a phenomenon similar to DLLM). Consequently, the entire instruction segment is cached and minimally updated, while the response segment is adaptively updated based on the "cosine similarity between value features and cached features." Combined with DFM's parallel decoding, this achieves a 1.2× speedup relative to AR architectures. On the training side, to avoid the waste of "uneven loads from random multimodal mixing," **only one modality is trained per batch**. Joint objectives are achieved through multi-task interleaving and gradient accumulation, resulting in a 1.4× training efficiency gain.

## Key Experimental Results

### Main Results

Omnimodal Understanding (average across three benchmarks; the strongest AR competitor OpenOmni scored 36.5):

| Model | OmniBench(T+A+V) | WorldSense(T+A+V) | AV-Odyssey | AVG. |
|------|------|------|------|------|
| OpenOmni | 37.4 | 37.2 | 32.8 | 36.5 |
| **NExT-OMNI** | **40.7** | **40.5** | **36.4** | **39.7** |

Ours achieves a +3.2 absolute improvement on average over OpenOmni. In multi-turn spoken interaction (Spoken QA, S→T average), NExT-OMNI leads with 62.0. In multi-turn visual interaction (OpenING), it achieves an AVG. of 55.0, significantly outperforming the same-architecture MMaDA (47.7), FUDOKI (44.5), and the AR-based SEED-X (50.2).

### Ablation Study

| Paradigm | Rep. | DGS | Recon. | VQAv2(Und) | GenEval(Gen) | InfoSeek(Ret) | AVG. |
|------|------|------|------|------|------|------|------|
| AR | Decoupled | × | × | 55.2 | 53.4 | 28.3 | 41.4 |
| DFM | Decoupled | × | × | 52.3 | 59.8 | 29.6 | 42.6 |
| DFM | Unified | × | × | 51.7 | 59.2 | 32.8 | 43.0 |
| DFM | Unified | ✓ | × | 54.3 | 59.4 | 33.1 | 43.9 |
| DFM | Unified | ✓ | ✓ | **56.2** | **62.6** | **33.7** | **45.6** |

### Key Findings
- **Paradigm Trade-offs**: Switching from AR to DFM results in a slight drop in understanding but significant gains in generation and retrieval. Moving to unified representations further decreases understanding due to granularity conflicts, but retrieval benefits consistently—confirming that "unified representations are better for generalized applications."
- **DGS Improves Understanding**: The Dynamic Length Generation Strategy pulls understanding back to levels comparable with AR, improving VQAv2 by +2.6.
- **Reconstruction Objective Improves Generation and Retrieval**: Reusing the reconstruction loss provides low-level details to features and suppresses excessive bias toward high-level semantics, increasing GenEval by +3.2 and the overall AVG. to 45.6.
- **DFM/Diffusion > AR/Hybrid for Retrieval**: Bidirectional information encoding (acting like BERT-style feature extraction) aggregates context better than causal masking. Decoupling schemes suffer in retrieval due to feature separation.

## Highlights & Insights
- **A Paradigm "First"**: The first fully Discrete Flow Matching-based open-source omnimodal foundation model, extending DFM from unimodal language/image modeling to unified any-to-any across text-image-video-audio.
- **Extra Dividends of Unified Representation**: Using the unified "understanding + generation" representation directly for cross-modal retrieval reveals an application dimension sacrificed by decoupled routes. Retrieval performance serves as a probe to measure whether "representations are truly unified."
- **Engineering and Paradigm Complementarity**: Reusing reconstruction loss, GradNorm balancing, interleaved single-modal training, and adaptive caching are all targeted designs to "fix where DFM is weak," successfully translating paradigm advantages into speed and multi-task coverage.

## Limitations & Future Work
- **Understanding Still Requires DGS Remediation**: DFM's inherent weakness in variable-length text generation is mitigated by engineering strategies, suggesting that pure DFM has not yet naturally surpassed AR in language-intensive understanding.
- **Preliminary Video Generation**: Long video is handled by a simplified strategy of "sampling 8 frames as multiple images"; video and long-audio generation remain at a "preliminary" level.
- **Limited Retrieval Training Scale**: Retrieval fine-tuning was performed only on a 100K subset of M-BEIR; generalization to larger scales and more retrieval scenarios remains to be verified.
- **Future Work**: The authors aim to extend NExT-OMNI to broader fields such as action trajectory generation for VLA and physical video generation for world models.

## Related Work & Insights
- **Comparison with AR/Hybrid Unified Models**: Compared to Janus (AR decoupled), Show-o (AR + discrete diffusion), and Bagel (AR + Diffusion decoupled MoT), this work wins systematically in the retrieval dimension via "DFM + unified representation," providing empirical proof that "fusion is superior to decoupling."
- **Comparison with Same-Paradigm Work**: Compared to MMaDA (discrete diffusion) and FUDOKI (DFM but decoupled), NExT-OMNI proves that DFM with unified representations is more versatile than DFM with decoupling.
- **Insight**: Evaluation solely on understanding and generation tends to underestimate the cost of decoupling schemes. Introducing cross-modal retrieval as an evaluation dimension more fairly exposes whether "representations are truly unified," providing a valuable reference for the benchmark design of future unified multimodal models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First fully DFM-based omnimodal foundation model, extending DFM to any-to-any and linking it to retrieval via unified representations with clear motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four task categories (understanding, spoken interaction, visual interaction, retrieval) plus key component ablations with broad comparisons; however, video generation and retrieval scale are relatively weak.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation regarding "decoupling vs. fusion," with the Fig. 2 pipeline and ablations providing well-attributed insights.
- **Value**: ⭐⭐⭐⭐ Provides a reproducible non-AR route (open-source code and weights) for unified multimodal modeling and elevates cross-modal retrieval as a useful probe for unity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FlowBind: Efficient Any-to-Any Generation with Bidirectional Flows](flowbind_efficient_any-to-any_generation_with_bidirectional_flows.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/multimodal_vlm/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[CVPR 2026\] Describe Anything Anywhere At Any Moment](../../CVPR2026/multimodal_vlm/describe_anything_anywhere_at_any_moment.md)
- [\[ICLR 2026\] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)
- [\[ICLR 2026\] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception](omni-captioner_data_pipeline_models_and_benchmark_for_omni_detailed_perception.md)

</div>

<!-- RELATED:END -->
