---
title: >-
  [Paper Note] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
description: >-
  [CVPR 2026][Multimodal VLM][Diffusion Language Models] Addressing the gap where "current Multimodal Large Language Models (MLLMs) almost exclusively follow the autoregressive paradigm and the diffusion path remains unverified," this paper grafts visual instruction tuning onto the masked diffusion language model LLaDA to create a pure diffusion MLLM—LLaDA-V. By leveraging bidirectional attention to better capture visual-spatial relationships, it refreshes the SOTA for pure dif…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Diffusion Language Models"
  - "Multimodal Large Language Models"
  - "Visual Instruction Tuning"
  - "Masked Diffusion"
  - "Bidirectional Attention"
date: 2026-05-08
content_hash: c6ece775a059bbdf
---

# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/You_LLaDA-V_Large_Language_Diffusion_Models_with_Visual_Instruction_Tuning_CVPR_2026_paper.html)  
**Code**: https://github.com/ML-GSAI/LLaDA-V  
**Area**: Multimodal VLM / Diffusion Models  
**Keywords**: Diffusion Language Models, Multimodal Large Language Models, Visual Instruction Tuning, Masked Diffusion, Bidirectional Attention

## TL;DR
Addressing the gap where "current Multimodal Large Language Models (MLLMs) almost exclusively follow the autoregressive paradigm and the diffusion path remains unverified," this paper grafts visual instruction tuning onto the masked diffusion language model LLaDA to create a pure diffusion MLLM—LLaDA-V. By leveraging bidirectional attention to better capture visual-spatial relationships, it refreshes the SOTA for pure diffusion MLLMs across 18 benchmarks and outperforms the autoregressive baseline LLaMA3-V on 11 tasks using the same training data.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) can process image/audio/video alongside text to generate natural language responses. However, existing methods overwhelmingly rely on **autoregressive (AR)** language models (e.g., LLaVA, Qwen2-VL, which all perform next-token prediction), with minimal exploration of alternative probability modeling approaches. Recent attempts to incorporate diffusion into MLLMs either still rely on AR for language capabilities (hybrid approaches) or use discrete diffusion with weak language modeling performance, yielding suboptimal results.

**Limitations of Prior Work**: The pure diffusion approach has long remained unproven. It was not until LLaDA scaled masked diffusion language models to 8B—competing with LLaMA3-8B-Instruct on various downstream tasks—that diffusion language models first gained the potential to serve as LLM backbones. However, the potential of LLaDA for multimodal understanding has **remained completely unexplored**. The core research question is: Can a pure diffusion MLLM for both training and sampling be competitive with autoregressive models?

**Key Challenge**: The authors bet on a specific realization—**bidirectional attention is naturally better suited for multimodal understanding**. While the causal (left-to-right) attention in AR fits sequential text generation, visual inputs possess spatial relationships and contextual dependencies that require "looking at all positions simultaneously." Since diffusion language models inherently use bidirectional attention, they theoretically should provide a more unified modeling of image-text inputs and stronger multimodal understanding. The problem lies in extending LLaDA's unimodal masked diffusion objective to "multi-turn, image-inclusive" multimodal dialogues.

**Goal**: (1) Design a training objective for masked diffusion language models capable of handling multi-turn multimodal dialogues; (2) Determine whether to use causal or bidirectional attention in multimodal scenarios; (3) Adapt the inference workflow for multimodal dialogues; and (4) Design a multi-stage training strategy to progressively build language-vision alignment, instruction following, and multimodal reasoning.

**Key Insight**: Rather than inventing a new architecture, this work adopts the proven **visual instruction tuning framework (Vision Tower + MLP Connector + Language Tower)** effective in various AR-MLLMs. It replaces only the AR language tower with the diffusion-based LLaDA, thereby cleanly isolating "diffusion vs. autoregressive" as the single variable.

**Core Idea**: Use masked diffusion instead of next-token prediction for multimodal instruction tuning—image features and prompts remain unmasked, while responses are randomly masked. The model learns to recover masked tokens conditioned on clean image-text inputs, utilizing LLaDA’s global bidirectional attention to better capture visual spatial dependencies.

## Method

### Overall Architecture
LLaDA-V follows the classic tripartite visual instruction tuning architecture: a **Vision Tower** (SigLIP 2, siglip2-so400m-patch14-384), an **MLP Connector** (two-layer MLP projecting visual features into the LLaDA embedding space), and a **Language Tower** (LLaDA-8B-Instruct, a masked diffusion language model). The fundamental difference from AR-MLLMs lies in the generation process: while AR models use next-token prediction to generate responses given image-text conditions, LLaDA-V treats the response as a masked sequence and iteratively recovers it through the reverse process of masked diffusion (see Fig. 2 in the paper). The pipeline is understood in three steps: only responses are masked during training while image/text remain clean to learn a masked token prediction objective; inference starts from a fully masked response and iteratively reveals tokens as the timestep drops from 1 to 0; the training process progresses through three stages: "Language-Image Alignment → Visual Instruction Tuning → Multimodal Reasoning Enhancement." Regarding the attention mechanism, the authors found that **bidirectional (no mask) outperforms causal**, and thus bidirectional attention is employed throughout.

### Key Designs

**1. Masked Diffusion Training Objective for Multi-turn Multimodal Dialogue: Masking only responses and replacing next-token with diffusion loss**

This modification adapts LLaDA from pure text to multimodality, addressing the lack of existing multimodal instruction tuning objectives for diffusion language models. Given a single-image, two-turn dialogue $(v,p^1_0,r^1_0,p^2_0,r^2_0)$ (where $v$ is the image representation, $p$ is the prompt, and $r$ is the ground-truth response), the image features $v$ and all prompts $p$ remain clean during training. Only responses are randomly masked following a diffusion schedule using $[M]$ tokens. The objective is to maximize the log-likelihood of predicting the masked tokens:

$$L(\omega) = -\mathbb{E}\Big[\tfrac{1}{t}\sum_{i}\sum_{j}\mathbf{1}[r^{1,i}_t=[M]\ \text{or}\ r^{2,j}_t=[M]]\cdot\log p_\omega(r^{1,i}_0,r^{2,j}_0\mid v,p^1_0,r^1_t,p^2_0,r^2_t)\Big]$$

where $r^1_t, r^2_t$ are the responses masked at time $t$. This objective is proven to be an upper bound of the negative log-likelihood of the masked tokens. Intuitively, it involves "filling in the blanks" in the response given clean image-text conditions, bridging masked diffusion with the visual instruction tuning framework.

**2. Bidirectional (no-mask) Attention: Enabling the model to view the full dialogue context for better visual-spatial dependency**

This addresses the choice between causal and bidirectional attention for multimodal understanding. From a systems perspective, causal attention (where earlier turns cannot see later turns) is attractive for multi-turn inference as KV states can be reused. However, the authors argue that bidirectional attention integrated across the entire dialogue context allows for a better understanding during masked prediction (consistent with why recent video diffusion models use bidirectional attention for temporal consistency). Ablations (Tab. 3, 12 benchmarks) show that no-mask performs better on 7/12 benchmarks, with an average score (excluding MME) of $49.73 > 49.03$ for causal. Thus, LLaDA-V adopts bidirectional attention. Further analysis confirmed that LLaDA-V exhibits global bidirectional attention behavior, whereas LLaMA3-V is strictly local and causal. This structural advantage allows LLaDA-V to capture complex spatial dependencies. Combined with Fast-dLLM’s approximate cache reuse, the efficiency deficit of bidirectional attention is largely mitigated.

**3. Three-stage Training Strategy: Alignment → Instruction Tuning → Reasoning Enhancement**

This strategy defines how to evolve a pure language diffusion model into a strong multimodal model. The first two stages follow mature practices like LLaVA-NeXT, while the third stage adds reasoning enhancement:

- **Stage 1 LVM-Language Alignment**: Freezes the language and vision towers and trains only the MLP projector using LLaVA-Pretrain to align visual representations with the LLaDA embedding space.
- **Stage 2 Visual Instruction Tuning**: Unfreezes the full model for training on large-scale instruction data (MAmmoTH-VL), divided into two phases: first using SI-10M (10 million single-image samples) for basic understanding, then OV-2M (2 million mixed samples) for multi-image and video expansion.
- **Stage 3 Multimodal Reasoning Enhancement**: Uses VisualWebInstruct (900k QA pairs with Chain-of-Thought) for reasoning training. To prevent the model from becoming overly verbose, **balanced reasoning training** is applied—mixing VisualWebInstruct with OV-2M. Prompts for OV-2M include a `/no think` tag while 50% of reasoning data includes a `/think` tag, allowing the model to switch between direct answers and expanded reasoning.

**4. Masked Diffusion Reverse Sampling Inference + Fast-dLLM Acceleration: Iterative response revelation with adjustable precision-throughput**

During inference, given a new prompt, the model initializes the entire response sequence as $[M]$ tokens. It follows the reverse masked diffusion process from state $r_t$ to $r_s$ ($s < t$, decreasing masking level). Each step predicts all $[M]$ tokens conditioned on $v, p_0, r_t$, then re-masks a proportion $s/t$ of the predictions back to $[M]$. The re-masking strategy uses LLaDA’s **low-confidence policy**, prioritizing the re-masking of low-confidence tokens. Since diffusion cannot natively use KV cache and multi-token decoding often degrades performance, the authors adapted Fast-dLLM’s approximate KV cache. It includes a configurable cache refresh interval $r$; larger intervals increase throughput with minimal accuracy loss (on MathVista, increasing $r$ from 2 to 48 yielded a 3.3× speedup with only a 2.3% accuracy drop). On MathVerse, LLaDA-V (with Fast-dLLM) reached 32.4 tokens/s with 28.5% accuracy, comparable to the AR baseline LLaMA3-V (30.5 tokens/s, 29.0%).

## Key Experimental Results

The language tower is LLaDA-8B-Instruct, the vision tower is SigLIP 2, and the projector is a randomly initialized two-layer MLP. For fair comparison, the AR baseline **LLaMA3-V** replaces the language tower with LLaMA3-8B-Instruct while keeping all other settings and training protocols identical.

### Main Results: Comparison with AR Baselines and MLLMs (Partial)

| Model | Type | Language Tower | MMMU(val) | MMMU-Pro(std) | MMStar | MMB(en-dev) | MuirBench | MLVU | VideoMME |
|------|------|------|------|------|------|------|------|------|------|
| Qwen2-VL | AR | Qwen2-7B | 54.1 | 43.5 | **60.7** | - | - | - | - |
| LLaMA3-V (AR Baseline) | AR | LLaMA3-8B | 45.4 | 28.3 | 56.5 | 79.8 | 47.4 | 57.5 | 55.8 |
| LaViDa-L | Diff. | LLaDA-8B | 43.3 | - | - | 70.5 | - | - | - |
| Dimple | Diff. | Dream-7B | 45.2 | - | - | 74.6 | - | - | - |
| MMaDA | Diff. | LLaDA-8B | 30.2 | - | - | 68.5 | - | - | - |
| **LLaDA-V (Ours)** | Diff. | LLaDA-8B | **48.6** | **35.2** | 60.1 | **82.9** | **48.3** | **59.5** | **56.1** |

LLaDA-V sets new SOTA among all pure/hybrid diffusion MLLMs. Despite the LLaDA-8B language tower being slightly weaker than LLaMA3-8B, LLaDA-V outperforms LLaMA3-V on 6 out of 9 knowledge/math benchmarks and leads in multi-image (MuirBench) and video tasks (MLVU, VideoMME). Its score of 60.1 on MMStar approaches the strong AR model Qwen2-VL (60.7). Its weaknesses are in chart/document (AI2D, DocVQA) and real-world scenarios (RealworldQA), which rely more heavily on the language tower's inherent strength.

### Ablation Study: Attention Masking Strategy (Select 12 Benchmarks)

| Configuration | MMMU(val) | MMStar | MMB(en-dev) | MuirBench | Mean (w/o MME) |
|------|------|------|------|------|------|
| Dialogue Causal Mask | 42.89 | 49.60 | 75.42 | 28.69 | 49.03 |
| **No Mask (Bidirectional, Ours)** | **44.67** | **49.79** | **76.71** | **33.88** | **49.73** |

### Key Findings
- **Bidirectional attention is the key mechanism for outperformance**: No-mask performed better on 7/12 benchmarks with an average gain of +0.70, especially on MuirBench (multi-image), jumping from 28.69 to 33.88.
- **Architectural advantages outweigh weaker language backbones**: LLaDA-V's language tower (LLaDA-8B) is weaker than LLaMA3-8B, yet it outperforms LLaMA3-V on 11 tasks. Attention analysis suggests its global/bidirectional structure is superior for capturing spatial dependencies.
- **Good data scalability**: LLaDA-V shows consistent performance gains as instruction data increases. On knowledge benchmarks like MMMU/MMMU-Pro, its scaling behavior even surpasses LLaMA3-V.
- **No efficiency bottleneck**: Utilizing Fast-dLLM, MathVerse performance (32.4 vs 30.5 tokens/s) is comparable to the AR baseline, with a flexible trade-off knob via the refresh interval $r$.
- **Transparent limitations**: Understanding of charts/documents and real-world scenes remains inferior to LLaMA3-V, and overall performance lags behind Qwen2-VL, primarily due to the weaker LLaDA-8B backbone and lack of preference alignment.

## Highlights & Insights
- **"Replacement of the language tower" is a clean controlled experiment**: LLaMA3-V and LLaDA-V share the vision tower, projector, data, and training protocol. The only variable is the AR vs. Diffusion language tower, making the conclusion regarding diffusion's multimodal benefits highly credible.
- **Validating the intuition that bidirectional attention suits vision**: The dual evidence from ablations and attention pattern analysis—analogous to how video diffusion uses bidirectional attention for temporal consistency—provides a mechanistic explanation for why diffusion MLLMs excel at spatial/multi-image tasks.
- **Transferable paradigm**: The masked diffusion objective on responses while keeping image-text conditions clean can be applied to any scenario expanding diffusion language models to new multimodal instructions.
- **The "Aha!" moment**: A **notably weaker language tower** paired with a **more suitable attention structure** can outperform a stronger language tower in AR models for multimodal tasks, suggesting that MLLM performance is driven by modeling paradigms and attention structures as much as by language backbone strength.

## Limitations & Future Work
- **Bottlenecked by the language tower**: LLaDA-8B lacks preference alignment and is generally weaker than Qwen2-7B, causing LLaDA-V to trail Qwen2-VL on most benchmarks. Its ceiling is locked by the language backbone.
- **Structural efficiency shortcomings**: Diffusion language models cannot natively use KV cache, and multi-token decoding can lead to performance drops. While Fast-dLLM mitigates this, it essentially trades approximation for speed.
- **Exploratory nature**: The primary contribution is "verifying feasibility + providing insights" rather than proposing a revolutionary new architecture.
- **Weak real-world/document understanding**: These tasks require fine-grained OCR and complex linguistic reasoning that the current masked diffusion + weak language tower combination cannot yet master.

## Related Work & Insights
- **vs. LLaDA (Pure Text Diffusion)**: This work extends LLaDA to multimodality by adding a vision tower, MLP projector, and modified training objective. Interestingly, while the LLaDA backbone is weaker than LLaMA3, LLaDA-V performs better on many tasks, suggesting an inherent "multimodal dividend" for diffusion.
- **vs. LLaMA3-V (AR Baseline)**: Using identical setups, LLaDA-V outperforms it on 11 tasks (especially multi-image/video) due to bidirectional attention, though it falls short in charts/real-world scenes.
- **vs. Other Pure Diffusion MLLMs**: Works like D-DiT, LaViDa, and MMaDA also explore discrete diffusion, but LLaDA-V sets a new SOTA and provides more systematic evidence regarding scaling, attention patterns, and controlled AR comparisons.
- **vs. Hybrid AR-Diffusion**: Hybrid approaches still rely on AR for language. This work maintains **pure diffusion for both training and sampling**, proving that the pure diffusion route is competitive.

## Rating
- Novelty: ⭐⭐⭐⭐ First MLLM with total diffusion training and sampling; provides solid feasibility verification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 benchmarks + controlled AR comparisons + scaling analysis + attention ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear research questions, design, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐ Proves diffusion as a scalable alternative to AR-MLLMs and provides open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](../../ICML2025/multimodal_vlm/parrot_multilingual_visual_instruction_tuning.md)
- [\[CVPR 2026\] DEVA: Fine-tuning Multimodal Large Language Models for Visual Perception Tasks](deva_fine-tuning_multimodal_large_language_models_for_visual_perception_tasks.md)
- [\[CVPR 2026\] Streaming Video Instruction Tuning (Streamo)](streaming_video_instruction_tuning.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
