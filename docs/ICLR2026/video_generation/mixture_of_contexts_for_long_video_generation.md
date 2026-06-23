---
title: >-
  [Paper Note] Mixture of Contexts for Long Video Generation
description: >-
  [ICLR 2026][Video Generation][Diffusion Transformer] The authors reframe long video generation as an "internal information retrieval" problem and propose Mixture of Contexts (MoC) — a parameter-free yet trainable sparse attention routing module. It allows each query to dynamically select a few relevant chunks plus mandatory anchors (text + local window) while using causa
tags:
  - ICLR 2026
  - Video Generation
  - Diffusion Transformer
  - MoC
date: 2026-05-08
content_hash: ce8f39b27ded6175
---
# Mixture of Contexts for Long Video Generation

**Conference**: ICLR 2026  
**Code**: [https://primecai.github.io/moc/](https://primecai.github.io/moc/)  
**Area**: Video Generation / Long Video / Sparse Attention  
**Keywords**: Long Video Generation, Diffusion Transformer, Sparse Attention, Learnable Routing, Long-range Memory, MoC  

## TL;DR
The authors reframe long video generation as an "internal information retrieval" problem and propose Mixture of Contexts (MoC) — a parameter-free yet trainable sparse attention routing module. It allows each query to dynamically select a few relevant chunks plus mandatory anchors (text + local window) while using causal masking to avoid feedback loops. This maintains or even improves identity/motion/scene consistency in minute-long videos while pruning 85% of token pairs and reducing attention FLOPs by 7×.

## Background & Motivation
**Background**: Video generation based on Diffusion Transformers (DiT) can synthesize realistic multi-second clips. However, extending this to minutes or even hours reveals that the true bottleneck is **long-range memory** rather than image quality — the model must retrieve key events across long timelines without drift, collapse, or identity loss. The $O(L^2)$ cost of self-attention makes long sequences computationally and educationally infeasible: a 480p, 1-minute video expands to approximately 180,000 tokens.

**Limitations of Prior Work**: Previous works follow two paths to reduce costs. One is **compressing history into compact representations** (key frames, frame packs, or hidden states like FramePack/TTTVideo), which incurs details loss due to lossy compression. The second is **imposing fixed sparse/selection patterns** (Radial Attention, SparseVideoGen, etc.), but static sparsity cannot adapt to "which part of history is important right now." The closest work, LCT, expands the DiT context window to 8 shots but maintains full dense attention, causing FLOPs/VRAM to explode with $(8L_{shot})^2$.

**Key Challenge**: Efficiency and fidelity are encoded as a hard trade-off — compression summaries lose detail, and static sparsity selects inaccurately. Both limit long-range dependencies and narrative coherence.

**Goal**: Achieve minute-level memory at a cost close to short videos **without modifying the diffusion backbone or training recipe**, by allowing the model to learn to "recall the correct context at the correct moment."

**Core Idea**: **[Redefining the Task]** Long video generation is viewed as token-level "internal in-context retrieval" — each query accesses only the most relevant context chunks through learnable sparse routing; efficiency is a byproduct of retrieval rather than the primary goal.

## Method

### Overall Architecture
MoC replaces the dense attention in DiT with a content-aligned sparse routing layer that performs three tasks: (i) segmenting the multimodal token stream along frames/shots/captions into semantically homogeneous chunks; (ii) using a parameter-free top-k router for each query to route to a few relevant chunks plus mandatory anchors; (iii) using causal masks to force information to flow strictly forward. Selected keys are fed into a variable-length Flash-Attention kernel, skipping the rest to achieve near-linear rather than quadratic computation and memory.

```mermaid
flowchart LR
    A[Multimodal Token Stream<br/>Frames/Shots/Captions] --> B[Content-aligned Chunking]
    B --> C[Mean-pool each chunk<br/>into representative keys]
    C --> D[Query-pooled key<br/>Top-k selection]
    D --> E[Add mandatory anchors<br/>Full text caption + Local shot]
    E --> F[Causal mask<br/>Pruning j≥i edges → DAG]
    F --> G[Variable-length Flash-Attention<br/>Calculated on selected chunks only]
```

### Key Designs
**1. Dynamic Top-k Routing: A parameter-free dot-product engine as a retrieval engine.** Given a set of all chunks $\Phi$, each query $q_i$ selects only the $k$ most relevant chunks for attention: $\Omega(q_i)=\big[\arg\max_{\Omega^*}\sum_{\omega\in\Omega^*} q_i^\top \phi(K_\omega)\big]$, where $|\Omega^*|=k$. The descriptor $\phi$ is obtained by **mean-pooling** the keys within a chunk. Attention is then calculated only over the selected set: $\mathrm{Attn}(q_i,K,V)=\mathrm{Softmax}\!\big(q_i K_{\Omega(q_i)}^\top/\sqrt d\big)\cdot V_{\Omega(q_i)}$. Crucially, although top-k is non-differentiable, the router itself is **parameter-free**. Learning signals propagate back through the attention of the selected chunks — if the wrong chunk is selected, gradients flow back to its key/value to weaken its representation, thereby shaping query/key projections to be more discriminative. This "self-correction" relies on the inherent DiT features (as shown by DDAE, denoising autoencoders naturally learn semantically separable representations); thus, mean-pooling suffices to capture dominant semantics for video tokens that are often spatially and temporally redundant.

**2. Content-aligned Chunking + Dual Mandatory Anchors: Spending the sparsity budget on long-range dependencies.** Video DiTs use heterogeneous 3D+modality grids. Simple fixed-window slicing mixes static background patches with high-entropy motion tokens, polluting the mean-pooled keys. MoC instead segments along **frame/shot/modality boundaries**, making each chunk semantically homogeneous and geometrically local. Two **mandatory edges** are inserted alongside dynamic routing: (a) **Cross-modal sink** — every visual query must attend to all text tokens (text accounts for <1% of tokens but encodes style/identity/key actions, acting as a low-entropy anchor and gradient highway, significantly suppressing prompt-drift and the fading of rare attribute words); (b) **Local shot window** — every token always attends to its own shot, preserving local cues like object trajectories and lighting continuity. This allows the budget for long-range dependencies to be spent meaningfully on retrieval rather than redundant local modeling.

**3. Causal Routing Mask: Turning the interaction graph into a DAG to prevent loops.** Sparse routing naturally introduces directionality. Without order constraints, it can degenerate into pathological closed loops — ablation studies observed shot 9 strongly routing to shot 11, while shot 11 routed back to shot 9, forming an isolated two-node cycle. This causes these shots to lose connection with earlier shots, manifesting as frozen motion or repeated frames. MoC adds a causal mask during routing: any edge $(i\to j)$ satisfying $j\ge i$ is pruned before top-k. This transforms the routing graph into a Directed Acyclic Graph (DAG), ensuring information flows strictly forward, eliminating feedback pairs and facilitating richer long-range dependencies while stabilizing training.

**4. Robustness and Efficiency Engineering: drop-off/drop-in + Per-head routing + Flash kernel.** To address the "dead expert" problem common in MoE, **context drop-off** randomly discards $\lfloor p_{drop}\cdot k\rfloor$ chunks from the top-k (where $p_{drop}\sim\mathrm{Uniform}(0,p_{max})$) to force model robustness against occasional missing context. **Context drop-in** injects $m\sim\mathrm{Poisson}(\lambda)$ additional chunks to activate underused paths and balance routing distributions. Routing is performed **independently per layer per head**, acting as an ensemble of $L_{layers}\times H_{heads}$ routers. While a single head is strictly sparse, the union across heads/layers covers a much larger context, avoiding information bottlenecks of static global selection. Implementation-wise, `torch.bucketize` and prefix sum tables generate variable-length chunks, and `segment_reduce` performs online mean-pooling to avoid materializing entire blocks. These are packed into a single variable-length Flash-Attention call. All operations are head-independent and tensor-parallel compatible. Per-head FLOPs for MoC are approximately $Ld+2LCd+4Lk\bar m d$, compared to $4L^2d$ for dense attention; the ratio $\approx 2L/(Cd+2k\bar m)$ grows linearly with sequence length.

## Key Experimental Results

### Main Results Table
The base model is LCT (3B MMDiT, the only architecture supporting long multi-shot general scenarios), evaluated on 8-shot, 8-seconds-per-shot 480p 12fps (approx. 180k tokens / 64s scene) using VBench:

| Method | Subject↑ | Background↑ | Motion Smooth↑ | Dynamic Degree↑ | Aesthetic↑ | Image↑ | Sparsity↑ | FLOPs↓ |
|------|----------|-------------|----------------|-----------------|-----------|--------|-----------|--------|
| LCT (dense) | 0.9378 | 0.9526 | 0.9859 | 0.4583 | 0.5436 | 0.5140 | 0% | 1.7×10¹³ |
| **MoC (Ours)** | **0.9421** | **0.9535** | **0.9920** | **0.5625** | **0.5454** | 0.5003 | **85%** | **2.3×10¹²** |

At 85% sparsity, FLOPs are reduced by >7× with a 2.2× end-to-end speedup, while most metrics actually improve.

### Ablation Study Table

| Design | Phenomenon when removed |
|------|--------------|
| Causal Mask | Two-node loops occur (shot 9↔11), motion stagnation/repeated frames, identity drift (Fig. 2) |
| Content-aligned Chunking | Mean-pooled keys become polluted; top-k budget is wasted on internally inconsistent keys |
| Cross-modal sink / Local shot window | Prompt-drift intensifies, rare attribute words fade, semantic breakdown at scene cuts |

### Key Findings
- **Efficiency as a byproduct of retrieval**: After pruning 85% of token pairs, computation is redistributed from redundant frames to key visual events. Dynamic Degree increases from 0.46 to 0.56, while image quality decreases only slightly.
- **Mean-pooling is sufficient**: The first principal component of local tokens after patch embedding often explains >90% of the variance; the arithmetic mean serves as an estimate of this component. This works zero-shot when applied directly to pre-trained models.
- **Qualitatively indistinguishable from dense**: Even with over 3/4 of attention computation pruned, the results are visually nearly identical to LCT, maintaining fine details and abstract semantics across hundreds or thousands of frames.

## Highlights & Insights
- **Paradigm Shift**: The first to demonstrate that "learnable sparse context routing" can serve as a long-range memory engine, where memory/consistency **emerge** with data scale and progressive sparsification, without requiring 3D priors or explicit heuristics like FoV.
- **Parameter-free yet Learnable**: The router requires zero additional parameters; learning signals are backpropagated through attention, cleverly bypassing the non-differentiability of top-k.
- **Plug-and-play**: The method is engineering-friendly, allowing the replacement of attention layers for fine-tuning from LCT weights without changing the backbone or training recipe. Mandatory cross-modal edges also improve the controllability of text-guided editing.

## Limitations & Future Work
- **Dependency on LCT**: Experiments were only verified on the single available multi-shot architecture; generalization to other video DiTs remains to be tested.
- **Minor drop in appearance fidelity**: Image Quality slightly decreased from 0.514 to 0.500, suggesting a trade-off between increased motion budget and appearance fidelity.
- **Coarse mean-pooling descriptors**: Descriptors might lack discriminative power for high-entropy internal chunks; more robust chunk descriptors could further improve performance.
- **Limited evaluation dimensions**: Principally relies on VBench automatic metrics; systemic evaluations for human preference or narrative coherence are lacking. Tests beyond minute-level (e.g., hour-level) have not yet been conducted.

## Related Work & Insights
- **Long Video Generation**: TECO, NUWA-XL, CausVid, MAGI-1, SkyReels-V2 use autoregressive/hierarchical approaches; FramePack/TTTVideo use fixed-length hidden state compression (all lossy). LCT expands the dense window but at a quadratic cost.
- **Video Sparse Attention**: SparseVideoGen, STA, Radial Attention, VMoBA, VSA mostly prune dense maps or use fixed sparse priors, focusing on accelerating short videos. MoC learns end-to-end routing of context sources, focusing on long-range memory.
- **In-Context Learning in Visual Generation**: WorldMem, Context-as-Memory, and VMem use external memory banks + FoV retrieval. IC-LoRA/OminiControl/FLUX-Context demonstrate the in-context capabilities of DiT. MoC connects the idea of "end-to-end routing among multiple context sources" to this line of research.
- **Inspiration**: Sparse attention ideas from LLMs like MoBA/NSA can be transferred to heterogeneous multimodal videos, but chunking/anchors/causality must be redesigned for video structures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reframing long video generation as internal retrieval + parameter-free learnable routing, with emergent memory capabilities, is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid core comparisons and key ablations (causality/chunking), though limited to a single base model and lacking human evaluation or hour-level verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete derivations for formulas and FLOPs, and convincing visualization of loops in Fig. 2.
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play ability to achieve minute-level memory at short-video costs provides significant value for the deployment of long video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation](moga_mixture-of-groups_attention_for_end-to-end_long_video_generation.md)
- [\[ICLR 2026\] VMoBA: Mixture-of-Block Attention for Video Diffusion Models](vmoba_mixture-of-block_attention_for_video_diffusion_models.md)
- [\[ICLR 2026\] NarrLV: Towards a Comprehensive Narrative-Centric Evaluation for Long Video Generation](narrlv_towards_a_comprehensive_narrative-centric_evaluation_for_long_video_gener.md)
- [\[ICLR 2026\] LongLive: Real-time Interactive Long Video Generation](longlive_real-time_interactive_long_video_generation.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)

</div>

<!-- RELATED:END -->
