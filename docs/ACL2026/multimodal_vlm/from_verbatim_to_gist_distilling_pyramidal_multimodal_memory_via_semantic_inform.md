---
title: >-
  [Paper Note] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck
description: >-
  [ACL 2026][Multimodal VLM][Long Video Understanding] This paper proposes MM-Mem, a pyramidal multimodal memory architecture inspired by Fuzzy Trace Theory (FTT). The memory is organized into three hierarchical layers — a Sensory Buffer (vision-dominant), an Episodic Stream (event-level summaries), and a Symbolic Schema (knowledge graph) — and achieves SOTA performance on 4 long-video benchmarks by compressing redundancy bottom-up via SIB-GRPO (Semantic Information Bottleneck + RL) and retrieving top-down via entropy-driven adaptive depth selection.
tags:
  - ACL 2026
  - Multimodal VLM
  - Long Video Understanding
  - Multimodal Memory
  - Information Bottleneck
  - Fuzzy Trace Theory
  - Reinforcement Learning
date: 2026-05-08
content_hash: eb8f9decd7684410
---

# From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck

**Conference**: ACL 2026
**arXiv**: [2603.01455](https://arxiv.org/abs/2603.01455)
**Code**: [GitHub](https://github.com/EliSpectre/MM-Mem)
**Area**: Multimodal VLM
**Keywords**: Long Video Understanding, Multimodal Memory, Information Bottleneck, Fuzzy Trace Theory, Reinforcement Learning

## TL;DR

This paper proposes MM-Mem, a pyramidal multimodal memory architecture inspired by Fuzzy Trace Theory (FTT). The memory is organized into three hierarchical layers — a Sensory Buffer (vision-dominant), an Episodic Stream (event-level summaries), and a Symbolic Schema (knowledge graph) — and achieves SOTA performance on 4 long-video benchmarks by compressing redundancy bottom-up via SIB-GRPO (Semantic Information Bottleneck + RL) and retrieving top-down via entropy-driven adaptive depth selection.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) excel at short-term perception but are constrained by context window limits and static memory mechanisms in long-video understanding. Existing approaches fall into two extremes: vision-centric methods (e.g., LongVA, VideoRAG) densely sample visual frames, incurring high latency and redundancy; text-centric methods (e.g., Vgent) convert video into textual memory, losing fine-grained details and inducing hallucinations.

**Limitations of Prior Work**: (1) Vision-centric methods accumulate massive visual tokens, causing severe computational redundancy while neglecting high-level semantics; (2) text-centric methods perform lossy compression via captioning, discarding critical visual cues and introducing ambiguity; (3) existing memory mechanisms are static, unlike the dynamic organization of human memory; (4) dynamic memory management in multimodal settings remains severely underexplored.

**Key Challenge**: Long-video understanding simultaneously requires fine-grained visual detail retention (for precise verification) and high-level semantic abstraction (for cross-event reasoning), yet existing methods can only achieve one — there is a fundamental trade-off between visual fidelity and semantic abstraction.

**Goal**: Design a hierarchical multimodal memory architecture that enables progressive distillation from fine-grained perception to high-level cognition, while supporting dynamic memory compression and adaptive retrieval.

**Key Insight**: The cognitive psychology concept of Fuzzy Trace Theory (FTT) posits that human memory maintains two parallel traces: *verbatim* (precise perceptual details) and *gist* (abstract semantic meaning). Vision naturally corresponds to verbatim and text to gist — this cross-modal complementarity can be directly mapped onto a hierarchical memory architecture.

**Core Idea**: Construct a three-layer pyramidal memory that transitions progressively from vision-dominant to text-dominant bottom-up; leverage information bottleneck theory to guide compression (SIB-GRPO); and employ entropy-driven top-down retrieval to dynamically switch between abstraction and detail.

## Method

### Overall Architecture

MM-Mem takes a long video stream as input and constructs a three-layer pyramidal memory offline: (1) the **Sensory Buffer** retains visual representations of keyframes along with brief textual labels; (2) the **Episodic Stream** generates event-level representations via clustering and summarization; (3) the **Symbolic Schema** constructs an entity knowledge graph. At query time, retrieval proceeds top-down: the knowledge graph (gist) is consulted first; if uncertainty remains, the system descends to the episodic layer; only when uncertainty persists does it access the raw visual frames (verbatim).

### Key Designs

1. **Three-Layer Pyramidal Memory Structure**

    - **Function**: Achieves progressive abstraction from perception to cognition.
    - **Mechanism**: The **Sensory Buffer** $\mathcal{M}_{sens} = \{(v_{t,i}, l_{t,i}, \tau_{t,i})\}$ performs content-adaptive temporal segmentation (PySceneDetect), selects keyframes based on inter-frame variation, and stores visual representations, textual labels, and timestamps. The **Episodic Stream** organizes sensory items into compact event sequences via a decision operator $\psi(m_{t,i}, e^\star) \in \{ADD\_NEW, MERGE, DISCARD\}$, then applies K-means clustering to select representative prototypes. The **Symbolic Schema** constructs a knowledge graph $\mathcal{G} = (\mathcal{N}, \mathcal{E})$, where nodes comprise episodic units and entity prototypes, and edges encode semantic relations and grounding pointers — grounding edges anchor textual concepts back to concrete visual evidence.
    - **Design Motivation**: Each layer corresponds to a distinct abstraction level in FTT. The key innovation lies in the grounding edges, which ensure that high-level textual abstractions remain anchored to visual evidence, thereby mitigating the hallucination problem inherent to purely text-centric approaches.

2. **SIB-GRPO: Information Bottleneck-Driven Memory Compression**

    - **Function**: Achieves an optimal balance between compressing redundancy and preserving task-relevant semantics.
    - **Mechanism**: The sensory-to-episodic transition is modeled as a stochastic compression problem, optimizing the information bottleneck objective $\min_{p_\theta(m|x)} [I(X;M) - \beta I(M;Y)]$, where $X$ is the sensory memory, $M$ is the episodic representation, and $Y$ is the downstream VQA answer. A variational decoder and a quality-quantity prior $r(m) \propto p_{ref}(m) \cdot e^{-\lambda|m|}$ are introduced to balance expressive quality and length control. Since episodic traces are discretely generated, a GRPO-style RL procedure trains the memory manager: $G$ candidate traces are sampled, a scalar reward $r(s,m) = R_{vqa} - \beta_1 \cdot Length(m) - \beta_2 \cdot \log\frac{\pi_{\theta_{old}}}{\pi_{ref}}$ is computed, and within-group normalized advantages are used to optimize a PPO clipped surrogate objective.
    - **Design Motivation**: Classical information bottleneck assumes continuous variables and cannot be directly applied to discrete text generated by LLMs. GRPO translates the IB principle into an RL objective trainable with sequence-level feedback; the quality-quantity prior functions analogously to the trust-region constraint in RLHF.

3. **Entropy-Driven Top-Down Retrieval**

    - **Function**: Adaptively selects retrieval depth according to query difficulty.
    - **Mechanism**: Inspired by inverse hierarchical theory, retrieval begins at the Symbolic Schema layer (most abstract). The system maintains a posterior distribution over answer candidates $p_i^{(s)} = p(a_i | \mathcal{Q}, R_{\leq s})$ and computes entropy $H_s(\mathcal{Q}) = -\sum_i p_i^{(s)} \log p_i^{(s)}$. Retrieval halts when $H_s \leq \gamma$ or when the entropy reduction $\Delta H_s$ over consecutive steps falls below threshold $\epsilon$. High-level textual retrieval rapidly narrows the semantic search space, while low-level visual retrieval is triggered only under high uncertainty — realizing an adaptive computation-accuracy trade-off.
    - **Design Motivation**: Not all queries require access to raw visual frames. Temporal reasoning can be resolved at the knowledge graph level; fine-grained counting tasks require descending to the sensory layer. Adaptive depth avoids unnecessary computational overhead.

### Loss & Training

The SIB-GRPO objective is: $J_{SIB-GRPO}(\theta) = \mathbb{E}[\frac{1}{G}\sum_{i=1}^{G} \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)]$. The backbone model is Qwen3-VL-8B, fine-tuned using the SWIFT framework with $\beta_1=0.1$, $\beta_2=0.3$, and temperature $= 0.0$.

## Key Experimental Results

### Main Results

**Video-MME Long Video Understanding (Overall Accuracy)**

| Method | Type | w/o Subtitles | w/ Subtitles |
|--------|------|--------------|-------------|
| Gemini 1.5 Pro | Proprietary | 75.0 | 81.3 |
| Qwen2-VL-72B | Open-source 72B | 71.2 | 77.8 |
| Vgent | Agent | 68.9 | 74.3 |
| **MM-Mem (Ours)** | **Agent 8B** | **72.4** | **78.1** |

**Streaming Video VStream-QA-Ego**

| Method | Accuracy | Score |
|--------|----------|-------|
| Flash-VStream | 59.0 | 3.9 |
| **MM-Mem** | **62.5** | **4.1** |

### Ablation Study

**Video-MME w/o Subtitles — Component Ablation**

| Configuration | Short | Medium | Long | Overall |
|--------------|-------|--------|------|---------|
| Full (MM-Mem) | 81.5 | 69.6 | 66.1 | 72.4 |
| w/o SIB-GRPO | ~79 | ~68 | ~63 | ~70 |
| w/o Hierarchical Memory | ~77 | ~66 | ~61 | ~68 |

### Key Findings

- MM-Mem surpasses all open-source MLLMs (including 72B models) and most agent systems using only an 8B backbone.
- The largest gains are observed in the Long split — SIB-GRPO is particularly critical for compressing long-range temporal dependencies.
- On HD-EPIC++, MM-Mem (30.28%) outperforms Qwen3-VL-8B (25.88%) by 4.4 points, demonstrating superior fine-grained aggregation capability for egocentric long videos.
- Efficiency analysis: inference latency is only 5.35 s/min of video; VRAM consumption is 17.8 GB, below Qwen3-VL-8B's 22.8 GB.
- t-SNE visualizations confirm that the sensory layer preserves domain-specific visual details, while the episodic layer exhibits naturally emergent semantic clustering.

## Highlights & Insights

- The mapping from FTT to engineering implementation is remarkably natural — the correspondence of vision = verbatim and text = gist is elegant in its simplicity, and the grounding edges ensure the two traces remain coupled.
- The combination of information bottleneck theory with GRPO is broadly generalizable — any scenario requiring a trade-off between information retention and compression (e.g., chunk selection in RAG) can draw on this paradigm.
- Entropy-driven adaptive retrieval depth is a practically useful design choice, eliminating the need to manually engineer retrieval strategies for each query type.

## Limitations & Future Work

- The computational overhead of memory construction, though amortizable, remains non-trivial — further distillation is required for edge deployment scenarios.
- The approach depends on the quality of upstream visual encoders and captioners — noise introduced at the sensory layer propagates upward through the hierarchy.
- SIB-GRPO currently uses task-driven VQA rewards; defining appropriate rewards in unsupervised settings without explicit downstream tasks remains an open problem.
- Systematic evaluation on ultra-long videos (>2 hours) has not been conducted.

## Related Work & Insights

- **vs. Vgent**: Vgent relies on purely textual memory; MM-Mem outperforms it by 3.5 points on Video-MME (72.4 vs. 68.9), as Vgent's text compression discards fine-grained visual evidence.
- **vs. VideoRAG**: A vision-centric method with higher VRAM consumption (23.0 vs. 17.8 GB) and lower performance (60.5 vs. 72.4), owing to redundancy accumulated through dense visual storage.
- **vs. A-Mem**: A text-centric dynamic memory system lacking multimodal grounding, which cannot descend to the sensory layer for visual verification when required.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Triple contributions: FTT → pyramidal memory mapping; IB → GRPO theoretical innovation; entropy-driven retrieval.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 benchmarks (offline + streaming + egocentric) + ablation study + efficiency analysis + t-SNE visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear; the bridge from cognitive science to engineering is natural and well-motivated.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a reusable cognitive architecture paradigm for memory systems in long-video agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[AAAI 2026\] EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](../../AAAI2026/multimodal_vlm/em-kd_distilling_efficient_multimodal_large_language_model_w.md)

</div>

<!-- RELATED:END -->
