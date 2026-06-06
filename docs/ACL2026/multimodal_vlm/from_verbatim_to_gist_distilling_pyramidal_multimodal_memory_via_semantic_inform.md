---
title: >-
  [Paper Note] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck
description: >-
  [ACL 2026][Multimodal VLM][Long video understanding] This paper proposes MM-Mem, a pyramidal multi-modal memory architecture inspired by Fuzzy Trace Theory (FTT). It organizes memory into three levels: a Sensory Buffer l…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Long video understanding"
  - "Multi-modal memory"
  - "Information Bottleneck"
  - "Fuzzy Trace Theory"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: f73ae86fce1c1fa0
---

# From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck

**Conference**: ACL 2026  
**arXiv**: [2603.01455](https://arxiv.org/abs/2603.01455)  
**Code**: [GitHub](https://github.com/EliSpectre/MM-Mem)  
**Area**: Multi-modal VLM  
**Keywords**: Long video understanding, Multi-modal memory, Information Bottleneck, Fuzzy Trace Theory, Reinforcement Learning

## TL;DR

This paper proposes MM-Mem, a pyramidal multi-modal memory architecture inspired by Fuzzy Trace Theory (FTT). It organizes memory into three levels: a Sensory Buffer layer (visual-dominant), an Episodic Stream layer (event-level summaries), and a Symbolic Schema layer (knowledge graph). By compressing redundancy via SIB-GRPO (Semantic Information Bottleneck + Reinforcement Learning) in a bottom-up distillation and employing entropy-driven top-down retrieval, the model achieves SOTA performance across four long-video benchmarks.

## Background & Motivation

**Background**: Multi-modal Large Language Models (MLLMs) excel in short-term perception but are constrained by context window limits and static memory mechanisms in long video understanding. Existing approaches fall into two extremes: vision-centric methods (e.g., LongVA, VideoRAG) rely on dense frame sampling, leading to high latency and redundancy, while text-centric methods (e.g., Vgent) convert videos into text summaries, causing loss of detail and hallucinations.

**Limitations of Prior Work**: (1) Vision-centric methods accumulate excessive visual tokens, resulting in computational redundancy and neglect of high-level semantics; (2) Text-centric methods perform lossy compression via captioning, losing critical visual cues; (3) Existing memory mechanisms are static, unlike the dynamic organization of human memory; (4) Dynamic memory management in multi-modal scenarios remains severely under-explored.

**Key Challenge**: Long video understanding necessitates the simultaneous preservation of fine-grained visual details (for precise verification) and high-level semantic abstractions (for cross-event reasoning). However, current methods face a fundamental trade-off between visual fidelity and semantic abstraction.

**Goal**: To design a hierarchical multi-modal memory architecture that achieves progressive distillation from fine-grained perception to high-level cognition while supporting dynamic memory compression and adaptive retrieval.

**Key Insight**: Fuzzy Trace Theory (FTT) in cognitive psychology posits that human memory contains two parallel traces: verbatim (precise perceptual details) and gist (abstract semantic meaning). Visual data naturally corresponds to verbatim, while text corresponds to gist—this cross-modal complementarity can be directly mapped to a hierarchical memory architecture.

**Core Idea**: Construct a three-layer pyramidal memory that transitions from vision-dominant to text-dominant representation bottom-up. Use Information Bottleneck theory to guide compression (SIB-GRPO) and entropy-driven top-down retrieval to dynamically switch between abstraction and detail.

## Method

### Overall Architecture

MM-Mem takes a long video stream as input and constructs a three-layer pyramidal memory offline: (1) **Sensory Buffer** preserves visual representations of keyframes and short text labels; (2) **Episodic Stream** generates event-level representations through clustering and summarization; (3) **Symbolic Schema** builds an entity knowledge graph. During querying, it performs top-down retrieval: first querying the knowledge graph (gist), descending to the episodic layer when uncertain, and accessing visual frames (verbatim) only if uncertainty persists.

### Key Designs

1.  **Three-layer Pyramidal Memory Structure**:
    - **Function**: Achieve progressive abstraction from perception to cognition.
    - **Mechanism**: The **Sensory Buffer** $\mathcal{M}_{sens} = \{(v_{t,i}, l_{t,i}, \tau_{t,i})\}$ captures segments via content-adaptive temporal segmentation (PySceneDetect) and selects keyframes based on inter-frame variance. The **Episodic Stream** organizes sensory items into compact event sequences using a decision operator $\psi(m_{t,i}, e^\star) \in \{ADD\_NEW, MERGE, DISCARD\}$, followed by K-means clustering to select representative prototypes. The **Symbolic Schema** constructs a knowledge graph $\mathcal{G} = (\mathcal{N}, \mathcal{E})$ where nodes include episodic units and entity prototypes, and edges represent semantic relations and grounding pointers.
    - **Design Motivation**: These layers correspond to different abstraction levels in FTT. The key innovation lies in "grounding edges," which ensure that high-level text abstractions remain anchored to specific visual evidence, mitigating the hallucination issues of pure-text methods.

2.  **SIB-GRPO: Information Bottleneck Driven Memory Compression**:
    - **Function**: Balance the trade-off between redundancy compression and task-relevant semantic preservation.
    - **Mechanism**: The perception-to-episodic transition is modeled as a stochastic compression problem, optimizing the Information Bottleneck objective: $\min_{p_\theta(m|x)} [I(X;M) - \beta I(M;Y)]$, where $X$ is sensory memory, $M$ is episodic representation, and $Y$ is the downstream VQA answer. A variational decoder and a quality-quantity prior $r(m) \propto p_{ref}(m) \cdot e^{-\lambda|m|}$ are introduced. Since episodic traces are discrete, a GRPO-style reinforcement learning approach is used to train the memory manager. It samples $G$ candidate traces and calculates a scalar reward $r(s,m) = R_{vqa} - \beta_1 \cdot Length(m) - \beta_2 \cdot \log\frac{\pi_{\theta_{old}}}{\pi_{ref}}$, optimizing a PPO-clipped surrogate objective.
    - **Design Motivation**: Traditional IB assumes continuous variables and cannot be applied directly to discrete LLM generations. GRPO converts IB principles into RL objectives trainable with sequence-level feedback.

3.  **Entropy-driven Top-down Retrieval**:
    - **Function**: Adaptively select retrieval depth based on query difficulty.
    - **Mechanism**: Inspired by Reverse Hierarchy Theory, retrieval starts at the Symbolic Schema (most abstract). The model maintains a posterior distribution of answer candidates $p_i^{(s)} = p(a_i | \mathcal{Q}, R_{\leq s})$ and calculates entropy $H_s(\mathcal{Q}) = -\sum_i p_i^{(s)} \log p_i^{(s)}$. Retrieval stops when $H_s \leq \gamma$ or the entropy reduction $\Delta H_s$ falls below a threshold $\epsilon$. 
    - **Design Motivation**: Not all questions require raw visual frames. High-level text retrieval quickly narrows the semantic scope, while low-level visual retrieval is triggered only during high uncertainty, achieving an adaptive computation-accuracy trade-off.

### Loss & Training

The SIB-GRPO objective function is:  
$$J_{SIB-GRPO}(\theta) = \mathbb{E}[\frac{1}{G}\sum_{i=1}^{G} \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)]$$  
The base model is Qwen3-VL-8B, fine-tuned using the SWIFT framework with hyperparameters $\beta_1=0.1$, $\beta_2=0.3$, and temperature=0.0.

## Key Experimental Results

### Main Results

**Video-MME Long Video Understanding (Overall Accuracy)**

| Method | Type | w/o Subtitles | w/ Subtitles |
| :--- | :--- | :--- | :--- |
| Gemini 1.5 Pro | Commercial | 75.0 | 81.3 |
| Qwen2-VL-72B | Open-source 72B | 71.2 | 77.8 |
| Vgent | Agent | 68.9 | 74.3 |
| **MM-Mem (Ours)** | **Agent 8B** | **72.4** | **78.1** |

**Streaming Video VStream-QA-Ego**

| Method | Accuracy | Score |
| :--- | :--- | :--- |
| Flash-VStream | 59.0 | 3.9 |
| **MM-Mem** | **62.5** | **4.1** |

### Ablation Study

**Video-MME w/o Subtitles Component Ablation**

| Configuration | Short | Medium | Long | Overall |
| :--- | :--- | :--- | :--- | :--- |
| Full (MM-Mem) | 81.5 | 69.6 | 66.1 | 72.4 |
| w/o SIB-GRPO | ~79 | ~68 | ~63 | ~70 |
| w/o Hierarchical Memory | ~77 | ~66 | ~61 | ~68 |

### Key Findings

- MM-Mem surpasses all open-source MLLMs (including 72B models) and most Agent systems using only an 8B base.
- Significant gains are observed in the "Long" category, proving SIB-GRPO's efficacy in compressing long-range temporal dependencies.
- On HD-EPIC++, MM-Mem (30.28%) outperforms Qwen3-VL-8B (25.88%) by 4.4 points, demonstrating fine-grained aggregation capabilities for egocentric videos.
- Efficiency: Inference latency is 5.35s per minute of video with VRAM usage of 17.8GB (lower than Qwen3-VL-8B's 22.8GB).

## Highlights & Insights

- The mapping from FTT to engineering is highly intuitive—the correspondence of Vision=verbatim and Text=gist is elegant, and the grounding edges ensure the two modalities remain coupled.
- Combining Information Bottleneck theory with GRPO is a generalizable framework for any scenario requiring a trade-off between information retention and compression (e.g., chunk selection in RAG).
- Entropy-driven adaptive retrieval depth provides a practical design that avoids manually designing retrieval strategies for different query types.

## Limitations & Future Work

- Computational overhead for memory construction exists, though it can be amortized; edge deployment requires further distillation.
- Performance relies on the quality of upstream visual encoders and captioners; noise in the sensory layer may propagate upward.
- SIB-GRPO currently uses task-driven VQA rewards; defining rewards for unsupervised scenarios remains an open problem.
- Lack of systematic evaluation on ultra-long videos (>2h).

## Related Work & Insights

- **vs Vgent**: Pure text memory; MM-Mem outperforms Vgent by 3.5 points on Video-MME (72.4 vs 68.9) because text-only compression loses fine-grained visual evidence.
- **vs VideoRAG**: Vision-centric; VideoRAG has higher VRAM (23.0 vs 17.8 GB) and lower performance (60.5 vs 72.4) due to redundancy in dense visual accumulation.
- **vs A-Mem**: Text-centric dynamic memory; lacks multi-modal grounding and cannot descend to detailed layers for visual verification.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (FTT mapping + IB-GRPO innovation + Entropy retrieval)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks + Ablations + Efficiency + t-SNE)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivation and bridge to cognitive science)
- Value: ⭐⭐⭐⭐⭐ (Provides a reusable cognitive architecture for long-video agent memory)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[ACL 2026\] Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines](reducing_peak_memory_usage_for_modern_multimodal_large_language_model_pipelines.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)

</div>

<!-- RELATED:END -->
