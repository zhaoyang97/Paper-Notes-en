---
title: >-
  [Paper Note] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking
description: >-
  [CVPR 2026][Multimodal VLM][Interpretability] This paper proposes the first circuit tracing framework for VLMs, training per-layer transcoders on Gemma-3-4B and constructing attribution graphs to reveal the hierarchical…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Interpretability"
  - "Circuit Tracing"
  - "Transcoder"
  - "Attribution Graph"
  - "Feature Steering"
date: 2026-05-08
content_hash: b0831fe60dc38a8b
---

# Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking

**Conference**: CVPR 2026
**arXiv**: [2602.20330](https://arxiv.org/abs/2602.20330)  
**Code**: [github.com/UIUC-MONET/vlm-circuit-tracing](https://github.com/UIUC-MONET/vlm-circuit-tracing)  
**Area**: Multimodal VLM
**Keywords**: Interpretability, Circuit Tracing, Transcoder, Attribution Graph, Feature Steering

## TL;DR

This paper proposes the first circuit tracing framework for VLMs, training per-layer transcoders on Gemma-3-4B and constructing attribution graphs to reveal the hierarchical integration mechanisms underlying multimodal reasoning, visual arithmetic circuits, and the internal causes of six-finger hallucinations. The causal controllability of the discovered circuits is validated through feature steering and circuit patching.

## Background & Motivation

**Background**: VLMs (e.g., CLIP, LLaVA, GPT-4o) have achieved remarkable success on visual question answering, image captioning, and complex visual reasoning tasks, yet their internal working mechanisms remain opaque black boxes — a particularly critical issue in high-stakes applications such as medical imaging, autonomous driving, and content moderation.

**Limitations of Prior Work**: Mechanistic interpretability research on LLMs (e.g., circuit discovery, induction head analysis, activation patching) has advanced considerably in recent years, but these methods have been almost exclusively confined to text-only models. VLMs present unique challenges: they must integrate two modalities with distinct statistical properties and semantics, and meaningful visual-language correspondences must be identified. Existing VLM interpretability work has largely remained at a high level — attention visualization and probing — which is correlational rather than causal in nature.

**Key Challenge**: Virtually nothing is known about how VLMs bind visual features to tokens, how cross-modal reasoning is implemented, or how visual and linguistic attention are coordinated. Sparse autoencoders and transcoders have successfully decomposed polysemantic representations in LLMs but have never been applied in multimodal settings.

**Goal**: Establish the first complete circuit tracing framework for VLMs and systematically analyze the internal computational mechanisms of multimodal reasoning.

**Key Insight**: Extend the transcoder + attribution graph paradigm validated in LLMs to the multimodal setting, developing new methods tailored to VLM-specific challenges such as image token processing, bidirectional attention, and cross-modal information flow.

**Core Idea**: By inserting transcoders into each MLP layer of a VLM, polysemantic representations are decomposed into interpretable monosemantic features. Combined with attribution graphs that trace causal relationships between features, sparse computational circuits driving multimodal reasoning are discovered and verified.

## Method

### Overall Architecture

The framework comprises three core components: (1) training per-layer transcoders for each MLP in the VLM to decompose multimodal polysemantic representations into sparse, monosemantic, interpretable features; (2) constructing attribution graphs to trace causal relationships between features; and (3) combining attention analysis and human expert annotation to identify the minimal computational circuits driving specific behaviors. The target model is Gemma-3-4B-it, which uses a SigLIP visual encoder (patch size 14, input resolution 896×896, producing 4096 patch tokens pooled into 256 soft image tokens) and a 34-layer transformer decoder ($d_{model}=2560$, $d_{ff}=10240$).

### Key Designs

1. **Per-Layer Transcoder**:

    - **Function**: Replaces each MLP layer in the VLM, decomposing polysemantic internal representations into sparse monosemantic features while preserving computational equivalence and exposing feature-level structure.
    - **Mechanism**: The encoder maps the MLP input $x \in \mathbb{R}^{d_{model}}$ to a much higher-dimensional feature space via $z(x) = \text{ReLU}(W_{enc}x + b_{enc})$, applying TopK sparsification (retaining only the $k=48$ largest activations). The decoder reconstructs the MLP output as $\text{TC}(x) = W_{dec}z(x) + b_{dec}$. Each transcoder feature is defined by a paired encoder column and decoder row, contributing additively to the output.
    - **Design Motivation**: Compared to the $\ell_1$ penalty used in the original transcoder, TopK sparsification requires no tuning of a sparsity coefficient, resulting in more stable training and greater feature consistency. Compared to SAEs that directly reconstruct activations, transcoders replicate the MLP's input-output behavior, clearly exposing causal relationships between features and facilitating circuit discovery.
    - **Reconstruction Residual**: The error $e(x) = \text{MLP}(x) - \text{TC}(x)$ is explicitly tracked and incorporated into the circuit graph as an independent error node, preventing approximation errors from contaminating the analysis.

2. **Attribution Graph**:

    - **Function**: Traces causal contribution relationships between internal features, constructing a complete computational graph from input token embeddings to output logits.
    - **Mechanism**: The local linearity of the model on a given input is exploited by freezing all nonlinearities — ReLU, attention softmax, and LayerNorm — at their current values. The attribution between each source-target feature pair is $A_{s \to t} = a_s \cdot w_{s \to t}$, where the virtual weight $w_{s \to t} = f_{dec}^{(s)\top} J^\blacktriangledown_{(s) \to (t)} f_{enc}^{(t)}$ encodes the source feature's decoder vector, the frozen residual stream Jacobian, and the target feature's encoder vector.
    - **Design Motivation**: Because each node's pre-activation equals the sum of all incoming attribution edges ($h_t = \sum_{s} A_{s \to t}$), the attribution graph provides a complete additive explanation. Pruning small attribution edges ($|A_{s \to t}| < \epsilon$) yields a sparse, interpretable graph. Cumulative influence thresholds are set at 0.80 and 0.98, with a maximum of $m=7500$ feature nodes, covering at least 0.95 of the logit probability mass.

3. **Multimodal Feature Interpretation and Circuit Discovery**:

    - **Function**: Assigns interpretable semantics to unnamed features in the attribution graph and extracts minimal circuits driving specific behaviors from the full computational graph.
    - **Mechanism**: For text token features, common patterns among top-k activating examples are analyzed. For image token features, attention rollout from the SigLIP encoder (selecting the $q$ fraction of lowest-entropy attention heads across the last $K$ layers and multiplying them layer by layer) is used to visualize attended image regions. Human experts aggregate functionally similar features into nodes, with inter-node attribution computed as the sum of feature attributions.
    - **Design Motivation**: An ad hoc feature analysis strategy is adopted — only the approximately 1,000 features present in the current attribution graph are analyzed (rather than precomputing all features), substantially reducing computational and storage costs. For specific tasks (e.g., sea otter recognition), activations are additionally computed on 30 task-relevant images, markedly improving feature interpretability.

### Intervention and Validation Strategies

**Feature Steering**: During the forward pass, the activation value $v_{\ell,t,i}$ of a specific feature is modified by computing the offset $\Delta z = v - z(x)$ and updating the residual stream as $h_{\ell,t} \leftarrow h_{\ell,t} + \Delta z \cdot d_{\ell,i}$, then observing changes in the output.

**Circuit Patching**: Feature patches (activation values at specific layers and positions) from circuit A are transplanted into structurally analogous circuit B. For example, in the "Mars" circuit, Mars visual features at intermediate layers are suppressed while Earth visual features discovered in the "Earth" circuit are activated, verifying whether all subsequent feature activations and the final output shift to Earth-related concepts.

## Key Experimental Results

### Transcoder Training Configuration

| Component | Configuration |
|-----------|---------------|
| Training Data | SmoLIM2 text 144K + ImageNet images 144K + Cauldron QA 72K |
| Optimizer | AdamW, lr = $2 \times 10^{-4} \times \sqrt{2^{14} / (N_{latents} \times d_{model})}$ |
| Training Scale | batch size 12, 30K steps, 8×H100, ~60 hours |
| Sparsification | TopK, $k=48$ |
| Feature Dimension | $d_{feat} = N_{latents} \times d_{model} \times 34$ |

### Expansion Factor Comparison

| Expansion Factor $N_{latents}$ | Dead Feature Ratio Trend | Layer-wise Variation |
|-------------------------------|--------------------------|----------------------|
| 32 | Highest; many features unused | Dead feature ratio especially high in early layers (Layer 3) |
| **64 (adopted)** | Moderate; optimal balance between utilization and quality | Intermediate layers (Layer 15) exhibit densest activation patterns |
| 128 | Lowest; FVU slightly increases | Redundancy among high-layer features increases |

### FVU Comparison: Multimodal vs. Text-Only Training

| Training Data | Mid-layer FVU (~Layer 15) | High-layer FVU (~Layer 30) | Analysis |
|---------------|---------------------------|---------------------------|----------|
| Text-only (SmoLIM2) | Higher | Comparable to multimodal | Lack of visual constraints leads to insufficient explanation at intermediate layers |
| **Text + Images (Ours)** | **Significantly lower** | **Slightly lower** | Visual features provide additional constraints, especially at intermediate layers where visual information is integrated |

### Computational Cost

| Operation | Compute Resources |
|-----------|------------------|
| Attribution graph for a single QA task | ~20 minutes on a single H100 |
| Feature activation analysis for a single attribution graph (~1,000 features) | ~20 H100 GPU-hours |
| Attention map precomputation for 28K images | ~2 hours on a single H100; ~2TB storage |
| Full transcoder training | 8×H100, ~60 hours |

### Key Findings

- **Hierarchical Integration**: Features jointly encoding visual and semantic concepts appear only above approximately Layer 20; earlier layers maintain modality independence. This supports the "progressive binding hypothesis" — cross-modal associations are established gradually along network depth.
- **Visual Arithmetic Circuits**: For image-rendered arithmetic (e.g., $1+2$), the model partially computes within visual space — intermediate layers exhibit visual features corresponding to the result digit "3", activated consistently across contexts. Visual representations of numerical ranges and modular arithmetic patterns are also discovered, echoing analogous findings in LLM text circuits.
- **Six-Finger Hallucination Mechanism**: Rather than a single failure mode, the hallucination arises from the interaction of perceptual bias and internal circuit dynamics: (1) the SigLIP encoder produces embeddings that over-emphasize generic "hand" semantics; (2) internal model circuits further amplify hand-related features; and (3) visual features for the digit "6" are suppressed to levels comparable to irrelevant digits, while hand-related features strongly activate the "five" circuit. The model does possess a visual counting circuit, but it is overwhelmed by stronger semantic and perceptual signals.
- **Parallel Visual-Semantic Pathways with Late Convergence**: Gemma-3 maintains separate visual and semantic representation streams deep into the network — for instance, "space shuttle" association features triggered by a Mars image reflect visual associations independent of semantic content; visually similar species (sea otters, seals, beavers) consistently co-activate in high layers even when belonging to different semantic categories. The two pathways converge into a unified multimodal representation only at the final layers.

### Ablation Study

| Intervention | Experimental Setup | Result |
|--------------|--------------------|--------|
| Circuit Patching (Mars → Earth) | Suppress intermediate-layer Mars visual features; activate Earth visual features | All subsequent features and output shift to Earth-related concepts |
| Feature Steering | Modify specific feature activation values | Output changes predictably, validating circuit causality |
| Feature Ablation (zeroing) | Set target feature to zero | Associated behavior is precisely suppressed |
| Feature Amplification | Set target feature to a positive constant | Associated behavior is enhanced |

## Highlights & Insights

- This is the first work to achieve complete circuit tracing in VLMs, successfully extending Anthropic's LLM methodology to the multimodal setting.
- The mechanistic analysis of six-finger hallucinations is particularly insightful: it is not simply a case of "encoder failure" but the interactive result of three factors — encoder bias, internal circuit competition, and the suppression of the counting circuit.
- The existence of an independent visual representation space within the VLM's language model component is demonstrated: feature clustering and co-activation driven by visual similarity operate independently of semantic organization.
- The ad hoc feature analysis strategy is both practical and efficient: analyzing only features present in the current attribution graph, combined with a small set of task-relevant images, substantially reduces cost while improving interpretability.

## Limitations & Future Work

- Only a single model, Gemma-3-4B, is analyzed; the use of SigLIP with bidirectional attention may introduce model-specific complexities, and the generalizability of the findings has not been verified.
- Per-layer transcoders cannot capture cross-layer superposition, and the high feature density of image embeddings in VLMs leads to frequently near-duplicate visual features in attribution graphs.
- Visual encoder attention maps sometimes fail to localize relevant regions, limiting the annotation quality of image features.
- Circuit discovery relies on manual human expert annotation, making quantitative evaluation difficult and direct application to model fine-tuning non-trivial.
- Computational costs are high (complete analysis of a single attribution graph requires ~20 GPU-hours), and automated feature interpretation methods remain computationally prohibitive.
- The optimal transcoder training strategies for VLMs under different configurations (e.g., JumpReLU, BatchTopK) have not been thoroughly investigated.

## Related Work & Insights

- **Direct Extension of LLM Circuit Tracing**: Building on Anthropic's circuit tracing framework (Lindsey et al., Ameisen et al.) and the per-layer transcoder adaptation by Hanna et al., this work is the first to handle image tokens and cross-modal information flow.
- **Fundamental Distinction from Attention Visualization / Probing**: Conventional VLM interpretability methods are correlational analyses, whereas the circuit tracing presented here is causal — circuits are verified to drive behavior through intervention experiments.
- **Sparse Autoencoders vs. Transcoders**: SAEs reconstruct activations directly; transcoders replicate the MLP's input-output behavior. The latter is better suited for circuit discovery as it preserves computational equivalence.
- **Implications**: The existence of an independent visual representation space within VLMs suggests that visual and linguistic information may only truly fuse at the "last moment." The multi-factor causal account of six-finger hallucinations offers several intervention points for hallucination mitigation (encoder debiasing, circuit competition modulation, and counting circuit reinforcement).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First complete circuit tracing framework for VLMs, filling a gap in multimodal mechanistic interpretability.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional analysis with causal intervention validation, but limited to a single model with no quantitative benchmark comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ — Case studies are insightful and engaging; methodology is clearly articulated with rich illustrations.
- Value: ⭐⭐⭐⭐⭐ — Establishes a standardized analytical framework for VLM interpretability; insights such as the six-finger hallucination mechanism have direct practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2026\] Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU](responses_fall_short_of_understanding_gap_between_internal_representations_and_responses_in_vdu.md)
- [\[CVPR 2026\] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)
- [\[ICLR 2026\] Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](../../ICLR2026/multimodal_vlm/visual_symbolic_mechanisms_vlm.md)
- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)

</div>

<!-- RELATED:END -->
