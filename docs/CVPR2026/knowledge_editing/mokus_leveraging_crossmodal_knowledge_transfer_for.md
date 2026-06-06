---
title: >-
  [Paper Note] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
description: >-
  [CVPR 2026][Knowledge Editing][Concept Customization] This paper introduces a new task termed "knowledge-aware concept customization," and discovers that knowledge editing applied to LLM text encoders naturally transfers…
tags:
  - "CVPR 2026"
  - "Knowledge Editing"
  - "Concept Customization"
  - "Cross-Modal Knowledge Transfer"
  - "DiT"
  - "LLM Text Encoder"
date: 2026-05-08
content_hash: 7f47ce15fde46029
---

# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**Conference**: CVPR 2026
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)  
**Code**: None  
**Area**: Image Generation / Concept Customization / Knowledge Editing
**Keywords**: Concept Customization, Cross-Modal Knowledge Transfer, Knowledge Editing, DiT, LLM Text Encoder

## TL;DR
This paper introduces a new task termed "knowledge-aware concept customization," and discovers that knowledge editing applied to LLM text encoders naturally transfers to the visual generation modality (cross-modal knowledge transfer). Building on this finding, the paper proposes MoKus: a two-stage framework that first binds a rare token to a visual concept as an anchor representation via LoRA fine-tuning, then efficiently maps multiple natural-language knowledge statements onto the anchor representation via knowledge editing—requiring only ~7 seconds per knowledge update.

## Background & Motivation
Existing concept customization methods (e.g., DreamBooth, Textual Inversion) represent target concepts with rare tokens (e.g., `<sks>`) and suffer from two fundamental limitations: (1) **Unstable performance**—rare tokens appear infrequently in pretraining data, lack semantic grounding, and produce highly variable generation quality when combined with normal text prompts; (2) **Knowledge unawareness**—rare tokens encode only visual appearance and cannot carry intrinsic knowledge about a concept (e.g., "a bronze statue in the harbor of Copenhagen, Denmark" → the Little Mermaid statue), causing knowledge-rich prompts (e.g., "Little Mermaid Statue Denmark") to fail. Encoder-based methods (e.g., IP-Adapter, BLIP-Diffusion) require large-scale retraining to accommodate new knowledge.

## Core Problem
How can a generative model simultaneously understand *what a concept looks like* (visual appearance) and *what a concept represents* (associated knowledge), enabling high-fidelity customized image generation from knowledge-bearing text prompts? Furthermore, a single concept may be associated with multiple knowledge statements (objective descriptions, subjective impressions, etc.); how can all such knowledge be efficiently bound to the same concept?

## Method

### Overall Architecture
MoKus is built upon an LLM text encoder combined with a DiT generation backbone (Qwen-Image) and operates in two stages: (1) **Visual Concept Learning**, which learns an anchor representation of the concept's visual appearance by fine-tuning LoRA weights in the self-attention layers of MMDIT; and (2) **Textual Knowledge Updating**, which maps each knowledge statement into the text space occupied by the anchor representation via knowledge editing, thereby binding knowledge to the concept.

### Key Designs
1. **Cross-Modal Knowledge Transfer**: This is the core insight of MoKus. In text-to-image models that use an LLM as the text encoder, modifying knowledge within the LLM via knowledge editing techniques (e.g., changing the answer to "Beethoven's favorite instrument" from "piano" to "guitar") naturally causes the generated images to reflect the updated answer (generating a guitar instead of a piano). That is, knowledge modifications in the text modality **transfer naturally** to the visual generation modality. This phenomenon distinguishes MoKus from works such as GapEval and UniSandbox—those approaches directly fine-tune the LLM encoder and fail to observe significant transfer, whereas MoKus employs more precise knowledge editing methods (UltraEdit/AlphaEdit).
2. **Visual Concept Learning (Anchor Representation Learning)**: A rare token (e.g., `<sks> dog`) is used as text input, and LoRA weights in the self-attention layers of DiT are fine-tuned to learn the visual appearance of the target concept. The training objective is the standard Rectified Flow velocity matching loss: $\mathcal{L}(\theta_v) = \mathbb{E}[\|v_\theta(z_t, t, h) - (z_0 - z_1)\|^2]$. After fine-tuning, the rare token serves as an "anchor representation"—not used directly for generation, but as an intermediary linking the concept to its knowledge.
3. **Textual Knowledge Updating**: Each knowledge statement $k_i$ is reformulated as a question $q_i$, with the anchor representation $y$ as the expected answer. $q_i$ is fed into the LLM encoder to obtain the hidden state $h_i$ and gradient $\nabla_{\theta_t} y_i$, from which the update direction is computed as $v_i = -\eta \cdot \|h_i\|^2 \cdot \nabla y_i$. A closed-form solution is then obtained via regularized least squares: $\Delta\theta_t^* = (H^\top H + I)^{-1} H^\top V$, and the resulting parameter shift is applied to the MLP layers of the LLM encoder (specifically, the Gate Projection and Up Projection of layers 18–26, totaling 16 parameter matrices). Each knowledge update takes ~7 seconds; five updates take ~360 seconds in total.
4. **KnowCusBench**: The first benchmark for knowledge-aware concept customization, comprising 35 concepts, 5 knowledge statements per concept (from 6 perspectives: personal relationships / physical attributes / function / value / origin / emotion), and 199 generation prompts (from 4 perspectives: background variation / object insertion / style transfer / attribute modification), yielding 5,975 evaluation images in total.

### Loss & Training
- **Visual Concept Learning**: Standard Rectified Flow loss; lr = 2e-4; AdamW optimizer; only LoRA parameters are trained.
- **Textual Knowledge Updating**: No backpropagation training; parameter shifts are computed directly via a closed-form solution; scaling factor $\eta = 10^{-6}$; batch size = 1.
- Only the MLP layers (Gate Proj + Up Proj) of LLM encoder layers 18–26 are modified, totaling 16 parameter matrices.

## Key Experimental Results

| Task | Metric | MoKus | Naive-DB | Enc-FT | Note |
|------|--------|-------|----------|--------|------|
| Reconstruction | CLIP-I | 0.867 | 0.874 | 0.582 | Comparable to DB, far surpassing Enc-FT |
| Reconstruction | CLIP-I-Seg | 0.764 | 0.758 | 0.553 | Best (more accurate with segmentation) |
| Generation | CLIP-I-Seg | 0.718 | 0.717 | 0.562 | Best |
| Generation | CLIP-T | 0.305 | 0.291 | 0.197 | Best (prompt alignment) |
| Generation | Pick Score | 21.30 | 20.80 | 18.34 | Best (human preference) |
| Efficiency | Training Time | 6 min | 27 min | 10 min | Most efficient |
| WISE Subset | WiScore | 1.33 | — | 0.81 (baseline) | Significant gain in world knowledge |

### Ablation Study
- **Effect of knowledge quantity**: Increasing from 1 to 5 knowledge statements, CLIP-I-Seg fluctuates only between 0.761 and 0.764—extremely stable; each additional knowledge statement adds only ~7 seconds of training time (331s → 360s), indicating high efficiency.
- **Scaling factor $\eta$**: $\eta = 10^{-6}$ is optimal; a larger value ($10^{-5}$) severely distorts the encoder output distribution and causes generation collapse (similar to the failure mode of Enc-FT); a smaller value ($10^{-7}$) leads to insufficient updates.
- **Layer selection**: Modifying only the MLP of layers 18–26 is optimal; too few layers limit update capacity, while too many layers impair pretrained knowledge.

## Highlights & Insights
- Cross-modal knowledge transfer is a highly insightful discovery—knowledge editing, originally a technique from NLP, is shown to work naturally in multimodal generation, and more effectively than directly fine-tuning the LLM encoder.
- The two-stage decoupled design is extremely efficient: Visual Concept Learning is performed only once (~6 min), after which each new knowledge statement can be bound in ~7 seconds without retraining.
- KnowCusBench standardizes evaluation through an orthogonal design of 6 knowledge perspectives × 4 prompt perspectives, ensuring broad coverage.
- The method naturally extends to virtual concept creation and concept erasure—generation behavior can be controlled simply by modifying knowledge answers.
- Performance gains on the WISE world knowledge benchmark confirm that knowledge updates are genuinely "written into" the model.

## Limitations & Future Work
- Requires an LLM as the text encoder (e.g., Qwen-Image); models with traditional CLIP text encoders (e.g., SD1.5/2.1) cannot directly adopt this approach.
- Knowledge must be expressible in a question-answer format, which may limit applicability to knowledge that is difficult to formulate as questions (e.g., abstract stylistic preferences).
- Evaluation still relies on CLIP-based metrics, which may lack sensitivity to fine-grained visual differences.
- The method currently supports only the image domain; the authors propose extending it to video concept customization in future work.
- The identity-matrix regularization term in the closed-form solution may be insufficiently flexible; more sophisticated regularization strategies could potentially yield further improvements.

## Related Work & Insights
- **vs. DreamBooth (Naive-DB)**: DreamBooth requires a full retraining cycle for each knowledge statement (27 min) and conditions generation directly on rare tokens—resulting in unstable performance when combined with novel prompts. MoKus trains the anchor representation only once, with subsequent per-knowledge updates taking 7 seconds each, and conditions generation on natural-language knowledge rather than rare tokens, yielding better generalization.
- **vs. Enc-FT (direct fine-tuning of the LLM encoder)**: This is the strategy employed by GapEval and UniSandbox. Direct fine-tuning severely disrupts the encoder's output distribution, causing generation quality to collapse (CLIP-I: 0.582 vs. MoKus: 0.867). MoKus avoids this issue through precise knowledge editing that modifies only specific directions in specific layers.
- **vs. encoder-based methods (e.g., IP-Adapter)**: These methods require large-scale retraining of the encoder to accommodate new knowledge or concepts, offering limited flexibility. MoKus's knowledge updating is entirely parameter-efficient (closed-form solution, completed in seconds).

### Broader Implications
- The cross-modal knowledge transfer phenomenon suggests that the LLM text encoder plays a far richer role in multimodal models than merely "extracting text features"—it stores world knowledge that can directly influence visual generation.
- This paradigm can address the "knowledge blind spots" of diffusion models (e.g., correctly generating specific landmarks or celebrity features).
- The combination of knowledge editing and generative models opens a new direction for controllable generation—control is achieved not by modifying prompts but by modifying the model's "beliefs."
- Related idea: `20260316_concept_bottleneck_world_model.md` (concept-level knowledge representation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ [Introduces a novel task, discovers the cross-modal knowledge transfer phenomenon, and designs an efficient two-stage framework—highly innovative]
- Experimental Thoroughness: ⭐⭐⭐⭐ [Constructs a dedicated benchmark with thorough ablations, but validation is limited to a single generation backbone (Qwen-Image)]
- Writing Quality: ⭐⭐⭐⭐⭐ [Motivation is clear; the derivation from observation to method is natural and well-structured; figures and text are well integrated]
- Value: ⭐⭐⭐⭐ [Knowledge-aware customization is a practically valuable new direction; the cross-modal knowledge transfer finding carries far-reaching implications for understanding multimodal models]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](../../ICML2026/knowledge_editing/do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](../../ICLR2026/knowledge_editing/got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](../../ICML2026/knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[AAAI 2026\] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](../../AAAI2026/knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)
- [\[ACL 2026\] Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse](../../ACL2026/knowledge_editing/spectral_characterization_and_mitigation_of_sequential_knowledge_editing_collaps.md)

</div>

<!-- RELATED:END -->
