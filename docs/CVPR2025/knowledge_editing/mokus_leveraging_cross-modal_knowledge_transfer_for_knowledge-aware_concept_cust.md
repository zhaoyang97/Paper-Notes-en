---
title: >-
  [Paper Note] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
description: >-
  [CVPR2025][Knowledge Editing][Concept Customization] Proposes the MoKus framework, discovering and utilizing the "cross-modal knowledge transfer" phenomenon—where updating knowledge in an LLM text encoder propagates automatically to the visual generation end—to achieve knowledge-aware concept customization. It features a two-stage design: first learning a visual anchor representation, and then binding textual knowledge in seconds.
tags:
  - "CVPR2025"
  - "Knowledge Editing"
  - "Concept Customization"
  - "Cross-Modal Knowledge Transfer"
  - "LLM Text Encoder"
  - "DiT"
date: 2026-05-08
content_hash: 1bda35844f587934
---

# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**Conference**: CVPR2025  
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)  
**Code**: [GitHub](https://chenyangzhu1.github.io/MoKus)  
**Area**: Knowledge Editing  
**Keywords**: Concept Customization, Knowledge Editing, Cross-Modal Knowledge Transfer, LLM Text Encoder, DiT

## TL;DR
Proposes the MoKus framework, discovering and utilizing the "cross-modal knowledge transfer" phenomenon—where updating knowledge in an LLM text encoder propagates automatically to the visual generation end—to achieve knowledge-aware concept customization. It features a two-stage design: first learning a visual anchor representation, and then binding textual knowledge in seconds.

## Background & Motivation
**Background**: Concept Customization aims to generate new customized images based on concept images provided by the user. Existing methods (DreamBooth, Textual Inversion, etc.) represent the target concept using a rare token (such as `<sks>`).

**Limitations of Prior Work**:
   - **Instability**: Rare tokens rarely appear in pre-training data, which leads to unstable generation quality when combined with other texts.
   - **Knowledge-Unaware**: Rare tokens only bind visual appearances and cannot carry the inherent knowledge of the concept (e.g., "The Little Mermaid statue is in Denmark").

**Key Challenge**: How to efficiently bind multiple pieces of natural language knowledge to a target visual concept to achieve knowledge-aware customized generation?

**Key Insight**: **Cross-Modal Knowledge Transfer**: After updating the answer to a certain question using knowledge editing techniques in the LLM text encoder, the generated images will correspond to the updated answer. For example, if the answer to "Beethoven's favorite instrument" is updated to "guitar," the generated image will feature a guitar.

## Method

### Overall Architecture (two-stage)

**Stage 1: Visual Concept Learning**
- Associate the target concept with a rare token, serving as the "anchor representation".
- Train the DiT + LoRA with Rectified Flow to minimize the MSE between the predicted and ground-truth velocity fields.
- LoRA parameters are only added to the MMDiT self-attention layers for efficient fine-tuning.
- Output: Anchor representation $h = \phi(P)$, storing the visual information of the target concept.

**Stage 2: Textual Knowledge Updating**
- Convert each knowledge item $k_i$ into a question format $q_i$, where the expected output is the anchor representation $y$.
- Input the question into the LLM encoder to extract the hidden state $h_i$ and gradient.
- Compute the update direction:
  $$v_i = -\eta \cdot \|h_i\|^2 \cdot \nabla y_i$$
- Solve the regularized least squares problem to obtain the closed-form solution of the parameter shift:
  $$\Delta\theta_t^* = (H^\top H + I)^{-1} H^\top V$$
- Directly add the shift to the parameters of the updatable layer in the LLM encoder: $\hat{\theta}_t = \theta_t + \Delta\theta_t^*$.
- Only modify the MLP layers (Gate/Up Projection, layers 18-26); updating a single piece of knowledge takes only a few seconds.

### KnowCusBench Benchmark
- 35 concepts (from DreamBench, CustomConcept101, Unsplash).
- 5 pieces of knowledge per concept (spanning 6 aspects: personal relationship, physical attribute, functionality, value, origin, and emotion).
- 199 generation prompts (spanning 4 aspects).
- A total of 5,975 evaluation images.

## Key Experimental Results

### Main Results (Table 1)

| Method | Reconstruction CLIP-I↑ | Reconstruction CLIP-I-Seg↑ | Generation CLIP-I↑ | Generation CLIP-I-Seg↑ | Generation CLIP-T↑ | Pick Score↑ | Training Time↓ |
|------|-----------|----------------|-----------|----------------|-----------|-----------|----------|
| Naive-DB | **0.874** | 0.758 | 0.789 | 0.717 | 0.291 | 20.80 | ~27min |
| Enc-FT | 0.582 | 0.553 | 0.591 | 0.562 | 0.197 | 18.34 | ~10min |
| **Ours** | 0.867 | **0.764** | 0.761 | **0.718** | **0.305** | **21.30** | **~6min** |

- MoKus performs best on the more critical CLIP-I-Seg metric (assessing concept fidelity more accurately after filtering the background).
- Enc-FT (directly fine-tuning the LLM encoder) severely disrupts the output distribution, leading to a complete collapse across all indicators.

### Ablation Study on Number of Knowledge Items (Table 2)

| Number of Knowledge Items | Reconstruction CLIP-I-Seg | Generation CLIP-T | Training Time(s) |
|---------|----------------|-----------|------------|
| 1 | 0.761 | 0.304 | 331.3 |
| 3 | 0.761 | 0.305 | 345.1 |
| 5 | 0.764 | 0.305 | 360.0 |

- Increasing the number of knowledge items from 1 to 5 keeps performance stable or even slightly improved, adding only about 7 seconds per additional item.

### Ablation on Scaling Factor
- Performance severely collapses when $\eta=1e-4$ (CLIP-I drops to 0.557), while $\eta=1e-6$ is the optimal point.
- Performance is stable within the range of $\eta$ from $1e-5$ to $1e-8$, indicating low sensitivity to hyperparameters.

### Key Findings
- Extensible to virtual concept creation (creating new concepts with pure text descriptions) and concept erasure (deleting concepts by modifying appearance descriptions).
- Enhances model performance on the world knowledge benchmark WISE.

### Implementation Details
- Based on Qwen-Image model, using 8 H800 GPUs.
- Visual Concept Learning: $lr=2e-4$, AdamW optimizer, Diffusers default LoRA configuration.
- Knowledge Updating: Uses the UltraEdit method, modifying the Gate/Up Projection matrices (16 parameter matrices in total) of layers 18-26 in the LLM encoder, with a scaling factor of $\eta=1e-6$.
- Evaluation: Divided into reconstruction and generation parts, with 5 different random seeds for each part, totaling 5,975 images.

### Related Work & Insights
- GapEval and UniSandbox explored cross-modal knowledge transfer by directly fine-tuning the LLM text encoder but found no significant evidence.
- MoKus uses knowledge editing techniques (rather than direct fine-tuning) to achieve precise updates, successfully observing cross-modal transfer.

## Highlights & Insights
- **Discovery and Utilization of Cross-Modal Knowledge Transfer**: Proves systematically for the first time that knowledge updates in the LLM text encoder can transfer to visual generation, contrasting with the failed attempts of parallel works (GapEval, UniSandbox).
- **Second-Level Knowledge Binding**: The closed-form solution allows each knowledge update to be completed in seconds, significantly outperforming methods that require retraining.
- **New Task + New Benchmark**: Defines the Knowledge-Aware Concept Customization task and builds KnowCusBench.
- **Bridge Role of Anchor Representation**: Downgrades rare tokens from the "final representation" to an "intermediate bridge," making natural language knowledge the true carrier of concepts.

## Limitations & Future Work
- Relies on the knowledge editing capability of the LLM text encoder; editing interference (locality) may affect other knowledge.
- Knowledge is expressed in a textual Q&A format, which may not be suitable for formalizing all concept knowledge.
- Tested only on Qwen-Image; generalization to other T2I architectures remains to be verified.
- The number of concepts in KnowCusBench is limited (35), requiring larger-scale evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The observation of cross-modal knowledge transfer is novel; knowledge-aware customization is a brand-new task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Self-built benchmark, comparisons with multiple baselines, and expansion to multiple applications.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with a smooth narrative of observation-methodology-verification.
- Value: ⭐⭐⭐⭐ Opens up a new knowledge-aware direction for concept customization technology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](../../ICML2026/knowledge_editing/do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](../../ACL2025/knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](../../ACL2025/knowledge_editing/structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](../../ACL2025/knowledge_editing/adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[ICLR 2026\] SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings](../../ICLR2026/knowledge_editing/suit_knowledge_editing_with_subspace-aware_key-value_mappings.md)

</div>

<!-- RELATED:END -->
