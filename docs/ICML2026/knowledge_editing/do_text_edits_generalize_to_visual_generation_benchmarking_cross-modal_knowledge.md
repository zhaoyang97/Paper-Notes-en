---
title: >-
  [Paper Note] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs
description: >-
  [ICML 2026][Knowledge Editing][Paper Note] This paper proposes UniKE—the first "cross-modal knowledge editing" benchmark for Unified Multimodal Models (UMMs) (2,971 editing subjects, 5,535 VQA-verifiable instances). It systematically reveals a modality gap where the "text-side editing success rate is ~92%, yet image generation VQA is only ~18.5%." By using a "r
tags:
  - ICML 2026
  - Knowledge Editing
date: 2026-05-08
content_hash: 39591be94d0ead6e
---
# Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs

**Conference**: ICML 2026  
**arXiv**: [2606.00477](https://arxiv.org/abs/2606.00477)  
**Code**: https://github.com/gxx27/UniKE (Available)  
**Area**: Knowledge Editing / Cross-Modal / Unified Multimodal Models (UMM)  
**Keywords**: Knowledge editing, cross-modal transfer, unified multimodal models, reasoning augmentation, conditioning path

## TL;DR
This paper proposes UniKE—the first "cross-modal knowledge editing" benchmark for Unified Multimodal Models (UMMs) (2,971 editing subjects, 5,535 VQA-verifiable instances). It systematically reveals a modality gap where the "text-side editing success rate is ~92%, yet image generation VQA is only ~18.5%." By using a "reasoning-augmented parameter editing" protocol, it increases VQA accuracy by up to 18.6 percentage points and identifies the root cause as the LLM-to-DiT projection bottleneck using cosine drift metrics on the conditioning path.

## Background & Motivation
**Background**: Unified Multimodal Models (UMMs) compress image understanding and generation into the same transformer backbone, relying on shared parameters to achieve end-to-end synergy between text and images. Representative works include Ovis-U1, BLIP3o-4B, and OmniGen2. Meanwhile, text-only Knowledge Editing (KE) methods—ROME, MEMIT, PMET, AlphaEdit—have matured, allowing precise rewriting of MLP layer weights without retraining to change facts (e.g., "The founder of Apple is Jobs" to "is Tim Cook").

**Limitations of Prior Work**: Since UMMs share a backbone, the question of whether "editing a fact on the text side via KE automatically updates image generation" has not been systematically studied. Existing multimodal KE benchmarks (e.g., TMKE) only measure image-conditioned text answering (I2T), missing the most critical text→image (T2I) propagation path.

**Key Challenge**: Text-side editing only requires "flipping the next-token distribution," which has a low threshold. However, to affect image generation, the perturbation must pass through the entire conditioning path (LLM → Projection → DiT) without being attenuated or filtered. The required signal strength and directionality for both are on completely different scales.

**Goal**: (1) Construct a visually verifiable cross-modal KE benchmark; (2) Quantify the loss between "text-side editing and image generation"; (3) Find a method to improve transfer without modifying weights; (4) Answer "why it drops" through mechanistic analysis.

**Key Insight**: The authors hypothesize that the edit is actually modified in the parameters but remains "latent" within the weights, only being transmitted to the visual generation path when activated by an explicit text context.

**Core Idea**: First, let the model "speak out" the edited fact in text to transform latent parameter changes into explicit text conditions. Then, superimpose this text condition onto the image prompt and feed it into the generator—this is Reasoning-augmented Parameter Editing.

## Method

### Overall Architecture
This work addresses whether editing a fact on the text side changes image generation. It does not train new models but breaks the problem into three reproducible tasks: first, using the UniKE benchmark to turn "whether the edited fact exists in the image" into a binary verifiable metric; next, comparing "direct generation" and "speak-then-generate" via Direct and Reasoning-Augmented protocols; and finally, using cosine drift analysis on the conditioning path to locate where the signal attenuates. Each evaluation instance is formalized as $\mathcal{I}=(q, y, y', p_{img}, t_{vis}, q_{vqa})$—comprising the editing prompt, original answer, target answer, image generation prompt, visual target description, and VQA verification question. The generated images are judged by Qwen3-VL-235B (LLM-as-judge) with a 0/1 determination, forming a 9×2 evaluation matrix of 3 UMMs (Ovis-U1 / BLIP3o-4B / OmniGen2) × 3 Editors (MEMIT / PMET / AlphaEdit) × 2 protocols.

### Key Designs

**1. UniKE Benchmark: Making "Text Edit to Image Transfer" Quantifiable for the First Time**

The pain point is that previous benchmarks could not measure this path—text-only benchmarks (ZsRE / CounterFact / MQuAKE) do not touch images, and the existing multimodal benchmark TMKE only tests image-conditioned text answering (I2T). UniKE fills this gap using an answer-neutral image prompt paired with VQA-as-judge. Attribute edits are generated by a Gemini-3.0-Flash self-instruction pipeline into candidate $(q, y, y')$ triplets, divided into four progressive difficulty stages (Stage 1: Atomic objects; Stage 2: Real-world scene embedding; Stage 3: Multi-entity complex composition; Stage 4: Derived product/utility transfer). Relation edits extract triplets from CounterFact/MQuAKE and use LLM-as-judge to filter non-visualizable categories. The key constraint is the "answer-neutral" principle—prompts cannot leak original or target values, so any correct expression in the image must come from the model's internal edited knowledge. It covers 2,971 subjects and 5,535 instances, encompassing both attributes (color, material, shape, size, pattern) and relations (membership, creator, location, profession).

**2. Reasoning-augmented Parameter Editing: Activating Latent Edits via Text Inference**

The authors found that all editors have high text-side editing success rates (Eff. 55%–90%), but extremely low image VQA scores. This suggests the fact is updated within the LLM but fails to reach the generation path; the change is "latent" in the weights and requires explicit context for activation. The Reasoning-Augmented protocol addresses this: instead of feeding $p_{img}$ directly to the generator (Direct protocol), it uses a category-conditioned template $p_{rea}$ to trigger the edited model to generate a text rationale $r$ (model-produced, not oracle). This $r$ is prepended to $p_{img}$ as an additional condition. This rationale "explicitly manifests" the edited fact latent in the MLP weights as a token-level text constraint, essentially using a longer, more aligned conditioning vector to compensate for signal attenuation. Its main advantage is that it does not modify weights, making it orthogonal to any editor, and it raised VQA across all 9 model-editor pairs with a maximum gain of +18.6 pp.

**3. Conditioning Path Drift Analysis: Locating Bottlenecks at the LLM-to-DiT Projection**

Using only the Direct protocol cannot distinguish between "the edit failed to update the LLM" and "it updated but failed to propagate." Thus, the authors sampled 100 cases using PMET to quantify signals along the path. They defined a cosine drift operator $\Delta_{cos}(a,b)=1-a^\top b/(\|a\|\|b\|)$. At the LLM output, they used per-token average $d_{cos}^{tok}$ and relative Frobenius drift $r_F=\|\delta\|_F/\|C_{fresh}^{LLM}\|_F$ to measure the perturbation caused by parameter editing. At the DiT input, they used $d_{cos}^{dir}$ and $d_{cos}^{rea}$ on mean-pooled conditioning vectors to measure the actual drift received by the DiT. Results showed that Ovis-U1, which uses a frozen dimensionality-reduction projection, has an $r_F$ of only 0.078, while BLIP3o-4B is as high as 0.527—the former's projection acts as an "architectural filter." However, Ovis-U1 benefits most from reasoning augmentation ($d_{cos}^{rea}=0.154$ vs $d_{cos}^{dir}=0.018$, an 8x amplification), because the rationale-injected perturbation falls into directions preserved by the projection.

### Loss & Training
This paper does not train new models; all editors use their original objective functions (closed-form weight updates for MEMIT/PMET, null-space projection for AlphaEdit). Only middle MLP layers are edited in the three UMMs: layers 4–8 for Ovis-U1, and layers 6–10 for BLIP3o-4B and OmniGen2. For AlphaEdit on BLIP3o-4B / OmniGen2, the authors used $\alpha=0.7/0.6$ to interpolate the null-space projector with an identity matrix for a "softened" version (marked with an asterisk) to avoid damaging generation quality in the shared Qwen2.5-VL backbone. All edits were performed in a sequential editing setting.

## Key Experimental Results

### Main Results
Summary of overall metrics for 3 UMMs × 3 editors × 2 protocols (Eff. = text-side edit accuracy, VQA = image VQA accuracy, unit %):

| Model | Editor | Eff. (Direct) | VQA (Direct) | VQA (+Reasoning) | Gain (pp) |
|-------|--------|---------------|--------------|------------------|-----------|
| Ovis-U1 | PMET | 72.18 | 9.71 | 28.32 | +18.6 |
| Ovis-U1 | MEMIT | 59.84 | 8.70 | 24.41 | +15.7 |
| BLIP3o-4B | PMET | 76.30 | 18.51 | 19.29 | +0.8 |
| BLIP3o-4B | AlphaEdit* | 77.88 | 16.12 | 17.33 | +1.2 |
| OmniGen2 | PMET | 76.20 | 11.43 | 16.01 | +4.6 |
| OmniGen2 | AlphaEdit* | 76.37 | 11.50 | 17.90 | +6.4 |

The most striking finding is the modality gap: under the Direct protocol, VQA is only 1/8 to 1/4 of Eff. Reasoning-Augmented improved VQA across all 9 pairs, but the gain is heavily architecture-dependent.

### Ablation Study
Conditioning path drift analysis for PMET on 100 sampled cases (Source: Table 4):

| Model | LLM Output $d_{cos}^{tok}$ | LLM Output $r_F$ | DiT Input $d_{cos}^{dir}$ | DiT Input $d_{cos}^{rea}$ |
|-------|--------------------------|----------------|--------------------------|--------------------------|
| Ovis-U1 | 0.003 | 0.078 | 0.018 | 0.154 |
| BLIP3o-4B | 0.139 | 0.527 | 0.031 | 0.064 |
| OmniGen2 | 0.038 | 0.262 | 0.018 | 0.092 |

Ovis-U1 has the weakest implicit drift at the LLM output (filtered by projection), but reasoning augmentation amplifies the DiT-side drift by 8x. BLIP3o-4B has the largest implicit drift but fails to propagate it, reflecting that "large drift $\neq$ good alignment."

### Key Findings
- Text-side Eff. and image VQA accuracy are almost uncorrelated: high Eff. does not guarantee the edited fact appears in images, debunking the intuition that "unified backbone $\implies$ automatic cross-modal knowledge propagation."
- Category difficulty varies significantly: in attributes, "size" is easiest (VQA handles relative comparison well), while "shape" is hardest (precise geometric control is difficult); in relations, "occupation" is easiest (local visual proxies like uniforms/tools), while "creator" is hardest (authorship is inherently non-visual).
- Moving from Stage 1 to Stage 2, text Eff. drops by 70% on average, but reasoning accuracy only drops by ~10%, suggesting the edited fact is "in the weights" but sensitive to the raw edit template; rationales serve as a more robust retrieval interface.
- Conditioning path attenuation occurs mainly before the DiT, not within it—implying that future directions should focus on the joint design of editors and projection layers.

## Highlights & Insights
- **Quantifiable Benchmark for Cross-Modal KE**: The combination of answer-neutral image prompts and VQA-as-judge transforms "whether the edit appears in the image" from a subjective question into a reproducible binary metric.
- **Training-free Reasoning-Augmented Protocol**: A strong plug-in baseline that does not modify weights or favor specific editors. It essentially converts "latent parameter changes" into "explicit text constraints," providing insights for future multimodal CoT editing.
- **System Diagnosis via Cosine Drift**: Quantifying the "editing signal" at two stages along the LLM-DiT path is an elegant way to diagnose UMMs as "signal attenuation systems," a method applicable to any "backbone + projection + head" architecture.

## Limitations & Future Work
- Limitations acknowledged by authors: Only 3 UMMs and 3 editors were tested; the reasoning protocol provides limited gains for BLIP3o / OmniGen2, showing text activation is not a panacea; rationales themselves may introduce new errors.
- Additional observations: All edits were single-step in a sequential setting, without exploring lifelong/batch editing; using Qwen3-VL as the judge might introduce self-preference bias; it is unclear how much failure in Stage 4 (derived products) is due to editing vs. the UMM's inherent reasoning limits.
- Future directions: (1) Design modality-aware editors that constrain weight updates to subspaces preserved by projections; (2) Jointly optimize rationale generation and editing; (3) Explore editing cross-attention layers to directly influence visual conditioning.

## Related Work & Insights
- **vs. MEMIT / PMET / AlphaEdit (Text-only KE)**: These methods show high Eff. but low VQA on UniKE, proving their "success" is modality-limited and highlighting the need for cross-modal evaluation.
- **vs. TMKE (Multimodal KE Benchmark)**: TMKE focuses on I2T (image-to-text), whereas UniKE measures the T2I direction; they are complementary as one verifies the "understanding" side and the other verifies the "generation" side.
- **vs. T2I Editing (TIME / ReFACT / DiffQuickFix)**: These target modular diffusion components (text encoders/cross-attention) and are not directly applicable to monolithic UMMs. This paper suggests UMMs require a new editing paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of cross-modal KE in UMMs; first to measure signal attenuation along the conditioning path.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis across stages, categories, and mechanisms, though UMM count is small and lacks lifelong editing.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, and the flow from Table 1 to Table 4 is logical; the mechanism section is slightly dense with formulas.
- Value: ⭐⭐⭐⭐⭐ Establishes a benchmark for the new UMM editing track and provides a strong training-free baseline and clear research directions.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2025/knowledge_editing/mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)
- [\[ICML 2026\] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)
- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](../../ACL2025/knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)

</div>

<!-- RELATED:END -->
