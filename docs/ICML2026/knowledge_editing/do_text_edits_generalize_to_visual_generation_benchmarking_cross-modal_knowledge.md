---
title: >-
  [Paper Note] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs
description: >-
  [ICML 2026][Knowledge Editing][Paper Note] This paper proposes UniKE—the first "cross-modal knowledge editing" benchmark for Unified Multimodal Models (UMMs), comprising 2,971 edited subjects and 5,535 VQA-verifiable instances. It systematically reveals a "modality gap" where the text-side editing success rate is $\sim 92\%$ while image generation VQA is only $
tags:
  - ICML 2026
  - Knowledge Editing
date: 2026-05-08
content_hash: 3bb350415e3c32ca
---
# Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs

**Conference**: ICML 2026  
**arXiv**: [2606.00477](https://arxiv.org/abs/2606.00477)  
**Code**: https://github.com/gxx27/UniKE (Available)  
**Area**: Knowledge Editing / Cross-Modal / Unified Multimodal Models (UMM)  
**Keywords**: Knowledge editing, cross-modal transfer, unified multimodal models, reasoning augmentation, conditioning pathway

## TL;DR
This paper proposes UniKE—the first "cross-modal knowledge editing" benchmark for Unified Multimodal Models (UMMs), comprising 2,971 edited subjects and 5,535 VQA-verifiable instances. It systematically reveals a "modality gap" where the text-side editing success rate is $\sim 92\%$ while image generation VQA is only $\sim 18.5\%$. Through a "reasoning-augmented parameter editing" protocol, VQA accuracy is increased by up to 18.6 percentage points. Furthermore, the root cause is localized to the LLM-to-DiT projection bottleneck using a cosine drift metric along the conditioning pathway.

## Background & Motivation
**Background**: Unified Multimodal Models (UMM) integrate image understanding and generation into a single transformer backbone, relying on shared parameters for end-to-end synergy between text and images. Representative works include Ovis-U1, BLIP3o-4B, and OmniGen2. Meanwhile, pure-text Knowledge Editing (KE) methods—such as ROME, MEMIT, PMET, and AlphaEdit—have matured, allowing precise rewriting of specific MLP layer weights (e.g., changing "Apple's founder is Jobs" to "Tim Cook") without retraining.

**Limitations of Prior Work**: Since UMMs share a backbone, the question of whether "editing a fact on the text side via KE automatically translates to image generation" has not been systematically studied. Existing multimodal KE benchmarks (e.g., TMKE) only evaluate image-conditioned text answering (I2T), lacking the most critical text $\rightarrow$ image (T2I) propagation path.

**Key Challenge**: Text-side editing only requires flipping the "next token distribution," which has a low threshold. However, to influence image generation, the perturbation must traverse the entire conditioning pathway (LLM $\rightarrow$ projection layer $\rightarrow$ DiT) without being attenuated or filtered. The required signal strength and directionality for the latter are on a completely different scale.

**Goal**: (1) Construct a cross-modal KE benchmark that can be visually verified; (2) Quantify the performance drop between text-side editing and image generation; (3) Identify a method to improve transfer without modifying weights; (4) Answer "why it drops" through mechanistic analysis.

**Key Insight**: The authors hypothesize that the edit is actually modified in the parameters but remains "latent" within the weights, only being transmitted to the visual generation pathway when activated by explicit textual context.

**Core Idea**: First, let the model "speak out" the edited fact in text to convert latent parameter changes into explicit textual conditions. Then, superimpose this textual condition onto the image prompt and feed it into the generator—this is referred to as Reasoning-augmented Parameter Editing.

## Method

### Overall Architecture
This work addresses whether editing a fact on the text side causes corresponding changes in image generation. Instead of training new models, the problem is decomposed into three reproducible tasks: first, using the UniKE benchmark to turn "whether the edited fact exists in the image" into a binary verifiable metric; second, comparing Direct vs. Reasoning-Augmented protocols to measure the gap between "direct drawing" and "drawing after speaking"; and finally, using cosine drift analysis along the conditioning pathway to locate where the signal is attenuated. Each evaluation instance is formalized as $\mathcal{I}=(q, y, y', p_{img}, t_{vis}, q_{vqa})$ (edit prompt, original answer, target answer, image generation prompt, visual target description, and VQA verification question). Generated images are judged $0/1$ by Qwen3-VL-235B as an LLM-as-judge. The overall structure forms a $9 \times 2$ evaluation matrix (3 UMMs: Ovis-U1 / BLIP3o-4B / OmniGen2 $\times$ 3 editors: MEMIT / PMET / AlphaEdit $\times$ 2 protocols).

### Key Designs

**1. UniKE Benchmark: Quantifying Text-to-Image Edit Propagation**

Previous benchmarks could not measure this path: pure-text benchmarks (ZsRE / CounterFact / MQuAKE) ignore images, while multimodal benchmarks like TMKE only test I2T. UniKE fills this gap using "answer-neutral" image prompts combined with VQA-as-judge. Attribute edits are generated via a Gemini-3.0-Flash self-instruction pipeline into four stages of increasing difficulty (Stage 1: atomic object inquiry; Stage 2: real-world scene embedding; Stage 3: multi-entity complex composition; Stage 4: derived product/use transfer). Relation edits are sampled from CounterFact / MQuAKE, with non-visualizable categories (e.g., nationality) filtered out. The key constraint is the "answer-neutral" principle: prompts must not leak the original or target values, ensuring any correct expression in the image stems from internal edited knowledge. The benchmark covers 2,971 subjects and 5,535 instances across attributes (color, material, shape, size, pattern) and relations (affiliation, creator, location, profession).

**2. Reasoning-augmented Parameter Editing: Activating Latent Edits via Textual Reasoning**

The authors found that while text-side editing success (Eff.) is high ($55\%–90\%$), image VQA is extremely low, suggesting the fact is updated in the LLM but fails to reach the generation pathway. The Reasoning-Augmented protocol addresses this: instead of feeding $p_{img}$ directly to the generator (Direct protocol), it uses a category-conditioned template $p_{rea}$ to trigger the edited model to generate a textual rationale $r$ (produced by the model itself, not an oracle). This $r$ is prepended to $p_{img}$ as an extra condition. The rationale "explicitizes" the latent edit in the MLP weights into token-level text constraints, acting as a longer, better-aligned conditioning vector to compensate for signal attenuation. It is weight-free and orthogonal to any editor, improving VQA across all 9 model-editor pairs with a maximum gain of $+18.6$ pp.

**3. Conditioning Pathway Drift Analysis: Localizing Bottlenecks to LLM-to-DiT Projection**

To distinguish between "edit failed to modify the LLM" and "edit modified the LLM but failed to propagate," the authors sampled 100 cases using PMET to quantify signals along the pathway. They defined a cosine offset operator $\Delta_{cos}(a, b) = 1 - a^\top b / (\|a\| \|b\|)$. At the LLM output, they measured the per-token average $d_{cos}^{tok}$ and relative Frobenius drift $r_F = \|\delta\|_F / \|C_{fresh}^{LLM}\|_F$ to gauge the perturbation caused by parameter editing. At the DiT input, they used $d_{cos}^{dir}$ and $d_{cos}^{rea}$ on mean-pooled conditioning vectors to measure the actual offset received by the DiT. Results showed that Ovis-U1, which uses a frozen dimensionality-reduction projection, had an $r_F$ of only 0.078, whereas BLIP3o-4B reached 0.527. The former's projection acts as an "architectural filter" that blocks wide-distribution edit perturbations. However, Ovis-U1 benefited most from reasoning augmentation ($d_{cos}^{rea} = 0.154$ vs. $d_{cos}^{dir} = 0.018$, an $8\times$ amplification), because the rationale injects perturbations into directions preserved by the projection.

### Loss & Training
This work does not train new models; all editors follow their original objective functions (closed-form weight updates for MEMIT/PMET, null-space projection for AlphaEdit). Only middle MLP layers are edited: layers 4–8 for Ovis-U1 and layers 6–10 for BLIP3o-4B and OmniGen2. For AlphaEdit on BLIP3o-4B / OmniGen2, the authors used $\alpha=0.7/0.6$ to interpolate the null-space projector with the identity matrix to obtain a "softened" version (marked with an asterisk in the paper), preventing over-constrained parameter updates from damaging generation quality. All edits were performed in a sequential editing setting.

## Key Experimental Results

### Main Results
Summary of Overall metrics (Eff. = Text-side editing accuracy, VQA = Image VQA accuracy, in %):

| Model | Editor | Eff. (Direct) | VQA (Direct) | VQA (+Reasoning) | Gain (pp) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ovis-U1 | PMET | 72.18 | 9.71 | 28.32 | +18.6 |
| Ovis-U1 | MEMIT | 59.84 | 8.70 | 24.41 | +15.7 |
| BLIP3o-4B | PMET | 76.30 | 18.51 | 19.29 | +0.8 |
| BLIP3o-4B | AlphaEdit* | 77.88 | 16.12 | 17.33 | +1.2 |
| OmniGen2 | PMET | 76.20 | 11.43 | 16.01 | +4.6 |
| OmniGen2 | AlphaEdit* | 76.37 | 11.50 | 17.90 | +6.4 |

The most striking discovery is the modality gap: under the Direct protocol, VQA is only $1/8$ to $1/4$ of Eff. Reasoning-Augmentation improved VQA across all 9 pairs, though the gain is highly architecturally dependent (Ovis-U1 benefited most).

### Ablation Study
Conditioning pathway drift analysis for PMET on 100 sampled cases (Source: Table 4):

| Model | LLM Output $d_{cos}^{tok}$ | LLM Output $r_F$ | DiT Input $d_{cos}^{dir}$ | DiT Input $d_{cos}^{rea}$ |
| :--- | :--- | :--- | :--- | :--- |
| Ovis-U1 | 0.003 | 0.078 | 0.018 | 0.154 |
| BLIP3o-4B | 0.139 | 0.527 | 0.031 | 0.064 |
| OmniGen2 | 0.038 | 0.262 | 0.018 | 0.092 |

Ovis-U1 has the weakest implicit drift (filtered by the frozen projection), but reasoning augmentation amplifies DiT-side drift by $8\times$. BLIP3o-4B has the largest implicit drift but fails to propagate it, reflecting that "large drift $\neq$ good alignment."

### Key Findings
- Text-side Eff. is almost uncorrelated with image VQA accuracy: high Eff. does not guarantee that the edited fact is visible in the image, debunking the intuitive assumption that "unified backbone $\implies$ automatic cross-modal knowledge propagation."
- Significant difficulty variance by category: In attributes, "size" is easiest (VQA handles relative comparisons well), while "shape" is hardest (precise geometric control is difficult). In relations, "occupation" is easiest (due to local visual proxies like uniforms), while "creator" is hardest (authorship is non-visual).
- From Stage 1 to Stage 2, text Eff. drops by $70\%$ on average, but reasoning accuracy only drops $\sim 10\%$. This indicates the edited fact is "in the weights" but sensitive to the edit template; reasoning acts as a robust retrieval interface.
- Conditioning decay occurs primarily before the DiT (Appendix D.3), not inside it—meaning future research should focus on the synergistic design of editors and projection layers.

## Highlights & Insights
- **First quantifiable benchmark for cross-modal KE**: The combination of answer-neutral image prompts and VQA-as-judge transforms "whether the fact is in the image" from a subjective question into a reproducible binary judgment, applicable to cross-modal unlearning and alignment.
- **Training-free Reasoning-Augmented protocol is a strong plug-in baseline**: By converting "latent parameter changes" into "textual constraints," this paradigm offers insights for future multimodal CoT editing and test-time intervention.
- **Treating the UMM as a "signal attenuation system" via cosine drift analysis**: Decomposing signals along the LLM-DiT pathway allows for precise diagnosis of black-box UMM failures, a technique generalizable to any "backbone + projection + head" architecture.

## Limitations & Future Work
- **Limitations**: Only three UMMs and three editors were tested; reasoning gains are limited for BLIP3o/OmniGen2, suggesting textual activation is not a universal solution; rationales themselves can introduce new errors.
- **Additional Insights**: All evaluations used sequential single-edit settings, leaving lifelong/batch editing unexplored; the Qwen3-VL judge might favor its own family of models; the failure in Stage 4 (derived products) could be due to UMM reasoning deficiencies rather than editing failure.
- **Future Directions**: (1) Design modality-aware editors that constrain weight updates to subspaces preserved by projections; (2) Jointly optimize rationale generation with the editing process; (3) Explore editing cross-attention layers rather than MLP layers to more directly influence the visual pathway.

## Related Work & Insights
- **vs. MEMIT / PMET / AlphaEdit (Pure-text KE)**: These methods achieve high Eff. on UniKE but very low VQA, proving their "success" is limited by modality and highlighting the need for cross-modal evaluation.
- **vs. TMKE (Multimodal KE Benchmark)**: TMKE focuses on I2T (answering based on images), while UniKE focuses on T2I (generating images based on edited text). The former verifies the "understanding side," whereas the latter verifies the "generation side."
- **vs. T2I Editing (TIME / ReFACT / DiffQuickFix)**: These target modular diffusion models (text encoder/cross-attention) and are not directly applicable to monolithic UMMs. This paper suggests UMMs require a new editing paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of cross-modal KE in UMMs and first to measure signal attenuation along the conditioning pathway.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models $\times$ 3 editors $\times$ 2 protocols, including stage/category/mechanistic analysis. However, UMM count is low and lifelong editing is missing.
- Writing Quality: ⭐⭐⭐⭐ Motivation is very clear, but the mechanistic analysis involves heavy notation that may be challenging to read.
- Value: ⭐⭐⭐⭐⭐ Establishes a benchmark for the new UMM editing track and provides a training-free strong baseline and clear research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[CVPR 2026\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2026/knowledge_editing/mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [\[ICML 2026\] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)
- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](../../ACL2025/knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ACL 2025\] REP: Keys to Robust Edits — From Theoretical Insights to Practical Advances](../../ACL2025/knowledge_editing/rep_robust_knowledge_editing.md)

</div>

<!-- RELATED:END -->
