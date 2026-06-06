---
title: >-
  [Paper Note] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs
description: >-
  [ICML 2026][Knowledge Editing][Cross-Modal Transfer] This paper introduces UniKE—the first "Cross-Modal Knowledge Editing" benchmark for Unified Multimodal Models (UMMs), comprising 2,971 edited subjects and 5…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "Cross-Modal Transfer"
  - "Unified Multimodal Models"
  - "Reasoning Augmentation"
  - "Conditioning Path"
date: 2026-05-08
content_hash: ea50742627f13ab0
---

# Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs

**Conference**: ICML 2026  
**arXiv**: [2606.00477](https://arxiv.org/abs/2606.00477)  
**Code**: https://github.com/gxx27/UniKE (Yes)  
**Area**: Knowledge Editing / Cross-Modal / Unified Multimodal Models (UMM)  
**Keywords**: Knowledge Editing, Cross-Modal Transfer, Unified Multimodal Models, Reasoning Augmentation, Conditioning Path

## TL;DR
This paper introduces UniKE—the first "Cross-Modal Knowledge Editing" benchmark for Unified Multimodal Models (UMMs), comprising 2,971 edited subjects and 5,535 VQA-verifiable instances. It systematically reveals a significant modality gap, where the "text-side edit success rate is ~92% but image generation VQA accuracy is only ~18.5%." By employing a "Reasoning-augmented Parameter Editing" protocol, the authors improve VQA accuracy by up to 18.6 percentage points. Furthermore, they localize the root cause to the LLM-to-DiT projection bottleneck using a cosine drift metric along the conditioning path.

## Background & Motivation
**Background**: Unified Multimodal Models (UMM) integrate image understanding and generation into a single transformer backbone, relying on shared parameters for end-to-end synergy between text and images (e.g., Ovis-U1, BLIP3o-4B, OmniGen2). Concurrently, pure-text Knowledge Editing (KE) methods—such as ROME, MEMIT, PMET, and AlphaEdit—have matured, allowing precise rewriting of MLP weights (e.g., changing "Apple's founder is Jobs" to "Tim Cook") without retraining.

**Limitations of Prior Work**: Despite UMMs sharing a backbone, the question of whether "editing a fact on the text side automatically transfers to image generation" has not been systematically studied. Existing multimodal KE benchmarks (like TMKE) only measure image-conditioned text answering (I2T), omitting the critical text-to-image (T2I) propagation path.

**Key Challenge**: Text-side editing only requires shifting the "next token distribution," a relatively low bar. Conversely, to influence image generation, perturbations must traverse the entire "LLM → Projection Layer → DiT" conditioning path without being attenuated or filtered. The required signal strength and directionality for the latter are orders of magnitude higher.

**Goal**: (1) Construct a cross-modal KE benchmark that is visually verifiable; (2) Quantify the information loss between "text-side editing" and "image generation"; (3) Identify a weight-free method to improve transfer; (4) Provide a mechanistic analysis of why the gap exists.

**Key Insight**: The authors hypothesize that the knowledge is indeed modified within the parameters but remains "latent," only propagating to the visual generation path when activated by explicit textual context.

**Core Idea**: The model is first prompted to "verbalize" the edited fact in text, transforming latent parameter changes into explicit textual conditions. These conditions are then superimposed onto the image prompt for the generator—a method termed Reasoning-augmented Parameter Editing.

## Method

### Overall Architecture
The work consists of three components: the UniKE benchmark, two evaluation protocols (Direct / Reasoning-Augmented), and a conditioning path mechanistic analysis. Each evaluation instance is formalized as $\mathcal{I}=(q, y, y', p_{img}, t_{vis}, q_{vqa})$, representing the edit prompt, original answer, target answer, image generation prompt, visual target description, and VQA verification question. Images are evaluated using Qwen3-VL-235B as an LLM-as-judge for 0/1 binary classification. The evaluation matrix covers three UMMs (Ovis-U1 / BLIP3o-4B / OmniGen2) × three editors (MEMIT / PMET / AlphaEdit) × two protocols.

### Key Designs

1.  **UniKE Benchmark — A Visually Verifiable Cross-Modal KE Dataset**:
    *   **Function**: Provides 2,971 subjects and 5,535 instances covering two main edit categories: attribute (color/material/shape/size/pattern) and relation (membership/creator/location/occupation). Every instance is automatically verifiable via VQA.
    *   **Mechanism**: For attribute edits, the authors use a Gemini-3.0-Flash self-instruction pipeline to generate $(q, y, y')$ triplets across four progressive difficulty stages: Stage 1 (atomic objects), Stage 2 (real-world scenarios), Stage 3 (complex multi-entity compositions), and Stage 4 (derived products/uses). Relation edits are extracted from CounterFact/MQuAKE and filtered for visualizability. Image prompts follow an "answer-neutral" principle, ensuring the prompt does not leak the target value, thus forcing the model to rely on its internal edited knowledge.
    *   **Design Motivation**: Previous T2T benchmarks cannot test images, and I2T benchmarks omit the T2I direction. UniKE enables the quantification of how text edits affect image generation via answer-neutral prompts and VQA judges.

2.  **Reasoning-augmented Parameter Editing — Activating Latent Edits via Textual Reasoning**:
    *   **Function**: A two-stage "think-then-draw" protocol that improves VQA accuracy across all model-editor pairs (up to +18.6 pp) without modifying any weights.
    *   **Mechanism**: While the Direct protocol feeds $p_{img}$ directly into the generator, the Reasoning-Augmented protocol uses a category-conditioned template $p_{rea}$ to trigger the model to generate a text rationale $r$ (produced by the edited model itself). This $r$ is then prepended to $p_{img}$. The rationale explicitly converts latent MLP edits into token-level constraints, providing a stronger, aligned semantic condition for the DiT.
    *   **Design Motivation**: Observations showed relatively high text-side efficacy (55%–90%) but very low VQA accuracy. Reasoning augmentation compensates for signal attenuation in the conditioning path by using aligned condition vectors.

3.  **Conditioning Path Drift Analysis — Localizing LLM-to-DiT Bottlenecks via Cosine Distance**:
    *   **Function**: Analyzes 100 cases using PMET to measure "implicit drift in LLM output" versus "actual drift in DiT input vectors," localizing the gap to the projection layer/path alignment.
    *   **Mechanism**: Uses a cosine drift operator $\Delta_{cos}(a,b)=1-a^\top b/(\|a\|\|b\|)$, per-token average $d_{cos}^{tok}$, and relative Frobenius drift $r_F=\|\delta\|_F/\|C_{fresh}^{LLM}\|_F$ to quantify perturbations at the LLM output. It then compares this with $d_{cos}^{dir}$ and $d_{cos}^{rea}$ at the DiT input. Results show that Ovis-U1, with its frozen dimensionality-reduction projection, acts as an "architectural filter" with $r_F$ of only 0.078, whereas BLIP3o-4B reaches 0.527.
    *   **Design Motivation**: This allows distinguishing whether the edit failed to change the LLM or if the change failed to propagate. It reveals that the alignment of the conditioning path is more critical than the editor's raw drift magnitude.

### Loss & Training
No new models were trained. Editors follow their original objective functions (closed-form updates for MEMIT/PMET, null-space projection for AlphaEdit). For UMMs, editing is focused on intermediate MLP layers (layers 4–8 for Ovis-U1; layers 6–10 for BLIP3o-4B and OmniGen2). For AlphaEdit on BLIP3o-4B/OmniGen2, a softened version (indicated with an asterisk *) is used with $\alpha=0.7/0.6$ to prevent excessive parameter space contraction. All edits utilize sequential editing settings.

## Key Experimental Results

### Main Results
Summary of overall metrics for 3 UMMs × 3 Editors (Eff. = Text-side editing accuracy, VQA = Image VQA accuracy, in %):

| Model | Editor | Eff. (Direct) | VQA (Direct) | VQA (+Reasoning) | Gain (pp) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ovis-U1 | PMET | 72.18 | 9.71 | 28.32 | +18.6 |
| Ovis-U1 | MEMIT | 59.84 | 8.70 | 24.41 | +15.7 |
| BLIP3o-4B | PMET | 76.30 | 18.51 | 19.29 | +0.8 |
| BLIP3o-4B | AlphaEdit* | 77.88 | 16.12 | 17.33 | +1.2 |
| OmniGen2 | PMET | 76.20 | 11.43 | 16.01 | +4.6 |
| OmniGen2 | AlphaEdit* | 76.37 | 11.50 | 17.90 | +6.4 |

The most striking finding is the modality gap: VQA accuracy is only 1/8 to 1/4 of the text efficacy under the Direct protocol. Reasoning-Augmented improves VQA across all 9 pairs, though the gain is architecture-dependent.

### Ablation Study
Conditioning path drift analysis for PMET on 100 sampled cases (Source: Paper Table 4):

| Model | LLM Output $d_{cos}^{tok}$ | LLM Output $r_F$ | DiT Input $d_{cos}^{dir}$ | DiT Input $d_{cos}^{rea}$ |
| :--- | :--- | :--- | :--- | :--- |
| Ovis-U1 | 0.003 | 0.078 | 0.018 | 0.154 |
| BLIP3o-4B | 0.139 | 0.527 | 0.031 | 0.064 |
| OmniGen2 | 0.038 | 0.262 | 0.018 | 0.092 |

Ovis-U1 shows the weakest LLM drift (filtered by frozen projection) but its DiT drift is amplified 8x by reasoning augmentation. BLIP3o-4B has high LLM drift but fails to propagate it, showing that "high drift $\neq$ good alignment."

### Key Findings
*   Text-side Eff. and image VQA accuracy are almost uncorrelated. This refutes the intuitive assumption that a unified backbone leads to automatic cross-modal knowledge propagation.
*   Category difficulty varies significantly: In attributes, "size" is easiest (relative comparisons), while "shape" is hardest (precise geometric control). In relations, "occupation" is easiest (visual proxies like uniforms), while "creator" is hardest (non-visual identity).
*   Textual Eff. drops by ~70% from Stage 1 to Stage 2, but reasoning accuracy only drops by ~10%. This proves edited facts are "in the weights" but are sensitive to templates; rationales serve as more robust retrieval interfaces.
*   Conditioning attenuation primarily occurs before the DiT (Appendix D.3), suggesting that future editors should be co-designed with projection layers.

## Highlights & Insights
*   **Establishment of a Quantitative Cross-Modal KE Benchmark**: The combination of answer-neutral prompts and VQA-as-judge transforms "is the fact present in the image" from a subjective question into a reproducible binary evaluation.
*   **Plug-in Training-free Baseline**: The Reasoning-Augmented protocol is editor-agnostic and significantly improves performance by externalizing "latent" changes, offering insights for future multimodal CoT editing.
*   **Signal Decay Diagnosis**: Treating the UMM as a signal attenuation system and measuring drift along the LLM-DiT path provides a brilliant diagnostic framework for localized failure analysis in complex architectures.

## Limitations & Future Work
*   **Limitations**: The study is restricted to three UMMs and three editors. The reasoning protocol provides limited gains for BLIP3o/OmniGen2, and the rationale itself can introduce new errors.
*   **Future Work**: (1) Design modality-aware editors that constrain updates to subspaces preserved by projection; (2) Jointly optimize rationale generation with the editing process; (3) Explore editing cross-attention layers to directly influence visual conditioning.

## Related Work & Insights
*   **Vs. MEMIT / PMET / AlphaEdit**: These methods show high text Eff. but low VQA accuracy on UniKE. Success in pure-text KE is revealed to be modality-limited.
*   **Vs. TMKE**: TMKE focuses on I2T (answering based on images); UniKE targets T2I (editing text to change images), providing a complementary perspective.
*   **Vs. T2I Editing (TIME/ReFACT)**: These edit modular text encoders. This paper demonstrates that monolithic UMMs require entirely new editing paradigms.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ First systematic study of cross-modal KE in UMMs and the first to quantify signal decay along the conditioning path.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid coverage across models, editors, and diagnostic analysis, though limited to single-edit settings.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and execution, although the mechanistic analysis contains dense notation.
*   **Value**: ⭐⭐⭐⭐⭐ Establishes a benchmark for the new field of UMM editing while providing a strong, training-free baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2026/knowledge_editing/mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [\[ICML 2026\] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2026/knowledge_editing/mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [\[ICML 2026\] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)
- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](../../ACL2025/knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)

</div>

<!-- RELATED:END -->
