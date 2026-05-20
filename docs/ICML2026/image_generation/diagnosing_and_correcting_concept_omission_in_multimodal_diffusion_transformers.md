---
title: >-
  [Paper Note] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers
description: >-
  [ICML 2026][Image Generation][MM-DiT] This paper uses linear probes to discover that in MM-DiT (FLUX / SD3.5), certain attention heads in intermediate layers naturally encode a binary signal in the key vectors of text to…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "MM-DiT"
  - "Concept Omission"
  - "Linear Probe"
  - "Attention Key Intervention"
  - "Training-free Guidance"
date: 2026-05-08
content_hash: b1a4f6427dc001e1
---

# Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers

**Conference**: ICML 2026  
**arXiv**: [2605.14270](https://arxiv.org/abs/2605.14270)  
**Code**: No public repository link provided  
**Area**: Diffusion Models / Text-to-Image Generation / Representation Intervention  
**Keywords**: MM-DiT, Concept Omission, Linear Probe, Attention Key Intervention, Training-free Guidance

## TL;DR
This paper uses linear probes to discover that in MM-DiT (FLUX / SD3.5), certain attention heads in intermediate layers naturally encode a binary signal in the key vectors of text tokens, indicating whether a target concept will appear. Based on this, the authors propose Omission Signal Intervention (OSI): during inference, they inject the mean difference direction between "omission" and "existence" classes into the key vectors of the top-K heads with strength $\alpha\sigma\boldsymbol{\theta}$, thereby activating the model's "self-awareness" of missing concepts and prompting completion. On FLUX, GenEval 6-object accuracy improves from 0.18 → 0.40, without any fine-tuning.

## Background & Motivation
**Background**: T2I diffusion models have largely transitioned to the MM-DiT architecture (FLUX, SD3, SD3.5), which concatenates text tokens and image patch tokens into a single sequence for joint attention. Compared to U-Net + cross-attention's unidirectional injection, this is better suited for learning cross-modal semantic alignment.

**Limitations of Prior Work**: Despite architectural advances, concept omission (e.g., generating "cat and dog and book" but missing the book) remains a persistent issue in MM-DiT. Existing remedies either add optimization constraints to visual attention maps (Attend-and-Excite, A-STAR, Rassin et al.) or require extra training (GLIGEN, reward fine-tuning), both incurring high inference costs or disrupting the original model distribution.

**Key Challenge**: Prior research focuses only on visual embeddings; how text embeddings internally encode "concept presence" within diffusion models remains a black box. Chen et al. (2024) analyzed CLIP text outputs but did not trace down to diffusion's internal attention heads. In other words, the model may "know" what it has omitted, but we lack tools to query this.

**Goal**: (1) Use probing tools to check whether MM-DiT's internal text tokens truly carry "omission state" information; (2) If so, identify the specific layers/heads/timesteps carrying this signal; (3) Amplify this signal at inference time with minimal cost to force the model to generate missing concepts.

**Key Insight**: MM-DiT's joint attention allows text tokens to receive visual feedback from image tokens at every layer, making their key/value vectors dynamic reflections of what has already been generated in the latent. In intermediate layers (where image tokens carry semantics but generation is incomplete), the key vectors of text tokens may encode "has my concept appeared in the noisy latent?"

**Core Idea**: First, use a linear probe to explicitly extract this "omission signal" from attention keys (the mean difference direction between "absent" and "present"); then, during inference, apply a mass mean shift to move the key in the "reverse omission" direction, making the model "believe" it has omitted more, thus actively compensating for missing concepts.

## Method

### Overall Architecture
Two stages. **Diagnose**: On the GenEval two-object subset of FLUX.1-Dev, generate images and use Mask2Former + BLIP-VQA dual annotation to determine whether each concept token was successfully generated ($y\in\{0,1\}$); collect text token key vectors $\mathbf{k}_c^{(t,l,h)}$ at intermediate timesteps to form dataset $\mathcal{D}_{l,h}^{\text{train}}$, and train a linear probe for each $(l, h)$ to classify absent/present. **Correct (OSI)**: For each head, compute the mass mean shift direction $\boldsymbol{\delta}^{(l,h)} = \mathbb{E}[\mathbf{k}|y=0]-\mathbb{E}[\mathbf{k}|y=1]$, normalize to get $\boldsymbol{\theta}^{(l,h)}$; during inference, for only the top-K most accurate heads and only in the early denoising phase $t\in[t_{\text{stop}}, 1]$, shift the concept token's key vector as $\mathbf{k}_c \leftarrow \mathbf{k}_c + \alpha\sigma^{(l,h)}\boldsymbol{\theta}^{(l,h)}$. The addition is in the "absent - present" direction, effectively making the model internally "feel" it has omitted more, thus triggering its built-in correction mechanism.

### Key Designs

1. **Concept-aware Probing Dataset Construction with Noise Filtering**:

    - **Function**: Collects clean (key vector, present-label) pairs from the FLUX generation process, specifically for training linear probes to detect omission signals.
    - **Mechanism**: Uses the template "a photo of {obj1} and {obj2}" for batch generation, recording text token key vectors $\mathbf{k}^{(t,l,h)}$ at each step, layer, and head. Labels are derived from the final image—adopted only if both Mask2Former (mmdetection) and BLIP-VQA agree, avoiding single annotator noise. Two types of traps are filtered: (a) **Early timestep label misalignment**—the target hasn't appeared in steps 1-3, but is labeled $y=1$ due to eventual success (the key encodes "not yet generated" but the label says "will be generated," causing training contradiction); (b) **Late timestep weak signal**—in late diffusion, omission signal fades as details are refined. Table 1 quantifies this: early step agreement with final label is only 0.409, intermediate jumps to 0.965, late reaches 1.000. Thus, only intermediate interval $\mathcal{T}$ samples are kept for probe training, with a 4:1 train/val split.
    - **Design Motivation**: The probe is essentially a sanity check, so data must truly reflect "this token, at this moment, knows whether its concept will appear"—otherwise, identified "important heads" are just noise. Filtering data by generation dynamics embeds diffusion's temporal structure into probe design.

2. **Head-Localization via Linear Probing**:

    - **Function**: Precisely locates "expert heads" encoding the omission signal within the vast space of 1368 attention heads × multiple timesteps, avoiding indiscriminate intervention.
    - **Mechanism**: Trains a linear classifier for each $(l, h)$, inputting the text token key vector from that head and outputting absent vs present. Fig. 2(a) shows probe accuracy peaks at intermediate timesteps (matching Table 1). Fig. 2(b)'s head-wise heatmap reveals a counterintuitive pattern: **early layers are near chance** (text tokens haven't absorbed enough visual feedback), **intermediate layers rise sharply** (up to 91.0% accuracy), **late layers drop again**. Thus, the top-300 heads (22% of 1368, all above 80% accuracy) are selected as "omission signal heads." Figs. 3 and 4 use box plots to show probe outputs evolving with timestep: the first row shows predicted $\hat{x}_0$ visual objects gradually appearing, the second row shows the probe's "present" probability rising in sync—a vivid visualization of "model self-awareness" dynamics.
    - **Design Motivation**: The signal is sparse—not all heads encode omission, so per-head probing ensures intervention only where useful, avoiding broad disruption. Training a probe for each $(l, h)$ is laborious but yields precise localization, minimizing negative side effects of subsequent OSI.

3. **Mass Mean Shift Intervention (OSI)**:

    - **Function**: At inference, injects the "absent → present" direction into key vectors with minimal cost and no training, prompting the model to complete missing concepts.
    - **Mechanism**: For each top-K head, computes the mass mean shift direction $\boldsymbol{\delta}^{(l,h)} = \mathbb{E}[\mathbf{k}^{(t,l,h)}|y=0] - \mathbb{E}[\mathbf{k}^{(t,l,h)}|y=1]$ (note: "absent minus present"), normalizes to $\boldsymbol{\theta}^{(l,h)}$. During inference, modifies as $\mathbf{k}_c \leftarrow \mathbf{k}_c + \alpha\sigma^{(l,h)}\boldsymbol{\theta}^{(l,h)}$, where $\sigma^{(l,h)}$ is the standard deviation along this direction (adaptive scaling), and $\alpha$ is a scalar (5.0 for FLUX, 7.5 for SD3.5). Only active in the first 15 of 30 steps, $t\in[0.78, 1]$, as concept formation mainly occurs early. The addition is in the "omission direction" (not "present direction"); the paper explains this as "making the model believe it has omitted more → triggering stronger correction"—an intentional hallucination of severity.
    - **Design Motivation**: Traditional methods either modify attention maps with constraints or require fine-tuning; OSI uses the mass mean shift + steering vector paradigm (Li et al. 2023a) for fully training-free, surgical intervention. The counterintuitive choice to "amplify the absence signal" rather than "inject the presence signal" is validated by Table 4's direction ablation—reversing the direction ($-\boldsymbol{\theta}$) collapses performance (two object 0.81 → 0.72, six object 0.18 → 0.02), showing that only the correct direction enables recovery, while the wrong direction suppresses.

### Loss & Training
OSI is training-free: the model itself is unchanged, only the key vectors are modified during inference as described above. The only training required is for the 1368 lightweight linear probes ($\ll$ main model), using BCE loss. 30 total sampling steps, CFG=3.5 (FLUX)/7.0 (SD3.5), top-K=300 (FLUX)/100 (SD3.5), $\alpha=5.0/7.5$, $t_{\text{stop}}=0.78/0.76$, active for the first 15 steps. Token selection: GenEval uses template rules for parsing, T2I-CompBench uses Llama-3.1-8B to extract target spans.

## Key Experimental Results

### Main Results
GenEval multi-object (2-6 obj) + T2I-CompBench non-spatial subset (object omission):

| Backbone | Method | 2-obj | 3-obj | 4-obj | 5-obj | 6-obj | Avg | T2I non-spatial |
|----------|--------|-------|-------|-------|-------|-------|-----|-----------------|
| FLUX | Base | 0.81 | 0.63 | 0.44 | 0.29 | 0.18 | 0.47 | 0.3069 |
| FLUX | TACA | 0.89 | 0.68 | 0.56 | 0.31 | 0.20 | 0.53 | 0.3078 |
| FLUX | PLADIS | 0.87 | 0.71 | 0.56 | 0.31 | 0.20 | 0.53 | 0.3075 |
| FLUX | **OSI** | **0.92** | **0.71** | **0.64** | **0.40** | **0.40** | **0.61** | **0.3083** |
| SD3.5 | Base | 0.82 | 0.72 | 0.59 | 0.38 | 0.24 | 0.55 | 0.3155 |
| SD3.5 | TACA | 0.87 | 0.74 | 0.55 | 0.47 | 0.26 | 0.58 | 0.3164 |
| SD3.5 | **OSI** | **0.89** | **0.80** | **0.68** | 0.46 | **0.35** | **0.64** | 0.3159 |

Attribute neglect (T2I-CompBench attribute binding):

| Backbone | Method | Color | Shape | Texture |
|----------|--------|-------|-------|---------|
| FLUX | Base | 0.7923 | 0.4995 | 0.6419 |
| FLUX | TACA | 0.7742 | 0.5118 | 0.6493 |
| FLUX | **OSI** | **0.8014** | **0.5819** | **0.7039** |
| SD3.5 | Base | 0.7955 | 0.5820 | 0.7305 |
| SD3.5 | **OSI** | **0.8048** | **0.6119** | **0.7480** |

Most notably, the 6-object scenario improves from 0.18 → 0.40, more than doubling, demonstrating OSI's true power in "hard cases where the baseline nearly collapses."

### Ablation Study
Direction + head selection + token-specific (FLUX):

| Setting | 2-obj | 3-obj | 4-obj | 5-obj | 6-obj |
|---------|-------|-------|-------|-------|-------|
| Base | 0.81 | 0.63 | 0.44 | 0.29 | 0.18 |
| Direction = Opposite ($-\boldsymbol{\theta}$) | 0.72 | 0.40 | 0.15 | 0.06 | 0.02 |
| Direction = Random | 0.84 | 0.65 | 0.53 | 0.32 | 0.30 |
| Heads = Bottom-K | 0.81 | 0.60 | 0.47 | 0.22 | 0.15 |
| Heads = Random K | 0.82 | 0.67 | 0.55 | 0.31 | 0.17 |
| Heads = All 1368 | 0.90 | 0.74 | 0.50 | 0.33 | 0.32 |
| **Ours (top-K + θ)** | **0.92** | **0.71** | **0.64** | **0.40** | **0.40** |

Token-specific (100 FLUX omission failure cases):

| Method | Accuracy (omitted obj) | Probe Prob (omitted obj) |
|--------|------------------------|--------------------------|
| FLUX (no OSI) | 0.00 | 0.292 |
| OSI on omitted token | **0.70** | **0.658** |
| OSI on present token | 0.14 | 0.298 |

Intervention duration ($t_{\text{stop}}$):

| Step | 0 | 5 | 10 | **15 (Ours)** | 20 | 25 | 30 |
|------|---|---|----|---------------|----|----|---|
| Accuracy | 0.82 | 0.88 | 0.91 | **0.92** | 0.92 | 0.92 | 0.91 |
| MANIQA | 0.473 | 0.479 | 0.480 | **0.480** | 0.481 | 0.480 | 0.480 |

### Key Findings
- **Direction is everything**: Reversing the intervention causes 6-obj accuracy to collapse from 0.18 to 0.02, showing that the mass mean shift direction is not an arbitrary steering—it precisely corresponds to the "concept realization" semantic axis; the wrong direction is worse than no intervention.
- **Surgical intervention on top heads is essential**: Bottom-K is almost ineffective or even regressive, random is weak, all heads (1368) are strong but top-K surpasses them, indicating "few but precise" is better than "broad coverage," validating probe-based head selection.
- **OSI's effect is token-specific**: Applying the same intervention to "already successfully generated" tokens yields little improvement (0.14 vs 0.70), proving OSI is not a universal "image enhancer" but truly targeted—it amplifies the "concept absence signal," ineffective for concepts not absent. This establishes a "causal link to concept generation."
- **Major gains achieved in the first 5 steps**: At $t_{\text{stop}}=0.95$ (intervening only in the first 5 steps), accuracy is already 0.88, saturating at 0.92 by step 15; this confirms concept formation mainly occurs in early denoising, consistent with "perception prioritized" training theory. MANIQA remains stable, showing OSI does not degrade image quality.
- **Trained on objects, generalizes to attributes**: The probe is trained only with object-level labels, but OSI is effective for color/shape/texture attribute binding, indicating the omission signal is a more general "semantic realization" concept.

## Highlights & Insights
- **"The model actually knows what it has omitted" is a striking finding**: Using a 91% accuracy probe to quantify this internal awareness, then amplifying it into a controllable guidance signal via mass mean shift, the full chain of "diagnosis → intervention → verification" is established.
- **Training-free with minimal extra computation**: Only key vectors are modified, only in top-K heads, only in the first 15 steps, with negligible overhead; this "surgical inference-time intervention" paradigm is a major engineering advance over fine-tuning or adding extra loss.
- **"Amplifying the absence signal rather than injecting the presence signal" is counterintuitive but precise**: Intuitively, one might "make the model believe it has already drawn it," but in reality, "making the model believe it has omitted more" triggers its built-in correction mechanism. This suggests diffusion models may have learned an implicit "self-correction loop," and OSI simply amplifies its input signal.
- **Mass Mean Shift + Probe is a general recipe for Diffusion Steering**: This approach (find signal-carrying heads → compute inter-class mean difference → push on key) can be directly transferred to style control, composition correction, concept fusion, and other controllability problems; essentially, it brings ITI (Inference-Time Intervention) from LLMs to MM-DiT.
- **Data filtering tied to temporal structure**: The 0.409 / 0.965 / 1.000 agreement numbers precisely distinguish the usability of early/mid/late timestep data, embedding diffusion dynamics into representation analysis.

## Limitations & Future Work
- **Over-generation side effect** (Fig. 7): The authors acknowledge OSI sometimes "draws too much"—generating two or three objects when only one is needed. This requires tuning $\alpha$ and $K$, fundamentally due to the lack of a closed-loop stopping criterion.
- Relies on "binary absent/present labels," making it hard to apply to ambiguous or subjective concepts (aesthetic style, relative quantities like "two or three")—the signal itself cannot be binarized.
- Top-K and $\alpha$ are empirically set via grid search and must be retuned for different backbones; lacks automated calibration.
- Probe reliability in dense 4-6 object scenarios is not separately evaluated; whether head selection should be prompt-conditional when prompts contain 5 concepts is not discussed.
- The time window $[t_{\text{stop}}, 1]$ is static; theoretically, dynamic intervention could be implemented—monitoring real-time probe probabilities and extending intervention for tokens with persistently low probability, potentially further improving results.
- Experiments focus on FLUX and SD3.5, not validated on other MM-DiT derivatives like Imagen / DALLE; the location and number of signal heads may be architecture-specific.

## Related Work & Insights
- **vs Attend-and-Excite / A-STAR / Rassin et al.**: These methods add loss or optimization constraints to the cross-attention map, requiring backpropagation at each step and slowing inference; OSI only modifies keys during the forward pass, with negligible delay.
- **vs TACA / PLADIS**: TACA balances text-image token counts via LoRA, PLADIS uses sparse attention, both altering attention computation; OSI does not change the process, only specific key values, making it lighter.
- **vs Chen et al. (2024)**: They also analyze "concept confusion" in text embeddings but only at CLIP output (outside diffusion); OSI probes inside MM-DiT at the head level.
- **vs ITI (Inference-Time Intervention, Li et al. 2023a)**: This work directly adopts the mass mean shift idea and transfers it from LLMs to diffusion key vectors; the contribution is discovering that "intermediate layer + intermediate timestep" in MM-DiT is the signal-rich region.
- **Insights**: This paradigm of "probing semantic signals in internal representations → steering in the opposite direction at inference" is not only applicable to concept omission, but can also be extended to attribute binding (color leakage), spatial omission, negation understanding, long prompt consistency, and all controllability issues where "the model can sense but fails to execute."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically transfer the ITI / mass mean shift paradigm from LLMs to MM-DiT internal key vectors, and to discover the counterintuitive but effective "amplify absence signal" direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two MM-DiT backbones × 5 object densities × 3 attribute types × token-specific causal experiments × 4 hyperparameter ablations, covering a wide range.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth "diagnosis → intervention → verification" narrative, with intuitive dynamic visualizations of probe probabilities in Figs. 3-4; clear correspondence between formulas and experiments.
- Value: ⭐⭐⭐⭐⭐ Training-free, near-zero overhead, more than doubling FLUX 6-obj performance, with immediate value for industrial T2I deployment; paradigm extensible to other controllability problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](../../ICCV2025/image_generation/exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](../../AAAI2026/image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[ICML 2026\] Krause Synchronization Transformers](krause_synchronization_transformers.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)

</div>

<!-- RELATED:END -->
