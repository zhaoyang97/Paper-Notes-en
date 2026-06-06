---
title: >-
  [Paper Note] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers
description: >-
  [ICML 2026][Image Generation][MM-DiT] The paper utilizes linear probes to discover that in MM-DiT (FLUX / SD3.5), the key vectors of text tokens in certain middle-layer attention heads naturally encode a binary signal of…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "MM-DiT"
  - "Concept Omission"
  - "Linear Probes"
  - "Attention Key Intervention"
  - "Training-free Guidance"
date: 2026-05-08
content_hash: 0f2a5cc7cebab7f2
---

# Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers

**Conference**: ICML 2026  
**arXiv**: [2605.14270](https://arxiv.org/abs/2605.14270)  
**Code**: No public repository link provided in the paper  
**Area**: Diffusion Models / Text-to-Image Generation / Representation Intervention  
**Keywords**: MM-DiT, Concept Omission, Linear Probes, Attention Key Intervention, Training-free Guidance

## TL;DR
The paper utilizes linear probes to discover that in MM-DiT (FLUX / SD3.5), the key vectors of text tokens in certain middle-layer attention heads naturally encode a binary signal of "whether the target concept will appear." Based on this, it proposes Omission Signal Intervention (OSI): during inference, the mean difference direction of "absent class - present class" is injected into the key vectors of the Top-K heads with an intensity of $\alpha\sigma\boldsymbol{\theta}$. This stimulates the model's "self-awareness" of missing concepts to complete the generation. On FLUX, GenEval 6-object accuracy improves from 0.18 to 0.40 without any fine-tuning.

## Background & Motivation
**Background**: T2I diffusion models have largely transitioned to the MM-DiT architecture (FLUX, SD3, SD3.5), which concatenates text tokens and image patch tokens into a single sequence for joint attention. This is more suitable for learning cross-modal semantic alignment compared to the unidirectional injection of U-Net + cross-attention.

**Limitations of Prior Work**: Despite architectural progress, concept omission (e.g., failing to generate a "book" when prompted for "a cat, a dog, and a book") remains a persistent issue in MM-DiT. Existing mitigation strategies either add optimization constraints to visual attention maps (Attend-and-Excite, A-STAR, Rassin et al.) or require additional training (GLIGEN, reward fine-tuning), both of which entail high inference costs or disrupt the original model distribution.

**Key Challenge**: Existing research focuses primarily on visual embeddings; how text embeddings encode the "whether a concept will appear" signal **internally** within the diffusion model remains a black box. While Chen et al. (2024) analyzed CLIP text outputs, they did not track this to the level of diffusion-internal attention heads. In other words, the model might "know" what it has missed, but tools to query this state are lacking.

**Goal**: (1) Use probing tools to examine if internal text tokens in MM-DiT carry "omission state" information; (2) If so, identify the specific layers, heads, and timesteps carrying this signal; (3) Use minimal-cost inference-time intervention to amplify this signal and force the model to generate the missing concepts.

**Key Insight**: The joint attention of MM-DiT allows text tokens to receive visual feedback from image tokens at every layer. Consequently, the key/value of a text token is no longer a "static text description" but dynamically reflects what has already been generated in the current latent. In middle layers (where image tokens carry semantics but generation is incomplete), the text token's key vector may encode "has my corresponding concept appeared in the noisy latent."

**Core Idea**: First, a linear probe is used to explicitly extract this "omission signal" from the attention keys (the mean difference direction of "absent $\rightarrow$ present"). Then, a mass mean shift is applied during inference to shift the key in the **direction of the absent class**, effectively making the model "think it has missed more," thereby triggering it to actively remedy the missing concept.

## Method

### Overall Architecture
The method consists of two phases. **Diagnose**: Generations are run on the GenEval two-object subset using FLUX.1-Dev. Each image is annotated via Mask2Former + BLIP-VQA to determine if each concept token was successfully generated ($y\in\{0,1\}$). Text token key vectors $\mathbf{k}_c^{(t,l,h)}$ from intermediate timesteps are collected to form a dataset $\mathcal{D}_{l,h}^{\text{train}}$. A linear probe is trained for each $(l, h)$ to distinguish between absent and present states. **Correct (OSI)**: The mass mean shift direction $\boldsymbol{\delta}^{(l,h)} = \mathbb{E}[\mathbf{k}|y=0]-\mathbb{E}[\mathbf{k}|y=1]$ is calculated from the training set of each head and normalized to $\boldsymbol{\theta}^{(l,h)}$. During inference, for top-K heads with the highest accuracy during the early denoising stage ($t\in[t_{\text{stop}}, 1]$), the concept token key vector is shifted as $\mathbf{k}_c \leftarrow \mathbf{k}_c + \alpha\sigma^{(l,h)}\boldsymbol{\theta}^{(l,h)}$. Notably, adding the "absent - present" direction makes the model internalize that it has "missed more," triggering its built-in remedy mechanism.

### Key Designs

1.  **Concept-aware Probing Dataset Construction with Noise Filtering**:
    - **Function**: Collects clean (key vector, present-label) pairs from the FLUX generation process specifically for training linear probes to identify the omission signal.
    - **Mechanism**: Batch generations use the template "a photo of {obj1} and {obj2}," recording the text token key vector $\mathbf{k}^{(t,l,h)}$ at every step, layer, and head. Labels come from the final image, determined by dual annotation with Mask2Former (mmdetection) and BLIP-VQA; samples are only accepted if both agree to avoid single-annotator noise. Two traps are addressed: (a) **Early timestep label misalignment**—the target has not appeared in steps 1-3, but if it ultimately succeeds, it is labeled $y=1$. This creates conflict as the key encodes "not yet generated" while the label says "will generate." (b) **Late timestep signal weakening**—later stages focus on details, and the omission signal vanishes. Table 1 provides quantitative evidence: agreement between early steps and final labels is only 0.409, jumping to 0.965 at intermediate steps, and 1.000 at late steps. The paper only retains samples from the intermediate interval $\mathcal{T}$ for the probe training set.
    - **Design Motivation**: Probing is a sanity check; data must reflect "this token, at this moment, knows whether its concept will appear." Filtering data by generation dynamics embedding the diffusion model's temporal structure into the probe design.

2.  **Head-Localization via Linear Probing**:
    - **Function**: Accurately locates "specialist heads" encoding the omission signal within a massive search space of 1368 attention heads across multiple timesteps.
    - **Mechanism**: A linear classifier is trained for each $(l, h)$ taking the text token key vector as input to predict absent vs. present. Fig. 2(a) shows probe accuracy peaking at intermediate timesteps (aligned with Table 1). The head-wise heatmap in Fig. 2(b) reveals a counter-intuitive pattern: **early layers stay near chance** (as text tokens haven't absorbed visual feedback), **middle layers rise significantly** (up to 91.0% accuracy), and **late layers drop again**. The top-300 heads (22% of the 1368 total, all exceeding 80% accuracy) are selected as "omission signal heads." Fig. 3 and Fig. 4 use box plots to show probe outputs evolving over timesteps: as visual objects gradually appear in the predicted $\hat{x}_0$ (first row), the "present" probability from the probe rises synchronously (second row).
    - **Design Motivation**: Signals are sparse—not all heads encode omission. Head-wise probing ensures intervention is only performed on useful heads, avoiding degradation of other capabilities.

3.  **Mass Mean Shift Intervention (OSI)**:
    - **Function**: Injects the "absent $\rightarrow$ present" direction into key vectors during inference in a training-free manner to stimulate the completion of missing concepts.
    - **Mechanism**: For each top-K head, the mass mean shift direction $\boldsymbol{\delta}^{(l,h)} = \mathbb{E}[\mathbf{k}^{(t,l,h)}|y=0] - \mathbb{E}[\mathbf{k}^{(t,l,h)}|y=1]$ is computed and normalized into $\boldsymbol{\theta}^{(l,h)}$. During inference, the modification $\mathbf{k}_c \leftarrow \mathbf{k}_c + \alpha\sigma^{(l,h)}\boldsymbol{\theta}^{(l,h)}$ is applied, where $\sigma^{(l,h)}$ is the standard deviation of probe data in that direction (adaptive scale normalization) and $\alpha$ is a scalar (5.0 for FLUX, 7.5 for SD3.5). This is active during the first 15 steps (out of 30) within $t\in[0.78, 1]$. Adding the "omission direction" is explained as making the model perceive the omission as more severe, triggering a stronger correction impulse—an intentional hallucination of severity.
    - **Design Motivation**: Unlike traditional methods that modify attention maps or fine-tune, OSI uses the mass mean shift + steering vector paradigm for training-free surgical intervention. The choice to amplify the "absent" signal instead of the "present" signal is validated by Table 4, where the opposite direction ($-\boldsymbol{\theta}$) collapses performance.

### Loss & Training
OSI is training-free: the model itself is not modified; only key vectors are altered during inference. Only the 1368 lightweight linear probes (≪ main model) are trained using BCE loss. Sampling: 30 steps total, CFG=3.5 (FLUX)/7.0 (SD3.5), top-K=300 (FLUX)/100 (SD3.5), $\alpha=5.0/7.5$, $t_{\text{stop}}=0.78/0.76$, active for the first 15 steps. Token selection: GenEval uses template rules; T2I-CompBench uses Llama-3.1-8B to extract target spans.

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

The most significant gain is in the 6-object scenario, doubling from 0.18 to 0.40, demonstrating OSI's power in difficult scenarios where the baseline collapses.

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
|--------|------------------------|---------------------------|
| FLUX (no OSI) | 0.00 | 0.292 |
| OSI on omitted token | **0.70** | **0.658** |
| OSI on present token | 0.14 | 0.298 |

Intervention duration ($t_{\text{stop}}$):

| Step | 0 | 5 | 10 | **15 (Ours)** | 20 | 25 | 30 |
|------|---|---|----|---------------|----|----|---|
| Accuracy | 0.82 | 0.88 | 0.91 | **0.92** | 0.92 | 0.92 | 0.91 |
| MANIQA | 0.473 | 0.479 | 0.480 | **0.480** | 0.481 | 0.480 | 0.480 |

### Key Findings
- **Direction is Everything**: Reverse intervention collapses 6-obj performance from 0.18 to 0.02, indicating that the mass mean shift direction is not arbitrary—it precisely corresponds to the semantic axis of concept realization.
- **Surgical Intervention on Top Heads is Mandatory**: Intervention on bottom-K heads is ineffective, and "all heads" (1368) is outperformed by top-K. This validates the value of probe-based head selection.
- **Token-Specific Effects**: Applying intervention to "already present" tokens yields almost no gain, proving OSI is not a universal quality enhancer but a targeted treatment for concept omission signals.
- **Early Gains matter**: Intervention at $t_{\text{stop}}=0.95$ (first 5 steps) already achieves 0.88 accuracy, saturating at 0.92 by step 15. This aligns with training theories where perception is prioritized in early denoising.
- **Generalization from Objects to Attributes**: Probes trained only on object-level labels enable OSI to improve color, shape, and texture binding, suggesting the omission signal is a general semantic realization concept.

## Highlights & Insights
- **"The model knows what it missed" is a profound discovery**: Quantifying this internal awareness with a 91% accurate probe and amplifying it into a steerable signal completes a "Diagnosis -> Intervention -> Verification" pipeline.
- **Training-free with minimal computation**: Modifying only key vectors in top-K heads for 15 steps introduces negligible overhead, representing a major engineering leap over fine-tuning.
- **Amplifying the "Absent" signal is counter-intuitive but effective**: Instead of making the model believe it has already drawn the object, the design makes the model believe it has missed more, triggering its internal self-correction loop.
- **A General Formula for Diffusion Steering**: The methodology (identify signal heads -> compute mean difference -> steer keys) can be transferred to style control, composition, or concept fusion. 
- **Temporal Data Filtering**: Distinguishing data availability between early, middle, and late timesteps (0.409/0.965/1.000 agreement) provides a masterclass in embedding diffusion dynamics into representation analysis.

## Limitations & Future Work
- **Over-generation Side Effects** (Fig. 7): OSI can sometimes generate too many instances of an object (e.g., getting three when asking for one). This requires tuning $\alpha$ and $K$ due to a lack of a closed-loop stopping criterion.
- **Binary Label Dependency**: Hard to apply to ambiguous or subjective concepts (aesthetic style, relative quantity "a few") where signals cannot be easily binarized.
- **Empirical Hyperparameters**: Values for top-K and $\alpha$ are determined via grid search and may need re-tuning for different backbones.
- **High-Density Reliability**: Probe reliability in scenes with 4-6 objects hasn't been independently evaluated.
- **Static Time Window**: The window $[t_{\text{stop}}, 1]$ is static; monitoring real-time probe probabilities to dynamically adjust intervention duration could further improve results.

## Related Work & Insights
- **vs. Attend-and-Excite / A-STAR**: These methods add losses or optimization constraints to cross-attention maps, slowing inference due to backpropagation at each step. OSI operates on the forward pass with negligible latency.
- **vs. TACA / PLADIS**: TACA balances text-image token counts via LoRA; PLADIS uses sparse attention. Both change attention computation, whereas OSI is a lighter modification of key values.
- **vs. Chen et al. (2024)**: They analyzed concept confusion in CLIP outputs; OSI goes deeper into the internal head levels of MM-DiT.
- **vs. ITI (Li et al. 2023a)**: OSI adapts mass mean shift from LLMs to diffusion key vectors, discovering that middle layers and timesteps are the signal-rich zones for MM-DiT.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematically transfers ITI/mass mean shift to MM-DiT keys and discovers the effective counter-intuitive "absent signal" direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across backbones, object densities, attributes, and causality experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear narrative flow from diagnosis to verification; excellent dynamic visualization of probe probabilities.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, zero-overhead, significant gains (doubled 6-obj accuracy on FLUX). Highly practical for industrial T2I deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](orthogonal_concept_erasure_for_diffusion_models.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](../../ICCV2025/image_generation/exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](../../AAAI2026/image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[ICML 2026\] Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)

</div>

<!-- RELATED:END -->
