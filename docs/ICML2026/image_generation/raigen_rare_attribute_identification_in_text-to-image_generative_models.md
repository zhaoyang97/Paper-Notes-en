---
title: >-
  [Paper Note] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models
description: >-
  [ICML 2026][Image Generation][Diffusion Models] RAIGen utilizes Matryoshka Sparse Autoencoders to decompose T2I diffusion model bottleneck representations into interpretable neurons. By applying a combined score of "acti…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Sparse Autoencoders"
  - "Minority Attribute Discovery"
  - "Bias Auditing"
  - "T2I Generation"
date: 2026-05-08
content_hash: a64bf9ba2ebd8f70
---

# RAIGen: Rare Attribute Identification in Text-to-Image Generative Models

**Conference**: ICML 2026  
**arXiv**: [2602.06806](https://arxiv.org/abs/2602.06806)  
**Code**: https://vssilpa.github.io/RAIGen_webpage/ (Project Page)  
**Area**: Diffusion Models / Image Generation / Model Interpretability / Bias Auditing  
**Keywords**: Diffusion Models, Sparse Autoencoders, Minority Attribute Discovery, Bias Auditing, T2I Generation

## TL;DR
RAIGen utilizes Matryoshka Sparse Autoencoders to decompose T2I diffusion model bottleneck representations into interpretable neurons. By applying a combined score of "activation rarity × CLIP semantic distinctiveness," it identifies minority attribute neurons that are "internally encoded by the model but rarely appear during generation," thereby extending bias auditing from "known fairness categories" and "significant majority patterns" to label-free rare attribute discovery.

## Background & Motivation

**Background**: T2I diffusion models (Stable Diffusion, SDXL, FLUX, etc.) can generate high-fidelity images but simultaneously inherit and amplify attribute biases from training data. Existing debiasing methods follow two paths: closed-set methods (e.g., fair diffusion based on gender/race, classifier-free guidance, learnable projection modules) can only handle predefined categories; open-set methods (e.g., OpenBias) leverage external LLMs to propose candidate attributes and VQA for voting to discover unknown biases, but they primarily expose **majority attributes** (patterns that over-appear in generation).

**Limitations of Prior Work**: Closed-set methods rely on manual categories and fail to cover non-fairness-oriented minority patterns such as "rare attire/cultural symbols/compositional patterns." Open-set methods treat the model as a black box, using external world models for reverse inference, which reveals "what is generated frequently" rather than "what is suppressed." The authors empirically demonstrate in Appendix G.1 that merely suppressing majority attributes does not uniformly redistribute probability mass to minority attributes but instead redistributes it unevenly among minority groups.

**Key Challenge**: The presence of minority attributes cannot be inferred solely from the output. A model can **internally encode** a concept yet rarely output it during sampling (e.g., in LAION, "teacher" is nearly gender-balanced, but SD output is heavily male-biased). To find these "suppressed minority attributes," one must **examine internal model representations** rather than just observing generated images.

**Goal**: (1) Propose a label-free framework that extracts "encoded but systematically under-expressed" attributes directly from internal diffusion model representations without any minority category priors; (2) provide a quantitative minority score for each candidate neuron; (3) verify that these attributes can be targeted and amplified during generation.

**Key Insight**: Diffusion bottleneck representations are naturally entangled and unreadable. Recently, Matryoshka Sparse Autoencoders (MSAE) demonstrated hierarchical, interpretable concept decomposition capabilities on CLIP. The authors train an MSAE on the diffusion bottleneck and utilize only the **coarsest granularity layer** (finer levels tend to fragment single concepts into part-features, unnecessarily increasing the search space).

**Core Idea**: After decomposing representations into sparse neurons using MSAE, a "minority attribute neuron" should satisfy two conditions: **low activation frequency** (rarely fires) and **top-activating images significantly deviating from the overall semantic center in CLIP space** (distinct). The Minority Score is the product of both: $s(\mathbf{z}) = \mathbf{d} \odot (\mathbf{1} - \boldsymbol{\nu})$.

## Method

### Overall Architecture
Given a T2I diffusion model $G$ and prompt $\mathbf{c}$: (1) Generate $N$ images using $G$, extracting the bottleneck representation $\mathbf{h} \in \mathbb{R}^{h \times w \times n}$ at the last denoising step, treating each spatial position as an $n$-dimensional sample to form dataset $\mathcal{D}_c = \{(\mathbf{h}^{(j)}, \mathbf{x}^{(j)})\}$; (2) train an MSAE on these bottleneck vectors to obtain hierarchical sparse codes $\mathbf{z}^{(k_i)}$, using only the coarsest granularity $k_1$; (3) compute the Minority Score for each neuron $z_i$, and remove redundancy based on CLIP center distances; (4) use top-activating images and activation heatmaps for manual or MLLM (GPT-5.2) labeling to explain neuron semantics; (5) insert labels back into prompts for lightweight rewriting to verify amplification of rare attributes.

### Key Designs

1.  **MSAE on diffusion bottleneck (coarsest layer only)**:
    - **Function**: Disentangle the diffusion bottleneck $\mathbf{h}$ into sparse interpretable neurons $\mathbf{z} = \{z_1, \dots, z_d\}$.
    - **Mechanism**: MSAE trains a single encoder/decoder using multiple Top-$k$ operators $\{k_1 < k_2 < \dots < k_f = d\}$, with a loss function consisting of a weighted sum of reconstruction errors across layers: $\mathcal{L}_{\text{MSAE}} = \sum_i \alpha_i \|\mathbf{r} - \hat{\mathbf{r}}^{(k_i)}\|_2^2$. Any layer can be selected during inference. The authors specifically **use $k_1$**, as finer layers fragment "female doctor" into part-features like "white coat sleeves + curly hair + stethoscope," which increases search space and generates spurious rare neurons.
    - **Design Motivation**: Hierarchical sparsity makes "concept granularity" a controllable knob. Choosing the coarsest layer is an engineering trade-off of "interpretability over coverage"—preferring to miss some finer minority attributes to ensure the remaining neurons are stable, human-readable semantic units.

2.  **Minority Score = rarity × distinctiveness**:
    - **Function**: Assign a $[0, 1]$ score to each neuron $z_i$, where high scores indicate "minority attribute candidates."
    - **Mechanism**: (a) **Activation frequency** $\nu_i = |\{(\mathbf{h}, \mathbf{x}) \in \mathcal{D}_c : z_i(\mathbf{h}) > 0\}| / |\mathcal{D}_c|$, where lower is rarer; (b) **Semantic distinctiveness** $d_i$, defined as the cosine distance between the neuron's activation-weighted CLIP center $\mu_i = \sum z_i(\mathbf{h}) \cdot \text{CLIP}(\mathbf{x}) / \sum z_i(\mathbf{h})$ and the global CLIP center $\mu_{\mathcal{D}_c}$; (c) both are min-max normalized to $[0, 1]$ and element-wise multiplied: $s(\mathbf{z}) = \mathbf{d} \odot (\mathbf{1} - \boldsymbol{\nu})$. Toy experiments show frequency alone has a Spearman correlation of $\rho \approx 0.991$ with ground-truth rare features, but real-world scenarios contain noise, necessitating distinctiveness as a second-level filter.
    - **Design Motivation**: Decoupling "rarity" and "semantic uniqueness" into two observable measurements is necessary because neither is sufficient alone—low $\nu$ might be a noise neuron, while high $d$ might be a high-frequency but skewed semantic cluster. Only satisfying both defines a true "internally encoded, externally suppressed" minority attribute.

3.  **Neuron de-redundancy based on CLIP center distance**:
    - **Function**: Merge "high-scoring but semantically redundant" neurons to obtain a set of distinct minority attributes.
    - **Mechanism**: Iterate through neurons in descending order of Minority Score. For each retained neuron, prune all other neurons whose centroids $\mu_i$ have a cosine distance less than the threshold $\tau$. $\tau$ serves as a hyperparameter to control semantic redundancy.
    - **Design Motivation**: MSAE often distributes the same minority concept (e.g., "curly-haired female doctor") across multiple high-scoring neurons. Greedy NMS using centroid distance ensures a diverse final set without needing to preset the number of attributes.

### Loss & Training
The MSAE training objective is $\mathcal{L}_{\text{MSAE}} = \sum_{i=1}^{f} \alpha_i \|\mathbf{r} - \hat{\mathbf{r}}^{(k_i)}\|_2^2$. The Minority Score calculation involves no gradients. Bottleneck representations are extracted only at the last denoising step $t = T_{\text{final}}$ (where semantic information is most complete). On FLUX.1-schnell, the hook point is at `transformer.transformer_blocks.18` with 4-step sampling.

## Key Experimental Results

### Main Results

**Main Results (Attribute Presence, lower is rarer, compared to OpenBias majority attributes)**:

| Model | Method | WinoBias (↓) | COCO (↓) |
|------|------|--------------|----------|
| SD v1.4 | OpenBias (Majority) | 0.941 | 0.933 |
| SD v1.4 | **Ours (Minority)** | **0.205** | **0.220** |
| SDXL | OpenBias (Majority) | 0.941 | 0.933 |
| SDXL | **Ours (Minority)** | **0.194** | **0.199** |

Attributes identified by RAIGen appear with only $\sim 20\%$ frequency, whereas OpenBias majority attributes appear at $\sim 94\%$, indicating RAIGen extracts suppressed rare patterns rather than salient majorities. Rarity on SDXL is slightly lower than SD v1.4, suggesting that greater model capacity does not automatically equate to higher rare pattern coverage.

**Amplification via prompt revision (WinoBias)**:

| Model | Prompt | NLL (↑) | Dev. ratio (↓) | CLIP Align. (↑) |
|------|--------|---------|----------------|-----------------|
| SD v1.4 | Base | 1.917 | 0.50 | 20.30 |
| SD v1.4 | Ours-Revised | 1.935 | **0.22** | 19.80 |
| SDXL | Base | 1.812 | 0.49 | 27.26 |
| SDXL | Ours-Revised | 1.852 | **0.23** | 26.89 |

Injecting minority attribute labels discovered by RAIGen into prompts via Llama 4-Scout reduces distribution deviation from $\sim 0.5$ to $\sim 0.22$ (closer to uniform). NLL increases slightly (moving into low-density regions), while CLIP alignment drops only $\sim 0.5$, maintaining semantic integrity.

**User Study (25 participants, 5 professions, Top-6 minority attributes per profession, estimated occurrences out of 10)**:

| Profession | Avg. Occurrence (↓) | 95% CI |
|------|----------------|--------|
| Analyst | 1.35 | [1.03, 1.67] |
| **CEO** | **0.70** | [0.44, 0.96] |
| Doctor | 1.18 | [0.97, 1.39] |
| Salesperson | 1.45 | [0.99, 1.91] |
| Sheriff | 2.64 | [2.21, 3.07] |

Across all professions, RAIGen attribute occurrences are $< 3/10$, with CEO being the rarest ($0.70/10$), confirming humans perceive these as "internally encoded but rare in generation."

### Key Findings
- Frequency alone is sufficient in toy settings (Spearman $\rho \approx 0.991$) but must be paired with distinctiveness in real SD representations. Appendix G.10 shows removing either significantly drops rare neuron recall.
- Restricting to the MSAE coarsest layer is a critical engineering choice; finer layers generate fragmented "part-features," which the authors explicitly avoid.
- The framework is architecture-agnostic, working on U-Net (SD 1.4/2/XL) and transformer-based DiT (FLUX.1-schnell). FLUX shows high-score but weakly interpretable neuron ratios, attributed to the lack of explicit spatial alignment in transformer hook points compared to U-Net bottlenecks.
- Attributes revealed by RAIGen extend beyond fairness categories, including stylistic/compositional rare patterns like "doctor portrait in a frame" or "side-view train with motion blur."

## Highlights & Insights
- **Defining "what is not generated" as an independent task**: While prior fairness methods focus on "debiasing known categories" and OpenBias on "discovering unknown majority attributes," this work establishes "unknown minority attribute discovery" as a third niche with a label-free solution.
- **Clean rarity × distinctiveness design**: Rather than complexifying the frequency definition to handle noise, the authors introduce the orthogonal CLIP-centroid distance for filtering. This minimal design covers both "rarity" and "significance" necessary conditions.
- **Counter-intuitive but correct coarsest-layer trade-off**: Although MSAE's selling point is multi-granularity, the authors prove that "finer is not better" for minority attributes, as it pushes up false positives through concept fragmentation.
- **Discovery → Amplification closed loop**: Beyond discovery, using LLMs to re-inject labels into prompts to reduce distribution deviation by half validates that the identified attributes are actionable, positioning RAIGen as both an auditing tool and a mitigation precursor.

## Limitations & Future Work
- RAIGen only finds minority attributes the model has **already encoded**; social minorities that the model failed to learn entirely will still be missed.
- The Minority Score is asymmetric: "high score = minority" holds, but "low score = majority" does not necessarily hold (could be noise).
- The pipeline heavily relies on CLIP as a semantic prior; CLIP's own biases (e.g., weak encoding of certain ethnicities) could contaminate distinctiveness assessments.
- Hook point selection on Transformer diffusion (FLUX) lacks systematicity, currently fixed at `transformer_blocks.18`.

## Related Work & Insights
- **vs OpenBias (D'Incà et al., CVPR 2024)**: OpenBias uses LLMs for candidates and VQA for voting to find **majority** attributes; RAIGen looks inside the model for **minority** ones. They are complementary.
- **vs DiffLens / SAeUron (Cywiński & Deja, ICML 2025)**: These use SAEs to intervene on **predefined** sensitive attributes or unlearning; RAIGen is upstream discovery.
- **vs Fair Diffusion / Debiased Prompts**: Traditional methods require manual gender/race categories; RAIGen skips category priors to find non-fairness-oriented patterns (composition, culture).
- **vs Matryoshka SAE for CLIP (Pach et al., 2025)**: Borrows the Matryoshka concept but migrates it to the diffusion bottleneck and challenges the "more granularity = more gain" assumption.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes "label-free rare attribute discovery" as a task with an elegant scoring function.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered SD 1.4/2/XL and FLUX, but dataset scale and baseline quantity are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions, consistent notation, and a logical progression from toy to human studies.
- Value: ⭐⭐⭐⭐ Provides a complementary "eye" to OpenBias for T2I auditing with a proven discovery-mitigation loop.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Content-Style Identification via Differential Independence](content-style_identification_via_differential_independence.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](../../ICCV2025/image_generation/ale_attribute_leakage_free_editing.md)
- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)

</div>

<!-- RELATED:END -->
