---
title: >-
  [Paper Note] Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness
description: >-
  [ICML 2026][Multimodal VLM][semantic antibodies] This paper adapts the principle of "negative selection" from biological immune systems to VLMs such as CLIP. It utilizes an LLM to actively hallucinate a batch of "look-al…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "semantic antibodies"
  - "negative selection"
  - "vMF prototype"
  - "open-space risk"
  - "near-distribution OOD"
date: 2026-05-08
content_hash: 93a6392ed17449a3
---

# Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness

**Conference**: ICML 2026  
**arXiv**: [2605.30745](https://arxiv.org/abs/2605.30745)  
**Code**: No public repository provided  
**Area**: AI Safety / Open-World Recognition / OOD Detection / Vision-Language Models  
**Keywords**: semantic antibodies, negative selection, vMF prototype, open-space risk, near-distribution OOD  

## TL;DR
This paper adapts the principle of "negative selection" from biological immune systems to VLMs such as CLIP. It utilizes an LLM to actively hallucinate a batch of "look-alike but non-target" textual descriptions as semantic antibodies. A lightweight adapter is then used to push visual features away from these antibodies, significantly reducing "high-confidence misclassification" in open-world scenarios without retraining the backbone.

## Background & Motivation
**Background**: Large Vision-Language Models (VLMs) like CLIP, ALIGN, and LLaVA align visual features into a dense semantic manifold, achieving impressive zero-shot recognition capabilities and being widely deployed in open-world scenarios such as autonomous driving and medical diagnosis.

**Limitations of Prior Work**: The authors define the vulnerability of these models as "Hubris of Semantics" — when encountering out-of-distribution (OOD) samples, the model fails to say "I don't know" and instead forcefully maps them to the closest known class with extremely high confidence (e.g., classifying a "robot dog" as a "golden retriever").

**Key Challenge**: Traditional OOD defenses rely on discriminative thresholds (MSP, Energy, ASH, etc.) or reactive concept matching (MCM), which only perceive statistical bias after an error occurs. Conversely, GAN-based outlier generation methods synthesize anomalies in pixel space, which leads to combinatorial explosion and lack of scalability at the ImageNet scale.

**Goal**: To actively and densely characterize the boundaries of "known classes" on the semantic manifold without retraining the VLM backbone or relying on pixel-level outlier generation, thereby constraining open-space risk.

**Key Insight**: The biological immune system randomly generates T-cell receptors through "thymic negative selection" and eliminates all candidates that bind to "self," leaving behind a negative map of "non-self." The authors analogize this mechanism to VLMs: using an LLM as a "computational thymus" to generate "near-OOD" semantic descriptions as antibodies.

**Core Idea**: Use an LLM to actively hallucinate "semantic antibodies" in textual form to surround the semantic hyperspherical caps of known classes. Then, train a lightweight adapter using an adversarial push-pull loss to push visual embeddings away from antibody directions and pull them toward prototype directions.

## Method

### Overall Architecture
Immuno-VLM partitions "immunization" into a three-stage pipeline, with backbones $\phi_v, \phi_t$ frozen throughout:

1.  **Antigen Characterization (Phase 1)**: For each known class $k$, a prototype direction $\bm{\mu}_k$ of a vMF distribution is estimated. Visual means and text embeddings are interpolated using a spherical geodesic to mitigate the modality gap.
2.  **Antibody Generation (Phase 2)**: An LLM is used to generate two types of text antibodies for each class: hard semantic negatives (visually similar but different categories) and contextual anomalies (the same object appearing in impossible scenarios). A band-pass cosine similarity filter is then used to remove candidates that are too close or too far.
3.  **Vaccination (Phase 3)**: A residual adapter $f_\theta(\mathbf{z}) = \mathrm{Norm}(\mathbf{z}+\mathrm{MLP}(\mathbf{z}))$ is trained using a Pull term to pull samples toward $\bm{\mu}_k$ and a Push term to push samples away from any antibodies. During inference, a "Differential Immuno-Score" $S_{DIS}$ is provided, and EVT is used to fit a threshold for each class.

### Key Designs

1.  **Geodesic Antigen Alignment**:
    - **Function**: Defines the "self-center" of each known class on the hypersphere while fusing visual evidence with semantic priors.
    - **Mechanism**: Maximizes $\sum_{\mathbf{v}\in\mathcal{V}_k} \kappa_k \bm{\mu}_k^\top \mathbf{v}$ under the constraint $\arccos(\bm{\mu}_k^\top \mathbf{t}_k) \le \xi$. The Lagrange derivation yields the optimal solution $\bm{\mu}_k^* \propto (1-\alpha)\bar{\mathbf{v}}_k + \alpha \mathbf{t}_k$, which is a point on the line segment connecting the visual mean and text embedding, subsequently normalized back to the unit sphere.
    - **Design Motivation**: Pure MLE can be biased by visual shortcuts (e.g., "dogs are always on grass"), while pure text lacks intra-class nuances. Geodesic interpolation provides a provably optimal compromise, mitigating the CLIP modality gap.

2.  **LLM as Computational Thymus for Antibody Generation**:
    - **Function**: Samples "non-self" on the semantic manifold rather than in pixel space to avoid degradation of random sampling on higher-order hyperspheres.
    - **Mechanism**: The authors prove Theorem 3.5—the cosine similarity of a uniformly sampled vector on a unit hypersphere with any prototype decays exponentially to 0 as $2\exp(-d\epsilon^2/2)$. This implies that at $d=512$, random negative samples are almost entirely orthogonal and contribute nothing to boundary tightening. Instead, the LLM conditionally generates two types of antibodies: hard negatives $\mathcal{A}_{hard}(y)$ (e.g., wolf for husky, malamute, wolf-dog hybrid) and contextual anomalies $\mathcal{A}_{context}(y)$ (e.g., car underwater, flying car). A band-pass filter with Safety-Utility conditions $\delta_{safe} < \langle \phi_t(a), \bm{\mu}_k\rangle < \delta_{risk}$ prevents antibodies from degrading into synonyms or pure noise.
    - **Design Motivation**: Finding hard negatives purely by "randomness" in high-dimensional space is destined to fail; delegating antibody generation to an LLM, which is naturally dense in semantic space, reduces the curse of dimensionality to a "language generation diversity" problem.

3.  **Push-Pull Vaccination Loss and Differential Immuno-Score**:
    - **Function**: Locally warps the visual embedding space to create a "sterile zone" without moving the backbone, utilizing both "near-self" and "far-from-non-self" signals during inference.
    - **Mechanism**: The training loss is defined as $\mathcal{L}_{vac} = \mathcal{L}_{pull} + \lambda \mathcal{L}_{push} + \eta\|\theta\|_2^2$, where Pull is a classification softmax based on vMF likelihood, and Push is a hinge form $\max(0, \cos(f_\theta(\phi_v(x)), \phi_t(a))-m)^2$ that forces a margin $m$ (e.g., 0.2) between visual samples and antibodies. During inference, $S_{DIS}(x) = s^+(\mathbf{z}) - \lambda_{inf}\cdot s^-(\mathbf{z})$ is defined, where $s^+$ is the similarity to the nearest prototype and $s^-$ is the similarity to the most dangerous antibody. A Weibull distribution is used to fit the score tails for each class to obtain adaptive thresholds $\tau_{evt}$.
    - **Design Motivation**: Fixed thresholds cannot adapt to density differences (e.g., "dog classes have many neighbors, aircraft carrier classes have few"). EVT provides an independent tail model for each class, corresponding to the generalization bound of "$\delta$-cover + Lipschitz adapter" in Theorem 3.7.

### Theoretical Highlights
- **Antibody Coverage Bound (Theorem 3.3)**: If the antibody set $\mathcal{A}_\delta$ is a $\delta$-cover of the known class semantic boundary and the prototype maintains a margin $m > \epsilon_{align} + \delta$ from antibodies, the FPR is strictly bounded by a combination of $\epsilon_{align}$ and $\delta$. This provides a formal explanation of why dense antibodies can replace real OOD samples.
- **Open-Space Generalization Bound (Theorem 3.7)**: Decomposes the true open-space risk $R_{\mathcal{O}}$ into empirical risk on antibodies, Rademacher complexity $\mathfrak{R}_N(\mathcal{H})$, a coverage error of $\mathcal{O}(L\delta/\sqrt{M})$, and a first-order concentration term. It explicitly states that "denser antibodies, more antibodies, and a simpler adapter" lead to higher safety.

## Key Experimental Results

### Main Results
All methods are evaluated using the same CLIP-ViT-B/16 backbone on ImageNet-1K (ID) and three OOD benchmarks.

| Dataset | Metric | Ours (Immuno-VLM) | Prev. SOTA (MCM) | Gain |
|--------|------|------|----------|------|
| In-Distribution | ID-Acc ↑ | ~78.2 | 78.2 | No drop |
| ImageNet-O (Near-OOD) | AUROC ↑ | Significant > 74.5 | 74.5 | 16%+ Semantic Gain |
| iNaturalist (Fine-grained OOD) | FPR95 ↓ | Better than 42.1 | 42.1 | Significant decrease |
| Texture (Far-OOD) | AUROC ↑ | Better than 83.4 | 83.4 | Consistent lead |

> The paper explicitly states that the model achieves SOTA on multiple challenging benchmarks, with a 16%+ improvement in semantic shift detection over the zero-shot baseline while maintaining ID accuracy.

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| Full (Pull+Push+vMF+EVT) | Best AUROC | Complete framework |
| w/o Push step | Close to Energy baseline | Antibodies not used in training; adapter degrades to contrastive fine-tuning |
| w/o vMF / Using visual mean | ID & OOD drop | Lack of semantic prior; prototype biased by visual noise |
| w/o EVT adaptive threshold | FPR95 increases | Global threshold fails to adapt to class density differences |
| Antibodies as uniform noise | Degradation to MCM | Confirms Theorem 3.5's prediction of degradation |

### Key Findings
- Shifting the source of negative samples from pixel space to semantic space is the primary cause of the performance jump; using random noise for antibodies causes immediate degradation.
- Text-regularized vMF prototypes improve both ID and OOD performance, suggesting the modality gap is amplified by "prototype shift."
- EVT adaptive thresholds benefit fine-grained OOD (iNaturalist) the most, supporting the theoretical assertion that "semantic density varies by class."

## Highlights & Insights
- By explicitly modeling the "curse of dimensionality" as Theorem 3.5, the conclusion that "sampling must occur on the semantic manifold" becomes clear, showing a rare and strong causal link between theory and method.
- The "LLM = Computational Thymus" metaphor is not only easy to communicate but naturally explains why LLMs outperform GANs in generating hard negatives: their output already exists within the VLM's shared semantic space.
- Defining the inference score as $s^+ - \lambda s^-$ instead of only $s^+$ explicitly utilizes the "non-self" knowledge acquired during training, offering a new paradigm for future OOD scoring.

## Limitations & Future Work
- Antibody quality is entirely dependent on the LLM; the authors admit the system's safety is upper-bounded by the LLM's "imagination" (Wasserstein alignment).
- The $\delta_{safe}, \delta_{risk}$ in the band-pass filter and the $\tau_{evt}$ in EVT are empirical settings; tuning costs can be high when the number of classes is very large.
- The paper does not provide public code or antibody generation prompt templates, making reproduction difficult.
- Antibody generation is an offline, one-time step; if ID categories expand dynamically, the three-stage pipeline must be re-executed.

## Related Work & Insights
- **vs MCM (Ming et al., 2022)**: MCM only uses class-name text embeddings for maximum concept matching and is reactive; Immuno-VLM actively generates negative textual concepts to formalize boundaries.
- **vs Classical AIS / Negative Selection**: Classical AIS generates detectors in pixel or random bit space, which Theorem 3.5 proves non-scalable; this work migrates detectors to the semantic space.
- **vs Energy / ASH etc. (Discriminative OOD)**: These methods only look at activation magnitudes without explicitly using "non-self" information; the $S_{DIS}$ in this paper merges discriminative and generative lines of thought.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Adapting immunological negative selection to VLMs with a clear mathematical mapping is unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on ImageNet-1K and three OOD benchmarks with complete ablations; would be even stronger with ViT-L/14 and LLaVA-style MLLMs.
- **Writing Quality**: ⭐⭐⭐⭐ Vivid analogies with theorem-method correspondence; however, related work citations are a bit too dense, affecting readability.
- **Value**: ⭐⭐⭐⭐ Provides a practical path for "open-world VLM safety" without moving the backbone using an LLM as a thymus; high industrial deployability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](../../ACL2026/multimodal_vlm/geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](../../ICCV2025/multimodal_vlm/on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICML 2026\] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings](timespot_benchmarking_geo-temporal_understanding_in_vision-language_models_in_re.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)

</div>

<!-- RELATED:END -->
