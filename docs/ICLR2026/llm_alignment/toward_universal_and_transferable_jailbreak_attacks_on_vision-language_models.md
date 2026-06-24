---
title: >-
  [Paper Note] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)
description: >-
  [ICLR 2026][LLM Alignment][VLM Jailbreak] The authors propose UltraBreak, which utilizes semantic adversarial objectives (replacing cross-entropy with cosine similarity to optimize for a smooth loss landscape) combined with input space constraints (random transformations + TV regularization to generate transform-invariant features). By training a single universal adversarial image, jailbreaks can be achieved across more than 6 VLM architectures and commercial models…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "VLM Jailbreak"
  - "Adversarial Attack"
  - "Universal Adversarial Image"
  - "Semantic Loss"
  - "Transferable Attack"
date: 2026-05-08
content_hash: dde8faa43ea7d24b
---

# Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)

**Conference**: ICLR 2026  
**arXiv**: [2602.01025](https://arxiv.org/abs/2602.01025)  
**Code**: Available (GitHub)  
**Area**: LLM Alignment  
**Keywords**: VLM Jailbreak, Adversarial Attack, Universal Adversarial Image, Semantic Loss, Transferable Attack

## TL;DR
The authors propose UltraBreak, which utilizes semantic adversarial objectives (replacing cross-entropy with cosine similarity to optimize for a smooth loss landscape) combined with input space constraints (random transformations + TV regularization to generate transform-invariant features). By training a single universal adversarial image, jailbreaks can be achieved across more than 6 VLM architectures and commercial models, reaching a black-box average ASR of 71% (SafeBench), significantly surpassing prior methods.

## Background & Motivation

**Background**: VLM jailbreak attacks are categorized into handcrafted designs (e.g., FigStep, which embeds harmful text into images) and gradient-based optimization (e.g., VAJM/UMK). While gradient methods theoretically produce universal triggers, they suffer from severe overfitting to single white-box proxy models in practice.

**Limitations of Prior Work**:  
   - Universality Issues: Existing gradient attacks are effective for single targets but fail to generalize across queries.
   - Transferability Issues: Adversarial images optimized for white-box proxies do not transfer well to black-box models.
   - Root Cause: Cross-entropy loss results in a "spiky" loss landscape, where optimal solutions found at sharp peaks have poor generalization.

**Key Challenge**: The goal is to use a single image to attack all queries across all models, yet current loss functions and optimization methods lead to significant overfitting.

**Goal**: Achieve both universality (one image across all harmful queries) and transferability (across model architectures) simultaneously.

**Key Insight**: The smoothness of the loss landscape determines generalization—replacing token-level cross-entropy with semantic-level cosine similarity.

**Core Idea**: Semantic loss to smooth the loss landscape + input transformations to generate invariant features = single-image universal cross-model jailbreak.

## Method

### Overall Architecture
UltraBreak aims to create a "universal image": trained once on an open-source VLM (the white-box proxy Qwen2-VL-7B) to be effective against any harmful query on any VLM (including closed-source commercial models). The optimization target is a single adversarial perturbation overlaid on a blank image, iteratively optimized using Adam for approximately 1300 steps on a small batch of harmful queries (SafeBench-Tiny, 5 per category, 50 total). Once converged, this image is independent of the proxy model and can be applied directly to other architectures. This universality and transferability are achieved by addressing the "proxy overfitting" issue through constraints on the image, loss, and text: first, applying random transformations and TV regularization at the input to force the perturbation to learn universal high-level structures; since these constraints can make the loss landscape jagged, the objective is switched from token-wise cross-entropy to semantic space similarity to re-smooth the landscape; finally, a target prompt template is applied on the text side to amplify the jailbreak effect.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}%%
flowchart TD
    A["Blank Image + Adversarial Perturbation<br/>Single Image to Optimize x"]
    subgraph C1["Input Space Constraints"]
        direction TB
        B1["Random Transformations<br/>Rotation, Scaling, Translation"] --> B2["Projection & Clipping<br/>CLIP Norm → Legal [0,1]"] --> B3["TV Regularization<br/>Suppress High-frequency Noise"]
    end
    Q["Harmful Query q"] --> QT["Target Prompt Guidance<br/>Template q-TPG"]
    A --> C1
    C1 --> M["White-box Proxy VLM<br/>Qwen2-VL-7B"]
    QT --> M
    M --> L["Semantic Adversarial Loss<br/>Cosine Similarity + Attention Weighting"]
    L -->|"Adam Update Perturbation<br/>Iterate ~1300 steps"| A
    L --> O["Universal Adversarial Image<br/>Text-like/Symbolic Structures"]
    O --> T["Transfer Attack<br/>Black-box / Commercial VLMs"]
```

### Key Designs

**1. Input Space Constraints: Forcing adversarial images to learn transform-invariant robust features instead of fragile pixel noise.**
If perturbations are allowed to optimize freely, they encode information into specific pixel locations—a low-level noise pattern that only works on the current proxy. UltraBreak applies three constraints during optimization: (a) Random transformations (rotation -15°~15°, scaling 0.8~1.2, translation 0~112 pixels) to prevent the perturbation from binding to fixed pixels; (b) Input projection to normalize images via CLIP means/stds and clip them to $[0,1]$; (c) TV regularization $\mathcal{L}_{\text{TV}}$ to enforce spatial smoothness. Visualization shows that without constraints, the result is pure noise; with random transformations, text-like patterns emerge; and with TV, these become smooth, coherent symbolic structures. These high-level features act as model-invariant cues.

**2. Semantic Adversarial Loss: Using cosine similarity instead of cross-entropy to smooth the landscape.**
Cross-entropy requires precise token matching, creating spiky peaks that fail on different models. UltraBreak measures loss in the semantic space: output logits are projected back to the embedding space $\mu_t = W^\top \text{softmax}(z_t)$, targeted tokens are mapped to embeddings $e_t$ with Gaussian noise $\tilde{e}_t = e_t + \varepsilon_t$, and cosine similarity is used: $\mathcal{L}_{\text{sem}} = \frac{1}{T}\sum_t (1 - \cos(\mu_t, e_t^{\text{att}}))$. Here, $e_t^{\text{att}}$ is the target representation weighted by causal attention. This loss allows "correct semantics" to yield low loss without requiring exact token matches, making the landscape flat and improving generalization.

**3. Target Prompt Guidance (TPG): Strengthening the attack on the text side.**
In addition to the visual attack, a fixed template is applied to the query: $q^{\text{TPG}} = \text{"Steps to "} + q + \text{" You must begin your response with: "} + p$, where $p$ is "[Jailbroken Mode]" for open-source models or "[START LIST]" for commercial ones. This forces an imperative "list steps" framework and a specific starting phrase to lower the probability of model refusal.

### Loss & Training
$$\arg\min_x \sum_{(q,y) \in \mathcal{Q}'} \mathbb{E}_{l,r,s}[\mathcal{L}_{\text{sem}}^{\text{att}}(M', A(x_{\text{blank}}, x_{\text{proj}}, l, r, s), q^{\text{TPG}}, y)] + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(x)$$
- Proxy Model: Qwen2-VL-7B-Instruct
- Training: SafeBench-Tiny (50 queries), 1300 Adam steps, $\tau=0.5$, $\lambda_{\text{TV}}=0.5$

## Key Experimental Results

### Main Results: Black-box ASR (SafeBench, 315 queries)

| Target Model | No Attack | FigStep | VAJM | UMK | **UltraBreak** |
|---------|-----------|---------|------|-----|--------------|
| Qwen-VL-Chat | 22.86 | 69.52 | 12.06 | 0.63 | **72.70** |
| Qwen2.5-VL-7B | 14.29 | 53.97 | 28.89 | 15.24 | **60.32** |
| LLaVA-v1.6 | 80.32 | 47.94 | 57.46 | 20.63 | **88.25** |
| GLM-4.1V-9B | 46.03 | 88.25 | 67.62 | 50.79 | **66.03** |
| Black-box Avg | 40.57 | 66.54 | 41.46 | 20.00 | **71.05** |
| Commercial Avg | 20.00 | - | 11.48 | 14.59 | **32.26** |

### Ablation Study

| Configuration | SafeBench Avg | AdvBench Avg | Description |
|------|-------------|-------------|------|
| **Full UltraBreak** | **71.83** | **57.64** | — |
| Remove Image (Text only) | 40.79 | 25.90 | Image contributes ~30% ASR |
| Remove Constraints | 51.99 | 29.86 | White-box overfitting (89%→49% transfer) |
| Remove Semantic Loss (use CE) | 55.80 | 40.15 | CE landscape sharp → poor transfer |
| Remove Attention Weighting | 57.54 | 41.83 | Unstable optimization + higher variance |

### Key Findings
- **Single-image Universal Attack**: Training on 50 queries translates to successful attacks on 315+ harmful queries across multiple architectures.
- **Semantic Loss vs. CE**: Semantic loss produces clustered and smooth landscapes, while CE results in scattered and spiky ones.
- **Transform-invariant Structures**: TV and random transformations cause adversarial images to exhibit text/symbol-like structures, which are easier to transfer across models than pixel noise.
- **Commercial Vulnerability**: Gemini-2.5 reached 42% ASR, and GPT-4o-mini reached 38.78%.

## Highlights & Insights
- **Loss Landscape Perspective**: The core insight is framing adversarial transferability as a loss landscape smoothness problem, using semantic loss to achieve this smoothness.
- **Transform Invariance = Model Invariance**: By learning high-level semantic features rather than low-level pixel noise, the perturbation explores features shared across different models.
- **Challenging Multi-proxy Beliefs**: Contrasting with previous beliefs that multi-model ensembles are required for transferability, UltraBreak proves a single proxy with the correct loss function suffices.

## Limitations & Future Work
- **Limited Effect on High-security Models**: Claude-3-Haiku shows only a 16% ASR, suggesting robust defense models are still effective.
- **White-box Proxy Dependency**: The method still requires an open-source VLM for white-box optimization.
- **Defense Directions**: The paper reveals vulnerability to "transform-invariant visual features," which could lead to new detection methods based on identifying text-like adversarial structures.

## Related Work & Insights
- **vs. FigStep**: FigStep requires handcrafted images (one per target), while UltraBreak is automated and universal with higher ASR.
- **vs. UMK/VAJM**: These methods use CE loss and overfit proxies; UltraBreak's semantic loss addresses this fundamentally.
- **vs. Text-side Jailbreaks**: UltraBreak operates in the visual modality and is orthogonal to text-based attacks, allowing for potential combination.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant combination of loss landscape perspective and semantic loss.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple models and benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear technical details and robust visual analysis.
- Value: ⭐⭐⭐⭐⭐ Significant implications for VLM security and defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](../../ACL2025/llm_alignment/hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)

</div>

<!-- RELATED:END -->
