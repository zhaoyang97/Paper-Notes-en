---
title: >-
  [Paper Note] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models
description: >-
  [ACL 2025][LLM (Other)][diffusion language model] Proposes EdiText, a controllable text editing method based on latent diffusion models. It combines SDEdit for coarse-grained editing and self-conditioning for fine-grained editing, achieving multi-scale text editing control ranging from minor modifications to extensive rewriting.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "diffusion language model"
  - "text editing"
  - "SDEdit"
  - "self-conditioning"
  - "controllable generation"
date: 2026-05-08
content_hash: 93589a537fe31d86
---

# EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.19765](https://arxiv.org/abs/2502.19765)  
**Code**: -  
**Area**: NLP / Text Editing / Diffusion Models  
**Keywords**: diffusion language model, text editing, SDEdit, self-conditioning, controllable generation  

## TL;DR

Proposes EdiText, a controllable text editing method based on latent diffusion models. It combines SDEdit for coarse-grained editing and self-conditioning for fine-grained editing, achieving multi-scale text editing control ranging from minor modifications to extensive rewriting.

## Background & Motivation

**Background:** Text editing aims to modify a given reference text to satisfy target attributes. Existing methods include approaches based on autoregressive and non-autoregressive models. While diffusion models have demonstrated powerful multi-scale control capabilities in the image editing domain, their application to text editing remains under-explored.

**Limitations of Prior Work:** (1) Energy-based model approaches (e.g., Mireshghallah et al. 2022) only offer fine-tuning level control, with a limited range; (2) ParaGuide (Horvitz et al. 2024) uses classifier guidance to adjust editing intensity, but the control range remains narrow; (3) Autoregressive models (e.g., Qwen2.5) show limited response to instruction control regarding the editing intensity, and modifying prompts fails to significantly alter the editing scale.

**Core Problem:** How to achieve both coarse-grained (large-scale adjustment) and fine-grained (precise tuning) editing control in text editing?

## Method

### Overall Architecture

EdiText uses LD4LG (Latent Diffusion for Language Generation) as the backbone model. This model compresses discrete text into a fixed-length continuous latent representation via a Perceiver Resampler encoder, reconstructs the text using an autoregressive decoder, and trains a conditional diffusion model to model the distribution of latent representations. Two complementary editing techniques are integrated on top of this model.

### Key Designs

1. **SDEdit Coarse-Grained Editing (EdiText-CE):** The reference text is encoded into a latent representation $x_0$, which is diffused to timestep $t_{CE}$ during the forward process, followed by reverse denoising using the trained conditional diffusion model conditioned on target attributes. The parameter $t_{CE}$ controls the editing intensity: when $t_{CE}$ is close to $T$, more noise is added, less original text is preserved, and the edit is large; when $t_{CE}$ is close up to 0, less noise is added, more is preserved, and the edit is small.

2. **Self-Conditioning Fine-Grained Editing (EdiText-FE):** The self-conditioning mechanism is re-purposed: instead of using the model's own previous prediction during sampling, the latent representation of the reference text is injected as a condition. The reference text representation serves as the condition from $t=T$ down to $t_{FE}$, below which regular self-conditioning resumes. Smaller $t_{FE}$ values prolong the influence of the reference text, leading to smaller edits.

3. **Integrated Coarse-to-Fine Editing:** SDEdit provides large-scale but coarse control, while self-conditioning provides small-scale but precise control. When stacked together, SDEdit first establishes the overall editing scope, and self-conditioning then performs fine tuning within that range, achieving a complete multi-scale coverage.

### Loss & Training

- LD4LG Training Loss: $L(\theta) = \mathbb{E}_{t,x_0,\epsilon_t}[\lambda_t^{-1} \|x_\theta(x_t, t) - x_0\|_2^2]$, where $\lambda_t = 1 - \alpha_t$
- Additional training in self-conditioning mode: Alternates training between unconditional and conditional (previous step prediction) modes with a probability of $p=0.5$

## Experiments

### Main Results (Detoxifying)

| Method | Hamming ↓ | SacreBLEU ↑ | BERTScore ↑ | Moderation ↓ | PerspectiveAI ↓ |
|------|-----------|-------------|-------------|-------------|-----------------|
| ParaGuide (λ=200) | 25.3 | 14.9 | 0.903 | 0.446 | 0.321 |
| ParaGuide (λ=10K) | 27.2 | 11.0 | 0.889 | 0.335 | 0.229 |
| Qwen2.5-0.5B | 27.2 | 31.1 | 0.903 | 0.347 | 0.312 |
| EdiText-CE (t=175) | 17.4 | 34.7 | 0.923 | 0.576 | 0.450 |
| EdiText-CE (t=200) | 28.9 | 7.6 | 0.865 | 0.105 | 0.136 |
| EdiText-FE (t=25) | 24.7 | 14.9 | 0.881 | 0.117 | 0.121 |

### Sentiment Control Task (Neg → Pos)

| Method | Hamming ↓ | BERTScore ↑ | Accuracy ↑ |
|------|-----------|-------------|------------|
| ParaGuide (λ=10K) | 18.0 | 0.857 | 0.89 |
| Qwen2.5-0.5B | 23.9 | 0.881 | 0.60 |
| EdiText-CE (t=200) | 15.1 | 0.879 | 0.77 |
| EdiText-CE (t=225) | 19.5 | 0.846 | **0.90** |
| EdiText-FE (t=25) | 10.7 | **0.916** | 0.60 |

### Key Findings

- **Control Range:** EdiText-CE covers the full range from near zero edit to completely rewritten texts by adjusting $t_{CE}$, whereas ParaGuide and Qwen2.5 exhibit extremely limited control ranges.
- **Editing Quality:** Under the same preservation rates, the target attribute satisfaction rate of EdiText outperforms or matches the baselines.
- **Fine-Grained Control:** EdiText-FE provides smoother editing gradients, mitigating the abrupt changes observed in EdiText-CE.
- **Integration Advantage:** Combining coarse and fine controls enables continuous and seamless multi-scale editing coverage.

## Highlights & Insights

- Innovatively adopts SDEdit (a method from the image domain) to text editing to achieve coarse-grained control.
- Cleverly re-interprets self-conditioning, reframing it from "enhancing generation quality" to "anchoring the reference text representation."
- The dual-layered coarse/fine control mechanism is highly complementary, covering the entire range of edits.
- The approach is clean and elegant, bypassing the need for additional classifiers (unlike ParaGuide).

## Limitations & Future Work

- Since it is based on latent diffusion models, the generated text quality is still inferior to modern large-scale autoregressive models.
- LD4LG compresses text into a fixed-length latent representation, which might lose details in long texts.
- Evaluation is only conducted on toxicity detoxification and sentiment control tasks; generalizability remains to be verified.
- Optimal editing parameters ($t_{CE}$, $t_{FE}$) need empirical tuning for specific tasks.
- Inference speed of diffusion models is slower compared to instruction-guided editing with autoregressive LLMs.

## Related Work & Insights

- **Diffusion Language Models:** LD4LG (Lovelace et al. 2023) latent diffusion; MDLM (Sahoo et al. 2024) discrete diffusion.
- **Text Editing:** ParaGuide (Horvitz et al. 2024) classifier-guided; Mireshghallah et al. 2022 EBM-based.
- **Image-to-Text Editing Transference:** SDEdit (Meng et al. 2022) noise-denoise editing framework.
- **Controllable Text Generation:** Li et al. 2022 diffusion-based constrained generation; self-conditioning (Chen et al. 2023) for improved sampling quality.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 7 |
| Writing Quality | 7 |
| Value | 6 |
| Overall Score | 7.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
