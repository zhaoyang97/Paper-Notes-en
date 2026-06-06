---
title: >-
  [Paper Note] DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning
description: >-
  [ACL 2026][Multimodal VLM][MLLM] DRIFT treats the "parameter difference between a text reasoning expert and a multi-modal model" as a directional prior. During multi-modal SFT backpropagation…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "MLLM"
  - "reasoning transfer"
  - "gradient prior"
  - "SFT"
  - "model merging"
date: 2026-05-08
content_hash: 1e9907e7efd8a544
---

# DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning

**Conference**: ACL 2026  
**arXiv**: [2510.15050](https://arxiv.org/abs/2510.15050)  
**Code**: https://wikichao.github.io/DRIFT/ (Project Page available)  
**Area**: Multi-modal VLM / Fine-tuning / Reasoning Transfer  
**Keywords**: MLLM, reasoning transfer, gradient prior, SFT, model merging

## TL;DR
DRIFT treats the "parameter difference between a text reasoning expert and a multi-modal model" as a directional prior. During multi-modal SFT backpropagation, it applies lightweight biasing to the gradients (without moving weights). Using only 4K multi-modal CoT data and approximately 2 hours of training, it consistently pushes Qwen2.5-VL-7B past parameter merging baselines and heavy SFT/RL methods on benchmarks like MathVista, MathVerse, and WeMath.

## Background & Motivation

**Background**: The current mainstream routes for improving MLLM reasoning capabilities are twofold: large-scale multi-modal CoT SFT (e.g., R1-OneVision, OpenVLThinker) or running RL on multi-modal data (e.g., R1-VL). Both rely on expensive multi-modal reasoning data and multi-day training. Meanwhile, pure text reasoning models (DeepSeek-R1-Distill series, Qwen-Math, etc.) are easily obtained due to the abundance of CoT text data.

**Limitations of Prior Work**: MLLMs generally "see clearly but reason poorly"—perception is fine, but multi-step reasoning deviates. Conversely, text reasoning experts are powerful but lack vision. While merging their parameters (BR2V, Task Arithmetic, TIES, DARE, Layer Swap) seems like a free lunch, the authors found in Tab. 1 using four backbones that: LLaMA/Mistral series (with relatively close parameter spaces) show small gains of +1~2 points, but Qwen series (Qwen2-VL, Qwen2.5-VL) suffer from large parameter space distribution shifts, causing performance drops after merging (Qwen2.5-VL+R1 drops −8.2 on MathVerse).

**Key Challenge**: The success of parameter space merging depends entirely on the alignment of the two experts' distributions on the backbone. Once the magnitude/direction deviates significantly, linear interpolation destroys multi-modal alignment, causing instability or even gradient explosion. Finding an optimal interpolation coefficient $\beta$ requires loading all candidate models into VRAM simultaneously, which is highly expensive.

**Goal**: To find a lightweight mechanism that can stably "borrow" reasoning capabilities from text experts for MLLMs without stacking massive multi-modal CoT data.

**Key Insight**: The authors' key observation is that the parameter difference between the expert and the base essentially encodes "domain knowledge direction." Instead of interpolating directly in weight space (which destroys alignment), it is better to inject this directional prior into the **gradients** of SFT. This allows the optimization trajectory to be "gently pulled" toward the reasoning direction rather than forcing parameters to snap there.

**Core Idea**: Treat $\Delta = \phi_{\text{reason}} - \phi_{\text{VL}}$ as a directional prior. During backpropagation, bias the gradients via $\tilde{g} = g + \alpha \cdot \text{scale}(g, \Delta)$. This maintains the standard SFT pipeline while stably transferring text reasoning capabilities to the multi-modal domain.

## Method

### Overall Architecture
DRIFT embeds "reasoning injection" into the backpropagation of standard multi-modal SFT. The process consists of three stages:

1.  **Offline Computation of Reasoning Prior**: Take a text reasoning expert $\phi_{\text{reason}}$ (e.g., DeepSeek-R1-Qwen-Distill-7B) and a multi-modal variant $\phi_{\text{VL}}$ (e.g., the LLM backbone of Qwen2.5-VL-7B) derived from the same base LLM. Calculate the parameter difference $\Delta = \phi_{\text{reason}} - \phi_{\text{VL}}$ layer-by-layer and module-by-module. $\Delta$ is retained only for selected "reasoning-related" modules (defaulting to ATTN projections Q/K/V/O, with MLP/Norm/LM Head as options). The result is **stored on the CPU** and moved to the GPU as needed.
2.  **Conventional Multi-modal SFT Forward Pass**: Train Qwen2.5-VL-7B-Instruct using 4K high-quality multi-modal CoT data (distilled from ThinkLiteVL-11K + filtered for errors, with CoT wrapped in `<think></think>`). The forward pass, loss, and autograd remain unchanged.
3.  **Directional Prior Injection via Gradient Hooks**: Register hooks during `backward()`. For each selected parameter $w$, rewrite the original gradient $g$ as the guided gradient $\tilde{g} = g + \alpha \cdot \text{scale}(g, \Delta)$, then pass it to the optimizer. Training runs for 3 epochs with a learning rate of $1\times 10^{-6}$ and $\alpha=-1$, taking about 2 hours.

### Key Designs

1.  **Directional Prior in Gradient Space (Not Weight Space Interpolation)**:
    *   Function: Uses the parameter difference from expert to VL as a compass for "where to go," but only nudges the gradient direction without directly modifying weights.
    *   Mechanism: Traditional BR2V performs $\phi_{\text{VL}\oplus\text{reason}} = \phi_{\text{base}} + \beta(\phi_{\text{VL}}-\phi_{\text{base}}) + (1-\beta)(\phi_{\text{reason}}-\phi_{\text{base}})$, which is extremely sensitive to $\beta$. DRIFT modifies the gradient of each selected module at every SFT step via $\tilde{g} = g + \alpha \cdot \text{scale}(g, \Delta)$. This allows weights to stay dominated by the multi-modal loss while being gently pulled by $\Delta$.
    *   Design Motivation: The destructiveness of parameter-level merging comes from "jumping across" in one step. Gradient-level injection is "nudging at every step," which, coupled with multi-modal CoT data, naturally aligns perception and reasoning without breaking visual alignment.

2.  **Three Scale Variants (Determining how $\Delta$ Interacts with $g$)**:
    *   Function: Controls the intensity of the directional prior to match the current gradient scale.
    *   Mechanism: The authors compare three formulas: (i) **Absolute** $\tilde{g} = g + \alpha \Delta$, adding $\Delta$ directly to the gradient; (ii) **Grad-Norm** $\tilde{g} = g + \alpha \|g\| \frac{\Delta}{\|\Delta\|}$, taking only the direction of $\Delta$ while preserving the magnitude of $g$; (iii) **Grad-Norm w/ Adaptive $\alpha$** $\tilde{g} = g + \alpha' \|g\| \frac{\Delta}{\|\Delta\|}$ where $\alpha' = \alpha \cdot \frac{1 + \cos(g, \Delta)}{2}$. This pushes more when $g$ and $\Delta$ are aligned and less when they conflict.
    *   Design Motivation: Absolute pull forces weights toward the reasoning expert (dropping 3 points on MathVista and 19.7 on LogicVista), proving "absolute magnitude" destroys alignment. Grad-Norm scales intensity with the current gradient, ensuring stability. Adaptive $\alpha$ further adjusts based on directional consistency, proving most robust—directly validating that "borrowing direction, not magnitude" is key.

3.  **Module Selection: Injecting into Attention Projections is Most Stable**:
    *   Function: Determines which transformer sub-modules $\Delta$ is injected into.
    *   Mechanism: Ablations were performed on ATTN(Q/K/V/O), MLP, Norm, and LM Head. Results showed that selecting only {ATTN} was most stable (+3.8 on LogicVista, +2.4 on MathVerse). Adding MLP lowered performance; adding Norm introduced noise; extending to LM Head gave inconsistent gains.
    *   Design Motivation: Attention projections are core to "deciding where to look" between tokens, carrying long-range dependency routing required for reasoning. MLPs act more like "local knowledge retrieval" with high cross-domain variance and noise. Norm parameters are scale-sensitive and easily derail training.

### Loss & Training
-   **Goal**: Standard multi-modal SFT cross-entropy loss, no auxiliary losses, no new parameters.
-   **Data**: 4K multi-modal CoT (ThinkLiteVL-11K → ThinkLite distilled CoT → error filtering → `<think></think>` wrapping).
-   **Optimization**: 3 epochs, lr $1\times10^{-6}$, $\alpha=-1$ (meaning $g$ is biased along the $-\Delta$ direction; since $\Delta = \phi_{\text{reason}} - \phi_{\text{VL}}$, weight updates shift toward the reasoning expert).
-   **Engineering**: Based on LLaMAFactory, $\Delta$ resides on CPU and is moved to GPU as needed. Only backpropagation hooks are modified, with zero additional trainable parameters.

## Key Experimental Results

### Main Results

DeepSeek-R1-Qwen-Distill-7B → Qwen2.5-VL-7B-Instruct, comparing 5 parameter merging and 4 reasoning SFT methods (Combined Tab. 2 + Tab. 3):

| Method | MathVista | MathVision | MathVerse | WeMath-strict | LogicVista | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B (baseline) | 67.9 | 25.0 | 41.4 | 34.3 | 46.7 | 44.7 |
| Task Arithmetic | 65.8 (−2.1) | 22.7 (−2.3) | 33.2 (−8.2) | 30.1 (−4.2) | 42.0 (−4.7) | 40.8 |
| TIES | 63.6 | 23.1 | 39.5 | 33.4 | 42.1 | 42.2 |
| DARE-TIES | 66.3 | 23.6 | 38.3 | 33.7 | 42.0 | 42.8 |
| Layer Swap | 63.6 | 22.9 | 37.9 | 32.1 | 35.1 | 40.3 |
| Pure SFT (4K) | 68.7 | 25.1 | 42.0 | 33.3 | 45.6 | — |
| **DRIFT (Ours)** | **69.9 (+2.0)** | **26.6 (+1.6)** | **43.9 (+2.5)** | **38.5 (+4.2)** | **47.2 (+0.5)** | **47.7 (+3.0)** |

DRIFT is the only method that improves across all 5 benchmarks, slightly outperforming much heavier methods like OpenVLThinker / R1-OneVision / X-Reasoner in average score while using only 4K data and ~2h training.

### Ablation Study

Tab. 4 showing combinations of scale strategy and merge modules (SFT baseline: MathVista 68.7 / MathVerse 42.0 / LogicVista 45.6):

| Configuration | MathVista | MathVerse | LogicVista | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Absolute @ {ATTN, MLP} | 65.7 (−3.0) | 39.5 (−2.5) | 25.9 (**−19.7**) | Directly pulling weights destroys alignment |
| Grad-Norm @ {ATTN, MLP} | 68.8 (+0.1) | 43.9 (+1.9) | 46.1 (+0.5) | Stable |
| Grad-Norm + Adaptive $\alpha$ @ {ATTN, MLP} | 69.9 (+1.2) | 43.9 (+1.9) | 47.2 (+1.6) | **Full Model** |
| Grad-Norm @ {ATTN only} | 68.8 | 44.4 (+2.4) | **49.4 (+3.8)** | Best for MathVerse/LogicVista |
| Grad-Norm @ {MLP only} | 68.5 (−0.5) | 42.6 (+0.6) | 46.3 (+0.7) | Minimal gain |
| Grad-Norm @ {ATTN, MLP, Norm} | 68.6 (−0.1) | 43.0 (+1.0) | 46.8 (+1.2) | Norm dilutes results |

### Key Findings
- **Direction vs. Magnitude**: Absolute mode crashed by 19.7 points on LogicVista, proving that pulling weights directly shatters multi-modal alignment. Grad-Norm is stable by only borrowing direction. Adaptive $\alpha$ is most robust by utilizing $\cos(g, \Delta)$.
- **Module Sensitivity**: ATTN projection is the best carrier for reasoning transfer. Injecting only into ATTN is more stable than injecting into all modules, suggesting reasoning capabilities reside more in "attention routing" than "FFN knowledge storage."
- **Maintaining Perception**: Tab. 6 shows DRIFT maintains or slightly improves performance on HallusionBench/RealWorldQA/MMStar (RWQA 68.6→69.2, MMStar 64.7→65.6), while Pure SFT drops 1.83/1.90 points—proving gradient-level injection is "lossless" to original visual capabilities.
- **Cross-Family Generalization**: Tab. 5 shows DRIFT consistently outperforms SFT when applied to other pairs like LLaVA-Next-8B + DART or Qwen2.5-VL + Qwen2.5-Math.

## Highlights & Insights
- **Shift from "Weight Space Merging" to "Gradient Space Injection"**: Captures the root cause of merging failures—jumping across the manifold destroys it—whereas nudging along the direction at each step avoids $\beta$ tuning and constant VRAM pressure.
- **CPU-Resident $\Delta$ + Backward Hook Injection**: Fully decouples "merging" from the forward pass, loss, and trainable parameters. It is an engineering "plug-and-play" solution with a very low barrier to reproduction.
- **Adaptive $\alpha = \alpha \cdot \frac{1+\cos(g,\Delta)}{2}$**: This clever formula quantifies the geometric relationship between the "prior" and "current task gradient" into intensity modulation, avoiding hard pushes in conflicting directions. This trick can be transferred to any "prior vector + gradient" scenario (e.g., task vector distillation, anti-forgetting in Continual Learning).
- **Practical Conclusion on Attention**: The evidence that "attention projections carry the most reasoning" can guide future target module selection for LoRA / DoRA / selective fine-tuning.

## Limitations & Future Work
- **Mathematical Reasoning Only**: Main benchmarks are MathVista/MathVerse/MathVision/WeMath/LogicVista series. Effectiveness on multi-modal scientific reasoning, code, or agentic planning remains unknown.
- **Requirement for "Same Base LLM" Pairs**: Authors emphasize derivation from the same base (e.g., Qwen2.5-VL comes from Qwen2.5). If backbones are from different sources (e.g., LLaVA with DeepSeek), the meaning of $\Delta$ becomes unclear.
- **Empirical $\alpha=-1$**: No scanning curve for $\alpha$ is provided. It is uncertain if $\alpha$ needs retuning across backbones, although Adaptive $\alpha$ handles dynamic scaling.
- **No RL Comparison**: The tables compare with SFT/merge methods but not directly with RL post-training (GRPO/RLOO). The claim of "matching heavy training methods" is primarily based on SFT baselines like X-Reasoner / R1-OneVision.
- **Future Directions**: Upgrading $\Delta$ to weighted multi-experts (reasoning + code + tool-use) or decaying the directional prior along the training schedule are logical extensions.

## Related Work & Insights
- **vs. BR2V (Chen et al. 2025a)**: BR2V performs merging in weight space, causing significant drops for Qwen. DRIFT moves the same $\Delta$ to gradient space and avoids hard-merging instability through Adaptive scaling.
- **vs. Task Arithmetic / TIES / DARE / Layer Swap**: These are "post-training one-time merges" sensitive to parameter shift. DRIFT is a "continuous small-step bias during SFT," allowing it to tolerate larger $\Delta$ magnitudes.
- **vs. LoRA / DoRA**: LoRA introduces new parameters to learn increments. DRIFT introduces no parameters and only modifies gradient direction. They can be combined orthogonally.
- **vs. R1-OneVision / OpenVLThinker / X-Reasoner**: These methods require 59K+ multi-modal CoT data and days of RL/SFT. DRIFT's outperformance with 4K data and 2h SFT suggests "prior injection" is a valuable direction for low-resource reasoning transfer.
- **Inspiration**: Treating a "pre-trained other expert" as a directional prior to guide optimization can be generalized to: (i) cross-lingual transfer, (ii) cross-modal transfer (audio → vision), (iii) anti-forgetting in Continual Learning (using old task experts as $\Delta$).

## Rating
- Novelty: ⭐⭐⭐⭐ Moving model merging from weight space to gradient space is a clear and under-explored perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers main tables, multiple ablations, cross-backbone tests, and perception preservation. Missing RL head-to-head and $\alpha$ scanning.
- Writing Quality: ⭐⭐⭐⭐ The "merge failure case" in Tab. 1 is a compelling introduction. Method formulas are clear.
- Value: ⭐⭐⭐⭐ A low-resource reasoning transfer paradigm with minimal engineering overhead, compatible with existing SFT pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](../../NeurIPS2025/multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](../../NeurIPS2025/multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)

</div>

<!-- RELATED:END -->
