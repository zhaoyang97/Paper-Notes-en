---
title: >-
  [Paper Note] CARES: Context-Aware Resolution Selector for VLMs
description: >-
  [ACL2026][Multimodal VLM][Resolution Selection] CARES adds a lightweight query-aware resolution selector before the target VLM. It predicts the minimum input resolution "sufficient to answer" the query using low-resoluti…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Resolution Selection"
  - "Visual Token Compression"
  - "VLM Inference Acceleration"
  - "ANLS"
  - "Continuous Routing"
date: 2026-05-08
content_hash: 63deab99805d93dc
---

# CARES: Context-Aware Resolution Selector for VLMs

**Conference**: ACL2026 Oral  
**arXiv**: [2510.19496](https://arxiv.org/abs/2510.19496)  
**Code**: https://mkimhi.github.io/CARES/  
**Area**: Multimodal VLM / Inference Efficiency / Adaptive Resolution  
**Keywords**: Resolution Selection, Visual Token Compression, VLM Inference Acceleration, ANLS, Continuous Routing

## TL;DR
CARES adds a lightweight query-aware resolution selector before the target VLM. It predicts the minimum input resolution "sufficient to answer" the query using low-resolution images and text questions. It maintains accuracy across 9 multimodal benchmarks while saving approximately 65–85% of prefill computation costs on average.

## Background & Motivation
**Background**: To cover tasks like OCR, document understanding, natural image QA, and chart reasoning, general-purpose VLMs typically default to high-resolution or AnyRes/tiling inputs. Higher resolutions result in more visual tokens; in the prefill stage, visual tokens can account for up to 99% of the total tokens.

**Limitations of Prior Work**: Many user queries do not require high resolution. For example, "What breed is the dog" may only need a low-res image, whereas "What is the name on the collar" requires high resolution. Existing methods like token pruning, pooling, or merging mostly occur after visual encoding, meaning the high-resolution tokenization cost has already been paid, and these methods often lack awareness of the current text query.

**Key Challenge**: VLMs require high resolution to ensure quality on difficult samples, but processing all samples at the highest resolution wastes significant computation. The truly controllable lever is before tokenization: deciding how many pixels to use first.

**Goal**: Learn a preprocessing module placed before any VLM to predict the minimum sufficient resolution based on the image-query pair, reducing visual tokens, FLOPS, TTFT, or API costs without modifying the target VLM architecture or weights.

**Key Insight**: Instead of directly predicting "hard/easy," the authors generate supervision using the target VLM's actual response quality under multi-resolution rollouts: whichever minimum resolution achieves sufficient quality is used as the label.

**Core Idea**: Shift the VLM inference efficiency problem forward to input resolution selection, learning "just enough" pixel allocation through query-conditioned sufficiency labels.

## Method
The main version of CARES is a discriminative selector: it uses a truncated SmolVLM-500M to jointly encode the image and question at a low resolution, takes the representation of the last token from an intermediate layer, and connects it to a lightweight classification head to predict three resolution categories (384/768/1024). During inference, continuous resolution is obtained through probability weighting.

### Overall Architecture
Before training, the authors perform multi-resolution labeling on the samples `(x,q,gt)`: images are resized to candidate resolutions and fed into a fixed VLM to obtain answers. Answer quality is evaluated using ANLS or corresponding metrics, and the minimum sufficient resolution is selected as the label. During training, CARES only sees the low-resolution image and query to learn to predict this label. At deployment, CARES outputs a continuous resolution, and the target VLM only processes the scaled image and original query.

### Key Designs
1. **Multi-resolution Sufficiency Labeling**:
	- **Function**: Generates the "minimum sufficient resolution" supervision signal for each sample.
	- **Mechanism**: Run the target VLM on a discrete set $\mathcal R_d=\{384,768,1024\}$ and calculate $u_k=ANLS(F(x^{(r_k)},q),gt)$; the smallest $r_k$ satisfying $u_k\ge \tau$ and where higher resolutions provide improvements no greater than $\delta$ is selected. Defaults are $\tau=0.85$ and $\delta=0.1$.
	- **Design Motivation**: Directly searching for continuous optimal resolution is too costly; discrete rollouts provide stable labels while retaining the meaning of "stopping once performance converges."

2. **Lightweight Context-Aware Selector**:
	- **Function**: Uses a small model to judge how much visual detail the current image-query requires before calling the large VLM.
	- **Mechanism**: Employs SmolVLM-500M with the latter half of the layers removed, using only the intermediate representation up to layer 16. Given the image and text query at $r_{min}$, the hidden state of the last token is fed into a classifier to output probabilities for three resolution levels.
	- **Design Motivation**: Intermediate layers often retain rich perceptual and semantic information, and running only half of a small VLM incurs low extra cost. Query-aware joint encoding is also more suitable than dual-tower features (like SigLIP) for judging "which visual details this question needs."

3. **Discrete Training, Continuous Deployment**:
	- **Function**: Avoids jumping only between coarse resolution levels during deployment.
	- **Mechanism**: After the classifier outputs probability $p=softmax(\ell)$, a continuous resolution is obtained via $\tilde r=\sum_k p_k r_k$, which is then rounded up to the input size supported by the target backbone.
	- **Design Motivation**: Discrete labels are easier to annotate and train, while continuous outputs leverage model uncertainty for finer-grained compute-quality trade-offs.

### Loss & Training
CARES uses an 80K training set, with 20K samples each from TextVQA, ChartQA, DocVQA, and LLaVA-Multi, covering both document and natural images. The main selector is trained for 6 epochs with a learning rate of $10^{-3}$ and batch size of 32, using cross-entropy $\mathcal L(\theta)=CE(f_\theta(z),r^*)$ to supervise the three-class resolution classification, with 0.05 label smoothing to support continuous resolution deployment. The authors also implemented an autoregressive version using Granite-Docling-258M, trained for 3 epochs with LoRA rank 8, allowing the model to predict `<1>/<2>/<3>` resolution tokens.

## Key Experimental Results

### Main Results
| Target VLM | Native Avg. | CARES Avg. | Avg. Cost change | Note |
|------------|---------------|--------------|--------------|------|
| Granite-Vision-2B | 0.59 | 0.60 | -63% | Accuracy slightly increased and cost significantly decreased on small models |
| InternVL3-8B | 0.77 | 0.77 | -64% | Performance maintained across multiple benchmarks |
| Qwen2.5-VL-72B | 0.79 | 0.80 | -70% | Transfers well to large models |
| GPT-4o | 0.69 | 0.68 | -55% | API costs decreased with quality roughly maintained |

| Benchmark Range | Metric | CARES Setting |
|----------------|------|-------------|
| Ai2D / ChartQA / SeedBench-2 | exact-match accuracy | Evaluation of natural images, charts, and general VQA |
| DocVQA / OCRBench | ANLS | Testing if high resolution is still needed in document/OCR scenarios |
| MMMU / RealWorldQA / InfoVQA / MathVista | task score | Testing cross-domain generalization |
| DocVQA latency frontier | TTFT / TFLOPs | CARES approaches native performance with ~2.58 TFLOPs, while native is ~7.5 TFLOPs |

### Ablation Study
| Configuration | Key Metrics | Note |
|------|----------|------|
| SigLIP v2 feature | 56.1% resolution accuracy | Dual-tower features are inferior to joint VLM encoding |
| SmolVLM Mid | 63.3% / 0.35B params | Default choice, best balance of efficiency and accuracy |
| SmolVLM Last | 62.3% / 0.5B params | Last layer is slightly weaker and more costly |
| Qwen2.5-3B Mid | 67.2% / 2.3B params | Most accurate resolution classification but heavier |
| Two-level res {384,1024} | 96.2% class accuracy / 0.76 downstream | Simpler labels but control is too coarse |
| Three-level res {384,768,1024} | 67.2% class accuracy / 0.80 downstream | Classification is harder, but downstream effect is better |
| Continuous routing | Granite/InternVL FLOPS -63% | More efficient than discrete (-46%) with almost no score drop |
| Label smoothing | OCRBench 0.821 vs 0.811 | Improves continuous resolution probability calibration |

### Key Findings
- CARES generalizes across different target VLMs: Significant prefill savings are achieved on Granite, InternVL, Qwen2.5-VL, and GPT-4o with less than a 1 percentage point change in average quality.
- Sufficient resolution labels are not accidental preferences of a single teacher: Granite-Vision-2B and Qwen3-VL-235B show over 95% label consistency across 1000 samples, with a Pearson correlation of 0.908.
- Continuous resolution is not decorative. While discrete prediction saves computation, continuous prediction further reduces FLOPS and avoids excessive upscaling near hard classification boundaries.

## Highlights & Insights
- The biggest highlight of CARES is placing efficiency control "before pixels enter the VLM." Compared to token pruning, it avoids the "pay first, save later" issue of encoding at high resolution before deleting tokens.
- Generating sufficiency labels via multi-resolution rollout is practical: it does not require human judgment of "how clear this needs to be," but lets the target model's own performance curve define the sufficiency threshold.
- Query-awareness is key. Resolution requirements cannot be judged by the image alone; the same image requires a completely different pixel budget for coarse classification versus OCR questions.

## Limitations & Future Work
- Training labels require multi-resolution VLM rollouts for a large number of samples, incurring non-trivial offline annotation costs; if the target model is updated frequently, labels may need reconstruction or calibration.
- Currently, the method primarily handles single-image static input; resolution selection for video, multi-image reasoning, and interactive retrieval scenarios will be more complex.
- CARES only selects the input resolution and does not address visual token redundancy inside the model; it is complementary to token pruning/merging, but the cumulative error after combination requires further study.
- For extremely small text, sparse details, or safety-critical tasks, the risk of low resolution is higher; deployment may require task-level safety floors or confidence-based retreats.

## Related Work & Insights
- **vs HiRED / SparseVLM / PyramidDrop / VTW**: These methods reduce visual tokens after tokenization or encoding; CARES determines image resolution before tokenization, avoiding unnecessary high-resolution input costs.
- **vs TokenFLEX / Matryoshka / LLaVA-Mini**: These methods train models to adapt to different token budgets; CARES does not modify the target VLM and can serve as a front-end to be stacked with elastic token models.
- **vs AnyRes / tiling**: AnyRes preserves details through more tiles; CARES judges whether those details are needed based on the query, allowing it to bypass high-cost tiling for coarse queries.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Resolution selection itself is not new, but query-aware sufficiency rollout is very practical for VLM inference front-ends.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 benchmarks, 4 types of target VLMs, AR versions, and multiple ablations; evidence is sufficient.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological motivation and algorithms are clear; tables are somewhat dense but contain complete key information.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for practical VLM deployment, cost control, and dynamic visual computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICML 2026\] Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](../../ICML2026/multimodal_vlm/density-aware_translation_of_spurious_correlations_in_zero-shot_vlms.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling](stability_implies_redundancy_delta_attention_selective_halting_for_efficient_lon.md)
- [\[ACL 2026\] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation](vauq_vision-aware_uncertainty_quantification_for_lvlm_self-evaluation.md)

</div>

<!-- RELATED:END -->
