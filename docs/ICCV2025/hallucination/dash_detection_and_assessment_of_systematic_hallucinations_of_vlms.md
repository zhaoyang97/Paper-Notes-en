---
title: >-
  [Paper Note] DASH: Detection and Assessment of Systematic Hallucinations of VLMs
description: >-
  [ICCV 2025][Hallucination Detection][Vision-language models] This paper proposes DASH, a fully automated pipeline that systematically discovers false-positive object hallucination clusters in VLMs via two complementary strategies: LLM-based text query generation (DASH-LLM) and diffusion model optimization-based image query generation (DASH-OPT). Applied to ReLAION-5B, DASH uncovers 19k+ clusters and 950k+ images, and constructs the more challenging DASH-B benchmark.
tags:
  - "ICCV 2025"
  - "Hallucination Detection"
  - "Vision-language models"
  - "object hallucination"
  - "systematic error detection"
  - "diffusion model optimization"
  - "large-scale benchmark"
date: 2026-05-08
content_hash: 0867e0fb66ac665a
---

# DASH: Detection and Assessment of Systematic Hallucinations of VLMs

**Conference**: ICCV 2025  
**arXiv**: [2503.23573](https://arxiv.org/abs/2503.23573)  
**Code**: [https://YanNeu.github.io/DASH](https://YanNeu.github.io/DASH)  
**Area**: Hallucination Detection  
**Keywords**: Vision-language models, object hallucination, systematic error detection, diffusion model optimization, large-scale benchmark

## TL;DR
This paper proposes DASH, a fully automated pipeline that systematically discovers false-positive object hallucination clusters in VLMs via two complementary strategies: LLM-based text query generation (DASH-LLM) and diffusion model optimization-based image query generation (DASH-OPT). Applied to ReLAION-5B, DASH uncovers 19k+ clusters and 950k+ images, and constructs the more challenging DASH-B benchmark.

## Background & Motivation
VLMs (e.g., PaliGemma, LLaVA-NeXT) achieve strong performance on multimodal tasks, yet suffer from **object hallucination**—incorrectly identifying objects as present when they do not appear in the image. Existing benchmarks (POPE, AMBER) exhibit two fundamental limitations:

**Overly restricted datasets**: They rely on small-scale annotated datasets such as MSCOCO (only 80 object categories), failing to reflect VLM behavior in open-world deployment. The TNR of current models on POPE has reached 96%, approaching saturation.

**Lack of systematic evaluation**: They cannot distinguish between sporadic hallucinations and systematic model deficiencies. If a VLM consistently hallucinates on specific image types, this indicates a fundamental flaw requiring correction.

- **Key Challenge**: VLMs are deployed in open-world settings, yet evaluation benchmarks remain confined to closed datasets. Exhaustively testing all image–object combinations in ReLAION-5B is infeasible, necessitating an **effective search strategy** to surface systematic hallucinations.
- **Core Idea**: Construct a fully automated pipeline, DASH, that generates targeted queries (textual or visual) to retrieve real images from large-scale datasets that trigger VLM hallucinations, and employs clustering to identify semantically coherent systematic error patterns.

## Method

### Overall Architecture
DASH consists of four stages: query generation (DASH-LLM or DASH-OPT) → Exploration (kNN retrieval) → Exploitation (expanded retrieval) → Clustering. The entire pipeline is fully automated and requires no manual annotation.

### Key Designs
1. **DASH-LLM (Text-based Query Generation)**:

    - **Function**: Employs an LLM (Llama 3.1-70B) to generate 50 text queries per object category.
    - **Mechanism**: The LLM is prompted to produce scene descriptions likely to cause VLM misidentification (e.g., "fireboat" may induce hallucination of "water cannon"), without explicitly mentioning the target object.
    - **Design Motivation**: False-positive hallucinations commonly arise from object co-occurrence associations (e.g., Christmas decorations → Baumkuchen cake). LLMs trained on large-scale text corpora effectively capture such associations.
    - **Limitation**: Queries are model-agnostic (not tailored to model-specific errors), and CLIP-based retrieval from text queries may fail to surface sufficiently relevant images.

2. **DASH-OPT (Optimization-based Image Query Generation)**:

    - **Function**: Optimizes the input variables of a diffusion model to synthesize images that simultaneously fool the VLM while not containing the target object.
    - **Mechanism**: A single-step diffusion model (distilled LDM) is used, jointly optimizing two objectives:
        - VLM loss: $L_{\text{vlm}}(C) = -\log p_{\text{vlm}}(\text{"Yes"} \mid q(C), \text{qstnOBJ})$, maximizing the probability that the VLM responds "Yes."
        - Detector loss: $L_{\text{det}}(C) = -\log(1 - p_{\text{det}}(\text{OBJ} \mid q(C)))$, minimizing the open-world detector (OWLv2) confidence for the target object.
        - Overall objective: $\min_C L_{\text{vlm}}(C) + L_{\text{det}}(C)$
    - **Design Motivation**: Optimizing directly in pixel space produces adversarial examples rather than natural images. Optimizing in the latent space of a diffusion model constrains generated images to lie on the natural image manifold. Unlike DASH-LLM, DASH-OPT is model-specific, enabling discovery of more unexpected hallucination patterns.

3. **Exploration–Exploitation–Clustering Pipeline**:

    - **Exploration**: For each query, kNN retrieval (CLIP similarity) is performed over ReLAION-5B to obtain candidate images; images where the detector identifies the target object or where no VLM hallucination is triggered are filtered out.
    - **Exploitation**: Successful images from the exploration stage undergo a second kNN retrieval (50 neighbors per image) to verify whether hallucinations transfer to semantically similar images; DreamSim is used to remove near-duplicates.
    - **Clustering**: Hierarchical clustering (average linkage) in the CLIP embedding space merges similar pre-clusters.

### Loss & Training
- OWLv2 is applied with a very low detection threshold (conservative strategy) to ensure that images labeled as "not containing the target object" are as reliable as possible.
- Human verification shows that only 5.2% of DASH images actually contain the target object, compared to a 25.5% label noise rate in POPE.
- Fine-tuning strategy: for each object, 200 DASH images (training the model to respond "No") and 400 positive samples (training the model to respond "Yes") are sampled.

## Key Experimental Results

### Main Results

| Model | Method | Total Images | Total Clusters | Avg. Clusters/Object | Avg. Images/Cluster |
|-------|--------|-------------|---------------|---------------------|---------------------|
| PaliGemma | DASH-LLM | 99.3K | 1892 | 5.0 | 52.5 |
| PaliGemma | DASH-OPT | **221.7K** | **3895** | **10.3** | 56.9 |
| LLaVA-NeXT Vicuna | DASH-LLM | 162.4K | 3632 | 9.6 | 44.7 |
| LLaVA-NeXT Vicuna | DASH-OPT | **252.0K** | **4632** | **12.2** | 54.4 |
| LLaVA-NeXT Mistral | DASH-OPT | 133.8K | 3229 | 8.5 | 41.5 |

### Ablation Study / Transfer & Benchmark

| Configuration | DASH-B Acc. | DASH-B TNR | POPE TNR | Notes |
|---------------|-------------|------------|---------|-------|
| PaliGemma2-3B | 68.9% | 40.9% | 97.3% | DASH-B is far harder than POPE |
| Ovis2-8B | 71.4% | 44.8% | 94.9% | Hallucination remains severe |
| LLaVa-OneVision | 75.1% | 60.1% | 95.8% | — |
| GPT-4o-mini | **86.3%** | **76.7%** | — | Strongest model still shows 23% false positives |
| PaliGemma (before fine-tuning) | 56.4% | — | 87.2% | — |
| PaliGemma (after fine-tuning) | **68.0%** (+11.6%) | — | 86.4% | Fine-tuning on DASH data is effective |

### Key Findings
- **DASH-OPT substantially outperforms DASH-LLM**: it discovers approximately twice as many clusters and more diverse hallucination patterns (e.g., "leopard print" → leopard, "cartoon frog" → dam).
- **Hallucinations transfer across models**: images discovered via PaliGemma transfer to LLaVA-NeXT Vicuna at a rate of 43–49%.
- **LLM backbone significantly affects hallucination rate**: Vicuna > Mistral > Llama, positively correlated with the model's overall "Yes" tendency.
- **Visual encoder has a larger impact**: CLIP (60% transfer rate) > SigLIP (43%).
- **Model scale has limited impact**: Qwen2-VL 7B (19%) vs. 72B (18%), though the 72B model exhibits fewer hallucinations at the same TPR.
- **POPE benchmark is saturated**: current VLMs achieve 96% TNR on POPE, while TNR on DASH-B is only 48.6%, revealing a large volume of previously overlooked hallucinations.

## Highlights & Insights
1. **Open-world perspective**: First systematic evaluation of VLM hallucinations on a web-scale dataset (ReLAION-5B, 5 billion images), demonstrating that existing benchmarks severely underestimate the problem.
2. **Fully automated, annotation-free pipeline**: A detector replaces manual verification, achieving an error rate (5.2%) far below the label noise of POPE (25.5%).
3. **Elegant design of DASH-OPT**: A distilled single-step diffusion model enables computationally efficient generation; latent-space optimization ensures natural rather than adversarial images.
4. **Practical value**: Discovered systematic hallucination patterns can be directly used for fine-tuning (DASH-B accuracy improves by 11.6%).

## Limitations & Future Work
1. Although ReLAION-5B is large, it does not cover all natural image distributions; some scenarios may lack sufficient semantic neighbors to form clusters.
2. The conservative detector threshold may introduce bias when evaluating state-of-the-art VLMs whose capabilities approach those of the detector.
3. The current fine-tuning strategy is task-isolated and not integrated into the standard VLM training workflow (e.g., curriculum learning).
4. Only false-positive hallucinations (incorrectly responding "Yes") are addressed; false-negative hallucinations (incorrectly responding "No") are not systematically studied.

## Related Work & Insights
- **Relation to Spurious ImageNet**: The latter investigates spurious correlations in image classifiers; DASH extends this concept to VLMs in open-world settings.
- **Relation to DiG-IN**: DiG-IN also employs optimization-guided image generation to debug models, but uses a multi-step diffusion process and targets only classifier discrepancies.
- **Insight**: The discovery of systematic hallucinations suggests that strong co-occurrence biases exist in VLM training data—future work may address this fundamentally through data debiasing at the pre-training stage.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — A novel paradigm for systematic hallucination detection; DASH-OPT is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 380 object categories, multiple VLMs, human verification, transfer analysis, and fine-tuning experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and excellent visualizations; some experimental details require consulting the appendix.
- **Value**: ⭐⭐⭐⭐⭐ — Reveals that VLM hallucination is far more severe than previously recognized; DASH-B has potential to become a new standard benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/hallucination/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](../../ACL2025/hallucination/automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ICCV 2025\] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context](why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte.md)
- [\[ICCV 2025\] ChartCap: Mitigating Hallucination of Dense Chart Captioning](chartcap_mitigating_hallucination_of_dense_chart_captioning.md)

</div>

<!-- RELATED:END -->
