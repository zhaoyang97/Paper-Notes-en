---
title: >-
  [Paper Note] Scaling Language-Free Visual Representation Learning
description: >-
  [ICCV 2025][Self-Supervised Learning][visual self-supervised learning] By training DINOv2/MAE-series models (1B–7B parameters) on MetaCLIP's 2 billion web images, this work systematically demonstrates that purely visual self-supervised learning (SSL) exhibits superior scaling behavior compared to CLIP in both model and data dimensions, surpassing CLIP on average VQA performance at 5B+ parameters—including OCR/Chart tasks conventionally assumed to require language supervision.
tags:
  - ICCV 2025
  - Self-Supervised Learning
  - visual self-supervised learning
  - CLIP contrastive
  - scaling law
  - VQA evaluation
  - language-free supervision
date: 2026-05-08
content_hash: 0a03d78f43409218
---

# Scaling Language-Free Visual Representation Learning

**Conference**: ICCV 2025
**arXiv**: [2504.01017](https://arxiv.org/abs/2504.01017)
**Code**: [https://davidfan.io/webssl/](https://davidfan.io/webssl/)
**Area**: Self-Supervised Learning / Representation Learning
**Keywords**: visual self-supervised learning, CLIP contrastive, scaling law, VQA evaluation, language-free supervision

## TL;DR
By training DINOv2/MAE-series models (1B–7B parameters) on MetaCLIP's 2 billion web images, this work systematically demonstrates that purely visual self-supervised learning (SSL) exhibits superior scaling behavior compared to CLIP in both model and data dimensions, surpassing CLIP on average VQA performance at 5B+ parameters—including OCR/Chart tasks conventionally assumed to require language supervision.

## Background & Motivation

### Core Problem

**Core Problem**: **Background**: Visual representation learning has evolved along two tracks: CLIP trained on image-text pairs, and SSL methods (e.g., DINOv2, MAE) trained on images alone. Although SSL achieves strong performance on traditional vision tasks such as classification and segmentation, it falls far short of CLIP in multimodal LLM settings (e.g., VQA), especially on OCR and chart understanding. The community widely attributes this gap to the absence of semantic information provided by language supervision. However, an overlooked confound is that CLIP is trained on billions of web images, whereas SSL methods typically rely on millions of ImageNet images. The fundamental question therefore is: does SSL lag behind CLIP because of the *absence of language*, or because of *differences in training data*?

### Starting Point

**Goal**: Is the underperformance of visual SSL methods in multimodal settings caused by the lack of language supervision, or by disparities in training data scale and distribution? If trained on identical data, can SSL match or even surpass CLIP?

## Method

### Overall Architecture
Web-SSL is a family of purely visual self-supervised models (Web-DINO = DINOv2 trained on web data; Web-MAE = MAE trained on web data). Training data: 2 billion web images from MetaCLIP (MC-2B), using images only without text. Model scale ranges from 1B to 7B parameters. Evaluation: frozen visual encoders are assessed on 16 VQA benchmarks via the Cambrian-1 visual instruction tuning pipeline (2-stage MLP + LLM).

### Key Designs
1. **Controlled Experimental Design**: This is an empirical study rather than a methods paper. The core contribution lies in rigorous variable control: SSL and CLIP are trained on the identical MC-2B dataset, with the same model architectures (ViT-1B to 7B) and the same evaluation pipeline (Cambrian-1 + Llama-3 8B), thereby eliminating data disparity as a confounding factor.

2. **Superior Scaling Behavior of SSL over CLIP**: Web-DINO's VQA performance scales approximately log-linearly with model size without saturation (still improving at 7B), whereas CLIP largely plateaus beyond 3B parameters. Along the data axis, Web-DINO 7B improves continuously from 1B to 8B training samples, with particularly sustained gains on OCR & Chart tasks. This indicates substantial untapped scaling potential in SSL.

3. **OCR/Chart Improvement via Data Filtering**: By using SmolVLM2 to filter text-containing images from MC-2B, the authors find that training on just 1.3% text-dense images enables Web-DINO to outperform CLIP trained on the full dataset on OCR & Chart (+4.3%), while also achieving higher average VQA. This demonstrates that data composition—not language supervision—is the key driver of OCR capability.

4. **Emergent Language Alignment**: By computing the intrinsic alignment between SSL encoder features and LLM (Llama-3.1 8B/70B) text features, the study finds that as model scale and data volume increase, SSL encoders naturally learn visually grounded features that are increasingly aligned with language—without any language supervision. This provides empirical support for the Platonic Representation Hypothesis.

### Loss & Training
- **Web-DINO**: Standard DINOv2 training recipe; batch size 3,072; lr 3.5e-4; warmup 100K steps.
- **Web-MAE**: Standard MAE recipe; lr reduced to 1.6e-3 for larger models to prevent divergence.
- **CLIP**: Standard MetaCLIP recipe; batch size 32,768.
- All models trained at 224×224 resolution for a single epoch (each of the 2B images seen once).

## Key Experimental Results

| Model | Params | Data | Avg VQA | OCR & Chart | Vision-Centric | IN1k linear |
|-------|--------|------|---------|-------------|----------------|-------------|
| MetaCLIP ViT-G(HF) | 1B | 12.8B | 54.8 | 37.3 | 58.4 | 86.4 |
| SigLIP SO400M | 400M | 45.0B | 55.4 | 39.5 | 58.9 | 86.5 |
| DINOv2 ViT-g(HF) | 1B | 1.9B | 47.9 | 21.2 | 55.3 | 86.0 |
| **Web-DINO 7B** | **7B** | **8B** | **55.2→59.9** | **39.4→55.1** | **59.1→60.8** | **86.5** |

- Web-DINO at 5B+ parameters surpasses same-data CLIP on average VQA.
- Web-DINO 7B (8B data, 518px) achieves 59.9 average VQA, comparable to SigLIP2 384px (62.0) while using 5× less data.
- Web-DINO 2B with 1.3% text-filtered data outperforms full-data CLIP 2B on OCR & Chart by +4.3%.
- On traditional vision tasks: Web-DINO achieves 86.5% on IN1k linear probe and surpasses MetaCLIP on ADE20K segmentation.

### Ablation Study
- Web-DINO trained on ImageNet does not exhibit scaling behavior, confirming that data diversity is a prerequisite for scaling.
- Web-MAE shows similar scaling trends and stronger OCR & Chart performance, indicating these findings are not specific to DINOv2.
- Progressive resolution adaptation (224→378→518) consistently improves OCR & Chart performance.
- CLIP saturates across all VQA categories beyond 3B parameters, while SSL continues to improve.

## Highlights & Insights
- **Challenges the Consensus that Visual SSL Requires Language Supervision**: This is a paradigm-shift-level finding—language supervision is not necessary; data scale and distribution are the decisive factors.
- **A Visual Analogue of the Bitter Lesson**: Reducing inductive bias (eliminating language supervision) and relying on scale yields better scaling behavior.
- **Rigorous Experimental Design**: Variable control is exemplary—all models share the same data, architecture, and evaluation protocol, lending strong credibility to the conclusions.
- **Insightful Data Filtering Result**: Using only 1.3% text-dense images suffices to surpass CLIP's OCR capability, opening new directions for data composition optimization.
- **Emergent Alignment**: SSL models naturally align with LLMs upon scaling, providing empirical support for the Platonic Representation Hypothesis.

## Limitations & Future Work
- SSL does not support zero-shot classification (which must be achieved indirectly via MLLMs or LiT-style adapters).
- VQA validation is conducted only with Llama-3 8B; larger LLMs may yield different conclusions.
- A 7B encoder is computationally large; efficiency must be considered for practical deployment.
- Regimes beyond 7B parameters and 8B training samples have not yet been explored.
- Data filtering relies on MLLM-based annotation (which indirectly involves language); purely language-free data curation strategies remain to be investigated.

## Related Work & Insights
- **vs. DINOv2**: Same method, different data—Web-DINO trained on web data substantially outperforms ImageNet-trained DINOv2 on VQA, confirming that data is the bottleneck.
- **vs. CLIP/SigLIP**: Under fair comparison, SSL scales better; CLIP may be more data-efficient at small model sizes, but this advantage disappears at large scale.
- **vs. Cambrian-1**: The evaluation framework is adopted from Cambrian-1, while this work contributes new insights into the choice of vision encoders.

## Related Work & Insights
- The findings carry broad implications for the multimodal community: future MLLMs may no longer need to rely on CLIP and could instead adopt scaled-up SSL encoders.
- The data composition optimization strategy (adjusting the proportion of text-dense images) is transferable to other SSL pretraining pipelines.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A paradigm-shift-level finding that challenges the CLIP-dominated visual pretraining paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive scaling study across 1B–7B models × 1B–8B data; 16 VQA + traditional benchmarks; comparisons against CLIP/MAE/DINOv2; thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Findings are organized around five Research Questions; logic is clear; figures and tables are polished and intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Significantly influences the community's understanding of visual representation learning; models are planned for open release; charts a new direction for SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](../../NeurIPS2025/self_supervised/towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)
- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](../../NeurIPS2025/self_supervised/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)

</div>

<!-- RELATED:END -->
