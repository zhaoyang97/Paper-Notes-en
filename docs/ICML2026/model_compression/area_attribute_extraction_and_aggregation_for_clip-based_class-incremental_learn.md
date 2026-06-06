---
title: >-
  [Paper Note] AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning
description: >-
  [ICML 2026][Model Compression][CLIP] This paper decomposes forgetting in CLIP-based Class-Incremental Learning (CIL) into "attribute extraction drift" and "attribute aggregation drift." It proposes Area…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "CLIP"
  - "Class-Incremental Learning"
  - "Attribute Anchors"
  - "Principal Geodesic Analysis"
  - "Optimal Transport Routing"
date: 2026-05-08
content_hash: 7844205e687ddbdf
---

# AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning

**Conference**: ICML 2026  
**arXiv**: [2605.28809](https://arxiv.org/abs/2605.28809)  
**Code**: https://github.com/LAMDA-CL/ICML2026-AREA  
**Area**: Model Compression / Continual Learning  
**Keywords**: CLIP, Class-Incremental Learning, Attribute Anchors, Principal Geodesic Analysis, Optimal Transport Routing  

## TL;DR
This paper decomposes forgetting in CLIP-based Class-Incremental Learning (CIL) into "attribute extraction drift" and "attribute aggregation drift." It proposes Area, which uses Principal Geodesic Analysis (PGA) to fix visual/textual attribute anchors on the hypersphere, employs lightweight task experts with VIB regularization to stabilize aggregation, and utilizes Optimal Transport (OT) routing for inference, achieving significant improvements in average and final accuracy across nine benchmarks.

## Background & Motivation
**Background**: Class-Incremental Learning requires models to learn new categories sequentially while maintaining recognition of old ones. Vision-Language Models (VLMs) like CLIP provide a powerful shared embedding space. Consequently, many CIL methods freeze the CLIP backbone and only train prompts, adapters, LoRAs, or small task-specific modules to mitigate catastrophic forgetting.

**Limitations of Prior Work**: Classification in CLIP is often formulated as the cosine similarity between image and text embeddings. However, this similarity mixes two processes: which attributes the model extracts from the input and how it weights these attributes for discrimination. When training only on current task data, new classes pull both attribute extraction and weighting, causing attributes related to old classes to be diluted or recombined, leading to forgetting.

**Key Challenge**: Freezing the CLIP backbone reduces parameter drift but does not guarantee stability at the attribute level. As new tasks arrive, the model still needs to introduce new attributes (e.g., wheels, windows, colors, shapes) for new classes and update how these attributes are combined. Without constraints from old data, this update biases toward the current task, disrupting the evidentiary balance for old categories.

**Goal**: The authors aim to explicitly decompose the prediction mechanism of CLIP-CIL into attribute extraction and attribute aggregation. They design stability mechanisms for both: using geometric anchors to fix class-level visual/textual attributes at the extraction end, and using task experts with Information Bottleneck (IB) to reduce task shortcuts at the aggregation end, while avoiding expert mis-selection via distributed task routing.

**Key Insight**: CLIP embeddings are typically normalized to a unit hypersphere. Therefore, using Euclidean PCA to extract attribute directions ignores the spherical geometry. Area utilizes Principal Geodesic Analysis (PGA) to extract class-level attribute bases in the tangent space of the hypersphere, treating these bases as reusable anchors for subsequent tasks.

**Core Idea**: Anchor the visual and textual evidence of each class as a set of spherical attribute directions and allow lightweight experts to learn to aggregate these anchors stably across tasks, rather than repeatedly rewriting the CLIP backbone or old class representations.

## Method
The Area framework is built around two pillars: "extraction stability" and "aggregation stability." Extraction stability addresses the drift of attribute directions, while aggregation stability prevents task experts from learning shortcuts on current data that lead to incorrect attribute weighting.

### Overall Architecture
When the $b$-th task arrives, the model only accesses the current data $\mathcal{D}^b$ without replaying old samples. The CLIP vision encoder $g_v$ and text encoder $g_t$ remain frozen throughout. For each new class, Area obtains normalized visual embeddings from images and generates text embeddings by combining class prompts with fine-grained descriptions from MLLMs.

Subsequently, the model uses PGA to construct class prototypes and attribute bases for both modalities. Once generated, these bases are frozen and added to an attribute bank for reuse. During training, Area introduces lightweight experts for each task, consisting of an attribute scoring branch and a residual refinement branch to combine fixed anchors into task-relevant, drift-resistant discriminative representations.

During inference, which may involve images from any learned task, Area avoids simple point-to-point cosine similarity for task identification. Instead, it treats the input embedding as a Dirac source distribution and the attribute anchors of each task as target distributions. It utilizes Sinkhorn Optimal Transport distance to derive task routing probabilities for a soft fusion of expert predictions.

### Key Designs
1. **PGA Multi-modal Attribute Anchors**:
	- **Function**: Establish fixed attribute subspaces for each category in both modalities to suppress extraction drift.
	- **Mechanism**: For normalized CLIP features of class $c$, the Fréchet mean $\mu_c$ is calculated on the unit hypersphere. Samples are then mapped to the tangent space via a logarithmic map to compute the covariance and extract the top $K$ principal directions as attribute bases. The process is mirrored on the text side using fused prompt and MLLM captions.
	- **Design Motivation**: CLIP cosine embeddings naturally reside on a hypersphere; PGA respects this geometry better than Euclidean SVD. Freezing these directions ensures that fine-grained evidence for old classes is not re-extracted during subsequent training.

2. **Attribute Aggregation Experts and VIB Stabilization**:
	- **Function**: Enable tasks to learn attribute weighting while preventing experts from relying on coincidental shortcuts in the current task.
	- **Mechanism**: The score branch outputs instance-level weights for attributes, while the refinement branch adds detail corrections. Besides standard contrastive loss, the training objective includes two Variational Information Bottleneck (VIB) surrogates: intervention monotonicity (evidence should not increase after occlusion) and invariant compression (attribute evidence across augmented views should stay near the view mean).
	- **Design Motivation**: Forgetting stems from both representation drift and aggregation weights biasing toward new tasks. IB constraints force experts to retain class-relevant attributes while compressing view noise and task shortcuts.

3. **Optimal Transport-based Task Attribute Manifold Routing**:
	- **Function**: Select or blend the most matching task experts during inference to reduce mis-routing caused by cross-task semantic overlap.
	- **Mechanism**: The input embedding forms a source measure, and the attribute bases of each task form empirical target measures. The model calculates entropic OT / Sinkhorn distances using a cosine cost, converts them to task probabilities via a Boltzmann distribution, and performs a weighted sum of expert predictions.
	- **Design Motivation**: Point-to-point similarity is susceptible to local feature drift. OT compares distribution matching between the input and the entire task attribute manifold, making it more robust for task selection in long-sequence incremental learning.

### Loss & Training
The objective includes standard CLIP-style contrastive loss and stable aggregation regularization. The VIB takes the form of minimizing $-I(\mathcal{Z};Y)+\beta I(\mathcal{Z};X)$, optimized via intervention loss and compression loss. The former penalizes occluded views for producing stronger evidence than the original, and the latter constrains scores across multiple viewpoints to stabilize around the mean.

The total stabilization objective is $\mathcal{L}_{stab}=\lambda_{int}\mathcal{L}_{int}+\lambda_{comp}\mathcal{L}_{comp}+\mathcal{L}_{cont}$. ViT-B/16 is used as the frozen backbone. Area introduces approximately 0.52M trainable parameters, comparable to other prompt/adapter-based methods.

## Key Experimental Results

### Main Results
The paper evaluates the method on nine datasets: CIFAR100, CUB200, ObjectNet, ImageNet-R, Aircraft, Cars, Food101, SUN397, and UCF101. Metrics include average accuracy $\bar{\mathcal{A}}$ during the incremental process and final accuracy $\mathcal{A}_B$.

| Dataset / Setting | Metric | Area | Prev. SOTA Result | Gain / Note |
|--------|------|------|----------|------|
| Aircraft B0 Inc10 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 71.03 / 61.78 | RAPF 50.38 / 23.61 | Significant reduction in forgotten knowledge for fine-grained aircraft classification |
| Cars B0 Inc10 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 97.77 / 96.17 | MG-CLIP 88.21 / 79.73 | Attribute anchors are highly effective for fine-grained differences |
| CIFAR B0 Inc10 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 89.24 / 83.69 | MG-CLIP 89.74 / 82.78 | Average accuracy comparable to SOTA, with higher final accuracy |
| CUB B0 Inc20 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 87.69 / 82.14 | RAPF 79.09 / 62.77 | Clear advantage in final accuracy for fine-grained bird species |
| ObjectNet B0 Inc20 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 61.02 / 49.20 | RAPF 53.78 / 34.97 | More robust against strong domain shifts |
| UCF101 B0 Inc10 | $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | 95.54 / 88.71 | RAPF 92.28 / 80.33 | Retains benefits in image-based video action classification |

### Ablation Study
The paper validates designs through component ablation, caption sources, annotation coverage, and routing efficiency.

| Configuration / Analysis | Key Metric | Note |
|------|---------|------|
| Baseline ZS-CLIP | CIFAR B0 Inc10 drops significantly as tasks progress | Zero-shot CLIP alone cannot resist task distribution shifts |
| w/ Attribute | Significant improvement over baseline | Fixed attribute anchors provide stable references for old classes |
| w/ VIB Loss | Further improvement over "w/ Attribute" | IB reduces task shortcuts and view noise |
| w/ OT | Best overall performance | Distributed routing reduces task expert mis-selection |
| OT vs cosine routing | +3.39% accuracy at final stage (100 classes), +2.9 ms/sample overhead | OT provides a favorable accuracy-efficiency trade-off |

| Caption Setting | Aircraft $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | CIFAR $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ | CUB $\bar{\mathcal{A}}$ / $\mathcal{A}_B$ |
|------|---------|---------|---------|
| Area + GPT5 captions | 71.03 / 61.78 | 89.24 / 83.69 | 87.69 / 82.14 |
| Area + LLaVA captions | 70.89 / 60.95 | 88.98 / 83.24 | 86.86 / 81.22 |
| RAPF + GPT5 captions | 50.38 / 23.61 | 86.14 / 78.04 | 79.09 / 62.77 |
| ZS-CLIP + GPT5 captions | 26.66 / 17.22 | 81.81 / 71.38 | 74.38 / 63.06 |

### Key Findings
- Area improves both average and final accuracy across most datasets, particularly in fine-grained or domain-shifted sets like Aircraft, Cars, CUB, and ObjectNet, verifying that attribute anchors protect fine-grained knowledge.
- Performance is not fragile regarding caption sources. While GPT5 is best, switching to LLaVA-v1.6-34B or LLaVA-7B leads to only minor drops; annotating 20% of samples with LLaVA-7B still achieves strong final accuracy.
- PGA anchors are abstract directions in CLIP space rather than strictly human-interpretable labels. Nearest tokens often show coarse semantic links (e.g., "envelope" linked to "red", "inscription"), suggesting "attributes" here are representation directions.
- Efficiency is manageable. Inference latency increases from 16.4 ms/sample (100 classes) to 18.2 ms/sample (300 classes) because routing occurs over task manifolds rather than per-class matching.

## Highlights & Insights
- The core contribution is the decomposition of CLIP similarity into "what to extract" and "how to aggregate." This framing shifts the view of catastrophic forgetting from mere parameter drift to a drift in attribute geometry and evidentiary weighting.
- PGA is highly compatible with CLIP. Since CLIP features reside on a hypersphere via normalization, principal geodesic directions in the tangent space are more natural than Euclidean PCA and better preserve intra-class structure.
- The two VIB surrogates are practical: the intervention constraint suppresses shortcuts by ensuring occlusion doesn't increase evidence, while the compression constraint ensures attributes don't rely on stochastic augmentations.
- OT routing is a transferable trick for continual learning. Unlike common methods that compare a query to a single prototype, OT matches distributions, which is ideal for long-sequence settings with overlapping task boundaries.

## Limitations & Future Work
- The research is limited to CLIP-based CIL with a frozen backbone. Its stability remains to be verified for scenarios requiring massive backbone updates, generative multi-modal models, or complex modality gaps.
- The method depends on MLLM captions for textual attribute enhancement. While experiments show robustness to caption quality, costs and quality remain potential bottlenecks in privacy-sensitive or low-resource domains.
- Semantic interpretability of attribute anchors is coarse. They act more as representation directions; future work on interpretable continual learning would require stronger attribute alignment or manual verification.
- Although OT overhead is currently small, further compression (e.g., Sparse Sinkhorn or shared attribute banks) may be needed for higher task counts or mobile deployment.

## Related Work & Insights
- **vs prompt-based CIL**: Prompting methods primarily adapt the text side. Area decomposes evidence into multi-modal anchors and aggregation experts, addressing attribute drift more directly.
- **vs adapter / LoRA CIL**: These methods balance stability and plasticity via small parameter updates but may still inadvertently change attribute weighting. Area imposes explicit constraints on both extraction and aggregation.
- **vs replay-based CIL**: Replay uses real old samples for constraints, which is generally powerful but faces privacy and storage issues. Area is an exemplar-free approach utilizing attribute anchors to retain old knowledge.
- **vs MG-CLIP / RAPF**: While these methods also leverage VLM priors, Area differs by using hyperspherical PGA for a fixed attribute bank and OT for distributed routing over task manifolds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decouples CLIP-CIL forgetting into dual-drift components and integrates PGA, VIB, and OT effectively.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across nine datasets; more comprehensive tables for all ablation values would be a plus.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive examples, though the "attribute" concept requires the text's explanation to avoid literal interpretation.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for frozen VLM-based incremental learning, especially for fine-grained and exemplar-free scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](../../CVPR2025/model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
