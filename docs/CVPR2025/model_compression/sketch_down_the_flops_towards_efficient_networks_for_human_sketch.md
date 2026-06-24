---
title: >-
  [Paper Note] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch
description: >-
  [CVPR 2025][Model Compression][FG-SBIR] This paper is the first to design an efficient inference network tailored specifically to the unique properties of human sketch data. By introducing cross-modal knowledge distillation (SketchyNetV1), a large network is compressed into a lightweight network while maintaining FG-SBIR accuracy. Furthermore, an RL-driven adaptive canvas size selector (SketchyNetV2) is developed to leverage the sparse and abstract nature of sketches…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "FG-SBIR"
  - "Knowledge Distillation"
  - "Canvas Size Selection"
  - "Reinforcement Learning"
  - "Model Efficiency"
date: 2026-05-08
content_hash: 27d93ff4bd63c837
---

# Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch

**Conference**: CVPR 2025  
**arXiv**: [2505.23763](https://arxiv.org/abs/2505.23763)  
**Code**: None  
**Area**: Model Compression / Sketch Retrieval  
**Keywords**: FG-SBIR, Knowledge Distillation, Canvas Size Selection, Reinforcement Learning, Model Efficiency

## TL;DR

This paper is the first to design an efficient inference network tailored specifically to the unique properties of human sketch data. By introducing cross-modal knowledge distillation (SketchyNetV1), a large network is compressed into a lightweight network while maintaining FG-SBIR accuracy. Furthermore, an RL-driven adaptive canvas size selector (SketchyNetV2) is developed to leverage the sparse and abstract nature of sketches, reducing FLOPs even further. Ultimately, this achieves a 99.37% reduction in FLOPs (40.18G $\rightarrow$ 0.254G) with negligible loss in accuracy.

## Background & Motivation

**Background**: Efficiency optimization of models (such as MobileNet, EfficientNet, etc.) is highly mature for photo data. However, efficient inference research remains almost non-existent for human sketch data. Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) is one of the most thoroughly researched and commercially viable tasks in the sketch domain, matching a query hand-drawn sketch to its exact photo instance.

**Limitations of Prior Work**: Direct deployment of lightweight networks designed for photos (such as MobileNetV2) to the FG-SBIR task leads to severe drops in accuracy—MobileNetV2 exhibits a 37% drop in accuracy compared to a VGG-16 baseline. This occurs because the fine-grained nature of sketches (using sparse lines to convey precise visual differences) requires sufficient model capacity to establish cross-modal sketch-to-photo mapping relationships.

**Key Challenge**: Prior efficiency methods ignore two unique characteristics of sketch data: (1) sketches consist of sparse black-and-white lines instead of pixel-dense photos, meaning identical semantic details can be preserved at much lower resolutions; (2) different sketches vary in their levels of abstraction—some convey sufficient information with simple strokes, while others demand fine details. Resorting to a fixed resolution for all sketches is sub-optimal.

**Goal**: Design plug-and-play sketch-specific modules to render general photo-oriented lightweight networks compatible with sketch data. Specifically, the goal is to significantly reduce FLOPs while preserving FG-SBIR retrieval accuracy.

**Key Insight**: (1) Transfer the fine-grained cross-modal retrieval capability of a larger network to a lightweight counterpart via knowledge distillation. (2) Leverage the vector nature of sketch formats to dynamically select the optimal canvas rendering size without incurring interpolation overhead.

**Core Idea**: First, employ relational knowledge distillation (preserving cross-modal distance structures) to compress the model. Then, train a canvas size selector via reinforcement learning to dynamically determine the minimum sufficient resolution for each individual sketch based on its level of abstraction.

## Method

The proposed method consists of two stages. In the first stage (SketchyNetV1), the cross-modal retrieval capability of a VGG-16 teacher network is transferred to a MobileNetV2 student network via knowledge distillation. In the second stage (SketchyNetV2), a GRU-based canvas size selector is built on top of the student network and trained via reinforcement learning.

### Overall Architecture

The inputs are a sketch (either in vector format or rasterized image) and a photo; the outputs represent the sketch-photo joint embedding for retrieval. SketchyNetV1 uses a three-branch Siamese network topology with VGG-16 as the teacher and MobileNetV2 as the student, which is trained jointly via triplet loss and relational distillation loss. SketchyNetV2 prepends a canvas size selector $\psi_C$ before SketchyNetV1, taking the vector sketch as inputs to predict the optimal rendering resolution, which is then rasterized and passed to the retrieval network.

### Key Designs

1. **Cross-Modal Relational Knowledge Distillation (SketchyNetV1)**:

    - **Function**: Transfers the fine-grained cross-modal retrieval capability of a heavy network to a lightweight student network.
    - **Mechanism**: Rather than directly distilling the absolute embedding vectors of the teacher and student (since embedding spaces of different networks might have incompatible dimensionalities and structures), this approach distills the **pairwise distance relationships** among triplets (sketch, positive photo, negative photo) in the teacher's embedding space. Defining distances as $d_{sp}^T = \delta(f_s^T, f_p^T)$ etc., a Huber loss is applied to match corresponding teacher and student distances: $\mathcal{L}_{RKD}^{sp} = \mathcal{L}_\delta(d_{sp}^T, d_{sp}^{st})$. The overall distillation loss is formulated as $\mathcal{L}_{RKD} = \mathcal{L}_{RKD}^{sp} + \mathcal{L}_{RKD}^{sn} + \mathcal{L}_{RKD}^{pn}$. The student's total training objective is given by $\mathcal{L}_{trn}^{st} = \lambda \mathcal{L}_{Tri} + (1-\lambda) \mathcal{L}_{RKD}$. Additionally, the student is trained across diverse canvas sizes to enforce scale invariance.
    - **Design Motivation**: Standard logit distillation is effective for classification, but FG-SBIR is a cross-modal retrieval task, yielding continuous $d$-dimensional embeddings rather than discrete categories. Directly regressing feature vectors is prone to embedding space misalignment. Preserving the relationship structures is more robust—as long as the student space maintains miniature sketch-to-positive distances and large sketch-to-negative distances, accurate retrieval is guaranteed.

2. **Adaptive Canvas Size Selector (SketchyNetV2)**:

    - **Function**: Dynamically evaluates the abstraction level of each sketch to output the optimum rendering resolution.
    - **Mechanism**: A 1-layer GRU is utilized to encode the sketch's vector sequence $s_v = (v_1, ..., v_T)$ (where each $v_t = (x_t, y_t, q_t^1, q_t^2, q_t^3)$ captures coordinates and pen states). The final hidden state is mapped through a linear layer to predict the probabilities of $K$ canvas choices $p(c|s_v)$. The Douglas-Peucker algorithm is adopted to bound sequence lengths. Sampling $c_{pred} \sim \text{categorical}(p(c|s_v))$ yields the targeted size, which is rasterized and input to SketchyNetV1.
    - **Design Motivation**: Empirical results demonstrate that sketches remain highly performant at lower resolutions (dropping only 4% at 32×32 compared to a 15% drop for photos). Crucially, optimal resolutions vary across sketches—30% of sketches retrieve perfectly at 32×32. Since rasterization is non-differentiable, RL is utilized. Using a vector encoder is lightweight and mathematically suited to represent abstract visual semantics.

3. **Reinforcement Learning Reward Design**:

    - **Function**: Signals training targets that balance retrieval accuracy and computational efficiency.
    - **Mechanism**: The accuracy reward is formulated as $R_{acc} = \lambda_r(1/r) + \lambda_{Tri}(-\mathcal{L}_{Tri})$, where $r$ denotes retrieval rank. The computation reward is defined as $R_{comp} = -\mathcal{L}_F$, where $\mathcal{L}_F = \frac{\sum_j q_j \cdot \eta_j}{q_{max} - q_{min}}$ corresponds to the FLOPs-weighted normalized computation penalty. The total reward is $R_{Tot} = \lambda_F R_{comp} + (1-\lambda_F) R_{acc}$. Policy gradients are used to optimize the policy network: $\mathcal{L}_{PG}(\theta) = -\frac{1}{B}\sum_i \log p(c|s_v^i) \cdot R_{Tot}^i$.
    - **Design Motivation**: There are no ground-truth canvas sizes (ill-posed), making brute-force labeling infeasible. The RL reward neatly unifies high accuracy (which favors large canvases) and low computational FLOPs (which favor small canvases) into a single, straightforward optimization term.

### Loss & Training

SketchyNetV1 Training: $\mathcal{L}_{trn}^{st*} = \frac{1}{4}\sum_{i=1}^4 \mathcal{L}_{trn}^{st(c_i)}$, averaged across multiple canvas sizes. SketchyNetV2 Training: Freeze SketchyNetV1, and train only the GRU-based selector using Policy Gradients. VGG-16 (ImageNet pre-trained) serves as the teacher, and MobileNetV2 serves as the student.

## Key Experimental Results

### Main Results

| Method | Params | ShoeV2 Top1 | ShoeV2 FLOPs |
|------|--------|-------------|-------------|
| VGG-16 (Triplet-SN) | 14.71M | 28.71% | 40.18G |
| MobileNetV2 (Directly Trained) | 2.22M | 20.85% | 0.83G |
| **SketchyNetV1** (KD) | 2.22M | 28.46% (↓0.25) | 0.833G |
| **SketchyNetV2** (KD+RL) | 2.27M | 27.89% (↓0.82) | **0.264G** |
| VGG-16 (HOLEF-SN) | 9.31M | 31.74% | 5.758G |
| **SketchyNetV1** (HOLEF) | 2.22M | 31.59% (↓0.15) | 0.833G |

### Ablation Study

| Configuration | ShoeV2 Top1 | FLOPs |
|------|------------|-------|
| VGG-16 (Full Resolution) | 33.03% | 40.18G |
| MobileNetV2 (Directly Trained 256×256) | 20.85% | 0.833G |
| SketchyNetV1 (KD) | 28.46% | 0.833G |
| SketchyNetV2 (KD+RL) | 27.89% | 0.254G |
| **FLOPs Reduction Rate** | - | **99.37%** |
| **Parameter Reduction Rate** | - | **84.89%** |

### Key Findings

- Directly training MobileNetV2 for FG-SBIR yields a poor accuracy of 20.85%. However, relational KD recovers this to 28.46% (comparable to VGG-16 baseline of 28.71%), while reducing FLOPs by 48 times.
- The canvas selector further reduces FLOPs from 0.833G to 0.254G (3.28× compression), with only $0.57\%$ loss in Top-1 accuracy.
- Sketches exhibit significantly higher tolerance to lower resolutions compared to photos: at 32×32, sketch accuracy drops by 4%, while photo accuracy drops by 15%.
- Roughly 30% of the sketch samples retrieve successfully using the lowest resolution (32×32), validating the rationale for adaptive selections.
- The proposed scheme shows strong generalizeability across multiple baselines (Triplet-SN, HOLEF-SN) and datasets (ShoeV2, ChairV2, Sketchy, FSCOCO).

## Highlights & Insights

- Directly tackles the resource-heavy aspect of sketch-based vision—generic photo compression approaches do not translate well to sparse sketch inputs.
- Key Insight: Sketches are sparse by nature and inherently robust to lower-resolution inference, a property overlooked in previous research.
- Utilizes sketch vector format to perform resolution changes "for free"—avoiding interpolation artifacts common to photos.
- The two-stage training scheme is decoupled and straightforward; each stage can be deployed independently while achieving synergistic gains when combined.
- The reinforcement learning reward formulation naturally bridges the trade-off between speed and accuracy via a single control hyperparameter $\lambda_F$.

## Limitations & Future Work

- Evaluated exclusively on FG-SBIR. Its applicability to other sketch-based tasks (e.g., sketch generation, sketch segmentation) remains unexplored.
- The canvas selector utilizes a simple 1-layer GRU; substituting it with advanced sequence encoders (e.g., Transformers) may yield further performance gains.
- The distillation process still relies heavily on a high-fidelity teacher model, bounding student performance underneath the teacher's capability.
- Truncating longer sequences via the Douglas-Peucker algorithm potentially compromises fine details in complex stroke architectures.
- The student model is statically designated as MobileNetV2; alternative architectures (e.g., ShuffleNet, GhostNet) have not been evaluated.

## Related Work & Insights

- **MobileNet/EfficientNet**: Representative photo-centric compact architectures that yield sub-optimal results when applied out-of-the-box to sketches.
- **Knowledge Distillation**: Scaled from logit-level distillation to relational knowledge distillation (RKD), adapted here for cross-modal indexing.
- **RL in Sketches**: While RL has been exploited in sketch domains (for key stroke selection, abstraction modeling), this work successfully extends its use to resolution configuration.
- Insight: Efficient models are best derived by tailoring algorithms to the specific modality (e.g., sparseness, abstract style, vector format) instead of executing generic model pruning / quantization.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to investigate sketch efficiency, proposing an innovative canvas selector. |
| Experimental Thoroughness | 4 | Robust cross-dataset validation with thorough exploratory pilot studies. |
| Writing Quality | 4 | Clear motivation, backed by sound pilot data. |
| Value | 4 | Empowers low-cost deployment via a massive 99%+ FLOPs compression. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation](../../ICML2025/model_compression/sketch_to_adapt_fine-tunable_sketches_for_efficient_llm_adaptation.md)
- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](../../ICCV2025/model_compression/multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](../../ICCV2025/model_compression/vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](../../ICML2025/model_compression/lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)
- [\[CVPR 2025\] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)

</div>

<!-- RELATED:END -->
