---
title: >-
  [Paper Note] SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding
description: >-
  [ICML 2026][3D Vision][Spiking Neural Networks] SVL utilizes "3D-Image-Text" tri-modal contrastive pretraining to endow Spiking Neural Networks (SNNs) with open-world understanding capabilities. By "re-parameterizing" th…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Spiking Neural Networks"
  - "3D Open-World Understanding"
  - "Vision-Language Pretraining"
  - "Tri-modal Alignment"
  - "Neuromorphic Hardware"
date: 2026-05-08
content_hash: ffb01dfaadad9f52
---

# SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2505.17674](https://arxiv.org/abs/2505.17674)  
**Code**: Yes (Note: "Code is available at SVL")  
**Area**: 3D Vision / Multimodal VLM / Spiking Neural Networks  
**Keywords**: Spiking Neural Networks, 3D Open-World Understanding, Vision-Language Pretraining, Tri-modal Alignment, Neuromorphic Hardware

## TL;DR
SVL utilizes "3D-Image-Text" tri-modal contrastive pretraining to endow Spiking Neural Networks (SNNs) with open-world understanding capabilities. By "re-parameterizing" the text encoder into a set of classification weights, the inference phase completely detaches from the text tower, maintaining a pure spike-driven operation. It achieves 85.4% zero-shot classification on ModelNet40 while consuming only 0.5%–11% of the energy of comparable ANN methods.

## Background & Motivation

**Background**: SNNs are considered natural low-power alternatives for 3D spatio-temporal perception (point clouds, event streams) due to their event-driven and sparse addition properties. Neuromorphic chips like Speck can consume as little as 0.7 mW. However, compared to ANNs, SNNs remain in the stage of "training small models individually for specific tasks."

**Limitations of Prior Work**: Existing SNN pretraining routes exhibit significant flaws—STDP initialization fails rapidly as network/data complexity scales; knowledge distillation methods like SpikeBert/SpikeCLIP rely on ANN weight initialization and use LayerNorm, making them neuromorphic-unfriendly; SpikformerV2 / Spike-driven Transformer V3 improve scalability via masked image modeling but lack multimodal interfaces; classic 3D VLMs (OpenShape, ULIP series) provide open-world 3D classification but require a bulky text encoder (tens to hundreds of millions of parameters) during inference, negating the power efficiency of SNNs at the edge.

**Key Challenge**: There is a fundamental conflict between the "low power + sparse addition" nature of SNNs and the "heavy text tower + dense matrix multiplication" of classic three-encoder VLMs in the inference path. Sacrificing multimodal capability for efficiency or efficiency for zero-shot capability creates a dilemma.

**Goal**: (i) Design a pretraining framework capable of "unlabeled" 3D-Image-Text tri-modal alignment while remaining purely spike-driven; (ii) Entirely eliminate the text encoder during the inference phase; (iii) Develop a truly "fully-spiked" point cloud Transformer backbone as part of the framework.

**Key Insight**: Since CLIP has already aligned "Image ↔ Text," the spike-based 3D encoder only needs to be aligned to CLIP's image space (fine-grained) and text space (semantic-level). Furthermore, during zero-shot tasks, the text encoder essentially computes embeddings for a fixed set of category prompts repeatedly; this implies it can be "folded" offline into a layer of $K \times C$ linear classification weights, allowing the text tower to be discarded during deployment.

**Core Idea**: Use tri-modal contrastive loss to align spiking 3D features with the frozen CLIP image and text spaces (MTA), then re-parameterize the text embeddings as a classification weight layer (Rep-VLI). Combined with a fully-spiked point cloud Transformer (Spike-driven PointFormer), this realizes "multimodal training, pure-spike inference."

## Method

### Overall Architecture
The input treats point clouds and event streams uniformly as point sets $D^t=\{\mathcal{P}, \mathcal{F}\}$. Event streams use a sliding window to normalize timestamps as z-coordinates $z_i = (t_i - t_{\min})/(t_{\max}-t_{\min})$, forming "spatio-temporal point clouds." During pretraining, triplets $(D_i^t, I_i^t, T_i^t)$ are constructed with three parallel paths: CLIP text encoder $\mathcal{E}_\theta^T$ outputs $\mathcal{F}^T \in \mathbb{R}^C$, CLIP image encoder $\mathcal{E}_\theta^I$ outputs $\mathcal{F}^I \in \mathbb{R}^C$ (both frozen throughout), and the spiking 3D encoder $\mathcal{E}_\theta^S$ outputs $\mathcal{F}^S \in \mathbb{R}^{T \times C}$. The spike firing rate $\mathcal{F}^S/T$ is aligned to both text and image spaces—this is MTA. For downstream inference, candidate category prompts are fed into $\mathcal{E}_\theta^T$ offline to obtain classification weights $W^L \in \mathbb{R}^{K \times C}$ (Rep-VLI). The inference link then consists only of the spike encoder and a single addition-based classification head. Regarding the backbone, the authors propose the first "fully spike-driven" point Transformer (Spike-driven PointFormer), where 3D-SDSA spikes all QKV components, reducing attention operations to sparse additions.

### Key Designs

1.  **MTA — Multi-scale Tri-modal Alignment**:
    - **Function**: Pulls spiking 3D features into CLIP's image and text spaces simultaneously without category labels, granting SNNs open-world recognition capabilities in one step.
    - **Mechanism**: Symmetric InfoNCE between normalized spike firing rate $\mathbf{x}_i = (\mathcal{F}^S/T) / \|\mathcal{F}^S/T\|_2$ and normalized text features $\mathbf{y}_i$ for "semantic-level" alignment $\mathcal{L}^{\text{NCE}}_{(S,T)}$ (Eq. 6); InfoNCE with normalized image features $\mathbf{b}_i$ for "fine-grained" alignment $\mathcal{L}^{\text{NCE}}_{(S,I)}$ (Eq. 7); and an additional MSE loss $\mathcal{L}^{\text{MSE}}_{(S,I)} = \sum_i \|\mathcal{F}_i^S - \mathcal{F}_i^I\|^2$ between spikes and images to further solidify granularity. Total loss: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}^{\text{NCE}}_{(S,T)} + \lambda_2 \mathcal{L}^{\text{NCE}}_{(S,I)} + \lambda_3 \mathcal{L}^{\text{MSE}}_{(S,I)}$.
    - **Design Motivation**: Ablations show that using only spike-text alignment results in only 21.9% zero-shot accuracy (Objaverse-LVIS), and spike-image alone results in 24.8%. Combining both with MSE reaches 33.6%—indicating that text provides semantic anchors while images provide fine-grained shape information, both being essential. MSE addresses the coarse granularity of InfoNCE in small minibatches.

2.  **Rep-VLI — Folding the Text Encoder into Classification Weights**:
    - **Function**: Enables the inference phase to discard the text tower while maintaining zero-shot capabilities, keeping the inference path sparse/spike-driven for neuromorphic hardware.
    - **Mechanism**: The $K$ candidate category prompts $\{T_1,\dots,T_K\}$ are passed through the text encoder offline once to derive $W^L_j = \tau \mathcal{E}_\theta^T(T_j)$ (Eq. 10), synthesizing a $K\times C$ weight layer. Inference uses "spike count decision" instead of softmax: for input $D_i$, $\text{logits}_{i,j} = \frac{1}{T}\sum_{t=1}^T W^L_j \cdot \mathcal{E}_\theta^S(D_i^t)$ (Eq. 11), with argmax as the prediction.
    - **Design Motivation**: Traditional three-encoder VLMs often have text towers accounting for 70%+ of total parameters (e.g., ULIP-2's text tower is 202.5M vs. its point encoder at 21.9M). This consumes memory and breaks the sparse addition property. Rep-VLI compresses "text computation" into an offline projection, merging zero-shot capability from CLIP with SNN efficiency.

3.  **Spike-driven PointFormer + 3D-SDSA**:
    - **Function**: Fills the gap for a "fully spike-driven point cloud Transformer" that can handle large-scale pretraining while remaining end-to-end spike-driven.
    - **Mechanism**: Uses FPS+kNN for local neighborhoods $X = \text{KNN}(\text{FPS}(P))$; additive pointwise embedding + I-LIF neurons $\mathcal{SN}(\cdot)$ produce spike features $S = \mathcal{SN}(\text{MLP}(X))$; stacks $L$ layers of SDF residual blocks $f_\ell = \text{SDF}(f_{\ell-1}) + f_{\ell-1}$. For attention, $Q,K,V$ are spiked into binary matrices via $\mathcal{SN}$. 3D-SDSA computes $\mathcal{SN}(Q_S(K_S^\top V_S)) = \mathcal{SN}((Q_S K_S^\top) V_S)$ (Eq. 16). Since multipliers are 0/1 tensors, matrix multiplications degrade into address-event accumulation (sparse addition).
    - **Design Motivation**: Previous Spike Point Transformers used non-spike operators and temporal unfolding, wasting efficiency; Spike PointNet/E-3DSNN were limited by point-level or sparse convolutional inductive biases. 3D-SDSA migrates global modeling to the spike domain fully for the first time.

### Loss & Training
Pretraining loss is the weighted sum in Eq. 9 with $\lambda_1=\lambda_2=\lambda_3=1$. I-LIF neurons emit integers during training (Eq. 4, $D^t$ controls max amplitude) and are unfolded into binary spikes during inference. CLIP encoders are frozen. The default timestep is $T\times D = 1\times 4$. For downstream DVS tasks, $T$ is increased to 6 to capture temporal dynamics.

## Key Experimental Results

### Main Results

Zero-shot 3D Classification on ModelNet40 / Objaverse-LVIS (Excerpt from Tab 1, "Energy" is estimated inference link energy):

| Type | Method | Params (M, Point+Text) | Energy (mJ) | Obj. | M40. |
|------|------|---------------------|-----------|------|------|
| ANN | OpenShape (Sparseconv-L) | 41.3+202.5 | 73.8 | 43.4 | 83.4 |
| ANN | ULIP-2 (Point-BERT) | 21.9+202.5 | 152.3 | 50.6 | 84.7 |
| ANN | ULIP (Point-BERT) | 21.9+227.8 | 161.8 | 34.9 | 69.6 |
| SNN | SpikeCLIP* | 9.5+22.8 | 11.0 | 0.5 | 5.1 |
| SNN | Spike PointNet + SVL | 3.57 | **0.27** | 24.9 | 76.3 |
| SNN | Spike-driven PointFormer-L + SVL | 22.1 | 9.4 | 43.4 | 83.1 |
| SNN | E-3DSNN-L + SVL | 17.7 | 0.64 | 43.9 | 84.6 |
| SNN | E-3DSNN-H + SVL | 46.7 | **0.79** | **47.0** | **85.4** |

**Main Comparison**: E-3DSNN-H + SVL achieves 85.4%, surpassing both ULIP-2 (84.7%) and OpenShape (83.4%), while using only $\sim$0.5% of ULIP-2's energy (0.79 mJ vs. 152.3 mJ). Compared to OpenShape, it offers +2.0% accuracy with $\sim$11× energy efficiency.

3D Downstream Task Fine-tuning (Tab 3/4 Excerpt, "↑" denotes gain over same backbone without SVL):

| Backbone | M40 ↑ | ScanObjectNN ↑ | KITTI AP-E ↑ | SemanticKITTI mIoU ↑ | DVS Action ↑ | DVS Gesture ↑ |
|------|-------|----------------|--------------|---------------------|-------------|---------------|
| Spike PointNet + SVL | +1.9 (90.1) | +6.1 (76.1) | – | +2.1 (15.6) | +2.1 (80.5) | +1.6 (98.5) |
| E-3DSNN-L + SVL | +2.5 (93.7) | +2.8 (83.0) | +1.1 (90.7) | +1.2 (69.7) | – | – |
| Spike-driven PointFormer-L + SVL | +1.8 (93.9) | +1.7 (83.4) | – | – | – | – |

### Ablation Study

MTA Loss Terms (Tab 6, Obj. / M40 Zero-shot accuracy, Backbone E-3DSNN-S):

| $\mathcal{L}^{\text{NCE}}_{(S,T)}$ | $\mathcal{L}^{\text{NCE}}_{(S,I)}$ | $\mathcal{L}^{\text{MSE}}_{(S,I)}$ | Obj. | M40 | Note |
|:--:|:--:|:--:|------|-----|------|
| ✗ | ✗ | ✗ | 0.5 | 5.1 | No alignment (Random init) |
| ✗ | ✓ | ✗ | 24.8 | 73.1 | Image alignment only |
| ✓ | ✗ | ✗ | 21.9 | 70.1 | Text alignment only (weaker) |
| ✓ | ✓ | ✗ | 31.7 | 77.8 | Dual alignment |
| ✓ | ✓ | ✓ | **33.6** | **79.6** | +MSE refinement |

Timesteps / Firing Bits (Tab 7, E-3DSNN-S backbone):

| Configuration | Power (mJ) | Obj. | M40 |
|------|-----------|------|-----|
| ANN baseline | 0.13 | 34.1 | 81.3 |
| $T\times D = 1\times 2$ | 0.02 | 32.9 | 78.5 |
| $T\times D = 2\times 1$ | 0.03 | 32.7 | 78.0 |
| $T\times D = 2\times 2$ | 0.08 | 33.9 | 80.5 |
| $T\times D = 1\times 4$ | 0.04 | 33.6 | 79.6 |
| $T\times D = 4\times 1$ | 0.10 | 32.9 | 78.6 |

### Key Findings
- **Image alignment is more critical than text alignment**: In MTA ablations, removing spike-text alignment drops Obj. from 33.6→24.8 (−8.8), whereas removing spike-image drops it to 21.9 (−11.7). Image alignment alone yields 24.8—indicating shape priors in CLIP's image space contribute more to 3D representation than text-based category anchors.
- **MSE acts as a "glue"**: InfoNCE dual alignment reaches 31.7, while adding MSE gains +1.9. This suggests InfoNCE granularity is coarse in minibatches; MSE forces spike means to match image embeddings for point-to-point global alignment.
- **Increasing bits is more cost-effective than increasing timesteps**: $T=1, D=4$ consumes only 0.04 mJ for 33.6 Obj., while $T=4, D=1$ doubles energy to 0.10 mJ and drops accuracy to 32.9. Compressing information into "stronger" fewer spikes is more efficient than temporal extension.
- **SVL provides smaller gains for Spike-driven PointFormer**: Since PointFormer-L already achieves 92.1 alone, SVL only adds +1.8. In contrast, the weak Spike PointNet sees a +6.1 gain on ScanObjectNN, demonstrating SVL's "amplification effect" on smaller backbones.

## Highlights & Insights
- **"Re-parameterizing the text encoder" is the key engineering insight**: It transforms the open-vocabulary VLM from a symmetric dual-tower structure into "tri-tower training, single-tower + linear layer inference." This is transferable to any downstream model relying on CLIP text for zero-shot classification, especially under hardware constraints (mobile, edge IoT).
- **Freezing CLIP + Training only the 3D encoder** is significantly cheaper than joint training from scratch. This separates "acquiring open-world semantics" from "acquiring 3D geometry," allowing a lightweight SNN to handle just the "projection" task.
- **3D-SDSA exploits the 0/1 nature of $Q_S K_S^\top$ to restore attention to sparse accumulation**. This logic can be applied to all "naturally sparse" modalities like event cameras, SAR, and LiDAR.
- **Treating event streams as z-normalized point clouds** allows event and point cloud data to share a single backbone, a simple and effective unified representation.

## Limitations & Future Work
- **Freezing CLIP binds the 3D space to 2D CLIP's semantic structure**—CLIP's inherent biases (e.g., poor recognition of specific or long-tail classes) will propagate to the 3D side.
- **Zero-shot focus remains on single-object classification**: Benchmarks like Obj./M40 use "cleaned" single instances. Real-world "scene-level zero-shot segmentation/detection" lacks strong data; Spike PointNet+SVL achieves only 15.6% mIoU on SemanticKITTI.
- **Descriptions/QA use an "LLM via LLaVA pipeline"**: Energy calculations only consider the spike encoder, not the heavy LLM. Including a 13B Vicuna would dilute the SNN's efficiency advantage.
- **Rep-VLI assumes the candidate category set is fixed at deployment**: Moving to a truly "open-vocabulary" scenario where new classes appear post-deployment would require re-encoding and re-parameterization.
- **Future Directions**: Fine-tune CLIP (e.g., LoRA) to decouple 2D bias; extend Rep-VLI to "hierarchical prompt trees" for incremental classes; hybridize 3D-SDSA with sparse convolutions for dense scene prediction.

## Related Work & Insights
- **vs. OpenShape / ULIP-2**: All use tri-encoder alignment, but OpenShape/ULIP-2 must retain the heavy text tower (202M parameters, $\sim$70 mJ) during inference. SVL's Rep-VLI offline folding results in 100×+ energy efficiency with comparable accuracy.
- **vs. SpikeCLIP**: Both use spikes and CLIP, but SpikeCLIP merely distills the CLIP vision tower into an SNN for 2D classification (only 0.5% on Obj.). SVL addresses 3D and uses explicit tri-modal alignment and a fully-spiked Transformer to reach 47% Obj.
- **vs. E-3DSNN**: E-3DSNN is a task-specific sparse convolutional backbone without pretraining. SVL improves E-3DSNN by 2.8% on ScanObjectNN, proving the efficacy of general pretraining for SNNs.
- **vs. Spike Point Transformer (Wu 2025b)**: Contemporary work that still uses non-spike operators and temporal unfolding. SVL's Spike-driven PointFormer with 3D-SDSA is the "first in the direction" for fully spike-driven attention.

## Rating
- Novelty: ⭐⭐⭐⭐ Tri-modal alignment + Rep-VLM + Fully-spiked PointFormer are individual innovations that together form a comprehensive end-to-end SNN open-world framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers zero-shot/fine-tuning across four downstream task types and three backbone scales. However, scene-level metrics are weak and end-to-end LLM energy is missing.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formulas and clear structure. Some failure baseline comparisons are lacking, and energy details are in the appendix.
- Value: ⭐⭐⭐⭐⭐ Decisively links "Open-World VLM" and "Neuromorphic Hardware." The Rep-VLI concept is applicable across various CLIP-driven downstream models with high industrial potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](../../ICCV2025/3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](../../ICLR2026/3d_vision/egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](../../NeurIPS2025/3d_vision/towards_3d_objectness_learning_in_an_open_world.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
