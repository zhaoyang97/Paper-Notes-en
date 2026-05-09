---
title: >-
  [Paper Note] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts
description: >-
  [CVPR 2026][Autonomous Driving][FedDG-ReID] This paper proposes FedBPrompt, a framework that introduces a Body Distribution Aware Visual Prompts Mechanism (BAPM) dividing prompts into Body Part Alignment Prompts and Holistic Full Body Prompts, paired with a Prompt-based Fine-Tuning Strategy (PFTS) that freezes the ViT backbone and trains only lightweight prompt parameters (reducing communication to ~1%), achieving average mAP gains of 3.3% and Rank-1 gains of 4.9% on FedDG-ReID benchmarks.
tags:
  - CVPR 2026
  - Autonomous Driving
  - FedDG-ReID
  - Visual Prompts
  - Body Part Alignment
  - Parameter-Efficient Fine-Tuning
  - ViT
  - Federated Aggregation
date: 2026-05-08
content_hash: d1757d17a506db4d
---

# FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts

**Conference**: CVPR 2026
**arXiv**: [2603.12912](https://arxiv.org/abs/2603.12912)
**Code**: [leavlong/FedBPrompt](https://github.com/leavlong/FedBPrompt)
**Area**: Autonomous Driving
**Keywords**: FedDG-ReID, Visual Prompts, Body Part Alignment, Parameter-Efficient Fine-Tuning, ViT, Federated Aggregation

## TL;DR

This paper proposes FedBPrompt, a framework that introduces a Body Distribution Aware Visual Prompts Mechanism (BAPM) dividing prompts into Body Part Alignment Prompts and Holistic Full Body Prompts, paired with a Prompt-based Fine-Tuning Strategy (PFTS) that freezes the ViT backbone and trains only lightweight prompt parameters (reducing communication to ~1%), achieving average mAP gains of 3.3% and Rank-1 gains of 4.9% on FedDG-ReID benchmarks.

## Background & Motivation

Federated Domain Generalization Person Re-Identification (FedDG-ReID) requires learning domain-invariant representations from multiple decentralized camera domains under a privacy-preserving federated learning framework, enabling generalization to unseen target domains. While ViT has become a dominant backbone due to its strong representational capacity, its **global attention mechanism** exposes two critical failure modes in FedDG-ReID:

**Background distraction**: ViT's global self-attention processes all patch tokens indiscriminately, including both pedestrian and background regions. When background distributions differ significantly across clients (indoor/outdoor, mall/street), the model tends to attend to highly similar backgrounds, causing incorrect matching between pedestrians of different identities due to background similarity.

**Body part misalignment under viewpoint variation**: Camera angle differences across clients (top-down/eye-level, frontal/lateral) cause the spatial positions of body parts to vary substantially across images. ViT's global attention cannot perceive such spatial structural differences, leading to a sharp drop in cross-view feature similarity for the same pedestrian.

Both issues are further amplified by **inter-client data distribution heterogeneity** in the federated setting — each client observes only its local scene data and cannot learn cross-domain invariances through centralized training.

Existing FedDG-ReID methods (e.g., data augmentation in DACS, feature alignment in FedReID) primarily address diversity at the data level, without directly resolving background distraction and part misalignment at the level of model attention mechanisms.

## Method

### Overall Architecture

FedBPrompt is built on ViT-B/16 and consists of two core components:

1. **BAPM (Body Distribution Aware Visual Prompts Mechanism)**: Structured visual prompts that guide attention to focus on pedestrians and align body parts.
2. **PFTS (Prompt-based Fine-Tuning Strategy)**: Freezes the backbone and trains only prompt parameters, substantially reducing federated communication overhead.

### Module 1: BAPM — Body Distribution Aware Visual Prompts Mechanism

**Core Idea**: A set of learnable prompt tokens $\mathbf{P}$ (50 total, dimension $d$) is injected into each ViT layer and divided into **two groups of four subsets**, each serving a distinct function:

**Group 1: Body Part Alignment Prompts (15 tokens, addressing viewpoint misalignment)**

- $\mathbf{P}^{\text{upper}}$ (5 tokens): Interacts only with patch tokens from the upper half of the image → corresponds to head/shoulders.
- $\mathbf{P}^{\text{mid}}$ (5 tokens): Interacts only with patch tokens from the middle region → corresponds to torso.
- $\mathbf{P}^{\text{lower}}$ (5 tokens): Interacts only with patch tokens from the lower half → corresponds to legs/feet.

Spatial regions are defined using an **overlapping partition** strategy (non-rigid segmentation): assuming the image is divided into $H \times W$ patches, the regions are:

$$I_{\text{upper}} = \{j \mid 1 \leq j \leq n/2\}, \quad I_{\text{mid}} = \{j \mid n/4+1 \leq j \leq 3n/4\}, \quad I_{\text{lower}} = \{j \mid n/2+1 \leq j \leq n\}$$

The three regions share 25% overlap to avoid information loss at rigid boundaries.

**Group 2: Holistic Full Body Prompts (35 tokens, addressing background distraction)**

- $\mathbf{P}^{\text{Full}}$: Interacts with **all** image patch tokens without spatial constraints.
- Function: Captures the pedestrian's overall appearance while suppressing cross-client background noise.

**Constrained Attention Mechanism**: Spatial constraints are enforced via a structured attention mask $M$:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

$$M_{ij} = \begin{cases} -\infty & \text{if } (q_i, k_j) \in \mathcal{C}_{\text{mismatch}} \\ 0 & \text{otherwise} \end{cases}$$

where $\mathcal{C}_{\text{mismatch}}$ denotes mismatched pairs: i.e., a Body Part Prompt attending to patches outside its corresponding spatial region.

**Key Design: Free Communication Among Prompts**: The attention mask between any two prompt tokens is always zero, i.e., $M_{ij} = 0, \forall q_i, k_j \in \mathbf{P}$. This means Body Part Prompts and Full Body Prompts can interact freely — part prompts supply structured local information, while holistic prompts integrate it into a coherent full-body representation. This design differs from rigid partitioning methods such as PCB by preserving global consistency.

Each ViT layer maintains independent prompt parameters $\mathbf{P}_{i-1}$, with the inter-layer update rule:

$$[\mathbf{x}_i, \_, \mathbf{E}_i] = L_i([\mathbf{x}_{i-1}, \mathbf{P}_{i-1}, \mathbf{E}_{i-1}])$$

### Module 2: PFTS — Prompt-based Fine-Tuning Strategy

**Design Motivation**: ViT-B/16 contains approximately 86M parameters, making per-round communication costs prohibitive in federated learning for resource-constrained deployments.

**Approach**:
1. The server pre-trains a standard ReID model (without prompts) on centralized data.
2. The pre-trained model is distributed to all clients, with backbone parameters $\Theta_b$ **frozen**.
3. Each client is equipped with randomly initialized BAPM prompt parameters $\Theta_p$ (~0.46M).
4. Clients train only the prompt parameters, with the objective:

$$\mathcal{L}_k(\Theta_p) = \sum_{(x,y) \in D_k} \mathcal{L}_{\text{ReID}}(g(x; \Theta_b, \Theta_p), y)$$

5. After local training, only prompt parameters are uploaded to the server for data-volume-weighted aggregation:

$$\Theta_p^{t+1} = \sum_{k=1}^{K} \frac{|D_k|}{\sum_{j=1}^{K}|D_j|} \Theta_{p,k}^{t+1}$$

**Communication Efficiency**: Only 0.46M parameters are communicated per round (vs. 86M for the full model), reducing overhead to approximately **0.5%**. Significant performance gains are achieved within only a few aggregation rounds.

### Loss & Training

- Supports two modes: **Full-Parameter Training** (entire model + BAPM) and **PFTS Training** (prompts only).
- Standard ReID losses are used (cross-entropy + triplet loss).
- BAPM can be integrated as a plug-and-play module into any ViT-based FedDG-ReID framework.

## Key Experimental Results

### Datasets and Protocols

- **Datasets**: CUHK02 (C2), CUHK03 (C3), Market1501 (M), MSMT17 (MS)
- **Protocol-1**: Leave-One-Out — 3 domains for training, 1 for testing
- **Protocol-2**: Source-domain performance evaluation

### Main Results on Protocol-1 (Table 1)

Using the strongest baseline SSCU (MM 2025) as reference:

| Setting | →M mAP | →M Rank-1 | →C3 mAP | →C3 Rank-1 | →MS mAP | →MS Rank-1 | Avg mAP | Avg Rank-1 |
|---------|--------|-----------|---------|------------|---------|------------|---------|------------|
| SSCU (baseline) | 46.3 | 69.6 | 33.7 | 33.4 | 20.0 | 43.7 | 33.3 | 48.9 |
| +PFTS | 48.9(+2.6) | 72.4(+2.8) | 35.5(+1.8) | 35.8(+2.4) | 21.3(+1.3) | 46.0(+2.3) | 35.2(+1.9) | 51.4(+2.5) |
| +BAPM | **49.1**(+2.8) | **73.4**(+3.8) | **37.4**(+3.7) | **38.4**(+5.0) | **23.4**(+3.4) | **49.5**(+5.8) | **36.6**(+3.3) | **53.8**(+4.9) |

Gains are even more pronounced on weaker baselines — on FedProx, BAPM yields an average improvement of mAP +10.0% and Rank-1 +13.5%.

### Ablation Study (Table 3)

Using SSCU as baseline under the "C2+C3+M→MS" setting:

| Configuration | Holistic | Part Align | mAP | Rank-1 |
|---------------|----------|------------|-----|--------|
| Baseline | — | — | 20.0 | 43.7 |
| +Holistic Only | ✓ | — | 22.9 | 48.2 |
| +Part Align Only | — | ✓ | 22.7 | 48.5 |
| +BAPM (Full) | ✓ | ✓ | **23.4** | **49.5** |

Both prompt groups contribute independently, and their combination yields further improvement. Part Alignment Prompts contribute more prominently in scenarios with large viewpoint variation (→MS).

### Attention Quality Quantification (Table 4)

| Method | Class Token Ins. AUC | RISE Ins. AUC |
|--------|---------------------|---------------|
| SSCU | 0.6160 | 0.6516 |
| +VPs (vanilla prompts) | 0.7103 | 0.7494 |
| +BAPM | **0.7559** | **0.7737** |

Insertion AUC measures the faithfulness of attention maps — BAPM significantly outperforms vanilla (unstructured) visual prompts, confirming that its attention more precisely focuses on pedestrian regions.

### Protocol-2 Source-Domain Performance

BAPM improves cross-domain generalization without degrading source-domain performance; source-domain results also improve noticeably (e.g., FedPav+BAPM on C2: mAP from 66.5 → 74.3).

## Highlights & Insights

- **Elegant structured prompt design**: Unlike VPT and related methods that treat all prompts homogeneously, BAPM assigns explicit spatial semantics to each prompt group — part prompts govern *where to look*, while holistic prompts govern *what to look at*. The combination of functional specialization and free inter-prompt communication is more flexible than rigid partitioning.
- **Overlapping partitions prevent information fragmentation**: The 25% overlap between upper/mid/lower regions addresses the boundary information loss inherent in hard equal-thirds segmentation.
- **High practical value of PFTS**: 0.5% communication overhead with convergence within a few rounds makes this approach highly viable for real-world federated deployment. PFTS alone consistently yields positive gains.
- **Plug-and-play generality**: Consistent improvements across 6 different baseline methods with stable average gains indicate that BAPM addresses a common bottleneck shared across methods.
- **Intuitive and compelling attention visualization**: In Figure 3, part prompts precisely localize their corresponding body regions, holistic prompts cover the full body, while the baseline attention is diffuse — the visual evidence is convincing.

## Limitations & Future Work

1. **Spatial partition assumes upright pedestrian pose**: The fixed upper/mid/lower ratio assumes pedestrians are approximately upright and centered. For non-standard postures (bending, sitting, severe occlusion), fixed partitioning may fail. Adaptive partitioning or pose-estimation-guided dynamic partitioning could be explored.
2. **Evaluation limited to four ReID datasets**: CUHK02/03, Market1501, and MSMT17 are classic but relatively clean datasets. Performance under more challenging real-world conditions (low illumination, extreme weather, ultra-high crowd density) warrants further investigation.
3. **Sensitivity to prompt count**: The paper uses 50 prompts (split 15+35), and although the appendix includes a sensitivity analysis, the optimal allocation ratio may depend on dataset and domain count, with no adaptive adjustment mechanism proposed.
4. **Dependence on pre-trained model quality**: PFTS relies on a high-quality pre-trained model as initialization. Whether prompt fine-tuning alone can compensate for a poor initial model remains unclear.
5. **Extension to cross-modal and more domains**: Validation is conducted solely on RGB visible-light images; whether body distribution features remain effective in cross-modal ReID (infrared–visible) is worth exploring.

## Related Work & Insights

- **VPT (Visual Prompt Tuning)**: Seminal visual prompting method → FedBPrompt's key innovation lies in endowing prompts with spatial structural semantics.
- **PromptFL**: Prompt-based communication in federated learning → FedBPrompt transfers this concept from NLP to visual ReID with a task-tailored prompt structure.
- **PCB/MGN and other part-based models**: Traditional pedestrian part alignment methods → BAPM achieves soft partitioning via prompts and attention masks, offering greater flexibility than physical cropping.
- **SSCU (MM 2025)**: Current FedDG-ReID state of the art → BAPM achieves consistent gains of 3–5% on top of this strong baseline.
- **DACS (AAAI 2024)**: Data augmentation route → BAPM is complementary, operating at the model attention mechanism level.
- **Broader Inspiration**: The structured prompt paradigm may generalize to other vision tasks requiring spatial alignment (e.g., fine-grained recognition, medical image analysis). The extremely low communication overhead of PFTS has wide applicability to edge-device federated learning scenarios.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 3.5 | Combining visual prompts with body-part spatial semantics is a meaningful design, though the overall framework is a natural extension of VPT. |
| Value | 4.5 | 99%+ reduction in communication with plug-and-play usability yields high practical deployment value. |
| Experimental Thoroughness | 4.0 | Six baselines, two protocols, complete ablations, attention visualization and quantification; detailed computational cost comparisons are lacking. |
| Writing Quality | 4.0 | Problem formulation is clear, method description is well-structured, and equations are complete; the conclusion section is somewhat brief. |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](neural_distribution_prior_for_lidar_ood_detection.md)
- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

<!-- RELATED:END -->
