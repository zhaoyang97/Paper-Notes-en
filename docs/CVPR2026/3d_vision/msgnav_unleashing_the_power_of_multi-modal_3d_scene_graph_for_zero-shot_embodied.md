---
title: >-
  [Paper Note] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
description: >-
  [CVPR 2026][3D Vision][Embodied Navigation] This paper proposes a Multi-modal 3D Scene Graph (M3DSG) that replaces conventional text-based relation edges with dynamically assigned image edges to preserve visual informati…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Embodied Navigation"
  - "3D Scene Graph"
  - "Zero-Shot Navigation"
  - "Multi-modal Scene Graph"
  - "Viewpoint Decision"
  - "VLM"
date: 2026-05-08
content_hash: 3683bef5cd1344b0
---

# MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation

**Conference**: CVPR 2026
**arXiv**: [2511.10376](https://arxiv.org/abs/2511.10376)  
**Code**: N/A  
**Area**: 3D Vision / Embodied Navigation
**Keywords**: Embodied Navigation, 3D Scene Graph, Zero-Shot Navigation, Multi-modal Scene Graph, Viewpoint Decision, VLM

## TL;DR

This paper proposes a Multi-modal 3D Scene Graph (M3DSG) that replaces conventional text-based relation edges with dynamically assigned image edges to preserve visual information. Built upon M3DSG, the zero-shot navigation system MSGNav is constructed, and a Visibility-based Viewpoint Decision (VVD) module is introduced to address the "last-mile" navigation problem. The method achieves state-of-the-art performance on GOAT-Bench and HM3D-ObjNav.

---

## Background & Motivation

Embodied Navigation requires agents to autonomously explore unknown environments and navigate toward goals specified as object categories, language descriptions, or reference images. Traditional RL-based methods suffer from poor generalization and large sim-to-real gaps, making zero-shot approaches more suitable for real-world deployment as they require no training or fine-tuning.

Recent zero-shot navigation methods based on explicit 3D scene graphs and LLM reasoning (e.g., SG-Nav) have shown promising results. However, conventional 3D scene graphs overly abstract object relations into pure text labels (e.g., "top," "beside"), which introduces three critical problems:

**High construction cost**: Frequent MLLM queries for relation inference incur significant token and time overhead.

**Irreversible loss of visual information**: Compressing rich visual observations into text labels increases ambiguity and sensitivity to perceptual errors.

**Limited vocabulary**: Novel object categories outside a predefined vocabulary cannot be represented, constraining generalization.

Additionally, the authors identify a previously overlooked "last-mile" problem: **knowing the target location does not guarantee finding a suitable final navigation viewpoint**. Existing methods typically select the nearest navigable point as the endpoint, but overly close or occluded viewpoints cause task failures—statistics show that a substantial fraction of 3D-Mem failures occur when the agent stops within 0.25 m–1.0 m of the target.

**Core insight**: Visual information is indispensable for real-world navigation; the information bottleneck of text-based scene graphs and the blind spot in viewpoint selection constitute the two primary barriers to further performance improvement.

---

## Method

### Overall Architecture

MSGNav adopts a pipeline of "incremental scene graph construction → efficient reasoning → optimal viewpoint decision," comprising five core modules:

- **M3DSG (Multi-modal 3D Scene Graph)**: A novel scene graph that replaces text relation edges with image edges.
- **KSS (Key Subgraph Selection)**: Extracts goal-relevant subgraphs from a large scene graph.
- **AVU (Adaptive Vocabulary Update)**: Dynamically expands the vocabulary using visual evidence.
- **CLR (Closed-Loop Reasoning)**: Introduces a decision memory for closed-loop reasoning.
- **VVD (Visibility-based Viewpoint Decision)**: Resolves the last-mile problem via visibility scoring.

At each timestep $t$, the agent receives an RGB-D observation $\mathcal{I}_t$ and incrementally updates the scene graph $\mathbf{S}_t$. MSGNav extracts a key subgraph via KSS and drives a VLM to localize the target or determine a frontier exploration direction, with VVD selecting the optimal navigation viewpoint.

### Key Design 1: M3DSG — Multi-modal 3D Scene Graph

**Structural definition**: The scene graph is defined as $\mathbf{S}=(\mathbf{O}, \mathbf{E})$, where $\mathbf{O}$ is the set of objects and $\mathbf{E}$ is the set of edges. The critical distinction is that **each edge stores a set of RGB-D images** $\mathbf{I}_j$ recording the visual context when the corresponding object pair co-occurs, rather than text labels.

**Incremental construction** consists of two sub-processes:

- **Object update**: Open-vocabulary detection with YOLO-W, instance masks with SAM, and visual embeddings with CLIP are applied per frame. Each object stores 8 attributes: ID, category, 3D coordinates, bounding box, mask, point cloud, visual features, and room location. Objects from new frames are matched and merged with existing ones via spatial and visual similarity.
- **Edge update**: For co-occurring object pairs within a distance threshold $\theta$ in the current frame, the frame image is appended to the corresponding edge's image set. A reverse mapping $\mathbf{H}$ from images to object pairs is maintained. **The entire process requires no VLM queries**, achieving high efficiency.

**Three advantages**: (1) Efficient construction—eliminates MLLM relation-inference queries; (2) Visual supplementation—retains raw image evidence for improved robustness; (3) Unbounded vocabulary—supports dynamic vocabulary expansion through visual context.

### Key Design 2: KSS — Key Subgraph Selection

As exploration progresses, the scene graph grows rapidly, making direct VLM input inefficient. KSS extracts goal-relevant subgraphs via three steps:

1. **Compress**: Simplify the scene graph into an adjacency list containing only (ID, category).
2. **Focus**: Feed the compressed graph to the VLM to select the top-$k$ goal-relevant object set $\mathbf{O}^{rel}$.
3. **Pruning**: Apply a greedy dynamic assignment algorithm (Algorithm 1) to iteratively select images covering the most edges, leveraging the reverse mapping $\mathbf{H}$ for efficient set-cover computation.

The resulting key subgraph requires on average **approximately 4 images** to represent the goal-relevant scene context, reducing token overhead by over 95%.

### Key Design 3: AVU — Adaptive Vocabulary Update

Predefined vocabularies (e.g., ScanNet-200) limit open-world generalization. During exploration, AVU prompts the VLM to inspect edge images, compare them with existing objects, and dynamically propose new vocabulary terms $\hat{V}_t$, which are incorporated into the global vocabulary $V_t = V_{t-1} \cup \hat{V}_t$, enabling progressive vocabulary expansion.

### Key Design 4: CLR — Closed-Loop Reasoning

A decision memory $\mathbf{M}$ is introduced to store each exploration decision $\mathcal{R}_t$ for reference in subsequent steps. This is formalized as:

$$\mathcal{R}_t, \hat{V}_t = \text{VLM}(\mathbf{S}^k, \mathbf{M}_t, \mathbf{F}, g, t)$$

By conditioning on historical feedback, CLR forms a closed reasoning loop that prevents repeated erroneous decisions.

### Key Design 5: VVD — Visibility-based Viewpoint Decision

To address the last-mile problem, VVD uniformly samples candidate viewpoints around the target object $\bar{o}$ at multiple radii $\mathbf{R}$. For each candidate viewpoint $\mathbf{v}_i$, visibility to the target point cloud $\mathcal{PC}_{\bar{o}}$ is evaluated via ray casting: points are sampled along the line of sight, and occlusion is determined by checking whether all sampled points have a distance to the nearest scene point cloud greater than an occlusion threshold $\tau$. The visibility score is defined as the fraction of visible points in the target point cloud:

$$S_{\mathbf{v}_i} = \frac{1}{|\mathcal{PC}_{\bar{o}}|} \sum_{\mathbf{p} \in \mathcal{PC}_{\bar{o}}} \mathbb{1}_{\mathcal{E}(\mathbf{v}_i, \mathbf{p})}$$

The viewpoint with the highest score is selected as the navigation endpoint.

### Loss & Training

MSGNav is a zero-shot method requiring no training or fine-tuning. GPT-4o (2024-08-06) is used as the VLM backbone, YOLO-W for open-vocabulary detection, SAM for instance segmentation, and CLIP for visual embeddings. The success distance threshold is 0.25 m for GOAT-Bench and 1.0 m for HM3D-ObjNav.

---

## Key Experimental Results

### Main Results: GOAT-Bench (Multi-modal Lifelong Open-Vocabulary Navigation)

| Method | Training-free | SR(%) ↑ | SPL(%) ↑ |
|---|:---:|:---:|:---:|
| SenseAct-NN Skill Chain | ✗ | 29.5 | 11.3 |
| VLMnav | ✓ | 20.1 | 9.6 |
| DyNaVLM | ✓ | 25.5 | 10.2 |
| 3D-Mem | ✓ | 28.8 | 15.8 |
| TANGO | ✓ | 32.1 | 16.5 |
| MTU3D (trained SOTA) | ✗ | 47.2 | 27.7 |
| **MSGNav (Ours)** | **✓** | **52.0** | **29.6** |

MSGNav surpasses the trained method MTU3D by +4.8% SR and +1.9% SPL without any training.

### Main Results: HM3D-ObjNav

| Method | Training-free | SR(%) ↑ | SPL(%) ↑ |
|---|:---:|:---:|:---:|
| SG-Nav | ✓ | 49.6 | 25.5 |
| VLFM | ✗ | 62.6 | 31.0 |
| DORAEMON | ✓ | 66.5 | 20.6 |
| WMNav | ✓ | 72.2 | 33.3 |
| **MSGNav (Ours)** | **✓** | **74.1** | **33.4** |

### Ablation Study: Module Contributions (GOAT-Bench Val Unseen, First Episode Round)

| M3DSG | VVD | AVU | CLR | Overall SR(%) | Overall SPL(%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | | 28.8 | 20.2 |
| ✓ | | | | 43.8 (+15.0) | 28.0 (+7.8) |
| ✓ | ✓ | | | 56.3 (+12.5) | 34.7 (+6.7) |
| ✓ | ✓ | ✓ | | 55.3 | 36.7 |
| ✓ | ✓ | | ✓ | 53.2 | 32.9 |
| ✓ | ✓ | ✓ | ✓ | **60.0** | **37.0** |

- M3DSG contributes the largest gain (+15.0% SR), followed by VVD (+12.5% SR).
- AVU and CLR show limited or even negative effects individually, but are **complementary**: AVU provides additional perceptual information to supplement CLR's strict decision-making, while CLR filters the noisy perception introduced by AVU.

### Ablation Study: Scene Graph Type Comparison

| Scene Graph Type | Overall SR(%) | Overall SPL(%) |
|---|:---:|:---:|
| Node-only (no relation edges) | 51.8 | 31.2 |
| Traditional graph (text edges) | 56.2 | 32.7 |
| **M3DSG (image edges)** | **60.0** | **37.0** |

Image edges improve over text edges by **+3.8% SR and +4.3% SPL**, with particularly notable gains on Language and Image goal types.

### VVD Module Performance at Different Success Thresholds

| Success Threshold d(m) | w/o VVD SR(%) | w/ VVD SR(%) | Gain |
|:---:|:---:|:---:|:---:|
| 0.25 (standard) | 33.91 | 51.97 | +18.06 |
| 0.55 | 57.44 | 63.03 | +5.59 |
| 1.00 | 62.38 | 66.52 | +4.14 |

### Key Findings

1. M3DSG is the largest contributor, yielding +15.0% SR over the 3D-Mem baseline.
2. VVD recovers approximately 18% absolute SR at the standard 0.25 m threshold, confirming that many failures are indeed caused by suboptimal final viewpoints.
3. AVU and CLR are complementary—open-vocabulary expansion introduces noise, which closed-loop reasoning can correct.
4. Image edges substantially outperform text edges on Language and Image goal types.
5. Viewpoints with VVD visibility scores above 0.6 consistently approximate ground-truth viewpoints.

---

## Highlights & Insights

- **Replacing text edges with image edges** is a concise and effective design choice: it simultaneously eliminates frequent MLLM inference calls and preserves rich visual context. "Less is more"—rather than laboriously abstracting observations into text, allowing the VLM to directly interpret relations from raw images proves more effective.
- **The identification and formalization of the last-mile problem** is a valuable contribution. Navigation success depends not only on reaching the vicinity of the target but on reaching a "good" position, an aspect frequently overlooked in practical robot deployment.
- **The greedy subgraph selection algorithm** achieves 95% token compression, requiring on average only 4 images per query while balancing efficiency and informativeness.
- Surpassing trained methods in a zero-shot setting (52.0% vs. 47.2%) demonstrates the substantial potential of the scene-representation + VLM-reasoning paradigm.
- The overall system exhibits high modularity, with each component independently verifiable through ablation, reflecting mature engineering design.

---

## Limitations & Future Work

1. **Inference efficiency bottleneck**: Online inference with multiple VFMs and VLMs (YOLO-W + SAM + CLIP + GPT-4o) makes real-time deployment challenging. Lighter-weight graph construction and reasoning solutions warrant exploration.
2. **Last-mile problem not fully resolved**: VVD mitigates but does not eliminate the issue; notable gains remain when thresholds are relaxed. Active perception via RL is a promising direction.
3. **VLM dependency**: The system relies heavily on GPT-4o's reasoning capability and API access, incurring high costs and network dependency. Only Qwen-VL-Max is validated in supplementary materials.
4. **Image edge storage overhead**: The memory cost of retaining large numbers of RGB-D images is not discussed and may become a concern during extended exploration.
5. **Simulation-only evaluation**: Real-world physical environment experiments are absent.

---

## Related Work & Insights

- **3D-Mem**: Emphasizes the value of raw images for navigation; M3DSG builds upon this by introducing scene graph structure to organize visual information.
- **ConceptGraphs** (ICRA 2024): A conventional open-vocabulary 3D scene graph using text relation edges; ablation experiments in this paper clearly demonstrate the advantage of image edges.
- **SG-Nav**: Also leverages hierarchical scene graphs for navigation using text-prompted LLMs, but is constrained by text-based relation representations.
- **GOAT-Bench** (CVPR 2024): A multi-modal lifelong navigation benchmark defining three goal types: category, language, and image.

**Insight**: Explicit 3D scene graphs combined with VLM reasoning have become the dominant paradigm for zero-shot navigation. M3DSG suggests that representation design (image vs. text) may be more critical than model selection—preserving raw perceptual data at construction time and allowing the VLM to interpret it on demand at inference time is a design principle with broad applicability.

---

## Rating

| Dimension | Score (1–5) | Notes |
|---|:---:|---|
| Novelty | 4 | The image-edge design is concise and effective; the formalization of the last-mile problem is original. |
| Technical Quality | 4 | Module design is well-motivated, algorithms are clear, and ablations are thorough. |
| Experimental Thoroughness | 4.5 | Two benchmarks + comprehensive ablations + per-category analysis + VVD statistical validation. |
| Writing Quality | 4 | Problem definition is clear, figures are intuitive, and the narrative is logically coherent. |
| Practicality | 3.5 | Inference efficiency remains limited by online calls to multiple large-scale models. |
| **Overall** | **4.0** | A solid, systematic contribution; the M3DSG design is elegant and yields substantial empirical gains. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](../../ICCV2025/3d_vision/diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[CVPR 2026\] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)

</div>

<!-- RELATED:END -->
