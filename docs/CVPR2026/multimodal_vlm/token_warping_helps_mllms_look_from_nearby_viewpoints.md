---
title: >-
  [Paper Note] Token Warping Helps MLLMs Look from Nearby Viewpoints
description: >-
  [CVPR 2026][Multimodal VLM][viewpoint transformation] This paper proposes performing spatial warping on ViT image tokens within MLLMs—rather than conventional pixel-level warping—to simulate viewpoint changes. It is foun…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "viewpoint transformation"
  - "token warping"
  - "spatial reasoning"
  - "mental imagery"
  - "MLLM"
date: 2026-05-08
content_hash: e52ae9f5c985d144
---

# Token Warping Helps MLLMs Look from Nearby Viewpoints

**Conference**: CVPR 2026
**arXiv**: [2604.02870](https://arxiv.org/abs/2604.02870)
**Code**: [https://token-warping-mllm.github.io/](https://token-warping-mllm.github.io/) (project page)
**Area**: Multimodal VLM
**Keywords**: viewpoint transformation, token warping, spatial reasoning, mental imagery, MLLM

## TL;DR
This paper proposes performing spatial warping on ViT image tokens within MLLMs—rather than conventional pixel-level warping—to simulate viewpoint changes. It is found that backward token warping maintains semantic consistency while remaining robust to depth estimation noise. The proposed method substantially outperforms pixel-level warping, specialized spatial-reasoning MLLMs, and generative warping approaches on the newly constructed ViewBench benchmark.

## Background & Motivation

**Background**: Multimodal large language models demonstrate strong visual reasoning capabilities, yet remain highly vulnerable to viewpoint changes. Even with near-perfect depth estimation, integrating predicted depth into MLLMs fails to yield genuine 3D understanding. MLLMs fine-tuned specifically for spatial reasoning (e.g., SpatialReasoner) show only marginal improvements on viewpoint transformation tasks.

**Limitations of Prior Work**: The conventional approach transforms source images to target viewpoints via pixel-level warping; however, pixel-level operations are extremely sensitive to minor errors in depth maps—even small inaccuracies cause pronounced geometric distortions and semantic degradation (e.g., deformed books, blurred objects) after warping. Generative novel-view synthesis methods (e.g., GenWarp) can synthesize complete images but may hallucinate non-existent objects or omit existing ones.

**Key Challenge**: Viewpoint transformation requires some form of internal representational transformation of the scene, yet there exists a fundamental tension in choosing the granularity of that transformation—object-level representations are too coarse and lose spatial detail, while pixel-level representations are too fine-grained and overly sensitive to noise. An intermediate-granularity representation is needed.

**Goal**: (1) Identify a viewpoint transformation representation robust to depth errors; (2) explore optimal warping strategies (forward/backward, nearest/adaptive); (3) construct a standardized benchmark for evaluating MLLMs' viewpoint reasoning capabilities.

**Key Insight**: Inspired by the "mental imagery" theory in cognitive science—Shepard, Minsky, Pylyshyn, Hinton, and others proposed that mental images rely on "part-based structural descriptions" rather than holistic representations. Image tokens in ViT operate at exactly the intermediate granularity between pixels and objects, making them natural "part-level" representational units.

**Core Idea**: Elevate the viewpoint transformation operation from the pixel level to the token level, leveraging image tokens as robust semantic units for viewpoint transformation to enable nearby-viewpoint reasoning in MLLMs.

## Method

### Overall Architecture
The input consists of a source-view image, its depth map, and source and target camera poses; the goal is to enable the MLLM to answer questions about the scene from the target viewpoint. The core of the method is performing geometric warping on image tokens at the ViT level within the MLLM, rather than operating at the pixel level. The entire pipeline requires no additional training and introduces only minimal warping computation at inference time.

### Key Designs

1. **Robustness of Tokens to Positional Perturbation**:

    - Function: Demonstrate that image tokens constitute an appropriate representational granularity for viewpoint transformation.
    - Mechanism: A "positional noise sensitivity test" is designed—Gaussian displacement perturbations (ranging from 0 to 20 pixels) are applied to the grid center coordinates of each token, and these perturbed positions are then used to fetch patches as ViT inputs. Experiments show that the accuracy of Qwen2.5-VL on CV-Bench-2D remains nearly unchanged even when perturbations approach the patch size. By contrast, applying the same perturbations to pixel-level representations leads to a noticeable performance drop.
    - Design Motivation: This provides theoretical grounding for subsequent token warping—since tokens are insensitive to positional information, positional offsets introduced by depth errors during warping will not seriously impair MLLM understanding.

2. **Backward Token Warping (Core Method)**:

    - Function: Rearrange tokens from the source viewpoint onto a regular grid defined in the target viewpoint.
    - Mechanism: A dense regular grid is defined from the target viewpoint's perspective. Each target grid point is mapped back to the source image plane via the inverse projection function $f_{T \to S}$, locating the corresponding patch/token in the source image. Concretely, a lightweight 3D proxy mesh is constructed from the source image depth map, and ray casting is performed from each grid position in the target viewpoint toward the source image to determine the corresponding source coordinates. Compared to forward warping, backward warping guarantees that tokens at the target viewpoint are densely and regularly arranged—which is critical for MLLMs trained on regular grids.
    - Design Motivation: Forward warping (projecting source tokens onto the target plane) produces sparse, irregular token distributions with numerous holes, causing the MLLM to receive out-of-distribution inputs and suffer severe performance degradation. Backward warping, by starting from the target viewpoint's regular grid, naturally ensures density and regularity.

3. **Nearest vs. Adaptive Fetching**:

    - Function: Determine how to fetch the token corresponding to a target grid point from the source image.
    - Mechanism: Nearest fetching directly selects the existing token in the source image whose grid center is closest (in Euclidean distance) to the mapped coordinate; adaptive fetching re-crops the source image centered at the mapped coordinate and encodes it as a new token. Experiments show that the two approaches achieve comparable performance—nearest fetching is simple and efficient yet loses nothing to adaptive, further validating token robustness to positional offsets.
    - Design Motivation: Nearest fetching avoids the overhead of re-encoding, while adaptive fetching is theoretically more precise but computationally more expensive. The comparable performance of the two provides practical guidance for deployment.

### Loss & Training
This method requires no training and operates purely at inference time—only a single warping transformation of image tokens is applied prior to MLLM inference, with negligible computational overhead.

## Key Experimental Results

### Main Results

Experiments are conducted on the newly constructed ViewBench benchmark, which is based on ScanNet real indoor scenes and evaluates three task categories: Text (spatially-tagged relationships), Shape (geometric shape relationships), and Object (target-view object description).

| Method | ViewBench-Text (5–15%) | ViewBench-Shape (5–15%) | ViewBench-Object (5–15%) |
|--------|----------------------|------------------------|-------------------------|
| SpatialReasoner | 46.73 | 33.72 | — |
| VLM-3R | 63.82 | 49.22 | — |
| GenWarp | 69.35 | 53.10 | 4.32 |
| Pixel Backward | 71.86 | 62.40 | 4.53 |
| Token Backward-Nearest | 74.87 | 67.44 | 4.80 |
| **Token Backward-Adaptive** | **77.89** | **67.44** | **4.97** |
| Oracle (GT Target View) | 100.00 | 100.00 | 6.64 |

### Ablation Study

| Configuration | ViewBench-Text (5–15%) | ViewBench-Shape (5–15%) | Notes |
|---------------|----------------------|------------------------|-------|
| Token Forward | 60.30 | 55.04 | Forward warping produces irregular tokens |
| Token Backward-Nearest | 74.87 | 67.44 | Backward + nearest; strong performance |
| Token Backward-Adaptive | 77.89 | 67.44 | Backward + adaptive; higher cost, marginal gain |
| Pixel Forward | 70.85 | 56.20 | Pixel-level forward |
| Pixel Backward | 71.86 | 62.40 | Pixel-level backward |

### Key Findings
- **Backward > Forward** is the most critical design choice: backward token warping outperforms forward warping by 14.57% on Text (5–15%), as MLLMs require a dense, regular token grid.
- **Token-level > Pixel-level**: Backward token warping surpasses backward pixel warping by ~6% on Text and ~5% on Shape, owing to tokens' greater robustness to depth noise.
- Nearest fetching and adaptive fetching achieve comparable performance, indicating that the robustness of token representations makes precise alignment unnecessary.
- The performance gap between predicted depth and GT depth is minimal, further validating the method's robustness to depth errors.
- All specialized spatial-reasoning MLLMs (SpatialReasoner, VLM-3R, ViLaSR) underperform token warping, demonstrating that spatial fine-tuning cannot substitute for explicit viewpoint transformation.

## Highlights & Insights
- **Elegant integration of cognitive science and engineering design**: The notion of "part-based representations" is drawn from mental imagery theory and mapped to ViT patch tokens, achieving a clean translation from cognitive theory to engineering methodology. This analogy is not merely explanatory—it directly guides the method design.
- **Training-free inference-time augmentation**: The entire method requires no additional training; a single token warping operation at inference time yields substantial improvements in viewpoint reasoning—a "free lunch" approach with high practical value.
- **Importance of regular dense token grids**: The finding that MLLMs are highly sensitive to the spatial distribution pattern of tokens—sparse, irregular tokens (produced by forward warping) constitute severely out-of-distribution inputs—is transferable to other tasks that require manipulating token layouts.

## Limitations & Future Work
- The method handles only **nearby viewpoint transformations** (viewpoints with overlapping coverage); large-angle viewpoint changes cause warping to fail due to extensive occlusions and hole regions.
- A depth map (GT or predicted) is required; while the method is robust to depth noise, the dependency on depth input restricts applicable scenarios.
- ViewBench is based on indoor scenes (ScanNet); generalization to outdoor or dynamic scenes has not been validated.
- Experiments are conducted solely on Qwen2.5-VL; the robustness to token perturbations may vary across MLLM architectures.
- Combination with spatial-reasoning fine-tuning methods remains unexplored—whether token warping combined with SpatialReasoner fine-tuning could yield further improvements is an open question.

## Related Work & Insights
- **vs. SpatialReasoner / VLM-3R**: These methods acquire spatial reasoning capabilities by fine-tuning MLLMs on spatial data, yet the paper finds that fine-tuning cannot replace explicit viewpoint transformation. Token warping substantially outperforms them without any additional training.
- **vs. GenWarp (generative warping)**: Generative methods synthesize target-view images using diffusion models but are prone to hallucinating non-existent objects. Token warping does not generate new pixels but merely rearranges existing tokens, thereby avoiding hallucination.
- **vs. Pixel warping**: A classical 3D vision approach, but sensitive to depth noise. Token warping exploits the coarse granularity of ViT patches to naturally tolerate positional errors.

## Rating
- Novelty: ⭐⭐⭐⭐ The token warping idea grounded in cognitive science is highly creative, though the technical implementation is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ ViewBench is well-designed with comprehensive ablations, but evaluation is limited to indoor scenes and a single MLLM.
- Writing Quality: ⭐⭐⭐⭐⭐ The exposition is clear, the logical chain from theory to experiments is complete, and the figures are intuitive.
- Value: ⭐⭐⭐⭐ Training-free inference-time augmentation carries strong practical value, though applicability is constrained to nearby viewpoint transformations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)
- [\[CVPR 2026\] EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[CVPR 2026\] Asking like Socrates: Socrates helps VLMs understand remote sensing images](asking_like_socrates_socrates_helps_vlms_understand_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
