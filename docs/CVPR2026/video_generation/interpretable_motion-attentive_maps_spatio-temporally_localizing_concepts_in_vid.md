---
title: >-
  [Paper Note] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers
description: >-
  [CVPR 2026][Video Generation][Video Diffusion Models] This paper proposes IMAP (Interpretable Motion-Attentive Maps), a training-free framework that extracts spatio-temporal saliency maps for motion concepts from Video DiTs via two modules: GramCol for spatial localization and motion head selection for temporal localization. IMAP surpasses existing methods on motion localization and zero-shot video semantic segmentation benchmarks.
tags:
  - CVPR 2026
  - Video Generation
  - Video Diffusion Models
  - Interpretability
  - Motion Localization
  - Attention Analysis
  - Saliency Maps
date: 2026-05-08
content_hash: 45d67b7cd3ea6f51
---

# I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2603.02919](https://arxiv.org/abs/2603.02919)
**Code**: [https://github.com/youngjun-jun/IMAP](https://github.com/youngjun-jun/IMAP)
**Area**: Video Generation
**Keywords**: Video Diffusion Models, Interpretability, Motion Localization, Attention Analysis, Saliency Maps

## TL;DR
This paper proposes IMAP (Interpretable Motion-Attentive Maps), a training-free framework that extracts spatio-temporal saliency maps for motion concepts from Video DiTs via two modules: GramCol for spatial localization and motion head selection for temporal localization. IMAP surpasses existing methods on motion localization and zero-shot video semantic segmentation benchmarks.

## Background & Motivation
1. **Background**: Video Diffusion Transformers (e.g., CogVideoX/HunyuanVideo) are capable of generating high-quality videos, yet their internal mechanisms remain poorly understood. Existing interpretability work is predominantly focused on image DiTs.
2. **Limitations of Prior Work**: ConceptAttention provides only spatial disentanglement and does not address motion or temporal information; DiTFlow/DiffTrack focus on inter-frame visual token correspondence but do not analyze how text is translated into motion. A core question remains unanswered: do Video DiTs genuinely understand and create motion?
3. **Key Challenge**: The defining characteristic of video over image is temporal motion information, yet existing saliency map methods perform only spatial localization and cannot answer the critical question of "when and which object is moving."
4. **Goal**: To construct spatio-temporally localized interpretable saliency maps for motion concepts within Video DiTs.
5. **Key Insight**: Analysis of multi-head attention in Video DiTs reveals that QK matching exhibits strong spatial localization capability, and the degree of inter-frame embedding separation correlates with motion localizability. Different attention heads assume distinct roles—certain heads specialize in temporal motion features.
6. **Core Idea**: GramCol is employed for spatial localization (text proxy tokens + Gram matrix), while frame separation scoring selects motion heads for temporal localization.

## Method

### Overall Architecture
The pipeline operates on the MM-Attn modules of a Video DiT. Given a concept word: (1) the most relevant visual token is identified as a text proxy via QK matching; (2) GramCol computes a Gram-matrix-based spatial saliency map; (3) for motion concepts, an additional motion head selection step is applied—the Calinski-Harabasz index (CHI) quantifies inter-frame feature separation, and only the top-$k$ motion heads are retained for IMAP computation. The entire process requires neither gradient computation nor parameter updates.

### Key Designs
1. **GramCol Spatial Localization**:
    - Function: Generates per-frame spatial saliency maps for arbitrary text concepts.
    - Mechanism: For each frame $f_i$, the visual token most aligned with concept $c$ is identified as a text proxy via QK matching: $s_{f_i}^c = \arg\max_p \text{row}_p(q_{f_i})k_c^\top$. GramCol is defined as the $s_{f_i}^c$-th column of the visual Gram matrix $G = h_x h_x^\top \in \mathbb{R}^{P \times P}$, i.e., the similarity vector between all visual tokens and the proxy token. The final map is averaged over selected timesteps, layers, and heads.
    - Design Motivation: Compared to ConceptAttention (which multiplies cross-modal features), GramCol computes similarity within the same modality space, naturally ensuring interpretable "positive highlighting"—regions similar to the proxy token receive large positive values. It also does not require softmax over a concept list and operates on a single concept.

2. **Motion Head Selection**:
    - Function: Identifies attention heads that specialize in motion processing within Video DiTs, enabling temporal localization.
    - Mechanism: For each attention head, visual tokens are partitioned into $F$ clusters by frame, and the CHI is computed to measure inter-frame feature separation. A higher CHI indicates greater inter-frame variation in that head, implying richer temporal motion information. The top-5 highest-CHI heads per layer are selected, and GramCol is computed exclusively from their features to produce IMAP. A Pearson correlation of $r = 0.60$ validates the positive association between CHI and motion localization score.
    - Design Motivation: Motion is inherently inter-frame variation; heads encoding strong motion information naturally exhibit greater inter-frame feature divergence. Compared to aggregating all heads, motion head selection removes the interference of spatial heads, yielding sharper motion localization.

3. **Layer and Timestep Selection**:
    - Function: Narrows the analysis scope to feature-rich layers and timesteps.
    - Mechanism: Early timesteps (close to pure noise, semantically uninterpretable, and prone to memorization-related artifacts such as watermarks) are excluded. Layer selection is based on the second-largest eigenvalue $\lambda_2$ of the attention matrix—under the DTMC framework, a larger $\lambda_2$ indicates a more informative transition matrix. Layers with $\lambda_2 > 0.7$ are selected for CogVideoX and $\lambda_2 > 0.75$ for HunyuanVideo.
    - Design Motivation: The $L \times T$ space of layers and timesteps is large; aggregating all of them dilutes the signal. Automatic selection based on $\lambda_2$ avoids manual hyperparameter tuning.

### Loss & Training
IMAP is entirely training-free and gradient-free, requiring no additional training or parameter updates. For real videos, features are extracted via a noise-adding and denoising process. GramCol requires only a single column of the Gram matrix—the computational cost is an $O(Pd)$ matrix multiplication followed by an $O(P)$ indexing operation. CHI computation is similarly lightweight (ratio of inter-frame to intra-frame variance). The overall additional overhead relative to Video DiT inference is negligible. In practice, the full analysis of a 49-frame video completes within seconds. Implementation details: layers with $\lambda_2 > 0.7$ are used for CogVideoX and $\lambda_2 > 0.75$ for HunyuanVideo; motion head selection is fixed at top-5; only dual-stream MM-DiT blocks are used (single-stream blocks in HunyuanVideo are excluded).

## Key Experimental Results

### Main Results (Motion Localization)

| Method | Backbone | SL | TL | PR | SS | OBJ | Avg |
|--------|----------|-----|-----|-----|-----|-----|------|
| ViCLIP | ViT-H | 0.33 | 0.17 | 0.35 | 0.29 | 0.28 | 0.28 |
| DAAM | VideoCrafter2 | 0.36 | 0.17 | 0.38 | 0.32 | 0.35 | 0.32 |
| ConceptAttn | CogVideoX-5B | 0.50 | 0.32 | 0.51 | 0.47 | 0.47 | 0.45 |
| **IMAP** | CogVideoX-5B | **0.58** | **0.65** | **0.64** | **0.52** | **0.59** | **0.60** |
| ConceptAttn | HunyuanVideo | 0.42 | 0.26 | 0.44 | 0.35 | 0.34 | 0.36 |
| **IMAP** | HunyuanVideo | **0.60** | **0.41** | **0.62** | **0.50** | **0.62** | **0.55** |

### Ablation Study

| Configuration | Avg Score | Note |
|---------------|-----------|------|
| Cross-Attention Map | 0.34 | Baseline attention map |
| GramCol (all heads) | ~0.45 | Spatial localization effective but temporal imprecise |
| GramCol + Layer Selection | ~0.50 | Improvement after excluding low-information layers |
| IMAP (GramCol + Motion Heads) | 0.54–0.60 | Motion head selection yields temporal localization breakthrough |

### Key Findings
- Temporal localization (TL) is IMAP's greatest advantage: on CogVideoX-2B, TL improves from 0.56 (Cross-Attn) to 0.62; on HunyuanVideo, from 0.26 to 0.41.
- GramCol is more stable than ConceptAttention: heterogeneous cross-head behavior in ConceptAttention leads to instability, whereas GramCol's same-modality similarity avoids this issue.
- The effectiveness of motion head selection is validated by the positive correlation between CHI and MLS ($r = 0.60$); random head selection leads to significant performance degradation.
- IMAP also demonstrates strong performance on zero-shot video semantic segmentation.

## Highlights & Insights
- **Elegant design of the text proxy token**: Rather than directly computing cross-modal similarity with text tokens, QK matching is used to identify visual tokens that best represent the textual concept, effectively converting a cross-modal problem into a same-modal one. This paradigm generalizes to any scenario requiring cross-modal localization.
- **Simple assumption: motion equals inter-frame difference**: Cluster separation is used to quantify motion information content with minimal computational overhead (CHI is a lightweight operation), yet proves highly effective. This demonstrates that simple statistical metrics can outperform complex learned methods for feature selection.
- **Insights into Video DiT internal mechanisms**: The paper reveals that distinct attention heads exhibit clear specialization (spatial vs. motion), and layers with larger $\lambda_2$ are more semantically informative—findings that provide guidance for future Video DiT design and optimization.

## Limitations & Future Work
- **Evaluation relies on LLM scoring**: OpenAI o3-pro is used for MLS evaluation with detailed rubrics, but reproducibility and consistency of LLM-based evaluation remain a concern. Human evaluation baselines are absent.
- The localization capability for highly **subtle motion** (e.g., micro-expression changes, slow gradual transitions) is unverified—CHI separation may fail to capture such fine-grained inter-frame differences.
- Validation is currently limited to CogVideoX (2B/5B) and HunyuanVideo; applicability to other architectures (single-stream DiT, cross-attention architectures) requires further experimentation.
- The top-$k = 5$ setting for motion head selection is globally fixed; different videos or motion types may warrant different numbers of heads. Adaptive $k$ selection is a natural improvement direction.
- The $\lambda_2$ layer selection thresholds (0.7 for CogVideoX, 0.75 for HunyuanVideo) are manually specified, lacking an automated selection strategy.
- IMAP is an analysis tool rather than a generation control tool; leveraging discovered motion heads for motion generation and editing control is a promising but unexplored direction.
- The current benchmark (504 videos, 150 motion types) is limited in scale; large-scale evaluation remains to be constructed.
- The ability to disentangle motions of multiple simultaneously moving objects (e.g., two interacting persons) warrants further investigation.

## Related Work & Insights
- **vs. ConceptAttention**: ConceptAttention addresses only spatial disentanglement, and the cross-modal similarity $h_x h_c^\top$ exhibits heterogeneous behavior across heads; GramCol resolves these issues via same-modality Gram matrices and extends the framework to temporal localization. ConceptAttention's softmax operation introduces multi-concept competition, which GramCol avoids entirely.
- **vs. DAAM**: DAAM uses cross-attention maps from U-Net architectures, making it unsuitable for joint-attention DiT architectures; IMAP is designed specifically for DiTs, leveraging MM-Attn QK matching and head-level analysis.
- **vs. DiTFlow/DiffTrack**: These methods focus on inter-frame visual token correspondence (optical flow/tracking), whereas IMAP addresses which visual regions correspond to a specific motion concept described in text. The two approaches are complementary and potentially combinable.
- **Connection to attention head pruning research**: The discovery of motion vs. spatial heads corroborates inference acceleration research on Video DiTs (sparse head pruning), suggesting that more intelligent pruning strategies can be developed without sacrificing motion information.
- **Connection to TokenRank**: This paper adopts the DTMC perspective and $\lambda_2$ importance metric from TokenRank, but extends its application from per-state weighting to per-layer selection—a novel use of the framework.
- **Implications for video editing and control**: The motion heads discovered by IMAP can potentially be leveraged inversely for motion editing—manipulating motion head features to control motion in generated videos without affecting spatial appearance.
- **Insights for video understanding**: This paper is the first to demonstrate that Video DiTs internally contain attention heads specialized for motion processing, which has important implications for understanding the internal mechanisms of video generative models.
- **Potential for zero-shot video semantic segmentation**: GramCol's strong performance on zero-shot video semantic segmentation suggests that internal representations of Video DiTs are valuable for perception tasks, offering a lightweight video understanding tool.
- **504-video/150-motion-type benchmark**: The motion localization evaluation benchmark constructed in this paper is itself a contribution, filling an evaluation gap in the field. Videos are annotated using Qwen3-VL and filtered to exclude non-motion clips, ensuring evaluation quality.

## Rating
- Novelty: ⭐⭐⭐⭐ GramCol and motion head selection constitute an elegant and novel design; this is the first systematic study of motion interpretability in Video DiTs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three Video DiT models are evaluated with ablations and zero-shot segmentation; benchmark construction is rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis is structured hierarchically—progressively narrowing from timestep → layer → head—with theoretical justification and experimental validation at each step.
- Value: ⭐⭐⭐⭐ Opens the motion dimension for Video DiT interpretability research; both GramCol and IMAP have practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_wi.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)

</div>

<!-- RELATED:END -->
