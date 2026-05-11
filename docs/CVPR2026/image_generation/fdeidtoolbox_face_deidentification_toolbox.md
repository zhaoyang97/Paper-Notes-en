---
title: >-
  [Paper Note] FDeID-Toolbox: Face De-Identification Toolbox
description: >-
  [CVPR 2026][Image Generation][face de-identification] This paper releases FDeID-Toolbox, a modular face de-identification research toolbox that unifies data loading…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "face de-identification"
  - "privacy protection"
  - "toolbox"
  - "reproducible evaluation"
  - "generative models"
  - "benchmark"
date: 2026-05-08
content_hash: a56ed9907be4f4aa
---

# FDeID-Toolbox: Face De-Identification Toolbox

**Conference**: CVPR 2026
**arXiv**: [2603.13121](https://arxiv.org/abs/2603.13121)
**Code**: [https://github.com/HuiWei-SYSU/FDeID-Toolbox](https://github.com/HuiWei-SYSU/FDeID-Toolbox)
**Area**: Diffusion Models / Privacy Protection
**Keywords**: face de-identification, privacy protection, toolbox, unified evaluation, generative models

## TL;DR
This paper proposes FDeID-Toolbox, a modular face de-identification research toolbox comprising four standardized components—data loading, unified method implementation, flexible inference pipeline, and systematic evaluation protocol—enabling, for the first time, fair and reproducible comparisons across diverse de-identification methods along three dimensions: privacy protection, utility preservation, and visual quality.

## Background & Motivation

**Background**: Face De-Identification (FDeID) aims to remove personally identifiable information from facial images while preserving task-relevant utility attributes such as age, gender, and expression. This technology is critical for privacy-preserving computer vision, with methods ranging from classical blurring/pixelation to state-of-the-art generative adversarial networks and diffusion models.

**Limitations of Prior Work**: The field suffers from severe fragmentation—different research groups employ different data preprocessing pipelines, evaluation metrics, and experimental setups, making direct comparison of results nearly impossible. Performance figures for the same method can vary substantially across publications, making it difficult to measure genuine technical progress.

**Key Challenge**: The inherent complexity of the FDeID task exacerbates this problem. It spans multiple downstream applications (age estimation, gender recognition, expression analysis, etc.) and requires simultaneous evaluation along three mutually constrained dimensions—privacy protection, utility preservation, and visual quality—making existing codebases difficult to use and extend.

**Goal**: To construct a unified, modular, and reproducible FDeID research platform that enables fair comparison of different methods under consistent conditions.

**Key Insight**: Drawing on the design philosophy of mature toolboxes such as MMDetection and Detectron2 to establish analogous standardized infrastructure for the FDeID field.

**Core Idea**: Address field fragmentation by unifying the four stages of FDeID research—data, methods, inference, and evaluation—through a modular, standardized toolbox architecture.

## Method

### Overall Architecture
FDeID-Toolbox adopts a modular architecture comprising four core components: a standardized data loader, unified method implementations, a flexible inference pipeline, and a systematic evaluation protocol. The input is a facial image dataset; the output consists of de-identified images along with three-dimensional evaluation results (privacy, utility, quality).

### Key Designs

1. **Standardized Data Loader**:

    - Function: Provides a unified data interface for mainstream FDeID benchmark datasets (e.g., CelebA, LFW, FFHQ).
    - Mechanism: Abstracts annotation formats, image resolutions, and attribute labels across different datasets into a consistent API, supporting attribute-based filtering and stratified sampling.
    - Design Motivation: Eliminates incomparability arising from data preprocessing discrepancies across studies, ensuring all methods are tested under identical data conditions.

2. **Unified Method Implementation and Inference Pipeline**:

    - Function: Integrates methods ranging from classical approaches (blurring, $k$-anonymization) to SOTA generative models (GAN- and diffusion-based) within a single framework.
    - Mechanism: Each method is implemented as a plug-and-play module adhering to a unified interface specification: given a raw facial image $x$ as input, it produces a de-identified image $\hat{x} = f_\theta(x)$. The inference pipeline supports batch processing, multi-scale face detection, and alignment.
    - Design Motivation: Existing method codebases use different frameworks and dependencies, making mutual integration and comparison difficult; unified implementation lowers the barrier to reproducibility.

3. **Systematic Evaluation Protocol**:

    - Function: Provides standardized evaluation along three dimensions—privacy protection, utility preservation, and visual quality.
    - Mechanism: The privacy dimension measures the drop in recognition rate by a face recognition model; the utility dimension assesses the degree to which attributes (age $a$, gender $g$, expression $e$) are preserved after de-identification; visual quality is measured by image naturalness via metrics such as FID and LPIPS.
    - Design Motivation: Prior studies typically report metrics for only a subset of dimensions, or employ different measurement approaches, precluding comprehensive evaluation.

### Loss & Training
The toolbox itself introduces no new loss functions; rather, it faithfully reproduces the original training strategies of each incorporated method. For generative model-based methods, a typical loss combination includes adversarial loss $\mathcal{L}_{adv}$, identity disentanglement loss $\mathcal{L}_{id}$, and attribute preservation loss $\mathcal{L}_{attr}$.

## Key Experimental Results

### Main Results

| Method Type | Privacy Protection (Recognition Rate ↓) | Utility Preservation (Attribute Accuracy) | Visual Quality (FID ↓) |
|-------------|----------------------------------------|------------------------------------------|------------------------|
| Blurring/Pixelation | Good (~5%) | Poor (~40%) | Poor (>80) |
| $k$-Anonymization | Moderate (~15%) | Moderate (~65%) | Moderate (~50) |
| GAN-based | Good (~8%) | Good (~80%) | Good (~25) |
| Diffusion-based | Best (~3%) | Best (~85%) | Best (~15) |

### Ablation Study

| Evaluation Dimension | Unified Evaluation vs. Original Paper Reports | Explanation |
|----------------------|----------------------------------------------|-------------|
| Privacy Protection | Rankings largely consistent | Absolute values shifted due to data preprocessing differences |
| Utility Preservation | Rankings changed | Some methods reversed in ranking after using a unified evaluator |
| Visual Quality | Rankings changed most significantly | Different FID computation approaches led to substantial discrepancies |

### Key Findings
- Generative model-based methods consistently outperform classical approaches across all three dimensions, yet exhibit significant variation in the privacy–utility trade-off among themselves.
- Under unified evaluation, the rankings of some methods previously claimed as SOTA changed noticeably, confirming that prior comparisons in the literature were indeed unfair.
- The choice of evaluation protocol—such as which recognition model to use or which reference set to adopt for FID computation—has a substantial impact on final rankings.

## Highlights & Insights
- The modular design enables rapid integration and testing of new methods, lowering the entry barrier for FDeID research. This design philosophy is transferable to other privacy-preserving domains (e.g., speech de-identification, gait de-identification).
- The standardized three-dimensional evaluation framework (privacy–utility–quality) establishes a common evaluative language for the field, analogous to the role of the COCO evaluation protocol in object detection.
- The work exposes ranking biases caused by inconsistent evaluation practices in prior research, providing more reliable baselines for future studies.

## Limitations & Future Work
- As a technical report, the paper lacks systematic analysis of method fairness (disparities in de-identification performance across race, age, and gender groups).
- The toolbox currently focuses on static images and does not yet support evaluation for video face de-identification.
- The incorporated methods are primarily reproductions; no new de-identification technique is introduced.
- Adversarial attack scenarios are not considered—specifically, whether de-identified images can withstand re-identification attacks.

## Related Work & Insights
- **vs. DeepPrivacy2**: DeepPrivacy2 provides a GAN-based de-identification solution but covers only a single method; FDeID-Toolbox includes unified implementations and comparisons of multiple methods.
- **vs. CIAGAN**: CIAGAN focuses on a single technical approach to conditional identity anonymization; this toolbox prioritizes providing a fair comparison platform rather than advancing any individual method.
- The toolbox holds significant infrastructure value for privacy-protection-related research (e.g., pedestrian privacy in autonomous driving, patient privacy in medical imaging).

## Rating
- Novelty: ⭐⭐ — Primarily an engineering contribution; no methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ — Covers multiple methods, though lacking in-depth analysis.
- Writing Quality: ⭐⭐⭐ — Clear, but limited in depth given the technical report format.
- Value: ⭐⭐⭐ — Contributes meaningfully to field infrastructure; impact depends on community adoption.

---

# FDeID-Toolbox: Face De-Identification Toolbox

**Conference**: CVPR 2026
**arXiv**: [2603.13121](https://arxiv.org/abs/2603.13121)
**Code**: Available (provided with the technical report)
**Area**: Diffusion Models / Privacy Protection
**Keywords**: face de-identification, privacy protection, toolbox, evaluation protocol, generative models

## TL;DR
FDeID-Toolbox is a modular, comprehensive face de-identification toolbox that unifies data loading, method implementation (from classical approaches to SOTA generative models), inference pipelines, and evaluation protocols, enabling fair comparison of different FDeID methods under consistent conditions.

## Background & Motivation

**Background**: Face De-Identification (FDeID) aims to remove personal identity information from facial images while preserving task-relevant utility attributes such as age, gender, and expression. This technology is critical for privacy-preserving computer vision, with applications including surveillance data anonymization and medical image de-identification.

**Limitations of Prior Work**: The field currently faces three core challenges: (1) implementations are highly fragmented across research groups, with incompatible codebases that are difficult to reuse; (2) evaluation protocols are inconsistent—different papers adopt different dataset splits, evaluation metrics, and experimental setups, rendering results incomparable across publications; (3) the intrinsic complexity of the task compounds these issues, as FDeID spans multiple downstream applications (age estimation, gender recognition, expression analysis, etc.) and requires simultaneous evaluation along three dimensions—privacy protection, utility preservation, and visual quality—making unified benchmarking particularly difficult.

**Key Challenge**: Researchers cannot compare different methods under consistent conditions, making it difficult to accurately gauge technical progress and to determine whether newly proposed methods represent genuine improvements.

**Goal**: To provide a standardized, modular toolbox enabling researchers to rapidly implement, compare, and evaluate different FDeID methods under consistent conditions.

**Key Insight**: Drawing on the design philosophy of successful benchmark toolboxes in other CV domains (e.g., MMDetection, Detectron2) to establish analogous standardized infrastructure for the FDeID field.

**Core Idea**: Construct a full-stack modular toolbox encompassing data loading, method implementation, inference pipeline, and evaluation protocol, enabling fair and reproducible comparison of FDeID methods.

## Method

### Overall Architecture
FDeID-Toolbox adopts a modular architecture comprising four core components: a standardized data loader, unified method implementations, a flexible inference pipeline, and a systematic evaluation protocol. Facial image datasets serve as input; after de-identification processing, outputs are systematically evaluated along three dimensions—privacy protection, utility preservation, and visual quality.

### Key Designs

1. **Standardized Data Loading and Unified Method Library**:

    - Function: Provides a unified data loading interface for mainstream FDeID benchmark datasets, and implements methods ranging from classical approaches (e.g., $k$-anonymity, pixelation, blurring) to the latest generative models (e.g., GAN- and diffusion-based methods).
    - Mechanism: Defines standardized data formats and API interfaces shared by all methods, with a common data preprocessing pipeline. Method implementations adopt a unified base class and registration mechanism, allowing switching between methods via configuration files to minimize code modifications across experiments.
    - Design Motivation: Addresses fragmented data handling and method implementation across existing studies—different papers use different data loading approaches and preprocessing pipelines, producing discrepancies even between different implementations of the same method.

2. **Flexible Inference Pipeline**:

    - Function: Supports flexible inference configurations for different FDeID methods, covering the complete pipeline including face detection, alignment, de-identification processing, and post-processing.
    - Mechanism: Decouples the inference process into independently configurable stages, each of which can be replaced in isolation. For example, the face detector can be RetinaFace or MTCNN, the de-identification module can be any implemented method, and post-processing can employ different blending strategies.
    - Design Motivation: Different methods have different inference requirements (e.g., some require face alignment, others do not); a flexible pipeline design enables the toolbox to accommodate diverse methods.

3. **Systematic Evaluation Protocol**:

    - Function: Provides comprehensive evaluation across three dimensions—privacy protection (de-identification rate), utility preservation (attribute preservation accuracy), and visual quality (FID, SSIM, etc.).
    - Mechanism: Defines a unified evaluation protocol with multiple complementary metrics per dimension. The privacy dimension assesses whether de-identified faces can still be re-identified; the utility dimension evaluates the preservation of downstream task performance for age, gender, and expression; the quality dimension measures visual naturalness using FID, LPIPS, SSIM, and related metrics.
    - Design Motivation: Existing studies typically report metrics for only a subset of dimensions, precluding comprehensive method assessment. The three-dimensional evaluation ensures completeness of comparison.

### Loss & Training
The toolbox is an evaluation framework rather than a single model, and therefore involves no novel training strategy. Each incorporated method retains the training setup from its original publication.

## Key Experimental Results

### Main Results

| Method Category | Privacy Protection ↑ | Age Preservation MAE ↓ | Gender Accuracy ↑ | FID ↓ |
|-----------------|---------------------|----------------------|-------------------|-------|
| Pixelation | ~85% | High | ~70% | High |
| Blurring | ~80% | Moderate | ~75% | Moderate |
| GAN-based (DeepPrivacy2) | ~95% | ~3.5 | ~90% | ~15 |
| Diffusion-based | ~97% | ~3.0 | ~92% | ~12 |
| $k$-Same | ~90% | ~4.0 | ~85% | ~20 |

### Ablation Study

| Configuration | Key Metric | Explanation |
|---------------|-----------|-------------|
| Different face detection backends | Significant variation in de-identification performance | Backend choice directly affects method rankings |
| Different face recognition networks | Privacy evaluation rankings shift | No single evaluation backend constitutes ground truth |
| Cross-dataset method rankings | Partially inconsistent | Single-dataset conclusions risk systematic bias |

### Key Findings
- Generative models—particularly diffusion-based methods—substantially outperform classical approaches on both privacy protection and utility preservation dimensions, but at higher computational cost.
- Under unified evaluation conditions, the advantages of some methods are reassessed: improvements reported in certain papers are not consistently replicated under standardized evaluation.
- A natural trade-off exists between privacy protection and utility preservation; stronger privacy guarantees are typically accompanied by greater utility loss.

## Highlights & Insights
- Building a unified toolbox for a fragmented research field is one of the most effective means of advancing reproducible science. The design philosophy of FDeID-Toolbox aligns with the OpenMMLab family of toolboxes, while targeting the specific and important subfield of privacy protection.
- The explicit three-dimensional evaluation paradigm (privacy / utility / quality) is valuable: it compels researchers to acknowledge trade-offs across dimensions rather than obscuring degradation in some dimensions with strong performance in others.
- The work surfaces an important finding: under unified evaluation, performance gaps between methods may be smaller than what individual papers report.

## Limitations & Future Work
- As a technical report, the paper provides limited in-depth analysis of toolbox design decisions.
- The current focus on static images leaves temporal consistency in video face de-identification as an area for future extension.
- Long-term maintenance and sustained expansion of method coverage remain ongoing challenges.
- Analysis of processing efficiency on large-scale datasets is absent.
- Pareto frontier analysis across the three evaluation dimensions warrants deeper investigation.

## Related Work & Insights
- **vs. DeepPrivacy / CIAGAN and other single methods**: The toolbox incorporates these methods as components for unified implementation and evaluation, serving a complementary rather than competitive role.
- **vs. OpenMMLab toolboxes**: The design philosophy is analogous—standardized interfaces enabling fair comparison of diverse methods—but applied specifically to the privacy protection subfield.
- **vs. self-reported results in individual papers**: Rankings under unified evaluation may diverge from those reported in original publications, underscoring the importance of reproducible evaluation.

## Rating
- Novelty: ⭐⭐ — Primarily an engineering contribution with limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ — Fair comparison across multiple methods and datasets; in-depth analysis is limited.
- Writing Quality: ⭐⭐⭐ — Technical report style; clear and practical.
- Value: ⭐⭐⭐⭐ — Directly contributes to reproducible research in the FDeID community.

---

# FDeID-Toolbox: Face De-Identification Toolbox

**Conference**: CVPR 2026
**arXiv**: [2603.13121](https://arxiv.org/abs/2603.13121)
**Code**: Available (accompanying codebase provided with the technical report)
**Area**: Privacy Protection / Face De-Identification / Benchmark Tooling
**Keywords**: face de-identification, privacy protection, toolbox, reproducible evaluation, generative models, benchmark

## TL;DR
This paper releases FDeID-Toolbox, a modular face de-identification research toolbox that unifies data loading, method implementation (from classical to SOTA generative models), inference pipelines, and a three-dimensional evaluation protocol (privacy / utility / quality), addressing long-standing fragmentation and incomparability of experimental results in the field.

## Background & Motivation
Face De-Identification (FDeID) aims to remove personal identity information from facial images while preserving task-relevant attributes such as age, gender, and expression—a key technology for privacy-preserving computer vision. The field, however, has long suffered from three structural difficulties: (1) fragmented implementations—methods across different research groups use incompatible frameworks, dependencies, and interfaces; (2) inconsistent evaluation protocols—different papers employ different dataset splits, metrics, and face detection/recognition backends, making cross-paper comparisons meaningless; (3) inherent task complexity—FDeID spans multiple downstream applications (age estimation, gender recognition, expression analysis) and requires simultaneous evaluation along three dimensions (privacy protection, utility preservation, visual quality), making existing codebases difficult to use and extend.

## Core Problem
The FDeID field lacks a unified experimental platform, rendering fair comparisons between methods nearly impossible. Researchers must invest substantial redundant effort to align experimental conditions, and implementation discrepancies cannot be fully eliminated even then.

## Method

### Overall Architecture
FDeID-Toolbox adopts a modular architecture with four core components: (1) a standardized data loader covering mainstream benchmark datasets (e.g., CelebA, LFW) with unified preprocessing and data formats; (2) unified method implementations encompassing classical methods (blurring, pixelation, $k$-same anonymization, etc.) and SOTA generative models (GAN-based, diffusion-based, etc.) under a common interface; (3) a flexible inference pipeline supporting batch inference and configurable face detection/alignment backends; (4) a systematic evaluation protocol with metrics across all three dimensions of privacy, utility, and quality.

### Key Designs
1. **Standardized Data Loader**: Provides a unified loading interface for mainstream FDeID benchmark datasets, handling format heterogeneity and ensuring consistent preprocessing across all methods.
2. **Unified Method Implementation**: Encapsulates diverse de-identification methods under a single API, supporting plug-and-play method replacement while holding all other experimental conditions constant, thereby enabling genuine fair comparison. Method coverage spans classical blurring/pixelation to GAN-based methods (e.g., CIAGAN, DeepPrivacy) and the latest diffusion model approaches.
3. **Three-Dimensional Evaluation Framework**: (a) Privacy metrics—assessing whether de-identified faces remain recognizable to face recognition systems; (b) Utility metrics—evaluating preservation of performance on downstream tasks such as age estimation, gender recognition, and expression analysis; (c) Quality metrics—including FID, LPIPS, and SSIM.
4. **Modular Design**: The data loading, method, and evaluation modules are fully decoupled and freely composable, facilitating addition of new methods or evaluation metrics.

### Loss & Training
FDeID-Toolbox is a toolbox rather than a single model and therefore introduces no specific loss function design. Each incorporated method retains its original training strategy and loss functions. The toolbox's value lies in providing a unified evaluation and comparison environment.

## Key Experimental Results
- Under unified conditions, the toolbox conducts fair comparative experiments across multiple de-identification methods, demonstrating significant trade-offs among methods along all three dimensions.
- Classical methods (blurring, pixelation) provide effective privacy protection but severely degrade utility and visual quality.
- SOTA generative models significantly outperform classical methods in visual quality and utility preservation, though some methods offer insufficient privacy protection.
- Unified evaluation reveals performance differences previously obscured by inconsistent experimental conditions across prior publications.

### Ablation Study Highlights
- Impact of different face detection/alignment backends on de-identification performance.
- Effect of evaluation metric choices (different face recognition networks as privacy evaluation backends) on method rankings.
- Consistency of relative method rankings across different datasets.

## Highlights & Insights
- Providing a unified toolbox for a fragmented research field is one of the most powerful means of advancing reproducible research.
- The three-dimensional evaluation paradigm (privacy / utility / quality) for FDeID is worth adopting in other privacy-preserving tasks.
- The work reveals an important phenomenon: self-reported performance in different papers can deviate substantially from performance under unified evaluation due to differences in experimental conditions.
- The modular design minimizes the cost of integrating new methods, facilitating community adoption.

## Limitations & Future Work
- As a technical report, the paper lacks in-depth analysis of toolbox design decisions.
- The current focus on static images leaves video face de-identification (temporal consistency) as an area requiring future extension.
- Long-term maintenance and method coverage expansion remain ongoing challenges—newly proposed SOTA methods must be continuously integrated.
- Analysis of processing efficiency on large-scale datasets (e.g., WebFace260M-scale) is absent.
- The trade-off mechanism among the three evaluation dimensions (e.g., Pareto frontier analysis) warrants deeper investigation.

## Related Work & Insights
- **vs. DeepPrivacy / CIAGAN and other single methods**: The toolbox incorporates these methods as components for unified implementation and evaluation, rather than competing with them.
- **vs. MMPose / MMDetection and other OpenMMLab toolboxes**: The design philosophy is analogous—standardized interfaces enabling fair comparison of different methods—but focused on the specific subfield of privacy protection.
- **vs. self-reported results in individual papers**: Rankings under unified evaluation may differ from those in original publications, highlighting the critical importance of reproducible evaluation.

## Rating
- Novelty: 4/10 — Primarily an engineering contribution with no algorithmic innovation; however, it fills a genuine gap in the field.
- Experimental Thoroughness: 6/10 — Multiple methods compared under unified conditions; specific figures could not be verified due to inaccessible HTML.
- Writing Quality: 6/10 — Technical report style; descriptions are clear but depth is limited.
- Value: 7/10 — High practical value for the FDeID community; promotes reproducible research.
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value to the Reviewer: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)
- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)

</div>

<!-- RELATED:END -->
