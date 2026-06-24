---
title: >-
  [Paper Note] Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images
description: >-
  [ICCV 2025][Multimodal VLM][Large-Scale Image Analysis] This paper proposes the Visual Chronicles system, which is the first to leverage Multimodal Large Language Models (MLLMs) to analyze a massive database of over 20 million street view images. Through a bottom-up hierarchical strategy (local change detection + trend discovery) and an efficient text-embedding-MLLM hybrid verification algorithm, the system discovers unlabeled, open-ended visual change trends across cities ov…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Large-Scale Image Analysis"
  - "Urban Change Detection"
  - "Visual Trend Discovery"
  - "MLLM System Design"
  - "Street View Imagery"
date: 2026-05-08
content_hash: 1ef76e50d0bc93ae
---

# Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images

**Conference**: ICCV 2025  
**arXiv**: [2504.08727](https://arxiv.org/abs/2504.08727)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Large-Scale Image Analysis, Urban Change Detection, Visual Trend Discovery, MLLM System Design, Street View Imagery

## TL;DR

This paper proposes the Visual Chronicles system, which is the first to leverage Multimodal Large Language Models (MLLMs) to analyze a massive database of over 20 million street view images. Through a bottom-up hierarchical strategy (local change detection + trend discovery) and an efficient text-embedding-MLLM hybrid verification algorithm, the system discovers unlabeled, open-ended visual change trends across cities over a decade (e.g., newly added solar panels in San Francisco, overpasses being painted blue) without labels. It reduces MLLM inference costs by 2000-fold while maintaining a verification accuracy of 93.9%.

## Background & Motivation

### Problem Definition

Given massive time-series imagery of a city (e.g., 20 million Google Street View images spanning 10+ years), the goal is to automatically discover frequently occurring visual trends—meaning similar change events that occur at least $N$ times across different locations—and provide visual evidence. The core requirements are:
- **Open-ended queries**: No predefined target categories (such as "cars" or "buildings"), supporting discovery at any semantic level.
- **No labels**: No training data required.
- **Large-scale**: Needs to handle tens of millions of images.

### Limitations of Prior Work

**Pre-trained recognition models cannot handle open-ended queries**: For example, a vehicle detector can only find car-related changes, failing to find open-semantic changes like "restaurants adding outdoor dining areas."

**Unsupervised image analysis tools lack semantic understanding**: Methods using HoG features, color histograms, or CLIP embeddings achieve only 16-27% AP on change detection tasks, failing to understand changes requiring global semantic understanding, such as "a shop closing down."

**Learning-based methods require labels**: For instance, demographic vehicle analysis (Gebru et al.) requires annotated vehicle attribute data.

**MLLM context limitations**: Even the strongest model, Gemini-1.5 Pro, can only handle a context of about 88K images, which is far from 20 million.

### Core Motivation

**Key Insight**: MLLMs exhibit exceptionally strong open-ended semantic analysis capabilities on small-scale image sets (20-40 images)—not only detecting semantic temporal changes (e.g., "a corner shop hanging a for-sale sign") but also automatically ignoring irrelevant changes (such as lighting and seasonal variations). However, MLLMs cannot directly handle massive amounts of data. Therefore, a hierarchical system must be designed to break down the massive analysis problem into sub-problems that MLLMs can handle efficiently.

## Method

### Overall Architecture

Visual Chronicles employs a bottom-up, two-step analysis framework:
1. **Local Change Detection**: For the temporal image sequence of each location, an MLLM is used to detect local visual changes and generate text descriptions.
2. **Trend Discovery**: Millions of local changes are aggregated to propose trend candidates through Canopy clustering. Then, a hybrid algorithm combining "text embedding ranking + MLLM verification" is used to efficiently confirm which trends actually exist.

### Key Designs

#### 1. **MLLM-Driven Local Change Detection**

- **Function**: Detects which semantic changes occurred in image sequences of approximately the same perspective at different times of each location.
- **Mechanism**:
    - Time-series images of each location are sorted chronologically and fed into an MLLM (Gemini-1.5 Pro).
    - A carefully designed prompt asks the MLLM to answer two questions: (1) What change occurred? (2) Between which two images did the change occur?
    - The MLLM outputs a textual description of each change and the corresponding evidence images.
    - Processing 883K locations in NYC and 943K locations in SF generates approximately 2.9 million and 3.6 million change records respectively.
- **Design Motivation**: MLLMs are naturally suited for such small-scale (average of 20+ images per location) open-ended semantic analyses—they can understand complex scene semantics (e.g., "a store closed") while automatically filtering out irrelevant seasonal and lighting variations. It achieves an AP of 76.56%, far exceeding the best embedding methods (CLIP at 26.52%).

#### 2. **Efficient Hybrid Trend Verification Algorithm**

- **Function**: Identifies frequent trends that appear at least $N$ times from millions of local changes.
- **Mechanism**:

  **Step 1 - Trend Candidate Proposal**:
  - Embed all change descriptions into a vector space.
  - Use Canopy Clustering (with a loose threshold) to find potential trend candidates.

  **Step 2 - Hybrid Verification** (Core Innovation):
  - Deploying embedding distance alone for classification is inaccurate (AP 73.13%, failing to capture subtle semantic nuances).
  - Directly utilizing the MLLM to verify every single candidate sequentially is too slow (more than 1 year for 200 trends × 3 million changes).
  
  *Hybrid Scheme* (Algorithm 1):
  1. Compute the distance of all changes to the trends using text embeddings.
  2. Retrieve the top-$k$ nearest changes ($k = 3N$).
  3. Use the MLLM on only these $k$ changes to perform binary classification: "Does this change belong to this trend?"
  4. If the MLLM confirms $\ge N$ changes belong to the trend, the trend is validated.

  This reduces the MLLM inference cost by $2000\times$ (verifying 200 trends takes only 4.6 hours instead of 380 days).

- **Design Motivation**: Although text embeddings are not precise enough for final classification (due to subtle phrasing variations and differing concept granularities), they serve as an efficient ranking tool to filter candidates most likely to belong to a trend. The precise semantic judgment of the MLLM is then deployed for final verification, allowing the two methods to complement each other.

#### 3. **Extended Query Support**

- **Function**: Supports time-conditioned queries ("What has happened since 2020?") and topic-conditioned queries ("What changes happened to retail stores?").
- **Mechanism**:
    - Temporal queries: Filter the timestamps of change records, retaining only changes within the specified time range, and rerun trend discovery.
    - Topic queries: Compute the similarity between change descriptions and the topic text using text embeddings, select the top-$k$ closest changes for MLLM filtering, and then run trend discovery.
    - Non-temporal queries: Replace the first step with asking a single image "Is there anything unusual in this image?", while keeping subsequent steps unchanged.
- **Design Motivation**: The modular design allows the system to flexibly adapt to different types of analysis demands without modifying the core algorithm.

### Loss & Training

This paper presents a zero-shot system and does not involve model training. Core hyperparameters include:
- MLLM: Gemini-1.5 Pro
- Trend validation threshold: $N = 500$
- Hybrid verification nearest neighbors: $k = 1500$ (i.e., $3N$)
- Number of trend candidates: 200-500
- Text embedding: NV-Emb

## Key Experimental Results

### Main Results

**Local Change Detection (200 locations, 3036 images)**:

| Method | Type | AP |
|------|------|-----|
| HoG Features | Image Feature | 16.44%|
| Color Histogram | Image Feature | 16.76% |
| Remote Sensing Methods | Remote Sensing | 18.51% |
| CLIP | Semantic Embedding | 26.52% |
| NV-Emb | Text Embedding | 23.75% |
| **Gemini (ours)** | **MLLM** | **76.56%** |

Additional MLLM metrics: Precision 81.34%, Recall 89.87%

**Trend Discovery Verification (50 candidates, 2000 trend-change pairs)**:

| Method | Type | Scalable | AP |
|------|------|--------|-----|
| Random | — | — | 47.70% |
| CLIP | Image Embedding | ✔ | 54.78% |
| NV-Emb | Text Embedding | ✔ | 73.13% |
| **Gemini** | **MLLM** | ✘ | **86.63%** |

### Ablation Study

**Hybrid Verification vs. Alternatives (1000 trend candidates)**:

| Method | Acc@50 | Acc@100 | Acc@200 |
|------|--------|---------|---------|
| AllTrue Baseline | 72.7% | 54.1% | 28.9% |
| NV-Emb Threshold | 77.9% | 69.6% | 81.8% |
| Random Selection + MLLM | 31.8% | 49.9% | 74.9% |
| **Hybrid Scheme (ours)** | **93.9%** | **94.6%** | **98.3%** |

### Key Findings

1. **MLLM far outperforms all baselines in local change detection**: 76.56% AP vs. CLIP's 26.52%, proving that the semantic understanding of MLLMs is irreplaceable for open-ended change detection.
2. **Directly asking MLLMs to predict trends is infeasible**: Without looking at the images, MLLMs can only provide abstract answers (e.g., "economic growth") and fail to discover specific trends.
3. **The hybrid algorithm achieves both high accuracy and efficiency**: It achieves 98.3% accuracy at $N=200$ while reducing inference costs by 2000-fold.
4. **Discovered unexpected trends**: E.g., overpasses being painted "Coronado Blue" in San Francisco (481 occurrences), 745 new security cameras added in New York, etc.
5. **Supports multiple query modes**: Temporal filtering identified the surge of outdoor dining trends during the post-COVID period (1482 occurrences in 2020-2022 vs. 668 in 2017-2019).

## Highlights & Insights

1. **First to use MLLMs for analysis on millions of images**: Pioneered a new paradigm of using MLLMs for large-scale visual data mining.
2. **"MLLM strong at small scale + system design solving large scale" paradigm**: Instead of forcing the MLLM to process all data directly, the massive problem is decomposed into smaller tasks that the MLLM excels at.
3. **Elegant design of the hybrid algorithm**: Leverages embeddings for coarse-grained ranking (efficiency) + MLLMs for fine-grained ranking (accuracy), making the two fully complementary.
4. **Discovered trends carry real-world value**: E.g., urban planning departments can evaluate infrastructure developments, and the retail industry can analyze storefront opening/closing trends.
5. **Zero-annotation, open-ended discovery**: No predefined categories or labels are required, truly achieving open-world visual data mining.

## Limitations & Future Work

1. **Sampling Bias**: Spatial and temporal sampling of street view cameras is uneven—e.g., more rooftop solar panels might be visible from overpasses, which does not necessarily imply more installations in that area.
2. **MLLM Biases and Errors**: The 81.34% precision implies that about 1/5 of the detections are false, and these errors might systematically bias towards certain types of changes.
3. **High Computational Cost**: Even with the hybrid algorithm reducing costs by 2000 times, processing a single city still requires hours of MLLM inference (4.6 hours under 64 parallel MLLM instances).
4. **Lack of Statistical Rigor**: No framework for hypothesis testing has been established to quantify the significance of trends, nor has multi-comparison correction been applied.
5. **Primarily a Proof of Concept**: As the first system of its kind, the design space is far from fully explored (e.g., the impact of different MLLMs or clustering strategies).

## Related Work & Insights

- Difference from VisDiff: VisDiff handles only thousands of images to find differences between datasets, whereas Visual Chronicles processes tens of millions to discover temporal trends.
- Difference from traditional remote sensing change detection: Remote sensing methods require labels or specific change categories, while this method is completely open-ended.
- Insight: The greatest value of MLLMs may not lie in directly answering user questions, but in serving as a "semantic engine" inside carefully engineered pipelines to solve previously impossible tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering problem definition and system design, first to employ MLLMs on tens of millions of images.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative evaluations (change detection + trend discovery + hybrid verification), but primarily relies on a single MLLM (Gemini).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Outstanding storytelling, seamlessly weaving together motivation, methods, and practical applications.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes the new paradigm of "MLLM + System Engineering = Large-Scale Visual Data Mining," offering broad prospects for future applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](../../ACL2025/multimodal_vlm/finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)

</div>

<!-- RELATED:END -->
