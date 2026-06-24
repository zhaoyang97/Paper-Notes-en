---
title: >-
  [Paper Note] Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning
description: >-
  [ACL 2025][VLM Reasoning][Visual Forgetting] This paper identifies a severe visual forgetting phenomenon in MLLMs during long CoT reasoning—removing the image halfway through reasoning only causes a ~2% drop in accuracy, indicating that models rely excessively on self-generated text while ignoring visual evidence. To address this, the authors propose a Take-along Visual Conditioning (TVC) strategy. It injects an image review mechanism via Dynamic Visual Reaffirmation (DVR) du…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Visual Forgetting"
  - "Take-along Visual Conditioning"
  - "Long CoT Reasoning"
  - "Dynamic Visual Reaffirmation"
  - "Periodic Visual Calibration"
date: 2026-05-08
content_hash: 85571151be08bb74
---

# Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning

**Conference**: ACL 2025  
**arXiv**: [2503.13360](https://arxiv.org/abs/2503.13360)  
**Code**: [TVC Project Page](https://sun-hailong.github.io/projects/TVC)  
**Area**: Multimodal Reasoning  
**Keywords**: Visual Forgetting, Take-along Visual Conditioning, Long CoT Reasoning, Dynamic Visual Reaffirmation, Periodic Visual Calibration

## TL;DR

This paper identifies a severe visual forgetting phenomenon in MLLMs during long CoT reasoning—removing the image halfway through reasoning only causes a ~2% drop in accuracy, indicating that models rely excessively on self-generated text while ignoring visual evidence. To address this, the authors propose a Take-along Visual Conditioning (TVC) strategy. It injects an image review mechanism via Dynamic Visual Reaffirmation (DVR) during the training phase, and compresses and re-injects visual tokens via Periodic Visual Calibration (PVC) during the inference phase. TVC outperforms the previous SOTA by 3.4 points on average (43.4 vs 40.0) across five mathematical reasoning benchmarks.

## Background & Motivation

**Background**: LLM reasoning capabilities have evolved from CoT prompting to product-grade solutions (o1/DeepSeek-R1), and multimodal reasoning has also progressed through data-driven approaches (Math-LLaVA/MAmmoTH-VL).

**Limitations of Prior Work**: (1) While text LLMs can maintain problem context by repeating key terms, visual inputs in MLLMs are restricted to the initial processing stage, leaving subsequent reasoning steps unable to re-examine the image; (2) As the reasoning chain grows, the model's attention to visual inputs decays exponentially, leading to an over-reliance on self-generated text ("visual forgetting"); (3) This decay triggers hallucinations and failures in spatial relationship verification.

**Key Challenge**: In current MLLM architectures, visual information is injected only once at the input stage, whereas long-chain reasoning requires continuous visual-textual interaction. Akin to humans repeatedly inspecting figures when solving problems, models also need to re-focus on visual inputs during the reasoning process.

**Goal**: To diagnose and mitigate the visual forgetting phenomenon in MLLM long-chain reasoning.

**Key Insight**: Quantifying the degree of visual forgetting through "progressive image removal" experiments, and then designing a dual-phase training-and-inference scheme for visual re-injection.

**Core Idea**: Re-injecting compressed visual tokens at critical stages of the reasoning chain to simulate human "image-relooking" behavior and sustain visual attention.

## Method

### Overall Architecture

TVC comprises two stages: training and inference. In the training phase, an iterative distillation pipeline (QVQ-72B $\rightarrow$ Qwen2-VL) is used to construct a long-chain reasoning dataset, and DVR is employed to inject visual caption reactivation at self-reflection points within the training data. In the inference phase, PVC compresses visual tokens (via $4 \times 4$ average pooling) midway through reasoning and re-injects them after resetting the KV cache.

### Key Designs

1. **Quantitative Diagnosis of Visual Forgetting**
    - **Function**: Quantitatively reveals the degree of visual forgetting through progressive image removal experiments.
    - **Mechanism**: Universally divides the reasoning process into $K=8$ stages, resets the KV cache at different positions $k$ to remove image tokens, and compares the accuracy difference between normal reasoning and inference after image removal.
    - **Key Findings**: (a) Removing the image at $k=4$ (halfway through reasoning) only reduces accuracy by 2.2% (40.9 vs 43.1); (b) The forgetting effect closely approximates exponential decay $\Delta_{\text{visual}}(k) \propto e^{-k}$; (c) Early removal ($k=0$) causes a ~20% drop, showing that visual information is indeed utilized in the early stages; (d) Attention matrix visualization confirms that visual attention weakens significantly after approximately 20% of the tokens.
    - **Design Motivation**: It is necessary to quantitatively understand the severity and patterns of the problem before proposing solutions.

2. **Dynamic Visual Reaffirmation (DVR) — Training Phase**
    - **Function**: Injects visual reactivation points into the training data to teach the model how to look back at the image.
    - **Mechanism**: (a) Employs QVQ-72B as the teacher model to distill long-chain CoT data, obtaining ~200K high-quality samples through dual-temperature sampling ($\tau=0$ for initial sampling + $\tau=1$ for error correction, best-of-8); (b) Manually injects bridging prompts (e.g., "Let me see the image again") and visual caption regeneration at self-reflection intervals (such as the reasoning midpoint $r_1=0.5L$); (c) Fine-tunes both LLM parameters and the cross-modal connector during training, while freezing the vision encoder.
    - **Data Quality Assurance**: Dynamic token truncation (200-8000 tokens) filters out excessively long or short reasoning chains; reflection token pruning (capped at 25 reflection markers) reduces useless metacognitive loops.
    - **Design Motivation**: The QVQ model itself lacks the ability to iteratively reference visual inputs during reasoning, necessitating the explicit embedding of this mechanism in the training data.

3. **Periodic Visual Calibration (PVC) — Inference Phase**
    - **Function**: Periodically re-injects compressed visual tokens during the inference process.
    - **Mechanism**: (a) Token Compression—uses $4 \times 4$ average pooling to compress the number of visual tokens by 16 times while retaining spatial semantics; (b) Visual Cache Reset—prepends bridging prompt instructions before self-reflection intervals, resets the KV cache, and re-injects the compressed image tokens.
    - **Design Motivation**: Compression is essential—too many visual tokens can cause the model to forget prior textual reasoning steps; KV cache resetting ensures that newly injected visual information can participate effectively in subsequent attention computations.

## Key Experimental Results

### Main Results — Comparison with SOTA Methods (5 Math Reasoning Benchmarks)

| Method | Size | MathVista | MathVision | MathVerse | Dynamath | OlympiadBench | Average |
|------|:----:|:---------:|:----------:|:---------:|:--------:|:-------------:|:----:|
| Qwen2-VL | 7B | 60.9 | 16.3 | 24.6 | 11.0 | 3.2 | 23.2 |
| InternVL2.5 | 8B | 64.5 | 17.0 | 22.8 | 9.4 | 0.1 | 22.8 |
| LLaVA-COT | 11B | 52.5 | 19.9 | 22.6 | 7.8 | - | - |
| QVQ-72B-preview | 72B | 71.4 | 35.9 | 41.5 | 30.7 | 20.4 | 40.0 |
| **TVC** | **7B** | **68.1** | **22.7** | **38.9** | **15.1** | **9.8** | **30.9** |
| **TVC** | **72B** | **72.2** | **41.9** | **48.8** | **30.0** | **24.3** | **43.4** |

### Ablation Study (Qwen2-VL-7B Benchmark)

| Configuration | MathVista | MathVision | MathVerse | Average |
|------|:---------:|:----------:|:---------:|:----:|
| Baseline | 60.9 | 16.3 | 24.6 | 33.9 |
| Vanilla SFT | 63.5 | 19.8 | 31.6 | 38.3 |
| TVC w/o PVC | 66.7 | 21.8 | 35.6 | 41.4 |
| TVC w/o DVR | 66.2 | 22.3 | 34.7 | 41.0 |
| **TVC Full** | **68.1** | **22.7** | **38.9** | **43.2** |

### Key Findings

- TVC-72B outperforms QVQ-72B-preview (the teacher model) by 3.4 points on average (43.4 vs 40.0), showing the student surpassing the teacher.
- The largest gains are observed on MathVision and MathVerse (+6.0 and +7.3), which require continuous visual reasoning.
- TVC-7B even outperforms several 72B models on MathVerse (38.9).
- Both DVR and PVC contribute comparably and complement each other: removing either individually drops the performance by ~2 points, while the full combination yields the greatest gain.
- The $4 \times 4$ average pooling visual token compression not only improves inference efficiency but also slightly enhances performance (43.2 vs 43.1 without compression).
- Scaling up the dataset (from 50K to 200K) consistently brings improvements without showing saturation.

## Highlights & Insights

1. **Rigorous Diagnosis of the Visual Forgetting Phenomenon**: Dual validation via progressive image removal experiments and attention heatmap visualization intuitively and convincingly demonstrates the severity of the problem (removing the image only causes a 2% drop!).
2. **"Looking Back at Images" Analogous to Human Behavior**: The core intuition of TVC directly corresponds to how humans repeatedly inspect figures when solving geometry problems, making it simple yet effective.
3. **Integrated Training-and-Inference Design**: DVR teaches the model when and how to look back, while PVC executes the looking-back policy during inference; both are indispensable.
4. **Multi-level Quality Control in Data Engineering**: Dual-temperature sampling, answer-centric reject sampling, dynamic truncation, and reflection token pruning construct a complete high-quality data pipeline for long-chain reasoning.

## Limitations & Future Work

1. For highly complex reasoning tasks, simply increasing the number of visual look-backs is insufficient; the reasoning capability of the model itself needs enhancement.
2. The method assumes that visual processing can be deferred, which is not suitable for real-time applications (such as robot navigation).
3. The localization of the visual reaffirmation point (midpoint $r_1=0.5L$) is empirical, and adaptive triggering mechanisms have not been explored.
4. The evaluation is restricted to mathematical reasoning benchmarks; its effectiveness on other visual reasoning scenarios such as chart understanding and document analysis has not been tested.
5. Training requires 64 $\times$ H20 GPUs for 4 days (for the 72B model), which incurs high computational costs.

## Related Work & Insights

- The visual forgetting phenomenon may be prevalent in all multimodal scenarios requiring long-context reasoning (e.g., long video understanding, multi-page document analysis).
- The "compression + re-injection" concept of PVC can be extended to other long-sequence attention optimization scenarios.
- Unlike FastV (which prunes redundant visual tokens based on attention weights), TVC targets dynamic visual maintenance during the reasoning process.
- The quality control pipeline used in distillation (dual-temperature + best-of-N + dynamic truncation) provides valuable references for constructing other reasoning datasets.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: The quantitative diagnosis of the visual forgetting phenomenon is novel and important, and the TVC solution is highly intuitive.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Comprehensive experiments across 5 benchmarks, multi-scale models (7B/72B), ablation studies, and data scaling curves.
- **Writing Quality** ⭐⭐⭐⭐: Clear logical flow from problem diagnosis to solution design, with high-quality visualizations.
- **Value** ⭐⭐⭐⭐: Visual forgetting is an important bottleneck in multimodal reasoning; TVC provides an effective engineering solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)
- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)
- [\[CVPR 2026\] Visual-Aware CoT: Achieving High-Fidelity Visual Consistency in Unified Models](../../CVPR2026/vlm_reasoning/visual-aware_cot_achieving_high-fidelity_visual_consistency_in_unified_models.md)
- [\[CVPR 2026\] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](../../CVPR2026/vlm_reasoning/virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](../../CVPR2025/vlm_reasoning/insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
