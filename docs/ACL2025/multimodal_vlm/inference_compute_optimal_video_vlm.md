---
title: >-
  [Paper Note] Inference Compute-Optimal Video Vision Language Models
description: >-
  [ACL 2025 (Long Paper)][Multimodal VLM][inference compute-optimal] This paper presents the first systematic study on the optimal allocation of inference compute budget for video VLMs. Under a fixed inference FLOPs constraint, through large-scale training sweeps (~100k A100 hours) and add-interact parametric modeling ($R^2$=0.98), the optimal trade-off strategy across three dimensions—language model size $x_N$, frame count $x_T$, and visual token count per frame $x_V$—is ident…
tags:
  - "ACL 2025 (Long Paper)"
  - "Multimodal VLM"
  - "inference compute-optimal"
  - "video VLM"
  - "scaling laws"
  - "frame count"
  - "visual token count"
  - "model size"
  - "parametric modeling"
date: 2026-05-08
content_hash: 9b5366d2066ff216
---

# Inference Compute-Optimal Video Vision Language Models

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2505.18855](https://arxiv.org/abs/2505.18855)  
**Code**: [github/vvlm_inference_scaling](https://github.com/tt6746690/vvlm_inference_scaling)  
**Area**: Multimodal VLM  
**Keywords**: inference compute-optimal, video VLM, scaling laws, frame count, visual token count, model size, parametric modeling  
**Authors**: Peiqi Wang (MIT), ShengYun Peng (Georgia Tech), Xuewen Zhang, Hanchao Yu, Yibo Yang, Fujun Liu, Qifan Wang (Meta), Lifu Huang (UC Davis)

## TL;DR

This paper presents the first systematic study on the optimal allocation of inference compute budget for video VLMs. Under a fixed inference FLOPs constraint, through large-scale training sweeps (~100k A100 hours) and add-interact parametric modeling ($R^2$=0.98), the optimal trade-off strategy across three dimensions—language model size $x_N$, frame count $x_T$, and visual token count per frame $x_V$—is identified.

## Background & Motivation

**Background**: Video VLMs (e.g., LLaVA-Video, Qwen2-VL) have been widely deployed in industrial scenarios such as recommendation systems and content moderation, processing millions of videos daily. For these applications, the pre-training costs have been absorbed by the open-source community, and fine-tuning costs are negligible compared to inference costs (the paper estimates that monthly inference costs can reach up to 340 times the fine-tuning cost). Consequently, inference FLOPs have become the dominant operational expense.

**Limitations of Prior Work**: (a) Prior to deployment, video VLMs require the determination of three key design parameters: language model size $x_N$, frame count $x_T$, and token count per frame $x_V$. However, there has been a lack of systematic allocation strategies. (b) Existing work (e.g., Du et al., 2024) only studies the trade-off of $x_T$ vs $x_V$, ignoring the critical dimension of model size $x_N$. (c) Prior studies generally ignore the inference compute cost of the vision encoder, calculating only the LM cost, which leads to an overestimation of the benefit of increasing the frame count. (d) No work has explored how the fine-tuning data size $n$ interacts with the scaling factors $x$ to affect the optimal frontier.

**Key Challenge**: Practical deployment requires making an optimal three-dimensional allocation decision within a tight inference budget; however, complex interaction effects exist among the three factors, making the independent optimization of any single factor suboptimal.

**Goal**: Given a fixed inference compute budget $c$ and fine-tuning data size $n$, how can one select the optimal $(x_N, x_T, x_V)$ to maximize downstream task performance?

**Key Insight**: This work adapts the research paradigm of Chinchilla on training compute-optimality—consisting of large-scale sweeps, parametric modeling, and constrained optimization—and applies it to the inference compute-optimal problem. A critical distinction is that while the fine-tuning data size $n$ does not affect inference cost, it potentially influences the optimal configuration.

## Method

### Overall Architecture

The inference compute optimization problem is formulated as:

$$x^*(c; n) = \arg\min_{x \in \mathcal{X},\; c(x) \leq c} f(x, n)$$

where $f(x, n)$ represents the downstream task error, and $c(x)$ represents the inference compute cost. The overall workflow consists of three steps: (1) gathering empirical data points $(x, n, f)$ via large-scale training sweeps; (2) fitting a parametric performance model; and (3) solving a constrained discrete optimization problem to obtain the optimal frontier $x^*(c; n)$.

### Key Designs

1. **Inference Cost Model Incorporating the Vision Encoder**:

    - Inference FLOPs are calculated as: $c(x) = 2x_T(x_M \cdot x_W + x_N \cdot x_V)$
    - where $x_M$ is the parameter count of the vision model and $x_W$ is the number of visual features. For SoViT-400M, this becomes $c(x) = 2x_T(0.43\text{e}9 \cdot 768 + x_N \cdot x_V)$
    - **Key Difference**: Prior work only calculated the LM cost $c_\text{LM} = 2x_N x_T x_V$ while ignoring $c_\text{ViT} = 2x_M x_T x_W$. However, for a 7B model configuration with $x_V \approx 50$, the vision model accounts for approximately 50% of the total inference FLOPs.
    - **Design Motivation**: Neglecting the cost of the vision encoder systematically overestimates the benefits of increasing the frame count (since each additional frame requires extra computation from the vision encoder).

2. **add-interact Parametric Performance Model**:

    - The task error is modeled as an additive power-law formulation with an interaction term:

    $$f(x, n) = \sum_k \alpha_k x_k^{-a_k} + \sum_k \beta_k x_k^{b_k} n^{-d} + \xi n^{-d} + \varepsilon$$

    - Meaning of terms: $\alpha_k x_k^{-a_k}$ represents the error reduction as $x_k$ increases (power-law decay under infinite data); $\beta_k x_k^{b_k} n^{-d}$ is the interaction term between scaling factors and data size—when $b_k > 0$, a larger $x_k$ implies richer visual information, requiring more data to fully exploit but yielding a higher marginal gain per sample; $\xi n^{-d}$ is the error term depending solely on the data size; and $\varepsilon$ is the irreducible error.
    - **Design Motivation**: Simple additive power-laws (add) or multiplicative power-laws (mult) fail to capture the interaction effect between $x$ and $n$. The add-interact model captures the critical phenomenon that "more visual details make the data richer yet more complex" via the interaction term $\beta_k x_k^{b_k} n^{-d}$.

3. **Two Types of Training Sweep Strategies**:

    - **Star sweep**: Centered around a high-computation configuration of $(7.5\text{B}, 32, 196)$, it fixes two factors and varies the third, fine-tuning across three data sizes: $\{0.25\text{M}, 0.5\text{M}, 1\text{M}\}$. This avoids expensive full-grid searches and provides more accurate estimations of each factor's scaling exponent.
    - **IsoFLOP sweep**: Adjusts $(x_N, x_T, x_V)$ under fixed inference FLOPs (2, 5, 15, 30 TFLOPs), fine-tuning on $n=2\text{M}$. It is used to identify the optimal configuration under a given budget and serves as a validation set to evaluate the extrapolation capability of the parametric model.
    - Sweep ranges: $x_N \in \{1\text{B}, 2.8\text{B}, 7.5\text{B}\}$, $x_T \in \{4, 8, 12, 16, 32\}$, $x_V \in \{4, 16, 25, 36, 49, 100, 196\}$.

### Model Fitting & Validation

- **Fitting Method**: Minimize MSE in log space using L-BFGS optimization, selecting the best solution from 500 random initializations.
- **Bootstrap Aggregation**: Performs 100 bootstrap resamplings to train base models and aggregates using the median, addressing the high variance when fitting parametric models over ~100 data points.
- **Model Selection**: Compares four parametric forms: mult, add, add-interact-s, and add-interact. The add-interact model performs best in both in-distribution CV and extrapolation.

### Elasticity Analysis

The concept of "elasticity" from economics is introduced to quantify the impact of data size on the optimal frontier:

$$e_k(c, n) = \frac{\partial x_k^*(c; n)}{\partial n} \cdot \frac{n}{x_k^*(c; n)}$$

$e_T = 0.1$ means that when the data size increases by 1%, the optimal frame count increases by 0.1%. Forward difference is used to approximate the derivative, averaged over 300 inference budgets and 100 data sizes.

## Key Experimental Results

### Parametric Model Comparison

| Model | CV MSE | CV $E\%$ | CV $R^2$ | Extrapolation MSE | Extrapolation $E\%$ | Extrapolation $R^2$ |
|------|--------|----------|----------|----------|-----------|-----------|
| mult | 1.21 | 1.62 | 0.88 | 6.73 | 3.55 | 0.45 |
| add | 0.56 | 1.11 | 0.94 | 2.04 | 2.15 | 0.83 |
| add-interact-s | 0.24 | 0.80 | 0.97 | 0.94 | 1.32 | 0.92 |
| **add-interact** | **0.20** | **0.77** | **0.98** | **0.95** | **1.33** | **0.92** |

### Key Findings

1. **Diminishing Returns Across Three Dimensions**: Individually increasing $x_N$, $x_T$, or $x_V$ shows diminishing marginal returns, with the rate of decay varying across tasks.
2. **Joint Scaling is Crucial**: Scaling from 15 to 30 TFLOPs yields negligible gains for the $x_N=1\text{B}$ model (due to a model capacity bottleneck), whereas $x_N=7.5\text{B}$ obtains significant improvements. Similar bottleneck effects exist for $x_T$ and $x_V$.
3. **Varying Scaling Rates**: The marginal gains of $x_T$ (frame count) are generally greater than those of $x_V$ (token count). Improving vision model efficiency is more valuable than reducing the LM cost of processing each frame.
4. **Strong Task Specificity**: Long-form video understanding (LongVideoBench) favors increasing $x_T$, while fine-grained perception (PerceptionTest) favors increasing $x_V$—**there is no universal allocation strategy**.
5. **Data Size Shifts the Optimal Frontier**: As the data size increases, elasticity analysis reveals $e_N < 0$ (smaller model), $e_T > 0$ (higher frame count), and $e_V > 0$ (higher token count). The trend is consistent across tasks, although the magnitudes vary.
6. **Advantage of Compute-Optimal Configurations**: Under equivalent inference FLOPs, optimal configurations improve average task performance by 5-15% compared to naive configurations.

### Elasticity (Data Sensitivity)

| Factor | Average Elasticity | Meaning |
|------|---------|------|
| $x_N$ | -0.22 | Data ↑ $\rightarrow$ Optimal model size ↓ |
| $x_T$ | +0.17 | Data ↑ $\rightarrow$ Optimal frame count ↑ |
| $x_V$ | +0.79 | Data ↑ $\rightarrow$ Optimal token count ↑ |

## Highlights & Insights

- **"Inference-focused Chinchilla" Positioning**: Successfully adapts the classic training compute-optimal paradigm—training sweeps, parametric modeling, and constrained optimization—to the inference stage with methodological rigor and high reproducibility.
- **Vision Encoder Cost Correction**: Curates the systematic bias in prior work (Li et al., 2024; Du et al., 2024) that ignored vision encoder cost, which is particularly significant for small-model and low-token configurations.
- **Crucial Interaction Term**: The core improvement of add-interact over add lies in modeling the interaction between scaling factors and data size—"more frames/tokens make data richer but also more complex"—which cannot be captured by add or mult models and directly influences how the optimal frontier depends on data size.
- **Highly Practical Elasticity Analysis**: The economic concept of elasticity provides intuitive and comparable quantitative metrics—$e_V = 0.79$ vs $e_T = 0.17$ indicates that the optimal token count is much more sensitive to data size than the frame count.
- **Practical Actionable Guidance**: Offers clear guidance for industrial deployment: (1) instead of merely scaling up the model size, scale all three factors jointly; (2) allocate to smaller models, fewer frames, and fewer tokens for low budgets, and scale all three together for high budgets; (3) as fine-tuning data is accumulated over time, gradually increase visual information density and scale down the model size appropriately.

## Limitations & Future Work

- **Evaluation Restricted to LLaVA-like Architectures**: Uses Llama-3.2 series (1B/2.8B/7.5B). The lack of available pre-trained models between 8B and 70B constrains a comprehensive characterization of language model scaling effects. It remains unknown whether the findings apply to dynamic-resolution architectures like Qwen2-VL.
- **Heuristically Selected Parametric Form**: The functional form of add-interact is selected from only four candidates, which may not be the optimal representation. Furthermore, it shows relatively high extrapolation error ($E\% \geq 5\%$) on certain tasks (LongVideoBench, Next-QA).
- **Unconsidered Scaling Factors**: Factors such as vision encoder size, downsampling methods, and training strategies are held constant, which could represent important degrees of freedom.
- **Simplified Inference Cost Estimation**: Only theoretical FLOPs during the prefill phase are considered. Real-world deployment issues like hardware utilization, memory-bound decoding, model quantization, and speculative decoding are not addressed.
- **High Resource Barrier**: The study demands ~100k A100 hours of experimental investment, which is non-reproducible for the vast majority of researchers.

## Related Work & Insights

- **vs Chinchilla (Hoffmann et al., 2022)**: Chinchilla optimizes the allocation of training compute between model size and pre-training data size. This work optimizes the allocation of inference compute among $x_N$, $x_T$, and $x_V$. The key difference is that in the inference scenario, $n$ does not affect compute costs but does affect the optimal configuration.
- **vs Du et al. (2024)**: Studies only the trade-off of $x_T$ vs $x_V$; (1) fails to include $x_N$, (2) ignores vision encoder costs, and (3) assumes a parametric model without directly observing trends from empirical training sweeps.
- **vs Li et al. (2024b)**: Concludes that "VLMs require fewer visual tokens and more parameters" but neglects vision encoder costs and the interaction with data size.
- **vs EfficientNet (Tan & Le, 2019)**: EfficientNet introduces compound scaling for jointly scaling CNN width, depth, and resolution. This work extends a similar concept to $x_N$, $x_T$, and $x_V$ for video VLMs.
- **vs Sardana et al. (2024)**: Also targets deployment costs, but focuses on LM pre-training rather than the fine-tuning scenarios of video VLMs.

## Insights & Connections

- This "inference compute-optimal" paradigm can be extended to any multimodal model, i.e., how to allocate inference budget among visual resolution, encoder size, and LLM size.
- The finding that "data size shifts the optimal frontier" implies that in scenarios with continuous data collection, the deployment configuration should be dynamically adjusted, unlike standard static deployment settings.
- The observation that $x_T$ yields higher scaling benefits than $x_V$ suggests that while token compression/merging techniques reduce $x_V$, allocating the saved budget to more frames may yield better performance.
- It raises concerns regarding fair comparison with methods like AuroraCap: fixing $x_T x_V$ (total token count) while having inconsistent allocations of $x_T$ vs $x_V$ leads to unfair comparisons.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first systematic study of inference compute-optimal allocation in video VLMs, featuring a comprehensive methodology and a unique niche.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale training sweeps utilizing ~100k A100 hours, covering 8 video tasks, comparing 4 parametric models, with thorough ablation and elasticity analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and clear mathematical modeling, with comparisons to Chinchilla easing comprehension, and a highly detailed Appendix (6 sections covering FAQ, implementations, sweeps, fitting, elasticity, and results).
- Value: ⭐⭐⭐⭐⭐ Provides direct and practical guidance for the industrial deployment of video VLMs, with a methodology transferable to other multimodal scaling problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SITCOM: Scaling Inference-Time COMpute for VLAs](../../NeurIPS2025/multimodal_vlm/sitcom_scaling_inference-time_compute_for_vlas.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ICCV 2025\] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](../../ICCV2025/multimodal_vlm/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)

</div>

<!-- RELATED:END -->
