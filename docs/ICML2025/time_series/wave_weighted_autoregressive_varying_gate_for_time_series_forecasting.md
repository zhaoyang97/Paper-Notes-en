---
title: >-
  [Paper Note] WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting
description: >-
  [ICML 2025][Time Series][Time Series Forecasting] Introduces the classic statistical ARMA (autoregressive moving average) structure into the autoregressive Transformer attention mechanism. By employing an indirect MA weight generation method, it decouples short- and long-term temporal patterns without increasing time complexity or parameter count, significantly improving time series forecasting performance.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Time Series Forecasting"
  - "ARMA"
  - "Autoregressive Attention"
  - "Moving Average"
  - "Linear Attention"
date: 2026-05-08
content_hash: 300ce65a93d6150e
---

# WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting

**Conference**: ICML 2025  
**arXiv**: [2410.03159](https://arxiv.org/abs/2410.03159)  
**Code**: Yes  
**Area**: Time Series  
**Keywords**: Time Series Forecasting, ARMA, Autoregressive Attention, Moving Average, Linear Attention

## TL;DR

Introduces the classic statistical ARMA (autoregressive moving average) structure into the autoregressive Transformer attention mechanism. By employing an indirect MA weight generation method, it decouples short- and long-term temporal patterns without increasing time complexity or parameter count, significantly improving time series forecasting performance.

## Background & Motivation

**Background**: Recently, decoder-only autoregressive Transformers have achieved great success in fields such as NLP, CV, and audio. However, in the field of Time Series Forecasting (TSF), state-of-the-art models are still dominated by encoder-only Transformers (e.g., PatchTST, iTransformer), MLPs (e.g., TiDE, CATS), or even linear models (DLinear, FITS). The few explorations into autoregressive models mainly focus on using pretrained LLMs for few-shot/zero-shot forecasting, with rarely any direct evaluation on the performance of end-to-end trained AR Transformers.

**Limitations of Prior Work**:
1. Autoregressive Transformers face **error accumulation** issues in long-term forecasting—iterative step-by-step forecasting leads to cumulative error amplification.
2. The **exponentially decaying gating** introduced in existing efficient linear attention mechanisms (e.g., gated linear attention) can enhance local pattern modeling, but weakens the ability to capture long-term and periodic patterns.
3. The coupling of **long-term periodic patterns and short-term local effects** in time series is a key conflict, which a simple EMA mechanism cannot decouple effectively.

**Key Challenge**: The decay factor in gated linear attention forces AR weights to focus on recent tokens, but TSF data often contains stable seasonal effects that should not decay over time. The main challenge is how to allow the model to model short-term fluctuations while capturing long-term stable periodic patterns.

**Ours**: Inspired by the classic Box-Jenkins ARMA model, a complete MA (moving average) term is introduced into the existing AR attention mechanism to construct WAVE attention. The MA term specifically processes short-term fluctuations, allowing the AR term to focus on long-term and periodic patterns.

**Key Insight**: This work first demonstrates that with appropriate tokenization, AR Transformers can achieve SOTA performance, and then progressively improves it through the ARMA structure. An indirect MA weight generation method is designed to avoid explicitly computing an $N \times N$ MA weight matrix, thereby maintaining $O(N)$ time complexity.

## Method

### Overall Architecture

A GPT-2 style decoder-only Transformer architecture is adopted. After channel-independent processing and RevIN normalization of the input time series, non-overlapping patching is used for tokenization (patch size = $L_P$, which is the forecasting length), dividing the input into $N = (L_I + P) / L_P$ tokens. Each token is linearly projected into a $d$-dimensional space, added with a learnable positional embedding, and is fed into $m$ layers of WAVE Transformers. The output of the last token corresponds to the forecasting output of length $L_P$, which avoids the error accumulation issue of iterative step-by-step forecasting.

### Key Designs

1. **Appropriate Tokenization to Avoid Error Accumulation**: Borrowing the patch strategy from PatchTST, the patch size is set to the forecasting length $L_P$, ensuring that the autoregressive "next token prediction" covers the entire forecasting horizon. Thus, a single-step prediction completes the entire forecasting process without iteration. Meanwhile, a channel-independent approach is adopted, where each channel is predicted independently and RevIN normalization is applied. Core formula:

    $N = \frac{L_I + P}{L_P}$

   where $P$ represents zero-padding to ensure exact division. **Design Motivation**: Eliminate the error accumulation problem of decoder-only architectures, rendering their performance comparable with encoder-only models.

2. **ARMA Attention Structure (WAVE Attention)**: The standard attention output is decomposed into an AR term and an MA term. The AR term is computed by the original attention mechanism, while the MA term models the short-term patterns of predictions residuals:

    $\bm{v}_{t+1} = \underbrace{\sum_{i=1}^{t} \mathbf{w}_{t,i} \odot \bm{v}_i}_{\text{AR term } \bm{o}_t^{AR}} + \underbrace{\sum_{j=1}^{t-1} \bm{\theta}_{t-1,j} \odot \bm{\epsilon}_j}_{\text{MA term } \bm{o}_t^{MA}} + \bm{\epsilon}_t$

   where $\bm{\epsilon}_t$ is the residual error after introducing the MA term, and $\bm{\theta}_{t-1,j}$ represents the MA weights. This design originates from the classic ARMA model—the AR term captures long-term dependencies and periodic patterns, and the MA term captures short-term fluctuations and local effects, achieving effective decoupling between the two.

3. **Indirect MA Weight Generation Method**: Directly calculating the MA weights requires inverting an $N \times N$ matrix ($\bm{\epsilon} = (\mathbf{I} + \mathbf{\Theta})^{-1} \mathbf{r}$), which brings the complexity back to $O(N^2)$. The core innovation of this paper is to replace $\bm{\epsilon}_j$ with the AR residuals $\bm{r}_j = \bm{v}_{j+1} - \bm{o}_j^{AR}$ as the value input to the MA term:

    $\bm{o}_t^{MA} = \sum_{j=1}^{t-1} \bm{\beta}_{t-1,j} \odot \bm{r}_j$

   where $\bm{\beta}_{t-1,j} = \phi_q^{MA}(\bm{q}_{t-1}^{MA}) \phi_k^{MA}(\bm{k}_j^{MA})^\top$ is efficiently computed using the linear attention formulation. The relationship between the indirectly generated weight matrix $\mathbf{B}$ and the implicit MA weight matrix $\mathbf{\Theta}$ is:

    $$\mathbf{B} = \mathbf{\Theta} \cdot (\mathbf{I} + \mathbf{\Theta})^{-1}, \quad \mathbf{\Theta} = \mathbf{B} \cdot (\mathbf{I} - \mathbf{B})^{-1}$$

   **Design Motivation**: Achieve $O(N)$ complexity for linear attention while generating valid MA weights.

4. **Activation Function Selection to Ensure MA Weight Properties**: The MA term is expected to model short-term effects, thus requiring the implicit $\mathbf{\Theta}$ to present a pattern where **near-diagonal elements are large and elements decay away from the diagonal**. Expanding $\mathbf{\Theta}$ as $\mathbf{B} + \mathbf{B}^2 + \mathbf{B}^3 + \cdots$, if the mean of $\beta$ is $b$, we have:

    $$\theta_{ij} = b(1+b)^{i-j-1}, \quad i > j$$

   To guarantee decay, it is required that $b \in (-1, 0)$. The final selection is:
    - Key activation: $\phi_k^{MA}(\bm{k}_j^{MA}) = \sigma(\alpha \bm{k}_j^{MA} / \sqrt{d})$ (sigmoid, $\alpha=0.05$)
    - Query activation: $\phi_q^{MA}(\bm{q}_t^{MA}) = -\text{LeakyReLU}(-\bm{q}_t^{MA} / \sqrt{d})$ (negative slope of 0.02)

   **Design Motivation**: LeakyReLU provides flexibility—the majority of negative outputs ensure the negative smoothing effect of the MA term, while a few positive values enhance the modeling flexibility.

5. **Parameter Sharing Strategy**: The MA term introduces additional projection matrices $\mathbf{W}_q^{MA}, \mathbf{W}_k^{MA}, \mathbf{W}_v^{MA}$. For a fair comparison, the AR and MA terms share $\mathbf{W}_q$, and $\mathbf{W}_v$ for the MA term is set as the identity matrix. The final trainable parameters are $\mathbf{W}_q, \mathbf{W}_k^{AR}, \mathbf{W}_k^{MA}, \mathbf{W}_o$—equivalent to the parameter count of a pure AR model.

### Loss & Training

- **Loss Function**: Standard MSE loss using a next-step prediction objective. The loss on the last token is multiplied by a weighting factor $N$ (only minimally impacts performance on small ETT datasets).
- **Optimizer**: AdamW ($\beta=(0.9, 0.95)$, weight decay = 0.1), following the GPT-2 setup.
- **Learning Rate**: Linear warm-up during the first 5 epochs ($6\times10^{-5} \to 6\times10^{-4}$), followed by stepwise decay.
- **Regularization**: 0.1 dropout is applied to both the AR and MA terms.
- **Model Dimension**: $d = 16\lfloor\sqrt{C}\rfloor$ ($C$ is the number of channels), $m=3$ layers, 8 heads.
- **Early Stopping**: patience = 12 epochs, maximum 100 epochs.
- **Normalization**: RevIN + RMSNorm.

## Key Experimental Results

### Main Results

Short-term forecasting evaluation on 12 datasets ($L_P \in \{12, 24, 48, 96\}$, $L_I=512$):

| Dataset | Metric (MSE) | WAVE Lin Attn | Pure AR Lin Attn | PatchTST | iTransformer | Gain (vs AR) |
|--------|-----------|---------------|---------------|----------|--------------|------------|
| Weather | Avg MSE | 0.100 | 0.104 | 0.107 | 0.117 | 3.8% |
| ETTm1 | Avg MSE | 0.222 | 0.238 | 0.244 | 0.259 | 6.7% |
| Traffic | Avg MSE | 0.330 | 0.337 | 0.358 | 0.330 | 2.1% |
| PEMS08 | Avg MSE | 0.116 | 0.119 | 0.121 | 0.117 | 2.5% |
| Solar | Avg MSE | 0.119 | 0.122 | 0.150 | 0.145 | 2.5% |

**Rank Statistics**: WAVE Lin Attn achieves an average rank of 2.333 (#Top1 = 25/48), far surpassing all baselines and the pure AR model.

### Ablation Study

| Configuration | Weather MSE | ETTm1 MSE | Description |
|------|-------------|-----------|------|
| WAVE (m=3) vs AR (m=1~8) | 0.100 vs 0.102~0.109 | 0.222 vs 0.230~0.241 | 3-layer WAVE outperforms AR with any number of layers |
| Lin Attn AR | 0.104 | 0.238 | Base linear attention |
| Lin Attn +ARMA | 0.100 | 0.222 | Consistent improvement after adding the MA term |
| GLin Attn AR | 0.119 | 0.407 | Gated decay damages long-term patterns |
| GLin Attn +ARMA | 0.105 | 0.260 | ARMA yields the most significant gain for gated attention |
| MEGA (EMA) | 0.121 | 0.412 | EMA is not as effective as ARMA in decoupling |

### Key Findings

1. **With appropriate tokenization, AR Transformers can rival SOTA models**: Once the patch tokenization strategy is applied to pure AR Transformers, their performance becomes comparable to established baselines like PatchTST and iTransformer.
2. **Consistent improvement from ARMA**: Across all 5 attention mechanisms evaluated (Softmax, Linear, Gated Linear, Element-wise Linear, Fixed), introducing the MA term consistently improves forecasting accuracy.
3. **Linear attention outperforms Softmax**: In TSF tasks, simpler linear attention mechanisms coupled with unnormalized input shortcuts demonstrate superior generalization capabilities.
4. **Gated attention benefits the most**: WAVE provides the most substantial improvement to gated linear attention, as the MA term assumes the responsibility of modeling local effects, thereby freeing the decay factor to perform its intended AR forgetting mechanism.
5. **No performance degradation when scaling up lookback window length**: When $L_I$ increases from 512 to 4096, the performance of both AR and WAVE models continuously improves, whereas baseline models typically suffer from performance degradation, highlighting the advantage of AR architectures in utilizing long-term dependencies.
6. **Extremely low computational overhead**: Parameter sharing ensures that the MA term does not introduce additional parameters, and the extra FLOPs are negligible (e.g., on ETTm1, Lin Attn rises minimally from 7.387M to 7.415M FLOPs).

## Highlights & Insights

1. **Elegant marriage of classic statistics and deep learning**: Successfully integrating the core concept of ARMA models (decoupling long- and short-term effects) into Transformer attention, backed by clear theoretical motivation and solid empirical results.
2. **Indirect MA weight generation as a key innovation**: Smartly utilizing $\bm{r}_j$ instead of $\bm{\epsilon}_j$ as the value input avoids matrix inversion and retains linear complexity, exhibiting outstanding skill and practicality.
3. **Theoretically grounded activation function design**: By analyzing the cumulative behavior of $\mathbf{\Theta} = \mathbf{B} + \mathbf{B}^2 + \cdots$, a constraint of $b \in (-1, 0)$ is derived, which guides the choice of the LeakyReLU+Sigmoid combination rather than relying on heuristic tuning.
4. **Deep analysis of EMA and gating mechanisms**: Points out the destructive effect of exponential decay on stable periodic patterns in TSF, explaining why gated linear attention alone is inferior to standard linear attention. This provides valuable insights for future studies in this field.
5. **Parameter sharing to maintain fairness**: By sharing $\mathbf{W}_q$ and setting $\mathbf{W}_v^{MA}=\mathbf{I}$, any skepticism regarding performance gains resulting from added parameters is thoroughly eliminated.

## Limitations & Future Work

1. **Limited to channel-independent mode**: The integration with multivariate relationship modeling (e.g., cross-variable attention in iTransformer) was not explored, potentially missing critical correlation information across variables.
2. **Evaluated solely on TSF tasks**: It remains unverified whether WAVE attention is applicable to general sequence modeling tasks such as NLP and audio processing.
3. **Lack of large-scale validation**: Tests on massive datasets (e.g., large-scale NLP pre-training) are absent, leaving the performance of the ARMA structure within large models to be determined.
4. **Fixed order of the MA term**: The current MA term resembles an infinite-order or full-order MA. Whether restricting the MA order (as in standard ARMA(p,q)) could further improve efficiency or performance remains unstudied.
5. **Sensitivity of the $\alpha$ parameter**: The key activation parameter $\alpha=0.05$ controls the degree to which the MA weights focus on long- or short-term information, and its sensitivity is not fully investigated in the paper.

## Related Work & Insights

- **PatchTST** (Nie et al., 2022): This work directly inherits the patch tokenization strategy from PatchTST, demonstrating that patching is equally effective in AR architectures.
- **Gated Linear Attention** (Yang et al., 2024): Based on this, the paper points out the limitations of pure decay mechanisms in TSF, which inspired the introduction of the ARMA structure.
- **MEGA** (Ma et al., 2022): Utilizes EMA to enhance gated attention; however, ARMA decouples long-term and short-term effects more effectively than EMA.
- **ARMA Cell** (Schiele et al., 2022): Attempted to introduce the ARMA structure into RNNs, yet failed to guarantee the short-term modeling properties of MA weights, resulting in performance that fell short of modern attention models.
- **RetNet** (Sun et al., 2023): Combines linear attention with retention mechanisms, complementing WAVE's efficient linear attention approach.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of incorporating the ARMA structure into the attention mechanism is novel, and the indirect MA weight generation method is clever, although the core is still based on existing linear attention frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive cross-validations across 12 datasets and 5 attention mechanisms, along with depth ablations, lookback window scaling analysis, computational cost comparisons, and visualization analyses—highly thorough.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured, with rigorous mathematical derivations and intuitive visualizations, although some portions feature dense notations that require careful reading.
- Value: ⭐⭐⭐⭐ Establishes a solid theoretical and practical foundation for the application of autoregressive Transformers in TSF; the ARMA structure holds great potential as a general plug-and-play module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[AAAI 2026\] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](../../AAAI2026/time_series/scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [\[ICLR 2026\] Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context–Detail Interaction](../../ICLR2026/time_series/enhancing_sparse_event_detection_in_healthcare_time-series_via_adaptive_gate_of_.md)
- [\[ICLR 2026\] Efficient Autoregressive Inference for Transformer Probabilistic Models](../../ICLR2026/time_series/efficient_autoregressive_inference_for_transformer_probabilistic_models.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)

</div>

<!-- RELATED:END -->
