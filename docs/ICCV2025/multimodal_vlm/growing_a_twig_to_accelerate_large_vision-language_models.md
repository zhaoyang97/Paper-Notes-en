---
title: >-
  [Paper Note] Growing a Twig to Accelerate Large Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][VLM acceleration] This paper proposes TwigVLM, which attaches a lightweight twig module to the early layers of a VLM to simultaneously enable twig-guided visual token pruning (TTP…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VLM acceleration"
  - "visual token pruning"
  - "self-speculative decoding"
  - "lightweight module"
  - "inference efficiency"
date: 2026-05-08
content_hash: 52ccbc4efcb61c89
---

# Growing a Twig to Accelerate Large Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2503.14075](https://arxiv.org/abs/2503.14075)  
**Code**: [github.com/MILVLG/twigvlm](https://github.com/MILVLG/twigvlm)  
**Area**: Multimodal VLM
**Keywords**: VLM acceleration, visual token pruning, self-speculative decoding, lightweight module, inference efficiency

## TL;DR

This paper proposes TwigVLM, which attaches a lightweight twig module to the early layers of a VLM to simultaneously enable twig-guided visual token pruning (TTP, for prefilling acceleration) and self-speculative decoding (SSD, for decoding acceleration). On LLaVA-1.5-7B, TwigVLM retains 96% accuracy after pruning 88.9% of visual tokens and achieves a 154% speedup in long-answer generation, substantially outperforming existing methods in both accuracy and speed.

## Background & Motivation

### Problem Definition

Large vision-language models (VLMs) achieve strong performance on open-world multimodal understanding, but their high computational overhead severely hinders practical deployment. VLM inference consists of two phases—prefilling (processing input tokens) and decoding (generating answer tokens one by one)—and both phases must be accelerated to achieve practically usable inference efficiency.

### Limitations of Prior Work

**Poor attention signal quality in early layers**: Methods such as FastV use attention maps from the 2nd layer of the VLM to guide visual token pruning, but early-layer attention is task-insensitive—nearly identical tokens are selected across different prompts—leading to significant accuracy degradation after pruning.

**Only prefilling is accelerated**: Existing token pruning methods (FastV, SparseVLM, VisionZip, etc.) primarily target prefilling acceleration, whereas the decoding phase accounts for the largest share of inference latency. When generating $\geq 32$ tokens, decoding time already far exceeds prefilling time.

**KV-cache limits the decoding benefit of pruning**: Due to the KV-cache mechanism, visual token pruning yields limited decoding speedup for self-attention and provides no speedup at all for the FFN layers, which dominate computation.

### Core Motivation

Two carefully designed pilot experiments reveal key insights:

**Pilot Experiment 1 (Attention quality)**: Using attention maps from layers at depth $D$ to guide token pruning at layer $K=2$ shows that deep-layer attention ($D=18$) outperforms shallow-layer attention ($D=2$) by approximately 14 percentage points in RelAcc. Deeper layers are closer to the prediction head and their attention maps more accurately capture inter-token multimodal relationships.

**Pilot Experiment 2 (Latency analysis)**: When the generation length $S \geq 32$, decoding time far exceeds prefilling time. Although FastV effectively reduces prefilling time, its decoding speedup is negligible (RelSpd of only 104.3%).

**Core question**: Can both problems—poor early-layer attention quality and insufficient decoding acceleration—be addressed within a unified framework?

## Method

### Overall Architecture

The core idea of TwigVLM is elegant and concise: a lightweight twig module consisting of $T$ transformer layers is attached after the $K$-th layer of a pretrained VLM, forming a shallow sub-network $\mathcal{M}_s$. After training, this twig module serves two purposes simultaneously:
1. **Prefilling phase**: The deep attention from the twig's last layer (rather than shallow VLM attention) guides visual token pruning (TTP).
2. **Decoding phase**: The shallow network acts as a draft model and the full VLM as a target model for self-speculative decoding (SSD).

### Key Designs

#### 1. **Twig Module Architecture and Training**

- **Function**: Append $T$ transformer blocks after the $K$-th layer of the VLM to form a shallow sub-network.
- **Mechanism**:
    - The full VLM is denoted $\mathcal{M}_b = \{\mathcal{T}_l\}_{l=1}^L$; the twig module is $\{\mathcal{G}_t\}_{t=1}^T$.
    - The shallow network is $\mathcal{M}_s = \{\mathcal{T}_k\}_{k=1}^K \cup \{\mathcal{G}_t\}_{t=1}^T$, where $K+T \ll L$.
    - Initialization: the $T$ twig layers are copied from VLM layers $K+1$ through $K+T$ (optimal choice, as the input distribution is best aligned).
    - During training, all VLM parameters are frozen; only the twig module ($T$ layers + prediction head) is trained using the same data and standard autoregressive loss as the VLM.
    - Training cost is approximately 10% of that required to train the VLM.
- **Design Motivation**: Initializing from adjacent VLM layers ensures that the twig's input–output distribution is aligned with the VLM, allowing the twig to acquire near-deep-layer semantic understanding at minimal training cost.

#### 2. **Twig-guided Token Pruning (TTP)**

- **Function**: Use the attention map from the twig's last layer to guide visual token pruning at layer $K$.
- **Mechanism**:

$$\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)} = \mathcal{P}(\mathbf{X}_{\mathcal{M}_b}^{(K)}, \mathbf{A}_{\mathcal{M}_s}^{(K+T)}, R)$$

  where $\mathcal{P}$ is a FastV-style TopR selection function, but the attention source is replaced from layer $K$ of the VLM to layer $K+T$ of the twig.

  An additional **FinalWipe** strategy is introduced: all visual tokens are removed after the $K_f$-th VLM layer (e.g., layer 24). The average number of retained tokens is redefined as:

$$\bar{R} = [M \times K + R \times (K_f - K)] / L$$

  This allows a larger $R$ to be used at the same $\bar{R}$, retaining more visual tokens in intermediate layers and thereby improving accuracy.

- **Design Motivation**: Pilot Experiment 1 demonstrates that deep attention is more effective than shallow attention for token selection. Although the twig layer is at the same depth as VLM layer $K+T$, its attention maps are of higher quality (96.0% vs. 86.2% RelAcc) because it is directly connected to the prediction head.

#### 3. **Self-Speculative Decoding (SSD)**

- **Function**: Use the shallow network to rapidly generate draft tokens, which are then verified in parallel by the deep network to accelerate decoding.
- **Mechanism**:
    - The shallow network $\mathcal{M}_s$ (draft model) autoregressively generates multiple draft tokens.
    - The deep network $\mathcal{M}_b$ (target model) verifies these tokens in parallel in a single forward pass.
    - Accepted tokens are used directly as final outputs; rejected tokens are resampled from the rejection point.
    - Key advantage: the draft and target models share the computation and KV-cache of the first $K$ layers, further improving efficiency.
    - SSD produces outputs **identical** to those of the target model—accuracy is unaffected.
- **Design Motivation**: Pilot Experiment 2 shows that decoding is the primary bottleneck in long-answer scenarios. TwigVLM naturally contains both a shallow and a deep sub-network, enabling speculative decoding without any additional model.

### Loss & Training

- **Loss function**: Standard autoregressive language modeling loss.
- **Training strategy**: Only the twig blocks are trained; all VLM parameters are frozen.
- **Default hyperparameters**: $T=3$, $K=2$, $K_f=24$.
- **Training data**: Same multimodal instruction-tuning data as the base VLM.
- **Hardware**: 8× A100 GPUs.

## Key Experimental Results

### Main Results

**88.9% visual token pruning on LLaVA-1.5-7B (average 64 tokens retained)**:

| Method | GQA | MMB | MME | VQA-T | SQA-I | VQA-V2 | RelAcc |
|--------|-----|-----|-----|-------|-------|--------|--------|
| LLaVA-1.5-7B (upper bound) | 61.9 | 64.7 | 1862 | 58.2 | 69.5 | 78.5 | 100% |
| FastV | 44.1 | 45.9 | 1218 | 50.7 | 70.0 | 52.0 | 77.0% |
| VisionZip‡ | 57.0 | 61.5 | 1756 | 56.0 | 68.8 | 74.2 | 95.2% |
| **TwigVLM** | **58.8** | **60.4** | **1760** | **55.8** | **70.0** | **75.6** | **96.0%** |

**Generation speed comparison (MM-Vet long answers, $\bar{S}$ ≈ 100 tokens)**:

| Method | RelSpd |
|--------|--------|
| FastV ($\bar{R}$=64) | ~104% |
| VisionZip ($\bar{R}$=64) | ~106% |
| **TwigVLM ($\bar{R}$=64)** | **~154%** |

**Video-LLaVA video understanding (135 tokens retained)**:

| Method | TGIF | MSVD | MSRVTT | ActivityNet | RelAcc |
|--------|------|------|--------|-------------|--------|
| FastV | 23.1 | 38.0 | 19.3 | 30.6 | 52.1% |
| VisionZip | 42.4 | 63.5 | 52.1 | 43.0 | 93.2% |
| **TwigVLM** | **44.7** | **68.3** | **54.6** | 41.5 | **96.3%** |

### Ablation Study

| Configuration | RelAcc | RelSpd | Notes |
|---------------|--------|--------|-------|
| Attention from VLM layer $K$ | 82.3% | — | Shallow attention signal is poor |
| Attention from VLM layer $K+T$ | 86.2% | — | Same depth but far from prediction head |
| **Attention from twig last layer** | **96.0%** | — | Close to prediction head; highest quality |
| Token pruning only (FastV) | — | 104.3% | Prefilling acceleration only |
| SSD only | — | 146.7% | Decoding acceleration only |
| **TTP + SSD** | — | **153.6%** | Complementary; maximum speedup |
| Random initialization | 87.2% | 120.4% | Lacks VLM knowledge |
| Initialized from last $T$ VLM layers | 90.4% | 131.4% | Distribution mismatch |
| **Initialized from VLM layers $K$:$K+T$** | **96.0%** | **153.6%** | Optimal; best distribution alignment |
| $T=1$ | 93.9% | 154.1% | Insufficient accuracy |
| **$T=3$** | **96.0%** | **153.6%** | Optimal accuracy–speed trade-off |
| $T=4$ | 95.8% | 145.4% | Reduced speed |

### Key Findings

1. **Attention depth determines pruning quality**: Twig-layer attention surpasses same-depth VLM-layer attention by 9.8% RelAcc, confirming that attention maps closer to the prediction head are more effective.
2. **TTP and SSD are perfectly complementary**: TTP accelerates prefilling while SSD accelerates decoding; their combination achieves an overall speedup of 153.6%.
3. **Initialization strategy is critical**: Initializing from adjacent VLM layers outperforms random initialization by 8.8% RelAcc and 33.2% RelSpd.
4. **FinalWipe improves accuracy**: Removing visual tokens in later layers does not degrade accuracy (since the contribution of visual tokens diminishes in deeper layers) and allows more tokens to be retained in intermediate layers.
5. **Strong generalization to video tasks**: TwigVLM improves RelAcc on Video-LLaVA from 93.2% (VisionZip) to 96.3%.

## Highlights & Insights

1. **Pilot-experiment-driven design**: Two pilot experiments precisely identify two key weaknesses of prior methods; the proposed method is designed directly around these findings, resulting in a clear and coherent narrative.
2. **One module, two uses**: The twig module serves dual purposes—token selection and speculative decoding—making the architecture elegantly unified.
3. **SSD incurs no accuracy loss**: Self-speculative decoding produces outputs identical to those of the original model; accuracy is completely unaffected by SSD.
4. **Breakthrough in long-answer scenarios**: In benchmarks requiring long answers such as MM-Vet, TwigVLM's speedup (154%) far exceeds that of token-pruning-only methods (~104%).
5. **Training efficiency**: Only a small fraction of the VLM's parameters need to be trained, with training time approximately 10% of that required for the full VLM.

## Limitations & Future Work

1. **Additional computation from the twig module**: Although the twig is shallow (3 layers), it still introduces additional attention computation during the prefilling phase.
2. **Requires post-training**: A twig module must be trained separately for each base VLM, making the approach less flexible than purely inference-time methods such as FastV.
3. **Hyperparameter sensitivity**: The optimal combination of $K$, $T$, and $K_f$ must be tuned individually for each VLM.
4. **Larger models not explored**: Main experiments focus on 7B models; performance on larger models (e.g., 72B) remains to be validated.
5. **Draft token acceptance rate is a bottleneck**: When the distribution gap between the twig and the VLM is large, the acceptance rate of draft tokens decreases.

## Related Work & Insights

- **Relation to FastV**: TwigVLM addresses two fundamental limitations of FastV—insensitive shallow-layer attention and inability to accelerate decoding.
- **Distinction from VisionZip**: VisionZip prunes tokens at the visual encoding stage without access to the text prompt, whereas TwigVLM prunes within the LLM using textual context to select task-relevant visual tokens.
- **Distinction from standalone speculative decoding methods**: Conventional speculative decoding requires an independent smaller model, whereas the draft and target models in TwigVLM share the first $K$ layers, requiring no additional model.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — An elegant unified framework that simultaneously addresses token pruning and decoding acceleration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three VLM benchmarks (image + video), six ablation dimensions, and detailed speed comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative logic from pilot experiments → motivation → method → experiments is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ — First to highlight the importance of decoding acceleration for VLM deployment, offering a concise and effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

</div>

<!-- RELATED:END -->
