---
title: >-
  [Paper Note] DiffPO: Diffusion Alignment with Direct Preference Optimization
description: >-
  [ACL 2025][LLM Alignment][diffusion] DiffPO is proposed to reformulate LLM alignment as a sentence-level diffusion denoising process. Through parallel decoding, it achieves efficient inference-time alignment and serves as a plug-and-play module that enhances the alignment quality of any base model.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "diffusion"
  - "inference-time alignment"
  - "parallel decoding"
  - "model-agnostic"
  - "preference optimization"
date: 2026-05-08
content_hash: dcfbb7aaa3540b36
---

# DiffPO: Diffusion Alignment with Direct Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2503.04240](https://arxiv.org/abs/2503.04240)  
**Code**: Yes (Mentioned in the paper)  
**Area**: LLM Alignment  
**Keywords**: diffusion, inference-time alignment, parallel decoding, model-agnostic, preference optimization

## TL;DR

DiffPO is proposed to reformulate LLM alignment as a sentence-level diffusion denoising process. Through parallel decoding, it achieves efficient inference-time alignment and serves as a plug-and-play module that enhances the alignment quality of any base model.

## Background & Motivation

**Background**: RLHF and DPO are the mainstream alignment methods, but they require separate training for each policy, which is computationally expensive. Inference-time alignment avoids retraining by directly adjusting the output distribution, but still relies on policy-specific value functions.

**Limitations of Prior Work**: Existing inference-time alignment methods (such as ARGS, BoN) have limited scalability (requiring policy-specific components) and high inference latency (token-by-token generation). Training-time alignment methods (such as DPO, SimPO) require separate training for each base model.

**Key Challenge**: Alignment is sentence-level (focusing on holistic features like style and format), whereas generation is token-level (next-token prediction). This granularity mismatch increases the learning difficulty.

**Goal**: To achieve an efficient, model-agnostic alignment method that can be applied to multiple base models with a single training run, while reducing inference latency.

**Key Insight**: Inspired by the global controllability of diffusion models, the alignment process is analogized to sentence-level denoising: progressively correcting an unaligned sentence $y(0)$ to an aligned sentence $y(T)$, performing sentence-level prediction instead of token-by-token generation at each step.

**Core Idea**: LLM alignment = sentence-level diffusion denoising process, using a DiffPO model for plug-and-play alignment enhancement.

## Method

### Overall Architecture

Training phase: Collect alignment trajectories (multiple responses with varying degrees of alignment for the same prompt), and train the DiffPO model to map inputs of any alignment level to highly aligned outputs. Inference phase: The base model generates an initial response, which DiffPO receives and corrects at the sentence level to output the final aligned response.

### Key Designs

1. **Alignment Trajectory Construction**: For each prompt in the UltraFeedback dataset, different base models generate $T=6$ responses, which are ranked by the ArmoRM reward model. The highest-scoring response is designated as the alignment target $y(T)$, and the remaining ones as intermediate states of varying alignment levels $y(0) \sim y(T-1)$.
2. **Consistency Optimization**: The core training objective is to enable DiffPO to directly predict $y(T)$ from any intermediate state $y(t)$. This is achieved using Consistency Loss (KL divergence to make the output distribution for input $y(t)$ close to the distribution for input $y(T)$, where the latter uses stop-gradient) + AR Loss (standard autoregressive loss to maintain generation quality).
3. **Model Agnosticism**: DiffPO optimizes sentence-level correction capabilities and does not depend on specific base model parameters. Training data comprises responses from multiple models, allowing DiffPO to learn universal alignment correction patterns. During inference, access to the base model parameters is not required, making it compatible with API-based models.

### Loss & Training

Total loss $L(\theta) = L_{\text{AR}} + \omega \cdot L_{\text{Con}}$, where $\omega = 10^3$. $L_{\text{Con}}$ uses forward KL divergence, and $L_{\text{AR}}$ uses standard cross-entropy. The backbone uses Gemma-2-it-2B/9B or Llama-3-8B-Instruct, with a maximum generation length of $N = 256$.

## Key Experimental Results

### Main Results

Llama-3-8B-Instruct as the base model:

| Method | MT-bench | AlpacaEval2 LC(%) | AlpacaEval2 WR(%) | HH-RLHF Helpful |
|------|----------|-------------------|-------------------|-----------------|
| Base | 6.78 | 36.83 | 42.12 | 0.67 |
| w. DPO | 6.90 | 47.20 | 53.56 | 0.74 |
| w. SimPO | 7.05 | 52.57 | 58.33 | 0.75 |
| w. DiffPO-9B | **7.40** | **55.84** | **61.88** | 0.72 |

DiffPO-9B cross-model enhancement:

| Base Model | AlpacaEval2 LC(%) Original→+DiffPO |
|---------|-------------------------------|
| Llama-3-70B-Instruct | 46.14 $\rightarrow$ **58.18** (+12.04) |
| Qwen2.5-7B-Instruct | 45.03 $\rightarrow$ **57.89** (+12.86) |
| Llama-3.2-1B-Instruct | 15.57 $\rightarrow$ **50.70** (+35.13) |

### Ablation Study

Effects of different DiffPO scales:

| DiffPO Scale | Llama-3-8B AlpacaEval2 LC(%) |
|------------|------------------------------|
| DiffPO-8B | 36.44 |
| DiffPO-9B | 55.84 |

The 9B model significantly outperforms the 8B counterpart, indicating that the capability of DiffPO itself is crucial for correction quality. When combined with pre-aligned base models like DPO/SimPO, DiffPO can bring further improvements.

### Key Findings

- DiffPO-9B can boost Llama-3.2-1B's AlpacaEval2 LC by 35 percentage points (15.57 $\rightarrow$ 50.70), demonstrating strong weak-to-strong enhancement capability.
- Train-once-deploy-many: The same DiffPO model can be applied across different model families, such as Llama, Mistral, and Qwen.
- Harmlessness scores improve significantly (e.g., Llama-3-SFT from 0.91 to 0.98), indicating that DiffPO also has positive effects on safety.
- Inference latency is superior to other inference-time methods (ARGS, BoN), achieving the optimal point in the alignment-latency trade-off.

## Highlights & Insights

- Elegantly transfers the intuition of diffusion processes to LLM alignment: sentence-level denoising vs. token-level reward optimization, offering a fresh perspective.
- Offers true "train-once, deploy-to-many" capability, which tremendously reduces the total cost of multi-model alignment.
- Consistency training enables DiffPO to complete alignment correction in a single step or a few steps, avoiding multi-step iterations.
- Can be superposed on top of existing DPO/SimPO for further improvement, showing complementarity with training-time alignment methods.

## Limitations & Future Work

- The maximum generation length is limited to 256 tokens, offering insufficient coverage for long-text responses.
- DiffPO itself requires sufficiently strong language capability (2B is far weaker than 9B); smaller DiffPO models show limited efficacy.
- The alignment-trajectory construction depends on an external reward model (ArmoRM), which introduces reward model bias.
- Sentence-level corrections may alter the factual content of the original responses; ensuring faithfulness requires further research.

## Related Work & Insights

- Compared with post-hoc alignment methods like Aligner/MetaAligner, the parallel decoding strategy of DiffPO is more efficient.
- Represents an innovative application of consistency model (LCM/CLCM) concepts in LLM alignment.
- Highly complementary to RLHF/DPO pipelines: RLHF/DPO performs basic alignment, while DiffPO handles inference-time enhancement.
- Model agnosticism makes it particularly suited for the alignment enhancement of API-only models (e.g., GPT-4o).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The alignment method from a diffusion perspective is a brand-new paradigm, with an ingenious application of parallel decoding
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated across 8 base models, 3 benchmarks, and multiple dimensions of ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework diagrams and complete theoretical derivations
- **Value**: ⭐⭐⭐⭐⭐ — Model-agnostic, plug-and-play alignment enhancement with high practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ICML 2026\] Boosting Direct Preference Optimization with Penalization](../../ICML2026/llm_alignment/boosting_direct_preference_optimization_with_penalization.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)

</div>

<!-- RELATED:END -->
