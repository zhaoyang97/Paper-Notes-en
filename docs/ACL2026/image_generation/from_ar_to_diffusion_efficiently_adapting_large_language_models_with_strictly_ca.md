---
title: >-
  [Paper Note] From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons
description: >-
  [ACL2026][Image Generation][Diffusion Language Models] This paper proposes FLUID, which effectively adapts pre-trained autoregressive (AR) LLMs into diffusion-based parallel generation models using strictly causal attent…
tags:
  - "ACL2026"
  - "Image Generation"
  - "Diffusion Language Models"
  - "Autoregressive Adaptation"
  - "Causal Attention"
  - "Dynamic Stride"
  - "Parallel Generation"
date: 2026-05-08
content_hash: f1ff1d1b81d9dc20
---

# From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons

**Conference**: ACL2026  
**arXiv**: [2605.27387](https://arxiv.org/abs/2605.27387)  
**Code**: https://huggingface.co/MYTH-Lab/FLUID  
**Area**: LLM Generation / Diffusion Language Models / Inference Acceleration  
**Keywords**: Diffusion Language Models, Autoregressive Adaptation, Causal Attention, Dynamic Stride, Parallel Generation  

## TL;DR
This paper proposes FLUID, which effectively adapts pre-trained autoregressive (AR) LLMs into diffusion-based parallel generation models using strictly causal attention and entropy-aware Elastic Horizons. With only 2.7B adaptation tokens, it achieves performance in reasoning and code generation close to strong AR models and superior to existing diffusion baselines.

## Background & Motivation
**Background**: Mainstream LLMs are based on autoregressive (AR) next-token prediction, generating one token at a time. While logically stable and mature in training, their latency during long-sequence reasoning and deployment grows linearly with length. Discrete diffusion language models attempt to break this serial bottleneck by iteratively denoising and generating multiple tokens in parallel.

**Limitations of Prior Work**: Standard diffusion language models mostly employ bidirectional attention to utilize global noisy contexts, which contradicts the causal priors of pre-trained AR backbones like GPT or LLaMA. Consequently, developing high-quality diffusion LMs often requires expensive large-scale pre-training from scratch. Existing "block diffusion" methods use fixed-size blocks as a compromise, but these cannot adapt to variations in local entropy within natural language.

**Key Challenge**: The strength of AR models stems from strictly causal dependencies, whereas the advantage of diffusion models lies in parallel denoising. Direct application of bidirectional diffusion breaks the AR prior. Conversely, using fixed blocks leads to over-aggression in high-entropy reasoning segments and over-conservatism in low-entropy template segments.

**Goal**: The authors aim to adapt existing AR checkpoints into parallel-generative diffusion models at a low cost while maintaining causal consistency, reasoning capabilities, and structural integrity for code generation.

**Key Insight**: The paper decomposes the problem into two types of mismatch: a structural mismatch between bidirectional diffusion and AR causal priors, and a dynamic mismatch between fixed generation windows and local information density. FLUID addresses these via Strictly Causal Alignment and Elastic Horizons, respectively.

**Core Idea**: Restrict diffusion denoising under strictly causal attention and enable the model to predict a "safe" parallel denoising horizon based on hidden states, rather than using a fixed block size.

## Method
FLUID stands for Flexible Unidirectional Inference Diffusion. Rather than training a diffusion LM from scratch, it performs parameter-efficient adaptation on AR backbones such as openPangu-Embedded-7B. Overall, it retains the AR structure (looking only at history) while introducing a variable-length mask span after the current position for parallel recovery via diffusion.

### Overall Architecture
Given a prefix, FLUID uses a causal Transformer to obtain the current hidden state, which is then used by a K-Head to predict a horizon $K_t$. The model places masks at the next $K_t$ positions and performs parallel denoising under strictly causal constraints. After prediction, the immediate next token is always accepted, while subsequent tokens are accepted only if their confidence exceeds a threshold $\gamma$. If a low-confidence token is encountered, the cursor stops and replans. Thus, FLUID takes multiple steps in low-entropy segments and automatically reverts to fine-grained generation in high-entropy reasoning segments.

### Key Designs
1. **Strictly Causal Diffusion Backbone**:
    - **Function**: Aligns the diffusion denoising process with the causal priors of pre-trained AR models.
    - **Mechanism**: Injects a lower triangular attention mask into the Transformer, where position $i$ can only access tokens $j \le i$, and future positions are set to $-\infty$. Recovering token $x_i$ depends only on the noisy history $\mathbf{x}_{t,<i}$, preventing any "peeking" at future noisy tokens.
    - **Design Motivation**: Standard bidirectional diffusion injects noisy future context into current predictions during inference, which can break deductive chains. Strictly causal constraints allow for smooth initialization from GPT-style checkpoints.

2. **Elastic Horizon Modeling**:
    - **Function**: Dynamically determines the parallel generation distance based on local uncertainty.
    - **Mechanism**: A lightweight K-Head is added to map the final hidden state $h_t$ to a categorical distribution $k \in \{1, \dots, K_{max}\}$. The oracle horizon $K_t^*$ is determined by the future loss sequence and a competence boundary $\tau$: the model is considered capable of safe parallel recovery within the maximum span where the average loss is below $\tau$.
    - **Design Motivation**: Natural language information density is non-uniform. The model can move aggressively through function boilerplate or simple arithmetic but should shorten its stride during complex mathematical reasoning. Fixed blocks cannot capture this variance.

3. **Two-stage Adaptation and Dynamic Causal Decoding**:
    - **Function**: Enables the backbone to learn causal diffusion denoising first, followed by the K-Head learning to reliably plan the horizon.
    - **Mechanism**: Stage I freezes the K-Head and trains the backbone using a joint AR and diffusion denoising loss, with 10% stochastic restoration noise injected into the mask span. Stage II freezes the backbone and trains the two-layer MLP K-Head to match the predicted distribution to a Gaussian soft target centered at $K_t^*$. During inference, a confidence gate decides the actual number of tokens accepted.
    - **Design Motivation**: Training the backbone and horizon planner simultaneously can be unstable. Decoupling "what to generate" from "how far to generate" by calibrating generation capability before learning its boundaries is more effective.

### Loss & Training
The goal of Stage I is to mix AR and diffusion: $\mathcal{L}_{Stage1} = \mathcal{L}_{AR} + \mathcal{L}_{Diff}$. $\mathcal{L}_{AR}$ maintains the next-token language prior of the prefix, while $\mathcal{L}_{Diff}$ recovers the mask span under strictly causal constraints. During training, $K \sim U[1, K_{max}]$ is sampled, and 10% noise is injected into the mask span to improve robustness against imperfect intermediate states.

The goal of Stage II is horizon distribution matching: $\mathcal{L}_{Stage2} = D_{KL}(\mathcal{Q} \| P_\phi(\cdot|h_t))$. $\mathcal{Q}$ is a Gaussian soft target centered at $K_t^*$ derived from the competence boundary. In experiments, $K_{max}=16$ and $\tau=2.8$. Stage I runs for 32,000 iterations, and Stage II for 2,000 steps. The backbone uses Rank-16 LoRA with an input length of 1024, a global batch size of 80, and a learning rate of $2 \times 10^{-4}$.

## Key Experimental Results

### Main Results
| Model / Method | Type | Adaptation/Training Tokens | MMLU | IFEval | GSM8K | MATH500 | HumanEval | MBPP |
|----------------|------|---------------------------|------|--------|-------|---------|-----------|------|
| FLUID-7B | Diff | 2.7B | 67.8 | 57.7 | 91.9 | 61.8 | 60.4 | 53.6 |
| Qwen-2.5-7B | AR | - | - | - | 91.6 | - | - | - |
| Dream-7B | Diff | - | - | - | 81.0 | - | - | - |
| LLaDA-8B | Diff | - | - | - | 78.6 | - | - | - |

### Ablation Study
| Configuration | Causal | Elastic | GSM8K | MATH500 | HumanEval | Note |
|---------------|--------|---------|-------|---------|-----------|------|
| Baseline | ✗ | ✗ | 82.0 | 51.2 | 42.2 | Bidirectional fixed-block baseline |
| Baseline + Elastic | ✗ | ✓ | 82.5 | 53.6 | 42.8 | Dynamic horizon only; still suffers from acausal future interference |
| Baseline + Causal | ✓ | ✗ | 90.6 | 59.2 | 54.9 | Causal constraint restores reasoning chain, but fixed blocks truncate structure |
| FLUID | ✓ | ✓ | 91.9 | 61.8 | 60.4 | Strictly causal and dynamic horizon are complementary |

### Key Findings
- FLUID reaches 91.9 on GSM8K, which is 10.9 points higher than Dream-7B, 13.3 points higher than LLaDA-8B, and comparable to Qwen-2.5-7B (91.6).
- Elastic Horizon is crucial for code tasks. A fixed $K=16$ causes semantic fracture; full FLUID is 5.5 points higher than the causal-only fixed-block version on HumanEval.
- A 10% stochastic restoration noise ratio is optimal: GSM8K / MATH500 / HumanEval scores are 91.9 / 61.8 / 60.4 respectively, compared to 91.0 / 60.8 / 59.8 at 0% and 91.1 / 61.5 / 60.0 at 15%.
- The competence boundary $\tau=2.8$ performs best. Smaller values are too conservative, while larger values lead to over-expansion of the horizon.
- Regarding inference efficiency, FLUID achieves approximately $2\times$ speedup over standard diffusion baselines. On MMLU, the average stride is 6.5, reaching 18.82 tokens/s, higher than the 17.52 tokens/s of fixed $K=16$. On GSM8K, the average stride can expand to 13.1.
- Training cost is approximately 320 GPU-hours, with Stage I accounting for the vast majority. Stage II trains only the lightweight K-Head, with negligible overhead.

## Highlights & Insights
- The most ingenious aspect of the paper is not treating "bidirectional" as a prerequisite for diffusion, but rather redefining causal diffusion: parallel recovery of a future span while each position remains restricted to historical context.
- Elastic Horizon transforms parallel decoding from a fixed engineering hyperparameter into a model-predicted problem. This is more natural than tuning a block size and explains why a single model should use different strides for GSM8K versus MMLU.
- Confidence gating ensures that horizon prediction is not a "hard" commitment. The K-Head proposes a plan, but tokens are vetted by a confidence check, reducing the risk of a single miscalculation contaminating the sequence.
- From an engineering perspective, 2.7B adaptation tokens and 320 GPU-hours are very low compared to training a diffusion LM from scratch, demonstrating that AR-to-diffusion adaptation is a viable practical path.

## Limitations & Future Work
- The performance ceiling of FLUID is constrained by the source AR backbone. If the source model (e.g., openPangu) has inherent hallucinations or reasoning flaws, strictly causal diffusion will not automatically fix those gaps.
- Current experiments focus on general LLM tasks, math, code, and instructions. Specialized domains like medicine, law, and specific structured generation have not yet been fully validated.
- It remains to be seen if MoE architectures, multimodal LLMs, or longer context scenarios are equally compatible with this adaptation method.
- Elastic Horizon relies on the calibration of the K-Head and competence boundary. Future work could explore fine-grained uncertainty estimation, online adaptive thresholds, and integration with KV caching or speculative decoding.

## Related Work & Insights
- **vs. LLaDA / Dream**: These diffusion language models show potential for parallel generation but typically rely on bidirectional attention and massive pre-training. FLUID focuses on reusing AR checkpoints and avoiding future noise interference via strictly causal mechanisms.
- **vs. Block Diffusion / semi-AR methods**: Fixed-block methods compromise between quality and speed, but the block size cannot adapt to local entropy. FLUID uses a K-Head to dynamically set the horizon and utilizes confidence gating for actual truncation.
- **vs. Fast-DLLM style inference acceleration**: Training-free acceleration is often an inference trick; FLUID changes the adaptation objective and structural constraints. While the cost is higher, it systematically improves reasoning, code, and semantic quality.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of strictly causal diffusion and Elastic Horizon addresses the core contradictions of AR-to-diffusion adaptation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers general, math, code, efficiency, and multiple ablations, though specialized domains and larger scale models are still to be validated.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear explanation of structural mismatch and the entropy-horizon dilemma. Some main tables in the source are slightly difficult to parse.
- **Value**: ⭐⭐⭐⭐⭐ Highly insightful for low-cost parallel LLM generation, particularly for converting AR checkpoints into lower-latency generators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](../../NeurIPS2025/image_generation/non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](../../ICML2026/image_generation/esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/image_generation/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](../../CVPR2026/image_generation/causal_motion_diffusion_models_for_autoregressive_motion_generation.md)

</div>

<!-- RELATED:END -->
