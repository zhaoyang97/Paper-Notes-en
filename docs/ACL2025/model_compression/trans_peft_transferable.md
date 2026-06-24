---
title: >-
  [Paper Note] Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models
description: >-
  [ACL 2025][Model Compression][PEFT] Trans-PEFT discovers that base model updates (e.g., Qwen2→Qwen2.5) primarily alter task knowledge stored in FFN layers while minimally affecting task patterns in Attention layers. Based on this insight, it proposes two strategies—intra-layer knowledge masking and cross-layer knowledge dropping—enabling PEFT modules trained on older versions to be directly transferred to newer versions without re-fine-tuning…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "PEFT"
  - "LoRA Transferability"
  - "Base Model Evolving"
  - "FFN Knowledge Masking"
  - "Adapter"
date: 2026-05-08
content_hash: e94a7e8beb21d636
---

# Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models

**Conference**: ACL 2025  
**arXiv**: [2506.06844](https://arxiv.org/abs/2506.06844)  
**Code**: [https://github.com/gccnlp/Trans-PEFT](https://github.com/gccnlp/Trans-PEFT)  
**Area**: Model Compression  
**Keywords**: PEFT, LoRA Transferability, Base Model Evolving, FFN Knowledge Masking, Adapter  

## TL;DR
Trans-PEFT discovers that base model updates (e.g., Qwen2→Qwen2.5) primarily alter task knowledge stored in FFN layers while minimally affecting task patterns in Attention layers. Based on this insight, it proposes two strategies—intra-layer knowledge masking and cross-layer knowledge dropping—enabling PEFT modules trained on older versions to be directly transferred to newer versions without re-fine-tuning, yielding performance gains of up to 30%.

## Background & Motivation

**Background**: PEFT (e.g., LoRA, Adapter) has become the mainstream method for fine-tuning LLMs, allowing a single base model to serve multiple users by dynamically switching PEFT modules. Base models undergo periodic updates (e.g., Qwen2→Qwen2.5, InternLM2→InternLM2.5) to refresh knowledge and enhance capabilities.

**Limitations of Prior Work**: Following base model updates, PEFT modules fine-tuned on older versions suffer severe performance degradation (direct transfer fails). For large-scale deployments featuring numerous PEFT modules, re-fine-tuning incurs massive computational overhead and raises concerns regarding long-term storage and privacy of user data.

**Key Challenge**: PEFT modules form tight coupling with the FFN layers of the base model—PEFT learns to leverage specific knowledge storage patterns, whereas model updates inevitably alter these patterns.

**Goal**: Enable direct transfer of PEFT modules fine-tuned on older versions to newer versions without re-fine-tuning.

**Key Insight**: An analysis of inner activation distribution shifts before and after model updates reveals that task patterns in Attention layers remain stable across versions, while knowledge storage in FFN layers changes significantly. Consequently, PEFT should reduce its dependence on FFN knowledge and focus on capturing version-invariant task patterns in Attention layers.

**Core Idea**: Randomly mask FFN layer outputs during training to force PEFT to capture cross-version invariant task patterns in Attention rather than version-specific FFN knowledge.

## Method

### Overall Architecture

During PEFT fine-tuning on the older base model version, two random interventions are applied to the FFN layers: (1) intra-layer knowledge masking dynamically zeros out intermediate FFN dimensions; (2) cross-layer knowledge dropping randomly discards the output of entire FFN layers. During inference on the newer version, the PEFT module is directly applied without any extra operations.

### Key Designs

1. **Intra-layer Knowledge Masking**:

    - **Function**: Randomly masks intermediate dimensions after the FFN activation function.
    - **Mechanism**: Introduces a Bernoulli mask $m \sim \text{Bernoulli}(1-p_i)$ for each FFN layer, multiplying it element-wise with the activation value: $\text{FFN}(\mathbf{X}) = \sigma(\mathbf{X}(\mathbf{W}_{fc1}+\Delta\mathbf{W}_{fc1})) \odot m \cdot (\mathbf{W}_{fc2}+\Delta\mathbf{W}_{fc2})$
    - **Design Motivation**: Since the knowledge storage distribution inside the FFN shifts after an update (observed experimentally), the randomness introduced by masking prevents PEFT from relying on specific dimensions of knowledge.

2. **Cross-layer Knowledge Dropping**:

    - **Function**: Randomly drops the outputs of entire FFN layers with a probability $p_c$.
    - **Mechanism**: $\widetilde{\text{FFN}}(\mathbf{X}) = z \cdot \text{FFN}(\mathbf{X})$, where $z \sim \text{Bernoulli}(1-p_c)$
    - **Design Motivation**: Experiments reveal that the impact of different FFN layers on activation magnitudes also shifts (or even exhibits opposite trends) after updates, necessitating a cross-layer perspective to reduce dependency.

3. **Theoretical Guarantees**:

    - Provides an upper bound on loss discrepancy: $|\mathcal{L}(\theta; \mathcal{M}_1) - \mathcal{L}(\theta; \mathcal{M}_0)|$ is bounded by the FFN perturbation magnitude $\rho$ and the gradient norm of the PEFT with respect to FFN.
    - Trans-PEFT reduces the upper bound on transfer loss by minimizing $\|\nabla_{\theta_{ffn}} \mathcal{L}\|$.

### Loss & Training
Standard task loss (e.g., cross-entropy) is used. Masking and dropping are only applied during training and omitted during inference. The masking rate $p_i$ and dropping rate $p_c$ are hyperparameters. The method is compatible with both LoRA and Adapter, requiring no changes to model architectures.

## Key Experimental Results

### Main Results

7 base models (Qwen2/2.5-7B, InternLM2/2.5-7B, Llama3/3.1/3.2, etc.), 12 datasets (mathematical reasoning, code generation, commonsense reasoning).

| Transfer Setting | Direct Transfer Acc | Trans-PEFT Acc | Gain |
|---------|-------------|---------------|------|
| Qwen2→Qwen2.5 (GSM8K) | 54.2 | **71.3** | +17.1 |
| InternLM2→InternLM2.5 (GSM8K) | 48.7 | **63.5** | +14.8 |
| Qwen2→Qwen2.5 (HumanEval) | 42.1 | **55.8** | +13.7 |
| Average Gain | - | - | **~30% (Max)** |

Comparison with direct fine-tuning on the newer version:

| Method | GSM8K Acc | Description |
|------|----------|------|
| Re-fine-tuned on Qwen2.5 | 73.8 | Upper bound (requires retraining) |
| Trans-PEFT Transfer | **71.3** | No retraining required |
| Direct Transfer | 54.2 | Significant performance degradation |

### Ablation Study

| Configuration | GSM8K Acc | Description |
|------|----------|------|
| Trans-PEFT (Full) | **71.3** | Intra-layer Masking + Cross-layer Dropping |
| Intra-layer Masking Only | 67.5 | Without Cross-layer Dropping |
| Cross-layer Dropping Only | 65.8 | Without Intra-layer Masking |
| Direct Transfer | 54.2 | No strategy applied |

### Key Findings
- **The two strategies are complementary**: Using either strategy alone yields major improvements, while combining them yields the best performance.
- **Trans-PEFT not only maintains performance but also leverages improvements in newer versions**: In some settings, post-transfer performance even exceeds the original performance on the older version.
- **The method is effective for both LoRA and Adapter**: It does not depend on a specific PEFT approach.
- **Sensitivity to masking and dropping rates is moderate**: $p_i$ in the range of 0.1-0.3 and $p_c$ in the range of 0.05-0.15 yield optimal results.

## Highlights & Insights
- **The "Attention remains invariant while FFN changes" observation is highly insightful**: This observation simplifies the intractable "model update" problem into an "FFN variation" problem, paving the way for the solution.
- **Dropout-like idea but with a precise target**: While conceptually similar to Dropout, the deeper logic is to reduce the upper bound on transfer loss by minimizing the gradient of PEFT with respect to FFN knowledge, backed by theoretical support.
- **Extremely high transfer value**: Given the frequent iterations of LLMs (e.g., Llama3→3.1→3.2→3.3), avoiding re-fine-tuning all PEFT modules upon each release offers immense engineering value.

## Limitations & Future Work
- Evaluates only architecture-preserving version updates (e.g., iterations within the same family); cross-architecture transfer is not addressed.
- Theoretical analysis relies on the assumption that Attention weights remain approximately invariant, which might not hold under substantial base model updates.
- Adaptive masking rates (dynamically adjusted based on the degree of variation in each layer) are left unexplored.
- Validated only on 7B-scale models; performance on larger models (e.g., 70B+) remains to be verified.

## Related Work & Insights
- **vs Qin et al. (2023)**: They demonstrated that direct PEFT transfer fails but proposed no solution. Trans-PEFT is the first systematic solution.
- **vs Dropout**: Trans-PEFT resembles Dropout but targets a specific goal—reducing dependency on FFN knowledge. While standard Dropout is applied to all modules, Trans-PEFT specifically targets FFN layers.
- **vs LoRAHub/Model Merging**: These approaches focus on combining different PEFT modules. Trans-PEFT addresses cross-version transfer, targeting a different problem.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation that "FFN changes while Attention remains invariant" is novel, and the solution design is highly reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with 7 models × 12 datasets × 2 PEFT methods.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis with up to cohesive logical chain from observation to method, theory, and experiments.
- Value: ⭐⭐⭐⭐⭐ Solves a practical and urgent engineering challenge, contributing significantly to the LLM ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

</div>

<!-- RELATED:END -->
