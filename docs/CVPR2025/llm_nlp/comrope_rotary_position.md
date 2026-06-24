---
title: >-
  [Paper Note] ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices
description: >-
  [CVPR 2025][LLM (Other)][Position Encoding] This paper proposes ComRoPE, which generalizes RoPE into a rotary position embedding parameterized by trainable commuting angle matrices. It theoretically proves that the pairwise commutativity of angle matrices is a necessary and sufficient condition for RoPE to satisfy relative position dependency, outperforming the state-of-the-art LieRE method by 1.6% (on training resolution) and 2.9% (on higher resolutions) on ImageNet-1K.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Position Encoding"
  - "Rotary Position Embedding"
  - "Transformer"
  - "Trainable Matrix"
  - "Scalability"
date: 2026-05-08
content_hash: db135d578b6da669
---

# ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices

**Conference**: CVPR 2025  
**arXiv**: [2506.03737](https://arxiv.org/abs/2506.03737)  
**Code**: [https://github.com/Longin-Yu/ComRoPE](https://github.com/Longin-Yu/ComRoPE)  
**Area**: LLM/NLP  
**Keywords**: Position Encoding, Rotary Position Embedding, Transformer, Trainable Matrix, Scalability

## TL;DR
This paper proposes ComRoPE, which generalizes RoPE into a rotary position embedding parameterized by trainable commuting angle matrices. It theoretically proves that the pairwise commutativity of angle matrices is a necessary and sufficient condition for RoPE to satisfy relative position dependency, outperforming the state-of-the-art LieRE method by 1.6% (on training resolution) and 2.9% (on higher resolutions) on ImageNet-1K.

## Background & Motivation
1. **Background**: RoPE encodes position information by applying rotary transformations to embedding vectors in the attention mechanism, which has been widely adopted in LLMs (such as LLaMA) and ViT.
2. **Limitations of Prior Work**: (1) Existing RoPE uses manually defined 2D rotation matrices, which limits the flexibility and adaptability of models in high-dimensional spaces; (2) most rotation matrices require manual design and yield suboptimal performance due to limited capacity; (3) prior attempts to scale to higher-dimensional rotation groups struggle to consistently preserve relative position dependency.
3. **Key Challenge**: RoPE must satisfy $R(x)^T R(y) = R(y-x)$ (depending solely on the relative position), but this property is difficult to maintain when expanding the rotation group from 2D to higher dimensions.
4. **Goal**: Expand the rotation group of RoPE from 2D to a larger subgroup of special orthogonal groups while maintaining consistent position shift behavior and enabling trainable rotation matrices.
5. **Key Insight**: Formalize the RoPE equations and identify the necessary and sufficient conditions for the angle matrices to satisfy the RoPE equation.
6. **Core Idea**: Pairwise commutativity of angle matrices = necessary and sufficient condition of the RoPE equation $\rightarrow$ two parameterized schemes for trainable commuting angle matrices.

## Method

### Overall Architecture
Features are divided into multiple blocks $\rightarrow$ each block defines a set of angle matrices $\mathcal{A} = \{A_1, ..., A_N\}$ (skew-symmetric matrices) $\rightarrow$ the rotation matrix is defined as $R(\mathbf{x}; \mathcal{A}) = \exp(\sum_i A_i x_i)$ $\rightarrow$ rotation is applied to queries and keys $\rightarrow$ attention computation depends only on relative positions.

### Key Designs

1. **RoPE Equation and Commutativity Theorem**:

    - Function: Provides theoretical guarantees for the correct operation of RoPE—the rotation matrices must satisfy relative position dependency.
    - Mechanism: Formalizes the definitions of the RPE equation (Def 3) and the RoPE equation (Def 4), and proves Theorem 1: the RoPE function parameterized by the angle matrix set $\mathcal{A}$ satisfies the RoPE equation if and only if the matrices in $\mathcal{A}$ are pairwise commutative, i.e., $A_i A_j = A_j A_i$ for all $i, j$.
    - Design Motivation: Previous RoPE extensions (such as LieRE) were only heuristically designed and did not prove whether they satisfy relative position dependency. The theoretical foundation of ComRoPE ensures the correctness of the position encoding.

2. **Two Trainable Commuting Angle Matrix Schemes**:

    - Function: Provides parameterization schemes for trainable matrices that satisfy commutativity constraints.
    - Mechanism: (1) Diagonal scheme: Angle matrices are block-diagonal, where rotations occur within each 2×2 block (generalizing standard RoPE), with trainable parameters controlling rotation frequencies per block; (2) Simultaneously diagonalizable scheme: All angle matrices share the same orthogonal transformation $P$, i.e., $A_i = P D_i P^T$, where $D_i$ are skew-symmetric diagonal matrices. Both schemes satisfy the commutativity condition.
    - Design Motivation: Scheme 1 is simple and efficient, while Scheme 2 offers higher representational degrees of freedom ($O(d^2)$ vs. $O(d)$ parameters).

3. **Unified Theoretical Framework**:

    - Function: Unifies multiple existing RoPE variants (standard RoPE, 2D-RoPE, RoPE-Mixed, LieRE) under a single theoretical framework.
    - Mechanism: Proves that existing methods are special cases of ComRoPE—standard RoPE uses fixed diagonal angle matrices, whereas LieRE utilizes Lie groups but does not guarantee commutativity.
    - Design Motivation: Provides a theoretical perspective to understand the strengths and weaknesses of different RoPE variants.

### Loss & Training
The angle matrix parameters are trained end-to-end alongside the remaining model parameters. Standard classification training is performed under the DeiT-III framework. Angle matrices are initialized to the diagonal form of standard RoPE. Experiments are conducted on the ImageNet-1K classification task with ViT-S models (22M parameters) under training and testing resolutions of 224² and 384², respectively.

## Key Experimental Results

### Main Results

| Method | ImageNet Top-1 (224²) | ImageNet Top-1 (384²) | Parameters |
|------|---------------------|---------------------|--------|
| DeiT-III (APE) | 81.8 | 82.4 | 22M |
| RoPE-Axial | 82.0 | 82.8 | 22M |
| LieRE | 82.4 | 83.1 | 22M |
| **ComRoPE-Type1** | **83.5** | **85.2** | 22M |
| **ComRoPE-Type2** | **84.0** | **86.0** | 22M |

### Ablation Study

| Configuration | Top-1 (224²) | Description |
|------|-------------|------|
| ComRoPE-Type2 | 84.0 | Fully trainable |
| Fixed angle matrices | 82.1 | Trainability contribution +1.9% |
| Non-commuting matrices | 81.5 | Commutativity constraint is important |
| Standard RoPE | 82.0 | ComRoPE improvement +2.0% |

### Key Findings
- ComRoPE significantly outperforms LieRE on both training and higher resolutions, demonstrating the advantages of richer position representations.
- Commutativity constraints are crucial for performance—removing the constraints leads to performance degradation (the position encoding no longer depends on relative positions).
- Type 2 (simultaneously diagonalizable) is stronger than Type 1 (block-diagonal) due to more degrees of freedom.
- The advantages are even more pronounced under resolution transfer scenarios (+2.9% vs. +1.6%), indicating that the trainable angle matrices learn more robust position representations.
- Position perturbation robustness experiments: APE is highly sensitive to position perturbations (+19.5% when perturbation intensity scales 0 $\rightarrow$ 1), while ComRoPE-LD only increases by 2.9%, showing that the design of RoPE itself has better position robustness.
- ComRoPE can be seamlessly integrated into the fine-tuning phase—even if ComRoPE is not used during pre-training, standard Attention can be replaced and pre-trained weights can be loaded during fine-tuning (all-zero angle matrices are equivalent to standard Attention).

## Highlights & Insights
- **Theoretical elegance**: Unifies the RoPE theory with the simple algebraic concept of commutativity; the proof of the necessary and sufficient conditions provides explicit guidelines for subsequent designs.
- **Balance of trainability and constraint**: Instead of leaving the matrices completely unconstrained (which would violate RoPE properties), the expressive capacity is maximized under the commutativity constraint.
- **Guidance value for future RoPE research**: Any new RoPE design should satisfy the commutativity condition, which serves as a verifiable theoretical standard.
- **Unification**: Standard RoPE, 2D-RoPE, RoPE-Mixed, and LieRE are all proven to be special cases of ComRoPE. When angle matrices are all-zero, RoPE Attention degenerates to standard Attention; when the block size is 2, ComRoPE-AP degenerates to the RoPE Attention commonly used in LLMs.

## Limitations & Future Work
- Currently only verified on ViT image classification, not yet evaluated in LLM long-sequence scenarios.
- The number of parameters in the Type 2 scheme increases by $O(d^2)$, which may introduce overhead for extremely large models.
- Matrix exponentiation introduces additional computation, requiring efficient implementations (such as Padé approximation or eigenvalue decomposition).
- Future work can explore applications in more spatial dimensions such as 3D data and video.
- The initialization strategy of angle matrices affects convergence speed; currently initialized to standard RoPE diagonal form, superior initialization schemes may exist.
- The commutativity condition might only be approximately satisfied after discretization (numerical computation), and its impact on accuracy remains unanalyzed.
- Features are processed in multiple divided blocks, where each block defines an independent set of angle matrices $\mathcal{A}$, and the independence among blocks guarantees computational parallelizability.
- Rigorously formalized the RPE equation (Def 3) and the RoPE equation (Def 4), with the proof of the necessary and sufficient conditions in Theorem 1 serving as a theoretical anchor for all future designs.

## Related Work & Insights
- **vs LieRE**: LieRE is built on Lie groups but does not prove whether it satisfies the RoPE equation; ComRoPE provides theoretical guarantees. ComRoPE-Type2 reaches 86.0% under 384² resolution, outperforming LieRE's 83.1% by +2.9%.
- **vs Standard RoPE**: Standard RoPE uses fixed 2D rotations, whereas ComRoPE employs trainable high-dimensional rotations, offering stronger capacity.
- **vs iRPE/RPB**: These methods utilize relative position bias instead of rotation, while ComRoPE preserves the rotational benefits of RoPE.
- **vs APE (Absolute Position Encoding)**: APE is extremely sensitive to position perturbations (+19.5%), while ComRoPE is naturally robust (only +2.9%).

## Rating

### Implementation Details
Using the DeiT-III framework, ViT-S model (22M parameters), ImageNet-1K classification.
Training resolution 224², testing resolutions 224² and 384². Angle matrices are initialized to the diagonal form of standard RoPE.
- Novelty: ⭐⭐⭐⭐⭐ The theoretical contribution of the necessary and sufficient condition of commutativity is highly significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ ImageNet classification + resolution transfer, but lacks NLP verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clear and systematic definitions.
- Value: ⭐⭐⭐⭐⭐ Foundational contribution to position encoding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](../../ACL2025/llm_nlp/probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)
- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](../../ACL2025/llm_nlp/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [\[ACL 2025\] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](../../ACL2025/llm_nlp/mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)
- [\[ACL 2025\] Computation Mechanism Behind LLM Position Generalization](../../ACL2025/llm_nlp/computation_mechanism_behind_llm_position_generalization.md)
- [\[ACL 2025\] Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](../../ACL2025/llm_nlp/a3cg_esg_greenwashing.md)

</div>

<!-- RELATED:END -->
