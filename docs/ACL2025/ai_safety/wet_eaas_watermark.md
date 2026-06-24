---
title: >-
  [Paper Note] WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark
description: >-
  [ACL 2025][AI Safety][EaaS Watermarking] This work reveals that existing EaaS embedding watermarking methods (EmbMarker/WARDEN) can be bypassed by paraphrasing attacks. It proposes WET (Watermark via Linear Transformation), which injects watermarks by applying linear transformations to embeddings using a secret circulant matrix. Theoretical analysis and empirical results demonstrate its robustness against paraphrasing attacks, achieving a verification AUC near 100%.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "EaaS Watermarking"
  - "Embedding Protection"
  - "Paraphrase Attacks"
  - "Linear Transformation"
  - "Intellectual Property Protection"
date: 2026-05-08
content_hash: de3f98fe977431de
---

# WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark

**Conference**: ACL 2025  
**arXiv**: [2409.04459](https://arxiv.org/abs/2409.04459)  
**Code**: [https://github.com/anudeex/WET.git](https://github.com/anudeex/WET.git)  
**Area**: AI Security  
**Keywords**: EaaS Watermarking, Embedding Protection, Paraphrase Attacks, Linear Transformation, Intellectual Property Protection  

## TL;DR
This work reveals that existing EaaS embedding watermarking methods (EmbMarker/WARDEN) can be bypassed by paraphrasing attacks. It proposes WET (Watermark via Linear Transformation), which injects watermarks by applying linear transformations to embeddings using a secret circulant matrix. Theoretical analysis and empirical results demonstrate its robustness against paraphrasing attacks, achieving a verification AUC near 100%.

## Background & Motivation
**Background**: Embeddings-as-a-Service (EaaS) is an embedding extraction API provided by LLM developers. Adversaries can clone the service through imitation attacks, where they query the API to get embeddings and train a surrogate model on them. Existing watermarking methods (EmbMarker, WARDEN) inject verifiable signals into embeddings via a trigger-word mechanism.

**Limitations of Prior Work**: These methods rely on trigger words to activate watermark injection—watermarks are added only when the input contains predefined trigger words. This implies that adversaries can change the presence of trigger words by paraphrasing the input text, thereby diluting the watermark.

**Key Challenge**: The trigger-word mechanism is a core design of EmbMarker/WARDEN but also their fatal weakness. After paraphrasing, trigger words are replaced by synonyms, reducing the watermark weight. When multiple paraphrased versions are averaged, the watermark signal is almost completely diluted.

**Goal**: (1) To verify the effectiveness of paraphrasing attacks, and (2) to design an embedding watermarking method robust against paraphrasing.

**Key Insight**: Instead of relying on trigger words, a linear transformation is applied to all output embeddings. Due to the commutativity of linear transformations, averaging after paraphrasing is equivalent to transforming the pseudo-aggregated embedding—thus, the watermark signal will not be diluted by averaging.

**Core Idea**: A secret circulant matrix $\mathbf{T}$ is used to perform a linear transformation on the original embedding to generate the watermarked embedding. During verification, the pseudoinverse $\mathbf{T}^+$ is used to reconstruct the original embedding and compare the similarity.

## Method

### Overall Architecture
Watermark injection: Upon receiving a query, EaaS first generates the original embedding $\mathbf{e}_o$ using the original model, and then applies a linear transformation $\mathbf{e}_p = \text{Norm}(\mathbf{T} \cdot \mathbf{e}_o)$ using a secret transformation matrix $\mathbf{T}$ to return to the user. Verification: For a suspicious service's embedding $\mathbf{e}'_p$, the pseudoinverse $\mathbf{T}^+$ is used to recover the original embedding $\mathbf{e}'_o = \mathbf{T}^+ \cdot \mathbf{e}'_p$, which is then evaluated against the original embedding via cosine similarity.

### Key Designs

1. **Paraphrase Attacks (New Attack Vector)**:

    - Function: To evaluate whether paraphrasing can bypass existing EaaS watermarks.
    - Mechanism: For each input text, $P$ paraphrased versions are generated (using GPT-3.5/DIPPER/round-trip translation). These are queried to the EaaS to retrieve watermarked embeddings, which are then averaged to serve as the training targets for the surrogate model. Since paraphrasing alters the presence of trigger words, the averaged watermark weight drops significantly.
    - Design Motivation: Existing watermarking methods only inject watermarks when trigger words are present, making paraphrasing the most natural bypass method. It is theoretically proven that as $P$ increases, the probability of obtaining samples with high watermark weights decreases exponentially.

2. **WET (Watermark via Linear Transformation)**:

    - Function: To design an embedding watermarking method robust to paraphrasing.
    - Mechanism: Key Theorem—Linear transformations are commutative: $\text{avg}(f(\{\mathbf{e}_p^i\})) = f(\text{avg}(\{\hat{\mathbf{e}}_o^i\}))$. Even if the attacker paraphrases and averages the embeddings, the watermark transformation $\mathbf{T}$ still exists in the aggregated embedding and will not be eliminated by the averaging operation. Additionally, watermark injection does not rely on trigger words and uniformly transforms all embeddings.
    - Design Motivation: The algebraic properties of linear transformations inherently resist averaging operations, which is impossible to achieve with the additive watermarks of EmbMarker/WARDEN.

3. **Circulant Matrix Construction**:

    - Function: To generate a transformation matrix that satisfies invertibility and condition number requirements.
    - Mechanism: The first row is randomly generated ($k$ non-zero positions, values sampled from $U(0,1)$ and normalized), and subsequent rows are sequentially shifted. When the FFT values of the circulant matrix are non-zero, full rank is guaranteed, allowing accurate computation of the pseudoinverse. Hyperparameters $w$ (watermark dimension) and $k$ (number of non-zero entries per row) control information preservation and watermark strength.
    - Design Motivation: The circulant structure ensures that all original dimensions contribute equally to the watermark, preventing information concentration in only a few dimensions.

## Key Experimental Results

### Main Results (Impact of Paraphrasing Attacks on Existing Methods)
GPT-3.5 paraphrasing attack on the Enron dataset:

| Method | ACC↑ | AUC (Verification)↓ | Attack Successful |
|------|------|-------------|-------------|
| WARDEN (No Attack) | 94.50 | 97.40 | ✗ |
| WARDEN + GPT-3.5 Paraphrase | 92.81 | **62.43** | ✓ (AUC drops significantly) |
| WET (No Attack) | 90.50 | 100.0 | ✗ |
| WET + GPT-3.5 Paraphrase | 89.80 | **99.98** | ✗ (AUC remains virtually unchanged) |

### WET vs Baselines (Average of 4 Datasets)

| Method | Downstream Task ACC | Verification AUC | Robust to Paraphrase |
|------|-------------|----------|-----------|
| EmbMarker | ~93% | ~95% (No Attack) → ~55% (After Attack) | ✗ |
| WARDEN | ~93% | ~97% (No Attack) → ~63% (After Attack) | ✗ |
| WET | ~90% | **~100%** (No Attack) → **~100%** (After Attack) | ✓ |

### Key Findings
- **Paraphrase attacks effectively dismantle existing watermarks**: WARDEN's verification AUC drops from 97.4 to 62.4 (near random), with minimal loss in downstream task performance.
- **WET is fully robust to paraphrasing**: The verification AUC remains near 100% across all paraphrasing methods (GPT-3.5, DIPPER, round-trip translation).
- **WET's utility cost is manageable**: Downstream task accuracy drops by about 3-4 percentage points, representing a reasonable security-utility trade-off.
- **Verification with a single sample**: WET near-perfectly verifies with only 1 queried sample, whereas EmbMarker/WARDEN require a large number of samples.
- The choice of a circulant matrix is critical—non-circulant random matrices suffer from poor condition numbers, leading to a significant drop in verification performance.

## Highlights & Insights
- **Complete chain of attack and defense**: Rather than simply proposing a new defense, this work first systematically validates the effectiveness of the paraphrasing attack (a new contribution) before designing a targeted defense.
- **Theoretical guarantee is the greatest advantage**: Theorem 1 proves the robustness of linear transformation to averaging from a linear algebra perspective, avoiding heuristic design. This makes WET not only empirically effective but also theoretically sound.
- **Fundamental improvement by avoiding trigger words**: Transitioning watermarking from "conditional injection" to a "global transformation" fundamentally eliminates the possibility of selective attacks.

## Limitations & Future Work
- The utility loss in downstream tasks (~3-4%) might be unacceptable for certain high-precision applications.
- The method was only evaluated with BERT as the surrogate model; surrogate models with different architectures (e.g., Transformer variants) might exhibit different behaviors.
- The security of the transformation matrix $\mathbf{T}$ completely relies on confidentiality—if leaked, the watermark can be removed.
- This work does not consider the impact of potential non-linear post-processing by attackers (such as non-linear layers in fine-tuning) on the watermark.

## Related Work & Insights
- **vs EmbMarker (Peng et al., 2023)**: EmbMarker uses a single target embedding + trigger words, making it easy to recover the target embedding using adversarial methods. WET does not use trigger words, and its transformation matrix is much larger than a single embedding, making it far harder to reverse engineer.
- **vs WARDEN (Shetty et al., 2024)**: WARDEN improves on EmbMarker by using multiple target embeddings but still relies on trigger words, which this paper proves can be bypassed by paraphrasing attacks.
- **vs Text Watermarking (Kirchenbauer et al., 2023)**: Text watermarking has also been shown to be vulnerable to paraphrasing, but the issue is more insidious for embedding watermarks, as operations in the embedding space are invisible to users.

## Rating
- Novelty: ⭐⭐⭐⭐ The linear transformation watermark is a novel design, and the discovery of the paraphrasing attack is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 datasets, 3 paraphrasing methods, and multiple hyperparameter ablations, backed by theoretical proofs.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, defense and attack logic is rigorous, and the theoretical derivation is concise.
- Value: ⭐⭐⭐⭐ Direct application value for EaaS intellectual property protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PECCAVI: Overcoming the Brittleness of AI Image Watermarking Under Visual Paraphrasing Attacks](../../CVPR2026/ai_safety/peccvai_overcoming_the_brittleness_of_ai_image_watermarking_under_visual_paraphr.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](../../NeurIPS2025/ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](../../NeurIPS2025/ai_safety/nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[AAAI 2026\] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service](../../AAAI2026/ai_safety/regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)
- [\[AAAI 2026\] Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data](../../AAAI2026/ai_safety/privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)

</div>

<!-- RELATED:END -->
