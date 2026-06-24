---
title: >-
  [Paper Note] Language Models Resist Alignment: Evidence From Data Compression
description: >-
  [ACL 2025 (Best Paper Award)][Model Compression][elasticity] This paper proposes the concept of LLM "elasticity" from a data compression perspective, proving that the change in compression rate under fine-tuning perturbations is inversely proportional to the dataset size. Because the pre-training data is vastly larger than the alignment data, alignment effects are preferentially "forgotten." This fundamentally explains the fragility of LLM alignment from an information-theore…
tags:
  - "ACL 2025 (Best Paper Award)"
  - "Model Compression"
  - "elasticity"
  - "inverse alignment"
  - "compression theory"
  - "alignment fragility"
  - "Hooke's Law analogy"
date: 2026-05-08
content_hash: e0d25296f19369da
---

# Language Models Resist Alignment: Evidence From Data Compression

**Conference**: ACL 2025 (Best Paper Award)  
**arXiv**: [2406.06144](https://arxiv.org/abs/2406.06144)  
**Code**: [pku-lm-resist-alignment.github.io](https://pku-lm-resist-alignment.github.io)  
**Area**: Model Compression  
**Keywords**: elasticity, inverse alignment, compression theory, alignment fragility, Hooke's Law analogy  

## TL;DR

This paper proposes the concept of LLM "elasticity" from a data compression perspective, proving that the change in compression rate under fine-tuning perturbations is inversely proportional to the dataset size. Because the pre-training data is vastly larger than the alignment data, alignment effects are preferentially "forgotten." This fundamentally explains the fragility of LLM alignment from an information-theoretic standpoint.

## Background & Motivation

**Background**: Current mainstream LLM alignment practices use methods such as SFT, RLHF, and DPO to fine-tune pre-trained models on relatively small amounts of high-quality data to align model behavior with human intent and safety guidelines. These approaches have achieved significant practical success, with models like GPT-4 and Llama 2 undergoing meticulous safety alignment.

**Limitations of Prior Work**: Numerous studies have demonstrated that the effects of alignment are highly fragile. Yang et al. (2023) showed that even a minor amount of malicious fine-tuning can cause a safe model to become unsafe again; Qi et al. (2024) even found that fine-tuning on non-malicious data can compromise safety mechanisms. More concerningly, Hubinger et al. (2024) demonstrated the "sleeper agent" phenomenon, where models can retain hidden harmful behaviors even after safety training.

**Key Challenge**: A massive order-of-magnitude discrepancy exists between the volume of alignment data (usually thousands to tens of thousands of samples) and pre-training data (typically trillions of tokens). What does this vast difference in data volume imply? Qi et al. (2024) proposed the concept of "shallow safety alignment," suggesting that alignment fails to penetrate into the deep representations of models. However, this remains a phenomenological description and lacks a fundamental mechanistic or theoretical explanation.

**Goal**: (1) Are the effects of alignment fine-tuning deep or superficial? (2) Is there an inherent mechanism that makes LLMs resist alignment? (3) If so, how does this mechanism enable "inverse alignment"? (4) How do model scale and pre-training data volume influence this resistance?

**Key Insight**: The authors start with the classic equivalence of "language modeling is data compression"—minimizing training loss is equivalent to minimizing the compression rate. If training and alignment are modeled as the joint compression of different datasets, how does the discrepancy in data volume affect the change in compression rate for each dataset? This perspective is promising, as compression theory provides mature mathematical tools to analyze such asymmetry.

**Core Idea**: By modeling LLM training as a data compression process, they prove that the change in compression rate of each dataset under fine-tuning perturbations is inversely proportional to its size. Consequently, the "hardness" of pre-training data is vastly higher than that of alignment data, leading to an intrinsic resistance to alignment within LLMs.

## Method

### Overall Architecture

The methodological framework of this paper consists of three layers: (1) **Modeling Layer**—formulates LLM training and alignment as data compression problems, introducing Token Trees and compression protocols; (2) **Theoretical Layer**—derives the elasticity theorem (Theorem 4.2) under the Pareto distribution assumption, proving the inverse relationship between compression rate change and dataset size; (3) **Validation Layer**—systematically verifies two sub-phenomena of elasticity, namely resistance and rebound, through experiments across various models, algorithms, and scales. The overall inputs are the pre-training dataset $\mathcal{D}_p$, alignment dataset $\mathcal{D}_a$, and perturbation dataset $\mathcal{D}_t$, while the outputs are theoretical and experimental conclusions regarding the laws governing the changes in the compression rates of each dataset.

### Key Designs

1. **Token Tree Modeling and Compression Protocol**:

    - **Function**: Establishes a rigorous formal framework of compression theory for the LLM training process.
    - **Mechanism**: Represents all possible responses in a dataset as a Token Tree $\mathcal{T}$, where each node has 0/1 child nodes and an EOS leaf node, with leaf node weights representing the probability of the corresponding response. Model training is equivalent to learning the node weights of the tree. Since models with finite parameters cannot precisely model node weights at arbitrary depths, it is assumed that the depth $d$ that can be accurately modeled increases monotonically with the model scale. The compression protocol consists of two steps: first pruning the Token Tree to depth $d$, and then compressing the pruned tree using Huffman coding. Defining the compression rate $\gamma_{p_\theta}^{\mathcal{D}_i}$ as the ratio of the compressed encoding length to the original length, minimizing training loss is equivalent to minimizing the compression rate.
    - **Design Motivation**: Traditional training loss analysis struggles to reveal the competitive relationship between different datasets, whereas a compression perspective naturally supports the joint analysis of multiple datasets—the node weights of the joint dataset $p_l^{\mathcal{D}} = \sum_i p_l^{\mathcal{D}_i} |\mathcal{D}_i| / \sum_i |\mathcal{D}_i|$ explicitly capture the weighting effect of data volume.

2. **Derivation of the Elasticity Theorem (Theorem 4.2)**:

    - **Function**: Mathematically and rigorously proves that the change in compression rate under fine-tuning perturbations is inversely proportional to dataset size.
    - **Mechanism**: Defines the normalized compression rate $\gamma_{p_\theta}^{\mathcal{D}_i/\mathcal{D}} = \gamma_{p_\theta}^{\mathcal{D}_i} - \log M$ (where $M$ is the number of leaf nodes after pruning) and introduces the concept of mass distribution to transform the entropy of response distributions into the entropy of leaf node probability random variables. Under the assumption that the mass distribution follows a Pareto distribution (supported by Zipf's Law), it derives that when the perturbation data volume $|\mathcal{D}_t|$ increases, $d\gamma_{p_\theta}^{\mathcal{D}_a/\mathcal{D}} / dl = \Theta(-k \cdot d\gamma_{p_\theta}^{\mathcal{D}_p/\mathcal{D}} / dl)$, where $k = |\mathcal{D}_p|/|\mathcal{D}_a|$. That is, the rate of change of the compression rate of the alignment dataset is $k$ times that of the pre-training dataset (where $k$ is typically on the order of $10^3 \sim 10^6$).
    - **Design Motivation**: Relying solely on the intuition that "larger data volume is more important" is insufficient; rigorous mathematical derivation is required to quantify the exact extent of this discrepancy. The inverse relationship implies that the degradation rate of alignment effects under perturbations is several orders of magnitude faster than that of pre-training effects, which is the root cause of alignment fragility.

3. **Hooke's Law Analogy Framework**:

    - **Function**: Provides an intuitively understandable physical analogy for the elasticity theorem and identifies elasticity invariants.
    - **Mechanism**: Analogizes the elasticity of LLMs to a series spring system. The dataset size $|\mathcal{D}_i|$ corresponds to the spring constant $k_i$, and the KL divergence change $\Delta D_{KL}(\mathcal{P}_{p_\theta} \| \mathcal{P}_{\mathcal{D}_i})$ corresponds to the spring deformation $\Delta l_i$. The elastic force $F \propto |\mathcal{D}_i| \cdot \Delta D_{KL}(\mathcal{P}_{p_\theta} \| \mathcal{P}_{\mathcal{D}_i})$ corresponds to Hooke's Law $F = k \cdot \Delta l$. In a series spring system, under the same external force, the spring with lower stiffness undergoes greater deformation—corresponding to the smaller-volume alignment dataset changing more under perturbation.
    - **Design Motivation**: While the mathematical form of the elasticity theorem is rigorous, the physical analogy makes it intuitive. The series spring model also reveals an important elasticity invariant—$|\mathcal{D}_i| \cdot \Delta D_{KL}$ remains constant across different datasets, providing a testable prediction for subsequent validation.

### Loss & Training

This paper does not propose new training methods but rather analyzes the failure mechanisms of existing alignment training. Experiments utilize the standard SFT loss $\mathcal{L}_{SFT}(\theta; \mathcal{D}) = -\mathbb{E}_{(x,y) \sim \mathcal{D}}[\log p_\theta(y|x)]$ for forward and inverse alignment. To verify resilience, additional tests are conducted using various alignment algorithms, including RLHF/PPO, DPO, KTO, and SimPO, demonstrating that the elasticity phenomenon is independent of the specific alignment algorithm.

## Key Experimental Results

### Main Results

**Resistance Validation (Table 1)**: Comparison of training losses between forward alignment and inverse alignment on Alpaca/TruthfulQA/Beavertails datasets.

| Dataset | Base Model | $\theta_2 \to \theta_1$ vs $\theta_1 \to \theta_2$ | $\theta_3 \to \theta_2$ vs $\theta_2 \to \theta_3$ | $\theta_3 \to \theta_1$ vs $\theta_1 \to \theta_3$ |
|--------|----------|------|------|------|
| Alpaca | Llama2-7B | 0.159↓ vs 0.202↑ | 0.195↓ vs 0.214↑ | 0.167↓ vs 0.235↑ |
| Alpaca | Llama2-13B | 0.177↓ vs 0.196↑ | 0.215↓ vs 0.241↑ | 0.184↓ vs 0.235↑ |
| Alpaca | Llama3-8B | 0.254↓ vs 0.257↑ | 0.227↓ vs 0.323↑ | 0.234↓ vs 0.323↑ |

In all experiments, the training loss of inverse alignment is lower than that of forward alignment, verifying the existence of resistance.

### Ablation Study

**KL Divergence Rebound Validation (Table 3)**: The amount of unsafe data required to reduce KL divergence below $\epsilon = 0.01$ after alignment with varying amounts of safety data.

| Base Model | Safety Data (1000) | Safety Data (2000) | Safety Data (5000) | Safety Data (10000) |
|----------|:---:|:---:|:---:|:---:|
| Llama2-7B Post-alignment KL | 0.21 | 0.22 | 0.26 | 0.27 |
| Llama2-7B Unsafe Data Required for Reversal | 961 | 844 | 801 | 729 |
| Gemma-2B Post-alignment KL | 0.18 | 0.21 | 0.24 | 0.25 |
| Gemma-2B Unsafe Data Required for Reversal | 923 | 853 | 709 | 598 |

Models aligned with more safety data conversely require less unsafe data to be reversed—perfectly matching the prediction of the elasticity theorem.

### Key Findings

- **Elasticity correlates positively with model scale**: Across Qwen 0.5B $\to$ 4B $\to$ 7B, larger models rebound faster. This resembles an "inverse scaling law"—the stronger the model's capabilities, the easier it is to reverse its alignment.
- **Elasticity correlates positively with pre-training data volume**: For the TinyLlama series trained from 0.1T to 3.0T, more pre-training data leads to stronger elasticity. Elasticity is barely observable at 0.1T but emerges significantly at 0.5T, suggesting a critical threshold data volume for its emergence.
- **Elasticity is independent of alignment algorithms**: Consistent rebound phenomena are observed across SFT, PPO, DPO, KTO, and SimPO, demonstrating that elasticity is an inherent property of the model rather than a flaw in specific algorithms.
- **Elasticity exhibits bidirectional symmetry**: Reverse experiments (training first on negative data, then reversing with positive data) observe the same elasticity, ruling out experimental setup bias.
- **Rebound follows a two-stage pattern**: The initial stage exhibits rapid performance degradation (moving away from the pre-training distribution $\to$ rapid rebound), followed by a slower decline in later stages (closer to the pre-training distribution $\to$ stabilization), which is consistent with the predictions of the spring model.

## Highlights & Insights

- **Exquisite physical analogy**: The series spring analogy transforms complex information-theoretic derivations into intuitive understanding—pre-training data acts as a "hard spring" (high stiffness) and alignment data as a "soft spring" (low stiffness). The same external force causes much larger deformation in the soft spring. This interdisciplinary analogy lowers the comprehension barrier and inspires new research directions.
- **Perfect closed loop of theoretical prediction and experimental validation**: The elasticity theorem predicts that the "amount of change is inversely proportional to dataset size." This prediction is precisely verified across multiple dimensions (model scale, data volume, algorithm type), and a critical threshold for elasticity emergence (0.1T~0.5T) is discovered. This paradigm of theory-driven experimental discovery is highly exemplary.
- **Revealing the fundamental dilemma of alignment research**: If elasticity is caused by an unavoidable structural factor—namely, the discrepancy between pre-training and alignment data volume—solely improving alignment algorithms may not fundamentally solve the problem. This insight holds strategic guidance for the entire AI safety field, suggesting either a drastic increase in the scale of alignment data to make it comparable to pre-training data, or the pursuit of entirely different alignment paradigms.

## Limitations & Future Work

- **Unverified Pareto distribution assumption**: The core theoretical assumption (Assumption A.7) is that the mass distribution of leaf nodes in the Token Tree follows a Pareto distribution. Although indirectly supported by Zipf's Law, this assumption has not been directly verified on actual LLM training data.
- **Lack of coverage of the full pre-training life cycle**: Due to computational cost constraints, the experiments could not systematically verify elasticity throughout the entire process of pre-training from scratch followed by alignment, using instead existing pre-trained models for subsequent experiments.
- **Mitigation strategies remain purely conceptual**: Appendix C.2 proposes adjusting data ratios based on the elasticity theorem to mitigate the risk of inverse alignment, but these schemes have not been implemented or validated.
- **Insufficient quantification of the elasticity threshold**: Although elasticity is found to emerge between 0.1T and 0.5T, the exact critical point cannot be precisely determined due to the lack of fine-grained pre-training data checkpoints.
- **Failure to explore multimodal settings**: It remains unclear whether elasticity is present in multimodal scenarios like vision-language models, which is crucial for evaluating the security of multimodal alignment.

## Related Work & Insights

- **vs Qi et al. (2024) "Shallow Safety Alignment"**: They proposed that alignment should go beyond surface-level tokens and penetrate into inner representations. This paper provides a theoretical explanation for why alignment is destined to be shallow—the order-of-magnitude difference between pre-training and alignment data is the structural root cause, not poor design of algorithms. The advantage of this work lies in offering a quantitative theoretical framework.
- **vs Hubinger et al. (2024) "Sleeper Agents"**: They showed from an adversarial training perspective that aligned models can retain hidden harmful behaviors. This paper offers a more fundamental explanation from an information-theoretic view—elasticity is an inherent property of LLMs; no adversarial design is needed, and data volume discrepancy alone is sufficient to make alignment fragile.
- **vs Wei et al. (2024) weight-attribution-based analysis**: They isolated safety-critical regions and capability-critical regions from a weight perspective, which represents local-level analysis. This paper provides a complementary explanatory framework from a global data compression perspective. The two can be combined—elasticity theory explains "why," while weight attribution analyzes "where."
- **Implications for open-source LLM ecosystems**: If meticulously aligned models can be reversed at a very low cost, the attack-defense balance in the open-source community faces fundamental challenges, necessitating the development of alignment methods that are "irreversible through fine-tuning."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulating a theoretical explanation of alignment fragility from a data compression perspective for the first time; the concept of elasticity and the Hooke's Law analogy are highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation spanning 4 model families (Llama 2/3, Gemma, Qwen), 5 alignment algorithms (SFT/PPO/DPO/KTO/SimPO), and multiple data scales.
- Writing Quality: ⭐⭐⭐⭐ While the theoretical section requires some mathematical background, the physical analogy and clear experimental design significantly lower the comprehension barrier.
- Value: ⭐⭐⭐⭐⭐ Recipient of the ACL 2025 Best Paper Award, exposing fundamental challenges in AI alignment and providing strategic guidance for safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](../../ACL2026/model_compression/alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)
- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)

</div>

<!-- RELATED:END -->
