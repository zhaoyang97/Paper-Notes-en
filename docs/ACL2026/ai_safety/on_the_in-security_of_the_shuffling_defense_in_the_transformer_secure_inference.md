---
title: >-
  [Paper Note] On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference
description: >-
  [ACL2026][AI Safety][Secure Inference] This paper points out that the "disclosing intermediate activations after shuffling" defense commonly used in Transformer secure inference is insecure. It proposes an attack that fi…
tags:
  - "ACL2026"
  - "AI Safety"
  - "Secure Inference"
  - "Linear Layer Encryption"
  - "Shuffling Defense"
  - "Model Extraction"
  - "Activation Alignment"
date: 2026-05-08
content_hash: 03e72cfd9b30100c
---

# On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference

**Conference**: ACL2026  
**arXiv**: [2605.04901](https://arxiv.org/abs/2605.04901)  
**Code**: No public code  
**Area**: Model Security / Privacy Computing  
**Keywords**: Secure Inference, Linear Layer Encryption, Shuffling Defense, Model Extraction, Activation Alignment

## TL;DR
This paper points out that the "disclosing intermediate activations after shuffling" defense commonly used in Transformer secure inference is insecure. It proposes an attack that first aligns activations under different random permutations and then solves linear equations to extract weights. The attack can recover approximately usable model weights for Pythia-70m and GPT-2 with a query cost of approximately \$1.

## Background & Motivation
**Background**: The goal of Transformer secure inference is to ensure that the client only receives the final output while the server does not see the user's input, and the server's model weights are not exposed to the client. Traditional full-model encrypted inference places both linear and non-linear layers within MPC or homomorphic encryption protocols, which has the cleanest privacy goals but is very costly in Transformers, especially for non-linear layers like softmax, GELU, and LayerNorm.

**Limitations of Prior Work**: Existing systems have found that the secure computation of non-linear layers often requires multiple communication rounds, truncation, comparison, or polynomial approximation, with communication latency accounting for 75% to 90% of the total delay. To make secure inference practical, a class of Linear Layer Encryption (LOE) schemes encrypts only the linear layers, hands the intermediate activations to the client to compute non-linear layers in plaintext, and then re-shares the secret results. While this LOE inference provides orders of magnitude speedup, it hands part of the information of the linear layer inputs and outputs to the client.

**Key Challenge**: If the client can see the activations before and after the linear layer, the weights satisfy an approximate linear relationship $X^{(l+1)} = X^{(l)}W^{(l)}$. Weights can be extracted by solving a linear system after collecting enough input-output pairs. Existing work uses random shuffling to disrupt activation positions, arguing that for a vector of length $h$, there are $h!$ permutations, making it almost impossible for an attacker to guess the true permutation. However, this argument only excludes "directly guessing the permutation" and does not exclude "aligning different shuffling results across queries."

**Goal**: The authors aim to answer a very specific question: when a client can only see activations after each random permutation and does not know any true permutations, can these activations still be restored to a common coordinate system to further recover the linear layer weights?

**Key Insight**: The key observation of the paper is that an attacker does not need to know the original permutation; they only need to make the activation values from multiple queries sufficiently close. Shuffling changes positions but not values; if two activation vectors themselves are very close, the elements in the same dimension will still be close to each other, and matching across vectors can be transformed into a minimum distance matching problem.

**Core Idea**: Use the random 1-bit error in secure truncation protocols to create "slight activation perturbations under the same prompt," then use Hungarian matching to align the shuffled activations, and finally solve for the row-column permuted version of the weights under an unknown common permutation.

## Method
This paper does not design a new secure inference system but analyzes what the LOE inference interface can leak from an attacker's perspective. The attacker is a semi-honest client: it follows the protocol and does not tamper with intermediate secret shares but can freely choose input prompts and observe the shuffled activations and final output probabilities that the protocol would normally return to the client.

### Overall Architecture
The attack process can be divided into four steps.

First, the attacker fixes an input sequence and repeatedly queries the LOE inference interface. Since secure inference encodes floating-point numbers as fixed-point numbers and executes truncation after multiplication, common protocols naturally introduce a random 1-bit truncation error; even with exactly the same prompt, multiple executions will produce small perturbations in the intermediate activations.

Second, for each layer, collect shuffled activations returned by multiple queries. For the $l$-th linear layer, what the attacker obtains is not $x_i^{(l)}$, but $sf(x_i^{(l)}, \pi_i^{(l)})$, where the permutation $\pi_i^{(l)}$ for each query might be different.

Third, within each layer, select an arbitrary shuffled vector as the reference permutation and align the other shuffled vectors to the same unknown permutation. There is no need to recover the original permutation; it is only necessary to ensure that the internal coordinates of all samples are consistent.

Fourth, substitute the aligned input and output activation matrices into a linear system and recover the weights via pseudo-inverse. The resulting weights differ from the original weights by row and column permutations, but as long as the permutations between consecutive layers are consistent, the forward propagation functionality remains equivalent.

### Key Designs
1.  **Shuffling Activation Alignment Based on Nearest Distance**:
    - **Function**: Aligns two activation vectors that have undergone different random permutations to the same permutation, allowing elements of the same hidden dimension to correspond again.
    - **Mechanism**: If the original vectors $x_a$ and $x_b$ are sufficiently close, the numerical distance between corresponding elements remains minimal even if the coordinate order in the shuffled $x_a'$ and $x_b'$ is different. The authors formulate the alignment as $\min_M \|x_a' - x_b'M\|_2$, where $M$ is a permutation matrix; this is then converted into a standard assignment problem by constructing a cost matrix $D[i,j]=(x_a'[i]-x_b'[j])^2$ and using the Hungarian algorithm to find the minimum total cost match.
    - **Design Motivation**: Existing shuffling defenses assume attackers must guess one of $h!$ permutations, but the attack here does not guess the global permutation at all; it uses numerical proximity for element-wise matching. As long as the perturbation is small enough not to disrupt the neighborhood relationship, the exponential permutation space collapses into an efficiently solvable bipartite matching problem.

2.  **Constructing Neighboring Activations via Secure Truncation Errors**:
    - **Function**: Creates a set of intermediate activations that are close to each other but not identical, despite the constraint that Transformer inputs are discrete tokens and input vectors cannot be arbitrarily fine-tuned.
    - **Mechanism**: Transformer token inputs pass through embeddings first, so attackers cannot directly add $\epsilon$ perturbations as in continuous input tasks. Instead, the authors leverage fixed-point truncation in secure inference: floating-point values are typically encoded as $x=\lfloor x_f\cdot 2^p\rceil$, and truncation is performed after multiplication for precision, where protocols like ABY3, CrypTFlow2, and SecretFlow-SPU may generate random 1-bit errors. Using the default 18-bit precision as an example, this corresponds to a micro-perturbation of about $2^{-18}$; by repeatedly submitting the same prompt, the attacker can obtain a set of alignable neighboring activations.
    - **Design Motivation**: This step is the most critical "realization" part of the paper. Simply claiming that neural networks are Lipschitz continuous is insufficient because GPT inputs are discrete; truncation error turns protocol implementation details into the random perturbation source needed for the attack, while remaining within the semi-honest threat model.

3.  **Extracting Functionally Equivalent Weights Under Unknown Permutations**:
    - **Function**: Restores the weights of each linear layer using aligned activation input-output pairs and explains why not knowing the true permutation does not hinder the value of the attack.
    - **Mechanism**: For the $l$-th layer, the original relationship is $X^{(l+1)}=X^{(l)}W^{(l)}$. The attacker actually solves for $W'^{(l)} = \pi^{(l)^{-1}}W^{(l)}\pi^{(l+1)}$, which is the original weight with row permutations in the input dimension and column permutations in the output dimension. Since non-linearities like GELU, softmax, and LayerNorm can maintain equivalence under permutations of the corresponding dimensions, using mutually compatible permutations for consecutive layers allows the entire forward propagation to output intermediate representations subjected to the same coordinate transformations, ultimately maintaining functional consistency.
    - **Design Motivation**: This directly addresses a possible counter-argument: whether the extracted matrices are useful even if they are not the original ones. The paper points out that neural network weights possess permutation symmetry, and row-column permuted weights can still serve as proxy models, for analyzing model information, or as initialization for subsequent black-box attacks.

### Loss & Training
The paper does not train a new model or optimize a neural network loss; the core computation involves two numerical problems.

In the alignment phase, the minimum cost matching is solved with the objective of minimizing $\sum_{i,j}M[i,j]D[i,j]$, subject to the constraint that each row and column of $M$ contains exactly one 1. This is solved precisely by the Hungarian algorithm.

In the weight extraction phase, a linear system is solved. Theoretically, the number of samples must at least reach the rank of the weight input dimension; in experiments, to mitigate ill-conditioned matrices, the authors use 16 times the maximum dimension for the number of queries. Due to the poor singular value distribution of the neighboring activation matrix, a condition number threshold $C$ is set when calculating the pseudo-inverse, discarding singular values smaller than $\sigma_{max}/C$; in most experimental scenarios, $C=10^7$ is the most stable.

## Key Experimental Results

### Main Results
Experiments were verified on Pythia-70m and GPT-2, covering major linear layer types in Transformers, including $W_{qkv}$, $W_o$ of attention, and $W_{h1}$, $W_{h2}$ of FFN. Fixed-point precision was tested at 14, 16, and 18 bits, where 18 bits corresponds to the common default precision in frameworks like SecretFlow-SPU.

| Model | Max Dim | Queries | Weight Recovery Error | Proxy Model Performance |
| :--- | :--- | :--- | :--- | :--- |
| Pythia-70m | 2048 | 32768 | L1 difference for most layers is ~$10^{-4}$ to $10^{-2}$; overall lower for smaller models | Wikitext PPL from 31.81 (orig), 44.46 (stolen), to 32.43 (after fine-tuning) |
| GPT-2 | 3072 | 49512 | L1 difference for most layers is ~$10^{-4}$ to $10^{-2}$; harder to recover for larger dimension layers | Wikitext PPL from 21.11 (orig), 47.92 (stolen), to 21.15 (after fine-tuning) |

| Layer Dim | FXP Prec | Input Matches / Dim | Input MSE | Output Matches / Dim | Output MSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 512 -> 2048 | 14 | 508 / 512 | 9.4E-07 | 1996 / 2048 | 1.2E-06 |
| 512 -> 2048 | 18 | 512 / 512 | 0.0E+00 | 2046 / 2048 | 4.0E-08 |
| 2048 -> 512 | 14 | 2004 / 2048 | 3.0E-06 | 504 / 512 | 5.9E-06 |
| 2048 -> 512 | 18 | 2046 / 2048 | 5.5E-08 | 511 / 512 | 1.5E-08 |
| 768 -> 3072 | 14 | 756 / 768 | 6.0E-07 | 3052 / 3072 | 1.2E-07 |
| 768 -> 3072 | 18 | 766 / 768 | 2.5E-08 | 3070 / 3072 | 3.7E-09 |
| 3072 -> 768 | 14 | 3056 / 3072 | 2.5E-07 | 762 / 768 | 5.8E-08 |
| 3072 -> 768 | 18 | 3070 / 3072 | 1.2E-08 | 764 / 768 | 5.0E-09 |

### Ablation Study
The analytical experiments in the paper mainly revolve around fixed-point precision, condition number threshold, and model scale.

| Config | Key Metric | Description |
| :--- | :--- | :--- |
| FXP 18 bit | Alignment MSE can reach $10^{-9}$ to $10^{-8}$ | Higher precision results in smaller truncation perturbations, making corresponding elements easier to match correctly |
| FXP 14 bit | Alignment error is typically larger, but individual weight recovery is sometimes better | Lower precision brings larger perturbations, which harms matching but increases the diversity of the input matrix, mitigating the ill-conditioned pseudo-inverse problem |
| Condition Threshold $C=10^7$ | Lowest L1 weight difference in most experiments | A threshold too small loses useful singular values, while one too large amplifies numerical noise |
| Pythia-70m vs GPT-2 | Pythia-70m weight recovery is more accurate | Attack difficulty increases with input dimension and model scale; layers like $W_{h2}$ with input dimensions of $4d_{model}$ are typically harder |

### Key Findings
- Shuffling does indeed have a destructive effect on "single activation structure" but cannot stop cross-query alignment; as long as the attacker can obtain a batch of neighboring activations, element-wise correspondence can be recovered through numerical distance.
- The alignment quality is very high. The mismatch ratio for most configurations in Table 1 is below 2%, with MSE between $10^{-9}$ and $10^{-6}$, indicating that the subsequent linear system is not built on rough guesses.
- The query cost is not excessive. The authors mitigate ill-conditioning using 16 times the maximum dimension in queries, approximately 33,000 for Pythia-70m and 50,000 for GPT-2; estimated at short-answer API token rates, the cost is around \$1.
- Extracted weights are not just "numerically close." The PPL of the stolen models worsens significantly on Wikitext without fine-tuning, but after at most 6 minutes of short fine-tuning, both Pythia-70m and GPT-2 approach the perplexity of the original models, proving they can serve as practical proxy models.
- Fixed-point precision has a dual role: high precision favors alignment, while low precision sometimes favors matrix inversion stability. This suggests that defenses cannot be solved simply by adjusting precision; protocol-level random errors and numerical conditions must be analyzed together.

## Highlights & Insights
- The most valuable part of this paper is deconstructing the security intuition that "the permutation space is large." A real attack does not need to recover the original permutation used by the server; it only needs to place samples into the same unknown coordinate system, and the difficulty of the defense argument shifts from factorial-level search to polynomial-time matching.
- The utilization of secure truncation errors is very clever. They are not malicious perturbations outside the protocol but are natural randomness introduced by many secure inference implementations for fixed-point calculations, so the attack remains valid in the semi-honest model without secretly relaxing the threat model.
- The "row-column permutation equivalence" of weights makes the attack results more severe. Even if the recovered matrices are not arranged in the original neuron order, model functionality can be maintained as long as the permutations for adjacent layers are consistent, explaining why unknown permutations still lead to effective model extraction.
- The paper provides a strong insight for privacy computing systems: analyzing the entropy or correlation of a single output is not enough. Once a secure interface supports repeated queries, cross-query correlation, numerical error, and protocol randomness all become attack surfaces.
- This line of thought can be migrated to other schemes that "publicize intermediate representations after scrambling." For example, in Vision Transformers, edge inference, or distributed privacy inference, as long as plaintext non-linear layer computation relies on permutation invariance, one should check if neighboring samples can be realigned.

## Limitations & Future Work
- The authors' experiments only verified relatively small-scale models like Pythia-70m and GPT-2. The paper admits that as model size increases, the fidelity of recovered weights will drop; whether a sufficiently strong proxy model can still be recovered at a similar cost for ultra-large LLMs still requires large-scale empirical evidence.
- The attack relies on the client's ability to observe shuffled activations of enough layers and to repeatedly query the same prompt. If actual systems limit the intermediate information returned, aggregate non-linear computation interfaces, or audit repeated queries, the cost and feasibility of the attack might change.
- Alignment relies on the element spacing of neighboring activations being sufficiently separable. If defenses introduce additional noise, inter-batch mixing, randomizing dimension groups, or irreversible perturbations, the mismatch rate of Hungarian matching might rise, but these changes would also affect plaintext non-linear computation accuracy and protocol efficiency.
- There are obvious numerical ill-conditioning problems in the weight recovery phase. The authors mitigate this by discarding singular values through a condition number threshold, but this remains an empirical choice; future work could study the impact of high-precision linear algebra, iterative solvers, or regularization methods on recovery quality.
- On the defense side, simple shuffling should no longer be considered sufficiently secure. A more robust path might involve reducing intermediate activation exposure, designing faster cryptographic protocols for non-linear layers, or establishing formal security definitions for LOE systems against multi-query attacks.

## Related Work & Insights
- **vs Full-model encryption secure inference**: FME keeps all model computations within the encrypted domain, offering stronger privacy goals but at a very high cost for non-linear layers. This paper attacks the LOE route that discloses shuffled activations for efficiency, showing that performance optimization sacrifices not just abstract risks but a practically exploitable weight leakage surface.
- **vs Early plaintext non-linear schemes (e.g., GELU-net, Bayhenn)**: Early schemes reduced leakage risk by limiting queries or using Bayesian networks, but subsequent work has proven that models can still be extracted. The difference in this paper is attacking the more modern "shuffling defense" version, showing that hiding coordinate order alone is insufficient.
- **vs Shuffling-based LOE inference (e.g., PermLLM, PP-Stream, Centaur)**: These works exploit the fact that non-linear layers are insensitive to element positions by handing randomly permuted activations to the client. This paper proves that attackers can recover a consistent coordinate system across queries, necessitating a re-evaluation of the security claims of these systems.
- **vs Carlini/Jagielski series model extraction attacks**: Traditional model extraction often learns proxy models from black-box logits or output probabilities. This paper additionally utilizes the intermediate activations leaked by the LOE interface, reducing the problem to more direct linear algebra recovery. The insight is that the secure inference interface itself may expose stronger attack signals than a standard API.
- **vs Privacy shuffling related work**: Some privacy mechanisms use shuffling as a means to reduce correlation. This paper reminds us that whether shuffling is secure depends on whether an attacker can obtain neighboring samples and repeated observations; when numerical values themselves are retained, anonymizing coordinates is not equivalent to deleting information.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Does not simply reproduce existing model extraction but targets core assumption vulnerabilities in the shuffling defense, converting permutation recovery into neighborhood matching and linear solving.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two Transformers, three levels of fixed-point precision, alignment error, weight error, and PPL proxy effects, but lacks verification on larger models and real deployed systems.
- Writing Quality: ⭐⭐⭐⭐ The paper structure is clear, and the threat model, algorithms, and equivalence proofs are well-connected; however, some weight error numbers in certain figures are not fully tabulated, and reproduction details could be more transparent.
- Value: ⭐⭐⭐⭐⭐ Highly meaningful for privacy-preserving Transformer inference, directly challenging the common shuffling security claim in LOE systems and providing explicit pressure tests for future defense design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute](../../AAAI2026/ai_safety/secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[CVPR 2026\] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](../../CVPR2026/ai_safety/all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](../../CVPR2026/ai_safety/decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[NeurIPS 2025\] OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction](../../NeurIPS2025/ai_safety/omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)

</div>

<!-- RELATED:END -->
