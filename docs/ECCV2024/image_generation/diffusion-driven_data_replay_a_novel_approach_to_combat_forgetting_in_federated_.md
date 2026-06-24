---
title: >-
  [Paper Note] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning
description: >-
  [ECCV 2024][Image Generation][Federated Class Continual Learning] This work proposes the DDDR framework, which is the first to introduce pre-trained diffusion models into Federated Class Continual Learning (FCCL). Through Federated Class Inversion technology, it learns a compact class embedding for each category, using the diffusion model to perform high-quality replay of historical data to combat catastrophic forgetting, and employs contrastive learning to bridge the domain…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Federated Class Continual Learning"
  - "Catastrophic Forgetting"
  - "Diffusion Models"
  - "Data Replay"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: e367844ec9ea2575
---

# Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning

**Conference**: ECCV 2024  
**arXiv**: [2409.01128](https://arxiv.org/abs/2409.01128)  
**Code**: [Yes (GitHub)](https://github.com/jinglin-liang/DDDR)  
**Area**: Diffusion Models / Federated Learning  
**Keywords**: Federated Class Continual Learning, Catastrophic Forgetting, Diffusion Models, Data Replay, Contrastive Learning

## TL;DR

This work proposes the DDDR framework, which is the first to introduce pre-trained diffusion models into Federated Class Continual Learning (FCCL). Through Federated Class Inversion technology, it learns a compact class embedding for each category, using the diffusion model to perform high-quality replay of historical data to combat catastrophic forgetting, and employs contrastive learning to bridge the domain gap between generated and real data.

## Background & Motivation

**Background**: Federated Learning (FL) is crucial in privacy-sensitive domains like healthcare and finance. However, in real-world applications, clients continuously introduce data from new classes, leading to the emerging problem of Federated Class Continual Learning (FCCL). The core challenge of FCCL is catastrophic forgetting, where the model loses its memory of old tasks while learning new ones.

**Limitations of Prior Work**:

**Empirical Replay Unavailable**: The most effective approach in continual learning is storing and replaying old task data. However, in federated scenarios, privacy protection requirements prohibit clients from retaining historical data long-term, and client dropouts can cause the loss of stored data.

**Unstable GAN Training**: FedCIL uses federally trained ACGANs to generate historical data, but GAN training itself is notoriously unstable, a problem that is further amplified in federated settings.

**Poor Data-Free Distillation Quality**: Target and MFCL train generators using data-free knowledge distillation, but the generated samples often resemble adversarial examples, exhibiting a large gap from the real data distribution and offering limited guidance to the model.

**Key Challenge**: Under the premise of not storing any real historical data, how can historical class data be reconstructed with high quality to combat forgetting?

**Key Insight**: Leveraging the powerful generation capabilities of pre-trained diffusion models: instead of training the generative model itself, it searches for a conditional embedding (class embedding) for each category in its input space, substantially reducing computation and communication costs while ensuring generation quality.

## Method

### Overall Architecture

DDDR consists of two phases: (1) **Federated Class Inversion Phase** — reverse-engineering a class embedding for each new category using a frozen pre-trained LDM; (2) **Replay-Augmented Training Phase** — using the class embeddings of historical tasks to generate replay data, which is combined with real data from the new task to train the classifier.

### Key Designs

1. **Federated Class Inversion**

   Core Idea: Instead of training the diffusion model, it performs reverse-engineering using a frozen pre-trained Latent Diffusion Model (LDM). The frozen text prompt "a photo of" is encoded by the text encoder to obtain $c_\theta(p)$, which is then concatenated with a learnable class embedding $v$ to form the conditioning guide $[c_\theta(p); v]$. The optimization objective is:

   $v_i^* = \arg\min_v \mathbb{E}_{z \sim \mathcal{E}(X_i), p, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\alpha_t}z + \sqrt{1-\alpha_t}\epsilon, t, [c_\theta(p); v])\|_2^2 \right]$

   where $X_i$ is the set of images for the $i$-th class, $\mathcal{E}$ is the encoder of the LDM, and $\epsilon_\theta$ is the frozen denoising model.

   **Design Motivation**: The class embedding can be viewed as a compressed representation of the category. Only this small vector (instead of the entire generative model) needs to be optimized and communicated, which dramatically reduces computation and communication resources. Furthermore, since the LDM parameters are not modified, the probability of generating images identical to the original data is extremely low, thereby enhancing privacy protection.

2. **Global Class Embedding Aggregation**

   In the federated setting, after each client locally optimizes the class embedding, the server aggregates them using FedAvg: $v_i = \frac{1}{k}\sum_{j=1}^k v_i^{(j)}$. Local training and global aggregation are performed iteratively until convergence, and the class embeddings are saved for subsequent data replay.

   **Design Motivation**: Transmitting only the class embedding instead of the data itself protects privacy while effectively aggregating the category knowledge distributed across different clients.

3. **Contrastive Learning Constraint**

   A domain gap exists between generated and real data. A supervised contrastive learning loss is used to close the distance between generated and real data of the same class in the feature space:

   $\mathcal{L}_{SCL} = \mathbb{E}_{e_i, e_p \sim P(e_i)} \left[ \log \frac{\exp(sim(e_i, e_p)/\tau)}{\sum_{i \neq j} \exp(sim(e_i, e_j)/\tau)} \right]$

   where $P(e_i)$ is the set of positive samples belonging to the same class as $e_i$, and $sim$ calculates the similarity after mapping to an $l_2$-normalized space via an MLP.

   **Design Motivation**: To enhance the classifier's generalization capability across both generated and real domains, indirectly boosting the representational capacity of the generated data relative to the real data.

### Loss & Training

The final objective function for classifier training is:

$$\mathcal{F}^* = \arg\min_\mathcal{F} \underbrace{\mathcal{L}_{CE}}_{\text{Current Task}} + w_1 \underbrace{\mathcal{L}_{SCL}}_{\text{Contrastive Learning}} + w_2 \underbrace{\mathcal{L}_{PCE}}_{\text{Historical Task CE}} + w_3 \underbrace{\mathcal{L}_{KD}}_{\text{Knowledge Distillation}}$$

where $w_1=1, w_2=0.5, w_3=10$. $\mathcal{L}_{CE}$ calculates the cross-entropy on both real and generated data of the current task; $\mathcal{L}_{PCE}$ calculates the cross-entropy on the generated data of historical tasks; $\mathcal{L}_{KD}$ uses KL divergence to distill knowledge from the previous round's model into the current model.

Training details: ResNet-18 is used as the classifier with 5 clients, and the LDM is pre-trained on LAION-400M. The Class Inversion phase consists of 10 communication rounds $\times$ 50 local training steps; the Replay-Augmented phase consists of 100 rounds $\times$ 5 local training epochs. Additionally, data is also generated for the current task to alleviate non-IID issues.

## Key Experimental Results

### Main Results

**CIFAR-100 Comparison Experiments (Table 1):**

| Method | IID T=5 Acc↑ | IID T=5 FM↓ | IID T=10 Acc↑ | IID T=10 FM↓ | non-IID T=5 Acc↑ | non-IID T=5 FM↓ |
|---|---|---|---|---|---|---|
| Finetune | 17.33 | 0.83 | 9.03 | 0.88 | 16.48 | 0.81 |
| FedEWC | 21.35 | 0.69 | 11.76 | 0.73 | 20.96 | 0.70 |
| Target | 34.40 | 0.48 | 22.95 | 0.49 | 34.35 | 0.48 |
| MFCL | 42.67 | 0.37 | 31.35 | 0.46 | 41.19 | 0.34 |
| **Ours** | **51.04** | **0.29** | **43.45** | **0.32** | **48.45** | **0.26** |

**Tiny-ImageNet Comparison Experiments (Table 2):**

| Method | IID T=5 Acc↑ | IID T=5 FM↓ | non-IID T=10 Acc↑ | non-IID T=10 FM↓ |
|---|---|---|---|---|
| Finetune | 12.29 | 0.60 | 6.58 | 0.64 |
| Target | 17.56 | 0.45 | 11.28 | 0.42 |
| MFCL | 15.11 | 0.52 | 8.54 | 0.51 |
| **Ours** | **25.47** | **0.36** | **16.65** | **0.27** |

On CIFAR-100 IID T=5, it outperforms the SOTA (MFCL) by **+8.37%** in accuracy and reduces forgetting by **0.08**; on CIFAR-100 IID T=10, the improvement is up to **+12.10%**. It also leads comprehensively on Tiny-ImageNet.

### Ablation Study

**CIFAR-100, T=5, non-IID (Table 3):**

| ID | Historical Gen Data | Current Gen Data | Contrastive Learning | Acc↑ | FM↓ | Note |
|---|---|---|---|---|---|---|
| 1 | ✓ | ✓ | ✓ | **48.45** | **0.26** | Full Model |
| 2 | ✗ | ✓ | ✓ | 17.63 | 0.84 | No historical replay $\rightarrow$ almost total forgetting |
| 3 | ✓ | ✗ | ✓ | 44.29 | 0.36 | No current gen data $\rightarrow$ non-IID effect |
| 4 | ✓ | ✓ | ✗ | 45.34 | 0.28 | No contrastive learning $\rightarrow$ domain gap untouched |
| 8 | ✗ | ✗ | ✗ | 16.48 | 0.81 | Equivalent to Finetune |

### Key Findings

- **Historical replay data is the core component**: Removing it results in almost total forgetting (Row 2 vs. Row 1: Acc 48.45 $\rightarrow$ 17.63), and the FM sky-rockets from 0.26 to 0.84.
- **Current task generated data alleviates non-IID**: Increases Acc by +4.16% and decreases FM by 0.10, mitigating non-IID effects by sharing identically distributed generated data.
- **Contrastive learning needs to couple with current generated data**: Effective on its own (Row 1 vs. 4: +3.11% Acc), but becomes detrimental in the absence of current generated data (Row 3 vs. 6: -0.84%), showing that its effectiveness comes from bridging the domain gap between generated and real data.
- Generation quality visualization indicates that DDDR's generated images are much closer to the real distribution, whereas those generated by Target/MFCL look more like adversarial examples.

## Highlights & Insights

1. **Clever Reuse of Pre-trained Models**: Instead of training the diffusion model, it only searches for conditional embeddings in the input space—shifting the computation from "training the entire model" to "optimizing a single vector," which is highly resource-efficient.
2. **Class Embedding as a Compressed Class Representation**: A single small vector can represent all visual information of a category, making the concept simple yet powerful.
3. **Naturalness of Privacy Protection**: Transmitting only the class embedding without modifying the LDM parameters means the probability of generating identical copies of the training images is extremely low.
4. **Conditional Effectiveness of Contrastive Learning**: Experiments show that contrastive learning is only effective when coupled with current task generated data, an insight that provides valuable reference for future work.

## Limitations & Future Work

- Reliance on the pre-trained LDM (LAION-400M); performance may degrade when there is a large gap between the target domain and the LDM's pre-training domain.
- Validation is restricted to CIFAR-100 and Tiny-ImageNet, lacking experiments on larger-scale or more diverse datasets.
- Class Inversion requires additional communication rounds (10 rounds), which could be a bottleneck in bandwidth-constrained scenarios.
- Continual learning scenarios under heterogeneous tasks (e.g., from classification to detection) have not been explored.
- The inference cost of diffusion models is high; a large number of replay samples need to be generated every time a new task is trained.

## Related Work & Insights

- **Textual Inversion / DreamBooth**: The concepts of personalized generation models inspired Class Inversion—searching in the input space of the diffusion model rather than fine-tuning the model itself.
- **FedCIL / Target / MFCL**: Representative works in FCCL that utilize GANs and data-free distillation for replay, though suffering from limited generation quality.
- **Insight**: Pre-trained foundation models (e.g., diffusion models, LLMs) can act as a "knowledge bank" in continual learning, where knowledge can be extracted and replayed via lightweight inversion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — For the first time, a pre-trained diffusion model is introduced into FCCL; the Class Inversion idea is clever and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensively covers two datasets $\times$ four settings (IID/non-IID $\times$ T=5/10), with in-depth and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The problem definition is clear, the methodology is described in detail, and the ablation analysis is insightful.
- **Value**: ⭐⭐⭐⭐ — Offers an absolute improvement of 8-12% on CIFAR-100 and nearly 10% on Tiny-ImageNet, significantly advancing the SOTA of FCCL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ICML 2025\] Autoencoder-Based Hybrid Replay for Class-Incremental Learning](../../ICML2025/image_generation/autoencoder-based_hybrid_replay_for_class-incremental_learning.md)
- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[ECCV 2024\] Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)
- [\[ECCV 2024\] Prompting Future Driven Diffusion Model for Hand Motion Prediction](prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)

</div>

<!-- RELATED:END -->
