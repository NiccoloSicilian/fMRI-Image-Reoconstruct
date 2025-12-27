# Brain-to-Vision Reconstruction: RelaxedAsCoal 

**Advanced Machine Learning Project (2025/26)** *Reconstructing visual stimuli from fMRI activity using Semantic Alignment and Diffusion Models.*

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Abstract
This project replicates and simplifies the **MindEye** architecture to reconstruct natural images from fMRI brain activity. Using the **Natural Scenes Dataset (NSD)**, we implemented a pipeline that maps 15,724 fMRI voxels into the 1,280-dimensional **Kandinsky-CLIP** latent space.

Unlike standard regression, we utilize a **Metric Learning approach** (Contrastive + Triplet + Cosine Loss) to learn the true semantic geometry of the latent space. The predicted embeddings are then decoded into images using the pre-trained **Kandinsky 2.1** diffusion model.

##  Authors
* **Niccolò Siciliano** (1958541)
* **Luiz Eduardo Leite Filho** (2219191)
* **Gabriele Gimelli** (1950107)
* **Gabriele Moretti** (1958932)

---

## Architecture

Our pipeline consists of three main stages:

1.  **fMRI Encoder (The "Funnel"):** A hierarchical MLP that progressively distills signal noise.
    * Input: `15,724` voxels
    * Hidden Layers: `12k` $\to$ `10k` $\to$ `8k` $\to$ `4k` $\to$ `2k`
    * Output: `1,280` dimensions (CLIP space)
2.  **Metric Learning Optimization:** We train the encoder using a combined loss function to ensure semantic alignment.
3.  **Generative Decoder:** We use the **Kandinsky 2.1** diffusion model (frozen) to generate images conditioned on the predicted CLIP embeddings.

```mermaid
graph LR
    A[fMRI Voxel Data] -->|MLP Encoder| B(Predicted CLIP Embedding)
    B -->|Conditioning| C{Kandinsky Diffusion}
    C --> D[Reconstructed Image]
