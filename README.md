# AI-Driven Background Segmentation for 3D Plant Phenotyping

---

## Paper and Citation

```
To be updated after publication.
```

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [The Preprocessing Pipeline](#the-preprocessing-pipeline)
- [Final Trained Model](#final-trained-model)
- [Source Code Usage](#source-code-usage)
  - [Running Preprocessing](#running-preprocessing)
  - [Running Inference](#running-inference)
  - [Training or Retraining the Model](#training-or-retraining-the-model)
- [Example Outputs](#example-outputs)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Project Overview

Accurate background segmentation in 3D plant phenotyping is critical for extracting reliable plant traits.  
Traditional methods like height thresholding often lead to data loss, especially for small or prostrate plants.

In this project, we propose a simple, yet powerful AI-driven approach using a **Multi-Layer Perceptron (MLP)** model trained on RGB, XYZ (spatial), and NIR (near-infrared) features.  
The model achieved:

- **99.93%** classification accuracy
- Significant improvement in **leaf area (LA) estimation**
- High **generalization capability** to external datasets

---

## Repository Structure

```bash
MLP-3D-PlantSeg/
│
├── README.md
├── LICENSE.md
│
├── data/
│   ├── raw/
│   ├── preprocessed/
│   ├── trained_model/
│   └── results/
│
├── code/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── inference.py
│   └── utils.py
│
├── figures/
│   ├── overview_pipeline.png
│   ├── model_architecture.png
│   ├── results_la_mape.png
│   ├── results_la_r2.png
│   ├── example_segmentation.png
│   ├── daily_pointcounts.png
│   ├── paris_lille_result.png
│   └── README_figures.md
│
└── requirements.txt
```

---

## The Preprocessing Pipeline

Our preprocessing transforms raw 3D scanner data into ready-to-use input for model training.

**Steps:**
1. **Rotation** to align plant surfaces.
2. **Merging** two scanner views into a unified point cloud.
3. **Voxelization** to standardize point density.
4. **Color Smoothing** to reduce noise in RGB and NIR channels.

Each point cloud file after preprocessing contains:
- X, Y, Z coordinates
- RGB color values
- NIR reflectance values

**Workflow Overview:**  
![Overview Pipeline](figures/overview_pipeline.png)

This standardization significantly improves model performance and reduces noise.

---

## Final Trained Model

The final background segmentation model is a **Multi-Layer Perceptron (MLP)** with the following configuration:

- **Input:** 7 features (RGB + XYZ + NIR)
- **Hidden layers:** 3 layers (10 - 50 - 50 neurons)
- **Activation:** ReLU
- **Output:** 1 neuron (sigmoid activation)

The model was optimized using **Bayesian Optimization** and trained with early stopping to avoid overfitting.

**Model Architecture:**  
![Model Architecture](figures/model_architecture.png)

The final model achieved:
- **99.93% classification accuracy**
- Very low false positives and false negatives

---

## Source Code Usage

> ⚠️ **Note:** `.py` source code files are currently placeholders.  
> The full code will be uploaded after paper acceptance and publication.

Despite this, the project structure and intended workflow are ready.

---

### Running Preprocessing

Preprocess raw point cloud data:

```bash
python code/preprocessing.py --input_folder data/raw --output_folder data/preprocessed
```

This script will perform:
- Rotation
- Merging
- Voxelization
- Color smoothing

Preprocessed files are saved into `data/preprocessed/`.

---

### Running Inference

Run background segmentation inference on preprocessed data:

```bash
python code/inference.py --model_path data/trained_model/final_mlp_model.h5 --input_folder data/preprocessed --output_folder data/results
```

The output will be point cloud files with plant/background labels.

---

### Training or Retraining the Model

Train the MLP model from scratch or fine-tune it:

```bash
python code/train_model.py --data_folder data/preprocessed --save_model_to data/trained_model
```

Model hyperparameters can be tuned using [Keras Tuner Documentation](https://keras.io/keras_tuner/).

Recommended configurations:
- 3 hidden layers
- ReLU activations
- Adam optimizer

---

## Example Outputs

### 🔹 Leaf Area Estimation Results


<p align="center">
  <img src="figures/results_la_mape.png" width="45%" />
  <img src="figures/results_la_r2.png" width="45%" />
</p>

<p align="center">
  <img src="figures/example_segmentation.png" width="90%" />
</p>
---

### 🔹 Example Segmentation Comparison

Comparison between coordinate-based and AI-based segmentation:

![Example Segmentation](figures/example_segmentation_1.png)
![](figures/example_segmentation_2.png)
---

 
---

### 🔹 Generalization on Paris-Lille Dataset

Testing model generalization to non-agricultural 3D point clouds:

![Paris-Lille Result](figures/paris_lille_result.png)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).

---

## Acknowledgements

- CZU Prague
- ICRISAT
- Phenospex
- Segments.ai

---

## Contact

- **Serkan Kartal** – [Your Email]
- **Jan Masner** – masner@pef.czu.cz
- **Jana Kholová** – kholova@pef.czu.cz

---
_"Enhancing 3D plant phenotyping through efficient and robust AI-based background segmentation."_
