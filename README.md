# Validating neural networks for spectroscopic classification on a universal synthetic dataset
To aid the development of machine learning models for automated spectroscopic data classification, we created a universal synthetic dataset for the validation of their performance.
The dataset mimics the characteristic appearance of experimental measurements from techniques such as X-ray diffraction, nuclear magnetic resonance, and Raman spectroscopy among others.
We applied eight neural network architectures to classify artificial spectra, evaluating their ability to handle common experimental artifacts. 
While all models achieved over 98\% accuracy on the synthetic dataset, misclassifications occurred when spectra had overlapping peaks or intensities. 
We found that non-linear activation functions, specifically ReLU in the fully-connected layers, were crucial for distinguishing between these classes, while adding more sophisticated components, such as residual blocks or normalization layers, provided no performance benefit.
Based on these findings, we summarize key design principles for neural networks in spectroscopic data classification and publicly share all scripts used in this study.

## Concept
Spectroscopic and diffraction signals are visually similar with characteristic intensity peaks when zoomed to matching segment lengths.

![Similarity of spectra and diffraction signals from different techniques](./figures/comparison_spectra_zoom.png)

Correspondingly, a synthetic dataset is formed which incorporates the characteristics of the different signals. The dataset contains multiple unique classes with distinct patterns (number, position and height of peaks). To account for realistic artifacts, the ideal spectra information is varied and multiple samples are generated per class.

![Variants of spectra](./figures/class_variation.png)

Then, multiple established neural network architectures are trained on the synthetic spectra and their performance and classification behavior is evaluated in detail.

## Usage

Clone the repository and install the required Python packages as defined in the requirements.txt file

```
git clone https://github.com/jschuetzke/synthetic-spectra-benchmark
cd synthetic-spectra-benchmark
pip install -r requirement.txt
```

## Organization of the Repo

Training data with synthetic spectra is available here:
https://figshare.com/articles/dataset/Synthetic_spectra_training_data/21581619

Test data for benchmarking:
https://figshare.com/articles/dataset/Synthetic_spectra_test_data/21601749

Documentation of the training runs can be found here:
https://wandb.ai/jschuetzke/synthetic-benchmark/overview
