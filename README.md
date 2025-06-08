# MSGEN: Hierarchical Multi-Scale Molecular Conformer Generation with Structural Awareness

The official implementation of MSGEN framework, which can incorporate multi-scale structural priors to ensure chemically valid 3D geometries.

<p align="center">
  <img src="Picture/Method.png" alt="Illustration of the MSGEN framework with three stages." width="600">
</p>

## Environments

### Install via Conda (Recommended)
```bash
# Clone the environment
conda env create -f env.yml
# Activate the environment
conda activate MSGEN
> ⚠️ **Note:** This environment file is configured for the default GeoDiff backbone.
> If you plan to use a different backbone model (e.g., ET-Flow), please switch to the corresponding environment file in their github.
```

## Dataset

### Offical Dataset
The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Preprocessed dataset
You can also use the prepocessed datasets (GEOM) from GeoDiff [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). After downleading the dataset, it should be put into the folder path as specified in the `dataset` variable of config files `./configs/*.yml`.


### Prepare your own GEOM dataset from scratch (optional)

You can also download origianl GEOM full dataset and prepare your own data split. A guide is available at previous work ConfGF's [[github page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).