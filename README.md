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
> ⚠️ **Note:** This environment file is configured for the default backbone.
# > If you plan to use a different backbone model (e.g., ET-Flow), please switch to the corresponding environment file in their github.
```

## Dataset
<!-- 
### Offical Dataset
The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).
### Preprocessed dataset
-->


We use the prepocessed datasets (GEOM) [[zenedo driver]](https://zenodo.org/records/15627485). After downleading the dataset, it should be put into the folder path as specified in the `dataset` variable of config files `./configs/*.yml`.

<!-- 
### Prepare your own GEOM dataset from scratch (optional)

You can also download origianl GEOM full dataset and prepare your own data split. A guide is available at previous work ConfGF's [[github page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).
-->

## Training

The hyper-parameters and training detail for stage 1 and stage 2 are provided in config files(`./configs/*.yml`), and feel free to tune these parameters.

You can train the model with the following commands:

```bash
# Default settings for stage 1
python train_s1.py ./config/qm9_1.yml
python train_s1.py ./config/drugs_1.yml
# Default setting for stage 2 
python train_s2.py ./config/qm9_2.yml
python train_s2.py ./config/drugs_2.yml
```

## Generation
We provide a 2-stage GeoDiff model designed using the MSGEN framework, i.e., `drugs_1` and `drugs_2`. Note that, please put the checkpoints `*.pt` into paths like `${log}/${model}/checkpoints/`, and also put corresponding configuration file `*.yml` into the upper level directory `${log}/${model}/`.

You can generate conformations for entire or part of test sets by:
```bash
python test.py ${log}/${model}/checkpoints/${iter_stage1}.pt ${log}/${model}/checkpoints/${iter_stage2}.pt \
    --start_idx 800 --end_idx 1000
```
All hyper-parameters related to sampling can be set in `test.py` files.

## Evaluation

After generating conformations following the obove commands, the results of all benchmark tasks can be calculated based on the generated data.

### Task 1. Conformer Generation

The `COV` and `MAT` scores on the GEOM datasets can be calculated using the following commands:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```

### Task 2. Property Prediction

For the property prediction, we use the split provided in the [[zenedo driver]](https://zenodo.org/records/15627485).
```bash
python test.py ${log}/${model}/checkpoints/${iter_stage1}.pt ${log}/${model}/checkpoints/${iter_stage2}.pt \       --num_confs 50 --start_idx 0 --test_set data/GEOM/QM9/qm9_property.pkl

python eval_prop.py --generated ${log}/${model}/${sample}/sample_all.pkl
```
