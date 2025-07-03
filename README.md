# Code for the ACRO Face evaluation

This is code to train models on facial images to detect presence or absance of some attribute (in this case diagnose acromegaly).

The code is organized 

## Installation

The suggest installation method uses the `pyproject.toml` to manage dependencies, ideally with a tool like astral `uv`. First install it by following [the instructions](https://docs.astral.sh/uv/getting-started/installation/).
One `uv` is installed, go to the project root and run:

```shell
$ uv sync
```

This creates the virtual environment for the project in `.venv`. Initialize it by running:
```shell
$ source .venv/bin/activate
```

Now we add this project as a package to that python environment:
```shell
$ uv pip install -e .
```




## Pretrained models

This code relies on a number of pre-trained models, which will be automatically downloaded into `pretrained_models`.


## Configuration

This code relies heavily on configuration files to adjust its behavior. The configuration files are typically python files which contain instantiations of specific configuration objects. You can find the configuration files in `configs`. 
Noteworthy files are:
- `configs/preprocess/` - Is used by the preprocessing script to determine how to generate images from the videos. This mainly entails detailing what angles the images should be taken from.
 - `configs/datasets/` - Tells the training framework how to load the files which belong to the dataset (e.g. what Dataset class to use). This can be used to specify that different views of the data should be used for training vs. evaluation (e.g. training using 20 video frames, evaluation using 3).
 - `configs/[model]_experiment.py` - Entrypoints for the actual experiments. They detail the experiment process, what dataset config and model config to use, as well as how the dataset splitting and hyper parameter optimization should be performed.
 - `configs/models/` - This directory contain the configuration for different models, this determines what model classes are used by the experiment running framework, and their [hyper]parameters.

## Data preprocessing

Data is assumed to be in video files, with the following naming scheme:

`[Site code][study code]-[A or K]-[video name].mp4`

The side code determines what site the data comes from, while the study code identifies the pair of acromegaly-control. `A` or `K` denotes the label of the image, for Acromegaly (A) or Control (K). The video identifier determines which video of this subject this is, in case multiple videos have been captured, but this is mainly used for the manual data management.

An example could be:

`LU01-A-capture_1.mp4`

Which would be at the site `LU`, of the first acromegaly patient (or corresponding control). In this case, the `A` denotes this as being a video of an acromegaly participant.

### Extracting yaw frames

All source video files should be placed in the same folder, called `raw`, placed in a root dataset folder (e.g. `datasets`). If you put the videos in `datasets/raw/`, then the following will produce the dataset containing the evaluation files (three images of 90, 45 and 0 degrees angle).

```shell
$ python scripts/preprocess_data.py datasets/raw configs/preprocess/three_yaw_dataset.py
```

To produce the training files, run the same command with a different pre-processing config:

```shell
$ python scripts/preprocess_data.py datasets/raw configs/preprocess/9_degree_yaw_dataset.py`
```

These two preprocessing runs has created a number of different directories containing frames from the original videos under `datasets/processed`:
```
masked_yaws_-90_81_20  masked_yaws_0_90_3  yaws  yaws_-90_81_20  yaws_0_90_3
```

These contain partial results from the different steps. The directory you are interested in are the ones prefixed by `masked`, as these will contain the frames from the desired yaws, where clothing has been automatically masked. 

### Preparing images

You might want to do manual inspection of the resulting images, at least ensure that the evaluation images seem correct. By default the images will be masked automatically, but this mask is imperfect either not masking enough of the clothes or masking parts of the face. The masks are in the alpha channel of the image, so can easily be corrected in an image manipulation program. 

After you are happy with the clothing masks, you need to collapse the alpha channel of the images:

```shell
$ python scripts/remove_alpha_channel.py datasets/processed/masked_yaws_0_90_3`
$ python scripts/remove_alpha_channel.py datasets/processed/masked_yaws_-90_81_20`
```

This produces two new directories which are the ones we want to use for training:

```
masked_yaws_0_90_3_no-alpha  masked_yaws_-90_81_20_no-alpha 
```

The number tells us the starting angle and the stop angle, and how many frames there are. In this case we have one folder with 3 frames per video and one with 20. We'll use the one with 3 frames for evaluation, and the one with 20 for training.

## Preparing configurations
Before we start training, we need to make the training framework aware of our data. We do this by setting up config files.

### Datasets
Before we start training, we need to configure the datasets. There is a default dataset config in `configs/datasets/acroface/acroface_ml` which is the recommended one. This assumes there are two directories for the dataset, `dataset/acroface_ml_dataset/training_dataset` and `dataset/acroface_ml_dataset/test_dataset`. You can create these two directories and copy the files you want over to them, but creating symlinks allow you to more clearly denote which result from the preprocessing you use:

```shell
$ mkdir -p dataset/acroface_ml_dataset
$ ln -s  $PWD/datasets/processed/masked_yaws_-90_81_20_no-alpha dataset/acroface_ml_dataset/training_dataset
$ ln -s  $PWD/datasets/processed/masked_yaws_0_90_3_no-alpha dataset/acroface_ml_dataset/test_dataset
```

#### Limiting participants

If you wish to limit which subjects (videos) are part of the datasets, you can add a `subjects.txt` file in the experiment dataset root (e.g. `dataset/acroface_ml_dataset` in the example above). This text file should have one subject code

## Experiments
Now that all data is set up, we can start running our experiments. They are divided according to the pre-trained model we'll use for our classification, with the following ones:

```
densenet_experiment_pretrain_20hp.py  farl_experiment_pretrain_10hp.py  inceptionv3_experiment_pretrain_20hp.py  resnet_experiment_pretrain_20hp.py 
```
The experiments look almost exactly the same, the difference is what model config file to load, and the different number of hyper parameter trials to perform. FaRL is significantly more costly to train, so is alotted fewer number of trials.

Before we run the experiments, we need to set up the dataset splits. This is done beforehand for two main reasons: determinism across model runs (the models will be trained on the same cross validation folds) and checking (easier to inspect the dataset split to assure proper disjoint splits). The dataset split script uses an experiment config to determine how to split the data, since this is the same in all our experiments we can use any one of them:

```shell
$ python scripts/setup_data_splits.py configs/farl_experiment_pretrain_10hp.py
```

This creates a JSON file describing the splits in `configs/dataset_splits/base_splits.json`. If you like, you can inspect this file manually. In this example, it will be a series of nested splits, due to the nested cross validation.

### Running experiments

An experiment will actually be a great number of independent training runs (as determined by the dataset splits). The experiment framework in this project decouples each of these runs from each other by defining each of them as a _work package_. These will automatically be distributed over multiple processes if the experiment config has a list of `multiprocessing_envs` (see below for details). To start an experiment run:


```shell
$ python scripts/run_experiment.py configs/farl_experiment_pretrain_10hp.py
```

This will create the work packages and start processing them, saving all results as files in the folder `experiments/EXPERIMENT_NAME/TIMESTAMP`, an example could be `experiments/farl_experiment_pretrain_10hp/2024-04-17T05.20.14`. Most files are human readable, so you can inspect training results as they are created.

### Running experiments in parallel

The script has rudimentary support for running parallel experiments. This is achieved by starting multiple processes and then make them process different work packages. To configure this, change the `multiprocessing_envs` environmental variable to be a list of dictionaries. Each dictionary lists the environmental variables and their values which a process should be started with, so if there are two dictionaries, two worker processes will be started with their own environments. This is useful to explicitly limit which devices (i.e. GPUs) those processes will see. Here's an example which would make the framework use two CUDA GPUs:

```python
experiment_config = ExperimentConfig(name='farl_experiment_pretrain_10hp', 
                                     dataset_config_path=dataset_config_path,
                                     data_split_path='configs/dataset_splits/base_splits.json',
                                     model_config_path=model_config_path,
                                     resample_strategy=top_resample_strategy,
                                     nested_resample_strategy=nested_resample_strategy,
                                     hpo_config=hpo_config,
                                     multiprocessing_envs=[{'CUDA_VISIBLE_DEVICES': '0'},
                                                           {'CUDA_VISIBLE_DEVICES': '1'},
                                                           #{'CUDA_VISIBLE_DEVICES': '2'},
                                                           #{'CUDA_VISIBLE_DEVICES': '3'},
                                                           #{'CUDA_VISIBLE_DEVICES': '4'},
                                                           #{'CUDA_VISIBLE_DEVICES': '5'},
                                                           #{'CUDA_VISIBLE_DEVICES': '6'},
                                                           #{'CUDA_VISIBLE_DEVICES': '7'},
                                                           ]
)
```

### Aborted experiments
If the experiment run failed for some reason (e.g. power loss), you can continue running an experiment with `scripts/continue_experiment.py`, for example with the experiment config from above:

```shell
$ python scripts/continue_experiment.py configs/farl_experiment_pretrain_10hp.py experiments/farl_experiment_pretrain_10hp/TIMESTAMP
```

This will contiunue processing the incomplete work packages and finnish the experiment.

### Experiment results

The results of experiments are saved to the experiment folder. Since experiments are based on nested cross validation, the experiment results folder structure mimics this. In the root experiment folder there will be a `children` directory which contain the topmost cross-validation folds. These are the level at which the final models will be created, and each folder in that toplevel `children` directory will correspond to one test dataset fold (i.e. all models under `experiments/farl_experiment_pretrain_10hp/TIMESTAMP/children/01` will have the same test data).
Each such folder will in turn have multiple children, corresponding to the nested cross validation (they will be trained with different subsamples of training data vs. development data).

You can automatically extract the final results from these top-level experiments by running:

```shell
$ python analysis/collate_ml_predicitons.py experiments/farl_experiment_pretrain_10hp/TIMESTAMP
```

This by default creates a file under `analysis/annotations/deep_learning_annotations` containing a Comma Seperated Values (CSV) file with subjects, ground truth label and model predictions. This file can then be used as input to the further analysis.
To create bootstrapped predictions run:

```shell
$ python analysis/make_bootstrap_statistics.py analysis/annotations/deep_learning_annotations
```

This creates a new file `analysis/bootstraped_statistics/bootstrapped_statistics.json` which has bootstrapped the 
predictions per subject from all predictors in a annotation file. Based on this we can then summarize the results with

```shell
$ python analysis/make_performance_table.py analysis/bootstraped_statistics/bootstrapped_statistics.json
```

This creates the files `analysis/results_tables/bootstrapped_results_human_readable_ci.xlsx` and `analysis/results_tables/bootstrapped_results_separate_ci.xlsx` which summarizes the results.
