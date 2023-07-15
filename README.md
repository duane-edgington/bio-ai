![license-GPL](https://img.shields.io/badge/license-GPL-blue)

# About

*biodiversity-model-trainer*  a command line tool for downloading and training machine learning models for 
the MBARI biodiversity project.

## Installation 

### Create the Anaconda environment

The fastest way to get started is to use the Anaconda environment.  This will create a conda environment called *bio*.
```shell
git clone http://github.com/mbari-org/biodiversity-model-trainer.git
cd biodiversity-model-trainer
conda env create 
conda activate bio
```

## Commands

* bio download --help` - Download data for training an object detection model 
* `bio -h` - Print help message and exit.
  
## Usage

Setup a .env file with the following contents:

```ini
TATOR_API_HOST=http://DiGiTS-Dev-Box-Fish.shore.mbari.org
TATOR_API_TOKEN=15afoobaryourmediatoken
```

## Data Download

Download data for model training in a format the [deepsea-ai module expects](https://docs.mbari.org/deepsea-ai/data/) with the download command, e.g.

Note - if your leave of the concepts option, the default is to fetch **all** concepts.

```shell
python bio.py download --generator vars-labelbot --version Baseline --concepts "Krill molt, Eusergestes similis"
```

Download data format is saved to a directory with the following structure e.g. for the Baseline version:

```
── Baseline
│   │   ├── images
│   │   │   └── image1.jpg
│   │   │   └── image2.jpg
│   │   ├── labels
│   │   │   └── image1.txt
│   │   │   └── image2.txt
labels.txt 
```
 
Once data is downloaded, split the data and continue to the [training command](https://docs.mbari.org/deepsea-ai/commands/train/). This requires setting up the AWS account.
This should be done by an AWS administrator if you are not already setup.

### Options

| Option | Description                                                                                                              |
| --- |--------------------------------------------------------------------------------------------------------------------------|
 | --help | Show the help message and exit.                                                                                          |
| --base-dir | The base directory to save the downloaded data.  The default is the current directory.                                   |
| --generator| The name of the generator to download data from.                                                                         |
| --version| The version of the generator to download data from. Default is Baseline.                                                 |
| --concepts| A comma separated list of concepts to download data for.  This is the name of the concept, e.g. Krill molt, Nanomia.     |
| --cifar| Download data in the CIFAR format.  The default is to download data in the format to train a YOLO object detection model |
| --group| The name of the group to download data for.  The default is to download data for all groups.                             |

### CIFAR data format

Use the optional --cifar flag to download data in the[ [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) format, e.g.

The CIFAR data is saved in a npy file with the following structure, e.g. for the data version Baseline:
```shell 

── Baseline
|   |   |-- labels.txt
│   │   ├── cifar
│   │   │   └── labels.npy 
│   │   │   └── images.npy
```

Read the data with the following code:

```python
import numpy as np
images = np.load('images.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)
```


## Object Detection Training

Training an object detection model requires the [deepsea-ai](https://github.com/mbari-org/deepsea-ai) module
 

### Setup the deepsea-ai module

The deepsea-ai module uses AWS for training and inference.  Add the appropriate AWS credentials to your environment using the aws command line tool, e.g.
to setup a profile specific to this project, e.g. 901103-bio

```
pip install awscli
aws configure --profile 901103-bio
``` 

Setup AWS accounting by setting up a .ini file with the following contents:
975513124282 is the "901103 Biodiversity and Biooptics" MBARI project AWS account number

e.g. ~/.aws/bio.ini
```ini
[aws]
sagemaker_arn = arn:aws:iam::975513124282:role/DeepSeaAI
yolov5_ecr = mbari/deepsea-yolov5:1.1.2
deepsort_ecr = mbari/deepsort-yolov5:1.3.5
strongsort_ecr = mbari/strongsort-yolov5:1.5.0
yolov5_model_s3 = s3://902005-public/models/yolov5x_mbay_benthic_model.tar.gz
deepsort_track_config_s3 = s3://902005-public/models/deep_sort_benthic.yaml
strongsort_track_config_s3 = s3://902005-public/models/strong_sort_benthic.yaml
[database]
site = http://deepsea-ai.shore.mbari.org
gql = %(site)s/graphql
[tags]
organization = mbari
project_number = 901103
stage = prod 
application = detection
```

Then run the setup command.  This will setup the appropriate AWS permissions and mirror the images used in the commands


```shell
deepsea-ai setup --mirror --config ~/.aws/bio.ini
```
---

### Train a model

Now that the data is downloaded, you can train a model.  Train a model by first splitting the data first, e.g.

**Note** this will randomly split 85% of the data for training, 10% for validation and 5% as a hold out for testing.

```shell
deepsea-ai split --input Baseline --output BaselineSplit
```

Then train the model

```shell
deepsea-ai train --images BaselineSplit/images.tar.gz  --labels BaselineSplit/labels.tar.gz --model yolov5x --epochs 50 --labels labels.txt --instance-type ml.p3.16xlarge  --batch-size 32 --input-s3 901103-bio-data --output-s3 901103-bio-ckpt
```
