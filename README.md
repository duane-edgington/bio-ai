[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/downloads/)

*bio-ai* is a command line tool to run models and manage data, etc. for the MBARI BioDiversity project.

## Installation 

### Create the Anaconda environment

The fastest way to get started is to use the Anaconda environment.  This will create a conda environment called *bio*.
```shell
git clone http://github.com/mbari-org/bio-ai.git
cd bio-ai
conda env create 
conda activate bio
```

## Commands

* bio download --help` - Download data for training an object detection model 
* `bio -h` - Print help message and exit.
  
## Usage

Setup a .env file with the following contents:

```ini
TATOR_API_HOST=http:/mantis.shore.mbari.org
TATOR_API_TOKEN=15afoobaryouraccesstoken
```

## Data Download

Download data for model training in a format the [deepsea-ai module expects](https://docs.mbari.org/deepsea-ai/data/) with the download command, e.g.

Note - if your leave of the concepts option, the default is to fetch **all** concepts.

```shell
python bio.py download --generator vars-labelbot --version Baseline --concepts "Krill molt, Eusergestes similis"
```

Download data format is saved to a directory with the following structure e.g. for the Baseline version:

```
Baseline
    ├── labels.txt
    ├── images
    │   ├── image1.png
    │   ├── image2.png 
    ├── labels
    │   ├── image1.txt
    │   ├── image2.txt 
```
 
Once data is downloaded, split the data and continue to the [training command](https://docs.mbari.org/deepsea-ai/commands/train/). This requires setting up the AWS account.
This should be done by an AWS administrator if you are not already setup.

### PASCAL VOC data format

If you want to download data also in the PASCAL VOC format, use the optional --voc flag, e.g.

```shell
python bio.py download --generator cluster --version Baseline --concepts "Krill molt, Eusergestes similis" --voc
```

Download data format is saved to a directory with the following structure e.g. for the Baseline version:
```
Baseline
    ├── labels.txt
    ├── voc
    │   ├── image1.xml
    │   ├── image2.xml 
```
 
### COCO data format

Use the optional --coco flag to download data in the [COCO](https://cocodataset.org/#home) format, e.g.

```shell
download --generator vars-annotation --version Baseline --group MERGE_CLASSIFY --base-dir VARSi2MAP --concepts "Atolla" --coco
```

Download data format is saved to a directory with the following structure e.g. for the Baseline version:
```
Baseline
    ├── labels.txt
    ├── coco
    │   └── coco.json
```
### CIFAR data format

Use the optional --cifar flag to download data in the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) format, e.g.

```shell
download --generator vars-annotation --version Baseline --group MERGE_CLASSIFY --base-dir VARSi2MAP --concepts "Atolla" --cifar --voc --cifar-size 128
```

The CIFAR data is saved in a npy file with the following structure, e.g. for the data version Baseline:
```shell 

Baseline
    ├── labels.txt
    ├── cifar
    │   ├── images.npy
    │   └── labels.npy
```

Read the data (and optionally visualize) with the following code:

```python
import numpy as np
import matplotlib.pyplot as plt
images = np.load('Baseline/cifar/images.npy')
labels = np.load('Baseline/cifar/labes.npy')
 
# Visualize a few images from the CIFAR data
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
```
 
![ Image link ](img/atolla_cifar128.png)


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
[docker]
yolov5_container = mbari/deepsea-yolov5:1.1.2
strongsort_container = mbari/strongsort-yolov5:6f35769

[aws]
account_id = 548531997526
sagemaker_arn = arn:aws:iam::548531997526:role/DeepSeaAI
model = s3://deepsea-ai-548531997526-models/yolov5x_mbay_benthic_model.tar.gz
track_config = s3://deepsea-ai-548531997526-track-conf/strong_sort_benthic.yaml
videos = s3://deepsea-ai-548531997526-videos
models = s3://deepsea-ai-548531997526-models
tracks = s3://deepsea-ai-548531997526-tracks

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
