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

### Setup the deepsea-ai module

The deepsea-ai module is used for training.

Add the appropriate AWS credentials to your environment using the aws command line tool.  
To setup a profile specific to this project, e.g. 901103-bio

```
pip install awscli
aws configure --profile 901103-bio
``` 

Setup the AWS accounting by setting up a .ini file with the following contents:
Replacing 548531997526 with the bio AWS account number

```ini
[aws]
sagemaker_arn = arn:aws:iam::548531997526:role/DeepSeaAI
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
deepsea-aio setup --mirror
```
---

## Commands

* bio download --help` - Download data for training an object detection model 
* `bio -h` - Print help message and exit.
  