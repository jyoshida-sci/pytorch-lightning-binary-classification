## About this codes

This script is to train a image classifier using CNN.

J. Yoshida et al.,
CNN-based event classification of alpha-decay events in nuclear emulsion
DOI: https://doi.org/10.1016/j.nima.2020.164930


## Dependency

torch==1.5.1
torchvision==0.6.1
pytorch-lightning==0.8.5
tensorboard

imbalanced-dataset-sampler
pytorch-randaugment

See also the Dockerfile and requirements.txt



## Usage
python classification_fine_tuning_binary_classification.py --root_data_path  ~/image_dataset_alpha_or_not  --gpu 0  --out_dir_name   --rand_augment_t 4  --rand_augment_n 2  --rand_augment_m  6  --batch_size 96



Run this command in output directory to monitor or manage the results.
The port number is arbitrary.
$tensorboard --logdir=. --bind_all --port 6006



## References

* https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags
* https://pytorch.org/get-started/previous-versions/
* https://github.com/PyTorchLightning/pytorch-lightning
* https://github.com/ufoym/imbalanced-dataset-sampler
* https://github.com/ildoonet/pytorch-randaugment
