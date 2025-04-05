# necessary imports
import torch
# import os, sys
# sys.path.append('./OpenOOD/')
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

# load the model
net = ResNet18_32x32(num_classes=10)
net.load_state_dict(
    torch.load('./cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
)
net.cuda()
net.eval()


#@title choose an implemented postprocessor
postprocessor_name = "react" #@param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}

# Notes:
# 1) The evaluator will automatically download the required datasets given the
# ID dataset specified by `id_name`

# 2) Passing the `postprocessor_name` will use an implemented postprocessor. To
# use your own postprocessor, just make sure that it inherits the BasePostprocessor
# class (see openood/postprocessors/base_postprocessor.py) and pass it to the
# `postprocessor` argument.

# 3) `config_root` points to the directory with OpenOOD's configurations for the
# postprocessors. By default the evaluator will look for the configs that come
# with the OpenOOD module. If you want to use custom configs, clone the repo locally
# and make modifications to OpenOOD/configs.

# 4) As you will see when executing this cell, during the initialization the evaluator
# will automatically run hyperparameter search on ID/OOD validation data (if applicable).
# If you want to use a postprocessor with specific hyperparams, you need
# to clone the OpenOOD repo (or just download the configs folder in the repo).
# Then a) specify the hyperparams and b) set APS_mode to False in the respective postprocessor
# config.

evaluator = Evaluator(
    net,
    id_name='cifar10',                     # the target ID dataset
    data_root='./data',                    # change if necessary
    config_root=None,                      # see notes above
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    postprocessor=None,                    # if you want to use your own postprocessor
    batch_size=200,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=2)                         # could use more num_workers outside colab


# let's do standard OOD detection
# full-spectrum detection is also available with
# `fsood` being True if you are evaluating on ImageNet

# the returned metrics is a dataframe which includes
# AUROC, AUPR, FPR@95 etc.
metrics = evaluator.eval_ood(fsood=False)

# there is some useful information stored as attributes
# of the evaluator instance

# evaluator.metrics stores all the evaluation results
# evaluator.scores stores OOD scores and ID predictions

# for more details please see OpenOOD/openood/evaluation_api/evaluator.py

print('Components within evaluator.metrics:\t', evaluator.metrics.keys())
print('Components within evaluator.scores:\t', evaluator.scores.keys())
print('')
print('The predicted ID class of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][0][:5])
print('The OOD score of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][1][:5])