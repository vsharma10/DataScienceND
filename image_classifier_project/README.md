## Developing an Image Classifier with Deep Learning

# Part 1: Image classifier implemented with PyTorch

In this first part of the project, an image classifier is implemented with PyTorch. A workspace with a GPU was provided for working on this project.

# Part 2 - Building the command line application

The deep neural network built and trained on the flower data set, was converted into an application that others can use. The application consists of a pair of Python scripts that run from the command line. For testing, the checkpoint saved in the first part was used.

* Specifications:

The project submission includes two files train.py and predict.py. The first file, train.py, trains a new network on a dataset and saves the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. A file just for functions, classes, and utility relating to the model and loading data and preprocessing images is also created.

    Train a new network on a data set with train.py
        Basic usage: python train.py data_directory
        Prints out training loss, validation loss, and validation accuracy as the network trains
        Options:
            Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
            Choose architecture: python train.py data_dir --arch "vgg13"
            Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
            Use GPU for training: python train.py data_dir --gpu

    Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
        Basic usage: python predict.py /path/to/image checkpoint
        Options:
            Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu
