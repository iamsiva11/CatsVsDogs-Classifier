#Cat Vs Dogs Classification

[Dogs vs. Cats Kaggle challenge](https://www.kaggle.com/c/dogs-vs-cats), which was held from September 25, 2013 to February 1st, 2014.

The task is straightforward: given an image, determine if it contains a cat or a dog.This is easy for humans, dogs, and cats. But our computer will find it a bit more difficult.

We’ll be  training a small network from Scratch (as a baseline)

We’ll be making use of the following methods in Keras

* fit_generator for training Keras a model using Python data generators
* ImageDataGenerator for real-time data augmentation


###Up and Running


Training time : 70*50 = 3500/60 -> Nearly 1 hour on a CPU Machine

>Pertained model file:

catsdogs-baseline.h5

>Run

python cat-test.py


<Train keras screenshot image>

Dataset : only 2000 training examples (1000 per class)

Data Folder: Training and Validation



A training data directory and validation data directory containing one subdirectory per image class, filled with .png or .jpg images:

<Data folder image>


We will use two sets of pictures, which we got from Kaggle: 1000 cats and 1000 dogs (although the original dataset had 12,500 cats and 12,500 dogs, we just took the first 1000 images for each class). We also use 400 additional samples from each class as validation data, to evaluate our models.

We can now use these generators to train our model. Each epoch takes 20-30s on GPU and 300-400s on CPU. So it's definitely viable to run this model on CPU if you aren't in a hurry.

This approach gets us to a validation accuracy of 0.79-0.81 after 50 epochs (a number that was picked arbitrarily --because the model is small and uses aggressive dropout, it does not seem to be overfitting too much by that point).






