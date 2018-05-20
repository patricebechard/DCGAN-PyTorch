# DCGAN implementation in PyTorch

We use a DCGAN on the CelebA dataset.

### Usage

After cloning the repo on your local machine, first run the `download.py` file to download and preprocess the data using the following command :

```
python download.py
```

You can easily change where the dataset and the results are saved in the `constants.py` file.

To train the model, run the `train.py` file using the following command:

```
python train.py
```

You can change the hyperparameters for the training and the model in the `params.json` file.

Here are some samples obtained using the trained model :

![Samples](https://raw.githubusercontent.com/patricebechard/DCGAN-PyTorch/master/results/samples.png)

**You can download a pretrained model using the deconvolutional generator [here](https://drive.google.com/file/d/1OjKEHRPYyk8pVCGgiM6SIpashBm4OK83/view?usp=sharing)**



### TODO

* Provide interactive notebook to show results