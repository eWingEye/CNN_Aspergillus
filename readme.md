# CNN_Aspergillus UserGuide

Read our literature from https://link.springer.com/article/10.1007/s12275-021-1013-z

**CNN_Aspergillus** ：
 CNN_ Mold is a very preliminary attempt,  it uses CNN method to identify Aspergillus .
This project consist of a trained Xception model and example pictures captured by dissecting microscopy, which was used to classify various Aspergillus species. (Aspergillus fumigatus A293, Aspergillus. nidulans TN02A7, Aspergillus flavus-AI1; Aspergillus niger-AI1, Aspergillus terreus-AI1, Aspergillus flavus-AI2, and Aspergillus clavatus-AI1) in a study. 
It's a long way from mature software, but our research shows that the trained Xception model exhibited an overall classification accuracy of 98.2% and classification accuracy of each type strain from 92.4% to 100% on a set of raw images. Users who have interests can go through our model using example pictures. More detailed description of the methodology can be found in our publication.
This project contains the data and source code that we used in the paper writing. We describe the use of the code in detail to ensure the verifiability of our article.
You can use all data from this project for free.
- **Anaconda Environment** ：
 Anaconda is a popular tool for Python data science. It manages the packages for python environment.
 Please download from https://www.anaconda.com/products/individual#Downloads and use the very version for your computer. 
- **Tensoflow2** ：Tensorflow2 was used as the engine for CNN training and validation. 
There are multiple changes in TensorFlow2 to make TensorFlow users more productive. TensorFlow 2 removes redundant APIs, makes APIs more consistent (Unified RNNs, Unified Optimizers), and better integrates with the Python runtime with Eager execution.
Many RFCs have explained the changes that have gone into making TensorFlow2. This guide presents a vision for what development in TensorFlow 2 should look like. It's assumed you have some familiarity with TensorFlow 1.x.
Please install TensorFlow2 other than TensorFlow1 for using CNN_Aspergillus .

-------------------


###Download dataset and codes
Use Github to download the source code and data,  and unpack them as a whole.

### Setup the environment
1. Install the Anaconda and ensure it works.
2. From conda prompt(windows)  or system prompt(linux|mac), run the following commands :
```
conda env create -f environment.yml
```
3. When it was finished,  use the commands below to enter the proper conda environment.

For windows.
```
conda activate CNN_Aspergillus 
```
For linux/mac.
```
source activate CNN_Aspergillus 
```
### Train the model
```
python training.py

```
### Validation the trained model
``` 
python evaluate.py

```

### Recognition for a specific Aspergillus  picture
``` 
python  predict.py
```

###
