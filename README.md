Point Cloud Classification and Segmentation
========================
**Name: Omkar Chittar**  
**UID - 119193556**
------------------------
```
PointCloud_Classification_and_Segmentation
+-checkpoints
+-data
+-logs
+-output
+-output_cls
+-output_cls_numpoints
+-output_cls_rotated
+-output_seg
+-output_seg_numpoints
+-output_seg_rotated
+-README.md
+-report
data_loader.py
eval_cls_numpoints.py
eval_cls_rotated.py
eval_cls.py
eval_seg_numpoints.yml
eval_seg_rotated.py
eval_seg.py
models.py
train.py
utils.py
```

# **Installation**

- Download and extract the files.
- Make sure you meet all the requirements given on: https://github.com/848f-3DVision/assignment2/tree/main
- Or reinstall the necessary stuff using 'environment.yml':
```bash
conda env create -f environment.yml
conda activate pytorch3d-env
```
## Data Preparation
Download zip file (~2GB) from https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing. Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.
- The **data** folder consists of all the data necessary for the code.
- There are 6 output folders: 
    1. **output_cls** folder has all the images/gifs generated after running ```eval_cls.py```.
    2. **output_seg** folder has all the images/gifs generated after running ```eval_seg.py```.
    3. **output_cls_numpoints** folder has all the images/gifs generated after running ```eval_cls_numpoints.py```.
    4. **output_seg_numpoints** folder has all the images/gifs generated after running ```eval_seg_numpoints.py```.
    5. **output_cls_rotated** folder has all the images/gifs generated after running ```eval_cls_rotated.py```.
    6. **output_seg_rotated** folder has all the images/gifs generated after running ```eval_seg_rotated.py```.
- All the necessary instructions for running the code are given in **README.md**.
- The folder **report** has the html file that leads to the webpage.


# **1. Classification Model**
- After making changes to:
    1. `models.py` 
    2. `train.py`
    3. `eval_cls.py`

Run the code:  
```bash
python train.py --task cls
```
The code trains the model for the classification task. 

Evaluate the trained model by running the code:
```bash
python eval_cls.py
```
Evaluates the model for the classification task by rendering point clouds named with their ground truth class and their respective predicted class. Displays the accuracy of the trained model in the terminal. The rendered point clouds are saved in the **output_cls** folder.


# **2. Segmentation Model**
- After making changes to:
    1. `models.py` 
    2. `train.py`
    3. `eval_seg.py`

Run the code:  
```bash
python train.py --task seg
```
The code trains the model for the Segmentation task. 

Evaluate the trained model by running the code:
```bash
python eval_seg.py
```
Evaluates the model for the Segmentation task by rendering point clouds with segmented areas with different colors. The rendered point clouds are saved in the **output_seg** folder. Displays the accuracy of the trained model in the terminal.


# **3. Robustness Analysis**
## **3.1. Rotating the point clouds** 
Here we try to evaluate the accuracy of the classification as well as the segmentation models by rotating the point clouds around any one axis (x/y/z) or their permutations.

### 3.1.1. Classification
Run the code:
```bash
python eval_cls_rotated.py
```
Evaluates the model with rotated inputs for the classification task by rendering point clouds named with their rotated angle, ground truth class and their respective predicted class. Displays the accuracy of the trained model in the terminal. The rendered point clouds are saved in the **output_cls_rotated** folder.

### 3.1.2. Segmentation
Run the code:
```bash
python eval_seg_rotated.py
```
Evaluates the model with rotated inputs for the segmentation task by rendering point clouds named with their rotated angle, and prediction accuracy. Displays the accuracy of the trained model in the terminal. The rendered point clouds are saved in the **output_seg_rotated** folder.


## **3.2. Varying the sampled points in the point clouds** 
Here we try to evaluate the accuracy of the classification as well as the segmentation models by varying the number of sampled points in the point clouds.

### 3.2.1. Classification
Run the code:
```bash
python eval_cls_numpoints.py
```
Evaluates the model with varying number of sampled points inputs for the classification task by rendering point clouds named with their index, number of points, ground truth class and their respective predicted class. Displays the accuracy of the trained model in the terminal. The rendered point clouds are saved in the **output_cls_numpoints** folder.

### 3.2.2. Segmentation
Run the code:
```bash
python eval_seg_numpoints.py
```
Evaluates the model with varying number of sampled points inputs for the segmentation task by rendering point clouds named with their index, number of points, and their respective predicted class accuracy. Displays the accuracy of the trained model in the terminal. The rendered point clouds are saved in the **output_seg_numpoints** folder.


# **4. Webpage**
The html code for the webpage is stored in the *report* folder along with the images/gifs.
Clicking on the *webpage.md.html* file will take you directly to the webpage.




