# MetaVOS
Official code for the paper titled "Meta Learning Deep Visual Words for Fast Video Object Segmentation"
If you use this code, please cite this [Meta Learning Deep Visual Words for Fast Video Object Segmentation](https://arxiv.org/abs/1812.01397):


	@article{DBLP:journals/corr/abs-1812-01397,
	  author    = {Harkirat Singh Behl and
	               Mohammad Najafi and
	               Philip H. S. Torr},
	  title     = {Meta Learning Deep Visual Words for Fast Video Object Segmentation},
	  journal   = {CoRR},
	  year      = {2018}
	}


## Pre Requisites

### STEP-1 Installing required libraries


### STEP-2 Downloading and preparing the dataset

**DAVIS-17**
Download the DAVIS-17 Train and Val dataset from [link](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip).
After downloading the dataset, extract it within the 'metavos' directory.

**For your own dataset**


### STEP-3 Downloading the trained Model
The trained model for DAVIS-17 can be downloaded from [link](https://unioxfordnexus-my.sharepoint.com/:u:/r/personal/engs1635_ox_ac_uk/Documents/research/segmentation/metavos_data/DAVIS_2017_prototypical_MODES_train_max_109000.pth?csf=1&e=VvEG1G).

After downloading the weights, please put it in the 'snapshots' folder


## TESTING
Run the file 'main.py'

	python main..py


## TRAINING