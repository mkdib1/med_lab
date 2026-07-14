# 3D Brain stroke segmentation using U-NET
Starting from ATLAS R3.0 dataset (https://docs.google.com/forms/d/e/1FAIpQLSclH8padHr9zwdQVx9YY_yeM_4OqD1OQFvYcYpAQKaqC6Vscg/viewform), the aim of the project is to build a neural network model for brain stroke segmentation encompassing the following steps:
- preprocess brain volumes in order to face data variability and outliers handling
- build a simple U-NET model from scratch using `monai` libraries
- applying data augmentation to train data, with a random search for hyperparameter tuning
- retrain a new model with tuned-augmented data and making model comparison


## 1. Structure
```
|- notebook/
|  |- main.ipynb/  # main python file to perform eda and visualize the results
|- output/  # intermediate results
|  |- best_augm_param.json  # best augmentation parameter configuration after random search
|  |- shapes.csv  # table with axes dimension statistics of all the brain volumes
|  |- voxel_stats.csv  # table with voxels dimension statistics of all the brain volumes
|- utils/  # helpers .py dir
|  |- __init__.py
|  |- augmentation.py # hyperparameter tuning for data augmentation
|  |- models.py  # u-net model architecture and training loop def
|  |- utils.py helper functions 
|- references/  # papers downloaded from google scholar					
|  |- gomes2013.pdf   
|  |- ...		
|- results/  
|  |- simple_model.pth  
|  |- history_simple_model.pth  
|  |- augm_model.pth  
|  |- history_augm_model.pth  
|- .gitignore
|- .dslab_presentation.pptx # power point presentation of this project  				
|- requirements.txt  # packages needed to lunch .ipynb and .py files   				
|  |- README.md 				# this file
```

## 2. Clone the repository
```sh
git clone https://github.com/mkdib1/med_lab.git
cd med_lab
```

## 3. Create the virtual env and install packages
After downloading the folder, please create your virtual env using `python=3.12` and run:
```sh
pip install -r requirements.txt
```

