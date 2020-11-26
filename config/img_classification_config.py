import os
from pathlib import Path

class ImageClassificationConfig:
    '''
    ## Experiment Params: These are the experimental params which are
    usually constant for a Project
    '''
    Image_Depth = 3
    Project_name = 'cassava-leaf-disease-classification'
    Unstructured_Data_Path = ''
    Base_Data_Path = '../input/cassava-leaf-disease-classification'
    Path_Curr_Dir = Path(__file__).parent.absolute()
    Path_Parent_Dir = str(Path(Path_Curr_Dir).parents[0])
    Gt_Path = os.path.join(Path_Parent_Dir, 'input', 'cassava-leaf-disease-classification', 'train.csv')
    CheckPoints_Dir = os.path.join(Path_Parent_Dir, 'checkpoints')
    Img_Ext = '.jpg'
    Num_Classes = 5
    CheckPoints_Path = ''
    Mode = 'train'
    Validation_Fraction = 0.2
    Dataset_Shuffle = True
    Cyclic_LR = False
    No_System_Threads = 8
    Device = "cuda"
    Number_GPU = 1
    '''
    ## Hyper Parameters: These are the standard tuning params 
    for an experiment.
    '''
    learning_rate = 0.00008
    batch_size_per_gpu = 2
    batch_size = batch_size_per_gpu * Number_GPU
    optimizer = 'radam'
    epochs = 1
    img_dim = 224
    img_channels = 3
    '''
    ## Callbacks and model-checkpoints parameters
    '''
    early_stopping_patience = 5
    model_chkpoint_period = 2
    one_cycle_lr_policy = False

ConfigObj = ImageClassificationConfig()
