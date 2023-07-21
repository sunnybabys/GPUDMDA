import torch

class hyperparams:
    '''rameters'''
    # model
    epoch_num = 1500
    number = 20  #五折交叉验证次数
    learning_rate = 0.001
    col_num = 128
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")