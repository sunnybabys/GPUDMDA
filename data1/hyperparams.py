import torch

class hyperparams:
    '''rameters'''
    # model
    epoch_num = 300
    number = 20  #五折交叉验证次数
    learning_rate = 0.001
    col_num = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")