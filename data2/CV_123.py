import numpy as np
import pandas as pd
from data_input import  Neg_DataLoader, Non_Neg_DataLoader
from net import transNet
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from hyperparams import hyperparams as params

class CV():
    def __init__(self, cv, n_repeats, inc_matrix) -> None:
        self.cv = cv
        self.inc_matrix = inc_matrix
        self.n_repeats = n_repeats
        self.i, self.j = inc_matrix.shape
        self.trains, self.tests = [], []

    def bance(self, index, type="train"):
        if self.cv == 1:  # 行
            t = self.inc_matrix.loc[index]  # 取出inc_matrix中对应index的行（即训练集所对应的行，在相互作用矩阵中）
        elif self.cv == 2:  # 列
            t = self.inc_matrix.loc[:, index]
        elif self.cv == 3:  # hl
            inc = self.inc_matrix.stack().reset_index()
            inc = inc.loc[index]

        if self.cv == 1 or self.cv == 2:
            inc = t.stack().reset_index()
        # 分别获取值为1，0的索引

        if(type=="train"):

            s1 = inc[inc.loc[:, 0].values == 1].index
            s0 = inc[inc.loc[:, 0].values != 1].index


            s1 = np.vstack((inc.loc[s1, 'level_0'].values, (inc.loc[s1, 'level_1'].values))).T
            s0 = np.vstack((inc.loc[s0, 'level_0'].values, (inc.loc[s0, 'level_1'].values))).T
            s = np.vstack((s1, s0))

        if (type == "test"):
            s1 = inc[inc.loc[:, 0].values == 1].index
            s0 = inc[inc.loc[:, 0].values == 0].index


            s1 = np.vstack((inc.loc[s1, 'level_0'].values, (inc.loc[s1, 'level_1'].values))).T
            s0 = np.vstack((inc.loc[s0, 'level_0'].values, (inc.loc[s0, 'level_1'].values))).T
            s = np.vstack((s1, s0))

        # print(len(s1),len(s0),len(s))
        return s  # 返回[[行号，列号]]的二维ndarry

    def cv_1(self):
        # 行
        print('cv1 {}行'.format(self.i))
        lens = self.i
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):  # 将列表[0,1...lens-1]分割为5分，即按行索引分5份
            self.trains.append(self.bance(train_index,"train"))
            self.tests.append(self.bance(test_index,"test"))
        return self.trains, self.tests

    def cv_2(self):
        # 列
        print('cv2 {}列'.format(self.j))
        lens = self.j
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index,"train"))
            self.tests.append(self.bance(test_index,"test"))
        return self.trains, self.tests

    def cv_3(self):
        print('cv3')
        lens = self.i * self.j
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index,"train"))
            self.tests.append(self.bance(test_index,"test"))
        return self.trains, self.tests


    @classmethod
    def get_cv(cls, cv, n_repeats, inc_matrix):

        if cv == 1:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_1()  # 初始化CV类，即上面的class CV()
        elif cv == 2:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_2()
        elif cv == 3:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_3()

        return trains, tests




def get_data(data,index):
    dataset = []
    for i in range(index.shape[0]):
            dataset.append(np.hstack((data.disease_feature[index[i,0]], data.microbe_feature[index[i,1]], data.interaction.iloc[index[i,0],index[i,1]])))
    reslut = pd.DataFrame(dataset).values
    return  reslut


if __name__ == '__main__':
    data =  Neg_DataLoader("./trans_data/data2")
    dataset = data.interaction
    dataset.columns=list(range(data.interaction.shape[1]))
    dataset.index=list(range(data.interaction.shape[0]))
    n_acc = []
    n_precision = []
    n_recall = []
    n_f1 = []
    n_AUC = []
    n_AUPR = []
    trains, tests = CV.get_cv(cv=3,n_repeats= params.number ,inc_matrix=dataset)
    for i in range(len(trains)):
        print(len(trains), trains[i].shape, tests[i].shape)
        train = get_data(data,trains[i])
        test = get_data(data,tests[i])
        feature_train = train[:,0:-1]
        target_train = train[:,-1].reshape(-1)
        feature_test = test[:,0:-1]
        target_test = test[:,-1].reshape(-1)
        print('begin training:')
        model = transNet( params.col_num, 100, 1).to(params.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        loss_fn = torch.nn.MSELoss().to(params.device)
        for epoch in range(params.epoch_num):
            model.train()
            model.type = "train"
            feature_train = torch.FloatTensor(feature_train)
            target_train = torch.FloatTensor(target_train)
            train_x = feature_train.to(params.device)
            train_y = target_train.to(params.device)
            pred = model(train_x)
            loss = loss_fn(pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(loss.item())

        model.eval()
        model.type = "test"
        feature_test = torch.FloatTensor(feature_test)
        target_test = torch.LongTensor(target_test)
        test_x = feature_test.to(params.device)
        test_y = target_test.to(params.device)
        pred = model(test_x)
        pred = pred.cuda().data.cpu().numpy()
        KT_y_prob_1 = np.arange(0, dtype=float)
        for i in pred:
            KT_y_prob_1 = np.append(KT_y_prob_1, i)
        light_y = []
        for i in KT_y_prob_1:  # 0 1
            if i > 0.5:
                light_y.append(1)
            else:
                light_y.append(0)
        n_acc.append(accuracy_score(target_test, light_y))
        n_precision.append(precision_score(target_test, light_y))
        n_recall.append(recall_score(target_test, light_y))
        n_f1.append(f1_score(target_test, light_y))

        fpr, tpr, thresholds = roc_curve(target_test, KT_y_prob_1)
        prec, rec, thr = precision_recall_curve(target_test, KT_y_prob_1)
        n_AUC.append(auc(fpr, tpr))
        n_AUPR.append(auc(rec, prec))

        print('--------------------------------------结果---------------------------------------------')
        print("accuracy:%.4f" % accuracy_score(target_test, light_y))
        print("precision:%.4f" % precision_score(target_test, light_y))
        print("recall:%.4f" % recall_score(target_test, light_y))
        print("F1 score:%.4f" % f1_score(target_test, light_y))
        print("AUC:%.4f" % auc(fpr, tpr))
        print("AUPR:%.4f" % auc(rec, prec))



        f = open("./result/data2/cv3_auc.csv", mode="a")
        for j in range(len(fpr)):
            f.write(str(fpr[j]))
            f.write(",")
            f.write(str(tpr[j]))
            f.write(",")
            f.write(str(auc(fpr, tpr)))
            f.write("\n")
        f.write("END__{}".format(i))
        f.write("\n")
        f.write("\n")
        f.close()

        f = open("./result/data2/cv3_aupr.csv", mode="a")
        for j in range(len(prec)):
            f.write(str(rec[j]))
            f.write(",")
            f.write(str(prec[j]))
            f.write(",")
            f.write(str(auc(rec, prec)))
            f.write("\n")
        f.write("END__{}".format(i))
        f.write("\n")
        f.write("\n")
        f.close()

mean_acc = np.mean(n_acc)
mean_precision = np.mean(n_precision)
mean_recall = np.mean(n_recall)
mean_f1 = np.mean(n_f1)
mean_AUC = np.mean(n_AUC)
mean_AUPR = np.mean(n_AUPR)

std_acc = np.std(n_acc)
std_precision = np.std(n_precision)
std_recall = np.std(n_recall)
std_f1 = np.std(n_f1)
std_AUC = np.std(n_AUC)
std_AUPR = np.std(n_AUPR)

print('--------------------------------------平均结果---------------------------------------------')
print("accuracy:%.4f" % mean_acc)
print("precision:%.4f" % mean_precision)
print("recall:%.4f" % mean_recall)
print("F1 score:%.4f" % mean_f1)
print("AUC:%.4f" % mean_AUC)
print("AUPR:%.4f" % mean_AUPR)

print('--------------------------------------平均std---------------------------------------------')
print("accuracy:%.4f" % std_acc)
print("precision:%.4f" % std_precision)
print("recall:%.4f" % std_recall)
print("F1 score:%.4f" % std_f1)
print("AUC:%.4f" % std_AUC)
print("AUPR:%.4f" % std_AUPR)