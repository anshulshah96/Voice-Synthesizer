from utils import * 
from numpy import zeros
from sklearn.naive_bayes import GaussianNB

class GaussNB():
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def predict(self, df_cross):
        df_cross.loc[:,self.feature_list] = self.scaler.transform(df_cross[self.feature_list])
        dfg = df_cross.groupby('fname')
        pred_file_pid = pd.DataFrame(columns=('fname', 'id'))

        for fname, group in dfg:
            x_test = group[self.feature_list]
            model = self.model
            output = (model.predict_log_proba(x_test)).transpose()
            plist = [sum(output[i]) for i in range(len(output))]
            pred_file_pid = pred_file_pid.append(pd.DataFrame(data={
                'fname':fname,
                'id':[np.argmax(plist)]
            }))
        return pred_file_pid
    
    def fit(self, df_train):
        self.scaler = preprocessing.StandardScaler().fit(df_train[self.feature_list])
        df_train.loc[:,self.feature_list] = self.scaler.transform(df_train[self.feature_list])
    
        dfg = df_train.groupby('id')
        x_train = df_train.loc[:,self.feature_list]
        y_train = list()
        for name, group in dfg:
            for i in range(len(group[self.feature_list])):
                 y_train.append(name)
        
        gnb = GaussianNB()
        model = gnb.fit(x_train, y_train)
        self.model = model

    def predict_df(self, df_cross):
        pred_file_pid = self.predict(df_cross)

        pred_file_pid.sort_values('fname', inplace=True)

        orig_file_pid = df_cross[['id','fname']].drop_duplicates().sort_values('fname')
        print(metrics.classification_report(orig_file_pid['id'], pred_file_pid['id']))