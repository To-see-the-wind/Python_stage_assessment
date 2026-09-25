import pandas as pd

train=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
train.LoanAmount=train.LoanAmount.fillna(train.LoanAmount.mean())
X=train[['ApplicantIncome','CoapplicantIncome','LoanAmount']].to_numpy()
X_norm=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
mean0=X_norm.mean(axis=0)#axis=0表示一列一列算，每一列计算一个均值
std0=X_norm.std(axis=0)
print((X_norm[:,0]>mean0[0]+2*std0[0]).sum())
#回答5：shape是（614，3），能乘，结果是（614，1）