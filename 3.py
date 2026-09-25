import pandas as pd

train=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
train.LoanAmount=train.LoanAmount.fillna(train.LoanAmount.mean())
df=train.loc[:,['ApplicantIncome','CoapplicantIncome','LoanAmount']]
df['ApplicantIncome']=(df['ApplicantIncome']-df['ApplicantIncome'].min())/(df['ApplicantIncome'].max()-df['ApplicantIncome'].min())
df['CoapplicantIncome']=(df['CoapplicantIncome']-df['CoapplicantIncome'].min())/(df['CoapplicantIncome'].max()-df['CoapplicantIncome'].min())
df['LoanAmount']=(df['LoanAmount']-df['LoanAmount'].min())/(df['LoanAmount'].max()-df['LoanAmount'].min())
mean0=df.mean(axis=0)#axis=0表示一列一列算，每一列计算一个均值
mean1=df.mean(axis=1)
std0=df.std(axis=0)
std1=df.std(axis=1)
print((df['ApplicantIncome']>mean0['ApplicantIncome']+2*std0['ApplicantIncome']).sum())
#回答5：shape是（614，3），能乘，结果是（614，1）