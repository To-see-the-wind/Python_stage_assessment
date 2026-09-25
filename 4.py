import pandas as pd

df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
print(df.info())#主要问题：多列有数据缺失

print(df.Credit_History.value_counts())#发现绝大多数no，因此统一fillna写no
df.Self_Employed=df.Self_Employed.fillna('no')
print(df.LoanAmount.value_counts())#发现贷款金额差异较大，参差不齐，并且可能对结果影响较大，因此全部删除
df=df.dropna(subset=['LoanAmount'])
print(df.Credit_History.value_counts())#信用达标对结果影响较大，但是缺失数据较多，因此填写unknown
df.Credit_History=df.Credit_History.fillna('Unknown')

df['Loan_Status']=df['Loan_Status'].replace({'Y':1,'N':0})
print(df.groupby(['Gender']).Loan_Status.mean())#得到女0.66男0.7对结果影响不大，男性略高一点点，也很少
print(df.groupby(['Married']).Loan_Status.mean())#得到yes0.72no0.63已经有一定影响了，差距接近百分之10了
print(pd.pivot_table(df, values='Loan_Status', index='Gender', columns='Married', aggfunc='mean'))#结果显示女性无论是否结婚都比男性更容易通过一点，推翻先前结论，而且性别差距明显小于是否结婚的差距，应重点考虑后者而非前者

def collectiveincome(row):
    return row['ApplicantIncome']+row['CoapplicantIncome']
df['Collectiveincome']=df.apply(collectiveincome,axis=1)