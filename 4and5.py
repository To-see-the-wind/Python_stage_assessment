import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
print(df.info())#主要问题：多列有数据缺失

print(df.Self_Employed.value_counts())#发现绝大多数no，因此统一fillna写no
df.Self_Employed=df.Self_Employed.fillna('No')
print(df.LoanAmount.value_counts())#发现贷款金额对结果影响较大，且数量较少，因此删除这些行
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
bars=df.groupby('Education').Loan_Status.mean()
series=df.groupby('Loan_Amount_Term')['Loan_Status'].mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(bars.index, bars.values)
axes[0].set_title('学历与通过率关系')
axes[0].set_xlabel('学历')
axes[0].set_ylabel('通过率')

axes[1].hist(df['Collectiveincome'], bins=100, edgecolor='white')
axes[1].set_title('共同收入分布')
axes[1].set_xlabel('收入')
axes[1].set_ylabel('人数')

axes[2].scatter(series.index, series.values)
axes[2].set_title('贷款期限与通过率关系')
axes[2].set_xlabel('贷款期限/月')
axes[2].set_ylabel('通过率')

plt.tight_layout()
plt.show()