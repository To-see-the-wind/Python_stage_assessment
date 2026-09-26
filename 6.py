import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

print(df.Self_Employed.value_counts())#发现绝大多数no，因此统一fillna写no
df.Self_Employed=df.Self_Employed.fillna('No')
print(df.LoanAmount.value_counts())#发现贷款金额对结果影响较大，且数量较少，因此删除这些行
df=df.dropna(subset=['LoanAmount'])
print(df.Credit_History.value_counts())#信用达标对结果影响较大，但是缺失数据较多，因此填写unknown
df.Credit_History=df.Credit_History.fillna('Unknown')
df.Gender=df.Gender.fillna('Male')#大多数是男人且对结果影响较小
df=df.dropna(subset=['Married','Loan_Amount_Term','Dependents'])#对结果影响大，而且缺失条数少
df['Loan_Status']=df['Loan_Status'].replace({'Y':1,'N':0}).astype(int)
def collectiveincome(row):
    return row['ApplicantIncome']+row['CoapplicantIncome']
df['Collectiveincome']=df.apply(collectiveincome,axis=1)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'3+': 3,'2':2,'1':1,'0':0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Credit_History'] = df['Credit_History'].replace({'Unknown':2}).astype(int)
df['Property_Area'] = df['Property_Area'].map({'Semiurban': 3,'Urban':2,'Rural':1})
df=df.drop(columns=['Loan_ID'])
X=df.drop(columns=['Loan_Status'])
y=df['Loan_Status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)#划分出一部分用于自检，可以得到更真实的模型能力，避免挖掘过深，死记硬背
models = {
    '逻辑回归': make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
    '决策树': DecisionTreeClassifier(max_depth=4, random_state=42),
    '随机森林': RandomForestClassifier(max_depth=4, random_state=42),
}
for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    print(accuracy_score(y_train, y_pred))#交叉验证准确率均比训练集低，0.02，0。05，0.03，过拟合轻微
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: 平均={scores.mean():.4f}, 标准差={scores.std():.4f}')#随机森林最准，逻辑回归最稳定
# #逻辑回归适合总结原因，关系清晰
#决策树规则直观,能画出来，且速度比随机森林快
#随机森林数多，更稳定，不容易犯偏执的错