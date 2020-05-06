import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['SimHei','Arial']})
original=pd.read_csv('C:\\Users\\1994y\\Desktop\\data_1.csv')
b=original.dropna()
c=b["名称"].str.strip()
cleaned=pd.DataFrame([])
cleaned['状态']=c.str.split('·').str[0]
cleaned['规格']=c.str.split(' ').str[-2]
cleaned['楼层']=b['内容'].str.strip().str.split(' ').str[-25]
cleaned['面积']=b['内容'].str.strip().str.split('/').str[1].str.strip()
cleaned['朝向']=b['内容'].str.strip().str.split('/').str[2].str.strip()
cleaned['价格']=b['价格'].str.split('元').str[0]
cleaned['地区']=b["地区"]
cleaned['地址']=b["地址"]
cleaned['来源']=b['来源'].str.strip()
house_count = cleaned.groupby('地区')['价格'].count().sort_values(ascending=False).reset_index()#重新调整为默认索引
f, [ax1,ax2] = plt.subplots(2,1,figsize=(20,15))

sns.barplot(x='地区', y='价格', palette="Greens_d",data=house_count, ax=ax1)
ax1.set_title('天津各区租房数量对比',fontsize=15)
ax1.set_xlabel('地区')
ax1.set_ylabel('数量')

sns.boxplot(x=cleaned['地区'].astype('str'),y=cleaned['价格'].astype('int'),ax=ax2)
ax2.set_title('天津各区租房房租',fontsize=15)
ax2.set_xlabel('地区')
ax2.set_ylabel('租金')
move=['LOFT','合租·龙悦花园','合租·洪湖雅园','合租·万顺温泉花园','合租·谊城公寓']
cleaned=cleaned[~cleaned['规格'].isin(move)]
cleaned=cleaned[~cleaned['朝向'].isin(['3室1厅1卫'])]
move_direction=['南 北','东 西','东 南','南 西','西南 西','东南 西','东 北','南 西北','东南 北']
correct_direction=['南北','东西','东南','西南','西南','东西南','东北','南西北','东南北']
def direction_correct(x):
    for i in range(len(move_direction)):
        if x == move_direction[i]:
            x=correct_direction[i]     
    return x
cleaned['朝向']=cleaned['朝向'].apply(direction_correct)
cleaned['面积']=cleaned['面积'].str[:-1]
cleaned['价格']=pd.qcut(cleaned['价格'],4).astype('object')
cleaned=cleaned.reset_index()
del cleaned['index']
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)         # 属性
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
house,house_columns=one_hot_encoder(cleaned)#独热编码
colormap = plt.cm.RdBu
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(20,20))
sns.heatmap(house.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

from sklearn.model_selection import train_test_split
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
cleaned.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in cleaned.columns.values]
x=cleaned.iloc[:,:-1]
y=cleaned.iloc[:,5] 
x_dum=pd.get_dummies(x)    #独热编码
x_train,x_test,y_train,y_test = train_test_split(x_dum,y,test_size = 0.3,random_state = 1)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressormodels=[XGBRegressor(),RandomForestRegressor(),GradientBoostingRegressor()]
models_str=['XGBoost','RandomForest','GradientBoost','Bagging']
score_=[]
def metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score
for name,model in zip(models_str,models):
    print('开始训练模型：'+name)
    model=model   #建立模型
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)  
    score=metric(y_test,y_pred)
    score_.append(str(score))
    print(name +' 得分:'+str(score))
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
other_params = {'eta': 0.3, 'n_estimators': 200, 'gamma': 0.37, 'max_depth': 9, 'min_child_weight': 0,
                'colsample_bytree': 1, 'colsample_bylevel':0.9, 'subsample': 0.2, 'reg_lambda': 1, 'reg_alpha': 0,
                'seed': 33}
def fit_model(X, y,params):
    cross_validator = KFold(10, shuffle=True)
    regressor = XGBRegressor(**other_params)
    cv_params =params
    scoring_fnc = make_scorer(metric)
    grid = GridSearchCV(estimator = regressor, param_grid = cv_params, scoring = scoring_fnc, refit=True,cv = cross_validator)
    grid = grid.fit(X, y)
    print("参数的最佳取值：:", grid.best_params_)
    print("最佳模型得分:", grid.best_score_)
fit_model(x_train, y_train,{'n_estimators': np.linspace(100, 1000, 10, dtype=int)})
fit_model(x_train, y_train,{'n_estimators': np.linspace(100, 300, 10, dtype=int)})
fit_model(x_train, y_train,{'max_depth': np.linspace(1, 10, 10, dtype=int)})
fit_model(x_train, y_train,{'min_child_weight': np.linspace(0, 10, 10, dtype=int)})
fit_model(x_train, y_train,{'gamma': np.linspace(0.1, 0.5, 10)})
fit_model(x_train, y_train,{'subsample': np.linspace(0, 1, 11)})
fit_model(x_train, y_train,{'colsample_bytree': np.linspace(0, 2, 10)[1:]})
fit_model(x_train, y_train,{'reg_lambda': np.linspace(0, 100, 11)})   
fit_model(x_train, y_train,{'reg_alpha': np.linspace(0, 10, 11)})
fit_model(x_train, y_train,{'eta': np.logspace(-2, 0, 10)})


