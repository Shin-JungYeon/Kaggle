# 데이터 분석

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("https://raw.githubusercontent.com/developer-sdk/kaggle-python-beginner/master/datas/kaggle-titanic/train.csv")
train.head(5)

train.isnull().sum()

def show_pie_chart(df, col_name):
    colname_survived = survived_crosstab(train, col_name)
    pie_chart(colname_survived)
    return colname_survived

def survived_crosstab(df, col_name):
    '''col_name과 Survived간의 교차도표 생성'''
    feature_survived = pd.crosstab(df[col_name], df['Survived'])
    feature_survived.columns = feature_survived.columns.map({0:"Dead", 1:"Alive"})
    return feature_survived

def pie_chart(feature_survived):
    '''
    pie_chart 생성
    pcol, prow = 차트를 출력할 개수. pcol * prow 만큼의 차트 출력 
    '''
    frows, fcols = feature_survived.shape
    pcol = 3
    prow = (frows/pcol + frows%pcol)
    plot_height = prow * 2.5
    plt.figure(figsize=(8, plot_height))

    for row in range(0, frows):
        plt.subplot(prow, pcol, row+1)

        index_name = feature_survived.index[row]
        plt.pie(feature_survived.loc[index_name], labels=feature_survived.loc[index_name].index, autopct='%1.1f%%')
        plt.title("{0}' survived".format(index_name))

    plt.show()

# =============================================
print("성별과 생존률 관계:")
c = show_pie_chart(train, 'Sex')
c

# =============================================
print("탑승항과 생존률 관계:")
c = show_pie_chart(train, 'Embarked')
c

# =============================================
print("호칭과 생존률 관계:")
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.')
train['Title'].value_counts()

train['Title'] = train['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer', 'Lady','Major', 'Rev', 'Sir'], 'Other')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'].value_counts()

c = show_pie_chart(train, 'Title')
c

# =============================================
print("나이와 생존률 관계:")
meanAge = train[['Title', 'Age']].groupby(['Title']).mean()
for index, row in meanAge.iterrows():
    nullIndex = train[(train.Title == index) & (train.Age.isnull())].index
    train.loc[nullIndex, 'Age'] = row[0]

train['AgeCategory'] = pd.qcut(train.Age, 8, labels=range(1, 9))
train.AgeCategory = train.AgeCategory.astype(int)

c = show_pie_chart(train, 'AgeCategory')
c

# =============================================
print("방 번호와 생존률 관계:")
train.Cabin.fillna('N', inplace=True)
train["CabinCategory"] = train["Cabin"].str.slice(start=0, stop=1)  
train["CabinCategory"] = train['CabinCategory'].map({ "N": 0, "C": 1, "B": 2, "D": 3, "E": 4, "A": 5, "F": 6, "G": 7, "T": 8 })

c = show_pie_chart(train, 'CabinCategory')
c

# =============================================
print("운임과 생존률 관계:")
train.Fare.fillna(0)  # test.csv 데이터에 결측치가 존재함.
train['FareCategory'] = pd.qcut(train.Fare, 8, labels=range(1, 9))
train.FareCategory = train.FareCategory.astype(int)

c = show_pie_chart(train, 'FareCategory')
c

# =============================================
print("가족 여부와 생존률 관계:")
train['Family'] = train['SibSp'] + train['Parch'] + 1
train.loc[train["Family"] > 4, "Family"] = 5

train['IsAlone'] = 1
train.loc[train['Family'] > 1, 'IsAlone'] = 0


c = show_pie_chart(train, 'Family')
c

c = show_pie_chart(train, 'IsAlone')
c


# =============================================
print("티켓 정보와 생존률 관계:")
# STON/O2. 3101282를 ['STON/O2.', '3101282']로 변경하고, '3101282'의 첫 번째 3을 선택 
train['TicketCategory'] = train.Ticket.str.split() # 공백으로 분리 
train['TicketCategory'] = [i[-1][0] for i in train['TicketCategory']] # 
train['TicketCategory'] = train['TicketCategory'].replace(['8', '9', 'L'], '8')
train['TicketCategory'] = pd.factorize(train['TicketCategory'])[0] + 1

c = show_pie_chart(train, 'TicketCategory')
c
