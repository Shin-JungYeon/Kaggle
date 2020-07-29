Kaggle 입문하기 on 20. 07. 27

---

#### 1. Kaggle 문제 풀이

- 문제에 datasets 주어짐
- pandas 라이브러리를 이용해 외부 파일(datasets) 읽어서 데이터 확인

  - 데이터의 구성, 결측치(null data) 확인
- matplotlib / seaborn / numpy 등의 라이브러리를 이용해 데이터 시각화 및 분석

  - feature들을 개별적 분석
  - feature들 간의 상관관계
- 분석한 데이터를 학습시키기 좋은 형태로 정리하는 전처리 작업 수행
- scikit-learn 라이브러리를 이용해 전처리한 데이터에 머신러닝 알고리즘 적용

  - 다양한 머신러닝 알고리즘 모듈 제공 (간단하게 사용 가능)

---

- 라이브러리

  - Pandas : 추상적인 자료구조와 데이터 분석 도구 제공. NumPy, scikit-learn, matplotlib와 함께 많이 사용.

    - 자료구조

      - **데이터프레임(Dataframe)** 
        - 표 형식의 자료구조.
        - 시리즈의 모음.
        - 컬럼(columns), 인덱스(index), 값(values)으로 구성.
      - **시리즈(Series)** 
        - 배열 형식의 자료구조.
        - 값(values)과 인덱스(index)로 구성.
        - 데이터프레임은 시리즈의 모음.

      |   RDB   |        Pandas         |
      | :-----: | :-------------------: |
      | 테이블  |     데이터프레임      |
      |  컬럼   |  데이터프레임의 컬럼  |
      | 행 번호 | 데이터프레임의 인덱스 |
      |   값    |   데이터프레임의 값   |

      

      <img src="https://wikidocs.net/images/page/75004/dataframe_sample_01.png" alt="dataframe" style="zoom: 25%;" />

    - 기본 함수

      - 데이터 조작 / 결측치 확인 / 형 변환 / 시계열 조작 / 인터벌 조작 / 평가 / 해싱에 관련된 기본 함수 제공.
      
    - SQL 쿼리문을 사용하는 것처럼 데이터 조회 및 수정 가능.

      ```python
      # 1. 대괄호 사용하여 조건식 입력.
      #    SELECT A, B
      DataFrame[["A", "B"]]
      # UPDATE D = 5 WHERE D = 1
      DataFrame.ix[DateFrame.ix[:,'D'] == 1, "D"] = 5
      
      ...
      # 2. query() 메서드 사용.
      #    expr : 조건식. 문자열로 입력.
      #    inplace=True : query에 의해 출력된 데이터로 원본 데이터를 대체.
      #    대용량의 데이터 처리할 경우 성능면에서 더 우월함.
      DataFrame.query(expr, inplace=False)
      ```
    
  - NumPy : **다차원 배열(ndarray)**을 이용하여 연산이 편리하고, 파이썬 기본 자료구조에 비하여 매우 빠른 작업 속도를 제공. 다차원 배열은 각 값에 대한 연산을 할 때 벡터화 연산을 이용하여 편리하게 계산. 데이터를 연속된 메모리 블록에 저장, C로 구현되어 있어 오버헤드 없이 메모리를 직접 조작.
  
  - sciki-learn : 파이썬으로 구현된 기계학습 라이브러리.
  
    - 분류(Classification) / 회귀(Regression) / 군집(Clustering) / 차원 축소(Dimensionality Reduction) / 모델 선택(Model selection) / 전처리(Preprocessing) 알고리즘 제공.
    - 지도학습, 비지도학습 모듈
    - 모델 선택 및 평가 모듈
    - 데이터 변환 및 데이터 불러오기 위한 모듈
    - 계산 성능 향상을 위한 모듈
  
    ```python
    # 지도학습 모델 import - 입력에 대한 결과값이 있어 이를 학습하고, 새로운 데이터에 대한 결과 예측.
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # data 분리 - train / test(validation)
    data = train.drop('Gender', axis=1).values
    target = train['Gender'].values
    # test_size: 분리 비율 설정. default=0.25
    # stratify: 분리 기준이 될 데이터. default=None
    # random_state: 랜덤 seed
    # shuffle : split 해주기 전에 섞을 지 여부. default=True
    x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.4, stratify=target, random_state=0)
    
    # 모델 적용 함수 
    def ml_fit(model):
        model.fit(x_train, y_train) # 학습
        prediction = model.predict(x_valid) # 예측
        accuracy = accuracy_score(prediction, y_valid) # 학습한 모델 평가
        print(model)
        print(f'총 {y_valid.shape[0]}개 데이터 중 {accuracy * 100:.3f}% 정확도로 맞춤')
      print('\n')
        return model
  
    # 기본 설정으로만 테스트 
    model = ml_fit(RandomForestClassifier(n_estimators=100)) # n_estimators=100 : 의사결정나무 갯수.
    model = ml_fit(LogisticRegression(solver='lbfgs')) # solver='lbfgs' 명시해주지 않으면 경고 출력.
    model = ml_fit(SVC(gamma='scale', C=0.02)) # C=0.02 값이 클수록 하드마진(오류 허용 안 함), 작을수록 소프트마진(오류를 허용함).
    model = ml_fit(KNeighborsClassifier())
    model = ml_fit(GaussianNB())
    model = ml_fit(DecisionTreeClassifier())
    ```
  
  - matplotlib : 파이썬으로 구현된 **시각화를 위한 도구**.
  
  - seaborn : matplotlib를 기반으로 다양한 색상 테마, 통계용 차트 등의 기능을 추가한 **시각화 패키지**. 기본적인 시각화 기능은 matplotlib 패키지에 의존하며 통계 기능은 statsmodels 패키지에 의존

---

- 학습법

  - 지도 학습 : **주어진 입력에 대한 결과값이 있어 이를 학습**하고, 새로운 데이터에 대한 결과 예측. 주요 알고리즘은 회귀(Regression, 연속된 값 예측)와 분류(Classification, 종류 예측)
    - Linear Regression
    - **Logistic Regression (Binary Classification)**
    - Multinomial Classification
    - **Decision Tree**
    - **Random Forest**
    - **KNN (K-Nearest Neighbors)**
  - 비지도형 학습 : 주어진 입력에 대한 결과값 없이, **학습 데이터 속에 패턴이나 규칙을 찾아서 학습**. 기존에 알지 못했던 새로운 특징을 추출하거나 서로 관련이 높은 그룹끼리 자동으로 분류.
    - K-means
    - Apriori
  - 강화 학습 : **상과 벌을 통해 훈련의 감독관이 원하는 방향으로 학습**을 진행. 더 나은 방향으로 개선시키는 학습 방법. 데이터 없이도 학습 가능.
    - Markov Decision Process

---

- 머신러닝 알고리즘

  - Boosting
    
    - 가능성이 높은 규칙들을 결합시켜 예측 모델을 만들어 내는 것.
    
  - Bagging
    - 전체 모집단의 분포를 확실하게 알 수 없는 경우, **표본을 취한 후 그 표본이 전체 집단을 대표한다고 가정**.
    - 표본으로부터 많은 횟수에 걸쳐 (동일한 갯수의) 샘플을 복원 추출한 후 각 샘플들에 대한 분포를 구함.
    - 전체 표본의 분포와 샘플들 간의 분포의 관계를 통해 전체 집단의 분포 유추.
    
  - **Support Vector Machine (SVM)**
    
    - 분류(classification)나 회귀 분석(regression)에 사용 가능. 분류 쪽의 성능이 뛰어나 주로 분류에 사용.
    
    - 지도 학습 알고리즘.
  
    - **분류를 할 때 최고의 마진을 가져가는 방향으로 분류** 수행.
    
    - kernel trick : mapping을 통해 고차원으로 변환시킨 후 선형적으로 구별이 가능하도록 하는 방법. 이 때 사용하는 함수가 kernel 함수.
    
      <img src="https://i1.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm06.png?fit=1024%2C768" alt="img" style="zoom:33%;" />
    
  - **Decision Tree**
    
    - 기회 비용에 대한 고려, 기대 이익 계산, 자원의 효과적인 사용이나 위험 관리 등 효율적인 결정이 필요한 분야에서 사용.
    
  - 어떤 항목에 대한 **관측값(observation)에 대하여 가지(branch) 끝에 위치하는 기대값(target)과 연결**시켜 주는 예측 모델.
    
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/CART_tree_titanic_survivors_KOR.png/350px-CART_tree_titanic_survivors_KOR.png" style="zoom: 67%;" />
    
  - **Logistic Regression(classification)**
  
    - 결과값이 특정 분류임. (ex. 생존 or 죽음 / 일반 메일 or 스팸 메일 / ...)
    - 결과값이 linear한 Linear Regression과 달리 **0 또는1로 표현되는 Discrete한 값을 예측**하는데 사용.
  
  - **Naive Bayes**
    
    - 특징이 너무 많은 경우 이 특징들 간의 연관 관계를 모두 고려하면 너무 복잡해지는 경향이 있기 때문에 단순화 시켜 쉽고 빠르게 판단을 내릴 때 주로 사용.
    
    - **특정 개체가 특정 집단에 속할 확률**.
  
      <img src="https://lh3.googleusercontent.com/proxy/lQaMQ56xpihvdcezFxt6RasrI5pqs-87N3WcY7aSQX9wqiM97Z7VegQc1AOHtVjlrojtbwT0zNkQWnzoUbAozH1BV5hzgaPgJc5UgbnPM_RxNXa-Qu94cRrXOdrmXLJ3Lbts7wtRqV-Vwghshl3a32NzHzNgKhO73_Bl-yIyRKil6QBmQjQC0k65v4Poug" alt="013_Naive Bayes Classifier : 네이버 블로그" style="zoom:50%;" />
    
  - **k-Nearest Neighbor (kNN)**
  
    - 최근접(k)을 정하고, 최근접 거리에 k개가 들어오게 해서 예측. k는 숫자가 작은 홀수일수록 좋음.
  
      <img src="https://t1.daumcdn.net/cfile/tistory/99992E365B683CC309" alt="img" style="zoom: 50%;" />
  
  - **Random Forest**
    
    - Bagging process를 통해 여러 개의 의사결정나무를 생성. ensemble(앙상블) 학습법 사용.
    
      <img src="https://t1.daumcdn.net/cfile/tistory/99E1D44D5BED2D4719" alt="img" style="zoom: 50%;" />

---

- scikit-learn 라이브러리가 제공하는 예측 함수 정보(매개변수 정보)

  - RandomForestClassifier : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  - LogisticRegression : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
  - SVC : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  - KNeighborsClassifier : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  - GaussianNB : https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
  - DecisionTreeClassifier : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

  

