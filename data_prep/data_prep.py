# https://github.com/ada-k/BankTermDepositPrediction
import pandas as pd

# read and process data
class DataPrep():

  # read the data
  def read_data(self, path):
    data = pd.read_csv(path, sep = ';')
    return data

  # preprocessing
  # def preprocess_data(self, data):
    # categorical + numerical + timestamp columns

  def treat_null(self, data):
    global categorical, discrete, continous, cols
    categorical = []
    discrete = []
    continous = []
    for col in data.columns:
      if data[col].dtype == object:
        categorical.append(col)
      elif data[col].dtype in ['int16', 'int32', 'int64']:
        discrete.append(col)
      elif data[col].dtype in ['float16', 'float32', 'float64']:
        continous.append(col)

    cols = discrete + categorical + continous
    data = data[cols]

    # null values
    # data = preprocess_data(data)
    indices = []
    for col in cols:
      k = data.columns.get_loc(col)
      indices.append(k)

    for col in indices:
      if data.columns[col] in discrete:
        x = data.iloc[:, col].values
        x = x.reshape(-1,1)
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer = imputer.fit(x)
        x = imputer.transform(x)
        data.iloc[:, col] = x

      if data.columns[col] in continous:
        x = data.iloc[:, col].values
        x = x.reshape(-1,1)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(x)
        x = imputer.transform(x)
        data.iloc[:, col] = x

      elif data.columns[col] in categorical:
        x = data.iloc[:, col].values
        x = x.reshape(-1,1)
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer = imputer.fit(x)
        x = imputer.transform(x)
        data.iloc[:, col] = x

    return data

    # outlier detection + treatment
    def outlier_correcter(self, data):
      # data = treat_null(data)
      for col in discrete + continous:
        data[col] = data[col].clip(lower=data[col].quantile(0.10), upper=data[col].quantile(0.90))
      return data

    # feature generation
    def generate_features(self, data):
      data['both_loans'] = 0  # default to 0
      data.loc[data['housing'] == 'yes', 'both_loans'] = 1
      data.loc[data['loan'] == 'no', 'both_loans'] = 1  # change to 1 if one has both loans
      data['total_contacts'] = data['campaign'] + data['previous']

      def squares(data, ls):
        m = data.shape[1]
        for l in ls:
          # data = data.assign(newcol=pd.Series(np.log(1.01+data[l])).values)
          data = data.assign(newcol=pd.Series(data[l] * data[l]).values)
          data.columns.values[m] = l + '_sq'
          m += 1
        return data

      log_features = ['duration', 'cons.price.idx', 'emp.var.rate', 'cons.conf.idx', 'euribor3m']

      data = squares(data, log_features)

      return data

    # scaling numerical
    def scaler(self, data):
      # data = outlier_correcter(data)
      indices = []
      for col in discrete + continous + ['total_contacts', 'duration_sq', 'cons.price.idx_sq', 'emp.var.rate_sq',
                                         'cons.conf.idx_sq', 'euribor3m_sq']:
        k = data.columns.get_loc(col)
        indices.append(k)

        for col in indices:
          x = data.iloc[:, col].values
          x = x.reshape(-1, 1)
          imputer = StandardScaler()
          imputer = imputer.fit(x)
          x = imputer.transform(x)
          data.iloc[:, col] = x

      return data

    # encoding categorical
    def encoder(self, data):
      # data = scaler(data)
      cols = categorical.copy()
      cols.remove('y')
      data = pd.get_dummies(data, columns=cols)
      return data

    # class imbalance
    def over_sample(self, data):
      # data = scaler(data)
      subscribers = data[data.y == 'yes']
      non_subscribers = data[data.y == 'no']

      subscribers_upsampled = resample(subscribers, replace=True,  # sample with replacement
                                       n_samples=len(non_subscribers),  # match number in majority class
                                       random_state=42)  # reproducible results

      data = pd.concat([subscribers_upsampled, non_subscribers])
      return data

# dataset http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

d = DataPrep()
path = '/content/bank-additional-full.csv'
data = d.read_data(path)
data = d.treat_null(data)
data = d.outlier_correcter(data)
data = d.generate_features(data)
data = d.scaler(data)
print('After scaling:', data.shape)
data = d.encoder(data)
data = d.over_sample(data)
data.head()


# split the data to have the predictor and predicted variables
x = data.drop(['y'], axis = 1)
y = data[['y']]
# Encode labels in target df.
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# get the sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train ,y_train, test_size = 0.20, random_state = 42)


regressor = LogisticRegression()
grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
model = GridSearchCV(regressor, param_grid=grid_values)
model.fit(x_train,y_train)
print(model.best_score_)
print(model.best_params_)

xgb = XGBClassifier(silent = True,max_depth = 6, n_estimators = 200)
xgb.fit(x_train, y_train)
# using kfolds
print('xgb mean score on the original dataset (kfold):', overall_score(xgb, x_train))
# stratified KFold
print('xgb mean score on the original dataset (stratified kfold):', overall__stratified_score(xgb, x_train))


