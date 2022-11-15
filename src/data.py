import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__all__ = ['DATA_PATH', 'data_preprocess']

DATA_PATH = "./data/loan_approval_data.csv"

def data_preprocess(filepath):
    #Read CSV data
    data = pd.read_csv(filepath)

    train_data = data.copy()
    train_data['Gender'].fillna(train_data['Gender'].value_counts().idxmax(), inplace=True)
    train_data['Married'].fillna(train_data['Married'].value_counts().idxmax(), inplace=True)
    train_data['Dependents'].fillna(train_data['Dependents'].value_counts().idxmax(), inplace=True)
    train_data['Self_Employed'].fillna(train_data['Self_Employed'].value_counts().idxmax(), inplace=True)
    train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(skipna=True), inplace=True)
    train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
    train_data['Credit_History'].fillna(train_data['Credit_History'].value_counts().idxmax(), inplace=True)

    # Convert some object data type to int64
    gender_stat = {"Female": 0, "Male": 1}
    yes_no_stat = {'No' : 0,'Yes' : 1}
    y_n_stat = {'N' : 0,'Y' : 1}
    dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
    education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
    property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}

    train_data['Gender'] = train_data['Gender'].replace(gender_stat)
    train_data['Married'] = train_data['Married'].replace(yes_no_stat)
    train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)
    train_data['Education'] = train_data['Education'].replace(education_stat)
    train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)
    train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)
    train_data['Loan_Status'] = train_data['Loan_Status'].replace(y_n_stat)

    scaler = MinMaxScaler()
    scalables = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    train_data[scalables] = scaler.fit_transform(train_data[scalables])

    return train_data

def split_data(data):
    # Separate into data and labels
    split = int(len(data)*0.8)
    x_train = data.iloc[:split,1:12]
    y_train = data.iloc[:split,12]

    x_test = data.iloc[split:,1:12]
    y_test = data.iloc[split:,12]

    return (x_train, y_train), (x_test, y_test)
