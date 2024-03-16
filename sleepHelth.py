import pandas as pd
 
path= r"Sleep_health_and_lifestyle_dataset.csv"
df=pd.read_excel(path)
# print(df)
df.isna().sum()
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No Sleep Disorder')
 
df.isna().sum()
 
 
 
 
########1st objective
import matplotlib.pyplot as plt
korelasi = df['Sleep Duration'].corr(df['Quality of Sleep'])
print('relationship between sleep duration and sleep quality: ', korelasi)
 
plt.scatter(df['Sleep Duration'], df['Quality of Sleep'])
plt.xlabel('Sleep duration')
plt.ylabel('Quality of sleep')
plt.title('relationship between sleep duration and sleep quality')
plt.show()
 
 
#2nd objective
 
 
import seaborn as sns
sns.boxplot(x='Physical Activity Level', y='Quality of Sleep', data=df)
plt.xlabel('Physical activity level')
plt.ylabel('Quality of sleep')
plt.title('The Relationship between Physical Activity and Sleep Quality')
plt.show()

#3rd objective
 
stres = df['Stress Level']
kualitas_tdr = df['Quality of Sleep']
 
kor = stres.corr(kualitas_tdr)
print(kor)
 
plt.scatter(stres, kualitas_tdr)
plt.xlabel('Stress Level')
plt.ylabel('Quality of Sleep')
plt.title('The relationship between stress levels and sleep quality')
plt.show()
 
#######
sleep_duration=df['Sleep Duration'].mean()
age = df['Age'].mean()
print('average sleep duration')
print(sleep_duration)
print('age average')
print(age)
 
from pandas._libs.tslibs.period import DIFFERENT_FREQ
import scipy
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']
 
mean_dur_male = df_male['Sleep Duration'].mean()
mean_dur_female = df_female['Sleep Duration'].mean()
 
t_statistic, p_value = scipy.stats.ttest_ind(df_male['Sleep Duration'], df_female['Sleep Duration'])
 
print("men's average sleep duration: ",mean_dur_male)
print("women's average sleep duration: ", mean_dur_female)
print("statistic: ", t_statistic)
print("p_value: ", p_value)
 
 
######
plt.figure(figsize=(10, 5))
sns.boxplot(x='Occupation', y='Sleep Duration', data=df)
plt.title('Comparison of sleep duration with work')
plt.xlabel('job')
plt.ylabel('duration')
plt.xticks(rotation=45)
plt.show()
 
#####
plt.figure(figsize=(10, 5))
sns.boxplot(x='Occupation', y='Quality of Sleep', data=df)
plt.title('comparison of sleep quality with work')
plt.xlabel('job')
plt.ylabel('quality')
plt.xticks(rotation=45)
plt.show()
 
 
#####4th objectives
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE

path= r"Sleep_health_and_lifestyle_dataset.csv"
df=pd.read_excel(path)
print(df)
df.isna().sum()
# Encode categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category','Blood Pressure','Heart Rate','Daily Steps'])

# Handle missing values
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No Sleep Disorder')


df.isna().sum()
X = df.drop('Sleep Disorder',axis=1)
y = df['Sleep Disorder']
 
# Pembagian dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Fitur seleksi menggunakan SelectKBest
selector = SelectKBest(score_func=f_classif,k=5)
X_train_selected = selector.fit_transform(X_train,y_train)
X_test_selected = selector.transform(X_test)
 
 
# Model Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_selected, y_train)
lr_accuracy = lr_model.score(X_test_selected, y_test)
 
# Fitur seleksi menggunakan Recursive Feature Elimination (RFE)
estimator = DecisionTreeClassifier()
rfe_selector = RFE(estimator, n_features_to_select=5)
X_train_selected_rfe = rfe_selector.fit_transform(X_train, y_train)
X_test_selected_rfe = rfe_selector.transform(X_test)
 
# Model Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_selected_rfe, y_train)
dt_accuracy = dt_model.score(X_test_selected_rfe, y_test)
 
# Model Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_selected_rfe, y_train)
rf_accuracy = rf_model.score(X_test_selected_rfe, y_test)
 
# Model Support Vector Machines (SVM)
svm_model = SVC()
svm_model.fit(X_train_selected_rfe, y_train)
svm_accuracy = svm_model.score(X_test_selected_rfe, y_test)
 
# Menampilkan akurasi model
print("Logistic Regression Accuracy:", lr_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("SVM Accuracy:", svm_accuracy)