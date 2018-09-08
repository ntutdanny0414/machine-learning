import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv(r'D:\dataset\Salary_Data.csv')
x = dataset.iloc[0:,0:1].values
y = dataset.iloc[:, 1].values
#plt.scatter(dataset.YearsExperience,dataset.Salary, color='blue')
plt.scatter(x,y, color='blue')
#plt.plot(dataset.YearsExperience,dataset.Salary, color='red')
plt.plot(x,y, color='red')
plt.title('YearsExperience&Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.savefig(r'D:\dataset\before.png')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3 , random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train, color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title('YearsExperience&Salary(training)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.savefig(r'D:\dataset\training.png')
plt.show()

plt.scatter(x_test,y_test, color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title('YearsExperience&Salary(training)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.savefig(r'D:\dataset\test.png')
plt.show()