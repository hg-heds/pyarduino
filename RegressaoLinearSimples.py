import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.close('all')

# Importando a base de dados
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Matriz de correlação
mc = dataset.corr()
print('\n>>> Matriz de correlação:\n')
print(mc)

# Visualizando os dados
plt.scatter(X, y, color = 'red')
plt.title('Salário vs Experiência', fontsize = 20)
plt.xlabel('Anos de Experiência', fontsize = 20)
plt.ylabel('Salário', fontsize = 20)
plt.savefig('MinhaFigura.svg', bbox_inches = 'tight')
plt.show()

# Mãos à obra: criando o modelo
regressor = LinearRegression()
regressor.fit(X, y)

# Predizendo os resultados
y_pred = regressor.predict(X)

# Visualizando os resultados
plt.figure()
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Salário vs Experiência', fontsize = 20)
plt.xlabel('Anos de Experiência', fontsize = 20)
plt.ylabel('Salário', fontsize = 20)
plt.show()

# Avaliando a qualidade do modelo
print('\n>>> Qualidade do modelo:')
#EQM = sum((y_train-y_pred_train)**2)/len(y_train) --> mean_squared_error(yreal, yobtido)
# Pode-se utlizar regressor.score(X,y)
print('\n    Coeficiente de determinação:', r2_score(y, y_pred))
print('\n    Erro quadrático médio:', mean_squared_error(y_pred, y))

# Parâmetros do modelo
print('\n>>> Parâmetros do modelo (considerando y = Theta_0 + Theta_1*x):')
print('\n    Theta_0:', regressor.intercept_)
print('\n    Theta_1:', regressor.coef_[0])