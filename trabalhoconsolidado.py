import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Análise incial do arquivo de PIB
pib = pd.read_csv('pibconvertido.csv')
fig, ax = plt.subplots()
ax.plot(pib.iloc[:, 0], pib.iloc[:, 1], 'b-')
plt.xlabel('Ano')
plt.ylabel('Variacao real do PIB YoY')
plt.title('Analisando a variacao do PIB anual descontando a inflacao')
plt.show()

#Traçando os gráficos com os dados já manipulados de Resultado Primário do Governo Central
data = pd.read_csv('updated copy.csv')
columns = data.columns
dates = data[columns[0]]
values = columns[1:]
dates = pd.to_datetime(dates, format='%m/%Y')

plt.figure(figsize=(10, 6))
for value in values:
    plt.plot(dates, data[value], label=value)
plt.xlabel('Data')
plt.ylabel('Quantidade em milhões de R$')
plt.title('Resultado Primário do Governo Central')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
plt.savefig('resultadoprimario.png')
#Traçando gráficos com os dados de Balança Comercial

balanca = pd.read_csv('balancacomercial.csv')
balanca_columns = balanca.columns
datas_balanca = balanca[balanca_columns[0]]
datas_balanca = pd.to_datetime(datas_balanca, format='%m/%Y')
dados_balanca = balanca_columns[1:]

plt.figure(figsize=(10, 6))
for value in dados_balanca:
    plt.plot(datas_balanca, balanca[value], label=value)

plt.xlabel('Data')
plt.xticks(rotation=45)
plt.ylabel('Valores em US$ milhões')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('balançacomercial.png')
#Mudando o padrão dos dados de data da soja de ano-mes-dia para dia/mes/ano

"""
soja_acerto = pd.read_csv('preco_soja.csv')
soja_acerto['Date'] = pd.to_datetime(soja_acerto['Date']).dt.strftime('%m/%y')
soja_acerto.to_csv('preco_soja.csv',index=False)
"""
#Como já foi rodado, colocamos como comentado, dado que já salvou em um novo csv.


#Mudando o padrão dos dados de data do Brent de ano-mes-dia para dia/mes/ano
"""
brent_acerto = pd.read_csv('BrentOilPrices.csv')
brent_acerto['Date'] = pd.to_datetime(brent_acerto['Date']).dt.strftime('%d/%m/%Y')
brent_acerto.to_csv('BrentOilPrices.csv',index=False)

Como já foi rodado, colocamos como comentado, dado que já salvou em um novo csv.
"""

#Mudando o padrão dos dados de data do Iron Ore de ano-mes-dia para dia/mes/ano
"""
iron_acerto = pd.read_csv('PIORECRUSDM.csv')
iron_acerto['Date'] = pd.to_datetime(iron_acerto['Date']).dt.strftime('%m/%y')
iron_acerto.to_csv('ironore.csv',index=False)
#Como já foi rodado, colocamos como comentado, dado que já salvou em um novo csv.

"""
#Agora temos nossas 3 commodities a ser analisadas com padrão correto de datas

#Vamos traçar gráficos com os preços das 3 commodities, para analisarmos se movimentam de forma semelhante
"""
"""

soja = pd.read_csv('preco_soja.csv')
iron = pd.read_csv('ironore.csv')
brent = pd.read_csv('finalbrent.csv')

fig, ax = plt.subplots()

soja['Date'] = pd.to_datetime(soja['Date'], format='%m/%y')
iron['Date'] = pd.to_datetime(iron['Date'], format='%m/%y')
brent['Date'] = pd.to_datetime(brent['Date'], format='%m/%y')

ax.plot(soja['Date'], soja['Price'], label='Soja')
ax.plot(iron['Date'], iron['Price'], label='Ferro')
ax.plot(brent['Date'], brent['Price'], label='Brent')

ax.set_xlabel('Data')
ax.set_ylabel('Preço')
ax.set_title('Analisando o preço das commodities')

ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.show()
plt.savefig('commodities.png')
#Agora vamos calcular a correlação entre PIB e variação do preço de cada commodity

correlacao_soja_pib = pib.iloc[:, 1].corr(soja['Price'])
print(f"Coeficiente de correlação entre PIB e preço da soja: {correlacao_soja_pib}")

correlacao_iron_pib = pib.iloc[:,1].corr(iron['Price'])
print(f"Coeficiente de correlação entre PIB e preço do ferro: {correlacao_iron_pib}")

correlacao_brent_pib = pib.iloc[:, 1].corr(brent['Price'])
print(f"Coeficiente de correlação entre PIB e preço do brent: {correlacao_brent_pib}")

#Correlação entre Receita do Governo Central e a variação do preço de cada commodity

correlacao_soja_receita = data.iloc[:,1].corr(soja['Price'])
print(f"Coeficiente de correlação entre Receita e preço da soja: {correlacao_soja_receita}")

correlacao_iron_receita = data.iloc[:,1].corr(iron['Price'])
print(f"Coeficiente de correlação entre Receita e preço do ferro: {correlacao_iron_receita}")

correlacao_brent_receita = data.iloc[:,1].corr(brent['Price'])
print(f"Coeficiente de correlação entre Receita e preço do Brent: {correlacao_brent_receita}")

#Vamos agora traçar gráficos analisando Receita e as commodities juntas

scaler = MinMaxScaler()
soja['Price_norm'] = scaler.fit_transform(soja['Price'].values.reshape(-1, 1))
iron['Price_norm'] = scaler.fit_transform(iron['Price'].values.reshape(-1, 1))
brent['Price_norm'] = scaler.fit_transform(brent['Price'].values.reshape(-1, 1))
data['Receita_norm'] = scaler.fit_transform(data[' Receita'].values.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(10, 6))

soja['Date'] = pd.to_datetime(soja['Date'], format='%m/%y')
iron['Date'] = pd.to_datetime(iron['Date'], format='%m/%y')
brent['Date'] = pd.to_datetime(brent['Date'], format='%m/%y')
data['Date'] = pd.to_datetime(data['Data'], format='%m/%Y')

ax.plot(soja['Date'], soja['Price_norm'], label='Soja')
ax.plot(iron['Date'], iron['Price_norm'], label='Ferro')
ax.plot(brent['Date'], brent['Price_norm'], label='Brent')
ax.plot(data['Date'], data['Receita_norm'], label='Receita')

ax.set_xlabel('Data')
ax.set_ylabel('Valores normalizados')
ax.set_title('Receita x Commodities')

ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()

plt.show()


#Vamos traçar uma regressão polinomial para traçar uma análise mais detalhada entre as variáveis



brent_anual= brent.groupby(brent['Date'].dt.year)['Price'].mean().reset_index()
pib_filtrado = pib[(pib['Data'] >= brent_anual['Date'].min()) & (pib['Data'] <= brent_anual['Date'].max())]
merged_data = pd.merge(pib_filtrado, brent_anual, left_on='Data', right_on='Date')

# Definindo as variáveis independentes (Brent) e dependente (PIB)
X = merged_data['Price']
y = merged_data['Variacao anual PIB real']


poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X.values.reshape(-1, 1))

model = LinearRegression()
model.fit(X_poly, y)
coefficients = model.coef_
intercept = model.intercept_

print(f"Coeficientes da regressão: {coefficients}")
print(f"Intercepto da regressão: {intercept}")
