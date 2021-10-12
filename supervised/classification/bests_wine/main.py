import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

def learningWines():
    print('Lendo dataset...')
    arq = pd.read_csv('wine_dataset.csv')

    print('Transformando de dados para validação binária...')
    # Style = string(Red, White) para int(0, 1)
    arq['style'] = arq['style'].replace('white', 1)
    arq['style'] = arq['style'].replace('red', 0)

    # Variável Alvo
    y = arq['style']

    # Variável Preditora
    x = arq.drop('style', axis=1)

    # print(y)
    # print(x)

    # Criação de dados para treino e teste do modelo
    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.2)# Pegar 20% do treino para execução de teste

    # print(x.shape)
    # print(y.shape)

    # Criação de modelo
    modelo = ExtraTreesClassifier()
    print('Treinando Modelo...')
    modelo.fit(xTreino, yTreino)

    resultado = modelo.score(xTeste, yTeste)
    print(f"Porcentagem de Apredizado em Treinamento: {float(str(resultado)[0:4])*100}%")

    # print(xTeste[0:3])
    print(f'{"-"*50}\nDados Corretos:\n{yTeste[0:3]}')

    predicao = modelo.predict(xTeste[0:3])
    print(f'{"-"*50}\nPredição do Modelo: {predicao}')
    # return f'{"-"*50}\nPredição do Modelo: {predicao}'

if __name__ == '__main__':
    learningWines()
