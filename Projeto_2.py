import streamlit as st
import time
import numpy as np

#pandas: biblioteca utilizada para manipulação dos dados
import pandas as pd
#seaborn e matplotlib: bibliotecas utilizadas na visualização gráfica.
import seaborn as sns
import matplotlib.pyplot as plt
#ydata_profiling:
from ydata_profiling import ProfileReport

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree

from sklearn.metrics import mean_squared_error



st.set_page_config(
     page_title="Análise exploratória previsão de renda",
     page_icon="C:/Users/Cris/Downloads/dados.png", #:?:
     layout="wide",
)

with st.spinner(text='Carregando... Aguarde!'):
    time.sleep(3)
    
st.write('# Análise exploratória previsão de renda')

#Lendo o arquivo csv
renda = pd.read_csv('C:/Users/Cris/Documents/Python Scripts/Curso/Telemarketing/input/previsao_de_renda.csv')

renda.data_ref = pd.to_datetime(renda.data_ref)

min_data = renda.data_ref.min()
max_data = renda.data_ref.max()

#campos filtros de data:
data_inicial = st.sidebar.date_input('Data Inicial', 
                value = min_data,
                min_value = min_data,
                max_value = max_data)
data_final = st.sidebar.date_input('Data Final', 
                value = max_data,
                min_value = min_data,
                max_value = max_data)    

st.sidebar.write('Data Inicial = ', data_inicial)
st.sidebar.write('Data Final = ', data_final)
#Filtrando o dataset conforme a seleção das datas
renda  = renda[(renda['data_ref'] <= pd.to_datetime(data_final)) & (renda['data_ref'] >=pd.to_datetime(data_inicial) )]

with st.spinner(text='Carregando... Aguarde!'):
    time.sleep(3)

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader('Proporção de Clientes por Sexo')
        #plt.figure(figsize=(0.5, 0.5))

        df_sexo = renda['sexo'].value_counts()
        data = [df_sexo['F'], df_sexo['M']]
        keys = ['Feminino', 'Masculino']

        palette_color = sns.color_palette('pastel')   
        fig = plt.pie(data, labels=keys, colors=palette_color, explode=[0, 0.1], autopct='%.0f%%')
        
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)
      
    with col2:
        st.subheader('Média Salárial por Sexo')
        #plt.figure(figsize=(1, 2))
        fig = sns.barplot(data=renda, x='sexo', y='renda')
        #plt.xticks(rotation=20)
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)
       
    
    with col3:
        st.subheader('Média Salárial por Escolaridade')
        fig = sns.barplot(data=renda, x='renda', y='educacao')
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)  

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader('Quantidade de pessoas na mesma residência')
        fig = sns.countplot(data=renda, x='qt_pessoas_residencia')
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)  
      
    with col2:
        st.subheader('Quantidade de pessoas por Idade')
        fig = sns.histplot(renda['idade'], color='#A1C9F4', label='Idade', kde=False)
        plt.xticks(rotation=50)
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)  

    with col3:
        st.subheader('Média Salárial por Tipo de Renda')
        fig = sns.barplot(data=renda, x='tipo_renda', y='renda') 
        plt.xticks(rotation=20)
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 1, 1])
        
    with col1:
        st.subheader('Renda timeline')
        fig = sns.lineplot(x='data_ref',y='renda', data=renda)
        plt.xticks(rotation=20)
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)  
        
    with col2:
        st.subheader('Renda por Tempo de Emprego')
        fig = sns.scatterplot(data=renda, x="tempo_emprego", y="renda")
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)    
        
    with col3:
        st.subheader('Renda por Quantidade de filhos')
        fig = sns.scatterplot(data=renda, x="qtd_filhos", y="renda")
        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)       

st.write('# Modelo de Machine Learning utilizando o algoritmo Random Forest ')
st.text('Random Forest basicamente cria várias arvores de decisão aleatórias e com várias combinações. ')
st.text('O resultado é uma média do resultado dessas arvores, essa média é o valor da previsão,  isso da um pouco mais de acuracia ao modelo.')
st.text('Nesse momento vamos dividir a base em 2 partes, base de treino e base de teste.')
st.text('Na base de treino vamos treinar o nosso modelo, fazemos com que o algoritimo entenda a relação entre as varáveis e assim faça a previsão dos dados.')
st.text('Utilizamos a base de teste para avaliar o desempenho do modelo. Comparamos as saídas que já foram observadas e as previstas pelo modelo. ')
st.text('Assim conseguir saber a precisão e o quão bem o modelo consegue explicar essas saídas.')


with st.container(border=True):
    col1, col2= st.columns([1, 1])
        
        
    with col1:
        

        st.subheader('Modelos de Treino')  
        # TRATAMENTOS DOS DADOS E CRIAÇÃO DO MODELO 
        
        #removendo as colunas data_ref, id_cliente e Unnamed
        renda.drop(columns= ['data_ref', 'id_cliente','Unnamed: 0'], axis=1, inplace=True)
        
        #remove as linhas nulas
        renda = renda.dropna()

        #remove as linhas duplicadas
        renda = renda.drop_duplicates()

        renda['possui_filhos'] = renda['qtd_filhos'] != 0 

        renda = pd.get_dummies(renda, columns=['sexo', 'tipo_renda', 'educacao', 'tipo_residencia', 'estado_civil','possui_filhos'])

        # Separando o dataset em variáveis dependentes e independentes
        # Y variável dependente, é o dado que queremos prever 
        # X variável independente, são todas as demais colunas que serão utilizadas para explicar o modelo
        y = renda['renda']
        X = renda.drop(['renda'], axis=1).copy()

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=100)
        modelo1 = RandomForestRegressor(max_depth=2)
        modelo2 = RandomForestRegressor(max_depth=8)

        modelo1.fit(X_train, y_train)
        modelo2.fit(X_train, y_train)
        
        # Fazendo previsões no conjunto de treino
        y_pred1 = modelo1.predict(X_train)
        y_pred2 = modelo2.predict(X_train)  
        
        st.write('Criamos 2 modelos de testes. O Modelo1 com profundidade máxima da árvore de decisão de 2. O Modelo2 com profundidade máxima da árvore de decisão de 8.')
        st.write('<b>Calculando o Erro Quadrático Médio (MSE) para ambos os modelos:</b>',unsafe_allow_html=True)
      
        MSE1 = mean_squared_error(y_train, y_pred1)
        MSE2 = mean_squared_error(y_train, y_pred2)
        
        st.write(f'MSE do modelo 1 (max_depth=2) é {MSE1}')
        st.write(f'MSE do modelo 2 (max_depth=8) é {MSE2}')
        st.write('<b><i>MSE Avalia a precisão do modelo prever os dados já observados.</i></b>',unsafe_allow_html=True)
     
        df_avaliacao1 = pd.DataFrame({'Valores Reais':y_train, 'Valores Preditos':y_pred1 })
        df_avaliacao2 = pd.DataFrame({'Valores Reais':y_train, 'Valores Preditos':y_pred2 })

        # Plotando o gráfico
        plt.figure(figsize=(10,6))
        fig = sns.scatterplot(x='Valores Reais', y='Valores Preditos', data=df_avaliacao2, color='orange')

        # Adicionando uma linha de tendência (diagonal perfeita para comparação)
        max_valor2 = max(df_avaliacao2.max()) # Pegar o maior valor entre reais e preditos para ajustar a linha
        fig = plt.plot([0, max_valor2], [0, max_valor2], color='blue', linestyle='--') # Linha 45º

        # Títulos e rótulos
        plt.title('Valores Reais vs Valores Preditos')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')

        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)     

    with col2:
        
        st.subheader('Modelo de Teste')   
        # Rodando o modelo de teste
        modelo3 = RandomForestRegressor (max_depth=8, min_samples_leaf=20)
        modelo3.fit(X_test, y_test) 

        y_pred = modelo3.predict(X_test)

        MSE3 = mean_squared_error(y_test, y_pred)
        st.write(f'MSE do Modelo de Teste (max_depth=8) é {MSE3}')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
        
        df_avaliacao = pd.DataFrame({'Valores Reais':y_test, 'Valores Preditos':y_pred })
        #st.dataframe(df_avaliacao.style.highlight_max(axis=0))
         
        # Plotando o gráfico
        plt.figure(figsize=(10,6))
        fig = sns.scatterplot(x='Valores Reais', y='Valores Preditos', data=df_avaliacao, color='orange')

        # Adicionando uma linha de tendência (diagonal perfeita para comparação)
        max_valor = max(df_avaliacao.max()) # Pegar o maior valor entre reais e preditos para ajustar a linha
        fig = plt.plot([0, max_valor], [0, max_valor], color='blue', linestyle='--') # Linha 45º

        # Títulos e rótulos
        plt.title('Valores Reais vs Valores Preditos')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')

        st.pyplot(fig=plt, clear_figure=True, use_container_width=True)       
           

