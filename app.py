import streamlit as st
from pandas import DataFrame
import pickle

# Ler dados do modelo treinado
modelo = pickle.load(open('data/modelo_titanic.pkl', 'rb'))
logit = modelo['resultados']
escala = modelo['escala']

# Titulo do APP
st.title("Classificador de sobrevivência no Titanic")
with st.expander('Modelo', expanded=False):
    st.markdown('''
            ## Apresentação do Protótipo
            Este Aplicativo possibilita que se faça previsões sobre a sobrevivência no desastre marítmo do Titanic em 1912 usando uma modelagem de `Machine Learning`.

            ## Modelo canônico
            Regressão logística
            
            $$P(Y=1) = \dfrac{exp(Xb+u)}{1+exp(Xb+u)}$$

            ''')

st.header("Dados de Entrada")
# Formulário
sexo = st.selectbox("Informe seu sexo (SEX):", ['Feminino', 'Masculino'])
idade = st.number_input("Informe sua idade (AGE):", value=30)
sibsp = st.number_input("Informe a quantidade de parentes (SIBSP):", value=0,
                        min_value=0, max_value=8)

idade_n = (idade - escala[0]['age'][0])/(escala[0]['age'][1] - escala[0]['age'][0])

sibsp_n = (sibsp - escala[1]['sibsp'][0])/(escala[1]['sibsp'][1] - escala[1]['sibsp'][0])

st.write(f"Idade declarada {idade}. Sexo declarado: {sexo}. Parentes declarados: {sibsp}")

# Data frame: atributos do modelo treinado
data = DataFrame({
        'age': [idade_n], 'sibsp': [sibsp_n], 'parch': [0], 'fare': [0.412503],
        'pclass_2': [False], 'pclass_3': [False], 
        'sex_male': [sexo=="Masculino"],
        'embarked_Q': [False], 'embarked_S': [True]
    })

st.table(data)

if st.button('Simular'):
    st.header('Resultados')
    classe = logit.predict(data)
    probabilidade = logit.predict_proba(data)

    r = DataFrame({'Probabilidade': probabilidade[0],
                   'Classe': ['Morreria', 'Sobreviveria']})

    resposta = "Morreria"
    if classe==1:
        resposta = "Sobreviveria"
        st.balloons()

    st.write(f"Você {resposta.upper()} ao desastre do Titanic")
    st.bar_chart(data=r, x="Classe", y="Probabilidade",
                 color=["#FF0000"])

