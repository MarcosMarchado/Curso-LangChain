import streamlit as st

st.title("Meu App com Streamlit")

nome = st.text_input("Digite seu nome:")
if nome:
    st.write(f"Olá, {nome}!")

numero = st.slider("Escolha um número", 0, 100, 25)
st.write(f"O quadrado do número é {numero**2}")
