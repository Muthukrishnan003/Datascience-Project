import streamlit as st

st.header('Product Recommendation')

col = ['Rice','Wheat','Dhal','Groundnut','Almond','Icecream']

selected_movie = st.selectbox( "Type or select a Product from the dropdown", col )

st.button('Show Recommendation')
