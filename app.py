import streamlit as st
import os

st.title("3D画像生成")

if(st.button("セットアップ")):
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)



    