import streamlit as st
import os
import shutil


st.title("3D画像生成")

temp_folder = "./tmp"
result_folder = "./result"

if(st.button("セットアップ")):
    
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)



if(st.button("削除")):
    shutil.rmtree(temp_folder)
    shutil.rmtree(result_folder)
