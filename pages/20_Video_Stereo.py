import streamlit as st
import src.image_geneate as ImageGenerate
import os

st.title("3D画像生成")

input_video_path = st.text_input("動画パス")
output_dir = st.text_input("出力画像パス")
model_name = st.selectbox("モデル", options=['vits', 'vitb', 'vitl'])
reverse = st.checkbox("Reverse SBS有効化")

if(st.button("Depth生成")):
    ImageGenerate.generate_stereo_video(input_video_path, output_dir, model_name)