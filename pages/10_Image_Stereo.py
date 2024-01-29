import streamlit as st
import src.image_geneate as ImageGenerate
import os

st.title("3D画像生成")

input_dir = st.text_input("入力画像パス")
output_dir = st.text_input("出力画像パス")
model_name = st.selectbox("モデル", options=['vits', 'vitb', 'vitl'])
reverse = st.checkbox("Reverse SBS有効化")

if(st.button("3D画像生成実行")):
    output_depth_path = os.path.join(output_dir, "depth")
    if not os.path.exists(output_depth_path):
        os.makedirs(output_depth_path)

    output_stereo_path = os.path.join(output_dir, "stereo")
    if not os.path.exists(output_stereo_path):
        os.makedirs(output_stereo_path)


    image_files = ImageGenerate.get_image_filenames(input_dir)
    depth_files = ImageGenerate.generate_depth_image(image_files, output_depth_path, model_name)
    ImageGenerate.generate_all_stereo_images(image_files, depth_files, output_stereo_path, reverse)
    
    st.write("実行完了")


