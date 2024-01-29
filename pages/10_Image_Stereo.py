import streamlit as st
from src.image_geneate import ImageGenerate, Utils
import shutil
import os

st.title("3D画像生成")

temp_folder = "./tmp"
result_folder = "./result"

uploaded_files = st.file_uploader("アップロード画像選択", type=["jpg", "png"], accept_multiple_files=True)


encoder = st.selectbox("モデル", options=['vits', 'vitb', 'vitl'])
reverse = st.checkbox("Reverse SBS有効化")

if(st.button("3D画像生成実行")):
    utils = Utils()
    input_rand_num_ = utils.generate_random_string(10)
    output_rand_num_ = utils.generate_random_string(10)
    
    input_dir = os.path.join(temp_folder, input_rand_num_)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    output_dir = os.path.join(temp_folder, output_rand_num_)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(uploaded_files) == 0:
        st.error("No file were uploaded")

    for i in range(len(uploaded_files)):
        bytes_data = uploaded_files[i].read()  # read the content of the file in binary
        with open(os.path.join(input_dir, uploaded_files[i].name), "wb") as f:
            f.write(bytes_data)  # write this content elsewhere

    image_generate = ImageGenerate()
    image_generate.set_depth_anything(encoder)
    image_generate.set_reverse(reverse)
    image_generate.set_image_filenames(input_dir)
    image_generate.set_output_directory(output_dir)
    image_generate.generate_stereo_image()

    zip_basename = utils.generate_random_string(10)
    zip_filename = os.path.join(result_folder, zip_basename + '.zip')
    shutil.make_archive(os.path.join(result_folder, zip_basename), 'zip', output_dir)

    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

    with open(zip_filename, "rb") as f:
        st.download_button(
            label="Download ZIP",
            data=f,
            file_name=zip_basename + '.zip',
            mime="application/zip"
        )
    
    st.write("実行完了")


