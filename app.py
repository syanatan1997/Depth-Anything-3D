import streamlit as st
import image_geneate_depthany as imd
import os

st.title("3D画像生成")

reverse = st.checkbox("逆")

if(st.button("実行")):
    img_path = "assets\inputs"
    outdir = "assets\outputs"

    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames]
        filenames.sort()
    

    depth_paths = imd.generate_depth(filenames, outdir)
    imd.generate_all_stereo_images(filenames, depth_paths, outdir, reverse)

    st.write("実行完了")


