import ffmpeg
import streamlit as st
import os

input_video = st.text_input("動画パス")  # 入力動画ファイルのパス
output_video = st.text_input("出力動画パス")
output_fps = st.number_input("フレームレート", value=24)

if st.button("実行"):
    (
        ffmpeg
        .input(input_video)
        .output(output_video, r=output_fps)
        .run()
    )
    st.write("実行完了")