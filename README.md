# Depth-Anything-3D
モデルダウンロードリンク：https://huggingface.co/LiheYoung/depth_anything_vitl14

Depth Anything：https://github.com/LiheYoung/Depth-Anything

## Pythonライブラリ
torchがデフォルトだとおそらくCPU版が入るので、自分でGPU版をインストールしてください。
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 実行方法
```
streamlit run app.py
```
