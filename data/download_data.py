"""Download the chest X-ray dataset (Kaggle: paultimothymooney/chest-xray-pneumonia).

Setup once: kaggle.com -> Settings -> API -> Create New Token, then save kaggle.json to
  Linux/Mac: ~/.kaggle/kaggle.json     Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
Run:  python download_data.py     -> creates ../chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}
"""
import subprocess
import sys

cmd = ["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia",
       "--unzip", "-p", ".."]
sys.exit(subprocess.call(cmd))
