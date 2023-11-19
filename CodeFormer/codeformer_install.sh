python ./CodeFormer/scripts/download_pretrained_models.py CodeFormer
python ./CodeFormer/scripts/download_pretrained_models.py facelib
python ./CodeFormer/scripts/download_pretrained_models.py dlib
pip install lpips

cd CodeFormer && python ./basicsr/setup.py develop
