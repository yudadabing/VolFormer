Requirement: 
torch >= 1.11.0
h5py >=3.11.0
einops >= 0.8.0
numpy >= 1.22.4
pillow >= 9.1.1
tensorboardX >= 2.6.2.2
scikit-image >= 0.21.0
timm >= 0.9.16
torchnet >= 0.0.4
tqdm >= 4.61.2
opencv-python >= 4.9.0.80


Train:
python mains.py train 

Test:
python mains.py test 

VolFormer.py  denotes the shared Transformer and contains the volumetric self-attention.

