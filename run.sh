python infer.py name='same_norm' datamodule.aug=0
python run.py name='gaussian_blur' datamodule.aug=2
python run.py name='resize224' datamodule.aug=1 datamodule.resize=[224,224] datamodule.batch_size=64 epoch=10 resume=True path.pretrained='/home/youngkim21/dacon/dacon-sem/output/weights/resize224-04:17:45:09/11.pth'
python infer.py name='resize224_infer' datamodule.aug=1 datamodule.resize=[224,224] datamodule.batch_size=64 resume=True path.pretrained='/home/youngkim21/dacon/dacon-sem/output/weights/resize224-05:01:13:06/21_last.pth'