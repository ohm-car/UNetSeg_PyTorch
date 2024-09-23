import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabHead

class VOCModel(nn.Module):
	"""docstring for VOCModel"""
	def __init__(self, arg, num_classes = 20, im_res = 224):
		super(VOCModel, self).__init__()
		self.arg = arg
		self.num_classes = num_classes

		self.parts = deeplabv3_resnet50(aux_loss=True)

		self.backbone = self.parts.backbone
		self.classifiers = list()

		for i in self.num_classes:
			_c = DeepLabHead(2048, 2)
			self.classifiers.append(_c)

	def forward(self, x, a):

		_temp = self.backbone(x)
		_features = _temp['out']
		recon_imgs = _temp['aux']
		masks = list()

		for i in xrange(self.num_classes):
			if a[i] == 1:
				masks.append(self.classifiers[i](_features))

		return masks, recon_imgs
