import torch
from torch import nn
import torch.nn.functional as F

class DCGAN(nn.Module):
	"""
	General DCGAN wrapper to contain both the generator and the discriminator.
	"""
	def __init__(self, generator_type='deconv'):
		"""
		Parameters
		----------
		generator_type : string
		    Type of generator used. 
		    Has to be in ['deconv', 'nearest_neighbor', 'bilinear']
		    default : 'deconv'
		"""
		super(DCGAN, self).__init__()

		if generator_type == 'deconv':
		    self.generator = DeconvGenerator()
		elif generator_type == 'nearest_neighbor':
		    self.generator = NNGenerator()
		elif generator_type == 'bilinear':
		    self.generator = BilinearGenerator()
		    
		self.name = 'dcgan_' + generator_type
		    
		self.discriminator = Discriminator()
    
	def forward(self, x, fake):
		"""
		Parameters
		----------
		fake : boolean
		    If the data is fake, we pass it through the generator.
		"""

		if fake:
		    x = self.generator(x)
		    
		x = self.discriminator(x)

		return x

class DeconvGenerator(nn.Module):
	"""
	Generator using Deconvolution (transposed convolution)
	"""
	def __init__(self):
		super(DeconvGenerator, self).__init__()

		# output size is 4*4*1024
		self.project = nn.Linear(in_features=100, out_features=16384)

		# 1st conv layer
		self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
		                                kernel_size=5, stride=2, padding=2,
		                                output_padding=1)
		self.batchnorm1 = nn.BatchNorm2d(num_features=512)

		# 2nd conv layer
		self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
		                                kernel_size=5, stride=2, padding=2,
		                                output_padding=1)
		self.batchnorm2 = nn.BatchNorm2d(num_features=256)

		# 3rd conv layer
		self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
		                                kernel_size=5, stride=2, padding=2,
		                                output_padding=1)
		self.batchnorm3 = nn.BatchNorm2d(num_features=128)

		# 4th conv layer
		self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=3,
		                                kernel_size=5, stride=2, padding=2,
		                                output_padding=1)
    
	def forward(self, x):
		assert x.shape[1] == 100

		x = self.project(x).view(-1, 1024, 4, 4)

		# 1st conv layer
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.relu(x)

		# 2nd conv layer
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = F.relu(x)

		# 3rd conv layer
		x = self.conv3(x)
		x = self.batchnorm3(x)
		x = F.relu(x)

		# 4th conv layer
		x = self.conv4(x)

		# to print images, they have to be bounded between 0 and 1
		x = F.sigmoid(x)
		# x = F.tanh(x)

		return x

class NNGenerator(nn.Module):
	"""
	Generator using Nearest-Neighbor Upsampling
	"""
	def __init__(self):
		super(NNGenerator, self).__init__()

		# output size is 4*4*1024
		self.project = nn.Linear(in_features=100, out_features=16384)

		# 1st conv layer
		self.upsample1 = nn.Upsample(size=4, scale_factor=2, mode='nearest')
		self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=5,
		                       padding=2)
		self.batchnorm1 = nn.BatchNorm2d(num_features=512)

		# 2nd conv layer
		self.upsample2 = nn.Upsample(size=8, scale_factor=2, mode='nearest')
		self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5,
		                       padding=2)
		self.batchnorm2 = nn.BatchNorm2d(num_features=256)

		# 3rd conv layer
		self.upsample3 = nn.Upsample(size=16, scale_factor=2, mode='nearest')
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5,
		                       padding=2)
		self.batchnorm3 = nn.BatchNorm2d(num_features=128)

		# 4th conv layer
		self.upsample4 = nn.Upsample(size=32, scale_factor=2, mode='nearest')
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5,
		                       padding=2)
      
	def forward(self, x):
		assert x.shape[1] == 100

		x = self.project(x).view(-1, 1024, 4, 4)

		# 1st conv layer
		x = self.upsample1(x)
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.relu(x)

		# 2nd conv layer
		x = self.upsample2(x)
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = F.relu(x)

		# 3rd conv layer
		x = self.upsample3(x)
		x = self.conv3(x)
		x = self.batchnorm3(x)
		x = F.relu(x)

		# 4th conv layer
		x = self.upsample4(x)
		x = self.conv4(x)

		# to print images, they have to be bounded between 0 and 1
		x = F.sigmoid(x)
		# x = F.tanh(x)

		return x

class BilinearGenerator(nn.Module):
	"""
	Generator using bilinear upsampling
	"""
	def __init__(self):
		super(BilinearGenerator, self).__init__()

		# output size is 4*4*1024
		self.project = nn.Linear(in_features=100, out_features=16384)
		#self.batchnorm0 = nn.BatchNorm2d(num_features=1024)
		#self.dropout = nn.Dropout()

		# 1st conv layer
		self.upsample1 = nn.Upsample(size=4, scale_factor=2, mode='bilinear')
		self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=5,
		                       padding=2)
		self.batchnorm1 = nn.BatchNorm2d(num_features=512)

		# 2nd conv layer
		self.upsample2 = nn.Upsample(size=8, scale_factor=2, mode='bilinear')
		self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5,
		                       padding=2)
		self.batchnorm2 = nn.BatchNorm2d(num_features=256)

		# 3rd conv layer
		self.upsample3 = nn.Upsample(size=16, scale_factor=2, mode='bilinear')
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5,
		                       padding=2)
		self.batchnorm3 = nn.BatchNorm2d(num_features=128)

		# 4th conv layer
		self.upsample4 = nn.Upsample(size=32, scale_factor=2, mode='bilinear')
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5,
		                       padding=2)
      
	def forward(self, x):
		assert x.shape[1] == 100

		x = self.project(x).view(-1, 1024, 4, 4)
		#x = self.batchnorm0(x)
		x = F.relu(x)

		# 1st conv layer
		x = self.upsample1(x)
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.relu(x)

		# 2nd conv layer
		x = self.upsample2(x)
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = F.relu(x)

		# 3rd conv layer
		x = self.upsample3(x)
		x = self.conv3(x)
		x = self.batchnorm3(x)
		x = F.relu(x)

		# 4th conv layer
		x = self.upsample4(x)
		x = self.conv4(x)

		# to print images, they have to be bounded between 0 and 1
		x = F.sigmoid(x)
		#x = F.tanh(x)
		return x

class Discriminator(nn.Module):
	"""
	Discriminator using normal convolutions. It is symmetric to the different generators.
	"""
	def __init__(self):
		super(Discriminator, self).__init__()

		# 1st conv layer
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5,
		                       stride=2, padding=2)
		self.batchnorm1 = nn.BatchNorm2d(num_features=128)

		# 2nd conv layer
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5,
		                       stride=2, padding=2)
		self.batchnorm2 = nn.BatchNorm2d(num_features=256)

		# 3rd conv layer
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5,
		                       stride=2, padding=2)
		self.batchnorm3 = nn.BatchNorm2d(num_features=512)

		# 4th conv layer
		self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5,
		                       stride=2, padding=2)
		self.batchnorm4 = nn.BatchNorm2d(num_features=1024)

		#classifier
		self.clf = nn.Linear(in_features=16384, out_features=1)
    
	def forward(self, x, negative_slope=0.2):
        
		# 1st conv layer
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.leaky_relu(x, negative_slope=negative_slope)

		# 2nd conv layer
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = F.leaky_relu(x, negative_slope=negative_slope)

		# 3rd conv layer
		x = self.conv3(x)
		x = self.batchnorm3(x)
		x = F.leaky_relu(x, negative_slope=negative_slope)

		# 4th conv layer
		x = self.conv4(x)
		x = self.batchnorm4(x)
		x = F.leaky_relu(x, negative_slope=negative_slope)

		#classifier
		x = self.clf(x.view(-1, 16384))

		# we use BCEWithLogitsLoss which includes a sigmoid layer instead of BCELoss
		# x = F.sigmoid(x)

		return x

if __name__ == "__main__":
	model = DCGAN()
