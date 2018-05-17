import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

class DCGAN(nn.Module):

	def __init__(self, input_size, output_size):

		self.generator = Generator()
		self.discriminator = Discriminator()

	def forward(self, x):

		pass

class Generator(nn.Module):

	def __init__(self, input_size, output_size):

		self.input_size = input_size
		self.output_size = output_size

		self.conv1 = nn.ConvTranspose2D(in_channels=1, out_channels=1024,
							   kernel_size=4)
		self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
			                   kernel_size=5, stride=2)
		self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
							   kernel_size=5, stride=2)
		self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
							   kernel_size=5, stride=2)
		self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=3,
							   kernel_size=5, stride=2)

	def forward(self, x):

		x = self.fc(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)

		return x

class Discriminator(nn.Module):

	def __init__(self, input_size, output_size=2):

		pass

	def forward(self, x):

		pass


if __name__ == "__main__":
	model = DCGAN()
