import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams

class InspectionRevNet(nn.Module):
	def __init__(self, features=1024, features_cond=256, kernel_size=31):
		super().__init__()
		features = features // 2
		self.features = features
		self.f1 = nn.Sequential(
			nn.Conv1d(features + features_cond, features * 4, 1),
			nn.GLU(-2),
			nn.Conv1d(features * 2, features * 2, padding="same", kernel_size=kernel_size, groups=features * 2),
			nn.GLU(-2),
			nn.Conv1d(features, features, 1)
		)
		self.f2 = nn.Sequential(
			nn.Conv1d(features + features_cond, features * 4, 1),
			nn.GLU(-2),
			nn.Conv1d(features * 2, features * 2, padding="same", kernel_size=kernel_size, groups=features * 2),
			nn.GLU(-2),
			nn.Conv1d(features, features, 1)
		)
	def forward(self, x, cond):
		x1, x2 = torch.split(x, [self.features, self.features], dim=-2)
		x2 = x2 + self.f1(torch.cat([x1, cond], dim=-2))
		x1 = x1 + self.f2(torch.cat([x2, cond], dim=-2))
		return torch.cat([x1, x2], dim=-2) # Y
	@torch.no_grad()
	def inverse(self, y, cond):
		y1, y2 = torch.split(y, [self.features, self.features], dim=-2)
		y1 = y1 - self.f2(torch.cat([y2, cond], dim=-2))
		y2 = y2 - self.f1(torch.cat([y1, cond], dim=-2))
		return torch.cat([y1, y2], dim=-2)
	@torch.enable_grad()
	def gradient(self, x, y, cond, grad_y):
		x = x.clone().detach().requires_grad_(True)
		cond = cond.clone().detach().requires_grad_(True)
		gradable = self.forward(x, cond)
		gradable.backward(grad_y)
		grad_x = x.grad.clone().detach()
		grad_cond = cond.grad.clone().detach()
		return grad_x, grad_cond

def GradOS_Metaclass(modules):
	class Inspection_GradOS(torch.autograd.Function):
		cv_modules = modules
		@classmethod
		def forward(self, ctx, x, cond):
			# print(self.cv_modules)
			y = x
			for layer in self.cv_modules:
				y = layer(y, cond)
			ctx.save_for_backward(y, cond)
			return y
		@classmethod
		def backward(self, ctx, grad_output):
			x, cond = ctx.saved_tensors
			grad = grad_output
			grad_cond = []
			for layer in (self.cv_modules[::-1]):
				y = x
				x = layer.inverse(x, cond)
				grad, _grad_cond = layer.gradient(x, y, cond, grad)
				grad_cond.append(_grad_cond)
			grad_cond = torch.sum(torch.stack(grad_cond, dim=0), dim=0)
			return grad, grad_cond
	return Inspection_GradOS

class Inspection(nn.Module):
	def __init__(self, in_dims, n_feats, *, num_channels=1024, num_layers=35, kernel_size=31):
		super().__init__()
		self.in_dims = in_dims
		self.n_feats = n_feats
		self.spec_projection = nn.Conv1d(in_dims * n_feats, num_channels, 1)
		self.diffusion_projection = nn.Sequential(
			SinusoidalPosEmb(hparams['hidden_size']),
			nn.Linear(hparams['hidden_size'], hparams['hidden_size'] * 4),
			nn.GLU(-1),
			nn.Linear(hparams['hidden_size'] * 2, hparams['hidden_size'] * 2),
			nn.GLU(-1)
		)
		self.out_projection = nn.Conv1d(num_channels, in_dims * n_feats, 1)
		self.w_modules = nn.ModuleList([
			InspectionRevNet(num_channels, hparams['hidden_size'], kernel_size) for _ in range(0, num_layers)
		])
	def forward(self, spec, step, cond):
		if self.n_feats == 1:
			x = spec[:, 0]
		else:
			x = spec.flatten(start_dim=1, end_dim=2)
		
		x = self.spec_projection(x)
		cond = cond + self.diffusion_projection(step).unsqueeze(-1)
		
		if self.training:
			x = GradOS_Metaclass(self.w_modules).apply(x, cond)
		else:
			for layer in self.w_modules:
				x = layer(x, cond)
		
		x = self.out_projection(x)
		
		if self.n_feats == 1:
			x = x[:, None, :, :]
		else:
			x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
		return x
