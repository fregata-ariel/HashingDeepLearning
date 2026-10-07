import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class TripletNet(nn.Module):
	"""Learn a projection used as a data-dependent hash function.

	Paper mapping:
	    MONGOOSE (ICLR 2021), Section 3.3 "Learnable LSH" and Section 3.3.1.

	Implementation note:
	    This is the SLIDE-side experimental learner in the released code. The
	    projection has K*L outputs, corresponding to hash-function components
	    across L tables. Its forward objective is pairwise/BCE based; it is not
	    a literal implementation of the margin-based triplet Equation 3 used by
	    the Reformer-side TripletLSHAttention implementation.

	Traceability:
	    MONGOOSE-SLIDE-PAIRWISE-LOSS.

	Paper:
	    https://openreview.net/forum?id=wWK7yXkULyh
	"""

	def __init__(self, margin, K, L, layer_size):
		super(TripletNet, self).__init__()
		self.K = K
		self.L = L
		self.dense1 = nn.Linear(layer_size, K*L)
		self.init_weights(self.dense1.weight, self.dense1.bias)
		self.dense1.bias.requires_grad = False
		self.margin = margin
		self.device = device

	def init_weights(self, weight, bias):
		weight.data.normal_(0,1)
		bias.data.fill_(0)

	def forward(self, arc, pair, label):
		"""Compute the differentiable hash-agreement loss for labeled pairs.

		The linear projection is split into L table-sized chunks, passed through
		tanh as a smooth sign/hash surrogate, and compared by inner product.
		Binary cross entropy encourages the supplied positive/negative pair
		labels. This realizes the paper's Section 3.3 idea of learning hash
		functions from training signals, but with this repository's own pairwise
		objective.

		Traceability:
		    MONGOOSE-SLIDE-PAIRWISE-LOSS.
		"""

		emb_arc = self.dense1(arc)
		emb_pair = self.dense1(pair)

		# embedding chunk - L1
		emb_arc_chunk = torch.cat(torch.chunk(emb_arc, self.L, dim = 1 ))
		emb_pair_chunk = torch.cat(torch.chunk(emb_pair, self.L, dim = 1 ))

		label_chunk = label.repeat(self.L)
		assert emb_arc_chunk.size()==emb_pair_chunk.size()
		assert emb_arc_chunk.size()[0]==label_chunk.size()[0]
		
		beta = 1
		alpha = 0.5
		emb_arc_chunk = torch.tanh( beta * emb_arc_chunk)
		emb_pair_chunk = torch.tanh( beta * emb_pair_chunk)

		# agree_x_p = torch.mean(  ((emb_arc_chunk > 0)  == (emb_pair_chunk > 0 ))[label_chunk==1].float()  )
		# agree_x_n = torch.mean(  ((emb_arc_chunk > 0)  == (emb_pair_chunk > 0 ))[label_chunk==0].float()  )
		#
		# print("\nagree_x_p", agree_x_p)
		# print("agree_x_n", agree_x_n)

		output = torch.sum(emb_arc_chunk * emb_pair_chunk, dim= 1)
		assert output.size()==label_chunk.size()

		output_loss = F.binary_cross_entropy(F.sigmoid(output), label_chunk)
		# reg_loss=torch.mean(torch.norm(self.dense1.weight,dim=1))
		# loss = output_loss+0.05*reg_loss
		loss = output_loss
		return loss

