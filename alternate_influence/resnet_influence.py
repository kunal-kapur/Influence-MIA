'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Influence(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Influence, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if return_features:
            return out, features
        return out

    def get_last_layer_grad(self, x, y, criterion):
        """Returns a flat gradient vector for self.linear params."""
        self.eval()
        self.zero_grad(set_to_none=True)
        out, _ = self.forward(x, return_features=True)
        loss = criterion(out, y)
        grads = torch.autograd.grad(loss, tuple(self.linear.parameters()),
                                    create_graph=False)
        return torch.cat([g.reshape(-1) for g in grads])


    def compute_last_layer_hessian(self, loader, device, damping=1e-4):
        """
        Builds the exact empirical Hessian over the linear classifier (including bias).
        Vectorized using einsum to run efficiently on A10 GPUs.
        """
        self.eval()
        
        # Dimensions: 10 classes, 513 features (512 + 1 for bias)
        num_classes = self.num_classes
        num_features = self.linear.in_features + 1
        param_dim = num_classes * num_features
        
        # Initialize H on the GPU
        H = torch.zeros((param_dim, param_dim), device=device)
        n = 0
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits, feats = self.forward(x, return_features=True)
                probs = torch.softmax(logits, dim=1)  # [B, C]
                B = feats.size(0)
                
                # Augment features with 1 for bias term: [B, 513]
                feats_aug = torch.cat([feats, torch.ones(B, 1, device=device)], dim=1)
                
                # 1. Compute P_cov for the entire batch at once: [B, C, C]
                P_cov_batch = torch.diag_embed(probs) - torch.einsum('bi,bj->bij', probs, probs)
                
                # 2. Compute the batched Kronecker product and sum over the batch dimension
                # 'bij' is the [B, C, C] class covariance
                # 'bk' and 'bl' are the [B, F] augmented features
                # Resulting shape 'ikjl' directly maps to the Kronecker layout [C, F, C, F]
                H_batch = torch.einsum('bij,bk,bl->ikjl', P_cov_batch, feats_aug, feats_aug)
                
                # Reshape to [C*F, C*F] and accumulate
                H += H_batch.reshape(param_dim, param_dim)
                n += B

        H = H / n
        # Damping for numerical stability
        H += damping * torch.eye(H.size(0), device=device)
        return H


    

    def influence_score(self, x_train, y_train, x_test, y_test,
                        criterion, H_inv, device):
        """
        I(z_train, z_test) = -grad_test^T @ H_inv @ grad_train
        Positive means removing z_train would HURT performance on z_test.
        Negative means removing z_train would HELP  performance on z_test.
        """
        g_train = self.get_last_layer_grad(x_train, y_train, criterion)
        g_test  = self.get_last_layer_grad(x_test,  y_test,  criterion)
        return -(g_test @ H_inv @ g_train).item()
    
    def _collect_feats_and_labels(self, dataloader, device):
        """Pass data through the backbone and return (all_feats, all_labels, all_feats_aug)."""
        self.eval()
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                _, feats = self.forward(x, return_features=True)
                all_feats.append(feats)
                all_labels.append(y.to(device))
        all_feats = torch.cat(all_feats, dim=0)    # [N, F]
        all_labels = torch.cat(all_labels, dim=0)  # [N]
        N = all_feats.size(0)
        all_feats_aug = torch.cat([all_feats, torch.ones(N, 1, device=device)], dim=1)  # [N, F+1]
        return all_feats, all_labels, all_feats_aug

    def _get_last_layer_grad_matrices(self, all_feats_aug, all_labels):
        """
        Single linear pass returning both gradient matrices.

        G_loss[n] = (p_n - e_{y_n}) ⊗ f_n        (gradient of CE loss)
        G_lira[n] = G_loss[n] / (p_true_n - 1)   (gradient of LiRA logit statistic)

        Returns:
            G_loss : [N, P]   P = num_classes * (in_features + 1)
            G_lira : [N, P]
        """
        with torch.no_grad():
            logits = self.linear(all_feats_aug[:, :-1])        # [N, C]
            probs  = torch.softmax(logits, dim=1)               # [N, C]
            y_onehot = F.one_hot(all_labels, num_classes=self.num_classes).float()
            d_logits = probs - y_onehot                         # [N, C]  ∇_logits CE

            p_true = probs.gather(1, all_labels.unsqueeze(1))   # [N, 1]
            p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            d_logits_lira = d_logits / (p_true - 1.0)          # [N, C]  ∇_logits LiRA

            N = all_feats_aug.size(0)
            G_loss = torch.einsum('nc,nf->ncf', d_logits,      all_feats_aug).reshape(N, -1)
            G_lira = torch.einsum('nc,nf->ncf', d_logits_lira, all_feats_aug).reshape(N, -1)
        return G_loss, G_lira  # each [N, P]

    def compute_influence_matrices(self, dataloader, H_inv, train_dataset_size, device):
        """
        Computes both influence matrices in a single forward pass, sharing H_inv @ G_loss.T:

          C_lira = -(1/N_train) * G_lira @ H_inv @ G_loss.T   (LiRA-vs-loss)
          C_loss = -(1/N_train) * G_loss @ H_inv @ G_loss.T   (loss-vs-loss)

        Returns:
            C_lira : [N, N]
            C_loss : [N, N]
        """
        self.eval()
        _, all_labels, all_feats_aug = self._collect_feats_and_labels(dataloader, device)
        G_loss, G_lira = self._get_last_layer_grad_matrices(all_feats_aug, all_labels)
        # Compute H_inv @ G_loss.T once and reuse for both matrices
        H_inv_Gl_T = H_inv @ G_loss.T                           # [P, N]
        C_lira = -(G_lira @ H_inv_Gl_T) / train_dataset_size
        C_loss = -(G_loss @ H_inv_Gl_T) / train_dataset_size
        return C_lira, C_loss

    # Thin wrapper kept for any external callers that only want C_lira.
    def compute_influence_matrix(self, dataloader, H_inv, train_dataset_size, device):
        C_lira, _ = self.compute_influence_matrices(dataloader, H_inv, train_dataset_size, device)
        return C_lira
    
    
    def get_lira_statistics(self, dataloader, device):
        """
        Computes the base LiRA statistic for every point in the dataloader.
        Returns a 1D tensor of length N containing the scaled logits.
        """
        self.eval()
        t_bases = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass to get raw logits [Batch_Size, Num_Classes]
                logits = self.forward(x, return_features=False)
                
                # Get the probabilities using softmax
                probs = torch.softmax(logits, dim=1)
                
                # Extract the probability of the *true* class for each image
                # gather() pulls the probability at the index specified by y
                p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                
                # Clamp p_true to avoid log(0) or log(1) resulting in infinity
                p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
                
                # Compute the LiRA scaled logit: log( p / (1-p) )
                t = torch.log(p_true / (1.0 - p_true))
                
                t_bases.append(t)
                
        # Concatenate into a single 1D tensor of size N
        return torch.cat(t_bases, dim=0)




def ResNet18_Influence():
    return ResNet_Influence(BasicBlock, [2, 2, 2, 2])



def test():
    net = ResNet18_Influence()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()