import torch

import train_func as tf
import utils

from itertools import combinations


class RoughLearning(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01, alpha=0):
        super(RoughLearning, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.alpha = 0

    def compute_discrimn_loss_rough(self, Z):
        """Empirical Discriminative Loss."""
        d, m = Z.shape
        scalar = d / (m ** (1 + self.alpha) * self.eps)

        if d > m:
            Z = Z.T
        I = torch.eye(Z.shape[0]).cuda()

        logdet = torch.logdet(I + self.gam1 * scalar * Z.matmul(Z.T))
        return logdet / 2.

    def compute_compress_loss_rough(self, Z, Pi):
        """Empirical Compressive Loss."""
        d, m = Z.shape
        k, _, _ = Pi.shape

        compress_loss = 0.
        for j in range(k):
            Z_j = Z.matmul(Pi[j])
            # delete the zero column
            non_zero_columns = torch.any(Z_j != 0, dim=0)
            Z_j = Z_j[:, non_zero_columns]

            m_j = Z_j.shape[1]
            if m_j == 0:
                continue

            scalar = d / (m_j ** (1 + self.alpha) * self.eps)

            if d > m_j:
                Z_j = Z_j.T
            I = torch.eye(Z_j.shape[0]).cuda()
            log_det = torch.logdet(I + scalar * Z_j.matmul(Z_j.T))
            compress_loss += log_det * m_j / m
        return compress_loss / 2.

    def compute_discrimn_loss_precise(self, Z):
        """Theoretical Discriminative Loss."""
        d, m = Z.shape
        scalar = d / (m * self.eps)

        if d > m:
            Z = Z.T
        I = torch.eye(Z.shape[0]).cuda()

        logdet = torch.logdet(I + scalar * Z.matmul(Z.T))
        return logdet / 2.

    def compute_compress_loss_precise(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, m = Z.shape
        k, _, _ = Pi.shape

        compress_loss = 0.
        for j in range(k):
            Z_j = Z.matmul(Pi[j])
            # delete the zero column
            non_zero_columns = torch.any(Z_j != 0, dim=0)
            Z_j = Z_j[:, non_zero_columns]

            m_j = Z_j.shape[1]
            if m_j == 0:
                continue
            scalar = d / (m_j * self.eps)

            if d > m_j:
                Z_j = Z_j.T
            I = torch.eye(Z_j.shape[0]).cuda()

            log_det = torch.logdet(I + scalar * Z_j.matmul(Z_j.T))
            compress_loss += m_j / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        Z = X.T
        Pi = tf.label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_rough = self.compute_discrimn_loss_rough(Z)
        compress_loss_rough = self.compute_compress_loss_rough(Z, Pi)

        discrimn_loss_precise = self.compute_discrimn_loss_precise(Z)
        compress_loss_precise = self.compute_compress_loss_precise(Z, Pi)

        total_loss_rough = self.gam2 * -discrimn_loss_rough + compress_loss_rough
        total_loss_precise = -discrimn_loss_precise + compress_loss_precise

        return (total_loss_rough,
                [discrimn_loss_rough.item(), compress_loss_rough.item()],
                [discrimn_loss_precise.item(), compress_loss_precise.item()], total_loss_precise)
