import torch
from torch import nn


class MseDirectionLoss(nn.Module):
    """Define MSE + Direction loss."""
    def __init__(self, lamda):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_pred, output_real):
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[
            6], output_pred[9], output_pred[12]
        y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[
            9], output_real[12]

        # different terms of loss
        abs_loss_0 = self.criterion(y_pred_0, y_0)
        loss_0 = torch.mean(1 - self.similarity_loss(
            y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1)))
        abs_loss_1 = self.criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - self.similarity_loss(
            y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = self.criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - self.similarity_loss(
            y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = self.criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - self.similarity_loss(
            y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))

        total_loss = loss_0 + loss_1 + loss_2 + loss_3 + self.lamda * (
            abs_loss_0 + abs_loss_1 + abs_loss_2 + abs_loss_3)

        return total_loss


class DirectionOnlyLoss(nn.Module):
    """Define Direction loss."""
    def __init__(self):
        super(DirectionOnlyLoss, self).__init__()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_pred, output_real):
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[
            6], output_pred[9], output_pred[12]
        y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[
            9], output_real[12]

        # different terms of loss
        loss_0 = torch.mean(1 - self.similarity_loss(
            y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1)))
        loss_1 = torch.mean(1 - self.similarity_loss(
            y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        loss_2 = torch.mean(1 - self.similarity_loss(
            y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        loss_3 = torch.mean(1 - self.similarity_loss(
            y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))

        total_loss = loss_0 + loss_1 + loss_2 + loss_3

        return total_loss
