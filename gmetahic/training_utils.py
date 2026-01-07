import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime


def train_bulk_only(
        train_loader, model, criterion, optimizer, device, min_stripe_signal
):
    model.train()
    running_loss = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for X1, X1_rev, X2, X2_rev, y_true, y_true_rev in tepoch:
            optimizer.zero_grad()

            # left interactions
            X1 = X1.to(device)
            X2 = X2.to(device)
            X = torch.cat((X1, X2), axis=1)
            y_true = y_true.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y_true, min_stripe_signal)

            # right interactions
            X1_rev = X1_rev.to(device)
            X2_rev = X2_rev.to(device)
            X_rev = torch.cat((X1_rev, X2_rev), axis=1)
            y_true_rev = y_true_rev.to(device)
            y_hat_rev = model(X_rev)
            loss += criterion(y_hat_rev, y_true_rev, min_stripe_signal)

            running_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss


def validate_bulk_only(val_loader, model, criterion, device, min_stripe_signal):
    model.eval()
    running_loss = 0

    for X1, X1_rev, X2, X2_rev, y_true, y_true_rev in val_loader:
        # left interactions
        X1 = X1.to(device)
        X2 = X2.to(device)
        X = torch.cat((X1, X2), axis=1)
        y_true = y_true.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y_true, min_stripe_signal)

        # right interactions
        X1_rev = X1_rev.to(device)
        X2_rev = X2_rev.to(device)
        X_rev = torch.cat((X1_rev, X2_rev), axis=1)
        y_true_rev = y_true_rev.to(device)
        y_hat_rev = model(X_rev)
        loss += criterion(y_hat_rev, y_true_rev, min_stripe_signal)

        running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader.dataset)

    return model, epoch_loss


def train(train_loader, model, criterion, optimizer, device, min_stripe_signal):
    model.train()
    running_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for Z2, Z2_rev, X1, X1_rev, X2, X2_rev, y_true, y_true_rev in tepoch:
            optimizer.zero_grad()

            # left interactions
            X1 = X1.to(device)
            X2 = X2.to(device)
            X = torch.cat((X1, X2), axis=1)
            Z2 = Z2.to(device)
            y_true = y_true.to(device)

            y_hat = model(Z2, X)
            loss = criterion(y_hat, y_true, min_stripe_signal)

            # right interactions
            X1_rev = X1_rev.to(device)
            X2_rev = X2_rev.to(device)
            X_rev = torch.cat((X1_rev, X2_rev), axis=1)
            Z2_rev = Z2_rev.to(device)
            y_true_rev = y_true_rev.to(device)

            y_hat_rev = model(Z2_rev, X_rev)
            loss += criterion(y_hat_rev, y_true_rev, min_stripe_signal)

            running_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        return model, optimizer, epoch_loss


def validate(val_loader, model, criterion, device, min_stripe_signal):
    """
    Function for the validation step of the training loop
    """
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for Z2, Z2_rev, X1, X1_rev, X2, X2_rev, y_true, y_true_rev in val_loader:
            # left interactions
            X1 = X1.to(device)
            X2 = X2.to(device)
            X = torch.cat((X1, X2), dim=1)
            Z2 = Z2.to(device)
            y_true = y_true.to(device)

            y_hat = model(Z2, X)
            loss = criterion(y_hat, y_true, min_stripe_signal)

            # right interactions
            X1_rev = X1_rev.to(device)
            X2_rev = X2_rev.to(device)
            X_rev = torch.cat((X1_rev, X2_rev), axis=1)
            Z2_rev = Z2_rev.to(device)
            y_true_rev = y_true_rev.to(device)

            y_hat_rev = model(Z2_rev, X_rev)
            loss += criterion(y_hat_rev, y_true_rev, min_stripe_signal)

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader.dataset)

    return model, epoch_loss


def training_loop(
        train,
        validate,
        model,
        ct_list,
        seed,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        epochs,
        patience,
        save_path,
        device,
        min_stripe_signal,
        print_every=1,
        clip_grad=None
):
    """
    Function defining the entire training loop
    """
    # set objects for storing metrics
    counter = 0
    best_loss = 1e10
    best_train_loss = 1e10
    train_losses = []
    valid_losses = []
    current_time = datetime.now()
    now_date = str(current_time.date())
    now_time = str(current_time.time()).split(".")[0].split(":")[0]
    ct_num = len(ct_list)

    SAVEPATH = os.path.join(
        save_path,
        "{}_CTCFmotifScore_seed{}_GMetaHiC_use_no_GB_{}cells_{}_{}h.pth.tar".format("_".join(ct_list), seed, ct_num, now_date,
                                                                           now_time)
    )

    # Train model
    for epoch in range(epochs):

        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device, min_stripe_signal
        )
        train_losses.append(train_loss)

        # validation
        model, valid_loss = validate(
            valid_loader, model, criterion, device, min_stripe_signal
        )
        valid_losses.append(valid_loss)

        counter += 1

        if valid_loss < best_loss:
            counter = 0
            best_loss = valid_loss
            print("++++saving+++++")
            torch.save(model.state_dict(), SAVEPATH)

        print(
            f"{datetime.now().time().replace(microsecond=0)} --- "
            f"Epoch: {epoch}\t"
            f"Train loss: {train_loss:.4f}\t"
            f"Valid loss: {valid_loss:.4f}\t"
        )

        if counter > patience:
            break

        scheduler.step()

    return model, optimizer, (train_losses, valid_losses)


def huber_loss(input, target, delta=1.0):
    """
    Huber损失函数 - 对异常值更鲁棒
    """
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)


def smooth_l1_loss(input, target):
    """
    Smooth L1损失函数 - PyTorch内置
    """
    return F.smooth_l1_loss(input, target, reduction='mean')


def focal_mse_loss(input, target, alpha=2.0, gamma=0.5):
    """
    聚焦MSE损失 - 对难样本给予更多关注
    """
    mse = (input - target) ** 2
    focal_weight = (1 + mse.detach() / alpha) ** gamma
    return torch.mean(focal_weight * mse)


def weighted_mse_loss(input, target, min_stripe_signal=-99999):
    """
    保持与原始版本相同的损失计算方式
    """
    # 保持原始的权重计算方式
    weight = target.sum(1) > min_stripe_signal
    loss = (input - target) ** 2
    loss = (weight * loss.T).T
    loss = torch.sum(loss)  # 返回总损失，与原版本一致
    return loss


def weighted_mse_loss_normalized(input, target, min_stripe_signal=-99999):
    """
    归一化版本的加权MSE损失 - 如果您希望使用平均损失
    """
    weight = (target.sum(1) > min_stripe_signal).float()
    loss = (input - target) ** 2
    weighted_loss = weight.unsqueeze(-1) * loss

    # 计算有效样本数量进行归一化
    valid_samples = weight.sum()
    if valid_samples > 0:
        return weighted_loss.sum() / valid_samples
    else:
        return weighted_loss.sum()


def robust_weighted_mse_loss(input, target, min_stripe_signal=-99999):
    """
    鲁棒版本的加权MSE损失 - 保持总损失的计算方式
    """
    weight = target.sum(1) > min_stripe_signal
    # 使用Smooth L1损失替代MSE，但保持总损失计算方式
    loss = nn.functional.smooth_l1_loss(input, target, reduction='none', beta=1.0)
    loss = (weight * loss.T).T
    loss = torch.sum(loss)
    return loss