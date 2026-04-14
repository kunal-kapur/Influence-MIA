import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(args, params):
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        return optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if optimizer_name == "adam":
        return optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    schedule = args.lr_schedule.lower()
    if schedule == "none":
        return None
    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if schedule == "step":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.step_milestones,
            gamma=args.step_gamma,
        )
    raise ValueError(f"Unsupported lr_schedule: {args.lr_schedule}")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum()
        total += inputs.size(0)

    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct / total).item()
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.detach() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum()
            total += inputs.size(0)

    return (running_loss / total).item(), (correct / total).item()


def train(model, train_loader, val_loader, args, device, eval_interval=5):
    """Full training loop with configurable optimizer and scheduler."""
    criterion = nn.CrossEntropyLoss()
    # Hyperparameters are configured in cifar10.yaml (lr, batch_size, optimizer, scheduler, dropout, num_workers, etc.)
    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    val_loss, val_acc = float("nan"), float("nan")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_interval == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

    return model
