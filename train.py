import torch
from my_model import LinearLayer, EmbeddingLayer, RMSNormLayer, PositionWiseFFN, RoPELayer, SoftmaxLayer, ScaledDotProductAttentionLayer,CausalMHSA, TransformerBlock, TransformerModel
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
import argparse
import os
from logger import ExperimentLogger
from Generate import generate

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_stable=logits-torch.max(logits,dim=-1,keepdim=True).values
        # Cancel out log and exp whenever possible.
        log_p=logits_stable-torch.log(torch.sum(torch.exp(logits_stable),dim=-1,keepdim=True))
        loss = -1*torch.sum(log_p[torch.arange(logits.size(0)),targets])/logits.size(0)
        return loss

class CEPerplexity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ce_total: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.sum(ce_total)/ce_total.size(0))

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1) * grad
                state["t"] = t + 1

        return loss
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]
                
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                lr_t = lr * math.sqrt(1-beta2**(t))/(1-beta1**(t))
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * (grad**2)
                p.data -= lr_t * m/(torch.sqrt(v)+eps)
                
                state["m"] = m
                state["v"] = v
        return loss

class CosinelrScheduler():
    def __init__(self, it: int,max_learning_rate: float, min_learning_rate: float, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.it=it

    def step(self):
        self.it +=1
        return self.calculate_lr()

    def calculate_lr(self) -> float:
        if self.it < self.warmup_steps:
            lr_now = self.max_learning_rate * (self.it / self.warmup_steps)
        elif self.it < self.total_steps:
            lr_now = (self.max_learning_rate-self.min_learning_rate)*0.5*(1+math.cos(math.pi*(self.it-self.warmup_steps)/(self.total_steps-self.warmup_steps)))+self.min_learning_rate
        else:
            lr_now = self.min_learning_rate
        return lr_now

class GradientClipping():
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def clip_gradients(self, parameters: Iterable[torch.nn.Parameter]):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = torch.sum(p.grad.data ** 2)
                total_norm += param_norm.item()
        total_norm = total_norm ** 0.5
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data = p.grad.data * clip_coef

class DataLoader():
    def __init__(self, x: np.ndarray, batch_size: int, context_length: int, device: str):
        self.x = x
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self):
        n = len(self.x)
        max_start = n - self.context_length - 1
        if max_start < 0:
            raise ValueError(
                f"x length ({n}) must be at least context_length+1 ({self.context_length+1})"
            )

        # Sample batch_size unique starting indices without replacement
        starts = np.random.choice(max_start + 1, size=self.batch_size, replace=False)

        # Build inputs and targets as lists of arrays
        inputs = [self.x[start:start + self.context_length] for start in starts]
        targets = [self.x[start + 1:start + self.context_length + 1] for start in starts]

        # Convert to torch tensors on the requested device
        inputs = torch.tensor(np.stack(inputs), dtype=torch.long, device=self.device)
        targets = torch.tensor(np.stack(targets), dtype=torch.long, device=self.device)

        return (inputs, targets)

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_data = np.memmap(args.train_data, dtype=np.int32, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.int32, mode='r') 

    model = TransformerModel(vocab_size=args.vocab_size, d_model=args.d_model, d_ff=args.d_ff, theta=args.theta, num_heads=args.num_heads, num_layers=args.num_layers, max_seq_len=args.max_seq_len).to(args.device)

    config = vars(args)  # 将所有命令行参数作为配置
    logger = ExperimentLogger(
        config=config,
        log_dir=args.log_dir,  # 新增命令行参数: --log_dir ./experiments/run1
        use_wandb=True,        # 可改为 False 以禁用 WandB
        wandb_project="my-transformer-exp",
        experiment_name=f"lr_{args.lr}_bs_{args.batch_size}",
    )

    # 跟踪模型梯度（可选）
    logger.watch_model(model, log_gradients=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosinelrScheduler(
            it=0,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps
        )

    train_loader = DataLoader(train_data, args.batch_size, args.context_length, args.device)
    val_loader = DataLoader(val_data, args.batch_size, args.context_length, args.device)

    criterion = CrossEntropyLoss()

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(
            model, optimizer, args.resume_from,
        )
        scheduler.it = start_iter
        logger.log_metrics({"resumed_from_iter": start_iter}, step=start_iter)

    iteration = start_iter
    while iteration < args.total_steps:
        # 获取一个批次
        inputs, targets = train_loader.get_batch()
        optimizer.zero_grad()

        # 前向传播
        logits = model(inputs)  # 形状: (batch_size, context_length, vocab_size)
        loss = criterion(logits.view(-1, args.vocab_size), targets.view(-1))

        # 反向传播与优化
        loss.backward()
        optimizer.step()
        iteration += 1

        # 更新学习率（调度器步进）
        lr = scheduler.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        if iteration % args.log_interval == 0:
            logger.log_metrics({
                "train_loss": loss.item(),
                "lr": lr,
                "iteration": iteration,
            }, step=iteration)

        # 验证
        if iteration % args.eval_interval == 0:
            model.eval()
            val_loss_sum = 0.0
            num_val_batches = 5  # 固定使用 5 个 batch 快速评估
            with torch.no_grad():
                for _ in range(num_val_batches):
                    val_inputs, val_targets = val_loader.get_batch()
                    val_logits = model(val_inputs)
                    val_loss = criterion(val_logits.view(-1, args.vocab_size), val_targets.view(-1))
                    val_loss_sum += val_loss.item()
            avg_val_loss = val_loss_sum / num_val_batches
            logger.log_metrics({
                "val_loss": avg_val_loss,
                "iteration": iteration,
            }, step=iteration)
        model.train()

        # 保存检查点
        if iteration % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_iter_{iteration}.pt")
            save_checkpoint(model, optimizer, scheduler, iteration, ckpt_path)
            logger.save_checkpoint(model, optimizer, scheduler, iteration, args.checkpoint_dir)

    logger.finish()
    print("Training completed successfully!")

if __name__=="__main__":
    main()

    