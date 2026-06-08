import math

import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, base_lr, warmup_ratio):
        self.device =  torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.warmup_ratio = warmup_ratio
        self.base_lr = base_lr
        self.global_step = 0
        self.writer = SummaryWriter(log_dir="logs")

    def get_lr_scale(self, step: int, warmup_steps: int, total_steps: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        
        # Cosine decay
        progress = ((step - warmup_steps)/ max(1, total_steps - warmup_steps))
        
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def train(self, epochs):
        """
        Full training loop for language modeling.
        Tracks:
        - Train Loss
        - Validation Loss
        - Train Perplexity
        - Validation Perplexity
        """
        best_valid_perplexity = float("inf")
        self.total_steps = len(self.train_loader) * epochs
        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
        
        print(f"Total Steps   : {self.total_steps}")
        print(f"Warmup Steps  : {self.warmup_steps}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for input_ids, target_ids, attention_mask in self.train_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                 # LR Schedule 
                scale = self.get_lr_scale(step=self.global_step, warmup_steps=self.warmup_steps, total_steps=self.total_steps)
                current_lr = self.base_lr * scale
                for pg in self.optimizer.param_groups:
                    pg["lr"] = current_lr
                
                
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(input_ids, attention_mask)

                # Compute loss
                loss = self.criterion(logits.reshape(-1, logits.size(-1)),target_ids.reshape(-1))

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update weights
                self.optimizer.step()
                self.global_step += 1
                 
                # Accumulate loss
                train_loss += loss.item()
            
            # Average train loss
            train_loss /= len(self.train_loader)
            
            # Average train loss
            train_perplexity = math.exp(train_loss)
            
            
            
            # =========== Validation =========
            self.model.eval()
            valid_loss = 0.0
            

            with torch.no_grad():

                for input_ids, target_ids, attention_mask in self.valid_loader:
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)

                    # Compute loss
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)),target_ids.reshape(-1))

                    # Accumulate loss
                    valid_loss += loss.item()

            # Average validation loss
            valid_loss /= len(self.valid_loader)
            valid_perplexity = math.exp(valid_loss)
            
            self.writer.add_scalars("Loss", {"Train": train_loss,"Eval": valid_loss}, epoch)
            self.writer.add_scalars("Perplexity", {"Train": train_perplexity,"Eval": valid_perplexity}, epoch)
            
            # self.scheduler.step(valid_perplexity)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            print(
                f"Epoch: {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Train PPL: {train_perplexity:.4f} | "
                f"Valid PPL: {valid_perplexity:.4f} | " 
                f"LR: {current_lr:.6f}" 
            )
            
            # Save best model
            if valid_perplexity < best_valid_perplexity:
                best_valid_perplexity = valid_perplexity 
                torch.save(self.model.state_dict(), "mini_gpt_wikitext2.pth")
                
        self.writer.close()
        print(f"\nBest Validation Perplexity: {best_valid_perplexity:.4f}")