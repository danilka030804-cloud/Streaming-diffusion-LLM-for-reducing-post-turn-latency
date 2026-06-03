from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class StatefulConfig:
    canvas_length: int = 64
    remasking_policy: str = "confidence"
    remasking_strategy: str = "none"
    compute_budget_default: int = 6
    cleanup_steps_default: int = 24
    top_k: int = 40
    top_p: float = 0.92
    remasking_decay: str = "cosine"
    
    early_threshold: float = 0.7
    early_start_step: float = 0.3
    late_remask_until: float = 0.7
    adaptive_variance_window: float = 0.6 



class StatefulDiffusionEvaluator:
    canvas_length = 64
    remasking_policy = "confidence"
    remasking_strategy = "none"
    compute_budget = 6
    
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        config = StatefulConfig(
            canvas_length=self.__class__.canvas_length,
            remasking_policy=self.__class__.remasking_policy,
            remasking_strategy=self.__class__.remasking_strategy,
            compute_budget_default=self.__class__.compute_budget,
            cleanup_steps_default=24
        )
        
        self.canvas_length = config.canvas_length
        self.mask_token_id = 126336
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.config = config
        
        self.total_nfe = 0
        self.hidden_nfe = 0
        self.visible_nfe = 0
        self.last_output = ""
        
        self.logit_variances = []
        self.confidence_history = []
        
        self.reset()
    
    def set_system_prompt(self, system_text: str):
        self.system_prefix_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
    
    def reset(self):
        self.canvas_tokens = torch.full(
            (1, self.canvas_length), self.mask_token_id, 
            dtype=torch.long, device=self.device
        )
        self.dynamic_history_ids = []
        self.total_nfe = 0
        self.hidden_nfe = 0
        self.visible_nfe = 0
        self.last_output = ""
        self.logit_variances = []
        self.confidence_history = []
    
    def top_k_top_p_filtering(self, logits, filter_value=-float("Inf")):
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def compute_remasking_decay(self, step: int, total_steps: int) -> float:
        progress = (step + 1) / total_steps
        if self.config.remasking_decay == "cosine":
            decay = math.cos(0.5 * math.pi * progress)
        elif self.config.remasking_decay == "linear":
            decay = 1.0 - progress
        elif self.config.remasking_decay == "exponential":
            decay = math.exp(-3 * progress)
        else:
            decay = 1.0 - progress
        return max(0.0, decay)
    
    def compute_strategy_factor(self, step: int, total_steps: int, 
                                 confidence: torch.Tensor = None,
                                 logits: torch.Tensor = None) -> float:
        strategy = self.config.remasking_strategy
        progress = step / total_steps
        
        if strategy == "none":
            return 1.0
        
        elif strategy == "early":
            if progress < self.config.early_start_step:
                return 1.2
            else:
                return 0.4
        
        elif strategy == "late":
            if progress < self.config.late_remask_until:
                return 1.5
            else:
                return 0.8
        
        elif strategy == "adaptive":
            if confidence is not None and len(self.confidence_history) > 0:
                avg_confidence = np.mean(self.confidence_history[-10:]) if self.confidence_history else 0.5
                
                if avg_confidence > self.config.early_threshold:
                    return 0.6
                elif avg_confidence < 0.4:
                    return 1.3
            
            if logits is not None:
                logit_var = logits.var(dim=-1).mean().item()
                self.logit_variances.append(logit_var)
                
                if len(self.logit_variances) > 5:
                    recent_var = np.mean(self.logit_variances[-5:])
                    if recent_var > self.config.adaptive_variance_window:
                        return 1.2
            
            return 1.0
        
        else:
            return 1.0
    
    def apply_remasking(self, logits, pred_tokens, num_to_mask):
        policy = self.config.remasking_policy
        
        if policy == "none" or num_to_mask <= 0:
            return pred_tokens
        
        log_probs = F.log_softmax(logits, dim=-1)
        confidence = log_probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)
        
        self.confidence_history.append(confidence.mean().item())
        
        if policy == "confidence":
            _, low_conf_indices = torch.topk(confidence, k=num_to_mask, largest=False, dim=-1)
            pred_tokens.scatter_(1, low_conf_indices, self.mask_token_id)
        
        elif policy == "entropy":
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            _, high_entropy_indices = torch.topk(entropy, k=num_to_mask, dim=-1)
            pred_tokens.scatter_(1, high_entropy_indices, self.mask_token_id)
        
        elif policy == "random":
            mask_positions = torch.randperm(self.canvas_length, device=self.device)[:num_to_mask]
            pred_tokens[0, mask_positions] = self.mask_token_id
        
        return pred_tokens
    
    @torch.no_grad()
    def generate(self, prompt_ids: list, steps_budget: int, is_hidden: bool = True,
                 step_callback=None) -> str:
        local_canvas = self.canvas_tokens.clone()
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = int(prompt_tensor.shape[1])
        
        self.logit_variances = []
        self.confidence_history = []
        
        for step in range(steps_budget):
            input_ids = torch.cat([prompt_tensor, local_canvas], dim=-1)
            attention_mask = torch.ones_like(input_ids, device=self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            self.total_nfe += 1
            
            if is_hidden:
                self.hidden_nfe += 1
            else:
                self.visible_nfe += 1
            
            logits = outputs.logits[:, prompt_len:, :].clone()
            
            logits[..., self.mask_token_id] = -float('inf')
            if self.tokenizer.eos_token_id is not None:
                logits[..., self.tokenizer.eos_token_id] = -float('inf')
            logits[..., 100000:] = -float('inf')
            
            filtered_logits = self.top_k_top_p_filtering(logits)
            probs = F.softmax(filtered_logits, dim=-1)
            b, s, v = probs.shape
            pred_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, s)
            
            log_probs = F.log_softmax(logits, dim=-1)
            confidence = log_probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)
            
            decay = self.compute_remasking_decay(step, steps_budget)
            base_num_to_mask = int(self.canvas_length * decay)
            
            strategy_factor = self.compute_strategy_factor(
                step, steps_budget, confidence, logits
            )
            num_to_mask = int(base_num_to_mask * strategy_factor)
            num_to_mask = max(0, min(num_to_mask, self.canvas_length))
            
            if num_to_mask > 0:
                pred_tokens = self.apply_remasking(logits, pred_tokens, num_to_mask)
            
            local_canvas = pred_tokens
            
            if step_callback:
                flat_tokens = local_canvas.cpu().tolist()[0]
                clean_tokens = [t for t in flat_tokens if t != self.mask_token_id]
                canvas_text = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
                step_callback(step, canvas_text)
        
        self.canvas_tokens = local_canvas
        
        flat_tokens = local_canvas.cpu().tolist()[0]
        clean_tokens = [t for t in flat_tokens if t != self.mask_token_id]
        result = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
        self.last_output = result
        return result
    
    def process_chunk(self, prompt_ids: list, steps_budget: int = None, tracker=None) -> str:
        if steps_budget is None:
            steps_budget = self.config.compute_budget_default
        
        def on_step(step, canvas_text):
            if tracker and hasattr(tracker, 'update_hidden_canvas'):
                tracker.update_hidden_canvas(canvas_text)
        
        return self.generate(prompt_ids, steps_budget, is_hidden=True, step_callback=on_step)
    
    def finalize(self, prompt_ids: list, cleanup_steps: int = None, tracker=None) -> str:
        if cleanup_steps is None:
            cleanup_steps = self.config.cleanup_steps_default
        
        def on_step(step, canvas_text):
            if tracker and hasattr(tracker, 'update_hidden_canvas'):
                tracker.update_hidden_canvas(canvas_text)
        
        return self.generate(prompt_ids, cleanup_steps, is_hidden=False, step_callback=on_step)