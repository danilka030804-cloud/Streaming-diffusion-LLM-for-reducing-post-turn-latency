import torch
import time
import math
import torch.nn.functional as F


class OfflineDiffusionEvaluator:
    def __init__(self, model, tokenizer, canvas_length: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.canvas_length = canvas_length
        self.device = model.device
        self.mask_token_id = 126336
        self.total_nfe = 0
        self.hidden_nfe = 0
        self.visible_nfe = 0
        self.last_output = ""

    def top_k_top_p_filtering(self, logits, top_k=40, top_p=0.92, filter_value=-float("Inf")):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            topk_values = torch.topk(logits, top_k, dim=-1)[0]
            thresh = topk_values[..., -1, None]
            indices_to_remove = logits < thresh
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: list, steps_budget: int, is_hidden: bool = True, 
                 step_callback=None) -> str:
        local_canvas = torch.full((1, self.canvas_length), self.mask_token_id, dtype=torch.long, device=self.device)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = int(prompt_tensor.shape[1])
        
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
            
            decay = math.cos(0.5 * math.pi * ((step + 1) / steps_budget))
            num_to_mask = int(self.canvas_length * decay)
            
            if num_to_mask > 0:
                log_probs = F.log_softmax(logits, dim=-1)
                conf = log_probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)
                _, low_conf_indices = torch.topk(conf, k=num_to_mask, largest=False, dim=-1)
                pred_tokens.scatter_(1, low_conf_indices, self.mask_token_id)
                
            local_canvas = pred_tokens
            
            if step_callback:
                flat_tokens = local_canvas.cpu().tolist()[0]
                clean_tokens = [t for t in flat_tokens if t != self.mask_token_id]
                canvas_text = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
                step_callback(step, canvas_text)

        flat_tokens = local_canvas.cpu().tolist()[0]
        clean_tokens = [t for t in flat_tokens if t != self.mask_token_id]
        result = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
        self.last_output = result
        return result

    def process_chunk(self, prompt_ids: list, steps_budget: int = 6, tracker=None) -> str:
        if tracker and hasattr(tracker, 'update_hidden_canvas'):
            tracker.update_hidden_canvas("")
        return ""
    
    def finalize(self, prompt_ids: list, cleanup_steps: int = 24, tracker=None) -> str:
        def on_step(step, canvas_text):
            if tracker and hasattr(tracker, 'update_hidden_canvas'):
                tracker.update_hidden_canvas(canvas_text)
        
        result = self.generate(prompt_ids, cleanup_steps, is_hidden=False, step_callback=on_step)
        return result
    
    def run_full_offline_inference(self, full_prompt_ids: list, total_steps: int = 32, tracker=None) -> tuple:
        start_time = time.time()
        
        def on_step(step, canvas_text):
            if tracker and hasattr(tracker, 'update_hidden_canvas'):
                tracker.update_hidden_canvas(canvas_text)
        
        result = self.generate(full_prompt_ids, total_steps, is_hidden=False, step_callback=on_step)
        post_turn_latency = time.time() - start_time
        
        return result, post_turn_latency
    
    def reset(self):
        self.total_nfe = 0
        self.hidden_nfe = 0
        self.visible_nfe = 0
        self.last_output = ""