import torch

class StandardARModel:
    def __init__(self, model, tokenizer, device="cuda", base_tokens=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.base_tokens = base_tokens
        self.full_prompt_tokens = []
        self.last_output = ""
        
    def process_speech_chunk(self, chunk_text: str, compute_budget: int = 2, tracker=None):
        chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
        self.full_prompt_tokens.extend(chunk_tokens)
        
        if tracker:
            tracker.register_hidden_pass(1)
            tracker.register_canvas_state(torch.tensor([self.full_prompt_tokens]))
        
    def finalize(self, additional_steps: int = 4, tracker=None) -> str:
        if not self.full_prompt_tokens:
            return ""
        
        if tracker:
            tracker.start_post_turn_phase()
        
        prompt_tensor = torch.tensor([self.full_prompt_tokens], dtype=torch.long, device=self.device)
        
        max_allowable_tokens = 256 
        
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_allowable_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        new_tokens = outputs[0][prompt_tensor.shape[1]:]
        
        if tracker and hasattr(tracker, 'latency'):
            for _ in range(len(new_tokens)):
                tracker.latency.add_visible_pass()
        
        clean_tokens = [t for t in new_tokens.tolist() if t != self.tokenizer.eos_token_id]
        self.last_output = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
        
        if tracker:
            tracker.end_post_turn_phase()
        
        return self.last_output

    
    def set_system_prompt(self, system_prompt: str):
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        self.full_prompt_tokens = system_tokens.copy()
    
    @property
    def canvas_tokens(self):
        return torch.tensor([self.full_prompt_tokens], dtype=torch.long, device=self.device)
    
    @property
    def last_logits(self):
        return None