import torch

class ARStreamingThinkingModel:
    def __init__(self, model, tokenizer, device="cuda", 
                 base_think_tokens=8, base_final_tokens=32,
                 think_strategy="tokens"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.base_think_tokens = base_think_tokens
        self.base_final_tokens = base_final_tokens
        self.think_strategy = think_strategy
        
        # Состояние
        self.full_prompt_tokens = []
        self.thought_tokens = []
        self.last_output = ""
        
        # Специальные токены
        self.think_start = tokenizer.encode(" [THINK]", add_special_tokens=False)[0] if "[THINK]" in tokenizer.get_vocab() else 29871
        self.think_end = tokenizer.encode(" [/THINK]", add_special_tokens=False)[0] if "[/THINK]" in tokenizer.get_vocab() else 29872
        
    def process_speech_chunk(self, chunk_text: str, compute_budget: int = 2, tracker=None):
        chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
        self.full_prompt_tokens.extend(chunk_tokens)
        
        if tracker:
            tracker.register_canvas_state(torch.tensor([self.full_prompt_tokens + self.thought_tokens]))
        
        thoughts_generated = self._generate_thoughts(compute_budget, tracker)
        
        if tracker:
            tracker.register_hidden_pass(thoughts_generated)
            tracker.register_canvas_state(torch.tensor([self.full_prompt_tokens + self.thought_tokens]))
        
    def _generate_thoughts(self, num_tokens: int, tracker=None) -> int:
        if num_tokens <= 0:
            return 0
        
        full_context = self.full_prompt_tokens + self.thought_tokens
        
        if not self.thought_tokens:
            full_context.append(self.think_start)
        
        input_tensor = torch.tensor([full_context], dtype=torch.long, device=self.device)
        
        generated = 0
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]
                
                if tracker:
                    tracker.register_thinking_token(-1, logits)
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                self.thought_tokens.append(next_token)
                full_context.append(next_token)
                input_tensor = torch.tensor([full_context], dtype=torch.long, device=self.device)
                generated += 1
                
                if next_token == self.think_end:
                    break
        
        return generated
    
    def finalize(self, additional_steps: int = 4, tracker=None) -> str:
        if tracker:
            tracker.start_post_turn_phase()
        
        if self.thought_tokens and self.thought_tokens[-1] != self.think_end:
            self.thought_tokens.append(self.think_end)
        
        full_context = self.full_prompt_tokens + self.thought_tokens
        full_context.extend(self.tokenizer.encode("\n[ANSWER]", add_special_tokens=False))
        
        input_tensor = torch.tensor([full_context], dtype=torch.long, device=self.device)
        
        if self.think_strategy == "tokens":
            max_new_tokens = self.base_final_tokens + additional_steps * 4
            temperature = 0.7
            
        elif self.think_strategy == "temperature":
            max_new_tokens = self.base_final_tokens
            temperature = 0.5 + additional_steps * 0.1
            temperature = min(temperature, 1.5)
            
        elif self.think_strategy == "both":
            max_new_tokens = self.base_final_tokens + additional_steps * 2
            temperature = 0.5 + additional_steps * 0.05
            temperature = min(temperature, 1.2)
        else:
            max_new_tokens = self.base_final_tokens
            temperature = 0.7
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        new_tokens = outputs[0][input_tensor.shape[1]:]
        clean_tokens = [t for t in new_tokens.tolist() if t != self.tokenizer.eos_token_id]
        self.last_output = self.tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
        
        if tracker:
            tracker.register_visible_pass(1)
            tracker.end_post_turn_phase()
        
        return self.last_output
    
    def set_system_prompt(self, system_prompt: str):
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        think_instruction = self.tokenizer.encode(" Think step by step before answering.", add_special_tokens=False)
        self.full_prompt_tokens = system_tokens + think_instruction
        self.thought_tokens = []
    
    @property
    def canvas_tokens(self):
        return torch.tensor([self.full_prompt_tokens + self.thought_tokens], dtype=torch.long, device=self.device)
    
    @property
    def last_logits(self):
        return None