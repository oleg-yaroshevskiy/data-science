"""
LLM Sampling Strategies
Implement different decoding/sampling methods for language models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 4
    pad_token_id: int = 0
    eos_token_id: int = 2


class LLMSampler:
    """Base class for LLM sampling strategies"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the model and tokenizer.
        Uses a HuggingFace model for actual LM forward passes.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the language model.
        This is provided for you - focus on implementing the sampling methods.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            logits: Logits for next token of shape (batch_size, vocab_size)
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Get logits for the last position
            logits = outputs.logits[:, -1, :]
        return logits
    
    def greedy_search(self, prompt: str, config: GenerationConfig) -> str:
        """
        Implement greedy decoding: always select the token with highest probability.
        
        Algorithm:
        1. Encode the prompt
        2. For each generation step:
            a. Get logits from the model
            b. Select token with highest probability
            c. Append to sequence
            d. Check for stopping conditions
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for _ in range(config.max_length):
            logits = self(input_ids)
            logits = logits / config.temperature
            next_token = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == config.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def top_k_sampling(self, prompt: str, config: GenerationConfig) -> str:
        """
        Implement top-k sampling: sample from the k most likely tokens.
        
        Algorithm:
        1. Encode the prompt
        2. For each generation step:
            a. Get logits from the model
            b. Keep only top-k logits, set others to -inf
            c. Convert to probabilities with softmax
            d. Sample from the distribution
            e. Append to sequence
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for _ in range(config.max_length):
            logits = self(input_ids)
            logits = logits / config.temperature
            
            # Filter to top-k tokens
            top_k_indices = torch.topk(logits, config.top_k, dim=-1).indices
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, top_k_indices, 0)
            logits = logits + mask
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == config.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def top_p_sampling(self, prompt: str, config: GenerationConfig) -> str:
        """
        Implement nucleus (top-p) sampling: sample from smallest set of tokens 
        whose cumulative probability exceeds p.
        
        Algorithm:
        1. Encode the prompt
        2. For each generation step:
            a. Get logits from the model
            b. Convert to probabilities and sort
            c. Compute cumulative probabilities
            d. Find cutoff where cumsum > p
            e. Mask tokens beyond cutoff
            f. Re-normalize and sample
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for _ in range(config.max_length):
            logits = self(input_ids)
            logits = logits / config.temperature
            
            # Nucleus (top-p) filtering
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = (cumulative_probs > config.top_p).nonzero(as_tuple=True)[1][0]

            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, sorted_indices[:, :cutoff_index + 1], 0)
            logits = logits + mask
            probs = F.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == config.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def beam_search(self, prompt: str, config: GenerationConfig) -> str:
        """
        Implement beam search: maintain top-k hypotheses at each step.
        
        Algorithm:
        1. Encode the prompt
        2. Initialize num_beams sequences with the prompt
        3. For each generation step:
            a. For each beam, get logits
            b. Compute log probabilities for all possible next tokens
            c. Find top num_beams sequences by total log probability
            d. Update beams with new sequences
            e. Check for completed sequences (EOS)
        4. Return the sequence with highest score
        
        Note: This is more complex - you need to track:
        - Multiple sequences (beams)
        - Their cumulative scores
        - Completed vs active sequences
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        num_beams = config.num_beams
        
        # Initialize beams
        beam_sequences = input_ids.repeat(num_beams, 1)
        beam_scores = torch.zeros(num_beams, device=self.device).unsqueeze(1)
        finished_sequences = []
        
        for step in range(config.max_length):
            all_logits = self(beam_sequences)
            log_probs = F.log_softmax(all_logits, dim=-1)
            beam_scores = beam_scores + log_probs
            
            # Get top num_beams candidates
            flat_scores = beam_scores.view(-1)
            top_scores, top_indices = torch.topk(flat_scores, num_beams)
            
            # Convert flat indices to (beam_idx, token_idx)
            vocab_size = all_logits.shape[-1]
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Create new beam sequences
            new_sequences = []
            for beam_idx, token_idx in zip(beam_indices, token_indices):
                old_seq = beam_sequences[beam_idx]
                new_seq = torch.cat([old_seq, token_idx.unsqueeze(0)])
                new_sequences.append(new_seq)
            
            beam_sequences = torch.stack(new_sequences)
            beam_scores = top_scores.unsqueeze(1)
            
            # Track finished sequences (those that generated EOS)
            for i, token_idx in enumerate(token_indices):
                if token_idx.item() == config.eos_token_id:
                    finished_sequences.append((beam_sequences[i].clone(), beam_scores[i].item()))
        
        # Select best sequence
        best_sequence = sorted(finished_sequences, key=lambda x: x[1], reverse=True)[0][0] if finished_sequences else beam_sequences[0]
        
        return self.tokenizer.decode(best_sequence, skip_special_tokens=True)


def main():
    """Test your implementations"""
    sampler = LLMSampler(model_name="gpt2")
    config = GenerationConfig(
        max_length=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_beams=4
    )
    
    prompt = "The future of artificial intelligence is"
    
    print("=" * 80)
    print("Testing Sampling Strategies")
    print("=" * 80)
    print(f"\nPrompt: {prompt}\n")
    
    # Test greedy search
    print("\n[1] Greedy Search:")
    print("-" * 80)
    result = sampler.greedy_search(prompt, config)
    print(result)
    
    # Test top-k sampling
    print("\n[2] Top-K Sampling (k=50):")
    print("-" * 80)
    result = sampler.top_k_sampling(prompt, config)
    print(result)
    
    # Test top-p sampling
    print("\n[3] Top-P Sampling (p=0.9):")
    print("-" * 80)
    result = sampler.top_p_sampling(prompt, config)
    print(result)
    
    # Test beam search
    print("\n[4] Beam Search (num_beams=4):")
    print("-" * 80)
    result = sampler.beam_search(prompt, config)
    print(result)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
