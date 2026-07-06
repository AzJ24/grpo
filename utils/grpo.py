import torch
import json
import re 
import logging
import os
import numpy as np

# efficient batched sampling for GRPO style generation
def generate_grpo_samples_batched(model, tokenizer, prompts, num_samples_per_prompt, system_prompt=None, do_sample=True,
                                  max_new_tokens=50, temperature=1.0, model_name=None,
                                  enable_thinking=True, top_p=0.95):
    all_texts = []
    for prompt in prompts:
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        if model_name == "qwen_3.5_9b":
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        all_texts.append(text)
    
#    print(all_texts) # Debug print to verify prompts being processed
    model_inputs = tokenizer(all_texts, return_tensors="pt", padding=True).to(model.device)
    
    prompt_lengths = model_inputs.attention_mask.sum(dim=1).tolist()
    
    batched_input_ids = model_inputs.input_ids.repeat_interleave(num_samples_per_prompt, dim=0)
    batched_attention_mask = model_inputs.attention_mask.repeat_interleave(num_samples_per_prompt, dim=0)
    # Ensure all tensors are on the model's device
    batched_input_ids = batched_input_ids.to(model.device)
    batched_attention_mask = batched_attention_mask.to(model.device)
    
    batched_prompt_lengths = [length for length in prompt_lengths for _ in range(num_samples_per_prompt)]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
        ).to(model.device)

    
    all_sequences = []
    all_generated_tokens = []
    
    for sequence, original_attention_mask, prompt_length in zip(outputs, batched_attention_mask, batched_prompt_lengths):
        num_padding = (original_attention_mask == 0).sum().item()
        clean_sequence = sequence[num_padding:]
        
        prompt_part = clean_sequence[:prompt_length]
        generated_part = clean_sequence[prompt_length:]
        
        eos_positions = (generated_part == eos_token_id).nonzero(as_tuple=True)[0]
        
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            generated_part = generated_part[:first_eos + 1]
        
        full_sequence = torch.cat([prompt_part, generated_part], dim=0)
        
        all_sequences.append(full_sequence.unsqueeze(0))
        all_generated_tokens.append(generated_part.unsqueeze(0))
    
    return all_sequences, all_generated_tokens, batched_prompt_lengths

def compute_batched_log_probs(model, sequences, prompt_lengths, tokenizer, micro_batch_size=4, temperature=1.0,
                              return_per_token=False):
    """
    Compute log probs in micro-batches to reduce memory usage.
    temperature should match the generation temperature to keep log-probs consistent with the sampling distribution.

    return_per_token=False: returns seq_log_probs (sum over generated tokens), shape (N,).
    return_per_token=True:  returns (token_log_probs, generated_masks), each shape
                            (N, max_seq_len - 1), padded with zeros.  Needed for
                            token-normalised PG loss and per-token KL estimators.
    """
    all_seq_log_probs = []
    all_token_log_probs = []
    all_gen_masks = []
    target_width = max(seq.shape[1] for seq in sequences) - 1 if return_per_token else None

    # Process in micro-batches
    for i in range(0, len(sequences), micro_batch_size):
        micro_sequences = sequences[i:i+micro_batch_size]
        micro_prompt_lengths = prompt_lengths[i:i+micro_batch_size]
        
        # Same logic but on smaller batch
        max_len = max(seq.shape[1] for seq in micro_sequences)
        
        padded_sequences = []
        attention_masks = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        for sequence in micro_sequences:
            seq_len = sequence.shape[1]
            padding_len = max_len - seq_len
            
            padded_seq = torch.cat([
                sequence,
                torch.full((1, padding_len), pad_token_id, dtype=sequence.dtype, device=sequence.device)
            ], dim=1)
            
            mask = torch.cat([
                torch.ones(1, seq_len, device=sequence.device),
                torch.zeros(1, padding_len, device=sequence.device)
            ], dim=1)
            
            padded_sequences.append(padded_seq)
            attention_masks.append(mask)
        
        batched_sequences = torch.cat(padded_sequences, dim=0)  
        batched_masks = torch.cat(attention_masks, dim=0)       
        
        # Forward pass on micro-batch
        outputs = model(batched_sequences, attention_mask=batched_masks)
        
        logits = outputs.logits[:, :-1, :]      
        target_tokens = batched_sequences[:, 1:]
        
        log_probs = torch.log_softmax(logits / temperature, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Clean up large tensors immediately
        del outputs, logits, log_probs
        
        generated_masks = []
        for j, prompt_length in enumerate(micro_prompt_lengths):
            mask = torch.zeros(max_len - 1, device=model.device)
            gen_start = prompt_length - 1  
            gen_end = micro_sequences[j].shape[1] - 1  
            mask[gen_start:gen_end] = 1
            generated_masks.append(mask)
        
        generated_masks = torch.stack(generated_masks)
        if return_per_token:
            pad_w = target_width - token_log_probs.shape[1]
            if pad_w > 0:
                token_log_probs = torch.cat([
                    token_log_probs,
                    torch.zeros(token_log_probs.shape[0], pad_w, dtype=token_log_probs.dtype, device=token_log_probs.device),
                ], dim=1)
                generated_masks = torch.cat([
                    generated_masks,
                    torch.zeros(generated_masks.shape[0], pad_w, device=generated_masks.device),
                ], dim=1)
            all_token_log_probs.append(token_log_probs)
            all_gen_masks.append(generated_masks)
            del batched_sequences, batched_masks
        else:
            masked_log_probs = token_log_probs * generated_masks
            seq_log_probs = masked_log_probs.sum(dim=1)
            all_seq_log_probs.append(seq_log_probs)
            del token_log_probs, batched_sequences, batched_masks, masked_log_probs, generated_masks

    if return_per_token:
        return torch.cat(all_token_log_probs, dim=0), torch.cat(all_gen_masks, dim=0)

    # Concatenate results - seq_log_probs is just [batch_size], no shape mismatch!
    seq_log_probs = torch.cat(all_seq_log_probs, dim=0)

    return seq_log_probs



def extract_json_outputs(text):
    """
    Extracts the response text, aleatoric, and epistemic uncertainty from a JSON string.
    """
    
    # Find the JSON block in the text.
    start_match = re.search(r'\{', text)
    end_match = re.search(r'\}', text[::-1])
    
    if start_match is None or end_match is None:
        return None, None, None 

    start_index = start_match.start()
    end_index = len(text) - end_match.start()
    
    json_string = text[start_index:end_index].strip()
    
    # Simple cleanup of potential trailing commas or markdown fences
    json_string = json_string.strip('`').strip()
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        # Failed to parse JSON
        return None, None, None 

    # Extract answer
    response_text = data.get("response") # Retain the full generated text

    # Extract uncertainty values
    predicted_alea = data.get("aleatoric_uncertainty")
    predicted_epi = data.get("epistemic_uncertainty")

    # Ensure uncertainty values are numerical (float/int)
    try:
        predicted_alea = float(predicted_alea)
    except (TypeError, ValueError):
        predicted_alea = None
        
    try:
        predicted_epi = float(predicted_epi)
    except (TypeError, ValueError):
        predicted_epi = None
    
    # Return the full response text instead of a specific MC selection
    return response_text, predicted_alea, predicted_epi


def setup_training_logger(log_dir: str, log_name: str = "training_samples.log"):
    """
    Set up a logger for detailed training sample logging.
    
    Args:
        log_dir: Directory to save log file
        log_name: Name of the log file
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    # Create a dedicated logger (not the root logger)
    logger = logging.getLogger("training_samples")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Simple format for readability
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    print(f"Training sample logger initialized: {log_path}")
    return logger


def log_training_step(logger, step: int, batch: list, generated_tokens: list, 
                      ground_truths: list, tokenizer, num_samples_per_prompt: int):
    """
    Log detailed information about each training step.
    
    Args:
        logger: Logger instance
        step: Current training step
        batch: List of batch items (questions)
        generated_tokens: Generated token sequences
        ground_truths: Ground truth data (repeated for each sample)
        tokenizer: Tokenizer for decoding
        num_samples_per_prompt: Number of samples per question
    """
    from grpo import extract_response_components  # Import here to avoid circular import
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP {step}")
    logger.info(f"{'='*80}")
    
    num_questions = len(batch)
    
    for q_idx in range(num_questions):
        question_data = batch[q_idx]
        start_idx = q_idx * num_samples_per_prompt
        end_idx = start_idx + num_samples_per_prompt
        
        logger.info(f"\n{'-'*60}")
        logger.info(f"GROUP {q_idx + 1}/{num_questions}")
        logger.info(f"{'-'*60}")
        logger.info(f"Question: {question_data['question']}")
        logger.info(f"Target Answer(s): {question_data['gt_answer']}")
        logger.info(f"Target Aleatoric: {question_data['u_aleatoric']:.4f}")
        logger.info(f"Target Epistemic: {question_data['u_epistemic']:.4f}")
        logger.info(f"")
        
        # Log each sample in the group
        for s_idx, sample_idx in enumerate(range(start_idx, end_idx)):
            gen_tokens = generated_tokens[sample_idx]
            text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            parsed = extract_response_components(text)
            
            logger.info(f"  Sample {s_idx + 1}/{num_samples_per_prompt}:")
            
            if parsed is not None:
                logger.info(f"    Predicted Answer: {parsed['answer'][:100]}{'...' if len(parsed['answer']) > 100 else ''}")
                logger.info(f"    Predicted Aleatoric: {parsed['aleatoric']:.4f}")
                logger.info(f"    Predicted Epistemic: {parsed['epistemic']:.4f}")
                
                # Calculate errors
                ale_error = abs(parsed['aleatoric'] - question_data['u_aleatoric'])
                epi_error = abs(parsed['epistemic'] - question_data['u_epistemic'])
                logger.info(f"    Aleatoric Error: {ale_error:.4f}")
                logger.info(f"    Epistemic Error: {epi_error:.4f}")
            else:
                logger.info(f"    [PARSE FAILED] Raw output: {text[:150]}...")
            
            logger.info(f"")
        
        # Summary for this group
        group_preds_ale = []
        group_preds_epi = []
        for sample_idx in range(start_idx, end_idx):
            text = tokenizer.decode(generated_tokens[sample_idx][0], skip_special_tokens=True)
            parsed = extract_response_components(text)
            if parsed:
                group_preds_ale.append(parsed['aleatoric'])
                group_preds_epi.append(parsed['epistemic'])
        
        if group_preds_ale:
            logger.info(f"  Group Summary:")
            logger.info(f"    Aleatoric preds: {group_preds_ale} (std={np.std(group_preds_ale):.4f})")
            logger.info(f"    Epistemic preds: {group_preds_epi} (std={np.std(group_preds_epi):.4f})")
            unique_ale = len(set([round(x, 2) for x in group_preds_ale]))
            unique_epi = len(set([round(x, 2) for x in group_preds_epi]))
            logger.info(f"    Unique values: ale={unique_ale}/{len(group_preds_ale)}, epi={unique_epi}/{len(group_preds_epi)}")

