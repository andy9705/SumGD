import copy
import warnings
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GreedySearchOutput
import transformers
import spacy

# Load spacy model
nlp = spacy.load('en_core_web_sm')

def process_tokens(tokens):
    """Combine tokens, removing special characters and subword symbols."""
    full_text = ""
    for token in tokens:
        if token.startswith("▁"):
            full_text += " " + token[1:]
        elif token in ["<0x0A>", "</s>", "<s>", "<unk>"]:
            continue
        else:
            full_text += token
    return full_text.strip()

def align_probabilities(llama_tokens, nltk_tokens):
    """Align LLaMA token indices with nltk/spacy tokens."""
    llama_index, llama_indices, combined_token, nltk_save = 0, [], "", ""

    for nltk_token in nltk_tokens:
        llama_temp = []
        while llama_index < len(llama_tokens):
            llama_token = llama_tokens[llama_index].replace("▁", "").replace("<0x0A>", "").replace("</s>", "").replace("<unk>", "").replace("<s>", "")
            if llama_token == "":
                llama_index += 1
                continue

            combined_token += llama_token
            nltk_save += nltk_token
            llama_temp.append(llama_index)
            llama_index += 1

            if combined_token == nltk_token or combined_token == nltk_save:
                nltk_save = ""
                break
            elif len(combined_token) > len(nltk_token):
                llama_index -= 1
                break

        llama_indices.append(llama_temp)
        combined_token = ""

    return llama_indices

def generate_pos_tags(tokenizer, generated_ids):
    """Generate POS tags for the generated text."""
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist()[0])
    full_text = process_tokens(generated_tokens)
    doc = nlp(full_text)
    spacy_tokens = [token.text for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    llama_indices = align_probabilities(generated_tokens, spacy_tokens)
    token_pos_pairs = [(prob, word, tag) for (prob, (word, tag)) in zip(llama_indices, pos_tags)]
    return token_pos_pairs

def generate_summary_caption(self, tokenizer, caption, model_kwargs, device, eos_token_id=None, 
                           pad_token_id=None, output_attentions=None, output_hidden_states=None):
    """
    Generate a summary of the provided caption using the model.
    
    Args:
        self: The model instance
        tokenizer: Tokenizer to use for encoding/decoding text
        caption: The caption text to summarize
        model_kwargs: Model keyword arguments to pass to the model
        device: The device to run the model on
        eos_token_id: End of sequence token ID(s)
        pad_token_id: Padding token ID
        output_attentions: Whether to output attention weights
        output_hidden_states: Whether to output hidden states
        
    Returns:
        summarized_caption: Tensor containing the token IDs of the summarized caption
        summary_text: String containing the decoded summarized caption
    """
    # Create EOS token tensor if provided
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(device) if eos_token_id is not None else None
    
    # Create summary prompt template and tokenize
    summary_template = f"USER: Summarize the following caption briefly.\nCaption: {caption} ASSISTANT:"
    summary_input_ids = tokenizer(
        summary_template, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to(device)['input_ids']
    
    # Keep track of which sequences are already finished
    unfinished_sequences = torch.ones(summary_input_ids.shape[0], dtype=torch.long, device=device)
    input_token_len = summary_input_ids.shape[1]
    
    # Create stopping criteria to prevent infinite loops
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    stopping_criteria = StoppingCriteriaList()
    
    # Reset model_kwargs for summary generation
    model_kwargs_summary = copy.deepcopy(model_kwargs)
    model_kwargs_summary['inputs_embeds'] = None
    model_kwargs_summary['inputs_embeds_llm'] = None
    model_kwargs_summary['attention_mask'] = torch.ones_like(summary_input_ids)
    
    while True:
        # Prepare model inputs using the language model preparation method
        inputs = self.prepare_inputs_for_generation_llm(summary_input_ids, **model_kwargs_summary)

        # Generate next token
        outputs = self(
            **inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Handle finished sequences
        if pad_token_id is not None and eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        # Update input_ids for next iteration
        summary_input_ids = torch.cat([summary_input_ids, next_tokens[:, None]], dim=-1)
        
        # Update model kwargs for next iteration
        model_kwargs_summary = self._update_model_kwargs_for_generation(
            outputs, model_kwargs_summary, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # Update unfinished sequences
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            
            # Stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        # Stop if we exceed the maximum length
        if stopping_criteria(summary_input_ids, None):
            break

    # Decode the generated summary
    summary_text = tokenizer.batch_decode(summary_input_ids[:, input_token_len:], skip_special_tokens=True)[0]
    
    # Convert to token IDs for return
    summarized_caption = tokenizer(
        summary_text, 
        return_tensors='pt', 
        add_special_tokens=False
    ).to(device)['input_ids']

    return summarized_caption, summary_text

def next_tokens_by_summary_guided_decoding(
    model,
    tokenizer,
    base_input_embeds,
    summarized_caption,
    input_ids,
    input_token_len,
    first_end_point,
    roll_back_index,
    model_kwargs,
    logits_processor,
    output_attentions=False,
    output_hidden_states=False,
    alpha=1,
):
    """Generate next tokens using summary-guided decoding."""
    device = input_ids.device
    
    # Get the current sentence being generated
    current_sentence = input_ids[:, first_end_point + 1 : roll_back_index]
    
    # Prepare summary-guided input
    summary_input_ids = torch.cat([summarized_caption, current_sentence], dim=-1)
    lvlm_input_ids = input_ids[:, :roll_back_index]
    
    # Prepare embeddings for different inputs
    # For LVLM input (up to roll_back_index)
    new_input_ids = lvlm_input_ids[:, 1:]  # Skip the first token as base_input_embeds already includes it
    changed_inputs_embeds = model.get_input_embeddings()(new_input_ids)
    inputs_embeds_lvlm = torch.cat([base_input_embeds, changed_inputs_embeds], dim=1)
    
    # For summary input
    changed_inputs_embeds_summary = model.get_input_embeddings()(summary_input_ids)
    inputs_embeds_summary = torch.cat([base_input_embeds, changed_inputs_embeds_summary], dim=1)
    
    # Update model kwargs
    model_kwargs_lvlm = copy.deepcopy(model_kwargs)
    model_kwargs_lvlm['inputs_embeds'] = inputs_embeds_lvlm
    size = model_kwargs_lvlm['inputs_embeds'].size()
    middle_value = size[1]
    ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(device)
    model_kwargs_lvlm['attention_mask'] = ones_tensor
    
    model_kwargs_summary = copy.deepcopy(model_kwargs)
    model_kwargs_summary['inputs_embeds'] = inputs_embeds_summary
    size = model_kwargs_summary['inputs_embeds'].size()
    middle_value = size[1]
    ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(device)
    model_kwargs_summary['attention_mask'] = ones_tensor
    
    # Prepare inputs
    lvlm_inputs = model.prepare_inputs_for_generation(new_input_ids, **model_kwargs_lvlm)
    summary_inputs = model.prepare_inputs_for_generation(summary_input_ids, **model_kwargs_summary)
    
    # Forward pass
    lvlm_outputs = model(**lvlm_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    summary_outputs = model(**summary_inputs, return_dict=True, output_attentions=False, output_hidden_states=output_hidden_states)
    
    # Get logits
    lvlm_next_token_logits = lvlm_outputs.logits[:, -1, :]
    summary_next_token_logits = summary_outputs.logits[:, -1, :]
    
    # Apply summary guidance
    diffs = lvlm_next_token_logits + alpha * (summary_next_token_logits - lvlm_next_token_logits)
    
    # Process logits and get next token
    contrastive_logits = logits_processor(lvlm_input_ids, diffs)
    next_tokens = torch.argmax(contrastive_logits, dim=-1)
    summary_output_tokens = tokenizer.convert_ids_to_tokens(next_tokens)
    
    # Update input_ids
    new_input_ids = torch.cat([lvlm_input_ids, next_tokens[:, None]], dim=-1)
    
    return summary_output_tokens, new_input_ids

def summary_guided_decoding(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    """
    Summary-Guided Decoding for InstructBLIP models.
    
    This method implements the summary-guided decoding algorithm that uses
    self-generated summaries to guide the generation process, particularly
    for content words (nouns, adjectives, numbers, proper nouns).
    """
    
    # Initialize values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )
    
    # Initialize attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    # Keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False  # used by synced_gpus only
    
    # Initialize tokenizer - you should update this path to your InstructBLIP tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models--lmsys--vicuna-7b-v1.1', use_fast=False, truncation_side="left")
    
    # Initialize tracking variables
    roll_back_index = -1
    generated_tokens = []
    input_token_len = model_kwargs['inputs_embeds'].shape[1]
    first_end_point = -1
    summary_check_point = -999
    check_input_embeds = -1
    
    # Save original embeddings
    saved_original_inputs_embeds = model_kwargs['inputs_embeds']
    
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break
        
        # Prepare model inputs
        if check_input_embeds == -1:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        else:
            # Update embeddings for new tokens
            new_input_ids = input_ids[:, -1].unsqueeze(0)
            inputs_embeds = model_kwargs['inputs_embeds']
            changed_inputs_embeds = self.get_input_embeddings()(new_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, changed_inputs_embeds], dim=1)
            model_kwargs['inputs_embeds'] = inputs_embeds
            size = model_kwargs['inputs_embeds'].size()
            middle_value = size[1]
            ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(input_ids.device)
            model_kwargs['attention_mask'] = ones_tensor
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        check_input_embeds = 1
        
        # Forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens_candidate = torch.argmax(next_tokens_scores, dim=-1)
        
        # Check if we need to apply summary-guided decoding
        generated_ids = torch.cat([input_ids, next_tokens_candidate[:, None]], dim=-1)
        generated_tokens = generated_ids.tolist()[0]
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
        token_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        check_finish = next_tokens_candidate.item()
        flag = 0
        
        # Check for sentence boundaries
        if token_text.endswith(".") or check_finish == 2 or check_finish == 1:
            # Generate POS tags
            result = generate_pos_tags(tokenizer, generated_ids)
            
            # Find content word indices
            indices = []
            for index_, word, tag in result:
                if tag in ['NOUN', 'ADJ', 'NUM', 'PROPN']:
                    indices.extend(index_)
            
            # Update roll_back_index
            for i_ in indices:
                if i_ > roll_back_index:
                    roll_back_index = i_
                    flag = 1
                    break
            
            # Handle first sentence
            if first_end_point == -1:
                first_end_point = result[-1][0][0]
                roll_back_index = result[-1][0][0]
            elif len(indices) == 0:
                first_end_point = result[-1][0][0]
            elif roll_back_index > indices[-1] or flag == 0:
                first_end_point = result[-1][0][0]
            elif roll_back_index <= first_end_point:
                roll_back_index = first_end_point
            else:
                # Apply summary-guided decoding
                if summary_check_point != first_end_point:
                    summary_check_point = first_end_point
                    
                    # Generate summary
                    to_be_summary_caption = tokenizer.batch_decode([generated_ids[0, 1:first_end_point+1]], skip_special_tokens=True)[0]
                    summarized_caption, self_summary_output_text = generate_summary_caption(
                        self,
                        tokenizer,
                        to_be_summary_caption,
                        model_kwargs,
                        input_ids.device,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states
                    )
                    
                    this_peer_finished = False
                
                # Apply summary guidance for next token
                summary_output_tokens, input_ids = next_tokens_by_summary_guided_decoding(
                    model=self,
                    tokenizer=tokenizer,
                    base_input_embeds=saved_original_inputs_embeds,
                    summarized_caption=summarized_caption,
                    input_ids=input_ids,
                    input_token_len=input_token_len,
                    first_end_point=first_end_point,
                    roll_back_index=roll_back_index,
                    model_kwargs=model_kwargs,
                    logits_processor=logits_processor,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    alpha=1,
                )
                
                if summary_output_tokens[0] == ".":
                    first_end_point = roll_back_index
                
                continue
        
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        
        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        
        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True
        
        if this_peer_finished and not synced_gpus:
            break
    
    if streamer is not None:
        streamer.end()
    
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def register_summary_guided_decoding():
    """Register the summary-guided decoding method to transformers GenerationMixin."""
    print("InstructBLIP Summary-Guided Decoding registered successfully!")
    transformers.generation.utils.GenerationMixin.summary_guided_decoding = summary_guided_decoding