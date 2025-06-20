import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from transformers.generation.utils import GreedySearchOutput

import torch
import torch.nn.functional as F
import spacy

nlp = spacy.load('en_core_web_sm')

def process_tokens(tokens):
    full_text = ""
    for token in tokens:
        if token.startswith("▁"):
            full_text += " " + token[1:]
        elif token in ["<0x0A>", "</s>", "<s>", "<unk>"]:
            full_text += ""
        else:
            full_text += token
    return full_text.strip()

def align_probabilities(llama_tokens, nltk_tokens):
    llama_index = 0
    llama_indexs = []
    combined_token = ""
    nltk_save = ""
    
    for nltk_token in nltk_tokens:
        token_prob_sum = 0
        token_length = 0
        llama_temp = []
        
        while llama_index < len(llama_tokens):
            llama_token = llama_tokens[llama_index].replace("▁", "").replace("<0x0A>", "").replace("</s>", "").replace("<unk>", "").replace("<s>", "")
            if llama_token == "":
                llama_index += 1
                continue
            combined_token += llama_token
            nltk_save = nltk_save + nltk_token

            llama_temp.append(llama_index)
            llama_index += 1

            if combined_token == nltk_token or combined_token == nltk_save:
                nltk_save = ""
                break
            elif len(combined_token) > len(nltk_token):
                llama_index = llama_index - 1
                break

        llama_indexs.append(llama_temp)
        combined_token = ""

    return llama_indexs

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

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )
    
    model_kwargs_llm = copy.deepcopy(model_kwargs)
    model_kwargs_llava_summary = copy.deepcopy(model_kwargs)
    
    model_kwargs_llm['inputs_embeds'] = model_kwargs.get("inputs_embeds_llm")
    size = model_kwargs_llm['inputs_embeds'].size()
    middle_value = size[1]
    ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(input_ids.device)
    model_kwargs_llm['attention_mask'] = ones_tensor

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    first_end_point = -1
    roll_back_index = -1
    summary_check_point = -999

    input_token_len = model_kwargs['inputs_embeds'].shape[1]
    input_llm_token_len = model_kwargs_llm['inputs_embeds'].shape[1]

    saved_original_inputs_embeds = model_kwargs['inputs_embeds']
    saved_original_inputs_embeds_llm = model_kwargs_llm['inputs_embeds']

    from transformers import LlamaTokenizer
    my_tokenizer = LlamaTokenizer.from_pretrained('models--lmsys--vicuna-7b-v1.1', use_fast=False, truncation_side="left")
    
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    last_checkpoint = "disttled_flan_t5_checkpoint"
    
    finetuned_summarize_model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    summarize_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    
    my_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    my_tokenizer.add_special_tokens({'bos_token': '</s>'})
    my_tokenizer.add_special_tokens({'eos_token': '</s>'})
    my_tokenizer.add_special_tokens({'unk_token': '</s>'})
    
    check_input_embeds = -1
    
    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break
        
        if check_input_embeds == -1:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        else:
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

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens_candidate = torch.argmax(next_tokens_scores, dim=-1)
        generated_ids = torch.cat([input_ids, next_tokens_candidate[:, None]], dim=-1)
        generated_tokens = generated_ids.tolist()[0]
        generated_tokens = my_tokenizer.convert_ids_to_tokens(generated_tokens)
        token_text = my_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        check_finish = next_tokens_candidate.item()
        
        flag = 0
        if token_text.endswith(".") or check_finish == 2 or check_finish == 1:
            indices = []
            remain_indices = []
            full_text = process_tokens(generated_tokens)
            doc = nlp(full_text)
            spacy_tokens = [token.text for token in doc]
            pos_tags = [(token.text, token.pos_) for token in doc]
            llama_indexs = align_probabilities(generated_tokens, spacy_tokens)
            result = [(prob, word, tag) for (prob, (word, tag)) in zip(llama_indexs, pos_tags)]
            
            for index_, word, tag in result:
                if tag in ['NOUN', 'ADJ', 'NUM', 'PROPN']:
                    indices.extend(index_)
            
            for i_ in indices:
                if i_ > roll_back_index:
                    roll_back_index = i_
                    flag = 1
                    break

            if first_end_point == -1:
                first_end_point = result[-1][0][0]
                roll_back_index = result[-1][0][0]
            elif len(indices) == 0:
                first_end_point = result[-1][0][0]
            elif roll_back_index > indices[-1] or flag == 0:
                first_end_point = result[-1][0][0]
                pass
            elif roll_back_index <= first_end_point:
                roll_back_index = first_end_point
            else:
                if summary_check_point != first_end_point:
                    summary_check_point = first_end_point
                    split_token_text = token_text.split(".")
                    
                    to_be_summary_caption = my_tokenizer.batch_decode([generated_ids[0, 1:first_end_point+1]], skip_special_tokens=True)[0]
                    prompt = to_be_summary_caption + "\nWhat is a summary of this text?"
                    summarized_inputs_ = summarize_tokenizer(prompt, return_tensors='pt')
                    
                    output = summarize_tokenizer.decode(
                        finetuned_summarize_model.generate(
                            summarized_inputs_["input_ids"], 
                            max_new_tokens=512,
                        )[0], 
                        skip_special_tokens=True
                    )
                    
                    summarized_caption = my_tokenizer(output, return_tensors='pt', add_special_tokens=False).to(input_ids.device)
                    summarized_caption = summarized_caption['input_ids']
                
                model_kwargs_llava_summary_now = copy.deepcopy(model_kwargs_llava_summary)
                model_kwargs_llava_summary_now['inputs_embeds'] = None
                model_kwargs_llava_summary_now['inputs_embeds_llm'] = None    
                model_kwargs_llava_summary_now['attention_mask'] = None
                
                current_sentence = input_ids[:, first_end_point+1:roll_back_index]
                new_summarized_input_ids = torch.cat([summarized_caption, current_sentence], dim=-1)

                lvlm_input_ids = input_ids[:, :roll_back_index]
                lvlm_input_text = my_tokenizer.batch_decode(lvlm_input_ids, skip_special_tokens=True)
                lvlm_input_text = [text.split('###')[0].strip() for text in lvlm_input_text]

                summary_input_text = my_tokenizer.batch_decode(new_summarized_input_ids, skip_special_tokens=True)
                summary_input_text = [text.split('###')[0].strip() for text in summary_input_text]

                new_input_ids = lvlm_input_ids[:, 1:]
                changed_inputs_embeds = self.get_input_embeddings()(new_input_ids)
                inputs_embeds = torch.cat([saved_original_inputs_embeds, changed_inputs_embeds], dim=1)
                model_kwargs['inputs_embeds'] = inputs_embeds
                size = model_kwargs['inputs_embeds'].size()
                middle_value = size[1]
                ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(input_ids.device)
                model_kwargs['attention_mask'] = ones_tensor
                lvlm_inputs = self.prepare_inputs_for_generation(new_input_ids, **model_kwargs)

                changed_inputs_embeds = self.get_input_embeddings()(new_summarized_input_ids)
                inputs_embeds_summary = torch.cat([saved_original_inputs_embeds, changed_inputs_embeds], dim=1)
                model_kwargs_llava_summary_now['inputs_embeds'] = inputs_embeds_summary
                size = model_kwargs_llava_summary_now['inputs_embeds'].size()
                middle_value = size[1]
                ones_tensor = torch.ones(1, middle_value, dtype=torch.int32).to(input_ids.device)
                model_kwargs_llava_summary_now['attention_mask'] = ones_tensor
                summary_inputs = self.prepare_inputs_for_generation(new_summarized_input_ids, **model_kwargs_llava_summary_now)

                lvlm_outputs = self(
                    **lvlm_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                summary_outputs = self(
                    **summary_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                summary_next_token_logits = summary_outputs.logits[:, -1, :]
                alpha = 1
                diffs = summary_next_token_logits
                contrastive_logits = diffs
                next_tokens = torch.argmax(contrastive_logits, dim=-1)
                
                summary_output_tokens = my_tokenizer.convert_ids_to_tokens(next_tokens)
                input_ids = torch.cat([lvlm_input_ids, next_tokens[:, None]], dim=-1)
                
                if summary_output_tokens[0] == ".":
                    first_end_point = roll_back_index

                continue   

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

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

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

def summary_guided_decoding_function():
    
    transformers.generation.utils.GenerationMixin.summary_guided_decoding = summary_guided_decoding