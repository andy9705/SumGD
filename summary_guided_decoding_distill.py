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
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist()[0])
    full_text = process_tokens(generated_tokens)
    doc = nlp(full_text)
    spacy_tokens = [token.text for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    llama_indices = align_probabilities(generated_tokens, spacy_tokens)
    token_pos_pairs = [(prob, word, tag) for (prob, (word, tag)) in zip(llama_indices, pos_tags)]
    return token_pos_pairs


def generate_summary_caption(self, summarize_tokenizer,llava_tokenizer, distilled_summarize_model,to_be_summary_caption, device):
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
    
    
    # Create summary prompt template and tokenize
    summary_template = to_be_summary_caption+"\nWhat is a summary of this text?"
    summarized_inputs_ = summarize_tokenizer(summary_template, return_tensors='pt')
    output = summarize_tokenizer.decode(
                distilled_summarize_model.generate(
                    summarized_inputs_["input_ids"], 
                    max_new_tokens=512,
                )[0], 
                skip_special_tokens=True
            )
    
    summarized_caption = llava_tokenizer(output, return_tensors='pt', add_special_tokens=False).to(device)
    summarized_caption = summarized_caption['input_ids']

    return summarized_caption


def next_tokens_by_summary_guided_decoding(
    model,
    tokenizer,
    base_input,
    summarized_caption,
    input_ids,
    input_ids_llm,
    input_token_len,
    input_llm_token_len,
    first_end_point,
    roll_back_index,
    model_kwargs,
    logits_processor,
    output_attentions=False,
    output_hidden_states=False,
    alpha=1,
):
    device = input_ids.device
    
    
    current_sentence = input_ids[:, input_token_len + first_end_point + 1 : input_token_len + roll_back_index]

    
    summary_input_ids = torch.cat([base_input, summarized_caption, current_sentence], dim=-1)
    lvlm_input_ids = input_ids[:, :input_token_len + roll_back_index]
    llm_input_ids = input_ids_llm[:, :input_llm_token_len + roll_back_index]

    
    def decode_text(input_ids, token_len):
        texts = tokenizer.batch_decode(input_ids[:, token_len:], skip_special_tokens=True)
        return [text.split('###')[0].strip() for text in texts]

    lvlm_input_text = decode_text(lvlm_input_ids, input_token_len)
    llm_input_text = decode_text(llm_input_ids, input_llm_token_len)
    summary_input_text = decode_text(summary_input_ids, input_token_len)

    
    attention_mask_summary = torch.ones_like(summary_input_ids).to(device)
    attention_mask_lvlm = torch.ones_like(lvlm_input_ids).to(device)
    attention_mask_llm = torch.ones_like(llm_input_ids).to(device)

    model_kwargs_summary = copy.deepcopy(model_kwargs)
    model_kwargs_summary['attention_mask'] = attention_mask_summary
    model_kwargs_lvlm = copy.deepcopy(model_kwargs)
    model_kwargs_lvlm['attention_mask'] = attention_mask_lvlm
    model_kwargs_llm = copy.deepcopy(model_kwargs)
    model_kwargs_llm['attention_mask'] = attention_mask_llm

    
    lvlm_inputs = model.prepare_inputs_for_generation(lvlm_input_ids, **model_kwargs_lvlm)
    summary_inputs = model.prepare_inputs_for_generation(summary_input_ids, **model_kwargs_summary)
    llm_inputs = model.prepare_inputs_for_generation_llm(llm_input_ids, **model_kwargs_llm)

    
    lvlm_outputs = model(**lvlm_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    summary_outputs = model(**summary_inputs, return_dict=True, output_attentions=False, output_hidden_states=output_hidden_states)
    llm_outputs = model(**llm_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)

    
    lvlm_next_token_logits = lvlm_outputs.logits[:, -1, :]
    summary_next_token_logits = summary_outputs.logits[:, -1, :]
    llm_next_token_logits = llm_outputs.logits[:, -1, :]

    
    diffs = lvlm_next_token_logits + alpha * (summary_next_token_logits - lvlm_next_token_logits)

    
    contrastive_logits = logits_processor(lvlm_input_ids, diffs)
    next_tokens = torch.argmax(contrastive_logits, dim=-1)
    summary_output_tokens = tokenizer.convert_ids_to_tokens(next_tokens)

    
    new_input_ids = torch.cat([lvlm_input_ids, next_tokens[:, None]], dim=-1)
    new_input_ids_llm = torch.cat([llm_input_ids, next_tokens[:, None]], dim=-1)
    

    return summary_output_tokens,  new_input_ids, new_input_ids_llm

def summary_guided_decoding(
        self,
        input_ids: torch.LongTensor,
        input_ids_llm: torch.LongTensor,
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
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
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

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        model_kwargs_llm = copy.deepcopy(model_kwargs)
        model_kwargs_llm['attention_mask']=torch.ones_like(input_ids_llm).to(input_ids.device)
        this_peer_finished = False  # used by synced_gpus only

        
        from transformers import AutoTokenizer
        llava_tokenizer = AutoTokenizer.from_pretrained('/home/kyungmin/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965', use_fast=False)
        

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        ## Distilled_summarizer_load
        last_checkpoint = "/home/kyungmin/my_project/OPERA/caption-summary-training-flan-t5-base-20240912-real/checkpoint-20910"  
        distilled_summarize_model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint) 
        summarize_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')


        roll_back_index=-1
        generated_tokens = []
        base_input =input_ids
        input_token_len = input_ids.shape[1]
        input_llm_token_len = input_ids_llm.shape[1]
        first_end_point=-1
        summary_check_point = -999

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_kwargs['attention_mask']=torch.ones_like(input_ids).to(input_ids.device)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_kwargs_llm['attention_mask']=torch.ones_like(input_ids_llm).to(input_ids.device)
            model_inputs_llm = self.prepare_inputs_for_generation_llm(input_ids_llm, **model_kwargs_llm)
            

            now_input_ids_len = input_ids.shape[1] - input_token_len
            
            
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False, #output_attentions
                output_hidden_states=output_hidden_states,
            )
            outputs_llm = self(
                **model_inputs_llm,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_candidate = torch.argmax(next_token_logits, dim=-1)
            


            generated_ids = torch.cat([input_ids[:, input_token_len:], next_tokens_candidate[:, None]], dim=-1)
            generated_tokens = generated_ids.tolist()[0]
            generated_tokens = llava_tokenizer.convert_ids_to_tokens(generated_tokens)
            token_text = llava_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            token_text = token_text[0]
            flag=0
            

            
            result = generate_pos_tags(llava_tokenizer, generated_ids)

            if roll_back_index > result[-1][0][0]:
                pass
            elif next_tokens_candidate.item() == 2:
                
                flag_point_second = -99
                for _result_index, (roll_index, __word, __pos) in enumerate(result):
                    
                    if roll_back_index in roll_index:
                        if __pos in ['NOUN','ADJ','NUM','PROPN']:
                            
                            flag_point_second=1
                            
                            if summary_check_point != first_end_point:
                                
                                summary_check_point = first_end_point
                                split_token_text = token_text.split(".")
                                
                                to_be_summary_caption=""
                                for i in range(len(split_token_text)-1):
                                    to_be_summary_caption=to_be_summary_caption+split_token_text[i]+"."

                                
                                summarized_caption = generate_summary_caption(
                                    self,
                                    summarize_tokenizer,
                                    llava_tokenizer,
                                    distilled_summarize_model,
                                    to_be_summary_caption,
                                    input_ids.device
                                )
                                
                                
                                

                            


                            
                            summary_output_tokens,  input_ids, input_ids_llm = next_tokens_by_summary_guided_decoding(
                                model=self,
                                tokenizer=llava_tokenizer,
                                base_input=base_input,
                                summarized_caption=summarized_caption,
                                input_ids=input_ids,
                                input_ids_llm=input_ids_llm,
                                input_token_len=input_token_len,
                                input_llm_token_len=input_llm_token_len,
                                first_end_point=first_end_point,
                                roll_back_index=roll_back_index,
                                model_kwargs=model_kwargs,
                                logits_processor=logits_processor,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                alpha=1,
                            )
                            if summary_output_tokens[0] == ".":
                                first_end_point=roll_back_index
                        if __word == ".":
                            first_end_point = roll_index[0]
            
                

                roll_back_index = roll_back_index + 1
                
                if flag_point_second == 1:
                    continue
                
                flag_point_fourth = -99
                for plpl in result[-1][0]:
                    if roll_back_index <= plpl:
                        flag_point_fourth=1
                if flag_point_fourth == 1:
                    continue

            elif token_text.endswith(".") and first_end_point == -1: ## Original Decoding for first Sentence. Summary-GUided Decoding Apply for second sentence.
                first_end_point=result[-1][0][0]
                roll_back_index = result[-1][0][0]
            
            
            elif first_end_point != -1:
                
                    

                
                flag_point_third = -99
                for _result_index, (roll_index, __word, __pos) in enumerate(result):
                    
                    if roll_back_index in roll_index:
                        flag_point_third = 1
                if flag_point_third == -99:
                    roll_back_index = roll_back_index+1
                    continue
                
                my_flag_point = -99
                for _result_index, (roll_index, __word, __pos) in enumerate(result):
                    
                    if roll_back_index in roll_index:
                        
                        if _result_index +2 < len(result):
                            my_flag_point = 1
                
                if my_flag_point == 1:
                    
                    flag_point_second = -99
                    for _result_index, (roll_index, __word, __pos) in enumerate(result):
                        
                        if roll_back_index in roll_index:
                            if __pos in ['NOUN','ADJ','NUM','PROPN']:
                                ## 이때 summary-guided decoding 진행.
                                flag_point_second=1
                            


                                

                                
                                
                                



                                if summary_check_point != first_end_point:
                                    
                                    summary_check_point = first_end_point
                                    
                                    split_token_text = token_text.split(".")
                                    
                                    to_be_summary_caption=""
                                    for i in range(len(split_token_text)-1):
                                        to_be_summary_caption=to_be_summary_caption+split_token_text[i]+"."

                                
                                    summarized_caption = generate_summary_caption(
                                    self,
                                    summarize_tokenizer,
                                    llava_tokenizer,
                                    distilled_summarize_model,
                                    to_be_summary_caption,
                                    input_ids.device
                                    )

                                


                                summary_output_tokens,  input_ids, input_ids_llm = next_tokens_by_summary_guided_decoding(
                                model=self,
                                tokenizer=llava_tokenizer,
                                base_input=base_input,
                                summarized_caption=summarized_caption,
                                input_ids=input_ids,
                                input_ids_llm=input_ids_llm,
                                input_token_len=input_token_len,
                                input_llm_token_len=input_llm_token_len,
                                first_end_point=first_end_point,
                                roll_back_index=roll_back_index,
                                model_kwargs=model_kwargs,
                                logits_processor=logits_processor,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                alpha=1,
                                )
                                if summary_output_tokens[0] == ".":
                                    first_end_point=roll_back_index

                    
                            if __word == ".":
                                first_end_point = roll_index[0]
                
                    
                    roll_back_index = roll_back_index + 1   
                    if flag_point_second == 1:
                        continue

            next_token_logits_llm = outputs_llm.logits[:,-1,:]

            

            
            
            
            
            
            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            
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
            input_ids_llm = torch.cat([input_ids_llm, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_kwargs_llm = self._update_model_kwargs_for_generation(
                outputs_llm, model_kwargs_llm, is_encoder_decoder=self.config.is_encoder_decoder
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

def summary_guided_decoding_function():
    
    transformers.generation.utils.GenerationMixin.summary_guided_decoding = summary_guided_decoding