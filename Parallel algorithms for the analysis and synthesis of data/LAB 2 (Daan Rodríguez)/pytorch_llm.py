import os
import time
import torch.distributed as dist

import torch
import transformers
from transformers import AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights


def init_model(rank):
    """
    Here we have some initialization of LLM
    Using accelerate framework with transformers we can load 7B model in 8GB GPU
    For demonstration we will use a 1.3B model, what will be loaded without GPU (it will take some time)
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B", trust_remote_code=True)
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config)

    if torch.cuda.device_count() > 1:
        device_map = infer_auto_device_map(model, max_memory={'{}'.format(rank): '8000MiB', 'cpu': '50GiB'},
                                           no_split_module_classes=["GPTNeoBlock"], dtype=torch.int8)
    else:
        device_map = infer_auto_device_map(model, max_memory={'cpu': '50GiB'}, no_split_module_classes=["GPTNeoBlock"],
                                           dtype=torch.float32)

    device_map = {k: 'cuda:' + v if 'cpu' not in v else v for k, v in device_map.items()}

    if torch.cuda.device_count() > 1:
        model_eval = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                                       device_map=device_map, load_in_8bit=True,
                                                                       torch_dtype=torch.float16)
    else:
        model_eval = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                                       device_map=device_map, load_in_8bit=False,
                                                                       torch_dtype=torch.float32)

    return model_eval, tokenizer


def init_distributed():
    try:
        dist_url = "tcp://127.0.0.1:29500"
        dist_backend = "nccl"
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    except Exception as e:
        print(f"Error during distributed environment initialization: {e}")
        raise

    return local_rank


if __name__ == "__main__":
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    local_rank = init_distributed()

    model, tokenizer = init_model(local_rank)
    # some initial parameters and "queries"
    max_length_result = 50
    input_texts = ['Hi! I want to tell you about', 'No, I don\'t want to', 'What???', 'Ha-ha-ha,']
    
    # split texts over different processes
    texts_per_process = len(input_texts) // local_world_size
    splits = [i for i in range(0, len(input_texts), texts_per_process)]
    input_ids = [tokenizer(input_text) for input_text in input_texts[splits[local_rank]: splits[local_rank] + texts_per_process]]
    
    batch_size = len(input_ids)
    max_length = max([len(i['input_ids']) for i in input_ids])
    
    # prepare input for the model
    for batch in input_ids:
        if len(batch['input_ids']) < max_length:
            batch['input_ids'] = torch.tensor([2] * (max_length - len(batch['input_ids'])) + batch['input_ids'])[None]
            batch['attention_mask'] = torch.tensor([2] * (max_length - len(batch['attention_mask'])) + batch['attention_mask'])[None]
        else:
            batch['input_ids'] = torch.tensor(batch['input_ids'])[None]
            batch['attention_mask'] = torch.tensor(batch['attention_mask'])[None]
            
    batch = {}
    batch['input_ids'] = torch.cat([i['input_ids'] for i in input_ids])
    batch['attention_mask'] = torch.cat([i['attention_mask'] for i in input_ids])
        
    start_time = time.time()
    for token_num in range(max_length, max_length_result):
        # here we don't want to calculate and use gradients
        with torch.no_grad():
            output = model(**batch, output_hidden_states=True)
            
        # perform some simple sampling from generated probabilities
        logits = output.logits[:, -1] # get the last token
        tokens_to_zero = logits < torch.topk(logits.float(), 150)[0][..., -1, None] # select 150 most probable tokens and remove others
        
        logits[tokens_to_zero] = -torch.inf # replace non top-150 tokens with -inf
        logits_probs = torch.nn.functional.softmax(logits.float(), -1) # move to 0 to 1 values
        
        sampled_tokens = torch.multinomial(logits_probs, num_samples=1)
        new_input = torch.cat([batch['input_ids'], sampled_tokens], 1)
        new_attention = torch.cat([batch['attention_mask'], torch.ones((batch_size, 1))], 1)
        
        batch['input_ids'] = new_input
        batch['attention_mask'] = new_attention
        
    buffer = torch.cat([batch['input_ids'] for _ in range(local_world_size)])
    dist.all_gather(tensor_list=[buffer], tensor=buffer)
        
    end_time = time.time()
    buffer = torch.cat(buffer)
    
    if local_rank == 0:
        print('Generation time: {}'.format(end_time - start_time))
        print('Rank: {}'.format(local_rank))
        texts = tokenizer.batch_decode(buffer)
        
        for text in texts:
            print(text.replace('\n', '\t'), '\n')