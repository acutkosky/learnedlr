from datasets import load_from_disk, load_dataset

def load_wikitext(config, tokenizer, split, stream=False):
    # ds_path = f"data/manual_saves/wikitext-2-v1/bs-{config.batch_size}-ml-{config.context_length}-{split}"
    # try:
    #     wikitext = load_from_disk(ds_path)
    #     print("loaded!")
    # except FileNotFoundError:   
    # print("file not found, generating from scratch")
    wikitext = load_dataset('wikitext', 'wikitext-2-v1', split=split, streaming=stream, cache_dir=f'~/temp/stream{stream}')
    wikitext = wikitext.filter(lambda x: len(x['text']) > 1)
    # wikitext.shuffle()
    wikitext = wikitext.map(lambda examples: tokenizer(examples["text"], 
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=config.context_length),
                            remove_columns=["text"],
                            batched=True,
                            batch_size=config.batch_size)
    # wikitext.save_to_disk(ds_path)
    return wikitext

def load_c4(config, tokenizer, split):
    ds_path = f"/projectnb/aclab/datasets/c4/en/"
    c4 = load_dataset('c4', 'en', data_dir=ds_path, streaming=True, split=split)
    c4 = c4.filter(lambda x: len(x['text']) > 1)
    c4 = c4.map(lambda examples: tokenizer(examples["text"], 
                                                        padding=True,
                                                        truncation=True,
                                                        max_length=config.context_length),
                                remove_columns=["text", "timestamp", "url"],
                                batched=True,
                                batch_size=config.batch_size)
    return c4


