import json
from pathlib import Path
from .common import digest,save,sha


def prepare(files,tokenizer,output,samples,copies,context,max_new_tokens,prompt_field='prompt'):
    if samples < 1 or copies < 1:
        raise ValueError('samples and copies must be positive')
    if Path(output).exists():
        raise FileExistsError(output)
    subsets = {};audit = {};seen_names = set()
    for filename in files:
        path = Path(filename);name = path.stem
        if name in seen_names:
            raise ValueError('Duplicate subset name: '+name)
        seen_names.add(name)
        records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if len(records) < samples:
            raise ValueError(f'{name} has{len(records)} records, requested{samples}')
        selected = []
        for i,row in enumerate(records[:samples]):
            prompt = row[prompt_field]
            if not isinstance(prompt,str) or not prompt.strip():
                raise ValueError(f'{name}[{i}]: missing nonempty prompt')
            ids = tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,return_dict=False)
            if hasattr(ids,'keys'):
                ids = ids['input_ids']
            if not isinstance(ids,list) or not ids or any(type(t) is not int for t in ids):
                raise ValueError('Tokenizer must return one list[int]')
            if len(ids)+max_new_tokens+16 > context:
                raise ValueError(f'{name}[{i}] does not fit context; increase --context, no silent truncation')
            selected.append(dict(original_index=i,question_id=str(row.get('task_id',row.get('question_id',i))),
                                 prompt_sha256=digest(prompt),input_ids=ids))
        data = [dict(row,copy_index=copy) for copy in range(copies) for row in selected]
        subsets[name] = data
        audit[name] = dict(source_file=path.name,source_sha256=sha(path),original_count=samples,copies=copies,
            requests=len(data),min_input_tokens=min(len(r['input_ids']) for r in data),max_input_tokens=max(len(r['input_ids']) for r in data),
            input_sha256=digest([r['input_ids'] for r in data]))
    overlap = []
    for i,a in enumerate(subsets):
        for b in list(subsets)[i+1:]:
            common = {r['prompt_sha256'] for r in subsets[a]} & {r['prompt_sha256'] for r in subsets[b]}
            if common:
                overlap.append(dict(a=a,b=b,unique_prompt_overlap=len(common)))
    prepared = dict(format_version=1,tokenizer_name=getattr(tokenizer,'name_or_path','unknown'),
        chat_template_sha256=digest(tokenizer.chat_template),context=context,max_new_tokens=max_new_tokens,
        subsets=subsets,audit=audit,prompt_overlap=overlap)
    tokenizer_root = Path(prepared['tokenizer_name'])
    prepared['tokenizer_file_hashes'] = {p.name:sha(p) for p in tokenizer_root.iterdir() if p.is_file() and p.suffix in ['.json','.model','.tiktoken']} if tokenizer_root.is_dir() else {}
    save(output,prepared)
    return prepared

