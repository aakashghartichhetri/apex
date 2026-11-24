"""Collect actual paired model outputs on GSM8K; no simulated results here."""
import argparse,json,re,time
from pathlib import Path

def answer(text):
    matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    return matches[-1].replace(',','') if matches else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--small',required=True);p.add_argument('--large',required=True);p.add_argument('--split',choices=['train','test'],default='test');p.add_argument('--offset',type=int,default=0);p.add_argument('--n',type=int,default=50);p.add_argument('--out',required=True);a=p.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM
    data=load_dataset('openai/gsm8k','main',split=a.split)
    if a.offset<0 or a.n<1 or a.offset+a.n>len(data):p.error('invalid sample range')
    records=[dict(id=i,question=data[i]['question'],gold=answer(data[i]['answer'])) for i in range(a.offset,a.offset+a.n)]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for label,model_id in [('small',a.small),('large',a.large)]:
        tok=AutoTokenizer.from_pretrained(model_id);model=AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        for row in records:
            prompt='Solve the problem. Finish with #### followed by the numerical answer.\n'+row['question']
            inputs=tok(prompt,return_tensors='pt').to(device)
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
            if device=='cuda':torch.cuda.synchronize()
            duration=time.perf_counter()-start
            tokens=output[0,inputs['input_ids'].shape[1]:];text=tok.decode(tokens,skip_special_tokens=True)
            row[label]=dict(model=model_id,text=text,answer=answer(text),correct=answer(text) is not None and answer(text)==row['gold'],seconds=duration,tokens=len(tokens))
        del model
        if device=='cuda':torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(dict(split=a.split,device=device,torch=torch.__version__,records=records),indent=2)+'\n')
if __name__=='__main__':main()
