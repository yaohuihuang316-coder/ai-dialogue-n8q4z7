"""Build prefix coding inputs from canonical source JSON.

Input: [{"source_key": str, "messages": [{"message_id": str,
        "role": "user"|"assistant", "content": str}, ...]}, ...]
Output: one task-opening prefix and consecutive user-action prefixes per source.
Goal segmentation and behavioral labels are supplied by the coding prompt.
"""
from pathlib import Path
import argparse, hashlib, json

def build(records):
    units=[];seen=set()
    for source in records:
        key=source['source_key'];assert key not in seen;seen.add(key)
        messages=source['messages'];ids=[m['message_id'] for m in messages]
        assert len(ids)==len(set(ids))
        previous=None
        for index,message in enumerate(messages):
            if message['role']!='user' or not message['content'].strip():continue
            prefix=messages[:index+1]
            payload=json.dumps(prefix,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            units.append(dict(source_key=key,kind='opening' if previous is None else 'action',previous_user_message_id=previous,target_user_message_id=message['message_id'],messages=prefix,input_hash=hashlib.sha256(payload.encode()).hexdigest()))
            previous=message['message_id']
    return units

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    records=json.loads(a.input.read_text(encoding='utf-8'))
    units=build(records)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(units,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(json.loads(a.output.read_text(encoding='utf-8')))==len(units)
    print(json.dumps(dict(sources=len(records),prefixes=len(units))))
