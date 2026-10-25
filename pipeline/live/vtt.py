import re, sys
def clean(path, every=15):
    txt=open(path).read()
    out=[]; seen=set(); cur=None; buf=[]
    for block in txt.split('\n\n'):
        m=re.search(r'(\d+):(\d+):(\d+)\.\d+ -->',block)
        if not m: continue
        t=int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))
        lines=[re.sub(r'<[^>]+>','',l).strip() for l in block.split('\n')[1:]]
        for l in lines:
            if l and l not in seen and '-->' not in l:
                seen.add(l)
                b=t//every*every
                if b!=cur:
                    if buf: out.append(f'[{cur//60}:{cur%60:02d}] '+' '.join(buf))
                    cur=b; buf=[]
                buf.append(l)
    if buf: out.append(f'[{cur//60}:{cur%60:02d}] '+' '.join(buf))
    return '\n'.join(out)
if __name__=='__main__':
    print(clean(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 15))
