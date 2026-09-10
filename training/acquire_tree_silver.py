"""Persist a bounded MLCPD sample and generate Tree-sitter silver annotations."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .streaming_sources import SourceLedger, iter_mlcpd_sources
from .tree_sitter_teacher import annotate
from .teachers import validate_annotation_result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True); p.add_argument('--max-examples',type=int,default=1000)
    p.add_argument('--max-bytes',type=int,default=262144); p.add_argument('--ledger',required=True)
    p.add_argument('--mlcpd-file',action='append',dest='files')
    a=p.parse_args(); root=Path(a.output_dir); source_dir=root/'sources'; source_dir.mkdir(parents=True,exist_ok=True)
    ann=root/'annotations.jsonl'; files=a.files or ['python_parsed_1.parquet','c_parsed_1.parquet','cpp_parsed_1.parquet','java_parsed_1.parquet','javascript_parsed_1.parquet','go_parsed_1.parquet','rust_parsed_1.parquet','typescript_parsed_1.parquet']
    count=0
    with ann.open('w',encoding='utf-8') as out, SourceLedger(a.ledger) as ledger:
        for source in iter_mlcpd_sources(files,a.max_bytes,ledger,a.max_examples):
            if count>=a.max_examples: break
            digest=source.content_sha256; path=source_dir/(digest+'.'+(source.spec.language or 'unknown'))
            path.write_bytes(source.raw)
            result=validate_annotation_result(annotate({'language':source.spec.language or 'unknown','source':source.text}),source.text)
            row={'uri':str(path.resolve()),'language':source.spec.language,'licenseId':source.spec.license_id,'sourceId':source.spec.source_id,'split':'train','contentSha256':digest,**result}
            out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n'); out.flush()
            ledger.record(source,'PERSISTED_ANNOTATED',labels={'teacher':result['teacher']['name'],'definitions':len(result['definitions']),'usages':len(result['usages'])},teacher_versions={'annotation':result['teacher']})
            count+=1
    print(json.dumps({'examples':count,'annotations':str(ann),'sources':str(source_dir),'ledger':a.ledger}))
    return 0 if count else 2
if __name__=='__main__': raise SystemExit(main())
