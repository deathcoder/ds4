from pathlib import Path
import json,hashlib,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];a=json.loads((w/'selector-v5-acceptance.json').read_text());assert a['passed'] and a['rows'][0]==a['rows'][1];d=root/'rust-star/fixtures/prefill-indexer-selector-4128-v5';d.mkdir();old=json.loads((root/'rust-star/fixtures/prefill-layer23-continuation-second-tile-4128-v1/manifest.json').read_text());tensors=[]
for name,dtype,shape,role in [('scores-tile.f32le.bin','f32',[32,2048],'input'),('topk-tile.i32le.bin','i32',[32,512],'output')]:
 b=(w/'selector-v5-a'/name).read_bytes();assert b==(w/'selector-v5-b'/name).read_bytes();(d/name).write_bytes(b);tensors.append({'name':name.split('.')[0],'hook':'indexer_scores' if role=='input' else 'indexer_topk','role':role,'dtype':dtype,'shape':shape,'encoding':'little-endian-ieee754-binary32' if dtype=='f32' else 'little-endian-signed-integer32','path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
m={'schema':'rust-star-differential-fixture-v1','fixture_id':'dwarfstar-oracle-v5-prefill-indexer-selector-4128','captured_at_utc':a['captured_at_utc'],'oracle':{'id':'oracle-v5','repository':'https://github.com/deathcoder/ds4.git','commit':a['source_commit'],'tree':'a0a33f4a00c3c78cd40350785485203e7c6b0b1d','capture_executable_sha256':a['executable_sha256']},'model':old['model'],'capture':dict(a,backend='metal',machine='Apple M1 Ultra,48 GPU cores,128 GB unified memory',performance_enabled=False),'scope':{'kind':'layer-segment','phase':'prefill','layer':2,'position':4159,'captured_position_range':[4128,4159],'tile_rows':32},'operations':[{'name':'deterministic_indexer_selection','kernel':'kernel_argsort_f32_i32_desc+kernel_argsort_merge_f32_i32_desc'}],'tensors':tensors};(d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=root/'rust-star/runtime/src/metal.rs';s=p.read_text();oldblock='''        let expected_second_l2_topk = second_i32!(
            "layer2-indexer-topk-tile2.i32le.bin",
            "second-tile layer-2 indexer top-k"
        );''';newblock='''        // The selector's tie permutation has its own v5 target; the other
        // second-tile tensors retain their historical producer identities.
        let expected_second_l2_topk = decode_i32_fixture(
            include_bytes!("../../fixtures/prefill-indexer-selector-4128-v5/topk-tile.i32le.bin"),
            "oracle-v5 second-tile layer-2 indexer top-k",
        )?;''';assert oldblock in s;s=s.replace(oldblock,newblock);needle='    crate::artifact::write_json_string(output, report.second_tile_fixture_id)?;';assert s.count(needle)==1;s=s.replace(needle,needle+'''\n    write!(output, ", \\\"selector_fixture\\\": ")?;
    crate::artifact::write_json_string(output, "dwarfstar-oracle-v5-prefill-indexer-selector-4128")?;''');p.write_text(s)
