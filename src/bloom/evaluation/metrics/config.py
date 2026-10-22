"""Explicit native-panel bindings and conservative V5 header discovery."""
from pathlib import Path
import json
import re
from .registry import canonical_identity

def plain(x):
    if isinstance(x,bytes): return x.decode('utf-8')
    if hasattr(x,'tolist'): return plain(x.tolist())
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x

def load_config(path):
    path=Path(path).expanduser().resolve()
    data=json.loads(path.read_text())
    if data.get('schema_version')!=1: raise ValueError('Expected metrics config schema_version=1')
    for section in ('raw_sources','training_logs','training_summaries'):
        for row in data.get(section,[]):
            p=Path(row['path']).expanduser()
            row['path']=str((path.parent/p).resolve() if not p.is_absolute() else p)
            if row.get('reference_path'):
                p=Path(row['reference_path']).expanduser()
                row['reference_path']=str((path.parent/p).resolve() if not p.is_absolute() else p)
            for binding in row.get('references',{}).values():
                if binding.get('reference_path'):
                    p=Path(binding['reference_path']).expanduser()
                    binding['reference_path']=str((path.parent/p).resolve() if not p.is_absolute() else p)
    data['legacy_inputs']=[str((path.parent/Path(p)).resolve()) if not Path(p).expanduser().is_absolute() else str(Path(p).expanduser()) for p in data.get('legacy_inputs',[])]
    data['_config_path']=str(path)
    return data

def inspect_header(source):
    import h5py
    row=dict(source)
    dataset,panel,model,family=canonical_identity(row['dataset'],row['panel'],row['model'],row.get('family'))
    row.update(dataset=dataset,panel=panel,model=model,family=family)
    with h5py.File(row['path'],'r') as h:
        a={k:plain(v) for k,v in h.attrs.items()}
        for key in ['prediction_expression','truth_expression','gene_names','prediction_time','truth_time','prediction_cell_index','truth_cell_index']:
            if key not in h: raise ValueError(f'Missing canonical paired dataset {key}')
        if a.get('temporal_direction','forward')!='forward': raise ValueError('Only measured forward prediction is supported')
        if a.get('model_version')=='v7': raise ValueError('Bidirectional BLOOM is excluded')
        raw_dataset=str(a.get('dataset_id','')).lower()
        aliases={'big':'big','big_dataset':'big','big_prenatal':'big','big_prenatal_scalea':'big','small':'small','small_dataset':'small','small_visual_cortex':'small','sted':'sted','sted_dataset':'sted','sted-ec':'sted','sted_ec':'sted'}
        observed=aliases.get(raw_dataset,raw_dataset)
        if observed and observed!=dataset: raise ValueError(f'Header dataset {observed} conflicts with {dataset}')
        ap=str(a.get('gene_panel','')).lower()
        ap='hvg' if ap in {'200','hvg200','hvg_200','hvg'} else ap
        if ap and ap!=panel: raise ValueError('Header native panel conflicts with slot')
        if a.get('model_family') and str(a['model_family']).upper()!=family: raise ValueError('Header model family conflicts with slot')
        observed_model=str(a.get('model_name',''))
        if family=='TRANS' and observed_model and observed_model not in {model,model.removesuffix('_200').removesuffix('_author')}:
            raise ValueError('Transformer header architecture conflicts with slot')
        for field,keys in [('split',['split','evaluation_split','eval_split']),('normalization',['normalization']),('readout',['readout','fm_readout']),('seed',['seed','random_seed','training_seed'])]:
            for key in keys:
                if key in a:
                    if field=='normalization' and str(a[key]).lower() in {'','unknown','unspecified','training_input_scale_unspecified'}:
                        if not row.get(field):row[field]=a[key]
                        continue
                    if row.get(field) not in (None,'','unknown') and str(row[field])!=str(a[key]): raise ValueError(f'Explicit {field} conflicts with H5 metadata')
                    row[field]=a[key]; break
        # Export provenance carries the original dataset settings if the H5 does not.
        run=Path(row.get('run_root',Path(row['path']).parent.parent))
        resolved=run/'00_provenance/resolved_dataset_config.json'
        if resolved.is_file():
            original=json.loads(resolved.read_text())
            if not row.get('normalization') and original.get('normalization'):
                row['normalization']=original['normalization']
        norm=str(row.get('normalization','')).lower()
        if norm in {'','unknown','unspecified','training_input_scale_unspecified'} and a.get('schema_version')=='v4.canonical_prediction.1' and a.get('model_family') in {'FM','TRANS'} and a.get('prediction_unit')=='unique_target_cell_mean':
            # The supplied V5 adapters save P and GT in the preprocessed input
            # matrix's units. This states their origin, not counts/log1p units.
            # Complete numerical GT equality is still required by the raw hash.
            row['normalization']='v5_saved_training_input_matrix_units'
            row['scale_provenance']='supplied_V5_canonical_exporter_training_input_matrix_units; physical_transform_unspecified'
        row.setdefault('split','unknown'); row.setdefault('normalization','unknown'); row.setdefault('readout','direct_forecast' if family=='TRANS' else 'unknown')
        row.setdefault('weighting','unique_target')
        row.setdefault('scenario','measured')
        row['n_cells']=h['prediction_expression'].shape[0]; row['n_genes']=h['prediction_expression'].shape[1]
        row['header_metadata']=a
    return row

def discover(v5_roots=(), log_roots=(), legacy_inputs=(), fm_readout='blend_0.85', fm_result_tag=None, normalization=None, split=None):
    config={'schema_version':1,'raw_sources':[],'training_logs':[],'training_summaries':[],'legacy_inputs':[], 'options':{'chunk_size':16384,'pcc_eps':1e-12,'pcc_nan_policy':'zero','pcc_min_valid_fraction':.05,'max_distribution_cells':2000,'max_gene_distribution_cells':256,'max_covariance_cells':5000,'max_covariance_genes':500,'variogram_gene_pairs':20000,'pca_components':30,'distribution_seed':42,'cpu_reserve':1}}
    audit=[]; seen=set()
    for value in v5_roots:
        root=Path(value).expanduser().resolve()
        if not root.is_dir(): audit.append({'path':str(root),'status':'missing_root'}); continue
        for dataset in ('big','small','sted'):
            for family in ('FM','TRANS'):
                parent=root/dataset/family
                if not parent.is_dir(): continue
                for run in sorted(parent.iterdir()):
                    if not run.is_dir(): continue
                    try:
                        panel='author' if run.name.endswith('_author') else 'hvg'
                        identity=canonical_identity(dataset,panel,run.name,family)
                    except ValueError: continue
                    file=run/'01_predictions/measured_forward.h5'
                    if not file.is_file():
                        audit.append({'path':str(file),'dataset':dataset,'panel':panel,'model':run.name,'status':'missing_measured_h5'});continue
                    key=str(file)
                    if key in seen:continue
                    seen.add(key)
                    source=dict(zip(('dataset','panel','model','family'),identity),path=str(file),run_root=str(run))
                    if normalization:source['normalization']=normalization
                    if split:source['split']=split
                    try:
                        source=inspect_header(source)
                        config['raw_sources'].append(source)
                        audit.append({'path':key,'status':'header_ready','dataset':dataset,'panel':panel,'model':run.name})
                    except Exception as e:audit.append({'path':key,'status':'blocked_header','reason':str(e)})
    log_seen=set()
    for value in log_roots:
        root=Path(value).expanduser().resolve()
        paths=[root] if root.is_file() else sorted(root.rglob('*.log')) if root.is_dir() else []
        if not paths:audit.append({'path':str(root),'status':'no_logs_found'})
        for file in paths:
            if str(file) in log_seen:continue
            log_seen.add(str(file))
            text=file.read_text(errors='replace')
            if not re.search(r'\bRESULT\s+(FM|TRANS)\s',text):continue
            config_lines={m.group(1):m.group(2).strip() for m in re.finditer(r'\|\s*INFO\s*\|\s*([A-Z][A-Z0-9_]+):\s*(.*)',text)}
            name=str(file)
            candidates=set(re.findall(r'(?:^|[/_])(big|small|sted)(?=[/_\.])',name))
            if len(candidates)!=1:
                audit.append({'path':str(file),'status':'needs_explicit_log_binding','reason':'dataset identity unresolved'});continue
            dataset=candidates.pop()
            assay=name+' '+config_lines.get('ADATA_VAR_PATH','')+' '+config_lines.get('GENE_PANEL','')
            author=bool(re.search(r'author',assay,re.I)); hvg=bool(re.search(r'hvg|HVG_200',assay,re.I))
            if author==hvg:
                audit.append({'path':str(file),'status':'needs_explicit_log_binding','reason':'native panel unresolved/conflicting'});continue
            panel='author' if author else 'hvg'
            has_fm=bool(re.search(r'\bRESULT\s+FM\s',text))
            if has_fm and ('v7' in file.name or config_lines.get('FM_TEMPORAL_DIRECTION','forward').strip("'\"")!='forward'):
                audit.append({'path':str(file),'status':'excluded_bidirectional_or_reverse'});continue
            if has_fm and not re.search(r'v6',name,re.I):
                audit.append({'path':str(file),'status':'needs_explicit_log_binding','reason':'unidirectional v6 code identity not established'});continue
            entry={'path':str(file),'dataset':dataset,'panel':panel}
            if has_fm:
                entry.update(fm_model='v6_author' if panel=='author' else 'v6_200_new',fm_readout=fm_readout)
                if fm_result_tag:entry['fm_result_tag']=fm_result_tag
            config['training_logs'].append(entry)
            audit.append({'path':str(file),'status':'log_binding_candidate','dataset':dataset,'panel':panel,'reason':'Exact result selection and metadata will be checked by audit/run'})
    for value in legacy_inputs:
        p=Path(value).expanduser().resolve()
        if p.is_dir():
            candidates=list(p.rglob('ALL_METRICS_LONG.csv'))+list(p.rglob('ALL_MODEL_METRIC_VALUES.tsv'))
            if len(candidates)!=1:
                audit.append({'path':str(p),'status':'needs_explicit_legacy_file','reason':f'Found {len(candidates)} candidate tables'});continue
            p=candidates[0]
        if p.is_file():config['legacy_inputs'].append(str(p))
        else:audit.append({'path':str(p),'status':'missing_legacy_input'})
    return config,audit
