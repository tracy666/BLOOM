"""CSV-only pipeline; no model execution or plots."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import time
from .registry import TRANS, METRICS, get_metric, display_model, canonical_identity
from .config import inspect_header

COLUMNS=['dataset','panel','model','model_label','family','metric','value','status','scope','day','gene','day_kind','cohort_id','route','definition_id','aggregation','weighting','readout','seed','split','normalization','comparison_status','n_cells','n_genes','n_days','target_days','valid_fraction','source_path','source_line','source_sha256','result_tag','source_key','pcc_nan_policy','pcc_eps','min_valid_fraction','variance_ddof','inference_unit','independence_attested','reason']

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write_csv(path,rows,preferred=()):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys([*preferred,*[k for row in rows for k in row]]))
    if not fields:fields=['status','reason']
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            row=clean(row)
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def resources(jobs,workers='auto',cpu_reserve=1,estimated_gb=.8):
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    quota=affinity
    try:
        limit,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if limit!='max':quota=min(quota,max(1,math.ceil(int(limit)/int(period))))
    except (OSError,ValueError):pass
    available=None
    try:
        mem=dict(re.findall(r'^(\w+):\s+(\d+)',Path('/proc/meminfo').read_text(),re.M));available=int(mem['MemAvailable'])/1024**2
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':available=min(available,max(.1,(int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))/1024**3))
    except (OSError,KeyError,ValueError):pass
    cpu=max(1,quota-int(cpu_reserve));memory=max(1,int(.7*available/estimated_gb)) if available is not None else cpu
    n=min(max(1,jobs),cpu,memory) if str(workers)=='auto' else min(max(1,jobs),max(1,int(workers)))
    return n,dict(cpu_affinity=affinity,cpu_quota=quota,cpu_reserve=cpu_reserve,available_ram_gb=available,estimated_gb_per_worker=estimated_gb,workers=n)

def _population_source(source,options,raw_result):
    import h5py
    import numpy as np
    from .population import compute_population
    path=source['path'];rawrows=raw_result['rows'];template=next(r for r in rawrows if r['scope']=='global')
    output=[];audits=[];day_values={}
    seed=int(options.get('distribution_seed',42));cap=max(int(options.get('max_distribution_cells',2000)),int(options.get('max_gene_distribution_cells',256)),int(options.get('max_covariance_cells',5000)),int(options.get('deg_max_cells',10000)) if source.get('reference_verified') else 0)
    if cap<2:raise ValueError('Population sample limit must be at least two')
    params={k:options[k] for k in ('max_distribution_cells','max_gene_distribution_cells','max_covariance_cells','max_covariance_genes','variogram_gene_pairs','pca_components','projections','deg_fdr','deg_min_effect','deg_max_cells') if k in options}
    params.setdefault('max_gene_distribution_cells',256)
    params.update(seed=seed,paired_rows=True)
    definition='population_audited_v1:'+json.dumps(params,sort_keys=True,separators=(',',':'))
    with h5py.File(path,'r') as h:
        times=np.asarray(h['prediction_time']);ids=np.asarray(h['prediction_cell_index']).astype(str)
        genes=[v.decode() if isinstance(v,bytes) else str(v) for v in np.asarray(h['gene_names'])]
        gene_order=np.argsort(genes);genes=[genes[i] for i in gene_order]
        params['gene_names']=genes
        for day in np.unique(times):
            positions=np.flatnonzero(times==day)
            positions=positions[np.argsort(ids[positions],kind='stable')]
            if len(positions)>cap:
                # Identical native target identities give identical row samples across models.
                positions=positions[np.sort(np.random.default_rng(seed).choice(len(positions),cap,replace=False))]
            positions=np.sort(positions)
            x=np.asarray(h['prediction_expression'][positions],float)[:,gene_order]
            y=np.asarray(h['truth_expression'][positions],float)[:,gene_order]
            canonical=np.argsort(ids[positions],kind='stable')
            x,y=x[canonical],y[canonical]
            reference=None;local=dict(params);refaudit={'status':'no_verified_prior_reference'}
            binding=source.get('references',{}).get(str(int(day)) if float(day).is_integer() else str(day),source if source.get('reference_path') else {})
            if binding.get('reference_path') and binding.get('reference_verified') is True and binding.get('reference_kind')=='observed_prior_stage':
                with h5py.File(binding['reference_path'],'r') as ref:
                    rnorm=binding.get('normalization',ref.attrs.get('normalization',''))
                    if isinstance(rnorm,bytes):rnorm=rnorm.decode()
                    if str(rnorm)!=str(template['normalization']):raise ValueError('Observed reference normalization must be explicitly identical to measured expression units')
                    if binding.get('time_kind')!='ordinal_stage':raise ValueError('Observed reference time_kind=ordinal_stage is required to match prediction_time')
                    key=binding.get('reference_key','reference_expression');tkey=binding.get('reference_time_key','reference_time')
                    rg=[v.decode() if isinstance(v,bytes) else str(v) for v in np.asarray(ref['gene_names'])]
                    if len(set(rg))!=len(rg) or set(rg)!=set(genes):raise ValueError('Observed reference has a different native gene set')
                    rt=np.asarray(ref[tkey],float)
                    if len(rt)!=len(ref[key]) or not np.isfinite(rt).all():raise ValueError('Observed reference times invalid')
                    prior=np.unique(rt[rt<float(day)])
                    if len(prior):
                        age=prior[-1]; ri=np.flatnonzero(rt==age)
                        if len(ri)>cap:ri=np.sort(np.random.default_rng(seed).choice(ri,cap,replace=False))
                        reference=np.asarray(ref[key][ri],float)[:,[rg.index(g) for g in genes]]
                        local.update(reference_verified=True,reference_kind='observed_prior_stage')
                        refaudit={'status':'verified_observed_prior_stage','path':binding['reference_path'],'time':float(age),'n_reference_cells':len(ri),'reference_key':key}
            result=compute_population(x,y,reference,local)
            audits.append({'day':float(day),'n_saved_day_cells':int((times==day).sum()),'n_initial_sample_cells':len(x),'reference':refaudit,'population_audit':result['audit']})
            values={k:v for k,v in result['values'].items() if not k.startswith('pseudobulk_')}
            day_values[float(day)]=values
            for metric,value in values.items():
                finite=value is not None and np.isfinite(value)
                output.append({**template,'route':'population_sampled','metric':metric,'value':float(value) if finite else float('nan'),'scope':'day','day':float(day),'gene':'','definition_id':definition+':'+metric,'aggregation':'within_day_population','n_cells':int((times==day).sum()),'sample_cells':len(x),'n_days':1,'status':'available' if finite else 'missing_reference_or_undefined','reason':result['audit'].get('invalid_reasons',{}).get(metric,''),'valid_fraction':'','variance_ddof':1})
            for data in result['genes']:
                for key,value in data.items():
                    if key in {'gene','gene_index'} or isinstance(value,(str,list,dict)):continue
                    if isinstance(value,(float,int,np.generic)):
                        output.append({**template,'route':'population_sampled','metric':key,'value':value,'scope':'day_gene','day':float(day),'gene':data.get('gene',genes[int(data.get('gene_index',0))]),'definition_id':definition+':component:'+key,'aggregation':'sampled_gene_diagnostic','n_cells':len(x),'n_days':1,'status':'available' if np.isfinite(value) else 'undefined','reason':'Sampled population/DE component; not an independent replicate','valid_fraction':''})
    metrics=sorted({k for values in day_values.values() for k in values})
    for metric in metrics:
        vals=[values.get(metric,float('nan')) for values in day_values.values()]
        complete=all(v is not None and np.isfinite(v) for v in vals)
        output.append({**template,'route':'population_sampled','metric':metric,'value':float(np.mean(vals)) if complete else float('nan'),'scope':'global','day':'','gene':'','definition_id':definition+':'+metric,'aggregation':'equal_day_mean','status':'available' if complete else 'missing_reference_or_undefined','reason':'' if complete else 'At least one target day lacks a defined metric; no favorable-day subset selected','valid_fraction':sum(v is not None and np.isfinite(v) for v in vals)/len(vals)})
    # Daily per-gene distances/components get explicit equal-day companions.
    # P/q values and truth DEG calls are never averaged across days.
    gene_metrics={'gene_wasserstein_1d','gene_mmd_u_1d','gene_covariance_row_squared_error','gene_covariance_relative_frobenius_component','gene_covariance_profile_pcc','gene_variogram_incident_error'}
    gene_groups={}
    for row in output:
        if row['scope']=='day_gene' and row['metric'] in gene_metrics:
            gene_groups.setdefault((row['gene'],row['metric']),[]).append(row)
    for (gene,metric),group in gene_groups.items():
        complete=len(group)==len(day_values) and all(np.isfinite(row['value']) for row in group)
        output.append({**template,'route':'population_sampled','metric':metric,'value':float(np.mean([row['value'] for row in group])) if complete else float('nan'),'scope':'gene','day':'','gene':gene,'definition_id':definition+':component:'+metric,'aggregation':'equal_day_gene_diagnostic_mean','status':'available' if complete else 'undefined_or_incomplete_day_coverage','reason':'Distinct univariate metric or incident-gene component; not the joint metric and not an independent replicate','valid_fraction':sum(np.isfinite(row['value']) for row in group)/len(day_values)})
    # A prior-reference binding is part of DE's comparison contract. Different
    # observed-reference artifacts cannot share a DE benchmark stratum.
    ref_contract=[{'day':a['day'],'reference':a['reference']} for a in audits]
    pop_cohort=hashlib.sha256(json.dumps({'raw_cohort':template['cohort_id'],'references':ref_contract},sort_keys=True).encode()).hexdigest()[:24]
    for row in output:
        row['cohort_id']=pop_cohort
        row['variance_ddof']=1
    return output,audits

def _task(kind,entry,options):
    start=time.time()
    try:
        if kind=='raw':
            from .raw import measure_source
            source=inspect_header(entry)
            result=measure_source(source,options)
            if options.get('mode','full')=='full':
                try:
                    pop,audit=_population_source(source,options,result)
                    result['rows'].extend(pop);result['audit']['population']=audit
                except Exception as exc:
                    result['audit']['population_error']=f'{type(exc).__name__}: {exc}'
                    result['audit']['population_status']='blocked_population_core_metrics_retained'
        elif kind=='log':
            from .logs import parse_log
            result=parse_log(entry)
        elif kind=='summary':
            from .summaries import parse_summary
            result=parse_summary(entry)
        else:
            from .legacy import import_legacy
            result=import_legacy(entry['path'])
        for row in result['rows']:
            row.setdefault('model_label',display_model(row['model']))
            row.setdefault('day_kind','ordinal_stage')
            if 'min_valid_fraction' in row:row.setdefault('pcc_min_valid_fraction',row['min_valid_fraction'])
            for field in ['inference_unit','independence_attested','run_id','reported_group']:
                if field in entry:row[field]=entry[field]
        result.update(status='partial' if result['audit'].get('population_error') else 'complete',kind=kind,elapsed_seconds=time.time()-start)
        return result
    except Exception as e:
        return {'rows':[],'records':[],'audit':{'path':entry.get('path'),'kind':kind,'status':'blocked','reason':f'{type(e).__name__}: {e}'},'status':'blocked','kind':kind,'elapsed_seconds':time.time()-start}

def jobs_from(config,dataset='all',panel='all'):
    jobs=[]
    for kind,section in [('raw','raw_sources'),('log','training_logs'),('summary','training_summaries')]:
        for row in config.get(section,[]):
            row=dict(row)
            # Explicit excluded sources cannot be turned into v6 via aliases.
            if kind=='raw':canonical_identity(row['dataset'],row['panel'],row['model'],row.get('family'))
            if dataset!='all' and row.get('dataset')!=dataset:continue
            p='hvg' if row.get('panel')=='200' else row.get('panel')
            if panel!='all' and p!=panel:continue
            jobs.append((kind,row))
    jobs.extend(('legacy',{'path':p}) for p in config.get('legacy_inputs',[]))
    return jobs

def audit_config(config,out):
    from .logs import parse_log
    from .legacy import import_legacy
    records=[]
    for kind,entry in jobs_from(config):
        try:
            if kind=='raw':
                header=inspect_header(entry)
                records.append({'kind':kind,'path':entry['path'],'status':'header_ready','dataset':header['dataset'],'panel':header['panel'],'model':header['model'],'n_cells':header['n_cells'],'n_genes':header['n_genes'],'split':header['split'],'normalization':header['normalization'],'note':'Full pairing, numeric GT and drawing-free calculations checked by run'})
            else:
                result=_task(kind,entry,{'mode':'core'})
                records.append({'kind':kind,'path':entry['path'],'status':result['status'],'metric_rows':len(result['rows']),'details':result['audit']})
        except Exception as e:records.append({'kind':kind,'path':entry.get('path'),'status':'blocked','reason':str(e)})
    write_json(Path(out)/'INPUT_AUDIT.json',records);write_csv(Path(out)/'INPUT_AUDIT.csv',records)
    print(f'Audited {len(records)} inputs; blocked={sum(r["status"]=="blocked" for r in records)}; {out}')
    return records

def _write_outputs(rows,records,audits,out,resources_info,config,seconds):
    from .comparison import build_comparisons,reported_training_ordering
    write_csv(out/'ALL_METRICS_LONG.csv',rows,COLUMNS)
    scopes={'global':'GLOBAL_METRICS.csv','day':'PER_DAY_METRICS.csv','gene':'PER_GENE_METRICS.csv','day_gene':'PER_DAY_GENE_METRICS.csv'}
    for scope,name in scopes.items():write_csv(out/name,[r for r in rows if r.get('scope')==scope],COLUMNS)
    write_csv(out/'TRAINING_NUMERIC_RECORDS.csv',records)
    comparisons=build_comparisons(rows)
    names={'pairwise':'GLOBAL_TRANS_VS_BLOOM.csv','summaries':'GLOBAL_METRIC_SUMMARY.csv','recommendations':'METRIC_SELECTION.csv','significance':'SIGNIFICANCE.csv','issues':'COMPARISON_ISSUES.csv'}
    for key,name in names.items():write_csv(out/name,comparisons.get(key,[]))
    write_csv(out/'TRAINING_REPORTED_ORDERING.csv',reported_training_ordering(rows))
    inventory=[]
    for dataset in ['big','small','sted']:
        for panel in ['hvg','author']:
            fm='v6_200_new' if panel=='hvg' else 'v6_author';suffix='_200' if panel=='hvg' else '_author'
            for model in [fm,*[m+suffix for m in TRANS]]:
                for metric in ['train_log_mean_day_pcc_cell','train_log_mean_day_pcc_gene']:
                    match=[r for r in rows if r['dataset']==dataset and r['panel']==panel and r['model']==model and r['metric']==metric and r['scope']=='global' and r.get('status')=='available']
                    inventory.append(dict(dataset=dataset,panel=panel,model=model,model_label=display_model(model),metric=metric,status='available' if match else 'missing_training_result',n_observations=len(match),reason='' if match else 'No selected matching RESULT/summary supplied; saved evaluation scores are not substituted'))
    write_csv(out/'TRAINING_METRIC_COVERAGE.csv',inventory)
    support=[]
    for metric in sorted(set(METRICS)|{r['metric'] for r in rows}):
        meta=get_metric(metric);actual=[r for r in rows if r['metric']==metric and r.get('status')=='available']
        support.append({**meta,'global_values':sum(r['scope']=='global' for r in actual),'per_day_values':sum(r['scope']=='day' for r in actual),'per_gene_values':sum(r['scope']=='gene' for r in actual),'per_day_gene_values':sum(r['scope']=='day_gene' for r in actual)})
    write_csv(out/'METRIC_SUPPORT.csv',support)
    aux=[]
    for key in sorted(set(str(r.get('source_key',r.get('metric',''))) for r in records)):
        local=[r for r in records if str(r.get('source_key',r.get('metric','')))==key]
        aux.append({'source_field':key,'numeric_records':len(local),'per_day_source_field':bool(re.match(r'^day\d+_',key)),'benchmark_role':'diagnostic; consult source definition and split','note':'Validation, runtime, composition and gate fields are not automatically ranked against TRANS'})
    write_csv(out/'TRAINING_AUXILIARY_SUPPORT.csv',aux)
    grouped={}
    for row in rows:
        key=(row['dataset'],row['panel'],row['route'],row['cohort_id'])
        grouped.setdefault(key,[]).append(row)
    for (dataset,panel,route,cohort),local in grouped.items():
        slug=hashlib.sha256(str(cohort).encode()).hexdigest()[:12]
        folder=out/'datasets'/dataset/panel/route/slug
        write_json(folder/'COHORT.json',{'dataset':dataset,'panel':panel,'route':route,'cohort_id':cohort,'comparison_status':sorted(set(r.get('comparison_status','') for r in local)),'readouts':sorted(set(str(r.get('readout','')) for r in local))})
        for scope,name in scopes.items():write_csv(folder/name,[r for r in local if r['scope']==scope],COLUMNS)
        glob=[r for r in local if r['scope']=='global' and r.get('status')=='available']
        wide={}
        for r in glob:
            key=(r['model'],str(r.get('seed','')),r.get('readout',''),r.get('source_path',''),r.get('run_id',''))
            wide.setdefault(key,{'model':r['model'],'model_label':r['model_label'],'seed':r.get('seed',''),'readout':r.get('readout',''),'source_path':r.get('source_path',''),'run_id':r.get('run_id','')})[r['metric']]=r['value']
        write_csv(folder/'GLOBAL_WIDE.csv',list(wide.values()))
        write_csv(folder/'GLOBAL_DEFINITIONS.csv',[{k:r.get(k,'') for k in ['metric','definition_id','aggregation','weighting','normalization','split']} for r in glob])
        for metric in sorted(set(r['metric'] for r in glob)):
            values=[r for r in glob if r['metric']==metric]
            meta=get_metric(metric)
            values=sorted(values,key=lambda r:(r.get('definition_id',''),r.get('aggregation',''),-(float(r['value'])) if meta['direction']=='higher' else float(r['value'])))
            if meta['direction'] in {'higher','lower'}:
                strata={}
                for row in values:strata.setdefault((row['definition_id'],row['aggregation'],str(row.get('seed',''))),[]).append(row)
                for group in strata.values():
                    rankvalues=sorted(set(float(r['value']) for r in group),reverse=meta['direction']=='higher')
                    for row in group:
                        row['numerical_rank_within_seed']=rankvalues.index(float(row['value']))+1
                        row['ranking_evidence']=row.get('comparison_status','unverified')
            write_csv(folder/'metrics'/re.sub(r'[^A-Za-z0-9_.-]','_',metric)/'values.csv',values,COLUMNS)
    write_json(out/'SOURCE_AUDIT.json',audits)
    write_json(out/'RUN_CONFIG.json',config)
    status={'status':'complete_with_blocked_inputs' if any(a.get('status') in {'blocked','partial'} for a in audits) else 'complete','inputs':len(audits),'blocked_inputs':sum(a.get('status') in {'blocked','partial'} for a in audits),'population_blocked_inputs':sum(bool(a.get('population_error')) for a in audits),'metric_rows':len(rows),'global_rows':sum(r['scope']=='global' for r in rows),'source_processing_seconds':seconds,'resources':resources_info,'statistical_claim':'Only explicitly attested independent matched replicates with verified cohorts can receive p/q values. Legacy rankings remain descriptive.'}
    write_json(out/'RUN_STATUS.json',status)
    (out/'README.md').write_text('# Metric CSV results\n\nDatasets and HVG/author are separate in `datasets/`. Source routes are never substituted.\n\n- `GLOBAL_METRIC_SUMMARY.csv`: BLOOM rank, all stronger/weaker TRANS and strongest-TRANS margin.\n- `GLOBAL_TRANS_VS_BLOOM.csv`: one pairwise comparison per global metric/model.\n- `METRIC_SELECTION.csv`: complete global descriptive/verified candidates across evaluation aspects; unfavorable values remain.\n- `METRIC_SUPPORT.csv`: theoretical and actual day/gene support.\n- `TRAINING_NUMERIC_RECORDS.csv`: all printed RESULT values, including unselected readouts and validation diagnostics.\n- `TRAINING_METRIC_COVERAGE.csv`: requested fields available/missing for all 138 model settings.\n- `SOURCE_AUDIT.json`, `COMPARISON_ISSUES.csv`: missing data and scientific-contract issues.\n\nThe uploaded log has only FM; aggregate logs cannot supply per-gene values or prove matched FM/TRANS evaluation. Do not call a descriptive win statistically significant. Source-dependent definitions, including variance ddof and equal-day versus pooled error, stay in separate strata.\n')
    return status

def run(config,out_root,tag=None,workers='auto',cpu_reserve=1,mode='full',dataset='all',panel='all'):
    jobs=jobs_from(config,dataset,panel)
    if not jobs:raise ValueError('No inputs selected; use init or provide explicit config bindings')
    root=Path(out_root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    label=tag or datetime.now(timezone.utc).strftime('benchmark_%Y%m%dT%H%M%S_%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',label):raise ValueError('Output tag must use letters/numbers/underscore/dash/dot')
    out=root/label
    if out.exists():raise ValueError(f'Output folder already exists: {out}; choose a new --tag or omit --tag')
    out.mkdir(exist_ok=False)  # Fresh result tree every run; explicit reuse never overwrites.
    options={**config.get('options',{}),'mode':mode}
    n,res=resources(len(jobs),workers,cpu_reserve)
    print(f'CSV export: inputs={len(jobs)}, processes={n}, mode={mode}; native datasets/panels; output={out}',flush=True)
    rows=[];records=[];audits=[];start=time.time()
    from tqdm import tqdm
    def collect(result,entry):
        audit={**result['audit'],'kind':result['kind'],'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}
        audits.append(audit);rows.extend(r for r in result['rows'] if (dataset=='all' or r['dataset']==dataset) and (panel=='all' or r['panel']==panel));records.extend(result.get('records',[]))
        # Incremental audit remains available during long raw jobs.
        write_json(out/'SOURCE_AUDIT.json',audits)
        return f"{result['kind']} {Path(entry['path']).name}: {result['status']}"
    if n==1:
        with tqdm(total=len(jobs),unit='source',dynamic_ncols=True) as bar:
            for kind,entry in jobs:
                result=_task(kind,entry,options);bar.set_postfix_str(collect(result,entry));bar.update()
    else:
        with ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('spawn')) as pool,tqdm(total=len(jobs),unit='source',dynamic_ncols=True) as bar:
            futures={pool.submit(_task,kind,entry,options):entry for kind,entry in jobs}
            for future in as_completed(futures):
                entry=futures[future]
                try:result=future.result()
                except Exception as e:result={'rows':[],'records':[],'kind':'worker','status':'blocked','elapsed_seconds':0,'audit':{'path':entry['path'],'reason':str(e)}}
                bar.set_postfix_str(collect(result,entry));bar.update()
    # Stable export ordering makes parallel and serial results byte-comparable.
    def row_order(row):
        day=row.get('day','')
        try:age=(0,float(day))
        except (ValueError,TypeError):age=(1,str(day))
        return (*[str(row.get(k,'')) for k in ['dataset','panel','route','cohort_id','model','seed','scope']],age,*[str(row.get(k,'')) for k in ['gene','metric','definition_id']])
    rows.sort(key=row_order)
    records.sort(key=lambda r:(str(r.get('source_path','')),int(r.get('source_line',0) or 0),str(r.get('source_key',''))))
    status=_write_outputs(rows,records,audits,out,res,config,time.time()-start)
    print(f"Finished {len(jobs)} inputs; rows={len(rows)}; blocked={status['blocked_inputs']}; {out}",flush=True)
    return out,status
