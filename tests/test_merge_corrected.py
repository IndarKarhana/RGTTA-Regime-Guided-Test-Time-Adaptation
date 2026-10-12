import copy
import pytest
from runpod.merge_corrected import merged_results, AFFECTED, UNCHANGED, PROTOCOL


def fixture():
    metric={'mse_mean':1.0,'mae_mean':0.5,'total_time':2.0,'n_batches_evaluated':1,
            'batch_metrics':[{'forecast_origin':'a','target_start':'b','target_end':'c'}]}
    job={'model':'m','dataset':'d','forecast_horizon':96,'seed':0,**{p:copy.deepcopy(metric) for p in AFFECTED+UNCHANGED}}
    data={'metadata':{'evaluation_protocol':PROTOCOL,'complete':True,'expected_experiments':1},'experiments':[job]}
    return data,{'data_sha256':{'d':'same'}}


def test_replace_only_affected_policies():
    old,manifest=fixture();new=copy.deepcopy(old)
    for p in AFFECTED: new['experiments'][0][p]['mse_mean']=0.25
    merged=merged_results(old,new,manifest,manifest)
    for p in AFFECTED: assert merged['experiments'][0][p]['mse_mean']==0.25
    for p in UNCHANGED: assert merged['experiments'][0][p]==old['experiments'][0][p]
    assert old['experiments'][0]['rgtta']['mse_mean']==1.0


@pytest.mark.parametrize('problem',['incomplete','origin','data','keys'])
def test_reject_incompatible_results(problem):
    old,manifest=fixture();new=copy.deepcopy(old);other=copy.deepcopy(manifest)
    if problem=='incomplete': new['metadata']['complete']=False
    if problem=='origin': new['experiments'][0]['rgtta']['batch_metrics'][0]['target_start']='future'
    if problem=='data': other['data_sha256']['d']='different'
    if problem=='keys': new['experiments'][0]['seed']=1
    with pytest.raises(ValueError): merged_results(old,new,manifest,other)


def test_archive_merge_generates_tables(tmp_path, monkeypatch):
    import io
    import json
    import tarfile
    import csv
    from runpod.merge_corrected import main
    data,manifest=fixture()
    paths=[]
    for name,folder in [('legacy','full_matrix'),('corrected','corrected_rg')]:
        path=tmp_path/f'{name}.tar.gz'; paths.append(path)
        with tarfile.open(path,'w:gz') as archive:
            for filename,content in [('unified_results.json',data),('manifest.json',manifest)]:
                payload=json.dumps(content).encode()
                entry=tarfile.TarInfo(f'benchmarks/results/{folder}/{filename}');entry.size=len(payload)
                archive.addfile(entry,io.BytesIO(payload))
    output=tmp_path/'merged'
    monkeypatch.setattr('sys.argv',['merge','--legacy',str(paths[0]),'--corrected',str(paths[1]),'--output',str(output)])
    main()
    assert json.loads((output/'status.json').read_text())['complete']
    assert len(list(csv.DictReader((output/'tables.csv').open())))==6
    assert len(list(csv.DictReader((output/'paired_comparisons.csv').open())))==3
