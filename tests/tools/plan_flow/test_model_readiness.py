import hashlib
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.model_backends.huggingface import HuggingFaceModelBackend


def fixture(tmp_path):
    binding = {'lock_id':'fixture', 'model_id':'fixture/model', 'revision':'a'*40,
               'files':[{'path':name, 'size':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
                        for name,data in [('weights.bin',b'weight'),('labels.json',b'{}')]]}
    root=tmp_path/'models--fixture--model'/'snapshots'/binding['revision']
    root.mkdir(parents=True); (root/'weights.bin').write_bytes(b'weight')
    return binding,root,HuggingFaceModelBackend(cache_root=tmp_path)


def test_partial_cache_repairs_only_missing_file_with_plain_file(tmp_path,monkeypatch):
    binding,root,backend=fixture(tmp_path); calls=[]
    def download(**kwargs):
        calls.append(kwargs['filename']); path=Path(kwargs['local_dir'])/kwargs['filename']
        path.write_bytes(b'{}');return str(path)
    monkeypatch.setattr('huggingface_hub.hf_hub_download',download)
    assert backend.prepare(binding,offline=False)==root
    assert calls==['labels.json']
    assert not (root/'labels.json').is_symlink()
    assert backend.prepare(binding,offline=True)==root
    assert calls==['labels.json']


def test_missing_file_can_be_restored_from_verified_blob_without_network(tmp_path,monkeypatch):
    binding,root,backend=fixture(tmp_path)
    blobs=root.parent.parent/'blobs';blobs.mkdir();(blobs/'opaque-git-blob-id').write_bytes(b'{}')
    monkeypatch.setattr('huggingface_hub.hf_hub_download',lambda **kw:pytest.fail('network download'))
    assert backend.prepare(binding,offline=False)==root
    assert (root/'labels.json').read_bytes()==b'{}'


def test_readiness_is_read_only_and_reports_exact_missing_bytes(tmp_path):
    binding,root,backend=fixture(tmp_path)
    result=backend.inspect(binding,offline=True)
    assert result['download_bytes']==2
    assert result['missing_files']==['labels.json']
    assert result['can_prepare'] is False
    assert result['verified'] is False
    assert not (root/'labels.json').exists()
    with pytest.raises(PlanFlowError) as error:backend.prepare(binding,offline=True)
    assert error.value.details['model_id']=='fixture/model'


def test_bad_download_is_never_published(tmp_path,monkeypatch):
    binding,root,backend=fixture(tmp_path)
    def download(**kwargs):
        path=Path(kwargs['local_dir'])/kwargs['filename'];path.write_bytes(b'XX');return str(path)
    monkeypatch.setattr('huggingface_hub.hf_hub_download',download)
    with pytest.raises(PlanFlowError):backend.prepare(binding,offline=False)
    assert not (root/'labels.json').exists()
    assert (root/'weights.bin').read_bytes()==b'weight'


def test_hash_change_invalidates_readiness_cache(tmp_path):
    binding,root,backend=fixture(tmp_path);(root/'labels.json').write_bytes(b'{}')
    assert backend.inspect(binding)['verified'] is True
    (root/'labels.json').write_bytes(b'XX')
    assert backend.inspect(binding)['invalid_files']==['labels.json']
