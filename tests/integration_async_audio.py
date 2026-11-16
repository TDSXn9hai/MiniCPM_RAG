"""Real same-GPU ASR/MiniCPM overlap and delayed-retrieval listen-gate check.

Run explicitly. The retrieval hold is an intentional scheduling fixture;
its duration is not a retrieval performance measurement.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_audio_rag import ASRRetrievalWorker, AsyncAudioRAGPipeline, LocalWhisperASR
from official_runtime import OfficialDuplexRuntime
from rag_pipeline import RAGRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/async_audio_gate_test')
    parser.add_argument('--asr-model', type=Path, default=ROOT/'models/whisper-small')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import librosa
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    release = threading.Event()
    real_retriever = RAGRetriever(ROOT/'knowledge/sample.json')
    class GatedRetriever:
        def retrieve(self, query):
            if not release.wait(60):
                raise TimeoutError('test retrieval gate was not released')
            return real_retriever.retrieve(query)
    runtime = OfficialDuplexRuntime(model_path=ROOT/'model',
        ref_audio_path=ROOT/'assets/ref_audio.wav', device=args.device)
    worker = ASRRetrievalWorker(LocalWhisperASR(args.asr_model, args.device), GatedRetriever())
    pipeline = AsyncAudioRAGPipeline(worker, runtime)
    events, waves = [], []
    owner = threading.get_ident()
    document = {'status': 'running', 'simulated_retrieval_hold_until_wait_units': 2,
                'device': args.device, 'cases': []}
    def event(kind, **data):
        events.append({'type': kind, **data})
        if kind == 'audio.listen' and data['phase'] == 'waiting':
            waits = [e for e in events if e['type'] == kind and e['phase'] == 'waiting']
            if len(waits) >= 2:
                release.set()
        print(kind, json.dumps(data, ensure_ascii=False), flush=True)
    def chunk(value):
        assert any(e['type'] == 'rag.injected' for e in events)
        if value['audio'] is not None:
            waves.append(value['audio'].copy())
        print('answer', repr(value['text']), value['end_of_turn'], flush=True)
    try:
        result = pipeline.answer(audio, source=str(args.audio.resolve()), on_event=event, on_chunk=chunk)
        result['name'] = 'delayed_retrieval_audio_gate'
        result['wav'] = str(out/'answer.wav') if waves else None
        if waves:
            sf.write(result['wav'], np.concatenate(waves), 24000, subtype='FLOAT')
        document['cases'].append(result)
        assert result['status'] == 'answered', result['text']
        assert result['asr']['device'] == args.device
        assert result['background']['worker_thread_id'] != owner
        assert result['citations'][0]['document_id'] == 'plant-zombies'
        listens = [e for e in events if e['type'] == 'audio.listen']
        assert all(e['mode'] == 'AUDIO' and e['forced_listen'] for e in listens)
        assert sum(e['phase'] == 'waiting' for e in listens) >= 2
        injection = next(e for e in events if e['type'] == 'rag.injected')
        assert injection['mode'] == 'AUDIO' and injection['forced_listen']
        assert injection['started_s'] >= result['background']['retrieval_finished_s']
        a, b = result['background']['asr_started_s'], result['background']['asr_finished_s']
        overlap = sum(max(0, min(b, e['finished_s'])-max(a, e['started_s'])) for e in listens)
        document['asr_minicpm_host_interval_overlap_s'] = overlap
        assert overlap > 0, 'no host-side inference intervals overlap'
        assert all(u['mode'] == 'AUDIO' for u in result['units'])
        document['keyword_hits'] = [w for w in ['阳光', '坚果', '冷却', '草坪'] if w in result['text']]
        document['status'] = 'passed'
        print('passed:', out, flush=True)
    except BaseException as exc:
        document.update(status='failed', error=str(exc))
        raise
    finally:
        release.set()
        try:
            worker.close()
        finally:
            runtime.close()
            (out/'results.json').write_text(json.dumps(document, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
