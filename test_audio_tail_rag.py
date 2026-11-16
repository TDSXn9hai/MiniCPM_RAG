"""Controlled GPU experiment: separate text versus final audio+text RAG unit.

Run directly, not collected as pytest tests. No production/model source edits.
The refreshed case explicitly intervenes in pending_logits for diagnosis only.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import librosa
import numpy as np
import soundfile as sf
import torch
from official_runtime import OfficialDuplexRuntime, build_rag_prompt
from rag_pipeline import RAGRetriever
from run_rag import transcribe_file
from run_probe import PrefillTrace, seed_all

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--asr-model', type=Path, default=ROOT/'models/whisper-small')
    args = parser.parse_args()
    out = ROOT / 'outputs/audio_tail_rag'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    audio_path = args.audio
    query, audio, asr = transcribe_file(audio_path, args.asr_model, 'cpu')
    retrieval = RAGRetriever(ROOT / 'knowledge/sample.json').retrieve(query)
    assert retrieval.usable
    prompt = build_rag_prompt(query, retrieval.evidence.text)
    runtime = OfficialDuplexRuntime(model_path=ROOT/'model', ref_audio_path=ROOT/'assets/ref_audio.wav')
    runtime._load()
    duplex = runtime._duplex
    trace = PrefillTrace(duplex)
    document = {'asr': asr, 'retrieval': retrieval.to_dict(), 'prompt': prompt,
                'audio_samples': len(audio), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                'system_prompt': runtime.system_prompt, 'cases': []}

    def save():
        (out/'results.json').write_text(json.dumps(document, ensure_ascii=False, indent=2))

    def listen(chunk):
        previous = duplex.force_listen_count
        duplex.force_listen_count = 10**9
        try:
            pre = trace.prefill(audio_waveform=chunk)
            assert pre['result']['success'], pre
            result = duplex.streaming_generate(decode_mode='greedy', max_new_speak_tokens_per_chunk=20)
            assert result['is_listen']
            return {'mode': duplex.current_mode, 'is_listen': bool(result['is_listen']),
                    'end_of_turn': bool(result['end_of_turn']), 'text': result.get('text', '')}
        finally:
            duplex.force_listen_count = previous

    chunks = [np.pad(audio[i:i+16000], (0, max(0, i+16000-len(audio)))).astype(np.float32)
              for i in range(0, len(audio), 16000)]
    try:
        for name in ['separate_text', 'tail_audio_only', 'tail_mixed', 'tail_mixed_refreshed',
                     'separate_text_after_tail_listen', 'text_without_audio']:
            seed_all(42)
            runtime._session_ready = False
            runtime._prepare()
            case = {'name': name, 'diagnostic_logits_intervention': name.endswith('refreshed'),
                    'listening_units': [], 'units': []}
            document['cases'].append(case)
            if name != 'text_without_audio':
                before = chunks if name in ['separate_text', 'separate_text_after_tail_listen'] else chunks[:-1]
                for chunk in before:
                    case['listening_units'].append(listen(chunk))
            if name in ['separate_text', 'separate_text_after_tail_listen', 'text_without_audio']:
                pre = trace.prefill(text_list=[prompt])
            elif name == 'tail_audio_only':
                pre = trace.prefill(audio_waveform=chunks[-1])
            else:
                pre = trace.prefill(audio_waveform=chunks[-1], text_list=[prompt],
                                    refresh=name.endswith('refreshed'))
            assert pre['result']['success'], pre
            case['injection'] = pre
            waves = []
            start = time.perf_counter()
            for i in range(24):
                if i:
                    p = trace.prefill(audio_waveform=np.zeros(16000, np.float32))
                    assert p['result']['success'], p
                offset = len(duplex.total_ids)
                result = duplex.streaming_generate(decode_mode='greedy', max_new_speak_tokens_per_chunk=20)
                ids = duplex.total_ids[offset:]
                wave = result.get('audio_waveform')
                if hasattr(wave, 'detach'):
                    wave = wave.detach().float().cpu().numpy()
                if wave is not None and not result['is_listen']:
                    waves.append(np.asarray(wave, np.float32).reshape(-1))
                case['units'].append({'unit': i, 'mode': duplex.current_mode,
                    'text': result.get('text', ''), 'is_listen': bool(result['is_listen']),
                    'end_of_turn': bool(result['end_of_turn']),
                    'generated_token_ids': ids, 'generated_tokens': duplex.tokenizer.decode(ids)})
                print(name, i, repr(result.get('text')), result['is_listen'], result['end_of_turn'], flush=True)
                if result['end_of_turn']:
                    break
            case['text'] = ''.join(u['text'] for u in case['units'])
            case['ended_naturally'] = case['units'][-1]['end_of_turn']
            case['wall_s'] = time.perf_counter()-start
            waveform = np.concatenate(waves) if waves else np.zeros(0, np.float32)
            case['wav'] = str(out/f'{name}.wav') if waveform.size else None
            case['audio_samples'] = int(waveform.size)
            if waveform.size:
                sf.write(case['wav'], waveform, 24000, subtype='FLOAT')
            save()
    finally:
        trace.close()
        runtime.close()
        save()

if __name__ == '__main__':
    main()
