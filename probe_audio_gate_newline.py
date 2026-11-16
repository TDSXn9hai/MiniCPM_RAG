"""GPU matrix for trailing newline and forced-listen RAG delivery.

Run explicitly; leaves official model and production runtime unmodified.
The retrieval delay is simulated by two silence units, not wall-clock sleeping.
"""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
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
    out = ROOT / 'outputs/audio_gate_newline'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    query, audio, asr = transcribe_file(
        args.audio,
        args.asr_model, 'cpu')
    retrieval = RAGRetriever(ROOT/'knowledge/sample.json').retrieve(query)
    assert retrieval.usable
    base = build_rag_prompt(query, retrieval.evidence.text)
    runtime = OfficialDuplexRuntime(model_path=ROOT/'model', ref_audio_path=ROOT/'assets/ref_audio.wav')
    runtime._load()
    duplex = runtime._duplex
    trace = PrefillTrace(duplex)
    silence = np.zeros(16000, np.float32)
    chunks = [np.pad(audio[i:i+16000], (0, max(0, i+16000-len(audio)))).astype(np.float32)
              for i in range(0, len(audio), 16000)]
    doc = {'asr': asr, 'retrieval': retrieval.to_dict(), 'base_prompt': base,
           'decode_mode': 'greedy', 'seed': 42, 'wait_units': 2,
           'logits_modified': False, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
           'cases': []}

    def save():
        (out/'results.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2))

    def distribution(logits):
        if logits is None:
            return None
        values = logits.detach().float().reshape(-1)
        probs = torch.softmax(values, dim=-1)
        top = torch.topk(probs, 5)
        eot = duplex.tokenizer.convert_tokens_to_ids('<|turn_eos|>')
        return {'eot_probability': float(probs[eot]),
                'turn_terminators': [{'id': int(i), 'token': duplex.tokenizer.decode([int(i)]),
                                     'probability': float(probs[i])} for i in duplex.turn_terminator_token_ids],
                'top5': [{'id': int(i), 'token': duplex.tokenizer.decode([int(i)]), 'probability': float(p)}
                         for p, i in zip(top.values.tolist(), top.indices.tolist())]}

    def generate():
        offset = len(duplex.total_ids)
        result = duplex.streaming_generate(decode_mode='greedy', max_new_speak_tokens_per_chunk=20)
        ids = duplex.total_ids[offset:]
        record = {'mode': duplex.current_mode, 'forced_listen': duplex.force_listen_count > 0,
                  'text': result.get('text') or '', 'is_listen': bool(result['is_listen']),
                  'end_of_turn': bool(result['end_of_turn']), 'token_ids': ids,
                  'tokens': duplex.tokenizer.decode(ids)}
        return record, result.get('audio_waveform')

    def forced_input(chunk, text=None):
        duplex.force_listen_count = 10**9
        kw = {'audio_waveform': chunk}
        if text is not None:
            kw['text_list'] = [text]
        pre = trace.prefill(**kw)
        assert pre['result']['success'], pre
        record, _ = generate()
        assert record['is_listen'] and not record['text'], record
        return record

    settings = [(path, n) for path in ['text_release', 'audio_release', 'audio_hold'] for n in (0, 1, 2)]
    settings += [('audio_no_evidence', 0)]
    try:
        for path, n in settings:
            seed_all(42)
            runtime._session_ready = False
            runtime._prepare()
            duplex.force_listen_count = 10**9
            case = {'name': f'{path}_nl{n}', 'path': path, 'trailing_newlines': n,
                    'input_units': [], 'units': []}
            doc['cases'].append(case)
            start = time.perf_counter()
            for chunk in chunks:
                case['input_units'].append(forced_input(chunk))
            for _ in range(2):
                case['input_units'].append(forced_input(silence))
            text = base + '\n'*n
            case['prompt_repr'] = repr(text)
            case['prompt_tail_token_ids'] = duplex.tokenizer.encode(text, add_special_tokens=False)[-12:]
            kwargs = {'text_list': [text]}
            if path != 'text_release':
                kwargs['audio_waveform'] = silence
            if path == 'audio_no_evidence':
                kwargs.pop('text_list')
            pre = trace.prefill(**kwargs)
            assert pre['result']['success'], pre
            case['injection'] = pre
            case['pending_distribution'] = distribution(duplex.pending_logits)
            case['final_text_feed_distribution'] = distribution(trace.last_logits)
            if path == 'audio_hold':
                record, _ = generate()
                assert record['is_listen'] and not record['text']
                case['injection_forced_listen_unit'] = record
                pre = trace.prefill(audio_waveform=silence)
                assert pre['result']['success'], pre
                case['release_distribution'] = distribution(duplex.pending_logits)
            duplex.force_listen_count = 0
            waves = []
            for i in range(24):
                if i:
                    pre = trace.prefill(audio_waveform=silence)
                    assert pre['result']['success'], pre
                record, wave = generate()
                case['units'].append(record)
                if wave is not None and not record['is_listen']:
                    if hasattr(wave, 'detach'):
                        wave = wave.detach().float().cpu().numpy()
                    waves.append(np.asarray(wave, np.float32).reshape(-1))
                print(case['name'], i, repr(record['text']), record['is_listen'], record['end_of_turn'], flush=True)
                if record['end_of_turn']:
                    break
            case['text'] = ''.join(u['text'] for u in case['units'])
            case['keyword_hits'] = [w for w in ['阳光', '坚果', '冷却', '草坪'] if w in case['text']]
            case['ended_naturally'] = case['units'][-1]['end_of_turn']
            case['wall_s'] = time.perf_counter()-start
            wave = np.concatenate(waves) if waves else np.zeros(0, np.float32)
            case['wav'] = str(out/f"{case['name']}.wav") if wave.size else None
            case['audio_samples'] = int(wave.size)
            if wave.size:
                sf.write(case['wav'], wave, 24000, subtype='FLOAT')
            save()
    finally:
        trace.close()
        runtime.close()
        save()

if __name__ == '__main__':
    main()
