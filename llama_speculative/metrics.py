import math
from .common import digest


def recompute(outputs,elapsed,cap,method):
    if not math.isfinite(elapsed) or elapsed <= 0 or not outputs:
        raise ValueError('Invalid elapsed time or empty outputs')
    budget = None if method == 'ar' else int(''.join(c for c in method if c.isdigit()))
    tokens = accepted = verify = proposed = capped = 0
    histogram = [0]*(budget+1) if budget else None
    for row in outputs:
        meta = row['meta_info'];n = len(row['output_ids'])
        assert n == meta['completion_tokens'] and n <= cap
        tokens += n;capped += n >= cap
        if budget is None:
            assert all(meta.get(k) in [None,0] for k in ['spec_verify_ct','spec_num_correct_drafts','spec_num_proposed_drafts'])
            continue
        h = meta['spec_correct_drafts_histogram']
        assert sum(h) == meta['spec_verify_ct']
        assert sum(i*v for i,v in enumerate(h)) == meta['spec_num_correct_drafts']
        assert not any(h[budget+1:])
        assert meta['spec_num_proposed_drafts'] == budget*meta['spec_verify_ct']
        accepted += meta['spec_num_correct_drafts'];verify += meta['spec_verify_ct'];proposed += meta['spec_num_proposed_drafts']
        for i,v in enumerate(h[:budget+1]):
            histogram[i] += v
    if budget and verify == 0:
        raise ValueError('No speculative verification rounds observed')
    return dict(count=len(outputs),elapsed_seconds=elapsed,output_tokens=tokens,tokens_per_second=tokens/elapsed,
        capped_requests=capped,accepted_drafts=accepted if budget else None,verify_rounds=verify if budget else None,
        proposed_drafts=proposed if budget else None,mean_accepted_draft=accepted/verify if budget else None,
        acceptance=accepted/proposed if budget else None,histogram=histogram,
        output_ids_sha256=digest([r['output_ids'] for r in outputs]))
