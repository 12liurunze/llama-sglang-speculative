"""Pilot-only observations. No changes to logits, tree decisions or caches."""
import functools
import json
import os
from pathlib import Path


def install():
    out = Path(os.environ['LLAMA_TP_AUDIT_OUT']);out.mkdir(parents=True,exist_ok=True)
    def emit(value):
        with (out/f'{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(dict(pid=os.getpid(),**value))+'\n')
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import EAGLEDraftCudaGraphRunner
    for cls,event in [(DecodeCudaGraphRunner,'decode_graph'),(PrefillCudaGraphRunner,'prefill_graph'),(EAGLEDraftCudaGraphRunner,'draft_graph')]:
        original = cls.execute
        def make(fn,event):
            @functools.wraps(fn)
            def execute(self,batch,*a,**kw):
                result = fn(self,batch,*a,**kw)
                emit(dict(event=event,bs=int(batch.batch_size),draft=bool(self.model_runner.is_draft_worker)))
                return result
            return execute
        cls.execute = make(original,event)
    from sglang.srt.layers.logits_processor import LogitsProcessor
    original = LogitsProcessor.forward
    @functools.wraps(original)
    def forward(self,*a,**kw):
        result = original(self,*a,**kw)
        logits = getattr(result,'next_token_logits',None)
        if logits is not None:
            width = int(logits.shape[-1]);expected = int(self.vocab_size)
            assert width == expected,('Logits are not full vocabulary',width,expected)
            emit(dict(event='full_vocab_logits',width=width,vocab_size=expected,
                      tp_gather=bool(self.do_tensor_parallel_all_gather)))
        return result
    LogitsProcessor.forward = forward
    emit(dict(event='installed'))

