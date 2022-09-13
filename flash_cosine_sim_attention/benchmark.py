import torch
from torch.cuda import synchronize, Event
from functools import wraps, partial

timer = partial(Event, enable_timing = True)

def benchmark(
    fn,
    *,
    num_times = 10,
    warmup_iters = 10,
    forwards = True,
    backwards = False
):
    assert forwards or backwards

    @wraps(fn)
    def inner(*args, **kwargs):
        # warmup

        for _ in range(warmup_iters):
            loss = fn(*args, **kwargs)

            if backwards:
                loss.sum().backward()

        # average across number of function calls

        all_measured_times_ms = 0.

        for _ in range(num_times):
            start_event = timer()
            end_event = timer()

            if forwards:
                start_event.record()

            o = fn(*args, **kwargs)

            if not backwards:
                end_event.record()

            if not forwards:
                start_event.record()

            if backwards:
                loss = o.sum()
                loss.backward()
                end_event.record()

            synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            all_measured_times_ms += elapsed_time_ms

        return all_measured_times_ms / num_times

    return inner
