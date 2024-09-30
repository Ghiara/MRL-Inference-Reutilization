def cyclic_schedule(step: int) -> float:
    step = step % 20_000
    if step < 10_000:
        return step / 10_000
    else:
        return 1.0

def cyclic_schedule_regularization(step: int) -> float:
    step = step % 20_000
    if step < 10_000:
        return 10 * step / 10_000
    else:
        return 10.0
