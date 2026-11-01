"""FCFS queue with exact time integration and an arrival-cohort estimator."""
from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Config:
    arrival_rate: float = 0.8
    service_rate: float = 1.0
    warmup: float = 2000.0
    duration: float = 20000.0

    def __post_init__(self):
        for name in ('arrival_rate', 'service_rate', 'duration'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.warmup) or self.warmup < 0:
            raise ValueError('warmup must be finite and nonnegative')
        if (not math.isfinite(self.warmup + self.duration)
                or self.warmup + self.duration <= self.warmup):
            raise ValueError('observation endpoint must be finite and advance the clock')

    @property
    def stable(self):
        return self.arrival_rate < self.service_rate


@dataclass
class Customer:
    identifier: int
    arrival: float
    observed: bool
    start: float | None = None


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    customer: int


@dataclass
class Result:
    mean_wait: float
    mean_sojourn: float
    mean_number: float
    utilization: float
    arrivals: int
    observed_arrival_rate: float
    little_residual: float
    end_time: float
    trace: list[Event]

    def metrics(self):
        return {k: getattr(self, k) for k in
                ('mean_wait', 'mean_sojourn', 'mean_number', 'utilization')}


def theory(config: Config) -> dict[str, float]:
    if not config.stable:
        raise ValueError('Stationary M/M/1 theory requires arrival_rate < service_rate')
    lam, mu = config.arrival_rate, config.service_rate
    return dict(mean_wait=lam / (mu * (mu - lam)),
                mean_sojourn=1 / (mu - lam), mean_number=lam / (mu - lam),
                utilization=lam / mu)


def simulate(config: Config, seed: int | np.random.SeedSequence = 0, *,
             record_trace: bool = False,
             arrival_draw: Callable[[], float] | None = None,
             service_draw: Callable[[], float] | None = None) -> Result:
    """Observe [warmup, warmup+duration); finish all customers arriving in it.

    Subsequent arrivals are omitted during draining: under FCFS they cannot
    affect this cohort. Integration is clipped to the observation window.
    Draw overrides are for deterministic verification, not M/M/1 inference.
    Equal-time departures precede arrivals. Service starts occur immediately.
    """
    streams = (seed if isinstance(seed, np.random.SeedSequence)
               else np.random.SeedSequence(seed)).spawn(2)
    arrival_rng, service_rng = map(np.random.default_rng, streams)
    arrival_draw = arrival_draw or (lambda: arrival_rng.exponential(1 / config.arrival_rate))
    service_draw = service_draw or (lambda: service_rng.exponential(1 / config.service_rate))
    end = config.warmup + config.duration
    events: list[tuple[float, int, int]] = []
    waiting: deque[Customer] = deque()
    active: Customer | None = None
    trace: list[Event] = []
    now = 0.0
    area = busy_area = wait_sum = sojourn_sum = 0.0
    observed = pending = 0
    next_id = 0

    def log(kind, customer):
        if record_trace:
            trace.append(Event(now, kind, customer.identifier))

    def schedule_arrival(previous):
        nonlocal next_id
        delta = float(arrival_draw())
        # Infinity is a supported sentinel for finite deterministic traces.
        if math.isnan(delta) or delta <= 0:
            raise ValueError('Interarrival draws must be positive')
        time = previous + delta
        if time <= previous:
            raise ValueError('Interarrival time does not advance the clock')
        if time < end:
            heapq.heappush(events, (time, 1, next_id))
            next_id += 1

    def start_service(customer):
        nonlocal active
        customer.start = now
        active = customer
        log('service_start', customer)
        service = float(service_draw())
        if (not math.isfinite(service) or service <= 0
                or not math.isfinite(now + service) or now + service <= now):
            raise ValueError('Service draws must be finite, positive and advance the clock')
        heapq.heappush(events, (now + service, 0, customer.identifier))

    def integrate(until):
        nonlocal area, busy_area
        width = max(0.0, min(until, end) - max(now, config.warmup))
        area += width * (len(waiting) + int(active is not None))
        busy_area += width * int(active is not None)

    schedule_arrival(0.0)
    while events:
        time, priority, identifier = heapq.heappop(events)
        if time >= end and pending == 0:
            integrate(end)
            now = end
            break
        integrate(time)
        now = time
        if priority == 1:
            in_window = config.warmup <= now < end
            customer = Customer(identifier, now, in_window)
            observed += int(in_window)
            pending += int(in_window)
            log('arrival', customer)
            if active is None:
                start_service(customer)
            else:
                waiting.append(customer)
            schedule_arrival(now)
        else:
            assert active is not None and active.identifier == identifier
            log('departure', active)
            if active.observed:
                wait_sum += active.start - active.arrival
                sojourn_sum += now - active.arrival
                pending -= 1
            active = None
            if waiting:
                start_service(waiting.popleft())
    if now < end:
        integrate(end)
        now = end
    mean_wait = wait_sum / observed if observed else math.nan
    mean_sojourn = sojourn_sum / observed if observed else math.nan
    mean_number = area / config.duration
    rate = observed / config.duration
    return Result(mean_wait, mean_sojourn, mean_number, busy_area / config.duration,
                  observed, rate, mean_number - rate * mean_sojourn, now, trace)
