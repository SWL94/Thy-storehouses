"""Nonpreemptive G/G/c with exact area integration and arrival-cohort follow-up."""
from dataclasses import dataclass, field
import heapq
import math
from typing import Callable

import numpy as np

from .distributions import Distribution


@dataclass(frozen=True)
class Config:
    arrival: Distribution = field(default_factory=lambda: Distribution('exponential', {'mean': 1.25}))
    service: Distribution = field(default_factory=Distribution)
    servers: int = 1
    discipline: str = 'FCFS'
    warmup: float = 2000
    duration: float = 20000
    followup: float = 20000
    max_events: int = 5_000_000
    priority_values: tuple[float, ...] = (0, 1)
    priority_probabilities: tuple[float, ...] = (0.5, 0.5)

    def __post_init__(self):
        for name in ('servers', 'max_events'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.discipline not in ('FCFS', 'LCFS', 'PR', 'SJF'):
            raise ValueError('discipline must be FCFS, LCFS, PR or SJF (all nonpreemptive)')
        for name in ('warmup', 'duration', 'followup'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == 'duration' and value == 0):
                raise ValueError(f'Invalid {name}')
        end = self.warmup + self.duration
        if not math.isfinite(end + self.followup) or end <= self.warmup:
            raise ValueError('Time endpoints must be finite and advance the clock')
        if (not self.priority_values or len(self.priority_values) != len(self.priority_probabilities)
                or not all(math.isfinite(x) for x in self.priority_values)
                or not all(math.isfinite(p) and p >= 0 for p in self.priority_probabilities)
                or not math.isclose(sum(self.priority_probabilities), 1, abs_tol=1e-12, rel_tol=0)):
            raise ValueError('Priority values/probabilities must be finite, matched and sum to one')

    @property
    def load(self):
        return self.service.mean / (self.servers * self.arrival.mean)


@dataclass
class Customer:
    identifier: int
    arrival: float
    service: float
    priority: float
    observed: bool
    start: float | None = None
    departure: float | None = None
    server: int | None = None

    @property
    def wait(self):
        return None if self.start is None else self.start - self.arrival

    @property
    def sojourn(self):
        return None if self.departure is None else self.departure - self.arrival


@dataclass
class Result:
    mean_number: float
    mean_sojourn: float
    mean_wait: float
    mean_workload: float
    utilization: float
    little_residual: float
    customers: list[Customer]
    end_time: float
    observation_complete: bool
    stop_reason: str
    trace: list[tuple[float, str, int, int | None]]

    @property
    def censored_waits(self):
        return sum(c.start is None for c in self.customers)

    @property
    def censored_sojourns(self):
        return sum(c.departure is None for c in self.customers)

    def metrics(self):
        return {key: getattr(self, key) for key in
                ('mean_number', 'mean_sojourn', 'mean_wait', 'mean_workload',
                 'utilization', 'little_residual')}


def simulate(config: Config, seed=0, *, record_trace=False,
             arrival_draw: Callable[[], float] | None = None,
             service_draw: Callable[[], float] | None = None,
             priority_draw: Callable[[], float] | None = None) -> Result:
    """Continue arrivals in follow-up: LCFS/PR/SJF can be delayed by later jobs.

    At the follow-up or event limit, incomplete customer outcomes are censored;
    their means are undefined, never estimated from completers alone.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    arng, srng, prng = map(np.random.default_rng, root.spawn(3))
    arrival_draw = arrival_draw or (lambda: config.arrival.sample(arng))
    service_draw = service_draw or (lambda: config.service.sample(srng))
    priority_draw = priority_draw or (lambda: prng.choice(config.priority_values,
                                                        p=config.priority_probabilities))
    end = config.warmup + config.duration
    limit = end + config.followup
    events = []  # (time, departure-before-arrival, customer ID, server ID)
    waiting = []
    free = list(range(config.servers))
    active = {}
    samples = []
    trace = []
    now = workload = number_area = workload_area = busy_area = 0.0
    pending = next_id = event_count = 0
    stop_reason = 'cohort_complete'

    def log(kind, customer):
        if record_trace:
            trace.append((now, kind, customer.identifier, customer.server))

    def schedule_arrival(previous):
        nonlocal next_id
        interval = float(arrival_draw())
        # +inf is a deterministic-test sentinel; model distributions are finite.
        time = previous + interval
        if math.isnan(interval) or interval <= 0 or time <= previous:
            raise ValueError('Interarrival draws must be positive and advance time')
        if time < limit:
            heapq.heappush(events, (time, 1, next_id, -1))
            next_id += 1

    def advance(time):
        nonlocal now, workload, number_area, workload_area, busy_area
        busy = len(active)
        left, right = max(now, config.warmup), min(time, end)
        if right > left:
            width = right - left
            number_area += (len(waiting) + busy) * width
            busy_area += busy * width
            work_at_left = max(0.0, workload - busy * (left - now))
            workload_area += max(0.0, work_at_left * width - busy * width**2 / 2)
        workload = max(0.0, workload - busy * (time - now))
        now = time

    def start(customer):
        server = heapq.heappop(free)
        customer.start, customer.server = now, server
        finish = now + customer.service
        if not math.isfinite(finish) or finish <= now:
            raise ValueError('Service time must advance a finite clock')
        active[server] = customer
        heapq.heappush(events, (finish, 0, customer.identifier, server))
        log('service_start', customer)

    schedule_arrival(0)
    while events:
        time, kind, identifier, server = events[0]
        if now >= end and pending == 0:
            break
        if time >= end and pending == 0:
            advance(end)
            break
        if event_count >= config.max_events:
            stop_reason = 'max_events'
            break
        if time > limit:
            advance(limit)
            stop_reason = 'followup_limit'
            break
        heapq.heappop(events)
        advance(time)
        event_count += 1
        if kind == 1:
            service = float(service_draw())
            priority = float(priority_draw())
            if not math.isfinite(service) or service <= 0 or not math.isfinite(priority):
                raise ValueError('Service must be positive finite; priority must be finite')
            customer = Customer(identifier, now, service, priority, config.warmup <= now < end)
            if customer.observed:
                samples.append(customer)
                pending += 1
            workload += service
            if not math.isfinite(workload):
                raise ValueError('Workload overflow')
            log('arrival', customer)
            if free:
                start(customer)
            else:
                key = {'FCFS': identifier, 'LCFS': -identifier,
                       'PR': priority, 'SJF': service}[config.discipline]
                heapq.heappush(waiting, (key, identifier, customer))
            schedule_arrival(now)
        else:
            customer = active.pop(server)
            customer.departure = now
            pending -= int(customer.observed)
            log('departure', customer)
            heapq.heappush(free, server)
            if waiting:
                start(heapq.heappop(waiting)[2])
    else:
        # With no events there is no remaining work. Advance the empty clock.
        if now < end:
            advance(end)
    complete = now >= end
    if pending and stop_reason == 'cohort_complete':
        stop_reason = 'followup_limit'
    nan = math.nan
    n = len(samples)
    mean_wait = (sum(c.wait for c in samples) / n
                 if complete and n and all(c.wait is not None for c in samples) else nan)
    mean_sojourn = (sum(c.sojourn for c in samples) / n
                    if complete and n and all(c.sojourn is not None for c in samples) else nan)
    mean_number = number_area / config.duration if complete else nan
    return Result(mean_number, mean_sojourn, mean_wait,
                  workload_area / config.duration if complete else nan,
                  busy_area / (config.duration * config.servers) if complete else nan,
                  mean_number - n / config.duration * mean_sojourn,
                  samples, now, complete, stop_reason, trace)


def theory(config: Config):
    """Exact FCFS M/G/1 (PK) or M/M/c; no generic G/G/c exact claim."""
    if config.load >= 1 or config.discipline != 'FCFS' or config.arrival.name != 'exponential':
        return {}
    lam, service, rho = 1 / config.arrival.mean, config.service.mean, config.load
    if config.servers == 1:
        wait = lam * config.service.second_moment / (2 * (1 - rho))
        workload = wait
    elif config.service.name == 'exponential':
        a = lam * service
        erlang_b = 1.0
        for k in range(1, config.servers + 1):
            erlang_b = a * erlang_b / (k + a * erlang_b)
        delay = erlang_b / (1 - rho + rho * erlang_b)
        wait = delay * service / (config.servers * (1 - rho))
        workload = lam * (wait + service) * service
    else:
        return {}
    return dict(mean_wait=wait, mean_sojourn=wait + service,
                mean_number=lam * (wait + service), mean_workload=workload, utilization=rho)
