"""Positive renewal interarrival and independent service-time distributions."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Distribution:
    name: str = 'exponential'
    parameters: dict[str, float] = field(default_factory=lambda: {'mean': 1.0})

    def __post_init__(self):
        schemas = {'exponential': {'mean'}, 'gamma': {'shape', 'scale'},
                   'erlang': {'shape', 'scale'}, 'weibull': {'shape', 'scale'},
                   'lognormal': {'meanlog', 'sigma'}, 'uniform': {'low', 'high'},
                   'deterministic': {'value'}}
        p = self.parameters
        if self.name not in schemas or set(p) != schemas[self.name]:
            raise ValueError(f'Invalid distribution/parameters: {self.name}; expected {schemas}')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p.values()):
            raise ValueError('Distribution parameters must be finite numbers')
        if self.name == 'uniform':
            valid = 0 <= p['low'] < p['high']
        elif self.name == 'lognormal':
            valid = p['sigma'] > 0
        else:
            valid = all(v > 0 for v in p.values())
        if self.name == 'erlang':
            valid = valid and float(p['shape']).is_integer()
        if not valid:
            raise ValueError('Invalid distribution parameter range')
        try:
            valid_moments = math.isfinite(self.mean) and math.isfinite(self.second_moment)
        except (OverflowError, ValueError):
            valid_moments = False
        if not valid_moments or self.mean <= 0:
            raise ValueError('Distribution moments must be finite and representable')

    @property
    def mean(self):
        p = self.parameters
        if self.name == 'exponential': return p['mean']
        if self.name in ('gamma', 'erlang'): return p['shape'] * p['scale']
        if self.name == 'weibull': return p['scale'] * math.gamma(1 + 1 / p['shape'])
        if self.name == 'lognormal': return math.exp(p['meanlog'] + p['sigma']**2 / 2)
        if self.name == 'uniform': return (p['low'] + p['high']) / 2
        return p['value']

    @property
    def second_moment(self):
        p = self.parameters
        if self.name == 'exponential': return 2 * p['mean']**2
        if self.name in ('gamma', 'erlang'): return p['shape'] * (p['shape'] + 1) * p['scale']**2
        if self.name == 'weibull': return p['scale']**2 * math.gamma(1 + 2 / p['shape'])
        if self.name == 'lognormal': return math.exp(2 * p['meanlog'] + 2 * p['sigma']**2)
        if self.name == 'uniform': return (p['low']**2 + p['low'] * p['high'] + p['high']**2) / 3
        return p['value']**2

    def sample(self, rng):
        p = self.parameters
        if self.name == 'exponential': return rng.exponential(p['mean'])
        if self.name in ('gamma', 'erlang'): return rng.gamma(p['shape'], p['scale'])
        if self.name == 'weibull': return p['scale'] * rng.weibull(p['shape'])
        if self.name == 'lognormal': return rng.lognormal(p['meanlog'], p['sigma'])
        if self.name == 'uniform': return rng.uniform(p['low'], p['high'])
        return p['value']
