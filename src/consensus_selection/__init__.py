"""Verifier-free selection from normalized action prefixes; no ML dependencies.

Paper equations (1), (2), (S1): Euclidean distance on flattened, frozen-normalized
executed prefixes, median positive distance + 1e-6 bandwidth, Gaussian density.
This is a paper-based implementation, not a copy of the authors' source package.
"""
import math
import statistics


def normalize_openvla(actions, stats):
    """Invert frozen OpenVLA q01/q99 unnormalization on raw [N, D] actions.

    Mask=False coordinates (e.g. gripper) already inhabit the model's normalized
    space and are left unchanged. Constant masked coordinates become zero and
    cannot influence selection. No fitting, clipping, or target data is used.
    """
    low, high = stats['q01'], stats['q99']
    mask = stats.get('mask', [True] * len(low))
    if not low or not len(low) == len(high) == len(mask):
        raise ValueError('Inconsistent action statistics')
    if any(not math.isfinite(float(v)) for v in [*low, *high]):
        raise ValueError('Non-finite action statistics')
    if any(h < l for l, h in zip(low, high)):
        raise ValueError('Negative action range')
    result = []
    for action in actions:
        if len(action) != len(low) or any(not math.isfinite(float(x)) for x in action):
            raise ValueError('Invalid action dimension or non-finite action')
        row = []
        for x, lo, hi, use_range in zip(action, low, high, mask):
            if not use_range:
                row.append(float(x))
            elif hi == lo:
                if not math.isclose(x, lo, rel_tol=0, abs_tol=1e-8):
                    raise ValueError('Constant action dimension has unexpected value')
                row.append(0.0)
            else:
                row.append(2 * (float(x) - lo) / (hi - lo) - 1)
        result.append(row)
    if not result:
        raise ValueError('Empty candidate pool')
    return result


def select(normalized_prefixes, epsilon=1e-6):
    """Select CS and Euclidean medoid indices from an [N, R, D] array/list.

    Inputs must already use a fixed, meaningful normalization. Exact ties select
    the first sampled index. Self-similarity is included in CS scores (constant
    for every candidate). Diagnostics allow all choices to be audited offline.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    vectors = []
    shape = None
    for prefix in normalized_prefixes:
        if len(prefix) == 0 or len(prefix[0]) == 0:
            raise ValueError('Empty action prefix')
        this_shape = (len(prefix), len(prefix[0]))
        if shape is None:
            shape = this_shape
        if this_shape != shape or any(len(action) != shape[1] for action in prefix):
            raise ValueError('Candidate prefixes must share [R, D] shape')
        vector = [float(v) for action in prefix for v in action]
        if not all(math.isfinite(v) for v in vector):
            raise ValueError('Non-finite normalized candidate')
        vectors.append(vector)
    if not vectors:
        raise ValueError('Empty candidate pool')
    n = len(vectors)
    distances = [[0.0] * n for _ in range(n)]
    positive = []
    for i in range(n):
        for j in range(i):
            distance = math.dist(vectors[i], vectors[j])
            if not math.isfinite(distance):
                raise ValueError('Non-finite pairwise distance')
            distances[i][j] = distances[j][i] = distance
            if distance > 0:
                positive.append(distance)
    h = (statistics.median(positive) if positive else 1.0) + epsilon
    scores = [math.fsum(math.exp(-0.5 * (d / h) ** 2) for d in row)
              for row in distances]
    medoid_costs = [math.fsum(row) for row in distances]
    return {
        'cs_index': max(range(n), key=scores.__getitem__),
        'medoid_index': min(range(n), key=medoid_costs.__getitem__),
        'bandwidth': h, 'cs_scores': scores, 'medoid_costs': medoid_costs,
        'num_unique': len({tuple(v) for v in vectors}),
        'mean_pairwise_distance': math.fsum(positive) / (n * (n - 1) / 2) if n > 1 else 0.0,
        'max_pairwise_distance': max(positive, default=0.0),
    }
