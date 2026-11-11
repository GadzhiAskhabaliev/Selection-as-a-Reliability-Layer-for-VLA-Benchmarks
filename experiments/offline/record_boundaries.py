"""Restore episode boundaries from outer TFDS records, before model loading.

Some NYU records have two consecutive is_last=True frames. Flattening and then
splitting on those flags invents extra episodes. Keep every observation/action,
preserve source flags, and set effective is_last only on each record's last step.
The old inference runtime and selectors remain unmodified.
"""
import json
import types


def validate_records(records, protocol):
    """Validate source-record metadata, independent of TensorFlow and the model."""
    lengths = []
    for index, record in enumerate(records):
        length = record['num_steps']
        if length <= 0 or record['is_first_indices'] != [0]:
            raise ValueError(f'Invalid source episode boundary at record {index}')
        if not record['is_last_indices'] or record['is_last_indices'][-1] != length - 1:
            raise ValueError(f'Source record {index} has no final is_last marker')
        lengths.append(length)
    if len(lengths) != protocol['split_episodes'] or sum(lengths) != protocol['full_split_expected_steps']:
        raise ValueError('Source TFDS record/step count differs from protocol before inference')
    if lengths != protocol['source_episode_lengths']:
        raise ValueError('TFDS record ordering/lengths differ from independent TFRecord audit')
    return {
        'status': 'PASS', 'boundary_policy': 'tfds_outer_record_v2',
        'num_source_episodes': len(lengths), 'num_source_steps': sum(lengths),
        'raw_is_last_count': sum(len(r['is_last_indices']) for r in records),
        'effective_is_last_count': len(lengths),
        'interior_is_last_markers': sum(len(r['is_last_indices']) - 1 for r in records),
        'dropped_steps': 0, 'episode_lengths': lengths, 'records': records,
    }


def restore_boundaries(episodes, tf, protocol, save_json):
    """Return equivalent nested episodes with corrected bookkeeping flags."""
    records = []
    for episode in episodes:
        # Project only flags. Image/action tensors are never passed through Python.
        flags = list(episode['steps'].map(
            lambda step: (step['is_first'], step['is_last'], step['is_terminal'])
        ).as_numpy_iterator())
        records.append({
            'episode': len(records), 'num_steps': len(flags),
            'is_first_indices': [i for i, f in enumerate(flags) if bool(f[0])],
            'is_last_indices': [i for i, f in enumerate(flags) if bool(f[1])],
            'is_terminal_indices': [i for i, f in enumerate(flags) if bool(f[2])],
        })
    try:
        audit = validate_records(records, protocol)
    except ValueError as exc:
        save_json('data_boundary_audit.json', {'status': 'FAIL', 'error': str(exc), 'records': records})
        raise
    lengths = tf.constant(audit['episode_lengths'], dtype=tf.int64)

    def fix_episode(episode_index, episode):
        length = tf.gather(lengths, episode_index)

        def fix_step(step_index, step):
            return {**step, 'source_is_last': step['is_last'],
                    'source_episode_index': episode_index, 'source_step_index': step_index,
                    'is_last': tf.equal(step_index, length - 1)}

        return {**episode, 'steps': episode['steps'].enumerate().map(fix_step)}

    corrected = episodes.enumerate().map(fix_episode)
    # Execute the actual mapped pipeline before accepting it for flattening.
    corrected_lengths = []
    for episode_index, episode in enumerate(corrected):
        flags = list(episode['steps'].map(
            lambda step: (step['is_last'], step['source_is_last'],
                          step['source_episode_index'], step['source_step_index'])
        ).as_numpy_iterator())
        record = records[episode_index]
        if len(flags) != record['num_steps']:
            raise ValueError('Boundary repair changed episode length')
        if [i for i, f in enumerate(flags) if bool(f[0])] != [len(flags) - 1]:
            raise ValueError('Effective boundary is not the final step')
        if [i for i, f in enumerate(flags) if bool(f[1])] != record['is_last_indices']:
            raise ValueError('Source boundary flags were not preserved')
        if any(int(f[2]) != episode_index or int(f[3]) != i for i, f in enumerate(flags)):
            raise ValueError('Source step mapping changed')
        corrected_lengths.append(len(flags))
    if corrected_lengths != audit['episode_lengths']:
        raise ValueError('Corrected dataset coverage changed')
    audit['mapped_pipeline_verified'] = True
    save_json('data_boundary_audit.json', audit)
    print('DATA BOUNDARY AUDIT PASS', json.dumps({k: v for k, v in audit.items()
                                               if k not in ('records', 'episode_lengths')}), flush=True)
    return corrected


def prepare_with_boundaries(baseline, original_prepare, protocol, kind):
    """Inject only the corrected dataset at the frozen runtime's builder boundary."""
    baseline.OUT.mkdir(parents=True, exist_ok=True)
    baseline.SCRATCH.mkdir(parents=True, exist_ok=True)
    # On Kaggle's host Python this re-execs the same bundle in pinned Python 3.11.
    # Heavy imports must stay after this point.
    baseline.bootstrap()
    import tensorflow as tf
    import tensorflow_datasets as tfds

    original_builder = tfds.builder_from_directory

    def builder_with_boundaries(*args, **kwargs):
        builder = original_builder(*args, **kwargs)

        def as_dataset(*ds_args, **ds_kwargs):
            if ds_kwargs.get('split') != protocol['split'] or ds_kwargs.get('shuffle_files') is not False:
                raise ValueError('Boundary repair requires the fixed, unshuffled protocol split')
            episodes = builder.as_dataset(*ds_args, **ds_kwargs)
            return restore_boundaries(episodes, tf, protocol, baseline.save_json)

        return types.SimpleNamespace(as_dataset=as_dataset)

    tfds.builder_from_directory = builder_with_boundaries
    try:
        runtime = original_prepare(kind=kind)
    finally:
        tfds.builder_from_directory = original_builder
    runtime.manifest['episode_boundary_policy'] = protocol['episode_boundary_policy']
    return runtime


def install(baseline, protocol):
    if protocol['episode_boundary_policy'] != 'tfds_outer_record_v2':
        raise ValueError('Unknown boundary policy')
    original_prepare = baseline.prepare_runtime
    baseline.prepare_runtime = lambda kind: prepare_with_boundaries(baseline, original_prepare, protocol, kind)
