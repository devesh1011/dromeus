#!/usr/bin/env python3
"""Verify this portable M3 retained-log package, without training or network access.

Default use: python3 verification/verify-retained-logs.py
Use --output verification/retained-log-check.json to explicitly write a receipt.
Retained row hashes check preserved bytes; they are not independent attestation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--index', default='audit/log-index.json')
    parser.add_argument('--output', type=Path, help='Explicitly write a JSON receipt; otherwise read-only')
    args = parser.parse_args()
    root = args.root.resolve()
    failures = []
    inputs = {}
    workers = []
    groups = collections.defaultdict(set)
    totals = collections.Counter()

    def check(ok, message):
        if not ok:
            failures.append(message)
        return ok

    def local_path(value):
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Input path escapes package root: ' + str(value))
        return path

    def read(value):
        data = local_path(value).read_bytes()
        inputs[str(value)] = digest(data)
        return data

    def read_csv(value):
        return list(csv.DictReader(read(value).decode('utf-8').splitlines()))

    try:
        index = json.loads(read(args.index))
        refs = index.get('reference_inputs', {})
        digit_nodes_path = refs.get('digit_nodes_csv', 'results/privacy-utility/worker-results.csv')
        digit_nodes = {(r['run_id'], r['node_id']): r for r in read_csv(digit_nodes_path)}
        trajectory_path = refs.get('digit_trajectory_csv', 'results/privacy-utility/learning-trajectory.csv')
        trajectory_rows = read_csv(trajectory_path)
        digit_trajectory = {(r['run_id'], int(r['rank']), int(r['logical_steps'])): r for r in trajectory_rows}
        check(len(digit_trajectory) == len(trajectory_rows) == 528, 'Expected 528 unique accepted-digit trajectory rows')
        warning_nodes = {}
        warning_nodes_path = refs.get('warning_nodes_csv', 'results/warning-detection/worker-assessment.csv')
        warning_nodes = {(r['run_id'], r['node_id']): r for r in read_csv(warning_nodes_path)}
        entries = index['files']
        check(bool(entries), 'Index contains no logs')
        seen_paths = set()
        seen_workers = set()
        for entry in entries:
            name = entry['extract_path']
            label = name
            try:
                check(name not in seen_paths, 'Duplicate extract path: ' + name)
                seen_paths.add(name)
                stored = read(name)
                encoding = entry.get('storage_encoding', 'gzip-jsonl')
                if encoding not in ('jsonl', 'gzip-jsonl'):
                    raise ValueError('Unsupported storage encoding: ' + str(encoding))
                raw = stored if encoding == 'jsonl' else gzip.decompress(stored)
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                counts = collections.Counter(q.get('event', q.get('type', '<no discriminator>')) for q in rows)
                check(digest(stored) == entry['extract_sha256'], label + ': stored-file SHA256 mismatch')
                check(len(stored) == entry['extract_bytes'], label + ': stored-file byte count mismatch')
                check(digest(raw) == entry['retained_rows_sha256'], label + ': retained-row SHA256 mismatch')
                stats = entry['stats']
                check(len(rows) == stats['kept'], label + ': retained row count mismatch')
                check(dict(counts) == stats['kept_events'], label + ': event counts mismatch')
                metrics_rows = [q for q in rows if q.get('event') in ('round_metrics', 'task_round_metrics')]
                run_ids = {q['run_id'] for q in metrics_rows}
                node_ids = {q['node_id'] for q in metrics_rows}
                manifests = {q['manifest_hash'] for q in metrics_rows}
                if not check(len(run_ids) == len(node_ids) == len(manifests) == 1, label + ': inconsistent metric identities'):
                    continue
                run_id, node_id, manifest = next(iter(run_ids)), next(iter(node_ids)), next(iter(manifests))
                key = (run_id, node_id)
                check(key not in seen_workers, label + ': duplicate worker')
                seen_workers.add(key)
                for field, value in [('run_id', run_id), ('node_id', node_id), ('manifest_hash', manifest)]:
                    check(entry.get(field) == value, label + ': index ' + field + ' mismatch or missing')
                rounds = [q['round_id'] for q in metrics_rows]
                evaluations = [q for q in metrics_rows if q['event'] == 'task_round_metrics']
                check(len(rounds) == 75 and set(rounds) == set(range(75)), label + ': expected exactly rounds 0..74')
                check(sorted(q['round_id'] for q in evaluations) == [14, 29, 44, 59, 74], label + ': evaluation cadence mismatch')
                observations = [q for q in rows if q.get('event') == 'consensus_observation']
                distances = [q for q in rows if q.get('event') == 'node_consensus_distance']
                if entry.get('cohort') == 'digit_calibration' and run_id.startswith('cal-'):
                    check(not observations, label + ': unexpected modern observations in legacy calibration')
                    check(len(distances) == 75 and {q['round_id'] for q in distances} == set(range(75)), label + ': legacy consensus-distance coverage mismatch')
                    consensus_schema = 'legacy node_consensus_distance; consensus_observation not emitted'
                else:
                    check(len(observations) == 75 and {q['round_id'] for q in observations} == set(range(75)), label + ': consensus coverage mismatch')
                    consensus_schema = 'consensus_observation'
                check(counts['node_complete'] == 1, label + ': expected one completion')
                for q in observations + evaluations + distances:
                    check(q.get('run_id') == run_id and q.get('node_id') == node_id and q.get('manifest_hash') == manifest, label + ': typed identity mismatch')
                if key in digit_nodes:
                    cohort = 'original_digit_final'
                    rank = int(digit_nodes[key]['rank'])
                    check(entry.get('rank') == rank, label + ': CSV rank mismatch')
                    for q in evaluations:
                        point = digit_trajectory[(run_id, rank, (q['round_id'] + 1) * 10)]
                        for source in ('mnist', 'usps', 'svhn', 'optdigits'):
                            check(q['evaluation_metrics'].get('accuracy_' + source) == float(point['accuracy_' + source]), label + ': full-precision accuracy mismatch')
                            totals['exact_digit_value_comparisons'] += 1
                        check(q['evaluation_metrics'].get('accuracy_macro_source') == float(point['mean_accuracy']), label + ': macro accuracy mismatch')
                        totals['exact_digit_value_comparisons'] += 1
                    final = next(q for q in evaluations if q['round_id'] == 74)
                    for source in ('mnist', 'usps', 'svhn', 'optdigits'):
                        check(final['evaluation_metrics']['accuracy_' + source] == float(digit_nodes[key]['accuracy_' + source]), label + ': node CSV endpoint mismatch')
                        totals['digit_node_csv_endpoint_comparisons'] += 1
                    totals['digit_evaluation_rows'] += len(evaluations)
                elif run_id.startswith('repair-assessment-'):
                    cohort = 'warning_assessment'
                    check(bool(warning_nodes) and key in warning_nodes, label + ': absent from warning nodes CSV')
                elif run_id.startswith('repair-calibration-'):
                    cohort = 'warning_calibration'
                else:
                    cohort = entry.get('cohort', 'digit_calibration')
                groups[cohort].add(run_id)
                workers.append({'path': name, 'cohort': cohort, 'run_id': run_id, 'node_id': node_id, 'manifest_hash': manifest, 'rounds': len(rounds), 'evaluation_rows': len(evaluations), 'consensus_observations': len(observations), 'node_consensus_distances': len(distances), 'consensus_schema': consensus_schema, 'events': dict(counts)})
                totals['files'] += 1
                totals['stored_bytes'] += len(stored)
                totals['retained_rows'] += len(rows)
            except Exception as exc:
                failures.append(label + ': ' + type(exc).__name__ + ': ' + str(exc))
        cohort_counts = collections.Counter(w['cohort'] for w in workers)
        check(len(groups['original_digit_final']) == 22 and cohort_counts['original_digit_final'] == 88, 'Expected 22 accepted digit groups / 88 workers')
        check(len(groups['warning_assessment']) == 14 and cohort_counts['warning_assessment'] == 56, 'Expected 14 warning assessment groups / 56 workers')
        check(len(groups['warning_calibration']) == 12 and cohort_counts['warning_calibration'] == 48, 'Expected 12 warning calibration groups / 48 workers')
        check(len(groups['digit_calibration']) == 10 and cohort_counts['digit_calibration'] == 40, 'Expected 10 original digit calibration groups / 40 workers')
        check(len(entries) == totals['files'] == 232, 'Expected exactly 232 verified worker logs')
        check(totals['digit_evaluation_rows'] == 440, 'Expected 440 accepted-digit evaluation rows')
    except Exception as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
    receipt = {'version': 1, 'status': 'pass' if not failures else 'fail', 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verifier_sha256': digest(Path(__file__).read_bytes()), 'scope': 'Read-only local verification of retained original record extracts; no training, network, model or privacy-ledger access. Complete source-log bytes and model-export replay are not established.', 'input_sha256': inputs, 'totals': dict(totals), 'groups': {k: sorted(v) for k, v in groups.items()}, 'workers': workers, 'failures': failures}
    if args.output is not None:
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'totals': receipt['totals'], 'groups': {k: len(v) for k, v in groups.items()}, 'failures': failures}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
