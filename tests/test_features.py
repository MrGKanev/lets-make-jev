"""Regression checks for persisted policies, diagnostics and evaluation-only data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.analyze import classification_report, evaluate_policy, pick_thresholds, wilson
from jev.decide import Jev, build_prompt, prompt_fingerprint
from jev.generate import generate_label
from tests.smoke_fake import Tok


class Features(unittest.TestCase):
    def setUp(self):
        self.choices = {'billing': 'fees', 'other': 'unrelated'}

    def test_matched_prompt_and_one_token(self):
        response = unittest.mock.Mock()
        response.json.return_value = {'choices': [{'message': {'content': 'A'}}]}
        with patch('jev.generate.http.post', return_value=response) as post:
            r = generate_label('m', 'task', 'text', self.choices)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['messages'][0]['content'], build_prompt('task', 'text', self.choices)[0])
        self.assertEqual(body['max_tokens'], 1)
        self.assertEqual(r['choice'], 'billing')

    def test_policy_roundtrip_and_mismatch(self):
        policy = {'model': 'm', 'prompt_version': '1', 'temperature': 2, 'min_p': .8, 'min_margin': .2,
                  'prompt_fingerprint': prompt_fingerprint('task', self.choices)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy.json'
            path.write_text(json.dumps(policy))
            j = Jev('m', tok=Tok(), policy=path)
            with patch.object(j, '_scores', return_value=[.9, .1]):
                r = j.decide('task', 'text', self.choices)
                batch = j._result('task', self.choices, {'billing': .9, 'other': .1})
                self.assertAlmostEqual(r['probabilities']['billing'], .75)
                self.assertEqual(r, batch)
                self.assertEqual(r['route'], 'review')
                self.assertIn('low_confidence', r['review_reasons'])
                self.assertEqual(r['raw_probabilities']['billing'], .9)
                with self.assertRaises(ValueError):
                    j.decide('different task', 'text', self.choices)
            with self.assertRaises(ValueError):
                Jev('wrong-model', tok=Tok(), policy=path)

    def test_temporary_server_error_does_not_change_backend(self):
        import requests
        response = unittest.mock.Mock(ok=False, status_code=503)
        response.raise_for_status.side_effect = requests.HTTPError('503')
        j = Jev('m', tok=Tok())
        with patch('jev.decide.http.post', return_value=response):
            with self.assertRaises(requests.HTTPError):
                j.decide('task', 'text', self.choices)
        self.assertEqual(j.backend, 'score')

    def test_category_limits(self):
        for choices in ({}, {'a': ''}, {str(i): '' for i in range(11)}):
            with self.assertRaises(ValueError):
                build_prompt('task', 'text', choices)

    def test_confusion_and_intervals(self):
        rows = [{'id': '1', 'text': 'one', 'label': 'a', 'probs': {'a': .9, 'other': .1}},
                {'id': '2', 'text': 'two', 'label': 'other', 'probs': {'a': .99, 'other': .01}},
                {'id': '3', 'text': 'three', 'label': 'other', 'probs': {'a': .1, 'other': .9}}]
        r = classification_report(rows)
        self.assertEqual(r['confusion_matrix']['other']['a'], 1)
        self.assertEqual(r['per_class']['a']['precision'], .5)
        self.assertEqual(r['per_class']['other']['recall'], .5)
        self.assertEqual(r['errors'][0]['id'], '2')
        e = evaluate_policy(rows, .8, .2)
        self.assertEqual(e['n_auto'], 2)
        self.assertEqual(e['n_correct'], 1)
        self.assertAlmostEqual(wilson(100, 100)[0], .963, places=3)
        self.assertEqual(wilson(0, 0), [0, 1])
        small = [rows[0]] * 20
        self.assertIsNone(pick_thresholds(small))
        self.assertIsNone(pick_thresholds(small * 2, .95, criterion='wilson'))
        self.assertIsNotNone(pick_thresholds(small * 6, .95, criterion='wilson'))

    def test_demo_score_and_generate_endpoints(self):
        import threading
        import requests
        from http.server import ThreadingHTTPServer
        from ui.serve import make_handler
        j = Jev('m', tok=Tok())
        case = {'task': 'task', 'text': 'text', 'choices': self.choices, 'label': 'billing'}
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(j, '', [case], (.8, .2)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(j, '_scores', return_value=[.9, .1]), patch('ui.serve.generate', return_value={'choice': 'billing', 'text': 'billing'}):
                for lane in ('score', 'generate'):
                    response = requests.post(f'http://127.0.0.1:{server.server_port}/{lane}', json={'i': 0}, timeout=5)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json()['ok'])
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_challenge_integrity(self):
        for domain in ('support', 'news'):
            root = Path('data')
            challenge = [json.loads(l) for l in (root / f'{domain}_challenge.jsonl').read_text().splitlines()]
            seen = set()
            for split in ('train', 'dev', 'test'):
                seen.update(json.loads(l)['text'] for l in (root / f'{domain}_{split}.jsonl').read_text().splitlines())
            self.assertFalse(seen & {r['text'] for r in challenge})
            self.assertEqual(len({r['id'] for r in challenge}), len(challenge))
            self.assertTrue(all(r['label'] in r['choices'] for r in challenge))
            self.assertEqual({t for r in challenge for t in r['tags']},
                             {'bulgarian', 'mixed_language', 'typos', 'near_domain_oos', 'multi_intent', 'injection', 'long_text'})


if __name__ == '__main__':
    unittest.main()
