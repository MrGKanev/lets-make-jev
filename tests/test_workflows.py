"""HTTP resilience, data checks, experiment isolation and full review/CLI roundtrip."""
import contextlib
import io
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer

import requests
from bench.experiment import experiment
from data.validate import audit
from jev import http
from jev.cli import main, apply_reviews, review_queue
from jev.decide import Jev, validate_scores
from tests.smoke_fake import Fake, Tok


class Workflows(unittest.TestCase):
    def test_server_shutdown_kills_stalled_process(self):
        import subprocess
        from bench.full import stop_server
        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired('server', 1), 0]
        stop_server(proc, timeout=1)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)

    def test_server_start_interruption_cleans_up(self):
        from bench.full import start_server
        with tempfile.TemporaryDirectory() as directory:
            proc = Mock()
            proc.poll.return_value = None
            with patch('bench.full.subprocess.Popen', return_value=proc), \
                 patch('bench.full.requests.get', side_effect=KeyboardInterrupt), \
                 patch('bench.full.stop_server') as stop:
                with self.assertRaises(KeyboardInterrupt):
                    start_server('model', 30000, Path(directory) / 'server.log')
                stop.assert_called_once_with(proc)

    def test_cli_rejects_incomplete_batch_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.jsonl'
            output = Path(directory) / 'output.jsonl'
            row = {'task': 'route', 'text': 'fees', 'choices': {'a': 'fees', 'other': 'other'}}
            source.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with patch('jev.cli.Jev') as engine, contextlib.redirect_stderr(io.StringIO()):
                engine.return_value.decide_many.return_value = []
                with self.assertRaises(SystemExit) as error:
                    main(['predict', '--input', str(source), '--output', str(output), '--model', 'm', '--batch', '2'])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_http_retries_and_pooling(self):
        client = Mock()
        transient = Mock(status_code=503)
        success = Mock(status_code=200)
        client.post.side_effect = [requests.Timeout(), transient, success]
        with patch('jev.http.session', return_value=client), patch('jev.http.time.sleep') as sleep:
            self.assertIs(http.post('http://test', json={}), success)
        self.assertEqual(client.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        transient.close.assert_called_once()
        self.assertIs(http.session(), http.session())
        client.post.reset_mock()
        client.post.side_effect = [Mock(status_code=401)]
        with patch('jev.http.session', return_value=client):
            self.assertEqual(http.post('http://test').status_code, 401)
        self.assertEqual(client.post.call_count, 1)

    def test_probability_and_batch_validation(self):
        for scores in ([.8], [.8,.8], [float('nan'),.2], [float('inf'),0], [-.1,1.1], [True,False]):
            with self.assertRaises(http.ResponseError):
                validate_scores(scores,2)
        j = Jev('m',tok=Tok())
        response = Mock(ok=True)
        response.json.return_value = {'scores': [[.8,.2]]}
        with patch('jev.http.post', return_value=response):
            with self.assertRaises(http.ResponseError):
                j.decide_many('task',['one','two'], {'a':'a','other':'other'})
        self.assertEqual(j.backend,'score')
        self.assertEqual(j.decide_many('task',[],{'a':'a','other':'other'}),[])

    def test_data_leakage_schema_and_shared_domains(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'task':'task','text':'An example','choices':{'a':'a','other':'other'},'label':'a'}
            def write(name, rows):
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write('one_train.jsonl',[row]);write('two_train.jsonl',[row])
            r=audit(root.glob('*.jsonl'))
            self.assertTrue(r['ok']);self.assertEqual(len(r['warnings']),1)
            write('two_test.jsonl',[{**row,'text':' AN   EXAMPLE '}])
            self.assertFalse(audit(root.glob('*.jsonl'))['ok'])
            write('one_train.jsonl',[row,row])
            self.assertTrue(any('duplicate' in e for e in audit(root.glob('*.jsonl'))['errors']))
            write('two_train.jsonl',[{**row,'choices':{'a':'different','other':'other'}}])
            write('one_dev.jsonl',[{**row,'text':'another','label':'bad'}])
            self.assertTrue(any('label' in e for e in audit(root.glob('*.jsonl'))['errors']))

    def test_experiment_success_failure_and_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.jsonl';source.write_text('original')
            with experiment(root/'runs',{'seed':0},[source]) as first:
                source.write_text('changed')
            with self.assertRaises(RuntimeError):
                with experiment(root/'runs',{},[source]) as second:
                    raise RuntimeError('failed inference')
            self.assertNotEqual(first,second)
            self.assertEqual((first/'data/input.jsonl').read_text(),'original')
            self.assertEqual(json.loads((first/'manifest.json').read_text())['status'],'complete')
            self.assertEqual(json.loads((second/'manifest.json').read_text())['status'],'failed')
            self.assertNotIn('git', (first/'manifest.json').read_text().lower())

    def test_benchmark_main_isolates_runs_and_reads_snapshots(self):
        import bench.full as full
        with tempfile.TemporaryDirectory() as d:
            def run(model, url, args, domain):
                snapshot = Path(args.data_dir) / f'{domain}_test.jsonl'
                self.assertTrue(snapshot.exists())
                self.assertIn(d, str(snapshot))
                return {'policy': None, 'sanity': {'warn': False}}
            arguments = ['bench.full', '--models', 'm', '--url', 'http://fake', '--domains', 'support', '--out', d, '--no-plot']
            with patch('sys.argv', arguments), patch.object(full, 'run_model', side_effect=run), patch('bench.report.write_report'), contextlib.redirect_stdout(io.StringIO()):
                full.main(); full.main()
            runs = list(Path(d).iterdir())
            self.assertEqual(len(runs), 2)
            for run_path in runs:
                manifest = json.loads((run_path/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'complete')
                self.assertNotIn('data_dir', manifest['config'])
                self.assertTrue((run_path/'m__support.json').exists())
                self.assertTrue((run_path/'data-quality.json').exists())

    def test_cli_predict_review_apply_reevaluate(self):
        from jev.decide import Jev as Engine
        srv=ThreadingHTTPServer(('127.0.0.1',0),Fake)
        threading.Thread(target=srv.serve_forever,daemon=True).start()
        url=f'http://127.0.0.1:{srv.server_port}'
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                source=root/'input.jsonl';pred=root/'pred.jsonl';queue=root/'queue.jsonl';labels=root/'labels.jsonl';again=root/'again.jsonl'
                row={'id':'one','task':'route','text':'fees','choices':{'a':'fees','other':'other'}}
                source.write_text(json.dumps(row)+'\n')
                def engine(model,base_url,policy=None):
                    return Engine(model,base_url,tok=Tok(),policy=policy)
                with patch('jev.cli.Jev',side_effect=engine), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['predict','--input',str(source),'--output',str(pred),'--model','m','--url',url,'--batch','2']),0)
                    predictions=[json.loads(l) for l in pred.read_text().splitlines()]
                    # Force review for this workflow fixture, preserving model evidence.
                    predictions[0]['route']='review'
                    pred.write_text(json.dumps(predictions[0])+'\n')
                    main(['review','export','--predictions',str(pred),'--output',str(queue)])
                    review=json.loads(queue.read_text());review['review_label']='other'
                    queue.write_text(json.dumps(review)+'\n')
                    main(['review','apply','--predictions',str(pred),'--reviews',str(queue),'--output',str(labels)])
                    main(['predict','--input',str(labels),'--output',str(again),'--model','m','--url',url])
                    self.assertEqual(json.loads(again.read_text())['label'],'other')
                    with self.assertRaises(SystemExit):
                        main(['predict','--input',str(source),'--output',str(pred),'--model','m'])
                tampered={**review,'text':'changed'}
                with self.assertRaises(ValueError):
                    apply_reviews(predictions,[tampered])
                with self.assertRaises(ValueError):
                    apply_reviews(predictions,[{**review,'review_label':'invalid'}])
        finally:
            srv.shutdown();srv.server_close()


if __name__=='__main__':
    unittest.main()
