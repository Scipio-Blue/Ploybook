import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ['PLOYBOOK_OFFLINE'] = '1'
sys.path.insert(0,str(ROOT / 'src'))
from ploybook import Ploybook, score_policy
from run_phase2 import aggregate


class PolicyTests(unittest.TestCase):
    def test_lower_and_severity_boundaries(self):
        self.assertFalse(score_policy(.349999, 1., False)['blocked'])
        self.assertTrue(score_policy(.35, .70, False)['blocked'])
        self.assertFalse(score_policy(.35, .699999, False)['blocked'])

    def test_core_and_fast_block_paths(self):
        self.assertTrue(score_policy(.5, 0., False)['blocked'])
        self.assertTrue(score_policy(.70, 0., False)['blocked'])
        self.assertEqual(score_policy(0.,0.,True)['p_harmful'],1.)
        self.assertFalse(aggregate([])['blocked'])

    def test_rounding_does_not_change_verdict(self):
        a = score_policy(.499999, 0., False)
        self.assertEqual(a['p_harmful'],.5)
        self.assertFalse(a['blocked'])

    def test_severity_may_come_from_another_hypothesis(self):
        a = aggregate([{'p_fidelity':1.,'p_harm_intent':.4,'p_sev_bin':.1},
                       {'p_fidelity':.1,'p_harm_intent':.1,'p_sev_bin':.8}])
        self.assertEqual(a,{'p_harmful':.70,'blocked':True})

    def test_screening_does_not_load_models(self):
        with patch('ploybook.jev_gate',return_value={'jb':.85,'harm_raw':.85}), \
             patch.object(Ploybook,'_load',side_effect=AssertionError('model loaded')):
            a = Ploybook().inspect('test evidence')
        self.assertTrue(a['blocked'])
        self.assertEqual(a['path'],'screening')

    def test_client_offline_guard_blocks_network(self):
        import jev_client
        with patch.object(jev_client,'_cache_get',return_value=None), \
             patch('urllib.request.urlopen',side_effect=AssertionError('network called')):
            with self.assertRaisesRegex(RuntimeError,'Offline mode'):
                jev_client.system_one({'test':'missing-cache'}, {})

    def test_navigation_extraction_and_four_hypothesis_decision(self):
        import numpy as np
        vectors = np.load(ROOT/'benchmarks/RAD-wiki/page_emb.npy')
        class Encoder:
            def encode(self, texts): return vectors[:1]
        class Model:
            def generate(self, prompts, **kwargs):
                self.prompt = prompts[0]
                return ['request for test information.']
        model = Model()
        with patch('ploybook.jev_gate',return_value={'jb':.1,'harm_raw':.1}), \
             patch('ploybook.jev_battery',return_value={
                 'p_fidelity':.7,'p_harm_intent':.5,'p_sev_bin':.8}) as battery:
            a = Ploybook(encoder=Encoder(),language_model=model).inspect('test evidence')
        self.assertEqual(battery.call_count,4)
        self.assertEqual(len(a['hypotheses']),4)
        self.assertEqual(a['hypotheses'][-1]['page_id'],'plain')
        self.assertIn('UNTRUSTED input evidence:',model.prompt)
        self.assertTrue(a['blocked'])


if __name__ == '__main__': unittest.main()
