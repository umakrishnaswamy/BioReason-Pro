import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "train_protein_grpo.py"


def load_grpo_module():
    module_name = "train_protein_grpo_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GRPO = load_grpo_module()


class TrainProteinGrpoContractsTest(unittest.TestCase):
    def test_extract_go_ids_preserves_order_and_deduplicates(self):
        text = "GO:0008150 and GO:0003674 and GO:0008150 again"
        self.assertEqual(GRPO.extract_go_ids(text), ["GO:0008150", "GO:0003674"])

    def test_extract_reasoning_and_answer_parses_sections(self):
        text = "<think>first infer signaling loss</think><answer>GO:0007165, GO:0005515</answer>"
        parsed = GRPO.extract_reasoning_and_answer(text)
        self.assertEqual(parsed["reasoning"], "first infer signaling loss")
        self.assertEqual(parsed["final_answer"], "GO:0007165, GO:0005515")

    def test_build_target_go_ids_merges_all_aspects(self):
        sample_meta = {
            "go_bp": "GO:0007165",
            "go_mf": "GO:0005515",
            "go_cc": "GO:0005737",
            "ground_truth_go_terms": "GO:0007165, GO:0009987",
        }
        self.assertEqual(
            GRPO.build_target_go_ids(sample_meta),
            ["GO:0007165", "GO:0005515", "GO:0005737", "GO:0009987"],
        )

    def test_standardize_group_rewards_returns_zeroes_for_constant_group(self):
        self.assertEqual(GRPO.standardize_group_rewards([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_compute_group_rewards_combines_named_components(self):
        completion = "<think>reasoning</think><answer>GO:0007165</answer>"
        sample_meta = {"go_bp": "GO:0007165"}

        totals, components = GRPO.compute_group_rewards(
            [completion],
            sample_meta,
            ["strict_format", "answer_nonempty", "go_overlap"],
            [1.0, 1.0, 2.0],
        )

        self.assertEqual(components["strict_format"], [1.0])
        self.assertEqual(components["answer_nonempty"], [1.0])
        self.assertEqual(components["go_overlap"], [1.0])
        self.assertEqual(totals, [4.0])

    def test_parse_reward_weights_validates_expected_count(self):
        with self.assertRaises(ValueError):
            GRPO.parse_reward_weights("1.0,2.0", 3)

    def test_parse_args_defaults_to_train_rl_contract(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model"])

        self.assertEqual(args.wandb_job_type, "train_rl")
        self.assertEqual(args.dataset_config, "disease_temporal_hc_reasoning_v1")
        self.assertEqual(args.reasoning_dataset_config, "disease_temporal_hc_reasoning_v1")
        self.assertEqual(args.checkpoint_artifact_name, "train-rl-output")
        self.assertEqual(args.output_dir, "data/artifacts/models/train_rl_output")
        self.assertEqual(args.max_eval_samples, 100)
        self.assertEqual(args.eval_sample_strategy, "stratified_aspect_profile")
        self.assertFalse(args.ablation_from_paper_rl)


if __name__ == "__main__":
    unittest.main()
