import json
import tempfile
import unittest
from pathlib import Path

from redundancy_project.src.temporal_measurements import (
    get_temporal_measurement_epochs,
)
from redundancy_project.src.metric_definitions import ordered_sample_identity
from redundancy_project.temporal_dashboard import (
    write_dashboard,
    write_experiment_dashboard,
)


class TemporalDashboardTests(unittest.TestCase):
    def test_measurement_cadence_always_includes_final_epoch(self):
        self.assertEqual(get_temporal_measurement_epochs(10, 1), list(range(1, 11)))
        self.assertEqual(get_temporal_measurement_epochs(10, 4), [4, 8, 10])
        self.assertEqual(get_temporal_measurement_epochs(3, 5), [3])
        with self.assertRaises(ValueError):
            get_temporal_measurement_epochs(10, 0)

    def test_writes_self_contained_interactive_dashboard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            train_positions = [1, 0]
            validation_positions = [3, 1]
            train_identity = ordered_sample_identity(
                "train", train_positions, "positions_in_fixed_training_subset"
            )
            validation_identity = ordered_sample_identity(
                "validation",
                validation_positions,
                "positions_in_fixed_validation_split",
            )
            metadata = {
                "run_id": "run-test",
                "seed": 7,
                "requested_config": {
                    "architecture": "CNN",
                    "activation": "tanh",
                    "initializer": "uniform",
                    "optimizer": "sgd",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "dropout": 0.0,
                    "dataset": {"train_fraction": 0.02},
                    "training": {"batch_size": 128, "epochs": 3},
                },
                "temporal_measurement_samples": {
                    "train_count": 2,
                    "train_positions_in_training_subset": train_positions,
                    "train_dataset_indices": [9, 4],
                    "train_identity": train_identity,
                    "validation_count": 2,
                    "validation_positions": validation_positions,
                    "validation_split_indices": validation_positions,
                    "validation_identity": validation_identity,
                },
                "train_subset_indices": [4, 9, 15],
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            metrics = {
                "training_iteration": 1,
                "train_eval_loss": 0.5,
                "train_eval_acc": 0.8,
                "validation_eval_loss": 0.6,
                "validation_eval_acc": 0.7,
            }
            (run_dir / "result.json").write_text(
                json.dumps(metrics) + "\n", encoding="utf-8"
            )
            measure = {
                "training_iteration": 1,
                "measurement_seconds": 1.25,
                "P_L": [[1.0, 0.2], [0.2, 1.0]],
                "P_C": 1.5,
                "metric_definition_version": 2,
                "p_c_definition": "spectral_energy_participation_ratio_of_row_normalized_W_in",
                "fc_intervention": "centered_pca_projection",
                "train": {
                    "R_L": [[1.0, None], [None, 1.0]],
                    "R_C": 1.4,
                    "F_L": [0.1, 0.2],
                    "reference_sample": train_identity,
                    "reference_activation_mean": [0.5, 0.5],
                    "activation_covariance_eigenvalues": [0.2, 0.1],
                    "F_C": {
                        str(k): {
                            "low_variance": 0.1,
                            "random": [0.2, 0.3],
                            "R_C_remaining_lowvar": 1.0,
                        }
                        for k in (0, 1, 5, 10)
                    },
                },
                "validation": {
                    "R_L": [[1.0, -0.2], [-0.2, 1.0]],
                    "R_C": 1.2,
                    "F_L": [0.3, 0.4],
                    "reference_sample": validation_identity,
                    "reference_activation_mean": [0.6, 0.4],
                    "activation_covariance_eigenvalues": [0.3, 0.1],
                    "F_C": {
                        str(k): {
                            "low_variance": 0.3,
                            "random": [0.4, 0.5],
                            "R_C_remaining_lowvar": 1.0,
                        }
                        for k in (0, 1, 5, 10)
                    },
                },
            }
            (run_dir / "temporal_measurements.jsonl").write_text(
                json.dumps(measure) + "\n", encoding="utf-8"
            )

            output = write_dashboard(run_dir)

            html = output.read_text(encoding="utf-8")
            self.assertIn("const DATA=", html)
            self.assertIn("<h2>P_L</h2>", html)
            self.assertIn("<h2>F_C</h2>", html)
            self.assertIn("−1: opposite directions<br>0: neutral score<br>+1: aligned", html)
            self.assertIn(
                "−1 anti<br>0 neutral<br>+1 corr.<br>black N/A",
                html,
            )
            self.assertIn('v===null?"#111111":color(v,min,max)', html)
            self.assertIn('name="rlSplit" value="train"', html)
            self.assertIn('name="rlSplit" value="validation"', html)
            self.assertIn('class="matrix-content"', html)
            self.assertIn('class="measure-panel matrix-panel"', html)
            self.assertIn(".matrix-panel .matrix-content{position:absolute;inset:3px 8px}", html)
            self.assertIn(".matrix-panel .matrix-help{padding-top:20px}", html)
            self.assertIn("w-70,h-12", html)
            self.assertIn("if(epochs.length===1)", html)
            self.assertIn("only one epoch measured", html)
            self.assertIn("w-70", html)
            self.assertIn(".rl-help{flex:0 0 66px;max-width:66px;min-width:66px", html)
            self.assertIn(".split-options{display:flex;flex-direction:column", html)
            self.assertIn("NLL (nats/example)", html)
            self.assertIn("Effective dimensions", html)
            self.assertIn("createLinearGradient", html)
            self.assertIn('right_ylabel:"Remaining R_C (···)"', html)
            self.assertIn('legendGroups:[', html)
            self.assertIn('name:"Train",color:"#2166ac"', html)
            self.assertIn('name:"Validation",color:"#b2182b"', html)
            self.assertIn('name:"Low-variance F_C"', html)
            self.assertIn('name:"Random F_C"', html)
            self.assertIn('legendSamplesBelow:true', html)
            self.assertIn('const slotWidth=(w-12)/Math.max(1,row.length)', html)
            self.assertIn('c.fillText(item.name,centre,labelY,Math.max(1,slotWidth-4))', html)
            self.assertIn('const labelY=24,sampleY=35', html)
            self.assertIn('top=opts.legendSamplesBelow?43', html)
            self.assertIn('legendGroups:[[', html)
            self.assertNotIn('name:"Remaining R_C"', html)
            self.assertNotIn('name:"Remaining R_C (right axis)"', html)
            self.assertIn('dash:[1,3]', html)
            self.assertIn('legendBelow:true', html)
            self.assertIn('axis:"right"', html)
            self.assertIn("c.setLineDash(s.dash||[])", html)
            self.assertIn(
                "Object.keys(trainFC).map(Number)",
                html,
            )
            self.assertIn('if(opts.xlabel==="Epoch")', html)
            self.assertIn("measure=>measure.training_iteration===selected().training_iteration", html)
            self.assertIn("c.setLineDash([4,3])", html)
            self.assertNotIn("markSnapshot", html)
            self.assertIn("const chartTop=top+4,chartBottom=top+ph-4", html)
            self.assertIn("chartBottom-chartTop", html)
            self.assertNotIn("const selected=run.accuracy.findIndex", html)
            self.assertIn("epochSlider", html)
            self.assertEqual(
                html.count('class="measure-panel" tabindex="0"')
                + html.count('class="measure-panel matrix-panel" tabindex="0"'),
                8,
            )
            self.assertIn("setPanelExpanded", html)
            self.assertIn(
                '?`, ${status.constant_unit_count} constant units`:"";',
                html,
            )
            self.assertIn('event.key==="Escape"', html)
            self.assertIn("grid.clientHeight", html)
            self.assertIn("position:absolute;left:0;z-index:5;width:100%", html)
            self.assertIn("const oneLine=!opts.legendBelow&&Boolean(opts.ylabel)", html)
            self.assertIn("const legendRows=[]", html)
            self.assertIn(
                "ylabelWidth+12+legendWidth",
                html,
            )
            self.assertIn('"run_details":["CNN","SGD","tanh","uniform"', html)
            self.assertIn("lr=0.001", html)
            self.assertNotIn("__DATA__", html)
            changed_epoch = json.loads(json.dumps(measure))
            changed_epoch["training_iteration"] = 2
            changed_epoch["train"]["reference_sample"][
                "ordered_indices_sha256"
            ] = "different-sample"
            (run_dir / "temporal_measurements.jsonl").write_text(
                json.dumps(measure) + "\n" + json.dumps(changed_epoch) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identity changed"):
                write_dashboard(run_dir)

            (run_dir / "temporal_measurements.jsonl").write_text(
                json.dumps(measure) + "\n", encoding="utf-8"
            )
            measure["metric_definition_version"] = 1
            (run_dir / "temporal_measurements.jsonl").write_text(
                json.dumps(measure) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "legacy records cannot be compared"
            ):
                write_dashboard(run_dir)

    def test_dashboard_accepts_and_labels_degenerate_reference_statuses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            train_identity = ordered_sample_identity(
                "train", [0, 1], "positions_in_fixed_training_subset"
            )
            validation_identity = ordered_sample_identity(
                "validation", [0, 1], "positions_in_fixed_validation_split"
            )
            metadata = {
                "run_id": "run-degenerate",
                "seed": 1,
                "requested_config": {
                    "architecture": "CNN",
                    "activation": "relu",
                    "initializer": "uniform",
                    "optimizer": "sgd",
                    "learning_rate": 0.001,
                    "dataset": {"train_fraction": 0.02},
                    "training": {"batch_size": 2, "epochs": 3},
                },
                "temporal_measurement_samples": {
                    "train_count": 2,
                    "train_positions_in_training_subset": [0, 1],
                    "train_dataset_indices": [0, 1],
                    "train_identity": train_identity,
                    "validation_count": 2,
                    "validation_positions": [0, 1],
                    "validation_split_indices": [0, 1],
                    "validation_identity": validation_identity,
                },
                "train_subset_indices": [0, 1],
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            records = []
            metrics = []
            for epoch, state in enumerate(
                ("all_constant", "single_nonconstant_unit", "rank_deficient"),
                start=1,
            ):
                split_records = {}
                for split, identity in (
                    ("train", train_identity),
                    ("validation", validation_identity),
                ):
                    status = state if split == "train" else "ok"
                    total_variance = 0.0 if status == "all_constant" else 1.0
                    rank = {
                        "all_constant": 0,
                        "single_nonconstant_unit": 1,
                        "rank_deficient": 1,
                        "ok": 20,
                    }[status]
                    nonconstant_count = {
                        "all_constant": 0,
                        "single_nonconstant_unit": 1,
                        "rank_deficient": 2,
                        "ok": 20,
                    }[status]
                    nullity = 20 - rank
                    fc = None if status == "all_constant" else {
                        str(k): {
                            "low_variance": 0.0 if k <= nullity else 0.2,
                            "random": [0.3] * 50,
                            "R_C_remaining_lowvar": 1.0 if k == 0 else 0.0,
                            "contains_zero_variance_directions": (
                                k > 0 and nullity > 0
                            ),
                            "zero_variance_directions_individually_identifiable": nullity == 0,
                            "nullspace_ablation_effect_on_reference_sample": (
                                "zero_to_numerical_precision"
                                if nullity > 0 else "not_applicable"
                            ),
                        }
                        for k in (0, 1, 19, 20)
                    }
                    split_records[split] = {
                        "R_L": None,
                        "R_C": None if rank == 0 else 1.0,
                        "F_L": [0.0] * 20,
                        "F_C": fc,
                        "reference_sample": identity,
                        "reference_activation_mean": [0.0] * 20,
                        "activation_covariance_eigenvalues": [0.0] * 20,
                        "representation_status": status,
                        "constant_unit_count": 20 - nonconstant_count,
                        "nonconstant_unit_count": nonconstant_count,
                        "numerical_rank": rank,
                        "nullity": nullity,
                        "total_activation_variance": total_variance,
                        "zero_variance_directions_individually_identifiable": nullity == 0,
                        "centered_nullspace_ablation_effect": (
                            "not_applicable" if rank == 0
                            else "zero_to_numerical_precision"
                        ),
                    }
                records.append({
                    "training_iteration": epoch,
                    "measurement_seconds": 0.1,
                    "metric_definition_version": 2,
                    "p_c_definition": "spectral_energy_participation_ratio_of_row_normalized_W_in",
                    "fc_intervention": "centered_pca_projection",
                    "activation_width": 20,
                    "P_L": [[0.0]],
                    "P_C": 1.0,
                    **split_records,
                })
                metrics.append({
                    "training_iteration": epoch,
                    "train_eval_loss": 1.0,
                    "train_eval_acc": 0.5,
                    "validation_eval_loss": 1.0,
                    "validation_eval_acc": 0.5,
                })
            (run_dir / "temporal_measurements.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (run_dir / "result.json").write_text(
                "".join(json.dumps(record) + "\n" for record in metrics),
                encoding="utf-8",
            )

            html = write_dashboard(run_dir).read_text(encoding="utf-8")
            for status in (
                "all_constant",
                "single_nonconstant_unit",
                "rank_deficient",
            ):
                self.assertIn(status, html)
            self.assertIn('zero_variance_directions_individually_identifiable', html)
            self.assertIn('PCA basis non-unique, centred removal has ~0 effect', html)
            self.assertIn('if(!leftValues.length&&!rightValues.length)', html)

    def test_experiment_dashboard_has_shared_epoch_slider_and_run_panels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_id in ("run-a", "run-b"):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "dashboard.html").write_text(
                    "<html></html>", encoding="utf-8"
                )
                (run_dir / "status.json").write_text(
                    json.dumps({"status": "completed"}), encoding="utf-8"
                )
                (run_dir / "run_metadata.json").write_text(
                    json.dumps({
                        "seed": 22,
                        "requested_config": {
                            "architecture": "CNN",
                            "optimizer": "adam",
                            "activation": "tanh",
                            "initializer": "uniform",
                            "learning_rate": 0.001,
                            "dataset": {"train_fraction": 0.02},
                            "training": {"epochs": 5, "batch_size": 128},
                        },
                    }),
                    encoding="utf-8",
                )
                measurements = [
                    {"training_iteration": epoch}
                    for epoch in (2, 4, 5)
                ]
                (run_dir / "temporal_measurements.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in measurements),
                    encoding="utf-8",
                )
                (run_dir / "result.json").write_text(
                    "".join(
                        json.dumps({
                            "training_iteration": epoch,
                            "train_eval_acc": 0.7 + epoch / 100,
                            "validation_eval_acc": 0.6 + epoch / 100,
                        }) + "\n"
                        for epoch in range(1, 6)
                    ),
                    encoding="utf-8",
                )

            output = write_experiment_dashboard(root)
            html = output.read_text(encoding="utf-8")

            self.assertIn('"epochs":[2,4,5]', html)
            self.assertIn('"run_id":"run-a"', html)
            self.assertIn('"run_id":"run-b"', html)
            self.assertIn("Shared epoch", html)
            self.assertIn("postMessage", html)
            self.assertIn("runs/run-a/dashboard.html", html)
            self.assertIn('"accuracy":[{"training_iteration":1', html)
            self.assertIn('"train_eval_acc":0.71', html)
            self.assertIn(
                "grid-template-columns:repeat(auto-fill,minmax(250px,1fr))",
                html,
            )
            self.assertIn("width:100%;margin:0 auto", html)
            self.assertIn("grid-auto-rows:170px", html)
            self.assertIn("function drawAccuracy", html)
            self.assertIn('className="accuracy-card"', html)
            self.assertIn('className="run-panel"', html)
            self.assertIn('classList.toggle("expanded")', html)
            self.assertIn("const tintHue=(runElements.length*137.508)%360", html)
            self.assertIn("hsl(${tintHue} 65% 88%)", html)
            self.assertIn("section.style.backgroundColor=tint", html)
            self.assertIn("frame.style.backgroundColor=tint", html)
            self.assertIn("grid-template-columns:repeat(auto-fill,minmax(250px,1fr))", html)
            self.assertIn("plotTop=top+13,plotBottom=h-bottom-13", html)
            self.assertIn('frame.dataset.dashboard=run.dashboard', html)
            self.assertIn('if(expanded&&!frame.dataset.loaded)', html)
            self.assertIn("point=>point.training_iteration===epoch", html)
            self.assertIn("c.setLineDash([4,3])", html)
            self.assertNotIn("Click to open detailed panel", html)
            self.assertNotIn("c.fillText(`Epoch ${epoch}`", html)
            self.assertIn(".run.expanded{grid-column:span 2;grid-row:span 4}", html)
            self.assertIn(".run.expanded .accuracy-chart{display:none}", html)
            self.assertIn(".run.expanded .run-panel{display:block}", html)
            self.assertIn("for(const item of runElements)", html)
            self.assertNotIn("<h1>Vue d’ensemble des runs</h1>", html)
            self.assertNotIn("scroll horizontally", html)
            self.assertIn('"details":["CNN","Adam","tanh","uniform"', html)
            self.assertIn("run-config", html)
            self.assertFalse((root / "dashboard-panels.html").exists())

    def test_experiment_dashboard_rejects_different_measurement_epochs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_id, epochs in (("run-a", [1, 2]), ("run-b", [1, 3])):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "dashboard.html").write_text(
                    "<html></html>", encoding="utf-8"
                )
                (run_dir / "status.json").write_text(
                    json.dumps({"status": "completed"}), encoding="utf-8"
                )
                (run_dir / "temporal_measurements.jsonl").write_text(
                    "".join(
                        json.dumps({"training_iteration": epoch}) + "\n"
                        for epoch in epochs
                    ),
                    encoding="utf-8",
                )
                (run_dir / "result.json").write_text(
                    "".join(
                        json.dumps({
                            "training_iteration": epoch,
                            "train_eval_acc": 0.5,
                            "validation_eval_acc": 0.4,
                        }) + "\n"
                        for epoch in epochs
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(
                ValueError, "different measurement epochs"
            ):
                write_experiment_dashboard(root)


if __name__ == "__main__":
    unittest.main()
