
import pytest

from src.immune.risk_signals import RiskSignals
from src.immune.immune_input import to_immune_input


def make_signals():
    return RiskSignals(
        relation_risks={
            "odor": 3,
            "contamination": 1,
            "physical": 0,
        },
        final_risk=3,
        hard_forbidden=False,
        constraint_status="unknown",
        relation_probabilities={
            "odor": [0.1, 0.1, 0.2, 0.6],
            "contamination": [0.7, 0.2, 0.1, 0.0],
            "physical": [0.8, 0.1, 0.1, 0.0],
        },
        relation_embeddings={
            "odor": [0.2, 0.3],
            "contamination": [0.4, 0.5],
            "physical": [0.6, 0.7],
        },
        uncertainty=0.2,
        context_delta={"leak_risk_delta": 0.1},
    )


def test_oracle_labels_not_exposed():
    result = to_immune_input(make_signals())

    assert "relation_risks" not in result
    assert "final_risk" not in result


def test_prediction_inputs_preserved():
    result = to_immune_input(make_signals())

    assert result["relation_probabilities"]["odor"] == [
        0.1, 0.1, 0.2, 0.6
    ]
    assert result["relation_embeddings"]["physical"] == [
        0.6, 0.7
    ]
    assert result["context_delta"]["leak_risk_delta"] == 0.1


def test_missing_gat_predictions_rejected():
    signals = make_signals()
    signals.relation_probabilities = {}

    with pytest.raises(ValueError):
        to_immune_input(signals)


def test_missing_embeddings_rejected():
    signals = make_signals()
    signals.relation_embeddings = {}

    with pytest.raises(ValueError):
        to_immune_input(signals)


def test_invalid_constraint_status_rejected():
    signals = make_signals()
    signals.constraint_status = "allowed"

    with pytest.raises(ValueError):
        to_immune_input(signals)
