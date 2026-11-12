"""Explicit lead0 saved-atom reader; no bank generation or formal admission."""
from .formal_physical_inputs import FormalPhysicalBank, LEDGER_KEYS, _need
from .onset_planning import OnsetPlanningContext
from .validation.validate_s01_replay import network_arcs


class OnsetPhysicalBank(FormalPhysicalBank):
    """Original numeric source/stream/physical checks on onset's own H and state."""
    lead_h = 0.
    ledger_keys = LEDGER_KEYS | {'information_contract', 'forecast_dependency_sha256'}

    def _make_context(self, root):
        return OnsetPlanningContext(root, network_arcs(root))

    def _prepare_target(self, context, target, bank):
        return context.prepare_target(target, bank,
            basic_soc=bank['config']['basic_soc'], tau_blk_h=bank['config']['tau_blk_h'])

    def _verify_extra_ledger(self, ledger):
        _need(ledger['information_contract'] == 'OR_ONSET_SAME_KNOWN_FORECAST_v1'
            and ledger['forecast_dependency_sha256'] == self.prepared.tasks.dependency_sha256,
            'Onset information/forecast dependency differs')
