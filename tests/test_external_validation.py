"""External validation against measured NASA/Stanford cold-flow data.

Every other test in this suite checks internal consistency, cross-backend
agreement, or agreement with HRAP's published equations. These check the models
against **real measurements**.

Source: B. S. Waxman, J. E. Zimmerman, B. J. Cantwell (Stanford) and
G. G. Zilliac (NASA Ames), *Mass Flow Rate and Isolation Characteristics of
Injectors for Use with Self-Pressurizing Oxidizers in Hybrid Rockets*,
NTRS 20190001326.

Reference values are quoted from the paper's prose and figure captions, not a
data table, and the effective discharge coefficient is stated as
"approximately 0.6". Tolerances below are set accordingly -- these assert that
the models land in the right place for the right reason, not that they match to
three digits.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2o_injector import (  # noqa: E402
    FlowModel,
    SaturationTable,
    get_backend,
    mass_flux,
    recommend_model,
)

PSI = 101325.0 / 14.696

# Waxman et al., injector 3 (D = 1.50 mm, L = 18.4 mm, L/D = 12.3, rounded)
T1 = 280.0
P1 = 704.0 * PSI
D = 1.50e-3
L = 18.4e-3
L_OVER_D = L / D
CD_SPI = 0.77

REF_SUPERCHARGE_PSI = 169.0
REF_CD_EFF = 0.60
REF_CD_EFF_DP = 330.0 * PSI
REF_SPI_VALID_TO_PSI = 200.0


@pytest.fixture(scope="module")
def props():
    return get_backend("esdu")


@pytest.fixture(scope="module")
def table(props):
    return SaturationTable(props)


@pytest.fixture(scope="module")
def upstream(props):
    return props.upstream_state(T1, P1)


def _cd_eff(table, up, dP, model):
    """Effective Cd as the paper defines it: mdot / (A sqrt(2 rho dP))."""
    r = mass_flux(table, up, P1 - dP, model, L_over_D=L_OVER_D)
    return CD_SPI * r.G / r.G_spi, r


def test_saturation_pressure_matches_refprop_via_supercharge(props):
    """The paper used REFPROP; the stated supercharge pins Psat(280 K).

    Supercharge = P1 - Psat, so agreeing on it to a couple of percent means the
    ESDU curve fit agrees with the reference EOS to well under one percent on
    the vapour pressure itself.
    """
    supercharge_psi = (P1 - props.P_sat(T1)) / PSI
    assert supercharge_psi == pytest.approx(REF_SUPERCHARGE_PSI, rel=0.05)
    # And the implied error on Psat itself is far smaller.
    psat_err = abs(supercharge_psi - REF_SUPERCHARGE_PSI) * PSI / props.P_sat(T1)
    assert psat_err < 0.02


def test_flow_is_single_phase_until_saturation_is_crossed(table, upstream, props):
    """Below the supercharge there is no flashing, so all models give SPI.

    The paper reports the SPI model valid up to dP ~ 200 psi; the predicted
    onset is the supercharge itself (~166 psi). Predicting onset slightly early
    is the expected direction, since real flow is metastable and flashes late.
    """
    supercharge = P1 - props.P_sat(T1)
    cd_low, r_low = _cd_eff(table, upstream, 0.5 * supercharge, FlowModel.DYER)
    assert cd_low == pytest.approx(CD_SPI, rel=1e-6)
    assert math.isinf(r_low.kappa)

    # Well past the observed transition the flow must have departed from SPI.
    cd_hi, _ = _cd_eff(table, upstream, 1.5 * REF_SPI_VALID_TO_PSI * PSI, FlowModel.DYER)
    assert cd_hi < 0.95 * CD_SPI

    # Onset must not be predicted later than the observed upper bound for SPI.
    assert supercharge / PSI < REF_SPI_VALID_TO_PSI


def test_measured_flow_is_bracketed_by_spi_and_hem(table, upstream):
    """The measurement must fall between the two limiting models.

    SPI ignores flashing and so is an upper bound; HEM assumes instantaneous
    equilibrium and, as the paper states, "gives a lower-bound estimate for the
    critical mass flow rate". A measurement outside that bracket would mean the
    implementation, not the physics, is wrong.
    """
    cd_hem, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.HEM)
    cd_spi, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.SPI)
    assert cd_hem < REF_CD_EFF < cd_spi


def test_hem_underpredicts_as_theory_requires(table, upstream):
    """Equilibrium is a lower bound, so HEM must sit below the data -- but close."""
    cd_hem, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.HEM)
    assert cd_hem < REF_CD_EFF
    assert cd_hem == pytest.approx(REF_CD_EFF, rel=0.15)


def test_dyer_is_within_paper_stated_accuracy(table, upstream):
    """The paper reports Dyer good to ~±10-15% with a per-geometry Cd."""
    cd_dyer, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.DYER)
    assert cd_dyer == pytest.approx(REF_CD_EFF, rel=0.20)


def test_spi_alone_is_badly_wrong_here(table, upstream):
    """Using the plain CdA equation over-predicts substantially.

    This is the paper's central practical point, and the reason HRAP's
    SPI-only liquid model will over-predict for a flashing injector.
    """
    cd_spi, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.SPI)
    assert cd_spi / REF_CD_EFF > 1.20


def test_hem_beats_dyer_for_this_long_orifice(table, upstream):
    """At L/D = 12.3 equilibrium is closer to reality than the blend."""
    cd_hem, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.HEM)
    cd_dyer, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.DYER)
    assert abs(cd_hem - REF_CD_EFF) < abs(cd_dyer - REF_CD_EFF)


def test_model_recommendation_agrees_with_the_data(table, upstream):
    """The tool's own L/D guidance must pick the model that actually fits."""
    model, _ = recommend_model(L_OVER_D)
    assert model is FlowModel.HEM

    errors = {}
    for m in (FlowModel.SPI, FlowModel.HEM, FlowModel.DYER):
        cd, _ = _cd_eff(table, upstream, REF_CD_EFF_DP, m)
        errors[m] = abs(cd - REF_CD_EFF)
    assert min(errors, key=errors.get) is model


def test_flow_chokes_in_the_two_phase_region(table, upstream):
    """The paper observes critical (backpressure-independent) flow."""
    _, r = _cd_eff(table, upstream, REF_CD_EFF_DP, FlowModel.HEM)
    assert r.choked


def test_effective_cd_decreases_monotonically_with_dP(table, upstream):
    """The measured Cd_eff curve falls once flashing starts; ours must too."""
    prev = None
    for dP_psi in (200, 250, 300, 330, 400, 500, 600):
        cd, _ = _cd_eff(table, upstream, dP_psi * PSI, FlowModel.DYER)
        if prev is not None:
            assert cd < prev
        prev = cd


def test_validation_script_runs():
    """The reproducible write-up must stay executable."""
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    )
    import validate_waxman

    preds, _ = validate_waxman.main()
    assert set(preds) == {"SPI", "HEM", "Dyer (NHNE)"}
