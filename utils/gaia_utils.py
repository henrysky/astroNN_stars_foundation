import io
import requests
import numpy as np
from zero_point import zpt
from astropy.io import votable
from astroquery.gaia import Gaia
from numpy.typing import NDArray
from astroNN.gaia import mag_to_absmag, fakemag_to_absmag

from stellarperceptron.model import StellarPerceptron

from mygaiadb.utils import radec_to_ecl
from gaiaxpy.core.config import _load_xpmerge_from_xml, _load_xpsampling_from_xml

zpt.load_tables()
xp_sampling_grid, xp_merge = _load_xpmerge_from_xml()
xp_design_matrices = _load_xpsampling_from_xml()


def mag_to_flux(mag, band="g"):
    """
    Reference:
    https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html
    https://www.cosmos.esa.int/web/gaia/dr3-passbands
    """
    band = band.lower()
    if band not in ["g", "bp", "rp", "rvs", "bad"]:
        raise ValueError("Unknown photometric band")
    else:
        if band == "g":
            factor = 25.6873668671 / 2.5
            factor_error = 0.0027553202 / 2.5
        elif band == "bp":
            factor = 25.3385422158 / 2.5
            factor_error = 0.0027901700 / 2.5
        elif band == "rp":
            factor = 24.7478955012 / 2.5
            factor_error = 0.0037793818 / 2.5
        elif band == "rvs":
            factor = 21.317 / 2.5
            factor_error = 0.002 / 2.5
        elif band == "bad":
            factor = 8.5
            factor_error = 0.0
        return 10 ** (factor - mag / 2.5)


def crop_xp(arr, num, mode="front"):
    """
    Get the first ``num`` of ceofficients in BP and RP, assuming you are giving this
    function a single array of 55 BP and 55 RP ceofficients

    The input array must be 110 elements wide and so ``num`` must be less than 110

    If the input array has size (N, 110) and num=M, the output array will have shape (M*2, 110)
    """
    assert num <= 110, "num cannot be larger than 110"
    assert arr.shape[1] == 110, "arr must be exactly 110 wide"
    if mode.lower() == "front":
        return np.hstack([arr[:, : num // 2], arr[:, 55 : 55 + num // 2]])
    elif mode.lower() == "end":
        return np.hstack([arr[:, 55 - (num // 2) : 55], arr[:, -num // 2 :]])
    else:
        raise NameError(f"Unknown Mode: {mode.lower()}")


def gaia_plx_zero_point(
    phot_g_mean_mag,
    nu_eff_used_in_astrometry,
    pseudocolour,
    ect_lat,
    astrometric_params_solved,
    leung2023_correction: bool=True,
) -> NDArray[np.float64]:
    parallax_zp = zpt.get_zpt(
        phot_g_mean_mag,
        nu_eff_used_in_astrometry,
        pseudocolour,
        ect_lat,
        astrometric_params_solved,
        _warnings=False,  # suppress warnings so that out of range values are set to nan
    )
    if leung2023_correction:
        leung2023_zp = 0.005 * np.clip(phot_g_mean_mag - 13.0, 0.0, 17.0 - 13.0)
        # parallax zero point correction is negative and overcorrected, so we add positive number
        parallax_zp += leung2023_zp
    return parallax_zp


def xp_spec_online(
    gdr3_source_id: int, absolute_flux: bool = False, return_info: bool = False
):
    """
    Look up the XP spectrum of a Gaia DR3 source ID online

    Parameters
    ----------
    gdr3_source_id: int
        Gaia DR3 Source ID
    absolute_flux: bool
        If True, return absolute flux instead of normalized flux
    return_info: bool
        If True, return a dictionary of information
    """
    assert isinstance(
        gdr3_source_id, int
    ), "gdr3_source_id must be an integer for Gaia DR3 Source ID"
    url = f"https://gea.esac.esa.int/data-server/data?ID=Gaia+DR3+{gdr3_source_id}&RETRIEVAL_TYPE=XP_CONTINUOUS"
    temp_data = requests.get(url).content  # download to memory
    votable_data = votable.parse(io.BytesIO(temp_data))
    gaia_xp_data = votable_data.get_first_table().array

    job = Gaia.launch_job(
        f"SELECT * from gaiadr3.gaia_source as G where G.source_id = {gdr3_source_id}"
    )
    sqlresult = job.get_results()[0]
    flux = mag_to_flux(sqlresult["phot_g_mean_mag"])

    ect_lon, ect_lat = radec_to_ecl(sqlresult["ra"], sqlresult["dec"])

    if absolute_flux:
        parallax = sqlresult["parallax"] - gaia_plx_zero_point(
            sqlresult["phot_g_mean_mag"],
            sqlresult["nu_eff_used_in_astrometry"],
            sqlresult["pseudocolour"],
            ect_lat,
            sqlresult["astrometric_params_solved"],
        )
        absmag = mag_to_absmag(
            sqlresult["phot_g_mean_mag"],
            parallax,
        )
        abs_flux = mag_to_flux(absmag)
    else:
        abs_flux = 1.0

    bprp_coeffs = {"bp": np.zeros(55), "rp": np.zeros(55)}
    bprp_coeffs["bp"][:] = gaia_xp_data["bp_coefficients"][0].data / flux * abs_flux
    bprp_coeffs["rp"][:] = gaia_xp_data["rp_coefficients"][0].data / flux * abs_flux

    if return_info:
        return bprp_coeffs, {
            "mag": sqlresult["phot_g_mean_mag"],
            "plx": parallax,
            "absmag": absmag,
        }
    else:
        return bprp_coeffs


def nn_xp_coeffs(
    nn_model: StellarPerceptron,
    absolute_flux: bool = False,
    return_df: bool = False,
    truncation: int = 55,
    **kwargs,
):
    """
    This function turns stellar parameters into physical XP spectra

    Parameters
    ----------
    nn_model: StellarPerceptron
        The neural network model to use
    num1 : int
        First number to add.
    **kwargs
        Additional keyword arguments passed to `otherApi`

    Returns
    -------
    int
        The sum of ``num1`` and ``num2``.

    """
    bprp_coeffs = {"bp": np.zeros(55), "rp": np.zeros(55)}
    nn_model.perceive(
        np.array([[kwargs[i] for i in kwargs.keys()]]), [list(kwargs.keys())]
    )
    pred_df = nn_model.request(
        [
            *[f"bp{i}" for i in range(truncation)],
            *[f"rp{i}" for i in range(truncation)],
            "g_fakemag",
            "teff",
            "logg",
            "m_h",
        ]
    )
    nn_model.clear_perception()
    bprp_coeffs["bp"][:truncation] = pred_df[
        [f"bp{i}" for i in range(truncation)]
    ].to_numpy()
    bprp_coeffs["rp"][:truncation] = pred_df[
        [f"rp{i}" for i in range(truncation)]
    ].to_numpy()

    if absolute_flux:
        g_absmag = fakemag_to_absmag(pred_df["g_fakemag"].to_numpy())
        bprp_coeffs["bp"] *= mag_to_flux(g_absmag)
        bprp_coeffs["rp"] *= mag_to_flux(g_absmag)
    if return_df:
        return bprp_coeffs, pred_df
    else:
        return bprp_coeffs


def xp_coeffs_phys(bprp_coeffs: dict) -> NDArray[np.float64]:
    """
    Turn the coefficients into physical spectra
    """
    xp_design_matrices = _load_xpsampling_from_xml()
    _, xp_merge = _load_xpmerge_from_xml()
    bp_spec = bprp_coeffs["bp"].dot(xp_design_matrices["bp"])
    rp_spec = bprp_coeffs["rp"].dot(xp_design_matrices["rp"])
    return np.add(
        np.multiply(bp_spec, xp_merge["bp"]), np.multiply(rp_spec, xp_merge["rp"])
    )


def nn_xp_coeffs_phys(nn_model: StellarPerceptron, return_df: bool=False, **kwargs) -> NDArray[np.float64]:
    """
    Get physical spectra with NN model
    """
    if return_df:
        bprp_coeffs, pred_df = nn_xp_coeffs(nn_model, return_df=return_df, **kwargs)
        spectrum = xp_coeffs_phys(bprp_coeffs)
        return spectrum, pred_df
    else:
        bprp_coeffs = nn_xp_coeffs(nn_model, return_df=return_df, **kwargs)
        spectrum = xp_coeffs_phys(bprp_coeffs)
        return spectrum
