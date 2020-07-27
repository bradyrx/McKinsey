import cftime
import geopandas as gpd
import numpy as np
import numpy.polynomial.polynomial as poly
import regionmask
import xarray as xr
from dask.diagnostics import ProgressBar
from eofs.xarray import Eof


def calculate_anomaly(ds):
    """Calculates anomaly by removing the linear trend and monthly climatology from a
    dataset.

    Args:
        ds (xarray object): Dataset or DataArray to generate anomalies from.

    Returns:
        xarray object: Dataset or DataArray with linear trend and seasonal cycle
        removed.
    """
    ds = detrend(ds)
    # Removes monthly climatology.
    gb = ds.groupby("time.month")
    clim = gb.mean(dim="time")
    return gb - clim


def compute_npgo_index():
    """Computes the North Pacific Gyre Oscillation (NPGO) index from SST output.

    References:
        * Di Lorenzo, E. and N. Mantua, 2016: Multi-year persistence of the 2014/15
          North Pacific marine heatwave. Nature Climate Change, 6(11) 1042-+,
          doi:10.1038/nclimate3082.
    """
    sst = load_single_variable("SST")
    area = xr.open_dataarray("data/area.nc")
    # Extract North Pacific region for Principal Component Analysis (PCA).
    sst = extract_region(sst, sst.TLONG, sst.TLAT, [180, 250, 25, 62])
    area = extract_region(area, area.TLONG, area.TLAT, [180, 250, 25, 62])
    sst_anom = calculate_anomaly(sst)
    month = sst_anom["time.month"]
    # Calculate PCA over the winter (JFM) annual means.
    JFM = month <= 3
    # EOF package requires `time` to be the leading dimension.
    sst_anom_winter = (
        sst_anom.where(JFM).resample(time="A").mean("time").transpose("time", ...)
    )
    # Compute PCA (EOF) over the annual winter averages, weighted by the grid cell area.
    solver = Eof(sst_anom_winter, weights=area, center=False)
    # Reconstruct the monthly index of SSTa by projecting the SSTa values onto the
    # annual principal component time series.
    pseudo_pc = solver.projectField(
        sst_anom.transpose("time", ...), neofs=2, eofscaling=1
    )
    # Mode 0 is the Pacific Decadal Oscillation. We're interested in its orthogonal
    # mode, which is the NPGO. The NPGO still explains ~20% of the variance of the
    # North Pacific.
    return pseudo_pc.isel(mode=1)


def compute_sensitivity_terms(ds):
    """Calculate sensitivity terms for Linear Taylor Expansion.

    .. note::
        These terms either originate through laboratory experiments (such as SST
        having a sensitivity of ~4% of the standing pCO2 stock) or through
        differentiation of the model equations.

    Args:
        ds (Dataset): Dataset containing:
            * DIC (dissolved inorganic carbon)
            * ALK (dissolved alkalinity)
            * SALT (salinity)
            * SST (sea surface temperature)
            * pCO2SURF (surface pCO2)

    Returns:
        Dataset: Sensitivity values for each of the component variables to variations
        in pCO2.
    """
    DIC, ALK, SALT, pCO2 = ds["DIC"], ds["ALK"], ds["SALT"], ds["pCO2SURF"]

    # The "buffer factor" or "revelle factor" describes the ocean's buffer capacity
    # for the carbonate system. It depends on a complex relationship between alkalinity
    # and dissolved inorganic carbon stocks.
    buffer_factor = dict()
    buffer_factor["ALK"] = -(ALK ** 2) / ((2 * DIC - ALK) * (ALK - DIC))
    buffer_factor["DIC"] = (3 * ALK * DIC - 2 * DIC ** 2) / (
        (2 * DIC - ALK) * (ALK - DIC)
    )

    # These sensitivities result from laboratory experiments or differentiation of
    # model equations with respect to pCO2.
    sensitivity = dict()
    sensitivity["SST"] = 0.0423
    sensitivity["SALT"] = 1 / SALT
    sensitivity["ALK"] = (1 / ALK) * buffer_factor["ALK"]
    sensitivity["DIC"] = (1 / DIC) * buffer_factor["DIC"]
    sensitivity = xr.Dataset(sensitivity) * pCO2
    return sensitivity


def detrend(ds, dim="time"):
    """Removes a linear trend over a specified dimension from a Dataset or DataArray.

    Args:
        ds (xarray object): Dataset or DataArray to remove linear trend from.
        dim (str, optional): Dimension to remove linear trend over.

    Returns:
        xarray object: Dataset or DataArray with linear trend removed over `dim`.
    """

    def _detrend(x, y):
        coefs = poly.polyfit(x, y, 1)
        linear_fit = poly.polyval(x, coefs)
        return y - linear_fit

    # I convert from `cftime` to numeric time for the numpy stats equations to work
    # and to ensure that we account for the difference in lengths of months.
    x = return_numeric_time(ds)

    # This allows for dask arrays and vectorizes the operation across the model grid.
    return xr.apply_ufunc(
        _detrend,
        x,
        ds,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )


def extract_region(ds, xgrid, ygrid, coords, lat_dim="nlat", lon_dim="nlon"):
    """Extracts a region from a curvilinear mesh given the coordinates of the corners.

    Args:
        ds (xarray object): Dataset or DataArray from which to extract the sub-region.
        xgrid, ygrid (DataArray): Longitude and latitude from the curvilinear mesh.
        coords (list): x0, x1, y0, y1 for the corners of the sub-region.
        lat_dim, lon_dim (str, optional): Names of the latitude and longitude
            dimensions. Defaults to `nlat` and `nlon` for the POP ocean grid.

    Returns:
        xarray object: Dataset or DataArray subset over the region of interest.
    """
    x0, x1, y0, y1 = coords
    a, c = find_indices(xgrid, ygrid, x0, y0)
    b, d = find_indices(xgrid, ygrid, x1, y1)
    subset_data = ds.isel({lat_dim: slice(a, b + 1), lon_dim: slice(c, d + 1)})
    return subset_data


def find_indices(xgrid, ygrid, xpoint, ypoint):
    """Find the closest i, j indices of a singular point on a curvilinear mesh.

    Args:
        xgrid, ygrid (DataArray): Longitude and latitude from the curvilinear mesh.
        xpoint, ypoint (int, float): Longitude and latitude coordinate to locate on
            the mesh.

    Returns:
        int: i and j indices (zero-based) of the point on the grid closest to
            `xpoint` and `ypoint`.
    """
    dx = xgrid - xpoint
    dy = ygrid - ypoint
    reduced_grid = abs(dx) + abs(dy)
    min_ix = np.nanargmin(reduced_grid)
    i, j = np.unravel_index(min_ix, reduced_grid.shape)
    return i, j


def linear_slope(x, y, dim="time"):
    """Return the linear slope with x predicting y.

    Args:
        x, y (xarray object): Predictor (x) and predictand (y) for the linear
            regression.
        dim (str, optional): Name of time dimension. Defaults to "time".

    Returns:
        xarray object: DataArray or Dataset of linear slopes from regression of y
            onto x.
    """
    # This allows for dask arrays and vectorizes the operation across the model grid.
    return xr.apply_ufunc(
        lambda x, y: np.polyfit(x, y, 1)[0],
        x,
        y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )


def load_decomposition_vars():
    """Loads in the necessary variables to compute the Linear Taylor Expansion for pCO2.

    Returns:
        Dataset: xarray Dataset containing surface pCO2, alkalinity, dissolved
            inorganic carbon, sea surface temperature, and sea surface salinity.
    """
    varlist = ["pCO2SURF", "ALK", "DIC", "SST", "SALT"]
    ds = xr.Dataset()
    for var in varlist:
        single_var = load_single_variable(var)
        ds[var] = single_var
    return ds


def load_single_variable(varname):
    """Loads a single variable from the `data/` directory.

    Args:
        varname (str): Name of the variable to load in.

    Returns:
        DataArray: N-D array of the desired variable in `xarray` format.
    """
    filename = f"data/b.e11.B20TRC5CNBDRD.f09_g16.009.pop.h.{varname}.192001-200512.nc"
    ds = xr.open_dataset(filename).squeeze()
    ds = ds[varname].drop_vars(["ULAT", "ULONG",])
    # There's an issue with `xarray` interpreting the POP ocean grid time, due to issues
    # with their NetCDF metadata. This is just a manual fix.
    ds["time"] = xr.cftime_range(
        start="1920-01", periods=ds["time"].size, freq="MS", calendar="noleap"
    )
    return ds


def mask_california_current(ds):
    """Return a dataset with a mask applied over the California Current Large
    Marine Ecosystem.

    Args:
        ds (xarray object): Dataset or DataArray to mask over the California Current.

    Returns:
        xarray object: Dataset or DataArray with the California Current Large Marine
            Ecosystem mask applied, with all-nan rows and columns dropped.
    """
    # There are many Large Marine Ecosystems spanning continental coastlines around
    # the world. Here, we select the California Current.
    shpfile = gpd.read_file("data/shpfiles/LMEs66.shp")
    polygon = shpfile[shpfile["LME_NAME"] == "California Current"]
    polygon = polygon.geometry[:].unary_union
    CalCS = regionmask.Regions([polygon])
    mask = CalCS.mask(ds.TLONG, ds.TLAT)
    masked = ds.where(mask == 0, drop=True)
    return masked


def return_numeric_time(ds, dim="time"):
    """Returns a numeric time version of `cftime` from an xarray object.

    Args:
        ds (xarray object): Dataset or DataArray from which to convert time axis.
        dim (str, optional): Name of time dimension. Defaults to "time".

    Returns
        DataArray: Time index converted to numeric time to make stats functions
            work and to account for differences in lengths of e.g., months.
    """
    # Days since 1990-01-01 is arbitrary, but creates a consistent numeric x axis
    # for regressions that accounts for differences in the lengths of months.
    x = cftime.date2num(ds[dim], "days since 1990-01-01", calendar="noleap")
    x = xr.DataArray(x, dims=dim, coords=[ds[dim]])
    return x


if __name__ == "__main__":
    ds = load_decomposition_vars()
    # Chunk for out-of-memory and parallel operations.
    ds = ds.chunk({"time": -1, "nlat": "auto", "nlon": "auto"})
    ds = mask_california_current(ds)
    area = xr.open_dataset("data/area.nc")["TAREA"]
    area = mask_california_current(area)
    # Area-weight over the California Current for a single time series view of
    # the carbonate system's response to the NPGO.
    ds = (ds * area).sum(["nlat", "nlon"]) / area.sum(["nlat", "nlon"])
    with ProgressBar():
        ds = ds.compute()
    # Calculate the sensitivity of each driver variable to variations in pCO2 that
    # result from the NPGO.
    pco2_sensitivity = compute_sensitivity_terms(ds)
    ds_anom = calculate_anomaly(ds)
    npgo = compute_npgo_index()
    # Calculate the regression terms between the NPGO index and driver variable
    # anomalies.
    slopes = linear_slope(npgo, ds_anom)
    # Compute the Linear Taylor Expansion for each term. This shows how each term
    # contributes toward the modeled pCO2 anomaly. The approximation is linear and
    # does not account for cross-derivative terms.
    result = slopes * pco2_sensitivity.mean()
    approx = result.to_array().sum()
    # The direct linear regression of ocean pCO2 to the NPGO. This gives us a sense
    # of how close our approximation is.
    direct = linear_slope(npgo, ds_anom["pCO2SURF"])
    print("Response of California Current pCO2 to 1σ NPGO")
    print("=====================================================")
    print(f"Approximate Response of pCO2:  {np.round(approx.values, 3)} uatm")
    print(f"Direct Response of pCO2:  {np.round(direct.values, 3)} uatm")
    print("=====================================================")
    print(f'Alkalinity: {np.round(result["ALK"].values, 3)} uatm')
    print(f'Dissolved Inorganic Carbon: {np.round(result["DIC"].values, 3)} uatm')
    print(f'Sea Surface Temperature: {np.round(result["SST"].values, 3)} uatm')
    print(f'Sea Surface Salinity: {np.round(result["SALT"].values, 3)} uatm')
