# McKinsey & Company Data Scientist, Climate Analytics Coding Sample

**Riley X. Brady**

**July 27th, 2020**

## Data

The data used in this sample is available freely through the [Earth System Grid Federation](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.CESM_CAM5_BGC_LE.ocn.proc.monthly_ave.html?df=true). You can also download the files used here directly from [Dropbox](https://drive.google.com/drive/folders/1YkPna1nKNyrmq8ByBIwkoWz1iBotZjC9?usp=sharing). Make sure to move them into the `data` folder.

## Run

First install the conda environment:

```bash
conda env update -f environment.yml
```

Then run

```bash
conda activate mckinsey-position-brady
python submission.py
```

## Background

The sample code demos a simplified analysis from my [2019 paper](https://bg.copernicus.org/articles/16/329/2019/) in _Biogeosciences_ on the response of coastal air-sea CO2 fluxes to modes of climate variability. Here, I perform a Linear Taylor Expansion on the ocean pCO2 response to the North Pacific Gyre Oscillation (NPGO), a prominent mode of climate variability that influences the California Current. The result of the analysis is a quantification of how variability in driver variables of pCO2 (temperature, salinity, dissolved inorganic carbon, and alkalinity) cause the response of pCO2 to the NPGO. Since it is a linear expansion, it is not a perfect approximation and negates cross-derivative terms. See, e.g., Figure 6 from the linked paper for a final result with the air-sea CO2 flux (which is influenced largely by pCO2).

## Contact

riley.brady@colorado.edu

https://www.rileyxbrady.com
