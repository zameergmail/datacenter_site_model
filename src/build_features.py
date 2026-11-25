"""Build a ZIP (ZCTA) level feature table for data-center site suitability.

Usage:  python3 -I src/build_features.py <project_dir>
Reads   <project_dir>/data/raw/*
Writes  <project_dir>/data/processed/zip_features.csv
        <project_dir>/data/processed/datacenters_combined.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
R_KM = 6371.0088

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}
# NOAA climdiv uses its own alphabetical state codes (01-48 = CONUS, 50 = AK)
NOAA_STATES = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
               "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
NOAA_TO_ABBR = {f"{i + 1:02d}": s for i, s in enumerate(NOAA_STATES)} | {"50": "AK"}
ABBR_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}
HYPERSCALERS = ["google", "microsoft", "amazon", "aws", "meta", "facebook", "apple", "oracle",
                "qts", "digital realty", "equinix", "coresite", "cyrusone", "vantage", "stack",
                "aligned", "switch", "compass", "edgecore", "cologix", "flexential", "iron mountain"]


def to_rad(lat, lon):
    return np.deg2rad(np.c_[np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])


# ---------------------------------------------------------------- ZIP base
z = pd.read_csv(RAW / "2023_Gaz_zcta_national.txt", sep="\t", dtype={"GEOID": str})
z.columns = z.columns.str.strip()
z = z.rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon",
                      "ALAND_SQMI": "land_sqmi", "AWATER_SQMI": "water_sqmi"})
z = z[["zip", "lat", "lon", "land_sqmi", "water_sqmi"]]

# primary county = county with the largest land overlap
rel = pd.read_csv(RAW / "zcta_county_rel_2020.txt", sep="|", dtype=str, encoding="utf-8-sig")
rel = rel.dropna(subset=["GEOID_ZCTA5_20"])
rel["AREALAND_PART"] = rel["AREALAND_PART"].astype(float)
rel = rel.sort_values("AREALAND_PART", ascending=False).drop_duplicates("GEOID_ZCTA5_20")
rel = rel.rename(columns={"GEOID_ZCTA5_20": "zip", "GEOID_COUNTY_20": "county_fips",
                          "NAMELSAD_COUNTY_20": "county_name"})[["zip", "county_fips", "county_name"]]
z = z.merge(rel, on="zip", how="left")
z["state"] = z["county_fips"].str[:2].map(STATE_FIPS)

# ---------------------------------------------------------------- county population
pop = pd.read_csv(RAW / "census_county_pop_2024.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
pop = pop[pop.SUMLEV == 50]
pop["county_fips"] = pop.STATE + pop.COUNTY
pop["county_pop_2024"] = pop.POPESTIMATE2024
pop["county_pop_growth_20_24"] = pop.POPESTIMATE2024 / pop.POPESTIMATE2020 - 1
cg = pd.read_csv(RAW / "2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str})
cg.columns = cg.columns.str.strip()
pop = pop.merge(cg[["GEOID", "ALAND_SQMI"]], left_on="county_fips", right_on="GEOID", how="left")
pop["county_pop_density"] = pop.county_pop_2024 / pop.ALAND_SQMI
z = z.merge(pop[["county_fips", "county_pop_2024", "county_pop_growth_20_24", "county_pop_density"]],
            on="county_fips", how="left")

# ---------------------------------------------------------------- electricity price (state, latest year)
# EIA Electric Power Monthly table 5.6.B: year-to-date average price by state
NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
pr = pd.read_excel(RAW / "eia_epm_table_5_06_b.xlsx", header=None, skiprows=4)
pr["state"] = pr[0].astype(str).str.strip().map(NAME_TO_ABBR)
pr = pr.dropna(subset=["state"])[["state", 5, 3]]
pr.columns = ["state", "elec_price_industrial_c_kwh", "elec_price_commercial_c_kwh"]
pr = pr.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name != "state" else s)
z = z.merge(pr, on="state", how="left")

# ---------------------------------------------------------------- power plants (EIA-860)
plants = pd.read_excel(RAW / "eia860/2___Plant_Y2025.xlsx", header=1)
gens = pd.read_excel(RAW / "eia860/3_1_Generator_Y2025.xlsx", header=1)
gens = gens[gens.Status.astype(str).str.upper().eq("OP")]
clean = gens["Energy Source 1"].isin(["SUN", "WND", "WAT", "GEO", "NUC"])
cap = gens.assign(clean_mw=np.where(clean, gens["Nameplate Capacity (MW)"], 0)).groupby("Plant Code").agg(
    mw=("Nameplate Capacity (MW)", "sum"), clean_mw=("clean_mw", "sum"))
plants = plants.merge(cap, left_on="Plant Code", right_index=True)
plants = plants.dropna(subset=["Latitude", "Longitude"])
plants["kv"] = pd.to_numeric(plants["Grid Voltage (kV)"], errors="coerce")
plants["gas_pipeline"] = plants["Natural Gas Pipeline Name 1"].notna()

zr = to_rad(z.lat, z.lon)
pt = BallTree(to_rad(plants.Latitude, plants.Longitude), metric="haversine")
for km in (50, 100):
    idx = pt.query_radius(zr, r=km / R_KM)
    z[f"gen_mw_{km}km"] = [plants.mw.values[i].sum() for i in idx]
    z[f"clean_mw_{km}km"] = [plants.clean_mw.values[i].sum() for i in idx]
idx50 = pt.query_radius(zr, r=50 / R_KM)
z["n_plants_50km"] = [len(i) for i in idx50]
z["max_grid_kv_50km"] = [np.nanmax(plants.kv.values[i]) if len(i) and np.isfinite(plants.kv.values[i]).any()
                         else 0 for i in idx50]
z["clean_share_100km"] = (z.clean_mw_100km / z.gen_mw_100km).fillna(0)
big = plants[plants.mw >= 500]
d, _ = BallTree(to_rad(big.Latitude, big.Longitude), metric="haversine").query(zr, k=1)
z["dist_500mw_plant_km"] = d[:, 0] * R_KM
gp = plants[plants.gas_pipeline]
d, _ = BallTree(to_rad(gp.Latitude, gp.Longitude), metric="haversine").query(zr, k=1)
z["dist_gas_pipeline_plant_km"] = d[:, 0] * R_KM

# ---------------------------------------------------------------- connectivity + data-center locations
fac = pd.DataFrame(json.load(open(RAW / "peeringdb_fac.json"))["data"])
fac = fac.dropna(subset=["latitude", "longitude"])
ixfac = pd.DataFrame(json.load(open(RAW / "peeringdb_ixfac.json"))["data"])
fac["hosts_ix"] = fac.id.isin(ixfac.fac_id)

osm = json.load(open(RAW / "osm_datacenters.json"))["elements"]
rows = []
for e in osm:
    c = e.get("center", e)
    if "lat" not in c:
        continue
    t = e.get("tags", {})
    rows.append({"source": "osm", "name": t.get("name"), "operator": t.get("operator"),
                 "lat": c["lat"], "lon": c["lon"]})
osm = pd.DataFrame(rows)
dcs = pd.concat([
    osm,
    pd.DataFrame({"source": "peeringdb", "name": fac.name, "operator": fac.org_name,
                  "lat": fac.latitude, "lon": fac.longitude}),
], ignore_index=True)
# de-duplicate points within ~300 m of each other
dcs["k"] = (dcs.lat.round(3).astype(str) + "," + dcs.lon.round(3).astype(str))
dcs = dcs.drop_duplicates("k").drop(columns="k").reset_index(drop=True)
op = (dcs.operator.fillna("") + " " + dcs.name.fillna("")).str.lower()
dcs["major_operator"] = op.apply(lambda s: any(h in s for h in HYPERSCALERS))
dcs.to_csv(OUT / "datacenters_combined.csv", index=False)

ft = BallTree(to_rad(fac.latitude, fac.longitude), metric="haversine")
d, _ = ft.query(zr, k=1)
z["dist_peeringdb_fac_km"] = d[:, 0] * R_KM
ix = fac[fac.hosts_ix]
d, _ = BallTree(to_rad(ix.latitude, ix.longitude), metric="haversine").query(zr, k=1)
z["dist_ix_km"] = d[:, 0] * R_KM
idx = ft.query_radius(zr, r=50 / R_KM)
z["networks_50km"] = [fac.net_count.values[i].sum() for i in idx]

dt = BallTree(to_rad(dcs.lat, dcs.lon), metric="haversine")
for km in (10, 25):
    z[f"dc_count_{km}km"] = [len(i) for i in dt.query_radius(zr, r=km / R_KM)]
mj = dcs[dcs.major_operator]
z["major_dc_count_25km"] = [len(i) for i in
                            BallTree(to_rad(mj.lat, mj.lon), metric="haversine").query_radius(zr, r=25 / R_KM)]
d, _ = dt.query(zr, k=1)
z["dist_nearest_dc_km"] = d[:, 0] * R_KM

# ---------------------------------------------------------------- climate (NOAA climdiv, county, 1995-2024 mean)
def climdiv(path, name, agg):
    w = [(0, 2), (2, 5), (5, 7), (7, 11)] + [(11 + 7 * i, 18 + 7 * i) for i in range(12)]
    df = pd.read_fwf(path, colspecs=w, header=None, dtype={0: str, 1: str})
    df.columns = ["st", "cty", "el", "year"] + list(range(1, 13))
    df = df[(df.year >= 1995) & (df.year <= 2024)]
    m = df[list(range(1, 13))].replace(-9999.0, np.nan).replace(-99.99, np.nan)
    df[name] = m.sum(axis=1, min_count=12) if agg == "sum" else m.mean(axis=1)
    df["county_fips"] = df.st.map(NOAA_TO_ABBR).map(ABBR_TO_FIPS) + df.cty
    return df.groupby("county_fips")[name].mean()

clim = pd.concat([
    climdiv(RAW / "noaa_cddccy.txt", "cooling_degree_days", "sum"),
    climdiv(RAW / "noaa_tmaxcy.txt", "avg_tmax_f", "mean"),
    climdiv(RAW / "noaa_pcpncy.txt", "annual_precip_in", "sum"),
    climdiv(RAW / "noaa_pdsicy.txt", "drought_pdsi", "mean"),
], axis=1)
z = z.merge(clim, left_on="county_fips", right_index=True, how="left")

# ---------------------------------------------------------------- target proxies
z["has_dc_10km"] = (z.dc_count_10km > 0).astype(int)
z["has_major_dc_25km"] = (z.major_dc_count_25km > 0).astype(int)

# ---------------------------------------------------------------- curated hyperscale-campus label
import unicodedata, re

def norm(s, strip=True):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    if not strip:
        return s
    return re.sub(r"\s+(city|town|village|cdp|borough|municipality|city and borough|"
                  r"unified government \(balance\)|metro government \(balance\)|\(balance\))$", "", s).strip()

camp = pd.read_csv(ROOT / "data/labels/hyperscale_campuses.csv")
places = pd.read_csv(RAW / "2023_Gaz_place_national.txt", sep="\t", dtype=str)
places.columns = places.columns.str.strip()
places["key"] = places.USPS + "|" + places.NAME.map(norm)
places = places.sort_values("ALAND_SQMI", key=lambda s: s.astype(float), ascending=False).drop_duplicates("key")
cg["key"] = cg.USPS.str.strip() + "|" + cg.NAME.map(lambda x: norm(x, False))
pl = places.set_index("key")[["INTPTLAT", "INTPTLONG"]].astype(float)
ct = cg.set_index("key")[["INTPTLAT", "INTPTLONG"]].astype(float)
lat, lon, how = [], [], []
for _, r in camp.iterrows():
    if pd.notna(r.place) and f"{r.state}|{norm(r.place, False)}" in pl.index:
        p = pl.loc[f"{r.state}|{norm(r.place, False)}"]; how.append("place")
    elif pd.notna(r.county) and f"{r.state}|{norm(r.county, False)}" in ct.index:
        p = ct.loc[f"{r.state}|{norm(r.county, False)}"]; how.append("county")
    else:
        p = pd.Series({"INTPTLAT": np.nan, "INTPTLONG": np.nan}); how.append("FAILED")
    lat.append(p.INTPTLAT); lon.append(p.INTPTLONG)
camp["lat"], camp["lon"], camp["geocode"] = lat, lon, how
failed = camp[camp.geocode == "FAILED"]
if len(failed):
    print("Geocode failures:\n", failed[["operator", "campus", "place", "county", "state"]])
camp = camp.dropna(subset=["lat"])
camp.to_csv(OUT / "hyperscale_campuses_geocoded.csv", index=False)

def near(sub, km):
    t = BallTree(to_rad(sub.lat, sub.lon), metric="haversine")
    return np.array([len(i) for i in t.query_radius(zr_all, r=km / R_KM)]), t.query(zr_all, k=1)[0][:, 0] * R_KM

zr_all = to_rad(z.lat, z.lon)
cnt, dist = near(camp, 25)
z["hyperscale_count_25km"], z["dist_hyperscale_km"] = cnt, dist
z["has_hyperscale_25km"] = (cnt > 0).astype(int)
cnt_op, _ = near(camp[camp.status == "operating"], 25)
z["has_hyperscale_operating_25km"] = (cnt_op > 0).astype(int)
cnt_gw, _ = near(camp[camp.scale == "1GW+"], 50)
z["has_gw_campus_50km"] = (cnt_gw > 0).astype(int)
print(f"Campuses geocoded: {len(camp)} ({(camp.geocode=='place').sum()} by place, "
      f"{(camp.geocode=='county').sum()} by county)")

z = z[z.state.notna() & (z.state != "PR")]
z.to_csv(OUT / "zip_features.csv", index=False)
print(z.shape)
print(z.describe().T.round(2).to_string())
print("DC points:", len(dcs), "| major:", int(dcs.major_operator.sum()))
