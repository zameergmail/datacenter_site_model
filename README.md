# Mega Data Center Site Suitability by ZIP Code

Side project: given a US ZIP code, estimate how suitable it is for a mega (≥100 MW) data center using only public data.

## Layout
```
datacenter_site_model/
├── datacenter_site_model.ipynb    # main notebook (already executed, outputs saved)
├── src/build_features.py          # raw → processed (python3 src/build_features.py .)
├── src/make_notebook.py           # regenerates the notebook skeleton
└── data/
    ├── labels/hyperscale_campuses.csv  # 116 curated ≥100 MW campuses (hand-maintained)
    ├── raw/                       # untouched public downloads (~190 MB)
    └── processed/
        ├── zip_features.csv       # 33,642 ZIPs × 40 columns, model-ready
        ├── zip_scores.csv         # same + model score, percentile, cluster
        ├── datacenters_combined.csv  # 2,598 deduped data-center points (OSM + PeeringDB)
        └── hyperscale_campuses_geocoded.csv  # curated campuses + lat/lon
```

## Research summary: what drives data center siting
Industry site-selection studies (CBRE, JLL, Cushman & Wakefield market reports) consistently rank:
1. **Power**: available capacity and the time it takes to get it (interconnection queues now run 3–7 yrs in hot markets), price (¢/kWh), and access to clean energy for corporate PPAs.
2. **Fiber/latency**: proximity to long-haul routes and internet exchanges. This matters less for AI-training campuses than for cloud regions.
3. **Land**: large flat parcels (100–1,000+ acres), zoning, and price.
4. **Water and climate**: cooling load (cooling degree days) and water availability or drought risk.
5. **Risk**: earthquake, flood, tornado, hurricane, and wildfire.
6. **Policy**: sales and use tax exemptions on equipment, property tax abatements, and permitting speed. Community pushback and moratoria are growing (for example in Loudoun and Prince William, VA).

## Data dictionary (`zip_features.csv`)
| Column | Meaning | Source |
|---|---|---|
| zip, lat, lon, land_sqmi, water_sqmi | ZCTA code, internal point, area | Census 2023 Gazetteer |
| county_fips, county_name, state | Primary county (largest land overlap) | Census 2020 ZCTA–County rel. |
| county_pop_2024, county_pop_density, county_pop_growth_20_24 | Land-pressure / labour proxies | Census Pop. Estimates V2024 |
| elec_price_industrial_c_kwh, elec_price_commercial_c_kwh | State avg price, YTD Sep 2026 | EIA EPM Table 5.6.B |
| gen_mw_50km, gen_mw_100km | Operating nameplate generation within radius | EIA-860 2025 |
| clean_mw_50km/100km, clean_share_100km | Solar+wind+hydro+geo+nuclear MW and share | EIA-860 2025 |
| n_plants_50km, max_grid_kv_50km | Plant count; highest plant grid voltage nearby (transmission proxy) | EIA-860 2025 |
| dist_500mw_plant_km, dist_gas_pipeline_plant_km | Distance to big plant / gas-pipeline-served plant | EIA-860 2025 |
| dist_peeringdb_fac_km, dist_ix_km, networks_50km | Connectivity (**leaky vs. label**) | PeeringDB |
| cooling_degree_days, avg_tmax_f, annual_precip_in, drought_pdsi | County climate means 1995–2024 | NOAA nClimDiv |
| dc_count_10km, dc_count_25km, major_dc_count_25km, dist_nearest_dc_km | Existing data centers | OSM + PeeringDB |
| has_dc_10km, has_major_dc_25km | v0 labels (OSM + PeeringDB, mostly metro colo) | derived |
| hyperscale_count_25km, dist_hyperscale_km | Curated hyperscale campuses nearby | labels/hyperscale_campuses.csv |
| **has_hyperscale_25km** | **Primary v1 label**: campus (any status) within 25 km | derived |
| has_hyperscale_operating_25km, has_gw_campus_50km | Stricter variants: operating only; 1 GW+ within 50 km | derived |

### Curated campus list (`data/labels/hyperscale_campuses.csv`)
Contains 116 campuses: Google, Microsoft, AWS, Meta, Apple, OpenAI/Oracle Stargate, xAI, QTS, Switch, Vantage, Aligned, and others. Each row has a status (operating, under_construction or announced), a scale (100MW+ or 1GW+) and a confidence rating. The list was compiled from public announcements and trade press up to Nov 2026, and each campus is geocoded to its city or county centroid with Census gazetteers. **Verify rows before you rely on them.** Rows marked `medium` or `low` confidence in particular should be checked against primary sources (company press releases, county economic-development filings). To add a campus, append a row and rerun `python3 src/build_features.py .`.

Known limitations: Connecticut ZIPs have no county population, because CT replaced counties with planning regions in 2022. AK and HI have no NOAA climate data. PO-box-only ZIPs have no ZCTA.

## Not yet included (manual or keyed downloads)
- FEMA National Risk Index (county hazards): download from hazards.fema.gov/nri/data-resources; scripted access is blocked.
- ZIP-level ACS (median home value, income): needs a free Census API key (api.census.gov/data/key_signup.html).
- FCC Broadband Data Collection (fiber): needs a free account.
- Substations and transmission lines: HIFLD Open was retired; use OSM `power=substation` through Overpass instead.
- WRI Aqueduct water stress, USDA land values, state data-center tax-incentive table, LBNL interconnection queues.
