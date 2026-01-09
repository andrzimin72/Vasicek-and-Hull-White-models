# Vasicek-and-Hull-White-models
This is fully integrated and enhanced versions of scripts, incorporating features:
- Vasicek and Hull-White models;
- bootstrap confidence intervals;
- caplet breakdown plots;
- swaption pricing;
- yield curve export (ZCB, spot, forward);
- parameter export to CSV;
- robust vectorization and error handling.

I think the provided scripts are a comprehensive implementation for calibrating short-rate models (Vasicek and CIR), pricing interest rate instruments (zero-coupon bonds, caps, caplets), and simulating the Vasicek model using Monte Carlo methods. I suppose this is a correct, robust, and insightful short-rate modeling framework. The plots are the expected signature of each model’s behavior. May be Hull-White fits the market because it’s designed to; Vasicek approximates it because it’s simpler. I think we’re now ready to use these models for real-world pricing and risk analysis.

## Recommendations:
I think these scripts could be used to:
1. For pricing caps/swaptions: use Hull-White - it’s designed for this.
2. For scenario analysis / stress testing: use Vasicek - easier to interpret parameters.
3. If we want Vasicek to fit better:
- add more market points (especially at intermediate maturities);
- try calibrating to caps/swaptions instead of just LIBOR/SWAP;
- use multi-factor Vasicek (but that’s advanced).
4. We should check our Hull-White sigma. If we manually set sigma=0.005, try letting it calibrate freely (remove the floor) - we’ll get higher cap prices and possibly better fit.

## How to Run:
1. Save both files in the same directory
2. Install dependencies:
```
pip install numpy matplotlib scipy pandas
```
3. Run:
```
python3 main.py
```

## Outputs:
- calibrated_parameters.csv;
- yield_curves.csv;
- three plots: market fit, ZCB curves, caplet breakdown.
