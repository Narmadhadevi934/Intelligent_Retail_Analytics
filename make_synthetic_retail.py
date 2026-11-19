
import numpy as np
import pandas as pd
import os


def make_synthetic_retail(
    seed=42,
    n_customers=1500,
    n_products=120,
    start="2023-01-01",
    end="2024-12-31"
):
    rng = np.random.default_rng(seed)

    cats = [
        "Home Decor", "Kitchen", "Stationery",
        "Toys", "Garden", "Gifts", "Bath"
    ]

    cat_of = rng.integers(0, len(cats), n_products)

    price = np.round(
        rng.lognormal(1.0, 0.6, n_products) + 0.5, 2
    )

    codes = [f"P{1000+i}" for i in range(n_products)]

    names = [
        f"{cats[c].upper()} ITEM {i+1}"
        for i, c in enumerate(cat_of)
    ]

    pop = rng.pareto(1.3, n_products) + 1
    pop = pop / pop.sum()

    bundles = [
        rng.choice(n_products, 3, replace=False)
        for _ in range(12)
    ]

    pref = [
        rng.choice(len(cats), 2, replace=False)
        for _ in range(n_customers)
    ]

    aff = np.ones((n_customers, n_products))

    for c in range(n_customers):
        aff[c, np.isin(cat_of, pref[c])] = 4.0

    month_mult = np.ones((13, n_products))

    for m in range(1, 13):
        for ci, cn in enumerate(cats):
            f = 1.0

            if cn in ("Toys", "Gifts") and m in (11, 12):
                f = 2.5
            if cn == "Garden" and m in (4, 5, 6, 7):
                f = 2.0
            if cn == "Home Decor" and m in (10, 11, 12):
                f = 1.5
            if cn == "Bath" and m in (1, 2):
                f = 1.4

            month_mult[m, cat_of == ci] = f

    t0 = pd.Timestamp(start)
    rows = []
    inv_no = 500000

    for d in pd.date_range(start, end):

        base = (
            22 * (1 + 0.5 * (d.month in (11, 12)))
            * (0.45 if d.weekday() == 5 else 1.0)
            * (1 + 0.0004 * (d - t0).days)
        )

        for _ in range(rng.poisson(base)):

            customer = int(rng.integers(0, n_customers))

            weights = (
                pop * aff[customer] * month_mult[d.month]
            )
            weights = weights / weights.sum()

            selected = list(
                rng.choice(
                    n_products,
                    size=int(rng.integers(1, 5)),
                    replace=False,
                    p=weights
                )
            )

            if rng.random() < 0.25:
                bundle = bundles[
                    int(rng.integers(0, len(bundles)))
                ]
                selected.extend(bundle.tolist())

            inv_no += 1

            for product in set(selected):

                quantity = int(rng.integers(1, 5))

                rows.append({
                    "InvoiceNo": str(inv_no),
                    "StockCode": codes[product],
                    "Description": names[product],
                    "Quantity": quantity,
                    "InvoiceDate": d + pd.Timedelta(
                        minutes=int(rng.integers(0, 1440))
                    ),
                    "UnitPrice": float(price[product]),
                    "CustomerID": 10000 + customer,
                    "Country": "United Kingdom"
                })

    df = pd.DataFrame(rows)

    if len(df) > 100:
        dirty = df.sample(10, random_state=seed).copy()
        dirty["CustomerID"] = np.nan
        df = pd.concat([df, dirty], ignore_index=True)

    print("Synthetic dataset generated:", len(df), "records")

    return df


if __name__ == "__main__":

    df = make_synthetic_retail()

    os.makedirs("outputs", exist_ok=True)

    output_file = os.path.join(
        "outputs",
        "synthetic_retail_dataset.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nDataset created successfully!")
    print("Total records:", len(df))
    print("Saved at:", os.path.abspath(output_file))
    print("\nFirst five records:")
    print(df.head())
