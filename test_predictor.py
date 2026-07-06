from predictor import predict_transaction

sample_transaction = {
    "amount": 25000,
    "channel": "Mobile",
    "merchant_category": "Retail",
    "bank": "Access Bank",
    "location": "Lagos",
    "age_group": "26-35",

    "hour": 14,
    "day_of_week": 3,
    "month": 6,

    "is_weekend": 0,
    "is_peak_hour": 1,

    "tx_count_24h": 5,
    "amount_sum_24h": 50000,

    "amount_mean_7d": 12000,
    "amount_std_7d": 3000,

    "tx_count_total": 300,
    "amount_mean_total": 10000,
    "amount_std_total": 2500,

    "channel_diversity": 3,
    "location_diversity": 2,

    "amount_vs_mean_ratio": 2.5,
    "online_channel_ratio": 0.80,
    "velocity_score": 4.8,

    "merchant_risk_score": 0.70,
    "composite_risk": 0.75
}

result = predict_transaction(sample_transaction)

print(result)