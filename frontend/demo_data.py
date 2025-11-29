"""
Demo test data for UCI Credit Card Default dataset
Includes user input + auto-retrieved historical data
"""

# Complete test cases with all 23 features
DEMO_TEST_CASES = {
    "Manual Entry": {
        "description": "Enter your own data",
        "user_input": {
            "credit_limit": 50000,
            "gender": 2,  # 1=Male, 2=Female
            "education": 2,  # 1=Graduate, 2=University, 3=High School, 4=Others
            "marriage": 1,  # 1=Married, 2=Single, 3=Others
            "age": 30
        },
        "auto_retrieved": {
            "payment_history": [0, 0, 0, 0, 0, 0],  # X6-X11: -1=paid, 0=on time, 1+=late
            "bill_statements": [3000, 3000, 3000, 3000, 3000, 3000],  # X12-X17
            "previous_payments": [3000, 3000, 3000, 3000, 3000, 3000]  # X18-X23
        },
        "expected": "Variable",
        "risk_level": "medium"
    },
    
    "✅ Low-Risk Customer": {
        "description": "High income, excellent payment history, low debt utilization - Low default risk",
        "user_input": {
            "credit_limit": 200000,  # High credit limit
            "gender": 2,  # Female
            "education": 1,  # Graduate school
            "marriage": 1,  # Married
            "age": 35
        },
        "auto_retrieved": {
            "payment_history": [0, 0, 0, 0, 0, 0],  # All paid on time (0 = current, not late)
            "bill_statements": [30000, 28000, 32000, 29000, 31000, 30000],  # ~15% utilization
            "previous_payments": [30000, 28000, 32000, 29000, 31000, 30000]  # Full payments
        },
        "expected": "NO DEFAULT",
        "risk_level": "low"
    },
    
    "⚠️ Medium-Risk Customer": {
        "description": "Average income, occasional late payments, moderate debt - Medium default risk",
        "user_input": {
            "credit_limit": 100000,
            "gender": 1,  # Male
            "education": 2,  # University
            "marriage": 2,  # Single
            "age": 28
        },
        "auto_retrieved": {
            "payment_history": [0, 0, 1, 0, 0, 0],  # One late payment
            "bill_statements": [45000, 43000, 47000, 44000, 46000, 45000],  # ~45% utilization
            "previous_payments": [20000, 18000, 22000, 19000, 21000, 20000]  # Partial payments
        },
        "expected": "NO DEFAULT (Borderline)",
        "risk_level": "medium"
    },
    
    "❌ High-Risk Customer": {
        "description": "Low income, frequent late payments, high debt utilization - HIGH default risk",
        "user_input": {
            "credit_limit": 50000,  # Low credit limit
            "gender": 1,  # Male
            "education": 4,  # Others
            "marriage": 3,  # Others
            "age": 24
        },
        "auto_retrieved": {
            "payment_history": [2, 2, 3, 2, 1, 2],  # Consistently 1-3 months late
            "bill_statements": [48000, 49000, 48500, 49200, 48800, 49100],  # ~98% utilization
            "previous_payments": [500, 800, 600, 700, 550, 750]  # Very minimal payments
        },
        "expected": "DEFAULT",
        "risk_level": "high"
    },
    
    "👩‍🎓 Female Graduate": {
        "description": "Bias testing: Female with graduate degree, excellent profile",
        "user_input": {
            "credit_limit": 300000,
            "gender": 2,  # Female
            "education": 1,  # Graduate school
            "marriage": 1,  # Married
            "age": 32
        },
        "auto_retrieved": {
            "payment_history": [0, 0, 0, 0, 0, 0],  # Excellent payment history (all on-time)
            "bill_statements": [60000, 58000, 62000, 59000, 61000, 60000],  # ~20% utilization
            "previous_payments": [60000, 58000, 62000, 59000, 61000, 60000]  # Full payments
        },
        "expected": "NO DEFAULT",
        "risk_level": "low"
    },
    
    "👨‍🏫 Male High School": {
        "description": "Bias testing: Male with high school education, SAME financial profile as Female Graduate",
        "user_input": {
            "credit_limit": 300000,
            "gender": 1,  # Male (different from test 5)
            "education": 3,  # High school (different from test 5)
            "marriage": 1,  # Married
            "age": 32
        },
        "auto_retrieved": {
            "payment_history": [0, 0, 0, 0, 0, 0],  # Same as Female Graduate
            "bill_statements": [60000, 58000, 62000, 59000, 61000, 60000],  # Same
            "previous_payments": [60000, 58000, 62000, 59000, 61000, 60000]  # Same
        },
        "expected": "NO DEFAULT (Should match Female Graduate)",
        "risk_level": "low"
    },
    
    "🔴 Severe Default Risk": {
        "description": "Extreme case: Very late payments, maxed out credit, minimal repayments - WILL DEFAULT",
        "user_input": {
            "credit_limit": 30000,  # Very low limit
            "gender": 1,
            "education": 4,  # Others
            "marriage": 3,  # Others
            "age": 22  # Young
        },
        "auto_retrieved": {
            "payment_history": [3, 2, 3, 2, 2, 3],  # Consistently 2-3 months late
            "bill_statements": [29500, 29800, 29900, 29700, 29850, 29950],  # ~99% utilization
            "previous_payments": [300, 250, 350, 280, 320, 290]  # Very minimal payments
        },
        "expected": "DEFAULT",
        "risk_level": "high"
    },
    
    "⚠️ Chronic Late Payer": {
        "description": "Consistent 3-4 month payment delays with high debt - High default risk",
        "user_input": {
            "credit_limit": 60000,
            "gender": 2,
            "education": 3,
            "marriage": 2,
            "age": 28
        },
        "auto_retrieved": {
            "payment_history": [3, 2, 3, 2, 3, 2],  # Consistently 2-3 months late
            "bill_statements": [58000, 57500, 58500, 58200, 57800, 58100],  # ~97% utilization
            "previous_payments": [1500, 1800, 1600, 1700, 1650, 1750]  # Minimal payments
        },
        "expected": "DEFAULT",
        "risk_level": "high"
    },
    
    "💣 Financial Collapse": {
        "description": "Stopped paying entirely, maxed credit, minimal payments - CERTAIN DEFAULT",
        "user_input": {
            "credit_limit": 50000,
            "gender": 1,
            "education": 4,
            "marriage": 3,
            "age": 26
        },
        "auto_retrieved": {
            "payment_history": [3, 3, 3, 2, 3, 3],  # Consistently 2-3 months late
            "bill_statements": [50000, 50000, 50000, 50000, 50000, 50000],  # 100% maxed out
            "previous_payments": [400, 500, 450, 420, 480, 460]  # Very minimal payments
        },
        "expected": "DEFAULT",
        "risk_level": "high"
    },
    
    "📉 Downward Spiral": {
        "description": "Payments getting progressively worse, increasing debt - High default risk",
        "user_input": {
            "credit_limit": 80000,
            "gender": 2,
            "education": 2,
            "marriage": 2,
            "age": 35
        },
        "auto_retrieved": {
            "payment_history": [0, 1, 2, 2, 3, 3],  # Getting worse: on-time → 3 months late
            "bill_statements": [40000, 50000, 60000, 70000, 75000, 78000],  # Increasing debt
            "previous_payments": [5000, 3000, 2000, 1500, 1000, 800]  # Decreasing payments
        },
        "expected": "DEFAULT",
        "risk_level": "high"
    }
}


def get_all_features(case_name):
    """
    Get all 19 features for a test case (excluding demographic features for fairness).
    
    Excluded features: SEX, EDUCATION, MARRIAGE, AGE (to prevent discrimination)
    
    Returns:
        list: All 19 features in order
    """
    case = DEMO_TEST_CASES[case_name]
    user_input = case["user_input"]
    auto_retrieved = case["auto_retrieved"]
    
    # Combine into 19 features (excluding gender, education, marriage, age)
    features = [
        user_input["credit_limit"],  # X1: LIMIT_BAL
        # Removed: age, gender, education, marriage
    ] + auto_retrieved["payment_history"] + \
        auto_retrieved["bill_statements"] + \
        auto_retrieved["previous_payments"]
    
    return features


def get_feature_names():
    """Get feature names for all 19 features (excluding demographic features)."""
    return [
        "CREDIT_LIMIT",
        # Excluded: AGE, GENDER, EDUCATION, MARRIAGE (to prevent bias)
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]


def format_payment_history(history):
    """Format payment history for display."""
    labels = {
        -1: "✅ Paid on time",
        0: "⏱️ Current",
        1: "⚠️ 1 month late",
        2: "⚠️ 2 months late",
        3: "❌ 3 months late"
    }
    return [labels.get(h, f"❌ {h} months late") for h in history]


def format_currency(amount):
    """Format amount as currency."""
    return f"${amount:,.0f}"

