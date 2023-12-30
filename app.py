import matplotlib.pyplot as plt

# Sample data
customers = [
    {"age": 25, "income": 30000, "spending_score": 70},
    {"age": 40, "income": 60000, "spending_score": 50},
    {"age": 35, "income": 50000, "spending_score": 80},
    {"age": 28, "income": 40000, "spending_score": 75},
    {"age": 50, "income": 80000, "spending_score": 45},
    {"age": 22, "income": 35000, "spending_score": 85},
]

# Segment customers based on age
age_segments = {"Young": 0, "Middle-aged": 0, "Senior": 0}
for customer in customers:
    if customer["age"] <= 30:
        age_segments["Young"] += 1
    elif 30 < customer["age"] <= 50:
        age_segments["Middle-aged"] += 1
    else:
        age_segments["Senior"] += 1

# Plotting the age segmentation
plt.bar(age_segments.keys(), age_segments.values())
plt.title("Customer Age Segmentation")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.show()

# Segment customers based on income
income_segments = {"Low": 0, "Medium": 0, "High": 0}
for customer in customers:
    if customer["income"] <= 40000:
        income_segments["Low"] += 1
    elif 40000 < customer["income"] <= 60000:
        income_segments["Medium"] += 1
    else:
        income_segments["High"] += 1

# Plotting the income segmentation
plt.bar(income_segments.keys(), income_segments.values())
plt.title("Customer Income Segmentation")
plt.xlabel("Income Group")
plt.ylabel("Number of Customers")
plt.show()

# Segment customers based on spending score
spending_score_segments = {"Low": 0, "Medium": 0, "High": 0}
for customer in customers:
    if customer["spending_score"] <= 50:
        spending_score_segments["Low"] += 1
    elif 50 < customer["spending_score"] <= 70:
        spending_score_segments["Medium"] += 1
    else:
        spending_score_segments["High"] += 1

# Plotting the spending score segmentation
plt.bar(spending_score_segments.keys(), spending_score_segments.values())
plt.title("Customer Spending Score Segmentation")
plt.xlabel("Spending Score Group")
plt.ylabel("Number of Customers")
plt.show()
