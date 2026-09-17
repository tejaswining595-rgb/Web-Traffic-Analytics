# Web-Traffic-Analytics
# ============================================================
#              WEB TRAFFIC ANALYTICS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Put your downloaded CSV file in the same folder.
# Change the filename if required.

file_name = "web_traffic.csv"

df = pd.read_csv(file_name)

print("====================================================")
print("              WEB TRAFFIC ANALYTICS")
print("====================================================")

print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================================
# STEP 2: BASIC DATA INFORMATION
# ============================================================

print("\n====================================================")
print("              DATASET INFORMATION")
print("====================================================")

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# STEP 3: CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())

# ============================================================
# STEP 4: REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nNumber of duplicate records:", duplicates)

df = df.drop_duplicates()

print("Duplicates removed successfully.")

# ============================================================
# STEP 5: CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "session_duration",
    "pageviews",
    "converted"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ============================================================
# STEP 6: HANDLE MISSING VALUES
# ============================================================

for column in df.select_dtypes(
    include=["int64", "float64"]
).columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

for column in df.select_dtypes(
    include=["object"]
).columns:

    df[column] = df[column].fillna(
        "Unknown"
    )

print("\nMissing values handled.")

# ============================================================
# STEP 7: BASIC STATISTICS
# ============================================================

print("\n====================================================")
print("              BASIC STATISTICS")
print("====================================================")

if "session_duration" in df.columns:

    print(
        "\nAverage session duration:",
        round(
            df["session_duration"].mean(),
            2
        )
    )

if "pageviews" in df.columns:

    print(
        "Average pageviews:",
        round(
            df["pageviews"].mean(),
            2
        )
    )

# ============================================================
# STEP 8: CALCULATE BOUNCE RATE
# ============================================================

print("\n====================================================")
print("                 BOUNCE RATE")
print("====================================================")

# Assumption:
# A session with only 1 pageview is considered a bounce.

if "pageviews" in df.columns:

    df["is_bounce"] = np.where(
        df["pageviews"] <= 1,
        1,
        0
    )

    total_sessions = len(df)

    bounced_sessions = df["is_bounce"].sum()

    bounce_rate = (
        bounced_sessions /
        total_sessions
    ) * 100

    print(
        "Total sessions:",
        total_sessions
    )

    print(
        "Bounced sessions:",
        bounced_sessions
    )

    print(
        "Overall bounce rate: {:.2f}%".format(
            bounce_rate
        )
    )

else:

    bounce_rate = 0

# ============================================================
# STEP 9: AVERAGE SESSION DURATION
# ============================================================

print("\n====================================================")
print("          AVERAGE SESSION DURATION")
print("====================================================")

if "session_duration" in df.columns:

    average_duration = (
        df["session_duration"].mean()
    )

    print(
        "Average session duration: {:.2f}".format(
            average_duration
        )
    )

else:

    average_duration = 0

# ============================================================
# STEP 10: PAGEVIEWS PER USER
# ============================================================

print("\n====================================================")
print("              PAGEVIEWS PER USER")
print("====================================================")

if "user_id" in df.columns and "pageviews" in df.columns:

    user_pageviews = df.groupby(
        "user_id"
    )["pageviews"].sum()

    average_pageviews_per_user = (
        user_pageviews.mean()
    )

    print(
        "Average pageviews per user: {:.2f}".format(
            average_pageviews_per_user
        )
    )

else:

    average_pageviews_per_user = df[
        "pageviews"
    ].mean()

    print(
        "Average pageviews per session: {:.2f}".format(
            average_pageviews_per_user
        )
    )

# ============================================================
# STEP 11: USER COHORT ANALYSIS
# ============================================================

print("\n====================================================")
print("             USER COHORT ANALYSIS")
print("====================================================")

if "cohort" in df.columns:

    cohort_analysis = df.groupby(
        "cohort"
    ).agg(
        users=("cohort", "count"),
        average_pageviews=("pageviews", "mean"),
        average_session_duration=(
            "session_duration",
            "mean"
        )
    )

    print(cohort_analysis)

else:

    print(
        "Cohort column not available in dataset."
    )

# ============================================================
# STEP 12: CONVERSION RATE
# ============================================================

print("\n====================================================")
print("              CONVERSION ANALYSIS")
print("====================================================")

if "converted" in df.columns:

    total_users = len(df)

    converted_users = df[
        df["converted"] == 1
    ].shape[0]

    conversion_rate = (
        converted_users /
        total_users
    ) * 100

    print(
        "Total users:",
        total_users
    )

    print(
        "Converted users:",
        converted_users
    )

    print(
        "Overall conversion rate: {:.2f}%".format(
            conversion_rate
        )
    )

else:

    conversion_rate = 0

# ============================================================
# STEP 13: CONVERSION BY TRAFFIC SOURCE
# ============================================================

print("\n====================================================")
print("        CONVERSION BY TRAFFIC SOURCE")
print("====================================================")

if "traffic_source" in df.columns:

    source_analysis = df.groupby(
        "traffic_source"
    ).agg(
        total_users=("traffic_source", "count"),
        conversions=("converted", "sum")
    )

    source_analysis[
        "conversion_rate"
    ] = (
        source_analysis["conversions"] /
        source_analysis["total_users"]
    ) * 100

    print(source_analysis)

else:

    print(
        "traffic_source column not available."
    )

# ============================================================
# STEP 14: BOUNCE RATE BY TRAFFIC SOURCE
# ============================================================

print("\n====================================================")
print("        BOUNCE RATE BY TRAFFIC SOURCE")
print("====================================================")

if "traffic_source" in df.columns:

    source_bounce = df.groupby(
        "traffic_source"
    )["is_bounce"].mean() * 100

    print(
        source_bounce
    )

# ============================================================
# STEP 15: SESSION DURATION BY TRAFFIC SOURCE
# ============================================================

print("\n====================================================")
print("    SESSION DURATION BY TRAFFIC SOURCE")
print("====================================================")

if "traffic_source" in df.columns:

    source_duration = df.groupby(
        "traffic_source"
    )["session_duration"].mean()

    print(
        source_duration
    )

# ============================================================
# STEP 16: CONVERSION FUNNEL
# ============================================================

print("\n====================================================")
print("               CONVERSION FUNNEL")
print("====================================================")

# Funnel stages:
#
# Visitors
# ↓
# Engaged Users
# ↓
# Product/Service Page
# ↓
# Checkout
# ↓
# Converted

total_visitors = len(df)

if "pageviews" in df.columns:

    engaged_users = len(
        df[df["pageviews"] >= 2]
    )

else:

    engaged_users = total_visitors

if "pageviews" in df.columns:

    product_page_users = len(
        df[df["pageviews"] >= 3]
    )

else:

    product_page_users = engaged_users

if "pageviews" in df.columns:

    checkout_users = len(
        df[df["pageviews"] >= 4]
    )

else:

    checkout_users = product_page_users

if "converted" in df.columns:

    converted_users = len(
        df[df["converted"] == 1]
    )

else:

    converted_users = 0

funnel = pd.DataFrame({
    "Stage": [
        "Visitors",
        "Engaged Users",
        "Product Page",
        "Checkout",
        "Converted"
    ],

    "Users": [
        total_visitors,
        engaged_users,
        product_page_users,
        checkout_users,
        converted_users
    ]
})

print(funnel)

# ============================================================
# STEP 17: FUNNEL DROP-OFF
# ============================================================

print("\n====================================================")
print("              FUNNEL DROP-OFF")
print("====================================================")

users = funnel["Users"].tolist()

stages = funnel["Stage"].tolist()

for i in range(len(users) - 1):

    current_users = users[i]

    next_users = users[i + 1]

    if current_users > 0:

        dropoff = (
            (current_users - next_users) /
            current_users
        ) * 100

    else:

        dropoff = 0

    print(
        stages[i],
        "->",
        stages[i + 1],
        ": {:.2f}% drop-off".format(
            dropoff
        )
    )

# ============================================================
# STEP 18: TOP DROP-OFF PAGE
# ============================================================

print("\n====================================================")
print("              TOP DROP-OFF POINT")
print("====================================================")

dropoffs = []

for i in range(len(users) - 1):

    if users[i] > 0:

        dropoff = (
            (users[i] - users[i + 1]) /
            users[i]
        ) * 100

    else:

        dropoff = 0

    dropoffs.append(dropoff)

if len(dropoffs) > 0:

    maximum_dropoff = max(dropoffs)

    maximum_index = dropoffs.index(
        maximum_dropoff
    )

    print(
        "Highest drop-off occurs between:",
        stages[maximum_index],
        "and",
        stages[maximum_index + 1]
    )

    print(
        "Drop-off rate: {:.2f}%".format(
            maximum_dropoff
        )
    )

# ============================================================
# STEP 19: BAR CHART - TRAFFIC SOURCES
# ============================================================

if "traffic_source" in df.columns:

    traffic_count = (
        df["traffic_source"]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    traffic_count.plot(
        kind="bar"
    )

    plt.title(
        "Website Traffic by Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Number of Users"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "01_traffic_sources.png"
    )

    plt.show()

# ============================================================
# STEP 20: BAR CHART - CONVERSION RATE
# ============================================================

if "traffic_source" in df.columns:

    plt.figure(figsize=(8, 5))

    source_analysis[
        "conversion_rate"
    ].plot(
        kind="bar"
    )

    plt.title(
        "Conversion Rate by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Conversion Rate (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "02_conversion_by_source.png"
    )

    plt.show()

# ============================================================
# STEP 21: BAR CHART - BOUNCE RATE
# ============================================================

if "traffic_source" in df.columns:

    plt.figure(figsize=(8, 5))

    source_bounce.plot(
        kind="bar"
    )

    plt.title(
        "Bounce Rate by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Bounce Rate (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "03_bounce_rate_by_source.png"
    )

    plt.show()

# ============================================================
# STEP 22: BAR CHART - SESSION DURATION
# ============================================================

if "traffic_source" in df.columns:

    plt.figure(figsize=(8, 5))

    source_duration.plot(
        kind="bar"
    )

    plt.title(
        "Average Session Duration by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Average Session Duration"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "04_session_duration_by_source.png"
    )

    plt.show()

# ============================================================
# STEP 23: FUNNEL CHART
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    funnel["Stage"],
    funnel["Users"]
)

plt.title(
    "Website Conversion Funnel"
)

plt.xlabel(
    "Number of Users"
)

plt.ylabel(
    "Funnel Stage"
)

plt.tight_layout()

plt.savefig(
    "05_conversion_funnel.png"
)

plt.show()

# ============================================================
# STEP 24: PAGEVIEWS DISTRIBUTION
# ============================================================

if "pageviews" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["pageviews"],
        bins=10
    )

    plt.title(
        "Distribution of Pageviews"
    )

    plt.xlabel(
        "Pageviews"
    )

    plt.ylabel(
        "Number of Users"
    )

    plt.tight_layout()

    plt.savefig(
        "06_pageviews_distribution.png"
    )

    plt.show()

# ============================================================
# STEP 25: SESSION DURATION DISTRIBUTION
# ============================================================

if "session_duration" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["session_duration"],
        bins=10
    )

    plt.title(
        "Distribution of Session Duration"
    )

    plt.xlabel(
        "Session Duration"
    )

    plt.ylabel(
        "Number of Users"
    )

    plt.tight_layout()

    plt.savefig(
        "07_session_duration_distribution.png"
    )

    plt.show()

# ============================================================
# STEP 26: IDENTIFY TRAFFIC PERFORMANCE
# ============================================================

print("\n====================================================")
print("            TRAFFIC PERFORMANCE SUMMARY")
print("====================================================")

if "traffic_source" in df.columns:

    print("\nTraffic source performance:")

    print(source_analysis)

    # Highest conversion rate

    highest_conversion_source = (
        source_analysis[
            "conversion_rate"
        ].idxmax()
    )

    highest_conversion_rate = (
        source_analysis[
            "conversion_rate"
        ].max()
    )

    print(
        "\nHighest conversion rate:",
        highest_conversion_source
    )

    print(
        "Conversion rate: {:.2f}%".format(
            highest_conversion_rate
        )
    )

    # Highest bounce rate

    highest_bounce_source = (
        source_bounce.idxmax()
    )

    highest_bounce_rate = (
        source_bounce.max()
    )

    print(
        "\nHighest bounce rate:",
        highest_bounce_source
    )

    print(
        "Bounce rate: {:.2f}%".format(
            highest_bounce_rate
        )
    )

    # Longest session

    longest_session_source = (
        source_duration.idxmax()
    )

    longest_session = (
        source_duration.max()
    )

    print(
        "\nLongest average session:",
        longest_session_source
    )

    print(
        "Average duration: {:.2f}".format(
            longest_session
        )
    )

# ============================================================
# STEP 27: BUSINESS RECOMMENDATIONS
# ============================================================

print("\n====================================================")
print("            BUSINESS RECOMMENDATIONS")
print("====================================================")

print("""
1. Improve landing pages:
   High bounce rates indicate that some visitors
   leave without exploring the website.

2. Improve website navigation:
   Make important pages easy to find so users can
   move through the conversion funnel smoothly.

3. Optimize the checkout process:
   A large drop-off near checkout may indicate
   usability or payment-related problems.

4. Improve high-performing traffic sources:
   Analyze sources that generate more conversions
   and maintain their successful strategies.

5. Optimize low-performing traffic sources:
   Review advertising messages, landing pages and
   audience targeting for sources with low conversion.

6. Increase user engagement:
   Provide useful content and clear calls-to-action
   to increase pageviews and session duration.

7. Monitor bounce rate regularly:
   Compare bounce rates over time to identify
   improvements or new website problems.

8. Track the conversion funnel:
   Monitoring every funnel stage helps identify
   where visitors leave the website.
""")

# ============================================================
# STEP 28: SAVE ANALYSIS REPORT
# ============================================================

report = open(
    "web_traffic_analysis_report.txt",
    "w"
)

report.write(
    "====================================================\n"
)

report.write(
    "             WEB TRAFFIC ANALYTICS REPORT\n"
)

report.write(
    "====================================================\n\n"
)

report.write(
    "Total Visitors: {}\n".format(
        total_visitors
    )
)

report.write(
    "Overall Bounce Rate: {:.2f}%\n".format(
        bounce_rate
    )
)

report.write(
    "Average Session Duration: {:.2f}\n".format(
        average_duration
    )
)

report.write(
    "Average Pageviews per User: {:.2f}\n".format(
        average_pageviews_per_user
    )
)

report.write(
    "Overall Conversion Rate: {:.2f}%\n\n".format(
        conversion_rate
    )
)

report.write(
    "----------------------------------------------------\n"
)

report.write(
    "CONVERSION BY TRAFFIC SOURCE\n"
)

report.write(
    "----------------------------------------------------\n"
)

if "traffic_source" in df.columns:

    report.write(
        source_analysis.to_string()
    )

report.write("\n\n")

report.write(
    "----------------------------------------------------\n"
)

report.write(
    "FUNNEL ANALYSIS\n"
)

report.write(
    "----------------------------------------------------\n"
)

report.write(
    funnel.to_string(index=False)
)

report.write("\n\n")

report.write(
    "----------------------------------------------------\n"
)

report.write(
    "BUSINESS RECOMMENDATIONS\n"
)

report.write(
    "----------------------------------------------------\n"
)

report.write("""
1. Improve landing pages.
2. Improve website navigation.
3. Optimize the checkout process.
4. Improve high-performing traffic sources.
5. Optimize low-performing traffic sources.
6. Increase user engagement.
7. Monitor bounce rate regularly.
8. Track the conversion funnel.
""")

report.close()

# ============================================================
# STEP 29: SAVE ANALYZED DATA
# ============================================================

df.to_csv(
    "web_traffic_analyzed.csv",
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n====================================================")
print("             ANALYSIS COMPLETED")
print("====================================================")

print("""
Files generated:

1. web_traffic_analyzed.csv

2. web_traffic_analysis_report.txt

Charts generated:

1. 01_traffic_sources.png
2. 02_conversion_by_source.png
3. 03_bounce_rate_by_source.png
4. 04_session_duration_by_source.png
5. 05_conversion_funnel.png
6. 06_pageviews_distribution.png
7. 07_session_duration_distribution.png

You can use these charts and the report to prepare
your Web Analytics Dashboard PDF or presentation.
""")

print("Thank you!")