---
title: Transforming and Analyzing Job Data Using SQL
pubDate: 12/31/2022 12:05
author: "Dennis Okwechime"
tags:
  - SQL
  - WebDev
  - Programming

img: 'sql.png'
imgUrl: '../../assets/blog_covers/sql.png'
description: This project explores telecom customer churn using data analysis in Microsoft SQL Server. By exploring factors like call behavior, service usage, and international call patterns, we uncover insights to help retain customers and optimize services.
layout: '../../layouts/BlogPost.astro'
category: App
---

# Transforming and Analyzing Job Data Using SQL

## Introduction
This project demonstrates how to clean and analyze job data stored in a SQL database. The primary objective is to prepare the dataset for insightful analysis by splitting columns, removing unnecessary data, and performing various SQL queries to extract meaningful information about job trends, salaries, company ratings, and industry characteristics.

## Explanation of the Code

### Step 1: Initial Data Inspection
Before diving into data cleaning and transformation, we start by taking a quick look at the columns in the `Uncleaned_jobs` table to understand its structure.

```sql
USE [Personal Project];
SELECT * FROM dbo.Uncleaned_jobs;
```

### Step 2: Splitting the Salary_Estimate Column
The `Salary_Estimate` column contains ranges that we need to split into minimum and maximum values for better analysis.

```sql
-- Remove '(Glassdoor est.)' from Salary_Estimate
UPDATE Uncleaned_jobs
SET Salary_Estimate = REPLACE(Salary_Estimate, '(Glassdoor est.)', '')
WHERE Salary_Estimate LIKE '%(Glassdoor est.)%';

-- Add new columns for Min and Max Salary Estimates
ALTER TABLE Uncleaned_jobs
ADD Min_Salary_Estimates VARCHAR(50),
    Max_Salary_Estimates VARCHAR(50);

-- Populate new columns with split data
UPDATE Uncleaned_jobs
SET Min_Salary_Estimates = SUBSTRING(NULLIF(Salary_Estimate, ''), 1, CHARINDEX('-', NULLIF(Salary_Estimate, '')) - 1),
    Max_Salary_Estimates = SUBSTRING(NULLIF(Salary_Estimate, ''), CHARINDEX('-', NULLIF(Salary_Estimate, '')) + 1, LEN(NULLIF(Salary_Estimate, '')) - CHARINDEX('-', NULLIF(Salary_Estimate, '')));
```

### Step 3: Dropping Unnecessary Columns
We remove columns that are not needed for our analysis to simplify the dataset.

```sql
ALTER TABLE Uncleaned_jobs
DROP COLUMN Min_Salary_Estimate,
             Max_Salary_Estimate,
             Competitors,
             Revenue,
             Salary_Estimate,
             Sector,
             Job_Description;
```

### Step 4: Data Cleaning
Round off the ratings to one decimal place and clean up other text fields.

```sql
UPDATE Uncleaned_jobs
SET Rating = ROUND(Rating, 1);

UPDATE Uncleaned_jobs
SET Type_of_ownership = LTRIM(SUBSTRING(Type_of_ownership, CHARINDEX(' - ', Type_of_ownership) + 3, LEN(Type_of_ownership) - CHARINDEX(' - ', Type_of_ownership) - 2))
WHERE Type_of_ownership LIKE 'company% -%';

ALTER TABLE Uncleaned_jobs
ALTER COLUMN Company_Name VARCHAR(100);

UPDATE Uncleaned_jobs
SET Company_Name = LEFT(Company_Name, CHARINDEX(' ', Company_Name) - 1)
WHERE Company_Name LIKE '% %';

DELETE FROM Uncleaned_jobs
WHERE Rating = -1;

UPDATE Uncleaned_jobs
SET Industry = 'Not Defined'
WHERE Industry = '-1';
```

### Step 5: Analyzing the Cleaned Data
With the dataset cleaned, we can now perform various analyses to gain insights.

#### Job Titles with the Highest Number of Openings
Identify which job titles have the most openings.

```sql
SELECT Job_Title, 
       COUNT(Job_Title) AS Job_Title_Count
FROM Uncleaned_jobs
GROUP BY Job_Title
ORDER BY Job_Title_Count DESC;
```

#### Minimum and Maximum Salaries for Each Job Title
Determine the salary range for different job titles.

```sql
SELECT Job_Title, 
       MIN(Min_Salary_Estimates) as Minimum, 
       MAX(Max_Salary_Estimates) AS Maximum
FROM Uncleaned_jobs
GROUP BY Job_Title
ORDER BY Maximum;
```

#### Most Popular Type of Ownership
Find out which type of ownership is most common among the companies.

```sql
SELECT Type_of_ownership, 
       COUNT(Type_of_ownership) AS Ownership_Count
FROM Uncleaned_jobs
GROUP BY Type_of_ownership
ORDER BY Ownership_Count DESC;
```

#### Companies with Ratings Higher than the Average
Identify companies that have higher-than-average ratings.

```sql
SELECT Company_Name, 
       Industry,
       Rating
FROM Uncleaned_jobs
WHERE Rating > (SELECT AVG(Rating) FROM Uncleaned_jobs)
ORDER BY Rating DESC;
```

#### Industries with Average Ratings Above 4
Analyze which industries have average ratings above 4.

```sql
SELECT Industry, 
       ROUND(AVG(Rating), 1) AS AVERAGE_RATING
FROM Uncleaned_jobs
WHERE Rating > 4
GROUP BY Industry;
```

#### Industries That Pay the Highest Salary
Identify industries that offer the highest salaries.

```sql
SELECT TOP 10 Industry, 
       MAX(Max_Salary_Estimates) AS Highest_Possible_Pay
FROM Uncleaned_jobs
GROUP BY Industry
ORDER BY Industry DESC;
```

#### Influence of Ownership Type on Average Ratings
Analyze how the type of ownership influences the average ratings of companies.

```sql
SELECT TOP 5 Type_of_Ownership, 
       ROUND(AVG(Rating), 1) AS AverageRatings
FROM Uncleaned_jobs
GROUP BY Type_of_ownership
ORDER BY AverageRatings DESC;
```

#### Percentage of Companies Founded in the Last 5 Years
Calculate the percentage of companies founded in the last 5 years for each industry.

```sql
WITH CompanyCounts AS (
    SELECT
        Industry,
        COUNT(*) AS TotalCompanies,
        SUM(CASE WHEN FOUNDED >= YEAR(GETDATE()) - 5 THEN 1 ELSE 0 END) AS CompaniesLast5Years
    FROM
        Uncleaned_jobs
    GROUP BY
        Industry
)
SELECT
    Industry,
    CompaniesLast5Years,
    TotalCompanies,
    ROUND(CompaniesLast5Years * 100.0 / TotalCompanies, 1) AS PercentageLast5Years
FROM
    CompanyCounts;
```


#### Calculating Overall Churn Rate:

   ```sql
   SELECT ROUND((SUM(churn) * 100.00 / COUNT(*)), 2) AS churn_rate_percentage
   FROM dbo.[telecom-churn];
   ```

This code calculates the overall churn rate as a percentage.


let's explore potential factors influencing churn:

#### Churn Rate by State

```sql
SELECT 
  state, ROUND((SUM(churn) * 100.00 / COUNT(*)), 2) AS churn_rate_percentage
FROM dbo.[telecom-churn]
GROUP BY state
ORDER BY churn_rate_percentage DESC;
```

This query groups churn data by state and calculates the churn rate for each state. Sorting by churn rate (descending) reveals which states have the highest churn.

#### Account Length and Churn Correlation:**

```sql
SELECT
    (SUM(account_length * churn) - SUM(account_length) * SUM(churn) / COUNT(*)) /
    (SQRT((SUM(account_length * account_length) - POWER(SUM(account_length), 2) / COUNT(*)) *
          (SUM(churn * churn) - POWER(SUM(churn), 2) / COUNT(*)))) AS correlation_coefficient
FROM
    dbo.[telecom-churn];
```

This code calculates the correlation coefficient between account length and churn, indicating a positive or negative relationship. A positive coefficient suggests customers with longer tenures churn less, while a negative coefficient suggests the opposite.

#### International Plan and Churn Rate

```sql
WITH Int_ChurnRates AS (
    SELECT
        international_plan,
        AVG(CAST(churn AS FLOAT)) AS churn_rate
    FROM
        dbo.[telecom-churn]
    GROUP BY
        international_plan
)

SELECT
    (SELECT churn_rate FROM Int_ChurnRates WHERE international_plan = 1) AS intl_plan_churn_rate,
    (SELECT churn_rate FROM Int_ChurnRates WHERE international_plan = 0) AS no_intl_plan_churn_rate,
    (SELECT churn_rate FROM Int_ChurnRates WHERE international_plan = 1) -
    (SELECT churn_rate FROM Int_ChurnRates WHERE international_plan = 0) AS Int_churn_rate_difference;
```

This code first creates a Common Table Expression (CTE) named `Int_ChurnRates` that calculates the average churn rate for customers with and without international plans. Then, it displays the churn rates and their difference.

#### 4. Customer Service Calls and Churn

```sql
SELECT
    customer_service_calls,
    AVG(CAST(churn AS FLOAT)) AS churn_rate
FROM
    dbo.[telecom-churn]
GROUP BY
    customer_service_calls
ORDER BY
    customer_service_calls;
```

This query groups data by the number of customer service calls and calculates the average churn rate for each group. Analyzing this could reveal if customers making more calls are more likely to churn.

#### 5. Voicemail Messages and Churn Correlation

```sql
SELECT
    (SUM((number_vmail_messages - avg_vmail_messages) * (churn - avg_churn)) / (COUNT(*) - 1)) /
    (SQRT((SUM(POWER(number_vmail_messages - avg_vmail_messages, 2)) / (COUNT(*) - 1)) *
          (SUM(POWER(churn - avg_churn, 2)) / (COUNT(*) - 1)))) AS correlation_coefficient
FROM (
    SELECT
        AVG(number_vmail_messages) AS avg_vmail_messages,
        AVG(CAST(churn AS FLOAT)) AS avg_churn
    FROM
        dbo.[telecom-churn]
) AS Corr_avg_data, dbo.[telecom-churn];

```




**Explanation:**

- This code calculates the correlation coefficient between the number of voicemail messages and churn.
- It first creates a subquery named `Corr_avg_data` that calculates the average number of voicemail messages and the average churn rate across all customers.
- The main query then subtracts the average voicemail message count and churn rate from each individual customer's data points.
- These differences are then multiplied by each other (voicemail difference * churn difference) for all customers.
- The sum of these products is divided by the total number of customers minus 1 (to account for the average already removed).
- This result represents the covariance between voicemail messages and churn.
- To calculate the correlation coefficient, the covariance is divided by the product of the standard deviations of voicemail messages and churn (calculated similarly but using the `POWER` function for squaring the differences).
- The final result is the correlation coefficient, indicating a positive or negative relationship between voicemail usage and customer churn.

**Following sections analyze churn rate based on various call minute categories:**

#### Daytime Minutes and Churn Rate

This section investigates the relationship between customer churn and the total number of minutes used for daytime, evening, and nighttime calls. We aim to understand if customers with higher call minute usage in different time ranges (day, evening, night) are more or less likely to churn.


There are three code snippets, each analyzing a specific call minute category:

   - **Daytime Minutes:**
     ```sql
     SELECT
         CASE
             WHEN total_day_minutes <= 100 THEN '0-100 Minutes'
             WHEN total_day_minutes <= 200 THEN '101-200 Minutes'
             WHEN total_day_minutes <= 300 THEN '201-300 Minutes'
             ELSE '301+ Minutes'
         END AS day_minutes_range,
         AVG(CAST(churn AS FLOAT)) AS churn_rate
     FROM
         dbo.[telecom-churn]
     GROUP BY
         CASE
             WHEN total_day_minutes <= 100 THEN '0-100 Minutes'
             WHEN total_day_minutes <= 200 THEN '101-200 Minutes'
             WHEN total_day_minutes <= 300 THEN '201-300 Minutes'
             ELSE '301+ Minutes'
         END
     ORDER BY
         MIN(total_day_minutes);
     ```

   - **Evening Minutes (Similar Structure):**

   - **Night Minutes (Similar Structure):**

**Breakdown of the Daytime Minutes Code:**

   - **`CASE` Statement:** This statement categorizes total daytime minutes into ranges:
      - 0-100 Minutes
      - 101-200 Minutes
      - 201-300 Minutes
      - 301+ Minutes
   - These ranges are assigned to a new column named `day_minutes_range`.
   - **`AVG(CAST(churn AS FLOAT)) AS churn_rate`:** This calculates the average churn rate for each call minute range.
   - **`GROUP BY`:** The data is grouped by the `day_minutes_range` categories.
   - **`ORDER BY MIN(total_day_minutes)`:** This sorts the results by the minimum value in the `total_day_minutes` column within each range, ensuring a clear order from lowest to highest usage.

**Expected Outcome:**

The resulting table will show the `day_minutes_range` and the corresponding average churn rate (`churn_rate`) for each range. Analyzing this data can reveal patterns:

   - If customers with higher daytime minute usage (301+ Minutes) have a significantly higher churn rate than those with lower usage (0-100 Minutes), it might suggest a link between high daytime call volumes and customer dissatisfaction.
   - Conversely, a similar churn rate across all ranges might indicate no clear correlation between daytime call minutes and churn.

The same analysis logic applies to evening and nighttime minutes, helping you understand if customers with higher call volumes during those specific times are more likely to churn.


#### Analyzing Churn Rate Based on Call Charges and Service Calls

This section dives deeper into customer behavior by analyzing churn rate in relation to:

- International Minute Usage
- Daytime, Evening, Nighttime Call Charges
- Customer Service Call Frequency

**1. International Minutes and Churn Rate:**

```sql
-- 	Is there a pattern between the total international minutes used and the churn rate?
SELECT
  CASE
    WHEN total_intl_minutes <= 10 THEN '0-10 Minutes'
    WHEN total_intl_minutes <= 20 THEN '11-20 Minutes'
    WHEN total_intl_minutes <= 30 THEN '21-30 Minutes'
    ELSE '31+ Minutes'
  END AS intl_minutes_range,
  AVG(CAST(churn AS FLOAT)) AS churn_rate
FROM
  dbo.[telecom-churn]
GROUP BY
  CASE
    WHEN total_intl_minutes <= 10 THEN '0-10 Minutes'
    WHEN total_intl_minutes <= 20 THEN '11-20 Minutes'
    WHEN total_intl_minutes <= 30 THEN '21-30 Minutes'
    ELSE '31+ Minutes'
  END
ORDER BY
  MIN(total_intl_minutes);
```

This code investigates if customers using more international minutes are more likely to churn. It categorizes international minute usage and calculates the average churn rate for each range. Analyzing the results can reveal:

   - If customers with higher international minute usage (31+ Minutes) have a significantly higher churn rate, it might suggest a need for more competitive international calling plans to retain them.

**2. Call Charges and Churn Rate:**

The next set of queries analyze churn rate in relation to call charges for daytime, evening, nighttime, and international calls. Here's the structure:

```sql
-- Are customers who are charged more likely to churn?
SELECT
  CASE
    WHEN total_day_charge <= 20 THEN '0-20'
    WHEN total_day_charge <= 30 THEN '21-30'
    WHEN total_day_charge <= 40 THEN '31-40'
    ELSE '41+'
  END AS charge_range_day,
  AVG(CAST(churn AS FLOAT)) AS churn_rate
FROM
  dbo.[telecom-churn]
GROUP BY
  CASE
    WHEN total_day_charge <= 20 THEN '0-20'
    WHEN total_day_charge <= 30 THEN '21-30'
    WHEN total_day_charge <= 40 THEN '31-40'
    ELSE '41+'
  END
ORDER BY
  charge_range_day;
```

Similar logic applies to evening, nighttime, and international charges. These queries categorize customers based on their call charge ranges and calculate the average churn rate for each group. The goal is to understand:

   - If customers with higher charges for a specific call type (daytime, evening, etc.) churn more, it might indicate a need for revised pricing structures or targeted promotions.

## Conclusion
This project showcases the process of transforming raw job data into a cleaner and more analyzable format using SQL. By splitting columns, removing unnecessary data, and conducting various analyses, we were able to extract meaningful insights about job titles, salaries, company ratings, and industry trends. This approach can be applied to similar datasets to facilitate better decision-making and data-driven insights.