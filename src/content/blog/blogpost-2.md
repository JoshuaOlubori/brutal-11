---
title: Transforming and Analyzing Job Data Using SQL
pubDate: 12/31/2022 12:05
author: "Tunde Mark"
tags:
  - SQL
  - WebDev
  - Programming
imgUrl: '/src/assets/blog_covers/sql.png'
description: Lorem markdownum longo os thyrso telum, continet servat fetus nymphae, vox nocte sedesque, decimo. Omnia esse, quam sive; conplevit illis indestrictus admovit dedit sub quod protectus, impedit non.
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

## Conclusion
This project showcases the process of transforming raw job data into a cleaner and more analyzable format using SQL. By splitting columns, removing unnecessary data, and conducting various analyses, we were able to extract meaningful insights about job titles, salaries, company ratings, and industry trends. This approach can be applied to similar datasets to facilitate better decision-making and data-driven insights.