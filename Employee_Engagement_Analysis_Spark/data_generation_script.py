import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of records
num_records = 100

# Define Departments and Job Titles
departments = ['Sales', 'IT', 'HR', 'Finance', 'Marketing', 'Support']
job_titles = ['Manager', 'Executive', 'Analyst', 'Developer', 'Coordinator', 'Support']

# Initialize lists to store data
employee_ids = list(range(1, num_records + 1))
department_list = []
job_title_list = []
satisfaction_rating_list = []
engagement_level_list = []
reports_concerns_list = []
provided_suggestions_list = []

# Define departments that need to have >50% high satisfaction and engagement
target_departments = ['Finance', 'Marketing']

# Calculate number of departments
num_departments = len(departments)

# Assign number of employees per department (approx equal)
employees_per_dept = num_records // num_departments
remaining = num_records % num_departments

dept_employee_counts = {dept: employees_per_dept for dept in departments}
for i in range(remaining):
    dept_employee_counts[departments[i]] += 1

# First, assign departments and job titles
for dept in departments:
    count = dept_employee_counts[dept]
    for _ in range(count):
        department_list.append(dept)
        # Assign JobTitle based on department
        if dept == 'Finance':
            # Finance has more Managers and Analysts
            job_title = np.random.choice(['Manager', 'Analyst'], p=[0.4, 0.6])
        elif dept == 'Marketing':
            # Marketing has more Executives and Coordinators
            job_title = np.random.choice(['Executive', 'Coordinator'], p=[0.5, 0.5])
        elif dept == 'IT':
            # IT has more Developers and Managers
            job_title = np.random.choice(['Developer', 'Manager'], p=[0.7, 0.3])
        elif dept == 'Sales':
            # Sales has more Executives and Managers
            job_title = np.random.choice(['Executive', 'Manager'], p=[0.6, 0.4])
        elif dept == 'HR':
            # HR has more Executives and Analysts
            job_title = np.random.choice(['Executive', 'Analyst'], p=[0.5, 0.5])
        elif dept == 'Support':
            # Support has more Support and Coordinators
            job_title = np.random.choice(['Support', 'Coordinator'], p=[0.7, 0.3])
        job_title_list.append(job_title)

# Create a DataFrame to manipulate data easily
df = pd.DataFrame({
    'EmployeeID': employee_ids,
    'Department': department_list,
    'JobTitle': job_title_list
})

# Function to assign SatisfactionRating
def assign_satisfaction(dept, job_title):
    if dept in target_departments:
        # Higher satisfaction ratings for target departments
        return np.random.choice([5,4,3,2,1], p=[0.5, 0.3, 0.1, 0.05, 0.05])
    else:
        # General distribution
        return np.random.choice([5,4,3,2,1], p=[0.2, 0.3, 0.3, 0.15, 0.05])

df['SatisfactionRating'] = df.apply(lambda row: assign_satisfaction(row['Department'], row['JobTitle']), axis=1)

# Function to assign EngagementLevel
def assign_engagement(dept, job_title):
    if job_title == 'Manager':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.7, 0.2, 0.1])
    elif job_title == 'Executive':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.6, 0.3, 0.1])
    elif job_title == 'Developer':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.4, 0.4, 0.2])
    elif job_title == 'Analyst':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
    elif job_title == 'Coordinator':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.5, 0.3])
    elif job_title == 'Support':
        return np.random.choice(['High', 'Medium', 'Low'], p=[0.1, 0.3, 0.6])
    else:
        return 'Medium'

df['EngagementLevel'] = df.apply(lambda row: assign_engagement(row['Department'], row['JobTitle']), axis=1)

# Assign ReportsConcerns
df['ReportsConcerns'] = np.random.choice([True, False], size=num_records, p=[0.3, 0.7])

# Ensure that at least 25% of employees have SatisfactionRating >=4 and ProvidedSuggestions == False
# Calculate number needed
num_valued_no_suggestions = 25
# Randomly select employees to fulfill this condition
valued_indices = df[(df['SatisfactionRating'] >=4)].index.tolist()
if len(valued_indices) < num_valued_no_suggestions:
    raise ValueError("Not enough employees with SatisfactionRating >=4 to assign ProvidedSuggestions=False")
selected_indices = np.random.choice(valued_indices, size=num_valued_no_suggestions, replace=False)

# Initialize ProvidedSuggestions as True with 70% probability, False otherwise
df['ProvidedSuggestions'] = np.random.choice([True, False], size=num_records, p=[0.7, 0.3])

# Set ProvidedSuggestions=False for selected_indices
df.loc[selected_indices, 'ProvidedSuggestions'] = False

# Optionally, adjust other records to maintain overall balance
# For example, ensure that overall proportion of ProvidedSuggestions=False is around 30%
current_false = df['ProvidedSuggestions'].value_counts().get(False, 0)
desired_false = int(0.3 * num_records)
additional_false_needed = desired_false - current_false
if additional_false_needed > 0:
    remaining_indices = df[~df.index.isin(selected_indices) & (df['ProvidedSuggestions'] == True)].index
    if len(remaining_indices) >= additional_false_needed:
        indices_to_flip = np.random.choice(remaining_indices, size=additional_false_needed, replace=False)
        df.loc[indices_to_flip, 'ProvidedSuggestions'] = False
elif additional_false_needed < 0:
    # Need to flip some False to True
    excess = abs(additional_false_needed)
    false_indices = df[df['ProvidedSuggestions'] == False].index.tolist()
    indices_to_flip = np.random.choice(false_indices, size=excess, replace=False)
    df.loc[indices_to_flip, 'ProvidedSuggestions'] = True

# Save the DataFrame to CSV
df.to_csv('employee_data.csv', index=False)

print("employee_data.csv has been generated successfully.")

# Optional: Display the first few rows
print(df.head(10))
