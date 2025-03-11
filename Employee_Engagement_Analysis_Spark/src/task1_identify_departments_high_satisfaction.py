# task1_identify_departments_high_satisfaction.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 50% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame containing departments meeting the criteria with their respective percentages.
    """
    # 1. Filter employees with SatisfactionRating > 4 and EngagementLevel == 'High'
    filtered_df = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))
    
    # 2. Calculate total employees per department
    total_by_dept = df.groupBy("Department").count().withColumnRenamed("count", "TotalEmployees")
    
    # 3. Calculate high satisfaction employees per department
    high_sat_by_dept = filtered_df.groupBy("Department").count().withColumnRenamed("count", "HighSatCount")
    
    # 4. Join the two DataFrames and compute the percentage
    joined_df = total_by_dept.join(high_sat_by_dept, on="Department", how="left").fillna(0)
    percentage_df = joined_df.withColumn("Percentage", spark_round((col("HighSatCount") / col("TotalEmployees")) * 100, 2))
    
    # 5. Filter for departments with more than 7.5% of such employees and select desired columns
    result_df = percentage_df.filter(col("Percentage") > 7.5).select("Department", "Percentage")
    
    return result_df

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-pavandantu18/Employee_Engagement_Analysis_Spark/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-pavandantu18/Employee_Engagement_Analysis_Spark/outputs/task1/departments_high_satisfaction.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
