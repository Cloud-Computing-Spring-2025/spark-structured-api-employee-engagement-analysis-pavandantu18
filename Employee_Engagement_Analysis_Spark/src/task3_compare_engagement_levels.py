# task3_compare_engagement_levels.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, round as spark_round
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType

def initialize_spark(app_name="Task3_Compare_Engagement_Levels"):
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
    # You can also use a StructType schema if desired.
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def map_engagement_level(df):
    """
    Map EngagementLevel from categorical to numerical values.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame with an additional column for numerical EngagementScore.
    """
    # Mapping:
    # 'Low' -> 1, 'Medium' -> 2, 'High' -> 3
    df_with_score = df.withColumn("EngagementScore", 
                                  when(col("EngagementLevel") == "Low", 1)
                                  .when(col("EngagementLevel") == "Medium", 2)
                                  .when(col("EngagementLevel") == "High", 3)
                                  .otherwise(0))
    return df_with_score

def compare_engagement_levels(df):
    """
    Compare engagement levels across different job titles and identify the top-performing job title.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data with numerical EngagementScore.

    Returns:
        DataFrame: DataFrame containing JobTitle and their average EngagementScore.
    """
    # Group by JobTitle, calculate the average EngagementScore, and round the result to two decimal places.
    result_df = df.groupBy("JobTitle").agg(spark_round(avg("EngagementScore"), 2).alias("AvgEngagementScore"))
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
    Main function to execute Task 3.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-pavandantu18/Employee_Engagement_Analysis_Spark/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-pavandantu18/Employee_Engagement_Analysis_Spark/outputs/task3/engagement_levels_job_titles.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Map EngagementLevel to a numerical EngagementScore
    df_mapped = map_engagement_level(df)
    
    # Compare engagement levels across job titles
    result_df = compare_engagement_levels(df_mapped)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
