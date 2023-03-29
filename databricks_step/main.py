from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
import argparse
import os
import json
import mlflow


def parse():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')
    parser.add_argument('--input')
    parser.add_argument('--output')

    # The AZUREML_SCRIPT_DIRECTORY_NAME argument will be filled in if the DatabricksStep
    # was run using a local source_directory and python_script_name
    parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')

    # Remaining arguments are filled in for all databricks jobs and can be used to build the run context
    parser.add_argument('--AZUREML_RUN_TOKEN')
    parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
    parser.add_argument('--AZUREML_RUN_ID')
    parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
    parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
    parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
    parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
    parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
    parser.add_argument('--AZUREML_WORKSPACE_ID')
    parser.add_argument('--AZUREML_EXPERIMENT_ID')

    (args, extra_args) = parser.parse_known_args()
    os.environ['AZUREML_RUN_TOKEN'] = args.AZUREML_RUN_TOKEN
    os.environ['AZUREML_RUN_TOKEN_EXPIRY'] = args.AZUREML_RUN_TOKEN_EXPIRY
    os.environ['AZUREML_RUN_ID'] = args.AZUREML_RUN_ID
    os.environ['AZUREML_ARM_SUBSCRIPTION'] = args.AZUREML_ARM_SUBSCRIPTION
    os.environ['AZUREML_ARM_RESOURCEGROUP'] = args.AZUREML_ARM_RESOURCEGROUP
    os.environ['AZUREML_ARM_WORKSPACE_NAME'] = args.AZUREML_ARM_WORKSPACE_NAME
    os.environ['AZUREML_ARM_PROJECT_NAME'] = args.AZUREML_ARM_PROJECT_NAME
    os.environ['AZUREML_SERVICE_ENDPOINT'] = args.AZUREML_SERVICE_ENDPOINT
    os.environ['AZUREML_WORKSPACE_ID'] = args.AZUREML_WORKSPACE_ID
    os.environ['AZUREML_EXPERIMENT_ID'] = args.AZUREML_EXPERIMENT_ID
    return args

def setup_access():
    tenant_id = dbutils.secrets.get('mykeyvault','tenant-id')
    client_id =  dbutils.secrets.get('mykeyvault','client-id')
    client_secret = dbutils.secrets.get('mykeyvault','client-secret')
    storage='<storage>'
    os.environ["AZURE_TENANT_ID"] = tenant_id 
    os.environ["AZURE_CLIENT_ID"] = client_id 
    os.environ["AZURE_CLIENT_SECRET"] = client_secret 

    spark.conf.set(f"fs.azure.account.auth.type.{storage}.dfs.core.windows.net", "OAuth")
    spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage}.dfs.core.windows.net", client_id)
    spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage}.dfs.core.windows.net", client_secret)
    spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage}.dfs.core.windows.net", f"https://login.microsoftonline.com/{tenant_id}/oauth2/token")

def setup_mlflow():
    endpoint = os.environ['AZUREML_SERVICE_ENDPOINT'].split('://')[-1]
    uri=(f'azureml://{endpoint}/mlflow/v1.0'
         f'/subscriptions/{os.environ["AZUREML_ARM_SUBSCRIPTION"]}'
         f'/resourceGroups/{os.environ["AZUREML_ARM_RESOURCEGROUP"]}'
         f'/providers/Microsoft.MachineLearningServices/workspaces/{os.environ["AZUREML_ARM_WORKSPACE_NAME"]}')
    experiment_id = os.environ['AZUREML_EXPERIMENT_ID']
    run_id = os.environ['AZUREML_RUN_ID']
    os.environ['MLFLOW_TRACKING_URI'] = uri
    os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
    os.environ['MLFLOW_RUN_ID'] = run_id

    return uri, experiment_id, run_id

def main(args):
    uri, experiment_id, run_id = setup_mlflow()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_id=run_id):
        mlflow.autolog()
        df = spark.read.parquet(args.input)
        data = df.select("passengerCount", "tripDistance", "fareAmount")
        assembler = VectorAssembler(inputCols=["passengerCount", "tripDistance"], outputCol="features")
        data = assembler.transform(data)
        (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=100)
        lr = LinearRegression(featuresCol="features", labelCol="fareAmount")
        model = lr.fit(trainingData)
        predictions = model.transform(testData)
        evaluator = RegressionEvaluator(labelCol="fareAmount", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("RMSE: %.3f" % rmse)
        newData = spark.createDataFrame([(2, 5), (3, 7)], ["passengerCount", "tripDistance"])
        newData = assembler.transform(newData)
        predictions = model.transform(newData)
        coeffs = []
        for x in model.coefficients:
            coeffs.append(x)
        print(f"coefficients are: {coeffs}")
    # End of mlflow run
    agg = df.groupBy("passengerCount").agg(
        F.avg("tripDistance").alias('avgTripDistance')
    )
    agg.write.format("parquet").save(args.output)



if __name__=='__main__':
    args = parse()
    setup_access()
    main(args)
