# Azure Machine Learning - Azure Databricks Integration

This tutorial walks you through a simple scenario of creating 
a pipeline in Azure ML that uses Azure Databricks to perform
some spark tasks and passes the processed data to the rest
of the pipeline. 

## Prerequisites

To perform this tutorial, you need the following resources.
- An Azure ML workspace
- An Azure Databricks workspace
- An Azure Data Lake Storage (ADLS gen2) account
    - Grant your Azure ML's managed identity `Storage Blob Data Contributor` access to this account.
    > Note that Azure ML by default creates a Blob storage (not ADLS gen2) and in many situations, that might be enough. However, since we are working with Databricks and Spark, an ADLS is preferred. 
    - Create a container for your data (recommended name: `data`) 
- A Service Principal with the following access
    - `Contributor` access to Azure ML Workspace
    - `Contributor` access to Azure Databricks Workspace
    - `Storage Blob Data Contributor` access to ADLS storage
    > To create a service principal, using portal, go to Azure Active Directory > App registration ([see here](https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal)).
    > 
    > To create a SP using `az` cli, use the `az ad sp create-for-rbac` command ([see here](https://learn.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli)).

> In this tutorial, we assume all resources are under one resource group. This simplifies access. 
> If your AML and Databricks workspaces are in different resource groups, make sure to adjust the code.

## Steps

1. In your Azure ML workspace, 
    - create a compute cluster (recommended name `amlcluster`). This will be used by submitted Azure ML jobs.
    - create a compute instance. This will be used for creating and submitting the pipeline and interactive work with notebooks.
    - Connect your Azure Databricks to Azure ML via "Attached Cluster" menu (recommended name `adbcluster`).
    To get an access token, follow these steps:
        * Launch Databricks workspace.
        * On the top-right, click on your user name and select "User settings".
        * Click on "Generate new token".
        * Give your token a name and click "generate".
        * Grab the new token and paste in in your AML window.
1. In you Azure ML's attached Key Vault:
    - First, give yourself permission to all secrets. You can do this via `Access policies` or `Access Control (IAM)`. 
    - Then,  insert the service principal information into the Azure ML's Key Vault as secrets.
    We recomment to use `client-id`, `tenant-id`, and `client-secret` as the key names.
1. In your Azure Databricks workspace, set up secret scope, which connects to an Azure Key Vault to read secrets. You can find the steps [here](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes). We recommend connecting to the same Key Vault that Azure ML uses.
    - Take note of the scope name you assign. (In this repo, we assume the name is `mykeyvault`.)
1. Clone or copy this repo in your Azure ML.
    - Using your compute instance, open a terminal.
    - Run `git clone <address of this repo>` in terminal.
1. Run `copy-data.ipynb` notebook. This notebook performs the following: 
    > Update the variables in the first block as needed.
    - Downloads the New York taxi dataset into the ADLS storage account.
    - Creates a Datastore in Azure ML pointing to the ADLS.
    - Creates two datasets:
        - `nyctlcraw`, which points to the raw data copied in the step above;
        - `nyctlccleaned`, which points to the cleaned data. This data does not exist yet, but will be created by Databricks.
1. Run `pipeline.ipynb` notebook to submit a pipeline job with two steps.
    > Update the variables in the first block as needed.
    - Step 1 submits the python script in `databricks_step` folder to your Azure Databricks cluster.
    This script, reads the data that we generated in previous step and processes it into another dataset that can be used for training.
    - Step 2 submits the python script in `azureml_step` folder to your Azure ML cluster.
    This script reads the processed data from Step 1 and performs training on it.
1. Check the status of your job:
    - To see overal pipeline's status, use Jobs view from Azure ML.
    - To see internal logs of the Databricks step, double-click on the databricks step and click on the link.  


## References
- [Azure ML `DatabricksStep` ](https://learn.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.databricks_step.databricksstep?view=azure-ml-py)
- [Example notebook for submitting jobs to Databricks](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-use-databricks-as-compute-target.ipynb)
