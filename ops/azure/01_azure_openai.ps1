<#
Creates an Azure OpenAI resource and one chat model deployment for the Regulatory Copilot.

Run once, from PowerShell, after `az login`:

  # 1. See which models you can deploy in a region (pick a small model with JSON-schema output):
  .\ops\azure\01_azure_openai.ps1 -Location centralindia -ListModels

  # 2. Create the resource and the deployment:
  .\ops\azure\01_azure_openai.ps1 -Location centralindia -AccountName aoai-regcopilot-<your-suffix> `
      -ModelName <model> -ModelVersion <version>

It prints the three lines to add to .env. The account name must be globally unique (it becomes
https://<AccountName>.openai.azure.com).
#>
param(
    [string]$ResourceGroup = "rg-regulatory-copilot",
    [string]$Location = "centralindia",
    [string]$AccountName,
    [string]$ModelName,
    [string]$ModelVersion,
    [string]$DeploymentName = "copilot-chat",
    [string]$Sku = "GlobalStandard",
    [int]$Capacity = 10,            # thousands of tokens per minute
    [switch]$ListModels
)
$ErrorActionPreference = "Stop"
# az and gcloud are native commands: PowerShell does not stop on their failures by itself.
function Assert-Ok([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit code $LASTEXITCODE)" } }

if ($ListModels) {
    az cognitiveservices model list --location $Location `
        --query "[?model.format=='OpenAI'].{model:model.name, version:model.version, skus:join(',', model.skus[].name)}" `
        --output table
    exit 0
}
if (-not ($AccountName -and $ModelName -and $ModelVersion)) {
    throw "Pass -AccountName, -ModelName and -ModelVersion (run with -ListModels first to see the options)."
}

Write-Host "Resource group $ResourceGroup in $Location"
az group create --name $ResourceGroup --location $Location --output none
Assert-Ok "Resource group"

Write-Host "Azure OpenAI resource $AccountName"
$exists = az cognitiveservices account list --resource-group $ResourceGroup --query "[?name=='$AccountName'] | length(@)" --output tsv
if ($exists -eq "0") {
    az cognitiveservices account create --name $AccountName --resource-group $ResourceGroup --location $Location `
        --kind OpenAI --sku S0 --custom-domain $AccountName --yes --output none
    Assert-Ok "Azure OpenAI resource"
}

Write-Host "Deployment $DeploymentName -> $ModelName $ModelVersion ($Sku, $Capacity K tokens/min)"
az cognitiveservices account deployment create --name $AccountName --resource-group $ResourceGroup `
    --deployment-name $DeploymentName --model-name $ModelName --model-version $ModelVersion --model-format OpenAI `
    --sku-name $Sku --sku-capacity $Capacity --output none
Assert-Ok "Model deployment"

$endpoint = az cognitiveservices account show --name $AccountName --resource-group $ResourceGroup --query properties.endpoint --output tsv
$key = az cognitiveservices account keys list --name $AccountName --resource-group $ResourceGroup --query key1 --output tsv

Write-Host ""
Write-Host "Add these to .env (do not commit .env):"
Write-Host "AZURE_OPENAI_ENDPOINT=$endpoint"
Write-Host "AZURE_OPENAI_DEPLOYMENT=$DeploymentName"
Write-Host "AZURE_OPENAI_API_KEY=$key"
Write-Host ""
Write-Host "Then look up the deployment's input and output prices per 1M tokens and add:"
Write-Host "AZURE_OPENAI_INPUT_PRICE_PER_MTOK_USD=<input price>"
Write-Host "AZURE_OPENAI_OUTPUT_PRICE_PER_MTOK_USD=<output price>"
