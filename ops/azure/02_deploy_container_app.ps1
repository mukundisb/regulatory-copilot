<#
Runs the Regulatory Copilot API on Azure Container Apps, from the same image CI pushes to
Google Artifact Registry. The image is copied into an Azure Container Registry first, because
Container Apps needs a registry it can always pull from when it scales up from zero.

Azure differences from Cloud Run:
  - LLM_PROVIDER=azure_openai: recommendations come from your Azure OpenAI deployment.
  - CLASSIFIER_BACKEND=tfidf: no cross-cloud call to Vertex AI, so no GCP credentials on Azure.
  - Scales to zero (min replicas 0); the first request after idle waits for a cold start.

Needs: az (logged in), gcloud (logged in, with read access to Artifact Registry), and .env values
from 01_azure_openai.ps1 in the environment or passed as parameters.

  .\ops\azure\02_deploy_container_app.ps1 -AcrName acrregcopilot<suffix> -ImageTag <git sha or latest>

Rerun the same command to roll out a new image tag; existing resources are reused.
#>
param(
    [string]$ResourceGroup = "rg-regulatory-copilot",
    [string]$Location = "centralindia",
    [Parameter(Mandatory = $true)][string]$AcrName,       # globally unique, letters and digits only
    [string]$EnvironmentName = "cae-regulatory-copilot",
    [string]$AppName = "regulatory-copilot",
    [string]$GcpImage = "asia-south1-docker.pkg.dev/regulatory-copilot-506507/regulatory-copilot/regulatory-copilot",
    [string]$ImageTag = "latest",
    [string]$AzureOpenAIEndpoint = $env:AZURE_OPENAI_ENDPOINT,
    [string]$AzureOpenAIDeployment = $env:AZURE_OPENAI_DEPLOYMENT,
    [string]$AzureOpenAIApiKey = $env:AZURE_OPENAI_API_KEY,
    [string]$InputPricePerMTok = $env:AZURE_OPENAI_INPUT_PRICE_PER_MTOK_USD,
    [string]$OutputPricePerMTok = $env:AZURE_OPENAI_OUTPUT_PRICE_PER_MTOK_USD
)
$ErrorActionPreference = "Stop"
# az and gcloud are native commands: PowerShell does not stop on their failures by itself.
function Assert-Ok([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit code $LASTEXITCODE)" } }

# Fill missing values from .env in the repository root, so the key never has to be typed on the command line.
$envFile = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) ".env"
if (Test-Path $envFile) {
    $dotenv = @{}
    Get-Content $envFile | Where-Object { $_ -match '^\s*([A-Z0-9_]+)\s*=\s*(.*)$' } | ForEach-Object {
        $dotenv[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
    if (-not $AzureOpenAIEndpoint)   { $AzureOpenAIEndpoint   = $dotenv["AZURE_OPENAI_ENDPOINT"] }
    if (-not $AzureOpenAIDeployment) { $AzureOpenAIDeployment = $dotenv["AZURE_OPENAI_DEPLOYMENT"] }
    if (-not $AzureOpenAIApiKey)     { $AzureOpenAIApiKey     = $dotenv["AZURE_OPENAI_API_KEY"] }
    if (-not $InputPricePerMTok)     { $InputPricePerMTok     = $dotenv["AZURE_OPENAI_INPUT_PRICE_PER_MTOK_USD"] }
    if (-not $OutputPricePerMTok)    { $OutputPricePerMTok    = $dotenv["AZURE_OPENAI_OUTPUT_PRICE_PER_MTOK_USD"] }
}
if (-not ($AzureOpenAIEndpoint -and $AzureOpenAIDeployment -and $AzureOpenAIApiKey)) {
    throw "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY are needed (run 01_azure_openai.ps1)."
}

Write-Host "== Azure CLI extension and resource providers"
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait --output none
az provider register --namespace Microsoft.OperationalInsights --wait --output none
az group create --name $ResourceGroup --location $Location --output none
Assert-Ok "Resource group"

Write-Host "== Container registry $AcrName"
$acrExists = az acr list --resource-group $ResourceGroup --query "[?name=='$AcrName'] | length(@)" --output tsv
if ($acrExists -eq "0") {
    az acr create --name $AcrName --resource-group $ResourceGroup --location $Location --sku Basic --admin-enabled true --output none
    Assert-Ok "Container registry"
}

Write-Host "== Copying $GcpImage`:$ImageTag into $AcrName"
$gcpToken = gcloud auth print-access-token
Assert-Ok "gcloud access token"
az acr import --name $AcrName --source "$GcpImage`:$ImageTag" --image "regulatory-copilot:$ImageTag" `
    --username oauth2accesstoken --password $gcpToken --force --output none
Assert-Ok "Image import"
$image = "$AcrName.azurecr.io/regulatory-copilot:$ImageTag"

Write-Host "== Container Apps environment $EnvironmentName (creates a Log Analytics workspace)"
$envExists = az containerapp env list --resource-group $ResourceGroup --query "[?name=='$EnvironmentName'] | length(@)" --output tsv
if ($envExists -eq "0") {
    az containerapp env create --name $EnvironmentName --resource-group $ResourceGroup --location $Location --output none
    Assert-Ok "Container Apps environment"
}

$envVars = @(
    "LLM_PROVIDER=azure_openai",
    "CLASSIFIER_BACKEND=tfidf",
    "AZURE_OPENAI_ENDPOINT=$AzureOpenAIEndpoint",
    "AZURE_OPENAI_DEPLOYMENT=$AzureOpenAIDeployment",
    "AZURE_OPENAI_API_KEY=secretref:aoai-key"
)
if ($InputPricePerMTok -and $OutputPricePerMTok) {
    $envVars += "AZURE_OPENAI_INPUT_PRICE_PER_MTOK_USD=$InputPricePerMTok"
    $envVars += "AZURE_OPENAI_OUTPUT_PRICE_PER_MTOK_USD=$OutputPricePerMTok"
}

$appExists = az containerapp list --resource-group $ResourceGroup --query "[?name=='$AppName'] | length(@)" --output tsv
if ($appExists -eq "0") {
    Write-Host "== Creating container app $AppName"
    $acrUser = az acr credential show --name $AcrName --query username --output tsv
    $acrPassword = az acr credential show --name $AcrName --query "passwords[0].value" --output tsv
    az containerapp create --name $AppName --resource-group $ResourceGroup --environment $EnvironmentName `
        --image $image --target-port 8000 --ingress external `
        --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 `
        --registry-server "$AcrName.azurecr.io" --registry-username $acrUser --registry-password $acrPassword `
        --secrets "aoai-key=$AzureOpenAIApiKey" --env-vars $envVars --output none
    Assert-Ok "Container app create"
} else {
    Write-Host "== Updating container app $AppName to $image"
    az containerapp secret set --name $AppName --resource-group $ResourceGroup --secrets "aoai-key=$AzureOpenAIApiKey" --output none
    az containerapp update --name $AppName --resource-group $ResourceGroup --image $image --set-env-vars $envVars --output none
    Assert-Ok "Container app update"
}

$fqdn = az containerapp show --name $AppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn --output tsv
$workspace = az containerapp env show --name $EnvironmentName --resource-group $ResourceGroup `
    --query properties.appLogsConfiguration.logAnalyticsConfiguration.customerId --output tsv
$url = "https://$fqdn"

Write-Host "== Waiting for /health (the first start downloads the embedding model and builds the index)"
$healthy = $false
foreach ($attempt in 1..30) {
    try {
        $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 60
        Write-Host ("Health: status={0} llm_provider={1} backend={2}" -f $health.status, $health.llm_provider, $health.backend)
        $healthy = $true
        break
    } catch {
        Start-Sleep -Seconds 20
    }
}
if (-not $healthy) { Write-Warning "No healthy answer after 10 minutes. Check: az containerapp logs show -n $AppName -g $ResourceGroup --follow" }

Write-Host ""
Write-Host "URL:                     $url"
Write-Host "Log Analytics workspace: $workspace"
Write-Host "Measure it:  python scripts/measure_live.py send --url $url --tag azure"
