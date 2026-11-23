<#
Deletes everything the Azure scripts created: the resource group with the container app, its
environment and Log Analytics workspace, the container registry and the Azure OpenAI resource.
Run it when you are done, so nothing keeps billing. It cannot be undone.

  .\ops\azure\03_teardown.ps1
#>
param([string]$ResourceGroup = "rg-regulatory-copilot")
$ErrorActionPreference = "Stop"
az group delete --name $ResourceGroup --yes --no-wait
Write-Host "Deletion of $ResourceGroup started. Check with: az group exists --name $ResourceGroup"
Write-Host "Azure OpenAI resources are soft-deleted; purge to reuse the same account name:"
Write-Host "  az cognitiveservices account list-deleted --output table"
