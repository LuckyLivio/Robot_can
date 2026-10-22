param(
  [Parameter(Mandatory=$true)][string]$InputDocx,
  [Parameter(Mandatory=$true)][string]$OutputPdf,
  [Parameter(Mandatory=$true)][string]$Receipt
)
$ErrorActionPreference = 'Stop'
$designWord = $null
$designDocument = $null
$designOwned = $false
try {
  $designWord = New-Object -ComObject Word.Application
  if ($designWord.Documents.Count -ne 0) {
    throw 'Refusing a Word automation instance that already has open documents.'
  }
  $designOwned = $true
  $designWord.Visible = $false
  $designWord.DisplayAlerts = 0
  $designDocument = $designWord.Documents.Open([System.IO.Path]::GetFullPath($InputDocx), $false, $false)
  $null = $designDocument.Fields.Update()
  foreach ($designToc in $designDocument.TablesOfContents) { $null = $designToc.Update() }
  foreach ($designToc in $designDocument.TablesOfContents) {
    $designEndRange = $designDocument.Range($designToc.Range.End, $designToc.Range.End)
    $designEndParagraph = $designEndRange.Paragraphs.Item(1)
    if ($designEndParagraph.Range.Text.Trim([char]13, [char]7, [char]32).Length -eq 0) {
      $designEndParagraph.Range.Font.Size = 1
      $designEndParagraph.Format.LineSpacingRule = 4
      $designEndParagraph.Format.LineSpacing = 1
      $designEndParagraph.Format.SpaceBefore = 0
      $designEndParagraph.Format.SpaceAfter = 0
    }
  }
  $designDocument.Repaginate()
  foreach ($designToc in $designDocument.TablesOfContents) { $null = $designToc.UpdatePageNumbers() }
  $designDocument.Save()
  $designDocument.ExportAsFixedFormat([System.IO.Path]::GetFullPath($OutputPdf), 17)
  $designReceipt = [ordered]@{
    passed = $true
    backend = 'Microsoft Word native PDF export on Windows'
    page_count = $designDocument.ComputeStatistics(2)
    word_count = $designDocument.ComputeStatistics(0)
    office_math_count = $designDocument.OMaths.Count
    toc_count = $designDocument.TablesOfContents.Count
    fields_updated = $true
    source = [System.IO.Path]::GetFullPath($InputDocx)
    pdf = [System.IO.Path]::GetFullPath($OutputPdf)
  }
  $designReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding utf8
  $designReceipt | ConvertTo-Json -Compress
}
finally {
  if ($null -ne $designDocument) { $designDocument.Close($false); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($designDocument) }
  if ($null -ne $designWord) {
    if ($designOwned) { $designWord.Quit() }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($designWord)
  }
}
