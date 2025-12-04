function Invoke-DeployScript {
    [CmdletBinding()]
    param(
        # 添加参数
        [Parameter(Mandatory = $true)]
        [string[]]$SourcePaths,
        
        [Parameter(Mandatory = $false)]
        [string]$destinationPath = "./../SmartPlusDeploySDK/"
    )
    
    begin {
        Write-Verbose "开始执行脚本进行部署拷贝"
    }
    
    process {
        foreach ($sourcePath in $SourcePaths) {
            Write-Host "开始拷贝路径: $path" -ForegroundColor Yellow
            # =====================================================
            # 复制目录并保持完整目录结构的PowerShell脚本
            # 自动创建目标目录（如果不存在），递归复制所有子目录和文件
            # 设置源目录和目标目录路径
            # $sourcePath = "./lib/"  # 请修改为实际的源目录路径
            # $sourcePath = "./Smartplue/include/"
            # $sourcePath = "./bin/"
            # $destinationPath = "./../SmartPlusDeploySDK/"  # 请修改为实际的目标目录路径

            # 检查源目录是否存在
            if (-not (Test-Path $sourcePath -PathType Container)) {
                Write-Host "错误：源目录 '$sourcePath' 不存在或不是目录！" -ForegroundColor Red
                exit 1
            }

            # 获取当前工作目录
            $currentDir = Get-Location | Select-Object -ExpandProperty Path

            # 将源路径转换为绝对路径
            $sourcePath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($currentDir, $sourcePath))
            Write-Host "源目录: $sourcePath" -ForegroundColor White

            # 将目标路径转换为绝对路径
            $destinationPath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($currentDir, $destinationPath))
            Write-Host "目标路径: $destinationPath" -ForegroundColor White

            # 获取源目录的名称
            # $sourceDirName = Split-Path $sourcePath -Leaf
            # 计算相对路径
            $sourceDirName = [System.IO.Path]::GetRelativePath($currentDir, $sourcePath)
            Write-Host "源目录名称: $sourceDirName" -ForegroundColor Cyan

            # 计算最终的目标路径（包含源目录名称）
            $finalDestinationPath = Join-Path $destinationPath $sourceDirName
            Write-Host "最终目标路径: $finalDestinationPath" -ForegroundColor Cyan

            # 确保目标目录的父目录存在
            $parentDestinationPath = Split-Path $finalDestinationPath -Parent
            if (-not (Test-Path $parentDestinationPath)) {
                Write-Host "创建父目录: $parentDestinationPath" -ForegroundColor Yellow
                New-Item -ItemType Directory -Path $parentDestinationPath -Force | Out-Null
            }

            # 显示源目录结构
            Write-Host "`n源目录结构:" -ForegroundColor Green
            Get-ChildItem -Path $sourcePath -Recurse | ForEach-Object {
                $relativePath = $_.FullName.Substring($sourcePath.Length)
                if (-not [string]::IsNullOrEmpty($relativePath)) {
                    $depth = $relativePath.TrimStart('\').Split('\').Length - 1
                    $indent = "  " * $depth
                    if ($_.PSIsContainer) {
                        Write-Host "$indent[$($_.Name)]" -ForegroundColor Blue
                    }
                    else {
                        Write-Host "$indent$($_.Name)" -ForegroundColor Gray
                    }
                }
            }

            Write-Host "`n准备复制..." -ForegroundColor Yellow
            Write-Host "源目录: $sourcePath" -ForegroundColor White
            Write-Host "目标位置: $finalDestinationPath" -ForegroundColor White

            # 方法3：使用 PowerShell Copy-Item（最终备用方法）
            Write-Host "`最终验证并确保复制完整..." -ForegroundColor Cyan
            try {
                Write-Host "使用 PowerShell Copy-Item..." -ForegroundColor Yellow
  
                # 使用 Copy-Item 递归复制
                Copy-Item -Path $sourcePath -Destination $finalDestinationPath -Recurse -Force
        
                Write-Host "PowerShell Copy-Item 复制完成！" -ForegroundColor Green
            }
            catch {
                Write-Host "PowerShell Copy-Item 失败: $_" -ForegroundColor Red
            }

            # 验证复制结果
            Write-Host "`验证复制结果..." -ForegroundColor Cyan

            if (Test-Path $finalDestinationPath) {
                # 获取所有文件
                $sourceFiles = Get-ChildItem -Path $sourcePath -Recurse -File
                $destFiles = Get-ChildItem -Path $finalDestinationPath -Recurse -File
    
                $sourceFileCount = $sourceFiles.Count
                $destFileCount = $destFiles.Count
    
                Write-Host "源目录文件数: $sourceFileCount" -ForegroundColor White
                Write-Host "目标目录文件数: $destFileCount" -ForegroundColor White
    
                if ($sourceFileCount -eq $destFileCount) {
                    Write-Host "✓ 复制成功！文件数量一致。" -ForegroundColor Green
                }
                else {
                    Write-Host "⚠ 警告：源目录和目标目录文件数量不一致！" -ForegroundColor Yellow
                    Write-Host "差异: $($destFileCount - $sourceFileCount) 个文件" -ForegroundColor Yellow
        
                    # 显示缺少的文件
                    if ($sourceFileCount -gt $destFileCount) {
                        Write-Host "`n可能缺少以下文件:" -ForegroundColor Yellow
                        $sourceFileNames = $sourceFiles | ForEach-Object { 
                            $_.FullName.Substring($sourcePath.Length).TrimStart('\')
                        }
                        $destFileNames = $destFiles | ForEach-Object { 
                            $_.FullName.Substring($finalDestinationPath.Length).TrimStart('\')
                        }
            
                        $missingFiles = $sourceFileNames | Where-Object { $_ -notin $destFileNames }
                        foreach ($file in $missingFiles | Select-Object -First 10) {
                            Write-Host "  $file" -ForegroundColor Red
                        }
            
                        if ($missingFiles.Count -gt 10) {
                            Write-Host "  ... 还有 $($missingFiles.Count - 10) 个文件未显示" -ForegroundColor Red
                        }
                    }
                }
    
                # 显示目标目录结构
                Write-Host "`n目标目录结构:" -ForegroundColor Green
                $allItems = Get-ChildItem -Path $finalDestinationPath -Recurse
                if ($allItems.Count -eq 0) {
                    Write-Host "  (空目录)" -ForegroundColor Gray
                }
                else {
                    foreach ($item in $allItems) {
                        $relativePath = $item.FullName.Substring($finalDestinationPath.Length)
                        if (-not [string]::IsNullOrEmpty($relativePath)) {
                            $depth = $relativePath.TrimStart('\').Split('\').Length - 1
                            $indent = "  " * $depth
                            if ($item.PSIsContainer) {
                                Write-Host "$indent[$($item.Name)]" -ForegroundColor Blue
                            }
                            else {
                                $sizeKB = if ($item.Length -gt 0) { [Math]::Round($item.Length / 1KB, 2) } else { 0 }
                                Write-Host "$indent$($item.Name) ($sizeKB KB)" -ForegroundColor Gray
                            }
                        }
                    }
                }
            }
            else {
                Write-Host "错误：目标目录 '$finalDestinationPath' 不存在，复制失败！" -ForegroundColor Red
                exit 1
            }

            Write-Host "`脚本执行完成！" -ForegroundColor Green
            Write-Host "复制位置: $finalDestinationPath" -ForegroundColor Green

            # =====================================================
            Write-Host "完成拷贝路径: $path" -ForegroundColor Yellow
        }
    }
    
    end {
        Write-Verbose "进行部署拷贝执行完成"
    }
}

# =====================================================
$SourceMultiPaths = @(
    "./lib/",
    "./bin/",
    "./Smartplue/include/",
    "./Smartplue1/include/",
    "./Smartplue2/include/",
    "./Smartplue3/include/",
    "./Smartplue4/include/"
)
$destinationPath = "./../SmartPlusDeploySDK/"  # 请修改为实际的目标目录路径

Invoke-DeployScript -SourcePaths $SourceMultiPaths -destinationPath $destinationPath
