# we will run the executable "\gc-polyscope-project-template\build\bin\Release\gc_project.exe" with the mesh path as an argument
# This is to categorize the already flitered files based on: %faces-violating-inequality (delta= 0.001 * AvgEdgeLen) > 5% or not

$veryBadMeshPaths = @()
$kindaBadMeshPaths = @()
$invalidMeshPaths = @()


# read the validMeshPaths.txt, each line is a path to be run
$validMeshPaths = Get-Content "validMeshPaths.txt"

# get the path to the directory containing the executable
$executableDir = "gc-polyscope-project-template\build\bin\Release\gc_project.exe"

$i = 0
# run the executable on each mesh file in the meshes directory
foreach ($meshPath in $validMeshPaths) {
    $output = & $executableDir $meshPath

    # if the output is empty, then the executable failed to run, so such meshes will be added to the invalid meshes list, get the length of the output
    if ($output.length -eq 0) {
        $invalidMeshPaths += $meshPath
        continue
    }

    # first character of the output should be 0 or 1
    if ($output[0] -eq "1") {
        $invalidMeshPaths += $meshPath
    }
    elseif ($output[0] -eq "4") {
        $veryBadMeshPaths += $meshPath
    }
    elseif ($output[0] -eq "5") {
        $kindaBadMeshPaths += $meshPath
    }

    # increment the counter
    $i++

    if ($i % 100 -eq 0) {
        Write-Host $i
        # print the count of valid and invalid meshes
        Write-Host "Invalid meshes: " $invalidMeshPaths.Count
        Write-Host "Very bad meshes: " $veryBadMeshPaths.Count
        Write-Host "Kinda bad meshes: " $kindaBadMeshPaths.Count
    }
}
# print the count of valid and invalid meshes
Write-Host "Invalid meshes: " $invalidMeshPaths.Count
Write-Host "Very bad meshes: " $veryBadMeshPaths.Count
Write-Host "Kinda bad meshes: " $kindaBadMeshPaths.Count

# write the list of invalid mesh paths to a file
$invalidMeshPaths | Out-File "invalidMeshPaths2.txt"

# write the list of very bad mesh paths to a file
$veryBadMeshPaths | Out-File "veryBadMeshPaths.txt"

# write the list of kinda bad mesh paths to a file
$kindaBadMeshPaths | Out-File "kindaBadMeshPaths.txt"




